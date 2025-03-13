import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("project_root:",project_root)
sys.path.append(project_root)


import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from dataset import VideoDataset
from lutils.configuration import Configuration
from lutils.logging import to_video
from model import Model
from evaluation.sde import NeuralSDE
from einops import rearrange
import torchsde
import torch.nn as nn
import gc
import json
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from evaluation.eval_utils import save_tensors_to_h5

eval_space = {
    "num_completions":1,
    "num_test_videos":4,

    "dataset": ["clevrer"],
    "steps":["final_step_300000"],
    "model":['f_nll-d-g_nll'], #suppose run name is format like f-01-null, 01 means the noise level is 0.1, null means negitive likelihood loss
    "batch_size": 4,
    "num_workers": 7,
    "fps":3, #video fps
    "res":1, #video resolution=fps*res
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class ReplacementDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, seed=0, num_samples=None):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.seed = seed
        self.num_samples = num_samples if num_samples is not None else len(dataset)
        self.total_size = self.num_samples
        
    def __len__(self):
        return self.num_samples
        
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        indices = torch.randint(len(self.dataset), 
                              size=(self.num_samples,), 
                              generator=g).tolist()
                              
        indices = indices[self.rank:self.num_samples:self.num_replicas]
        return iter(indices)



@torch.no_grad()
def encode_decode_videos_batch(
        observations: torch.Tensor,
        ae: nn.Module = None,
        config: Configuration = None,
    ) -> torch.Tensor:
    """
    Generates num_frames frames conditioned on observations
    """
    # Get the original model if it's wrapped in DDP
    if isinstance(ae, DDP):
        ae = ae.module

    # Encode observations to latents
    ae.eval()
    if config["autoencoder"]["type"] == "ours":
        latents = ae(observations).latents
    elif config["autoencoder"]["type"] == "sd-vae-ft-mse":
        flat_input_frames = rearrange(observations, "b n c h w -> (b n) c h w")
        #sd-vae-ft-mse
        flat_latents = ae.encode(flat_input_frames).latent_dist.sample()
        latents = rearrange(flat_latents, "(b n) c h w -> b n c h w", n=observations.size(1))
    else:
        flat_input_frames = rearrange(observations, "b n c h w -> (b n) c h w")
        #orignal ae
        flat_latents = ae.encode(flat_input_frames)
        latents = rearrange(flat_latents, "(b n) c h w -> b n c h w", n=observations.size(1))

    b, n, c, h, w = latents.shape
    if n == 1:
        latents = latents[:, 1:]
    # Decode to image space
    latents = rearrange(latents, "b n c h w -> (b n) c h w")
    if config["autoencoder"]["type"] == "ours":
        reconstructed_observations = ae.backbone.decode_from_latents(latents)
    elif config["autoencoder"]["type"] == "sd-vae-ft-mse":
        #sd-vae-ft-mse
        reconstructed_observations = ae.decode(latents).sample
    else:
        #orignal ae
        reconstructed_observations = ae.decode(latents)
    reconstructed_observations = rearrange(reconstructed_observations, "(b n) c h w -> b n c h w", b=b).contiguous()

    return reconstructed_observations

@torch.no_grad()
def save_encode_decode_videos(dataset_name, dl, data_config):
    local_rank = int(os.environ["LOCAL_RANK"])
    # Calculate completions per GPU
    world_size = dist.get_world_size()
    total_completions = int(eval_space["num_completions"])
    completions_per_gpu = total_completions // world_size
    # Handle remaining completions
    extra_completions = 1 if local_rank < (total_completions % world_size) else 0
    local_completions = completions_per_gpu + extra_completions
    
    model_type='f'
    loss_func = 'mse' 
    step_str='final_step_300000'
    # Create a dictionary to store loaded models
    loaded_models = {}

    loaded_models[MODEL_CONFIGS[model_type]['name']] = load_model(
        model_type,
        project_root,
        dataset_name,
        step_str,
        loss_func,
        device=f"cuda:{local_rank}"
    )

    # Assign models to variables
    flow = loaded_models.get('flow')


    if flow is not None:
        ae=flow.ae
        ae = DDP(ae, device_ids=[local_rank])

    for completion_index in tqdm(range(local_completions), desc=f"cuda:{local_rank} encode epoch"): #tqdm
        encode_decode_target_videos = encode_decode_videos_epoch(dl, data_config, ae)
        path = f'{project_root}/evaluation/encode_decode_videos_nfe/{dataset_name}/cuda{local_rank}_{completion_index}.h5'
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_tensors_to_h5(encode_decode_target_videos, path)
        
        del encode_decode_target_videos
        encode_decode_target_videos = None

    # Clean up models
    if flow is not None:
        del flow
        del ae
        flow = None
        ae = None

    dist.barrier()  # Synchronize processes
    gc.collect()
    torch.cuda.empty_cache()

@torch.no_grad()
def encode_decode_videos_epoch(data_loader, data_config, ae=None):
    
    device=torch.cuda.current_device()
    encode_decode_videos = []
    target_videos = []
    for x in tqdm(
            data_loader, 
            total=eval_space["num_test_videos"]//eval_space["batch_size"],
            desc=f"cuda:{device} encode Batch"
        ):
        
        x = x.to(device, non_blocking=True)
        num_condition_frames = data_config["evaluation"]["condition_frames"]
        num_frames_to_generate = data_config["evaluation"]["frames_to_generate"]
        #num_frames_to_generate=eval_space["frames_to_generate"]
        encode_decode = encode_decode_videos_batch(
                            observations=x[:, :num_condition_frames+num_frames_to_generate],
                            ae=ae,
                            config=data_config["model"],   
                        ).to(device, non_blocking=True)
        encode_decode_videos.append(encode_decode)
        target_videos.append(x[:, :num_condition_frames+num_frames_to_generate])
    res={'encode_decode_videos':torch.cat(encode_decode_videos, dim=0).to(device).to(torch.float32),
         'target_videos':torch.cat(target_videos, dim=0).to(device).to(torch.float32)}
    return res

@torch.no_grad()
def generate_videos_batch(
        observations: torch.Tensor,
        verbose: bool = False,
        num_frames: int = None,
        ae: nn.Module = None,
        flow: nn.Module = None,
        denoiser: nn.Module = None,
        diffusion: nn.Module = None,
        denoising_magnitude: float = None,
        diffusion_magnitide: float = None,
        config: Configuration = None,
        atol: float = None,
        num_steps: int = None,
    ) -> torch.Tensor:
    """
    Generates num_frames frames conditioned on observations

    :param observations: [b, num_observations, num_channels, height, width]
    :param num_frames: number of frames to generate
    :param warm_start: part of the integration path to jump to
    :param steps: number of steps for sampling
    :param past_horizon: number of frames to condition on
    :param verbose: whether to display loading bar
    """

    # Get original models if they're wrapped in DDP
    if isinstance(ae, DDP):
        ae = ae.module
    if isinstance(flow, DDP):
        flow = flow.module
    if isinstance(denoiser, DDP):
        denoiser = denoiser.module
    if isinstance(diffusion, DDP):
        diffusion = diffusion.module

    device_id=torch.cuda.current_device()
    #verbose=True if device_id==0 else False
    # Encode observations to latents
    ae.eval()
    if config["autoencoder"]["type"] == "ours":
        latents = ae(observations).latents
    elif config["autoencoder"]["type"] == "sd-vae-ft-mse":
        flat_input_frames = rearrange(observations, "b n c h w -> (b n) c h w")
        #sd-vae-ft-mse
        flat_latents = ae.encode(flat_input_frames).latent_dist.sample()
        latents = rearrange(flat_latents, "(b n) c h w -> b n c h w", n=observations.size(1))
    else:
        flat_input_frames = rearrange(observations, "b n c h w -> (b n) c h w")
        #orignal ae
        flat_latents = ae.encode(flat_input_frames)
        latents = rearrange(flat_latents, "(b n) c h w -> b n c h w", n=observations.size(1))

    b, n, c, h, w = latents.shape
    num_input_frames=flow.num_input_frames
    state_size=flow.state_size
    if n == 1:
        latents = latents[:, [0, 0]]

    def f(t: torch.Tensor, y: torch.Tensor):
        # Calculate vectors
        dy_1=flow.vector_field_regressor(
            input_latents=y,

        ) #shape: [batch_size, c, h, w]
        dy_0=latents[:,-num_input_frames+1:]-latents[:,-num_input_frames:-1]
        dy=torch.cat([rearrange(dy_0, "b n c h w -> b (n c) h w"),dy_1],dim=1)

        denoise=denoiser.vector_field_regressor(input_latents=y)

        dy=dy+denoise*denoising_magnitude
        return dy
    
    sde_model = NeuralSDE(
        flow=flow.vector_field_regressor,
        diffusion=diffusion.vector_field_regressor if diffusion is not None else None,
        logit_range=diffusion.logit_range if diffusion is not None else None,
        denoiser=denoiser.vector_field_regressor if denoiser is not None else None,
        denoising_magnitude=denoising_magnitude,
        diffusion_magnitide=diffusion_magnitide,
        state_shape=(b,num_input_frames,c,h,w),
        latents=latents
    )


    res=eval_space["res"]
    gen = tqdm(range(num_frames), desc=f"cuda:{device_id} generating latents", disable=not verbose, leave=False)
    sde_model.count=0
    latents2show=latents

    for _ in gen:
        # Initialize
        y0=rearrange(latents[:, -num_input_frames:], "b n c h w -> b (n c) h w")
        

        # Solve SDE
        y0=rearrange(y0, "b c h w -> b (c h w)")
        sde_model.latents=latents
        next_latents = torchsde.sdeint(
            sde=sde_model,
            y0=y0,
            ts=torch.linspace(0,1,res+1).to(y0.device),
            method="milstein",  # 'srk', 'euler', 'milstein', etc.
            #dt=1.0/num_steps,
            dt=1e0,
            adaptive=True,
            rtol=1e-10,
            atol=atol,
            dt_min=1e-10,

        )
        #sde_model.count+=1
        next_latents = rearrange(next_latents, "n b (c h w) -> b n c h w", b=b,n=res+1,c=num_input_frames*c,h=h,w=w).contiguous()



        # #Solve ODE
        # next_latents = odeint(
        #     func=f,
        #     y0=y0,
        #     t=torch.linspace(0,1,2).to(y0.device),
        #     method="rk4"
        # )
        # next_latents = rearrange(next_latents, "n b c h w -> b n c h w")
        
        latents2show=torch.cat([latents2show,next_latents[:,1:,-state_size:]],dim=1)
        latents = torch.cat([latents, next_latents[:,-1:,-state_size:]], dim=1)
        sde_model.latents = latents
    gen.close()
    print(f"\n\nsde_model.count:{sde_model.count}\n\n")



    b, n, c, h, w = latents.shape
    if n == 1:
        latents = latents[:, 1:]
    # Decode to image space
    latents2show = rearrange(latents2show, "b n c h w -> (b n) c h w")
    if config["autoencoder"]["type"] == "ours":
        reconstructed_observations = ae.backbone.decode_from_latents(latents2show)
    elif config["autoencoder"]["type"] == "sd-vae-ft-mse":
        #sd-vae-ft-mse
        reconstructed_observations = ae.decode(latents2show).sample
    else:
        #orignal ae
        reconstructed_observations = ae.decode(latents2show)
    reconstructed_observations = rearrange(reconstructed_observations, "(b n) c h w -> b n c h w", b=b).contiguous()

    return reconstructed_observations,sde_model.count

@torch.no_grad()
def generate_videos_epoch(data_loader, data_config, f=None, d=None, g=None, ae=None,atol=None,num_steps=None):
    
    device=torch.cuda.current_device()
    generated_videos = []
    target_videos = []
    total_count=0
    for x in tqdm(
            data_loader, 
            total=eval_space["num_test_videos"]//eval_space["batch_size"],
            desc=f"cuda:{device} generate Batch"
        ):
        
        x = x.to(device, non_blocking=True)
        num_condition_frames = data_config["evaluation"]["condition_frames"]
        num_frames_to_generate = data_config["evaluation"]["frames_to_generate"]
        #num_frames_to_generate=eval_space["frames_to_generate"]

        generated,count = generate_videos_batch(
                            observations=x[:, :num_condition_frames],
                            num_frames=num_frames_to_generate,
                            verbose=True,
                            ae=ae,
                            flow=f,
                            denoiser=d,
                            diffusion=g,
                            denoising_magnitude=1.0,
                            diffusion_magnitide=1.0,
                            config=data_config["model"],   
                            atol=atol,
                            num_steps=num_steps,
                        )
        generated_videos.append(generated.to(device, non_blocking=True))
        total_count+=count
        target_videos.append(x[:, :num_condition_frames+num_frames_to_generate]) # e,b,t,c,h,w
    generated_videos=torch.cat(generated_videos, dim=0).to(device).to(torch.float32) #b,t,c,h,w
    target_videos=torch.cat(target_videos, dim=0).to(device).to(torch.float32) #b,t,c,h,w
    avg_count=torch.tensor(total_count/(eval_space["num_test_videos"]//eval_space["batch_size"]))
    res={'generated_videos':generated_videos,
         'target_videos':target_videos,
         'avg_count':avg_count}
    return res

@torch.no_grad()
def save_generated_videos(dataset_name, dl, data_config):
    local_rank = int(os.environ["LOCAL_RANK"])
    # Calculate completions per GPU
    world_size = dist.get_world_size()
    total_completions = int(eval_space["num_completions"])
    completions_per_gpu = total_completions // world_size
    # Handle remaining completions
    extra_completions = 1 if local_rank < (total_completions % world_size) else 0
    local_completions = completions_per_gpu + extra_completions
    for model_name in eval_space["model"]:
        models = [m for m in ['f', 'd', 'g'] if m in model_name] 
        loss_func = 'mse' if 'f_mse' in model_name else 'nll' if 'f_nll' in model_name else None 
        for step_str in eval_space["steps"]:
            # Create a dictionary to store loaded models
            loaded_models = {}
            # Main loading logic
            for model_type in models:
                if model_type in MODEL_CONFIGS:
                    loaded_models[MODEL_CONFIGS[model_type]['name']] = load_model(
                        model_type,
                        project_root,
                        dataset_name,
                        step_str,
                        loss_func,
                        device=f"cuda:{local_rank}"
                    )

            # Assign models to variables
            flow = loaded_models.get('flow')
            denoiser = loaded_models.get('denoiser')
            diffusion = loaded_models.get('diffusion')

            if flow is not None:
                ae=flow.ae
            if flow is not None:
                flow = DDP(flow, device_ids=[local_rank ])
            if denoiser is not None:
                denoiser = DDP(denoiser, device_ids=[local_rank])
            if diffusion is not None:
                diffusion = DDP(diffusion, device_ids=[local_rank])
            if ae is not None:
                ae = DDP(ae, device_ids=[local_rank])

            for completion_index in tqdm(range(local_completions), desc=f"cuda:{local_rank} generate epoch"): #tqdm
                #for atol in [5e-3]:
                for num_steps in [10]:
                    generated_target_videos = generate_videos_epoch(dl, data_config, flow, denoiser, diffusion, ae,atol,None)
                    path=f'{project_root}/evaluation/generated_videos/{dataset_name}/{model_name}/{step_str}/cuda{local_rank}_{completion_index}_{atol}.h5'
                    if not os.path.exists(os.path.dirname(path)):
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                    
                    save_tensors_to_h5(generated_target_videos, path)
                    save_gif(generated_target_videos['generated_videos'],cmap=None,fps=eval_space["fps"],base_path=os.path.dirname(path),res=eval_space["res"],gif_name=f"cuda{local_rank}_{completion_index}_{num_steps}")
                    #save_fig(generated_target_videos['generated_videos'],cmap=None,base_path=os.path.dirname(path),fig_name=f"cuda{local_rank}_{completion_index}_{atol}-0-15")
                    #save_gif(generated_target_videos['target_videos'],cmap=None,fps=eval_space["fps"],base_path=os.path.dirname(path),res=1,gif_name=f"cuda{local_rank}_{completion_index}_{atol}_target")
                    #save_fig(generated_target_videos['target_videos'],cmap=None,base_path=os.path.dirname(path),fig_name=f"cuda{local_rank}_{completion_index}_{atol}_target")
                    del generated_target_videos
                    generated_target_videos = None

            # Clean up models
            if flow is not None:
                del flow
                del ae
                flow = None
                ae = None
            if denoiser is not None:
                del denoiser
                denoiser = None
            if diffusion is not None:
                del diffusion
                diffusion = None

            dist.barrier()  # Synchronize processes
            gc.collect()
            torch.cuda.empty_cache()
    return

@torch.no_grad()
def save_all_videos():
    """Main evaluation function with distributed support"""
    # Initialize distributed training
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for distributed training")
    
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    
    # Calculate completions per GPU
    world_size = dist.get_world_size()

    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
    print(f"RANK: {os.environ.get('RANK', 'Not set')}")

    for dataset_name in eval_space["dataset"]:
            
        config_path = os.path.join("./configs", dataset_name+".yaml")
        data_config = Configuration(config_path)
        path = os.path.join(data_config["data"]["data_root"], "test")
        dataset = VideoDataset(
            data_path=path,
            input_size=data_config["data"]["input_size"],
            crop_size=data_config["data"]["crop_size"],
            frames_per_sample=data_config["data"]["frames_per_sample"],
            skip_frames=data_config["data"]["skip_frames"],
            random_horizontal_flip=False,
            aug=False,
            albumentations=False)

        samples_per_gpu = eval_space["num_test_videos"]

        sampler = ReplacementDistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=local_rank,
            seed=42,
            num_samples=samples_per_gpu * world_size  
        )

        dl = DataLoader(
            dataset=dataset,
            batch_size=eval_space["batch_size"],
            shuffle=False,
            num_workers=eval_space["num_workers"],
            sampler=sampler,
            pin_memory=True
        )
        #save_encode_decode_videos(dataset_name, dl, data_config)
        save_generated_videos(dataset_name, dl, data_config)
        

    # Gather and merge results from all processes
    dist.barrier()
    dist.destroy_process_group()
    return

def animate_diff(i, observations,axs,cmap):
    print(f'gif animating frame {i} of {observations.shape[1]}', end='\r')
    plots = []
    for col in range(observations.shape[0]):
        axs[col].clear()
        axs[col].set_xticks([])
        axs[col].set_yticks([])

        frame = observations[col][i]
        if frame.shape[0]==1:
            plots.append(axs[col].imshow(frame.squeeze(), cmap=cmap, vmin=-1.5, vmax=1.5))
        elif frame.shape[0]==3:
            plots.append(axs[col].imshow(frame.transpose(1,2,0)))  
    return plots

def save_gif(observations=None,cmap=None,fps=None,base_path=None,res=None,gif_name=None):
    observations=to_video(observations[:,3:])
    if observations.shape[0] == 1:
        fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(5,10))
        axs = axs.reshape(1, 1)  
    else:
        fig, axs = plt.subplots(nrows=1, ncols=observations.shape[0], sharex=True, sharey=True, figsize=(5*observations.shape[0],5))

    ani = FuncAnimation(fig, animate_diff, fargs=[np.array(observations),axs,cmap],  interval=1000/(fps*res), blit=False, repeat=True, frames=observations.shape[1]) 
    gif_path = os.path.join(base_path, f"{gif_name}.gif")
    ani.save(gif_path, dpi=100, writer=PillowWriter(fps=fps*res))
    for ax in axs:
        ax.clear()
    plt.close()

def save_fig(observations=None,cmap=None,base_path=None,fig_name=None):
    observations=to_video(observations[:,3:])
    # if observations.shape[1] == 13:
    #     observations=observations[:,0:3]
    if observations.shape[1] == 61:
        observations=observations[:,0:16]
    fig, axs = plt.subplots(nrows=observations.shape[0], ncols=observations.shape[1], sharex=True, sharey=True, figsize=(observations.shape[1],observations.shape[0]))
    for row in range(observations.shape[0]):
        for col in range(observations.shape[1]):
            frame=observations[row,col]
            axs[row,col].set_xticks([])
            axs[row,col].set_yticks([])
            if frame.shape[0]==1:
                axs[row,col].imshow(frame.squeeze(), cmap=cmap, vmin=-1.5, vmax=1.5)
            elif frame.shape[0]==3:
                axs[row,col].imshow(frame.transpose(1,2,0))  
    fig.savefig(os.path.join(base_path, f"{fig_name}.png"),dpi=370)
    if axs.ndim == 2:
        for row in axs:
            for ax in row:
                ax.clear()
    else:
        for ax in axs:
            ax.clear()
    plt.close()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)

MODEL_CONFIGS = {
    'f': {
        'name': 'flow',
        'run_suffix': 'f-{loss_func}',
        'config_suffix': 'f-01-{loss_func}',
        'needs_loss_func': True
    },
    'd': {
        'name': 'denoiser',
        'run_suffix': 'd-01',
        'config_suffix': 'd-01',
        'needs_loss_func': False
    },
    'g': {
        'name': 'diffusion',
        'run_suffix': 'g-01-{loss_func}',
        'config_suffix': 'g-01-{loss_func}',
        'needs_loss_func': True
    },
    'fm': {
        'name': 'fm',
        'run_suffix': 'fm',
        'config_suffix': 'fm',
        'needs_loss_func': False
    },
    'pfi': {
        'name': 'pfi',
        'run_suffix': 'pfi-10',
        'config_suffix': 'pfi-10',
        'needs_loss_func': False
    },
}

def load_model(model_type, project_root, dataset_name, step_str, loss_func=None, device=None):
    """
    Load a model based on its type and configuration
    """
    config = MODEL_CONFIGS[model_type]
    suffix = config['run_suffix'].format(loss_func=loss_func) if config['needs_loss_func'] else config['run_suffix']
    config_suffix = config['config_suffix'].format(loss_func=loss_func) if config['needs_loss_func'] else config['config_suffix']
    
    model_path = os.path.join(
        f'{project_root}/runs',
        f'{dataset_name}_run-{suffix}',
        'checkpoints',
        f'{step_str}.pth'
    )
    
    config_path = os.path.join(
        f'{project_root}/configs',
        f'{dataset_name}_{config_suffix}.yaml'
    )
    
    model_config = Configuration(config_path)
    model = Model(model_config["model"])
    model.load_from_ckpt(model_path)
    model = model.to(device)
    model.eval()
    
    return model


if __name__ == '__main__':

    set_seed(1)
    save_all_videos()

