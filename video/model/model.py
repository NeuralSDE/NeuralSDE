from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from torchdiffeq import odeint
from tqdm import tqdm

from lutils.configuration import Configuration
from lutils.dict_wrapper import DictWrapper
from model.vector_field_regressor import build_vector_field_regressor
from model.vqgan.taming.autoencoder import vq_f8_ddconfig, vq_f8_small_ddconfig, vq_f16_ddconfig, VQModelInterface
from model.vqgan.vqvae import build_vqvae
from diffusers import AutoencoderKL
import torchsde
from evaluation.sde import NeuralSDE

def interpolate_position(time, time_start, time_end, pos_start, pos_end):
    """
    Interpolate the position between start and end points based on the given time.

    Args:
        time (float): The time at which to interpolate.
        time_start (float): The start time.
        time_end (float): The end time.
        pos_start (np.ndarray): The position at the start time.
        pos_end (np.ndarray): The position at the end time.

    Returns:
        np.ndarray: The interpolated position.
    """
    weight = (time - time_start) / (time_end - time_start)
    position_interp = (1 - weight) * pos_start + weight * pos_end
    return position_interp

class SubModel(nn.Module):
    def __init__(self, vector_field_regressor):
        super(SubModel, self).__init__()
        self.vector_field_regressor = vector_field_regressor
    
    def forward(self, x):
        return self.vector_field_regressor(input_latents=x)

class Model(nn.Module):
    def __init__(self, config: Configuration):
        super(Model, self).__init__()

        self.config = config
        self.sigma = config["sigma"]

        if config["autoencoder"]["type"] == "ours":
            self.ae = build_vqvae(
                config=config["autoencoder"],
                convert_to_sequence=True)
            self.ae.backbone.load_from_ckpt(config["autoencoder"]["ckpt_path"])
        elif config["autoencoder"]["type"] == "sd-vae-ft-mse":
            self.ae=AutoencoderKL.from_pretrained("./stabilityai/sd-vae-ft-mse")
        else:
            if config["autoencoder"]["config"] == "f8":
                ae_config = vq_f8_ddconfig
            elif config["autoencoder"]["config"] == "f8_small":
                ae_config = vq_f8_small_ddconfig
            else:
                ae_config = vq_f16_ddconfig
            self.ae = VQModelInterface(ae_config, config["autoencoder"]["ckpt_path"])
        
        self.noise_std = torch.tensor(self.config["noise_std"])
        self.func=config["func"]
        self.num_input_frames=config["num_input_frames"]
        self.state_size=config["vector_field_regressor"]["state_size"]

        self.vector_field_regressor = build_vector_field_regressor(
            config=self.config["vector_field_regressor"],
            num_input_frames=self.num_input_frames,
            func=self.func
        )
        self.logit_range=None
        self.f=None
        self.d=None
        if self.func == "g":
            self.f = self._create_and_load_submodel("f", config)
            self.d = self._create_and_load_submodel("d", config)
            self.logit_range=config["logit_range"]

    def _create_and_load_submodel(self, func_type: str, config: Configuration):
        vector_field_regressor = build_vector_field_regressor(
            config=self.config["vector_field_regressor"],
            num_input_frames=self.num_input_frames,
            func=func_type
        )
        
        submodel = SubModel(vector_field_regressor)
        
        
        ckpt_path = config["f_ckpt_path"] if func_type == "f" else config["d_ckpt_path"]
        loaded_state = torch.load(ckpt_path, map_location="cpu",weights_only=True)
        state = loaded_state["model"]
        
        
        vector_field_state = {k.replace("vector_field_regressor.", ""): v for k, v in state.items() 
                              if k.startswith("vector_field_regressor.")}
        
        
        submodel.vector_field_regressor.load_state_dict(vector_field_state)
        
        return submodel

    def load_from_ckpt(self, ckpt_path: str):
        loaded_state = torch.load(ckpt_path, map_location="cpu",weights_only=True)
        
        # Get model state dict
        state = loaded_state["model"]
        
        # Check if current model is DDP
        current_model_is_ddp = isinstance(self, torch.nn.parallel.DistributedDataParallel)
        
        # Check if loaded state is from DDP model
        loaded_state_is_ddp = any(k.startswith("module.") for k in state)
        
        # Adjust key names based on current model and loaded state
        if loaded_state_is_ddp and not current_model_is_ddp:
            # If loading DDP state into non-DDP model, remove "module." prefix
            state = {k.replace("module.", ""): v for k, v in state.items()}
        elif not loaded_state_is_ddp and current_model_is_ddp:
            # If loading non-DDP state into DDP model, add "module." prefix
            state = {f"module.{k}": v for k, v in state.items()}
        
        # Get actual model instance
        model = self.module if current_model_is_ddp else self
        model.load_state_dict(state)

    def forward(
            self,
            observations: torch.Tensor) -> DictWrapper[str, Any]:
        """

        :param observations: [b, num_observations, num_channels, height, width]
        """

        batch_size = observations.size(0)
        num_observations = observations.size(1)
        assert num_observations > self.num_input_frames and self.num_input_frames > 1

        # Sample target frames and conditioning
        target_frames_indices = torch.randint(low=self.num_input_frames, high=num_observations, size=[batch_size])
        target_frames = observations[torch.arange(batch_size), target_frames_indices] #shape: [batch_size, C, H, W]
        target_frames = target_frames.unsqueeze(1)
        
        #input frames is from t-num_input_frames to t-1
        input_frames_indices_start = target_frames_indices - self.num_input_frames  
        input_frames_indices_end = target_frames_indices - 1
        
        # Create input frame indices sequence for each batch
        input_frames_indices = torch.stack([
            torch.arange(start, end + 1, device=observations.device) 
            for start, end in zip(input_frames_indices_start, input_frames_indices_end)
        ])  # shape: [batch_size, num_input_frames]
        
        # Use advanced indexing to get all input frames
        batch_indices = torch.arange(batch_size, device=observations.device).unsqueeze(1).expand(-1, self.num_input_frames)
        input_frames = observations[batch_indices, input_frames_indices]  # shape: [batch_size, num_input_frames, C, H, W]

        # #reference frame is t-1
        # reference_frames = observations[torch.arange(batch_size), input_frames_indices_end:input_frames_indices_end+1] #shape: [batch_size,1 C, H, W]

        #original context frame
        # conditioning_frames_indices = torch.cat(
        #     [torch.randint(low=0, high=s - 1, size=[1]) for s in target_frames_indices], dim=0)

        # #ablation context frame, context frame is always t-2
        # conditioning_frames_indices = torch.cat(
        #     [torch.randint(low=s-2, high=s - 1, size=[1]) for s in target_frames_indices], dim=0)
        # conditioning_frames = observations[torch.arange(batch_size), conditioning_frames_indices]

        # Encode observations to latent codes
        with torch.no_grad():
            all_frames = torch.cat([input_frames,target_frames], dim=1)
            self.ae.eval()
            if self.config["autoencoder"]["type"] == "ours":
                latents = self.ae(all_frames).latents
            elif self.config["autoencoder"]["type"] == "sd-vae-ft-mse":
                #sd-vae-ft-mse
                flat_all_frames = rearrange(all_frames, "b n c h w -> (b n) c h w")
                flat_latents = self.ae.encode(flat_all_frames).latent_dist.sample()
                latents = rearrange(flat_latents, "(b n) c h w -> b n c h w", n=self.num_input_frames+1).contiguous()
            else:
                # # 记录开始时间
                # start = torch.cuda.Event(enable_timing=True)
                # end = torch.cuda.Event(enable_timing=True)
                # start.record()
                
                flat_all_frames = rearrange(all_frames, "b n c h w -> (b n) c h w")
                #orignal ae
                flat_latents = self.ae.encode(flat_all_frames)
                latents = rearrange(flat_latents, "(b n) c h w -> b n c h w", n=self.num_input_frames+1).contiguous()
                
                # # 记录结束时间并计算耗时
                # end.record()
                # torch.cuda.synchronize()
                # elapsed_time = start.elapsed_time(end)
                # print(f"Encoding time: {elapsed_time:.10f} ms")
        target_latents = latents[:, -1:]
        input_latents = latents[:, :-1]

        pos_start=rearrange(input_latents, "b n c h w -> b (n c) h w") #shape: [batch_size, num_input_frames*c, h, w]
        pos_end=rearrange(torch.cat([input_latents[:,1:],target_latents],dim=1), "b n c h w -> b (n c) h w").contiguous() #shape: [batch_size, num_input_frames*c, h, w]

        timestamps = torch.rand(batch_size, 1, 1, 1).to(pos_start.dtype).to(pos_start.device)
        pos_interp = (1 - (1 - self.sigma) * timestamps) * pos_start + timestamps * pos_end

        delta_t = 1.0 



        noise=torch.randn_like(pos_interp)*self.noise_std.to(pos_interp.device)
        noisy_pos_interp = pos_interp + noise



        # Calculate target vectors
        target_vectors=None
        if self.func=="f_nll" or self.func=="f_mse":
            target_vectors = ((pos_end - (1 - self.sigma) * pos_interp) / (1 - (1 - self.sigma) * timestamps))[:,-self.state_size:]
        elif self.func=="d":
            target_vectors = -noise
        elif self.func=="g":
            dx_dt = ((pos_end - (1 - self.sigma) * pos_interp) / (1 - (1 - self.sigma) * timestamps))[:,-self.state_size:]
            with torch.no_grad():
                f_pred = self.f(noisy_pos_interp)
                target_vectors = (dx_dt - f_pred)**2*delta_t
        # Calculate time distances
        # index_distances = (reference_frames_indices - conditioning_frames_indices).to(reference_latents.device)

        # Predict vectors
        reconstructed_vectors = self.vector_field_regressor(
            input_latents=noisy_pos_interp,
            # index_distances=index_distances,
            # timestamps=timestamps.squeeze(3).squeeze(2).squeeze(1)
        )

        return DictWrapper(
            # Inputs
            observations=observations,

            # Data for loss calculation
            reconstructed_vectors=reconstructed_vectors,
            target_vectors=target_vectors)

    @torch.no_grad()
    def generate_frames(
            self,
            observations: torch.Tensor,
            num_frames: int = None,
            verbose: bool = False) -> torch.Tensor:
        """
        Generates num_frames frames conditioned on observations

        :param observations: [b, num_observations, num_channels, height, width]
        :param num_frames: number of frames to generate
        :param verbose: whether to display loading bar
        """

        # Encode observations to latents
        self.ae.eval()
        if self.config["autoencoder"]["type"] == "ours":
            latents = self.ae(observations).latents
        elif self.config["autoencoder"]["type"] == "sd-vae-ft-mse":
            flat_input_frames = rearrange(observations, "b n c h w -> (b n) c h w")
            #sd-vae-ft-mse
            flat_latents = self.ae.encode(flat_input_frames).latent_dist.sample()
            latents = rearrange(flat_latents, "(b n) c h w -> b n c h w", n=observations.size(1)).contiguous()
        else:
            flat_input_frames = rearrange(observations, "b n c h w -> (b n) c h w")
            #orignal ae
            flat_latents = self.ae.encode(flat_input_frames)
            latents = rearrange(flat_latents, "(b n) c h w -> b n c h w", n=observations.size(1)).contiguous()

        b, n, c, h, w = latents.shape
        if n == 1:
            latents = latents[:, [0, 0]]

        f=None
        if self.func=="f_nll" or self.func=="f_mse":
            sde_model = NeuralSDE(
                flow=self.vector_field_regressor,
                diffusion=None,
                logit_range=None,
                denoiser=None,
                denoising_magnitude=1.0,
                state_shape=(b,self.num_input_frames,c,h,w),
                latents=latents
            )
        elif self.func=="d":
            sde_model = NeuralSDE(
                flow=None,
                diffusion=self.vector_field_regressor,
                logit_range=None,
                denoiser=None,
                denoising_magnitude=1.0,
                state_shape=(b,self.num_input_frames,c,h,w),
                latents=latents
            )
        elif self.func=="g":
            sde_model = NeuralSDE(
                flow=self.f,
                diffusion=self.vector_field_regressor,
                logit_range=self.logit_range,
                denoiser=self.d,
                denoising_magnitude=1.0,
                state_shape=(b,self.num_input_frames,c,h,w),
                latents=latents
            )
        gen = tqdm(range(num_frames), desc="Generating frames", disable=not verbose, leave=False)
        for _ in gen:
            # Initialize 
            if self.func=="f_mse" or self.func=="f_nll":
                y0=rearrange(latents[:, -self.num_input_frames:], "b n c h w -> b (n c) h w")
            elif self.func=="d":
                y0=rearrange(latents[:, -self.num_input_frames:], "b n c h w -> b (n c) h w")
                y0=y0+torch.randn_like(y0)*self.noise_std
            elif self.func=="g":
                y0=rearrange(latents[:, -self.num_input_frames:], "b n c h w -> b (n c) h w")


            # Solve SDE
            y0=rearrange(y0, "b c h w -> b (c h w)")
            sde_model.latents=latents
            next_latents = torchsde.sdeint(
                sde=sde_model,
                y0=y0,
                ts=torch.linspace(0,1,2).to(y0.device),
                method="milstein",  # 'srk', 'euler', 'milstein', etc.
                dt=1e0,
                adaptive=True,
                rtol=1e-10,
                atol=1e-2,
                dt_min=1e-10,
            )
            next_latents = rearrange(next_latents, "n b (c h w) -> b n c h w", b=b,n=2,c=self.num_input_frames*c,h=h,w=w).contiguous()
            latents = torch.cat([latents, next_latents[:,-1:,-self.state_size:]], dim=1)
        gen.close()

        if n == 1:
            latents = latents[:, 1:]

        # Decode to image space
        latents = rearrange(latents, "b n c h w -> (b n) c h w")
        if self.config["autoencoder"]["type"] == "ours":
            reconstructed_observations = self.ae.backbone.decode_from_latents(latents)
        elif self.config["autoencoder"]["type"] == "sd-vae-ft-mse":
            #sd-vae-ft-mse
            reconstructed_observations = self.ae.decode(latents).sample
        else:
            #orignal ae
            reconstructed_observations = self.ae.decode(latents)
        reconstructed_observations = rearrange(reconstructed_observations, "(b n) c h w -> b n c h w", b=b).contiguous()

        return reconstructed_observations
