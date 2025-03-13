import os
from typing import Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_polynomial_decay_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau
from evaluation.evaluator import Evaluator
from lutils.configuration import Configuration
from lutils.constants import MAIN_PROCESS
from lutils.dict_wrapper import DictWrapper
from lutils.logger import Logger
from lutils.logging import to_video, make_observations_grid
from training.utils import check_ddp_consistency
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

class Trainer:
    """
    Class that handles the training
    """

    def __init__(
            self,
            rank: int,
            run_name: str,
            config: Configuration,
            dataset: Dataset,
            sampler: torch.utils.data.distributed.Sampler,
            num_gpus: int,
            device: torch.device):
        """
        Initializes the Trainer

        :param rank: rank of the current process
        :param config: training configuration
        :param dataset: dataset to train on
        :param sampler: sampler to create the dataloader with
        :param device: device to use for training
        """
        super(Trainer, self).__init__()

        self.config = config
        self.rank = rank
        self.is_main_process = self.rank == MAIN_PROCESS
        self.num_gpus = num_gpus
        self.device = device

        # Create folder for saving
        self.run_path = os.path.join("runs", run_name)
        os.makedirs(self.run_path, exist_ok=True)
        os.makedirs(os.path.join(self.run_path, "checkpoints"), exist_ok=True)

        # Setup dataloader
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config["batching"]["batch_size"],
            shuffle=False,
            num_workers=self.config["batching"]["num_workers"],
            sampler=sampler,
            pin_memory=True)

        # Setup losses
        self.flow_matching_loss = nn.MSELoss()

        # Optimizer will be defined in train_epoch
        self.optimizer = None

        # Scheduler will be defined in train_epoch
        self.lr_scheduler = None

        self.global_step = 0

    def init_optimizer(self, model: nn.Module):
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config["optimizer"]["learning_rate"],
            weight_decay=self.config["optimizer"]["weight_decay"],
            betas=(0.9, 0.999))
        lr_end=self.config["optimizer"]["learning_rate"]*1.0e-3
        self.lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.config["optimizer"]["num_warmup_steps"],
            num_training_steps=self.config["optimizer"]["num_training_steps"],
            power=0.5,
            lr_end=lr_end)
        # self.lr_scheduler=ReduceLROnPlateau(optimizer=self.optimizer,
        #     factor=0.5,
        #     patience=200,
        #     threshold=1.5e-1,
        #     threshold_mode='rel',
        #     cooldown=0,
        #     min_lr=2e-10,
        #     eps=1e-10)

    def get_lr(self):
        assert self.optimizer is not None

        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def train(
            self,
            model: nn.Module,
            logger: Logger,
            evaluator: Evaluator,
            scalar_logging_frequency: int = 100,
            media_logging_frequency: int = 5000,
            saving_frequency: int = 5000,
            evaluation_frequency: int = 20000,
            checkpointing_frequency: int = 20000):
        """
        Trains the model for one epoch

        """

        model.train()
        dmodel = model if not isinstance(model, nn.parallel.DistributedDataParallel) else model.module
        # Setup optimizer and scheduler if not yet
        if self.optimizer is None:
            self.init_optimizer(model)

        # Setup loading bar
        train_gen = tqdm(self.dataloader, desc="Batches", disable=not self.is_main_process, leave=False)
        for batch in train_gen:
            # # 创建计时器
            # data_timer = torch.cuda.Event(enable_timing=True)
            # forward_timer = torch.cuda.Event(enable_timing=True)
            # loss_timer = torch.cuda.Event(enable_timing=True)
            # backward_timer = torch.cuda.Event(enable_timing=True)
            # end_timer = torch.cuda.Event(enable_timing=True)

            # data_timer.record()
            # Fetch data
            observations = batch.cuda()
            batch_size = observations.size(0)
            num_observations = self.config["num_observations"]
            observations = observations[:, :num_observations]
            
            # forward_timer.record()
            # Forward the model
            model_outputs = model(observations)
            
            # loss_timer.record()
            # Compute the loss
            loss, auxiliary_output = self.calculate_loss(model_outputs, model.func,model.logit_range)
            
            # backward_timer.record()
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # end_timer.record()
            
            # # 确保所有操作完成并计算时间
            # torch.cuda.synchronize()
            # data_time = data_timer.elapsed_time(forward_timer)
            # forward_time = forward_timer.elapsed_time(loss_timer)
            # loss_time = loss_timer.elapsed_time(backward_timer)
            # backward_time = backward_timer.elapsed_time(end_timer)
            # total_time = data_timer.elapsed_time(end_timer)
            
            # print(f"Step {self.global_step} timing:\n"
            #       f"  Data prep: {data_time:.10f} ms\n"
            #       f"  Forward pass: {forward_time:.10f} ms\n"
            #       f"  Loss computation: {loss_time:.10f} ms\n"
            #       f"  Backward pass: {backward_time:.10f} ms\n"
            #       f"  Total time: {total_time:.10f} ms")

            # Uncomment this if nans appear in the loss during the training
            # self.reduce_gradients(model, self.num_gpus)

            # Optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()

            # Log scalars
            if self.global_step % scalar_logging_frequency == 0 and self.is_main_process:
                self.log_scalars(
                    auxiliary_output,
                    DictWrapper(
                        num_observations=num_observations),
                    logger)

            # Log media
            if self.global_step % media_logging_frequency == 0 and self.is_main_process:
                model_outputs["generated_observations"] = dmodel.generate_frames(
                    observations=observations[:min(4, batch_size), :self.config["condition_frames"]],
                    num_frames=self.config["frames_to_generate"],
                    verbose=self.is_main_process,
                )
                self.log_media(model_outputs, logger)

            # Finalize logs
            logger.finalize_logs(step=self.global_step)

            # Evaluate the model
            if self.global_step % evaluation_frequency == 0 and self.is_main_process:
                evaluator.evaluate(model=model, logger=logger, global_step=self.global_step)

            # Save checkpoint
            if self.global_step % checkpointing_frequency == 0:
                self.save_checkpoint(model, f"step_{self.global_step}.pth")
            elif self.global_step % saving_frequency == 0:
                self.save_checkpoint(model)

            self.global_step += 1

        # Close loading bar
        train_gen.close()

        # Save the model
        logger.info("Saving the trained model...")
        self.save_checkpoint(model, f"final_step_{self.global_step}.pth")

    def calculate_loss(
            self,
            results: DictWrapper[str, Any],
            func: str,
            logit_range: Optional[Tuple[float, float]] = None,
        ) -> Tuple[torch.Tensor, DictWrapper[str, Any]]:
        """
        Calculates the loss

        :param results: Dict with the model outputs
        :return: [1,] The loss value
        """

        # results.target_vectors shape: [batch_size, c, h, w]
        if func == "f_nll" or func == "f_mse":
            if func=="f_nll":
                squared_diff = (results.target_vectors - results.reconstructed_vectors)**2
                loss = (
                    torch.log(squared_diff + 1.0e-3) - 
                    torch.log(torch.tensor(1.0e-3).to(results.target_vectors.device))
                ).mean()  
            elif func=="f_mse":
                loss = nn.MSELoss()(results.reconstructed_vectors, results.target_vectors)
            flow_matching_loss = self.flow_matching_loss(results.reconstructed_vectors, results.target_vectors)

        elif func=="d":
            loss = nn.MSELoss()(results.reconstructed_vectors, results.target_vectors)
            flow_matching_loss = self.flow_matching_loss(results.reconstructed_vectors, results.target_vectors)
        
        elif func=="g":
            logit_min=logit_range[0]
            logit_max=logit_range[1]
            loss = nn.MSELoss()(
                torch.exp((torch.tanh(results.reconstructed_vectors) + 1.0) * 0.5 * (logit_max - logit_min) + logit_min), 
                torch.sqrt(results.target_vectors)
            ) + nn.MSELoss()(
                (torch.tanh(results.reconstructed_vectors) + 1.0) * 0.5 * (logit_max - logit_min) + logit_min, 
                0.5 * torch.log(results.target_vectors + 1.0e-6)
            )
            flow_matching_loss = self.flow_matching_loss(
                torch.exp((torch.tanh(results.reconstructed_vectors) + 1.0) * 0.5 * (logit_max - logit_min) + logit_min), 
                torch.sqrt(results.target_vectors)
            )
        # # Flow matching loss
        # flow_matching_loss = self.flow_matching_loss(
        #     results.reconstructed_vectors,
        #     results.target_vectors)

        # Sum up all the losses
        loss_weights = self.config["loss_weights"]
        loss = \
            loss_weights["flow_matching_loss"] * loss

        # DDP hack
        def add_zero_to_loss(value):
            if v is None:
                return loss
            return loss + value.mul(0).mean()

        for _, v in results.items():
            if isinstance(v, list):
                for ev in v:
                    loss = add_zero_to_loss(ev)
            else:
                loss = add_zero_to_loss(v)

        # Create auxiliary output
        auxiliary_output = DictWrapper(
            # Total loss
            total_loss=loss,

            # Loss terms
            flow_matching_loss=flow_matching_loss
        )

        return loss, auxiliary_output

    def log_scalars(self, loss_terms: DictWrapper[str, Any], other_data: DictWrapper[str, Any], logger: Logger):
        for k, v in loss_terms.items():
            logger.log(f"Training/Loss/{k}", v)

        # Log training stats
        logger.log(f"Training/Stats/learning_rate", self.get_lr())
        logger.log(f"Training/Stats/total_loss_is_nan", torch.isnan(loss_terms.total_loss).to(torch.int8))
        logger.log(f"Training/Stats/total_loss_is_inf", torch.isinf(loss_terms.total_loss).to(torch.int8))

        # Other stats
        for k, v in other_data.items():
            logger.log(f"Training/Stats/{k}", v)

    @staticmethod
    def log_media(results: DictWrapper[str, Any], logger: Logger):
        num_sequences = min(4, results.observations.size(0))

        # Log images grid
        grid = make_observations_grid(
            [
                results.observations,
                results.generated_observations,
            ],
            num_sequences=num_sequences)
        logger.log(f"Training/Media/reconstructed_observations", logger.wandb().Image(grid))

        # Log real videos
        real_videos = to_video(results.observations[:num_sequences])
        logger.log("Training/Media/real_videos", logger.wandb().Video(real_videos, fps=7))

        # Log generated videos
        generated_videos = to_video(results.generated_observations)
        logger.log("Training/Media/generated_videos", logger.wandb().Video(generated_videos, fps=7))

    @staticmethod
    def reduce_gradients(model: nn.Module, num_gpus: int):
        params = [param for param in model.parameters() if param.grad is not None]
        if len(params) > 0:
            flat = torch.cat([param.grad.flatten() for param in params])
            if num_gpus > 1:
                torch.distributed.all_reduce(flat)
                flat /= num_gpus
            torch.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
            grads = flat.split([param.numel() for param in params])
            for param, grad in zip(params, grads):
                param.grad = grad.reshape(param.shape)

    def save_checkpoint(self, model: nn.Module, checkpoint_name: str = None):
        if self.num_gpus > 1:
            check_ddp_consistency(model, r".*\..+_(mean|var|tracked)")

        if self.is_main_process:
            state_dict = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "model": model.state_dict(),
                "global_step": self.global_step
            }
            if checkpoint_name:
                torch.save(state_dict, os.path.join(self.run_path, "checkpoints", checkpoint_name))
            torch.save(state_dict, os.path.join(self.run_path, "checkpoints", "latest.pth"))

    def load_checkpoint(self, model: nn.Module, checkpoint_name: str = None):
        if checkpoint_name is None:
            checkpoint_name = "latest.pth"
        filename = os.path.join(self.run_path, "checkpoints", checkpoint_name)
        if not os.path.isfile(filename):
            raise Exception(f"Cannot load model: no checkpoint found at '{filename}'")

        # Init optimizer and scheduler if not yet
        if self.optimizer is None:
            self.init_optimizer(model)

        # Load checkpoint with proper device mapping
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        loaded_state = torch.load(filename, map_location=map_location)
        
        # Load optimizer and scheduler states
        self.optimizer.load_state_dict(loaded_state["optimizer"])
        self.lr_scheduler.load_state_dict(loaded_state["lr_scheduler"])

        # Get model state dict
        state = loaded_state["model"]
        
        # Check if current model is DDP
        current_model_is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
        
        # Check if loaded state is from DDP model
        loaded_state_is_ddp = any(k.startswith("module.") for k in state)
        
        # Adjust key names based on current model and loaded state
        if loaded_state_is_ddp and not current_model_is_ddp:
            # If loading DDP state into non-DDP model, remove "module." prefix
            state = {k.replace("module.", ""): v for k, v in state.items()}
        elif not loaded_state_is_ddp and current_model_is_ddp:
            # If loading non-DDP state into DDP model, add "module." prefix
            state = {f"module.{k}": v for k, v in state.items()}
        
        # Get actual model instance and load state
        actual_model = model.module if current_model_is_ddp else model
        actual_model.load_state_dict(state)

        # Restore global step
        self.global_step = loaded_state["global_step"]
