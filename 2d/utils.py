import os
import json
from datetime import datetime
import yaml
import numpy as np
import torch
import random


def get_weight(model):
    size_all_mb = sum(p.numel() for p in model.parameters()) / 1024**2
    return size_all_mb

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config, log_dir):
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as file:
        yaml.dump(config, file)

def print_num_parameters(model, model_name):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters in {model_name}: {num_params}")
    
def create_log_dir_path(config):
    # Create a unique directory name based on the current date and time
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    space="latent_" if config["dataset"]["in_latent_space"] else "original_"

    if (not config["training"]["flow"]["train_flow"]) and (not config["training"]["denoiser"]["train_denoiser"]) and (not config["training"]["diffusion"]["train_diffusion"]):
        raise ValueError("No training model specified")
    
    f_dir = None
    d_dir = None
    g_dir = None
    if config["training"]["flow"]["train_flow"]:
        space_model=space+"f"
        f_dir = os.path.join(config['training']['log_dir_base'], 
                            config['dataset']['name'],
                            space_model,
                            current_time)
    if config["training"]["denoiser"]["train_denoiser"]:
        space_model=space+"d"
        d_dir = os.path.join(config['training']['log_dir_base'], 
                            config['dataset']['name'],
                            space_model,
                            current_time)
    if config["training"]["diffusion"]["train_diffusion"]:
        space_model=space+"g"
        g_dir = os.path.join(config['training']['log_dir_base'], 
                            config['dataset']['name'],
                            space_model,
                            current_time)
    return f_dir, d_dir, g_dir

def create_eval_log_dir(config):
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    eval_log_dir = os.path.join(config['training']['log_dir_base'], 
                            config['dataset']['name'],
                            "eval",
                            current_time)
    os.makedirs(eval_log_dir, exist_ok=True)
    return eval_log_dir
# Append time dimension to the ground truth trajectory
def append_time_to_trajectory(trajectory, time_gap=0.1):
    num_samples = trajectory.shape[0]
    time_index = np.linspace(0, (num_samples - 1) * time_gap, num_samples).reshape(
        -1, 1
    )
    return np.hstack((time_index, trajectory))

def get_max_grad(model):
    max_grad = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            max_grad = max(max_grad, param.grad.abs().max().item())
    return max_grad

def log_max_grad(writer, model, model_name, epoch, max_grad):
    writer.add_scalar(f'Max_Grad/{model_name}', max_grad, epoch)

def check_for_nan(module, input, output):
    if isinstance(output, torch.Tensor):
        if torch.isnan(output).any():
            print(f'NaN detected in output of {module.__class__.__name__}')
    elif isinstance(output, tuple):
        for i, out in enumerate(output):
            if isinstance(out, torch.Tensor) and torch.isnan(out).any():
                print(f'NaN detected in output {i} of {module.__class__.__name__}')

def add_nan_hooks(model):
    for name, module in model.named_modules():
        module.register_forward_hook(check_for_nan)
        
def save_best_params(study, trial, config):
    if study.best_trial.number == trial.number:
        best_params = study.best_params
        best_value = study.best_value
        result = {
            "best_params": best_params,
            "best_value": best_value,
            "trial_number": trial.number
        }
        with open(os.path.join(config["training"]["log_dir_base"], config["dataset"]["name"], "best_params.json"), "w") as f:
            json.dump(result, f, indent=4)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def to_tensor(array, device=None):
    if device is None:
        device = get_device()
    return torch.tensor(array, dtype=torch.float32).to(device)


def print_flowmatching_dataset_stats(dataset):
    print("Size of dataset: ", len(dataset))
    
    # Sample 100 random trajectories
    sample_size = min(100, len(dataset))
    sample_indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    noisy_mins, noisy_maxs = [], []
    clean_mins, clean_maxs = [], []
    target_mins, target_maxs = [], []
    
    noisy_data_list, clean_data_list, target_data_list = [], [], []
    for idx in sample_indices:
        noisy_data, clean_data, target_data, noise, delta_t_tensor = dataset[idx]
        noisy_data_list.append(noisy_data.cpu())
        clean_data_list.append(clean_data.cpu())
        target_data_list.append(target_data.cpu())
    
    noisy_data_stacked = torch.stack(noisy_data_list)
    clean_data_stacked = torch.stack(clean_data_list)
    target_data_stacked = torch.stack(target_data_list)
    
    noisy_mins = torch.min(noisy_data_stacked, dim=0)[0].numpy()
    noisy_maxs = torch.max(noisy_data_stacked, dim=0)[0].numpy()
    clean_mins = torch.min(clean_data_stacked, dim=0)[0].numpy()
    clean_maxs = torch.max(clean_data_stacked, dim=0)[0].numpy()
    target_mins = torch.min(target_data_stacked, dim=0)[0].numpy()
    target_maxs = torch.max(target_data_stacked, dim=0)[0].numpy()
    
    # Convert lists of arrays to 2D numpy arrays
    noisy_mins = np.array(noisy_mins)
    noisy_maxs = np.array(noisy_maxs)
    clean_mins = np.array(clean_mins)
    clean_maxs = np.array(clean_maxs)
    target_mins = np.array(target_mins)
    target_maxs = np.array(target_maxs)

    # Print min and max values for each dimension
    for dim in range(noisy_mins.shape[0]):
        print(f"Dimension {dim}:")
        print(f"  Noisy Data - Min: {noisy_mins[dim]:.4f}, Max: {noisy_maxs[dim]:.4f}")
        print(f"  Clean Data - Min: {clean_mins[dim]:.4f}, Max: {clean_maxs[dim]:.4f}")
        print(f"  Target Data - Min: {target_mins[dim]:.4f}, Max: {target_maxs[dim]:.4f}")