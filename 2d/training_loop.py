import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from losses import improved_negative_log_likelihood,diffusion_regression_loss,denoiser_loss
from utils import get_max_grad, log_max_grad, add_nan_hooks
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import save_config, load_config
import random,math
from matplotlib.colors import LinearSegmentedColormap


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


class FlowMatchingDataset(Dataset):
    def __init__(
        self,
        trajectories=None,
        denoising=None,
        noise_std_denoiser=None,
        noise_std_flow=None,
        append_time=None,
        time_gap=None,
        use_hermite_spline=None,
        multi_noise_denoiser=None,
        multi_noise_flow=None,
        state_shape=None
    ):
        """
        A PyTorch Dataset for flow matching with optional denoising and time appending.

        Args:
            trajectories (list of np.ndarray): List of 2D arrays, where each array represents a trajectory.
                Each trajectory should have shape (num_samples, num_features), where:
                - num_samples is the number of time steps in the trajectory.
                - num_features is the number of features at each time step, including time and spatial dimensions.
                  The first feature should be the time (t), and the remaining features should be the spatial dimensions (e.g., x, y, z, ...).
            denoising (bool): Whether to apply denoising to the input data.
            noise_std (float): Standard deviation of noise for denoising.
            append_time (bool): Whether to append a time index to the first dimension of each trajectory.
            time_gap (float): The gap between consecutive time steps if append_time is True.

        Example:
            If you have a trajectory with time (t), and spatial dimensions (x, y, z), the shape of each trajectory should be (num_samples, 4).
            For instance:
                trajectory = np.array([
                    [0.0, 1.0, 2.0, 3.0],  # t, x, y, z at time step 0
                    [0.1, 1.1, 2.1, 3.1],  # t, x, y, z at time step 1
                    ...
                ])

        Attributes:
            trajectories (list of np.ndarray): List of input trajectories.
            denoising (bool): Whether denoising is applied.
            noise_std (float): Standard deviation of noise for denoising.
            append_time (bool): Whether a time index is appended.
            time_gap (float): The gap between consecutive time steps.
            data (list of tuples): Flattened list of trajectory segments, where each segment is a tuple of (start_point, end_point).
        """
        self.trajectories = trajectories.copy()  # List of trajectories
        self.denoising = denoising
        self.noise_std_denoiser = noise_std_denoiser
        self.noise_std_flow = noise_std_flow
        self.append_time = append_time
        self.time_gap = time_gap
        self.data = []  # Flattened list of all trajectory segments
        self.use_hermite_spline = use_hermite_spline
        self.multi_noise_denoiser = multi_noise_denoiser
        self.multi_noise_flow = multi_noise_flow
        self.state_shape = state_shape
        # Append time index if required
        if self.append_time:
            for i in range(len(self.trajectories)):
                num_samples = self.trajectories[i].shape[0]
                time_index = np.linspace(
                    0, (num_samples - 1) * self.time_gap, num_samples
                ).reshape(-1, 1)
                self.trajectories[i] = np.hstack((time_index, self.trajectories[i]))

        # Flatten the trajectories into a single dataset
        for i, trajectory in enumerate(self.trajectories):
            for idx in range(len(trajectory) - 1):
                self.data.append((trajectory[idx], trajectory[idx + 1], i))
        if self.use_hermite_spline:
            self.hermite_splines = [PchipInterpolator(traj[:, 0], traj[:, 1:]) for traj in self.trajectories]
        #shuffle
        random.shuffle(self.data)
                

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        start_point, end_point, traj_index = self.data[idx]  # Correct unpacking
        t_start = start_point[0]
        t_end = end_point[0]
        t_rand = np.random.uniform(t_start, t_end)
        pos_start = start_point[1:]
        pos_end = end_point[1:]

        if self.use_hermite_spline:
            pos_interp = self.hermite_splines[traj_index](t_rand)
            dpos_dt = self.hermite_splines[traj_index].derivative()(t_rand)
        else:
            pos_interp = interpolate_position(t_rand, t_start, t_end, pos_start, pos_end)
            dpos_dt = (pos_end - pos_start) / (t_end - t_start)

        delta_t = t_end - t_start  # Dynamically computed delta_t
        assert delta_t >= 1e-4, f"Delta t is too small: {delta_t}"

        clean_data = torch.tensor(pos_interp, dtype=torch.float32)
        target_data = torch.tensor(dpos_dt, dtype=torch.float32)
        delta_t = torch.tensor(delta_t, dtype=torch.float32)  # Return delta_t as a tensor

        if self.multi_noise_denoiser:
            noise_std_denoiser = torch.rand(1) * self.noise_std_denoiser
            # noise_std_denoiser = torch.exp(
            #     torch.rand(1) * torch.log(torch.tensor(1.0e8))
            # ) * 1.0e-6  
        else:
            noise_std_denoiser = self.noise_std_denoiser

        if self.multi_noise_flow:
            noise_std_flow = torch.rand(1) * self.noise_std_flow
        else:
            noise_std_flow = self.noise_std_flow

            noise_denoiser = torch.randn_like(clean_data) * noise_std_denoiser
            noise_flow = torch.randn_like(clean_data) * noise_std_flow
            noisy_data_denoiser = clean_data + noise_denoiser
            noisy_data_flow = clean_data + noise_flow
            #noise= noisy_data - clean_data
            slice=math.prod(self.state_shape[1:])
            return noisy_data_denoiser, noisy_data_flow, clean_data, target_data[-slice:], -noise_denoiser, delta_t,noise_std_denoiser

def training_loop(
    model_type,  # flow, denoiser and diffusion
    model,
    config,
    train_dataloader,
    val_dataloader,
    device,
    num_epochs,
    checkpoint_interval,
    log_dir,
    mixed_precision,
    grad_clip_value,
    debug_nan,
    start_epoch=0
):
    """Generic training function for flow and denoiser models
    
    Args:
        model_type: Type of model being trained ('flow' or 'denoiser')
        model: The model to train
        config: Training configuration dictionary
        train_dataloader: Training data loader
        val_dataloader: Validation data loader 
        device: Device to train on
        num_epochs: Number of epochs to train
        checkpoint_interval: Interval for saving checkpoints
        log_dir: Directory for logs and checkpoints
        mixed_precision: Whether to use mixed precision training
        grad_clip_value: Value for gradient clipping
        debug_nan: Whether to debug NaN values
        start_epoch: Starting epoch number
        
    Returns:
        Trained model
    """
    model = model.to(device)
    start_epoch = 0
    
    # Resume if needed
    if config["training"][model_type]["resume"]:
        model_path = f"{config['training'][model_type]['resume_run_path']}/{model_type}_epoch_{config['training'][model_type]['resume_epoch']}.pth"
        model = torch.load(model_path)
        model.to(device)
        start_epoch = config["training"][model_type]["resume_epoch"]

    if debug_nan:
        add_nan_hooks(model)    

    # Initialize optimizer
    if config["training"][model_type]["optimizer"] == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["training"][model_type]["learning_rate"],
            weight_decay=config["training"][model_type]["l2_reg"],
            momentum=0.9
        )
    elif config["training"][model_type]["optimizer"] == "adam":
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config["training"][model_type]["learning_rate"],
            weight_decay=config["training"][model_type]["l2_reg"]
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['training'][model_type]['optimizer']}")

    # Initialize scheduler
    scheduler = None
    if config["training"][model_type]["use_learning_schedule"]:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config["training"][model_type]["factor"],
            patience=config["training"][model_type]["patience"],
            verbose=True,
            threshold=1.0e-4,
            min_lr=1.0e-6 if model_type=="diffusion" else 1.0e-5
        )

    flow_model = None
    if model_type == "diffusion":
        flow_path = config["training"]["diffusion"]["flow_path"]
        if flow_path:
            flow_dir = os.path.dirname(flow_path)
            flow_config_path = os.path.join(flow_dir, "config.yaml")
            flow_config = load_config(flow_config_path)
            flow_model = torch.load(flow_path)
            flow_model.to(device)
            flow_model.eval()
        else:
            raise ValueError("Flow path is not specified")

    model_config = {"dataset": config["dataset"], 
                    "training": {
                        "batch_size": config["training"]["batch_size"],
                        "checkpoint_interval": config["training"]["checkpoint_interval"],
                        "seed": config["training"]["seed"],
                        "num_epochs": config["training"]["num_epochs"],
                        "save_model": config["training"]["save_model"],
                        "mixed_precision": config["training"]["mixed_precision"],
                        "log_dir_base": config["training"]["log_dir_base"],
                        model_type: config["training"][model_type]
                    }}

    os.makedirs(log_dir, exist_ok=True)
    save_config(model_config, log_dir)
    # Initialize writer and scaler
    writer = SummaryWriter(log_dir=log_dir)
    scaler = GradScaler() if mixed_precision else None

    optuna_score = 0.0
    for epoch in range(start_epoch, num_epochs):
        model.train()
        
        # Training metrics
        train_loss = 0.0
        train_max_error = 0.0
        max_grad = 0.0

        for batch in train_dataloader:
            optimizer.zero_grad()
            
            # Process batch data
            noisy_data_denoiser, noisy_data_flow, clean_data, target_data, noise_denoiser, delta_t_tensor, noise_std_denoiser = [
                x.to(device) for x in batch
            ]

            # Forward pass based on model type
            with autocast(enabled=mixed_precision):
                if model_type == "flow":
                    pred = model(clean_data if model_config["training"]["flow"]["train_on_clean_data"] else noisy_data_flow)
                    loss, max_error = improved_negative_log_likelihood(
                        pred,
                        target_data,
                        model_config["training"]["flow"]["loss_fn"],
                        model_config["training"]["flow"]["desingularization"]
                    )
                elif model_type == "denoiser":  # denoiser
                    pred = model(noisy_data_denoiser)
                    loss, max_error = denoiser_loss(pred, noise_denoiser)
                elif model_type == "diffusion":  # diffusion
                    log_g_pred = model(clean_data if flow_config["training"]["flow"]["train_on_clean_data"] else noisy_data_flow)
                    f_pred=flow_model(clean_data if model_config["training"]["diffusion"]["train_on_clean_data"] else noisy_data_flow).detach()
                    loss, max_error = diffusion_regression_loss(log_g_pred, f_pred, target_data, delta_t_tensor)

                train_loss += loss.item()
                train_max_error += max_error.item()

            # Backward pass and optimization
            if mixed_precision:
                with torch.autograd.set_detect_anomaly(debug_nan):
                    scaler.scale(loss).backward()
                if grad_clip_value > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                scaler.step(optimizer)
                scaler.update()
            else:
                with torch.autograd.set_detect_anomaly(debug_nan):
                    loss.backward()
                if grad_clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                optimizer.step()

            max_grad = max(max_grad, get_max_grad(model))

        # Log gradients
        log_max_grad(writer, model, f"{model_type.capitalize()}", epoch, max_grad)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_max_error = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                noisy_data_denoiser, noisy_data_flow, clean_data, target_data, noise_denoiser, delta_t_tensor, noise_std_denoiser = [
                    x.to(device) for x in batch
                ]

                if model_type == "flow":
                    pred = model(clean_data if model_config["training"]["flow"]["train_on_clean_data"] else noisy_data_flow)
                    loss, max_error = improved_negative_log_likelihood(
                        pred,
                        target_data,
                        model_config["training"]["flow"]["loss_fn"],
                        model_config["training"]["flow"]["desingularization"]
                    )
                    max_error = torch.max(torch.abs(pred - target_data))
                elif model_type == "denoiser":
                    pred = model(noisy_data_denoiser)
                    loss = nn.MSELoss()(pred, noise_denoiser)
                    max_error = torch.max(torch.abs(pred - noise_denoiser))
                elif model_type == "diffusion":
                    log_g_pred = model(clean_data if flow_config["training"]["flow"]["train_on_clean_data"] else noisy_data_flow)
                    f_pred=flow_model(clean_data if model_config["training"]["diffusion"]["train_on_clean_data"] else noisy_data_flow).detach()
                    loss, max_error = diffusion_regression_loss(log_g_pred, f_pred, target_data, delta_t_tensor)

                val_loss += loss.item()
                val_max_error += max_error.item()

        # Calculate averages
        avg_train_loss = train_loss / len(train_dataloader)
        avg_train_max_error = train_max_error / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_max_error = val_max_error / len(val_dataloader)

        # Update scheduler
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # Logging
        writer.add_scalar(f"{model_type}/LR", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar(f"{model_type}/Train/Loss", avg_train_loss, epoch)
        writer.add_scalar(f"{model_type}/Train/Max_Error", avg_train_max_error, epoch)
        writer.add_scalar(f"{model_type}/Val/Loss", avg_val_loss, epoch)
        writer.add_scalar(f"{model_type}/Val/Max_Error", avg_val_max_error, epoch)

        # Print progress
        if (epoch + 1) % 1 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"{model_type}/Train/Loss: {avg_train_loss:.7f}, "
                f"{model_type}/Train/Max_Error: {avg_train_max_error:.7f}, "
                f"{model_type}/Val/Loss: {avg_val_loss:.7f}, "
                f"{model_type}/Val/Max_Error: {avg_val_max_error:.7f}"
            )

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save(
                model,
                os.path.join(log_dir, f"{model_type}_ep_{epoch + 1}.pth"),
            )
            print(f"{model_type} checkpoint saved at epoch {epoch + 1}")

        # Calculate metrics every 50 epochs
        if (epoch + 1) % 20 == 0:
            optuna_score = calculate_and_log_metrics(model, model_type, train_dataloader, device, writer, epoch, config, flow_model)

    model.eval()
    writer.close()
    return model, optuna_score

def calculate_and_log_metrics(model, model_type, dataloader, device, writer, epoch, config, flow_model=None):
    """Calculate and log additional metrics for model evaluation"""
    with torch.no_grad():
        score = 0.0
        mse = 0.0
        percentile = 0.0
        max_error = 0.0
        
        for batch in dataloader:
            noisy_data_denoiser, noisy_data_flow, clean_data, target_data, noise_denoiser, delta_t_tensor, noise_std_denoiser = [
                x.to(device) for x in batch
            ]

            if model_type == "flow":
                pred = model(clean_data if config["training"]["flow"]["train_on_clean_data"] else noisy_data_flow)
                target = target_data
            elif model_type == "denoiser":
                pred = model(noisy_data_denoiser)
                target = noise_denoiser
            elif model_type == "diffusion":
                pred = model(clean_data if config["training"]["flow"]["train_on_clean_data"] else noisy_data_flow)
                f_pred=flow_model(clean_data if config["training"]["diffusion"]["train_on_clean_data"] else noisy_data_flow).detach()
                delta_t_tensor = delta_t_tensor.view(-1, 1).expand_as(target_data)
                target = 0.5 * torch.log((target_data - f_pred.detach()) ** 2 + 1e-6) + 0.5 * torch.log(delta_t_tensor)

            mse += nn.MSELoss()(target, pred).item()
            error = torch.abs(target - pred)
            error_flat = error.view(error.shape[0], -1)
            top_50_errors, _ = torch.topk(error_flat, k=min(50, error_flat.shape[1]), dim=1)
            top_1_errors, _ = torch.topk(error_flat, k=min(1, error_flat.shape[1]), dim=1)
            max_error += top_1_errors.mean().item()
            percentile += top_50_errors.mean().item()

        # Calculate averages
        mse /= len(dataloader)
        percentile /= len(dataloader)
        max_error /= len(dataloader)
        score = percentile * mse

        # Log metrics
        print(f"At epoch {epoch + 1}, {model_type}_score: {score}, {model_type}_mse: {mse}, "
              f"{model_type}_percentile: {percentile}, {model_type}_max_error: {max_error}")
        writer.add_scalar(f"{model_type}/Optuna/Score", score, epoch)
        writer.add_scalar(f"{model_type}/Optuna/MSE", mse, epoch)
        writer.add_scalar(f"{model_type}/Optuna/Percentile", percentile, epoch)
        writer.add_scalar(f"{model_type}/Optuna/Max_Error", max_error, epoch)
    return score

def train_model(
    train_dataset,
    val_dataset,
    flow_backbone,
    denoiser_backbone,
    diffusion_backbone,
    num_epochs=100,
    batch_size=32,
    checkpoint_interval=20,
    log_dir=None,  # Directory to save models and logs
    mixed_precision=False,
    grad_clip_value=1.0,  # Gradient clipping value
    debug_nan=False,
    config=None,
):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
    )

    flow=None
    denoiser=None
    diffusion=None
    flow_optuna_score = None
    denoiser_optuna_score = None
    diffusion_optuna_score = None
    f_dir, d_dir, g_dir = log_dir
    # Train flow model
    if config["training"]["flow"]["train_flow"]:

            
        flow, flow_optuna_score = training_loop(
            model_type="flow",
            model=flow_backbone,
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader, 
            device=device,
            num_epochs=num_epochs,
            checkpoint_interval=checkpoint_interval,
            log_dir=f_dir,
            mixed_precision=mixed_precision,
            grad_clip_value=grad_clip_value,
            debug_nan=debug_nan,
        )

    # Train denoiser model 
    if config["training"]["denoiser"]["train_denoiser"]:
        
        denoiser, denoiser_optuna_score = training_loop(
            model_type="denoiser",
            model=denoiser_backbone,
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader, 
            device=device,
            num_epochs=num_epochs,
            checkpoint_interval=checkpoint_interval,
            log_dir=d_dir,
            mixed_precision=mixed_precision,
            grad_clip_value=grad_clip_value,
            debug_nan=debug_nan,
        )
    
    #Train diffusion model
    if config["training"]["diffusion"]["train_diffusion"]:
        diffusion, diffusion_optuna_score= training_loop(
            model_type="diffusion",
            model=diffusion_backbone,
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader, 
            device=device,
            num_epochs=num_epochs,
            checkpoint_interval=checkpoint_interval,
            log_dir=g_dir,
            mixed_precision=mixed_precision,
            grad_clip_value=grad_clip_value,
            debug_nan=debug_nan,
        )




    return flow,flow_optuna_score,denoiser,denoiser_optuna_score,diffusion,diffusion_optuna_score

def visualize_field_canvas(
    model,
    ax,  # Existing matplotlib axes object
    diffusion_model=None,
    denoiser=None,
    grid_density=20,  # Control grid density across the canvas
    flow_color="red",
    denoiser_color="green",
    arrow_scale=0.2,  # Scale factor for arrow size
    arrow_alpha=0.3,  # Transparency of arrows
    denoising_magnitude=1.0,
    noise_std=0.05,
    show_flow=True,
    show_denoise=True,
    show_diffusion=True,
    show_composite=False,
):
    """Visualize flow field and denoiser field across the entire canvas
    
    Args:
        model: Trained flow model
        ax: Matplotlib axes object for plotting
        diffusion_model: Diffusion model for uncertainty estimation
        denoiser: Denoising model
        grid_density: Number of grid points in each dimension
        flow_color: Color for flow field arrows
        denoiser_color: Color for denoiser field arrows
        arrow_scale: Scaling factor for arrow sizes
        arrow_alpha: Transparency level for arrows
        denoising_magnitude: Magnitude of denoising effect
        noise_std: Standard deviation of noise
    """
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if diffusion_model:
        diffusion_model = diffusion_model.to(device)
    if denoiser:
        denoiser = denoiser.to(device)

    # Get current canvas bounds
    x_min, x_max = -0.5, 10.0
    y_min, y_max = -3.0, 3.0
    #extend the canvas bounds
    # x_min, x_max = x_min*1.2, x_max*1.2
    # y_min, y_max = y_min*1.2, y_max*1.2
    
    # Create uniform grid across the canvas
    x_grid = np.linspace(x_min, x_max, grid_density)
    y_grid = np.linspace(y_min, y_max, grid_density)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    x_grid_diffusion = np.linspace(x_min*10, x_max*10, grid_density*10)
    y_grid_diffusion = np.linspace(y_min*10, y_max*10, grid_density*10)
    X_grid_diffusion, Y_grid_diffusion = np.meshgrid(x_grid_diffusion, y_grid_diffusion)
    diffusion_field = np.zeros((grid_density*10, grid_density*10))
    if diffusion_model and show_diffusion:
        for i in range(grid_density*10):
            for j in range(grid_density*10):
                grid_coord = np.array([X_grid_diffusion[i, j], Y_grid_diffusion[i, j]])
                X_tensor = torch.tensor(grid_coord, dtype=torch.float32).to(device)
                
                diffusion_std = torch.exp(diffusion_model(X_tensor)).detach().cpu().numpy()
                diffusion_field[i, j] = math.prod(diffusion_std)
                # # Create an ellipse for each point with width and height proportional to the std deviation
                # ellipse = patches.Ellipse(
                #     (X_grid[i, j], Y_grid[i, j]),
                #     width=50 * diffusion_std[0],  # 2 * std for width
                #     height=50 * diffusion_std[1],  # 2 * std for height
                #     edgecolor="purple",
                #     facecolor="none",
                #     linestyle="--",
                # )
                # ax.add_patch(ellipse)
        colors = [
            (255/255, 255/255, 255/255),      # White
            (252/255, 250/255, 237/255),      # Light beige
            (255/255, 133/255, 118/255),      # Pink
            (255/255, 30/255, 54/255),        # Deep red
        ]
        n_bins = 1000  # Number of color steps for smooth gradient
        custom_cmap = LinearSegmentedColormap.from_list('custom_gradient', colors, N=n_bins)

        im = ax.pcolormesh(X_grid_diffusion, Y_grid_diffusion, diffusion_field, 
                           cmap=custom_cmap,
                           alpha=1.0,
                           shading='gouraud')
    # Draw flow and denoiser fields at each grid point
    for i in range(grid_density):
        for j in range(grid_density):
            grid_coord = np.array([X_grid[i, j], Y_grid[i, j]])
            X_tensor = torch.tensor(grid_coord, dtype=torch.float32).to(device)
            
            # # Calculate and draw flow field
            flow = model(X_tensor).detach().cpu().numpy()
            if show_flow:
                ax.arrow(
                    X_grid[i, j],
                    Y_grid[i, j],
                    flow[0] * arrow_scale*2.0,
                    flow[1] * arrow_scale*2.0,
                    color=flow_color,
                    head_width=0.04,
                    head_length=0.06,
                    width=0.002,
                    alpha=arrow_alpha,
                )
            
            # Calculate and draw denoiser field if available
            if denoiser:
                # Calculate denoising coefficient using diffusion model
                if denoising_magnitude == -1:
                    denoising_coefficient = 50*50*torch.exp(diffusion_model(X_tensor))**2
                else:
                    denoising_coefficient = denoising_magnitude
                
                # Get denoising direction and scale by coefficient
                denoise_correction = (denoising_coefficient * denoiser(X_tensor)).detach().cpu().numpy()
            
            if show_denoise:
                # Draw denoiser arrow
                ax.arrow(
                    X_grid[i, j],
                    Y_grid[i, j],
                    denoise_correction[0] * arrow_scale*0.3,
                    denoise_correction[1] * arrow_scale*0.3,
                    color=denoiser_color,
                    head_width=0.04,
                    head_length=0.06,
                    width=0.002,
                    alpha=arrow_alpha,
                )


            if show_composite:
                flow += denoise_correction
                ax.arrow(
                    X_grid[i, j],
                    Y_grid[i, j],
                    flow[0] * arrow_scale*0.5,
                    flow[1] * arrow_scale*0.5,
                    color='brown',
                    head_width=0.04,
                    head_length=0.06,
                    width=0.002,
                    alpha=arrow_alpha,
                )

    return ax