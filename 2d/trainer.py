import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from training_loop import (
    FlowMatchingDataset,
    train_model,
    visualize_field_canvas
)
from common import generate_trajectory
from models.mlp import FlowMLP, DiffusionMLP, DenoiserMLP
from datasets import get_dataset

from utils import (
    load_config,
    save_config,
    create_log_dir_path,
    set_seed,
    create_eval_log_dir,
)
from functools import partial
import math


def main(config):
    training = args.training
    set_seed(config["training"]["seed"])
 
    if training:
        f_dir, d_dir, g_dir = create_log_dir_path(config)
        # Save a copy of the configuration to the log directory

    # Load the dataset
    dataset = get_dataset(
        env_name=config["dataset"]["name"],
        config=config
    )
    train_trajs = dataset.get_train_data()
    val_trajs = dataset.get_val_data()
    state_shape = dataset.get_state_dim()
    
    time_gap = config["dataset"]["time_gap"]
    
    print("len(train_trajs):", len(train_trajs))
    print("len(val_trajs):", len(val_trajs))
    # Create the FlowMatchingDataset
    train_dataset = FlowMatchingDataset(
        train_trajs,
        noise_std_denoiser=config["training"]["denoiser"]["noise_std"],
        noise_std_flow=config["training"]["flow"]["noise_std"],
        append_time=True,
        time_gap=config["dataset"]["time_gap"],
        use_hermite_spline=config["dataset"]["use_hermite_spline"],
        multi_noise_denoiser=config["training"]["denoiser"]["multi_noise"],
        multi_noise_flow=config["training"]["flow"]["multi_noise"],
        state_shape=state_shape
    )

    val_dataset = FlowMatchingDataset(
        val_trajs,
        noise_std_denoiser=config["training"]["denoiser"]["noise_std"],
        noise_std_flow=config["training"]["flow"]["noise_std"],
        append_time=True,
        time_gap=config["dataset"]["time_gap"],
        use_hermite_spline=config["dataset"]["use_hermite_spline"],
        multi_noise_denoiser=config["training"]["denoiser"]["multi_noise"],
        multi_noise_flow=config["training"]["flow"]["multi_noise"],
        state_shape=state_shape
    )
    # print_flowmatching_dataset_stats(train_dataset)
    # print_flowmatching_dataset_stats(val_dataset)

    if training:
        # Train the SDE model using the pushT dataset
        flow, flow_optuna_score, denoiser, denoiser_optuna_score, diffusion, diffusion_optuna_score = train_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            flow_backbone=FlowMLP(
                input_dim=math.prod(state_shape),
                hidden_dim=config["training"]["flow"]["hidden_dim"],
                num_layers=config["training"]["flow"]["num_layers"],
                norm_type=config["training"]["flow"]["norm_type"],
                state_shape=state_shape
            ),
            denoiser_backbone=DenoiserMLP(
                input_dim=math.prod(state_shape),
                hidden_dim=config["training"]["denoiser"]["hidden_dim"],
                num_layers=config["training"]["denoiser"]["num_layers"],
                norm_type=config["training"]["denoiser"]["norm_type"],
                state_shape=state_shape
            ),
            diffusion_backbone=DiffusionMLP(
                input_dim=math.prod(state_shape),
                hidden_dim=config["training"]["diffusion"]["hidden_dim"],
                num_layers=config["training"]["diffusion"]["num_layers"],
                logit_min=config["training"]["diffusion"]["logit_min"],
                logit_max=config["training"]["diffusion"]["logit_max"],
                norm_type=config["training"]["diffusion"]["norm_type"],
                state_shape=state_shape
            ),

            num_epochs=config["training"]["num_epochs"],
            batch_size=config["training"]["batch_size"],
            checkpoint_interval=config["training"]["checkpoint_interval"],
            log_dir=(f_dir, d_dir, g_dir), 
            mixed_precision=config["training"]["mixed_precision"],
            config=config,
        )

        print("Training completed.")

        if config["training"]["save_model"]:
            # Save the entire models
            if flow is not None:
                torch.save(flow, os.path.join(f_dir, "flow_final.pth"))
                print(f"flow optuna score: {flow_optuna_score}")
            if denoiser is not None:
                torch.save(denoiser, os.path.join(d_dir, "denoiser_final.pth"))
                print(f"denoiser optuna score: {denoiser_optuna_score}")
            if diffusion is not None:
                torch.save(diffusion, os.path.join(g_dir, "diffusion_model_final.pth"))
                print(f"diffusion optuna score: {diffusion_optuna_score}")
            print("Models saved.")
            
        return flow_optuna_score, denoiser_optuna_score, diffusion_optuna_score
    else:
        # Construct the full paths to the model files
        f_path = config["eval"]["flow_path"]
        f_config = load_config(os.path.join(os.path.dirname(f_path), "config.yaml"))
        d_path = config["eval"]["denoiser_path"]
        d_config = load_config(os.path.join(os.path.dirname(d_path), "config.yaml"))
        g_path = config["eval"]["diffusion_path"]
        g_config = load_config(os.path.join(os.path.dirname(g_path), "config.yaml"))
        eval_config = config.copy()
        eval_config["training"]["flow"] = f_config["training"]["flow"]
        eval_config["training"]["denoiser"] = d_config["training"]["denoiser"]
        eval_config["training"]["diffusion"] = g_config["training"]["diffusion"]

        eval_log_dir = create_eval_log_dir(eval_config)
        save_config(eval_config, eval_log_dir)
        # Load the models
        f_model = torch.load(f_path)
        f_model.eval()
        d_model = torch.load(d_path)
        d_model.eval()
        g_model = torch.load(g_path)
        g_model.eval()
        print("Models loaded.")



        visualize_trajectory = eval_config["eval"]["visualize_trajectory"]
        num_trajectories = eval_config["eval"]["num_trajectories"]

        y0 = [traj[0] for traj in val_trajs[:num_trajectories]] 
        y0 = np.array(y0) #shape: [b, nd]

        num_points = eval_config["eval"]["num_points"]

        # Improved visualization of the generated trajectory
    if visualize_trajectory:
        if not (eval_config["dataset"]["name"] in ["movingmnist", "kth"]):
            # Create figure and axes
            fig, ax = plt.subplots(figsize=(21,12))
            
            # Generate trajectory
            generated_trajectory = generate_trajectory(
                f=f_model,
                g=g_model,
                d=d_model,
                y0=y0,
                num_points=num_points,
                state_shape=state_shape,
                rtol=1e-10,
                atol=1e-3,
                denoising_magnitude=eval_config["eval"]["denoising_magnitude"],
                diffusion_magnitide=eval_config["eval"]["diffusion_magnitide"],
                noise_std=eval_config["training"]["denoiser"]["noise_std"],
                delta_t=time_gap,
            )
                
            # Visualize flow and denoiser fields across the entire canvas
            ax = visualize_field_canvas(
                model=f_model,
                ax=ax,
                diffusion_model=g_model,
                denoiser=d_model,
                grid_density=21,
                flow_color="blue",
                denoiser_color="green",
                arrow_scale=1.0,
                arrow_alpha=0.5,
                denoising_magnitude=eval_config["eval"]["denoising_magnitude"],
                noise_std=eval_config["training"]["denoiser"]["noise_std"],
                show_flow=True,
                show_denoise= True,
                show_diffusion= eval_config["eval"]["diffusion_magnitide"]>0.0,
                show_composite=False
            )
            
            # Plot generated trajectory with time-based coloring
            #colors = plt.cm.viridis(np.linspace(0, 1, generated_trajectory.shape[1])) #shape: [batch_size, t,nd]
            colors = plt.cm.plasma(np.linspace(0, 1, generated_trajectory.shape[1])) #shape: [batch_size, t,nd]
            for i in range(generated_trajectory.shape[0]):
                for j in range(generated_trajectory.shape[1]):
                    ax.scatter(
                        generated_trajectory[i, j, -2].cpu().numpy(),
                        generated_trajectory[i, j, -1].cpu().numpy(),
                        color=colors[j],
                        s=4
                    )
            
            # Set plot labels and properties
            ax.set_title("Neural SDE",fontsize=24)
            ax.set_xlabel("x",fontsize=22)
            ax.set_ylabel("y",fontsize=22)
            ax.set_xlim(-0.5, 10.0)  
            ax.set_ylim(-3.0, 3.0)
            ax.set_xticks(np.arange(-0.5, 10.5, 2.0))
            ax.set_yticks(np.arange(-3.0, 3.0, 1.0))
            ax.tick_params(axis='both', which='major', labelsize=20)
            #ax.set_facecolor('#f0f0f0') 
            #ax.legend()
            plt.grid(True)
            plt.savefig(os.path.join(eval_log_dir, f"field_f+d_{eval_config['eval']['denoising_magnitude']}.png"),dpi=300)
            plt.show()
        


def objective(trial, base_config):
    # Define the hyperparameters to optimize
    config = base_config.copy()
    # config["model"]["hidden_dim"] = trial.suggest_int("hidden_dim", 256, 2048)
    # config["model"]["num_layers"] = trial.suggest_int("num_layers", 2, 10)
    config["training"]["flow"]["learning_rate"] = trial.suggest_float("learning_rate_flow", 1e-5, 1e-3,log=True)
    config["training"]["diffusion"]["learning_rate"] = trial.suggest_float("learning_rate_diffusion", 1e-5, 1e-3,log=True)
    config["training"]["denoiser"]["learning_rate"] = trial.suggest_float("learning_rate_denoiser", 1e-5, 1e-3,log=True)
    config["training"]["flow"]["patience"] = trial.suggest_int("patience_flow", 5, 20)
    config["training"]["diffusion"]["patience"] = trial.suggest_int("patience_diffusion", 5, 20)
    config["training"]["denoiser"]["patience"] = trial.suggest_int("patience_denoiser", 5, 20)
    config["training"]["flow"]["factor"] = trial.suggest_float("factor_flow", 0.9, 0.999,log=True)
    config["training"]["diffusion"]["factor"] = trial.suggest_float("factor_diffusion", 0.9, 0.999,log=True)
    config["training"]["denoiser"]["factor"] = trial.suggest_float("factor_denoiser", 0.9, 0.999,log=True)
    config["training"]["batch_size"] = trial.suggest_int("batch_size", 2,128)
    # config["training"]["norm"] = trial.suggest_categorical("norm", ["layer", "batch", "group"])
    #config["training"]["train_on_clean_data"] = trial.suggest_categorical("train_on_clean_data", [True, False])
    config["training"]["flow"]["l2_reg"] = trial.suggest_float("l2_reg_flow", 1e-8, 1e-3,log=True)
    config["training"]["denoiser"]["l2_reg"] = trial.suggest_float("l2_reg_denoiser", 1e-8, 1e-3,log=True)
    config["training"]["diffusion"]["l2_reg"] = trial.suggest_float("l2_reg_diffusion", 1e-8, 1e-3,log=True)
    config["dataset"]["noise_std"] = trial.suggest_float("noise_std", 1e-6, 1e-1,log=True)
    config["training"]["desingularization"]=trial.suggest_float("desingularization", 1e-5, 1e0,log=True)
    config["training"]["loss_fn"] = trial.suggest_categorical("loss_fn", ["mse", "log_mse"])
    if config["training"]["loss_fn"] == "log_mse":
        config["training"]["desingularization"] = trial.suggest_float("desingularization", 1e-4, 1e0,log=True)
    else:
        config["training"]["desingularization"] = None 

    
    # Call the main function with the trial's hyperparameters
    flow_optuna_score, denoiser_optuna_score, diffusion_optuna_score = main(config)
    # Return the metric you want to optimize (e.g., validation loss)
    return flow_optuna_score
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="./config/config.yaml", help="Path to the config file"
    )
    parser.add_argument("--tune", action="store_true", help="Whether to tune hyperparameters")
    parser.add_argument("--training", action="store_true", help="Whether to train the model")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.tune:
        import optuna
        from utils import save_best_params
        study = optuna.create_study(direction="minimize")
        save_best_params_with_config = partial(save_best_params, config=config)
        try:
            study.optimize(lambda trial: objective(trial, config), n_trials=1000, callbacks=[save_best_params_with_config])
            print("Best trial:")
            print(study.best_trial.params)
            save_best_params(study, study.best_trial, config)
        except KeyboardInterrupt:
            print("Optimization stopped early.")
        
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # The best parameters have already been saved to 'best_params.json'
        # by the callback function during optimization
        print("\nBest parameters have been saved to 'best_params.json'")

    else:
        main(config)

