import torch
import torch.nn as nn


# Updated negative log-likelihood function for the improved training strategy
def improved_negative_log_likelihood(f_pred, dx_dt, loss_fn="log_mse", desingularization=1e-4):
    """
    Compute the log-squared error objective for training the flow network.

    This loss function is derived from the original negative log-likelihood (NLL) of a
    Stochastic Differential Equation (SDE) model, where the drift function (f_pred)
    and the diffusion function (g_pred) are trained together.

    Derivation:
    1. The original NLL can be expressed as:
       NLL = sum(((dx/dt - f_pred) ** 2 / (2 * g_pred ** 2) + 0.5 * log(2 * pi * g_pred ** 2)) * delta_t)

    2. By analytically minimizing the NLL with respect to g_pred, we find that the
       optimal g_pred is given by:
       g_pred ** 2 = (dx/dt - f_pred) ** 2 * delta_t

    3. Substituting this optimal g_pred back into the NLL and removing constant terms,
       the resulting objective simplifies to:
       J(f_pred) = log((dx/dt - f_pred) ** 2)

    This new objective focuses solely on minimizing the difference between the observed
    trajectory (dx/dt) and the predicted drift (f_pred), without explicitly training the
    diffusion term. The log term helps in reducing the impact of large errors, promoting
    smoother convergence.

    Args:
        f_pred (torch.Tensor): The predicted drift values (f(x)).
        dx_dt (torch.Tensor): The observed trajectory finite-difference values.

    Returns:
        torch.Tensor: The computed log-squared error loss for training the flow network.
    """

    max_error = torch.max(torch.abs(dx_dt - f_pred))

    if loss_fn == "log_mse":
        log_sq_error = torch.sum(
            torch.log(torch.square(dx_dt - f_pred) + desingularization), dim=tuple(range(1, dx_dt.dim()))
        )/dx_dt.shape[1]
        assert (
            log_sq_error.dim() == 1
        ), f"log_sq_error should have a batch dimension, but got shape {log_sq_error.shape}"
        return torch.mean(log_sq_error), max_error

    elif loss_fn == "mse":
        sq_error = nn.MSELoss()(dx_dt, f_pred)
        return sq_error, max_error


def diffusion_regression_loss(log_g_pred, f_pred, dx_dt, delta_t):
    """
    Compute the regression loss for the diffusion network.
    g_pred^2 should match (dx/dt - f)^2 * delta_t,
    -> log(g_pred) should match 0.5 * log((dx/dt - f)^2) + 0.5 * log(delta_t)
    """
    # Detach f_pred to prevent backpropagation through f_theta
    delta_t_expanded = delta_t.view(-1, 1).expand_as(dx_dt)
    target_g = torch.sqrt((dx_dt - f_pred.detach()) ** 2 * delta_t_expanded)
    target_log_g = 0.5 * torch.log(
        (dx_dt - f_pred.detach()) ** 2 + 1e-6
    ) + 0.5 * torch.log(delta_t_expanded)
   
    loss = nn.MSELoss()(log_g_pred, target_log_g)+ nn.MSELoss()(torch.exp(log_g_pred), target_g)
    max_error = torch.max(torch.abs(log_g_pred - target_log_g))

    return loss, max_error

def denoiser_loss(denoise_pred, noise_target):
    loss = nn.MSELoss()(denoise_pred, noise_target)
    max_error = torch.max(torch.abs(denoise_pred - noise_target))
    return loss, max_error
