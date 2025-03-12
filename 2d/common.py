import torch
import torchsde
from scipy.integrate import solve_ivp
from utils import get_device, to_tensor, to_numpy
import matplotlib.pyplot as plt
import numpy as np
import os,math
from tqdm import tqdm
from einops import rearrange

class NeuralSDE(torchsde.SDEIto): 
    def __init__(
        self,
        f=None,
        g=None,
        d=None,
        denoising_magnitude=None,
        diffusion_magnitide=None,
        state_shape=None,
        observations=None,
        noise_std=None,
        delta_t=None,
    ):
        super().__init__(noise_type="diagonal")
        self.flow = f  # The drift model
        self.diffusion = g  # The diffusion model
        self.denoiser = d  # The denoising model
        self.denoising_magnitude = denoising_magnitude  # Scaling factor for denoising
        self.diffusion_magnitide = diffusion_magnitide
        self.noise_std = noise_std
        self.call_count = 0
        self.state_shape = state_shape # (n, c, h, w) or (n, d)
        self.observations = observations # (b, t,nchw ) or (b, t, nd)
        self.delta_t=delta_t
    # Drift term: f + c * denoising
    @torch.no_grad()
    def f(self, t, X):
        with torch.no_grad():
            dx_1 = self.flow(X)
            if dx_1.shape!=X.shape:
                dx_0 = (self.observations[:,-1,(-self.state_shape[0]+1)*self.state_shape[1]:] 
                    - self.observations[:,-1,(-self.state_shape[0])*self.state_shape[1]:-self.state_shape[1]])
                dx_0=dx_0/self.delta_t
                flow=torch.cat([dx_0,dx_1],dim=1) #shape: [n,d ]
            else:
                flow=dx_1
            #tqdm.write(f"t: {t.item():.2f}, fx:{flow[:, -2:].mean().item():.3f}, fy: {flow[:, -1:].mean().item():.3f}")
            if self.denoiser:
                # derivation of the denoising_coefficient involves using fokker-planck equation
                # and Ito's lemma to derive the evolution of logP, and enforcing that the
                # combined diffusion term is negative (probablility concentrate)
                # we need to enforce that self.denoising_magnitude >= 1.0
                denoising_coefficient = self.denoising_magnitude
                #tqdm.write(f"t: {t.item():.2f}, ax:{denoising_coefficient[:, -2:].mean().item():.5f}, ay: {denoising_coefficient[:, -1:].mean().item():.5f}")
                denoise=self.denoiser(X)
                denoise = denoise *denoising_coefficient
                #tqdm.write(f"t: {t.item():.2f}, dx:{denoise[:, -2:].mean().item():.3f}, dy: {denoise[:, -1:].mean().item():.3f}")
                flow = flow + denoise
            self.call_count += 1
            return flow

    # Diffusion term: g(X)
    @torch.no_grad()
    def g(self, t, X):
        with torch.no_grad():
            # coords=X.squeeze(0).detach().cpu().numpy()
            # if coords[0]>2.79 and coords[0]<3.21 and coords[1]>-0.21 and coords[1]<0.21:
            #     return 1.2*torch.ones_like(X)
            # if self.denoising_magnitude != -1:
            #     return torch.zeros_like(X)
            diffusion=torch.zeros_like(X)
            if self.diffusion:
                last_point_diffusion = torch.exp(self.diffusion(X))
                diffusion_coefficient=self.diffusion_magnitide
                #tqdm.write(f"t: {t.item():.2f}, gx:{last_point_diffusion[:, -2:].mean().item():.3f}, gy: {last_point_diffusion[:, -1:].mean().item():.3f}")
                last_point_diffusion=last_point_diffusion*diffusion_coefficient
                #tqdm.write(f"t: {t.item():.2f}, gx:{last_point_diffusion[:, -2:].mean().item():.5f}, gy: {last_point_diffusion[:, -1:].mean().item():.5f}")
                diffusion[:, -math.prod(self.state_shape[1:]):] = last_point_diffusion 
            return diffusion  # Return the learned diffusion coefficients


def generate_trajectory(
    f,
    g,  # Added diffusion model as a parameter
    d,
    y0,
    num_points,
    state_shape,
    rtol=1e-1,
    atol=1e-2,
    denoising_magnitude=None,  # Add denoising magnitude
    diffusion_magnitide=None,
    noise_std=None,
    delta_t=None,

):


    device = get_device()
    f = f.to(device)
    if g:
        g = g.to(device)
    if d:
        d = d.to(device)


    x0 = to_tensor(y0, device)
    observations=x0.unsqueeze(1) #shape: [b, 1, d]
    # Use SDE solver with call counting
    sde_model = NeuralSDE(
        f=f,
        g=g,
        d=d,
        denoising_magnitude=denoising_magnitude,
        diffusion_magnitide=diffusion_magnitide,
        noise_std=noise_std,
        state_shape=state_shape,
        observations=observations,
        delta_t=delta_t,
    ).to(device)
    gen = tqdm(range(num_points), desc="Generating points", disable=False, leave=False)
    for _ in gen:
        x0=observations[:,-1]
        sde_model.observations=observations
        trajectory = torchsde.sdeint(
            sde_model,
            x0,
            ts=torch.linspace(0,delta_t,2).to(x0.device),
            method="milstein",  # 'srk', 'euler', 'milstein', etc.
            dt=delta_t,
            adaptive=True,
            rtol=rtol,
            atol=atol,
            dt_min=1e-10,
        )
        trajectory=rearrange(trajectory,"t b d -> b t d")
        observations=torch.cat([observations,trajectory[:,-1:]],dim=1)


    print(f"Number of function calls: {sde_model.call_count}")
    return observations