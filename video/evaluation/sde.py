import torch
import torchsde
from einops import rearrange
class NeuralSDE(torchsde.SDEIto):  # Assuming Ito SDEs are supported by default

    def __init__(
        self,
        flow=None,
        diffusion=None,
        logit_range=None,
        denoiser=None,
        denoising_magnitude=1.0,
        diffusion_magnitide=1.0,
        state_shape=None,
        noise_std=1.0,
        latents=None,
    ):
        super().__init__(noise_type="diagonal")
        self.flow = flow  # The drift model
        self.diffusion = diffusion  # The learned diffusion model
        self.logit_min=logit_range[0] if logit_range is not None else None
        self.logit_max=logit_range[1] if logit_range is not None else None
        self.denoiser = denoiser
        self.denoising_magnitude = denoising_magnitude  # Scaling factor for denoising
        self.diffusion_magnitide = diffusion_magnitide  # Scaling factor for diffusion
        self.noise_std = noise_std
        self.state_shape = state_shape #b,nc,h,w
        self.latents = latents
        self.count=0
        self.prev_t = 0.0
        self.prev_dt = -1.0
    # Drift term: f + c * denoising
    @torch.no_grad()
    def f(self, t: torch.Tensor, X: torch.Tensor):
        # if t<1e-10:
        #     self.prev_t=0.0
        #     self.prev_t = 0.0
        #     self.prev_dt = -1.0
        # if t>self.prev_t+1.0e-10:
        #     self.count+=1
        #     self.prev_dt = t - self.prev_t
        #     self.prev_t = t
            
        


        X=rearrange(X, "b (nc h w) -> b nc h w", b=self.state_shape[0],nc=self.state_shape[1]*self.state_shape[2],h=self.state_shape[3],w=self.state_shape[4])
        flow=None
        if self.flow:
            dy_1 = self.flow(X)
            dy_0=self.latents[:,-self.state_shape[1]+1:]-self.latents[:,-self.state_shape[1]:-1]
            flow=torch.cat([rearrange(dy_0, "b n c h w -> b (n c) h w"),dy_1],dim=1) #shape: [batch_size, num_input_frames*c, h, w]
        print(f"f max:{torch.max(torch.abs(flow)).item()},f min:{torch.min(torch.abs(flow)).item()}")
        if self.denoiser:
            # derivation of the denoising_coefficient involves using fokker-planck equation
            # and Ito's lemma to derive the evolution of logP, and enforcing that the
            # combined diffusion term is negative (probablility concentrate)
            # we need to enforce that self.denoising_magnitude >= 1.0
            denoising_coefficient = torch.tensor(self.denoising_magnitude).to(X.device).to(X.dtype)
            denoise=self.denoiser(X)
            #print(f"d max:{torch.max(torch.abs(denoise)).item()},d min:{torch.min(torch.abs(denoise)).item()}")
            denoise_correction = denoise *denoising_coefficient

            # print("time:",t.item())
            # print("denoise_norm:",torch.norm(denoise, dim=-1, keepdim=True))
            print("d max:",torch.max(torch.abs(denoise_correction)).item(),"d min:",torch.min(torch.abs(denoise_correction)).item())
            # print(f"denoise_divide_flow:{torch.max(torch.abs(denoise_correction/flow)).item(),torch.min(torch.abs(denoise_correction/flow)).item()}")

            flow += denoise_correction
        flow = rearrange(flow, "b nc h w -> b (nc h w)")
        self.count+=1
        return flow

    # Diffusion term: g(X)
    @torch.no_grad()
    def g(self, t: torch.Tensor, X: torch.Tensor):
 
        X=rearrange(X, "b (nc h w) -> b nc h w", b=self.state_shape[0],nc=self.state_shape[1]*self.state_shape[2],h=self.state_shape[3],w=self.state_shape[4])
        diffusion=torch.zeros_like(X)
        if self.diffusion:
            last_frame_diffusion = torch.exp(
                (torch.tanh(self.diffusion(input_latents=X)) + 1.0) * 0.5 * 
                (self.logit_max - self.logit_min) + 
                self.logit_min
            )
            print("g max:",torch.max(torch.abs(last_frame_diffusion)).item(),"g min:",torch.min(torch.abs(last_frame_diffusion)).item())
            diffusion[:, -self.state_shape[2]:] = last_frame_diffusion
        diffusion = rearrange(diffusion, "b nc h w -> b (nc h w)")
        return diffusion  # Return the learned diffusion coefficients
    
