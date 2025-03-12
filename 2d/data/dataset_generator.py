import os
import numpy as np
import matplotlib.pyplot as plt
import random,math
import torch
import torchsde
#get current directory path
current_dir_path = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory path: {current_dir_path}")

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DoubleSDE(torchsde.SDEIto):
    def __init__(self, 
                 N1: int, N2: int, N3: int, 
                 Delta_t: float, 
                 theta: float,
                 nu: float,
                 threshold: float,
                 mode: str=None,
                 eps_y: float=None,
                 smooth: bool=False,
                 g_constant: float=0.0
):
        """
        SDE with piecewise-defined drift and Gaussian diffusion.

        Args:
            N1, N2, N3: Integers defining the length of each stage.
            Delta_t: Time step size.
            nu: Base velocity magnitude.
            mode: Which velocity profile to use in stage 2 and stage 3  (0, 1, or 2).
        """
        super().__init__(noise_type="diagonal")
        self.N1 = N1
        self.N2 = N2
        self.N3 = N3
        self.Delta_t = Delta_t
        self.nu = nu

        # Precompute mean and std for the Gaussian in g:
        self.mu = (N1 + N2/2)*Delta_t
        self.sigma = (N2/2)*Delta_t/2.0

        # Final angle after stage 2:
        # Stage 2 runs from t = N1*Delta_t to t = N1*Delta_t + N2*Delta_t.
        # degree to radian
        self.theta=theta*math.pi/180
        self.w=self.theta/(self.N2*self.Delta_t)

        self.threshold=threshold
        self.mode=mode
        self.eps_y=eps_y
        self.smooth=smooth
        self.g_constant=g_constant

    def f(self, t, X):
        x = X[..., 0]
        y = X[..., 1]

        t = t.to(X.device)
        stage1_end = self.N1*self.Delta_t
        stage2_end = self.N1*self.Delta_t + self.N2*self.Delta_t

        # Initialize tensors instead of float values
        fx = torch.zeros_like(x)
        fy = torch.zeros_like(y)

        if t < stage1_end:
            # First stage
            fx = torch.full_like(x, self.nu)  # Use full_like instead of direct float assignment
            fy = torch.zeros_like(y)
        elif t < stage2_end and self.smooth:
            # Second stage
            if self.mode=="0":
                fx = torch.full_like(x, self.nu)
                fy = torch.zeros_like(y)
            elif self.mode=="1":
                fx = self.nu-self.w*(y-self.eps_y) * torch.ones_like(x)
                fy = self.w*(x-self.nu*self.N1*self.Delta_t) * torch.ones_like(y)
            elif self.mode=="-1":
                fx = self.nu-self.w*(y-self.eps_y) * torch.ones_like(x)
                fy = -self.w*(x-self.nu*self.N1*self.Delta_t) * torch.ones_like(y)
        else:
            # Third stage
            if self.mode=="0":
                fx = torch.full_like(x, self.nu)
                fy = torch.zeros_like(y)
            elif self.mode=="1":
                fx = torch.full_like(x, self.nu * math.cos(self.theta))
                fy = torch.full_like(y, self.nu * math.sin(self.theta))
            elif self.mode=="-1":
                fx = torch.full_like(x, self.nu * math.cos(self.theta))
                fy = torch.full_like(y, -self.nu * math.sin(self.theta))

        return torch.stack((fx, fy), dim=-1)

    def g(self, t, X):
        return torch.ones_like(X)*self.g_constant

def double_sde(g_constant=0.01):
    # Parameters (example)
    N1, N2, N3 = 45, 10, 45
    Delta_t = 1.0
    nu = 0.1
    threshold=0.2
    sde = DoubleSDE(N1, N2, N3, Delta_t, 30,nu, threshold, mode=None,smooth=False,g_constant=g_constant)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sde.to(device)

    # Solve the SDE for one trajectory:
    ts = torch.linspace(0, (N1+N2+N3)*Delta_t, (N1+N2+N3)+ 1).to(device)
    # Generate 100 trajectories
    num_trajectories = 50
    all_trajectories = []
    
    for mode in ["1", "-1"]:
        for i in range(num_trajectories):
            # Add some random variation to initial positions for each trajectory
            #eps_y=np.random.uniform(-threshold, threshold)
            eps_y=0.0
            x0 = torch.tensor([[0.0 , 0.0 + eps_y]], device=device)
            sde.mode=mode
            sde.eps_y=eps_y
            trajectory = torchsde.sdeint(sde, x0, ts, method="milstein", 
                                    adaptive=True, atol=1e-4, rtol=1e-10, dt_min=1e-10)
            traj_np = trajectory.squeeze(1).cpu().detach().numpy()
            all_trajectories.append(traj_np)
    return np.array(all_trajectories)


if __name__ == "__main__":
    set_seed(2)
    all_trajectories = double_sde(g_constant=0.0)

    # Plot all trajectories
    plt.figure(figsize=(10, 8))
    for traj in all_trajectories:
        plt.plot(traj[:,0], traj[:,1], alpha=0.5)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Double ODE trajectories')
    plt.grid(True)
    plt.savefig(os.path.join(current_dir_path, 'double_ode.png'))
    plt.show()