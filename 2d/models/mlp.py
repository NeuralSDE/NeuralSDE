import torch.nn as nn
import torch.nn.init as init
import math
def initialize_weights(module):
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)

class ResidualBlock(nn.Module):
    def __init__(self, dim, norm_type="layer", num_groups=32):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

        if norm_type == "batch":
            self.norm1 = nn.BatchNorm1d(dim)
            self.norm2 = nn.BatchNorm1d(dim)
        elif norm_type == "layer":
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        elif norm_type == "group":
            self.norm1 = nn.GroupNorm(num_groups, dim)
            self.norm2 = nn.GroupNorm(num_groups, dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.norm2(out)
        out += residual
        return self.relu(out)


# Neural network to model f_theta(X)
class FlowMLP(nn.Module):
    def __init__(
        self, input_dim=None, hidden_dim=512, num_layers=4, norm_type="layer", num_groups=32,state_shape=None
    ):
        super(FlowMLP, self).__init__()
        self.num_layers = num_layers
        self.state_shape=state_shape

        # Initial linear layer to project input to hidden dimension
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Residual blocks
        layers = [
            ResidualBlock(hidden_dim, norm_type, num_groups) for _ in range(num_layers)
        ]
        self.network = nn.Sequential(*layers)

        # Output layer to project back to input dimension
        self.output_layer = nn.Linear(hidden_dim, math.prod(state_shape[1:]))

        # Print the number of parameters
        self.print_num_parameters()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.network(x)
        x = self.output_layer(x)
        return x

    def print_num_parameters(self):
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters in MLP: {num_params}")

class DenoiserMLP(FlowMLP):
    def __init__(
        self,
        input_dim=None,
        hidden_dim=512,
        num_layers=4,
        norm_type="layer",
        num_groups=32,
        state_shape=None,
    ):
        super(DenoiserMLP, self).__init__(input_dim, hidden_dim, num_layers, norm_type, num_groups,state_shape)
        self.output_layer = nn.Linear(hidden_dim, math.prod(state_shape))

class DiffusionMLP(FlowMLP):
    def __init__(
        self,
        input_dim=2,
        hidden_dim=512,
        num_layers=4,
        logit_min=-10.0,
        logit_max=2.0,
        norm_type="layer",
        num_groups=32,
        state_shape=None,
    ):
        super(DiffusionMLP, self).__init__(
            input_dim, hidden_dim, num_layers, norm_type, num_groups,state_shape
        )
        self.tanh = nn.Tanh()
        self.logit_min = logit_min
        self.logit_max = logit_max

        # Apply Xavier initialization
        # experiment shows that default init is better
        # self.apply(initialize_weights)

    def forward(self, x):
        # predict the log of the diffusion coefficient, this could be
        # more stable for training when diffusion is small
        x = super(DiffusionMLP, self).forward(x)
        x = self.tanh(x)

        # Rescale from [-1, 1] to [logit_min, logit_max]
        scaled_output = (x + 1) * 0.5 * (self.logit_max - self.logit_min) + self.logit_min


        # Optionally apply exponential if needed for larger dynamic range
        # scaled_output = torch.exp(scaled_output)
        return scaled_output
    
