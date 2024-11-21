import torch
from torch import nn 
from typing import Tuple, Optional

class NeRF(nn.Module):
    # Neural radiance fields module
    def __init__(self, d_input:int=3,
                 n_layers:int=8,
                 d_filter:int=256,
                 skip:Tuple[int]=(4,),
                 d_viewdirs:Optional[int]=None):
        super().__init__()
        self.d_input = d_input  # Dimension of input
        self.skip = skip    # Residual layer
        self.act = nn.functional.relu # Relu
        self.d_viewdirs = d_viewdirs # Dimension of view direction

        # create model layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.d_input, d_filter)] +
             [nn.Linear(d_filter + self.d_input, d_filter) if i in skip \
             else nn.Linear(d_filter, d_filter) for i in range(n_layers -1)]
        )
        # Bottleneck layers
        if self.d_viewdirs is not None:
            # If using viewdirs, split alpha and rgb
            self.alpha_out = nn.Linear(d_filter, 1)
            self.rgb_filters = nn.Linear(d_filter, d_filter)
            self.branch = nn.Linear(d_filter+self.d_viewdirs, d_filter//2) # rgb is function of x and viewdirs
            self.output = nn.Linear(d_filter//2, 3)
        else:
            # If no viewdirs, use simpler output
            self.output = nn.Linear(d_filter, 4)

    # Forward pass with optional view direction
    def forward(self, x:torch.Tensor,
                viewdirs: Optional[torch.Tensor]=None)->torch.Tensor:
        if self.d_viewdirs is None and viewdirs is not  None:
            raise ValueError('Cannot input x_direction if d_viewdirs was not given')
        # Apply forward passup to bottleneck
        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)

        # Apply bottleneck
        if self.d_viewdirs is not None:
            alpha = self.alpha_out(x)

            # Pass through bottleneck to get RGB
            x = self.rgb_filters(x)
            x = torch.concat([x, viewdirs], dim=-1)
            x = self.act(self.branch(x))
            x = self.output(x)

            # concatenate alphas to output
            x = torch.concat([x, alpha], dim=-1)
        else:
            # simple output
            x = self.output(x)
        return x