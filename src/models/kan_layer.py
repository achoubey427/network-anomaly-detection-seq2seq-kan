"""
Trainable Kolmogorov-Arnold Network (KAN) Layer Implementation.
Implements learnable univariate B-spline activation functions on network edges:
phi(x) = w_b * silu(x) + w_s * sum_k(c_k * B_k(x))
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network Linear / Bottleneck layer.
    Replaces standard affine transformation Wx + b with learnable B-spline activations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        # Initialize uniform B-spline knot grid
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # Trainable Parameters
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize base weights, spline coefficients, and scalers."""
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            self.spline_weight.data.uniform_(-self.scale_noise, self.scale_noise)
            nn.init.ones_(self.spline_scaler)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Cox-de Boor recursive B-spline basis functions of order `spline_order`.
        x: (batch_size, in_features)
        Returns: (batch_size, in_features, grid_size + spline_order)
        """
        grid: torch.Tensor = self.grid  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)  # (batch_size, in_features, 1)

        # 0-th order basis
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        # Higher order recursive basis
        for k in range(1, self.spline_order + 1):
            denom1 = grid[:, k:-1] - grid[:, : -(k + 1)]
            denom2 = grid[:, k + 1 :] - grid[:, 1:(-k)]
            denom1[denom1 == 0] = 1.0
            denom2[denom2 == 0] = 1.0

            bases = (
                (x - grid[:, : -(k + 1)]) / denom1 * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x) / denom2 * bases[:, :, 1:]
            )

        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: computes linear combination of base activation and B-splines.
        x: (batch_size, in_features)
        Returns: (batch_size, out_features)
        """
        # Base linear activation path: W_b * SiLU(x)
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # Spline activation path
        spline_basis = self.b_splines(x)  # (batch_size, in_features, num_splines)
        # Scaled spline combination
        spline_output = F.linear(
            spline_basis.view(x.size(0), -1),
            (self.spline_weight * self.spline_scaler.unsqueeze(-1)).view(self.out_features, -1),
        )

        return base_output + spline_output
