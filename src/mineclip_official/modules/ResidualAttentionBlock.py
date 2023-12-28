"""
Adapted from OpenAI CLIP implementation: https://github.com/openai/CLIP
"""
from __future__ import annotations

from collections import OrderedDict
import numpy as np
import torch
from torch import nn

class QuickGELU(nn.Module):
    """
    (Gaussian Error Linear Unit)
    Use Sigmoid instead of tangh. Faster and more memory efficient than GELU.
    https://paperswithcode.com/method/gelu
    """
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    """
    Initializes the ResidualAttentionBlock.

    Args:
        d_model (int): Dimensionality of the input features.
        n_head (int): Number of attention heads.
        attn_mask (torch.Tensor, optional): Attention mask for masking certain positions in attention computation.
    """
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        # Convert attn_mask to the same dtype and device as input tensor
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the ResidualAttentionBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing through the attention block.
        """
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x