import torch
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce

# functions

def exists(val):
    return val is not None

# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, dim_seq):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Conv1d(dim_seq, dim_seq, 1)
        nn.init.constant_(self.proj.bias, 1.)

    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        gate = self.norm(gate)
        return self.proj(gate) * x

# main classes

def gMLP(
    *,
    num_tokens = None,
    dim,
    depth,
    seq_len,
    ff_mult = 4
):
    dim_ff = dim * ff_mult

    return nn.Sequential(
        nn.Embedding(num_tokens, dim) if exists(num_tokens) else nn.Identity(),
        *[Residual(nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_ff * 2),
            nn.GELU(),
            SpatialGatingUnit(dim_ff, seq_len),
            nn.Linear(dim_ff, dim)
        )) for i in range(depth)]
    )

def gMLPVision(
    *,
    image_size,
    patch_size,
    num_classes,
    dim,
    depth,
    ff_mult = 4,
    channels = 3
):
    dim_ff = dim * ff_mult
    num_patches = (image_size // patch_size) ** 2

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = patch_size, p2 = patch_size),
        nn.Linear(channels * patch_size ** 2, dim),
        *[Residual(nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_ff * 2),
            nn.GELU(),
            SpatialGatingUnit(dim_ff, num_patches),
            nn.Linear(dim_ff, dim)
        )) for i in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n d -> b d', 'mean'),
        nn.Linear(dim, num_classes)
    )
