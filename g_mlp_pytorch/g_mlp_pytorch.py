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

class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out)

class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, dim_seq, attn_dim = None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Conv1d(dim_seq, dim_seq, 1)
        self.attn = Attention(dim * 2, dim, attn_dim) if exists(attn_dim) else None
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 1.)

    def forward(self, x):
        res, gate = x.chunk(2, dim = -1)
        gate = self.norm(gate)
        gate = self.proj(gate)
        if exists(self.attn):
            gate += self.attn(x)
        return gate * res

# main classes

def gMLP(
    *,
    num_tokens = None,
    dim,
    depth,
    seq_len,
    ff_mult = 4,
    attn_dim = None
):
    dim_ff = dim * ff_mult

    return nn.Sequential(
        nn.Embedding(num_tokens, dim) if exists(num_tokens) else nn.Identity(),
        *[Residual(nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_ff * 2),
            nn.GELU(),
            SpatialGatingUnit(dim_ff, seq_len, attn_dim),
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
    channels = 3,
    attn_dim = None
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
            SpatialGatingUnit(dim_ff, num_patches, attn_dim),
            nn.Linear(dim_ff, dim)
        )) for i in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n d -> b d', 'mean'),
        nn.Linear(dim, num_classes)
    )
