<img src="./gmlp.png" width="400px"></img>

## gMLP - Pytorch

Implementation of <a href="https://arxiv.org/abs/2105.08050">gMLP</a>, an all-MLP replacement for Transformers, in Pytorch

## Install

```bash
$ pip install g-mlp-pytorch
```

## Usage

For masked language modelling

```python
import torch
from torch import nn
from g_mlp_pytorch import gMLP

model = gMLP(
    num_tokens = 20000,
    dim = 512,
    depth = 6,
    seq_len = 256,
    act = nn.Tanh()        # activation for spatial gate (defaults to identity)
)

x = torch.randint(0, 20000, (1, 256))
logits = model(x) # (1, 256, 20000)
```

For image classification

```python
import torch
from g_mlp_pytorch import gMLPVision

model = gMLPVision(
    image_size = 256,
    patch_size = 16,
    num_classes = 1000,
    dim = 512,
    depth = 6
)

img = torch.randn(1, 3, 256, 256)
logits = model(img) # (1, 1000)
```

You can also add a tiny amount of attention (one-headed) to boost performance, as mentioned in the paper as `aMLP`, with the addition of one extra keyword `attn_dim`. This applies to both `gMLPVision` and `gMLP`

```python
import torch
from g_mlp_pytorch import gMLPVision

model = gMLPVision(
    image_size = 256,
    patch_size = 16,
    num_classes = 1000,
    dim = 512,
    depth = 6,
    attn_dim = 64
)

img = torch.randn(1, 3, 256, 256)
pred = model(img) # (1, 1000)
```

Non-square images and patch sizes

```python
import torch
from g_mlp_pytorch import gMLPVision

model = gMLPVision(
    image_size = (256, 128),
    patch_size = (16, 8),
    num_classes = 1000,
    dim = 512,
    depth = 6,
    attn_dim = 64
)

img = torch.randn(1, 3, 256, 128)
pred = model(img) # (1, 1000)
```

## Experimental

A independent researcher proposes using circulant matrices in gMLPs in <a href="https://zhuanlan.zhihu.com/p/395005917">a blogpost on Zhihu</a>. This allows you to scale gMLPs with increasing sequence length with linear parameter costs (as opposed to quadratic). My experiments show little performance degradation.

You can use it by setting one extra flag to `True`

```python
import torch
from torch import nn
from g_mlp_pytorch import gMLP

model = gMLP(
    num_tokens = 20000,
    dim = 512,
    depth = 6,
    seq_len = 256,
    causal = True,
    use_circulant_matrix = True  # set to True
)

x = torch.randint(0, 20000, (1, 256))
logits = model(x) # (1, 256, 20000)
```

## Citations

```bibtex
@misc{liu2021pay,
    title   = {Pay Attention to MLPs}, 
    author  = {Hanxiao Liu and Zihang Dai and David R. So and Quoc V. Le},
    year    = {2021},
    eprint  = {2105.08050},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

```bibtex
@software{peng_bo_2021_5196578,
    author       = {PENG Bo},
    title        = {BlinkDL/RWKV-LM: 0.01},
    month        = aug,
    year         = 2021,
    publisher    = {Zenodo},
    version      = {0.01},
    doi          = {10.5281/zenodo.5196578},
    url          = {https://doi.org/10.5281/zenodo.5196578%7D
}
```
