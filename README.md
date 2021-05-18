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
from g_mlp_pytorch import gMLP

model = gMLP(
    num_tokens = 20000,
    dim = 512,
    depth = 6,
    seq_len = 256
)

x = torch.randint(0, 20000, (1, 256))
emb = model(x) # (1, 256, 512)
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
pred = model(img) # (1, 1000)
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
