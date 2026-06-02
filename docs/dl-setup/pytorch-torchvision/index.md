---
sidebar_position: 4
title: Install PyTorch and TorchVision
description: Install PyTorch and TorchVision for Python using pre-built wheels on Ubuntu 24.04
---

# Install PyTorch and TorchVision

My suggestion is to install these packages from pre-built Python wheels directly. This saves a lot of time and trouble as long as you have installed the correct NVIDIA driver and CUDA toolkit.

```bash
sudo pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132 --break-system-packages
```

## Validation

Verify that PyTorch and TorchVision are installed correctly and that the GPU is accessible:

```python
import torch
torch.__version__    # '2.12.0+cu132'
torch.cuda.is_available()    # True

import torchvision
torchvision.__version__    # '0.27.0+cu132'
```

Expected output:

```
'2.12.0+cu132'
True
'0.27.0+cu132'
```
```
