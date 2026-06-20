---
sidebar_position: 1
title: Install CUDA
description: Install the NVIDIA CUDA Toolkit and configure system-wide environment variables on Ubuntu 24.04
---

# Install CUDA

Set the `os` variable below to match your Ubuntu version (e.g. `ubuntu2404` for 24.04, `ubuntu2604` for 26.04):

```bash
# Set the Ubuntu version
os=ubuntu2604

# Download and install the CUDA repository keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/${os}/x86_64/cuda-keyring_1.1-1_all.deb
sudo apt install ./cuda-keyring_1.1-1_all.deb

# Update package lists
sudo apt update
```

## Choosing a Driver: Pinned vs Rolling

Two ways to install the NVIDIA driver from this point:

- **`nvidia-driver-pinning-XXX`** — pins to a specific major driver version (e.g. `610`). Recommended in general, since CUDA Toolkit and TensorRT versions are tested against specific driver versions, and a pinned install won't silently jump to a newer (potentially incompatible) driver on a routine `apt upgrade`.
- **`nvidia-driver-open`** — always installs whatever the latest open-kernel-module driver is. Simpler, but less reproducible.

Check what version is currently available before deciding:

```bash
apt policy nvidia-driver-open
```

This shows the candidate version apt would install (e.g. `610.xx-xxx`). Use that major version number to pin:

```bash
sudo apt install nvidia-driver-pinning-610
```

:::tip
If you'd rather not pin and just want the latest driver, skip the pinning package and run `sudo apt install nvidia-driver-open` instead.
:::

## Install the CUDA Toolkit

```bash
sudo apt install cuda-toolkit
```

## Configure Environment Variables

Configure CUDA environment variables system-wide by creating `/etc/profile.d/cuda.sh`:

```bash
sudo tee /etc/profile.d/cuda.sh << 'EOF'
# Runtime paths — required to run CUDA binaries and load shared libraries
export PATH=/usr/local/cuda/bin${PATH:+:$PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Build-time paths — required when compiling C++ projects against CUDA (e.g., LibTorch)
export CPATH=/usr/local/cuda/include${CPATH:+:$CPATH}
export C_INCLUDE_PATH=/usr/local/cuda/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}
export LIBRARY_PATH=/usr/local/cuda/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}
EOF
```

Apply the changes by logging out and back in, or run:

```bash
source /etc/profile.d/cuda.sh
```

## Validation

```bash
nvidia-smi
nvcc --version
```
