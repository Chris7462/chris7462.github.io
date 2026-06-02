---
sidebar_position: 1
title: Install CUDA
description: Install the NVIDIA CUDA Toolkit and configure system-wide environment variables on Ubuntu 24.04
---

# Install CUDA

Update the Ubuntu version below if needed. This example was created for Ubuntu 24.04.

```bash
# Set the Ubuntu version
os=ubuntu2404

# Download and install the CUDA repository keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/${os}/x86_64/cuda-keyring_1.1-1_all.deb
sudo apt install ./cuda-keyring_1.1-1_all.deb

# Update package lists
sudo apt update
# If you want to have specific nvidia driver version
# sudo apt install nvidia-driver-pinning-610    # pinning on 610 driver
sudo apt install nvidia-driver-open
sudo apt install cuda-toolkit
```

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

