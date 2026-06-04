---
sidebar_position: 3
title: Install LibTorch and LibTorchVision
description: Build and install LibTorch and LibTorchVision from source for C++ inference on Ubuntu 24.04
---

# Install LibTorch and LibTorchVision

LibTorch and LibTorchVision are used for model deployment in C++. For LibTorch you can download a pre-built version, but for LibTorchVision you still have to build it from source. I build both from source.

## Install LibTorch

```bash
# If not already installed
sudo apt install libnccl2 libnccl-dev

cd ~/thirdparty/pytorch
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_SHARED_LIBS=ON \
         -DUSE_CUDA=ON \
         -DUSE_CUDNN=ON \
         -DUSE_CUDSS=ON \
         -DUSE_CUFILE=ON \
         -DUSE_CUSPARSELT=ON \
         -DUSE_SYSTEM_NCCL=ON \
         -DCMAKE_INSTALL_PREFIX=/opt/libtorch
cmake --build . -j$(nproc)
sudo cmake --install .
```

:::tip
If you encounter build issues, make sure the CUDA environment variables are set correctly. See the [Install CUDA](./cuda.md) page for the system-wide setup.
:::

Or, you can simply do:

```bash
cd pytorch
python tools/build_libtorch.py
```

Then, set up the environment variables system-wide via `/etc/profile.d/libtorch.sh`:

```bash
sudo tee /etc/profile.d/libtorch.sh << 'EOF'
export Torch_DIR=/opt/libtorch/share/cmake/Torch
export LD_LIBRARY_PATH=/opt/libtorch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
EOF
```

Re-login or run `source /etc/profile.d/libtorch.sh` to apply immediately.

## Install LibTorchVision

```bash
git clone --recursive https://github.com/pytorch/vision torchvision
cd torchvision
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_SHARED_LIBS=ON \
         -DWITH_CUDA=ON \
         -DCMAKE_PREFIX_PATH=/opt/libtorch \
         -DCMAKE_INSTALL_PREFIX=/opt/libtorchvision
cmake --build . -j$(nproc)
sudo cmake --install .
```

Set up the environment variables system-wide via `/etc/profile.d/libtorchvision.sh`:

```bash
sudo tee /etc/profile.d/libtorchvision.sh << 'EOF'
export TorchVision_DIR=/opt/libtorchvision/share/cmake/TorchVision
export LD_LIBRARY_PATH=/opt/libtorchvision/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
EOF
```

Re-login or run `source /etc/profile.d/libtorchvision.sh` to apply immediately.
