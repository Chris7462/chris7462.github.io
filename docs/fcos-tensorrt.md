---
sidebar_position: 1
title: FCOS Deployment to TensorRT
description: Step-by-step guide to deploy FCOS object detection to TensorRT on Ubuntu 24.04
---

# FCOS Deployment to TensorRT

*July 20, 2025*

## Install Nvidia CUDA

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo apt install ./cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install nvidia-driver-pinning-595    # pinning on 595 driver
sudo apt install nvidia-driver-open
sudo apt-get -y install cuda-toolkit-13-2
```

```bash
# Taken from: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions
echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
```

## Install Nvidia cuDNN and TensorRT

```bash
sudo apt install \
  libnvinfer-bin=10.16.0.72-1+cuda13.2 \
  libnvinfer-dev=10.16.0.72-1+cuda13.2 \
  libnvinfer-dispatch-dev=10.16.0.72-1+cuda13.2 \
  libnvinfer-dispatch10=10.16.0.72-1+cuda13.2 \
  libnvinfer-headers-dev=10.16.0.72-1+cuda13.2 \
  libnvinfer-headers-plugin-dev=10.16.0.72-1+cuda13.2 \
  libnvinfer-lean-dev=10.16.0.72-1+cuda13.2 \
  libnvinfer-lean10=10.16.0.72-1+cuda13.2 \
  libnvinfer-plugin-dev=10.16.0.72-1+cuda13.2 \
  libnvinfer-plugin10=10.16.0.72-1+cuda13.2 \
  libnvinfer-samples=10.16.0.72-1+cuda13.2 \
  libnvinfer-vc-plugin-dev=10.16.0.72-1+cuda13.2 \
  libnvinfer-vc-plugin10=10.16.0.72-1+cuda13.2 \
  libnvinfer10=10.16.0.72-1+cuda13.2 \
  libnvonnxparsers-dev=10.16.0.72-1+cuda13.2 \
  libnvonnxparsers10=10.16.0.72-1+cuda13.2 \
  python3-libnvinfer=10.16.0.72-1+cuda13.2 \
  python3-libnvinfer-dev=10.16.0.72-1+cuda13.2 \
  python3-libnvinfer-dispatch=10.16.0.72-1+cuda13.2 \
  python3-libnvinfer-lean=10.16.0.72-1+cuda13.2 \
  tensorrt=10.16.0.72-1+cuda13.2
```

## Install Torch, TorchVision, and TorchAudio for Python

My suggestion is to install these packages from pre-built Python wheels directly. This would save you a lot of time and trouble as long as you have installed the correct Nvidia and CUDA driver.

```bash
sudo pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

## Install LibTorch and LibTorchVision for C++

LibTorch and LibTorchVision are used for deployment in C++. For LibTorch you can download a pre-built version, but for LibTorchVision you still have to build it from source. I build both from source.

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

Or, you can simply do:

```bash
cd pytorch
python tools/build_libtorch.py
```

Setup the bash environment:

```bash
export Torch_DIR="/home/yi-chen/thirdparty/libtorch/share/cmake/Torch"
export LD_LIBRARY_PATH="/home/yi-chen/thirdparty/libtorch/lib:$LD_LIBRARY_PATH"
```

To install LibTorchVision, follow the same steps:

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
cmake --install .
```
