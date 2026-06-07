---
sidebar_position: 5
title: Install OpenCV
description: Build and install OpenCV with CUDA support from source on Ubuntu 24.04
---

# Install OpenCV

To enable GPU support for OpenCV's DNN module, OpenCV must be built from source with CUDA support.

## Clone the Repositories

Clone both OpenCV and the extra modules (opencv_contrib):

```bash
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
```

## Build from Source

```bash
cd opencv
mkdir build
cd build

OPENCV_CONTRIB_MODULES=/home/yi-chen/thirdparty/opencv_contrib/modules
OPENBLAS_INCLUDE_DIR=/usr/include/x86_64-linux-gnu/openblas-pthread
OPENBLAS_LIB=/usr/lib/x86_64-linux-gnu/libopenblas.so

cmake -DCMAKE_BUILD_TYPE=RELEASE \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      \
      -DWITH_CUDA=ON \
      -DENABLE_CUDA_FIRST_CLASS_LANGUAGE=ON \
      -DCUDA_ARCH_BIN="7.5;8.9;12.0" \
      -DWITH_CUBLAS=ON \
      -DWITH_CUDNN=ON \
      -DOPENCV_DNN_CUDA=ON \
      -DCUDA_FAST_MATH=ON \
      -DENABLE_FAST_MATH=ON \
      \
      -DBUILD_opencv_python3=ON \
      -DPYTHON3_EXECUTABLE="$(which python3)" \
      \
      -DWITH_TBB=ON \
      -DWITH_GSTREAMER=ON \
      -DWITH_FFMPEG=ON \
      -DWITH_OPENMP=ON \
      \
      -DWITH_OPENBLAS=ON \
      -DWITH_LAPACK=ON \
      -DOpenBLAS_LIB="${OPENBLAS_LIB}" \
      -DOpenBLAS_INCLUDE_DIR="${OPENBLAS_INCLUDE_DIR}" \
      \
      -DBUILD_opencv_sfm=ON \
      -DBUILD_opencv_cudacodec=ON \
      -DOPENCV_ENABLE_NONFREE=ON \
      -DOPENCV_EXTRA_MODULES_PATH="${OPENCV_CONTRIB_MODULES}" \
      \
      -DBUILD_EXAMPLES=ON \
      -DINSTALL_PYTHON_EXAMPLES=ON \
      -DINSTALL_C_EXAMPLES=ON \
      \
      ..
```

:::info
`CUDA_ARCH_BIN` depends on your NVIDIA GPU's compute capability. Check the [CUDA GPU list on Wikipedia](https://en.wikipedia.org/wiki/CUDA) for the correct value for your card. For example:
- RTX 3090 / A100 → `8.6` / `8.0`
- RTX 4090 → `8.9`
- RTX 5090 → `12.0`

You can turn off flags unrelated to CUDA, but the CUDA-related flags must be enabled and `CUDA_ARCH_BIN` must be specified.
:::

Then build and install:

```bash
cmake --build . -j$(nproc)
sudo cmake --install .
sudo ldconfig
```
