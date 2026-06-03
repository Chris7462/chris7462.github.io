---
sidebar_position: 4
title: Install manif
description: Build and install manif from source on Ubuntu 24.04
---

# Install manif

manif is a C++ header-only library for Lie group operations on manifolds (SO2, SO3, SE2, SE3, etc.), designed for state estimation and robotics applications.

## Dependencies

```bash
sudo apt install cmake libeigen3-dev
```

## Build and Install

```bash
git clone https://github.com/artivis/manif.git
cd manif
mkdir build
cd build
cmake ..
cmake --build . -j$(nproc)
sudo cmake --install .
```
