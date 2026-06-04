---
sidebar_position: 2
title: Install g2o
description: Build and install g2o from source on Ubuntu 24.04
---

# Install g2o

g2o (General Graph Optimization) is an open-source C++ framework for optimizing graph-based nonlinear error functions, widely used in SLAM and bundle adjustment.

## Dependencies

```bash
sudo apt install cmake libeigen3-dev
```

:::tip
g2o can optionally use **Ceres Solver**. See the [Install Ceres Solver](./ceres-solver) page if you need it.
:::

## Build and Install

```bash
git clone https://github.com/RainerKuemmerle/g2o.git
cd g2o
mkdir build
cd build
cmake ..
cmake --build . -j$(nproc)
sudo cmake --install .
```
