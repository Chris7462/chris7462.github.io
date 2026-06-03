---
sidebar_position: 1
title: Install Ceres Solver
description: Build and install Ceres Solver from source on Ubuntu 24.04
---

# Install Ceres Solver

## Dependencies

Install the required dependencies:

```bash
# CMake
sudo apt-get install cmake
# google-glog + gflags
sudo apt-get install libgoogle-glog-dev libgflags-dev
# Use ATLAS for BLAS & LAPACK
sudo apt-get install libatlas-base-dev
# Eigen3
sudo apt-get install libeigen3-dev
# SuiteSparse (optional)
sudo apt-get install libsuitesparse-dev
```

## Build and Install

```bash
git clone --recursive-submodule https://github.com/ceres-solver/ceres-solver.git
cd ceres-solver
mkdir build
cd build
cmake ..
cmake --build . -j$(nproc)
sudo cmake --install .
```

## Validation

You can verify the installation by running the bundled command-line bundle adjustment application against one of the included problems from the University of Washington's BAL dataset ([Agarwal](http://ceres-solver.org/bibliography.html#agarwal)):

```bash
bin/simple_bundle_adjuster ../data/problem-16-22106-pre.txt
```
