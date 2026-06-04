---
sidebar_position: 3
title: Install GeographicLib
description: Build and install GeographicLib from source on Ubuntu 24.04
---

# Install GeographicLib

GeographicLib is a C++ library for geodesic computations, coordinate conversions, and geographic projections — commonly used for GPS/GNSS data processing and localization.

## Dependencies

```bash
sudo apt install cmake libeigen3-dev
```

## Build and Install

```bash
git clone https://github.com/geographiclib/geographiclib.git
cd geographiclib
mkdir build
cd build
cmake ..
cmake --build . -j$(nproc)
sudo cmake --install .
```
