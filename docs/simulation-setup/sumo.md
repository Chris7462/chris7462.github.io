---
sidebar_position: 2
title: Install SUMO with JuPedSim
description: Build SUMO from source with JuPedSim integration on Ubuntu 24.04
---

# Install SUMO with JuPedSim

This guide covers building [Eclipse SUMO](https://github.com/eclipse-sumo/sumo) from source with JuPedSim integration, installing system-wide to `/usr/local`.

## Step 0. Clone the Repository

```bash
mkdir -p ~/thirdparty && cd ~/thirdparty
git clone --recursive https://github.com/eclipse-sumo/sumo
```

:::note
The `--recursive` flag is required to also clone SUMO's submodules.
:::

## Step 1. Install Dependencies

Use SUMO's provided script to install all required apt packages and build JuPedSim v1.3.1 from source:

```bash
cd ~/thirdparty/sumo
sudo bash build_config/install_dependencies.sh
```

This installs all apt dependencies and JuPedSim to `/usr/local` by default.

## Step 2. Install OpenSceneGraph

Required for SUMO's 3D visualization support:

```bash
sudo apt install libopenscenegraph-dev
```

## Step 3. Build SUMO

```bash
cd ~/thirdparty/sumo
mkdir -p build && cd build
cmake ..
cmake --build . -j$(nproc)
```

### Expected Enabled Features

After a successful build, CMake should report the following enabled features:

```
FMI Proj GUI FMT Intl SWIG tcmalloc Parquet Eigen GDAL FFmpeg OSG GL2PS JuPedSim
```

### Known Non-Issues

These CMake warnings are harmless and can be safely ignored:

- `Could NOT find protobuf (missing: protobuf_DIR)` — false alarm; cmake finds it via fallback immediately after
- `Boost not found, building without stacktrace support` — `boost_process` cmake config is missing from Ubuntu 24.04 packaging; only affects crash diagnostics in debug builds, not runtime functionality

## Step 4. Install to `/usr/local`

Install to the default `/usr/local` prefix. This avoids cmake path baking issues and means binaries are automatically on all users' `PATH` with no extra configuration.

```bash
sudo cmake --install .
```

Files are installed to:

| Path | Contents |
|------|----------|
| `/usr/local/bin` | Executables (`sumo`, `sumo-gui`, `netconvert`, etc.) |
| `/usr/local/lib` | SUMO and JuPedSim shared libraries |
| `/usr/local/include/jupedsim` | JuPedSim headers |
| `/usr/local/share/sumo` | Data files, tools, schemas (`$SUMO_HOME`) |
| `/etc/profile.d/sumo.sh` | System-wide `SUMO_HOME` environment variable |

## Step 5. System-wide Environment Configuration

`/usr/local/bin` is already on users' `PATH` by default, so only `SUMO_HOME` needs to be set. Create `/etc/profile.d/sumo.sh`:

```bash
sudo nano /etc/profile.d/sumo.sh
```

Add:

```bash
export SUMO_HOME=/usr/local/share/sumo
```

Make it executable:

```bash
sudo chmod +x /etc/profile.d/sumo.sh
```

Apply immediately:

```bash
source /etc/profile.d/sumo.sh
```

## Step 6. Verify Installation

```bash
sumo --version
sumo-gui --version
```

## Uninstall

### Uninstall SUMO

CMake generates an `install_manifest.txt` in the build directory listing every installed file. Use it to remove all installed files:

```bash
cd ~/thirdparty/sumo/build
sudo xargs rm -f < install_manifest.txt
```

Then clean up any empty directories left behind:

```bash
sudo rm -rf /usr/local/share/sumo
```

### Uninstall JuPedSim

```bash
sudo rm /usr/local/lib/libjupedsim.so
sudo rm -rf /usr/local/include/jupedsim
sudo rm -rf /usr/local/lib/cmake/jupedsim
sudo ldconfig
```

### Remove Environment Configuration

```bash
sudo rm /etc/profile.d/sumo.sh
```
