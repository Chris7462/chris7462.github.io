---
sidebar_position: 1
title: Install CARLA 0.9.16
description: Install CARLA 0.9.16 on Ubuntu 24.04 via SSH
---

# Install CARLA 0.9.16

This guide covers installing CARLA 0.9.16 on Ubuntu 24.04 accessed via SSH.

## Environment

| | |
|---|---|
| **Machine** | Lenovo Legion (Ubuntu 24.04) |
| **CARLA path** | `/opt/carla/0.9.16` |
| **Python** | System Python 3.12 (no need to install 3.10) |
| **Access** | SSH |

## Step 1. Download CARLA 0.9.16

:::note
If CARLA is already installed at `/opt/carla/0.9.16`, skip this step.
:::

```bash
mkdir -p ~/carla && cd ~/carla

# Main package (~7GB)
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.16.tar.gz

# Additional maps (optional but recommended for Town07 etc.)
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.16.tar.gz

# Create installation directory and extract
sudo mkdir -p /opt/carla/0.9.16
sudo tar -xzf CARLA_0.9.16.tar.gz -C /opt/carla/0.9.16
```

If you downloaded the additional maps, sync them manually:

```bash
sudo mkdir -p /opt/carla/tmp
sudo tar -xzf AdditionalMaps_0.9.16.tar.gz -C /opt/carla/tmp

sudo rsync -av /opt/carla/tmp/CarlaUE4/ /opt/carla/0.9.16/CarlaUE4/
sudo rsync -av /opt/carla/tmp/Engine/ /opt/carla/0.9.16/Engine/
```

## Step 2. Python Virtual Environment

CARLA 0.9.16 provides wheels up to Python 3.12 (`cp312`). Ubuntu 24.04 ships Python 3.12 by default, but Ubuntu 26.04 ships Python 3.14, which has no matching wheel. If you are on Ubuntu 26.04, install Python 3.12 first via the deadsnakes PPA:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev
```

:::note
Ubuntu 24.04 ships with Python 3.12 by default, so this step can be skipped on 24.04.
:::

Then create the virtual environment using Python 3.12 explicitly:

```bash
python3.12 -m venv ~/carla-env
source ~/carla-env/bin/activate

pip install /opt/carla/0.9.16/PythonAPI/carla/dist/carla-0.9.16-cp312-cp312-manylinux_2_31_x86_64.whl
pip install numpy pygame
```

If you are using a different Python version, pick the matching wheel from the `PythonAPI/carla/dist/` directory:

| Wheel File | Python |
|---|---|
| `carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl` | Python 3.10 |
| `carla-0.9.16-cp311-cp311-manylinux_2_31_x86_64.whl` | Python 3.11 |
| `carla-0.9.16-cp312-cp312-manylinux_2_31_x86_64.whl` | Python 3.12 |

## Step 3. Launch CARLA

:::warning
**X11 forwarding does not work with CARLA.** CARLA uses Vulkan for rendering, which is incompatible with `ssh -X` forwarding. Running `./CarlaUE4.sh` over SSH will crash with:

```
VK_ERROR_INITIALIZATION_FAILED
Segmentation fault (core dumped)
```
:::

**Recommended: Headless Mode**

Run CARLA without a display window. The Python API and ROS 2 bridge work fully in this mode:

```bash
/opt/carla/0.9.16/CarlaUE4.sh -RenderOffScreen
```

### Optional Launch Flags

| Flag | Purpose |
|---|---|
| `-RenderOffScreen` | No GUI window (required over SSH) |
| `-quality-level=Low` | Reduce GPU load |
| `-world-port=2000` | Change RPC port (default: 2000) |

## Step 4. Verify Installation

With CARLA running in one terminal, open another terminal and run:

```bash
source ~/carla-env/bin/activate
python3 -c "import carla; client = carla.Client('localhost', 2000); print(client.get_server_version())"
# Expected output: 0.9.16
```

## Step 5. Remote Visualization Options

Since `ssh -X` cannot forward the CARLA window, use one of these options if a GUI is needed:

| Option | Notes |
|---|---|
| **VNC** (`tigervnc`) | Full remote desktop; run CARLA inside VNC session |
| **NoMachine** | Smoother than VNC, free for personal use |
| **VirtualGL + TurboVNC** | Best performance, more complex setup |

:::tip
For most use cases (Python API, ROS 2 bridge, RViz), headless mode is sufficient. RViz itself does work over `ssh -X`.
:::

## Step 6. Running CARLA

Two approaches are available depending on your setup.

### Option A: System-wide Wrapper (Headless / SSH)

Suitable for SSH access or machines with a single GPU. Creates a `carla` command available to all users:

```bash
sudo vim /usr/local/bin/carla
```

Paste the following:

```bash
#!/bin/bash
exec /opt/carla/0.9.16/CarlaUE4.sh -RenderOffScreen "$@"
```

Make it executable:

```bash
sudo chmod +x /usr/local/bin/carla
```

Now any user can run:

```bash
carla                      # runs with -RenderOffScreen by default
carla -quality-level=Low   # additional flags can still be passed
```

### Option B: NVIDIA GPU (Local Display)

On machines with both an integrated GPU and a discrete NVIDIA GPU (e.g. Lenovo Legion), Vulkan may default to the iGPU. Add this alias to your `~/.bashrc` to force CARLA to use the NVIDIA GPU and render with a display window:

```bash
alias carla='VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
__NV_PRIME_RENDER_OFFLOAD=1 \
__NV_PRIME_RENDER_OFFLOAD_PROVIDER=NVIDIA-G0 \
__GLX_VENDOR_LIBRARY_NAME=nvidia \
__VK_LAYER_NV_optimus=NVIDIA_only \
/opt/carla/0.9.16/CarlaUE4.sh -ResX=640 -ResY=640'
```

Apply immediately:

```bash
source ~/.bashrc
```

Then run:

```bash
carla
```

:::note
This alias launches CARLA with a visible window at 640×640. Adjust `-ResX` and `-ResY` as needed.
:::

## Notes

- No bundled `libstdc++.so.6` found in `CarlaUE4/Binaries/Linux/` — this build uses the system library directly and is compatible with Ubuntu 24.04 out of the box.
- The `manylinux_2_31` wheel tag requires glibc ≥ 2.31; Ubuntu 24.04 ships glibc 2.39
