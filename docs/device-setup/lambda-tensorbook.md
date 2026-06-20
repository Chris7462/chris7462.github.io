---
sidebar_position: 5
title: Lambda Tensorbook — Install NVIDIA Driver
description: Install the pinned NVIDIA driver and fix a hybrid-graphics Xorg crash on a Lambda Tensorbook running Ubuntu 26.04
---

# Install NVIDIA Driver

This guide covers installing the NVIDIA proprietary driver on a **Lambda Tensorbook** running Ubuntu 26.04, and fixing a hybrid-graphics Xorg crash that shows up after a fresh install.

## System Specs

| Component | Details |
|-----------|---------|
| CPU | Intel Core i7-10875H |
| GPU | NVIDIA GeForce RTX 2080 Super Mobile / Max-Q (8GB) |
| iGPU | Intel UHD Graphics (CometLake-H) |
| RAM | 64 GB |
| BIOS | Insyde H2O (Clevo/Sager-based) |

## Step 1: Install the Driver

Follow the standard [Install CUDA](../dl-setup/cuda.md) guide to add the keyring and pin a driver version:

```bash
os=ubuntu2604
wget https://developer.download.nvidia.com/compute/cuda/repos/${os}/x86_64/cuda-keyring_1.1-1_all.deb
sudo apt install ./cuda-keyring_1.1-1_all.deb
sudo apt update
apt policy nvidia-driver-open
sudo apt install nvidia-driver-pinning-610
sudo reboot
```

## Step 2: Hybrid-Graphics Xorg Crash

After rebooting, the system may drop to a TTY with a fatal Xorg error instead of reaching the login screen:

```
(EE) modeset(G0): Failed to create pixmap
(EE) failed to create screen resources(EE)
Fatal server error:
(EE) failed to create screen resources(EE)
Server terminated with error (1). Closing log file.
```

### Diagnosis

First confirm the driver itself is fine — this is a Xorg config issue, not a driver problem:

```bash
nvidia-smi
```

If this shows the GPU correctly (driver version, temperature, memory), the driver and kernel module are working. The actual error is in `/var/log/Xorg.0.log`:

```bash
grep -E "\(EE\)|Failed to create pixmap" /var/log/Xorg.0.log
```

```
(EE) [drm] Failed to open DRM device for (null): -2
(EE) modeset(G0): Failed to create pixmap
(EE) failed to create screen resources(EE)
```

`-2` is `ENOENT` with a `(null)` device path. The Tensorbook is a hybrid-graphics laptop (`lshw -C display` shows both the NVIDIA dGPU on `driver=nvidia` and the Intel iGPU on `driver=i915`). Xorg's `AutoAddGPU` behavior tries to automatically attach the second GPU as an extra provider using the generic `modesetting` driver — even though the NVIDIA card already has its own driver bound — and fails to resolve a device path for it, crashing the whole server.

:::note
The Insyde H2O BIOS on this machine (and Clevo/Sager-based machines in general) does **not** expose a Discrete-Graphics-only mode in Setup Utility, even after unlocking the hidden Clevo menu via **Advanced Chipset Control → Setup Menu Insyde Full Show → Show**. So unlike the [Lenovo Legion](./lenovo-legion.md), this isn't fixable from the BIOS side — it has to be fixed in Xorg config.
:::

### Fix

Disable `AutoAddGPU` so Xorg lets each GPU's own driver (udev `OutputClass`) handle it instead of trying to bolt on an extra provider:

```bash
sudo mkdir -p /etc/X11/xorg.conf.d
sudo tee /etc/X11/xorg.conf.d/10-disable-autoaddgpu.conf << 'EOF'
Section "ServerFlags"
    Option "AutoAddGPU" "false"
EndSection
EOF
sudo reboot
```

This only affects whether Xorg auto-attaches extra GPU providers — it doesn't touch the NVIDIA driver or CUDA/`nvidia-smi` functionality. After rebooting, the system should reach the normal login screen.

## Validation

```bash
nvidia-smi
nvcc --version
```

You should see the GPU listed with driver and CUDA version, and a working desktop session.
