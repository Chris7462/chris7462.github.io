---
sidebar_position: 2
title: Lenovo Legion Pro 7i Gen 10 — Install NVIDIA Driver
description: Install NVIDIA GPU driver on a Lenovo Legion Pro 7i Gen 10 (RTX 5090) running Ubuntu 26.04
---

# Install NVIDIA Driver

This guide covers installing the NVIDIA proprietary driver on a **Lenovo Legion Pro 7i Gen 10** running Ubuntu 26.04. These steps disable the Nouveau open-source driver and configure the system to use the discrete NVIDIA GPU exclusively.

## System Specs

| Component | Details |
|-----------|---------|
| CPU | Intel Core Ultra 9 275HX |
| GPU | NVIDIA GeForce RTX 5090 Laptop (24GB) |
| RAM | ~64GB (62 GiB usable) |
| Storage | 1TB SSD + 1TB SSD |
| Display | 16" 2560×1600 OLED 240Hz |

:::warning
Do **not** use NVIDIA's apt repository at `developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64/` to install the driver on the Lenovo Legion. Those binaries will not work correctly on this machine — **you will get a black screen**. Use the `.run` installer from the NVIDIA Unix Drivers page instead.
:::

## Step 1: Download the NVIDIA Driver

Download the driver from the [NVIDIA Unix Drivers page](https://www.nvidia.com/en-us/drivers/unix/). Choose version **610** or **595** depending on your preference.

## Step 2: Blacklist the Nouveau Driver

The Nouveau open-source driver must be disabled before installing the NVIDIA proprietary driver:

```bash
sudo tee /etc/modprobe.d/blacklist-nouveau.conf > /dev/null << EOF
blacklist nouveau
options nouveau modeset=0
EOF
```

Then update the initramfs:

```bash
sudo update-initramfs -u
```

## Step 3: Configure BIOS for Discrete Graphics

Reboot and press **F2** to enter the BIOS. Navigate to the graphics settings and select **Discrete Graphics** (bottom right of the screen), then save and exit.

:::info
**Discrete Graphics** mode connects the laptop display directly to the NVIDIA RTX (dGPU). The Intel integrated graphics (iGPU) is inactive and Optimus is disabled. This avoids hybrid graphics conflicts that cause black screens during driver installation on Linux.

The three available modes are:
- **Dynamic Graphics** — Optimus is active; the system switches between iGPU and dGPU automatically
- **Discrete Graphics** — dGPU (NVIDIA RTX) only; iGPU inactive, no Optimus. Use this for Linux + NVIDIA driver
- **UMA Graphics** — iGPU (Intel) only; dGPU disabled
:::

## Step 4: Switch to TTY and Stop the Display Manager

At the login screen, press **Ctrl + Alt + F1** to switch to a TTY session, then stop the display manager:

```bash
# Ubuntu GNOME
sudo systemctl stop gdm3

# Unity
# sudo systemctl stop lightdm

# KDE
# sudo systemctl stop sddm
```

## Step 5: Run the NVIDIA Installer

Run the downloaded installer (adjust the filename to match the version you downloaded):

```bash
sudo NVIDIA-Linux-x86_64-610.43.02.run
# or
# sudo NVIDIA-Linux-x86_64-595.80.run
```

Follow the on-screen instructions to complete the installation.

## Step 6: Reboot

```bash
sudo reboot
```

## Validation

After rebooting, verify the driver is installed and the GPU is detected:

```bash
nvidia-smi
```

You should see your GPU listed with the driver version and CUDA version.

## Uninstall

Use this section to fully remove the NVIDIA proprietary driver and restore the Nouveau driver.

### Step 1: Switch to TTY and Stop the Display Manager

Press **Ctrl + Alt + F1** to switch to a TTY session, then stop the display manager:

```bash
# Ubuntu GNOME
sudo systemctl stop gdm3

# Unity
# sudo systemctl stop lightdm

# KDE
# sudo systemctl stop sddm
```

### Step 2: Run the Uninstaller

The `.run` installer ships with a built-in uninstall option:

```bash
sudo NVIDIA-Linux-x86_64-610.43.02.run --uninstall
# or whichever version you installed
# sudo NVIDIA-Linux-x86_64-595.80.run --uninstall
```

Alternatively, if the original `.run` file is no longer available, use the uninstaller that was placed on your system during installation:

```bash
sudo nvidia-uninstall
```

### Step 3: Re-enable the Nouveau Driver

Remove the blacklist file created during installation:

```bash
sudo rm /etc/modprobe.d/blacklist-nouveau.conf
```

Then update the initramfs to apply the change:

```bash
sudo update-initramfs -u
```

### Step 4: Restore BIOS Graphics Setting (Optional)

If you switched the BIOS to **Discrete Graphics** during installation and want to go back, reboot, press **F2**, and restore the graphics mode to your preferred setting (e.g. **Dynamic Graphics** for Optimus/hybrid, or **UMA Graphics** for iGPU only).

### Step 5: Verify NVIDIA Files Are Removed

Before rebooting, confirm that NVIDIA driver files have been cleaned up.

#### Libraries

```bash
find /usr/lib /usr/lib32 /usr/lib64 /usr/local/lib -name "libnvidia*" 2>/dev/null
```

Should return no output. If any `libnvidia*` files remain, remove them:

```bash
sudo rm /usr/lib/x86_64-linux-gnu/libnvidia*.so*
sudo ldconfig
```

#### Kernel modules

```bash
lsmod | grep nvidia
```

Should return no output.

```bash
find /lib/modules -name "nvidia*.ko*" 2>/dev/null
```

:::note
This may return files like `nvidiafb.ko.zst` under `/lib/modules/.../drivers/video/fbdev/nvidia/`. These are the generic **nvidiafb framebuffer driver** that ships with Ubuntu — not from the `.run` installer — and can be safely left in place.
:::

#### Binaries

```bash
find /usr/bin /usr/local/bin -name "nvidia*" 2>/dev/null
```

The following binaries are **safe to keep** — they belong to the NVIDIA Container Toolkit (Docker GPU support) and CUDA MPS, not the driver itself:

| Binary | Belongs to |
|--------|-----------|
| `nvidia-container-cli` | NVIDIA Container Toolkit |
| `nvidia-container-toolkit` | NVIDIA Container Toolkit |
| `nvidia-container-runtime` | NVIDIA Container Toolkit |
| `nvidia-container-runtime-hook` | NVIDIA Container Toolkit |
| `nvidia-cdi-hook` | NVIDIA Container Toolkit |
| `nvidia-ctk` | NVIDIA Container Toolkit |
| `nvidia-cuda-mps-control` | CUDA MPS |
| `nvidia-cuda-mps-server` | CUDA MPS |

The following are driver-specific and should be removed if present:

```bash
sudo rm -f /usr/bin/nvidia-smi /usr/bin/nvidia-persistenced /usr/bin/nvidia-debugdump
```

### Step 6: Reboot

```bash
sudo reboot
```

After rebooting, the system will fall back to the Nouveau open-source driver. Verify with:

```bash
lsmod | grep nouveau
```
