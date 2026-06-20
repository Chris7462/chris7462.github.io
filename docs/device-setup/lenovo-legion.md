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
Do **not** use NVIDIA's apt repository at `developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64/` to install the driver on the Lenovo Legion. Those binaries will not work correctly with the Legion's iGPU configuration — **you will get a black screen**. Use the `.run` installer from the NVIDIA Unix Drivers page instead.
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
This disables the Intel iGPU entirely so the display manager only talks to the NVIDIA RTX GPU — avoiding Optimus/hybrid graphics conflicts.
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
