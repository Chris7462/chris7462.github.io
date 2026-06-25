---
sidebar_position: 1
title: Lambda Tensorbook — Install NVIDIA Driver
description: Install the NVIDIA driver on a Lambda Tensorbook running Ubuntu 26.04
---

# Install NVIDIA Driver

This guide covers installing the NVIDIA proprietary driver on a **Lambda Tensorbook** running Ubuntu 26.04.

## System Specs

| Component | Details |
|-----------|---------|
| CPU | Intel Core i7-10875H |
| GPU | NVIDIA GeForce RTX 2080 Super Mobile / Max-Q (8GB) |
| iGPU | Intel UHD Graphics (CometLake-H) |
| RAM | 64 GB |
| BIOS | Insyde H2O (Clevo/Sager-based) |


## Step 1: Install the NVIDIA apt repo

Follow the standard [Install CUDA](../dl-setup/cuda.md) guide to add the keyring and pin a driver version:

```bash
os=ubuntu2604
wget https://developer.download.nvidia.com/compute/cuda/repos/${os}/x86_64/cuda-keyring_1.1-1_all.deb
sudo apt install ./cuda-keyring_1.1-1_all.deb
sudo apt update
```

## Step 2: Blacklist the Nouveau Driver

The Nouveau open-source driver must be disabled before rebooting into the NVIDIA driver:

```bash
sudo tee /etc/modprobe.d/blacklist-nouveau.conf > /dev/null << EOF
blacklist nouveau
options nouveau modeset=0
EOF
```
then update the initramfs image and reboot
```bash
sudo update-initramfs -u
sudo reboot
```

## Step 3: Switch to TTY and Stop the Display Manager

At the login screen, press **Ctrl + Alt + F1** to switch to a TTY session, then stop the display manager:

```bash
# Unity
# sudo systemctl stop lightdm

# Ubuntu GNOME
# sudo systemctl stop gdm3

# KDE
# sudo systemctl stop sddm
```

## Step 4: Install the NVIDIA driver

Check the latest available NVIDIA driver, check the version first with `apt policy` and and pinning the specific version
```bash
sudo apt policy nvidia-driver-open
sudo apt install nvidia-driver-pinning-610
sudo apt install nvidia-driver-open
```

## Step 5: Install the Xorg NVIDIA Video Driver

The CUDA repo package does not pull in `xserver-xorg-video-nvidia` automatically. Without it, X falls back to nouveau and fails to start:

```bash
sudo apt install xserver-xorg-video-nvidia
sudo reboot
```

## Validation

```bash
nvidia-smi
```

You should see the GPU listed with driver and CUDA version, and a working desktop session.
