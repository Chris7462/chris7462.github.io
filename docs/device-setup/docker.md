---
sidebar_position: 5
title: Install Docker with NVIDIA Support
description: Install Docker and the NVIDIA Container Toolkit on Ubuntu 26.04
---

# Install Docker with NVIDIA Support

This guide covers installing Docker and the NVIDIA Container Toolkit on Ubuntu 26.04. For NVIDIA driver and CUDA installation, see the [Install CUDA](../dl-setup/cuda.md) guide.

## Install Docker

### 0. Uninstall old versions

Before installing, remove any conflicting packages:

```bash
sudo apt remove $(dpkg --get-selections docker.io docker-compose docker-compose-v2 docker-doc podman-docker containerd runc | cut -f1)
```

### 1. Set up Docker's apt repository

```bash
# Install dependencies
sudo apt update
sudo apt install ca-certificates curl

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to apt sources
sudo tee /etc/apt/sources.list.d/docker.sources << EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF
```

### 2. Install the Docker packages

```bash
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### 3. Manage Docker as a non-root user

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### 4. Verify the installation

```bash
docker run hello-world
```

## Install NVIDIA Container Toolkit

### 5. Install the toolkit

```bash
sudo apt-get install nvidia-container-toolkit
```

### 6. Configure the Docker runtime

Edit the Docker daemon configuration file:

```bash
sudo vim /etc/docker/daemon.json
```

Make sure the following contents are present:

```json
{
  "runtimes": {
    "nvidia": {
      "args": [],
      "path": "nvidia-container-runtime"
    }
  },
  "data-root": "/home/docker"
}
```

### 7. Change the containerd root

```bash
sudo vim /etc/containerd/config.toml
```

Find the line:

```toml
#root = "/var/lib/containerd"
```

Uncomment it and change it to:

```toml
root = "/home/docker/containerd"
```

### 8. Restart Docker

```bash
sudo systemctl restart docker
```

## Validation

Pick an NVIDIA image from [Docker Hub](https://hub.docker.com/r/nvidia/cuda/tags) and verify GPU access:

```bash
docker run --gpus all nvidia/cuda:13.2.0-base-ubuntu26.04 nvidia-smi
```

:::tip
For more advanced Docker usage including GUI applications, X11 forwarding, and SLURM integration, see the [Docker Usage](../hpc/user/docker.md) guide.
:::
