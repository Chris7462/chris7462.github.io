---
title: Docker Usage
sidebar_label: Docker
sidebar_position: 4
---

# Docker Usage and Management

This document covers how to use Docker on the cluster, including GPU access, GUI applications, and SLURM integration.

---

## Which Machine Should I Use?

| Task | Machine | Workflow |
|------|---------|----------|
| Batch jobs, CPU compute, SLURM submissions | node01 | Submit via SLURM as usual |
| Docker containers, GPU workloads | node02 | SSH directly, run Docker freely |

**node02 is the dedicated Docker machine.** You do not need SLURM to run Docker on node02 — just SSH in and run your containers directly.

:::warning
**node01 (Control Node):** Docker containers using GPU or significant compute resources on node01 must still be launched inside a SLURM allocation. Docker processes running outside of SLURM on node01 will be automatically terminated.
:::

---

## Connecting to node02

Access to node02 is granted on request. Contact the system administrator to have your account enabled on node02. Once access is granted:

```bash
ssh USERNAME@192.168.220.76
```

Once logged in, you can run Docker containers directly without any SLURM steps.

:::note
The Docker daemon and image store are **local to each machine**. Since node01 and node02 have different hardware specifications, Docker images cannot be shared between them. If you have an image on node01, you will need to rebuild or re-pull it on node02.
:::

---

## Prerequisites

Docker usage requires membership in the `docker` group. Contact the system administrator to request access. You can verify your membership with:

```bash
groups $USER
```

If `docker` appears in the output, you are ready to go.

---

## Image Naming Convention

When building or pulling images, always **prefix the image name with your username**. This keeps the shared image list organized and makes it easy to identify who owns what.

```bash
# Building an image
docker build -t username_imagename:version .

# Examples
docker build -t username_mymodel:v1 .
docker build -t username_ros2:latest .
```

For pulled images, retag immediately after pulling and remove the original:

```bash
# After pulling, retag and remove the original
docker pull ros:humble
docker tag ros:humble username_ros:humble
docker rmi ros:humble
```

:::note
Docker image names must be **lowercase**. Use underscores as separators (e.g., `username_mymodel:v1`, not `Username_MyModel:v1`).
:::

:::warning
Images that do not follow the naming convention will be removed periodically, as the system administrator cannot identify their owner. Always prefix your images with your username to avoid losing your work.
:::

---

## Container Lifecycle: `--rm` vs Persistent

### Disposable Containers (Recommended Default)

Always use the `--rm` flag if you do not intend to reuse the container. This automatically removes the container when it exits, keeping the system clean.

```bash
# Container is removed after you exit
docker run -it --rm ubuntu:24.04 bash
```

### Persistent Containers

Omit `--rm` when you plan to stop and resume the container later — for example, a long-running development environment with installed packages or generated data you want to keep.

```bash
# Create and start a named container (no --rm)
docker run -it --name my_dev_env ubuntu:24.04 bash

# ... do some work, install packages, then exit
# The container still exists in a stopped state

# Restart and reattach later
docker start my_dev_env
docker attach my_dev_env
```

### Detaching from a Running Container

To leave a container running in the background without stopping it, use the detach key sequence:

```
Ctrl+P, Ctrl+Q
```

This returns you to the host shell while the container continues running. You can reattach later with `docker attach`.

:::warning
Typing `exit` or pressing `Ctrl+D` inside the container **stops** the container entirely (unless it has other processes keeping it alive). Use `Ctrl+P, Ctrl+Q` when you want the container to keep running.
:::

### Managing Persistent Containers

```bash
# List all containers (including stopped)
docker ps -a

# Remove a specific stopped container
docker rm my_dev_env

# Clean up all stopped containers
docker container prune
```

:::tip
Stopped containers still consume disk space. Periodically clean up containers you no longer need with `docker container prune`.
:::

---

## GPU Access in Docker

The NVIDIA Container Toolkit is pre-installed on node02. Use the `--gpus` flag to pass the GPU into a container.

```bash
# Verify GPU access inside a container
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

You can also select specific GPUs if needed:

```bash
# Use only GPU 0
docker run --rm --gpus '"device=0"' nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

---

## Running GUI Applications in Docker

To run graphical applications (e.g., RViz, Gazebo, CARLA) inside a Docker container, you need to forward the X11 display from the host.

```bash
docker run -it --rm \
    --privileged \
    --net=host \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=/tmp/.docker_xauth \
    -v $HOME/.Xauthority:/tmp/.docker_xauth:ro \
    osrf/ros:jazzy-desktop \
    bash
```

| Flag | Purpose |
|------|---------|
| `--privileged` | Grants extended permissions needed for display access |
| `--net=host` | Shares the host network stack (required for X11) |
| `-e DISPLAY=$DISPLAY` | Passes the host display variable into the container |
| `-e XAUTHORITY=/tmp/.docker_xauth` | Sets the X authority file path inside the container |
| `-v $HOME/.Xauthority:/tmp/.docker_xauth:ro` | Mounts the host X authority file as read-only |

---

## Combining GPU + GUI

For GPU-accelerated graphical workloads such as CARLA simulation, RViz, or Gazebo, combine the `--gpus` flag with the X11 forwarding flags:

```bash
docker run -it --rm \
    --gpus all \
    --privileged \
    --net=host \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=/tmp/.docker_xauth \
    -v $HOME/.Xauthority:/tmp/.docker_xauth:ro \
    my_image \
    bash
```

---

## SLURM + Docker on node01 (Control Node Only)

:::note
This workflow only applies if you are running Docker on **node01**. On node02, skip this entirely.
:::

On node01, Docker containers that use GPU or heavy compute must be launched inside a SLURM allocation.

### Step 1: Request Resources via `srun`

```bash
# Request shared GPU (2 shards), 8 CPUs, 32GB memory for 2 hours
# Note: Remember these numbers — you must use the same values in your docker run command
srun --gres=shard:2 --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash
```

### Step 2: Launch Docker with Matching Resource Limits

Inside the SLURM shell, launch your container with the same resource limits you requested:

```bash
# IMPORTANT: The --cpus and --memory values here must match what you requested in Step 1
#   SLURM --cpus-per-task=8  →  Docker --cpus=8
#   SLURM --mem=32G          →  Docker --memory=32g
docker run -it --rm \
    --gpus all \
    --cpus=8 \
    --memory=32g \
    --privileged \
    --net=host \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=/tmp/.docker_xauth \
    -v $HOME/.Xauthority:/tmp/.docker_xauth:ro \
    my_image \
    bash
```

### Step 3: Release Resources When Done

```bash
exit  # exit Docker container
exit  # exit SLURM allocation
```

:::warning
This is enforced on node01. Docker containers using GPU or heavy compute outside of a SLURM allocation will be automatically terminated.
:::

---

## Be a Good Neighbor

node02 is a shared machine with no automatic resource arbitration for Docker. Please:

- Check with your team before starting long or heavy GPU workloads
- Avoid leaving idle containers running with `--gpus all`
- Clean up stopped containers and unused images periodically

---

## Docker Quick Reference

```bash
# Run a disposable container
docker run -it --rm ubuntu:24.04 bash

# Run with GPU access
docker run -it --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 bash

# Run with GUI (X11 forwarding)
docker run -it --rm --privileged --net=host \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=/tmp/.docker_xauth \
    -v $HOME/.Xauthority:/tmp/.docker_xauth:ro \
    osrf/ros:jazzy-desktop bash

# Run with GPU + GUI
docker run -it --rm --gpus all --privileged --net=host \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=/tmp/.docker_xauth \
    -v $HOME/.Xauthority:/tmp/.docker_xauth:ro \
    my_image bash

# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Stop a running container
docker stop <container_name>

# Remove stopped containers
docker container prune

# Restart and attach to a stopped container
docker start <container_name>
docker attach <container_name>

# Detach from a running container without stopping it
# Press Ctrl+P, then Ctrl+Q (keyboard shortcut, not a command)

# Reattach to a running container
docker attach <container_name>

# List downloaded images
docker images

# Remove an image
docker rmi <image_name>
```
