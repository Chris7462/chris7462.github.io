---
title: Getting Started
sidebar_label: Getting Started
sidebar_position: 1
---

# Getting Started

This document covers how to connect to the HPC cluster and get oriented with the available resources.

---

## Prerequisites

- You must be connected to the **internal network** before accessing the cluster
- Your account must be created by the system administrator before you can log in
- Contact the system administrator to request an account

---

## Cluster Overview

The cluster consists of two workstations and a shared NAS storage:

| Node | IP Address | Role | Use For |
|------|------------|------|---------|
| Control Node (node01) | 192.168.220.75 | SLURM controller + compute | Batch jobs, CPU compute, SLURM submissions |
| Compute Node (node02) | 192.168.220.76 | SLURM compute + Docker | Docker containers, GPU workloads |
| NAS | 192.168.220.80 | Shared storage | Home directories (`/home`) |

:::note
**node02** is the dedicated Docker machine. You can SSH directly into node02 and run Docker containers freely without SLURM. See the [Docker](../docker/index.md) guide for details.

**node01** enforces SLURM for compute jobs. Any non-SLURM process running longer than 5 minutes will be automatically terminated.
:::

---

## Connecting via SSH

### Linux / macOS

```bash
ssh USERNAME@192.168.220.75
```

### Windows

Use [MobaXterm](https://mobaxterm.mobatek.net/) or [PuTTY](https://www.putty.org/) for SSH connections.

---

## Running GUI Applications (X11 Forwarding)

If you need to run graphical applications (e.g., `rviz2`) remotely, use X11 forwarding:

```bash
# Enable server access control on your local machine (first time only)
xhost +

# Connect with X11 forwarding
ssh -X USERNAME@192.168.220.75

# Test X11 forwarding
rviz2
```

If configured correctly, the GUI application will appear on your local machine while running on the remote server.

:::tip
For better performance with OpenGL applications like `rviz2` or Gazebo, use VirtualGL instead of plain X11 forwarding. See the [VirtualGL](../../admin/virtualgl/index.md) guide for setup details.
:::

---

## What to Do Next

Once logged in:

- Submit compute jobs through SLURM — see the [SLURM Tutorial](../slurm/index.md)
- Run Docker containers on node02 — see the [Docker](../docker/index.md) guide
- Transfer files to/from the cluster — see the [File Transfer](../file-transfer/index.md) guide

:::warning
**SLURM usage is mandatory for compute jobs on node01.** Any non-SLURM compute process running longer than 5 minutes will be automatically terminated. Always use `srun` or `sbatch` to submit jobs.
:::
