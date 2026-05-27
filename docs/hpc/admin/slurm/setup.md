---
title: SLURM Setup
sidebar_label: Setup
sidebar_position: 1
---

# SLURM Setup Guide

This document covers the SLURM installation and configuration for shared use of the cluster workstation.

---

## System Overview

| Component | Specification |
|-----------|---------------|
| CPU       | AMD Ryzen Threadripper PRO 9985WX — 64 cores / 128 threads @ up to 5.5 GHz |
| RAM       | 256 GB (8× 32GB DIMMs) |
| GPU       | NVIDIA RTX PRO 6000 Blackwell — ~96 GB VRAM |
| Storage   | Samsung 990 PRO 4TB NVMe |
| OS        | Ubuntu 24.04 |
| Hostname  | node01       |

---

## Prerequisites

Before starting, verify your hardware is detected correctly:

```bash
# CPU info
lscpu

# Memory
free -h

# GPU
nvidia-smi

# All hardware
sudo lshw -short
```

---

## Step 1: Install SLURM Packages

```bash
sudo apt update
sudo apt install slurm-wlm slurm-wlm-doc munge libmunge-dev
```

---

## Step 2: Configure Munge Authentication

Munge provides authentication between SLURM components.

```bash
# Generate munge key
sudo /usr/sbin/mungekey

# Fix permissions
sudo chown munge:munge /etc/munge/munge.key
sudo chmod 400 /etc/munge/munge.key

# Enable and start munge
sudo systemctl enable munge
sudo systemctl start munge

# Verify munge is working
munge -n | unmunge
```

If successful, you'll see output showing the credential was successfully decoded.

---

## Step 3: Create SLURM Configuration Files

### 3.1 Main Configuration (`slurm.conf`)

```bash
sudo vim /etc/slurm/slurm.conf
```

Paste the following configuration:

```bash
# ==============================================================================
# SLURM Configuration for Exxact Workstation
# ==============================================================================

# Cluster identification
ClusterName=node01

# Controller configuration
SlurmctldHost=node01
SlurmUser=slurm

# ------------------------------------------------------------------------------
# Paths and PIDs
# ------------------------------------------------------------------------------
SlurmctldPidFile=/run/slurmctld.pid
SlurmdPidFile=/run/slurmd.pid
SlurmdSpoolDir=/var/spool/slurmd
StateSaveLocation=/var/spool/slurmctld

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
SlurmctldDebug=info
SlurmdDebug=info

# ------------------------------------------------------------------------------
# Process Tracking and Task Management
# ------------------------------------------------------------------------------
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup,task/affinity

# ------------------------------------------------------------------------------
# Scheduling
# ------------------------------------------------------------------------------
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory

# ------------------------------------------------------------------------------
# GPU Support
# ------------------------------------------------------------------------------
GresTypes=gpu

# ------------------------------------------------------------------------------
# Job Defaults and Limits
# ------------------------------------------------------------------------------
DefMemPerCPU=2000
MaxJobCount=5000
MaxArraySize=10000

# Enforce time limit for better backfill scheduling
EnforcePartLimits=ALL

# ------------------------------------------------------------------------------
# Timeouts
# ------------------------------------------------------------------------------
SlurmctldTimeout=300
SlurmdTimeout=300
InactiveLimit=0
MinJobAge=300
KillWait=30
Waittime=0

# ------------------------------------------------------------------------------
# Node Definition
# ------------------------------------------------------------------------------
# AMD Ryzen Threadripper PRO 9985WX (64 cores / 128 threads)
# 256 GB RAM
# 1x NVIDIA RTX PRO 6000 Blackwell (~96GB VRAM)
NodeName=node01 \
      CPUs=128 \
      Boards=1 \
      SocketsPerBoard=1 \
      CoresPerSocket=64 \
      ThreadsPerCore=2 \
      RealMemory=250000 \
      Gres=gpu:rtx6000:1 \
      State=UNKNOWN

# ------------------------------------------------------------------------------
# Partition
# ------------------------------------------------------------------------------
# Single partition - users should always specify --time for efficient scheduling
# Max 7 days, default 30 minutes if not specified
PartitionName=main \
      Nodes=node01 \
      Default=YES \
      MaxTime=7-00:00:00 \
      DefaultTime=00:30:00 \
      State=UP
```

**Key configuration notes:**

- `SlurmUser=slurm` — Required to avoid UID mismatch errors
- `RealMemory=250000` — Reserves ~6GB for OS overhead from 256GB total
- `DefaultTime=00:30:00` — Jobs without `--time` get 30 minutes (prevents infinite jobs)
- `MaxTime=7-00:00:00` — Maximum job duration is 7 days
- Backfill scheduler automatically prioritizes shorter jobs when resources are available

### 3.2 GPU Configuration (`gres.conf`)

```bash
sudo vim /etc/slurm/gres.conf
```

```bash
# GPU: NVIDIA RTX PRO 6000 Blackwell
Name=gpu Type=rtx6000 File=/dev/nvidia0
```

:::tip Adding More GPUs
When adding more GPUs to the same node, update `gres.conf` with additional entries and update the node definition in `slurm.conf`:

```bash
# Example for 6 GPUs
Name=gpu Type=rtx6000 File=/dev/nvidia0
Name=gpu Type=rtx6000 File=/dev/nvidia1
Name=gpu Type=rtx6000 File=/dev/nvidia2
Name=gpu Type=rtx6000 File=/dev/nvidia3
Name=gpu Type=rtx6000 File=/dev/nvidia4
Name=gpu Type=rtx6000 File=/dev/nvidia5
```

Then update the node definition in `slurm.conf`:
```bash
Gres=gpu:rtx6000:6
```
:::

### 3.3 Cgroup Configuration (`cgroup.conf`)

```bash
sudo vim /etc/slurm/cgroup.conf
```

```bash
ConstrainCores=yes
ConstrainRAMSpace=yes
ConstrainDevices=yes
```

:::warning
Do **NOT** include `CgroupAutomount=yes` — this option is defunct in Ubuntu 24.04's SLURM version and will cause errors.
:::

---

## Step 4: Create Required Directories

```bash
sudo mkdir -p /var/spool/slurmd
sudo mkdir -p /var/spool/slurmctld
sudo mkdir -p /var/log/slurm

sudo chown slurm:slurm /var/spool/slurmd
sudo chown slurm:slurm /var/spool/slurmctld
sudo chown slurm:slurm /var/log/slurm
```

---

## Step 5: Configure Hostname Resolution

Ensure the hostname resolves correctly:

```bash
hostname
getent hosts node01
```

If the hostname doesn't resolve, add it to `/etc/hosts`:

```bash
echo "127.0.1.1 node01" | sudo tee -a /etc/hosts
```

:::warning
The head node hostname must **NOT** resolve to `127.0.1.1` if you are running a multi-node cluster. Replace it with the real network IP instead. See [Adding a Node](./adding-a-node) for details.
:::

---

## Step 6: Start SLURM Services

```bash
# Enable and start controller
sudo systemctl enable slurmctld
sudo systemctl start slurmctld

# Enable and start compute daemon
sudo systemctl enable slurmd
sudo systemctl start slurmd
```

---

## Step 7: Verify Installation

```bash
# Check node status
sinfo

# Expected output:
# PARTITION AVAIL TIMELIMIT NODES STATE NODELIST
# main*     up    7-00:00:00    1  idle node01

# Check detailed node info
scontrol show node node01

# Test a simple job
srun hostname

# Test GPU access
srun --gres=gpu:1 nvidia-smi
```

---

## Troubleshooting

### Check Service Status and Logs

```bash
# Service status
sudo systemctl status slurmctld
sudo systemctl status slurmd
sudo systemctl status munge

# Recent logs
sudo tail -50 /var/log/slurm/slurmctld.log
sudo tail -50 /var/log/slurm/slurmd.log
```

### Node is DOWN After Reboot

**Symptom:**
```
PARTITION AVAIL TIMELIMIT NODES STATE NODELIST
main*     up    7-00:00:00    1  down node01
```

**Cause:** SLURM automatically marks nodes as DOWN after unexpected reboots as a safety precaution.

**Solution:** If the node is healthy, resume it:
```bash
sudo scontrol update nodename=node01 state=resume
```

**Verify:**
```bash
sinfo
srun hostname
```

### "can't stat gres.conf file /dev/nvidia0: No such file or directory"

**Symptom** (in `/var/log/slurm/slurmd.log`):
```
error: Waiting for gres.conf file /dev/nvidia0
fatal: can't stat gres.conf file /dev/nvidia0: No such file or directory
```

**Cause:** `slurmd` started before the NVIDIA driver finished loading during boot.

**Immediate fix:**
```bash
sudo systemctl restart slurmd
sudo scontrol update nodename=node01 state=resume
```

**Permanent fix:** Add a systemd dependency so `slurmd` waits for the NVIDIA driver:
```bash
sudo systemctl edit slurmd
```

Add:
```ini
[Unit]
After=nvidia-persistenced.service
Wants=nvidia-persistenced.service
```

### "cred/munge: Unexpected uid"

**Symptom:**
```
error: cred/munge: Unexpected uid (64030) != Slurm uid (0)
```

**Solution:** Add `SlurmUser=slurm` to `slurm.conf` after `SlurmctldHost`.

### "CgroupAutomount is defunct"

**Symptom:**
```
error: The option "CgroupAutomount" is defunct, please remove it from cgroup.conf.
```

**Solution:** Remove `CgroupAutomount=yes` from `cgroup.conf`.

### "Header lengths are longer than data received"

**Symptom:**
```
srun: error: Task launch for StepId=X.0 failed on node: Header lengths are longer than data received
```

**Solution:** Usually indicates a version mismatch or cgroup issues. Verify all components are the same version:
```bash
slurmctld -V
slurmd -V
srun -V
```

### Hostname Not Resolving

**Solution:** Add to `/etc/hosts`:
```bash
echo "127.0.1.1 $(hostname)" | sudo tee -a /etc/hosts
```

---

## Restart Services After Configuration Changes

```bash
sudo systemctl restart slurmctld
sudo systemctl restart slurmd
```

---

## Quick Reference

| File | Location | Purpose |
|------|----------|---------|
| `slurm.conf` | `/etc/slurm/slurm.conf` | Main SLURM configuration |
| `gres.conf` | `/etc/slurm/gres.conf` | GPU resource definitions |
| `cgroup.conf` | `/etc/slurm/cgroup.conf` | Resource isolation settings |
| `munge.key` | `/etc/munge/munge.key` | Authentication key |

| Directory | Purpose |
|-----------|---------|
| `/var/spool/slurmd` | Slurmd spool directory |
| `/var/spool/slurmctld` | Controller state files |
| `/var/log/slurm/` | SLURM log files |

```bash
# Check cluster status
sinfo

# Show node details
scontrol show node node01

# Show partition details
scontrol show partition main

# View running/pending jobs
squeue

# Restart after config changes
sudo systemctl restart slurmctld && sudo systemctl restart slurmd
```
