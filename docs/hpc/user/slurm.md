---
title: SLURM Tutorial
sidebar_label: Tutorial
sidebar_position: 3
---

# SLURM User Guide

This guide covers how to use SLURM to submit and manage compute jobs on the cluster.

---

## Cluster Overview

The cluster consists of two nodes:

| Resource | Control Node (node01) | Compute Node (node02) |
|----------|-----------------------|-----------------------|
| CPU | Threadripper PRO 9985WX — 128 threads | Ryzen 9 9950X3D — 32 threads |
| RAM | 256 GB | 128 GB |
| GPU | 2× RTX PRO 6000 Blackwell Max-Q (~96 GB VRAM each, ~192 GB total) | RTX 5000 Ada (~32 GB VRAM) |
| Shards | 16 shards (~12 GB each) | 2 shards (~16 GB each) |
| Feature tag | `large` | `small` |

| Setting | Value |
|---------|-------|
| Max job time | 7 days |
| Default job time | 30 minutes |

---

## GPU Allocation Policy

The GPU(s) can be allocated a few ways:

| Allocation | Command | Use Case |
|------------|---------|----------|
| Shared (default) | `--gres=shard:N` | Development, inference, small training jobs |
| Exclusive (single GPU) | `--gres=gpu:1` | Large training jobs requiring one full GPU |
| Exclusive (multi-GPU) | `--gres=gpu:2` | Multi-GPU parallel training (e.g. PyTorch DDP) needing both GPUs on node01 |

### Shards per Node

| Node | GPU | Shards | VRAM per Shard |
|------|-----|--------|----------------|
| node01 | 2× RTX PRO 6000 Blackwell Max-Q | 16 shards | ~12 GB each |
| node02 | RTX 5000 Ada | 2 shards | ~16 GB each |

:::warning
SLURM does **not** enforce VRAM limits per shard. If you exceed your allocation, your job may crash or affect other users. **Be a good citizen!**
:::

### GPU Request Guidelines

**node01 (2× RTX PRO 6000 Blackwell Max-Q — 96GB each):**

| VRAM Needed | Request |
|-------------|---------|
| < 12 GB | `--gres=shard:1` |
| 12–24 GB | `--gres=shard:2` |
| 24–48 GB | `--gres=shard:4` |
| 48–96 GB | `--gres=gpu:1` (one full GPU) |
| Multi-GPU parallel training (DDP, model/data parallel) | `--gres=gpu:2` (both GPUs, exclusive) |

:::note
`--gres=gpu:1` gives you exclusive access to **one** of node01's two GPUs (96GB) — it no longer means "the whole node" now that there are two cards. If your job needs to spread work across both GPUs simultaneously (e.g. `torchrun --nproc_per_node=2`), request `--gres=gpu:2` instead. See [Multi-GPU Parallel Training](#multi-gpu-parallel-training-ddp-both-gpus) below for a full example.
:::

**node02 (RTX 5000 Ada — 32GB):**

| VRAM Needed | Request |
|-------------|---------|
| < 16 GB | `--gres=shard:1` |
| 16–32 GB | `--gres=gpu:1` (full GPU) |

---

## Node Features and Constraints

Each node is tagged with a feature label to help target jobs:

| Node | Feature | Use For |
|------|---------|---------|
| node01 | `large` | Jobs needing more VRAM, CPU threads, or RAM |
| node02 | `small` | Lighter inference, testing, smaller training runs |

By default, SLURM auto-schedules jobs across both nodes based on your resource request. Use `--constraint` only when your job genuinely requires a specific node.

```bash
# Run on the larger node (node01)
#SBATCH --constraint=large

# Run on the smaller node (node02)
#SBATCH --constraint=small
```

:::tip
For most jobs, omit `--constraint` entirely and let SLURM decide. If you request `--gres=shard:4`, SLURM will automatically avoid node02 (which only has 2 shards) — no constraint needed.
:::

---

## `srun` vs `sbatch`

| Command | Use Case | Behavior |
|---------|----------|----------|
| `srun` | Interactive jobs, quick tests | Blocks terminal until job completes |
| `sbatch` | Production jobs, long runs | Submits and returns immediately |

---

## `srun` — Interactive Jobs

Run commands directly on the cluster. Your terminal waits for the job to finish.

### Basic Usage

```bash
# Run a simple command
srun hostname

# Run a Python script
srun python train.py

# Start an interactive shell
srun --pty bash
```

### Requesting Resources

```bash
# Request 4 CPUs and 16GB memory for 30 minutes
srun --cpus-per-task=4 --mem=16G --time=00:30:00 python train.py

# Request shared GPU (2 shards, ~24GB VRAM)
srun --gres=shard:2 nvidia-smi

# Request full GPU (exclusive access)
srun --gres=gpu:1 nvidia-smi

# Combine CPU, memory, shared GPU, and time
srun --cpus-per-task=8 --mem=32G --gres=shard:2 --time=02:00:00 python train.py
```

### Interactive GPU Session

```bash
# Get a shell with shared GPU access (2 shards) for 2 hours
srun --gres=shard:2 --mem=32G --time=02:00:00 --pty bash

# Now you can run commands interactively
nvidia-smi
python train.py
exit  # release resources when done

# For large jobs needing full GPU
srun --gres=gpu:1 --mem=64G --time=04:00:00 --pty bash

# For multi-GPU jobs needing both GPUs on node01
srun --gres=gpu:2 --ntasks=2 --mem=128G --time=04:00:00 --pty bash
```

---

## `sbatch` — Batch Jobs

Submit jobs that run in the background. Use this for production workloads.

### Basic Job Script

Create a file called `job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Your commands here
echo "Job started on $(hostname)"
python train.py
echo "Job finished"
```

Submit it:

```bash
sbatch job.sh
```

### GPU Job Script (Shared GPU)

For most GPU jobs, use shards:

```bash
#!/bin/bash
#SBATCH --job-name=gpu_training
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=shard:2

# Load environment
source ~/venvs/myenv/bin/activate

# Run training
python train.py --epochs 100 --batch-size 32
echo "Training complete"
```

### GPU Job Script (Full GPU)

For large models requiring full GPU:

```bash
#!/bin/bash
#SBATCH --job-name=large_training
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --constraint=large
#SBATCH --gres=gpu:1

# Load environment
source ~/venvs/myenv/bin/activate

# Run large model training
python train.py --model large --batch-size 128
echo "Training complete"
```

Submit:

```bash
mkdir -p logs  # create logs directory first
sbatch job.sh
```

### GPU Job Script (Multi-GPU, Both GPUs)

For PyTorch DistributedDataParallel or other multi-GPU parallel training that needs **both** GPUs on node01 simultaneously:

```bash
#!/bin/bash
#SBATCH --job-name=ddp_training
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --constraint=large
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2

# Load environment
source ~/venvs/myenv/bin/activate

# torchrun spawns one process per GPU
torchrun --nproc_per_node=2 train_ddp.py --epochs 100 --batch-size 256
echo "Training complete"
```

:::note
`--gres=gpu:2` reserves both physical GPUs exclusively for the duration of the job — no other user's `shard` or `gpu` jobs can use either card until it finishes. Only request this when your job is actually structured to use both GPUs (e.g. via `torchrun`, `accelerate`, or `deepspeed`); otherwise use `--gres=gpu:1` or a shard request.
:::

Submit:

```bash
mkdir -p logs
sbatch job.sh
```

### Passing Arguments to Job Scripts

```bash
#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --output=logs/%x_%j.log
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

# $1, $2, etc. are command-line arguments
python train.py --lr $1 --epochs $2
```

Submit with arguments:

```bash
sbatch job.sh 0.001 50
```

### Common `#SBATCH` Options

| Option | Description | Example |
|--------|-------------|---------|
| `--job-name` | Job name (shows in queue) | `--job-name=training` |
| `--output` | Stdout file (`%j`=job ID, `%x`=job name) | `--output=logs/%x_%j.log` |
| `--error` | Stderr file | `--error=logs/%x_%j.err` |
| `--time` | Time limit (`HH:MM:SS` or `D-HH:MM:SS`) | `--time=04:00:00` |
| `--cpus-per-task` | Number of CPU threads | `--cpus-per-task=8` |
| `--mem` | Total memory | `--mem=32G` |
| `--gres=shard:N` | Shared GPU (N shards) | `--gres=shard:2` |
| `--gres=gpu:1` | One full GPU (exclusive access) | `--gres=gpu:1` |
| `--gres=gpu:2` | Both GPUs on node01 (exclusive, multi-GPU) | `--gres=gpu:2` |
| `--constraint` | Target node by capability | `--constraint=large` |

---

## Monitoring Jobs

### Check Queue Status

```bash
# View all jobs
squeue

# View only your jobs
squeue -u $USER

# Detailed job info
squeue -l
```

### Check Cluster Status

```bash
# Node availability
sinfo

# Detailed view with node list
sinfo -N -l

# Detailed node info (includes Feature tags)
scontrol show node node01
scontrol show node node02
```

### Check Job Details

```bash
# While job is running or pending
scontrol show job <job_id>

# After job completes (accounting info)
# Note: This feature is currently turned off. Please request if you really need it.
sacct -j <job_id> --format=JobID,JobName,Elapsed,State,MaxRSS,MaxVMSize
```

### Cancel Jobs

```bash
# Cancel a specific job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER

# Cancel all pending jobs
scancel -u $USER --state=pending
```

---

## Job Arrays

Run the same script with different parameters:

```bash
#!/bin/bash
#SBATCH --job-name=array_job
#SBATCH --output=logs/array_%A_%a.log
#SBATCH --array=1-10
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4

# SLURM_ARRAY_TASK_ID contains the array index (1, 2, 3, ... 10)
echo "Running task $SLURM_ARRAY_TASK_ID"
python experiment.py --seed $SLURM_ARRAY_TASK_ID
```

Submit:

```bash
sbatch array_job.sh  # submits 10 jobs
```

Useful array patterns:

```bash
#SBATCH --array=1-100        # 1 to 100
#SBATCH --array=1-100%10     # 1 to 100, max 10 running at once
#SBATCH --array=1,3,5,7      # specific values
#SBATCH --array=1-10:2       # 1,3,5,7,9 (step of 2)
```

---

## Targeting Specific Nodes

By default, SLURM automatically schedules jobs to the best available node. Use `--constraint` to target a specific node capability.

### Using `--constraint` (Recommended)

```bash
# Auto-schedule — SLURM decides (preferred for most jobs)
srun --gres=shard:2 --time=02:00:00 --pty bash

# Target the larger node (node01) by capability
srun --constraint=large --gres=shard:2 --time=02:00:00 --pty bash

# Target the smaller node (node02) by capability
srun --constraint=small --gres=shard:1 --time=02:00:00 --pty bash
```

In a batch script:

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --constraint=large   # target node01 by capability
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

python train.py
```

### Using `--nodelist` / `-w` (Admin/Debug Use Only)

Use `--nodelist` only when you need to pin to a specific hostname, such as for debugging a node-specific issue.

```bash
# Force job on node01 by hostname
srun -w node01 --gres=shard:2 --time=02:00:00 --pty bash

# Force job on node02 by hostname
srun -w node02 --gres=shard:1 --time=02:00:00 --pty bash
```

:::warning
Avoid hard-coding `--nodelist=node01` in production scripts. If a node is replaced or renamed, your scripts will break. Use `--constraint` instead.
:::

---

## Best Practices

### Use Shards by Default

```bash
# Good — use shards for most work
#SBATCH --gres=shard:2

# Only for large models needing >48GB VRAM on node01
#SBATCH --gres=gpu:1
```

### Let SLURM Schedule Unless You Have a Reason

```bash
# Good — SLURM picks the best available node
#SBATCH --gres=shard:2

# Only add constraint if your job truly needs the larger node
#SBATCH --constraint=large
#SBATCH --gres=shard:4
```

### Always Specify `--time`

Helps the scheduler run shorter jobs sooner:

```bash
# Good — scheduler knows job length
srun --time=00:30:00 python quick_test.py

# Less optimal — defaults to 30 minutes even if job takes 5 minutes
srun python quick_test.py
```

### Request Only What You Need

Over-requesting blocks resources from others:

```bash
# Bad — requesting all resources
#SBATCH --cpus-per-task=128
#SBATCH --mem=250G
#SBATCH --gres=gpu:1

# Good — request what you actually use
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=shard:2
```

Likewise, only request `--gres=gpu:2` when your job is genuinely structured to use both GPUs — it blocks both cards from every other user until it finishes.

### Test Interactively First

```bash
# Get interactive session with shared GPU
srun --gres=shard:2 --time=00:30:00 --pty bash

# Test your code works
python train.py --epochs 1

# Exit and submit real job
exit
sbatch job.sh
```

---

## Example: PyTorch Training Job

### Standard Training (Shared GPU, Auto-Scheduled)

```bash
#!/bin/bash
#SBATCH --job-name=pytorch_train
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=shard:2

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Setup environment
source ~/venvs/pytorch/bin/activate

# Run training
python train.py \
    --model resnet50 \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --output-dir results/$SLURM_JOB_ID

echo "End time: $(date)"
```

### Large Model Training (Full GPU, Large Node Required)

```bash
#!/bin/bash
#SBATCH --job-name=large_model_train
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --constraint=large
#SBATCH --gres=gpu:1

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Setup environment
source ~/venvs/pytorch/bin/activate

# Run large model training
python train.py \
    --model vit_large \
    --epochs 50 \
    --batch-size 256 \
    --learning-rate 0.0001 \
    --output-dir results/$SLURM_JOB_ID

echo "End time: $(date)"
```

### Multi-GPU Parallel Training (DDP, Both GPUs)

For models that need to be split across both GPUs, or that train faster with data-parallel DDP across both cards:

```bash
#!/bin/bash
#SBATCH --job-name=ddp_train
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --constraint=large
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Setup environment
source ~/venvs/pytorch/bin/activate

# torchrun spawns one training process per GPU and handles
# the DDP process group setup (rank, world size, etc.) automatically
torchrun --nproc_per_node=2 train_ddp.py \
    --model resnet50 \
    --epochs 100 \
    --batch-size 256 \
    --learning-rate 0.001 \
    --output-dir results/$SLURM_JOB_ID

echo "End time: $(date)"
```

:::tip
Test your DDP setup interactively first with a short run before submitting a long `sbatch` job — see [Test Interactively First](#test-interactively-first) above, using `srun --gres=gpu:2 --ntasks=2 --pty bash` instead of the shard version.
:::

---

## Quick Reference

```bash
# Interactive session with shared GPU (recommended)
srun --gres=shard:2 --mem=32G --time=02:00:00 --pty bash

# Interactive session with one full GPU (large jobs only)
srun --gres=gpu:1 --mem=64G --time=04:00:00 --pty bash

# Interactive session with both GPUs (multi-GPU DDP testing)
srun --gres=gpu:2 --ntasks=2 --mem=128G --time=04:00:00 --pty bash

# Interactive session on the larger node specifically
srun --constraint=large --gres=shard:2 --time=02:00:00 --pty bash

# Submit batch job
sbatch job.sh

# Check your jobs
squeue -u $USER

# Check node and GPU availability
sinfo -N -l

# Cancel a job
scancel <job_id>

# Sync files to node02 before running there
rsync -avz ~/myproject node02:~/

# View job output in real-time
tail -f logs/my_job_12345.log
```

---

## Getting Help

```bash
# Manual pages
man srun
man sbatch
man squeue

# Quick help
srun --help
sbatch --help
```
