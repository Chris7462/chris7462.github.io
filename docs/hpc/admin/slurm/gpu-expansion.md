---
title: "Adding a Second GPU"
sidebar_label: "GPU Expansion"
sidebar_position: 4
---

# Adding a Second GPU to node01 (RTX PRO 6000 Blackwell Max-Q)

> This page documents the full process of adding a second NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition GPU to the **node01** control node, including hardware verification, driver/CUDA compatibility checks, SLURM reconfiguration, and Grafana/Prometheus monitoring updates.

## 1. Overview

A second NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition GPU (identical to the existing card) was installed in **node01**, bringing the node from 1 GPU to 2. The goal was to support PyTorch DistributedDataParallel (DDP) training jobs that require exclusive access to both GPUs simultaneously, while preserving the existing GPU sharding setup for smaller dev/inference workloads.

Summary of changes:
- Physical card installed and detected at the OS level
- No driver or CUDA Toolkit changes required
- SLURM `gres.conf` and `slurm.conf` updated to expose both GPUs (exclusive `gpu` Gres and shared `shard` Gres)
- Grafana dashboard restructured to show per-GPU panels instead of a single merged panel

## 2. Hardware Verification

### 2.1 Physical Installation

Standard procedure: power down **node01**, seat the second RTX PRO 6000 Blackwell Max-Q card in the available PCIe slot, connect power cables, boot.

### 2.2 Confirming Detection

```bash
lspci | grep -i nvidia
```

```
11:00.0 VGA compatible controller: NVIDIA Corporation Device 2bb4 (rev a1)
11:00.1 Audio device: NVIDIA Corporation Device 22e8 (rev a1)
f1:00.0 VGA compatible controller: NVIDIA Corporation Device 2bb4 (rev a1)
f1:00.1 Audio device: NVIDIA Corporation Device 22e8 (rev a1)
```

```bash
nvidia-smi
```

```
Tue Jun 16 16:58:33 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.05              Driver Version: 595.71.05      CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX PRO 6000 Blac...    On  |   00000000:11:00.0 Off |                  Off |
| 30%   32C    P8             16W /  300W |      15MiB /  97887MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA RTX PRO 6000 Blac...    On  |   00000000:F1:00.0 Off |                  Off |
| 30%   34C    P8              7W /  300W |     122MiB /  97887MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

Both GPUs detected correctly, on separate PCIe root complexes (`11:00.0` and `f1:00.0`), consistent with the Threadripper PRO's multi-channel PCIe topology.

## 3. Driver & CUDA Compatibility Check

No driver update was required, since the existing driver (595.71.05) already supports Blackwell architecture.

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

```
name, compute_cap
NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 12.0
NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 12.0
```

```bash
nvcc --version
```

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```

```bash
python3 -c "import torch; print(torch.cuda.get_device_capability(0)); print(torch.cuda.get_device_capability(1))"
```

```
(12, 0)
(12, 0)
```

Both GPUs report compute capability `sm_120`, CUDA Toolkit is at 13.2 matching the driver's max supported version, and PyTorch correctly recognizes both cards. No toolkit, driver, or framework updates were needed.

## 4. SLURM Configuration

### 4.1 Design Decision: Exclusive vs Shared GPU Access

Large training jobs (e.g. PyTorch DistributedDataParallel) require exclusive, dedicated access to both physical GPUs simultaneously — NCCL communication between DDP processes assumes uncontested bandwidth and memory, so shard-based partial access is not viable for these jobs.

Smaller development and inference workloads continue to use the existing GPU sharding setup.

| Resource | Use Case | Example Command |
|---|---|---|
| `gpu:2` | Large training jobs (PyTorch DDP) needing both GPUs exclusively | `srun --gres=gpu:2 --ntasks=2 torchrun --nproc_per_node=2 train.py` |
| `gpu:1` | Single-GPU exclusive jobs | `srun --gres=gpu:1 python train.py` |
| `shard:N` | Smaller/dev/inference jobs, shared access | `srun --gres=shard:2 python inference.py` |

:::note
Since `gpu` and `shard` Gres types reference the same underlying device files, SLURM prevents a `shard` job from landing on a GPU that is currently held exclusively by a `gpu` job, and vice versa. Users submitting small `shard` jobs may queue behind a large exclusive training run.
:::

### 4.2 gres.conf Changes

**Before:**
```
# Full GPU - exclusive access
# Use: srun --gres=gpu:1 (for large training jobs needing full GPU)
Name=gpu Type=rtx6000 File=/dev/nvidia0

# GPU Shards - shared access
# Use: srun --gres=shard:2 (for smaller jobs, development, inference)
# 8 shards available, each represents ~12GB VRAM (96GB / 8)
# Note: VRAM is not enforced by SLURM, users must manage memory usage
Name=shard Count=8 File=/dev/nvidia0
```

**After:**
```
# ==============================================================================
# GRES (Generic Resources) Configuration
# ==============================================================================
# Full GPU - exclusive access
# Use: srun --gres=gpu:2 (for large training jobs needing both GPUs, e.g. PyTorch DDP)
# Use: srun --gres=gpu:1 (for single-GPU exclusive jobs)
Name=gpu Type=rtx6000 File=/dev/nvidia0
Name=gpu Type=rtx6000 File=/dev/nvidia1

# GPU Shards - shared access
# Use: srun --gres=shard:2 (for smaller jobs, development, inference)
# 16 shards available, each represents ~12GB VRAM (192GB / 16)
# Note: VRAM is not enforced by SLURM, users must manage memory usage
Name=shard Count=8 File=/dev/nvidia0
Name=shard Count=8 File=/dev/nvidia1
```

### 4.3 slurm.conf Changes

**Before:**
```
NodeName=node01 \
    CPUs=128 \
    Boards=1 \
    SocketsPerBoard=1 \
    CoresPerSocket=64 \
    ThreadsPerCore=2 \
    RealMemory=250000 \
    Gres=gpu:rtx6000:1,shard:8 \
    Feature=large \
    CoreSpecCount=1 \
    State=UNKNOWN
```

**After:**
```
NodeName=node01 \
    CPUs=128 \
    Boards=1 \
    SocketsPerBoard=1 \
    CoresPerSocket=64 \
    ThreadsPerCore=2 \
    RealMemory=250000 \
    Gres=gpu:rtx6000:2,shard:16 \
    Feature=large \
    CoreSpecCount=1 \
    State=UNKNOWN
```

Only the `Gres` line changed. CPU, RAM, Feature, and CoreSpecCount settings were unaffected since only the GPU count changed. The `node02` node definition was untouched.

### 4.4 Restart Services

```bash
sudo systemctl restart slurmctld
sudo systemctl restart slurmd
```

### 4.5 Troubleshooting: Stale Gres Count Drain

After restarting, the node was automatically marked `DOWN`/`DRAIN` because the reported GPU count temporarily mismatched the configured count:

```bash
scontrol show node node01
```

```
NodeName=node01 Arch=x86_64 CoresPerSocket=64
   ...
   Gres=gpu:rtx6000:2,shard:rtx6000:16
   ...
   State=IDLE+DRAIN ThreadsPerCore=2 TmpDisk=0 Weight=1 Owner=N/A MCS_label=N/A
   ...
   Reason=gres/gpu count reported lower than configured (1 < 2) [slurm@2026-06-16T23:07:44]
```

**Resolution:** a clean `slurmd` restart followed by a manual resume cleared the issue:

```bash
sudo systemctl restart slurmd
sleep 3
sudo scontrol update nodename=node01 state=resume
```

:::warning
After resuming, the `Reason` field may continue to display the old drain message even though the node is healthy and `IDLE`. This is cosmetic — SLURM does not always clear the `Reason` text automatically on resume. It can be manually cleared with:

```bash
sudo scontrol update nodename=node01 reason=""
```
:::

### 4.6 Verification

```bash
srun --gres=gpu:2 -w node01 nvidia-smi
```

```
+-----------------------------------------+------------------------+----------------------+
|   0  NVIDIA RTX PRO 6000 Blac...    On  |   00000000:11:00.0 Off |                  Off |
| 30%   31C    P8             16W /  300W |      15MiB /  97887MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA RTX PRO 6000 Blac...    On  |   00000000:F1:00.0 Off |                  Off |
| 30%   33C    P8              8W /  300W |     122MiB /  97887MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
```

Both GPUs successfully allocated and visible inside a SLURM-scheduled job, confirming the `gpu:2` request works end-to-end.

To view Gres allocation in `sinfo` (not shown by default):
```bash
sinfo -o "%N %G"
```

## 5. Monitoring (Grafana / Prometheus) Updates

### 5.1 Exporter Verification — No Config Changes Needed

The `nvidia_gpu_exporter` uses `nvidia-smi` under the hood and automatically reports all GPUs visible on the host. No changes to the Prometheus scrape target or exporter configuration were required — the existing `node01:9835` target picked up both GPUs automatically.

```bash
curl -s http://localhost:9835/metrics | grep -E "^nvidia_smi_utilization_gpu_ratio|^nvidia_smi_temperature_gpu"
```

```
nvidia_smi_temperature_gpu{uuid="23b4975b-2171-ea56-f5f7-f78bf0cd300a"} 31
nvidia_smi_temperature_gpu{uuid="d0b27d44-e215-eeda-b053-6a2342e854c5"} 32
nvidia_smi_utilization_gpu_ratio{uuid="23b4975b-2171-ea56-f5f7-f78bf0cd300a"} 0
```

### 5.2 UUID Mapping

```bash
nvidia-smi --query-gpu=index,uuid --format=csv
```

```
index, uuid
0, GPU-23b4975b-2171-ea56-f5f7-f78bf0cd300a
1, GPU-d0b27d44-e215-eeda-b053-6a2342e854c5
```

| GPU Index | PCIe Address | UUID (exporter format, no prefix) |
|---|---|---|
| 0 | 11:00.0 | `23b4975b-2171-ea56-f5f7-f78bf0cd300a` |
| 1 | F1:00.0 | `d0b27d44-e215-eeda-b053-6a2342e854c5` |

:::warning Gotcha
`nvidia-smi --query-gpu=uuid` reports UUIDs with a `GPU-` prefix (e.g. `GPU-23b4975b-...`), but the `nvidia_gpu_exporter` metrics label strips this prefix (e.g. `23b4975b-...`). PromQL filters using the prefixed UUID will silently return no data. Always verify the exact label format directly from the exporter's `/metrics` endpoint before writing dashboard queries.
:::

### 5.3 Dashboard Restructuring

The original dashboard had single gauge/stat panels per metric (GPU Utilization, GPU Memory Usage, GPU Temperature, GPU Power Draw, GPU Memory Used, GPU Memory Total), each querying `instance="node01:9835"` without filtering by GPU. With two GPUs reporting under the same instance label, these merged into cramped multi-series panels with unreadable raw-UUID labels.

The dashboard was restructured so each GPU has its own dedicated block of panels, filtered explicitly by UUID, with clean "GPU 0" / "GPU 1" labels. Layout per GPU (repeated for GPU 0 then GPU 1, stacked vertically):

| Position | Panel | Type |
|---|---|---|
| Top-left (spans both rows) | Utilization | Gauge |
| Top-center-left (spans both rows) | Memory Usage | Gauge |
| Top-right column 1, row 1 | Memory Used | Stat |
| Top-right column 2, row 1 | Temperature | Stat |
| Top-right column 1, row 2 | Memory Total (blue) | Stat |
| Top-right column 2, row 2 | Power Draw | Stat |

Time series panels (GPU Utilization Over Time, GPU Memory Over Time) were kept as combined multi-line panels, since time series visualizations handle multiple series natively, with legends updated to "GPU 0" / "GPU 1" instead of raw UUIDs.

Additionally, all "Total" stat panels across the dashboard (Disk Total, NAS Total, GPU Memory Total, Total RAM, Total CPU Cores) were standardized to a blue color scheme for visual consistency. `CPUs Idle` was set to green and `CPUs Total` to blue in the SLURM Jobs row.

Download the updated dashboard JSON file: [hpc-overview-nas-gpus.json](/hpc/hpc-overview-nas-gpus.json).

:::note
To import: go to **Dashboards → New → Import** in Grafana, upload the file, and confirm the Prometheus datasource UID (`afn5lu3umf75sf`) matches your current instance.
:::

## 6. Appendix

### 6.1 GPU Reference Table

| GPU Index | PCIe Address | Power Cap | UUID (no prefix) |
|---|---|---|---|
| 0 | 11:00.0 | 300W | `23b4975b-2171-ea56-f5f7-f78bf0cd300a` |
| 1 | F1:00.0 | 300W | `d0b27d44-e215-eeda-b053-6a2342e854c5` |

### 6.2 Related Pages

- [SLURM Setup Guide](./setup)
- [HPC Web Monitoring](../monitoring)
- [Cluster Overview](../overview)
