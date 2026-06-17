---
title: Cluster Overview
sidebar_label: Overview
sidebar_position: 1
---

# Cluster Overview

This document provides a high-level overview of the HPC cluster, including hardware specifications, network architecture, and the services running on each node.

---

## Network Architecture

The cluster consists of two workstations and a NAS, interconnected via a 10GbE internal network. All three devices connect through a **TP-Link TL-SX105** 5-port 10GbaseT unmanaged switch. The switch uplinks to the company network via a 1Gb wall port.

```
             External Network
                     │
                     │  1Gb uplink
                     │
                [Wall Port]
                     │
     [TP-Link 5-Port Unmanaged Switch]
        │            │             │
        │            │             │   10Gb Internal Network
        │            │             │
[Control Node] [Compute Node] [NAS Storage Array]
```

| Device       | Hostname | IP Address     | Network    |
|--------------|----------|----------------|------------|
| Control Node | node01   | 192.168.220.75 | 10Gb + 1Gb |
| Compute Node | node02   | 192.168.220.76 | 10Gb       |
| NAS          | QNAP NAS | 192.168.220.80 | 10Gb       |

---

## Hardware Specifications

### Control Node

| Component | Specification                                      |
|-----------|----------------------------------------------------|
| CPU       | AMD Threadripper PRO 9985WX — 64 cores / 128 threads @ up to 5.5 GHz |
| RAM       | 256 GB (8× 32GB DIMMs)                             |
| GPU       | 2× NVIDIA RTX PRO 6000 Blackwell Max-Q — ~96 GB VRAM each (~192 GB total) |
| Storage   | Samsung 990 PRO 4TB NVMe                           |
| OS        | Ubuntu 24.04                                       |
| IP        | 192.168.220.75                                     |

:::note
node01 was expanded from 1 to 2 GPUs — see [Adding a Second GPU](./slurm/gpu-expansion) for the full hardware, SLURM, and monitoring update walkthrough.
:::

### Compute Node

| Component | Specification                                      |
|-----------|----------------------------------------------------|
| CPU       | AMD Ryzen 9 9950X3D — 16 cores / 32 threads        |
| RAM       | 128 GB                                             |
| GPU       | NVIDIA RTX 5000 Ada Generation — ~32 GB VRAM       |
| OS        | Ubuntu 24.04                                       |
| IP        | 192.168.220.76                                     |

### NAS

| Component  | Specification                                     |
|------------|---------------------------------------------------|
| Model      | QNAP TS-855X-8G-US                                |
| CPU        | Intel Atom C5125 8-core                           |
| RAM        | 8 GB DDR4                                         |
| Network    | 1× 10GbE RJ45 + 2× 2.5GbE                         |
| Drives     | 4× WD Red Pro 18TB                                |
| RAID       | RAID 5 — ~47 TB usable                            |
| NVMe Cache | 2× Samsung 970 EVO Plus 1TB (M.2 PCIe)            |
| OS         | QuTS Hero h5.2.9 (ZFS)                            |
| IP         | 192.168.220.80                                    |

---

## Storage Layout

Each node has its own local partitions, with `/home` shared from the NAS via NFS.

| Partition | Control Node               | Compute Node               |
|-----------|----------------------------|----------------------------|
| `/`       | 1TB NVMe (local)           | 1TB NVMe (local)           |
| `/scratch`| 2.6TB NVMe (local)         | 2.6TB NVMe (local)         |
| `/home`   | NFS — 47TB from NAS        | NFS — 47TB from NAS        |
| Swap      | 10 GB                      | 10 GB                      |

:::tip
For deep learning training workloads, copy datasets to `/scratch/local` before training. Local NVMe (~3–5 GB/s) is significantly faster than NFS (~1.2 GB/s).
:::

---

## Services

| Service           | Control Node | Compute Node |
|-------------------|:------------:|:------------:|
| `slurmctld`       | ✓            |              |
| `slurmd`          | ✓            | ✓            |
| `munge`           | ✓            | ✓            |
| `prometheus`      | ✓            |              |
| `grafana`         | ✓            |              |
| `node_exporter`   | ✓            | ✓            |
| `nvidia_gpu_exporter` | ✓        | ✓            |
| `slurm_exporter`  | ✓            |              |
| `apache2`         | ✓            |              |
