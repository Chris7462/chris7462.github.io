---
title: Adding a Compute Node
sidebar_label: Adding a Node
sidebar_position: 3
---

# Adding a Compute Node to the SLURM Cluster

This document provides a step-by-step guide for adding a new compute node to the existing SLURM cluster.

---

## Architecture Overview

```
┌────────────────────────────┐     ┌────────────────────────────┐
│      Control Node          │     │      New Compute Node      │
│                            │     │                            │
│  slurmctld (controller)    │<───>│  slurmd (compute)          │
│  slurmd (compute)          │     │                            │
│  munge (auth)              │     │  munge (auth)              │
└────────────────────────────┘     └────────────────────────────┘
      Users SSH here                  Jobs scheduled here
      to submit jobs                  by the controller
```

The control node runs both `slurmctld` (controller) and `slurmd` (compute). New nodes run only `slurmd`.

| File | Same on All Nodes? |
|------|--------------------|
| `slurm.conf` | Yes — must be identical |
| `cgroup.conf` | Yes — must be identical |
| `gres.conf` | No — different per node (matches local GPU) |

---

## Step 1: Set Hostname

On the **new node**:

```bash
sudo hostnamectl set-hostname <new_hostname>
```

---

## Step 2: Network Configuration

Both machines must resolve each other by hostname to **real network IPs** (not loopback).

### Update `/etc/hosts` on BOTH Machines

```bash
sudo vim /etc/hosts
```

Add entries for all cluster nodes:

```
192.168.220.75  node01  # control node
192.168.220.76  node02  # compute node
```

:::warning
The control node hostname (`node01`) must **NOT** resolve to `127.0.1.1`. Replace any `127.0.1.1 node01` line with the real network IP.
:::

### Verify Connectivity from BOTH Sides

```bash
ping node01
ping node02
ssh node01
ssh node02
hostname -f
```

---

## Step 3: Match User Accounts

SLURM requires **identical UIDs and GIDs** on all nodes. Every user that submits or runs jobs must exist on both machines with the same numeric IDs.

### Check Existing UIDs on the Control Node

```bash
for u in user1 user2 user3 slurm; do
    id $u 2>/dev/null
done
```

### Create Groups on the New Node

```bash
sudo groupadd -g 1005 <group1>
sudo groupadd -g 1006 <group2>
```

### Create Users on the New Node

```bash
sudo useradd -u 1001 -g 1006 -m -s /bin/bash -c "User One" user1
sudo useradd -u 1002 -g 1006 -m -s /bin/bash -c "User Two" user2
sudo useradd -u 1003 -g 1005 -m -s /bin/bash -c "User Three" user3
```

### Create the SLURM System User

```bash
sudo groupadd -g 64030 slurm
sudo useradd -u 64030 -g 64030 -r -s /usr/sbin/nologin slurm
```

:::note
The `useradd` warning about UID exceeding `SYS_UID_MAX` is harmless — ignore it.
:::

### Handle UID Conflicts

If an existing local user on the new node has a UID in the 1001–1009 range, move them first:

```bash
# Example: move user 'localuser' from UID 1001 to 2001
sudo usermod -u 2001 localuser
sudo groupmod -g 2001 localuser
sudo find / -uid 1001 -exec chown 2001 {} + 2>/dev/null
sudo find / -gid 1001 -exec chgrp 2001 {} + 2>/dev/null
```

### Verify — Run on BOTH Machines and Compare

```bash
for u in user1 user2 user3 slurm; do
    id $u 2>/dev/null
done
```

Output must be identical (UIDs, primary GIDs). Secondary groups like `sudo`, `docker` can differ.

---

## Step 4: Install SLURM on the New Node

```bash
sudo apt update
sudo apt install -y slurmd slurm-client munge
```

:::warning
Do **NOT** install `slurmctld` — only the control node needs the controller.
:::

---

## Step 5: Sync Munge Key

SLURM uses Munge for authentication. Both nodes must share the **same key**.

### Export Key from Control Node

```bash
# On control node
sudo cat /etc/munge/munge.key | base64
```

### Import Key on New Node

```bash
# On new node
sudo systemctl stop munge

echo "<paste_base64_key_as_one_line>" | base64 -d | sudo tee /etc/munge/munge.key > /dev/null

sudo chown munge:munge /etc/munge/munge.key
sudo chmod 400 /etc/munge/munge.key
sudo systemctl enable --now munge
```

### Test

```bash
# From new node
munge -n | ssh node01 unmunge
```

Expected output: `STATUS: Success (0)`

---

## Step 6: Update `slurm.conf`

Edit `/etc/slurm/slurm.conf` on the **control node** to add the new node definition.

### Gather Specs from the New Node First

```bash
lscpu | grep -E "^CPU\(s\)|Core|Thread|Socket|Model name"
free -h | grep Mem
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

### Add the New Node Definition

```bash
# Node 2: node02 (compute only)
# AMD Ryzen 9 9950X3D (16 cores / 32 threads)
# 128 GB RAM
# 1x NVIDIA RTX 5000 Ada Generation (~32GB VRAM)
NodeName=node02 \
      CPUs=32 \
      Boards=1 \
      SocketsPerBoard=1 \
      CoresPerSocket=16 \
      ThreadsPerCore=2 \
      RealMemory=120000 \
      Gres=gpu:rtx5000:1,shard:2 \
      State=UNKNOWN
```

### Update the Partition to Include the New Node

```bash
PartitionName=main \
      Nodes=node01,node02 \
      Default=YES \
      MaxTime=7-00:00:00 \
      DefaultTime=00:30:00 \
      State=UP
```

### Update GPU Support Section

```bash
GresTypes=gpu,shard
```

### Intel Hybrid CPU Note

If the new node uses an Intel hybrid architecture (P-cores + E-cores), `slurmd -C` may auto-detect fewer CPUs than available. Add this line under `TaskPlugin`:

```bash
SlurmdParameters=config_overrides
```

This tells `slurmd` to trust `slurm.conf` values instead of auto-detection.

---

## Step 7: Create `gres.conf` on the New Node

Each node has its own `gres.conf` matching its local GPU. Create `/etc/slurm/gres.conf` on the new node:

```bash
# Full GPU - exclusive access
Name=gpu Type=rtx5000 File=/dev/nvidia0

# GPU Shards - shared access
# 2 shards available, each represents ~16GB VRAM (32GB / 2)
Name=shard Count=2 File=/dev/nvidia0
```

:::note
Adjust the GPU type, shard count, and VRAM per shard based on the actual GPU installed on the new node.
:::

---

## Step 8: Copy Configs to the New Node

`slurm.conf` and `cgroup.conf` must be **identical** on all nodes.

```bash
# From control node
scp /etc/slurm/slurm.conf <new_node>:/tmp/
scp /etc/slurm/cgroup.conf <new_node>:/tmp/
```

On the **new node**:

```bash
sudo cp /tmp/slurm.conf /etc/slurm/slurm.conf
sudo cp /tmp/cgroup.conf /etc/slurm/cgroup.conf
```

Verify `cgroup.conf` exists and matches:

```bash
ConstrainCores=yes
ConstrainRAMSpace=yes
ConstrainDevices=yes
```

---

## Step 9: Create Required Directories on the New Node

```bash
sudo mkdir -p /var/spool/slurmd /var/log/slurm
sudo chown slurm:slurm /var/spool/slurmd /var/log/slurm
```

---

## Step 10: Start Services

Start the new node first, then restart the control node:

```bash
# On new node
sudo systemctl enable --now slurmd

# On control node
sudo systemctl daemon-reload
sudo systemctl restart slurmctld
sudo systemctl restart slurmd
```

---

## Step 11: Verify the Cluster

```bash
# Check both nodes are visible
sinfo -N -l

# Test job on each node
srun -w node01 hostname
srun -w node02 hostname

# Test GPU on new node
srun -w node02 --gres=gpu:1 nvidia-smi

# Test shard on new node
srun -w node02 --gres=shard:1 nvidia-smi

# Let SLURM pick the node
srun --gres=shard:1 hostname
```

Expected `sinfo` output:

```
NODELIST  NODES  PARTITION  STATE  CPUS  S:C:T    MEMORY  TMP_DISK  WEIGHT  AVAIL_FE  REASON
node01    1      main*      idle   128   1:64:2   250000  0         1       (null)    none
node02    1      main*      idle   32    1:16:2   120000  0         1       (null)    none
```

---

## Removing a Compute Node

### Stop `slurmd` on the Compute Node

```bash
sudo systemctl stop slurmd
sudo systemctl disable slurmd
```

### Update `slurm.conf` on the Control Node

1. Remove the node's `NodeName=...` block
2. Remove the node from the `Nodes=` list in the partition
3. Copy updated `slurm.conf` to all remaining nodes

### Restart SLURM on the Control Node

```bash
sudo systemctl daemon-reload
sudo systemctl restart slurmctld
sudo systemctl restart slurmd
```

### Clear Stale Node State (if Grafana/sinfo Still Shows Old Node)

```bash
sudo systemctl stop slurmctld
sudo rm /var/spool/slurmctld/node_state
sudo systemctl start slurmctld
```

### Fully Remove SLURM from the Compute Node (Optional)

```bash
sudo apt remove --purge slurmd slurm-client munge
sudo rm -rf /etc/slurm /var/spool/slurmd /var/log/slurm /etc/munge
```

---

## Troubleshooting

### `slurmd` Fails to Start — "DNS SRV lookup failed"

The `slurm.conf` file is missing or misnamed on the compute node. Verify:

```bash
ls -la /etc/slurm/slurm.conf
```

### GPU Shows "No devices were found"

1. Verify `nvidia-smi` works outside SLURM
2. Check `gres.conf` matches the actual GPU device (`/dev/nvidia0`)
3. Check if `slurmd -C` detects fewer CPUs than configured (Intel hybrid CPU issue) — add `SlurmdParameters=config_overrides` to `slurm.conf`

### `slurmd` Detects Wrong CPU Count

Common with Intel hybrid architectures (P-cores + E-cores). Add to `slurm.conf`:

```bash
SlurmdParameters=config_overrides
```

### Munge Authentication Fails

Ensure both nodes have the same munge key:

```bash
# On control node
sudo md5sum /etc/munge/munge.key

# On compute node
sudo md5sum /etc/munge/munge.key
```

If they differ, re-copy the key from the control node.

### Node Shows as "down" in `sinfo`

```bash
# Check why
scontrol show node <nodename> | grep Reason

# Resume the node
sudo scontrol update NodeName=<nodename> State=resume
```
