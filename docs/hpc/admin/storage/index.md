---
title: NAS Storage Setup
sidebar_label: Storage
sidebar_position: 4
---

# NAS Storage Setup

This document describes the setup of the centralized NAS storage for the HPC cluster. The NAS serves as the shared `/home` filesystem for all cluster nodes via NFS over 10GbE.

---

## Architecture

```
┌────────────────┐     ┌────────────────┐
│  Control Node  │     │  Compute Node  │
│ 192.168.220.75 │     │ 192.168.220.76 │
│                │     │                │
│  /home (NFS)   │     │  /home (NFS)   │
└───────┬────────┘     └────────┬───────┘
        │                       │
        │        10GbE          │
        └──────────┬────────────┘
                   │
        ┌──────────▼────────────┐
        │       QNAP NAS        │
        │    192.168.220.80     │
        │   47TB RAID5 (ZFS)    │
        └───────────────────────┘
```

---

## Hardware

### QNAP NAS

| Item | Specification |
|------|---------------|
| Model | QNAP TS-855X-8G-US |
| IP Address | 192.168.220.80 |
| OS | QuTS Hero h5.2.9 (ZFS) |
| CPU | Intel Atom C5125 8-core |
| RAM | 8 GB DDR4 |
| Network | 1× 10GbE RJ45 + 2× 2.5GbE |
| Drives | 4× WD Red Pro 18TB |
| RAID | RAID 5 — ~47 TB usable |
| NVMe Cache | 2× Samsung 970 EVO Plus 1TB (M.2 PCIe) |

---

## QNAP Configuration

### Storage Pool

| Setting | Value |
|---------|-------|
| Pool name | Storage Pool 1 (System) |
| RAID type | RAID 5 |
| Total capacity | 47.19 TB |
| Usable capacity | 46.86 TB |
| Filesystem | ZFS (QuTS Hero) |
| Compression | Enabled |
| Deduplication | Disabled |
| Block size | 4K (Database/VM profile) |
| Snapshot schedule | Daily 01:00 — Smart Versioning (Hourly:24, Daily:7, Weekly:4, Monthly:12) |
| Encryption | Disabled |
| ACL inheritance | `aclinherit=discard` (set via ZFS CLI) |

### Shared Folder (`nfs-home`)

| Setting | Value |
|---------|-------|
| Folder name | nfs-home |
| Mount path on NAS | `/share/ZFS18_DATA/nfs-home` |
| Space allocation | Thin provisioning |
| Quota | 46.86 TB |

### NVMe Cache

| Setting | Value |
|---------|-------|
| Drives | 2× Samsung 970 EVO Plus 1TB (M.2 PCIe) |
| Cache type | Read Cache + ZIL Synchronous Write Log |
| Read cache | 1.73 TB, RAID 0 |
| ZIL write log | 9.97 GB, RAID 1 (mirrored) |
| Cache mode | Random I/O |
| Cached storage | Storage Pool 1 |

### NFS Configuration

| Setting | Value |
|---------|-------|
| NFS versions enabled | NFSv2/v3 and NFSv4 |
| Sync mode | sync (wdelay) |
| Control node | read/write, no_root_squash |
| Compute node | read/write, root_squash |

### ZFS ACL Fix

QNAP's default ACL inheritance caused new files to get 777 permissions. Fixed by setting:

```bash
zfs set aclinherit=discard zpool1/zfs18
```

---

## Cluster Node Configuration

### Control Node `/etc/fstab`

```bash
/dev/disk/by-uuid/15e33979-eea6-427a-bde2-0faa32650099 / ext4 defaults 0 1
/dev/disk/by-uuid/cc343fa9-7065-4ed8-9f87-7114e4092080 /scratch ext4 defaults 0 2
192.168.220.80:/nfs-home /home nfs rw,sync,hard,intr,rsize=1048576,wsize=1048576,nfsvers=4 0 0
```

### Compute Node `/etc/fstab`

```bash
/dev/disk/by-uuid/ec5ab975-e5de-4d8f-a287-85318aa7b56f / ext4 defaults 0 1
/dev/disk/by-uuid/35336e1c-4b62-4679-98bd-76b743725d65 /scratch ext4 defaults 0 2
192.168.220.80:/nfs-home /home nfs rw,sync,hard,intr,rsize=1048576,wsize=1048576,nfsvers=4 0 0
```

### Docker and Containerd

Docker `data-root` and containerd have been moved to `/scratch` on both nodes to avoid storing container data on the NFS `/home`.

| Service | Path |
|---------|------|
| Docker data-root | `/scratch/docker` |
| containerd root | `/scratch/docker/containerd` |

Config files updated:
- `/etc/docker/daemon.json` — `data-root` set to `/scratch/docker`
- `/etc/containerd/config.toml` — `root` set to `/scratch/docker/containerd`

---

## User Quotas

Quotas are enforced on the QNAP NAS via ZFS `userquota` by UID. Since `/home` is NFS, client-side Linux quota tools cannot be used — quota must be managed on the NAS where the filesystem lives.

### Quota Management Commands

SSH into the NAS first:

```bash
ssh user@192.168.220.80
```

Check all user quotas:

```bash
zfs userspace zpool1/zfs18
```

Set or update a user quota:

```bash
zfs set userquota@<UID>=100G zpool1/zfs18
```

Remove a quota (set to unlimited):

```bash
zfs set userquota@<UID>=none zpool1/zfs18
```

### Adding New Users

When a new cluster user is created on the nodes, set their quota on the NAS:

```bash
ssh user@192.168.220.80
zfs set userquota@<NEW_UID>=100G zpool1/zfs18
```

---

## Performance

| Configuration | Write Speed | Read Speed |
|---------------|-------------|------------|
| 1GbE (before 10GbE switch) | 105 MB/s | 105 MB/s |
| 10GbE (no NVMe cache) | 169 MB/s | 1.2 GB/s |
| 10GbE + NVMe cache | 673 MB/s | 1.2 GB/s |
| Local NVMe (reference) | ~2–3 GB/s | ~3–5 GB/s |

:::note
Read speed is limited by 10GbE line speed (~1.25 GB/s). NVMe cache primarily benefits write speed and repeated reads of frequently accessed files (conda environments, Python packages, source code).
:::

:::tip
For best deep learning training performance, copy datasets to local `/scratch` before training — local NVMe gives significantly higher throughput than NFS.
:::

---

## Local Data Directory (`/scratch/local`)

`/scratch/local` is a shared directory on both nodes for users to copy training datasets from NFS (`/home`) to local NVMe storage.

- Copy large datasets here before training to get local NVMe speed (~3–5 GB/s) instead of NFS speed (~1.2 GB/s)
- Shared between all users on the same node
- Data here is **node-local** — files on the control node are not visible on the compute node

### Permissions

Access is controlled via ACL — only users in the specific groups can read and write:

```bash
# View current ACL
getfacl /scratch/local
```

### Usage Example

```bash
# Copy dataset from NFS home to local scratch before training
cp -r ~/my_dataset /scratch/local/

# Or use rsync for large datasets
rsync -avP ~/my_dataset/ /scratch/local/my_dataset/
```

:::warning
- `/scratch/local` is **not backed up** — do not store original data here, only copies
- Clean up your data after training to free space for other users
- `/scratch` is a local partition — it will be lost if the disk fails
:::

---

## Maintenance Notes

### NVMe Cache

:::danger
**Never physically remove NVMe drives while cache is active** — data loss will occur even if the NAS is powered off.
:::

To safely remove NVMe drives:

1. Go to **Storage & Snapshots → Cache Acceleration → Manage**
2. Select **Disable cache**
3. Wait for the cache to fully flush
4. Power off the NAS before removing drives

### Snapshots

- **Schedule:** Daily at 01:00
- **Retention:** Hourly ×24, Daily ×7, Weekly ×4, Monthly ×12
- **Manage via:** Storage & Snapshots → Data Protection → Snapshot

### Firmware Updates

:::warning
- Set to **notify only** — never auto-update
- Never install beta firmware on a production NAS
- Apply updates during scheduled maintenance windows only
:::

### Home Directory Backup

A local backup of `/home` is kept on both nodes at `/scratch/home-backup`. This was created at migration time and is **not automatically updated**. For disaster recovery, the NAS snapshots are the primary backup mechanism.
