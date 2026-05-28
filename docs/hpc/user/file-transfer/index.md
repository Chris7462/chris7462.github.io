---
title: File Transfer
sidebar_label: File Transfer
sidebar_position: 2
---

# File Transfer

This document covers how to transfer files between your local machine and the cluster.

---

## SCP

SCP (Secure Copy) is the simplest way to transfer files over SSH.

### Copy a File from Local to Cluster

```bash
scp /local/path/to/file USERNAME@192.168.220.75:~/destination/
```

### Copy a Folder from Local to Cluster

```bash
scp -r /local/path/to/folder USERNAME@192.168.220.75:~/destination/
```

### Copy a File from Cluster to Local

```bash
scp USERNAME@192.168.220.75:~/path/to/file /local/destination/
```

---

## rsync

rsync is better than SCP for large transfers — it only copies files that have changed, and it can resume interrupted transfers.

### Sync a Folder from Local to Cluster

```bash
rsync -avz /local/path/to/folder/ USERNAME@192.168.220.75:~/destination/
```

### Sync a Folder from Cluster to Local

```bash
rsync -avz USERNAME@192.168.220.75:~/path/to/folder/ /local/destination/
```

### Common rsync Flags

| Flag | Description |
|------|-------------|
| `-a` | Archive mode — preserves permissions, timestamps, symlinks |
| `-v` | Verbose — shows files being transferred |
| `-z` | Compress data during transfer |
| `-P` | Show progress and allow resuming interrupted transfers |
| `--delete` | Delete files on destination that no longer exist on source |

:::tip
Use `-P` instead of `-v` for large transfers to see progress:

```bash
rsync -azP /local/path/to/folder/ USERNAME@192.168.220.75:~/destination/
```
:::

---

## SSHFS

SSHFS mounts the cluster's home directory as a local folder on your machine. This lets you browse and edit remote files as if they were local.

### Install SSHFS (One-time Setup)

```bash
# Ubuntu / Debian
sudo apt install sshfs
```

### Mount the Cluster Home Directory

```bash
# Create a local mount point
mkdir -p ~/mnt/cluster

# Mount the remote home directory
sshfs USERNAME@192.168.220.75:/home/USERNAME ~/mnt/cluster
```

You can now browse the cluster's home directory at `~/mnt/cluster`.

### Unmount When Done

```bash
# Linux
fusermount -u ~/mnt/cluster
```

:::note
SSHFS is convenient for browsing and editing files but is **not suitable for large data transfers** — use `rsync` or `scp` instead. Also note that SSHFS performance depends on your network connection.
:::
