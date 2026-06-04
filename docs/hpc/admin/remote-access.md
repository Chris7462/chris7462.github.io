---
title: Remote Access
sidebar_label: Remote Access
sidebar_position: 2
---

# Remote Access

This document covers how to configure remote access to the cluster nodes via SSH and RDP.

---

## SSH Server

### Install

```bash
sudo apt update
sudo apt install openssh-server
```

### Verify

```bash
sudo systemctl status ssh
```

The output should show the service as **active (running)** and enabled to start on boot.

### Firewall (Optional)

If the firewall is enabled on your system, open the SSH port:

```bash
sudo ufw allow ssh
```

That's it. The cluster nodes are now accessible via SSH from any machine on the network.

---

## RDP Server (Optional)

:::note
This section is only required if the machine has a desktop environment (X server) installed.
The control node (node01) runs a full desktop environment and may require RDP access.
Compute-only nodes without a display environment can skip this section.
:::

### Install

```bash
sudo apt install xrdp
```

### Enable and Start

```bash
sudo systemctl enable --now xrdp
```

### Verify

```bash
sudo systemctl status xrdp
```

The output should show the service as **active (running)** and enabled to start on boot.

### Firewall (Optional)

If the firewall is enabled, open port 3389 for incoming RDP traffic:

```bash
sudo ufw allow from any to any port 3389 proto tcp
```
