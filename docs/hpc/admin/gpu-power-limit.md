---
title: GPU Power Limit Enforcement
sidebar_label: GPU Power Limit
sidebar_position: 10
---

# GPU Power Limit Enforcement Setup

This document explains how to configure the workstation to automatically apply a power limit to all NVIDIA GPUs at boot, to keep total system draw within the capacity of the circuit and UPS the machine is on.

---

## Overview

### What It Does

- Applies an `nvidia-smi -pl <watts>` power cap to every GPU on the system at boot
- Runs once as a `oneshot` systemd service, after the NVIDIA driver is loaded
- Persists across reboots (the OS-level power limit itself is *not* persistent by default, so this service re-applies it every boot)

### Why It's Needed

`nvidia-smi -pl` power limits reset to the card's default on every reboot. On a multi-GPU workstation where total power draw needs to stay under a fixed ceiling (circuit capacity, UPS rated output, thermal/cooling budget), the limit has to be re-applied automatically rather than relying on someone to run the command by hand after every restart.

---

## Step 1: Create the Power Limit Script

```bash
sudo vim /usr/local/bin/set-gpu-power-limit.sh
```

Paste the following content:

```bash
#!/bin/bash
# ==============================================================================
# GPU Power Limit Setter
# Applies a power cap to all NVIDIA GPUs at boot (resets to default otherwise)
# ==============================================================================

# Power limit to apply to all GPUs, in watts
POWER_LIMIT=250

# Give the NVIDIA driver a moment to finish initializing
sleep 5

nvidia-smi -pl "$POWER_LIMIT"
```

Make it executable:

```bash
sudo chmod +x /usr/local/bin/set-gpu-power-limit.sh
```

---

## Step 2: Create the Systemd Service

```bash
sudo vim /etc/systemd/system/gpu-power-limit.service
```

Paste the following content:

```ini
[Unit]
Description=Set NVIDIA GPU power limit at boot
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/set-gpu-power-limit.sh
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable gpu-power-limit.service
sudo systemctl start gpu-power-limit.service
```

---

## Step 3: Verify Installation

Check the service is running:

```bash
sudo systemctl status gpu-power-limit.service
```

Expected output:

```
● gpu-power-limit.service - Set NVIDIA GPU power limit at boot
     Loaded: loaded (/etc/systemd/system/gpu-power-limit.service; enabled)
     Active: active (exited)
```

Confirm the limit was actually applied:

```bash
nvidia-smi --query-gpu=index,power.limit --format=csv
```

Expected output (for a 250W cap on three GPUs):

```
index, power.limit [W]
0, 250.00 W
1, 250.00 W
2, 250.00 W
```

---

## Step 4: Test the Enforcement

**Test 1: Reboot persistence**

```bash
sudo reboot
```

After the system comes back up:

```bash
nvidia-smi --query-gpu=index,power.limit --format=csv
```

Expected behavior: all GPUs still show the configured limit, not the card default.

**Test 2: Manual override during a session**

```bash
sudo nvidia-smi -pl 300
nvidia-smi --query-gpu=index,power.limit --format=csv
sudo systemctl restart gpu-power-limit.service
nvidia-smi --query-gpu=index,power.limit --format=csv
```

Expected behavior: the limit reverts to the configured value after the service restarts.

---

## Configuration Options

### Adjust the Power Limit

Edit `/usr/local/bin/set-gpu-power-limit.sh` and change `POWER_LIMIT` to the desired wattage, then restart the service:

```bash
sudo systemctl restart gpu-power-limit.service
```

Valid range depends on the card — check with:

```bash
nvidia-smi -q -d POWER
```

### Set Different Limits per GPU

Replace the single `nvidia-smi -pl "$POWER_LIMIT"` line with per-index calls:

```bash
nvidia-smi -i 0 -pl 250
nvidia-smi -i 1 -pl 280
nvidia-smi -i 2 -pl 250
```

Use `nvidia-smi -L` to confirm GPU index-to-card mapping before assigning different limits.

---

## Managing the Service

| Action | Command |
|--------|---------|
| Start | `sudo systemctl start gpu-power-limit.service` |
| Restart (re-apply limit) | `sudo systemctl restart gpu-power-limit.service` |
| Status | `sudo systemctl status gpu-power-limit.service` |
| View logs | `sudo journalctl -u gpu-power-limit.service` |
| Disable on boot | `sudo systemctl disable gpu-power-limit.service` |
| Enable on boot | `sudo systemctl enable gpu-power-limit.service` |

---

## Troubleshooting

### Service Fails or Limit Not Applied

Check for syntax errors and review the logs:

```bash
sudo bash -n /usr/local/bin/set-gpu-power-limit.sh
sudo journalctl -u gpu-power-limit.service -n 50
```

### `nvidia-smi -pl` Returns "Not Supported"

Some GPUs or driver/firmware combinations don't allow power limit changes, or require running as root. Confirm:

- The script is running with sufficient privileges (systemd services run as root by default, which is required here).
- The requested wattage is within `Min Power Limit` / `Max Power Limit` from `nvidia-smi -q -d POWER`.

### Limit Resets Unexpectedly

If something else on the system (a container runtime, another script, a driver reinstall) also touches power limits, it can override this service's setting after boot. Re-running:

```bash
sudo systemctl restart gpu-power-limit.service
```

re-applies the configured value at any time.

---

## Summary of Files

| File | Purpose |
|------|---------|
| `/usr/local/bin/set-gpu-power-limit.sh` | Script that applies the GPU power limit |
| `/etc/systemd/system/gpu-power-limit.service` | Systemd service definition |
