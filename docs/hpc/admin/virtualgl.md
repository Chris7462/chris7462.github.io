---
title: VirtualGL Setup
sidebar_label: VirtualGL
sidebar_position: 7
---

# VirtualGL Setup Guide

This document covers the setup of VirtualGL on the HPC workstation to enable GPU-accelerated remote GUI rendering over SSH.

---

## Overview

VirtualGL renders OpenGL on the server's NVIDIA GPU and forwards the output over SSH X11 forwarding. GUI applications appear as native windows on the client — no VNC, no remote desktop required.

```
Client (SSH -Y) ──► node01 (vglrun app) ──► NVIDIA GPU renders ──► frames sent back to client
```

**Why VirtualGL?** Without it, OpenGL applications like `rviz2` still work over SSH X11 forwarding but use slow indirect rendering (~70 FPS). With VirtualGL, frames are rendered on the GPU and compressed before sending (~115+ FPS).

---

## Prerequisites

Verify the NVIDIA driver is installed and working:

```bash
nvidia-smi
```

Verify Xorg is running (not Wayland):

```bash
echo $XDG_SESSION_TYPE
# Should output: x11
```

### Ensure Xorg is Available (Not Wayland)

Ubuntu 24.04 uses GDM with Wayland by default. VirtualGL requires Xorg.

```bash
sudo vim /etc/gdm3/custom.conf
```

Uncomment or add under `[daemon]`:

```ini
[daemon]
WaylandEnable=false
```

Reboot:

```bash
sudo reboot
```

Verify after reboot:

```bash
echo $XDG_SESSION_TYPE
# Should output: x11
```

---

## Step 1: Install VirtualGL

```bash
# Import GPG key
wget -q -O- https://packagecloud.io/dcommander/virtualgl/gpgkey | \
  sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/VirtualGL.gpg

# Add repository
sudo wget -q -O /etc/apt/sources.list.d/VirtualGL.list \
  https://raw.githubusercontent.com/VirtualGL/repo/main/VirtualGL.list

# Install
sudo apt update
sudo apt install -y virtualgl
```

Verify:

```bash
dpkg -l | grep virtualgl
ls /opt/VirtualGL/bin/
```

---

## Step 2: Configure VirtualGL

```bash
# Stop the display manager
sudo systemctl stop gdm3

# Configure VirtualGL
sudo /opt/VirtualGL/bin/vglserver_config
```

Answer the configuration questions as follows:

| Question | Answer |
|----------|--------|
| Restrict 3D X server access to `vglusers` group? | **Yes** |
| Restrict framebuffer device access to `vglusers` group? | **Yes** |
| Disable XTEST extension? | **Yes** |

Restart the display manager:

```bash
sudo systemctl start gdm3
```

---

## Step 3: Add Users to `vglusers` Group

```bash
sudo usermod -aG vglusers user1
sudo usermod -aG vglusers user2
sudo usermod -aG vglusers user3
```

:::note
Users must log out and back in (or reboot) for group changes to take effect.
:::

---

## Step 4: Enable SSH X11 Forwarding

Verify these lines are present and uncommented in `/etc/ssh/sshd_config`:

```bash
cat /etc/ssh/sshd_config | grep X11
```

Ensure the following are set:

```
X11Forwarding yes
X11DisplayOffset 10
X11UseLocalhost yes
```

If you made changes, restart sshd:

```bash
sudo systemctl restart sshd
```

---

## Step 5: Add VirtualGL to System PATH

```bash
sudo vim /etc/profile.d/virtualgl.sh
```

Add the following:

```bash
export PATH="/opt/VirtualGL/bin:$PATH"
export VGL_COMPRESS=proxy
export VGL_DISPLAY=:0
```

Make it executable:

```bash
sudo chmod +x /etc/profile.d/virtualgl.sh
```

**What these variables do:**

- `VGL_COMPRESS=proxy` — sends frames through the SSH X11 tunnel (not vglclient)
- `VGL_DISPLAY=:0` — tells VirtualGL which display has the GPU

:::note
No `__NV_PRIME_RENDER_OFFLOAD` or `__GLX_VENDOR_LIBRARY_NAME` needed — the NVIDIA GPU is the only GPU on this system (no hybrid switching).
:::

---

## Step 6: Add xauth Fix to Each User's `~/.bashrc`

Add the following near the top of each user's `~/.bashrc` (before any `return` statements):

```bash
sed -i '2i xauth merge /etc/opt/VirtualGL/vgl_xauth_key 2>/dev/null' ~/.bashrc
```

---

## Step 7: Verify

```bash
# Grant access to the X display
xauth merge /etc/opt/VirtualGL/vgl_xauth_key

# Check NVIDIA GPU is accessible via VirtualGL
/opt/VirtualGL/bin/glxinfo -display :0 -c | grep "OpenGL renderer"
# Should show: NVIDIA RTX PRO 6000
```

---

## Client Setup

**Requirements:** Ubuntu with SSH — no extra software needed on the client side.

### Connect and Run

```bash
# From client machine
ssh -Y user@node01

# Test GPU rendering
vglrun glxgears

# ROS applications
vglrun rviz2
vglrun gazebo

# Python (usually works without vglrun)
python3 my_plot_script.py
```

### SSH Config for Convenience (on Each Client)

```bash
vim ~/.ssh/config
```

Add:

```
Host node01
    HostName 192.168.220.75
    User <your-username>
    ForwardX11 yes
    ForwardX11Trusted yes
    Compression yes
```

Then simply:

```bash
ssh node01
vglrun rviz2
```

---

## Usage Tips

### Aliases (on the Server)

Add to `~/.bashrc`:

```bash
alias rviz2='vglrun rviz2'
alias gazebo='vglrun gazebo'
alias glxgears='vglrun glxgears'
```

### Performance

For best performance, use wired LAN. Add `-C` to SSH for compression over slower links:

```bash
ssh -Y -C user@node01
```

### Persistent CLI Work with tmux

GUI apps close when SSH disconnects, but terminal work can persist:

```bash
ssh user@node01
tmux new -s work
# Do work... detach with Ctrl+B then D
# Reconnect later:
tmux attach -s work
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Connect from client | `ssh -Y user@node01` |
| Run app with GPU | `vglrun <app>` |
| Test GPU rendering | `vglrun glxgears` |
| Check GPU in use | `vglrun glxinfo \| grep "OpenGL renderer"` |
| Fix "VGL client" errors | `export VGL_COMPRESS=proxy && export VGL_DISPLAY=:0` |

---

## Troubleshooting

### "Could not connect to VGL client" / "Connection refused"

`VGL_COMPRESS` is not set. Fix:

```bash
export VGL_COMPRESS=proxy
export VGL_DISPLAY=:0
vglrun rviz2
```

Verify: `echo $VGL_COMPRESS` should show `proxy`.

### "Invalid MIT-MAGIC-COOKIE-1 key" / "unable to open display :0"

X authentication issue. Fix:

```bash
xauth merge /etc/opt/VirtualGL/vgl_xauth_key
```

### "cannot open display" or "Error: unable to open display"

SSH was not started with `-X` or `-Y`. Reconnect with:

```bash
ssh -Y user@node01
```

### "X11 forwarding request failed on channel 0"

X11 forwarding is disabled on the server. Check `/etc/ssh/sshd_config` has `X11Forwarding yes`. Install `xauth` if missing:

```bash
sudo apt install xauth
```

### `glxinfo` Shows "Mesa" Instead of NVIDIA

- GDM is still using Wayland — check `WaylandEnable=false` in `/etc/gdm3/custom.conf`
- VirtualGL not configured — re-run `vglserver_config`
- User not in `vglusers` group — check with `groups $USER`

### RViz2 / Gazebo Shows Black 3D Viewport

Launched without `vglrun` — always prefix with `vglrun`. Verify:

```bash
vglrun glxinfo | grep "OpenGL renderer"
# Should show NVIDIA
```

### Apps Render but Are Very Slow

- Not using VirtualGL (indirect rendering) — use `vglrun`
- Use wired LAN, not WiFi
- Add `-C` to SSH for compression

---

## Uninstall

### 1. Unconfigure VirtualGL

```bash
sudo systemctl stop gdm3
sudo /opt/VirtualGL/bin/vglserver_config
# Select: Unconfigure server for use with VirtualGL
sudo systemctl start gdm3
```

### 2. Remove Users from `vglusers` Group

```bash
sudo deluser user1 vglusers
sudo deluser user2 vglusers
sudo deluser user3 vglusers
```

### 3. Uninstall VirtualGL

```bash
sudo apt purge -y virtualgl
sudo rm -f /etc/apt/sources.list.d/VirtualGL.list
sudo rm -f /etc/apt/trusted.gpg.d/VirtualGL.gpg
sudo apt update
```

### 4. Remove Config Files

```bash
sudo rm -f /etc/profile.d/virtualgl.sh
```

### 5. Remove xauth Line from Each User's `~/.bashrc`

```bash
sed -i '/vgl_xauth_key/d' ~/.bashrc
```

### 6. Re-enable Wayland (Optional)

```bash
sudo vim /etc/gdm3/custom.conf
```

Comment out or remove:

```ini
#WaylandEnable=false
```

### 7. Revert SSH Config (Only if You Changed It)

If `X11Forwarding` was not enabled before, edit `/etc/ssh/sshd_config` and set:

```
X11Forwarding no
```

```bash
sudo systemctl restart sshd
```

### 8. Reboot

```bash
sudo reboot
```
