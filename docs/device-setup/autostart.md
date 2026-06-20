---
sidebar_position: 6
title: Autostart Programs on Login
description: Automatically run a program when a user logs into the Ubuntu Unity desktop
---

# Autostart Programs on Login (Ubuntu Unity)

Ubuntu 26.04 removed the graphical **Startup Applications** tool, but the underlying mechanism it used to manage is still there: `.desktop` files in an autostart directory, read by the desktop session on login. This guide covers setting it up by hand.

## Per-User Autostart

To autostart a program for only your own account:

```bash
mkdir -p ~/.config/autostart
vim ~/.config/autostart/myprogram.desktop
```

```ini
[Desktop Entry]
Type=Application
Name=My Program
Exec=/path/to/your/script.sh
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
```

Save the file — no reboot needed, it takes effect on next login.

## System-Wide Autostart (All Users)

For shared/multi-account machines, place the file in `/etc/xdg/autostart` instead, so every account picks it up without separate per-user setup:

```bash
sudo vim /etc/xdg/autostart/myprogram.desktop
```

Same `.desktop` content as above.

## If `Exec` Points to a Shell Script

Make sure the script is executable, or the autostart entry will silently fail:

```bash
chmod +x /path/to/your/script.sh
```

## `.desktop` Field Reference

| Field | Required | Purpose |
|---|---|---|
| `Type` | Yes | Always `Application` for autostart entries |
| `Name` | Yes | Display name (shown in tools like GNOME Tweaks) |
| `Exec` | Yes | Full path to the program or script to run |
| `X-GNOME-Autostart-enabled` | Recommended | Set to `true` explicitly — some tools read this to show the on/off toggle |
| `Hidden` | No | Set to `true` to disable the entry without deleting the file |
| `NoDisplay` | No | Set to `true` to hide it from application menus while keeping it as an autostart entry |
| `Terminal` | No | Set to `true` if the program needs a terminal window |
| `Comment` | No | Short description (purely informational) |
| `Icon` | No | Icon name or path, used by GUI tools |

## Testing Before Relying on Autostart

Run the `Exec` line directly in a terminal first to confirm it works, before logging out to test the autostart entry itself:

```bash
/path/to/your/script.sh
```

## Optional: Graphical Management

If you'd rather not hand-edit `.desktop` files, GNOME Tweaks provides a Startup Applications-style panel that manages the same underlying files:

```bash
sudo apt install gnome-tweaks
```

:::note
This is different from running a program on **SSH login** to a server. Desktop autostart only applies to graphical login sessions. For something to run automatically on every SSH login instead, use a shell profile script under `/etc/profile.d/` — see [Install CUDA](../dl-setup/cuda.md) for an example of that pattern.
:::
