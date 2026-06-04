---
title: User Management
sidebar_label: User Management
sidebar_position: 5
---

# User Management

This document covers the complete process for managing users on the HPC cluster, including creation, group access, disk quotas, Docker access, and account lifecycle management.

---

## 1. User Creation

### Create a New User

```bash
# Create user with home directory and bash shell
sudo useradd -m -s /bin/bash -g <primary_group> USERNAME

# Set password
sudo passwd USERNAME
```

### Verify User Created

```bash
id USERNAME
getent passwd USERNAME
```

### Useful Commands

```bash
# Change primary group
sudo usermod -g <group> USERNAME

# Delete the auto-created user private group if not needed
sudo groupdel USERNAME

# Verify
groups USERNAME
ls -la /home | grep USERNAME
```

---

## 2. Hide User from GDM Login Screen

For SSH-only users who should not appear on the graphical login screen:

### Create AccountsService Override

```bash
sudo vim /var/lib/AccountsService/users/USERNAME
```

Add:

```ini
[User]
SystemAccount=true
```

### Apply Changes

```bash
sudo systemctl restart gdm3
```

:::note
Hidden users can still use SSH with X11 forwarding (`ssh -X`) for GUI applications.
:::

---

## 3. Set User Display Name

To show a full name on the login screen instead of the username:

```bash
sudo chfn -f "Full Name Here" USERNAME
```

Restart GDM to see changes:

```bash
sudo systemctl restart gdm3
```

---

## 4. Disk Quota Setup

:::info
This section applies to **local disk quotas only** (e.g. on a local `/home` partition). If `/home` is mounted from the NAS via NFS, quota management is handled on the NAS via ZFS `userquota`. See the [Storage](./storage.md) doc for details.
:::

### Prerequisites (One-time Setup)

Ensure `/etc/fstab` has the `usrquota` option on `/home`:

```bash
# Before
/dev/disk/by-uuid/YOUR-UUID /home ext4 defaults 0 2

# After
/dev/disk/by-uuid/YOUR-UUID /home ext4 defaults,usrquota 0 2
```

Enable quotas:

```bash
sudo mount -o remount /home
sudo quotacheck -cum /home
sudo quotaon /home
```

### Set User Quota

```bash
# Format: soft limit, hard limit (in KB), soft inodes, hard inodes
sudo setquota -u USERNAME SOFT_KB HARD_KB 0 0 /home
```

### Common Quota Sizes

| Size | Value (KB) | Command |
|------|------------|---------|
| 50 GB | 52428800 | `sudo setquota -u USERNAME 52428800 52428800 0 0 /home` |
| 100 GB | 104857600 | `sudo setquota -u USERNAME 104857600 104857600 0 0 /home` |
| 200 GB | 209715200 | `sudo setquota -u USERNAME 209715200 209715200 0 0 /home` |
| 500 GB | 524288000 | `sudo setquota -u USERNAME 524288000 524288000 0 0 /home` |

### Verify Quota

```bash
sudo quota -u USERNAME
```

### View All Quotas

```bash
sudo repquota /home
```

---

## 5. Group Management

### Create a New Group

```bash
sudo groupadd GROUPNAME
```

### Add User to a Group

```bash
sudo usermod -aG GROUPNAME USERNAME
```

### Add User to Multiple Groups

```bash
sudo usermod -aG group1,group2,group3 USERNAME
```

### Remove User from a Group

```bash
sudo gpasswd -d USERNAME GROUPNAME
```

### View User's Groups

```bash
groups USERNAME
```

### View All Members of a Group

```bash
getent group GROUPNAME
```

---

## 6. Standard Group Access

### Standard User (Non-Admin)

For regular HPC users with Docker access:

```bash
sudo usermod -aG docker USERNAME
```

### Administrator

For users who need full system administration capabilities:

```bash
sudo usermod -aG adm,cdrom,sudo,dip,plugdev,lxd,docker USERNAME
```

| Group | Purpose |
|-------|---------|
| `sudo` | Administrative (root) access |
| `adm` | Read system logs |
| `cdrom` | Access CD-ROM devices |
| `dip` | Network dial-up access |
| `plugdev` | Access removable devices |
| `lxd` | LXD container management |
| `docker` | Docker container access |

---

## 7. Shared Folder Access Between Users

### Option A: Shared Directory (Recommended)

Create a shared folder that multiple users can access:

```bash
# Create shared folder
sudo mkdir /home/SHARED_FOLDER

# Set group ownership
sudo chown root:GROUPNAME /home/SHARED_FOLDER

# Set permissions (group read/write, setgid for new files)
sudo chmod 2770 /home/SHARED_FOLDER
```

:::note
The `2` (setgid) ensures new files created inside the folder inherit the group.
:::

### Option B: Access Each Other's Home Directories

Allow users in the same group to access each other's home folders:

```bash
# Create shared group
sudo groupadd GROUPNAME

# Add users to group
sudo usermod -aG GROUPNAME user1
sudo usermod -aG GROUPNAME user2

# Change group ownership of home directories
sudo chgrp GROUPNAME /home/user1
sudo chgrp GROUPNAME /home/user2

# Set permissions
sudo chmod 750 /home/user1   # group can read/list
sudo chmod 750 /home/user2
```

### Permission Reference

| Permission | Owner | Group | Others | Use Case |
|------------|-------|-------|--------|----------|
| `700` | rwx | --- | --- | Private, no sharing |
| `750` | rwx | r-x | --- | Group can read/list |
| `770` | rwx | rwx | --- | Group can read/write |
| `755` | rwx | r-x | r-x | Public read access |

---

## 8. Docker Access

### Add User to Docker Group

```bash
sudo usermod -aG docker USERNAME
```

### Verify

```bash
groups USERNAME
```

### Test Docker Access

```bash
sudo -u USERNAME docker run hello-world
```

:::note
Docker images stored in `/home/docker` are **not** subject to user quotas. Monitor usage with `docker system df`.
:::

---

## 9. Admin Privileges

### Grant Sudo Access

```bash
sudo usermod -aG sudo USERNAME
```

### Remove Sudo Access

```bash
sudo gpasswd -d USERNAME sudo
```

### Check Sudoers Files

```bash
# Check for user-specific sudoers files
ls -la /etc/sudoers.d/

# Remove if exists
sudo rm /etc/sudoers.d/USERNAME
```

### Verify Sudo Status

```bash
sudo -l -U USERNAME
```

---

## 10. Delete User Groups

When users share a common group, their individual private groups may no longer be needed.

:::warning
You cannot delete a group that is a user's primary group. Change it first.
:::

```bash
# Step 1: Change primary group to shared group
sudo usermod -g NEWGROUP USERNAME

# Step 2: Verify primary group changed
id USERNAME

# Step 3: Delete old group
sudo groupdel USERNAME

# Step 4: Verify
groups USERNAME
```

---

## 11. Force Password Change on First Login

```bash
sudo chage -d 0 USERNAME
```

### Verify

```bash
sudo chage -l USERNAME
# Output should show: Last password change : password must be changed
```

### Set Password Expiry Policy

```bash
# Password expires after 90 days
sudo chage -M 90 USERNAME

# Warn user 7 days before expiry
sudo chage -W 7 USERNAME

# View all password policies
sudo chage -l USERNAME
```

### Password Policy Options

| Option | Description |
|--------|-------------|
| `-d 0` | Force change on next login |
| `-M DAYS` | Maximum days before password expires |
| `-m DAYS` | Minimum days between password changes |
| `-W DAYS` | Warning days before expiration |
| `-E DATE` | Account expiration date (YYYY-MM-DD) |

---

## 12. User Modification Reference

```bash
# Change password
sudo passwd USERNAME

# Change shell
sudo usermod -s /bin/zsh USERNAME

# Change home directory
sudo usermod -d /home/newhome -m USERNAME

# Rename user
sudo usermod -l NEWNAME OLDNAME

# Lock account
sudo usermod -L USERNAME

# Unlock account
sudo usermod -U USERNAME

# Set account expiration
sudo usermod -e 2026-12-31 USERNAME

# Remove expiration
sudo usermod -e "" USERNAME

# Delete user and home directory
sudo userdel -r USERNAME

# Remove AccountsService file if exists
sudo rm /var/lib/AccountsService/users/USERNAME
```

### View User Info

```bash
id USERNAME
groups USERNAME
getent passwd USERNAME
sudo chage -l USERNAME
```

---

## 13. Quick Setup Scripts

### Standard HPC User (SSH-only, with quota)

```bash
#!/bin/bash
# Usage: ./add_hpc_user.sh USERNAME "Full Name" QUOTA_GB

USERNAME=$1
FULLNAME=$2
QUOTA_GB=$3
QUOTA_KB=$((QUOTA_GB * 1024 * 1024))

# Create user
sudo useradd -m -s /bin/bash $USERNAME
sudo passwd $USERNAME

# Force password change on first login
sudo chage -d 0 $USERNAME

# Set display name
sudo chfn -f "$FULLNAME" $USERNAME

# Hide from GDM
echo -e "[User]\nSystemAccount=true" | sudo tee /var/lib/AccountsService/users/$USERNAME

# Set quota (local disk only — for NAS quota see Storage doc)
sudo setquota -u $USERNAME $QUOTA_KB $QUOTA_KB 0 0 /home

# Add to docker group
sudo usermod -aG docker $USERNAME

# Verify
echo "=== User Setup Complete ==="
id $USERNAME
groups $USERNAME
sudo quota -u $USERNAME
sudo chage -l $USERNAME | grep "Last password change"
```

### Add Users to Shared Group

```bash
#!/bin/bash
# Usage: ./setup_shared_group.sh GROUPNAME USER1 USER2 ...

GROUPNAME=$1
shift
USERS=$@

# Create group
sudo groupadd $GROUPNAME 2>/dev/null || echo "Group exists"

# Add users
for USER in $USERS; do
    sudo usermod -aG $GROUPNAME $USER
    sudo chgrp $GROUPNAME /home/$USER
    sudo chmod 750 /home/$USER
    echo "Added $USER to $GROUPNAME"
done

# Verify
echo "=== Group Members ==="
getent group $GROUPNAME
```
