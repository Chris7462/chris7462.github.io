---
title: User Management
sidebar_label: User Management
sidebar_position: 5
---

# User Management

This document covers the complete process for managing users on the HPC cluster, including creation, group access, disk quotas, Docker access, account lifecycle management, and syncing accounts across cluster nodes.

---

## 1. User Creation

### Create a New User

```bash
# Create user with home directory and bash shell
sudo useradd -m -s /bin/bash -g <primary_group> USERNAME

# Set password
sudo passwd USERNAME
```

:::note
This is run on **node01**, where `-m` creates the user's home directory on the NFS-shared `/home`. Since `/home` is shared, this directory becomes immediately visible on node02 as well — so on node02 the user should be created **without** `-m`/with `-M` (see [Syncing User Accounts](#14-syncing-user-accounts-between-node01-and-node02) below).
:::

:::note
After creating a user on node01, you must also replicate the account (UID/GID, groups, and password) on node02. See [Syncing User Accounts Between node01 and node02](#14-syncing-user-accounts-between-node01-and-node02) below.
:::

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

## 13. Adding to or Removing from `vglusers`

For users who need VirtualGL access for GPU-accelerated remote rendering:

```bash
# Add user
sudo usermod -aG vglusers USERNAME

# Remove user
sudo deluser USERNAME vglusers
```

:::note
Users must log out and back in (or reboot) for group changes to take effect. See the [VirtualGL](./virtualgl.md) guide for details.
:::

---

## 14. Syncing User Accounts Between node01 and node02

This section covers how to replicate a user account (UID/GID, group memberships, and password) from the head node (`node01`) to the compute node (`node02`) for the two-node SLURM cluster.

:::warning
Since `/home` is shared via NFS, user accounts must exist locally with **matching UID/GID** on both nodes — this is required for SLURM to function correctly (see [Adding a Node](./slurm/adding-a-node.md)). However, home directories should **not** be created by `useradd` on node02 (they already exist on the NFS share).
:::

### Step 1: List Users

To list all regular (non-system) usernames on a node:

```bash
awk -F: '$3 >= 1000 && $3 < 65534 {print $1}' /etc/passwd
```

### Step 2: Check User/Group Info

On `node01`, check the UID, primary GID, and supplementary groups for the new user:

```bash
id username
groups username
```

### Step 3: Create the User on node02

Ensure any required groups exist on `node02` with matching GIDs first:

```bash
sudo groupadd -g <gid> <groupname>
```

Then create the user **without** creating a home directory (NFS-shared `/home`):

```bash
sudo useradd -u <uid> -g <primary_gid> -s /bin/bash -c "Full Name" -d /home/username -M username
sudo usermod -aG docker,tools,... username
```

| Flag | Meaning |
|------|---------|
| `-M` | Do not create home directory |
| `-G` | Supplementary groups (replaces all existing supplementary groups) |

### Step 4: Sync Group Membership

Compare and align supplementary groups so they match between nodes:

```bash
for u in user1 user2 user3; do
  echo "== $u =="
  echo -n "groups: "; groups $u
  echo -n "id:     "; id $u
done
```

Run this on both `node01` and `node02` and compare output side by side. Fix mismatches with:

```bash
sudo usermod -G group1,group2,... username
```

:::warning
`usermod -G` **replaces** all supplementary groups for that user, so list all groups the user should belong to.
:::

### Step 5: Sync Passwords

Passwords are stored as hashes in `/etc/shadow`. Never edit `/etc/shadow` directly with `echo >>` — use `chpasswd -e` or `usermod -p`, which write safely with proper locking.

#### Get the Hash from node01

```bash
sudo grep ^username: /etc/shadow
```

This returns:

```
username:$y$j9T$.../...:lastchange:0:99999:7:::
```

Copy the `username:hash` portion (first two fields, separated by `:`).

#### Apply the Hash on node02

**Option A — `usermod -p` (single user, recommended for one-off changes):**

```bash
sudo usermod -p '$y$j9T$...' username
```

**Option B — `chpasswd` (heredoc, useful for syncing multiple users at once):**

```bash
sudo chpasswd -e <<'EOF'
username1:$y$j9T$...
username2:$y$j9T$...
EOF
```

#### Verify

Run on both nodes and compare:

```bash
for u in user1 user2 user3; do
  sudo grep ^$u: /etc/shadow | cut -d: -f1,2
done
```

### `/etc/shadow` Field Reference

The shadow file format is:

```
username:hash:lastchange:min:max:warn:inactive:expire:reserved
```

| Field | Meaning |
|---|---|
| `hash` | Encrypted password hash |
| `lastchange` | Days since Unix epoch (1970-01-01) when password was last changed |
| `min` | Minimum days before password can be changed again |
| `max` | Maximum days password is valid (`99999` ≈ never expires) |
| `warn` | Days before expiration to warn the user |
| `inactive` | Days after expiration before account is disabled |
| `expire` | Account expiration date (days since epoch) |

:::note
When using `chpasswd -e`, verify that `lastchange` updates correctly and isn't left blank or reset to `0`.
:::

### Long-Term Recommendation

For more than a handful of users, consider centralizing authentication with **LDAP/SSSD** so accounts only need to be created once and are automatically consistent across all nodes.
