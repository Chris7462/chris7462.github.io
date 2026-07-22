---
title: Migrating to Rootless Docker
sidebar_label: Rootless Docker Migration
sidebar_position: 9
description: Migrating node01 from a shared rootful Docker daemon to per-user rootless Docker
---

# Migrating to Rootless Docker

This document covers migrating a shared, rootful Docker installation to per-user
[rootless mode](https://docs.docker.com/engine/security/rootless/), including the
NFS `data-root` pitfall specific to this cluster's NFS-shared `/home` layout.

:::info Why rootless
Membership in the `docker` group is effectively root-equivalent under a rootful
daemon — any member can bind-mount the host filesystem into a container and
gain full host access. Rootless mode runs the daemon itself inside a user
namespace, with no shared daemon and no `docker` group requirement.
:::

---

## 1. Architecture Change

Rootless mode is **per-user**, not a config toggle on the existing shared daemon.
Each user runs their own `dockerd` instance, listening on their own socket at
`/run/user/<uid>/docker.sock` (or `~/.docker/run/docker.sock` on non-systemd
sessions — see [Troubleshooting](#troubleshooting)). There is no shared image
store across users; each user pulls/builds their own images independently.

This replaces the previous model of one shared rootful daemon with `docker`
group membership and a centralized `data-root` at `/home/docker` (or later
`/scratch/docker`).

---

## 2. Prerequisites

### subuid / subgid ranges

Each user needs at least 65,536 subordinate UIDs/GIDs in `/etc/subuid` and
`/etc/subgid`. On this cluster these are auto-assigned sequentially by
`useradd`, so most users already have a valid range. Check with:

```bash
cat /etc/subuid
cat /etc/subgid
```

Ranges must not overlap between users. If a user is missing an entry:

```bash
sudo usermod --add-subuids <range> --add-subgids <range> USERNAME
```

### uidmap package

```bash
which newuidmap newgidmap || sudo apt install -y uidmap
```

### Identify affected users

The full migration list is everyone currently in the `docker` group:

```bash
getent group docker | cut -d: -f4 | tr ',' '\n'
```

---

## 3. Decommission the Shared Rootful Daemon

:::warning
This stops Docker for every user immediately. Check for running containers
and logged-in users first, and give a heads-up before proceeding.
:::

```bash
docker ps -a
who
```

If clear:

```bash
sudo systemctl disable --now docker.service docker.socket
sudo rm /var/run/docker.sock
```

### Clean up the old rootful config

The rootful daemon and rootless daemons use **completely separate config
files** — `/etc/docker/daemon.json` and `/etc/containerd/config.toml` are
ignored by rootless Docker entirely. Once rootful is decommissioned, revert
them for hygiene:

```bash
sudo mv /etc/docker/daemon.json /etc/docker/daemon.json.bak-rootful
```

In `/etc/containerd/config.toml`, revert the custom containerd root back to
the commented default:

```toml
#root = "/var/lib/containerd"
```

### Disable the system-wide containerd service

Each rootless daemon bundles its own `containerd` instance
(`/run/user/<uid>/docker/containerd`), so the system-wide service becomes
unused. Confirm nothing else depends on it before disabling:

```bash
sudo ss -x | grep containerd
ls -la /run/containerd/containerd.sock
sudo systemctl disable --now containerd
```

### Clean up old rootful image data

```bash
du -sh /scratch/docker   # or wherever the old data-root pointed
sudo rm -rf /scratch/docker
```

Do this only after confirming no images need to be preserved — rootless
users will need to re-pull/rebuild anything they still need.

---

## 4. The NFS `data-root` Pitfall

:::danger Critical
`/home` on this cluster is NFS-mounted from the NAS. Rootless Docker's
default data directory, `~/.local/share/docker`, therefore sits on NFS.
**NFS is not a supported location for the Docker data-root** — unprivileged
overlayfs mounts fail there with:

```
failed to mount ... fstype: overlay ... err: permission denied
```

This is not a permissions bug to chase — it's an unsupported configuration.
The fix is to point `data-root` at local disk instead.
:::

Each user gets their own directory under the local `/scratch` filesystem
(`/dev/nvme0n1p5`, ext4 — not NFS):

```bash
sudo mkdir -p /scratch/docker/USERNAME
sudo chown USERNAME:<primary-group> /scratch/docker/USERNAME
```

:::warning Don't assume group == username
Look up each user's actual primary group — it does **not** always match
their username on this cluster (e.g. `itca`, `tools`, `kk0` are all in use
as primary groups here):

```bash
id -gn USERNAME
```
:::

This directory is referenced via `data-root` in the user's
`~/.config/docker/daemon.json` (see [step 6](#6-install-rootless-docker)).

---

## 5. NVIDIA GPU Support (CDI)

Rootless GPU access works but needs the [Container Device Interface
(CDI)](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)
path rather than the legacy `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=...`
combination, which is intended for rootful use.

### One-time, system-wide: generate the CDI spec

This only needs to run **once per host**, as root — not per user:

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo chmod 644 /etc/cdi/nvidia.yaml
```

### Per user: configure the NVIDIA runtime for rootless

```bash
nvidia-ctk runtime configure --runtime=docker --config=$HOME/.config/docker/daemon.json
```

### GPU test command

Use the native `--device` CDI syntax (no `--runtime=nvidia` needed) or the
familiar `--gpus all` flag — both work once CDI is enabled:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

:::tip
If a specific image tag pulls but fails with `exec format error`, check
whether it actually publishes an `amd64` build — some tags (e.g. certain
TensorRT devel images) are arm64-only. Validate GPU access first with a
well-known multi-arch tag like the one above before troubleshooting a
specific project image.
:::

---

## 6. Install Rootless Docker

:::danger Must be a real login session
Rootless Docker's systemd integration requires an actual login session
(SSH, console) — **not** `sudo su - USERNAME`. Switching users via `sudo su -`
does not create a D-Bus/systemd user session, causing the installer to fall
back to a non-systemd mode and `systemctl --user` commands to fail with
`Failed to connect to bus: No medium found`. Each user must SSH in directly
under their own account.
:::

Run as the target user (no `sudo`):

```bash
dockerd-rootless-setuptool.sh install
```

Add environment variables to `~/.bashrc`:

```bash
export PATH=/usr/bin:$PATH
export DOCKER_HOST=unix:///run/user/<uid>/docker.sock
```

Write `~/.config/docker/daemon.json` with both the NVIDIA runtime and the
local-disk `data-root`:

```json
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    },
    "data-root": "/scratch/docker/USERNAME"
}
```

Restart to apply:

```bash
systemctl --user restart docker
```

### Enable linger (admin step, once per user)

Without this, the rootless daemon stops the moment the user logs out:

```bash
sudo loginctl enable-linger USERNAME
```

### Verify

```bash
docker info | grep -i rootless
docker info | grep -i "docker root dir"
docker run --rm hello-world
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

---

## 7. Remove Docker Group Membership

Once a user's rootless setup is confirmed working, remove their now-unneeded
`docker` group membership (the group is only meaningful for the decommissioned
rootful daemon):

```bash
sudo gpasswd -d USERNAME docker
```

Confirm the group is shrinking as expected:

```bash
getent group docker
```

Note that an already-open shell for that user will still show `docker` in
`groups` output until their next login — this is cosmetic only, since the
rootful socket no longer exists to grant access through.

---

## 8. Multi-Node Clusters: Shared NFS Home Gotchas

Since `/home` (and therefore `~/.bashrc`, `~/.docker/`, `~/.config/docker/`)
is NFS-shared across nodes, state from a rootless install on one node is
**visible on every other node**, even before that node has its own rootless
daemon running. This caused two confusing failures when migrating node02
after node01 was already done:

### Stale `DOCKER_HOST` / CLI context

A user who completed setup on node01 will have `DOCKER_HOST` exported in
`~/.bashrc` and the Docker CLI context set to `rootless` in
`~/.docker/config.json` — both picked up automatically on node02 as well.
Until that user has *also* completed rootless setup on node02, `docker`
commands there fail with:

```
failed to connect to the docker API at unix:///run/user/<uid>/docker.sock:
... connect: no such file or directory
```

To check node02's actual (pre-migration) rootful state during the transition:

```bash
unset DOCKER_HOST
docker context use default
docker ps -a
```

:::tip
Once a user has completed rootless setup on **both** nodes, this resolves
itself permanently — the socket path convention (`/run/user/<uid>/docker.sock`)
is UID-based and consistent across nodes, so the inherited env var/context
becomes correct rather than stale. The failure only exists in the gap
between migrating node01 and node02 for a given user. Migrate a user's
remaining nodes in one continuous session where practical, to minimize this
window.
:::

### `data-root` may already differ from node to node

Don't assume every node's rootful `daemon.json` used the same `data-root`.
On this cluster, node01's rootful config pointed at `/home/docker` (on NFS —
the actual source of its overlay-mount failures), while node02's already
pointed at `/scratch/docker` (already local disk). Always check each node's
own `/etc/docker/daemon.json` before assuming the NFS pitfall applies there
too.

### `/etc/subuid` / `/etc/subgid` are per-node, not shared

Even though user accounts themselves are shared via NFS-backed `/home`,
subuid/subgid ranges are local files on each node and are **not**
synchronized. Always re-check ranges on each node before assigning a new
one — the highest existing block, and even whether a given user has a range
at all, can differ from node to node.

---

## 9. Migrating a Service Account (No Interactive Login)

Not every `docker`-group member is a human. Service accounts — like a CI
agent (e.g. `buildkite-agent`) — need a different approach, since rootless
Docker's install requires a genuine systemd user session, and service
accounts don't get one from a normal SSH login (they may have no reason to
ever be logged into interactively at all).

### Decision point: is rootless even the right fix here?

Before migrating a service account, consider whether it's warranted. The
threat model motivating this whole migration — a person misusing shared
`docker` group access — doesn't map the same way onto a single-purpose,
non-human account. Options, roughly in order of increasing effort:

1. **Leave the rootful daemon decommissioned and give the service account
   its own rootless daemon** (this section) — most consistent with the rest
   of the migration, but more setup.
2. **Move the service's Docker workload off this host** — sidesteps the
   problem if there's a more appropriate place to run it.
3. **Reintroduce a scoped rootful daemon just for this one account** — avoids
   the extra setup, but partially reverses the security improvement this
   migration was for. Generally the least preferable option.

This cluster went with option 1.

### Prerequisites specific to a service account

Check for a shell and a subuid/subgid range, same as any user:

```bash
grep SERVICE_ACCOUNT /etc/passwd    # confirm it has a real shell, not /nologin
grep ^SERVICE_ACCOUNT: /etc/subuid
grep ^SERVICE_ACCOUNT: /etc/subgid
```

If missing a range, assign one (check the current highest block on **this**
node first — see [section 8](#8-multi-node-clusters-shared-nfs-home-gotchas)):

```bash
sudo usermod --add-subuids <range> --add-subgids <range> SERVICE_ACCOUNT
```

Enable linger — this is what keeps a systemd user session alive for the
account with no one logged in, and is required for the service's daemon to
survive beyond an interactive session:

```bash
sudo loginctl enable-linger SERVICE_ACCOUNT
```

### Get a real session without SSH

Use `machinectl shell` rather than `sudo su -`, since `su` does not create
a systemd/D-Bus user session (see the warning in
[section 6](#6-install-rootless-docker)):

```bash
sudo machinectl shell SERVICE_ACCOUNT@ /bin/bash
```

If unavailable:
```bash
sudo apt install -y systemd-container
```

Confirm the session is real before installing:

```bash
systemctl --user status
```

### Check home directory ownership before installing

Service-account home directories (especially ones under `/var/lib/`,
installed by a package manager) may have subdirectories owned by `root`
even though the account itself owns the rest of its home — commonly
`.config`, if some other root-run process created it first. This causes the
rootless installer to fail with a permission error while creating
`.config/systemd`. Check and fix before installing:

```bash
ls -la /var/lib/SERVICE_ACCOUNT/
sudo chown SERVICE_ACCOUNT:$(id -gn SERVICE_ACCOUNT) /var/lib/SERVICE_ACCOUNT/.config
```

### Install (same as any user, from inside the `machinectl` session)

Follow [section 6](#6-install-rootless-docker) as normal — install, env
vars, `daemon.json` with local `data-root`, NVIDIA config if needed, test
with `hello-world`.

### Wire the actual service to the new socket

The interactive `machinectl` session only proves the daemon works — the
real service (its systemd unit) needs to be told to use it explicitly,
since a system-level unit does not source the account's `~/.bashrc`:

```bash
sudo systemctl edit SERVICE_ACCOUNT
```

Add an override:

```ini
[Unit]
After=user@<uid>.service

[Service]
Environment=DOCKER_HOST=unix:///run/user/<uid>/docker.sock
Environment=PATH=/usr/bin:/usr/local/bin:/bin:/usr/sbin:/sbin
```

The `After=user@<uid>.service` ordering matters: without it, the service
could start before the rootless Docker user session is up on a fresh boot.
The explicit `PATH` matters too if the original unit doesn't set one —
systemd's default minimal `PATH` may not include `/usr/bin`.

Apply and verify:

```bash
sudo systemctl daemon-reload
sudo systemctl restart SERVICE_ACCOUNT
sudo systemctl show SERVICE_ACCOUNT -p Environment
```

Confirm with a real workload from the service itself (e.g. an actual CI
build), not just the interactive test — that's the only way to validate the
full chain: linger → user session → rootless daemon → service unit → job.

---

## 10. Migration Checklist (per user)

1. Confirm subuid/subgid range exists
2. Admin: create and `chown` `/scratch/docker/USERNAME` (correct primary group)
3. User SSHes in directly (not `sudo su -`)
4. `dockerd-rootless-setuptool.sh install`
5. Add `PATH`/`DOCKER_HOST` to `~/.bashrc`
6. Write `~/.config/docker/daemon.json` (NVIDIA runtime + local `data-root`)
7. `systemctl --user restart docker`
8. Admin: `sudo loginctl enable-linger USERNAME`
9. Test: `docker run --rm hello-world` and the GPU test command
10. Admin: `sudo gpasswd -d USERNAME docker` once confirmed
