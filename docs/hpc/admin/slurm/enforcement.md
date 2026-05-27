---
title: SLURM Enforcement
sidebar_label: Enforcement
sidebar_position: 2
---

# SLURM Enforcement Setup

This document explains how to configure the workstation to enforce SLURM usage for compute jobs. Non-SLURM compute processes running longer than 5 minutes will be automatically terminated.

---

## Overview

### What It Does

- Monitors for compute-heavy processes (Python, Julia, MATLAB, etc.) running outside SLURM
- Warns users at 4 minutes with a terminal message
- Terminates the process at 5 minutes
- Logs all actions to the system journal

### How It Detects SLURM Processes

The enforcer uses layered detection to identify if a process is running under SLURM:

1. **Cgroup check** — Is the process in a SLURM cgroup?
2. **Environment check** — Does the process have `SLURM_JOB_ID` in its environment?
3. **Parent cgroup check** — Is the parent process in a SLURM cgroup?
4. **Grandparent cgroup check** — Is the grandparent in a SLURM cgroup?

This layered approach ensures processes spawned from IDEs like VS Code (running inside a SLURM session) are correctly identified as SLURM processes and not terminated.

### What It Ignores

- All desktop applications (browsers, file managers, terminals)
- System processes
- Processes running inside SLURM jobs
- Child processes of SLURM jobs (e.g., VS Code terminals within an `srun` session)

---

## Step 1: Create the Enforcer Script

```bash
sudo vim /usr/local/sbin/slurm-enforcer-daemon.sh
```

Paste the following content:

```bash
#!/bin/bash
# ==============================================================================
# SLURM Usage Enforcer Daemon
# Kills non-SLURM compute processes running longer than 5 minutes
# ==============================================================================

# Minimum UID to check (skip system users)
MIN_UID=1000

# Max allowed runtime in seconds outside SLURM
MAX_RUNTIME=300

# Warning time (seconds before kill)
WARN_TIME=240

# Check interval (seconds)
CHECK_INTERVAL=30

# Only target these compute-heavy processes
# Add more as needed for your workloads
# Note: Build tools (cmake, make, gcc, g++, nvcc) are excluded as they spawn many short-lived processes
COMPUTE_PROCS="python|python3|python3\.[0-9]+|julia|R|Rscript|matlab|MATLAB|java|node|ruby|perl|mpirun|mpiexec|torchrun|accelerate|deepspeed|horovodrun|ffmpeg|blender|openfoam|ansys|abaqus|comsol|docker|podman|containerd-shim"

# Track warned PIDs to avoid duplicate warnings
declare -A WARNED_PIDS

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1"
    logger -t slurm-enforcer "$1"
}

# Stop and remove containers started within the enforcement window
cleanup_user_containers() {
    local username=$1
    local max_age_minutes=$(( (MAX_RUNTIME + 60) / 60 ))

    local containers=$(docker ps -q --filter "status=running" 2>/dev/null)
    if [ -z "$containers" ]; then
        return
    fi

    for container_id in $containers; do
        local created=$(docker inspect --format='{{.Created}}' "$container_id" 2>/dev/null)
        if [ -z "$created" ]; then
            continue
        fi

        local created_epoch=$(date -d "$created" +%s 2>/dev/null)
        local now_epoch=$(date +%s)
        local age_seconds=$((now_epoch - created_epoch))
        local age_minutes=$((age_seconds / 60))

        if [ "$age_minutes" -le "$max_age_minutes" ]; then
            log "Stopping container $container_id (age: ${age_minutes}m) - started without SLURM"
            docker rm -f "$container_id" 2>/dev/null
        fi
    done
}

# Check if a process is running under SLURM using multiple methods
is_slurm_process() {
    local pid=$1

    # Method 1: Check if process cgroup contains slurm
    local cgroup=$(cat /proc/$pid/cgroup 2>/dev/null)
    if echo "$cgroup" | grep -qiE "slurm|node01__slu"; then
        return 0
    fi

    # Method 2: Check if SLURM_JOB_ID is in process environment
    if [ -r /proc/$pid/environ ]; then
        if tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -q "^SLURM_JOB_ID="; then
            return 0
        fi
    fi

    # Method 3: Check if parent process is in SLURM cgroup
    local ppid=$(ps -o ppid= -p $pid 2>/dev/null | tr -d ' ')
    if [ -n "$ppid" ] && [ "$ppid" -gt 1 ]; then
        local parent_cgroup=$(cat /proc/$ppid/cgroup 2>/dev/null)
        if echo "$parent_cgroup" | grep -qiE "slurm|node01_slu"; then
            return 0
        fi

        # Method 4: Check grandparent too (for deeply nested processes like VS Code terminals)
        local gpid=$(ps -o ppid= -p $ppid 2>/dev/null | tr -d ' ')
        if [ -n "$gpid" ] && [ "$gpid" -gt 1 ]; then
            local grandparent_cgroup=$(cat /proc/$gpid/cgroup 2>/dev/null)
            if echo "$grandparent_cgroup" | grep -qiE "slurm|node01_slu"; then
                return 0
            fi
        fi
    fi

    return 1
}

warn_user() {
    local uid=$1
    local pid=$2
    local comm=$3
    local etime=$4
    local username=$(id -nu "$uid" 2>/dev/null)
    local remaining=$((MAX_RUNTIME - etime))

    [ "${WARNED_PIDS[$pid]}" == "1" ] && return
    WARNED_PIDS[$pid]=1

    log "WARNING: User $username process '$comm' (PID $pid) will be killed in ~${remaining}s"

    local user_ttys=$(who | grep "^$username " | awk '{print $2}')
    for tty in $user_ttys; do
        echo -e "\n⚠️  WARNING: Your process '$comm' (PID $pid) has been running for ${etime}s outside SLURM.\nIt will be TERMINATED in ~${remaining} seconds.\n\nTo avoid this, use SLURM:\n  srun --pty bash\n  srun --gres=gpu:1 --time=02:00:00 --pty bash\n" | write "$username" "$tty" 2>/dev/null
    done
}

cleanup_warned_pids() {
    for pid in "${!WARNED_PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            unset WARNED_PIDS[$pid]
        fi
    done
}

check_processes() {
    ps -eo pid,uid,etimes,comm --no-headers 2>/dev/null | while read pid uid etime comm; do
        [ "$uid" -lt "$MIN_UID" ] && continue
        echo "$comm" | grep -qE "^($COMPUTE_PROCS)$" || continue

        if is_slurm_process "$pid"; then
            continue
        fi

        [ "$etime" -lt "$WARN_TIME" ] && continue

        local username=$(id -nu "$uid" 2>/dev/null)

        # Warning phase (4-5 minutes)
        if [ "$etime" -ge "$WARN_TIME" ] && [ "$etime" -lt "$MAX_RUNTIME" ]; then
            warn_user "$uid" "$pid" "$comm" "$etime"
            continue
        fi

        # Kill phase (>5 minutes)
        if [ "$etime" -ge "$MAX_RUNTIME" ]; then
            log "KILLING non-SLURM process: PID=$pid USER=$username CMD=$comm RUNTIME=${etime}s"

            if echo "$comm" | grep -qE "^(docker|podman)$"; then
                log "Cleaning up containers started by $username"
                cleanup_user_containers "$username"
            fi

            local user_ttys=$(who | grep "^$username " | awk '{print $2}')
            for tty in $user_ttys; do
                echo -e "\n🛑 TERMINATED: Process '$comm' (PID $pid) killed after ${etime}s.\nPlease use SLURM for compute jobs.\n" | write "$username" "$tty" 2>/dev/null
            done

            # Graceful termination
            kill -TERM "$pid" 2>/dev/null

            # Force kill after 5 seconds if still running
            (
                sleep 5
                if kill -0 "$pid" 2>/dev/null; then
                    kill -KILL "$pid" 2>/dev/null
                    log "Force killed PID=$pid (did not respond to SIGTERM)"
                fi
            ) &
        fi
    done
}

# Main loop
log "SLURM enforcer daemon started (MAX_RUNTIME=${MAX_RUNTIME}s, CHECK_INTERVAL=${CHECK_INTERVAL}s)"
log "Targeting compute processes: $COMPUTE_PROCS"
log "Detection methods: cgroup, SLURM_JOB_ID env, parent/grandparent cgroup"

while true; do
    check_processes
    cleanup_warned_pids
    sleep "$CHECK_INTERVAL"
done
```

Make it executable:

```bash
sudo chmod +x /usr/local/sbin/slurm-enforcer-daemon.sh
```

---

## Step 2: Create the Systemd Service

```bash
sudo vim /etc/systemd/system/slurm-enforcer.service
```

Paste the following content:

```ini
[Unit]
Description=SLURM Usage Enforcer - Terminates non-SLURM compute jobs
Documentation=man:slurmd(8)
After=network.target slurmd.service
Wants=slurmd.service

[Service]
Type=simple
ExecStart=/usr/local/sbin/slurm-enforcer-daemon.sh
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security hardening
NoNewPrivileges=no
ProtectSystem=strict
ProtectHome=read-only
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable slurm-enforcer
sudo systemctl start slurm-enforcer
```

---

## Step 3: Add Login Message (MOTD)

Create a message that users see when they log in:

```bash
sudo vim /etc/motd
```

Paste the following:

```
================================================================================
                        WORKSTATION USAGE POLICY
================================================================================

All compute jobs MUST be submitted through SLURM.

  Interactive session:    srun --pty bash
  Interactive with GPU:   srun --gres=gpu:1 --time=02:00:00 --pty bash
  Batch job:              sbatch job.sh

⚠️  Processes running outside SLURM for more than 5 minutes will be TERMINATED.

Run 'sinfo' to check cluster status.
Run 'squeue' to see running jobs.

================================================================================
```

---

## Step 4: Add SLURM Shell Prompt Indicator

Help users know when they're inside a SLURM job by modifying the system bashrc:

```bash
sudo vim /etc/bash.bashrc
```

Add the following at the end of the file:

```bash
# Add SLURM indicator to prompt
__slurm_prompt() {
    if [ -n "$SLURM_JOB_ID" ] && [[ "$PS1" != *"slurm-"* ]]; then
        if [ -n "$SLURM_GPUS_ON_NODE" ]; then
            PS1="${PS1//\\h/slurm-$SLURM_JOB_ID:gpu}"
        else
            PS1="${PS1//\\h/slurm-$SLURM_JOB_ID}"
        fi
    fi
}
PROMPT_COMMAND="__slurm_prompt${PROMPT_COMMAND:+;$PROMPT_COMMAND}"
```

This replaces the hostname in the prompt to show the SLURM job ID:

- Regular job: `user@node01:~$` → `user@slurm-12345:~$`
- GPU job: `user@node01:~$` → `user@slurm-12345:gpu:~$`

---

## Step 5: Verify Installation

Check the service is running:

```bash
sudo systemctl status slurm-enforcer
```

Expected output:

```
● slurm-enforcer.service - SLURM Usage Enforcer - Terminates non-SLURM compute jobs
     Loaded: loaded (/etc/systemd/system/slurm-enforcer.service; enabled)
     Active: active (running)
```

Watch the logs in real-time:

```bash
sudo journalctl -u slurm-enforcer -f
```

---

## Step 6: Test the Enforcement

**Test 1: Process outside SLURM (should be killed)**

```bash
python3 -c "import time; print('Starting...'); time.sleep(600)"
```

Expected behavior: warning message appears at 4 minutes, process terminated at 5 minutes.

**Test 2: Process inside SLURM (should NOT be killed)**

```bash
srun --time=00:10:00 python3 -c "import time; print('Starting...'); time.sleep(600)"
```

Expected behavior: process runs without interruption.

**Test 3: VS Code inside SLURM session (should NOT be killed)**

```bash
srun --mem=2G --time=00:30:00 --pty bash
code ./
# Open a terminal in VS Code and run:
# python3 -c "import time; print('Starting...'); time.sleep(600)"
```

Expected behavior: process runs without interruption. The enforcer detects `SLURM_JOB_ID` in the process environment.

---

## Configuration Options

### Adjust Time Limits

Edit `/usr/local/sbin/slurm-enforcer-daemon.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_RUNTIME` | 300 | Seconds before process is killed (5 min) |
| `WARN_TIME` | 240 | Seconds before warning is sent (4 min) |
| `CHECK_INTERVAL` | 30 | How often to check for processes (seconds) |

After editing, restart the service:

```bash
sudo systemctl restart slurm-enforcer
```

### Add New Compute Processes

To monitor additional programs, edit the `COMPUTE_PROCS` line in the script:

```bash
COMPUTE_PROCS="python|python3|julia|R|...|your_new_program"
```

Restart the service after changes.

---

## Managing the Service

| Action | Command |
|--------|---------|
| Start | `sudo systemctl start slurm-enforcer` |
| Stop | `sudo systemctl stop slurm-enforcer` |
| Restart | `sudo systemctl restart slurm-enforcer` |
| Status | `sudo systemctl status slurm-enforcer` |
| View logs | `sudo journalctl -u slurm-enforcer -f` |
| Disable on boot | `sudo systemctl disable slurm-enforcer` |
| Enable on boot | `sudo systemctl enable slurm-enforcer` |

---

## Troubleshooting

### Service Won't Start

Check for syntax errors:

```bash
sudo bash -n /usr/local/sbin/slurm-enforcer-daemon.sh
sudo journalctl -u slurm-enforcer -n 50
```

### Users Not Receiving Warnings

The `write` command requires:
- User must be logged in with a terminal
- `mesg y` must be enabled (default on most systems)

### Process Not Being Detected

Verify the process name matches the `COMPUTE_PROCS` list:

```bash
ps -eo comm | grep -E "python|julia|R"
```

The process name in `ps` output must match exactly.

### VS Code Processes Being Killed Inside SLURM Session

Verify the detection is working:

```bash
# Find the Python process PID
ps aux | grep python

# Check if SLURM_JOB_ID is in its environment
cat /proc/<PID>/environ | tr '\0' '\n' | grep SLURM

# Check parent process cgroup
ps -o ppid= -p <PID>
cat /proc/<PARENT_PID>/cgroup
```

If `SLURM_JOB_ID` is not being inherited, ensure the user launched VS Code from within the `srun` session, not from a separate terminal.

### Cgroup Pattern Not Matching

Different SLURM installations may use different cgroup names. Check what your system uses:

```bash
srun --time=00:05:00 bash -c 'cat /proc/self/cgroup'
```

If needed, update the cgroup pattern in the script:

```bash
# Find this line in the script
echo "$cgroup" | grep -qiE "slurm|node01_slu"

# Add your pattern
echo "$cgroup" | grep -qiE "slurm|node01_slu|your_pattern"
```

---

## Summary of Files

| File | Purpose |
|------|---------|
| `/usr/local/sbin/slurm-enforcer-daemon.sh` | Main enforcement script |
| `/etc/systemd/system/slurm-enforcer.service` | Systemd service definition |
| `/etc/motd` | Login message for users |
| `/etc/bash.bashrc` | Shell prompt indicator (append to end) |
