---
title: Ollama + Aider Local LLM Setup
sidebar_label: Local LLM
sidebar_position: 8
---

# Admin Guide: Ollama + Aider Local LLM Setup

## Architecture

- **Ollama server**: runs only on `node01` (control node), uses GPU (RTX PRO 6000 Blackwell, ~96GB VRAM)
- **Aider client**: runs on `node02` (or node01), connects to Ollama over network — no local Ollama install required
- **Model**: `qwen3-coder-next:latest` (51GB, 79.7B params)

---

## Setup Steps

### 1. Configure Ollama on node01 to listen on the network

Default Ollama binds to `127.0.0.1` only. Edit the systemd override:

```bash
sudo systemctl edit ollama
```

Resulting `/etc/systemd/system/ollama.service.d/override.conf`:

```ini
[Service]
Environment="OLLAMA_MODELS=/scratch/ollama/models"
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

Apply:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
sudo ss -tlnp | grep 11434   # should show *:11434
```

### 2. Install Aider as a shared system-wide tool

For a shared multi-user workstation, install Aider in a single shared venv owned by root, with the binary symlinked into `/usr/local/bin` so all users can run `aider` directly:

```bash
sudo python3 -m venv /opt/aider-venv
sudo /opt/aider-venv/bin/pip install --upgrade pip
sudo /opt/aider-venv/bin/pip install aider-chat
sudo ln -s /opt/aider-venv/bin/aider /usr/local/bin/aider
```

This avoids each user maintaining their own venv and ensures version consistency across the cluster.

### 3. Point client nodes at the Ollama server

On **node02** (and any other client nodes), set `OLLAMA_API_BASE` system-wide:

```bash
sudo tee /etc/profile.d/ollama.sh << 'EOF'
export OLLAMA_API_BASE=http://node01:11434
EOF
```

(Optional on node01 itself, pointing to `http://localhost:11434`, for consistency.)

Verify connectivity:

```bash
curl http://node01:11434/api/tags
```

This should return JSON listing the models available on node01.

### 4. Removing Ollama from a node (if mistakenly installed)

```bash
sudo systemctl stop ollama
sudo systemctl disable ollama
sudo rm /usr/local/bin/ollama
sudo gpasswd -d <username> ollama   # remove users from group first
sudo userdel ollama
sudo groupdel ollama
sudo rm /etc/systemd/system/ollama.service
sudo rm -rf /etc/systemd/system/ollama.service.d
sudo systemctl daemon-reload
sudo rm -rf /usr/share/ollama /scratch/ollama
```

---

## SLURM Enforcement Notes

- The Ollama server (systemd service, UID < 1000) is exempt from `slurm-enforcer` (`MIN_UID=1000`) — no action needed
- `aider` is a `python3` process and matches `COMPUTE_PROCS` — if run outside SLURM for >5 minutes it will be warned (4 min) and killed (5 min)
- Users must be instructed to wrap interactive Aider sessions in `srun` (see User Guide)
