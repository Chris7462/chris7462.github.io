---
title: Ollama Local LLM Setup
sidebar_label: Local LLM
sidebar_position: 8
---

# Admin Guide: Ollama Local LLM Setup

## Architecture

- **Ollama server**: runs only on `node01` (control node), uses GPU (RTX PRO 6000 Blackwell, ~96GB VRAM)
- **Client**: runs on `node01`, `node02`, or a user's laptop — connects to Ollama over the network
- **No client tool is installed system-wide** — users install their preferred coding assistant (Aider, OpenCode, etc.) in their own environment. See the User Guide for details.

### Available Models

| Model | Size | Parameters | Notes |
|---|---|---|---|
| `qwen3-coder-next:latest` | 51 GB | 79.7B | Default — best for complex coding tasks |
| `qwen3.6:27b` | 17 GB | 27.8B | Smaller/faster alternative |
| `deepseek-r1:70b` | 43 GB | 70.6B | Strong reasoning model |

---

## Setup Steps

:::note
This guide assumes Ollama is already installed on `node01` and models have been pulled. See [Install and Configure Ollama](../../local-llm-setup/ollama.md) for installation instructions.

The network-binding configuration below replaces the default single-node setup — do not set `OLLAMA_HOST` or system-wide environment variables as described in that guide.
:::

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

### 2. Verify the server is reachable

Check the native Ollama endpoint:

```bash
curl http://node01:11434/api/tags
```

Check the OpenAI-compatible endpoint (required by OpenCode and other tools):

```bash
curl http://node01:11434/v1/models
```

Both should return JSON listing the available models.

### 3. Removing Ollama from a node (if mistakenly installed)

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
- Client tools (Aider, OpenCode, etc.) are user-installed in their own environments — users are responsible for wrapping their sessions in `srun` (see User Guide)
- Any compute process matching `COMPUTE_PROCS` running outside SLURM for more than 5 minutes will be warned (4 min) and killed (5 min) regardless of which client tool is used
