---
sidebar_position: 1
title: Install and Configure Ollama
description: Install Ollama, configure shared model storage on NFS, and pull models for local LLM inference
---

# Install and Configure Ollama

This guide covers installing Ollama, optionally configuring shared model storage on an NFS home, and pulling models for local inference.

## Step 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify the installation:

```bash
ollama --version
# Expected output: ollama version is 0.30.x or above
```

## Step 2. Shared Model Storage on NFS (Optional)

By default, Ollama stores models under the system `ollama` user at `/usr/share/ollama/.ollama/models/` — not in your home directory. On a multi-node setup where `/home` is mounted over NFS, this means models pulled on one node are not visible on others.

To fix this, redirect Ollama's model storage to a shared path on NFS.

### Create the shared model directory

Run this once on either node (NFS so it propagates to all nodes automatically):

```bash
sudo mkdir -p /home/ollama
sudo chown -R ollama:ollama /home/ollama
```

### Create the systemd override on each node

Run the following on **node1** and **node2**:

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d/
sudo tee /etc/systemd/system/ollama.service.d/override.conf << 'EOF'
[Service]
Environment="OLLAMA_MODELS=/home/ollama/models"
EOF
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

Verify the service is running:

```bash
sudo systemctl status ollama
```

Models pulled on either node will now be stored at `/home/ollama/models` on the NAS and visible to all nodes.

## Step 3. Download Models

```bash
# Primary model (recommended, 17 GB, fits entirely in 24 GB VRAM)
ollama pull qwen3.6:27b

# Backup model (51 GB, will offload to RAM, slower)
ollama pull qwen3-coder-next
```

Verify the model list:

```bash
ollama list
```

## Step 4. Create a Modelfile to Limit Context Window (Optional)

If your GPU has limited VRAM (e.g. 24 GB), the default context window may cause the KV cache to overflow VRAM, offloading part of the weights to RAM and slowing inference significantly. You can limit it with a Modelfile.

```bash
cat << 'EOF' > /tmp/Modelfile
FROM qwen3.6:27b
PARAMETER num_ctx 8192
EOF

ollama create qwen3.6-8k -f /tmp/Modelfile
```

:::note
8192 tokens is sufficient for most coding tasks. If you need to load larger files, increase to `16384`.
:::

Verify the model was created:

```bash
ollama list
# Should show qwen3.6-8k:latest
```

## Step 5. Set System-wide Environment Variables

Create `/etc/profile.d/ollama.sh` to make `OLLAMA_API_BASE` available to all users:

```bash
sudo tee /etc/profile.d/ollama.sh << 'EOF'
export OLLAMA_API_BASE=http://localhost:11434
EOF
source /etc/profile.d/ollama.sh
```

## Monitor GPU Usage

```bash
# Check VRAM usage
nvidia-smi

# Check currently loaded Ollama models
ollama ps

# Stop a specific model to free VRAM
ollama stop qwen3-coder-next
```

## Model Comparison

| Model | Size | VRAM Required | Speed | Best For |
|---|---|---|---|---|
| `qwen3.6-8k` | 17 GB | ~17 GB | Fast | Daily C++/CUDA development (recommended) |
| `qwen3.6:27b` | 17 GB | ~34 GB (large context) | Medium | Tasks requiring longer context |
| `qwen3-coder-next` | 51 GB | Offloads to RAM | Slow | Ultra-long context (256K) tasks |

## Tips

- If `ollama ps` shows a high CPU/GPU offload ratio (e.g. `32%/68%`), reduce the context window size or stop other running models.
- Two models loaded simultaneously will compete for VRAM — make sure only one model is running at a time.
