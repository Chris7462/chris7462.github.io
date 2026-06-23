---
sidebar_position: 1
title: Install and Configure Ollama
description: Install Ollama, optionally configure model storage on /scratch/ollama/models, and pull models for local LLM inference
---

# Install and Configure Ollama

This guide covers installing Ollama, optionally changing the model storage location to `/scratch/ollama/models`, and pulling models for local inference.

## Step 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify the installation:

```bash
ollama --version
# Expected output: ollama version is 0.30.x or above
```

## Step 2. Change Model Storage Location (Optional)

By default, Ollama stores models under the system `ollama` user at `/usr/share/ollama/.ollama/models/`. To change the model storage location, redirect Ollama's model storage to `/scratch/ollama/models`.

### Create the model directory

```bash
sudo mkdir -p /scratch/ollama/models
sudo chown -R ollama:ollama /scratch/ollama/models
```

### Create the systemd override

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d/
sudo tee /etc/systemd/system/ollama.service.d/override.conf << 'EOF'
[Service]
Environment="OLLAMA_MODELS=/scratch/ollama/models"
EOF
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

Verify the service is running:

```bash
sudo systemctl status ollama
```

Models will now be stored at `/scratch/ollama/models`.

## Step 3. Download Models

```bash
# Primary model (recommended, 17 GB, fits entirely in 24 GB VRAM)
ollama pull qwen3.6:27b

# Backup model (51 GB, will offload to RAM, slower)
ollama pull qwen3-coder-next

# Reasoning model (43 GB, will offload to RAM, slower)
ollama pull deepseek-r1:70b
```

Verify the model list:

```bash
ollama list
```

Verify the model works:

```bash
ollama run qwen3.6:27b
# Type /bye to exit
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

## Monitor GPU Usage

```bash
# Check VRAM usage
nvidia-smi

# Check currently loaded Ollama models
ollama ps

# Stop a specific model to free VRAM
ollama stop qwen3-coder-next

# Remove a model from disk
ollama rm qwen3-coder-next
```

## Model Comparison

| Model | Size | VRAM Required | Speed | Best For |
|---|---|---|---|---|
| `qwen3.6-8k` | 17 GB | ~17 GB | Fast | Daily C++/CUDA development (recommended) |
| `qwen3.6:27b` | 17 GB | ~34 GB (large context) | Medium | Tasks requiring longer context |
| `qwen3-coder-next` | 51 GB | Offloads to RAM | Slow | Ultra-long context (256K) tasks |
| `deepseek-r1:70b` | 43 GB | Offloads to RAM | Slow | Complex reasoning tasks |

## Tips

- If `ollama ps` shows a high CPU/GPU offload ratio (e.g. `32%/68%`), reduce the context window size or stop other running models.
- Two models loaded simultaneously will compete for VRAM — make sure only one model is running at a time.
