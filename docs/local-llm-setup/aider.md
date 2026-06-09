---
sidebar_position: 1
title: Local LLM Coding Assistant with Aider
description: Set up Aider with Ollama and Qwen 3.6 27B for local C++/CUDA/ROS 2 development on Ubuntu
---

# Local LLM Coding Assistant with Aider

This guide covers setting up a fully local LLM coding assistant using [Ollama](https://ollama.com) and [Aider](https://aider.chat) — no cloud API required.

## Environment

- **Machine**: Lenovo Legion (Ubuntu 24.04)
- **GPU**: NVIDIA RTX 5090 (24 GB VRAM)
- **Model**: Qwen 3.6 27B
- **Use case**: Local C++/CUDA/ROS 2 development assistance

## Step 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify the installation:

```bash
ollama --version
# Expected output: ollama version is 0.30.x or above
```

## Step 2. Download Models

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

## Step 3. Create a Modelfile (Limit Context Window)

The default context window is too large and will cause the KV cache to overflow VRAM, offloading part of the weights to RAM and slowing inference significantly.

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

## Step 4. Install Python 3.12

This is for Ubuntu 26.04 only. Ubuntu 24.04 ships with Python 3.12 by default, so this step can be skipped. If you are on a newer Ubuntu that ships a later Python version incompatible with Aider, install 3.12 via the deadsnakes PPA:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv
```

Verify the version:

```bash
python3.12 --version
# Expected output: Python 3.12.x
```

## Step 5. Create a Virtual Environment and Install Aider

```bash
python3.12 -m venv ~/.aider
source ~/.aider/bin/activate

pip install "aider-chat>=0.80" --no-cache-dir
```

Verify the installation:

```bash
aider --version
# Expected output: Aider v0.86.x or above
```

## Step 6. Set Environment Variables

```bash
echo 'export OLLAMA_API_BASE=http://localhost:11434' >> ~/.bashrc
source ~/.bashrc
```

## Step 7. Launch Aider

Activate the virtual environment before each session:

```bash
source ~/.aider/bin/activate
```

Navigate to your project directory and launch Aider with `--no-git` mode (recommended for workspaces with submodules):

```bash
cd ~/kitti_ws
aider --model ollama/qwen3.6-8k --no-git
```

## Step 8. Basic Aider Commands

| Command | Description |
|---|---|
| `/add <file_path>` | Add a file for the model to read and edit |
| `/ls` | List currently loaded files |
| `/drop <file_path>` | Remove a file |
| `/exit` | Quit Aider |

Example:

```
/add src/perception/fcn_segmentation/src/fcn_segmentation.cpp
/add src/perception/scnn_trt_backend/src/scnn_trt_backend.cpp
```

## Step 9. Monitor GPU Usage

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

- **Do not let Aider push directly to GitHub.** Recommended workflow: Aider edits → review changes yourself → manually run `git commit` and `git push`.
- In `--no-git` mode, Aider will not auto-commit — all git operations must be done manually.
- If `ollama ps` shows a high CPU/GPU offload ratio (e.g. `32%/68%`), reduce the context window size or stop other running models.
- Two models loaded simultaneously will compete for VRAM — make sure only one model is running at a time.
