---
sidebar_position: 2
title: Local LLM Coding Assistant with Aider
description: Set up Aider with Ollama and Qwen 3.6 27B for local C++/CUDA/ROS 2 development on Ubuntu
---

# Local LLM Coding Assistant with Aider

This guide covers setting up [Aider](https://aider.chat) as a local LLM coding assistant — no cloud API required.

:::note
This guide assumes Ollama is already installed and a model is available. See [Install and Configure Ollama](./ollama.md) for setup instructions.
:::

## Environment

- **Machine**: Lenovo Legion (Ubuntu 24.04)
- **GPU**: NVIDIA RTX 5090 (24 GB VRAM)
- **Model**: Qwen 3.6 27B
- **Use case**: Local C++/CUDA/ROS 2 development assistance

## Step 1. Install Python 3.12

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

## Step 2. Create a Virtual Environment and Install Aider

```bash
python3.12 -m venv ~/.aider
source ~/.aider/bin/activate

pip install aider-chat
```

Verify the installation:

```bash
aider --version
# Expected output: Aider v0.86.x or above
```

## Step 3. Launch Aider

Activate the virtual environment before each session:

```bash
source ~/.aider/bin/activate
```

Navigate to your project directory and launch Aider with `--no-git` mode (recommended for workspaces with submodules):

```bash
cd ~/kitti_ws
aider --model ollama/qwen3.6-8k --no-git
```

## Step 4. Basic Aider Commands

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

## Tips

- **Do not let Aider push directly to GitHub.** Recommended workflow: Aider edits → review changes yourself → manually run `git commit` and `git push`.
- In `--no-git` mode, Aider will not auto-commit — all git operations must be done manually.
