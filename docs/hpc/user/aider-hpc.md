---
title: Aider on HPC
sidebar_label: Aider on HPC
sidebar_position: 5
---

# User Guide: Using the Local LLM Coding Assistant (Aider + Ollama) on HPC

## Overview

A local LLM coding assistant is available for development on the cluster. Aider runs on your shell and talks to a shared Ollama server running on **node01** — no cloud API, no API keys.

### Available Models

| Model | Size | Notes |
|---|---|---|
| `qwen3-coder-next:latest` | 51 GB | Default — used in examples below |
| `qwen3.6:27b` | 17 GB | Smaller/faster alternative |

To use a different model, swap the `--model` argument in the `aider` command, e.g. `--model ollama/qwen3.6:27b`.

:::tip
Want to run Aider from your own laptop instead? See [Aider on Your Laptop](./aider-laptop.md).
:::

---

## Running an Aider Session

:::warning
All interactive sessions must be run inside a SLURM allocation. Processes left running outside SLURM for more than 5 minutes will be automatically terminated.
:::

### 1. Request a SLURM allocation

```bash
srun --time=04:00:00 --cpus-per-task=2 --mem=4G --pty bash
```

- No GPU request needed — inference runs on node01's GPU over the network
- `--cpus-per-task=2` — Aider itself is lightweight (mostly waiting on network responses from the Ollama server), so 2 CPUs is generally sufficient unless you're also running other tools (linters, test suites, builds) in the same session
- `--mem=4G` — covers Aider plus typical editor/shell overhead; increase this if your project's build or test commands need more memory
- `--time=04:00:00` gives a 4-hour working window (adjust as needed)

`--cpus-per-task=2 --mem=4G` is a reasonable default for most Aider sessions. If you plan to also compile code or run tests in the same allocation, request more (e.g. `--cpus-per-task=8 --mem=16G`) based on your project's normal requirements.

### 2. Launch Aider

```bash
cd ~/your_project
aider --model ollama/qwen3-coder-next:latest --no-git
```

### 3. Basic Aider Commands

| Command | Description |
|---|---|
| `/add <file_path>` | Add a file for the model to read and edit |
| `/ls` | List currently loaded files |
| `/drop <file_path>` | Remove a file |
| `/exit` | Quit Aider |

Example:

```
/add src/main.cpp
```

### 4. Git Workflow

- Aider runs in `--no-git` mode — it will not auto-commit
- Review all changes yourself, then manually run `git commit` and `git push`
- Do not let Aider push directly to GitHub

### 5. Ending Your SLURM Session

First, exit Aider:

```
/exit
```

Then exit the SLURM allocation:

```bash
exit
```

This releases your SLURM allocation. Your shell prompt shows `slurm-<jobid>` while inside the allocation, confirming you're properly in a SLURM session. Exit the SLURM session, and your prompt changes back to `node01`.

---

## Troubleshooting

- **If Aider gets killed or you see a warning about being terminated**: you forgot to run `srun` first — restart your session per Step 1
- **If Aider can't connect to the model**: contact admin — the Ollama server may be down
