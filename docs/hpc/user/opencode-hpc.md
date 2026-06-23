---
title: OpenCode on HPC
sidebar_label: OpenCode on HPC
sidebar_position: 5
---

# User Guide: Using the Local LLM Coding Assistant (OpenCode + Ollama) on HPC

## Overview

A local LLM coding assistant is available for development on the cluster. OpenCode runs in your terminal and talks to a shared Ollama server running on **node01** — no cloud API, no API keys.

### Available Models

| Model | Size | Notes |
|---|---|---|
| `qwen3-coder-next:latest` | 51 GB | Best for complex coding tasks |
| `qwen3.6:27b` | 17 GB | Smaller/faster alternative |
| `deepseek-r1:70b` | 43 GB | Strong reasoning model |

Model selection is done inside the TUI with `/models`. To set a default, add `"model": "ollama/qwen3-coder-next:latest"` at the top level of `~/.config/opencode/opencode.json`.

:::tip
Want to run OpenCode from your own laptop instead? See [OpenCode on Your Laptop](./opencode-laptop.md).
:::

---

## Initial Setup (One-Time)

Before running OpenCode for the first time on node02, configure the Ollama provider.

### Install OpenCode

```bash
curl -fsSL https://opencode.ai/install | bash
```

Add the binary to your PATH if needed:

```bash
echo 'export PATH="$HOME/.opencode/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
opencode --version
```

### Configure the Ollama Provider

Create the config directory and file:

```bash
mkdir -p ~/.config/opencode
```

Create `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (HPC)",
      "options": {
        "baseURL": "http://node01:11434/v1"
      },
      "models": {
        "qwen3-coder-next:latest": {},
        "qwen3.6:27b": {},
        "deepseek-r1:70b": {}
      }
    }
  }
}
```

:::note
OpenCode uses `/v1` (the OpenAI-compatible endpoint), not Ollama's native `/api` endpoint. Omitting `/v1` will cause connection failures.
:::

Add a placeholder auth entry — OpenCode requires this even when no API key is needed:

```bash
mkdir -p ~/.local/share/opencode
cat > ~/.local/share/opencode/auth.json << 'EOF'
{
  "ollama": {
    "type": "api",
    "key": "ollama"
  }
}
EOF
```

Verify connectivity:

```bash
curl http://node01:11434/v1/models
```

---

## Running an OpenCode Session

:::warning
All interactive sessions must be run inside a SLURM allocation. Processes left running outside SLURM for more than 5 minutes will be automatically terminated.
:::

### 1. Request a SLURM allocation

```bash
srun --time=04:00:00 --cpus-per-task=2 --mem=4G --pty bash
```

- No GPU request needed — inference runs on node01's GPUs over the network
- `--cpus-per-task=2` — OpenCode itself is lightweight (mostly waiting on network responses from the Ollama server), so 2 CPUs is generally sufficient unless you're also running other tools (linters, builds, tests) in the same session
- `--mem=4G` — covers OpenCode plus typical shell overhead; increase if your project's build or test commands need more memory
- `--time=04:00:00` gives a 4-hour working window (adjust as needed)

If you plan to also compile code or run tests in the same allocation, request more resources:

```bash
srun --time=04:00:00 --cpus-per-task=8 --mem=16G --pty bash
```

### 2. Launch OpenCode

```bash
cd ~/your_project
opencode
```

OpenCode opens a full TUI in your terminal. On first launch it may download a small provider package (`@ai-sdk/openai-compatible`) — this is a one-time step.

### 3. Basic TUI Commands

| Command | Description |
|---|---|
| `/models` | Switch the active model |
| `/add <file_path>` | Add a file or folder into the context (also `@filename` inline) |
| `/init` | Analyse the project and generate an `AGENTS.md` context file |
| `/undo` | Revert the last AI-made file change (requires a Git repo) |
| `/redo` | Reapply a reverted change |
| `/share` | Generate a shareable link for the current session |
| `/exit` | Quit OpenCode |

**Mode toggle:** press `Tab` to switch between **Plan mode** (read-only, proposes changes without applying them) and **Build mode** (applies changes directly). OpenCode starts in Plan mode by default.

**Shell passthrough:** prefix a line with `!` to run a shell command directly, e.g. `!make -j4`.

**Session info panel:** the right-hand panel shows token context usage, LSP status, and current working directory. Toggle it with `Ctrl+X` then `I`.

### 4. Git Workflow

- Review all AI-made changes yourself before committing
- Manually run `git commit` and `git push` — do not let OpenCode push directly to GitHub
- Use `/undo` to revert the last AI-made change if needed (requires the project to be a Git repo)

### 5. Ending Your SLURM Session

First, exit OpenCode:

```
/exit
```

Then exit the SLURM allocation:

```bash
exit
```

This releases your SLURM allocation. Your shell prompt shows `slurm-<jobid>` while inside the allocation, confirming you're properly in a SLURM session. After exiting, your prompt changes back to `node01` or `node02`.

---

## Tips

- **Clipboard / text selection**: OpenCode automatically copies selected text to the clipboard, which can conflict with your terminal emulator. To restore normal terminal copy behavior:

  ```bash
  echo 'export OPENCODE_EXPERIMENTAL_DISABLE_COPY_ON_SELECT=true' >> ~/.bashrc
  source ~/.bashrc
  ```

- **Model selection**: `qwen3-coder-next:latest` (79.7B) is the most capable but slowest. Switch to `qwen3.6:27b` (27.8B) for faster responses on lighter tasks via `/models`.

- **Exploring a codebase**: run `/init` first to generate an `AGENTS.md` context file, then use `/add <folder>` or `@foldername` inline to load specific packages into context. Stay in Plan mode while exploring so OpenCode cannot modify any files.

---

## Troubleshooting

- **If OpenCode gets killed or you see a termination warning**: you forgot to run `srun` first — exit and restart your session per Step 1
- **`/models` returns `Not Found: 404`**: the provider config was not applied. Exit OpenCode, verify `~/.config/opencode/opencode.json` is correct, confirm connectivity with `curl http://node01:11434/v1/models`, then relaunch
- **Provider not showing in `/models` after editing config**: config changes are not picked up live — restart OpenCode
- **If OpenCode can't connect to the model**: confirm the Ollama server is up with `curl http://node01:11434/api/tags` — if it fails, contact admin
