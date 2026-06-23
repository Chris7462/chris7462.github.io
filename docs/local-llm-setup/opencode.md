---
sidebar_position: 2
title: Local LLM Coding Assistant with OpenCode
description: Set up OpenCode with Ollama for local C++/CUDA/ROS 2 development on Ubuntu
---

# Local LLM Coding Assistant with OpenCode

This guide covers setting up [OpenCode](https://opencode.ai) as a local LLM coding assistant — no cloud API required.

:::note
This guide assumes Ollama is already installed and a model is available. See [Install and Configure Ollama](./ollama.md) for setup instructions.
:::

## Environment

- **Machine**: Lenovo Legion (Ubuntu 24.04)
- **GPU**: NVIDIA RTX 5090 (24 GB VRAM)
- **Model**: Qwen 3.6 27B
- **Use case**: Local C++/CUDA/ROS 2 development assistance

## Step 1. Install OpenCode

OpenCode is distributed as a self-contained binary — no Python virtual environment required. Use the official install script:

```bash
curl -fsSL https://opencode.ai/install | bash
```

Verify the installation:

```bash
opencode --version
```

:::note
The install script places the binary under `~/.opencode/bin/`. If `opencode` is not found after installation, add it to your PATH:

```bash
echo 'export PATH="$HOME/.opencode/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```
:::

## Step 2. Configure the Ollama Provider

OpenCode connects to Ollama through its OpenAI-compatible API endpoint. Create the config directory and file:

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
      "name": "Ollama (Local)",
      "options": {
        "baseURL": "http://localhost:11434/v1"
      },
      "models": {
        "qwen3.6:27b": {},
        "qwen3.6-8k:latest": {},
        "qwen3-coder-next:latest": {}
      }
    }
  }
}
```

:::note
OpenCode uses `/v1` (the OpenAI-compatible endpoint), not Ollama's native `/api` endpoint. Omitting `/v1` will cause connection failures. Verify the endpoint is reachable before launching OpenCode:

```bash
curl http://localhost:11434/v1/models
```

This should return a JSON list of available models.
:::

### Add a Placeholder Auth Entry

OpenCode requires an auth entry for every configured provider even when no API key is needed. Create `~/.local/share/opencode/auth.json`:

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

## Step 3. Launch OpenCode

Navigate to your project directory and launch OpenCode:

```bash
cd ~/your_project
opencode
```

OpenCode opens a full TUI in your terminal. On first launch it may download a small provider package (`@ai-sdk/openai-compatible`) — this is a one-time step.

To select the model, type `/models` inside the TUI and choose from the Ollama (Local) provider entries.

## Step 4. Basic TUI Commands

| Command | Description |
|---|---|
| `/models` | Switch the active model |
| `/init` | Analyse the project and generate an `AGENTS.md` context file |
| `/undo` | Revert the last AI-made file change (requires a Git repo) |
| `/redo` | Reapply a reverted change |
| `/share` | Generate a shareable link for the current session |
| `/exit` | Quit OpenCode |

**Adding files to context:** use `@filename` inline in your prompt to reference a file or folder, e.g.:

```
@src/perception/fcn_segmentation/src/fcn_segmentation.cpp explain this file
```

**Mode toggle:** press `Tab` to switch between **Plan mode** (read-only, proposes changes without applying them) and **Build mode** (applies changes directly). OpenCode starts in Plan mode by default.

**Shell passthrough:** prefix a line with `!` to run a shell command directly, e.g. `!make -j4`.

**Session info panel:** the right-hand panel shows token context usage, LSP status, and current working directory. Toggle it with `Ctrl+X` then `I`.

## Tips

- **Do not let OpenCode push directly to GitHub.** Recommended workflow: OpenCode edits → review changes yourself → manually run `git commit` and `git push`.
- Use `/undo` to revert the last AI-made change if needed (requires the project to be a Git repo).
- **Clipboard / text selection**: OpenCode automatically copies selected text to the clipboard, which can conflict with your terminal emulator. To restore normal terminal copy behavior:

  ```bash
  echo 'export OPENCODE_EXPERIMENTAL_DISABLE_COPY_ON_SELECT=true' >> ~/.bashrc
  source ~/.bashrc
  ```

  Alternatively, hold `Shift` while selecting to use the terminal's native selection without any config change.
