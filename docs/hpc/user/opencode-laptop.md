---
title: OpenCode on Your Laptop
sidebar_label: OpenCode on Your Laptop
sidebar_position: 6
---

# User Guide: Running OpenCode from Your Own Laptop

This guide covers connecting to the shared Ollama server from your personal laptop instead of node02. For an overview of available models and basic OpenCode usage, see [OpenCode on HPC](./opencode-hpc.md).

:::note
This requires VPN access to the internal cluster network (`192.168.220.0/24`) so your laptop can reach `node01:11434`.
:::

## 1. Install OpenCode

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

## 2. Configure the Ollama Provider

OpenCode connects to Ollama through its OpenAI-compatible API endpoint. Create the config directory and file:

```bash
mkdir -p ~/.config/opencode
```

Create `~/.config/opencode/opencode.json` with the following content:

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
OpenCode uses `/v1` (the OpenAI-compatible endpoint), not Ollama's native `/api` endpoint. The difference matters — omitting `/v1` will cause connection failures. Verify the endpoint is reachable before launching OpenCode:

```bash
curl http://node01:11434/v1/models
```

This should return a JSON list of available models.
:::

### Add a Placeholder Auth Entry

OpenCode expects an auth entry for every configured provider even when no API key is required. Create `~/.local/share/opencode/auth.json`:

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

### Persist the Hostname (Optional)

If your laptop does not resolve `node01` by name, add the cluster node to `/etc/hosts`:

```bash
sudo tee -a /etc/hosts << 'EOF'
192.168.220.75  node01
EOF
```

## 3. Launch OpenCode

Navigate to your project directory and launch OpenCode:

```bash
cd ~/your_project
opencode
```

OpenCode opens a full TUI in your terminal. On first launch it may download a small provider package (`@ai-sdk/openai-compatible`) — this is a one-time step.

To select the model, type `/models` inside the TUI and choose from the Ollama (HPC) provider entries.

:::note
Unlike Aider, OpenCode does not take a `--model` flag on the command line. Model selection is done inside the TUI with `/models`, or set as a default in `opencode.json` by adding `"model": "ollama/qwen3-coder-next:latest"` at the top level.
:::

## 4. Basic TUI Commands

| Command | Description |
|---|---|
| `/models` | Switch the active model |
| `/init` | Analyse the project and generate an `AGENTS.md` context file |
| `/undo` | Revert the last AI-made file change (requires a Git repo) |
| `/redo` | Reapply a reverted change |
| `/share` | Generate a shareable link for the current session |
| `/exit` | Quit OpenCode |

**Mode toggle:** press `Tab` to switch between **Plan mode** (read-only, proposes changes without applying them) and **Build mode** (applies changes directly). Plan mode is useful for reviewing what OpenCode intends to do before committing to any edits.

**Shell passthrough:** prefix a line with `!` to run it as a shell command in the current directory, e.g. `!make -j4`.

**Session info panel:** the right-hand panel shows the session name, token context usage, LSP status, and current working directory. Toggle it with `Ctrl+X` then `I`, or search for it via the command palette (`Ctrl+P`).

## 5. Recommended Workflow for Exploring a Codebase

Start in **Plan mode** (the default) so OpenCode cannot modify any files during exploration.

**Step 1 — Generate project context:**

```
/init
```

This scans the workspace and writes an `AGENTS.md` file that persists context about your project structure across sessions.

**Step 2 — Add the folder you want to explore:**

```
/add filename
```

Or reference it inline directly in your prompt using `@`:

```
@filename explain the overall architecture
```

**Step 3 — Review, then switch to Build mode only when ready:**

Press `Tab` to switch to Build mode when you want OpenCode to apply changes. Switch back to Plan at any time to review without risk.

See [Basic TUI Commands](./opencode-hpc.md#3-basic-tui-commands) and [Git Workflow](./opencode-hpc.md#4-git-workflow) for usage details.

---

## Tips

- **Clipboard / text selection**: OpenCode automatically copies any selected text to the clipboard, which can conflict with your terminal emulator's own selection behavior. To disable this and restore normal terminal copy behavior, add the following to your shell profile:

  ```bash
  echo 'export OPENCODE_EXPERIMENTAL_DISABLE_COPY_ON_SELECT=true' >> ~/.bashrc
  source ~/.bashrc
  ```

  Alternatively, hold `Shift` while selecting text to bypass the TUI's mouse handling and use the terminal's native selection without any config change.

- **Model selection**: `qwen3-coder-next:latest` (79.7B) is the most capable model but slower. Switch to `qwen3.6:27b` (27.8B) for faster responses on lighter tasks using `/models` inside the TUI.

- **Do not let OpenCode push directly to GitHub.** Recommended workflow: OpenCode edits → review changes yourself → manually run `git commit` and `git push`.

---

## Troubleshooting

- **OpenCode can't find the `opencode` binary**: check `echo $PATH` and ensure `~/.opencode/bin` is included. Re-source your shell profile or open a new terminal.
- **`/models` returns `Not Found: 404`**: this means OpenCode launched before the provider config was fully applied, or the `baseURL` is incorrect. Exit and restart OpenCode — config changes in `opencode.json` are not picked up live. Verify the endpoint with `curl http://node01:11434/v1/models` before relaunching.
- **Provider not showing up in `/models`**: restart OpenCode after editing `opencode.json` — config changes are not picked up live.
- **Connection refused or timeout**: confirm you are connected to the VPN, then run `curl http://node01:11434/api/tags` to verify connectivity directly.
- **Tool calls not working (file edits, bash)**: this is most often a context window issue on the model side. Contact admin if the problem persists — the Ollama server configuration controls `num_ctx`.
- **TUI rendering looks broken**: OpenCode requires a terminal emulator with true color and Unicode support. Recommended options: Alacritty, WezTerm, iTerm2 (macOS), or Windows Terminal (WSL).
