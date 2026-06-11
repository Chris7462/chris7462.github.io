---
title: Aider on Your Laptop
sidebar_label: Aider on Your Laptop
sidebar_position: 6
---

# User Guide: Running Aider from Your Own Laptop

This guide covers connecting to the shared Ollama server from your personal laptop instead of node02. For an overview of available models and basic Aider usage, see [Aider on HPC](./aider-hpc.md).

:::note
This requires VPN access to the internal cluster network (`192.168.220.0/24`) so your laptop can reach `node01:11434`.
:::

## 1. Install Aider

Aider is not pre-installed on your laptop. Install it in a dedicated virtual environment:

```bash
python3 -m venv ~/.aider
source ~/.aider/bin/activate
pip install aider-chat
```

:::note
Activate this virtual environment (`source ~/.aider/bin/activate`) each time you want to use Aider.
:::

## 2. Set `OLLAMA_API_BASE`

```bash
export OLLAMA_API_BASE=http://node01:11434
```

Add this to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) to persist it across sessions.

### System-wide Setup (Optional)

If you'd rather configure this for all users on your laptop:

1. Add node01's IP to `/etc/hosts`:

```bash
sudo tee -a /etc/hosts << 'EOF'
192.168.220.75  node01
EOF
```

2. Add the environment variable system-wide via `/etc/profile.d/ollama.sh`:

```bash
sudo tee /etc/profile.d/ollama.sh << 'EOF'
export OLLAMA_API_BASE=http://node01:11434
EOF
```

Re-login or run `source /etc/profile.d/ollama.sh` to apply immediately.

Verify connectivity:

```bash
curl http://node01:11434/api/tags
```

## 3. Launch Aider

```bash
cd ~/your_project
aider --model ollama/qwen3-coder-next:latest --no-git
```

See [Basic Aider Commands](./aider-hpc.md#3-basic-aider-commands) and [Git Workflow](./aider-hpc.md#4-git-workflow) for usage details.

---

## Troubleshooting

- **If Aider can't connect to the model**: check `echo $OLLAMA_API_BASE` (should show `http://node01:11434`), confirm you're connected to the VPN, then `curl http://node01:11434/api/tags` to test connectivity directly
