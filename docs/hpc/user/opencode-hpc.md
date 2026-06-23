---
title: OpenCode on HPC
sidebar_label: OpenCode on HPC
sidebar_position: 6
---

# User Guide: Using OpenCode on HPC

OpenCode runs in your terminal and talks to a shared Ollama server on **node01** — no cloud API, no API keys. For installation, provider configuration, TUI commands, and general usage, see [OpenCode on Your Laptop](./opencode-laptop.md) — the setup is identical whether you are on node02 or your own machine.

This page covers the one HPC-specific requirement: **all OpenCode sessions must run inside a SLURM allocation.**

:::warning
Processes left running outside a SLURM allocation for more than 5 minutes will be automatically terminated.
:::

---

## 1. Request a SLURM Allocation

Before launching OpenCode, request an interactive allocation:

```bash
srun --time=04:00:00 --cpus-per-task=2 --mem=4G --pty bash
```

- No GPU request needed — inference runs on node01's GPUs over the network
- `--cpus-per-task=2` and `--mem=4G` are sufficient for OpenCode alone
- Adjust `--time` as needed; 4 hours is a reasonable working window

If you plan to compile code or run tests in the same session, request more:

```bash
srun --time=04:00:00 --cpus-per-task=8 --mem=16G --pty bash
```

Your shell prompt will show `slurm-<jobid>` while inside the allocation.

## 2. Launch OpenCode

```bash
cd ~/your_project
opencode
```

## 3. End the Session

Exit OpenCode first, then release the allocation:

```
/exit
```

```bash
exit
```

---

## Troubleshooting

- **OpenCode got killed or you see a termination warning**: you ran OpenCode outside a SLURM allocation — restart from Step 1.
- For all other issues (provider config, `/models` 404, connectivity), see the [Troubleshooting section](./opencode-laptop.md#troubleshooting) in the laptop guide.
