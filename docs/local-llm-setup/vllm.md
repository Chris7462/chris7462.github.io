---
sidebar_position: 3
title: Serve Qwen3.8-27B with vLLM on a Remote GPU
description: Install vLLM on a remote H100 machine, serve Qwen3.8-27B-FP8 with an OpenAI-compatible API, and connect from OpenCode over SSH tunnel
---

# Serve Qwen3.8-27B with vLLM on a Remote GPU

This guide covers installing vLLM on a remote GPU machine, serving [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (a dense, hybrid-attention, vision-language model released August 2026) through an OpenAI-compatible API, and connecting to it from [OpenCode](https://opencode.ai) on a local machine over an SSH tunnel.

:::note
This guide assumes OpenCode is already installed. See [Local LLM Coding Assistant with OpenCode](./opencode.md) for setup instructions. Unlike that guide (Ollama on a local RTX 5090), this one covers **vLLM on a remote, shared GPU machine**, which needs an extra networking step to reach from your local machine.
:::

## Environment

- **Remote machine**: `soc006`, NVIDIA H100 PCIe (80 GB VRAM), shared with other GPU workloads
- **Local machine**: `lambda-11037`
- **Model**: Qwen3.8-27B-FP8 (~28 GB checkpoint)
- **Use case**: Remote inference server for OpenCode, accessed over SSH tunnel

:::note
Qwen3.8-27B is built on the Qwen3.5 hybrid-attention architecture (`Qwen3_5ForConditionalGeneration`), which includes Mamba/gated-linear-attention (GDN) layers alongside regular attention. This architecture is newer than most stable vLLM releases, so this guide installs vLLM from the nightly wheel index rather than PyPI.
:::

## Step 1. Install vLLM

Use an isolated virtual environment to avoid conflicting with other Python tooling on the shared machine.

```bash
python3 -m venv ~/.vllm
source ~/.vllm/bin/activate
```

Qwen3.8's architecture isn't in stable vLLM releases yet, so install from the nightly index:

```bash
pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly
pip install -U "transformers>=5.4.0"
```

:::note
If `pip install` fails with `no such option: --torch-backend`, that flag is `uv pip`-only. Either drop it (as above) or install [uv](https://docs.astral.sh/uv/) first and use `uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly` — `uv` resolves the matching CUDA/torch build automatically and is worth switching to if you hit torch/CUDA mismatch errors.
:::

### Install Python headers

vLLM JIT-compiles Triton kernels for the model's GDN layers at first load. This requires the Python development headers, which are usually not installed by default:

```bash
sudo apt update
sudo apt install -y python3-dev build-essential
```

Without this step, the server fails at startup with `fatal error: Python.h: No such file or directory`.

## Step 2. Serve the Model

```bash
vllm serve Qwen/Qwen3.8-27B-FP8 \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key sk-local-something \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --max-num-seqs 256 \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

Flag notes:

| Flag | Why |
|---|---|
| `--host 127.0.0.1` | Only listen on localhost; combine with an SSH tunnel (Step 4) rather than exposing the port publicly |
| `Qwen/Qwen3.8-27B-FP8` | The FP8 checkpoint (~28 GB) leaves far more headroom for KV cache than the BF16 original (~56 GB), especially useful on a shared GPU |
| `--max-num-seqs 256` | Qwen3.8's Mamba/GDN layers each require a dedicated cache block per concurrent sequence. The default (`1024`) can exceed the blocks available after `--kv-cache-dtype fp8` and `--max-model-len` are accounted for, causing `max_num_seqs (1024) exceeds available Mamba cache blocks (...)` at startup. Lower this (or raise `--gpu-memory-utilization`) if you hit that error |
| `--kv-cache-dtype fp8` | Reduces KV cache memory footprint further; can affect accuracy slightly since the checkpoint may lack calibrated scaling factors (watch for `Using uncalibrated q_scale` warnings) |
| `--enable-auto-tool-choice --tool-call-parser qwen3_coder` | Required for OpenCode's tool calls (file edits, shell commands) to be parsed correctly. Omitting this breaks agentic tool use |
| `--reasoning-parser qwen3` | Qwen3.8 is a hybrid thinking model; this separates the reasoning trace from the final answer in the API response |

First launch downloads the checkpoint from Hugging Face (~29 GB) and takes several minutes for weight download, `torch.compile`, and CUDA graph capture. Subsequent restarts are faster due to compile caching.

### Verify the server is up

```bash
curl http://127.0.0.1:8000/v1/models -H "Authorization: Bearer sk-local-something"
```

This should return a JSON list containing `Qwen/Qwen3.8-27B-FP8`.

## Step 3. Keep the Server Running

Running `vllm serve` directly in an SSH session means the model process dies the moment that session disconnects. Use `screen` so it survives:

```bash
screen -S vllm
# run the vllm serve command inside this session
```

Detach with `Ctrl+A` then `D`. Reattach anytime with:

```bash
screen -r vllm
```

For a more permanent setup, wrap the same command in a systemd user service instead.

## Step 4. Connect from the Local Machine

### Open an SSH tunnel

On the **local** machine:

```bash
ssh -L 8000:localhost:8000 <user>@soc006
```

Keep this session open — closing it drops the tunnel (though the remote vLLM server keeps running if it's in `screen`). For a background tunnel that doesn't tie up a terminal:

```bash
ssh -fN -L 8000:localhost:8000 <user>@soc006
```

Verify the tunnel from the local machine:

```bash
curl http://127.0.0.1:8000/v1/models -H "Authorization: Bearer sk-local-something"
```

### Add the provider to OpenCode

Add a new entry under the existing `provider` object in `~/.config/opencode/opencode.jsonc` — it can sit alongside other providers (e.g. an Ollama server on a separate machine):

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "qwen-remote": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Qwen3.8-27B (H100)",
      "options": {
        "baseURL": "http://localhost:8000/v1",
        "apiKey": "sk-local-something"
      },
      "models": {
        "Qwen/Qwen3.8-27B-FP8": {
          "attachment": true,
          "modalities": {
            "input": ["text", "image"],
            "output": ["text"]
          }
        }
      }
    }
  }
}
```

The `attachment` and `modalities` fields declare vision support — without them, OpenCode sends image attachments as plain filenames instead of multimodal input, and the model only sees the file name.

Restart OpenCode, then run `/models` and select `qwen-remote` / `Qwen/Qwen3.8-27B-FP8`.

## Sending Images in OpenCode

Qwen3.8-27B is natively multimodal. In the OpenCode TUI:

- **Reference by path**: type `@` followed by the image filename (e.g. `@screenshot.jpg what's in this image?`) — this triggers fuzzy file search and attaches the image directly
- **Drag and drop**: drag an image file into the terminal window, if your terminal emulator supports it
- **Paste**: `Ctrl+V` with an image on the clipboard — unreliable over a plain SSH session since the terminal often can't reach the local clipboard; prefer `@` in that case

Image paths are resolved relative to the machine OpenCode is running on (the local machine, in this setup) — not the remote GPU machine.

:::note
Don't ask the model to "read" an image file path as a tool call — OpenCode's `Read` tool has a known issue passing image bytes through to vision models. Attaching via `@`, drag-and-drop, or paste bypasses that tool and sends the image directly as an API attachment, which works correctly.
:::

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Model architectures ['Qwen3_5ForConditionalGeneration'] are not supported for now` | Installed vLLM release doesn't include Qwen3.5/3.8 architecture support yet | Install vLLM from the nightly wheel index (Step 1) |
| `fatal error: Python.h: No such file or directory` | Missing Python dev headers, needed for Triton JIT compilation of GDN kernels | `sudo apt install python3-dev build-essential` |
| `max_num_seqs (1024) exceeds available Mamba cache blocks (...)` | Default concurrency setting needs more Mamba cache blocks than available VRAM allows | Add `--max-num-seqs 256` (or lower), or raise `--gpu-memory-utilization` |
| `-bash: /usr/bin/curl: Argument list too long` | Base64-encoded image passed as a command-line argument exceeds the shell's `ARG_MAX` | Write the JSON payload to a file and use `curl -d @payload.json` instead of `-d "..."` |
| Response is truncated / `content` is `null` with `finish_reason: "length"` | Qwen3.8 defaults to extra-high thinking effort, which can consume the entire `max_tokens` budget on reasoning before producing an answer | Raise `max_tokens`, or pass `chat_template_kwargs: {"enable_thinking": false}` (or `{"reasoning_effort": "low"}`) in the request |
| OpenCode says image attachment isn't supported / only sees the filename | Model entry in `opencode.jsonc` is missing vision capability declarations | Add `"attachment": true` and `"modalities": {"input": ["text", "image"], "output": ["text"]}` under the model config |

## Tips

- Prefer the FP8 checkpoint over BF16 on shared/limited-VRAM GPUs — it roughly halves the weight footprint with minimal accuracy loss, leaving more room for KV cache.
- Keep `vllm serve` in `screen` (or systemd) on the remote machine so an SSH disconnect doesn't kill inference.
- The SSH tunnel and the vLLM server are independent — closing the tunnel only breaks the local connection, not the remote server; reopen the tunnel to reconnect without restarting the model.
- Multiple OpenCode providers can coexist in the same `opencode.jsonc` — no need to remove an existing Ollama or other provider entry when adding a new one.
