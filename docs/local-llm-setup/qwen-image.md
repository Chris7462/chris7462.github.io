---
sidebar_position: 4
title: Serve Qwen-Image-2.1 with vLLM-Omni and Use It from OpenCode
description: Serve Qwen-Image-2.1 on a remote Blackwell GPU with vLLM-Omni, expose it to OpenCode through a small MCP server, and give the chat model vision so it can check its own output
---

# Serve Qwen-Image-2.1 with vLLM-Omni and Use It from OpenCode

This guide covers serving [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (Qwen's text-to-image and image-editing model, released September 2026) on a remote GPU with [vLLM-Omni](https://github.com/vllm-project/vllm-omni), then wiring it into [OpenCode](https://opencode.ai) through a small MCP server so the coding agent can generate and edit images on request.

:::note
This guide assumes OpenCode is already installed and connected to a chat model. See [Local LLM Coding Assistant with OpenCode](./opencode.md) and [Serve Qwen3.8-27B with vLLM on a Remote GPU](./vllm.md). Unlike Qwen3.8, which accepts images as **input** but still responds with text, Qwen-Image **outputs** images and cannot drive OpenCode's agent loop, which expects text and tool calls. So instead of adding it as a provider, the image API is exposed as an MCP tool that the chat model calls.
:::

## Environment

- **Remote machine**: `soc006`, NVIDIA H100 PCIe (80 GB VRAM), shared with other GPU workloads
- **Local machine**: `lambda-11037`
- **Image model**: Qwen-Image-2.1 (7B DiT, BF16, ~33 GB of weights including text encoder and VAE)
- **Chat model**: Qwen3.8-27B served by Ollama on `soc006` (`qwen3.8:27b`, `qwen3.8:27b-bf16`, and a locally imported `qwen3.8-uncensored:27b-q8kp`)
- **Use case**: Let OpenCode generate/edit images and inspect the results

The overall architecture:

```
OpenCode (lambda-11037)
  └─ chat model: Ollama on soc006 (needs tools; needs vision to inspect images)
       └─ tool call → qwen-image MCP server (local Python process on lambda-11037)
                         └─ HTTP → vLLM-Omni on soc006:8091 → Qwen-Image-2.1 (GPU 1)
```

:::warning
Qwen-Image-2.1 is released under the **Qwen Research License**, which permits non-commercial use only. Check the license before using generated images in a product or deliverable.
:::

## Step 1. Install vLLM-Omni

As of this writing, Qwen-Image-2.1 support lives in vLLM-Omni **PR #7759**, which has not been merged yet, and that branch targets **vLLM 0.29.0**.

### Prepare Python 3.12

vLLM recommends Python 3.12. Check the system version first:

```bash
python3 --version
```

If it isn't 3.12, install it from the deadsnakes PPA:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev
```

### Check out the PR branch

```bash
cd /scratch/thirdparty
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
git fetch origin pull/7759/head:qwen-image-2.1
git checkout qwen-image-2.1
```

### Create a virtual environment

```bash
python3.12 -m venv .vllm
source .vllm/bin/activate
pip install -U pip setuptools wheel
```

:::warning
Do **not** create the venv with `--system-site-packages`. On a shared machine, system packages (e.g. a `torchaudio` under `/usr/local/lib/python3.12/dist-packages`) can shadow the venv's own packages and cause CUDA version mismatches at import time. Verify with:

```bash
grep include-system .vllm/pyvenv.cfg   # should be false
```
:::

### Install vLLM and vLLM-Omni together

Install both in **one** command so pip resolves them together. Installing them separately can let `pip install -e .` silently upgrade vLLM to a newer, incompatible release:

```bash
pip install "vllm==0.29.0" -e .
```

### Verify versions and GPU support

```bash
pip show vllm | grep Version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_arch_list())"
```

For Blackwell GPUs (RTX 5090, RTX PRO 6000), check that:

- vLLM is `0.29.0`
- `get_arch_list()` includes `sm_120`
- the CUDA version is 12.8 or newer, and `is_available()` is `True`

This setup ended up with `torch 2.13.0+cu132`.

:::note
`uv pip install ... --torch-backend=auto` (as in the upstream instructions) picks the right CUDA build of PyTorch automatically. With plain `pip`, the check above is how you confirm you got a Blackwell-capable build.
:::

### Put the model cache in a different folder (optional)

By default, Hugging Face downloads models to `~/.cache/huggingface`. If that location works for you (a local disk with enough free space), skip this step.

Otherwise, point `HF_HOME` somewhere else. This is worth doing when:

- your home directory is on a network mount (NFS), which uses shared storage and slows down weight loading at startup
- your home partition doesn't have room for the ~33 GB checkpoint

Appending the variable to the venv's `activate` script applies it automatically whenever the environment is activated, without affecting other projects:

```bash
echo 'export HF_HOME=/scratch/local/hf_cache' >> /scratch/thirdparty/vllm-omni/.vllm/bin/activate
```

:::note
`HF_HOME` must be set in the same shell that runs `vllm serve`. Check with `echo $HF_HOME` before starting the server — if it's empty, the download silently goes to the default location.
:::

## Step 2. Serve the Model

### Start the server in `screen`

```bash
screen -S qwen-image
source /scratch/thirdparty/vllm-omni/.vllm/bin/activate
echo $HF_HOME        # make sure it is set before starting
vllm serve Qwen/Qwen-Image-2.1 --omni --port 8091
```

Detach with `Ctrl+A` then `D`; reattach with `screen -r qwen-image`.

Startup goes through: weight download → loading the text encoder, DiT, and VAE → CUDA graph capture → `Application startup complete.` To confirm the weights are landing in the right place, watch the cache from another terminal:

```bash
watch -n 10 du -sh /scratch/local/hf_cache/hub
```

### Verify the server

```bash
curl -s -X POST http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen-Image-2.1", "prompt": "A red pickup truck on a test track, sign reads \"ISUZU\"", "size": "1024x1024", "num_inference_steps": 40, "true_cfg_scale": 1.0, "seed": 42}' \
  -o resp.json

jq -r '.data[0].b64_json' resp.json | base64 -d > test.png
```

Saving the raw response first means an error shows up as readable JSON (`cat resp.json`) instead of a corrupt PNG. The first request is slower due to warm-up.

### Check reachability from the local machine

```bash
curl -s http://soc006:8091/v1/models
```

If that returns a JSON model list, no tunnel is needed. Otherwise, forward the port over SSH and use `http://localhost:8091` below:

```bash
ssh -fN -L 8091:localhost:8091 yi-chen@soc006
```

## Step 3. The Image API

vLLM-Omni exposes three OpenAI-compatible endpoints:

| Endpoint | Purpose | Request format | Image in response |
|---|---|---|---|
| `/v1/images/generations` | Text-to-image | JSON | `.data[0].b64_json` |
| `/v1/images/edits` | Image editing | **multipart/form-data** | `.data[0].b64_json` |
| `/v1/chat/completions` | Image editing via JSON | JSON | `.choices[0].message.content[0].image_url.url` |

Recommended parameters:

| Parameter | Value | Why |
|---|---|---|
| `num_inference_steps` | `40` | Send it on **every** request — the server default is 50 |
| `true_cfg_scale` | `1.0` | Default is 4.0, but CFG only applies with a `negative_prompt` |
| `negative_prompt` | omit | Enables CFG, which roughly doubles latency |
| `size` | `"WxH"` | e.g. `2048x2048`, `2752x1536` (16:9), `1536x2752` (9:16); rounded down to a multiple of 32 |
| `seed` | integer | Reproducible output |

### Text-to-image from Python

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://soc006:8091/v1", api_key="EMPTY")
resp = client.images.generate(
    model="Qwen/Qwen-Image-2.1",
    prompt='A neon shop sign that reads "QWEN IMAGE 2.1", rainy night',
    size="1024x1024",
    extra_body={"num_inference_steps": 40, "true_cfg_scale": 1.0, "seed": 42},
)
open("t2i.png", "wb").write(base64.b64decode(resp.data[0].b64_json))
```

### Image editing

```bash
curl -s -X POST http://soc006:8091/v1/images/edits \
  -F image=@input.png \
  -F 'prompt=Change the background to a sunset beach' \
  -F model=Qwen/Qwen-Image-2.1 \
  -F num_inference_steps=40 -F true_cfg_scale=1.0 \
  | jq -r '.data[0].b64_json' | base64 -d > edit.png
```

- This endpoint accepts **only** form data — sending JSON drops the connection.
- Up to 4 reference images per request; refer to them as `<image1>`, `<image2>`, … in send order.
- Upload images as files. Data URLs in form fields are capped at 1024 KB, which a single PNG can easily exceed.

For transparent backgrounds, start the prompt with `This is an RGBA image with transparency.` and end it with `The image has alpha channel and the background is transparent.`

## Step 4. Wrap the API as an MCP Server

The MCP server is a thin Python process that runs on the **local** machine and forwards tool calls to the image API. It needs no GPU.

```bash
python3 -m venv ~/.mcp
~/.mcp/bin/pip install -U pip requests 'mcp<2'
mkdir -p ~/tools
```

:::warning
Pin `mcp<2`. In mcp 2.x, `FastMCP` was renamed to `MCPServer` and other APIs changed, so the script below fails with `No module named 'mcp.server.fastmcp'`.
:::

Save the following as `~/tools/qwen_image_mcp.py`:

```python title="~/tools/qwen_image_mcp.py"
"""
qwen_image_mcp.py — MCP server that exposes Qwen-Image-2.1 (served by vLLM-Omni)
as tools for opencode or any other MCP client.

Env vars:
  QWEN_IMAGE_URL  vLLM-Omni base URL (default: http://localhost:8091)
  QWEN_IMAGE_OUT  output directory for PNGs (default: ./generated, relative to cwd)
"""
import base64
import os
import pathlib
import time

import requests
from mcp.server.fastmcp import FastMCP

BASE_URL = os.environ.get("QWEN_IMAGE_URL", "http://localhost:8091").rstrip("/")
OUT_DIR = pathlib.Path(os.environ.get("QWEN_IMAGE_OUT", "./generated")).resolve()
MODEL = "Qwen/Qwen-Image-2.1"
TIMEOUT = 600

mcp = FastMCP("qwen-image")


def _save(b64: str, prefix: str) -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{prefix}_{int(time.time() * 1000)}.png"
    path.write_bytes(base64.b64decode(b64))
    return str(path)


@mcp.tool()
def generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
) -> str:
    """Generate an image from a text prompt using Qwen-Image-2.1.

    Good at rendering text inside images (signs, posters, infographics).
    Supported sizes up to 2K, e.g. 2048x2048, 2752x1536 (16:9), 1536x2752 (9:16).
    For a transparent background, start the prompt with
    "This is an RGBA image with transparency." and end it with
    "The image has alpha channel and the background is transparent."

    Returns the absolute path of the saved PNG.
    """
    body = {
        "model": MODEL,
        "prompt": prompt,
        "size": f"{width}x{height}",
        "num_inference_steps": 40,
        "true_cfg_scale": 1.0,
    }
    if seed is not None:
        body["seed"] = seed

    r = requests.post(f"{BASE_URL}/v1/images/generations", json=body, timeout=TIMEOUT)
    r.raise_for_status()
    return _save(r.json()["data"][0]["b64_json"], "t2i")


@mcp.tool()
def edit_image(
    prompt: str,
    image_paths: list[str],
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
) -> str:
    """Edit or combine existing images with Qwen-Image-2.1.

    image_paths: 1-4 local image files. In the prompt, refer to them as
    <image1>, <image2>, ... in the order given.
    If width/height are omitted, output follows the last image's aspect ratio.

    Returns the absolute path of the saved PNG.
    """
    if not 1 <= len(image_paths) <= 4:
        raise ValueError("edit_image accepts 1 to 4 images per request")

    data = {
        "model": MODEL,
        "prompt": prompt,
        "num_inference_steps": "40",
        "true_cfg_scale": "1.0",
    }
    if width and height:
        data["size"] = f"{width}x{height}"
    if seed is not None:
        data["seed"] = str(seed)

    handles = [open(os.path.expanduser(p), "rb") for p in image_paths]
    try:
        files = [("image", (os.path.basename(h.name), h, "image/png")) for h in handles]
        # This endpoint only accepts multipart/form-data — never send JSON here.
        r = requests.post(f"{BASE_URL}/v1/images/edits", files=files, data=data, timeout=TIMEOUT)
    finally:
        for h in handles:
            h.close()

    r.raise_for_status()
    return _save(r.json()["data"][0]["b64_json"], "edit")


if __name__ == "__main__":
    mcp.run()  # stdio transport, which opencode's "local" MCP type expects
```

Test it the way OpenCode will launch it — with absolute paths and **without** activating the venv:

```bash
QWEN_IMAGE_URL=http://soc006:8091 /home/yi-chen/.mcp/bin/python /home/yi-chen/tools/qwen_image_mcp.py
```

If it sits there without an error, it is waiting on stdio as expected. Press `Ctrl+C` to exit.

## Step 5. Register the MCP Server in OpenCode

Add an `mcp` block to `~/.config/opencode/opencode.jsonc`. The full config used here, with the two existing Ollama providers for `soc006` (one via its LAN hostname, one via `soc006`):

```jsonc title="~/.config/opencode/opencode.jsonc"
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "soc006": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (HPC)",
      "options": {
        "baseURL": "http://soc006:11434/v1"
      },
      "models": {
        "qwen3.8:27b": {
          "attachment": true,
          "modalities": {
            "input": ["text", "image"],
            "output": ["text"]
          }
        },
        "qwen3.8:27b-bf16": {
          "attachment": true,
          "modalities": {
            "input": ["text", "image"],
            "output": ["text"]
          }
        },
        "qwen3.8-uncensored:27b-q8kp": {
          "attachment": true,
          "modalities": {
            "input": ["text", "image"],
            "output": ["text"]
          }
        },
        "qwen3-coder-next:latest": {
        }
      }
    }
  },
  "mcp": {
    "qwen-image": {
      "type": "local",
      "command": [
        "/home/yi-chen/.mcp/bin/python",
        "/home/yi-chen/tools/qwen_image_mcp.py"
      ],
      "environment": {
        "QWEN_IMAGE_URL": "http://soc006:8091"
      },
      "enabled": true
    }
  }
}
```

The `qwen3.8-uncensored:27b-q8kp` entries assume the vision-enabled import from [Step 6](#step-6-give-the-chat-model-vision-optional). If you import it without the projector, leave those entries empty (`{}`) like `qwen3-coder-next:latest`.

:::warning
`mcp` must be a **top-level** key, a sibling of `provider` — not nested inside it. If it ends up inside `provider`, OpenCode treats it as a model provider named `mcp`, and the tools never appear. A quick check (two-space indent means top level):

```bash
grep -n '^  "mcp"' ~/.config/opencode/opencode.jsonc
```
:::

Fully quit and restart OpenCode (MCP servers are only loaded at startup), then confirm the server is connected:

```bash
opencode mcp list
```

Try it with a chat model that supports tool calling:

```
Generate a 1024x1024 image of a rainy intersection at night with a street sign that reads "TEST ROUTE"
```

Images are saved under `generated/` in the directory where OpenCode was started.

## Step 6. Give the Chat Model Vision (Optional)

To let the agent inspect what it generated, the chat model needs **vision**, and OpenCode must know the model accepts images.

### Check model capabilities

```bash
ollama show qwen3.8:27b-bf16
```

`Capabilities` must include `tools` (to call the MCP server) and `vision` (to look at images).

### Keep the projector when importing a GGUF

Qwen3.8 is multimodal. Exporting an official model's Modelfile with `ollama show --modelfile qwen3.8:27b-bf16` shows **two** `FROM` lines:

| Size | Contents |
|---|---|
| Large (~50 GB) | Language model weights |
| Small (~1 GB) | Vision projector (mmproj) |

When importing a community GGUF, keep **both** lines: point the large one at the downloaded model and the small one at the repo's matching mmproj file. Dropping the second line gives you a text-only model.

This example imports the Q8_K_P quantization of [HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF](https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF). Its `K_P` quantization name is custom, so `ollama pull hf.co/...:Q8_K_P` rejects the tag; download the file and import it manually instead.

:::note
Use the mmproj shipped with the fine-tune rather than the official Qwen3.8 projector blob. The fine-tune modifies the language model weights, so its own projector is the one most likely to line up with them. The official blob is a fallback if Ollama rejects the repo's file.
:::

Find the mmproj file in the repo:

```bash
python3 -c "from huggingface_hub import list_repo_files; print('\n'.join(f for f in list_repo_files('HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF') if 'mmproj' in f.lower()))"
```

This returns `mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf`. Download the model and the projector:

```bash
mkdir -p /scratch/local/hf_cache/models/qwen3.8-uncensored
cd /scratch/local/hf_cache/models/qwen3.8-uncensored

hf download HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF \
  Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf \
  mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf \
  --local-dir .
```

Start from the official Modelfile so the chat template, renderer, parser, and sampling parameters carry over:

```bash
ollama show --modelfile qwen3.8:27b-bf16 > Modelfile
```

Replace only the two `FROM` lines:

```
FROM /scratch/local/hf_cache/models/qwen3.8-uncensored/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf
FROM /scratch/local/hf_cache/models/qwen3.8-uncensored/mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf
TEMPLATE {{ .Prompt }}
RENDERER qwen3.8
PARSER qwen3.5
PARAMETER presence_penalty 0
PARAMETER repeat_penalty 1
PARAMETER temperature 1
PARAMETER top_k 20
PARAMETER top_p 0.95
PARAMETER min_p 0
```

```bash
ollama create qwen3.8-uncensored:27b-q8kp -f Modelfile
ollama show qwen3.8-uncensored:27b-q8kp
```

Expected capabilities and projector:

```
  Capabilities
    completion
    vision
    tools
    thinking

  Projector
    architecture        clip
    parameters          460.73M
    embedding length    1152
    dimensions          5120
```

`Capabilities` should now list `vision`, and the `Projector` section's `dimensions` should match the model's `embedding length` (5120 for Qwen3.8-27B). Loading is not the same as working, so test with a real image:

```bash
ollama run qwen3.8-uncensored:27b-q8kp "Describe this image and read out all text in it /home/yi-chen/test.png"
```

In testing with the truck image from Step 2, the model correctly identified a red Isuzu pickup and read the "ISUZU" lettering, but it also reported text such as "SUV2" and a "SILVERADO" license plate that were likely invented. Scene-level understanding is reliable; treat fine-print checks with caution.

### Declare vision in OpenCode

Add `attachment` and `modalities` only to models that actually have vision. Add the entry under both the `soc006` and `soc006` providers:

```jsonc
"qwen3.8-uncensored:27b-q8kp": {
  "attachment": true,
  "modalities": {
    "input": ["text", "image"],
    "output": ["text"]
  }
}
```

For text-only models, leave the entry empty (`{}`). Declaring image input on a model without a projector produces a 400 error the moment OpenCode sends it an image.

:::note
As noted in the [vLLM guide](./vllm.md), OpenCode's `Read` tool has a known issue passing image bytes to vision models. If asking the agent to "look at the generated image" doesn't work, attach the file with `@generated/<file>.png` instead.
:::

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `vLLM and vLLM-Omni appear to have mismatched major/minor versions` | vLLM was upgraded past 0.29 during install | `pip install "vllm==0.29.0" -e .` |
| `PyTorch and TorchAudio were compiled with different CUDA versions`, with a path under `/usr/local/lib/.../dist-packages` | The venv sees system site-packages | Set `include-system-site-packages = false` in `pyvenv.cfg` and reinstall, or recreate the venv |
| Weights download to `~/.cache/huggingface` | `HF_HOME` not set in the shell that started the server | Stop, delete the partial download, set `HF_HOME`, restart |
| `No module named 'mcp.server.fastmcp'` | mcp 2.x installed | `pip install 'mcp<2'` |
| The agent says it has no `generate_image` tool | MCP server not loaded — often `mcp` nested inside `provider` | Move `mcp` to the top level, restart OpenCode, check `opencode mcp list` |
| `Multimodal data provided, but model does not support multimodal requests.` | Model declared as image-capable but has no vision projector | Add the projector (Step 6) or remove `modalities` from the model entry |
| Requests to `/v1/images/edits` drop the connection | Request was sent as JSON | Use multipart/form-data (`curl -F`) |
| The chat model never calls the tool, or tool calls are malformed | Ollama's context window is too short and the tool descriptions get truncated | Set `OLLAMA_CONTEXT_LENGTH=32768` (or higher) on the Ollama service |

## Tips

- The vLLM-Omni server has **no authentication** by default. Anyone who can reach port 8091 can use the GPU. Add `--api-key <key>` to `vllm serve` if the port is reachable beyond your own network, and send `Authorization: Bearer <key>` from the MCP script.
- The server holds its GPU for as long as it runs — coordinate on shared machines.
- Always send `num_inference_steps: 40` and `true_cfg_scale: 1.0`; the defaults are slower without improving quality.
- The first request after startup is a warm-up — exclude it from any benchmarks.
- For batch generation, add `--step-execution --max-num-seqs 8` to `vllm serve`.
