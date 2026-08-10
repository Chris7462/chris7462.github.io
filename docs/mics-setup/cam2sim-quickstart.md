---
sidebar_position: 1
title: Cam2Sim Quick Start
description: Reproduce Cam2Sim's Step 5C (Gaussian Splatting trajectory replay) using the precomputed reference_bag dataset, no ROS bag extraction or Splatfacto training required
---

# Cam2Sim Quick Start

This guide covers reproducing **Step 5C** (Gaussian Splatting trajectory
replay) of [Cam2Sim](https://github.com/ast-fortiss-tum/cam2sim) using the
precomputed `reference_bag` dataset — no ROS bag extraction, no COLMAP, and
no Splatfacto training required.

## Prerequisites

- Linux machine (tested on Ubuntu 20.04; 22.04/24.04 likely fine)
- NVIDIA GPU, compute capability ≥ 7.5 (RTX 20-series or newer)
- NVIDIA driver supporting CUDA 12.x
- ~30+ GB free disk space (precomputed dataset + CARLA + Nerfstudio env)

## Step 1. Clone the Repo

```bash
git clone https://github.com/ast-fortiss-tum/cam2sim.git
cd cam2sim
```

Run every command below from this `cam2sim/` project root.

## Step 2. Make Sure Python 3.10 Is Available

This guide uses `venv` instead of Conda. You'll need Python 3.10 installed
system-wide (Nerfstudio and `data_extraction_requirements.txt` are built
around it).

```bash
python3.10 --version
```

If that's not found, install it via your distro's package manager:

```bash
sudo apt install python3.10 python3.10-venv
```

## Step 3. Install CARLA 0.9.16

Follow the [official quick-start guide](https://carla.readthedocs.io/en/0.9.16/start_quickstart/)
to download and extract CARLA 0.9.16.

Then open `3_generate_simulation_data/utils/config.py` and set:

```python
CARLA_INSTALLATION_PATH = "/absolute/path/to/CARLA_0.9.16"
```

## Step 4. Install Nerfstudio (venv)

```bash
python3.10 -m venv ~/.nerfstudio
source ~/.nerfstudio/bin/activate
pip install --upgrade pip
```

Nerfstudio's own docs are [Conda-first](https://docs.nerf.studio/quickstart/installation.html),
but the underlying steps are plain pip installs:

1. Install a CUDA-matched PyTorch build — check `nvidia-smi` for your
   driver's CUDA version first, then grab the matching command from
   [pytorch.org](https://pytorch.org).
2. Install `tinycudann` and `gsplat` — these compile from source against
   your system's CUDA toolkit. This is the actual reason the repo
   recommends Conda: `conda install cudatoolkit` gives an isolated,
   version-matched toolkit per environment. With `venv` you need a
   system-wide CUDA toolkit (check with `nvcc --version`) that matches
   your PyTorch build, or these two will fail to compile.
3. `pip install nerfstudio`

:::note
Keep this venv dedicated to Nerfstudio — the pipeline scripts expect a
consistently-named environment (originally `nerfstudio`); use that name for
the venv folder too so it's easy to track.
:::

Verify:

```bash
ns-train --help
```

:::note
Splatfacto (the GS model used here) needs a CUDA-capable GPU, compute
capability ≥ 7.5.
:::

Now add the extra packages Step 5's GS scripts need (`5C_trajectory_replay.py`,
`5D_dave2.py` — talk to CARLA, run a pygame UI, project CARLA↔UTM coordinates):

```bash
pip install carla==0.9.16 pygame==2.6.1 pyproj==3.5.0 pyrender==0.1.45
```

Verify:

```bash
python -c "import carla, pygame, pyproj, pyrender; print('OK')"
```

## Step 5. Create the `data_extraction` Environment (venv)

Needed for the CARLA-side scripts (3C, 3F) that `step5.sh` launches.

```bash
python3.10 -m venv ~/.data_extraction
source ~/.data_extraction/bin/activate

pip install -U pip setuptools wheel
pip install -r data_extraction_requirements.txt
```

## Step 6. Download the Precomputed Dataset

This contains everything Steps 1–4 of the full pipeline would normally
produce: the CARLA-ready trajectory, OpenDRIVE map, parked-vehicle JSON, and
trained Gaussian Splatting models with alignment files.

```bash
source ~/.data_extraction/bin/activate
pip install -U gdown

gdown 1MmAYlxy67F1oxDKADHl3yUZochmifV1Q -O data.zip
unzip -o data.zip
rm data.zip
```

:::note
If `gdown` fails, use the [manual download link](https://drive.google.com/file/d/1MmAYlxy67F1oxDKADHl3yUZochmifV1Q/view?usp=sharing) instead.
:::

Verify the result matches this structure:

```
cam2sim/
├── data/
│   ├── data_for_carla/reference_bag/
│   │   ├── camera.json
│   │   ├── trajectory_positions_rear_odom_yaw.json
│   │   └── vehicle_data.json
│   ├── processed_dataset/reference_bag/maps/
│   │   └── map.xodr
│   └── data_for_gaussian_splatting/reference_bag/
│       ├── frame_positions_split_*_1_of_2.txt
│       ├── images_gs_split_*_1_of_2/
│       └── outputs/splatfacto_split_*/splatfacto/<timestamp>/
│           ├── config.yml
│           ├── nerfstudio_models/
│           └── utm_to_nerfstudio_transform.json
```

## Step 7. Fix Absolute Paths in the Gaussian Splatting Configs

Nerfstudio bakes absolute paths (username + project root of the training
machine) into every `config.yml`. Rewrite them to match your machine:

```bash
python 4_gaussian_splatting_preparation/4D_fix_paths.py
```

Run this once, from the project root.

## Step 8. Run It

`step5.sh` ships assuming Conda (sources `conda.sh`, then `conda activate
<env_name>` inside each spawned terminal). If you don't have conda
installed, patch it once: replace the conda-detection block with a check
that each venv's `bin/activate` exists, and swap every
`source '$CONDA_SH'; conda activate '$env_name';` for
`source '$env_path/bin/activate';`, changing `ENV_CARLA` / `ENV_GS` /
`ENV_DAVE` from conda env *names* to venv *paths*
(`$HOME/.data_extraction`, `$HOME/.nerfstudio`, `$HOME/.dave_2`).

Once patched:

```bash
bash 5_execute_simulation/step5.sh
```

This defaults to mode 5C (Gaussian Splatting trajectory replay), which is
what the Quick Start needs.

Or skip the wrapper entirely and run the three underlying scripts by hand,
in order:

```bash
source ~/.data_extraction/bin/activate
python 3_generate_simulation_data/3C_setup_carla.py
python 3_generate_simulation_data/3F_generate_carla_scenario.py

source ~/.nerfstudio/bin/activate
python 5_execute_simulation/5C_trajectory_replay.py
```

Either way, this runs the same three stages in sequence:

1. **CARLA server** — `3C_setup_carla.py` (env `data_extraction`)
2. **Map + parked vehicles** — `3F_generate_carla_scenario.py` (env `data_extraction`)
3. **Gaussian Splatting replay** — `5C_trajectory_replay.py` (env `nerfstudio`)

:::warning
CARLA must **not** already be running before you start this — the launcher
(or `3C_setup_carla.py`) starts it for you.
:::

### Hybrid-Graphics Laptops (NVIDIA Optimus/PRIME)

`3C_setup_carla.py` launches `CarlaUE4.sh` via `subprocess.run([...],
check=True)` with no `env=` argument, so it inherits whatever environment
the Python process runs in — the reliable fix is to build an `env` dict
inside the script itself (merging `os.environ.copy()` with the offload
variables below) and pass it as `env=carla_env` to that `subprocess.run`
call, so it works no matter how the script is invoked:

```python
carla_env = os.environ.copy()
carla_env["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"
carla_env["__NV_PRIME_RENDER_OFFLOAD"] = "1"
carla_env["__NV_PRIME_RENDER_OFFLOAD_PROVIDER"] = "NVIDIA-G0"
carla_env["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
carla_env["__VK_LAYER_NV_optimus"] = "NVIDIA_only"
```

A window should open showing **CARLA on the left, Gaussian-Splatted view on
the right**, replaying the recorded trajectory frame by frame.

## Step 9. Check the Output

Output frames are written to:

```
data/data_for_carla/reference_bag/replay_results/reference_bag_replay/
├── carla/
├── gs/
└── combined/
```

If you see populated `carla/`, `gs/`, and `combined/` folders with per-frame
images after the replay finishes, the Quick Start is successfully
reproduced.

## Troubleshooting

**CUDA compilation errors** (e.g. Ubuntu 24 + GCC 13 too new for the CUDA
toolkit): the pipeline scripts auto-export a compatible compiler when
`gcc-11` is present. If Nerfstudio/`gsplat` still fails to compile:

```bash
sudo apt install gcc-11 g++-11
```

:::note
Tested reference config: Ubuntu 20.04, RTX 4090 (24GB), driver 565.57.01,
CUDA 12.7, Intel Core Ultra 9, 32GB RAM. Other Ubuntu versions and GPUs with
compute capability ≥7.5 should work but aren't validated by the authors.
:::

**`pip install -r data_extraction_requirements.txt` fails to build
`pyliblzfse` / `fpsample`** (CMake errors about missing Python headers or a
missing C library): these packages compile native extensions, which Conda
normally papers over by bundling its own build toolchain. On a plain venv
you need the matching system dev packages:

```bash
sudo apt install python3.10-dev liblzfse-dev
```

Then re-run the `pip install -r data_extraction_requirements.txt` step.

**`Split: none` / black GS panel, with `[WARN] No GS models loaded -
falling back to only_carla mode`** buried in Terminal 3's output, plus a
`torch.load` / `weights_only` / `numpy.core.multiarray.scalar` warning just
above it: PyTorch ≥2.6 changed `torch.load()`'s default to
`weights_only=True`, which breaks loading Nerfstudio's Splatfacto
checkpoints. Fix by adding `weights_only=False` to the checkpoint-loading
call in Nerfstudio's own installed package (not in cam2sim's code):

```bash
grep -n "torch.load(load_path" ~/.nerfstudio/lib/python3.10/site-packages/nerfstudio/utils/eval_utils.py
```

Edit that line (`eval_load_checkpoint`, used for inference/replay — not the
`trainer.py` calls, which are for resuming *training*) to add
`weights_only=False`.

:::note
This patches a file inside the venv's `site-packages`, so it'll be wiped
out by any future `pip install --upgrade nerfstudio` or venv recreation —
worth re-checking if the GS panel goes black again later.
:::

**`ImportError: ... libtorch_cuda.so: undefined symbol: ncclCommResume`**
on `import torch`: a stray/mismatched NCCL package (e.g. a leftover
`nvidia-nccl-cu12` when `torch` actually wants `nvidia-nccl-cu13`, or vice
versa) sitting alongside torch's real dependency. Check what's installed:

```bash
pip show torch | grep Version
pip show nvidia-nccl-cu12 | grep Version   # or nvidia-nccl-cu13
```

Fix by letting pip re-resolve torch's NCCL dependency cleanly:

```bash
pip uninstall -y torch nvidia-nccl-cu12 nvidia-nccl-cu13
pip install torch==<your version>
```

Verify with:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

before rerunning the pipeline.

## Status

✅ Quick Start (Step 5C) successfully reproduced — Terminal 3 shows an
active GS split (not `none`) and the replay window renders CARLA + Gaussian
Splatting views side by side.

## What's Next

Once this reproduces cleanly, the full pipeline (raw ROS bag → COLMAP →
Splatfacto training → DAVE-2 closed-loop driving → validation) is documented
in the repo's "Replication" section — Steps 1 through 6. That's a separate,
much heavier undertaking (own ROS bag, manual COLMAP GUI work per route
segment, training from scratch) and only worth it if the precomputed replay
isn't sufficient for your purposes.
