---
sidebar_position: 2
title: Build CARLA ue4-dev from Source on Ubuntu 26.04
description: Build CARLA (ue4-dev branch) from source on Ubuntu 26.04
---

# Build CARLA ue4-dev from Source

This guide covers building CARLA from source (`ue4-dev` branch) on Ubuntu 26.04, including the required Unreal Engine fork.

:::warning
CARLA is officially tested only on Ubuntu 20.04/22.04. Ubuntu 26.04 is unsupported — expect to hit compiler/toolchain/Python version mismatches along the way. This guide will note fixes as they come up.
:::

## Environment

| | |
|---|---|
| **Unreal Engine path** | `~/thirdparty/UnrealEngine` |
| **CARLA source path** | `~/thirdparty/carla` |
| **Python** | 3.14 |

## Prerequisite: Align GCC and G++ Versions

Ubuntu 26.04 ships two GCC versions side by side:

```sh
$ ls /usr/lib/gcc/x86_64-linux-gnu
15  16
```

but only one matching G++ version:

```sh
$ ls /usr/include/c++
15
```

Clang-10 (the compiler this build relies on) always picks the highest-numbered GCC install it finds, without checking whether that install actually has usable C++ headers to go with it. Here, that means it settles on GCC 16, which has no matching G++ 16 headers installed — and that gap is what causes build failures like `fatal error: 'cstddef' file not found` later on.

To keep GCC and G++ aligned, hide GCC 16 from clang's autodetection so it falls back to GCC 15:

```sh
sudo mv /usr/lib/gcc/x86_64-linux-gnu/16 /usr/lib/gcc/x86_64-linux-gnu/16.bak
```

## Part 1. Build Unreal Engine

CARLA uses a modified fork of Unreal Engine maintained by CARLA.

:::note
Downloading this Unreal Engine fork requires a GitHub account that's linked to Epic Games' organization. If your account isn't linked yet, set that up first by following [Epic's guide](https://www.unrealengine.com/ue-on-github?lang=en-US) — otherwise the clone below will fail.
:::

### 1.1 Clone the Unreal Engine Fork

```sh
mkdir -p ~/thirdparty && cd ~/thirdparty
git clone --depth 1 -b carla git@github.com:CarlaUnreal/UnrealEngine.git
```

:::note
`--depth 1` fetches only the latest commit instead of the full history, and `-b carla` checks out the `carla` branch directly during the clone — together they avoid downloading the repo's full history and every other branch, which matters here since this repo is large.
:::

### 1.2 Set Up and Build Unreal Engine

From inside the `~/thirdparty/UnrealEngine` directory, run:

```sh
./Setup.sh && ./GenerateProjectFiles.sh && make
```

This may take an hour or two depending on your system.

:::warning
Do not add `-j` to use all processor cores (e.g. `make -j$(nproc)`) — this will cause the build to fail. Clang uses all available cores anyway.
:::

### 1.3 Verify the Editor Opens

Confirm Unreal Engine installed correctly by launching the editor:

```sh
cd ~/thirdparty/UnrealEngine/Engine/Binaries/Linux && ./UE4Editor
```

:::note
On first launch, the editor needs to compile shaders, and the log will fill with lines like:

```
[2026.07.30-11.13.07:778][685]LogShaderCompilers: Display: Worker (3/13): shaders left to compile 5
[2026.07.30-11.13.07:865][691]LogShaderCompilers: Display: Worker (10/13): shaders left to compile 4
[2026.07.30-11.13.07:888][692]LogShaderCompilers: Display: Worker (4/13): shaders left to compile 3
[2026.07.30-11.13.07:974][697]LogShaderCompilers: Display: Worker (8/13): shaders left to compile 2
[2026.07.30-11.13.07:977][697]LogShaderCompilers: Display: Worker (9/13): shaders left to compile 1
```

This can take a long time — the editor window will appear grayed out and unresponsive for most of it, since the main UI thread is waiting on the compile workers rather than processing window events. That's expected; don't cancel it, and let it run to completion.
:::

If the editor opens without errors, the Unreal Engine build is complete.

:::warning Troubleshooting: editor hangs at "shaders left to compile: 1"
Occasionally the launch gets genuinely stuck right at the very end, with no further progress for tens of minutes even though the process still shows some CPU usage. This looks like a flaky race condition on the last shader, not a fundamental build or driver problem. Don't leave it running unattended for hours hoping it recovers — kill it and relaunch promptly instead.

**Fix:** kill the process and relaunch:

```sh
kill <PID>
cd ~/thirdparty/UnrealEngine/Engine/Binaries/Linux && ./UE4Editor
```

Shader compilation results are cached, so later attempts have little or nothing left to recompile and typically finish quickly — it may take two or three tries before the editor launches cleanly, which is normal given Ubuntu 26.04 is unsupported for this engine version.
:::

### 1.4 Set the `UE4_ROOT` Environment Variable

CARLA needs to know where your Unreal Engine build lives, so this environment variable has to be set.

To set it for just the current shell session:

```sh
export UE4_ROOT=~/thirdparty/UnrealEngine
```

To make it persist across sessions, add the same line to `.bashrc` or `.profile`:

```sh
echo "export UE4_ROOT=~/thirdparty/UnrealEngine" >> ~/.bashrc
source ~/.bashrc
```

:::note
Adjust the path above if you cloned Unreal Engine somewhere other than your home directory.
:::

## Part 2. Build CARLA

### 2.1 Clone the CARLA Repository

Clone the `ue4-dev` branch of the CARLA repository into `~/thirdparty`:

```sh
mkdir -p ~/thirdparty && cd ~/thirdparty
git clone -b ue4-dev git@github.com:carla-simulator/carla.git
```

:::warning
The repository's default branch is `ue5-dev`, not `ue4-dev` — make sure you pass `-b ue4-dev` explicitly, or you'll end up on the wrong branch.
:::

### 2.2 Set the `CARLA_UE4_ROOT` Environment Variable

The build commands coming up reference the root of the CARLA repo, so it's worth pointing a `CARLA_UE4_ROOT` variable at wherever you cloned it:

```sh
export CARLA_UE4_ROOT=~/thirdparty/carla
```

As with `UE4_ROOT` earlier, add this to your `.bashrc` or `.profile` if you want it to stick around for future sessions rather than just the current shell.

### 2.3 Download the CARLA Content

CARLA also needs a separate repository of 3D assets — maps, vehicles, pedestrians, etc. — that isn't part of the code repo.

Cloning it directly into `${CARLA_UE4_ROOT}/Unreal/CarlaUE4/Content/Carla` is the recommended way to fetch it if you plan on pushing content changes back (either to the main CARLA content repo or your own fork):

```sh
git clone -b master https://bitbucket.org/carla-simulator/carla-content ${CARLA_UE4_ROOT}/Unreal/CarlaUE4/Content/Carla
```

:::note
If you're using your own fork of the content repo, swap in that remote URL instead.
:::

### 2.4 Build CARLA with Make

The remaining commands should all be run from the root of the `~/thirdparty/carla` repo. Building CARLA has two parts — the client and the server — and this step covers the client.

The build relies on `ninja-build`. If your system doesn't already have it, install it before moving on to the next step:

```sh
sudo apt install ninja-build
```

#### Build with ROS2 and Project Chrono

```sh
make setup ARGS="--ros2 --chrono"
```

On success, the output ends with:

```
Setup.sh: Success!
```

#### Build the CARLA Editor

```sh
make CarlaUE4Editor ARGS="--ros2 --chrono"
```

:::warning Troubleshooting: `error: use of undeclared identifier 'uintptr_t'` / `'uint32_t'`
This shows up while compiling `libosm2dr`, specifically in `OutputDevice_File.cpp` via the vendored `foreign/zstr/zstr.hpp` header that this step pulls in. The header uses `uintptr_t` and `uint32_t` without including `<cstdint>`, and nothing else in the include chain happens to pull that in first — it's a genuine missing-include bug in the vendored source, not an environment issue.

**Fix:** patch the header to add the missing include, then rebuild:

```sh
sed -i '11a #include <cstdint>' Build/libosm2dr-source/src/foreign/zstr/zstr.hpp
make CarlaUE4Editor ARGS="--ros2"
```
:::

On success, the output ends with:

```
BuildCarlaUE4.sh: Success!
```

#### Compile the Python API Client

The Python API client is what lets you drive the simulation from a script. You need to build it the first time and again whenever you pull updates; once it's built, you can run your Python scripts against the simulation.

First, install the Python build dependencies:

```sh
sudo apt install python3-auditwheel python3-build python3-distro python3-packaging python3-pyelftools python3-setuptools python3-wheel
```

Before building, override `LDSHARED` so the linker step doesn't fail:

```sh
export LDSHARED="${CARLA_UE4_ROOT}/Build/clang.sh -shared"
```

Then build the client:

```sh
make PythonAPI
```

:::warning
Without the `LDSHARED` override above, this build step is likely to fail with something like:

```
ld.lld: error: unknown argument '--package-metadata=...'
```

This happens during the final link of the Python API's `.so` file. Ubuntu's Python packaging tools automatically add a `--package-metadata=<json>` flag to `LDSHARED` for build-provenance tracking, and while Ubuntu's own patched linker understands that flag, `ld.lld` (the LLVM linker shipped with UE4's clang-10 toolchain) does not. Overriding `LDSHARED` to call UE4's own compiler wrapper directly, without that flag, avoids the problem.
:::

On success, the output ends with:

```
BuildPythonAPI.sh: Success!
```

With the build complete, install the CARLA Python package:

```sh
sudo pip3 install PythonAPI/carla/dist/carla-0.9.16-cp314-cp314-linux_x86_64.whl --break-system-packages
```

The `carla` wheel doesn't pull in any other dependencies, so installing it system-wide like this is low-risk. If you'd rather keep it scoped, drop `sudo` to install it for just your user, or install it inside a virtual environment instead.

#### Compile the Server

This command builds the server and, once done, launches the Unreal Engine editor directly. You'll run it every time you want to bring up the server or open the editor going forward:

```sh
make launch ARGS="--ros2 --chrono"
```

On the first launch, expect to see warnings about shaders and mesh distance fields — these are still loading in the background, and the map won't render correctly until they finish. Later launches skip most of this and come up noticeably faster.

#### Create the Package

To produce the final binary, run `Package.sh` directly rather than `make package`. The `make package` target resets several build options back to their defaults as part of packaging, which has the side effect of turning ROS2 support back off — undoing the `--ros2 --chrono` build you just did.

```sh
Util/BuildTools/Package.sh
```

This can take a while (potentially a couple of hours). On success, the output ends with something like:

```
Package.sh: Copying extra files to package Carla took 0 seconds.
Package.sh: Packaging CARLA release.
Package.sh: Archiving and compressing the project took 1785444335 seconds.
Package.sh: Cooking all other packages took 0 seconds.
Package.sh: Overall packaging took 11867 seconds.
Package.sh: CARLA release created at /home/yi-chen/thirdparty/carla/Dist/CARLA_edf3e9f-dirty.tar.gz
Package.sh: Success!
```

The `.tar.gz` is the packaged build — that's the file to share or move elsewhere. To actually launch the CARLA simulator from it, run `CarlaUE4.sh` inside the extracted package, with `--ros2` to enable ROS2 support:

```sh
~/thirdparty/carla/Dist/CARLA_Shipping_edf3e9f-dirty/LinuxNoEditor/CarlaUE4.sh --ros2
```

Your `Shipping` directory name will differ (it's derived from your own commit hash and build), but the path pattern is the same.

For convenience, set up a `carla` alias in `~/.bashrc` so you don't have to type the full path each time:

```sh
alias carla='/home/yi-chen/thirdparty/carla/Dist/CARLA_Shipping_edf3e9f-dirty/LinuxNoEditor/CarlaUE4.sh'
```

Then reload your shell config and launch with:

```sh
source ~/.bashrc
carla --ros2
```

:::note For laptops with hybrid Intel + NVIDIA graphics
If your laptop has both an Intel integrated GPU and an NVIDIA discrete GPU, you'll need to explicitly offload rendering to the NVIDIA card, or CARLA may end up running on the weaker integrated GPU instead. Use this expanded alias, which sets the environment variables needed for NVIDIA PRIME render offload before launching:

```sh
# Carla with GPU
alias carla='VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
__NV_PRIME_RENDER_OFFLOAD=1 \
__NV_PRIME_RENDER_OFFLOAD_PROVIDER=NVIDIA-G0 \
__GLX_VENDOR_LIBRARY_NAME=nvidia \
__VK_LAYER_NV_optimus=NVIDIA_only \
/home/yi-chen/thirdparty/carla/Dist/CARLA_Shipping_edf3e9f-dirty/LinuxNoEditor/CarlaUE4.sh'
```
:::

### Verify ROS2 Integration

With the simulator running, confirm ROS2 support is actually active by listing the available topics:

```sh
ros2 topic list
```

If `/carla/map` shows up alongside the usual topics, ROS2 is working correctly:

```
/carla/map
/clock
/parameter_events
/rosout
```

### Verify Chrono Integration

There's a chrono example available in `PythonAPI/examples/manual_control_chrono.py`:

```sh
python3 PythonAPI/examples/manual_control_chrono.py
```

If no erros show up in the terminal, Project Chrono is integrated correctly.

### Restore GCC 16

Once everything above is complete, remember to restore GCC 16 so other tools on the system that expect it aren't affected:

```sh
sudo mv /usr/lib/gcc/x86_64-linux-gnu/16.bak /usr/lib/gcc/x86_64-linux-gnu/16
```
