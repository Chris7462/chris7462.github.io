---
sidebar_position: 5
title: Build CARLA ue5-dev from Source with ROS2 Support on Linux
description: Build CARLA (ue5-dev branch) from source on Linux with native ROS2 support enabled, using the extended/manual build instructions
---

# Build CARLA ue5-dev from Source with ROS2 Support

This guide covers building CARLA UE5 (`ue5-dev` branch) from source on Linux
using the manual/extended build path, with native ROS2 support enabled.

:::warning
The Unreal Engine 5 version of CARLA requires **Ubuntu 22.04 at minimum**. It
has not been configured to build on older Ubuntu versions.
:::

:::note
The official docs also offer a one-shot `CarlaSetup.sh` script that installs
prerequisites, builds Unreal Engine, and builds CARLA all in one pass, with
ROS2 enabled by default. This guide instead follows the manual/extended path,
which gives more control and is what's needed if the setup script fails or
you want to build Unreal Engine or content separately.
:::

## Prerequisite: Install Build Tools

```sh
sudo apt update
sudo apt install build-essential ninja-build libvulkan1 python3 python3-dev python3-pip git git-lfs
```

## Part 1. Build Unreal Engine

CARLA UE5 uses a modified fork of Unreal Engine 5.5 maintained by CARLA.

:::note
Downloading this Unreal Engine fork requires a GitHub account that's linked
to Epic Games' organization. If your account isn't linked yet, set that up
first by following [Epic's guide](https://www.unrealengine.com/en-US/ue-on-github) —
otherwise the clone below will fail.
:::

### 1.1 Clone the Unreal Engine Fork

```sh
git clone --depth 1 -b ue5-dev-carla git@github.com:CarlaUnreal/UnrealEngine.git
cd UnrealEngine
```

### 1.2 Set Up and Build Unreal Engine

```sh
./Setup.sh && ./GenerateProjectFiles.sh && make
```

:::note
The first build of Unreal Engine can take up to 3 hours.
:::

### 1.3 Verify the Editor Opens

Confirm Unreal Engine installed correctly by launching the editor:

```sh
cd Engine/Binaries/Linux
./UnrealEditor
```

If the editor opens without errors, the Unreal Engine build is complete.

### 1.4 Set the `CARLA_UNREAL_ENGINE_PATH` Environment Variable

CARLA needs to know where your Unreal Engine build lives, so this
environment variable has to be set.

To set it for just the current shell session:

```sh
export CARLA_UNREAL_ENGINE_PATH=<PATH_TO_UNREAL_ENGINE_FOLDER>
```

(Optional) To make it persist across sessions — useful if you plan to
reconfigure or rebuild CARLA later in a fresh shell — add the same line to
`.bashrc` or `.profile`:

```sh
echo 'export CARLA_UNREAL_ENGINE_PATH=<PATH_TO_UNREAL_ENGINE_FOLDER>' >> ~/.bashrc
source ~/.bashrc
```

:::warning
If `CARLA_UNREAL_ENGINE_PATH` isn't set, the CARLA build step later will
instead download and build Unreal Engine itself, adding over an hour of build
time and roughly 225 GB of disk space.
:::

## Part 2. Build CARLA

### 2.1 Clone the CARLA Repository

Clone the `ue5-dev` branch of the CARLA repository:

```sh
git clone https://github.com/carla-simulator/carla.git
```

:::note
`ue5-dev` is the repository's default branch, so no `-b` flag is needed here.
:::

### 2.2 Clone the CARLA Content

Clone the content repository into `Unreal/CarlaUnreal/Content` (create the
folder first if it doesn't exist):

```sh
mkdir -p CARLA_ROOT/carla/Unreal/CarlaUnreal/Content
cd CARLA_ROOT/carla/Unreal/CarlaUnreal/Content
git clone --single-branch --depth 1 -b ue5-dev https://bitbucket.org/carla-simulator/carla-content.git Carla
```

:::note
This can take a while — it's downloading a large amount of asset data.
:::

### 2.3 Configure CARLA with ROS2 Enabled

From the root of the CARLA repo, set up a build preset with ROS2 support
enabled:

```sh
cmake --preset Release -DENABLE_ROS2=ON
```

:::warning Preset names
The CARLA docs sometimes reference preset names like `Linux-Release` — these
don't exist in `CMakePresets.json`. The actual preset names defined there are
`Debug`, `Development`, and `Release`.
:::

:::warning ROS2 is not enabled by any preset
None of the presets in `CMakePresets.json` set `ENABLE_ROS2` — it defaults to
off, which is why the flag above must be passed explicitly on the command
line to get native ROS2 support. CMake merges it with the preset's existing
cache variables.
:::

:::note
If ROS2 dependencies (Fast-DDS, Fast-CDR, etc.) aren't discoverable at
configure time, `ENABLE_ROS2=ON` may silently fail to pick them up. Check the
configure log for ROS2-related messages after reconfiguring.
:::

### 2.4 Build and Install the Python API

```sh
cmake --build Build/Release --target carla-python-api-install
```

### 2.5 Launch the Editor

```sh
cmake --build Build/Release --target launch
```

### 2.6 Build a Package

```sh
cmake --build Build/Release --target package
```

The package is generated in `$CARLA_PATH/Build/Package`. Use the
`package-development` target instead for a package with debug logging.

### 2.7 Run the Package

The built package is located under `Build/<preset>/Package/`, in a folder
named after the CARLA version and build type — for this build:

```
$CARLA_PATH/Build/Release/Package/Carla-0.10.0-Linux-Shipping/
```

From inside that package folder, run the simulator from the `Linux`
subfolder:

```sh
cd Linux
./CarlaUnreal.sh --ros2
```

To use the Python API, you need to install the wheel that matches this
package build — it's located under `PythonAPI/carla/dist/` inside that same
package folder:

```sh
sudo pip3 install PythonAPI/carla/dist/carla-***.whl --break-system-packages
```

## Verify ROS2 Integration

With the simulator running, list the available ROS2 topics:

```sh
ros2 topic list
```

A `/clock` topic confirms CARLA is publishing over ROS2 correctly:

```
/clock
/parameter_events
/rosout
```

If `/clock` is missing, native ROS2 support was **not** compiled in — go back
to Part 2.3 and confirm `ENABLE_ROS2=ON` was passed during configure, then
rebuild and repackage.
