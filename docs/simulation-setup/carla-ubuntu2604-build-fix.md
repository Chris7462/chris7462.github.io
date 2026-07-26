---
sidebar_position: 2
title: Fix CARLA PythonAPI Build on Ubuntu 26.04
description: Resolve clang-10/GCC 16 toolchain incompatibilities when building CARLA's PythonAPI on Ubuntu 26.04
---

# Fix CARLA PythonAPI Build on Ubuntu 26.04

This guide covers the toolchain issues encountered running `make PythonAPI` for
[CARLA](https://carla.org/) on Ubuntu 26.04, and how to resolve each one. CARLA
builds against UE4.26's bundled `clang-10.0.1` toolchain, which predates Ubuntu
26.04's GCC 15/16 by several years — most of the failures below trace back to
that version gap.

:::info Environment
- Ubuntu 26.04 (`resolute`), system GCC 15.2.0 default, GCC 16 also present
- UE4.26 bundled `clang-10.0.1` (`v17_clang-10.0.1-centos7`)
- Boost 1.90.0 (`boost-1.90.0-c10`)
:::

:::caution Root cause
Clang-10's GCC-installation autodetection always picks the **highest-numbered**
GCC install it finds on the host, without checking whether that install actually
has working C++ headers. On this system it kept selecting GCC 16 even when
GCC 16's headers either didn't exist yet or weren't compatible with a 2020-era
clang frontend. Nearly every fix below is a variation on working around this.
:::

## 1. `fatal error: 'cstddef' file not found`

Boost (`libs/container`, `libs/program_options`, ...) fails to compile with
clang-10 unable to find `<cstddef>`.

**Cause:** `/usr/lib/gcc/x86_64-linux-gnu/16` existed (compiler driver only), but
`libstdc++-16-dev` (the actual C++ headers) wasn't installed. Clang-10 still
selected GCC 16 as its preferred install and computed a malformed header search
path for it, finding nothing.

Confirm with:

```bash
clang++ -v -x c++ -E - < /dev/null 2>&1 | grep -A6 "search starts here"
# shows "ignoring nonexistent directory" for the GCC 16 c++ include paths
```

**Fix:** don't bother installing `libstdc++-16-dev` — even with real headers
present, GCC 16's `libstdc++` is too new for clang-10's frontend to parse
correctly (see step 2's original cause below). Skip straight to hiding GCC 16
from clang's autodetection entirely, so it falls back to GCC 15 on its own:

```bash
sudo mv /usr/lib/gcc/x86_64-linux-gnu/16 /usr/lib/gcc/x86_64-linux-gnu/16.bak
sudo mv /usr/include/c++/16 /usr/include/c++/16.bak
rm -rf Build/boost-1.90.0-c10-source/bin.v2   # clear stale partial build
make PythonAPI
```

Restore afterward if GCC 16 is needed for other work:

```bash
sudo mv /usr/lib/gcc/x86_64-linux-gnu/16.bak /usr/lib/gcc/x86_64-linux-gnu/16
sudo mv /usr/include/c++/16.bak /usr/include/c++/16
```

:::note Why not `libstdc++-16-dev` + `CPLUS_INCLUDE_PATH`?
An earlier pass at this fix installed `libstdc++-16-dev` and then forced GCC 15
headers onto clang via `CPLUS_INCLUDE_PATH` for the Boost build only. That works,
but `CPLUS_INCLUDE_PATH` is a shell-wide override — it leaks into later
`-stdlib=libc++` compiles (rpclib) and breaks those in a different way
(`no member named 'abort' in namespace 'std'`), so it has to be carefully
unset between build stages. Hiding GCC 16 outright avoids all of that: clang
never sees GCC 16 at any stage, so there's nothing to leak or unset, and the
same fix carries through Boost, rpclib, and later `libcarla` builds without
re-triggering.
:::

:::danger GCC 16 must stay hidden for any CARLA build target
Restoring GCC 16 mid-project (even after `make PythonAPI` succeeds) will break
`make package` the same way — it re-triggers the exact same
`__normal_iterator` incomplete-type error (see below), this time across dozens
of locations directly in `libcarla`'s own source (`Actor.cpp`, `Client.cpp`,
`Control.cpp`, ...) rather than just Boost/rpclib. `make package` recompiles
`libcarla.cpp` from scratch, so it hits clang-10's GCC-autodetection fresh
every time.

**Practical takeaway:** keep GCC 16 hidden (`.bak`'d) for the entire duration
of CARLA build work on this machine, not just for one target. Only restore it
when you're done building CARLA and need GCC 16 for something else — expect
to re-hide it before touching `make PythonAPI`, `make package`, or any other
CARLA build target again.
:::

:::tip Start-of-session checklist
The GCC 16 hide and the `LDSHARED` override (step 4) are both **per-session,
not one-time** fixes — `make package` re-derives its own compiler/linker flags
from scratch on every invocation, in a fresh terminal. Run both of these at
the start of any CARLA build session, before running `make PythonAPI` or
`make package`, to skip the failed-build round trip:

```bash
sudo mv /usr/lib/gcc/x86_64-linux-gnu/16 /usr/lib/gcc/x86_64-linux-gnu/16.bak 2>/dev/null
sudo mv /usr/include/c++/16 /usr/include/c++/16.bak 2>/dev/null
export LDSHARED="/home/yi-chen/thirdparty/carla/Build/clang.sh -shared"
```

(The `2>/dev/null` guards make the `mv` commands harmless to re-run if GCC 16
is already hidden from a previous session.)
:::

### Background: why `cstddef` and `__normal_iterator` were the same root problem

Even if `libstdc++-16-dev` had been installed to fix the missing-header error
above, a second error would have followed immediately:

```
error: member access into incomplete type '__gnu_cxx::__normal_iterator<...>'
```

surfacing deep inside `bits/stl_iterator.h`, `bits/stl_vector.h`, etc., in
Boost `filesystem` (`codecvt_error_category.cpp`), `container` (`dlmalloc.cpp`),
and later `msgpack`/`rpclib`.

**Cause:** GCC 16's `libstdc++` implementation of `__normal_iterator`'s
`operator-` uses a trailing-return-type `decltype` that requires two-phase
lookup/incomplete-type handling clang-10's frontend doesn't parse correctly —
a genuine compiler/header-version incompatibility, independent of whether the
headers are present. Hiding GCC 16 (above) avoids this error entirely, since
clang never gets a chance to select GCC 16's headers in the first place — for
Boost, for rpclib, or for `libcarla` itself later on.

## 2. `CMake Error: unable to find a build program corresponding to "Ninja"`

The `rpclib` CMake configure step fails; `CMAKE_C_COMPILER`/`CMAKE_CXX_COMPILER`
are also reported as not set.

**Cause:** `ninja-build` wasn't installed — the compiler errors were a
downstream symptom of the missing generator.

```bash
sudo apt install ninja-build
```

## 3. `error: use of undeclared identifier 'uintptr_t'` / `'uint32_t'`

`libosm2dr` fails compiling `OutputDevice_File.cpp`, via the vendored
`foreign/zstr/zstr.hpp`.

**Cause:** A genuine missing-include bug in the vendored header — it uses
`uintptr_t`/`uint32_t` without including `<cstdint>`, and nothing else in the
include chain pulls it in first.

```bash
sed -i '11a #include <cstdint>' \
  Build/libosm2dr-source/src/foreign/zstr/zstr.hpp
```

(Inserted after the existing `#include <cassert>` on line 11.)

:::note
`Build/libosm2dr-source` is a fresh clone — if it's ever re-cloned (e.g. a
`rm -rf Build/` and rebuild from scratch), this patch needs to be reapplied.
:::

## 4. `ld.lld: error: unknown argument '--package-metadata=...'`

The final link of `libcarla.cpython-314-x86_64-linux-gnu.so` (Python API wheel
build) fails.

**Cause:** Ubuntu's Python packaging (`sysconfig`/`distutils`) automatically
injects a `--package-metadata=<json>` linker flag for build-provenance tracking
into `LDSHARED`. Ubuntu's patched system linker understands it, but `ld.lld` —
the LLVM linker bundled with UE4's clang-10 toolchain — does not.

Override `LDSHARED` to drop the flag while still using the correct
(UE4-bundled) compiler wrapper:

```bash
export LDSHARED="/home/yi-chen/thirdparty/carla_UE4.26/Build/clang.sh -shared"
make PythonAPI
```

:::warning Use an absolute path, not `Build/clang.sh`
A relative path here (`export LDSHARED="Build/clang.sh -shared"`) works when
invoked directly, but fails with `command 'Build/clang.sh' failed: No such
file or directory` once `pip`'s isolated build environment (`* Creating
isolated environment: venv+pip...`) runs the actual `build_ext`/link step —
that subprocess runs from a different working directory, so the relative
path no longer resolves. Always export the full absolute path to
`Build/clang.sh`, matching the start-of-session checklist above.
:::

## 5. ROS2 topics don't appear even though the server runs fine with `--ros2`

`carla --ros2` launches without errors, `ros2_native.py` runs and spawns the
vehicle/sensors successfully, but `ros2 topic list` only ever shows the
default ROS2 topics (`/parameter_events`, `/rosout`) — nothing under
`/carla/...`.

**Cause:** CARLA's native ROS2 support (FastDDS-based server-side topic
publishing) is a **compile-time opt-in**, not just the `--ros2` runtime flag.
The `--ros2` CLI flag only takes effect if the binary was built with ROS2
support in the first place. A binary built via plain `make PythonAPI` /
`make package` (no `--ros2` at build time) silently accepts `--ros2` at
launch and does nothing — no error, no log entry, just no DDS participant.

Confirm via the editor's `OptionalModules.ini`:
```bash
grep -i ros2 Unreal/CarlaUE4/Config/OptionalModules.ini
# Ros2 OFF  <- confirms it wasn't built in
```

:::info Not documented in CARLA's official build guide
CARLA's official [Linux build page](https://carla.readthedocs.io/en/latest/build_linux/)
never mentions a build-time `--ros2` flag at all. The only place it shows
`--ros2` is as a *runtime* flag on an already-packaged binary, in the
"Running tests" section:
`./Dist/CARLA_<package_id>/LinuxNoEditor/CarlaUE4.sh --ros2 ...`.
The build-time flag documented below only surfaces by reading CARLA's own
`Util/BuildTools/Setup.sh` and `BuildCarlaUE4.sh` scripts directly — both
declare `ros2` as a recognized `getopt` long option, and `BuildCarlaUE4.sh`
is what actually writes `Ros2 ON`/`Ros2 OFF` into `OptionalModules.ini`.
:::

**Fix:** rebuild with `--ros2` passed to the relevant build stages. Two
scripts matter here — `Setup.sh` (fetches Fast-DDS/CycloneDDS/Zenoh as new
dependencies) and `BuildCarlaUE4.sh` (compiles `libcarla_ros2` and links it
into the editor, and writes `Ros2 ON` into `OptionalModules.ini`).
`BuildPythonAPI.sh` and `Package.sh` do **not** accept `--ros2` at all (ROS2
support is server-side only) — passing it to them errors on unrecognized
option, so build each stage separately rather than relying on `make
package`'s automatic prerequisite chaining (which would otherwise re-run
`BuildCarlaUE4.sh` *without* `--ros2` and flip `Ros2 OFF` again before
packaging):

```bash
make setup ARGS="--ros2"
make CarlaUE4Editor ARGS="--ros2"
make PythonAPI
Util/BuildTools/Package.sh
```

:::danger `make launch ARGS="--ros2"` needed too, if launching via the Editor
Compiling ROS2 in (`Ros2 ON` in `OptionalModules.ini`) is necessary but was
historically **not sufficient** for the Editor-launch path. Before
[carla-simulator/carla#9665](https://github.com/carla-simulator/carla/pull/9665)
(merged into `ue4-dev`, fixing [#9511](https://github.com/carla-simulator/carla/issues/9511)),
`--ros2` was consumed by `BuildCarlaUE4.sh` purely for the build-time
`OptionalModules.ini` write and never forwarded to the actual UE4 Editor
process — at runtime `CarlaSettings.cpp` checks `FParse::Param(TEXT("-ros2"))`
on the editor's own command line, which always failed because the flag never
reached it. Result: `Ros2 ON` confirmed, editor launched, but still no
`/carla/...` topics — because the running process was never actually told to
enable ROS2.

The fix appends `--ros2` (and a new `--dds-middleware=<value>` option) to
`EDITOR_FLAGS` in both `BuildCarlaUE4.sh` and `BuildCarlaUE4.bat`. With the
fix present, launching via the Editor requires **passing `--ros2` again at
launch time**, not just at the earlier `CarlaUE4Editor` build step:

```bash
make launch ARGS="--ros2"
```

This only affects the **Editor** launch path (`make launch`). A
**packaged** build (`Util/BuildTools/Package.sh` → `./Dist/.../CarlaUE4.sh
--ros2 ...`) reads the runtime `--ros2` flag directly and was never affected
by this bug — it's the more reliable path if you're unsure whether your
checkout includes this fix. Confirm with:
```bash
git log --oneline --all | grep -i 9665
git log -p --follow -- Util/BuildTools/BuildCarlaUE4.sh | grep -A5 "EDITOR_FLAGS.*ros2"
```
:::

:::note `make CarlaUE4Editor` and `make PythonAPI` are independent
Order doesn't matter between these two — `make PythonAPI` only runs
`Setup.sh` → `BuildLibCarla.sh` (Client.Release) → compiles `libcarla.cpp`.
It never touches `BuildCarlaUE4.sh` or `CarlaUE4Editor`, so it doesn't
rebuild the editor and doesn't reset `OptionalModules.ini`. ROS2 native
support is server-side only, so the Python client build doesn't need or use
`--ros2` at all. (Verify the exact target name for your checkout with
`make help` or `grep -n "^PythonAPI" Util/BuildTools/Linux.mk` — some forks
may expose a separate `PythonAPI.wheel` target.)
:::

:::tip `parse-options: unrecognized option '--ros2'` warning is benign
This warning can appear early in the output of `make CarlaUE4Editor
ARGS="--ros2"` (or `make setup ARGS="--ros2"`) without stopping the build.
It's a known, long-standing quirk of CARLA's Makefile — `$(ARGS)` gets
broadcast to more than one script in the build chain, and only the script(s)
that actually declared `ros2` as a valid option use it; another script
earlier in the chain prints the warning and continues anyway (same pattern
reported in [carla-simulator/carla#3766](https://github.com/carla-simulator/carla/issues/3766)
with a different flag). Don't rely on the warning's absence — confirm the
flag actually took effect:
```bash
grep -i ros2 Unreal/CarlaUE4/Config/OptionalModules.ini
# expect: ... Ros2 ON ...
```
:::

:::warning `Setup.sh` can silently half-install Fast-DDS
`Setup.sh` gates the whole Fast-DDS/foonathan-memory-vendor build behind
`if [[ -d ${FASTDDS_INSTALL_DIR} ]]`, but creates that directory with
`mkdir -p` **before** attempting the build — and never checks exit codes
along the way. If the build fails or is interrupted partway through, the
directory still exists, so every subsequent run reports `"FastDDS already
installed"` and skips the entire build forever, even though it never
finished. If `make CarlaUE4Editor ARGS="--ros2"` fails with something like
`libfoonathan_memory .a not found`, delete the partial install and force a
clean re-run:
```bash
rm -rf Build/fast-dds-install Build/foonathan-memory-vendor-source Build/fast-dds-lib-source
make setup ARGS="--ros2"
```
Verify it actually completed by checking for real build artifacts, not the
script's own "Success!" message:
```bash
find Build/fast-dds-install -iname "*.a"
# expect libfoonathan_memory-*.a, libfastcdr.a, libfastrtps.a, libcrypto.a, libssl.a
```
:::

Re-install the freshly built wheel (force-reinstall, since the version
string is unchanged) and re-package:
```bash
sudo pip3 install --break-system-packages --force-reinstall \
  PythonAPI/carla/dist/carla-0.9.16-cp314-cp314-linux_x86_64.whl
```

Confirm the fix by checking `OptionalModules.ini` shows `Ros2 ON`, then
re-launching and checking topics — via the Editor, `--ros2` must be passed
again at launch time (see the `#9665` note above); via a packaged build, the
Dist binary's own `--ros2` flag is sufficient:
```bash
grep -i ros2 Unreal/CarlaUE4/Config/OptionalModules.ini   # Ros2 ON

# Editor path
make launch ARGS="--ros2"

# or packaged path
./Dist/CARLA_<package_id>/LinuxNoEditor/CarlaUE4.sh --ros2 -RenderOffScreen \
  -nosound

ros2 topic list
# /carla/hero/... topics now appear alongside /clock, /tf, /carla/map
```

:::note Package.sh output has a bogus duration line
`Package.sh` may print something like
`Archiving and compressing the project took 1785078261 seconds.` — a bug in
its own timing calculation, not a real ~56-year archive step. Harmless;
ignore it.
:::

## Result

`make PythonAPI` completes successfully, producing:

```
PythonAPI/carla/dist/carla-0.9.16-cp314-cp314-linux_x86_64.whl
```

Install and verify:

```bash
sudo pip3 install --break-system-packages PythonAPI/carla/dist/carla-0.9.16-cp314-cp314-linux_x86_64.whl
python3 -c "import carla; print(carla.__file__)"
# -> /usr/local/lib/python3.14/dist-packages/carla/__init__.py
```

:::tip
Installed here with `sudo pip3` into system `dist-packages` for convenience. A
virtual environment is safer long-term if other projects on the machine need a
different `carla` version.
:::

## Quick reference

| Change | Needed for | Revert after? |
|---|---|---|
| `/usr/lib/gcc/x86_64-linux-gnu/16` and `/usr/include/c++/16` moved to `.bak` | All clang-10 build stages (Boost, rpclib, `libcarla`) that would otherwise hit GCC 16's incompatible headers | Optional — restore if GCC 16 needed elsewhere; must be hidden again before further CARLA builds |
| `ninja-build` installed | CMake/Ninja-based subprojects (rpclib) | Keep installed |
| `zstr.hpp` patched with `#include <cstdint>` | libosm2dr | Reapply if source is re-cloned |
| `LDSHARED` override, **absolute path** | Final Python API `.so` link | Only needed for that link step; relative path breaks inside pip's isolated build subprocess |
| `make setup ARGS="--ros2"` | Fetch/build Fast-DDS, CycloneDDS, Zenoh | Re-run any time `Build/fast-dds-install` is deleted |
| `make CarlaUE4Editor ARGS="--ros2"` | Compile `libcarla_ros2`, write `Ros2 ON` to `OptionalModules.ini` | Must re-run after any plain (non-`--ros2`) `CarlaUE4Editor`/`launch`/`package` build, which flips it back to `Ros2 OFF` |
| `parse-options: unrecognized option '--ros2'` warning | — | Benign; ignore, then verify `OptionalModules.ini` shows `Ros2 ON` |
| `make launch ARGS="--ros2"` (not just at `CarlaUE4Editor` build time) | Activating ROS2 at runtime when launching via the **Editor** | Per-launch; fixed by merged PR [#9665](https://github.com/carla-simulator/carla/pull/9665) — confirm your checkout includes it |

:::info Rebuilding from scratch later
If GCC 16 is still hidden, a clean rebuild should work without repeating any
of the above. If GCC 16 has been restored in the meantime, expect to hit
step 1's error again — consider pointing CARLA's Boost `user-config.jam` and
CMake toolchain files at GCC 15 explicitly instead of relying on clang's
autodetection, for a more permanent fix that doesn't depend on hiding
directories at all.
:::
