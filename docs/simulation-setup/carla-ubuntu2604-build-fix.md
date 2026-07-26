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

Install the missing headers:

```bash
sudo apt install libstdc++-16-dev
```

:::note
This resolves the missing-directory error, but exposes the next issue, since
GCC 16's actual headers are too new for clang-10 to parse correctly.
:::

## 2. `error: member access into incomplete type '__gnu_cxx::__normal_iterator<...>'`

Boost `filesystem` (`codecvt_error_category.cpp`), `container` (`dlmalloc.cpp`),
and later `msgpack`/`rpclib` fail deep inside `bits/stl_iterator.h`,
`bits/stl_vector.h`, etc.

**Cause:** GCC 16's `libstdc++` implementation of `__normal_iterator`'s
`operator-` uses a trailing-return-type `decltype` that requires two-phase
lookup/incomplete-type handling clang-10's frontend doesn't parse correctly —
a genuine compiler/header-version incompatibility.

Force clang to use GCC **15**'s headers instead, while building Boost
(`-stdlib=libstdc++`):

```bash
export CPLUS_INCLUDE_PATH=/usr/include/c++/15:/usr/include/x86_64-linux-gnu/c++/15:/usr/include/c++/15/backward
rm -rf Build/boost-1.90.0-c10-source/bin.v2   # clear stale partial build
make PythonAPI
```

:::warning
`CPLUS_INCLUDE_PATH` is a shell-wide override — it leaks into every subsequent
compile in that terminal session, including ones that don't want it (see
step 4). Unset it once Boost is done:

```bash
unset CPLUS_INCLUDE_PATH
```
:::

## 3. `CMake Error: unable to find a build program corresponding to "Ninja"`

The `rpclib` CMake configure step fails; `CMAKE_C_COMPILER`/`CMAKE_CXX_COMPILER`
are also reported as not set.

**Cause:** `ninja-build` wasn't installed — the compiler errors were a
downstream symptom of the missing generator.

```bash
sudo apt install ninja-build
```

## 4. `error: no member named 'abort' in namespace 'std'`

`rpclib` (built with `-stdlib=libc++`, using UE4's bundled LibC++ headers) fails
with dozens of "no member named X" errors (`abort`, `size_t`, `malloc`, ...)
while including `<cstdlib>`/`<cmath>`.

**Cause:** `CPLUS_INCLUDE_PATH` from step 2 was still exported and leaked into
this libc++-mode build. UE4's bundled libc++ `stdlib.h` does
`#include_next <stdlib.h>` expecting glibc's plain C header, but with
`/usr/include/c++/15` prepended, it instead resolved GCC's **C++** `stdlib.h`
wrapper (`using std::abort;`), which assumes libstdc++ already ran.

```bash
unset CPLUS_INCLUDE_PATH
make PythonAPI
```

Boost was already built and cached, so it didn't need `CPLUS_INCLUDE_PATH` again.

## 5. Same `__normal_iterator` error, now inside `msgpack`/`rpclib`

After unsetting `CPLUS_INCLUDE_PATH`, `rpclib`'s `msgpack` code (`unpack.hpp`,
`parse.hpp`, using `std::vector`) hits the same incomplete-type error as step 2.

**Cause:** With `CPLUS_INCLUDE_PATH` unset, clang's autodetection re-selected
GCC 16 (which now has real headers from step 1) — same GCC-16/clang-10
incompatibility as step 2, reached via a different code path.

Since `CPLUS_INCLUDE_PATH` collides with libc++ builds (step 4), hide GCC 16
from clang's autodetection instead so it falls back to GCC 15 on its own:

```bash
sudo mv /usr/lib/gcc/x86_64-linux-gnu/16 /usr/lib/gcc/x86_64-linux-gnu/16.bak
sudo mv /usr/include/c++/16 /usr/include/c++/16.bak
make PythonAPI
```

Restore afterward if GCC 16 is needed for other work:

```bash
sudo mv /usr/lib/gcc/x86_64-linux-gnu/16.bak /usr/lib/gcc/x86_64-linux-gnu/16
sudo mv /usr/include/c++/16.bak /usr/include/c++/16
```

:::caution GCC 16 has since been restored
After the build finished, GCC 16 was moved back into place (verified via
`clang++ -v -x c++ -E - < /dev/null 2>&1 | grep -A6 "search starts here"`,
which now correctly resolves `/usr/include/c++/16`). This means a **clean
rebuild from scratch will likely hit steps 2/5 again** — re-hide GCC 16 as
shown above if that happens.
:::

## 6. `error: use of undeclared identifier 'uintptr_t'` / `'uint32_t'`

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

## 7. `ld.lld: error: unknown argument '--package-metadata=...'`

The final link of `libcarla.cpython-314-x86_64-linux-gnu.so` (Python API wheel
build) fails.

**Cause:** Ubuntu's Python packaging (`sysconfig`/`distutils`) automatically
injects a `--package-metadata=<json>` linker flag for build-provenance tracking
into `LDSHARED`. Ubuntu's patched system linker understands it, but `ld.lld` —
the LLVM linker bundled with UE4's clang-10 toolchain — does not.

Override `LDSHARED` to drop the flag while still using the correct
(UE4-bundled) compiler wrapper:

```bash
export LDSHARED="Build/clang.sh -shared"
make PythonAPI
```

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
| `CPLUS_INCLUDE_PATH=/usr/include/c++/15:...` | Boost build only (`-stdlib=libstdc++`) | Yes — collides with libc++ builds |
| `/usr/lib/gcc/x86_64-linux-gnu/16` and `/usr/include/c++/16` moved to `.bak` | rpclib and any further clang-10 builds needing GCC headers | Optional — restore if GCC 16 needed elsewhere |
| `libstdc++-16-dev` installed | Superseded by hiding GCC 16 | No need to remove |
| `ninja-build` installed | CMake/Ninja-based subprojects (rpclib) | Keep installed |
| `zstr.hpp` patched with `#include <cstdint>` | libosm2dr | Reapply if source is re-cloned |
| `LDSHARED` override | Final Python API `.so` link | Only needed for that link step |

:::info Rebuilding from scratch later
If GCC 16 is still hidden (step 5), a clean rebuild should work without
repeating that step. If GCC 16 has been restored in the meantime, expect to
hit steps 2/5 again — consider pointing CARLA's Boost `user-config.jam` and
CMake toolchain files at GCC 15 explicitly instead of relying on clang's
autodetection, for a more permanent fix.
:::
