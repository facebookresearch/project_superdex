# Project SuperDex

> A unified platform for dexterous manipulation research

**Welcome to Project SuperDex.**

SuperDex brings together a purpose-built physics engine, robotics authoring
tools, and a scalable reinforcement learning interface in a unified simulation
platform, with VR-based teleoperation and additional capabilities planned for
future releases.

Visit the [Project SuperDex website](https://projectsuperdex.com/) for more.

| Threaded Screw | Manipulation |
| :-: | :-: |
| ![Threaded Screw](https://github.com/user-attachments/assets/ddac8a5f-2f72-4ca0-b555-a090513f224a) | ![Manipulation](https://github.com/user-attachments/assets/406ea05e-6117-4b81-8d0a-0d6ae1cbaebd)|

## Building blocks

- **SuperDex Physics** — A contact-first physics engine purpose-built for tactile
  manipulation, and applicable wherever stable contact and accurate sensing
  matter. This is the simulation backbone of Project SuperDex.
- **SuperDex Robotics** — A robotics SDK that provides robot definitions and
  composition, controllers, sensors, actuators, and the framework that
  aggregates them into complete simulation configs.
- **SuperDex Studio** — A lightweight desktop GUI authoring and visualization
  application for creating, editing, and validating the simulation assets —
  robots, meshes, task prefabs, and scenes.
- **SuperDex Lab** — The simulation harness that abstracts the Markov decision
  process and dynamics-constrained-optimization underpinning RL, MPC, and
  system-ID.

---

## Gallery

The following videos were recorded in real-time using SuperDex Teleop (available in Q4 2026).

| Soft Sponge | Fruit Bag |
| :-: | :-: |
| ![Soft Sponge](https://github.com/user-attachments/assets/b19c8848-0e06-42c9-80b2-9c58d81774f7) | ![Fruit Bag](https://github.com/user-attachments/assets/09aabf80-9e43-4855-99a6-c7bcac0c974b) |
| Cereal Box | Rope Braid |
| ![Cereal Box](https://github.com/user-attachments/assets/8827c424-0062-4c93-8a73-1586854674c2) | ![Rope Braid](https://github.com/user-attachments/assets/50531ef6-1db7-48a8-aa27-f935665ed723) |
| Contact Viz | Puzzle Cube |
| ![Contact Viz](https://github.com/user-attachments/assets/03601744-fb9d-4c2e-aa1c-a25294df3274) | ![Puzzle Cube](https://github.com/user-attachments/assets/6afa67b6-787d-401e-803b-a7a713c76220) |
| Compose Bots  | Soft Fingertips |
| ![Compose Bots](https://github.com/user-attachments/assets/bca2d36d-5ddd-4aa3-9863-e4903521b6b0) | ![Soft Fingertips](https://github.com/user-attachments/assets/ed08d3ab-af15-47a2-b7e2-d9b125e61b78) |

---

## SuperDex Studio

SuperDex Studio is the desktop application where you turn raw CAD and robot descriptions into native SuperDex Assets - bots, prefabs, and scenes. This is the GUI toolkit of Project SuperDex.

Click to learn more about [SuperDex Studio](https://projectsuperdex.com/studio/).

| SuperDex Studio | Apply Forces |
| :-: | :-: |
| ![SuperDex Studio](https://github.com/user-attachments/assets/4f3ef7df-9c22-4692-bd0b-71f23b4ff3a8) | ![Apply Forces](https://github.com/user-attachments/assets/d746b626-4aab-4a0e-8bc7-98c65c29af68) |

---

## SuperDex Lab (early preview)

SuperDex Lab connects simulation and policy development through a Gymnasium-style API for reinforcement learning. It is currently in early preview and will receive substantial improvements.

Click to learn more about [SuperDex Lab](https://projectsuperdex.com/lab/).

![Reinforcement Learning](https://github.com/user-attachments/assets/87baf336-9d57-4528-a1cf-a5382fa7a0d3)

---

## Q4 2026: SuperDex Teleop

SuperDex Teleop is the next major module coming to Project SuperDex. Initial components for Unreal Engine 5 based virtual teleoperation will be available in Q4 2026.

![Ben Virtual Teleop](https://github.com/user-attachments/assets/08a742a1-b7e5-4f77-a69a-8e24a761048e)

SuperDex Teleop runs natively on-device on Quest 3. No remote PC, no streaming. Includes support for both hand tracking and controllers including mixed mode. Runs in pure C++ for low-latency, real-time, scalable virtual teleop.

![Android Teleop](https://github.com/user-attachments/assets/53639e05-5f6c-4cc6-b693-b3ae6e87ff2f)

---

## Requirements
* **OS:** Linux (x86_64), Windows (x86_64), macOS (ARM)
* **Python** 3.12
   - Pre-built wheels are currently provided only for Python3.12. More flexible abi3 wheels will be available in a future release.

## Get the Source Code and Examples

* Clone: `git clone --branch stable https://github.com/facebookresearch/project_superdex.git`
    * Note: `stable` branch always matches latest published release

## Quick Start (Python)

Project SuperDex has first-class support for Python across the board. The quickest way to get started is with `uv` and `pypi` wheels.

1. Install pre-requisites
    * [uv](https://docs.astral.sh/uv/getting-started/installation/)
    * Linux Only: `SuperDex Studio` and `SuperDex Physics Debugger` require an available X11 display (native X11 or XWayland), a graphics driver supporting OpenGL 4.1, and the following runtime components:
       * Ubuntu/Debian: `sudo apt install libgl1 libx11-6 libx11-xcb1 libxcb1 libxext6 libxrandr2 libxinerama1 libxcursor1 libxi6 zenity`
       * Fedora/RHEL: `sudo dnf install libglvnd-glx libX11 libX11-xcb libxcb libXext libXrandr libXinerama libXcursor libXi zenity`
2. `cd` into the `project_superdex` source directory
3. Create venv `uv venv`
4. Pip Install: `uv pip install superdex`
5. Run
    * Optional: To run Python examples in double precision (float64), set the environment variable `SUPERDEX_PRECISION=double`; otherwise, single precision is used.
    * Physics example: `uv run --no-project superdex_physics/examples/example_tendon_comparison.py`
    * Robotics example: `uv run --no-project superdex_robotics/examples/control/example_osc_jsc_control.py`
    * SuperDex Studio: `uv run --no-project superdex-studio`
    * Note:  `--no-project` is required for `uv run` cmds within this repo or else it will build from source

## Building from Source

### Install Pre-requisites

* [CMake](https://cmake.org/download/) (v3.25 or newer)
* [Ninja](https://github.com/ninja-build/ninja/releases)
* [uv](https://docs.astral.sh/uv/getting-started/installation/) (for Python build)
* Linux:
    * [Clang](https://clang.llvm.org/get_started.html) (v17 or newer):
        * Check your installed version: `clang --version`. The executable may be versioned instead (for example, `clang-22 --version`).
        * If needed, install Clang:
            * Ubuntu/Debian: Install Clang 22 from the [official LLVM repository](https://apt.llvm.org/):
                ```bash
                wget https://apt.llvm.org/llvm.sh
                chmod +x llvm.sh
                sudo ./llvm.sh 22
                ```
            * Fedora/RHEL: `sudo dnf install clang`
    * Before building, set `CC` and `CXX` to the Clang executables you verified above so CMake uses them.
        Choose one:

        ```bash
        # Unversioned executables
        export CC=clang
        export CXX=clang++
        ```

        ```bash
        # Versioned executables: set this to your installed major version (17 or newer)
        CLANG_VERSION=22
        export CC="clang-${CLANG_VERSION}"
        export CXX="clang++-${CLANG_VERSION}"
        ```
        NOTE: If CMake already configured the build directory with different compilers, delete the build directory before trying again because CMake caches its compiler selection.
    * **SuperDex Studio** and **SuperDex Physics Debugger** source builds (`uv sync --extra gui`) additionally require:
        * Ubuntu/Debian: `sudo apt install libx11-dev libxext-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libxcb1-dev libgl1-mesa-dev`
        * Fedora/RHEL: `sudo dnf install libX11-devel libXext-devel libXrandr-devel libXinerama-devel libXcursor-devel libXi-devel libxcb-devel mesa-libGL-devel`
* macOS:
    * Install **Xcode Command Line Tools** for Xcode 26 or newer: `xcode-select --install`
    * Check "Software Update" for the latest version if already installed.
* Windows:
    * [MSVC Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) for VS2022 or newer (Visual Studio IDE is not necessary)
        * Select the **Desktop development with C++** workload and include the **C++ Clang tools for Windows** (ClangCL) component.
    * Or install from terminal:
        * `winget install --exact --id Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --add Microsoft.VisualStudio.Component.VC.Llvm.Clang --add Microsoft.VisualStudio.Component.VC.Llvm.ClangToolset --add Microsoft.VisualStudio.Component.Windows11SDK.26100"`
* Ensure `cmake`, `ninja`, `uv`, and the platform-specific Clang compiler are on PATH: `clang` and `clang++` on Linux/macOS (versioned names such as `clang-17` and `clang++-17` are also supported), or `clang-cl` on Windows.
* Other compilers such as GCC and MSVC are not officially supported or covered by CI, but can be used "at your own risk".

### Building from Source (Python)

1. Get the source code and install pre-requisites (see above)
2. Windows only: From the Start menu, launch the `x64 Native Tools Command Prompt` matching your installed Visual Studio version (e.g. `x64 Native Tools Command Prompt for VS 2022`).
3. `cd` into the `project_superdex` source directory
4. Build: `uv sync --extra gui -v`
5. Run:
    * Physics example: `uv run superdex_physics/examples/example_tendon_comparison.py`
    * Robotics example: `uv run superdex_robotics/examples/control/example_osc_jsc_control.py`
    * SuperDex Studio: `uv run superdex-studio`

Flags for `uv sync` are:

| uv sync flags (additive) | Build targets |
| :-- | :-- |
| none | build tools only |
| `--extra core` | physics, robotics, lab |
| `--extra gui` | core + physics-debugger, studio, mesh-cli |
| `--extra double` | core + physics-fp64, robotics-fp64 |
| `--all-extras` | everything |

NOTE: Running the examples above requires `--extra gui`.

NOTE: `--extra double` builds the double-precision bindings, but single precision is still the default at runtime. To run Python examples in double precision, set the environment variable `SUPERDEX_PRECISION=double`.

### Building from Source (C++)

Core modules such as SuperDex Physics and SuperDex Robotics are written in C++ and exposed to Python. Full C++ examples will be shared in the future. Users who are interested in calling C++ directly can get started via `CMake`.

1. Get the source code and install pre-requisites (see above)
2. Windows only: From the Start menu, launch the `x64 Native Tools Command Prompt` matching your installed Visual Studio version (e.g. `x64 Native Tools Command Prompt for VS 2022`).
3. `cd` into the `project_superdex` source directory
4. Configure lean Release build: `cmake -B build -DCMAKE_BUILD_TYPE=Release -DMOCHI_BUILD_DEBUGGER=OFF -DMOCHI_USE_PYBIND=OFF -G Ninja`
5. Build: `cmake --build build --parallel`

NOTE: To build the C++ libraries in double precision, add `-DMOCHI_USE_DOUBLE_PRECISION=ON` when configuring CMake. Otherwise, single precision will be used by default.

---
## Documentation

Documentation, getting started guides, and examples are available for all modules.

* [SuperDex Physics](https://projectsuperdex.com/physics/docs/overview/)
* [SuperDex Robotics](https://projectsuperdex.com/robotics/docs/overview/)
* [SuperDex Studio](https://projectsuperdex.com/studio/docs/overview/)
* [SuperDex Lab](https://projectsuperdex.com/lab/docs/overview/)

---

## How to Contribute
We welcome contributions! Go to [CONTRIBUTING](/CONTRIBUTING.md) and our [CODE OF CONDUCT](/CODE_OF_CONDUCT.md) for how to get started.

## License
First-party SuperDex source code is licensed under [Apache 2.0](/LICENSE). Assets and documentation are licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) except where otherwise noted. Third-party and derived code and assets retain their own terms, recorded in their accompanying `LICENSE` and `NOTICE` files. `superdex_mesh_cli` is an optional, standalone mesh processing CLI tool released under GPLv3 due to use of OCCT and CGAL.

Third-party license families represented in this repository include Apache-2.0, BSD-2-Clause, BSD-3-Clause, BSL-1.0, CC-BY-4.0, CC0-1.0, GPL-3.0-or-later, LGPL-2.1 with the OCCT exception, LGPL-3.0-or-later, MIT, MPL-2.0, SIL OFL-1.1, HDF5, libpng, public-domain, zlib, Zstandard, and component-specific commercial or asset terms. The accompanying license and notice files are authoritative for each component or asset.

Certain third party dependencies and third party assets in this repo are licensed for non-commercial/academic uses only. You must ensure that you, as the user, access and use such third party dependencies and assets in compliance with their respective licenses.

---

# Citation

If you use Project SuperDex in your research you may cite:

```
@misc{mochi2026,
  author        = {{The Mochi Team}},
  title         = {An Implicit Physics Engine for Contact-Rich Simulation},
  year          = {2026},
  eprint        = {},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  doi           = {},
  url           = {},
}
```
