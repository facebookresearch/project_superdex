#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PEP 723 inline metadata: `uv run tools/build_wheels.py` resolves these into a cached
# environment. A plain `python tools/build_wheels.py` ignores them.
# /// script
# requires-python = ">=3.12"
# dependencies = ["build", "cibuildwheel", "scikit-build-core", "setuptools", "wheel"]
# ///

"""Build every published SuperDex wheel into one wheelhouse.

    uv run tools/build_wheels.py --output wheelhouse           # publishable wheels
    uv run tools/build_wheels.py --output wheelhouse --fast    # iterating
    uv run tools/check_wheels.py wheelhouse

Run from the repository root.

The default path builds the seven native distributions with `cibuildwheel` and the two pure
ones with `python -m build`. That is the one that produces publishable artifacts: correct
platform tags, and the per-platform wheel repair.

`--fast` trades that away for iteration speed -- see `build_fast`.

`--host` is for macOS and Windows environments where cibuildwheel cannot provision its own
interpreter. On macOS that step writes into `/Library/Frameworks` through `sudo installer`;
on Windows it downloads CPython through nuget. `--host` builds with the interpreter it is
handed but still applies each pyproject's `repair-wheel-command`, so the wheels stay
publishable -- see `build_host`.

This script holds no build configuration. Every knob -- Python versions, CMake arguments,
repair commands, repair exclusions -- lives in each distribution's `pyproject.toml`, so a
CI workflow that shells out to the same `cibuildwheel` command cannot drift away from what
is built locally. What lives here is the list of distributions, and the local-machine
plumbing (container engine selection) that has no business being checked in.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
import venv
from pathlib import Path

# Stable build selectors used by CI and internal wheel jobs.
NATIVE_DISTRIBUTIONS = (
    "superdex_python",
    "superdex_physics_fp64",
    "superdex_robotics",
    "superdex_robotics_fp64",
    "superdex_mesh_cli",
    "superdex_physics_debugger",
    "superdex_studio",
)
PURE_DISTRIBUTIONS = (
    "superdex_lab",
    "superdex_meta",
)
ALL_DISTRIBUTIONS = NATIVE_DISTRIBUTIONS + PURE_DISTRIBUTIONS

DISTRIBUTION_DIRECTORIES = {
    "superdex_python": Path("superdex_physics/wheels/superdex-physics"),
    "superdex_physics_fp64": Path("superdex_physics/wheels/superdex-physics-fp64"),
    "superdex_robotics": Path("superdex_robotics/wheels/superdex-robotics"),
    "superdex_robotics_fp64": Path("superdex_robotics/wheels/superdex-robotics-fp64"),
    "superdex_mesh_cli": Path("superdex_mesh_cli"),
    "superdex_physics_debugger": Path("superdex_physics_debugger"),
    "superdex_studio": Path("superdex_studio"),
    "superdex_lab": Path("superdex_lab"),
    "superdex_meta": Path("superdex_meta"),
}

# The FP64 payloads. Dropping both halves the native build.
FP64_DISTRIBUTIONS = ("superdex_physics_fp64", "superdex_robotics_fp64")

# The distributions that build the GPL-isolated mesh helper, and so need CGAL, Boost and OCCT.
# The studio is here because `SUPERDEX_BUILD_STUDIO=ON` builds the helper even though it ships
# it as a dependency rather than in its own wheel.
MESH_CLI_DISTRIBUTIONS = ("superdex_mesh_cli", "superdex_studio")

# The distributions that link a full windowing stack. On Linux that means the X11 extension
# libraries (Xrandr, Xinerama, Xcursor, Xi), which the manylinux image does not carry.
RENDERER_DISTRIBUTIONS = ("superdex_physics_debugger", "superdex_studio")

# What a platform does not produce, and why. An entry is either a gap, where the wheel is meant
# to ship from here and does not yet, or a duplicate, where another platform already produces an
# identical artifact. Emptying a gap entry is how that wheel starts shipping.
SKIPPED_DISTRIBUTIONS = {
    "Darwin": (
        PURE_DISTRIBUTIONS,
        "they declare `setuptools>=77` for their PEP 639 metadata, only 70.0.0 is vendored, "
        "and a --host build has no network to resolve a newer one; Linux publishes them and "
        "they are `py3-none-any`, so nothing is lost",
    ),
    "Linux": (
        RENDERER_DISTRIBUTIONS,
        "they need the X11 extension libraries (Xrandr, Xinerama, Xcursor, Xi), which the "
        "manylinux image does not carry, so their `before-all` has to yum install them and "
        "that needs a container with a route out -- pass `--container-network`",
    ),
    "Windows": (
        PURE_DISTRIBUTIONS,
        "they are `py3-none-any` and Linux already publishes them in about 90 seconds; "
        "building them here means two `pip install setuptools>=77` runs into fresh virtual "
        "environments, which measured 18m48s -- longer than any compile on this platform -- "
        "for two files that are then discarded as duplicates of Linux's",
    ),
}

# Populated inside the project so cibuildwheel's copy carries it into the container. The
# name is covered by the `build*` rule already in .gitignore.
BUILD_REQUIREMENTS_DIR = "build-requirements"
CONTAINER_PROJECT_PATH = "/project"

# Same trick for the mesh helper's third-party sources. `build-*` is likewise gitignored.
MESH_CLI_SOURCES_DIR = "build-mesh-cli-sources"

# Driver-owned environment variables for staging superdex_mesh_cli's third-party sources.
# The generic names alongside them are the `MOCHI_*_SOURCE_DIR` cache variables in
# arvr/libraries/mochi/cmake/paths.cmake. The driver exports them into
# `SKBUILD_CMAKE_DEFINE` only after staging, so unrelated ambient shell variables do not
# accidentally opt a wheel build into this path.
MESH_CLI_SOURCES = (
    ("SUPERDEX_MESH_CLI_CGAL_SOURCE_DIR", "MOCHI_CGAL_SOURCE_DIR", "cgal"),
    ("SUPERDEX_MESH_CLI_BOOST_SOURCE_DIR", "MOCHI_BOOST_SOURCE_DIR", "boost"),
    (
        "SUPERDEX_MESH_CLI_RAPIDJSON_SOURCE_DIR",
        "MOCHI_RAPIDJSON_SOURCE_DIR",
        "rapidjson",
    ),
    ("SUPERDEX_MESH_CLI_OCCT_SOURCE_DIR", "MOCHI_OCCT_SOURCE_DIR", "occt"),
)

# scikit-build-core reads `[tool.scikit-build.cmake] define` from this variable as a
# ';'-separated list of `NAME=value` pairs and forwards each as a `-D` on the configure line.
MESH_CLI_DEFINE_VARIABLE = "SKBUILD_CMAKE_DEFINE"

# scikit-build-core adds these at `get_requires_for_build_wheel` time when the environment has
# no new-enough cmake or ninja, as in the manylinux images. They are computed rather than
# declared, so they cannot be read out of `[build-system].requires`.
DYNAMIC_BUILD_REQUIREMENTS = ("cmake>=3.25", "ninja>=1.5")

# What `--host` exports before invoking the backend, reproducing cibuildwheel's `setup_python`.
# On macOS cibuildwheel computes these from the build identifier and never from the interpreter,
# so a different interpreter still produces an equivalent wheel: `_PYTHON_HOST_PLATFORM` stamps
# the platform tag. A native Windows x64 build already tags itself from its own sysconfig.
HOST_BUILD_ENVIRONMENT = {
    "Darwin": {
        "MACOSX_DEPLOYMENT_TARGET": "11.0",
        "_PYTHON_HOST_PLATFORM": "macosx-11.0-arm64",
        "ARCHFLAGS": "-arch arm64",
    },
    "Windows": {},
}

# The `[tool.cibuildwheel.<table>]` a build on each platform reads.
CIBUILDWHEEL_TABLE = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}

# The CMake generator every native wheel must name in its own `[tool.scikit-build.cmake]`
# args, and the compiler each platform must pin in its cibuildwheel environment. See
# `_check_declared_toolchain` for why neither can be stated once, centrally.
REQUIRED_GENERATOR = "-GNinja"
DECLARED_COMPILERS = {"linux": {"CC": "clang", "CXX": "clang++"}}

# cibuildwheel's own defaults, from `resources/defaults.toml`. A distribution that declares its
# own `repair-wheel-command` overrides these.
DEFAULT_REPAIR_COMMAND = {
    "Darwin": "delocate-wheel --require-archs {delocate_archs} -w {dest_dir} -v {wheel}",
    "Windows": "delvewheel repair -w {dest_dir} -v {wheel}",
}

# `{delocate_archs}` for an arm64 build. Passed to every template regardless of platform; the
# Windows ones do not reference it.
DELOCATE_ARCHS = "arm64"

# A repair command names a console script, and a vendored package installs none. Map the script
# back to the module behind it.
REPAIR_MODULES = {
    "delocate-wheel": "delocate.cmd.delocate_wheel",
    "delvewheel": "delvewheel",
}

# What has to be importable for each no-isolation mode. `--fast` builds the pure wheels too, so
# it needs setuptools and wheel; `--host` builds only native ones, which declare
# scikit-build-core alone as their backend.
FAST_REQUIREMENTS = ("build", "scikit_build_core", "setuptools", "wheel")
HOST_REQUIREMENTS = ("build", "scikit_build_core")

# `--no-isolation` because there is no network to build an isolated environment from; the
# backend comes from PYTHONPATH. `--skip-dependency-check` because `build` checks
# `[build-system].requires` through importlib.metadata, and a vendored tree ships no
# `.dist-info`, so every requirement reads as missing even though all of them import.
HOST_NO_ISOLATION_FLAGS = ("--no-isolation", "--skip-dependency-check")


########################################################################################


def repository_root() -> Path:
    """The tree to build, which is the one this script was invoked out of.

    Deliberately `absolute()` and not `resolve()`. A staged tree can reach this file
    through a symlink into another checkout, and resolving it would silently retarget the
    entire build -- sources, `source-dir = ".."`, and output directory -- at that other
    checkout instead of the stage the caller asked for.
    """

    return Path(__file__).absolute().parent.parent


def distribution_path(root: Path, name: str) -> Path:
    return root / DISTRIBUTION_DIRECTORIES[name]


def _check_repository_root(root: Path) -> None:
    """Reject a tree that is not the repo root, or that cibuildwheel cannot copy.

    Native pyprojects use the repository CMake project, so cibuildwheel has to be invoked
    from the root naming a distribution directory. Invoking it on `.` would build
    `superdex-dev` instead.
    """

    missing = [
        str(DISTRIBUTION_DIRECTORIES[name])
        for name in ALL_DISTRIBUTIONS
        if not distribution_path(root, name).is_dir()
    ]
    if missing:
        raise SystemExit(
            f"{root} does not look like a SuperDex repository root; "
            f"missing {', '.join(missing)}"
        )

    # And the other direction: the lists above are what CI builds, so a distribution missing from
    # them is never built and never published, with nothing to notice.
    registered = set(DISTRIBUTION_DIRECTORIES.values())
    distribution_roots = (
        root,
        root / "superdex_physics" / "wheels",
        root / "superdex_robotics" / "wheels",
    )
    unregistered = sorted(
        str(path.relative_to(root))
        for distribution_root in distribution_roots
        for path in distribution_root.iterdir()
        if path.is_dir()
        and (path / "pyproject.toml").is_file()
        and path.relative_to(root) not in registered
    )
    if unregistered:
        raise SystemExit(
            f"{', '.join(unregistered)} look like distributions but are not in "
            "NATIVE_DISTRIBUTIONS or PURE_DISTRIBUTIONS, so nothing would build them. "
            "Add them there (and to tools/check_wheels.py)."
        )


def _check_no_symlinked_sources(root: Path) -> None:
    """Fail early on a symlink-staged tree, which the container build cannot see.

    cibuildwheel copies the project into its container with `tar`, which stores symlinks as
    symlinks. Links pointing outside the container then dangle, and the failure surfaces
    much later as a confusing CMake or import error.
    """

    if platform.system() != "Linux":
        return
    symlinked = [
        str(DISTRIBUTION_DIRECTORIES[name])
        for name in ALL_DISTRIBUTIONS
        if distribution_path(root, name).is_symlink()
    ]
    if (root / "CMakeLists.txt").is_symlink():
        symlinked.insert(0, "CMakeLists.txt")
    if symlinked:
        raise SystemExit(
            f"{root} is staged with symlinks ({', '.join(symlinked)}). The Linux build "
            "runs in a container and copies the tree with tar, so symlinks arrive "
            "dangling. Re-stage with real files (internally: `skip_it.py --copy`)."
        )


########################################################################################


def _container_engine() -> str | None:
    """Pick a container engine for the Linux build, if one is not already chosen.

    cibuildwheel defaults to Docker. Hosts that only have Podman need to be told, and that
    is a property of the machine, not of the wheels, so it does not belong in a pyproject.
    """

    if platform.system() != "Linux" or os.environ.get("CIBW_CONTAINER_ENGINE"):
        return None
    if shutil.which("docker"):
        return None
    if shutil.which("podman"):
        return "podman"
    raise SystemExit(
        "The Linux wheel build needs a container engine, but neither docker nor podman "
        "is on PATH."
    )


def _build_environment() -> dict[str, str]:
    environment = dict(os.environ)
    engine = _container_engine()
    if engine is not None:
        print(f"[build_wheels] using container engine: {engine}")
        environment["CIBW_CONTAINER_ENGINE"] = engine
    return environment


########################################################################################


def build_system_requires(root: Path, names: list[str]) -> list[str]:
    requirements = set()
    for name in names:
        pyproject = tomllib.loads(
            (distribution_path(root, name) / "pyproject.toml").read_text(
                encoding="utf-8"
            )
        )
        requirements.update(pyproject.get("build-system", {}).get("requires", []))
    return sorted(requirements)


def _download_python(workspace: Path) -> Path:
    """A throwaway interpreter with pip, for hosts whose system Python has none."""

    venv.EnvBuilder(with_pip=True, clear=True).create(workspace)
    for candidate in (
        workspace / "bin" / "python",
        workspace / "Scripts" / "python.exe",
    ):
        if candidate.exists():
            return candidate
    raise SystemExit(f"could not find the interpreter in {workspace}")


def prepare_offline_build_requirements(root: Path, names: list[str]) -> list[str]:
    """Vendor the PEP 517 build requirements so the container needs no network.

    The manylinux container has no network of its own. Giving it one means handing it the
    host's proxy, which not every host will do -- a corporate proxy can happily serve the
    host and refuse the container. Downloading the requirements here, where the network
    already works, sidesteps that entirely and makes the container build reproducible: it
    installs exactly the wheels we fetched, from a directory, with the index switched off.

    Returns the in-container environment assignments that point pip at them.
    """

    requirements = build_system_requires(root, names) + list(DYNAMIC_BUILD_REQUIREMENTS)
    destination = root / BUILD_REQUIREMENTS_DIR
    destination.mkdir(parents=True, exist_ok=True)
    print(f"[build_wheels] vendoring build requirements: {', '.join(requirements)}")
    python = _download_python(root / f"{BUILD_REQUIREMENTS_DIR}-venv")
    _run(
        [
            str(python),
            "-m",
            "pip",
            "--disable-pip-version-check",
            "download",
            "--only-binary=:all:",
            "--dest",
            str(destination),
            *requirements,
        ],
        cwd=root,
        environment=dict(os.environ),
    )
    find_links = f"{CONTAINER_PROJECT_PATH}/{BUILD_REQUIREMENTS_DIR}"
    return ["PIP_NO_INDEX=1", f"PIP_FIND_LINKS={find_links}"]


def _mesh_cli_source_directories() -> list[tuple[str, str, Path]] | None:
    """The configured mesh-helper source overrides, if the caller provided them.

    These are a set: mixing a local CGAL or Boost tree with a fetched OCCT tree is legal in
    CMake, but it makes the build origin depend on whichever subset happened to be exported
    into the environment. Accept either every SuperDex-owned variable above or none.
    """

    values = {
        input_variable: os.environ.get(input_variable)
        for input_variable, _, _ in MESH_CLI_SOURCES
    }
    if not any(values.values()):
        return None

    missing = [input_variable for input_variable, value in values.items() if not value]
    if missing:
        expected = ", ".join(
            input_variable for input_variable, _, _ in MESH_CLI_SOURCES
        )
        raise SystemExit(
            f"mesh helper source overrides are partial: set {expected} together, "
            f"or leave them all unset. Missing: {', '.join(missing)}"
        )

    directories = []
    invalid = []
    for input_variable, output_variable, name in MESH_CLI_SOURCES:
        value = values[input_variable]
        if value is None:
            raise AssertionError(f"{input_variable} unexpectedly unset")
        directory = Path(value)
        if not directory.is_dir():
            invalid.append(f"{input_variable}={directory}")
            continue
        directories.append((output_variable, name, directory))

    if invalid:
        raise SystemExit(
            "mesh helper source overrides must point at existing directories: "
            + ", ".join(invalid)
        )
    return directories


def stage_mesh_cli_sources(root: Path, names: list[str]) -> bool:
    """Copy the mesh helper's third-party sources into the tree.

    The mesh helper is the only part of SuperDex with third-party sources that CMake fetches
    at configure time rather than finding vendored. When callers provide explicit source
    directories through the `SUPERDEX_MESH_CLI_*_SOURCE_DIR` variables, copy those into
    the tree so the build can stay offline with no CMake change.

    True when everything is staged and the corresponding `MOCHI_*_SOURCE_DIR` cache
    variables can be pointed at it. False when there is nothing to do, or no overrides
    were configured.
    """

    if not any(name in MESH_CLI_DISTRIBUTIONS for name in names):
        return False

    sources = _mesh_cli_source_directories()
    if sources is None:
        configured = ", ".join(
            input_variable for input_variable, _, _ in MESH_CLI_SOURCES
        )
        print(
            f"[build_wheels] no {configured} overrides set; the mesh helper's "
            "sources will have to be downloaded by CMake"
        )
        return False

    destination = root / MESH_CLI_SOURCES_DIR
    for _, name, source in sources:
        staged = destination / name
        if not staged.is_dir():
            print(f"[build_wheels] staging {source} -> {MESH_CLI_SOURCES_DIR}/{name}")
            # `symlinks=False` on purpose: cibuildwheel tars this tree into the container,
            # where a link pointing back out at the checkout would dangle.
            shutil.copytree(source, staged, symlinks=False)
    return True


def prepare_offline_mesh_cli_sources(root: Path, names: list[str]) -> list[str]:
    """The staged sources as an in-container assignment, for `CIBW_ENVIRONMENT_LINUX`.

    Empty when nothing was staged. A build that reaches CMake without this is not broken,
    but it will try to download and will only work with `--container-network`.

    The value is quoted because `CIBW_ENVIRONMENT_LINUX` is a space-joined list of shell
    assignments: unquoted, the ';' separators would terminate the command.
    """

    if not stage_mesh_cli_sources(root, names):
        return []
    defines = ";".join(
        f"{variable}={CONTAINER_PROJECT_PATH}/{MESH_CLI_SOURCES_DIR}/{name}"
        for _, variable, name in MESH_CLI_SOURCES
    )
    return [f'{MESH_CLI_DEFINE_VARIABLE}="{defines}"']


def host_mesh_cli_environment(root: Path, names: list[str]) -> dict[str, str]:
    """The same override for a `--host` build, as absolute paths on this filesystem.

    The container build can name a fixed path because cibuildwheel mounts the project at
    `/project`. A host build has no such mount, so the value has to carry the real
    location of the staged copies. No shell is involved here, so it needs no quoting.

    Empty when no source overrides were configured. In that case CMake falls back to its
    vendored or fetched copies, which is the right generic behavior for a host build with
    network access. Callers that need this path to stay offline must set the variables first.
    """

    requested = [name for name in names if name in MESH_CLI_DISTRIBUTIONS]
    if not requested:
        return {}
    if not stage_mesh_cli_sources(root, names):
        return {}
    destination = root / MESH_CLI_SOURCES_DIR
    return {
        MESH_CLI_DEFINE_VARIABLE: ";".join(
            f"{variable}={destination / name}" for _, variable, name in MESH_CLI_SOURCES
        )
    }


def _run(command: list[str], *, cwd: Path, environment: dict[str, str]) -> None:
    print(f"[build_wheels] $ {' '.join(command)}")
    subprocess.run(command, cwd=cwd, env=environment, check=True)


########################################################################################


def interpreter() -> str:
    r"""`sys.executable`, without Windows' extended-length path prefix.

    Some launchers invoke Python through a `\\?\`-prefixed path, and Python then reports that
    form verbatim in `sys.executable`. Anything that re-invokes it as a subprocess fails,
    because `CreateProcess` will not launch an image through the prefix. cibuildwheel does
    exactly that when it bootstraps each build environment, and the failure surfaces as a bare
    "The system cannot find the path specified." with no indication of which path.

    Dropping the prefix is safe here: it exists only to lift the 260-character path limit,
    and an interpreter this far under it resolves identically either way.
    """

    prefix = "\\\\?\\"
    if sys.executable.startswith(prefix):
        return sys.executable[len(prefix) :]
    return sys.executable


def cibuildwheel_command() -> list[str]:
    """How to invoke cibuildwheel for the native wheels.

    Prefer the module form. `pip install cibuildwheel` also drops a console script on PATH,
    but an interpreter that merely has the package importable has no such script -- which is
    the case wherever the dependency is vendored rather than installed. Only the module form
    finds those.
    """

    probe = subprocess.run(
        [interpreter(), "-c", "import cibuildwheel"], capture_output=True, check=False
    )
    if probe.returncode == 0:
        return [interpreter(), "-m", "cibuildwheel"]
    executable = shutil.which("cibuildwheel")
    if executable is not None:
        return [executable]
    raise SystemExit(
        "The native wheels need `cibuildwheel`, which is neither importable from "
        f"{sys.executable} nor on PATH. Install it with `pip install cibuildwheel` "
        "(or `uv tool install cibuildwheel`)."
    )


def build_frontend() -> list[str]:
    """How to invoke the PEP 517 frontend for the pure-Python wheels.

    Prefer the running interpreter, so the wheels are built by the Python the caller chose.
    Fall back to a standalone `pyproject-build`, which is how `pipx`/`uv tool` install it --
    common on machines whose system Python has no `pip`.
    """

    probe = subprocess.run(
        [interpreter(), "-c", "import build"], capture_output=True, check=False
    )
    if probe.returncode == 0:
        return [interpreter(), "-m", "build"]
    executable = shutil.which("pyproject-build")
    if executable is not None:
        return [executable]
    raise SystemExit(
        "The pure-Python wheels need the `build` frontend, which is neither importable "
        f"from {sys.executable} nor on PATH as `pyproject-build`. Install it with "
        "`pip install build` (or `uv tool install build`)."
    )


def build_native(
    name: str, *, root: Path, output: Path, env: dict[str, str], command: list[str]
) -> None:
    _run(
        [
            *command,
            "--output-dir",
            str(output),
            str(DISTRIBUTION_DIRECTORIES[name]),
        ],
        cwd=root,
        environment=env,
    )


def build_pure(
    name: str, *, root: Path, output: Path, env: dict[str, str], frontend: list[str]
) -> None:
    _run(
        [
            *frontend,
            "--wheel",
            "--outdir",
            str(output),
            str(DISTRIBUTION_DIRECTORIES[name]),
        ],
        cwd=root,
        environment=env,
    )


########################################################################################


def build_fast(name: str, *, root: Path, output: Path, env: dict[str, str]) -> None:
    """Build one distribution with this interpreter, reusing the CMake build tree.

    `--no-isolation` is what makes this fast, and not for the reason it looks like.
    scikit-build-core records the directory it was imported from and deletes
    `CMakeCache.txt` whenever that changes (see `cmake.py`, "New isolated environment ...
    clearing cache"). Under normal isolated builds that directory is a fresh temporary
    environment every single time, so every build re-runs the compiler and header probes
    -- minutes per distribution, before a single source file is compiled. Building from
    one stable environment keeps the cache, so a rebuild only compiles what changed.

    What this gives up: no container, so the Linux wheel is tagged for the host's glibc
    rather than manylinux, and no repair step. Fine for testing, not publishable.
    """

    _run(
        [
            interpreter(),
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(output),
            str(DISTRIBUTION_DIRECTORIES[name]),
        ],
        cwd=root,
        environment=env,
    )


def _check_no_isolation_requirements(flag: str, modules: tuple[str, ...]) -> None:
    """`--no-isolation` means the build backends have to be importable right here."""

    missing = [module for module in modules if importlib.util.find_spec(module) is None]
    if missing:
        raise SystemExit(
            f"{flag} needs {', '.join(missing)} importable from {sys.executable}. "
            f"Run this through `uv run tools/build_wheels.py {flag}`, which resolves them "
            "from the script's inline metadata, or install them yourself."
        )


########################################################################################


def _check_host_supported() -> None:
    """Reject a `--host` build that would produce something not worth publishing."""

    if platform.system() not in HOST_BUILD_ENVIRONMENT:
        raise SystemExit(
            f"--host is not supported on {platform.system()}. On Linux it would tag the "
            "wheel for the host's glibc instead of manylinux, which is the one thing the "
            "container build exists to avoid."
        )
    # The macOS entry above hardcodes arm64 in all three variables, so an Intel host would
    # silently produce a wheel tagged `macosx_11_0_arm64` that cannot run anywhere.
    if platform.system() == "Darwin" and platform.machine() != "arm64":
        raise SystemExit(
            f"--host on macOS assumes arm64, but this host is {platform.machine()}. The "
            "wheel would be tagged arm64 regardless of what was actually compiled."
        )


def host_environment() -> dict[str, str]:
    """The ambient environment plus what cibuildwheel would have exported.

    `setdefault`, not assignment, mirroring cibuildwheel: a caller who deliberately raised
    `MACOSX_DEPLOYMENT_TARGET` keeps their value.
    """

    environment = dict(os.environ)
    for variable, value in HOST_BUILD_ENVIRONMENT[platform.system()].items():
        environment.setdefault(variable, value)
    return environment


def _repair_program(program: str) -> list[str]:
    """How to run `program`: straight off PATH, or as the vendored module behind it."""

    if shutil.which(program) is not None:
        return [program]
    module = REPAIR_MODULES.get(program)
    if module is None:
        raise SystemExit(
            f"the repair command needs `{program}`, which is not on PATH and has no known "
            "vendored module. Add it to REPAIR_MODULES, or put it on PATH."
        )
    probe = subprocess.run(
        [interpreter(), "-c", f"import {module}"], capture_output=True, check=False
    )
    if probe.returncode != 0:
        raise SystemExit(
            f"the repair command needs `{program}`, which is neither on PATH nor "
            f"importable as `{module}` from {interpreter()}."
        )
    return [interpreter(), "-m", module]


def _pyproject(root: Path, name: str) -> dict:
    """This distribution's parsed `pyproject.toml`."""

    return tomllib.loads((distribution_path(root, name) / "pyproject.toml").read_text())


def _cibuildwheel_table(root: Path, name: str, table: str) -> dict:
    """This distribution's `[tool.cibuildwheel.<table>]` block."""

    return _pyproject(root, name).get("tool", {}).get("cibuildwheel", {}).get(table, {})


def _declared_cmake_args(root: Path, name: str) -> list[str]:
    """This distribution's `[tool.scikit-build.cmake] args`."""

    tool = _pyproject(root, name).get("tool", {})
    table = tool.get("scikit-build", {}).get("cmake", {})
    return [str(argument) for argument in table.get("args", [])]


def _check_declared_toolchain(root: Path, names: list[str], table: str) -> None:
    """Refuse to build a wheel that does not declare the toolchain we publish with.

    Two declarations, in two places, for two reasons.

    The generator, in `[tool.scikit-build.cmake] args`. CMake fixes it before it reads a
    line of any CMakeLists.txt, so no CMake file can choose it, and scikit-build-core has
    no shared configuration -- one `-GNinja` per pyproject is the only way to say it. Every
    build path reads it, cibuildwheel included.

    The compiler, in `[tool.cibuildwheel.<platform>.environment]`, on Linux alone. Mochi's
    `cmake/compiler.cmake` selects Clang when nothing else has, but inside the manylinux
    container that selection does not fire and the build falls through to the container's
    gcc. Windows is absent because clang-cl selection does work there once the generator is
    Ninja; macOS because the system AppleClang is supported and there is nothing to pick.

    Either omission is quiet: the wheel is produced, repaired and tagged exactly as it
    should be, built with the wrong toolchain, and the only tell is a line of CMake output
    half an hour in. Cheaper to refuse up front.
    """

    problems = [
        f"  {name}: {REQUIRED_GENERATOR} missing from [tool.scikit-build.cmake] args"
        for name in names
        if REQUIRED_GENERATOR not in _declared_cmake_args(root, name)
    ]
    for key, value in DECLARED_COMPILERS.get(table, {}).items():
        problems += [
            f"  {name}: {key} is {declared.get(key) or 'unset'}, expected {value}"
            for name in names
            for declared in [
                _cibuildwheel_table(root, name, table).get("environment", {})
            ]
            if declared.get(key) != value
        ]
    if problems:
        raise SystemExit(
            "these distributions do not declare the toolchain to build with:\n"
            + "\n".join(sorted(problems))
        )


def host_repair_command(
    root: Path, name: str, *, wheel: Path, dest_dir: Path
) -> list[str] | None:
    """The repair cibuildwheel would run for this distribution, expanded here.

    Read out of `[tool.cibuildwheel.<platform>] repair-wheel-command` rather than
    reimplemented, so the wheel this produces and the one GitHub's cibuildwheel produces are
    repaired by the same command string. An explicitly empty command means "do not repair",
    which is how cibuildwheel spells it too.

    The template is split into tokens *before* the paths go in, never after. `shlex.split`
    treats a backslash as an escape character, so splitting an already-expanded command turns
    `C:\\Users\\facebook\\...` into `C:Usersfacebook...` -- which is exactly how the first
    Windows run failed, after building the wheel perfectly well. Substituting per token also
    means a path containing spaces needs no quoting.
    """

    block = _cibuildwheel_table(root, name, CIBUILDWHEEL_TABLE[platform.system()])
    template = block.get(
        "repair-wheel-command", DEFAULT_REPAIR_COMMAND[platform.system()]
    )
    if not template.strip():
        return None
    program, *arguments = [
        token.format(
            wheel=str(wheel), dest_dir=str(dest_dir), delocate_archs=DELOCATE_ARCHS
        )
        for token in shlex.split(template)
    ]
    return [*_repair_program(program), *arguments]


def build_host(name: str, *, root: Path, output: Path, env: dict[str, str]) -> None:
    """Build one native distribution with this interpreter, then repair it.

    Everything cibuildwheel does on macOS except provisioning its own interpreter, which is
    the one step it cannot do here: it installs the python.org framework with `sudo
    installer`, and some locked-down workers have neither passwordless sudo nor a writable
    `/Library/Frameworks`.

    See `HOST_NO_ISOLATION_FLAGS` for why neither frontend flag is optional here.
    """

    with tempfile.TemporaryDirectory(prefix=f"{name}-raw-") as workspace:
        raw = Path(workspace)
        _run(
            [
                interpreter(),
                "-m",
                "build",
                "--wheel",
                *HOST_NO_ISOLATION_FLAGS,
                "--outdir",
                str(raw),
                str(DISTRIBUTION_DIRECTORIES[name]),
            ],
            cwd=root,
            environment=env,
        )
        built = sorted(raw.glob("*.whl"))
        if len(built) != 1:
            raise SystemExit(
                f"expected exactly one wheel from {name}, got {len(built)}: "
                f"{', '.join(wheel.name for wheel in built) or 'none'}"
            )
        repair = host_repair_command(root, name, wheel=built[0], dest_dir=output)
        if repair is None:
            print(f"[build_wheels] {name}: no repair command, using the wheel as built")
            shutil.move(str(built[0]), output / built[0].name)
            return
        _run(repair, cwd=root, environment=env)


def build_host_wheelhouse(
    native: list[str], pure: list[str], *, root: Path, output: Path
) -> int:
    """Build the native distributions. The pure ones never reach here.

    `SKIPPED_DISTRIBUTIONS` filters them out first, on every `--host` platform. This raises
    rather than skipping quietly, so that emptying that entry without also teaching `--host`
    how to build them fails loudly instead of producing a short wheelhouse.
    """

    if pure:
        raise SystemExit(
            f"--host cannot build {', '.join(pure)}: they need setuptools>=77 for their PEP "
            "639 metadata and only 70.0.0 is vendored, so building them means resolving one "
            "over the network. Linux publishes them instead; see `SKIPPED_DISTRIBUTIONS`."
        )
    environment = host_environment()
    environment.update(host_mesh_cli_environment(root, native))
    for name in native:
        build_host(name, root=root, output=output, env=environment)
    return _report(output)


########################################################################################


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the published SuperDex wheels into one wheelhouse."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("wheelhouse"),
        help="Directory to collect the built wheels into. Defaults to ./wheelhouse.",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=ALL_DISTRIBUTIONS,
        metavar="SELECTOR",
        help=(
            "Build just this distribution; repeatable. "
            f"One of: {', '.join(ALL_DISTRIBUTIONS)}."
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Empty the output directory first.",
    )
    parser.add_argument(
        "--skip-fp64",
        action="store_true",
        help=(
            "Leave out the two FP64 distributions, halving the native build. "
            "The umbrella `superdex` pins them exactly, so it will not install from the "
            "resulting wheelhouse -- ask for `superdex-lab` instead, which pulls robotics "
            "and physics."
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Build with this interpreter instead of cibuildwheel: no container, no wheel "
            "repair, and the CMake build tree is reused between runs. For iterating. The "
            "wheels it produces are testable but not publishable."
        ),
    )
    parser.add_argument(
        "--host",
        action="store_true",
        help=(
            "Build with this interpreter instead of cibuildwheel, but still apply the "
            "platform environment and the `repair-wheel-command` from each pyproject. "
            "macOS and Windows only, where cibuildwheel cannot provision its own "
            "interpreter. Unlike --fast, the wheels this produces are repaired and "
            "publishable."
        ),
    )
    parser.add_argument(
        "--container-network",
        action="store_true",
        help=(
            "Let the Linux container fetch what it needs itself -- build requirements from "
            "PyPI, and the mesh helper's CGAL/Boost/OCCT sources from their upstreams -- "
            "instead of staging both here first. Only works where the container has a "
            "usable index and route out."
        ),
    )
    return parser.parse_args()


class _NothingToBuild(Exception):
    """Every requested distribution is skipped on this platform. Not a failure."""


def _drop_skipped(
    selected: tuple[str, ...], container_network: bool
) -> tuple[str, ...]:
    """Remove what this platform does not produce, saying so."""

    entry = SKIPPED_DISTRIBUTIONS.get(platform.system())
    if entry is None:
        return selected
    skipped, reason = entry
    # The Linux entry is the one condition a flag can lift: the renderers are skipped only
    # because their `before-all` cannot reach a package mirror, which `--container-network`
    # fixes. Nothing else here is recoverable that way.
    if container_network and platform.system() == "Linux":
        return selected
    dropped = [name for name in selected if name in skipped]
    if dropped:
        print(f"[build_wheels] not building {', '.join(dropped)}: {reason}")
    return tuple(name for name in selected if name not in skipped)


def _select_distributions(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    """The native and pure distributions to build, after `--only` and `--skip-fp64`."""

    selected = tuple(args.only) if args.only else ALL_DISTRIBUTIONS
    if args.skip_fp64:
        selected = tuple(name for name in selected if name not in FP64_DISTRIBUTIONS)
        print(
            "[build_wheels] skipping fp64; `superdex` pins those distributions exactly "
            "and will not install from this wheelhouse -- use `superdex-lab`"
        )
    remaining = _drop_skipped(selected, args.container_network)
    if not remaining:
        # A normal outcome, not an error: CI emits one action per distribution on every
        # platform, so the ones a platform skips have to no-op rather than fail the leg.
        print("[build_wheels] nothing to build on this platform; producing no wheels")
        raise _NothingToBuild()
    return (
        [name for name in NATIVE_DISTRIBUTIONS if name in remaining],
        [name for name in PURE_DISTRIBUTIONS if name in remaining],
    )


def _prepare_output(root: Path, args: argparse.Namespace) -> Path:
    output = (root / args.output).resolve()
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


def build_fast_wheelhouse(
    native: list[str], pure: list[str], *, root: Path, output: Path
) -> int:
    environment = dict(os.environ)
    for name in [*native, *pure]:
        build_fast(name, root=root, output=output, env=environment)
    return _report(output)


def build_cibuildwheel_wheelhouse(
    native: list[str],
    pure: list[str],
    *,
    root: Path,
    output: Path,
    command: list[str],
    container_network: bool,
) -> int:
    environment = _build_environment()
    if native and platform.system() == "Linux" and not container_network:
        # This variable replaces the pyproject environment rather than extending it, so
        # carry the validated compiler declarations into the offline override too.
        assignments = [
            f"{key}={value}" for key, value in DECLARED_COMPILERS["linux"].items()
        ]
        assignments += prepare_offline_build_requirements(
            root, native
        ) + prepare_offline_mesh_cli_sources(root, native)
        environment["CIBW_ENVIRONMENT_LINUX"] = " ".join(assignments)
    frontend = build_frontend() if pure else []
    for name in native:
        build_native(name, root=root, output=output, env=environment, command=command)
    for name in pure:
        build_pure(name, root=root, output=output, env=environment, frontend=frontend)
    return _report(output)


def main() -> int:
    args = _parse_args()
    if args.fast and args.host:
        raise SystemExit(
            "--fast and --host are alternatives: --fast skips the repair step, --host "
            "performs it. Pick one."
        )
    root = repository_root()
    _check_repository_root(root)
    # Printed because getting this wrong is silent otherwise: everything still runs, just against
    # a different tree than the caller meant.
    print(f"[build_wheels] building from {root}")

    try:
        native, pure = _select_distributions(args)
    except _NothingToBuild:
        _prepare_output(root, args)
        return 0

    native_command: list[str] = []
    if args.fast:
        _check_no_isolation_requirements("--fast", FAST_REQUIREMENTS)
    elif args.host:
        _check_host_supported()
        _check_no_isolation_requirements("--host", HOST_REQUIREMENTS)
    elif native:
        _check_no_symlinked_sources(root)
        native_command = cibuildwheel_command()

    # Policed on every path. The generator half applies everywhere, since it lives in
    # `[tool.scikit-build.cmake]`; the compiler half only bites under cibuildwheel, which is
    # the only path that exports the environment it checks.
    if native:
        _check_declared_toolchain(root, native, CIBUILDWHEEL_TABLE[platform.system()])

    output = _prepare_output(root, args)

    if args.fast:
        return build_fast_wheelhouse(native, pure, root=root, output=output)
    if args.host:
        return build_host_wheelhouse(native, pure, root=root, output=output)
    return build_cibuildwheel_wheelhouse(
        native,
        pure,
        root=root,
        output=output,
        command=native_command,
        container_network=args.container_network,
    )


def _report(output: Path) -> int:
    wheels = sorted(output.glob("*.whl"))
    print(f"\n[build_wheels] {len(wheels)} wheel(s) in {output}:")
    for wheel in wheels:
        print(f"  {wheel.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
