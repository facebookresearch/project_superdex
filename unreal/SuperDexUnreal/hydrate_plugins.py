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

"""Hydrate this SuperDexUnreal project's plugins with public native dependencies.

The plugins in Plugins/ are source plugins. Before Unreal can compile them, SuperDexPhysics and
SuperDexRobotics need the native SuperDex libraries (mochi_physics, marl, superdex_robotics,
superdex_recorder) and their headers in their ThirdParty folders, and TeleopNodeZenoh needs the
zenoh-pico and nanopb sources its modules compile. This script puts them there, so the project can
be opened and run as is. Native builds use Release and Ninja.

Usage
-----
Run from anywhere; paths are resolved relative to this file.

    python hydrate_plugins.py [--build-dir DIR] [--ue-root DIR]

Mochi is built from superdex_physics/libraries/mochi; set SUPERDEX_MOCHI_ROOT to use Mochi
sources elsewhere.

Every run configures CMake, builds incrementally, stages libraries and headers, and validates
the dependencies. Unchanged native libraries do not need to be rebuilt.

Building needs CMake 3.25+ and Ninja, plus clang-cl 17+ on Windows (it finds the installed MSVC
toolchain by itself). Mochi's CMake rejects other compilers. On Linux the libraries are built with
the engine's own clang toolchain against the engine's bundled libc++, as Unreal's modules are, so
the script needs the engine: pass --ue-root or set UE_ROOT. On Windows use `py -3` in place of
`python`.

Examples
--------
    # First-time setup on Windows, then open SuperDexUnreal.uproject (UE 5.5). Unreal compiles
    # the plugin modules the first time the project opens.
    py -3 hydrate_plugins.py

    # Keep the CMake tree somewhere persistent:
    py -3 hydrate_plugins.py --build-dir C:\\build\\superdex-unreal

    # Linux, against an installed engine:
    python3 hydrate_plugins.py --ue-root ~/UnrealEngine-5.5

What it writes (all ignored by .gitignore)
------------------------------------------
    Plugins/SuperDexPhysics/ThirdParty/mochi_physics/lib/<Platform>/         mochi_physics, marl
    Plugins/SuperDexRobotics/ThirdParty/superdex_robotics/lib/<Platform>/    superdex_robotics
    Plugins/SuperDexRobotics/ThirdParty/superdex_recorder/lib/Win64/         superdex_recorder_shared
    Plugins/*/Source/<Module>/ThirdParty/<library>/include                   only if missing or broken
    Plugins/TeleopNodePlugins/TeleopNodeZenoh/Source/TeleopNodeZenoh/Private/ZenohPico      zenoh-pico sources
    Plugins/TeleopNodePlugins/TeleopNodeZenoh/Source/TeleopNodeZenoh/Private/Nanopb   nanopb's C runtime

A source checkout may already provide the include trees as links into the library sources;
those are left alone. A missing or dangling tree is replaced with a copy of the headers.
TeleopNodeZenoh's sources are placed by configuring its own CMake project (see its README), which
downloads zenoh-pico and nanopb at pinned commits.
Windows (Win64) is the supported platform; Linux is for development.
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


logger: logging.Logger = logging.getLogger(__name__)

# Normalised but not symlink-resolved, so a tree assembled out of links reports its own layout.
PROJECT_DIR = Path(os.path.abspath(__file__)).parent

Platform = Literal["Win64", "Linux"]


@dataclass(frozen=True)
class Lib:
    """A native library. `stem` is the CMake target's output name and what Build.cs expects."""

    stem: str
    # Windows only: the import library is staged as `<stem>.dll.imp.lib`, the name Build.cs links.
    needs_import_lib: bool = False

    def staged_names(self, plat: Platform) -> list[tuple[str, str]]:
        """(build output name, staged name) pairs for one platform."""
        if plat == "Win64":
            names = [(f"{self.stem}.dll", f"{self.stem}.dll")]
            if self.needs_import_lib:
                names.append((f"{self.stem}.lib", f"{self.stem}.dll.imp.lib"))
            return names
        return [(f"lib{self.stem}.so", f"lib{self.stem}.so")]


@dataclass(frozen=True)
class LibDir:
    """Libraries staged into one folder, relative to the plugin root, for one platform."""

    dest: str
    libs: tuple[Lib, ...]


@dataclass(frozen=True)
class HeaderTree:
    """An include tree the plugin's Build.cs adds to its include paths."""

    # Relative to the plugin root.
    dest: str
    # Which source tree it comes from, and the path within it.
    root: Literal["mochi", "superdex"]
    source: str


@dataclass(frozen=True)
class CodePlugin:
    name: str
    lib_dirs: dict[str, tuple[LibDir, ...]]
    headers: tuple[HeaderTree, ...]

    def lib_dirs_for(self, plat: Platform) -> tuple[LibDir, ...]:
        return self.lib_dirs.get(plat, ())


PHYSICS = CodePlugin(
    name="SuperDexPhysics",
    lib_dirs={
        "Win64": (
            LibDir(
                "ThirdParty/mochi_physics/lib/Win64",
                (Lib("mochi_physics", needs_import_lib=True), Lib("marl")),
            ),
        ),
        "Linux": (
            LibDir(
                "ThirdParty/mochi_physics/lib/Linux",
                (Lib("mochi_physics"), Lib("marl")),
            ),
        ),
    },
    headers=(
        HeaderTree(
            "Source/SuperDexPhysics/ThirdParty/mochi_core/include",
            "mochi",
            "mochi_core/include",
        ),
        HeaderTree(
            "Source/SuperDexPhysics/ThirdParty/mochi_physics/include",
            "mochi",
            "mochi_physics/include",
        ),
        # SuperDexPhysics publishes this path and SuperDexRobotics compiles against it.
        HeaderTree(
            "Source/SuperDexPhysics/ThirdParty/superdex_physics/include",
            "superdex",
            "superdex_physics/include",
        ),
    ),
)

ROBOTICS = CodePlugin(
    name="SuperDexRobotics",
    lib_dirs={
        "Win64": (
            LibDir(
                "ThirdParty/superdex_robotics/lib/Win64",
                (Lib("superdex_robotics", needs_import_lib=True),),
            ),
            # SuperDexRecorder links this on Win64 only.
            LibDir(
                "ThirdParty/superdex_recorder/lib/Win64",
                (Lib("superdex_recorder_shared", needs_import_lib=True),),
            ),
        ),
        "Linux": (
            LibDir(
                "ThirdParty/superdex_robotics/lib/Linux", (Lib("superdex_robotics"),)
            ),
        ),
    },
    headers=(
        HeaderTree(
            "Source/SuperDexRobotics/ThirdParty/superdex_robotics/include",
            "superdex",
            "superdex_robotics/include",
        ),
        HeaderTree(
            "Source/SuperDexRecorder/ThirdParty/superdex_recorder/include",
            "superdex",
            "superdex_recorder/include",
        ),
    ),
)

# Physics first: Robotics links the import library staged into the Physics tree.
CODE_PLUGINS = (PHYSICS, ROBOTICS)


@dataclass(frozen=True)
class SourcePlugin:
    """A plugin whose own CMake project places the third-party sources its modules compile.
    Configuring that project is all it takes; nothing is built."""

    name: str
    # Relative to Plugins/.
    path: str
    # Trees the CMake project places, relative to the plugin root.
    placed: tuple[str, ...]


TELEOP_NODE_ZENOH = SourcePlugin(
    name="TeleopNodeZenoh",
    path="TeleopNodePlugins/TeleopNodeZenoh",
    placed=(
        "Source/TeleopNodeZenoh/Private/ZenohPico",
        "Source/TeleopNodeZenoh/Private/Nanopb",
    ),
)

SOURCE_PLUGINS = (TELEOP_NODE_ZENOH,)

# Header directories that never ship with the libraries, at any depth.
STRIPPED_DIR_NAMES = frozenset({"internal", "experimental"})


def log(msg: str, *args: object) -> None:
    logger.info(msg, *args)


def run(cmd: list[str | Path]) -> None:
    argv = [str(c) for c in cmd]
    log("$ %s", subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv))
    subprocess.run(argv, check=True)


def host_platform() -> Platform:
    system = platform.system()
    if system == "Windows":
        return "Win64"
    if system == "Linux":
        return "Linux"
    raise RuntimeError(f"Unsupported host platform: {system}. Use Windows or Linux.")


def cmake_targets(plat: Platform) -> tuple[str, ...]:
    targets = ["mochi_physics", "marl", "superdex_robotics"]
    if plat == "Win64":
        targets.append("superdex_recorder_shared")
    return tuple(targets)


# --- Roots ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Roots:
    superdex: Path
    mochi: Path

    def source_of(self, tree: HeaderTree) -> Path:
        return (self.mochi if tree.root == "mochi" else self.superdex) / tree.source

    def plugin_dir(self, plugin: CodePlugin) -> Path:
        return PROJECT_DIR / "Plugins" / plugin.name


def source_plugin_dir(plugin: SourcePlugin) -> Path:
    return PROJECT_DIR / "Plugins" / plugin.path


def _abspath(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _is_superdex_root(path: Path) -> bool:
    return (path / "CMakeLists.txt").is_file() and (path / "superdex_physics").is_dir()


def _is_mochi_root(path: Path) -> bool:
    return (path / "CMakeLists.txt").is_file() and (path / "mochi_core").is_dir()


def resolve_roots() -> Roots:
    """The project sits at <superdex>/unreal/SuperDexUnreal and Mochi at
    <superdex>/superdex_physics/libraries/mochi. Set SUPERDEX_MOCHI_ROOT to build against Mochi
    sources somewhere else."""
    superdex = PROJECT_DIR.parent.parent
    if not _is_superdex_root(superdex):
        raise RuntimeError(
            f"{superdex} is not the SuperDex source root. This script must live in a SuperDex checkout."
        )
    tried = [superdex / "superdex_physics" / "libraries" / "mochi"]
    override = os.environ.get("SUPERDEX_MOCHI_ROOT")
    if override:
        tried.insert(0, _abspath(Path(override)))
    mochi = next((c for c in tried if _is_mochi_root(c)), None)
    if mochi is None:
        raise RuntimeError(
            "Could not locate the mochi sources. Tried:\n  "
            + "\n  ".join(map(str, tried))
        )
    return Roots(superdex=superdex, mochi=mochi)


# --- Build ------------------------------------------------------------------------------


def linux_toolchain_args(ue_root: Path) -> list[str]:
    """CMake arguments that build like UnrealBuildTool does on Linux: the engine's bundled clang
    and sysroot, the engine's libc++ (the plugins call mochi's C++ API, so both sides must agree
    on std::), linked statically so the libraries carry no C++ runtime of their own."""
    sdks = ue_root / "Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64"
    toolchains = sorted(sdks.glob("*clang*/x86_64-unknown-linux-gnu"))
    libcxx = ue_root / "Engine/Source/ThirdParty/Unix/LibCxx"
    if not toolchains or not (libcxx / "include/c++/v1").is_dir():
        raise RuntimeError(
            f"{ue_root} does not look like an Unreal Engine 5.5 install: expected the clang "
            f"toolchain under {sdks} and libc++ under {libcxx}."
        )
    toolchain = toolchains[-1]
    libcxx_lib = libcxx / "lib/Unix/x86_64-unknown-linux-gnu"
    cxx_flags = [
        "-nostdinc++",
        f"-isystem{libcxx / 'include'}",
        f"-isystem{libcxx / 'include/c++/v1'}",
        # Mochi turns off std::pmr, which the engine's libc++ lacks.
        "-DMOCHI_UNREAL_LINUX=1",
        # The engine's libc++ gates C++20 ranges and format behind this.
        "-D_LIBCPP_ENABLE_EXPERIMENTAL",
    ]
    link_flags = f"-fuse-ld=lld -nostdlib++ -static-libgcc -L{libcxx_lib}"
    return [
        f"-DCMAKE_C_COMPILER={toolchain / 'bin/clang'}",
        f"-DCMAKE_CXX_COMPILER={toolchain / 'bin/clang++'}",
        f"-DCMAKE_SYSROOT={toolchain}",
        f"-DCMAKE_CXX_FLAGS={' '.join(cxx_flags)}",
        f"-DCMAKE_SHARED_LINKER_FLAGS={link_flags}",
        f"-DCMAKE_EXE_LINKER_FLAGS={link_flags}",
        "-DCMAKE_CXX_STANDARD_LIBRARIES=-l:libc++.a -l:libc++abi.a -lpthread",
        # Unversioned sonames, so the libraries find each other under the names staged into
        # the plugins (libmarl.so, not libmarl.so.1).
        "-DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON",
    ]


def build_natives(
    roots: Roots,
    build_dir: Path,
    plat: Platform,
    ue_root: Path | None = None,
) -> None:
    if not shutil.which("cmake"):
        raise RuntimeError("cmake not found on PATH (CMake 3.25 or newer is required).")
    log("Building the native libraries with CMake into %s", build_dir)
    configure: list[str | Path] = [
        "cmake",
        "-S",
        roots.superdex,
        "-B",
        build_dir,
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DSUPERDEX_MOCHI_SOURCE_DIR={roots.mochi}",
        "-DMOCHI_BUILD_SHARED=ON",
        "-DSUPERDEX_BUILD_ROBOTICS=ON",
        "-DMOCHI_USE_TRACY=OFF",
        "-DMOCHI_USE_PYBIND=OFF",
        "-DMOCHI_BUILD_DEBUGGER=OFF",
        "-DMOCHI_BUILD_TESTS=OFF",
        # MOCHI_USE_HDF5 stays ON (the default): the plugins read and write .h5 through the DLL.
    ]
    if plat == "Win64":
        # Without this the DLLs disagree with UnrealBuildTool on struct packing. Win64 only:
        # mochi rejects it with anything but MSVC or clang-cl.
        configure.append("-DMOCHI_UNREAL_ABI=ON")
        configure.append("-DSUPERDEX_BUILD_RECORDER=ON")
    else:
        if ue_root is None:
            raise RuntimeError(
                "Linux builds compile against the engine's libc++: pass --ue-root or set UE_ROOT."
            )
        configure += linux_toolchain_args(ue_root)
    run(configure)
    run(
        [
            "cmake",
            "--build",
            build_dir,
            "--config",
            "Release",
            "--parallel",
            str(os.cpu_count() or 1),
            "--target",
            *cmake_targets(plat),
        ]
    )


def _find_built(build_dir: Path, name: str) -> tuple[Path | None, str | None]:
    """Find the Release output called `name`, rejecting ambiguous candidates."""
    hits = sorted(h for h in build_dir.rglob(name) if h.is_file())
    if len(hits) > 1:
        configured = [
            h
            for h in hits
            if "release" in {p.casefold() for p in h.relative_to(build_dir).parts[:-1]}
        ]
        hits = configured if len(configured) == 1 else hits
    if not hits:
        return None, f"{name}: not found under {build_dir}"
    if len(hits) > 1:
        found = ", ".join(str(h.relative_to(build_dir)) for h in hits)
        return None, f"{name}: {len(hits)} candidates under {build_dir}: {found}"
    return hits[0], None


def _replace_file(source: Path, dest: Path) -> None:
    """Copy `source` to `dest`, removing whatever was at `dest` first. Removing (rather than
    writing through) matters: internally `dest` can be a link into a build cache."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    shutil.copy2(source, dest)


def stage_built_libs(roots: Roots, build_dir: Path, plat: Platform) -> list[str]:
    problems: list[str] = []
    for plugin in CODE_PLUGINS:
        for lib_dir in plugin.lib_dirs_for(plat):
            dest_dir = roots.plugin_dir(plugin) / lib_dir.dest
            for lib in lib_dir.libs:
                for built_name, staged_name in lib.staged_names(plat):
                    built, problem = _find_built(build_dir, built_name)
                    if problem:
                        problems.append(f"{plugin.name}: {problem}")
                        continue
                    assert built is not None
                    _replace_file(built, dest_dir / staged_name)
                    log("  %s/%s/%s", plugin.name, lib_dir.dest, staged_name)
    return problems


# --- Headers ----------------------------------------------------------------------------


def _tree_usable(path: Path) -> bool:
    """True for a directory (or a link to one) that resolves and holds at least one file."""
    try:
        return path.is_dir() and any(p.is_file() for p in path.rglob("*"))
    except OSError:
        return False


def _replace_tree(source: Path, dest: Path) -> None:
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    elif dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        dest,
        symlinks=False,
        ignore=lambda _d, names: {n for n in names if n in STRIPPED_DIR_NAMES},
    )


def stage_headers(roots: Roots) -> list[str]:
    problems: list[str] = []
    for plugin in CODE_PLUGINS:
        for tree in plugin.headers:
            dest = roots.plugin_dir(plugin) / tree.dest
            if _tree_usable(dest):
                continue
            source = roots.source_of(tree)
            if not source.is_dir():
                problems.append(f"{plugin.name}: header source not found: {source}")
                continue
            _replace_tree(source, dest)
            log("  %s/%s <- %s", plugin.name, tree.dest, source)
    return problems


# --- Plugin sources ---------------------------------------------------------------------


def compiler_args(plat: Platform, ue_root: Path | None) -> list[str]:
    """The compilers the native build uses, for CMake projects that only need to configure."""
    if plat == "Linux":
        if ue_root is None:
            raise RuntimeError(
                "Linux builds compile against the engine's libc++: pass --ue-root or set UE_ROOT."
            )
        return linux_toolchain_args(ue_root)
    # Mochi's CMake picks clang-cl for Ninja on Windows; do the same.
    clang_cl = shutil.which("clang-cl")
    if clang_cl is None:
        return []
    return [f"-DCMAKE_C_COMPILER={clang_cl}", f"-DCMAKE_CXX_COMPILER={clang_cl}"]


def place_plugin_sources(
    build_dir: Path, plat: Platform, ue_root: Path | None = None
) -> None:
    for plugin in SOURCE_PLUGINS:
        log("Placing the sources of %s", plugin.name)
        run(
            [
                "cmake",
                "-S",
                source_plugin_dir(plugin),
                "-B",
                build_dir / plugin.name,
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                *compiler_args(plat, ue_root),
            ]
        )


# --- Check ------------------------------------------------------------------------------


def missing(roots: Roots, plat: Platform) -> list[str]:
    """Everything the plugins need on `plat` that is not in place."""
    gaps: list[str] = []
    for plugin in CODE_PLUGINS:
        root = roots.plugin_dir(plugin)
        if not (root / f"{plugin.name}.uplugin").is_file():
            gaps.append(f"{plugin.name}: plugin not found at {root}")
            continue
        for lib_dir in plugin.lib_dirs_for(plat):
            for lib in lib_dir.libs:
                for _built, staged_name in lib.staged_names(plat):
                    path = root / lib_dir.dest / staged_name
                    if not path.is_file():
                        gaps.append(
                            f"{plugin.name}: missing {lib_dir.dest}/{staged_name}"
                        )
        for tree in plugin.headers:
            if not _tree_usable(root / tree.dest):
                gaps.append(f"{plugin.name}: missing or broken {tree.dest}")
    return gaps + _missing_plugin_sources()


def _missing_plugin_sources() -> list[str]:
    gaps: list[str] = []
    for plugin in SOURCE_PLUGINS:
        root = source_plugin_dir(plugin)
        if not (root / f"{plugin.name}.uplugin").is_file():
            gaps.append(f"{plugin.name}: plugin not found at {root}")
            continue
        for tree in plugin.placed:
            if not _tree_usable(root / tree):
                gaps.append(f"{plugin.name}: missing or empty {tree}")
    return gaps


# --- Driver -----------------------------------------------------------------------------


def hydrate(
    roots: Roots,
    *,
    plat: Platform,
    build_dir: Path,
    ue_root: Path | None = None,
) -> list[str]:
    """Stage public native dependencies into this project's plugins.

    Returns the problems found (empty on success). Also the entry point package_plugins.py uses.
    """
    log("project  : %s", PROJECT_DIR)
    log("superdex : %s", roots.superdex)
    log("mochi    : %s", roots.mochi)
    log("platform : %s", plat)
    build_natives(roots, build_dir, plat, ue_root)
    log("Staging libraries")
    problems = stage_built_libs(roots, build_dir, plat)
    log("Checking include trees")
    problems += stage_headers(roots)
    place_plugin_sources(build_dir, plat, ue_root)
    problems += [g for g in missing(roots, plat) if g not in problems]
    return problems


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ue-root",
        type=Path,
        default=os.environ.get("UE_ROOT") or None,
        help="Unreal Engine install, needed to build on Linux (default: $UE_ROOT)",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "superdex-unreal-build",
        help="CMake build directory (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="[hydrate] %(message)s", stream=sys.stdout
    )
    args = _parse_args(argv)
    try:
        roots = resolve_roots()
        plat = host_platform()
        problems = hydrate(
            roots,
            plat=plat,
            build_dir=_abspath(args.build_dir),
            ue_root=_abspath(args.ue_root) if args.ue_root else None,
        )
    except (
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        log("ERROR: %s", error)
        return 1
    if problems:
        log("FAILED with %d problem(s):", len(problems))
        for problem in problems:
            log("  %s", problem)
        return 1
    log("Done. Open %s", PROJECT_DIR / "SuperDexUnreal.uproject")
    return 0


if __name__ == "__main__":
    sys.exit(main())
