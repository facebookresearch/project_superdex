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

"""Package the adjacent SuperDexUnreal project's public plugins, one zip per plugin.

Each archive unzips into the `Plugins/` directory of an Unreal Engine 5.5 C++ project, where
UnrealBuildTool compiles the plugin's modules from source the first time the project opens.
Code plugins carry the native libraries they need; content-only plugins carry their content.
Hydration builds the public native libraries in Release with Ninja.

Usage
-----
    python package_plugins.py [--out-dir DIR] [options]

1. HYDRATE   runs SuperDexUnreal/hydrate_plugins.py, which builds the native libraries with
             CMake and stages them into SuperDexUnreal/Plugins. Afterwards the SuperDexUnreal
             project can be opened and run as is. (--skip-hydrate reuses what is there.)
2. EXTRACT   copies each plugin found under SuperDexUnreal/Plugins (SuperDexPhysics,
             SuperDexRobotics, SuperDexAssets, SuperDexExamples, ...) without build output,
             internal/ and experimental/ folders, or internal tooling files. Links are
             replaced by the files they point to.
3. UPLUGIN   sets EnabledByDefault so a consumer does not have to enable the plugin by hand.
4. VALIDATE  every library the plugin's Build.cs links is present, no source file was dropped,
             no link survived, nothing excluded leaked.
5. ARCHIVE   writes <archive>.zip beside this script by default, with the plugin folder as
             its single top-level entry.

Building needs CMake 3.25+, Ninja and clang-cl 17+ on Windows; on Linux it needs the engine
(--ue-root or UE_ROOT), whose clang and libc++ it builds with. See hydrate_plugins.py. On Windows use `py -3` in place of `python`.

Examples
--------
    # Release archives for Windows, written beside this script:
    py -3 package_plugins.py

    # Re-zip after editing plugin content, without rebuilding the native libraries:
    py -3 package_plugins.py --out-dir dist --skip-hydrate

    # Keep the CMake tree between runs:
    py -3 package_plugins.py --out-dir dist --build-dir C:\\build\\superdex-unreal

    # Linux development archives:
    python3 package_plugins.py --out-dir dist --ue-root ~/UnrealEngine-5.5

Archives (Windows): superdex_physics_unrealplugin_windows.zip,
superdex_robotics_unrealplugin_windows.zip, superdex_assets_unrealplugin.zip,
superdex_examples_unrealplugin.zip. They are not independent: SuperDexRobotics links the
mochi_physics import library staged inside SuperDexPhysics, and the content plugins depend on
both, so install them side by side. Each .uplugin declares its dependencies.

Windows is the release platform. On Linux the same steps produce *_linux archives for
development use.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import types
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger: logging.Logger = logging.getLogger(__name__)

# Normalised but not symlink-resolved: a tree staged out of links back into a source checkout
# must report its own layout, not the checkout's.
SCRIPT_DIR = Path(os.path.abspath(__file__)).parent
PROJECT_DIR = SCRIPT_DIR / "SuperDexUnreal"

# Never shipped: build output, editor scratch and build-system files.
EXCLUDED_NAMES = frozenset(
    {
        "Binaries",
        "Intermediate",
        "Saved",
        "DerivedDataCache",
        ".vs",
        "BUCK",
        "TARGETS",
        "PACKAGE",
        "__pycache__",
        ".gitkeep",
    }
)
# `.umap.txt` files are text exports, not loadable maps; `_BuiltData` is regenerated lighting.
EXCLUDED_SUFFIXES = ("_BuiltData.uasset", ".umap.txt", ".pdb", ".pyc")
# Folders that stay private, at any depth (the published repository strips them too).
STRIPPED_DIR_NAMES = frozenset({"internal", "experimental"})

# Archive names. Plugins not listed get superdex_<name>_unrealplugin from their folder name.
ARCHIVE_STEMS = {
    "SuperDexPhysics": "superdex_physics_unrealplugin",
    "SuperDexRobotics": "superdex_robotics_unrealplugin",
    "SuperDexAssets": "superdex_assets_unrealplugin",
    "SuperDexExamples": "superdex_examples_unrealplugin",
    "SuperDexOpenXRPoseSource": "superdex_openxr_pose_source_unrealplugin",
    "TeleopNodeZenoh": "superdex_teleop_node_zenoh_unrealplugin",
}


def log(msg: str, *args: object) -> None:
    logger.info(msg, *args)


def load_hydrate(project_dir: Path) -> types.ModuleType:
    """The project's hydrate_plugins.py, which owns the library and header layout."""
    path = project_dir / "hydrate_plugins.py"
    spec = importlib.util.spec_from_file_location("superdex_hydrate_plugins", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class PluginSource:
    name: str
    root: Path
    is_code: bool

    def archive_name(self, plat: str) -> str:
        stem = ARCHIVE_STEMS.get(self.name) or "superdex_{}_unrealplugin".format(
            self.name.removeprefix("SuperDex").lower()
        )
        if not self.is_code:
            return f"{stem}.zip"
        return f"{stem}_{'windows' if plat == 'Win64' else 'linux'}.zip"


def discover_plugins(project_dir: Path) -> list[PluginSource]:
    """Every plugin under Plugins/, skipping private folders and build output."""
    plugins: list[PluginSource] = []
    plugins_dir = project_dir / "Plugins"
    for parent, dirs, files in os.walk(plugins_dir):
        dirs[:] = sorted(
            d for d in dirs if d not in STRIPPED_DIR_NAMES and d not in EXCLUDED_NAMES
        )
        descriptors = [f for f in files if f.endswith(".uplugin")]
        if descriptors:
            root = Path(parent)
            name = descriptors[0][: -len(".uplugin")]
            plugins.append(PluginSource(name, root, (root / "Source").is_dir()))
            dirs[:] = []  # a plugin's own folders hold no further plugins
    return plugins


# --- Extract ----------------------------------------------------------------------------


def _ignore(_directory: str, names: list[str]) -> set[str]:
    return {
        n
        for n in names
        if n in EXCLUDED_NAMES
        or n in STRIPPED_DIR_NAMES
        or n.endswith(EXCLUDED_SUFFIXES)
    }


def extract_plugin(plugin: PluginSource, staging_root: Path) -> Path:
    dest = staging_root / plugin.name
    log("EXTRACT  %s -> %s", plugin.name, dest)
    if dest.exists():
        shutil.rmtree(dest)
    # symlinks=False copies what each link points at: the archive must hold real files.
    shutil.copytree(plugin.root, dest, symlinks=False, ignore=_ignore)
    return dest


def patch_uplugin(plugin: PluginSource, staged: Path) -> None:
    path = staged / f"{plugin.name}.uplugin"
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    # Left false in the repository so internal projects keep enabling plugins explicitly.
    data["EnabledByDefault"] = True
    path.write_text(json.dumps(data, indent="\t") + "\n", encoding="utf-8")


# --- Validate ---------------------------------------------------------------------------


def _skipped(rel: Path) -> bool:
    if any(part in EXCLUDED_NAMES for part in rel.parts):
        return True
    if rel.as_posix().endswith(EXCLUDED_SUFFIXES):
        return True
    return any(part in STRIPPED_DIR_NAMES for part in rel.parts)


def _dropped_sources(plugin: PluginSource, staged: Path) -> list[str]:
    source_dir = plugin.root / "Source"
    if not source_dir.is_dir():
        return []
    return [
        f"source file dropped: {rel.as_posix()}"
        for rel in (
            p.relative_to(plugin.root) for p in source_dir.rglob("*") if p.is_file()
        )
        if not _skipped(rel) and not (staged / rel).exists()
    ]


def _missing_natives(code: Any, staged: Path, plat: str) -> list[str]:
    problems: list[str] = []
    for lib_dir in code.lib_dirs_for(plat):
        for lib in lib_dir.libs:
            for _built, staged_name in lib.staged_names(plat):
                rel = Path(lib_dir.dest) / staged_name
                if not (staged / rel).is_file():
                    problems.append(f"library missing: {rel.as_posix()}")
    for tree in code.headers:
        if not any(p.is_file() for p in (staged / tree.dest).rglob("*")):
            problems.append(f"include tree missing or empty: {tree.dest}")
    return problems


def _leaks(staged: Path) -> list[str]:
    problems: list[str] = []
    for parent, dirs, files in os.walk(staged):
        for name in dirs + files:
            entry = Path(parent) / name
            rel = entry.relative_to(staged)
            if entry.is_symlink():
                problems.append(f"link survived: {rel.as_posix()}")
            if _skipped(rel):
                problems.append(f"excluded content leaked: {rel.as_posix()}")
    return problems


def validate(
    plugin: PluginSource,
    staged: Path,
    hydrate: types.ModuleType,
    plat: str,
) -> list[str]:
    problems = _dropped_sources(plugin, staged)
    code = {p.name: p for p in hydrate.CODE_PLUGINS}.get(plugin.name)
    if code is not None:
        problems += _missing_natives(code, staged, plat)
    return problems + _leaks(staged)


# --- Archive ----------------------------------------------------------------------------


def archive(plugin: PluginSource, staged: Path, out: Path) -> Path:
    log("ARCHIVE  %s", out.name)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(staged.rglob("*")):
            if path.is_file():
                # One top-level folder, so unzipping into Plugins/ lands as Plugins/<Name>/...
                zf.write(path, Path(plugin.name) / path.relative_to(staged))
    log("         %.1f MiB", out.stat().st_size / (1024 * 1024))
    return out


# --- Driver -----------------------------------------------------------------------------


def package(args: argparse.Namespace) -> int:
    project_dir = PROJECT_DIR
    hydrate = load_hydrate(project_dir)
    plat = hydrate.host_platform()
    roots = hydrate.resolve_roots()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    problems: list[str] = []
    if args.skip_hydrate:
        problems += hydrate.missing(roots, plat)
    else:
        log("HYDRATE  %s", project_dir)
        problems += hydrate.hydrate(
            roots,
            plat=plat,
            build_dir=Path(os.path.abspath(args.build_dir)),
            ue_root=Path(os.path.abspath(args.ue_root)) if args.ue_root else None,
        )
    if problems:
        log("Hydration incomplete:")
        for problem in problems:
            log("  %s", problem)
        return 1

    args.staging_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    for plugin in discover_plugins(project_dir):
        log("=== %s", plugin.name)
        staged = extract_plugin(plugin, args.staging_dir)
        patch_uplugin(plugin, staged)
        found = validate(plugin, staged, hydrate, plat)
        problems += [f"{plugin.name}: {p}" for p in found]
        # Written even when validation failed, so a failed run can be inspected.
        produced.append(
            archive(plugin, staged, args.out_dir / plugin.archive_name(plat))
        )

    if problems:
        log("FAILED with %d problem(s):", len(problems))
        for problem in problems:
            log("  %s", problem)
        return 1
    for out in produced:
        log("Done: %s", out)
    log("Install the archives side by side: the plugins depend on each other.")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    tmp = Path(tempfile.gettempdir())
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="ZIP output directory (default: beside this script)",
    )
    parser.add_argument(
        "--staging-dir", type=Path, default=tmp / "superdex-unreal-staging"
    )
    parser.add_argument("--build-dir", type=Path, default=tmp / "superdex-unreal-build")
    parser.add_argument(
        "--ue-root",
        type=Path,
        default=os.environ.get("UE_ROOT") or None,
        help="Unreal Engine install, needed to build on Linux (default: $UE_ROOT)",
    )
    parser.add_argument(
        "--skip-hydrate",
        action="store_true",
        help="package the plugins as they are (fails if a library is missing)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="[package-plugins] %(message)s", stream=sys.stdout
    )
    args = _parse_args(argv)
    try:
        return package(args)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        log("ERROR: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
