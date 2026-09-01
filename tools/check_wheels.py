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

# PEP 723 inline metadata makes this script's empty dependency environment explicit, so
# running it does not sync whichever uv project contains or invokes it. This script imports
# only the standard library.
# Auditing a Windows wheel's linkage additionally needs `delvewheel`, but it is run as a
# subprocess rather than imported, and only for `win_amd64` wheels, so it stays out of the list
# rather than being installed on the platforms that never reach that check.
# /// script
# requires-python = ">=3.12"
# dependencies = ["delvewheel==1.13.0"]
# ///

"""Verify a wheelhouse built by `tools/build_wheels.py`.

    python3 tools/check_wheels.py wheelhouse

Two layers. The static layer reads the wheels as zip files and checks that the six
distributions divide their contents the way the split requires. The runtime layer installs
them into throwaway virtual environments and imports them, at both precisions.

The static checks exist because the runtime ones cannot see these failures: a wheel that
wrongly ships `superdex/__init__.py`, or that vendors a private copy of a library another
wheel owns, imports perfectly well on the machine that built it and breaks somewhere else.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import hmac
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import venv
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import cast, NamedTuple

########################################################################################


class NativeDistribution(NamedTuple):
    """A native wheel and what its contents must and must not include."""

    wheel_name: str
    payload_dir: str
    extension: str
    own_libraries: tuple[str, ...]


class ToolDistribution(NamedTuple):
    """A native wheel whose payload is a program rather than an extension module.

    Checked differently from `NativeDistribution` for one reason: nothing imports these, so
    the runtime layer cannot see them at all. Whether the executable is present, and whether
    its shared libraries came along, is only ever a static question.
    """

    wheel_name: str
    payload_dir: str
    executable: str
    own_libraries: tuple[str, ...]
    console_script: str | None


# `marl` carries no `real`-typed API, so it is not precision-suffixed; it is still the
# physics wheel's payload rather than the robotics wheel's.
_PHYSICS_LIBRARIES = ("mochi_physics", "mochi_pybind_core", "marl")
_PHYSICS_LIBRARIES_FP64 = ("mochi_physics_double", "mochi_pybind_core_double", "marl")

NATIVE_DISTRIBUTIONS = (
    NativeDistribution(
        wheel_name="superdex_physics",
        payload_dir="superdex/physics/_native/",
        extension="mochi_physics",
        own_libraries=_PHYSICS_LIBRARIES,
    ),
    NativeDistribution(
        wheel_name="superdex_physics_fp64",
        payload_dir="superdex_physics_fp64/_native/",
        extension="mochi_physics_double",
        own_libraries=_PHYSICS_LIBRARIES_FP64,
    ),
    NativeDistribution(
        wheel_name="superdex_robotics",
        payload_dir="superdex/robotics/_native/",
        extension="superdex_robotics",
        own_libraries=(),
    ),
    NativeDistribution(
        wheel_name="superdex_robotics_fp64",
        payload_dir="superdex_robotics_fp64/_native/",
        extension="superdex_robotics_double",
        own_libraries=(),
    ),
)

# The GUI tools and the helper they spawn. Each carries a program plus every shared library that
# program needs, so it runs straight out of site-packages with no launcher setup.
TOOL_DISTRIBUTIONS = (
    ToolDistribution(
        wheel_name="superdex_physics_debugger",
        payload_dir="superdex_physics_debugger/_native/",
        executable="mochi_debugger",
        # No `mochi_physics`: the debugger is a client of a running simulation, not a host of
        # one. It reaches the engine over the debug protocol, and the only physics code it
        # links is `mochi_physics_dbg` -- a static library carrying that protocol and the
        # public headers -- so no shared `mochi_physics` is ever loaded.
        own_libraries=("mochi_renderer", "marl"),
        console_script="superdex-physics-debugger",
    ),
    ToolDistribution(
        wheel_name="superdex_studio",
        payload_dir="superdex_studio/_native/",
        executable="superdex_studio",
        own_libraries=(
            "mochi_physics",
            "mochi_renderer",
            "mochi_mesh",
            "superdex_robotics",
            "marl",
        ),
        console_script="superdex-studio",
    ),
    ToolDistribution(
        wheel_name="superdex_mesh_cli",
        payload_dir="superdex_mesh_cli/_native/",
        executable="superdex_mesh_cli",
        # Alone among the tools, this one carries no shared library: it links no compiled Mochi
        # library, and CGAL and OCCT are static.
        own_libraries=(),
        # Spawned by other programs, never run by a human, so it exposes no command.
        console_script=None,
    ),
)

PURE_WHEEL_NAMES = ("superdex_lab", "superdex")
FP64_WHEEL_NAMES = ("superdex_physics_fp64", "superdex_robotics_fp64")

# The studio's resource trees. It resolves both relative to its own executable, so they have to
# be inside the payload directory.
STUDIO_RESOURCE_DIRS = ("assets/", "processing_presets/")

# Exactly one distribution is GPL, and the boundary only holds if it stays that way.
GPL_WHEEL_NAME = "superdex_mesh_cli"

# This is a deliberately manual inventory of each wheel's native link closure. When a bundled
# dependency is added or removed, update its wheel's notice, checked-in license texts, and this
# declaration together. The build check verifies that the declared files are packaged; it does
# not hash or otherwise freeze third-party license text.
_CORE_THIRD_PARTY_LICENSE_FILES = (
    "cityhash/LICENSE",
    "eigen/COPYING.APACHE",
    "eigen/COPYING.BSD",
    "eigen/COPYING.MINPACK",
    "eigen/COPYING.MPL2",
    "eigen/COPYING.README",
    "eigen/LICENSE",
    "happly/LICENSE",
    "hdf5/COPYING",
    "hdf5/COPYING_LBNL_HDF5",
    "marl/LICENSE",
    "picojson/LICENSE",
    "tiny_obj_loader/LICENSE",
    "zlib/LICENSE",
)
_PHYSICS_THIRD_PARTY_LICENSE_FILES = _CORE_THIRD_PARTY_LICENSE_FILES + (
    "entt/LICENSE",
    "pybind11/LICENSE",
)
_ROBOTICS_THIRD_PARTY_LICENSE_FILES = _CORE_THIRD_PARTY_LICENSE_FILES + (
    "miniz/LICENSE",
    "pybind11/LICENSE",
    "tinyxml2/LICENSE",
    "xxhash/LICENSE",
)
_RENDERER_THIRD_PARTY_LICENSE_FILES = _CORE_THIRD_PARTY_LICENSE_FILES + (
    "filament/LICENSE",
    "filament-basisu/APACHE-2.0.txt",
    "filament-basisu/BSD.txt",
    "filament-basisu/LICENSE",
    "filament-basisu/ZLIB.txt",
    "filament-basisu/ZSTD_LICENSE",
    "filament-cgltf/LICENSE",
    "filament-draco/LICENSE",
    "filament-fsr/LICENSE",
    "filament-libpng/LICENSE",
    "filament-meshoptimizer/LICENSE",
    "filament-mikktspace/LICENSE",
    "filament-robin-map/LICENSE",
    "filament-sgsr/LICENSE",
    "filament-smol-v/LICENSE",
    "filament-spirv-headers/LICENSE",
    "filament-stb/LICENSE",
    "filament-tinyexr/LICENSE",
    "filament-vkmemalloc/LICENSE.txt",
    "glad/KHRONOS_LICENSE",
    "glad/LICENSE",
    "glfw/LICENSE.md",
    "imgui/LICENSE.txt",
    "implot/LICENSE",
    "native-fonts/FONT_AWESOME_LICENSE.txt",
    "native-fonts/ROBOTO_APACHE-2.0.txt",
    "stl_reader/LICENSE",
    "tinyxml2/LICENSE",
)
THIRD_PARTY_LICENSE_FILES = {
    "superdex_mesh_cli": (
        "NOTICE.md",
        "boost/LICENSE_1_0.txt",
        "cgal/LICENSE",
        "cgal/LICENSE.BSL",
        "cgal/LICENSE.GPL",
        "cgal/LICENSE.LGPL",
        "cgal/LICENSE.RFL",
        "eigen/COPYING.APACHE",
        "eigen/COPYING.BSD",
        "eigen/COPYING.MINPACK",
        "eigen/COPYING.MPL2",
        "eigen/COPYING.README",
        "eigen/LICENSE",
        "occt/LICENSE_LGPL_21.txt",
        "occt/NOTICE.md",
        "occt/OCCT_LGPL_EXCEPTION.txt",
        "rapidjson/LICENSE",
    ),
    "superdex_physics": ("NOTICE.md",) + _PHYSICS_THIRD_PARTY_LICENSE_FILES,
    "superdex_physics_fp64": ("NOTICE.md",) + _PHYSICS_THIRD_PARTY_LICENSE_FILES,
    "superdex_robotics": ("NOTICE.md",) + _ROBOTICS_THIRD_PARTY_LICENSE_FILES,
    "superdex_robotics_fp64": ("NOTICE.md",) + _ROBOTICS_THIRD_PARTY_LICENSE_FILES,
    "superdex_physics_debugger": ("NOTICE.md",) + _RENDERER_THIRD_PARTY_LICENSE_FILES,
    "superdex_studio": ("NOTICE.md",)
    + _RENDERER_THIRD_PARTY_LICENSE_FILES
    + (
        "entt/LICENSE",
        "imguizmo/LICENSE",
        "miniz/LICENSE",
        "tinyfiledialogs/LICENSE",
        "xxhash/LICENSE",
    ),
}

THIRD_PARTY_LICENSE_REQUIRED_TEXT = {
    "filament-fsr/LICENSE": ("Copyright (c) 2021 Advanced Micro Devices",),
    "filament-sgsr/LICENSE": ("Copyright (c) 2023, Qualcomm Innovation Center",),
    "miniz/LICENSE": ("Copyright 2016 Martin Raiber",),
    "stl_reader/LICENSE": ("Copyright (c) 2018-2023, Sebastian Reiter",),
}

# Every library that belongs to a physics wheel. A robotics wheel containing any of these has been
# given a private copy by the repair tool, which is the `g_context` split that the repair
# exclusions in the robotics pyprojects exist to prevent.
FOREIGN_LIBRARIES = frozenset(_PHYSICS_LIBRARIES + _PHYSICS_LIBRARIES_FP64)

_LIBRARY_FILE = re.compile(
    r"^(?:lib)?(?P<stem>[A-Za-z0-9_.+]+?)"
    # auditwheel and delvewheel append a content hash to libraries they vendor.
    r"(?:[-_][0-9a-f]{6,})?"
    # ELF puts the SOVERSION after the extension (libmarl.so.1); Mach-O puts it before
    # (libmarl.1.dylib), which the stem would otherwise swallow.
    r"\.(?:so|dylib|dll|pyd)(?:\.[0-9]+)*$"
)
_TRAILING_VERSION = re.compile(r"(?:\.[0-9]+)+$")


########################################################################################


def _library_stem(member: str) -> str | None:
    """Return the library or extension name a wheel member carries, or None.

    Normalizes away the decorations the three platforms and their repair tools add, so that
    `libmarl.so`, `libmarl.so.1`, `libmarl.1.dylib`, `marl.dll` and `libmarl-a1b2c3d4.so`
    all answer `marl`.
    """

    match = _LIBRARY_FILE.match(member.rsplit("/", 1)[-1])
    if match is None:
        return None
    return _TRAILING_VERSION.sub("", match.group("stem"))


def _wheel_distribution(wheel: Path) -> str:
    return wheel.name.split("-", 1)[0]


class WheelFilename(NamedTuple):
    distribution: str
    version: str
    python_tags: tuple[str, ...]
    abi_tags: tuple[str, ...]
    platform_tags: tuple[str, ...]


class ManifestTarget(NamedTuple):
    system: str
    arch: str
    python: str
    abi: str


def _normalize_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def _wheel_filename(wheel: Path) -> WheelFilename | None:
    if not wheel.name.endswith(".whl"):
        return None
    parts = wheel.name[:-4].rsplit("-", 3)
    if len(parts) != 4 or "-" not in parts[0]:
        return None
    name, version_and_build = parts[0].split("-", 1)
    version = version_and_build.split("-", 1)[0]
    return WheelFilename(
        distribution=name,
        version=version,
        python_tags=tuple(parts[1].split(".")),
        abi_tags=tuple(parts[2].split(".")),
        platform_tags=tuple(parts[3].split(".")),
    )


def _members(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as archive:
        return archive.namelist()


def _metadata(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        return archive.read(name).decode("utf-8")


def _member_is_safe(info: zipfile.ZipInfo) -> bool:
    name = info.filename
    if (
        not name
        or ":" in name
        or "\\" in name
        or "\x00" in name
        or name.startswith("/")
    ):
        return False
    component_path = name[:-1] if name.endswith("/") else name
    if any(component in {"", ".", ".."} for component in component_path.split("/")):
        return False
    mode = (info.external_attr >> 16) & 0xFFFF
    kind = stat.S_IFMT(mode)
    return kind in {0, stat.S_IFREG, stat.S_IFDIR}


def _stream_sha256(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    hasher = hashlib.sha256()
    with archive.open(info) as source:
        while chunk := source.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.digest()


def _record_problem(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    digest: str,
    size: str,
) -> str | None:
    if not size.isdecimal() or int(size) != info.file_size:
        return f"{info.filename}: RECORD size {size!r} != {info.file_size}"
    algorithm, separator, encoded = digest.partition("=")
    if separator != "=" or algorithm != "sha256" or not encoded:
        return f"{info.filename}: RECORD must contain a sha256 digest"
    try:
        expected = base64.b64decode(
            encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True
        )
    except ValueError:
        return f"{info.filename}: malformed RECORD digest"
    if len(expected) != hashlib.sha256().digest_size:
        return f"{info.filename}: malformed RECORD digest"
    if not hmac.compare_digest(_stream_sha256(archive, info), expected):
        return f"{info.filename}: RECORD content hash does not match"
    return None


def _expanded_filename_tags(filename: WheelFilename) -> set[str]:
    return {
        f"{python_tag}-{abi_tag}-{platform_tag}"
        for python_tag in filename.python_tags
        for abi_tag in filename.abi_tags
        for platform_tag in filename.platform_tags
    }


class DistInfoMembers(NamedTuple):
    metadata: zipfile.ZipInfo
    wheel: zipfile.ZipInfo
    record: zipfile.ZipInfo


def _safe_member_index(
    wheel: Path, infos: list[zipfile.ZipInfo]
) -> tuple[dict[str, zipfile.ZipInfo], list[str]]:
    names: dict[str, zipfile.ZipInfo] = {}
    problems = []
    for info in infos:
        if not _member_is_safe(info):
            problems.append(f"{wheel.name}: unsafe ZIP member {info.filename!r}")
            continue
        folded = info.filename.casefold()
        if folded in names:
            problems.append(
                f"{wheel.name}: duplicate or case-colliding ZIP member {info.filename!r}"
            )
            continue
        names[folded] = info
    return names, problems


def _dist_info_members(infos: list[zipfile.ZipInfo]) -> DistInfoMembers | None:
    by_suffix = [
        [info for info in infos if info.filename.endswith(suffix)]
        for suffix in (".dist-info/METADATA", ".dist-info/WHEEL", ".dist-info/RECORD")
    ]
    if any(len(matches) != 1 for matches in by_suffix):
        return None
    return DistInfoMembers(*(matches[0] for matches in by_suffix))


def _check_dist_info_metadata(
    wheel: Path,
    archive: zipfile.ZipFile,
    filename: WheelFilename,
    members: DistInfoMembers,
) -> list[str]:
    problems = []
    dist_info_roots = {
        info.filename.split("/", 1)[0]
        for info in (members.metadata, members.wheel, members.record)
    }
    expected_dist_info = (
        f"{_normalize_distribution(filename.distribution)}-{filename.version}.dist-info"
    )
    if dist_info_roots != {expected_dist_info}:
        problems.append(
            f"{wheel.name}: metadata files do not share the expected dist-info directory"
        )

    metadata = BytesParser(policy=policy.default).parsebytes(
        archive.read(members.metadata)
    )
    metadata_name = metadata.get("Name")
    metadata_version = metadata.get("Version")
    if (
        not isinstance(metadata_name, str)
        or _normalize_distribution(metadata_name)
        != _normalize_distribution(filename.distribution)
        or metadata_version != filename.version
    ):
        problems.append(
            f"{wheel.name}: filename distribution/version does not match METADATA"
        )

    wheel_metadata = BytesParser(policy=policy.default).parsebytes(
        archive.read(members.wheel)
    )
    if wheel_metadata.get("Wheel-Version") is None:
        problems.append(f"{wheel.name}: WHEEL has no Wheel-Version")
    wheel_tags = set(wheel_metadata.get_all("Tag", []))
    if wheel_tags != _expanded_filename_tags(filename):
        problems.append(f"{wheel.name}: WHEEL tags do not match the filename tags")
    return problems


def _recorded_entries(
    wheel: Path, rows: list[list[str]]
) -> tuple[dict[str, tuple[str, str, str]], list[str]]:
    recorded: dict[str, tuple[str, str, str]] = {}
    problems = []
    for row in rows:
        if len(row) != 3 or not row[0]:
            problems.append(f"{wheel.name}: RECORD row must have three fields")
            continue
        record_path, digest, size = row
        if not _member_is_safe(zipfile.ZipInfo(record_path)):
            problems.append(
                f"{wheel.name}: RECORD contains unsafe path {record_path!r}"
            )
            continue
        folded = record_path.casefold()
        if folded in recorded:
            problems.append(f"{wheel.name}: RECORD repeats path {record_path!r}")
            continue
        recorded[folded] = (record_path, digest, size)
    return recorded, problems


def _check_record_inventory(
    wheel: Path,
    archive: zipfile.ZipFile,
    infos: list[zipfile.ZipInfo],
    record_info: zipfile.ZipInfo,
    recorded: dict[str, tuple[str, str, str]],
    present_signatures: set[str],
) -> list[str]:
    problems = []
    file_names = {info.filename.casefold(): info for info in infos if not info.is_dir()}
    expected_rows = set(file_names) - present_signatures
    if set(recorded) != expected_rows:
        missing = sorted(expected_rows - set(recorded))
        extra = sorted(set(recorded) - expected_rows)
        problems.append(
            f"{wheel.name}: RECORD inventory mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )

    for folded, (record_path, digest, size) in recorded.items():
        info = file_names.get(folded)
        if info is None:
            continue
        if record_path != info.filename:
            problems.append(
                f"{wheel.name}: RECORD path spelling does not match {info.filename!r}"
            )
        if info.filename == record_info.filename:
            if digest or size:
                problems.append(
                    f"{wheel.name}: RECORD's own hash and size must be empty"
                )
            continue
        problem = _record_problem(archive, info, digest, size)
        if problem is not None:
            problems.append(f"{wheel.name}: {problem}")
    return problems


def check_record_integrity(wheel: Path) -> list[str]:
    """Require a safe archive and an authoritative, SHA-256-backed RECORD."""

    problems: list[str] = []
    try:
        with zipfile.ZipFile(wheel) as archive:
            infos = archive.infolist()
            names, problems = _safe_member_index(wheel, infos)
            if problems:
                return problems

            dist_info = _dist_info_members(infos)
            if dist_info is None:
                return [
                    f"{wheel.name}: expected exactly one METADATA, WHEEL, and RECORD"
                ]

            filename = _wheel_filename(wheel)
            if filename is None:
                return [f"{wheel.name}: malformed wheel filename"]
            problems += _check_dist_info_metadata(wheel, archive, filename, dist_info)

            try:
                record_text = archive.read(dist_info.record).decode("utf-8")
                rows = list(csv.reader(io.StringIO(record_text, newline="")))
            except (UnicodeDecodeError, csv.Error) as error:
                problems.append(f"{wheel.name}: RECORD is malformed: {error}")
                return problems

            recorded, record_problems = _recorded_entries(wheel, rows)
            problems += record_problems

            signature_paths = {
                f"{dist_info.record.filename}.jws".casefold(),
                f"{dist_info.record.filename}.p7s".casefold(),
            }
            present_signatures = signature_paths & names.keys()
            if len(present_signatures) > 1:
                problems.append(f"{wheel.name}: wheel contains both RECORD signatures")
            problems += _check_record_inventory(
                wheel,
                archive,
                infos,
                dist_info.record,
                recorded,
                present_signatures,
            )
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        return [f"{wheel.name}: cannot read wheel archive: {error}"]
    return problems


########################################################################################


def check_namespace_is_shared(wheel: Path, members: list[str]) -> list[str]:
    """`superdex/` is a PEP 420 namespace split across three distributions.

    An `__init__.py` at its root would make whichever distribution shipped it the sole
    owner of the name, and the other two would stop importing.
    """

    if "superdex/__init__.py" in members:
        return [f"{wheel.name}: ships superdex/__init__.py, breaking the namespace"]
    return []


def check_fp64_owns_no_namespace_paths(wheel: Path, members: list[str]) -> list[str]:
    """The fp64 wheels must stay entirely out of `superdex/`.

    They are payload only. Anything they placed under `superdex/` would collide in RECORD
    with the base distribution that owns it, and uninstalling either would take files the
    other still needs.
    """

    if _wheel_distribution(wheel) not in FP64_WHEEL_NAMES:
        return []
    intruders = [name for name in members if name.startswith("superdex/")]
    if intruders:
        return [
            f"{wheel.name}: ships {len(intruders)} path(s) under superdex/: {intruders[:3]}"
        ]
    return []


def check_tool_owns_no_namespace_paths(wheel: Path, members: list[str]) -> list[str]:
    """The tool wheels must stay entirely out of `superdex/`.

    Same rule as the fp64 payloads, for the same reason: that namespace belongs to
    `superdex-physics`, and anything these placed inside it would collide in RECORD and make
    uninstalling one take files the others still need.
    """

    if _wheel_distribution(wheel) not in {d.wheel_name for d in TOOL_DISTRIBUTIONS}:
        return []
    intruders = [name for name in members if name.startswith("superdex/")]
    if intruders:
        return [
            f"{wheel.name}: ships {len(intruders)} path(s) under superdex/: {intruders[:3]}"
        ]
    return []


def check_payload_layout(
    wheel: Path, members: list[str], distribution: NativeDistribution
) -> list[str]:
    """The extension has to be where `superdex.physics.loader` looks for it."""

    payload = [name for name in members if name.startswith(distribution.payload_dir)]
    if not payload:
        return [f"{wheel.name}: no payload under {distribution.payload_dir}"]

    stems = {_library_stem(name) for name in payload}
    problems = []
    if distribution.extension not in stems:
        problems.append(
            f"{wheel.name}: {distribution.payload_dir} has no {distribution.extension} "
            f"extension (found {sorted(s for s in stems if s)})"
        )
    for library in distribution.own_libraries:
        if library not in stems:
            problems.append(
                f"{wheel.name}: {distribution.payload_dir} is missing its own "
                f"{library} shared library"
            )
    return problems


def check_no_foreign_libraries(
    wheel: Path, members: list[str], distribution: NativeDistribution
) -> list[str]:
    """A wheel must not carry a library another wheel in the set owns."""

    owned = set(distribution.own_libraries)
    foreign = sorted(
        {
            stem
            for stem in (_library_stem(name) for name in members)
            if stem in FOREIGN_LIBRARIES and stem not in owned
        }
    )
    if foreign:
        return [
            f"{wheel.name}: vendors {', '.join(foreign)}, which the physics wheel owns; "
            "the repair exclusions did not take effect"
        ]
    return []


def check_tool_payload(
    wheel: Path, members: list[str], distribution: ToolDistribution
) -> list[str]:
    """The program and every library it needs are in the payload directory.

    A tool wheel that installs cleanly and then fails at launch with "error while loading
    shared libraries" is the failure this catches. Nothing imports these wheels, so no
    runtime check will.
    """

    payload = [name for name in members if name.startswith(distribution.payload_dir)]
    if not payload:
        return [f"{wheel.name}: no payload under {distribution.payload_dir}"]

    problems = []
    executables = {
        name.rsplit("/", 1)[-1].removesuffix(".exe")
        for name in payload
        if _library_stem(name) is None and not name.endswith("/")
    }
    if distribution.executable not in executables:
        problems.append(
            f"{wheel.name}: {distribution.payload_dir} has no {distribution.executable} "
            f"executable (found {sorted(executables)})"
        )

    stems = {_library_stem(name) for name in payload}
    for library in distribution.own_libraries:
        if library not in stems:
            problems.append(
                f"{wheel.name}: {distribution.payload_dir} is missing its own "
                f"{library} shared library, so the executable will not start"
            )
    return problems


def check_tool_entry_point(
    wheel: Path, members: list[str], distribution: ToolDistribution
) -> list[str]:
    """The command is declared, and declared as a gui-script.

    `gui_scripts` rather than `console_scripts` is what keeps a console window from
    appearing on Windows, and the distribution name has to match the command for the bare
    `uvx <name>` form to resolve.
    """

    if distribution.console_script is None:
        return []
    entry_points = next(
        (name for name in members if name.endswith(".dist-info/entry_points.txt")), None
    )
    if entry_points is None:
        return [f"{wheel.name}: declares no {distribution.console_script} command"]
    with zipfile.ZipFile(wheel) as archive:
        content = archive.read(entry_points).decode("utf-8")
    if distribution.console_script not in content:
        return [f"{wheel.name}: declares no {distribution.console_script} command"]
    if "[gui_scripts]" not in content:
        return [
            f"{wheel.name}: {distribution.console_script} is not a gui-script, so Windows "
            "will show a console window when it is launched"
        ]
    return []


def check_studio_resources(wheel: Path, members: list[str]) -> list[str]:
    """The studio's resources sit beside its executable, not elsewhere in the wheel."""

    if _wheel_distribution(wheel) != "superdex_studio":
        return []
    payload = "superdex_studio/_native/"
    missing = [
        directory
        for directory in STUDIO_RESOURCE_DIRS
        if not any(name.startswith(payload + directory) for name in members)
    ]
    if missing:
        return [
            f"{wheel.name}: {', '.join(missing)} missing from {payload}; the studio "
            "resolves them relative to its own executable"
        ]
    return []


def check_license_boundary(wheel: Path) -> list[str]:
    """Exactly one distribution is GPL, and it is the one that links CGAL.

    The whole point of splitting the mesh helper out is that the GPL obligation stops at its
    edge. A wheel that picks up a GPL declaration it should not have -- or the helper losing
    the one it should -- means that boundary has moved.
    """

    # Only the license headers. The body of METADATA is the README, and the READMEs that explain
    # this split mention the GPL without being under it.
    headers = _metadata(wheel).split("\n\n", 1)[0].splitlines()
    licenses = [
        line
        for line in headers
        if line.startswith(
            ("License:", "License-Expression:", "Classifier: License ::")
        )
    ]
    declares_gpl = any("GPL" in line for line in licenses)
    is_helper = _wheel_distribution(wheel) == GPL_WHEEL_NAME
    if is_helper and not declares_gpl:
        return [
            f"{wheel.name}: links CGAL but no longer declares a GPL license; the license "
            "boundary that keeps the rest of SuperDex permissive depends on it"
        ]
    if not is_helper and declares_gpl:
        return [
            f"{wheel.name}: declares a GPL license, but only {GPL_WHEEL_NAME} should; "
            "something GPL has leaked out of the helper"
        ]
    return []


def check_third_party_licenses(wheel: Path, members: list[str]) -> list[str]:
    """Every affected native wheel carries its attribution payload."""

    required = THIRD_PARTY_LICENSE_FILES.get(_wheel_distribution(wheel))
    if required is None:
        return []

    member_by_name = {
        name: next(
            (
                member
                for member in members
                if member.endswith(f".dist-info/licenses/thirdparty_licenses/{name}")
            ),
            None,
        )
        for name in required
    }
    missing = [name for name, member in member_by_name.items() if member is None]
    problems = []
    if missing:
        problems.append(
            f"{wheel.name}: missing third-party attribution files: {', '.join(missing)}"
        )

    if not wheel.exists():
        return problems

    with zipfile.ZipFile(wheel) as archive:
        for name, required_text in THIRD_PARTY_LICENSE_REQUIRED_TEXT.items():
            member = member_by_name.get(name)
            if member is None:
                continue
            content = archive.read(member).decode("utf-8", errors="replace")
            omitted = [text for text in required_text if text not in content]
            if omitted:
                problems.append(
                    f"{wheel.name}: {name} omits required notice text: "
                    + ", ".join(repr(text) for text in omitted)
                )
    return problems


# The oldest libstdc++ SuperDex supports. Ubuntu 22.04 ships GCC 12's libstdc++, which provides
# up to GLIBCXX_3.4.30; a wheel that needs anything newer installs cleanly and then fails at import
# with "GLIBCXX_x.y.z not found". Robotics users are overwhelmingly on 22.04 and 24.04, so 22.04 is
# the floor and this is the number that enforces it.
#
# The manylinux tag does not cover this. It encodes the glibc floor only, so a wheel can be tagged
# manylinux_2_28 and still be unloadable on 22.04 because of libstdc++.
MAX_GLIBCXX = (3, 4, 30)

_GLIBCXX_SYMBOL = re.compile(rb"GLIBCXX_(\d+)\.(\d+)\.(\d+)")


def check_glibcxx_floor(wheel: Path, members: list[str]) -> list[str]:
    """No Linux wheel may require a libstdc++ newer than the oldest distro we support."""

    if "linux" not in wheel.name:
        return []
    highest = (0, 0, 0)
    with zipfile.ZipFile(wheel) as archive:
        for name in members:
            # Chosen by ELF magic, not by filename. A tool wheel's payload is a bare executable
            # with no extension -- `superdex_mesh_cli` is the whole of its wheel -- so matching
            # on `.so` reads nothing at all there and reports the wheel clean.
            with archive.open(name) as payload:
                if payload.read(4) != b"\x7fELF":
                    continue
            # The versioned symbols a binary needs are literal strings in its .dynstr section, so
            # they can be read without an ELF parser. These payloads define no GLIBCXX versions of
            # their own -- libstdc++ itself is never bundled -- so every match is a requirement.
            for match in _GLIBCXX_SYMBOL.finditer(archive.read(name)):
                highest = max(highest, tuple(int(g) for g in match.groups()))
    if highest > MAX_GLIBCXX:
        needed = ".".join(str(part) for part in highest)
        floor = ".".join(str(part) for part in MAX_GLIBCXX)
        return [
            f"{wheel.name}: needs GLIBCXX_{needed}, above the GLIBCXX_{floor} that Ubuntu 22.04 "
            "provides; it would install and then fail at import"
        ]
    return []


# The Visual C++ runtime an extension module may rely on instead of carrying. `python.exe` links
# both, so they resolved before any wheel was imported. Nothing supplies `msvcp140*.dll`: it
# arrives with the Visual C++ Redistributable, which a fresh Windows install has no reason to
# have, so a wheel that needs it has to carry it.
#
# The exemption is the interpreter's, not Windows'. It therefore does not extend to a tool wheel,
# whose payload is an executable running as its own process, searching its own directory and
# never seeing the copy beside `python.exe`.
#
# The runtime layer cannot see any of this, for the same reason it cannot see the GLIBCXX floor
# above: every machine that builds or verifies these wheels has Visual Studio on it, so the
# import succeeds there whether or not the DLL shipped. The first machine to disagree is a user's.
PERMITTED_VC_RUNTIME = frozenset({"vcruntime140.dll", "vcruntime140_1.dll"})

# The redistributable's DLL families, as filename prefixes.
_VC_RUNTIME_PREFIXES = (
    "concrt",
    "mfc",
    "msvcp",
    "msvcr",
    "vcamp",
    "vccorlib",
    "vcomp",
    "vcruntime",
)


def _direct_dependencies(binary: Path) -> set[str]:
    """The DLL names a Windows binary imports, lowercased.

    `delvewheel needed` reads the import table out of the PE header and resolves nothing on
    disk, so it answers the same on any host. The wheel therefore need not match the host
    auditing it, which is what makes this a static check rather than a runtime one.
    """

    result = subprocess.run(
        [sys.executable, "-m", "delvewheel", "needed", str(binary)],
        capture_output=True,
        text=True,
        check=True,
    )
    return {line.strip().lower() for line in result.stdout.splitlines() if line.strip()}


def check_windows_runtime_is_carried(wheel: Path, members: list[str]) -> list[str]:
    """No Windows wheel may need a Visual C++ Redistributable DLL it does not carry."""

    if "win_amd64" not in wheel.name:
        return []
    binaries = [
        name for name in members if name.lower().endswith((".pyd", ".dll", ".exe"))
    ]
    if not binaries:
        return []
    if importlib.util.find_spec("delvewheel") is None:
        raise SystemExit(
            f"auditing {wheel.name} needs delvewheel; install it with "
            "`python -m pip install delvewheel`"
        )

    with tempfile.TemporaryDirectory(prefix="superdex-linkage-") as temp_dir:
        root = Path(temp_dir)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(root, members=binaries)
        needed = set()
        for name in binaries:
            needed |= _direct_dependencies(root / name)

    # Everything the wheel ships answers its own name, wherever in the wheel it sits: a DLL beside
    # the binary that imports it and one in `<distribution>.libs` both resolve, the first through
    # the loader's own directory search and the second through the patch delvewheel writes.
    carried = {name.rsplit("/", 1)[-1].lower() for name in binaries}
    # A tool wheel gets no exemption: see `PERMITTED_VC_RUNTIME`.
    permitted = (
        frozenset()
        if _wheel_distribution(wheel) in {d.wheel_name for d in TOOL_DISTRIBUTIONS}
        else PERMITTED_VC_RUNTIME
    )
    missing = sorted(
        name
        for name in needed - carried - permitted
        if name.startswith(_VC_RUNTIME_PREFIXES)
    )
    if missing:
        return [
            f"{wheel.name}: needs {', '.join(missing)} but does not carry it, so it will not "
            "load on a machine without the Visual C++ Redistributable"
        ]
    return []


def check_platform_tag(wheel: Path) -> list[str]:
    """An unrepaired `linux_*` tag is rejected by PyPI and will not install elsewhere."""

    if re.search(r"-linux_[a-z0-9_]+\.whl$", wheel.name):
        return [
            f"{wheel.name}: still has a bare linux tag; auditwheel repair did not run"
        ]
    return []


def check_metadata_line_endings(wheel: Path) -> list[str]:
    """METADATA must use LF, so a pure wheel is the same file on every platform.

    setuptools writes it through a text-mode file object, which makes every line CRLF on
    Windows. `tools/build_wheels.py` rewrites that back to LF; this is what stops a future
    setuptools, or a wheelhouse assembled by hand, from quietly reintroducing it and
    leaving two byte-different wheels competing for one `py3-none-any` filename.
    """

    with zipfile.ZipFile(wheel) as archive:
        name = next(
            (n for n in archive.namelist() if n.endswith(".dist-info/METADATA")), None
        )
        if name is None or b"\r\n" not in archive.read(name):
            return []
    return [
        f"{wheel.name}: METADATA has CRLF line endings, so this wheel differs "
        "byte-for-byte from the same wheel built elsewhere"
    ]


########################################################################################


def check_wheel_contents(wheels: list[Path]) -> list[str]:
    extensions = {
        distribution.wheel_name: distribution for distribution in NATIVE_DISTRIBUTIONS
    }
    tools = {
        distribution.wheel_name: distribution for distribution in TOOL_DISTRIBUTIONS
    }
    problems = []
    for wheel in wheels:
        integrity_problems = check_record_integrity(wheel)
        if integrity_problems:
            problems += integrity_problems
            continue
        members = _members(wheel)
        problems += check_namespace_is_shared(wheel, members)
        problems += check_fp64_owns_no_namespace_paths(wheel, members)
        problems += check_tool_owns_no_namespace_paths(wheel, members)
        problems += check_metadata_line_endings(wheel)
        problems += check_license_boundary(wheel)
        problems += check_third_party_licenses(wheel, members)
        problems += check_glibcxx_floor(wheel, members)
        problems += check_windows_runtime_is_carried(wheel, members)

        name = _wheel_distribution(wheel)
        distribution = extensions.get(name)
        if distribution is not None:
            problems += check_platform_tag(wheel)
            problems += check_payload_layout(wheel, members, distribution)
            problems += check_no_foreign_libraries(wheel, members, distribution)

        tool = tools.get(name)
        if tool is not None:
            problems += check_platform_tag(wheel)
            problems += check_tool_payload(wheel, members, tool)
            problems += check_tool_entry_point(wheel, members, tool)
            problems += check_studio_resources(wheel, members)
    return problems


# Wheels a platform is not expected to produce. `SKIPPED_DISTRIBUTIONS` in build_wheels.py is the
# build-side source of truth; this is the same set in wheel names rather than directory names, so
# the two have to be emptied together.
#
# Consulted only when `--target` names a platform, because it describes a wheelhouse holding one
# platform's build and nothing else -- what the Manifold upload gathers. A wheelhouse assembled
# from a full run is complete whatever host checks it: the GitHub verify job downloads the
# `py3-none-any` wheels alongside its own platform's, and its runtime layer needs them for
# `pip install superdex`. Inferring the target from the host excused exactly those two there.
#
# Linux has no entry: the renderer wheels build there whenever the container can reach a package
# mirror, which is every CI run. A wheelhouse missing them is now incomplete rather than expected,
# so a regression fails here instead of passing quietly.
UNBUILT_WHEELS = {
    "Darwin": ("superdex", "superdex_lab"),
    "Windows": ("superdex", "superdex_lab"),
}

# Listed rather than taken from the keys above, so that emptying an entry cannot silently
# remove a platform from `--target`.
TARGET_PLATFORMS = ("Darwin", "Linux", "Windows")


def check_wheelhouse_is_complete(
    wheels: list[Path],
    *,
    target: str | None = None,
    target_arch: str | None = None,
    target_python: str | None = None,
    target_abi: str | None = None,
) -> list[str]:
    expected = (
        {d.wheel_name for d in NATIVE_DISTRIBUTIONS}
        | {d.wheel_name for d in TOOL_DISTRIBUTIONS}
        | set(PURE_WHEEL_NAMES)
    )
    unbuilt = UNBUILT_WHEELS.get(target, ())
    if unbuilt:
        print(
            f"[check_wheels] not expecting {', '.join(unbuilt)} in a {target} wheelhouse"
        )
        expected -= set(unbuilt)
    by_distribution: dict[str, list[Path]] = {}
    for wheel in wheels:
        by_distribution.setdefault(_wheel_distribution(wheel), []).append(wheel)

    problems = []
    found = set(by_distribution)
    missing = sorted(expected - found)
    unexpected = sorted(found - expected)
    duplicates = sorted(
        name for name, candidates in by_distribution.items() if len(candidates) != 1
    )
    if missing:
        problems.append(f"wheelhouse is missing: {', '.join(missing)}")
    if unexpected:
        problems.append(
            f"wheelhouse has unexpected distributions: {', '.join(unexpected)}"
        )
    if duplicates:
        problems.append(
            f"wheelhouse must contain exactly one wheel per distribution: {', '.join(duplicates)}"
        )
    for wheel in wheels:
        problems += check_target_tag(
            wheel,
            target=target,
            target_arch=target_arch,
            target_python=target_python,
            target_abi=target_abi,
        )
    return problems


def _canonical_arch(target: str, arch: str) -> str:
    normalized = arch.lower().replace("-", "_")
    aliases = {
        ("Linux", "amd64"): "x86_64",
        ("Linux", "arm64"): "aarch64",
        ("Darwin", "aarch64"): "arm64",
        ("Windows", "x86_64"): "amd64",
        ("Windows", "x64"): "amd64",
    }
    return aliases.get((target, normalized), normalized)


def check_target_tag(
    wheel: Path,
    *,
    target: str | None,
    target_arch: str | None,
    target_python: str | None,
    target_abi: str | None,
) -> list[str]:
    """Reject foreign-platform and foreign-architecture wheels."""

    filename = _wheel_filename(wheel)
    if filename is None:
        return [f"{wheel.name}: malformed wheel filename"]
    platforms = filename.platform_tags
    if platforms == ("any",):
        if filename.python_tags != ("py3",) or filename.abi_tags != ("none",):
            return [f"{wheel.name}: pure wheel must be tagged py3-none-any"]
        return []

    if target_python is not None and target_python not in filename.python_tags:
        return [f"{wheel.name}: Python tag does not match target {target_python}"]
    if target_abi is not None and target_abi not in filename.abi_tags:
        return [f"{wheel.name}: ABI tag does not match target {target_abi}"]

    selected_target = target or platform.system()
    prefixes = {
        "Linux": (
            "manylinux_",
            "manylinux1_",
            "manylinux2010_",
            "manylinux2014_",
            "musllinux_",
        ),
        "Darwin": ("macosx_",),
        "Windows": ("win_",),
    }
    allowed_prefixes = prefixes.get(selected_target)
    if allowed_prefixes is None:
        return [f"{wheel.name}: unsupported target platform {selected_target}"]
    if any(not tag.startswith(allowed_prefixes) for tag in platforms):
        return [f"{wheel.name}: contains a platform tag foreign to {selected_target}"]
    if target_arch is None:
        return []
    expected_arch = _canonical_arch(selected_target, target_arch)
    if any(not tag.endswith(f"_{expected_arch}") for tag in platforms):
        return [
            f"{wheel.name}: platform tag does not match target architecture {target_arch}"
        ]
    return []


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> tuple[int, str]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            size += len(chunk)
            hasher.update(chunk)
    return size, hasher.hexdigest()


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"manifest {field} must be an object")
    return value


def _manifest_target(target: dict[str, object]) -> ManifestTarget:
    target_os = target.get("os")
    target_arch = target.get("arch")
    systems = {"linux": "Linux", "macos": "Darwin", "windows": "Windows"}
    if (
        not isinstance(target_os, str)
        or target_os not in systems
        or not isinstance(target_arch, str)
        or not target_arch
    ):
        raise ValueError("manifest target OS or architecture is invalid")
    target_python = target.get("python")
    target_abi = target.get("abi")
    if not isinstance(target_python, str) or not re.fullmatch(
        r"[A-Za-z0-9_]+", target_python
    ):
        raise ValueError("manifest target python is invalid")
    if not isinstance(target_abi, str) or not re.fullmatch(
        r"[A-Za-z0-9_]+", target_abi
    ):
        raise ValueError("manifest target abi is invalid")
    return ManifestTarget(systems[target_os], target_arch, target_python, target_abi)


def _is_positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _authenticated_manifest_root(
    raw: bytes, expected_manifest_sha256: str | None
) -> dict[str, object]:
    manifest = json.loads(raw, object_pairs_hook=_strict_json_object)
    root = _mapping(manifest, "root")
    if set(root) != {"schema", "source", "target", "wheels"}:
        raise ValueError("manifest has missing or unknown top-level fields")
    if root["schema"] != 1:
        raise ValueError("manifest schema is not 1")
    if raw != _canonical_json_bytes(root):
        raise ValueError("manifest is not canonically serialized")

    manifest_digest = hashlib.sha256(raw).hexdigest()
    if expected_manifest_sha256 is not None and not hmac.compare_digest(
        manifest_digest, expected_manifest_sha256
    ):
        raise ValueError("manifest SHA-256 does not match the approved digest")
    return root


def _validate_manifest_source(value: object) -> None:
    source = _mapping(value, "source")
    expected_fields = {
        "repository",
        "run_id",
        "workflow_id",
        "artifact_id",
        "artifact_name",
        "head_sha",
        "github_archive_digest",
        "archive_sha256",
        "archive_size",
    }
    if set(source) != expected_fields:
        raise ValueError("manifest source has missing or unknown fields")
    if source["repository"] != "facebookresearch/project_superdex":
        raise ValueError("manifest repository is not Project SuperDex")
    for field in ("run_id", "workflow_id", "artifact_id"):
        if not _is_positive_integer(source[field]):
            raise ValueError(f"manifest source {field} is invalid")
    artifact_name = source["artifact_name"]
    if not isinstance(artifact_name, str) or not artifact_name.strip():
        raise ValueError("manifest source artifact_name is invalid")
    head_sha = source["head_sha"]
    if not isinstance(head_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", head_sha):
        raise ValueError("manifest source head SHA is invalid")
    github_digest = source["github_archive_digest"]
    if not isinstance(github_digest, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", github_digest
    ):
        raise ValueError("manifest GitHub archive digest is invalid")
    archive_digest = source["archive_sha256"]
    if not isinstance(archive_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", archive_digest
    ):
        raise ValueError("manifest archive SHA-256 is invalid")
    if github_digest.removeprefix("sha256:") != archive_digest:
        raise ValueError("manifest archive digests disagree")
    if not _is_positive_integer(source["archive_size"]):
        raise ValueError("manifest archive size is invalid")


def _valid_manifest_wheel_entry(name: object, size: object, digest: object) -> bool:
    return (
        isinstance(name, str)
        and Path(name).name == name
        and "\\" not in name
        and ":" not in name
        and "\x00" not in name
        and name.endswith(".whl")
        and _is_positive_integer(size)
        and isinstance(digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
    )


def _manifest_wheels(value: object) -> dict[str, tuple[int, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("manifest wheels must be a non-empty list")
    listed: dict[str, tuple[int, str]] = {}
    listed_folded: set[str] = set()
    for entry_value in value:
        entry = _mapping(entry_value, "wheel entry")
        if set(entry) != {"name", "size", "sha256"}:
            raise ValueError("manifest wheel entry has missing or unknown fields")
        name, size, digest = entry["name"], entry["size"], entry["sha256"]
        if not _valid_manifest_wheel_entry(name, size, digest):
            raise ValueError("manifest wheel entry is invalid")
        name = cast(str, name)
        size = cast(int, size)
        digest = cast(str, digest)
        folded = name.casefold()
        if folded in listed_folded:
            raise ValueError(f"manifest repeats wheel {name!r}")
        listed[name] = (size, digest)
        listed_folded.add(folded)
    return listed


def _manifest_wheel_problems(
    wheelhouse: Path,
    wheels: list[Path],
    listed: dict[str, tuple[int, str]],
) -> list[str]:
    problems = []
    for wheel in wheels:
        if wheel.is_symlink() or not wheel.is_file() or wheel.parent != wheelhouse:
            problems.append(f"{wheel.name}: wheel is not a regular wheelhouse file")
            continue
        size, digest = _sha256_file(wheel)
        expected_size, expected_digest = listed[wheel.name]
        if size != expected_size or not hmac.compare_digest(digest, expected_digest):
            problems.append(
                f"{wheel.name}: bytes do not match the authenticated manifest"
            )
    return problems


def verify_manifest(
    wheelhouse: Path,
    wheels: list[Path],
    manifest_path: Path,
    expected_manifest_sha256: str | None,
) -> tuple[ManifestTarget | None, list[str]]:
    """Verify provenance and every exact wheel byte before inspecting archives."""

    if manifest_path.is_symlink() or not manifest_path.is_file():
        return None, [f"manifest is not a regular file: {manifest_path}"]
    try:
        raw = manifest_path.read_bytes()
        root = _authenticated_manifest_root(raw, expected_manifest_sha256)
        _validate_manifest_source(root["source"])

        target = _mapping(root["target"], "target")
        if set(target) != {"os", "arch", "python", "abi"}:
            raise ValueError("manifest target has missing or unknown fields")
        manifest_target = _manifest_target(target)
        listed = _manifest_wheels(root["wheels"])
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return None, [f"invalid wheel manifest: {error}"]

    actual_names = {wheel.name for wheel in wheels}
    if actual_names != set(listed):
        return manifest_target, [
            "wheelhouse filenames do not exactly match the authenticated manifest"
        ]
    return manifest_target, _manifest_wheel_problems(wheelhouse, wheels, listed)


########################################################################################


MODULE_ORIGIN_AUDIT = """
site_roots = {
    Path(path).resolve()
    for path in (sysconfig.get_path("purelib"), sysconfig.get_path("platlib"))
    if path
}
for module_name, module in tuple(sys.modules.items()):
    if not module_name.startswith("superdex"):
        continue
    origins = []
    module_file = getattr(module, "__file__", None)
    if module_file is not None:
        origins.append(Path(module_file).resolve())
    origins.extend(Path(path).resolve() for path in getattr(module, "__path__", ()))
    if not origins:
        # pybind11's def_submodule() registers e.g. `superdex_robotics.bots` through
        # PyImport_AddModule, so it has no file, no __path__ and no spec. No finder ever
        # located it: it is part of whatever binary its parent was loaded from, and that
        # parent is checked in its own right. A top-level name has no parent to inherit
        # from -- rpartition gives "", which is never a key in sys.modules -- so one with
        # no origin still fails here.
        assert module_name.rpartition(".")[0] in sys.modules, (
            f"{module_name} has no inspectable import origin"
        )
        continue
    for resolved in origins:
        assert any(resolved.is_relative_to(root) for root in site_roots), (
            f"{module_name} imported from outside this venv: {resolved}"
        )
"""


SMOKE_TEST = (
    """
import os
import sys
import sysconfig
from pathlib import Path

expect_double = os.environ["SUPERDEX_EXPECT_DOUBLE"] == "1"

import superdex.physics as physics
import superdex.robotics as robotics

"""
    + MODULE_ORIGIN_AUDIT
    + """

reported = physics.uses_double_precision()
assert reported == expect_double, f"expected double={expect_double}, got {reported}"

# create_context() calls mochi::CheckContext(), which fails unless robotics resolved the same
# physics Context this process initialized. The two extensions live in different wheels and must
# still share one g_context.
physics.initialize(num_worker_threads=0)
try:
    context = robotics.create_context()
    assert context is not None, "create_context() returned None"
    assert robotics.create_context() is context, "create_context() is not idempotent"
finally:
    physics.shutdown()

print("smoke ok:", "fp64" if reported else "fp32", physics.__file__)
"""
)

MISSING_PAYLOAD_TEST = """
import sys

try:
    import superdex.physics
except ImportError as error:
    message = str(error)
    assert "superdex-physics[fp64]" in message, f"unhelpful error: {message}"
    print("missing-payload error ok")
    sys.exit(0)

sys.exit("importing superdex.physics at FP64 should have failed")
"""


########################################################################################


def _create_venv(path: Path) -> Path:
    venv.EnvBuilder(with_pip=True, clear=True).create(path)
    for candidate in (path / "bin" / "python", path / "Scripts" / "python.exe"):
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"could not find the interpreter in the virtual environment {path}"
    )


def _pip(python: Path, *arguments: str) -> None:
    command = [
        str(python),
        "-m",
        "pip",
        "--isolated",
        "--disable-pip-version-check",
        "--no-cache-dir",
        *arguments,
    ]
    print(f"[check_wheels] $ {' '.join(command)}")
    subprocess.run(
        command,
        check=True,
        cwd=python.parent.parent,
        env=_base_environment(),
    )


def _run_script(python: Path, script: str, environment: dict[str, str]) -> None:
    print(f"[check_wheels] running check with {environment}")
    with tempfile.TemporaryDirectory(prefix="superdex-probe-") as empty_cwd:
        subprocess.run(
            [str(python), "-I", "-P", "-c", script],
            check=True,
            cwd=empty_cwd,
            env={**_base_environment(), **environment},
        )


def _base_environment() -> dict[str, str]:
    # Keep only OS/runtime inputs needed to launch Python and establish TLS. Source, virtualenv,
    # package-manager, dynamic-loader, credential, and precision inputs are intentionally absent.
    allowed = {
        "APPDATA",
        "HOME",
        "LANG",
        "LC_ALL",
        "LOCALAPPDATA",
        "PATH",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USERPROFILE",
        "WINDIR",
    }
    return {key: value for key, value in os.environ.items() if key in allowed}


########################################################################################


def third_party_requirements(wheels: list[Path]) -> tuple[list[str], list[str]]:
    """Requirements of the SuperDex wheels that are not SuperDex wheels themselves."""

    internal = {
        _normalize_distribution(distribution.wheel_name)
        for distribution in NATIVE_DISTRIBUTIONS + TOOL_DISTRIBUTIONS
    } | {_normalize_distribution(name) for name in PURE_WHEEL_NAMES}
    requirements = set()
    problems = []
    for wheel in wheels:
        message = BytesParser(policy=policy.default).parsebytes(
            _metadata(wheel).encode("utf-8")
        )
        for requirement in message.get_all("Requires-Dist", []):
            if "@" in requirement.split(";", 1)[0]:
                problems.append(
                    f"{wheel.name}: direct-reference dependency is not allowed: {requirement!r}"
                )
                continue
            match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
            if match is None:
                problems.append(
                    f"{wheel.name}: invalid Requires-Dist value {requirement!r}"
                )
                continue
            if _normalize_distribution(match.group(1)) not in internal:
                requirements.add(requirement.strip())
    return sorted(requirements), problems


def populate_dependencies(python: Path, wheels: list[Path], deps: Path) -> None:
    """Download the third-party dependencies so the install itself can be offline.

    The install runs `--no-index` so that it can only resolve against wheels we built --
    otherwise a missing local wheel is silently satisfied from PyPI and the check passes
    for the wrong reason. That also blocks numpy, gymnasium and friends, hence this step.
    """

    requirements, problems = third_party_requirements(wheels)
    if problems:
        _print_problems(problems)
        raise SystemExit(1)
    deps.mkdir(parents=True, exist_ok=True)
    print(f"[check_wheels] downloading {len(requirements)} third-party requirement(s)")
    # `--only-binary=:all:` because the install below is `--no-index`. Without it pip is free
    # to fetch an sdist, which then has to be built in an isolated environment that cannot
    # reach an index to get its own backend -- a confusing failure, several packages deep.
    # This turns that into an explicit download-time error naming the offending requirement.
    _pip(python, "download", "--only-binary=:all:", "--dest", str(deps), *requirements)


########################################################################################


def _install_offline(python: Path, wheelhouse: Path, deps: Path, *targets: str) -> None:
    _pip(
        python,
        "install",
        "--no-index",
        "--find-links",
        str(wheelhouse),
        "--find-links",
        str(deps),
        *targets,
    )
    _pip(python, "check")


def check_complete_install(workspace: Path, wheelhouse: Path, deps: Path) -> None:
    """`pip install superdex` resolves, and both precisions work from that one install."""

    python = _create_venv(workspace / "complete")
    _install_offline(python, wheelhouse, deps, "superdex")
    _run_script(python, SMOKE_TEST, {"SUPERDEX_EXPECT_DOUBLE": "0"})
    _run_script(
        python,
        SMOKE_TEST,
        {"SUPERDEX_EXPECT_DOUBLE": "1", "SUPERDEX_PRECISION": "fp64"},
    )


def check_fp32_install(workspace: Path, wheelhouse: Path, deps: Path) -> None:
    """Without the `fp64` extra, asking for FP64 explains itself."""

    python = _create_venv(workspace / "fp32")
    _install_offline(python, wheelhouse, deps, "superdex-physics", "superdex-robotics")
    _run_script(python, SMOKE_TEST, {"SUPERDEX_EXPECT_DOUBLE": "0"})
    _run_script(python, MISSING_PAYLOAD_TEST, {"SUPERDEX_PRECISION": "fp64"})


def check_fp64_extra_install(workspace: Path, wheelhouse: Path, deps: Path) -> None:
    """The `[fp64]` extra pulls in the FP64 wheel and makes FP64 work."""

    python = _create_venv(workspace / "extra")
    _install_offline(
        python,
        wheelhouse,
        deps,
        "superdex-physics[fp64]",
        "superdex-robotics[fp64]",
    )
    _run_script(
        python,
        SMOKE_TEST,
        {"SUPERDEX_EXPECT_DOUBLE": "1", "SUPERDEX_PRECISION": "fp64"},
    )


def check_legacy_double_extra_install(
    workspace: Path, wheelhouse: Path, deps: Path
) -> None:
    """The legacy `[double]` extra and precision value continue to select FP64."""

    python = _create_venv(workspace / "legacy-double-extra")
    _install_offline(
        python,
        wheelhouse,
        deps,
        "superdex-physics[double]",
        "superdex-robotics[double]",
    )
    _run_script(
        python,
        SMOKE_TEST,
        {"SUPERDEX_EXPECT_DOUBLE": "1", "SUPERDEX_PRECISION": "double"},
    )


########################################################################################


def report_linkage(wheels: list[Path]) -> None:
    """Print `auditwheel show` for every wheel carrying a binary, when auditwheel is available.

    Advisory: the pass/fail linkage rules are the static content checks above, which work
    on every platform. This just puts the external dependency list and the resolved
    manylinux policy in the log next to them.

    The tool wheels are included. They ship the two largest binaries SuperDex publishes --
    OCCT is linked into `superdex_mesh_cli` and Filament into `superdex_studio` -- so they
    are the ones whose symbol list is worth reading, not the ones to leave out.
    """

    if shutil.which("auditwheel") is None:
        return
    binary = {d.wheel_name for d in NATIVE_DISTRIBUTIONS} | {
        d.wheel_name for d in TOOL_DISTRIBUTIONS
    }
    for wheel in wheels:
        if _wheel_distribution(wheel) not in binary:
            continue
        result = subprocess.run(
            ["auditwheel", "show", str(wheel)],
            capture_output=True,
            text=True,
            check=False,
        )
        print(f"\n[check_wheels] auditwheel show {wheel.name}\n{result.stdout.strip()}")


########################################################################################


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install and smoke-test a SuperDex wheelhouse."
    )
    parser.add_argument("wheelhouse", type=Path, help="Directory of built wheels.")
    parser.add_argument(
        "--deps-dir",
        type=Path,
        help=(
            "Where to cache downloaded third-party wheels. Defaults to a sibling "
            "'<wheelhouse>-deps' directory, kept out of the wheelhouse so that stays "
            "exactly the set of publishable artifacts."
        ),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse an already-populated dependency directory.",
    )
    parser.add_argument(
        "--target",
        choices=TARGET_PLATFORMS,
        help=(
            "Which platform's wheelhouse this is, as `platform.system()` spells it. "
            "Needed because one platform's wheels are checked on the Linux worker that "
            "uploads them, not on the one that built them. Omit it for a wheelhouse "
            "assembled from a full run, which is expected to hold every distribution."
        ),
    )
    parser.add_argument(
        "--contents-only",
        action="store_true",
        help="Run the static wheel-content checks and stop.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Authenticated wheel-manifest.v1.json produced by the intake tool.",
    )
    parser.add_argument(
        "--manifest-sha256",
        help="Independently approved SHA-256 of the canonical manifest bytes.",
    )
    parser.add_argument(
        "--require-provenance",
        action="store_true",
        help="Fail unless both an authenticated manifest and its approved digest are supplied.",
    )
    return parser.parse_args()


def _run_static_checks(
    wheels: list[Path],
    *,
    target: str | None = None,
    target_arch: str | None = None,
    target_python: str | None = None,
    target_abi: str | None = None,
) -> None:
    problems = check_wheelhouse_is_complete(
        wheels,
        target=target,
        target_arch=target_arch,
        target_python=target_python,
        target_abi=target_abi,
    ) + check_wheel_contents(wheels)
    if problems:
        print("\n[check_wheels] FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        raise SystemExit(1)


def _validate_provenance_arguments(args: argparse.Namespace) -> None:
    if args.manifest_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", args.manifest_sha256
    ):
        raise SystemExit("--manifest-sha256 must be 64 lowercase hex characters")
    if (args.manifest is None) != (args.manifest_sha256 is None):
        raise SystemExit("--manifest and --manifest-sha256 must be supplied together")
    if args.require_provenance and (
        args.manifest is None or args.manifest_sha256 is None
    ):
        raise SystemExit(
            "--require-provenance requires --manifest and --manifest-sha256"
        )


def _print_problems(problems: list[str]) -> None:
    print("\n[check_wheels] FAILED:", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)


def _verified_manifest_target(
    args: argparse.Namespace, wheelhouse: Path, wheels: list[Path]
) -> tuple[ManifestTarget | None, list[str]]:
    if args.manifest is None:
        return None, []
    return verify_manifest(
        wheelhouse,
        wheels,
        args.manifest.absolute(),
        args.manifest_sha256,
    )


def _effective_target(
    requested_target: str | None, manifest_target: ManifestTarget | None
) -> str | None:
    if manifest_target is None:
        return requested_target
    if requested_target is not None and requested_target != manifest_target.system:
        raise SystemExit("--target does not match the authenticated manifest")
    return manifest_target.system


def _validate_runtime_target(
    effective_target: str | None, manifest_target: ManifestTarget | None
) -> None:
    if effective_target is not None and effective_target != platform.system():
        raise SystemExit(
            "runtime qualification must run on the authenticated target platform"
        )
    if manifest_target is not None and _canonical_arch(
        platform.system(), platform.machine()
    ) != _canonical_arch(platform.system(), manifest_target.arch):
        raise SystemExit(
            "runtime qualification must run on the authenticated target architecture"
        )


def _run_runtime_checks(
    args: argparse.Namespace, wheelhouse: Path, wheels: list[Path]
) -> None:
    deps = (args.deps_dir or wheelhouse.with_name(wheelhouse.name + "-deps")).resolve()
    with tempfile.TemporaryDirectory(prefix="superdex-check-") as temp_dir:
        workspace = Path(temp_dir)
        if not args.skip_download:
            populate_dependencies(_create_venv(workspace / "download"), wheels, deps)
        check_complete_install(workspace, wheelhouse, deps)
        check_fp32_install(workspace, wheelhouse, deps)
        check_fp64_extra_install(workspace, wheelhouse, deps)
        check_legacy_double_extra_install(workspace, wheelhouse, deps)


def main() -> int:
    args = _parse_args()
    wheelhouse = args.wheelhouse.resolve()
    wheels = sorted(wheelhouse.glob("*.whl"))
    if not wheels:
        raise SystemExit(f"no wheels found in {wheelhouse}")
    print(json.dumps([wheel.name for wheel in wheels], indent=2))

    _validate_provenance_arguments(args)
    manifest_target, manifest_problems = _verified_manifest_target(
        args, wheelhouse, wheels
    )
    if manifest_problems:
        _print_problems(manifest_problems)
        return 1
    effective_target = _effective_target(args.target, manifest_target)

    _run_static_checks(
        wheels,
        target=effective_target,
        target_arch=manifest_target.arch if manifest_target else None,
        target_python=manifest_target.python if manifest_target else None,
        target_abi=manifest_target.abi if manifest_target else None,
    )
    report_linkage(wheels)
    print(f"[check_wheels] wheel contents ok ({len(wheels)} wheels)")
    if args.contents_only:
        return 0

    _validate_runtime_target(effective_target, manifest_target)
    _run_runtime_checks(args, wheelhouse, wheels)

    print("\n[check_wheels] all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
