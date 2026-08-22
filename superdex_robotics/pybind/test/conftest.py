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

import json
import os
import unittest

import numpy as np
import superdex.physics as mochi
import superdex.robotics as bots  # noqa: F401


np_real = np.float64 if mochi.uses_double_precision() else np.float32

requires_hdf5 = unittest.skipUnless(mochi.uses_hdf5(), "Requires MOCHI_USE_HDF5=ON")

# True on an internal (Meta) build. The bots `test` target sets MOCHI_INTERNAL from
# mochi_is_internal(); on @mode/external it is "0", so the @requires_internal bot-scene
# tests skip (they load bot assets that only ship internally).
is_internal_build = os.environ.get("MOCHI_INTERNAL") == "1"

# Decorator that skips the test unless running against an internal build.
requires_internal = unittest.skipUnless(
    is_internal_build, "Requires internal build (MOCHI_INTERNAL=1)"
)

default_num_worker_threads = 0

mochi.initialize(num_worker_threads=default_num_worker_threads)


def find_repo_root() -> str | None:
    current_dir = os.path.abspath(os.getcwd())
    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, ".buckconfig")):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None


def find_assets_dir() -> str:
    env_path = os.environ.get("MOCHI_ASSETS_PATH")
    if env_path:
        return env_path if env_path.endswith("/") else env_path + "/"
    repo_root = find_repo_root()
    if repo_root:
        path = os.path.join(repo_root, "arvr", "libraries", "mochi", "assets") + "/"
        if os.path.isdir(path):
            return path
    raise RuntimeError("Cannot find Mochi assets directory")


assets_dir = find_assets_dir()


def find_bots_assets_dir() -> str:
    # Bot assets live under Superdex (outside the mochi assets tree), so they
    # are resolved separately from the mochi assets.
    # Case 1: MOCHI_BOTS_ASSETS_PATH environment variable (set by the test runner).
    env_path = os.environ.get("MOCHI_BOTS_ASSETS_PATH")
    if env_path:
        return env_path if env_path.endswith("/") else env_path + "/"
    # Case 2: running in fbsource — resolve relative to the repo root.
    repo_root = find_repo_root()
    if repo_root:
        path = os.path.join(repo_root, "arvr", "projects", "superdex", "assets") + "/"
        if os.path.isdir(path):
            return path
    raise RuntimeError("Cannot find bot assets directory")


bots_assets_dir = find_bots_assets_dir()


def write_assets_tag_root_marker(dir_path: str) -> None:
    """Write a .superdex_root in `dir_path` mapping @bots -> canonical assets/bots dir.

    A bot scene written to a temp dir needs this to resolve the @bots asset tag its bot
    paths use.
    """
    bots_dir = os.path.realpath(os.path.join(bots_assets_dir, "bots")).replace(
        "\\", "/"
    )
    with open(os.path.join(dir_path, ".superdex_root"), "w") as f:
        f.write(json.dumps({"@bots": bots_dir}))


def bot_tag_path(asset_relative_path: str) -> str:
    """Build an @bots/<rel> path for an asset under <assets>/bots/<rel>."""
    prefix = "bots/"
    assert asset_relative_path.startswith(prefix), asset_relative_path
    return "@bots/" + asset_relative_path[len(prefix) :]


class MochiTestBase(unittest.TestCase):
    pass
