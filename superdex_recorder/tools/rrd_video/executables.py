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

"""Packaged executables, which some packagers install without the execute bit."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def ensure_executable(path: Path, work_dir: Path) -> Path:
    """`path`, or an executable copy of it in this run's private `work_dir`."""
    if os.access(path, os.X_OK):
        return path
    copy = work_dir / "bin" / path.name
    copy.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutil.copyfile(path, copy)
    copy.chmod(0o700)
    return copy
