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

# pyre-strict

from pathlib import Path

from superdex.physics.utils.coordinate_systems import CoordinateSystem

class RerunLoggerCfg:
    application_id: str
    recording_id: str | None
    recording_name: str | None
    spawn: bool
    connect: bool
    connect_addr: str | None
    save_path: Path | str | None
    coordinate_system: CoordinateSystem | str | None
    log_meshes: bool
    log_transforms: bool
    log_debug_draw: bool
    use_timeline: bool
    mesh_assets_dir: str | None

    def __init__(
        self,
        *,
        application_id: str = ...,
        recording_id: str | None = ...,
        recording_name: str | None = ...,
        spawn: bool = ...,
        connect: bool = ...,
        connect_addr: str | None = ...,
        save_path: Path | str | None = ...,
        coordinate_system: CoordinateSystem | str | None = ...,
        log_meshes: bool = ...,
        log_transforms: bool = ...,
        log_debug_draw: bool = ...,
        use_timeline: bool = ...,
        mesh_assets_dir: str | None = ...,
    ) -> None: ...
