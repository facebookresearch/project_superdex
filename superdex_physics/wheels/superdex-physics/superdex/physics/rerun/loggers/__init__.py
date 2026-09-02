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

"""
Loggers for the Mochi rerun integration.

This module contains logger classes that convert Mochi data structures to rerun
entities and log them to the active recording.
"""

from superdex.physics.rerun.loggers.actor_logger import (
    ActorLogger,
    compute_face_normals,
    extract_actor_mesh,
)
from superdex.physics.rerun.loggers.base import Logger
from superdex.physics.rerun.loggers.debug_draw_logger import DebugDrawLogger

__all__ = [
    "ActorLogger",
    "DebugDrawLogger",
    "Logger",
    "compute_face_normals",
    "extract_actor_mesh",
]
