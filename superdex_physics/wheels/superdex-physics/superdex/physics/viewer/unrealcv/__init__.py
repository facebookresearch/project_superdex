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
UnrealCV-based viewer for SuperDex Lab gym environments.

This module provides an alternative viewer implementation that uses UnrealCV to render
mochi simulations in Unreal Engine. It requires a running Unreal Engine instance with
the UnrealCV plugin installed and configured.
"""

from __future__ import annotations

import importlib.util

# Check if unrealcv is available
UNREALCV_AVAILABLE = importlib.util.find_spec("unrealcv") is not None

# The viewer is only available if unrealcv is installed
UNREALCV_VIEWER_AVAILABLE = UNREALCV_AVAILABLE

# Only import viewer classes if unrealcv is available
if UNREALCV_AVAILABLE:
    from .unrealcv_viewer import UnrealCVViewer
    from .unrealcv_viewer_cfg import CaptureFormat, CaptureMode, UnrealCVViewerCfg
    from .unrealcv_viewer_state import (
        UnrealCVActorState,
        UnrealCVCameraState,
        UnrealCVSceneState,
        UnrealCVViewerState,
    )

__all__ = [
    "UNREALCV_VIEWER_AVAILABLE",
    "CaptureFormat",
    "CaptureMode",
    "UnrealCVViewer",
    "UnrealCVViewerCfg",
    "UnrealCVActorState",
    "UnrealCVCameraState",
    "UnrealCVSceneState",
    "UnrealCVViewerState",
]
