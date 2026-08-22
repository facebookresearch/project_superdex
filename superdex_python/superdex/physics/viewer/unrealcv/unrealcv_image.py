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

from __future__ import annotations

import cv2
import numpy as np


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Return an owning, contiguous RGB copy of a three- or four-channel image."""
    if image.dtype == np.uint8 and image.size > 0:
        conversion = cv2.COLOR_BGRA2RGB if image.shape[2] == 4 else cv2.COLOR_BGR2RGB
        return cv2.cvtColor(image, conversion)
    return image[:, :, 2::-1].copy()
