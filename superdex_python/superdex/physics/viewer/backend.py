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
Centralized polyscope backend module that handles optional polyscope imports and
provides API compatibility flags. This module isolates all polyscope dependencies to
make the library resilient to platforms that don't support it.
"""

from __future__ import annotations

try:
    import polyscope
    import polyscope.imgui as polyscope_imgui

    POLYSCOPE_AVAILABLE = True
except ImportError:
    polyscope = None
    polyscope_imgui = None
    POLYSCOPE_AVAILABLE = False

try:
    import polyscope.implot as polyscope_implot
except ImportError:
    polyscope_implot = None

########################################################################################

# Unfortunately Polyscope doesn't provide a versioning string, and some package
# management systems don't provide a way to query the version of a package
# installed. So, we have to rely on the presence of specific API functions to guess
# API compatibility. This is not ideal, but it's the best we can do for now.

# fmt: off
POLYSCOPE_VERSION_GE_2_5_0 = POLYSCOPE_AVAILABLE and hasattr(polyscope, "get_ui_scale")
"""Flag indicating if the Polyscope version is at least >= 2.5.0."""
# fmt: on

# For type-checking, use static imports so modules can be found (doesn't affect runtime behavior)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polyscope
    import polyscope.imgui as polyscope_imgui  # noqa: F401
    import polyscope.implot as polyscope_implot  # noqa: F401
