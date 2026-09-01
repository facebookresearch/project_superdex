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

import os

########################################################################################

# Location of special SuperDex directories.
ASSETS_PATH_ENV_VAR = "SUPERDEX_ASSETS_PATH"
"""Environment variable used to specify the location of the assets directory."""
LEGACY_ASSETS_PATH_ENV_VAR = "MOCHI_ASSETS_PATH"
"""Deprecated environment variable alias for the assets directory."""

DEBUGGER_PATH_ENV_VAR = "SUPERDEX_DEBUGGER_PATH"
"""Environment variable used to specify the mochi_debugger executable path."""
LEGACY_DEBUGGER_PATH_ENV_VAR = "MOCHI_DEBUGGER_PATH"
"""Deprecated environment variable alias for the debugger executable path."""

# Logging-related settings.
LOG_LEVEL_ENV_VAR = "SUPERDEX_PYTHON_LOG_LEVEL"
"""Environment variable used to specify the Python logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."""
LEGACY_LOG_LEVEL_ENV_VAR = "MOCHI_PYTHON_LOG_LEVEL"
"""Deprecated environment variable alias for the Python logging level."""

# Precision selection.
PRECISION_ENV_VAR = "SUPERDEX_PRECISION"
"""Environment variable used to select SuperDex bindings precision (single, double)."""
LEGACY_PRECISION_ENV_VAR = "MOCHI_PRECISION"
"""Deprecated environment variable alias for bindings precision."""


def get_env_var_value(primary: str, legacy: str) -> str | None:
    """Return a canonical env value, falling back to the deprecated alias.

    Membership rather than truthiness: an empty canonical value is how a caller clears an
    inherited setting, so it has to win over the legacy alias instead of reading as unset.
    """
    if primary in os.environ:
        return os.environ[primary]
    return os.environ.get(legacy)


def set_env_var_value(primary: str, legacy: str, value: str) -> None:
    """Write the canonical env value, mirroring to the deprecated alias only if in use.

    Mirroring unconditionally would introduce the deprecated name into a process that never
    had it and then leak it to every child, so a clean environment stays clean.
    """
    os.environ[primary] = value
    if legacy in os.environ:
        os.environ[legacy] = value
