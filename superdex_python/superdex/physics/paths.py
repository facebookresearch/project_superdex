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
import sys
from collections.abc import Iterator
from pathlib import Path

from superdex.physics.environment import (
    ASSETS_PATH_ENV_VAR,
    get_env_var_value,
    LEGACY_ASSETS_PATH_ENV_VAR,
)

########################################################################################

__all__ = ["get_assets_root", "resolve_asset", "resolve_asset_root"]


def _resolve_assets_path_from_env() -> Path | None:
    """Resolve the optional assets override from canonical or legacy env vars."""
    env_value = get_env_var_value(ASSETS_PATH_ENV_VAR, LEGACY_ASSETS_PATH_ENV_VAR)
    if not env_value:
        return None
    try:
        path = Path(env_value).expanduser().resolve()
    except (OSError, RuntimeError, TypeError):
        return None
    if path.exists() and path.is_dir() and os.access(path, os.R_OK):
        return path
    return None


def _source_tree_assets_root() -> Path | None:
    module_path = Path(__file__).resolve()
    for ancestor in module_path.parents:
        if not (ancestor / "superdex_python").is_dir():
            continue
        candidate = ancestor / "assets"
        if candidate.is_dir() and os.access(candidate, os.R_OK):
            return candidate.resolve()
    return None


def _find_assets_path() -> Path | None:
    """Resolve the SuperDex assets path from env override or a source checkout."""
    env_path = _resolve_assets_path_from_env()
    if env_path is not None:
        return env_path.resolve()
    source_tree_path = _source_tree_assets_root()
    if source_tree_path is not None:
        return source_tree_path.resolve()
    return None


def get_assets_root() -> Path:
    assets_path = _find_assets_path()
    if assets_path is not None:
        return assets_path
    raise FileNotFoundError(
        "Could not locate the SuperDex assets root. Set SUPERDEX_ASSETS_PATH or "
        "use a SuperDex source checkout with an `assets/` directory."
    )


def _script_assets_roots() -> Iterator[Path]:
    """Yields the ``assets`` directories above the running script, nearest first.

    Assets belong to whichever distribution ships them: the physics examples load from
    ``superdex_physics/assets``, the robotics examples from the top-level ``assets``.
    Walking up from the script finds the one that owns the caller without either of them
    naming a path. Anchored on the script rather than on this module, which sits in
    ``superdex_python`` and so would only ever reach the repository root.
    """

    script = sys.argv[0] if sys.argv else ""
    if not script:
        # No script to anchor on: an interactive session, or an embedding host.
        return
    try:
        start = Path(script).resolve()
    except (OSError, RuntimeError, ValueError):
        return
    for ancestor in start.parents:
        candidate = ancestor / "assets"
        if candidate.is_dir() and os.access(candidate, os.R_OK):
            # Resolved so a symlinked root compares equal to the other two, which resolve.
            yield candidate.resolve()


def _asset_roots() -> Iterator[Path]:
    """Yields the roots a relative asset may live under, in priority order."""

    env_root = _resolve_assets_path_from_env()
    if env_root is not None:
        # An explicit override is the whole search; falling back would mix asset trees.
        yield env_root
        return

    seen = set()
    for root in (*_script_assets_roots(), _source_tree_assets_root()):
        if root is None or root in seen:
            continue
        seen.add(root)
        yield root


def _require_asset(asset_path: Path) -> tuple[Path, Path]:
    """Finds a relative asset, returning the root it came from and the resolved path.

    The roots are tried in order and the first that holds the asset wins, so a
    distribution's own assets take precedence over the ones further up the tree.
    """

    searched = []
    for root in _asset_roots():
        candidate = root / asset_path
        if candidate.exists():
            return root, candidate.resolve()
        searched.append(str(root))
    if not searched:
        raise FileNotFoundError(
            f"Could not resolve asset {asset_path!s}: no assets root found. Set "
            f"{ASSETS_PATH_ENV_VAR}, or use a SuperDex source checkout."
        )
    raise FileNotFoundError(
        f"Could not resolve asset {asset_path!s} under any of: {', '.join(searched)}"
    )


def resolve_asset(path: str | Path) -> Path:
    """Resolves an asset to an absolute path. Absolute paths are returned as they are."""

    asset_path = Path(path)
    if asset_path.is_absolute():
        if asset_path.exists():
            return asset_path.resolve()
        raise FileNotFoundError(f"Asset path does not exist: {asset_path}")
    return _require_asset(asset_path)[1]


def resolve_asset_root(path: str | Path) -> Path:
    """Returns the assets root that ``path`` resolves under.

    A prefab carries paths relative to the root it was authored against, so loading one
    needs the root it came from rather than the directory it happens to sit in.
    """

    asset_path = Path(path)
    if asset_path.is_absolute():
        raise ValueError(
            f"resolve_asset_root needs a path relative to an assets root: {asset_path}"
        )
    return _require_asset(asset_path)[0]
