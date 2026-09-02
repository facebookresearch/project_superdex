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

import importlib
import importlib.util
import os
import sys
import sysconfig
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple, NoReturn

from superdex.physics.environment import (
    get_env_var_value,
    LEGACY_PRECISION_ENV_VAR,
    PRECISION_ENV_VAR,
    set_env_var_value,
)

_PACKAGED_NATIVE_SEARCH_PATHS: set[Path] = set()
_PACKAGED_NATIVE_DLL_DIR_HANDLES: list[object] = []


########################################################################################


def _get_use_double_precision() -> bool:
    """Resolve, validate, and normalize the process-wide precision selection.

    The normalized value is written back to the environment so child processes inherit
    the same precision.
    """

    DEFAULT_PRECISION = "fp32"

    value = get_env_var_value(PRECISION_ENV_VAR, LEGACY_PRECISION_ENV_VAR)
    value = (value or DEFAULT_PRECISION).strip().lower()
    if value in ("single", "float", "float32", "fp32", "32"):
        set_env_var_value(PRECISION_ENV_VAR, LEGACY_PRECISION_ENV_VAR, "fp32")
        return False
    if value in ("double", "float64", "fp64", "64"):
        set_env_var_value(PRECISION_ENV_VAR, LEGACY_PRECISION_ENV_VAR, "fp64")
        return True
    raise ImportError(
        f"Unknown SuperDex precision: {value!r}. "
        f"Set {PRECISION_ENV_VAR} to 'fp32' or 'fp64'."
    )


USE_DOUBLE_PRECISION = _get_use_double_precision()
"""Whether to use the FP64 variants for SuperDex Physics bindings."""

PRECISION_NAME: str = "fp64" if USE_DOUBLE_PRECISION else "fp32"
"""The selected precision name."""


########################################################################################


class NativePayload(NamedTuple):
    """Where one family of native modules is packaged, per precision.

    FP32 lives under the ``superdex`` namespace of the base distribution; FP64 is a
    standalone top-level package installed by the ``[fp64]`` extra.
    """

    subpackage: str
    distribution: str
    fp64_package: str


def _superdex_namespace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def site_packages_root() -> Path:
    return Path(sysconfig.get_paths()["platlib"])


def native_root_via_spec(package: str) -> Path | None:
    """The `_native/` payload directory of a separately-distributed package, if importable."""

    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError):
        # ValueError: the name is in sys.modules with a cleared __spec__.
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations))) / "_native"


def native_roots(package: str) -> list[Path]:
    """Where a sibling distribution's `_native/` payload may live, most preferred first.

    Sibling distributions get their own top-level directory rather than a place next to
    the package that uses them, so they have to be found through the import system. The
    site-packages fallback covers editable installs, where the importing module is still
    in the source tree.
    """

    roots = []
    spec_root = native_root_via_spec(package)
    if spec_root is not None:
        roots.append(spec_root)
    roots.append(site_packages_root() / package / "_native")
    return roots


def _first_directory(candidates: Sequence[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _fp32_native_roots(payload: NativePayload) -> list[Path]:
    """Where the FP32 payload may live, most authoritative first.

    A wheel puts it next to this facade; an editable install leaves the facade in the
    source tree, which has no `_native/`, and CMake installs the payload under site-packages.
    """

    return [
        _superdex_namespace_root() / payload.subpackage / "_native",
        site_packages_root() / "superdex" / payload.subpackage / "_native",
    ]


def _fp64_native_roots(payload: NativePayload) -> list[Path]:
    """Where the optional fp64 payload may live, most authoritative first.

    fp64 is a separate wheel, so it is located through the import system rather than
    assumed to sit next to the facade; editable installs need the site-packages fallback.
    """

    roots = []
    spec_root = native_root_via_spec(payload.fp64_package)
    if spec_root is not None:
        roots.append(spec_root)
    roots.append(site_packages_root() / payload.fp64_package / "_native")
    return roots


def _packaged_native_roots(payload: NativePayload) -> list[Path]:
    if USE_DOUBLE_PRECISION:
        return _fp64_native_roots(payload)
    return _fp32_native_roots(payload)


def _packaged_native_dir(payload: NativePayload) -> Path | None:
    # Each distribution carries exactly one precision, so the payload sits directly in
    # `_native/` with no precision-keyed subdirectory.
    return _first_directory(_packaged_native_roots(payload))


def _available_packaged_precisions(payload: NativePayload) -> list[str]:
    available = []
    if _first_directory(_fp32_native_roots(payload)) is not None:
        available.append("fp32")
    if _first_directory(_fp64_native_roots(payload)) is not None:
        available.append("fp64")
    return available


def _ensure_packaged_native_search_path(payload: NativePayload) -> Path | None:
    native_dir = _packaged_native_dir(payload)
    if native_dir is None:
        return None

    native_dir_str = str(native_dir)
    if native_dir_str not in sys.path:
        sys.path.insert(0, native_dir_str)
    if (
        sys.platform == "win32"
        and hasattr(os, "add_dll_directory")
        and native_dir not in _PACKAGED_NATIVE_SEARCH_PATHS
    ):
        _PACKAGED_NATIVE_DLL_DIR_HANDLES.append(os.add_dll_directory(native_dir_str))
    _PACKAGED_NATIVE_SEARCH_PATHS.add(native_dir)
    return native_dir


########################################################################################


def precision_variant_for_module(module_name: str) -> str:
    """Return a module name for the selected precision."""
    # Precision is process-wide for SuperDex Physics Python facades; every native module lookup uses
    # the same suffix rule so mixed FP32/FP64 extension state is not introduced.
    return f"{module_name}_double" if USE_DOUBLE_PRECISION else module_name


def module_is_present(module_name: str, *, payload: NativePayload) -> bool:
    """Return whether Python can resolve a runtime implementation of a module."""

    # Query the precision-selected runtime name; the facade name may be stable, but the
    # extension module it forwards to depends on USE_DOUBLE_PRECISION.
    precision_module_name = precision_variant_for_module(module_name)
    _ensure_packaged_native_search_path(payload)
    loaded_module = sys.modules.get(precision_module_name)
    if loaded_module is not None:
        # A cached namespace package from type stubs is not a usable native module.
        return not module_is_stub_only(loaded_module)

    # Namespace packages have no origin, so require a concrete origin before declaring
    # that runtime code is available.
    spec = importlib.util.find_spec(precision_module_name)
    return spec is not None and spec.origin is not None


def module_is_stub_only(module: ModuleType) -> bool:
    """Return whether import resolved a PEP 420 namespace without runtime code."""
    # Stub-only packages expose namespace search paths but no concrete file.
    return getattr(module, "__file__", None) is None and hasattr(module, "__path__")


########################################################################################


class NativeModuleNotFoundError(ImportError):
    """Raised when no requested native module has a runtime implementation."""


def _try_get_source_build_tools() -> ModuleType | None:
    """Return the enabled internal source-build helper, if it is available.

    The helper is excluded from public packages and imports Meta-specific machinery, so
    it is loaded lazily. A present but disabled helper is treated as unavailable. Errors
    raised while evaluating the opt-in state are propagated.
    """

    try:
        from superdex.physics.internal import source_build
    except ImportError:
        return None
    return source_build if source_build.build_enabled() else None


def _prepare_source_build(allow_source_build: bool) -> ModuleType | None:
    source_build = _try_get_source_build_tools() if allow_source_build else None
    if source_build is not None:
        source_build.prepare_prebuilt_env()
    return source_build


def _load_selected_runtime_module(
    precision_module_name: str,
) -> tuple[ModuleType | None, ModuleNotFoundError | None, bool]:
    missing_error = None
    module = sys.modules.get(precision_module_name)
    if module is None:
        try:
            module = importlib.import_module(precision_module_name)
        except ModuleNotFoundError as error:
            if error.name != precision_module_name:
                raise
            missing_error = error
            module = None

    if module is None:
        return None, missing_error, False
    if module_is_stub_only(module):
        return None, missing_error, True
    return module, missing_error, False


def _build_selected_runtime_module(
    source_build: ModuleType,
    module_name: str,
    precision_module_name: str,
) -> tuple[ModuleType | None, ModuleNotFoundError | None, bool]:
    extension_dir = source_build.build_module(module_name)
    if extension_dir is None:
        return None, None, False

    sys.path.append(str(extension_dir))
    try:
        module = importlib.import_module(precision_module_name)
    except ModuleNotFoundError as error:
        if error.name != precision_module_name:
            raise
        return None, error, False

    if module_is_stub_only(module):
        return None, None, True
    return module, None, False


def _missing_payload_guidance(payload: NativePayload) -> str:
    """Name the distribution that carries the payload for the selected precision.

    Only produced when the *other* precision is packaged; in a source tree no packaged
    payload is expected and the advice would be wrong. Keep it self-contained -- it
    surfaces partway down a chained traceback.
    """

    available_precisions = _available_packaged_precisions(payload)
    if not available_precisions:
        return ""
    if PRECISION_NAME in available_precisions:
        return ""

    installed = ", ".join(available_precisions)
    if USE_DOUBLE_PRECISION:
        return (
            f". The FP64 payload is not installed (installed: {installed}). "
            f"Run `pip install '{payload.distribution}[fp64]'`, or build from source "
            "with MOCHI_USE_DOUBLE_PRECISION=ON"
        )
    return (
        f". The FP32 payload is not installed (installed: {installed}). "
        f"Run `pip install {payload.distribution}`, or build from source with "
        "MOCHI_USE_DOUBLE_PRECISION=OFF"
    )


def _raise_missing_native_module(
    module_name: str,
    precision_module_name: str,
    *,
    payload: NativePayload | None,
    found_stub_only: bool,
    missing_error: ModuleNotFoundError | None,
    build_error: ModuleNotFoundError | None,
) -> NoReturn:
    message = f"Could not import native module {precision_module_name!r}"
    if found_stub_only:
        message += "; found only a stub namespace package"
    if build_error is not None:
        message += f"; build error: {build_error}"
    if payload is not None:
        message += _missing_payload_guidance(payload)
    # Preserve the last miss as the cause: prefer the build failure when a source build
    # was attempted, otherwise the original import miss. Parenthesized so the `from` cause
    # is unambiguously the result of the `or`, not a precedence surprise.
    raise NativeModuleNotFoundError(message) from (build_error or missing_error)


def import_module(
    module_name: str,
    *,
    payload: NativePayload | None = None,
    allow_source_build: bool = False,
) -> ModuleType:
    """Import one selected-precision native SuperDex module.

    With ``allow_source_build``, a missing extension is built with Buck and retried.
    """

    precision_module_name = precision_variant_for_module(module_name)
    if payload is not None:
        _ensure_packaged_native_search_path(payload)

    # A prior from-source build (here or in a parent) records sibling shared libraries and
    # extension dirs; applying them first lets a cached import resolve without rebuilding.
    source_build = _prepare_source_build(allow_source_build)

    module, missing_error, found_stub_only = _load_selected_runtime_module(
        precision_module_name
    )
    if module is not None:
        return module

    build_error = None
    if source_build is not None:
        module, build_error, build_found_stub_only = _build_selected_runtime_module(
            source_build,
            module_name,
            precision_module_name,
        )
        if module is not None:
            return module
        found_stub_only = found_stub_only or build_found_stub_only

    _raise_missing_native_module(
        module_name,
        precision_module_name,
        payload=payload,
        found_stub_only=found_stub_only,
        missing_error=missing_error,
        build_error=build_error,
    )


########################################################################################


def forward_module(
    source: ModuleType,
    destination: dict[str, Any],
    *,
    reserved_names: Sequence[str] = (),
) -> list[str]:
    """Copy a native module's public symbols into a facade and return its public names.

    The source docstring is appended to whatever docstring the facade already has (its own
    module docstring, or one contributed by an earlier forward), so merging several
    extensions into one facade keeps every contributing docstring for ``help()`` instead
    of the last call overwriting the rest.

    ``reserved_names`` lists public names an earlier call already forwarded; a source that
    would redefine one raises ``ImportError`` rather than silently shadowing it.
    """

    # Facade modules keep their own identity, import metadata, and docstring handling
    # separate from the forwarded symbols, so Python introspection still points at the
    # facade file. __doc__ is handled explicitly below (concatenated, not copied).
    PRESERVED_ATTRIBUTES = {
        "__name__",
        "__doc__",
        "__file__",
        "__package__",
        "__path__",
        "__spec__",
        "__loader__",
    }

    public_names = [name for name in source.__dict__ if not name.startswith("_")]
    collisions = sorted(set(reserved_names).intersection(public_names))
    if collisions:
        raise ImportError(
            f"Cannot forward {source.__name__!r}: it redefines already-forwarded "
            f"symbols {collisions}."
        )

    for name, value in source.__dict__.items():
        if name not in PRESERVED_ATTRIBUTES:
            destination[name] = value

    # Append the extension docstring for help(), preserving any docstring the facade
    # already carries so merging multiple extensions does not drop earlier help text.
    docstrings = [text for text in (destination.get("__doc__"), source.__doc__) if text]
    destination["__doc__"] = "\n\n".join(docstrings)
    return public_names


########################################################################################


def lazy_import_resolver(
    *,
    lazy_imports: Mapping[str, str],
    namespace: dict[str, object],
    module_name: str,
) -> Callable[[str], ModuleType]:
    """Create a module attribute resolver for lazily-imported modules."""

    def resolve(name: str) -> ModuleType:
        lazy_module_name = lazy_imports.get(name)
        if lazy_module_name is None:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        try:
            module = importlib.import_module(lazy_module_name)
        except (NativeModuleNotFoundError, ModuleNotFoundError) as error:
            if (
                isinstance(error, ModuleNotFoundError)
                and error.name != lazy_module_name
            ):
                raise
            raise AttributeError(
                f"module {module_name!r} has no attribute {name!r}"
            ) from error
        namespace[name] = module
        return module

    return resolve


########################################################################################

# The import machinery above is packaging plumbing that the facades call on the
# user's behalf. Only the resolved precision is part of the public surface.
__all__ = ["PRECISION_NAME", "USE_DOUBLE_PRECISION"]
