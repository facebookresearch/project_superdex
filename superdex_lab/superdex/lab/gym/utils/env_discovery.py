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

"""Automatic discovery and registration of SuperDex Gym environments.

Environments are found by scanning the env packages for modules named ``*_env.py``
and collecting the :class:`~superdex.lab.gym.envs.mochi_env.MochiEnv` subclass each defines.
The Gymnasium id is derived from the class name (``CartPoleEnv`` ->
``superdex_gym/CartPole-v0``), and its config class is the sibling ``<Name>EnvCfg``.

Every JSON file next to an env module is one of two things:

- ``<module>_<variant>.json`` -- a gym config *variant*, the only place env config may
  live (e.g. ``cartpole_env_actuate_on_pole.json``). Its schema is::

      {"description": "...optional...", "env_cfg": {<EnvCfg field>: value, ...}}

- ``<module>[_<variant>].<kind>.json`` -- a usage *recipe* for the base env or for one
  variant, where ``<kind>`` is ``train`` or ``benchmark``. A ``train`` recipe holds only
  training settings and may not configure the environment, so every configuration that
  gets trained is also a named, discoverable env. (A ``benchmark`` recipe does carry a
  measurement setup in an ``env_cfg`` section: it is a harness baseline that the benchmark
  scripts sweep permutations over, not a task anyone runs or trains.)

``<variant>`` is a snake_case token, with three rules on its segments (see
:func:`_is_valid_variant_name`): ``env`` is forbidden, which is what keeps a longer
sibling module's files unambiguous; ``train`` and ``benchmark`` are forbidden, so a
variant cannot be confused with a recipe; and a ``test`` segment marks the variant
test-only -- discovered and smoke-tested, but never registered with Gymnasium nor listed
in a CLI.

Because discovery is purely filesystem-based, environments (and variants) whose
modules are absent from a build are simply never discovered, so no build-specific
gating is required in code.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import pkgutil
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
from superdex.lab.gym.envs.mochi_env import MochiEnv

logger: logging.Logger = logging.getLogger(__name__)

########################################################################################

# Root packages scanned recursively for ``*_env`` modules, in registration order.
# Any subpackage present in the build is included automatically; subpackages absent
# from a build contribute nothing, with no configuration required either way.
DEFAULT_ENV_PACKAGES: tuple[str, ...] = (
    "superdex.lab.gym.envs.benchmarks",
    "superdex.lab.gym.envs.robots",
)

_ENV_ID_PREFIX = "superdex_gym/"
_ENV_ID_SUFFIX = "-v0"

# A gym config variant name is a snake_case token; see _is_valid_variant_name.
_VARIANT_NAME_RE: re.Pattern[str] = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")

# Segments a variant name may not use; see _is_valid_variant_name.
_RESERVED_VARIANT_SEGMENTS: frozenset[str] = frozenset({"env", "train", "benchmark"})

# A variant name containing this segment is test-only; see EnvEntry.test_only.
_TEST_VARIANT_SEGMENT = "test"


########################################################################################


@dataclass(frozen=True)
class EnvEntry:
    """A discovered environment (optionally a named config variant)."""

    env_id: str
    """Gymnasium id, e.g. ``superdex_gym/CartPole-v0``."""
    env_cls: type[MochiEnv]
    """Environment class."""
    cfg_cls: type
    """Paired ``<Name>EnvCfg`` config class."""
    variant: str | None = None
    """Variant name (e.g. ``"actuate_on_pole"``), or ``None`` for the base/default env."""
    cfg_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments for ``cfg_cls`` (empty for the base env, parsed JSON otherwise)."""
    description: str | None = None
    """Optional human-readable description (from the variant JSON)."""
    test_only: bool = False
    """Whether this variant exists only to be smoke-tested (see :data:`_TEST_VARIANT_SEGMENT`).

    Test-only variants are degenerate configurations (no gravity, no damping, ...) that
    are worth crash-checking but are not shippable tasks, so they are deliberately kept
    out of the Gymnasium registry and the CLI listings.
    """

    @property
    def short_name(self) -> str:
        """CLI-friendly snake_case name, e.g. ``cart_pole`` / ``cart_pole_actuate_on_pole``."""
        base = _camel_to_snake(_strip_env_suffix(self.env_cls.__name__))
        return f"{base}_{self.variant}" if self.variant else base

    def make_cfg(self) -> Any:
        """Instantiate the config class with this entry kwargs."""
        return self.cfg_cls(**self.cfg_kwargs)

    def make_env(self) -> MochiEnv:
        """Instantiate the environment with this entry config."""
        return self.env_cls(self.make_cfg())


########################################################################################
# Discovery
########################################################################################


def discover_envs(packages: Sequence[str] = DEFAULT_ENV_PACKAGES) -> list[EnvEntry]:
    """Scan ``packages`` recursively and return the discovered envs and their variants.

    Each root package is walked recursively, so ``*_env`` modules in any subpackage
    present in the build are found automatically; subpackages missing from a build
    contribute nothing. Results are ordered by package, then module, with the base env
    preceding its variants.
    """
    entries: list[EnvEntry] = []
    for package_name in packages:
        package = importlib.import_module(package_name)
        for module_info in pkgutil.walk_packages(
            package.__path__, prefix=f"{package_name}."
        ):
            if module_info.ispkg or not module_info.name.endswith("_env"):
                continue
            entries.extend(
                _entries_for_module(importlib.import_module(module_info.name))
            )

    return entries


def _entries_for_module(module: object) -> list[EnvEntry]:
    """Build the base entry and any variant entries for a single ``*_env`` module."""
    entries: list[EnvEntry] = []
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if not _is_env_class(cls, module):
            continue

        env_name = _strip_env_suffix(cls.__name__)
        cfg_cls = getattr(module, f"{cls.__name__}Cfg", None)
        if cfg_cls is None:
            logger.warning(
                "Env %s has no paired %sCfg; skipping.", cls.__name__, cls.__name__
            )
            continue

        entries.append(
            EnvEntry(
                env_id=f"{_ENV_ID_PREFIX}{env_name}{_ENV_ID_SUFFIX}",
                env_cls=cls,
                cfg_cls=cfg_cls,
            )
        )
        entries.extend(_variant_entries(cls, cfg_cls, env_name, module))

    return entries


def _is_valid_variant_name(variant: str) -> bool:
    """Whether ``variant`` from ``<module>_<variant>.json`` names a gym config variant.

    A variant name must be a snake_case token, and none of its segments may be reserved
    (see :data:`_RESERVED_VARIANT_SEGMENTS`). Three things ride on this:

    * snake_case is what :func:`_snake_to_pascal` expects when it builds the gym id, so
      anything else would produce a malformed id. It is also what keeps a *variant-scoped*
      usage config out of gym registration: ``foo_env_hard.train.json`` yields
      ``hard.train``, because ``Path.stem`` strips only the final suffix, and registering
      it would produce ``FooHard.train-v0`` built from a training recipe.
    * forbidding an ``env`` segment is what makes sibling env modules unambiguous. The
      ``<module>_*.json`` glob over-matches into a longer sibling's files, so
      ``foo_env_extra_env.json`` would otherwise read as variant ``extra_env`` of
      ``foo_env.py`` as much as a file belonging to ``foo_env_extra_env.py``. Since every
      env module ends in the ``_env`` segment, rejecting that segment resolves the
      ambiguity locally, with no cross-module knowledge.
    * forbidding ``train`` / ``benchmark`` segments keeps ``foo_env_train.json`` (a
      variant) from reading as ``foo_env.train.json`` (a recipe); they differ by one
      character, so the two are reserved rather than left to a typo.
    """
    if _VARIANT_NAME_RE.fullmatch(variant) is None:
        return False
    return not _RESERVED_VARIANT_SEGMENTS.intersection(variant.split("_"))


def _variant_entries(
    env_cls: type,
    cfg_cls: type,
    env_name: str,
    module: object,
) -> list[EnvEntry]:
    """Discover ``<module>_<variant>.json`` config variants next to the env module."""
    module_file = Path(inspect.getfile(module))
    module_stem = module_file.stem
    entries: list[EnvEntry] = []
    for json_path in sorted(module_file.parent.glob(f"{module_stem}_*.json")):
        variant = json_path.stem[len(module_stem) + 1 :]
        if not _is_valid_variant_name(variant):
            # Not a gym variant: a usage config, a file belonging to a longer sibling env
            # module, or some other JSON that merely shares the prefix.
            logger.debug(
                "Ignoring '%s': '%s' is not a valid gym config variant name.",
                json_path.name,
                variant,
            )
            continue
        data = json.loads(json_path.read_text())
        variant_pascal = _snake_to_pascal(variant)
        entries.append(
            EnvEntry(
                env_id=f"{_ENV_ID_PREFIX}{env_name}{variant_pascal}{_ENV_ID_SUFFIX}",
                env_cls=env_cls,
                cfg_cls=cfg_cls,
                variant=variant,
                cfg_kwargs=dict(data.get("env_cfg", {})),
                description=data.get("description"),
                test_only=_TEST_VARIANT_SEGMENT in variant.split("_"),
            )
        )
    return entries


def _is_env_class(cls: type, module: object) -> bool:
    """Whether ``cls`` is a concrete env class defined in ``module``."""
    return (
        issubclass(cls, MochiEnv)
        and cls is not MochiEnv
        and not inspect.isabstract(cls)
        and cls.__name__.endswith("Env")
        # Only classes *defined* here, not imported base classes re-exported into scope.
        and cls.__module__ == getattr(module, "__name__", None)
    )


########################################################################################
# Registration and lookup
########################################################################################

_cached_entries: list[EnvEntry] | None = None


def get_env_entries() -> list[EnvEntry]:
    """Return the discovered entries, computing (and caching) them on first use."""
    global _cached_entries
    if _cached_entries is None:
        _cached_entries = discover_envs()
    return _cached_entries


def invalidate_env_entries() -> None:
    """Drop the cached discovery result so the next call rescans.

    Discovery walks the filesystem and imports modules, so the first result is cached.
    Call this if the set of env modules can change within the process (a test that writes
    a config variant, a plugin installed at runtime, a reloaded package).
    """
    global _cached_entries
    _cached_entries = None


def register_all_envs() -> list[EnvEntry]:
    """Register every discovered environment (and public variant) with Gymnasium.

    Idempotent: ids already present in the Gymnasium registry are left untouched. Returns
    every discovered entry, including the test-only ones that were not registered.

    Test-only variants are skipped, which is also what keeps them out of the Ray Tune
    registry, since that is populated by fanning out the Gymnasium registry.

    Every registered entry is registered with a ``cfg`` kwarg -- ``{}`` for a base env, the
    parsed variant JSON otherwise. Env constructors take ``MochiEnvCfg | dict`` and build
    their default config from an empty dict, so this keeps ``gym.make(env_id)`` working for
    base ids as well as variants. Callers wanting explicit configuration still override it
    (direct construction, ``gym.make(..., cfg=...)``, or RLlib's ``env_config``).

    The registered dict is a copy: entries are cached for the life of the process, so
    handing the registry the entry's own dict would let any consumer that mutates
    ``gym.spec(env_id).kwargs["cfg"]`` silently reconfigure every later lookup.
    """
    entries = get_env_entries()

    for entry in entries:
        if entry.test_only or entry.env_id in gym.registry:
            continue
        gym.register(
            entry.env_id,
            entry_point=entry.env_cls,
            kwargs={"cfg": dict(entry.cfg_kwargs)},
        )

    return entries


def get_env_short_names(include_test_only: bool = False) -> dict[str, EnvEntry]:
    """Map CLI-friendly short names (see :attr:`EnvEntry.short_name`) to entries.

    Test-only variants are excluded by default, so a CLI listing this map advertises only
    the environments it is meaningful to run. Pass ``include_test_only=True`` to get them
    back when deliberately debugging one.
    """
    return {
        entry.short_name: entry
        for entry in get_env_entries()
        if include_test_only or not entry.test_only
    }


def load_entry_config(entry: EnvEntry, kind: str) -> dict[str, Any]:
    """Load the ``<kind>`` usage recipe for ``entry``, or ``{}`` when it has none.

    A base entry reads ``<module>.<kind>.json``; a variant reads
    ``<module>_<variant>.<kind>.json``. A variant deliberately does *not* fall back to the
    base recipe: a recipe is written for one configuration, so inheriting (say) a base
    env's ``stop_criteria`` into a different configuration would silently train against
    the wrong target.
    """
    module_file = Path(inspect.getfile(entry.env_cls))
    stem = module_file.stem
    if entry.variant is not None:
        stem = f"{stem}_{entry.variant}"
    return _load_json(module_file.with_name(f"{stem}.{kind}.json"))


def load_env_config(env_cls: type, kind: str) -> dict[str, Any]:
    """Load the ``<env_module>.<kind>.json`` usage recipe sitting next to ``env_cls``.

    Unlike the gym config *variants* (``<module>_<variant>.json``), these dot-separated
    files hold non-gym usage configs (e.g. ``benchmark`` or ``train`` settings) and are
    deliberately never registered as gym environments nor picked up by discovery. Returns
    the parsed JSON, or an empty dict when the file is absent.

    This is the module-level recipe, i.e. the one belonging to the base env. Use
    :func:`load_entry_config` when the recipe should follow a specific entry's variant.
    """
    module_file = Path(inspect.getfile(env_cls))
    return _load_json(module_file.with_name(f"{module_file.stem}.{kind}.json"))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


########################################################################################
# Name helpers
########################################################################################


def _strip_env_suffix(class_name: str) -> str:
    return class_name[: -len("Env")] if class_name.endswith("Env") else class_name


def _camel_to_snake(name: str) -> str:
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    name = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", name)
    return name.lower()


def _snake_to_pascal(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_") if part)
