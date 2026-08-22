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

"""Shared UnrealCV utilities for SuperDex Physics environments.

This module provides common configuration and utilities for environments that use
UnrealCV rendering with SuperDex Physics simulation.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from superdex.physics.viewer.unrealcv.unrealcv_client import UnrealCVClient

logger = logging.getLogger(__name__)

# Guard so the legacy ``hidden_actors`` deprecation warning is logged at most
# once per process instead of on every reset (set_actors_visibility runs each
# reset). Reset is process-scoped on purpose -- one nudge per run is enough.
_HIDDEN_ACTORS_DEPRECATION_WARNED = False

# Blueprint prefix indicator for actor mapping values
BLUEPRINT_PREFIX = "blueprint://"

# Prefab prefix indicator for actor mapping values. Unlike `blueprint://`, this
# is an internal-only marker produced by the bot-task flow (see
# `build_task_actor_mapping` in make_dataset.py); it is never hand-authored in
# YAML. The value after the prefix is a mochi prefab name that the Unreal side
# resolves to a render class via its prefab render registry.
PREFAB_PREFIX = "prefab://"

# Sentinel value sent to the UnrealCV vset .../rgba and vset .../hsva commands
# for individual channels that should be left unchanged. The C++ handler in
# ObjectHandler.cpp treats any channel value strictly less than -10000.0 as
# "keep the previous value of this channel" rather than overwriting it. This
# lets domain randomization randomize, e.g., only hue while leaving saturation
# and value as set on the underlying material/light.
KEEP_PREVIOUS_TOKEN = -1.0e6


@dataclass
class UnrealCVActorConfig:
    """Configuration for UnrealCV connection and rendering."""

    # Actors to hide at startup (e.g., extra shape objects in the UE level)
    hidden_actors: list[str] = field(default_factory=list)
    """DEPRECATED: legacy list of UE actor names to always hide.

    Superseded by ``visibility_overrides`` (a ``{name: bool | probability}``
    map). Kept for backward compatibility: ``set_actors_visibility`` migrates
    each entry here into ``visibility_overrides`` as ``False`` (always hide) at
    runtime, with explicit ``visibility_overrides`` entries taking precedence.

    Defaults to empty so the deprecation warning only fires for configs that
    explicitly opt into the legacy field. Prefer ``visibility_overrides`` in new
    configs (e.g. ``{name: false}`` to always hide)."""

    # Actor mapping for UE scene
    actor_mapping: dict[str, str] = field(
        default_factory=lambda: {
            # Map mochi actor names to Unreal Engine actor names
            # Scene objects
            "scene/Cross___Actor": "BP_Cross_Bryant_Render_C_2",
            "scene/Box___Actor": "BP_ShapeBox_Bryant_Render_C_1",
            "scene/Lid___Actor": "BP_ShapeBoxLid_Bryant_Render_C_1",
            "scene/Foam___Actor": "StaticMeshActor_5",
            "scene/Table___Actor": "StaticMeshActor_0",
            # Robot actor
            (
                "scene/BP_Robot_FR3_DG5F_Right_C_1___LuminMochiRobot___Articulation"
            ): "BP_Robot_Arroyo_C_1",
        }
    )
    """Mapping from mochi actor names to Unreal Engine actor instance names or blueprints.

    Values can be:
    - Pre-placed actor name: "BP_Cross_C_1" (no spawning, just reference)
    - Blueprint with auto name: "blueprint:///Game/Path/To/BP_Actor"
    - Blueprint with desired name: "blueprint:///Game/Path/To/BP_Actor@DesiredName"
    - Prefab by name: "prefab://<prefabName>" (resolved to a render class on the
      Unreal side). This is produced internally by the bot-task flow
      (``build_task_actor_mapping``), not hand-authored in YAML. The spawned UE
      name defaults to the mapping key unless a "@DesiredName" suffix is given.
    """

    articulated_actor_mapping: dict[str, dict] = field(
        default_factory=lambda: {
            # Mapping for articulated actors where joint angles drive bone rotations
            "scene/BP_Robot_FR3_DG5F_Right_C_1___LuminMochiRobot___Articulation": {
                "ue_actor": "BP_Robot_Arroyo_C_1",
                "link_to_bone": {
                    # Robot base
                    "base": "base",
                    "arm_base": "arm_base",
                    # Franka FR3 arm links
                    "fr3_link0": "fr3_link0",
                    "fr3_link1": "fr3_link1",
                    "fr3_link2": "fr3_link2",
                    "fr3_link3": "fr3_link3",
                    "fr3_link4": "fr3_link4",
                    "fr3_link5": "fr3_link5",
                    "fr3_link6": "fr3_link6",
                    "fr3_link7": "fr3_link7",
                    "fr3_link8": "fr3_link8",
                    # DG5F hand base/mount links
                    "right_base_link": "right_base_link",
                    "dg5f_link_mount": "dg5f_link_mount",
                    "dg5f_link_base": "dg5f_link_base",
                    "dg5f_link_palm": "dg5f_link_palm",
                    # Finger 5 (pinky)
                    "dg5f_link_5_1": "dg5f_link_5_1",
                    "dg5f_link_5_2": "dg5f_link_5_2",
                    "dg5f_link_5_3": "dg5f_link_5_3",
                    "dg5f_link_5_4": "dg5f_link_5_4",
                    "dg5f_link_5_tip": "dg5f_link_5_tip",
                    # Finger 4 (ring)
                    "dg5f_link_4_1": "dg5f_link_4_1",
                    "dg5f_link_4_2": "dg5f_link_4_2",
                    "dg5f_link_4_3": "dg5f_link_4_3",
                    "dg5f_link_4_4": "dg5f_link_4_4",
                    "dg5f_link_4_tip": "dg5f_link_4_tip",
                    # Finger 3 (middle)
                    "dg5f_link_3_1": "dg5f_link_3_1",
                    "dg5f_link_3_2": "dg5f_link_3_2",
                    "dg5f_link_3_3": "dg5f_link_3_3",
                    "dg5f_link_3_4": "dg5f_link_3_4",
                    "dg5f_link_3_tip": "dg5f_link_3_tip",
                    # Finger 2 (index)
                    "dg5f_link_2_1": "dg5f_link_2_1",
                    "dg5f_link_2_2": "dg5f_link_2_2",
                    "dg5f_link_2_3": "dg5f_link_2_3",
                    "dg5f_link_2_4": "dg5f_link_2_4",
                    "dg5f_link_2_tip": "dg5f_link_2_tip",
                    # Finger 1 (thumb)
                    "dg5f_link_1_1": "dg5f_link_1_1",
                    "dg5f_link_1_2": "dg5f_link_1_2",
                    "dg5f_link_1_3": "dg5f_link_1_3",
                    "dg5f_link_1_4": "dg5f_link_1_4",
                    "dg5f_link_1_tip": "dg5f_link_1_tip",
                },
            },
        }
    )
    """Mapping from mochi articulated actors to UE skeletal meshes.

    Each value is a dict with:
    - ue_actor: Instance name or blueprint (same format as actor_mapping)
    - link_to_bone: Dict mapping mochi link names to UE bone names

    Optional FK metadata (used by smynth_generate / make_dataset to drive
    multi-articulated FK; each entry's FK asset can be configured
    independently):
    - prefab_path: Path (relative to ``MOCHI_ASSETS_PATH``) to a
      ``.mochi_scene`` prefab containing this articulated actor.
    - bot_scene_path: Alternative to ``prefab_path``; path to a
      ``.mochi_bot_scene`` whose ``bots[*].bot_name`` entry is loaded for
      FK. Mirrors the eval-side ``bot_scene_path`` flow.
    - bot_name: For ``bot_scene_path``: the bot's name. For
      ``prefab_path``: a substring used to locate the articulated actor
      inside the prefab.
    - track_link: Optional mochi link name. When set, the UE actor is
      driven as a single rigid xform that follows the named link's
      world pose, instead of via ``poseable_xforms``. Use this when the
      UE skeletal mesh's bone-local frames don't match mochi's
      parent-relative bone frames (most hand-authored UE skeletons),
      which would otherwise cause motion in the wrong direction.
    - track_link_offset_rot_deg: Optional 3-element ``[rx, ry, rz]``
      Euler angles (degrees, XYZ extrinsic order) describing a constant
      rotation offset to apply to the tracked link's world rotation, in
      the link's local frame. Used to align a UE mesh whose authored
      "rest" orientation differs from the mochi link's by a fixed
      rotation (e.g. ``[0, 90, 0]`` for a 90 deg yaw correction).
      Only consulted when ``track_link`` is also set.
    - track_link_offset_rot_quat: Optional 4-element ``[qx, qy, qz, qw]``
      quaternion offset (overrides ``track_link_offset_rot_deg`` if both
      are present). Use when the required rotation cannot be expressed
      cleanly in Euler angles. At setup, smynth_generate logs the
      tracked link's rest-pose world rotation so you can copy its
      conjugate ``[-qx, -qy, -qz, qw]`` here to make the actor's rest
      orientation identity.
    """


def _parse_actor_value(value: str) -> tuple[str, str, str | None]:
    """Parse an actor mapping value.

    Args:
        value: Actor mapping value (e.g., "blueprint://path@name",
            "prefab://name", or a plain "ActorName")

    Returns:
        Tuple of (kind, path_or_name, desired_name)
        - kind: "actor" (pre-placed), "blueprint", or "prefab"
        - path_or_name: blueprint asset path (blueprint), mochi prefab name
          (prefab), or instance name (actor)
        - desired_name: Desired instance name (from "@name" suffix), or None
    """
    if value.startswith(BLUEPRINT_PREFIX):
        spec = value[len(BLUEPRINT_PREFIX) :]
        if "@" in spec:
            blueprint_path, desired_name = spec.split("@", 1)
            return "blueprint", blueprint_path, desired_name
        return "blueprint", spec, None

    if value.startswith(PREFAB_PREFIX):
        spec = value[len(PREFAB_PREFIX) :]
        if "@" in spec:
            prefab_name, desired_name = spec.split("@", 1)
            return "prefab", prefab_name, desired_name
        return "prefab", spec, None

    return "actor", value, None


def _get_mapped_instance_name(
    unrealcvactor_cfg: UnrealCVActorConfig, mochi_name: str
) -> tuple[str | None, str, str | None]:
    """Get the UE instance info for a SuperDex Physics actor.

    Checks both actor_mapping and articulated_actor_mapping for the instance name.

    Args:
        unrealcvactor_cfg: UnrealCV actor configuration
        mochi_name: SuperDex Physics actor name

    Returns:
        Tuple of (instance_name, kind, spawn_arg)
        - instance_name: Desired UE instance name (if specified), or None
        - kind: "actor", "blueprint", or "prefab"
        - spawn_arg: Blueprint asset path (blueprint) or mochi prefab name
          (prefab); None for pre-placed actors
    """
    value: str | None = None
    if mochi_name in unrealcvactor_cfg.actor_mapping:
        value = unrealcvactor_cfg.actor_mapping[mochi_name]
    elif mochi_name in unrealcvactor_cfg.articulated_actor_mapping:
        value = unrealcvactor_cfg.articulated_actor_mapping[mochi_name].get("ue_actor")

    if not value:
        return None, "actor", None

    kind, path_or_name, desired_name = _parse_actor_value(value)
    if kind == "blueprint":
        return desired_name, "blueprint", path_or_name
    if kind == "prefab":
        # The spawned UE name defaults to the mapping key (the mochi/spawn name)
        # when no explicit "@DesiredName" suffix is given.
        return desired_name or mochi_name, "prefab", path_or_name
    return path_or_name, "actor", None


def _spawn_blueprint(
    client: UnrealCVClient,
    blueprint_path: str,
    desired_name: str | None,
) -> str | None:
    """Spawn a blueprint in UE and return the instance name.

    Args:
        client: UnrealCV client
        blueprint_path: Blueprint asset path
        desired_name: Desired instance name (optional)

    Returns:
        Spawned instance name if successful, None otherwise
    """
    if desired_name:
        logger.info(
            f"Spawning blueprint '{blueprint_path}' with desired name '{desired_name}'"
        )
        response = client._request(
            f"vset /objects/spawn_blueprint {blueprint_path} {desired_name}"
        )
    else:
        logger.info(f"Spawning blueprint '{blueprint_path}' (auto-generated name)")
        response = client._request(f"vset /objects/spawn_blueprint {blueprint_path}")

    logger.info(f"Spawn response: '{response}'")

    if response and "error" not in response.lower():
        return response.strip()
    return None


def _spawn_prefab(
    client: UnrealCVClient,
    prefab_name: str,
    desired_name: str | None,
) -> str | None:
    """Spawn a prefab in UE by name and return the spawned instance name.

    The Unreal side resolves ``prefab_name`` to a render class via its prefab
    render registry (see ``vset /objects/spawn_prefab``).

    Args:
        client: UnrealCV client
        prefab_name: SuperDex Physics prefab name
        desired_name: Desired instance name (optional)

    Returns:
        Spawned instance name if successful, None otherwise
    """
    if desired_name:
        logger.info(
            f"Spawning prefab '{prefab_name}' with desired name '{desired_name}'"
        )
        response = client._request(
            f"vset /objects/spawn_prefab {prefab_name} {desired_name}"
        )
    else:
        logger.info(f"Spawning prefab '{prefab_name}' (auto-generated name)")
        response = client._request(f"vset /objects/spawn_prefab {prefab_name}")

    logger.info(f"Spawn response: '{response}'")

    if response and "error" not in response.lower():
        return response.strip()
    return None


def _update_actor_mappings(
    unrealcvactor_cfg: UnrealCVActorConfig,
    mochi_name: str,
    ue_instance_name: str,
) -> None:
    """Update actor_mapping and articulated_actor_mapping with spawned instance name.

    Replaces blueprint:// values with the actual spawned instance name.

    Args:
        unrealcvactor_cfg: UnrealCV actor configuration
        mochi_name: SuperDex Physics actor name
        ue_instance_name: Spawned UE instance name
    """
    GREEN = "\033[92m"
    RESET = "\033[0m"

    if mochi_name in unrealcvactor_cfg.articulated_actor_mapping:
        unrealcvactor_cfg.articulated_actor_mapping[mochi_name]["ue_actor"] = (
            ue_instance_name
        )
        logger.info(
            f"{GREEN}Updated articulated_actor_mapping['ue_actor'] to '{ue_instance_name}'{RESET}"
        )
    elif mochi_name in unrealcvactor_cfg.actor_mapping:
        unrealcvactor_cfg.actor_mapping[mochi_name] = ue_instance_name
        logger.info(f"{GREEN}Updated actor_mapping to '{ue_instance_name}'{RESET}")


def _process_single_actor(
    unrealcvactor_cfg: UnrealCVActorConfig,
    mochi_name: str,
    existing_ue_actors: set[str],
    client: UnrealCVClient,
) -> bool:
    """Process a single SuperDex Physics actor: verify existence or spawn blueprint/prefab.

    Args:
        unrealcvactor_cfg: UnrealCV actor configuration
        mochi_name: SuperDex Physics actor name
        existing_ue_actors: Set of UE actor names currently in the scene
        client: UnrealCV client

    Returns:
        True if actor was spawned, False otherwise
    """
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    logger.info(f"Processing actor '{mochi_name}'")

    # Get instance info
    mapped_instance, kind, spawn_arg = _get_mapped_instance_name(
        unrealcvactor_cfg, mochi_name
    )

    # Pre-placed actor: just reference it, nothing to spawn
    if kind == "actor":
        if mapped_instance and mapped_instance in existing_ue_actors:
            logger.info(
                f"{GREEN}'{mochi_name}' -> '{mapped_instance}' exists in UE (pre-placed){RESET}"
            )
        elif mapped_instance:
            logger.error(
                f"{RED}'{mochi_name}' -> '{mapped_instance}' NOT found in UE (expected pre-placed){RESET}"
            )
        else:
            logger.warning(f"{RED}'{mochi_name}' has no mapping (skipping){RESET}")
        return False

    # Blueprint / prefab spawning path
    if not spawn_arg:
        logger.error(f"{RED}'{mochi_name}' has {kind} prefix but no path/name{RESET}")
        return False

    # Check if already exists
    if mapped_instance and mapped_instance in existing_ue_actors:
        logger.warning(
            f"{YELLOW}'{mochi_name}' -> '{mapped_instance}' already exists, updating mapping{RESET}"
        )
        _update_actor_mappings(unrealcvactor_cfg, mochi_name, mapped_instance)
        return False

    # Spawn the blueprint or prefab
    logger.info(f"{YELLOW}Spawning {kind} '{spawn_arg}' for '{mochi_name}'{RESET}")
    if kind == "prefab":
        ue_instance_name = _spawn_prefab(client, spawn_arg, mapped_instance)
    else:
        ue_instance_name = _spawn_blueprint(client, spawn_arg, mapped_instance)

    if ue_instance_name:
        # Check name mismatch
        if mapped_instance and ue_instance_name != mapped_instance:
            logger.warning(
                f"{YELLOW}Requested name '{mapped_instance}' but UE spawned '{ue_instance_name}' (name in use?){RESET}"
            )

        # Update mappings with actual spawned name
        _update_actor_mappings(unrealcvactor_cfg, mochi_name, ue_instance_name)

        logger.info(f"{GREEN}✓ Spawned '{ue_instance_name}' for '{mochi_name}'{RESET}")
        return True
    else:
        logger.error(f"{RED}✗ Failed to spawn {kind} for '{mochi_name}'{RESET}")
        return False


def spawn_and_map_actors(
    unrealcvactor_cfg: UnrealCVActorConfig, client: UnrealCVClient
) -> int:
    """Spawn UE actors for SuperDex Physics actors marked with blueprint:// or prefab:// prefix.

    Processes all actors in actor_mapping and articulated_actor_mapping.
    Values starting with "blueprint://" or "prefab://" trigger spawning, others
    are assumed to be pre-placed actors and are skipped.

    Args:
        unrealcvactor_cfg: UnrealCV actor configuration
        client: UnrealCV client connected to UE

    Returns:
        Number of actors spawned
    """
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    # Collect all SuperDex Physics actor names from the mappings
    mochi_actor_names = set(unrealcvactor_cfg.actor_mapping.keys())
    mochi_actor_names.update(unrealcvactor_cfg.articulated_actor_mapping.keys())

    if not mochi_actor_names:
        logger.warning(f"{RED}No actors in config, nothing to spawn{RESET}")
        return 0

    logger.info(f"Processing {len(mochi_actor_names)} actors from config")

    # Query existing actors
    response = client._request("vget /objects")
    if not response:
        logger.error("Failed to get object list from UE")
        return 0

    existing_ue_actors = set(response.split())
    logger.info(f"Found {len(existing_ue_actors)} existing UE actors")

    # Process each actor
    spawned_count = 0
    for mochi_name in mochi_actor_names:
        if _process_single_actor(
            unrealcvactor_cfg, mochi_name, existing_ue_actors, client
        ):
            spawned_count += 1

    logger.info(f"{GREEN}Spawned {spawned_count} actors total{RESET}")
    return spawned_count


##########################################################################################
# Material Domain Randomization
##########################################################################################


def _resolve_override_value(
    override_val,
    global_range: tuple[float, float] | None,
    rng: np.random.Generator | None = None,
) -> float | None:
    """Resolve an override field to a concrete float.

    - "default" -> use the global range (or None if global is also null)
    - None      -> explicitly skip (always returns None)
    - float     -> use as-is (fixed value)
    - object with .low/.high -> sample uniformly from the per-actor range

    ``rng`` is the random source for range sampling; when None it falls back to
    the module-global ``np.random`` (legacy, non-deterministic). Determinism-
    critical callers pass a seeded ``Generator`` (e.g. ``keyed_rng(seed, key)``).
    """
    r = rng if rng is not None else np.random
    if override_val is None:
        return None
    if isinstance(override_val, str) and override_val == "default":
        if global_range is not None:
            return float(r.uniform(global_range[0], global_range[1]))
        return None
    if (
        isinstance(override_val, dict)
        and "low" in override_val
        and "high" in override_val
    ):
        return float(r.uniform(override_val["low"], override_val["high"]))
    if hasattr(override_val, "low") and hasattr(override_val, "high"):
        return float(r.uniform(override_val.low, override_val.high))
    return float(override_val)


def _material_cmds(
    actor_name: str,
    prop: str,
    values: str,
    component: str | list[str] | None,
) -> list[str]:
    """Build material ``vset`` command(s), optionally scoped to child component(s).

    Per-component routes take the component name as the first arg after the
    property (mirroring the ObjectHandler.cpp overloads):
    ``vset /object/<actor>/<prop> <component> <values>``. A None/empty component
    yields the whole-actor form ``vset /object/<actor>/<prop> <values>``.

    When ``component`` is a list, the SAME ``values`` are emitted once per named
    component, so every listed component receives an identical result (they move
    together). This differs from separate override entries for those components,
    which each sample independently.
    """
    if not component:
        return [f"vset /object/{actor_name}/{prop} {values}"]
    components = component if isinstance(component, list) else [component]
    return [f"vset /object/{actor_name}/{prop} {c} {values}" for c in components]


def _build_rgb_color_cmd(
    actor_name: str,
    override: Any,
    rng: np.random.Generator | None = None,
    component: str | list[str] | None = None,
) -> list[str]:
    r = _resolve_override_value(override.red, None, rng)
    g = _resolve_override_value(override.green, None, rng)
    b = _resolve_override_value(override.blue, None, rng)
    a = _resolve_override_value(override.alpha, None, rng)
    if r is None and g is None and b is None and a is None:
        return []
    # Channels resolved to None are sent as the keep-previous sentinel so the
    # ObjectHandler.cpp side reads the existing material color and only
    # overwrites the explicitly specified channels.
    r = r if r is not None else KEEP_PREVIOUS_TOKEN
    g = g if g is not None else KEEP_PREVIOUS_TOKEN
    b = b if b is not None else KEEP_PREVIOUS_TOKEN
    a = a if a is not None else KEEP_PREVIOUS_TOKEN
    return _material_cmds(actor_name, "rgba", f"{r} {g} {b} {a}", component)


def _build_hsv_color_cmd(
    actor_name: str,
    override: Any | None,
    hue_range: Any | None,
    sat_range: Any | None,
    val_range: Any | None,
    rng: np.random.Generator | None = None,
    component: str | list[str] | None = None,
) -> list[str]:
    h = _resolve_override_value(override.hue if override else "default", hue_range, rng)
    s = _resolve_override_value(
        override.saturation if override else "default", sat_range, rng
    )
    v = _resolve_override_value(
        override.value if override else "default", val_range, rng
    )
    a = _resolve_override_value(override.alpha if override else "default", None, rng)
    if h is None and s is None and v is None and a is None:
        return []
    # Channels resolved to None are sent as the keep-previous sentinel so the
    # ObjectHandler.cpp side reads the existing material color (in HSV) and
    # only overwrites the explicitly specified channels.
    h = h if h is not None else KEEP_PREVIOUS_TOKEN
    s = s if s is not None else KEEP_PREVIOUS_TOKEN
    v = v if v is not None else KEEP_PREVIOUS_TOKEN
    a = a if a is not None else KEEP_PREVIOUS_TOKEN
    return _material_cmds(actor_name, "hsva", f"{h} {s} {v} {a}", component)


def _classify_dr_response(
    cmd: str,
    resp: str | None,
) -> tuple[str, str | None]:
    """Classify a DR batch response into a category.

    Returns ``(category, resp)`` where category is one of
    ``"ok"``, ``"warning"``, ``"soft_fail"``, ``"failed"``.
    """
    if not resp:
        return ("failed", resp)
    lower = resp.lower()
    if lower == "ok":
        return ("ok", resp)
    if "missing" in resp and "/" in resp:
        if " 0/" in resp:
            return ("soft_fail", resp)
        return ("warning", resp)
    if lower.startswith("error"):
        return ("failed", resp)
    return ("warning", resp)


_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_ORANGE = "\033[38;5;208m"
_RED = "\033[91m"
_RESET = "\033[0m"


def _is_numeric_token(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _cmd_label(cmd: str) -> str:
    """Extract 'ActorName property [component]' from a vset command.

    Per-component material commands carry the target component name as the first
    arg after the property (a non-numeric token). Include it so a failure is
    attributable to the specific child component, not just the actor.
    """
    parts = cmd.split()
    if len(parts) >= 2:
        segments = parts[1].split("/")
        if len(segments) >= 4:
            label = f"{segments[2]} {segments[3]}"
            if len(parts) >= 3 and not _is_numeric_token(parts[2]):
                label += f" [{parts[2]}]"
            return label
    return cmd


def _resp_diag(resp: str) -> str:
    """Strip error prefix, keep diagnostics."""
    idx = resp.find("materials set")
    if idx >= 0:
        start = resp.rfind(":", 0, idx)
        return resp[start + 1 :].strip() if start >= 0 else resp
    return resp


# Actors already reported as absent from the loaded map. DR configs deliberately
# name actors from more than one map variant (e.g. the ``_CAT_N`` curtains) so a
# single config works on either level, so whichever name is not in the loaded map
# fails on every reset -- ~867 identical ERROR lines over a 200-rollout eval.
# Lazily initialized (see .llms/rules/python.md on module-scope state).
_ABSENT_ACTORS_REPORTED: set[str] | None = None


def _cmd_actor(cmd: str) -> str:
    """Actor name from a ``vset /object/<actor>/<prop> ...`` command."""
    parts = cmd.split()
    segments = parts[1].split("/") if len(parts) >= 2 else []
    return segments[2] if len(segments) >= 3 else cmd


def _split_absent_actor_failures(
    kind: str,
    failed: list[tuple[str, str | None]],
) -> list[tuple[str, str | None]]:
    """Report each absent actor once per process; return the other failures.

    Keyed on the actor, not the command, because every DR property targeting a
    missing actor reports the same single fact. A genuinely mistyped actor name
    is still surfaced -- once, at ERROR, naming the actor -- rather than hidden.
    """
    global _ABSENT_ACTORS_REPORTED
    if _ABSENT_ACTORS_REPORTED is None:
        _ABSENT_ACTORS_REPORTED = set()
    remaining: list[tuple[str, str | None]] = []
    for cmd, resp in failed:
        if not (resp and "can not find object" in resp.lower()):
            remaining.append((cmd, resp))
            continue
        actor = _cmd_actor(cmd)
        if actor in _ABSENT_ACTORS_REPORTED:
            continue
        _ABSENT_ACTORS_REPORTED.add(actor)
        logger.error(
            "%s%s: '%s' is not in the loaded map, so its DR commands are no-ops "
            "on every reset. Reported once per process. First command: %s%s",
            _RED,
            kind,
            actor,
            cmd,
            _RESET,
        )
    return remaining


def _classify_dr_responses(
    commands: list[str],
    responses: list[str | None],
) -> dict[str, list]:
    """Classify DR batch responses into buckets."""
    result: dict[str, list] = {
        "ok": [],
        "warning": [],
        "soft_fail": [],
        "failed": [],
    }
    for cmd, resp in zip(commands, responses):
        cat, r = _classify_dr_response(cmd, resp)
        if cat == "ok":
            result["ok"].append(cmd)
        else:
            result[cat].append((cmd, r))
    return result


def _pick_severity_color(
    failed: list,
    soft_fails: list,
    warnings: list,
) -> str:
    if failed:
        return _RED
    if soft_fails:
        return _ORANGE
    if warnings:
        return _YELLOW
    return _GREEN


def _log_material_dr_results(
    commands: list[str],
    responses: list[str | None],
    actor_names: list[str],
) -> None:
    buckets = _classify_dr_responses(commands, responses)
    succeeded = buckets["ok"]
    warnings = buckets["warning"]
    soft_fails = buckets["soft_fail"]
    failed = buckets["failed"]

    for cmd in succeeded:
        logger.info("%sMaterial DR OK: %s%s", _GREEN, _cmd_label(cmd), _RESET)
    for cmd, resp in warnings:
        logger.warning(
            "%sMaterial DR WARNING: %s — %s%s",
            _YELLOW,
            _cmd_label(cmd),
            _resp_diag(resp),
            _RESET,
        )
    for cmd, resp in soft_fails:
        logger.warning(
            "%sMaterial DR UNSUPPORTED: %s — %s%s",
            _ORANGE,
            _cmd_label(cmd),
            _resp_diag(resp),
            _RESET,
        )
    # A missing actor is a hard error, reported once per process by the helper.
    # Everything left (e.g. an existing actor whose material rejects the command)
    # stays at warning, since material DR is best-effort per field.
    for cmd, resp in _split_absent_actor_failures("Material DR", failed):
        logger.warning(
            "%sMaterial DR FAILED: %s — %s%s",
            _RED,
            _cmd_label(cmd),
            resp,
            _RESET,
        )

    total_ok = len(succeeded) + len(warnings)
    color = _pick_severity_color(failed, soft_fails, warnings)
    logger.info(
        "%sMaterial DR: %d/%d commands succeeded "
        "(%d warnings, %d unsupported) for %d actors%s",
        color,
        total_ok,
        len(commands),
        len(warnings),
        len(soft_fails),
        len(actor_names),
        _RESET,
    )


def _resolve_scalar_cmd(
    actor_name: str,
    prop: str,
    override_val: Any,
    global_range: Any,
    rng: np.random.Generator | None = None,
    component: str | list[str] | None = None,
) -> list[str]:
    val = _resolve_override_value(override_val, global_range, rng)
    if val is not None:
        return _material_cmds(actor_name, prop, f"{val}", component)
    return []


def _resolve_explicit_only_cmd(
    actor_name: str,
    prop: str,
    override_val: Any,
    global_range: Any,
    rng: np.random.Generator | None = None,
) -> str | None:
    """Resolve a command only when explicitly set (not 'default')."""
    if override_val is None or override_val == "default":
        return None
    val = _resolve_override_value(override_val, global_range, rng)
    if val is not None:
        return f"vset /object/{actor_name}/{prop} {val}"
    return None


def _get_global_ranges(material_cfg: Any) -> dict[str, Any]:
    if not material_cfg:
        return {}
    return {
        "hue": material_cfg.get_hue(),
        "saturation": material_cfg.get_saturation(),
        "value": material_cfg.get_value(),
        "roughness": material_cfg.get_roughness(),
        "metallic": material_cfg.get_metallic(),
        "specular": material_cfg.get_specular(),
        "light_intensity": material_cfg.get_light_intensity(),
        "opacity": material_cfg.get_opacity(),
    }


def _build_scalar_param_commands(actor_name: str, override: Any) -> list[str]:
    """Generic named scalar material params -> vset .../scalar_param <name> <v>.

    Escape hatch for material params not covered by the semantic fields. Each
    value is resolved like any override (fixed / range / null-skip); ``None``
    skips. Param names must be single-token (no spaces).
    """
    params = getattr(override, "scalar_params", None) or {}
    commands: list[str] = []
    for name, spec in params.items():
        val = _resolve_override_value(spec, None)
        if val is not None:
            commands.append(f"vset /object/{actor_name}/scalar_param {name} {val}")
    return commands


def _build_vector_param_commands(actor_name: str, override: Any) -> list[str]:
    """Generic named vector (RGB) material params -> vset .../vector_param <name> <r> <g> <b>.

    Each value is a 3-element ``[r, g, b]`` list of fixed floats and/or ranges.
    The C++ sets RGB directly (no keep-previous), so unresolved channels default
    to 0.0. Param names must be single-token (no spaces).
    """
    params = getattr(override, "vector_params", None) or {}
    commands: list[str] = []
    for name, channels in params.items():
        rgb: list[float] = []
        for spec in list(channels)[:3]:
            val = _resolve_override_value(spec, None)
            rgb.append(0.0 if val is None else val)
        rgb += [0.0] * (3 - len(rgb))
        commands.append(
            f"vset /object/{actor_name}/vector_param {name} {rgb[0]} {rgb[1]} {rgb[2]}"
        )
    return commands


def _build_exclusive_color_texture_cmds(
    actor_name: str,
    override: Any,
    ranges: dict,
    texture_probability: float,
    rng: np.random.Generator | None = None,
    component: str | list[str] | None = None,
) -> list[str]:
    """Exclusive per-rep hue-vs-texture selection for a randomizable material.

    Each rep picks EITHER a texture swap OR a flat hue tint, never both:

    - texture mode (``uniform() < texture_probability``): sample ``texture_index``
      from its range and reset the tint to neutral white (``hsva 0 0 1 1``) so the
      texture shows its true colors and does not inherit a hue tint left over from
      a previous hue-mode rep (MIDs persist between reps).
    - hue mode (otherwise): tint the flat-white slot with the sampled
      ``hue``/``saturation``/``value`` and send ``texture_index 0.0``, giving a
      flat pure-color surface.

    Assumes a material authored as ``FinalColor = SelectedTexture × ColorTint``
    with texture index 0 = flat white. ``roughness``/``metallic``/``specular``
    are orthogonal and emitted by the caller in both modes.
    """
    r = rng if rng is not None else np.random
    commands: list[str] = []
    if float(r.uniform()) < float(texture_probability):
        # Texture mode: sampled texture variant + neutral white tint reset.
        ov_tex = getattr(override, "texture_index", "default")
        if ov_tex is None or ov_tex == "default":
            ov_tex = {"low": 0.0, "high": 1.0}
        tex = _resolve_override_value(ov_tex, None, rng)
        if tex is not None:
            commands.extend(
                _material_cmds(actor_name, "texture_index", f"{tex}", component)
            )
        commands.extend(
            _material_cmds(actor_name, "hsva", "0.0 0.0 1.0 1.0", component)
        )
    else:
        # Hue mode: sampled tint on the flat-white texture slot (index 0).
        commands.extend(
            _build_hsv_color_cmd(
                actor_name,
                override,
                ranges.get("hue"),
                ranges.get("saturation"),
                ranges.get("value"),
                rng,
                component,
            )
        )
        commands.extend(_material_cmds(actor_name, "texture_index", "0.0", component))
    return commands


def _build_actor_material_commands(
    actor_name: str,
    override: Any,
    ranges: dict,
    rng: np.random.Generator | None = None,
    component: str | list[str] | None = None,
) -> list[str]:
    """Build all material DR commands for a single override.

    ``component`` scopes every command to one or more named child mesh
    components (a list applies ONE sampled result to all of them). ``None`` =
    whole-actor. Light-only properties (``light_intensity`` / ``source_radius``)
    have no per-component UnrealCV route, so they are emitted only whole-actor.

    When ``override.texture_probability`` is set, color + texture_index follow
    the exclusive per-rep hue-vs-texture contract (see
    ``_build_exclusive_color_texture_cmds``); otherwise hue/tint and
    texture_index are applied independently (the original behavior).
    """
    commands: list[str] = []

    texture_probability = (
        getattr(override, "texture_probability", None) if override else None
    )
    if texture_probability is not None:
        commands.extend(
            _build_exclusive_color_texture_cmds(
                actor_name,
                override,
                ranges,
                texture_probability,
                rng,
                component,
            )
        )
    elif override and override.uses_rgb:
        commands.extend(_build_rgb_color_cmd(actor_name, override, rng, component))
    else:
        commands.extend(
            _build_hsv_color_cmd(
                actor_name,
                override,
                ranges.get("hue"),
                ranges.get("saturation"),
                ranges.get("value"),
                rng,
                component,
            )
        )

    for prop, range_key in [
        ("roughness", "roughness"),
        ("metallic", "metallic"),
        ("specular", "specular"),
    ]:
        commands.extend(
            _resolve_scalar_cmd(
                actor_name,
                prop,
                getattr(override, prop, "default") if override else "default",
                ranges.get(range_key),
                rng,
                component,
            )
        )

    # Texture index (only when explicitly set). Normalized 0..1; the C++ side
    # maps it to a concrete index via the material's TextureCount parameter.
    # In exclusive mode texture_index is already handled above, so skip here.
    ov_tex = getattr(override, "texture_index", "default") if override else "default"
    if texture_probability is None and ov_tex is not None and ov_tex != "default":
        tex = _resolve_override_value(ov_tex, None, rng)
        if tex is not None:
            commands.extend(
                _material_cmds(actor_name, "texture_index", f"{tex}", component)
            )

    # Light-only properties: whole-actor only (no per-component UnrealCV route).
    if component is None:
        c = _resolve_explicit_only_cmd(
            actor_name,
            "light_intensity",
            override.light_intensity if override else "default",
            ranges.get("light_intensity"),
            rng,
        )
        if c is not None:
            commands.append(c)

        c = _resolve_explicit_only_cmd(
            actor_name,
            "source_radius",
            override.source_radius if override else "default",
            None,
            rng,
        )
        if c is not None:
            commands.append(c)

    c = _resolve_explicit_only_cmd(
        actor_name,
        "opacity",
        getattr(override, "opacity", None) if override else "default",
        ranges.get("opacity"),
    )
    if c is not None:
        commands.append(c)

    commands.extend(_build_scalar_param_commands(actor_name, override))
    commands.extend(_build_vector_param_commands(actor_name, override))

    return commands


def apply_material_dr(
    client: UnrealCVClient,
    unrealcvactor_cfg: UnrealCVActorConfig,
    material_cfg: Any = None,
    rng: np.random.Generator | None = None,
) -> None:
    """Apply material domain randomization to actors listed in material_overrides.

    For each actor in ``material_overrides``:

    1. If a property has a fixed value, use it.
    2. If a property has a RangeConfig, sample from that per-actor range.
    3. If a property is None, fall back to the global *material_cfg* range
       (or skip if global is also null).

    Color can be specified as HSV (hue/saturation/value) or RGB
    (red/green/blue). They are mutually exclusive per actor — if any RGB
    field is set, RGBA is used; otherwise HSVA.

    Args:
        client: Connected UnrealCVClient.
        unrealcvactor_cfg: Config containing ``material_overrides``.
        material_cfg: Optional global randomization ranges.
        rng: Optional seeded RNG for deterministic sampling; falls back to the
            module-global ``np.random`` when None.
    """
    overrides: dict = getattr(unrealcvactor_cfg, "material_overrides", {})
    if not overrides:
        return

    actor_names = list(overrides.keys())
    if not actor_names:
        return

    ranges = _get_global_ranges(material_cfg)
    commands: list[str] = []

    for actor_name, override in overrides.items():
        # Whole-actor material DR (original behavior).
        commands.extend(
            _build_actor_material_commands(actor_name, override, ranges, rng)
        )

        # Additive per-child / per-group overrides, applied on top. Each group's
        # ``names`` share ONE sampled result (fanned out to one command per
        # component); separate groups sample independently.
        for group in getattr(override, "components", None) or []:
            names = getattr(group, "names", None)
            if not names:
                logger.warning(
                    "%sDR config: material_overrides['%s'].components entry has "
                    "no 'names'; skipping.%s",
                    _YELLOW,
                    actor_name,
                    _RESET,
                )
                continue
            commands.extend(
                _build_actor_material_commands(
                    actor_name, group, ranges, rng, component=names
                )
            )

    if not commands:
        return

    result = client._request_batch(commands)
    responses: list[str | None] = result[0] if isinstance(result, tuple) else result
    _log_material_dr_results(commands, responses, actor_names)


def _resolve_scale_axis(
    per_axis: float | None,
    uniform: float | None,
) -> float:
    if per_axis is not None:
        return per_axis
    if uniform is not None:
        return uniform
    return 1.0


def _log_scale_dr_results(
    commands: list[str],
    responses: list[str | None],
    actor_names: list[str],
) -> None:
    succeeded = []
    failed = []
    for cmd, resp in zip(commands, responses):
        if resp and resp.lower() == "ok":
            succeeded.append(cmd)
        else:
            failed.append((cmd, resp))

    for cmd in succeeded:
        logger.info("%sScale DR OK: %s%s", _GREEN, cmd, _RESET)
    for cmd, resp in _split_absent_actor_failures("Scale DR", failed):
        logger.error("%sScale DR FAILED: %s -> %s%s", _RED, cmd, resp, _RESET)

    color = _GREEN if not failed else _YELLOW
    logger.info(
        "%sScale DR: %d/%d commands succeeded for %d actors%s",
        color,
        len(succeeded),
        len(commands),
        len(actor_names),
        _RESET,
    )


def apply_scale_dr(
    client: UnrealCVClient,
    unrealcvactor_cfg: UnrealCVActorConfig,
    rng: np.random.Generator | None = None,
) -> None:
    """Apply scale overrides to actors listed in scale_overrides.

    For each actor, resolves uniform ``scale`` and per-axis
    ``scale_x``/``scale_y``/``scale_z`` fields. Per-axis values take
    precedence; unspecified axes fall back to the uniform value (or 1.0).

    Actors whose override sets a non-null ``scale_group`` share a single
    scale sampled once per call: the first member of a group (in dict order)
    samples from its range, and later members with the same group reuse that
    exact ``(sx, sy, sz)`` instead of resampling. This keeps duplicate props
    that are the same size in reality (e.g. two foam pieces) identically
    scaled instead of drawing independent values.

    Args:
        client: Connected UnrealCVClient.
        unrealcvactor_cfg: Config containing ``scale_overrides``.
        rng: Optional seeded RNG for deterministic sampling; falls back to the
            module-global ``np.random`` when None.
    """
    overrides: dict = getattr(unrealcvactor_cfg, "scale_overrides", {})
    if not overrides:
        return

    commands: list[str] = []
    actor_names: list[str] = []
    group_scales: dict[str, tuple[float, float, float]] = {}

    for actor_name, override in overrides.items():
        actor_names.append(actor_name)
        if override is None:
            continue

        group = getattr(override, "scale_group", None)
        if group is not None and group in group_scales:
            sx, sy, sz = group_scales[group]
        else:
            uniform = _resolve_override_value(override.scale, None, rng)
            sx = _resolve_scale_axis(
                _resolve_override_value(override.scale_x, None, rng), uniform
            )
            sy = _resolve_scale_axis(
                _resolve_override_value(override.scale_y, None, rng), uniform
            )
            sz = _resolve_scale_axis(
                _resolve_override_value(override.scale_z, None, rng), uniform
            )
            if group is not None:
                group_scales[group] = (sx, sy, sz)
        commands.append(f"vset /object/{actor_name}/scale {sx} {sy} {sz}")

    if not commands:
        return

    result = client._request_batch(commands)
    responses: list[str | None] = result[0] if isinstance(result, tuple) else result
    _log_scale_dr_results(commands, responses, actor_names)


##########################################################################################
# MPC (Material Parameter Collection) Domain Randomization
##########################################################################################


def _resolve_mpc_value(val, rng: np.random.Generator | None = None) -> float | None:
    """Resolve a single MPC parameter value to a float.

    ``rng`` is the random source for range sampling; when None it falls back to
    the module-global ``np.random`` (legacy, non-deterministic).
    """
    r = rng if rng is not None else np.random
    if val is None:
        return None
    if isinstance(val, dict) and "low" in val and "high" in val:
        return float(r.uniform(val["low"], val["high"]))
    if hasattr(val, "low") and hasattr(val, "high"):
        return float(r.uniform(val.low, val.high))
    return float(val)


def _log_mpc_results(
    commands: list[str],
    responses: list[str | None],
    actor_names: list[str],
) -> None:
    """Log MPC DR command results."""
    _GREEN = "\033[92m"
    _RED = "\033[91m"
    _RESET = "\033[0m"

    succeeded = []
    failed = []
    for cmd, resp in zip(commands, responses):
        if resp and resp.lower() == "ok":
            succeeded.append(cmd)
        else:
            failed.append((cmd, resp))

    for cmd in succeeded:
        param_info = cmd.split("/mpc_scalar ")[1] if "/mpc_scalar " in cmd else cmd
        logger.info("%sMPC DR OK: %s%s", _GREEN, param_info, _RESET)
    for cmd, resp in _split_absent_actor_failures("MPC DR", failed):
        logger.error("%sMPC DR FAILED: %s -> %s%s", _RED, cmd, resp, _RESET)

    logger.info(
        "%sMPC DR: %d/%d commands succeeded for %d actors%s",
        _GREEN if not failed else _RED,
        len(succeeded),
        len(commands),
        len(actor_names),
        _RESET,
    )


def apply_mpc_dr(
    client: UnrealCVClient,
    unrealcvactor_cfg: UnrealCVActorConfig,
    rng: np.random.Generator | None = None,
) -> None:
    """Apply MPC scalar parameter overrides.

    For each actor in ``mpc_overrides``, sets scalar parameters on the
    MaterialParameterCollection owned by that actor.

    Args:
        client: Connected UnrealCVClient.
        unrealcvactor_cfg: Config containing ``mpc_overrides``.
        rng: Optional seeded RNG for deterministic sampling; falls back to the
            module-global ``np.random`` when None.
    """
    overrides: dict = getattr(unrealcvactor_cfg, "mpc_overrides", {})
    if not overrides:
        return

    commands: list[str] = []
    actor_names: list[str] = []

    for actor_name, params in overrides.items():
        actor_names.append(actor_name)
        if not params:
            continue
        for param_name, val in params.items():
            resolved = _resolve_mpc_value(val, rng)
            if resolved is None:
                continue
            commands.append(
                f"vset /object/{actor_name}/mpc_scalar {param_name} {resolved}"
            )

    if not commands:
        return

    result = client._request_batch(commands)
    responses: list[str | None] = result[0] if isinstance(result, tuple) else result
    _log_mpc_results(commands, responses, actor_names)


##########################################################################################
# Position Domain Randomization (absolute location and/or offset jitter)
##########################################################################################


def _get_position_field(override: Any, field_name: str) -> Any:
    """Read a field from a position override that may be a dataclass or dict."""
    if isinstance(override, dict):
        return override.get(field_name)
    return getattr(override, field_name, None)


def _parse_location(resp: str | None) -> tuple[float, float, float] | None:
    """Parse a ``vget /object/<name>/location`` response into an (x, y, z) tuple."""
    if not resp or resp.lower().startswith("error"):
        return None
    try:
        parts = resp.split()
        return float(parts[0]), float(parts[1]), float(parts[2])
    except (ValueError, IndexError):
        return None


def _capture_initial_positions(
    client: UnrealCVClient,
    actor_names: list[str],
    initial_positions: dict[str, tuple[float, float, float]],
) -> None:
    """Query and cache initial world positions for actors not yet in the cache."""
    missing = [n for n in actor_names if n not in initial_positions]
    if not missing:
        return

    commands = [f"vget /object/{name}/location" for name in missing]
    result = client._request_batch(commands)
    responses: list[str | None] = result[0] if isinstance(result, tuple) else result

    for name, resp in zip(missing, responses):
        loc = _parse_location(resp)
        if loc is not None:
            initial_positions[name] = loc
        else:
            logger.warning("Position DR: failed to query initial position for %s", name)


def _resolve_position_axis(
    override: Any,
    loc_field: str,
    offset_field: str,
    initial: float,
    rng: np.random.Generator | None = None,
) -> float:
    """Resolve one axis: ``location_*`` (or initial) baseline plus ``offset_*`` jitter."""
    loc = _get_position_field(override, loc_field)
    base = float(loc) if loc is not None else initial
    offset = (
        _resolve_override_value(_get_position_field(override, offset_field), None, rng)
        or 0.0
    )
    return base + offset


def _log_position_dr_results(
    commands: list[str],
    responses: list[str | None],
    actor_names: list[str],
) -> None:
    """Log position DR command results."""
    _GREEN = "\033[92m"
    _RED = "\033[91m"
    _RESET = "\033[0m"

    succeeded = []
    failed = []
    for cmd, resp in zip(commands, responses):
        if resp and resp.lower() == "ok":
            succeeded.append(cmd)
        else:
            failed.append((cmd, resp))

    for cmd in succeeded:
        logger.info("%sPosition DR OK: %s%s", _GREEN, cmd, _RESET)
    for cmd, resp in _split_absent_actor_failures("Position DR", failed):
        logger.error("%sPosition DR FAILED: %s -> %s%s", _RED, cmd, resp, _RESET)

    logger.info(
        "%sPosition DR: %d/%d commands succeeded for %d actors%s",
        _GREEN if not failed else _RED,
        len(succeeded),
        len(commands),
        len(actor_names),
        _RESET,
    )


def apply_position_dr(
    client: UnrealCVClient,
    unrealcvactor_cfg: UnrealCVActorConfig,
    initial_positions: dict[str, tuple[float, float, float]],
    rng: np.random.Generator | None = None,
) -> None:
    """Randomize actor world positions from ``position_overrides``.

    ``location_x/y/z`` set an explicit baseline; ``offset_x/y/z`` jitter from
    it. Missing ``location_*`` components fall back to the actor's initial UE
    position (queried and cached on first call). The two compose:
    ``final = baseline + offset``.

    Args:
        client: Connected UnrealCVClient.
        unrealcvactor_cfg: Config containing ``position_overrides``.
        initial_positions: Mutable cache of actor initial world positions.
            Callers should create one dict and pass it on every call.
        rng: Optional seeded RNG for the ``offset_*`` range draws; falls back to
            the module-global ``np.random`` when None. Determinism-critical
            callers pass ``keyed_rng(seed, "dr/scene/position")``.
    """
    overrides: dict = getattr(unrealcvactor_cfg, "position_overrides", {})
    if not overrides:
        return

    _capture_initial_positions(client, list(overrides.keys()), initial_positions)

    pos_fields = (
        "location_x",
        "location_y",
        "location_z",
        "offset_x",
        "offset_y",
        "offset_z",
    )
    loc_fields = ("location_x", "location_y", "location_z")
    commands: list[str] = []
    actor_names: list[str] = []
    for actor_name, override in overrides.items():
        if all(_get_position_field(override, f) is None for f in pos_fields):
            continue
        # Any axis without an explicit ``location_*`` falls back to the actor's
        # initial UE position as its baseline. If that initial is unavailable
        # (the ``vget`` failed — already warned in _capture_initial_positions),
        # skip this actor rather than teleporting it to world origin + offset.
        needs_initial = any(
            _get_position_field(override, f) is None for f in loc_fields
        )
        if needs_initial and actor_name not in initial_positions:
            logger.warning(
                "Position DR: skipping '%s' — initial position unavailable and "
                "not every axis sets an explicit location_*.",
                actor_name,
            )
            continue
        actor_names.append(actor_name)
        ix, iy, iz = initial_positions.get(actor_name, (0.0, 0.0, 0.0))
        x = _resolve_position_axis(override, "location_x", "offset_x", ix, rng)
        y = _resolve_position_axis(override, "location_y", "offset_y", iy, rng)
        z = _resolve_position_axis(override, "location_z", "offset_z", iz, rng)
        commands.append(f"vset /object/{actor_name}/location {x} {y} {z}")

    if not commands:
        return

    result = client._request_batch(commands)
    responses: list[str | None] = result[0] if isinstance(result, tuple) else result
    _log_position_dr_results(commands, responses, actor_names)


##########################################################################################
# Rotation Domain Randomization (absolute rotation and/or offset jitter)
##########################################################################################


def _rotation_cache_key(actor_name: str, component: str | None) -> str:
    """Cache key for an initial rotation: actor name, or ``actor/component``."""
    return actor_name if not component else f"{actor_name}/{component}"


def _capture_initial_rotations(
    client: UnrealCVClient,
    entries: list[tuple[str, str | None]],
    initial_rotations: dict[str, tuple[float, float, float]],
) -> None:
    """Query and cache initial rotations (pitch, yaw, roll) for uncached entries.

    Each entry is an ``(actor_name, component)`` pair. Actor-level entries
    (``component`` is None) query ``vget /object/<name>/rotation``; per-component
    entries query ``vget /object/<name>/rotation <component>``.
    """
    # Unique uncached (key -> vget command), preserving first-seen order.
    pending: dict[str, str] = {}
    for actor_name, component in entries:
        key = _rotation_cache_key(actor_name, component)
        if key in initial_rotations or key in pending:
            continue
        if component:
            pending[key] = f"vget /object/{actor_name}/rotation {component}"
        else:
            pending[key] = f"vget /object/{actor_name}/rotation"
    if not pending:
        return

    keys = list(pending.keys())
    commands = [pending[k] for k in keys]
    result = client._request_batch(commands)
    responses: list[str | None] = result[0] if isinstance(result, tuple) else result

    for key, resp in zip(keys, responses):
        rot = _parse_location(resp)
        if rot is not None:
            initial_rotations[key] = rot
        else:
            logger.warning("Rotation DR: failed to query initial rotation for %s", key)


def _resolve_rotation_axis(
    override: Any,
    rot_field: str,
    offset_field: str,
    initial: float,
    rng: np.random.Generator | None = None,
) -> float:
    """Resolve one axis: ``rotation_*`` (or initial) baseline plus ``offset_*`` jitter."""
    rot = _get_position_field(override, rot_field)
    base = float(rot) if rot is not None else initial
    offset = (
        _resolve_override_value(_get_position_field(override, offset_field), None, rng)
        or 0.0
    )
    return base + offset


def _log_rotation_dr_results(
    commands: list[str],
    responses: list[str | None],
    actor_names: list[str],
) -> None:
    """Log rotation DR command results."""
    _GREEN = "\033[92m"
    _RED = "\033[91m"
    _RESET = "\033[0m"

    succeeded = []
    failed = []
    for cmd, resp in zip(commands, responses):
        if resp and resp.lower() == "ok":
            succeeded.append(cmd)
        else:
            failed.append((cmd, resp))

    for cmd in succeeded:
        logger.info("%sRotation DR OK: %s%s", _GREEN, cmd, _RESET)
    for cmd, resp in _split_absent_actor_failures("Rotation DR", failed):
        logger.error("%sRotation DR FAILED: %s -> %s%s", _RED, cmd, resp, _RESET)

    logger.info(
        "%sRotation DR: %d/%d commands succeeded for %d actors%s",
        _GREEN if not failed else _RED,
        len(succeeded),
        len(commands),
        len(actor_names),
        _RESET,
    )


def apply_rotation_dr(
    client: UnrealCVClient,
    unrealcvactor_cfg: UnrealCVActorConfig,
    initial_rotations: dict[str, tuple[float, float, float]],
    rng: np.random.Generator | None = None,
) -> None:
    """Randomize actor or child-component rotations from ``rotation_overrides``.

    ``rotation_pitch/yaw/roll`` set an explicit baseline; ``offset_pitch/yaw/roll``
    jitter from it. Missing ``rotation_*`` components fall back to the actor's
    initial rotation (queried and cached on first call). The two compose:
    ``final = baseline + offset``. Axis mapping (UE Rotator convention):
    Pitch = about Y, Yaw = about Z, Roll = about X.

    Each actor maps to a single override or a list of overrides. An entry with a
    ``component`` field targets that named child component's *relative* rotation
    (``vset /object/<actor>/rotation <component> <p> <y> <r>``); otherwise the
    actor's world rotation is set (``vset /object/<actor>/rotation <p> <y> <r>``).
    Each (actor, component) pair caches its own initial rotation.

    Args:
        client: Connected UnrealCVClient.
        unrealcvactor_cfg: Config containing ``rotation_overrides``.
        initial_rotations: Mutable cache of initial rotations, keyed by actor
            name or ``actor/component``. Callers create one dict and pass it
            on every call.
        rng: Optional seeded RNG for the ``offset_*`` range draws; falls back to
            the module-global ``np.random`` when None. Determinism-critical
            callers pass ``keyed_rng(seed, "dr/scene/rotation")``.
    """
    overrides: dict = getattr(unrealcvactor_cfg, "rotation_overrides", {})
    if not overrides:
        return

    rot_fields = (
        "rotation_pitch",
        "rotation_yaw",
        "rotation_roll",
        "offset_pitch",
        "offset_yaw",
        "offset_roll",
    )
    rotation_fields = ("rotation_pitch", "rotation_yaw", "rotation_roll")

    # Flatten to (actor_name, component, override) entries that set something.
    entries: list[tuple[str, str | None, Any]] = []
    for actor_name, value in overrides.items():
        items = value if isinstance(value, list) else [value]
        for override in items:
            if all(_get_position_field(override, f) is None for f in rot_fields):
                continue
            component = _get_position_field(override, "component")
            entries.append((actor_name, component, override))

    if not entries:
        return

    _capture_initial_rotations(
        client, [(a, c) for a, c, _ in entries], initial_rotations
    )
    commands: list[str] = []
    labels: list[str] = []
    for actor_name, component, override in entries:
        key = _rotation_cache_key(actor_name, component)
        # Any axis without an explicit ``rotation_*`` falls back to the
        # initial rotation as its baseline. If that initial is unavailable
        # (the ``vget`` failed - already warned in _capture_initial_rotations),
        # skip this entry rather than orienting it to world-zero + offset.
        needs_initial = any(
            _get_position_field(override, f) is None for f in rotation_fields
        )
        if needs_initial and key not in initial_rotations:
            logger.warning(
                "Rotation DR: skipping '%s' - initial rotation unavailable and "
                "not every axis sets an explicit rotation_*.",
                key,
            )
            continue
        ip, iy, ir = initial_rotations.get(key, (0.0, 0.0, 0.0))
        pitch = _resolve_rotation_axis(
            override, "rotation_pitch", "offset_pitch", ip, rng
        )
        yaw = _resolve_rotation_axis(override, "rotation_yaw", "offset_yaw", iy, rng)
        roll = _resolve_rotation_axis(override, "rotation_roll", "offset_roll", ir, rng)
        if component:
            commands.append(
                f"vset /object/{actor_name}/rotation {component} {pitch} {yaw} {roll}"
            )
        else:
            commands.append(f"vset /object/{actor_name}/rotation {pitch} {yaw} {roll}")
        labels.append(key)

    if not commands:
        return

    result = client._request_batch(commands)
    responses: list[str | None] = result[0] if isinstance(result, tuple) else result
    _log_rotation_dr_results(commands, responses, labels)


##########################################################################################
# Sphere Placement Domain Randomization (random point on a sphere, aimed at center)
##########################################################################################


def _random_point_on_sphere_above(
    radius: float,
    z_floor: float,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Random point on a radius-``radius`` sphere centered at the origin, ``z >= z_floor``.

    Uses the UE5 convention that Z is world up. ``z_floor`` is clamped to
    ``[-radius, radius]``. Samples ``z`` uniformly in ``[z_floor, radius]``, then
    a uniform angle around the resulting horizontal (XY) circle slice. ``rng`` is
    the random source; when None it falls back to the module-global ``np.random``.
    """
    r = rng if rng is not None else np.random
    z_floor = max(min(z_floor, radius), -radius)
    z = float(r.uniform(z_floor, radius))
    slice_r = float(np.sqrt(max(0.0, radius * radius - z * z)))
    angle = float(r.uniform(0.0, 2.0 * np.pi))
    return (slice_r * float(np.cos(angle)), slice_r * float(np.sin(angle)), z)


def _random_aim_offset(
    max_offset: float,
    scalar: float | None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Random horizontal ``(x, y)`` offset added to the look-at target.

    Models imperfect aiming (e.g. a person adjusting a light). ``scalar`` selects
    the distribution within ``max_offset``: when set, each axis is drawn from
    ``N(0, max_offset * scalar)`` and the vector is clamped to ``max_offset`` -- an
    'accuracy'-stat model where small errors dominate and large misses are rare
    (``~0.33`` keeps the clamp rare). When None, the offset is drawn uniformly
    over the disk of radius ``max_offset`` (even coverage out to the cap). Returns
    ``(0.0, 0.0)`` when ``max_offset`` is non-positive (aim exactly at target).
    ``rng`` is the random source; when None it falls back to the module-global
    ``np.random``.
    """
    r = rng if rng is not None else np.random
    if max_offset <= 0.0:
        return (0.0, 0.0)
    if scalar is not None and scalar > 0.0:
        std = max_offset * scalar
        ox = float(r.normal(0.0, std))
        oy = float(r.normal(0.0, std))
        mag = float(np.hypot(ox, oy))
        if mag > max_offset:
            ox *= max_offset / mag
            oy *= max_offset / mag
        return (ox, oy)
    # Area-uniform over the disk: radius ~ max_offset * sqrt(U).
    angle = float(r.uniform(0.0, 2.0 * np.pi))
    radius = max_offset * float(np.sqrt(r.uniform(0.0, 1.0)))
    return (radius * float(np.cos(angle)), radius * float(np.sin(angle)))


def _resolve_sphere_center(
    actor_name: str,
    override: Any,
    ref_positions: dict[str, tuple[float, float, float]],
) -> tuple[float, float, float] | None:
    """Resolve the world sphere center, or None if unresolvable.

    ``center_offset`` is applied only when ``center`` is a named actor (it shifts
    from the queried actor position). When ``center`` is explicit coordinates a
    non-zero ``center_offset`` is meaningless — it is ignored with a warning;
    bake the offset into the coordinates instead.
    """
    center = _get_position_field(override, "center")
    off = _get_position_field(override, "center_offset") or (0.0, 0.0, 0.0)
    if isinstance(center, str):
        base = ref_positions.get(center)
        if base is None:
            return None
        return (
            base[0] + float(off[0]),
            base[1] + float(off[1]),
            base[2] + float(off[2]),
        )
    if center is not None:
        if any(float(o) != 0.0 for o in off):
            logger.warning(
                "Sphere placement DR: '%s' sets center_offset %s but center is "
                "explicit coordinates, not a named actor; the offset is ignored "
                "(bake it into the coordinates instead)",
                actor_name,
                tuple(float(o) for o in off),
            )
        return (float(center[0]), float(center[1]), float(center[2]))
    return None


def _resolve_sphere_min_z(
    actor_name: str,
    override: Any,
    ref_positions: dict[str, tuple[float, float, float]],
) -> float | None:
    """Resolve the world-space minimum Z (UE world up), or None if unresolvable.

    ``min_z_offset`` is applied only when ``min_z`` is a named actor (it shifts
    from the queried actor's world Z). When ``min_z`` is an explicit value a
    non-zero ``min_z_offset`` is meaningless — it is ignored with a warning;
    bake the offset into ``min_z`` instead.
    """
    min_z = _get_position_field(override, "min_z")
    off = float(_get_position_field(override, "min_z_offset") or 0.0)
    if isinstance(min_z, str):
        ref = ref_positions.get(min_z)
        if ref is None:
            return None
        return ref[2] + off
    if min_z is not None:
        if off != 0.0:
            logger.warning(
                "Sphere placement DR: '%s' sets min_z_offset %s but min_z is an "
                "explicit value, not a named actor; the offset is ignored "
                "(bake it into min_z instead)",
                actor_name,
                off,
            )
        return float(min_z)
    return None


def _build_sphere_placement_commands(
    actor_name: str,
    override: Any,
    ref_positions: dict[str, tuple[float, float, float]],
    rng: np.random.Generator | None = None,
) -> list[str]:
    """Sample a point on the sphere above min Z and aim the actor at the center."""
    radius = _get_position_field(override, "radius")
    if radius is None or float(radius) <= 0.0:
        logger.error(
            "%sSphere placement DR: '%s' has missing/invalid radius; skipping%s",
            _RED,
            actor_name,
            _RESET,
        )
        return []
    r = float(radius)

    center = _resolve_sphere_center(actor_name, override, ref_positions)
    world_min_z = _resolve_sphere_min_z(actor_name, override, ref_positions)
    if center is None or world_min_z is None:
        logger.error(
            "%sSphere placement DR: '%s' could not resolve sphere center or "
            "minimum Z (missing coords / reference actor position); skipping%s",
            _RED,
            actor_name,
            _RESET,
        )
        return []

    cx, cy, cz = center
    z_floor = world_min_z - cz
    if z_floor > r:
        logger.error(
            "%sSphere placement DR: '%s' minimum Z (%.2f) exceeds the sphere top "
            "(center Z %.2f + radius %.2f); no valid point, skipping%s",
            _RED,
            actor_name,
            world_min_z,
            cz,
            r,
            _RESET,
        )
        return []

    # Sample on an origin-centered sphere, then translate by the world center.
    lx, ly, lz = _random_point_on_sphere_above(r, z_floor, rng)
    px, py, pz = cx + lx, cy + ly, cz + lz

    # Aim the actor's forward (+X) axis at the target. Exact aim points back at
    # the center (dir = center - point = -local). Optional aim inaccuracy perturbs
    # the target by a random (ox, oy) in its horizontal plane (modelling imperfect
    # aiming), so dir = (center + (ox, oy, 0)) - point = (ox - lx, oy - ly, -lz).
    max_off = float(_get_position_field(override, "aim_inaccuracy_max") or 0.0)
    scalar = _get_position_field(override, "aim_inaccuracy_scalar")
    ox, oy = _random_aim_offset(max_off, None if scalar is None else float(scalar), rng)
    dx, dy, dz = ox - lx, oy - ly, -lz

    # UE5 convention: yaw is about +Z (world up) measured from +X; pitch is the
    # elevation toward +Z. So yaw uses the XY components, pitch uses Z vs. the XY radius.
    yaw = float(np.degrees(np.arctan2(dy, dx)))
    pitch = float(np.degrees(np.arctan2(dz, np.hypot(dx, dy))))

    return [
        f"vset /object/{actor_name}/location {px} {py} {pz}",
        f"vset /object/{actor_name}/rotation {pitch} {yaw} 0.0",
    ]


def _query_sphere_reference_positions(
    client: UnrealCVClient, overrides: dict
) -> dict[str, tuple[float, float, float]]:
    """Query world positions of all center/min-Z reference actors in one batch."""
    ref_actors = sorted(
        {
            value
            for override in overrides.values()
            for value in (
                _get_position_field(override, "center"),
                _get_position_field(override, "min_z"),
            )
            if isinstance(value, str)
        }
    )
    if not ref_actors:
        return {}

    commands = [f"vget /object/{name}/location" for name in ref_actors]
    result = client._request_batch(commands)
    responses: list[str | None] = result[0] if isinstance(result, tuple) else result

    positions: dict[str, tuple[float, float, float]] = {}
    for name, resp in zip(ref_actors, responses):
        loc = _parse_location(resp)
        if loc is not None:
            positions[name] = loc
        else:
            logger.error(
                "%sSphere placement DR: failed to query position for reference "
                "actor '%s'%s",
                _RED,
                name,
                _RESET,
            )
    return positions


def _log_sphere_placement_results(
    commands: list[str],
    responses: list[str | None],
    actor_names: list[str],
) -> None:
    """Log sphere placement DR command results."""
    _GREEN = "\033[92m"
    _RED = "\033[91m"
    _RESET = "\033[0m"

    failed = [
        (cmd, resp)
        for cmd, resp in zip(commands, responses)
        if not (resp and resp.lower() == "ok")
    ]
    for cmd, resp in _split_absent_actor_failures("Sphere placement DR", failed):
        logger.error(
            "%sSphere placement DR FAILED: %s -> %s%s", _RED, cmd, resp, _RESET
        )
    logger.info(
        "%sSphere placement DR: %d/%d commands succeeded for %d actors%s",
        _GREEN if not failed else _RED,
        len(commands) - len(failed),
        len(commands),
        len(actor_names),
        _RESET,
    )


def apply_sphere_placement_dr(
    client: UnrealCVClient,
    unrealcvactor_cfg: UnrealCVActorConfig,
    rng: np.random.Generator | None = None,
) -> None:
    """Place actors on a random sphere point (above a min Z) and aim them at the center.

    For each actor in ``sphere_placement_overrides``, samples a point on the
    configured sphere whose world Z (UE up) is at or above the configured
    minimum, moves the actor there, and rotates it so its +X axis points back at
    the center. Center / minimum-Z reference actor positions are queried fresh
    each call, so placement tracks reference actors that move (e.g. a
    height-randomized table).

    Args:
        client: Connected UnrealCVClient.
        unrealcvactor_cfg: Config containing ``sphere_placement_overrides``.
        rng: Optional seeded RNG for the point-on-sphere + aim-offset draws;
            falls back to the module-global ``np.random`` when None. Determinism-
            critical callers pass ``keyed_rng(seed, "dr/scene/sphere")``.
    """
    overrides: dict = getattr(unrealcvactor_cfg, "sphere_placement_overrides", {})
    if not overrides:
        return

    ref_positions = _query_sphere_reference_positions(client, overrides)

    commands: list[str] = []
    actor_names: list[str] = []
    for actor_name, override in overrides.items():
        actor_cmds = _build_sphere_placement_commands(
            actor_name, override, ref_positions, rng
        )
        if actor_cmds:
            actor_names.append(actor_name)
            commands.extend(actor_cmds)

    if not commands:
        return

    result = client._request_batch(commands)
    responses: list[str | None] = result[0] if isinstance(result, tuple) else result
    _log_sphere_placement_results(commands, responses, actor_names)


##########################################################################################
# Capture DR State
##########################################################################################


def _collect_dr_actors(cfg: UnrealCVActorConfig) -> set[str]:
    """Collect all UE actor names from config mappings and overrides."""
    actors: set[str] = set()
    for v in cfg.actor_mapping.values():
        if isinstance(v, str):
            if "@" in v:
                actors.add(v.split("@", 1)[1])
            elif not v.startswith("blueprint:") and not v.startswith("prefab:"):
                actors.add(v)
    for v in cfg.articulated_actor_mapping.values():
        ue = v.get("ue_actor", "") if isinstance(v, dict) else ""
        if ue:
            if "@" in ue:
                actors.add(ue.split("@", 1)[1])
            elif not ue.startswith("blueprint:") and not ue.startswith("prefab:"):
                actors.add(ue)
    actors.update(getattr(cfg, "material_overrides", {}))
    actors.update(getattr(cfg, "scale_overrides", {}))
    actors.update(getattr(cfg, "mpc_overrides", {}))
    actors.update(getattr(cfg, "position_overrides", {}))
    actors.update(getattr(cfg, "rotation_overrides", {}))
    return actors


def capture_dr_state(
    client: UnrealCVClient,
    unrealcvactor_cfg: UnrealCVActorConfig,
) -> dict[str, Any]:
    """Query the current DR state of all configured actors via batched vget.

    Issues one ``vget /object/<name>/dr_state`` per actor in a single
    ``_request_batch`` call, so the entire capture is one TCP round-trip.

    Returns:
        Dict keyed by actor name, values are property dicts (color_rgba,
        roughness, metallic, specular, texture_index, light_intensity,
        source_radius, light_color_rgba, scale, mpc).
    """
    actors = sorted(_collect_dr_actors(unrealcvactor_cfg))
    if not actors:
        return {}

    commands = [f"vget /object/{name}/dr_state" for name in actors]
    result = client._request_batch(commands)
    responses: list[str | None] = result[0] if isinstance(result, tuple) else result

    state: dict[str, Any] = {}
    for actor_name, resp in zip(actors, responses):
        if not resp or resp.lower().startswith("error"):
            continue
        try:
            parsed = json.loads(resp)
            if parsed:
                state[actor_name] = parsed
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse dr_state JSON for %s", actor_name)

    logger.debug("DR state captured (%d actors)", len(state))
    return state


##########################################################################################
# Visibility Domain Randomization (deterministic user-controlled toggles)
##########################################################################################


def _log_visibility_results(
    commands: list[str],
    responses: list[str | None],
    actor_names: list[str],
) -> None:
    """Log visibility DR command results."""
    _GREEN = "\033[92m"
    _RED = "\033[91m"
    _RESET = "\033[0m"

    succeeded = []
    failed = []
    for cmd, resp in zip(commands, responses):
        if resp and resp.lower() == "ok":
            succeeded.append(cmd)
        else:
            failed.append((cmd, resp))

    for cmd in succeeded:
        logger.info("%sVisibility DR OK: %s%s", _GREEN, cmd, _RESET)
    for cmd, resp in _split_absent_actor_failures("Visibility DR", failed):
        logger.error("%sVisibility DR FAILED: %s -> %s%s", _RED, cmd, resp, _RESET)

    logger.info(
        "%sVisibility DR: %d/%d commands succeeded for %d actors%s",
        _GREEN if not failed else _RED,
        len(succeeded),
        len(commands),
        len(actor_names),
        _RESET,
    )


def _resolve_visibility_overrides(
    unrealcvactor_cfg: UnrealCVActorConfig,
) -> dict[str, bool | float]:
    """Merge legacy ``hidden_actors`` into ``visibility_overrides``.

    Each legacy ``hidden_actors`` name becomes ``{name: False}`` (always hide).
    Explicit ``visibility_overrides`` entries take precedence on conflict.
    """
    global _HIDDEN_ACTORS_DEPRECATION_WARNED
    merged: dict[str, bool | float] = {}
    legacy_hidden = getattr(unrealcvactor_cfg, "hidden_actors", []) or []
    if legacy_hidden and not _HIDDEN_ACTORS_DEPRECATION_WARNED:
        _HIDDEN_ACTORS_DEPRECATION_WARNED = True
        logger.warning(
            "%sDEPRECATED: 'hidden_actors' is being phased out. Migrating %d "
            "entr%s (%s) to visibility_overrides=false. Please move these to "
            "'visibility_overrides' (a {name: bool | probability} map) in your "
            "config.%s",
            _ORANGE,
            len(legacy_hidden),
            "y" if len(legacy_hidden) == 1 else "ies",
            ", ".join(legacy_hidden),
            _RESET,
        )
    for name in legacy_hidden:
        merged[name] = False
    merged.update(getattr(unrealcvactor_cfg, "visibility_overrides", {}) or {})
    return merged


def apply_visibility_dr(
    client: UnrealCVClient,
    unrealcvactor_cfg: UnrealCVActorConfig,
    rng: np.random.Generator | None = None,
) -> None:
    """Show/hide actors from ``visibility_overrides`` (and legacy ``hidden_actors``).

    Each ``actor_name -> value`` entry emits ``vset /object/{name}/show`` or
    ``vset /object/{name}/hide``. The value is either:
    - a bool: ``True`` = show, ``False`` = hide (deterministic), or
    - a float in [0, 1]: probability of *showing* the actor (re-rolled each
      call); ``1.0`` ≡ ``True``, ``0.0`` ≡ ``False``.

    Legacy ``hidden_actors`` names are merged in as ``False`` (always hide),
    with explicit ``visibility_overrides`` entries winning on conflict. An
    empty result is a no-op.

    Args:
        client: Connected UnrealCVClient.
        unrealcvactor_cfg: Config with ``visibility_overrides`` / ``hidden_actors``.
        rng: Optional seeded RNG for the float-probability draws (bool entries
            are deterministic); falls back to the module-global ``np.random``
            when None. Determinism-critical callers pass
            ``keyed_rng(seed, "dr/scene/visibility")``.
    """
    overrides = _resolve_visibility_overrides(unrealcvactor_cfg)
    if not overrides:
        return

    r = rng if rng is not None else np.random
    commands: list[str] = []
    actor_names: list[str] = []

    for actor_name, value in overrides.items():
        if isinstance(value, bool):
            show = value
        else:
            show = float(r.uniform()) < float(value)
        actor_names.append(actor_name)
        commands.append(f"vset /object/{actor_name}/{'show' if show else 'hide'}")

    if not commands:
        return

    result = client._request_batch(commands)
    responses: list[str | None] = result[0] if isinstance(result, tuple) else result
    _log_visibility_results(commands, responses, actor_names)


##########################################################################################
# HDRI-driven lighting (IBL) + reflections, synced with the backdrop dome
##########################################################################################


def _log_named_dr_results(
    name: str,
    commands: list[str],
    responses: list[str | None],
    num_actors: int,
) -> None:
    """Log results of a batch of DR commands under a human-readable ``name``."""
    _GREEN = "\033[92m"
    _RED = "\033[91m"
    _RESET = "\033[0m"

    succeeded = []
    failed = []
    for cmd, resp in zip(commands, responses):
        if resp and resp.lower() == "ok":
            succeeded.append(cmd)
        else:
            failed.append((cmd, resp))

    for cmd in succeeded:
        logger.info("%s%s OK: %s%s", _GREEN, name, cmd, _RESET)
    for cmd, resp in _split_absent_actor_failures(name, failed):
        logger.error("%s%s FAILED: %s -> %s%s", _RED, name, cmd, resp, _RESET)

    logger.info(
        "%s%s: %d/%d commands succeeded for %d actors%s",
        _GREEN if not failed else _RED,
        name,
        len(succeeded),
        len(commands),
        num_actors,
        _RESET,
    )


def apply_hdri_lighting_dr(
    client: UnrealCVClient,
    unrealcvactor_cfg: UnrealCVActorConfig,
    initial_rotations: dict[str, tuple[float, float, float]],
    rng: np.random.Generator | None = None,
) -> None:
    """Drive HDRI-as-IBL lighting + reflections, synced with the backdrop dome.

    Each call samples ONE HDRI index and ONE yaw and uses the SAME cube asset
    to drive both:
    - the backdrop dome (``texture_cube_param`` + ``rotation``), and
    - a Movable, Specified-Cubemap SkyLight (``skylight_cubemap`` +
      ``skylight_angle``),

    so the visible backdrop and the scene's ambient illumination / reflections
    always show the same HDRI at the same orientation. There is no sun coupling:
    the SkyLight provides soft IBL only, and a randomized shadow-casting
    directional light is a separate, later workstream.

    Optional brightness knobs (all fixed float, ``{low, high}`` range, or
    omitted; batched atomically with the cubemap swap):
    - ``skylight_intensity`` — SkyLight IBL strength.
    - ``skylight_color`` — dict with optional ``red`` / ``green`` / ``blue``
      keys, each fixed or range; omitted channels default to 1.0.
    - ``skydome_brightness`` — backdrop emissive multiplier (requires a
      ``Brightness`` scalar parameter on the dome's material, e.g.
      ``M_HDRI_Randomizable``).

    Args:
        client: Connected UnrealCVClient.
        unrealcvactor_cfg: Config containing ``hdri_lighting_override``.
        initial_rotations: Mutable cache of initial rotations (shared with
            ``apply_rotation_dr``); the dome's baseline rotation is queried and
            cached on first call so the per-trial yaw composes onto it.
        rng: Optional seeded RNG for the cubemap pick + yaw / brightness / tint
            range draws; falls back to the module-global ``np.random`` when None.
            Determinism-critical callers pass ``keyed_rng(seed, "dr/scene/hdri")``.
    """
    override = getattr(unrealcvactor_cfg, "hdri_lighting_override", None)
    if override is None:
        return

    skylight = _get_position_field(override, "skylight_actor")
    dome = _get_position_field(override, "dome_actor")
    cubemap_paths = _get_position_field(override, "cubemap_asset_paths") or []
    dome_cube_param = _get_position_field(override, "dome_cube_param") or "EnvCube"
    offset_yaw_cfg = _get_position_field(override, "offset_yaw")
    dome_yaw_sign = _get_position_field(override, "dome_yaw_sign")
    if dome_yaw_sign is None:
        dome_yaw_sign = 1
    skylight_intensity_cfg = _get_position_field(override, "skylight_intensity")
    skylight_color_cfg = _get_position_field(override, "skylight_color")
    skydome_brightness_cfg = _get_position_field(override, "skydome_brightness")

    if not skylight or not dome or not cubemap_paths:
        logger.warning("HDRI lighting DR: incomplete override config; skipping")
        return

    # One shared sample drives BOTH the dome and the SkyLight from the same
    # cube asset, so backdrop and IBL stay in sync. ``np.random.Generator`` uses
    # ``.integers``; the legacy module-global uses ``.randint`` — pick per source.
    if rng is not None:
        idx = int(rng.integers(0, len(cubemap_paths)))
    else:
        idx = int(np.random.randint(0, len(cubemap_paths)))
    yaw = _resolve_override_value(offset_yaw_cfg, None, rng) or 0.0
    cube_path = cubemap_paths[idx]

    # Compose the per-trial yaw onto the dome's cached baseline rotation.
    _capture_initial_rotations(client, [(dome, None)], initial_rotations)
    base_pitch, base_yaw, base_roll = initial_rotations.get(
        _rotation_cache_key(dome, None), (0.0, 0.0, 0.0)
    )
    dome_yaw = base_yaw + dome_yaw_sign * yaw

    commands = [
        f"vset /object/{dome}/texture_cube_param {dome_cube_param} {cube_path}",
        f"vset /object/{dome}/rotation {base_pitch} {dome_yaw} {base_roll}",
        f"vset /object/{skylight}/skylight_cubemap {cube_path}",
        f"vset /object/{skylight}/skylight_angle {yaw}",
    ]

    # Optional brightness / tint knobs, batched with the cubemap swap so the
    # SkyLight recaptures exactly once with the new intensity + color applied.
    skylight_intensity = _resolve_override_value(skylight_intensity_cfg, None, rng)
    if skylight_intensity is not None:
        commands.append(
            f"vset /object/{skylight}/skylight_intensity {skylight_intensity}"
        )

    if skylight_color_cfg is not None:
        r = _resolve_override_value(
            _get_position_field(skylight_color_cfg, "red"), None, rng
        )
        g = _resolve_override_value(
            _get_position_field(skylight_color_cfg, "green"), None, rng
        )
        b = _resolve_override_value(
            _get_position_field(skylight_color_cfg, "blue"), None, rng
        )
        if r is not None or g is not None or b is not None:
            # Any channel omitted defaults to 1.0 (full white on that channel).
            r = r if r is not None else 1.0
            g = g if g is not None else 1.0
            b = b if b is not None else 1.0
            commands.append(f"vset /object/{skylight}/skylight_color {r} {g} {b}")

    skydome_brightness = _resolve_override_value(skydome_brightness_cfg, None, rng)
    if skydome_brightness is not None:
        commands.append(
            f"vset /object/{dome}/scalar_param Brightness {skydome_brightness}"
        )

    result = client._request_batch(commands)
    responses: list[str | None] = result[0] if isinstance(result, tuple) else result
    logger.info("HDRI lighting DR: idx=%d yaw=%.2f deg cube=%s", idx, yaw, cube_path)
    _log_named_dr_results("HDRI lighting DR", commands, responses, 2)


def apply_cast_shadow_dr(
    client: UnrealCVClient,
    unrealcvactor_cfg: UnrealCVActorConfig,
) -> None:
    """Toggle shadow casting on actors from ``cast_shadow_overrides``.

    Each ``actor_name -> bool`` entry emits ``vset /object/{name}/cast_shadow
    {1|0}`` (1 = cast shadows, 0 = no shadows). Primarily used to stop hidden
    lab-room geometry from casting phantom shadows under the (future)
    directional light. An empty mapping is a no-op.

    Args:
        client: Connected UnrealCVClient.
        unrealcvactor_cfg: Config containing ``cast_shadow_overrides``.
    """
    overrides: dict = getattr(unrealcvactor_cfg, "cast_shadow_overrides", {})
    if not overrides:
        return

    commands: list[str] = []
    actor_names: list[str] = []
    for actor_name, enable in overrides.items():
        actor_names.append(actor_name)
        commands.append(f"vset /object/{actor_name}/cast_shadow {1 if enable else 0}")

    if not commands:
        return

    result = client._request_batch(commands)
    responses: list[str | None] = result[0] if isinstance(result, tuple) else result
    _log_named_dr_results("Cast-shadow DR", commands, responses, len(actor_names))


def run_dr_preview_loop(
    apply_once: Callable[[int], None],
    *,
    iters: int,
    delay: float,
    base_seed: int,
) -> None:
    """Loop ``apply_once(seed)`` — one full scene randomization per iteration.

    Shared DR-preview driver for render/eval entrypoints' ``dr_preview`` mode:
    after the entrypoint has connected + spawned actors, it repeatedly re-applies
    the scene domain randomization here (each iteration reseeded to
    ``base_seed + i``) and then exits early, so DR variation can be eyeballed live
    against the exact production DR path.

    ``iters == 0`` loops until the user quits; ``delay == 0`` waits at a prompt
    between iterations (Enter re-randomizes; ``q``/``quit``/``exit`` stops),
    otherwise sleeps ``delay`` seconds. Ctrl-C also stops cleanly.
    """
    i = 0
    try:
        while iters == 0 or i < iters:
            seed = base_seed + i
            total = "" if iters == 0 else f"/{iters}"
            print(f"\n[dr-preview] randomization {i + 1}{total} (seed={seed})")
            apply_once(seed)
            i += 1
            if iters != 0 and i >= iters:
                break
            if delay > 0:
                time.sleep(delay)
            else:
                try:
                    reply = input(
                        "[dr-preview] Press Enter to re-randomize "
                        "(q/quit/exit to stop)... "
                    )
                except EOFError:
                    break
                if reply.strip().lower() in ("q", "quit", "exit"):
                    break
    except KeyboardInterrupt:
        pass
    print("[dr-preview] done.")
