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

"""Rebuild the scene that a recorded episode ran in."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import superdex.physics as sdp
from superdex.physics.utils.scene_helpers import load_bot_scene, load_bot_task_prefab

from . import _extension
from .reader import EpisodeReader
from .restore import _bind_actors

if TYPE_CHECKING:
    import superdex.robotics as sdr  # @manual

_BOT_ARCHIVE_SUFFIXES = (".superdex_bot_archive", ".mochi_bot_archive")
# Unreal recordings of small parts differ from rebuilt masses by up to 2.6e-5, while
# mismatched geometry differs by far more.
_MASS_RELATIVE_TOLERANCE = 1e-4
_PHYSICAL_COMPONENTS = (
    "superdex.physics.contact_params",
    "superdex.physics.density",
    "superdex.physics.mass",
)


@dataclass(frozen=True)
class RebuiltScene:
    """A bot scene rebuilt for a recorded episode."""

    bot_scene: sdr.BotScene
    """The loaded bot scene, which owns the physics scene."""
    spawned_actors: dict[str, list[str]]
    """Runtime names of the actors created for each bot-task spawn, keyed by spawn name."""


def rebuild_scene(
    recording: EpisodeReader,
    bot_scene_path: str | Path,
    robotics_context: sdr.RoboticsContext,
    *,
    bot_task_path: str | Path | None = None,
) -> RebuiltScene:
    """Load a bot scene and a bot task, then apply an episode's recorded setup to it.

    Bots that the recorded slot selections did not choose are removed, as in Unreal.
    Static actors move to their recorded poses, and every recorded body gets its
    recorded contact parameters and density (see :func:`apply_recorded_physics`).
    This reproduces domain randomization applied when the episode was recorded.
    Dynamic state is left to :class:`EpisodeRestorer`.

    When the recording names the scene archive or task file it was recorded with, as
    Unreal recordings do, the given files must have the same content hash. A spawn whose
    prefab the recording names, in a ``scene.prefab.<spawn>`` property, is rebuilt from
    that prefab rather than its nominal one.
    """
    properties = _episode_properties(recording)
    _check_recorded_file(properties, "scene.archive", Path(bot_scene_path))
    if bot_task_path is not None:
        _check_recorded_file(properties, "scene.task", Path(bot_task_path))
    bot_scene = load_bot_scene(str(bot_scene_path), robotics_context)
    _retire_unselected_bots(bot_scene, properties)
    scene = bot_scene.get_scene()
    assert scene is not None
    spawned = (
        {}
        if bot_task_path is None
        else _spawn_bot_task(
            bot_scene,
            scene,
            robotics_context,
            Path(bot_task_path),
            _recorded_prefabs(properties),
        )
    )
    _place_static_actors(recording, scene)
    apply_recorded_physics(recording, scene)
    return RebuiltScene(bot_scene, spawned)


def apply_recorded_physics(recording: EpisodeReader, scene: sdp.Scene) -> None:
    """Give every recorded body its recorded contact parameters and density.

    Afterwards each body's mass must match the recording, which also catches a scene
    whose geometry differs from the one that was recorded.
    """
    names = recording.read_static_value("superdex.physics.contact_param_names")
    template = sdp.ContactParams()
    unknown = [name for name in names if not hasattr(template, name)]
    if unknown:
        raise ValueError(f"The recording has unknown contact parameters {unknown}.")
    bodies = _recorded_bodies(recording, scene)
    samples = recording.component_samples_many(_PHYSICAL_COMPONENTS)

    def body(path: str) -> sdp.Actor:
        actor = bodies.get(path.lstrip("/"))
        if actor is None:
            raise ValueError(f"Scene is missing the recorded body at {path!r}.")
        return actor

    for sample in samples["superdex.physics.contact_params"]:
        actor = body(sample.entity_path)
        contact = actor.get_contact_params()
        for name, value in zip(names, sample.value, strict=True):
            setattr(contact, name, type(getattr(template, name))(value))
        # Recorded as float64, but stored at Mochi's precision, which may be float32.
        requested = [float(getattr(contact, name)) for name in names]
        actor.set_contact_params(contact)
        applied = actor.get_contact_params()
        if [float(getattr(applied, name)) for name in names] != requested:
            raise ValueError(
                f"{actor.get_name()!r} did not accept its recorded contact parameters."
            )
    for sample in samples["superdex.physics.density"]:
        body(sample.entity_path).set_density(sample.value[0])
    for sample in samples["superdex.physics.mass"]:
        actor = body(sample.entity_path)
        if not math.isclose(
            actor.get_mass(), sample.value[0], rel_tol=_MASS_RELATIVE_TOLERANCE
        ):
            raise ValueError(
                f"{actor.get_name()!r} has mass {actor.get_mass()} with its recorded "
                f"density, but {sample.value[0]} was recorded."
            )


def _episode_properties(recording: EpisodeReader) -> dict[str, str]:
    samples = recording.component_samples_many(
        ("superdex.episode.property_key", "superdex.episode.property_value")
    )
    values = {
        sample.entity_path: sample.value[0]
        for sample in samples["superdex.episode.property_value"]
    }
    return {
        sample.value[0]: values[sample.entity_path]
        for sample in samples["superdex.episode.property_key"]
    }


def _check_recorded_file(properties: dict[str, str], prefix: str, path: Path) -> None:
    recorded_hash = properties.get(f"{prefix}.hash")
    if recorded_hash is None:
        return
    algorithm = properties.get(f"{prefix}.hash_algorithm")
    if algorithm != "xxh3_64":
        raise ValueError(f"Unsupported {prefix} hash algorithm {algorithm!r}.")
    if _extension._hash_file(str(path)) != recorded_hash:
        raise ValueError(
            f"{path} differs from the {prefix} the episode was recorded with "
            f"({properties.get(f'{prefix}.filename')!r})."
        )


def _recorded_prefabs(properties: dict[str, str]) -> dict[str, str]:
    prefix = "scene.prefab."
    return {
        key.removeprefix(prefix): value
        for key, value in properties.items()
        if key.startswith(prefix)
    }


def _retire_unselected_bots(
    bot_scene: sdr.BotScene, properties: dict[str, str]
) -> None:
    """Remove the alternatives to the bots the recording selected for its slots.

    A scene slot offers several bots, which may share an articulated actor name. Unreal
    records each slot's choice as ``scene.slot.<slot>`` and retires the offered bots
    that no slot selected, so the rebuilt scene does the same. A bot stays if any slot
    selected it or a slot without a recorded choice offers it, so recordings made before
    slots were recorded keep every bot.
    """
    offered = {
        str(slot.name): [str(bot) for bot in slot.controlled_bots]
        for slot in bot_scene.get_bot_scene_prefab().slots
    }
    selected = {
        slot: properties[f"scene.slot.{slot}"]
        for slot in offered
        if f"scene.slot.{slot}" in properties
    }
    for slot, bot in selected.items():
        if bot not in offered[slot]:
            raise ValueError(
                f"The recording selects bot {bot!r} for slot {slot!r}, which offers "
                f"{offered[slot]}."
            )
    kept = set(selected.values()).union(
        *(bots for slot, bots in offered.items() if slot not in selected)
    )
    for slot in selected:
        for bot in offered[slot]:
            if bot not in kept and bot_scene.get_bot(bot) is not None:
                bot_scene.remove_bot(bot)


def _spawn_bot_task(
    bot_scene: sdr.BotScene,
    scene: sdp.Scene,
    robotics_context: sdr.RoboticsContext,
    task_path: Path,
    recorded_prefabs: dict[str, str],
) -> dict[str, list[str]]:
    import superdex.robotics as sdr  # @manual

    task = load_bot_task_prefab(str(task_path))
    prefab_paths = {
        str(entry.name): str(entry.path)
        for entry in bot_scene.get_bot_scene_prefab().scene.spawnable_prefabs
    }
    spawns = {str(spawn.name): spawn for spawn in task.reset_state}
    world_from_spawns: dict[str, sdp.TransformRT] = {}

    # The loader rejects parent cycles and unknown parents.
    def world_from_spawn(name: str) -> sdp.TransformRT:
        if name not in world_from_spawns:
            parent = str(spawns[name].parent)
            world_from_parent = (
                sdp.TransformRT()
                if parent in ("", "task_root")
                else world_from_spawn(parent)
            )
            world_from_spawns[name] = world_from_parent * spawns[name].parent_from_spawn
        return world_from_spawns[name]

    spawned = {}
    for name, spawn in spawns.items():
        prefab_name = recorded_prefabs.get(name, str(spawn.prefab_name))
        prefab_path = prefab_paths.get(prefab_name)
        if prefab_path is None:
            raise ValueError(
                f"Spawn {name!r} uses prefab {prefab_name!r}, which the bot "
                f"scene does not declare. Available: {sorted(prefab_paths)}."
            )
        pose = world_from_spawn(name)
        if prefab_path.endswith(_BOT_ARCHIVE_SUFFIXES):
            bot_prefab = sdr.load_bot_prefab_from_file(prefab_path)
            bot_prefab.name = name
            bot_prefab.world_from_root = pose
            bot = sdr.create_bot(scene, bot_prefab, robotics_context)
            actor = None if bot is None else bot.get_articulated_actor()
            if actor is None:
                raise ValueError(f"Could not create bot {name!r} from {prefab_path}.")
            spawned[name] = [actor.get_name()]
            continue
        params = sdp.prefab.PrefabParams()
        params.name = name
        params.translation = pose.translation
        params.rotation = pose.rotation
        params.apply_scene_settings = False
        result = sdp.prefab.add_to_scene(
            prefab_path, os.path.dirname(prefab_path), scene, params
        )
        spawned[name] = [actor.get_name() for actor in result.actors]
    return spawned


def _place_static_actors(recording: EpisodeReader, scene: sdp.Scene) -> None:
    columns = recording._component_columns(
        (
            "superdex.physics.root_translation_xyz",
            "superdex.physics.root_quaternion_xyzw",
        )
    )
    first_values: dict[str, dict[str, Any]] = {}
    for (path, component), rows in columns.items():
        first_value = rows["value"][0].as_py()
        first_values.setdefault(path.lstrip("/"), {})[component] = first_value
    for recorded in _bind_actors(recording, scene):
        if not recorded.is_static:
            continue
        pose = first_values.get(recorded.path, {})
        rotation = pose.get("superdex.physics.root_quaternion_xyzw")
        translation = pose.get("superdex.physics.root_translation_xyz")
        if rotation is None or translation is None:
            raise ValueError(
                f"Static actor {recorded.actor.get_name()!r} has no recorded pose."
            )
        recorded.actor.set_root_transform(
            sdp.TransformRT(rotation=rotation, translation=translation)
        )
        # A moved static actor sweeps to its new pose over the next step, which a zero
        # step never completes. Zeroing its velocity places it there instead.
        recorded.actor.set_velocity([0, 0, 0], [0, 0, 0])


def _recorded_bodies(
    recording: EpisodeReader, scene: sdp.Scene
) -> dict[str, sdp.Actor]:
    bodies: dict[str, sdp.Actor] = {}
    links_by_world_path: dict[str, dict[str, sdp.Actor]] = {}
    for recorded in _bind_actors(recording, scene):
        bodies[recorded.path] = recorded.actor
        if not recorded.is_articulated:
            continue
        links: dict[str, sdp.Actor] = {}
        for handle in recorded.actor.get_nested_link_actors():
            link = scene.get_actor(handle)
            if link is not None:
                links[link.get_name().rsplit("/", 1)[-1]] = link
        world_path = recorded.path.replace("data/physics/actors/", "world/", 1)
        links_by_world_path[world_path] = links
    for sample in recording.component_samples("superdex.physics.link_name"):
        path = sample.entity_path.lstrip("/")
        world_path = path.split("/links/", 1)[0]
        link = links_by_world_path.get(world_path, {}).get(sample.value[0])
        if link is not None:
            bodies[path] = link
    return bodies
