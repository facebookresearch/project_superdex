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

import gc
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import superdex.physics as sdp
import superdex.recorder as recorder
import superdex.robotics as sdr
from superdex.physics.paths import resolve_asset
from superdex.physics.utils.scene_helpers import load_bot_scene
from superdex.recorder.reader import EpisodeReader
from superdex.recorder.rebuild import _retire_unselected_bots

BOT_SCENE_PATH = resolve_asset(
    "bots/internal/tests/bot_scene/test_bot_scene.mochi_bot_scene"
)
SKINNED_HAND_PATH = resolve_asset(
    "bots/hands/wuji_hand2_beta1/left/wuji_hand2_beta1_skinned_left.superdex_bot"
)
SPAWN_NAME = "shape_filter"
# Another of `BOT_SCENE_PATH`'s spawnable prefabs, as a PrefabSelection could draw.
SELECTED_PREFAB_NAME = "clamp"
SPAWN_TRANSLATION = [0.05, 0.0, 0.0]
RANDOMIZED_SPAWN_TRANSLATION = [0.06, 0.01, 0.0]
FRICTION_SCALE = 1.4
DENSITY_SCALE = 0.8
NUM_FRAMES = 100
NUM_RESUME_FRAMES = 20
# Rebuilt bodies resumed from the last recorded frame stay this close to the live run
# [m]. Without the recorded friction and density, the check fails.
RESUME_TOLERANCE = 1e-5
TIME_STEP = 0.005
KNOB_ARCHIVE = (
    "internal/archives/task_items/Knob_M8_50mm_4273T84/"
    "Knob_M8_50mm_4273T84.superdex_bot_archive"
)
KNOB_SPAWN_NAME = "knob"
# `BOT_SCENE_PATH`'s knob pose, just above the foam.
KNOB_ROTATION = [0.0, -math.sqrt(0.5), -math.sqrt(0.5), 0.0]
KNOB_TRANSLATION = [0.0075, -0.245, 0.8355]
SLOT_ARM_PATH = "@bots/internal/t3/right/t3_right.superdex_bot"
# Alternatives for one slot that share the arm's articulated actor name.
SLOT_BOTS = ("t3_right", "t3_right_osc_v2")


def physics_scene(bot_scene: sdr.BotScene) -> sdp.Scene:
    scene = bot_scene.get_scene()
    assert scene is not None
    return scene


def scene_actors(scene: sdp.Scene) -> dict[str, sdp.Actor]:
    actors: dict[str, sdp.Actor] = {}
    scene.for_each_actor(lambda actor: actors.__setitem__(actor.get_name(), actor))
    return actors


def colliding_bodies(scene: sdp.Scene) -> dict[str, sdp.Actor]:
    return {
        name: actor
        for name, actor in scene_actors(scene).items()
        if actor.get_type() == sdp.ActorType.RIGID
        and actor.get_collider_type() != sdp.ColliderType.NONE
    }


def randomize_physics(
    scene: sdp.Scene,
    friction_scale: float = FRICTION_SCALE,
    density_scale: float = DENSITY_SCALE,
) -> None:
    for actor in colliding_bodies(scene).values():
        contact = actor.get_contact_params()
        contact.coulomb_friction_coefficient *= friction_scale
        contact.viscous_friction_coefficient *= friction_scale
        actor.set_contact_params(contact)
        if not actor.is_static():
            actor.set_density(density_scale * actor.get_density())


def physical_parameters(
    actor: sdp.Actor, contact_param_names: list[str]
) -> dict[str, list[float]]:
    contact = actor.get_contact_params()
    parameters = {"contact": [getattr(contact, name) for name in contact_param_names]}
    if actor.is_static():
        pose = actor.get_root_transform()
        parameters["pose"] = pose.translation.tolist() + pose.rotation.tolist()
    else:
        parameters["mass_properties"] = [actor.get_density(), actor.get_mass()]
    return parameters


class RebuildSceneTest(unittest.TestCase):
    def setUp(self) -> None:
        sdp.initialize(num_worker_threads=0)
        self.addCleanup(sdp.shutdown)
        self.addCleanup(gc.collect)
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.context = sdr.create_context()
        self.task_path = self.directory / "shape_filter.mochi_bot_task"
        self.task_path.write_text(
            json.dumps(
                {
                    "metadata": {"name": "shape_filter"},
                    "resetState": [
                        {
                            "name": SPAWN_NAME,
                            "prefabName": SPAWN_NAME,
                            "parent": "task_root",
                            "parentFromSpawn": {
                                "rotation": [0, 0, 0, 1],
                                "translation": SPAWN_TRANSLATION,
                            },
                        }
                    ],
                }
            )
        )

    def load_live_scene(
        self,
        spawn_translation: list[float] = SPAWN_TRANSLATION,
        scale: float = 1.0,
        prefab_name: str = SPAWN_NAME,
    ) -> tuple[sdr.BotScene, list[sdp.Actor]]:
        """Load the bot scene and spawn the task like Unreal does: randomized spawn poses
        apply as it spawns, and spawned bodies keep their prefab names. Return the scene
        and the spawned bodies."""
        bot_scene = load_bot_scene(str(BOT_SCENE_PATH), self.context)
        prefab_path = next(
            str(entry.path)
            for entry in bot_scene.get_bot_scene_prefab().scene.spawnable_prefabs
            if str(entry.name) == prefab_name
        )
        params = sdp.prefab.PrefabParams()
        params.translation = spawn_translation
        params.scale = scale
        params.apply_scene_settings = False
        result = sdp.prefab.add_to_scene(
            prefab_path, os.path.dirname(prefab_path), physics_scene(bot_scene), params
        )
        return bot_scene, list(result.actors)

    def record(
        self,
        scene: sdp.Scene,
        spawned: list[sdp.Actor],
        num_frames: int,
        metadata: recorder.EpisodeMetadata | None = None,
        spawn_name: str = SPAWN_NAME,
    ) -> tuple[Path, Path]:
        episode_recorder = recorder.Recorder(recorder.RecorderConfig(self.directory))
        episodes = [
            episode_recorder.begin_episode(metadata or recorder.EpisodeMetadata())
            for _ in range(2)
        ]
        configs = [recorder.PhysicsRecorderConfig() for _ in episodes]
        for config in configs:
            config.spawn_names = [(actor.get_handle(), spawn_name) for actor in spawned]
        configs[1].record_contact_forces = True
        recorders = [
            recorder.PhysicsRecorder(episode, scene, config)
            for episode, config in zip(episodes, configs, strict=True)
        ]
        for frame in range(num_frames + 1):
            if frame > 0:
                scene.step(TIME_STEP)
            for physics in recorders:
                physics.record(frame, scene.get_total_simulation_time())
        for episode in episodes:
            episode.commit()
        return episodes[0].output_path, episodes[1].output_path

    def test_rebuilt_scene_has_randomized_physical_parameters(self) -> None:
        live, spawned = self.load_live_scene(RANDOMIZED_SPAWN_TRANSLATION)
        live_scene = physics_scene(live)
        randomize_physics(live_scene)
        kinematic_path, _ = self.record(live_scene, spawned, NUM_FRAMES)
        live_bodies = colliding_bodies(live_scene)

        recording = EpisodeReader(kinematic_path)
        rebuilt = recorder.rebuild_scene(
            recording, BOT_SCENE_PATH, self.context, bot_task_path=self.task_path
        )
        scene = physics_scene(rebuilt.bot_scene)
        rebuilt_names = {
            actor.get_name(): f"{SPAWN_NAME}/{actor.get_name()}" for actor in spawned
        }
        rebuilt_bodies = colliding_bodies(scene)
        bodies = {
            name: rebuilt_bodies[rebuilt_names.get(name, name)] for name in live_bodies
        }
        self.assertCountEqual(
            rebuilt.spawned_actors[SPAWN_NAME], rebuilt_names.values()
        )
        contact_param_names = recording.read_static_value(
            "superdex.physics.contact_param_names"
        )
        for name, live_body in live_bodies.items():
            with self.subTest(body=name):
                self.assertEqual(
                    physical_parameters(bodies[name], contact_param_names),
                    physical_parameters(live_body, contact_param_names),
                )

    def test_rebuilt_scene_resumes_randomized_episode(self) -> None:
        live, spawned = self.load_live_scene(RANDOMIZED_SPAWN_TRANSLATION)
        live_scene = physics_scene(live)
        randomize_physics(live_scene)
        kinematic_path, _ = self.record(live_scene, spawned, NUM_FRAMES)
        live_bodies = {
            name: body
            for name, body in colliding_bodies(live_scene).items()
            if not body.is_static() and not body.is_nested_link_actor()
        }
        live_translations = []
        for _ in range(NUM_RESUME_FRAMES):
            live_scene.step(TIME_STEP)
            live_translations.append(
                {
                    name: body.get_center_of_mass_transform().translation.tolist()
                    for name, body in live_bodies.items()
                }
            )

        recording = EpisodeReader(kinematic_path)
        rebuilt_names = {
            actor.get_name(): f"{SPAWN_NAME}/{actor.get_name()}" for actor in spawned
        }

        def resume_error(undo_randomization: bool) -> float:
            rebuilt = recorder.rebuild_scene(
                recording, BOT_SCENE_PATH, self.context, bot_task_path=self.task_path
            )
            scene = physics_scene(rebuilt.bot_scene)
            if undo_randomization:
                randomize_physics(scene, 1 / FRICTION_SCALE, 1 / DENSITY_SCALE)
            rebuilt_bodies = colliding_bodies(scene)
            bodies = {
                name: rebuilt_bodies[rebuilt_names.get(name, name)]
                for name in live_bodies
            }
            recorder.EpisodeRestorer(recording, scene).restore_frame(NUM_FRAMES)
            error = 0.0
            for expected in live_translations:
                scene.step(TIME_STEP)
                for name, translation in expected.items():
                    actual = bodies[name].get_center_of_mass_transform().translation
                    error = max(
                        error,
                        float(np.abs(np.subtract(actual.tolist(), translation)).max()),
                    )
            return error

        self.assertLess(resume_error(undo_randomization=False), RESUME_TOLERANCE)
        self.assertGreater(resume_error(undo_randomization=True), RESUME_TOLERANCE)

    def test_applies_recorded_contact_parameters_of_a_skinned_hand(self) -> None:
        """A skinned articulation keeps its contact skin's parameters on the root."""
        prefab = sdr.load_bot_prefab_from_file(str(SKINNED_HAND_PATH))
        live_scene = sdp.create_scene("live")
        live_hand = sdr.create_bot(
            live_scene, prefab, self.context
        ).get_articulated_actor()
        contact = live_hand.get_contact_params()
        contact.coulomb_friction_coefficient *= FRICTION_SCALE
        live_hand.set_contact_params(contact)
        path, _ = self.record(live_scene, [], num_frames=1)

        rebuilt_scene = sdp.create_scene("rebuilt")
        rebuilt_hand = sdr.create_bot(
            rebuilt_scene, prefab, self.context
        ).get_articulated_actor()
        self.assertNotEqual(
            rebuilt_hand.get_contact_params().coulomb_friction_coefficient,
            contact.coulomb_friction_coefficient,
        )
        recording = EpisodeReader(path)
        recorder.apply_recorded_physics(recording, rebuilt_scene)

        names = recording.read_static_value("superdex.physics.contact_param_names")
        self.assertEqual(
            [getattr(rebuilt_hand.get_contact_params(), name) for name in names],
            [getattr(live_hand.get_contact_params(), name) for name in names],
        )

    def test_rebuilds_bot_spawned_by_task(self) -> None:
        # Bot scenes can't use absolute paths, so a tag locates the assets.
        (self.directory / ".superdex_root").write_text(
            json.dumps({"@bots": str(resolve_asset("bots"))})
        )
        bot_scene_path = self.directory / "knob.mochi_bot_scene"
        bot_scene_path.write_text(
            json.dumps(
                {
                    "metadata": {"name": "knob"},
                    "scene": {
                        "baseScene": "@bots/internal/tests/bot_scene/base_scene/"
                        "base_scene.mochi_scene",
                        "spawnablePrefabs": [
                            {"name": "knob", "path": f"@bots/{KNOB_ARCHIVE}"}
                        ],
                    },
                }
            )
        )
        task_path = self.directory / "knob.mochi_bot_task"
        task_path.write_text(
            json.dumps(
                {
                    "metadata": {"name": "knob"},
                    "resetState": [
                        {
                            "name": KNOB_SPAWN_NAME,
                            "prefabName": "knob",
                            "parent": "task_root",
                            "parentFromSpawn": {
                                "rotation": KNOB_ROTATION,
                                "translation": KNOB_TRANSLATION,
                            },
                        }
                    ],
                }
            )
        )
        live = load_bot_scene(str(bot_scene_path), self.context)
        live_scene = physics_scene(live)
        prefab = sdr.load_bot_prefab_from_file(
            str(resolve_asset(f"bots/{KNOB_ARCHIVE}"))
        )
        prefab.world_from_root = sdp.TransformRT(
            rotation=KNOB_ROTATION, translation=KNOB_TRANSLATION
        )
        bot = sdr.create_bot(live_scene, prefab, self.context)
        knob = None if bot is None else bot.get_articulated_actor()
        assert knob is not None
        randomize_physics(live_scene)
        kinematic_path, _ = self.record(
            live_scene, [knob], NUM_FRAMES, spawn_name=KNOB_SPAWN_NAME
        )

        recording = EpisodeReader(kinematic_path)
        rebuilt = recorder.rebuild_scene(
            recording, bot_scene_path, self.context, bot_task_path=task_path
        )
        self.assertEqual(rebuilt.spawned_actors, {KNOB_SPAWN_NAME: [KNOB_SPAWN_NAME]})
        scene = physics_scene(rebuilt.bot_scene)
        rebuilt_bodies = colliding_bodies(scene)
        contact_param_names = recording.read_static_value(
            "superdex.physics.contact_param_names"
        )
        for name, live_body in colliding_bodies(live_scene).items():
            rebuilt_name = name.replace(f"{knob.get_name()}/", f"{KNOB_SPAWN_NAME}/", 1)
            with self.subTest(body=name):
                self.assertEqual(
                    physical_parameters(
                        rebuilt_bodies[rebuilt_name], contact_param_names
                    ),
                    physical_parameters(live_body, contact_param_names),
                )

    def test_accepts_files_matching_recorded_hashes(self) -> None:
        live, spawned = self.load_live_scene()
        properties = {}
        for prefix, path in (
            ("scene.archive", BOT_SCENE_PATH),
            ("scene.task", self.task_path),
        ):
            properties[f"{prefix}.filename"] = path.name
            properties[f"{prefix}.hash"] = recorder._extension._hash_file(str(path))
            properties[f"{prefix}.hash_algorithm"] = "xxh3_64"
        kinematic_path, _ = self.record(
            physics_scene(live), spawned, 1, recorder.EpisodeMetadata(properties)
        )
        rebuilt = recorder.rebuild_scene(
            EpisodeReader(kinematic_path),
            BOT_SCENE_PATH,
            self.context,
            bot_task_path=self.task_path,
        )
        self.assertCountEqual(rebuilt.spawned_actors, [SPAWN_NAME])

    def test_rebuilds_spawn_from_its_recorded_prefab(self) -> None:
        live, spawned = self.load_live_scene(prefab_name=SELECTED_PREFAB_NAME)
        selected_path, _ = self.record(
            physics_scene(live),
            spawned,
            1,
            recorder.EpisodeMetadata(
                {f"scene.prefab.{SPAWN_NAME}": SELECTED_PREFAB_NAME}
            ),
        )
        rebuilt = recorder.rebuild_scene(
            EpisodeReader(selected_path),
            BOT_SCENE_PATH,
            self.context,
            bot_task_path=self.task_path,
        )
        self.assertCountEqual(
            rebuilt.spawned_actors[SPAWN_NAME],
            [f"{SPAWN_NAME}/{actor.get_name()}" for actor in spawned],
        )

        # Without the property the spawn falls back to its nominal prefab.
        unnamed_path, _ = self.record(physics_scene(live), spawned, 1)
        with self.assertRaisesRegex(sdp.Error, "missing recorded actor"):
            recorder.rebuild_scene(
                EpisodeReader(unnamed_path),
                BOT_SCENE_PATH,
                self.context,
                bot_task_path=self.task_path,
            )

    def test_rejects_bot_scene_that_differs_from_recording(self) -> None:
        live, spawned = self.load_live_scene()
        kinematic_path, _ = self.record(
            physics_scene(live),
            spawned,
            1,
            recorder.EpisodeMetadata(
                {
                    "scene.archive.filename": BOT_SCENE_PATH.name,
                    "scene.archive.hash": "0000000000000000",
                    "scene.archive.hash_algorithm": "xxh3_64",
                }
            ),
        )
        with self.assertRaisesRegex(ValueError, "differs from the scene.archive"):
            recorder.rebuild_scene(
                EpisodeReader(kinematic_path),
                BOT_SCENE_PATH,
                self.context,
                bot_task_path=self.task_path,
            )

    def test_rejects_scene_whose_masses_differ_from_recording(self) -> None:
        # Scaled bodies have the same density but more mass.
        live, spawned = self.load_live_scene(scale=1.1)
        kinematic_path, _ = self.record(physics_scene(live), spawned, 1)
        with self.assertRaisesRegex(ValueError, "with its recorded density"):
            recorder.rebuild_scene(
                EpisodeReader(kinematic_path),
                BOT_SCENE_PATH,
                self.context,
                bot_task_path=self.task_path,
            )

    def write_slot_scene(self, slots: list[dict[str, object]] | None = None) -> Path:
        """A bot scene whose right slot offers two bots built from the same arm."""
        (self.directory / ".superdex_root").write_text(
            json.dumps({"@bots": str(resolve_asset("bots"))})
        )
        path = self.directory / "slots.mochi_bot_scene"
        path.write_text(
            json.dumps(
                {
                    "metadata": {"name": "slots"},
                    "scene": {
                        "baseScene": "@bots/internal/tests/bot_scene/base_scene/"
                        "base_scene.mochi_scene"
                    },
                    "bots": [
                        {"name": name, "path": SLOT_ARM_PATH} for name in SLOT_BOTS
                    ],
                    "slots": slots
                    or [{"name": "right", "controlledBots": list(SLOT_BOTS)}],
                }
            )
        )
        return path

    def record_slot_choice(self, scene_path: Path, choice: str) -> Path:
        live = load_bot_scene(str(scene_path), self.context)
        # Unreal retires the alternatives before it records.
        for bot in SLOT_BOTS:
            if bot != choice:
                live.remove_bot(bot)
        path, _ = self.record(
            physics_scene(live),
            [],
            1,
            recorder.EpisodeMetadata({"scene.slot.right": choice}),
        )
        return path

    def test_keeps_only_the_bots_recorded_slots_selected(self) -> None:
        scene_path = self.write_slot_scene()
        path = self.record_slot_choice(scene_path, SLOT_BOTS[1])
        rebuilt = recorder.rebuild_scene(EpisodeReader(path), scene_path, self.context)
        self.assertIsNone(rebuilt.bot_scene.get_bot(SLOT_BOTS[0]))
        self.assertIsNotNone(rebuilt.bot_scene.get_bot(SLOT_BOTS[1]))

    def test_slots_without_a_recorded_choice_keep_their_bots(self) -> None:
        scene_path = self.write_slot_scene(
            [
                {"name": "right", "controlledBots": list(SLOT_BOTS)},
                {"name": "spare", "controlledBots": [SLOT_BOTS[1]]},
            ]
        )
        bot_scene = load_bot_scene(str(scene_path), self.context)
        _retire_unselected_bots(bot_scene, {"scene.slot.right": SLOT_BOTS[0]})
        for bot in SLOT_BOTS:
            self.assertIsNotNone(bot_scene.get_bot(bot))

    def test_rejects_slot_choice_the_scene_does_not_offer(self) -> None:
        live, spawned = self.load_live_scene()
        path, _ = self.record(
            physics_scene(live),
            spawned,
            1,
            recorder.EpisodeMetadata({"scene.slot.right": "t3_left"}),
        )
        with self.assertRaisesRegex(ValueError, "slot 'right', which offers"):
            recorder.rebuild_scene(
                EpisodeReader(path),
                BOT_SCENE_PATH,
                self.context,
                bot_task_path=self.task_path,
            )

    def test_requires_recorded_bot_task(self) -> None:
        live, spawned = self.load_live_scene()
        kinematic_path, _ = self.record(physics_scene(live), spawned, 1)
        with self.assertRaisesRegex(sdp.Error, "missing recorded actor"):
            recorder.rebuild_scene(
                EpisodeReader(kinematic_path), BOT_SCENE_PATH, self.context
            )

    def test_rejects_static_actor_without_recorded_pose(self) -> None:
        live = load_bot_scene(str(BOT_SCENE_PATH), self.context)
        episode = recorder.Recorder(
            recorder.RecorderConfig(self.directory)
        ).begin_episode()
        # Static poses are recorded with the first frame, which this episode never has.
        recorder.PhysicsRecorder(episode, physics_scene(live))
        episode.commit()
        with self.assertRaisesRegex(
            ValueError, r"Static actor '.+' has no recorded pose"
        ):
            recorder.rebuild_scene(
                EpisodeReader(episode.output_path), BOT_SCENE_PATH, self.context
            )
