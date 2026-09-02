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

import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from superdex.physics.viewer.unrealcv.unrealcv_actor_utils import (
    _build_actor_material_commands,
    _log_material_dr_results,
    _log_mpc_results,
    _log_visibility_results,
    _resolve_mpc_value,
    apply_cast_shadow_dr,
    apply_hdri_lighting_dr,
    apply_mpc_dr,
    apply_scale_dr,
)


@dataclass
class _FakeRangeConfig:
    low: float
    high: float


@dataclass
class _FakeMaterialOverride:
    """Minimal stand-in for ActorMaterialOverride for command-builder tests.

    Mirrors the fields ``_build_actor_material_commands`` reads (including the
    ``uses_rgb`` property) so tests do not depend on the fbcode config module.
    """

    hue: object = None
    saturation: object = None
    value: object = None
    red: object = None
    green: object = None
    blue: object = None
    alpha: object = None
    roughness: object = "default"
    metallic: object = "default"
    specular: object = "default"
    light_intensity: object = "default"
    source_radius: object = "default"
    texture_index: object = "default"
    texture_probability: object = None

    @property
    def uses_rgb(self) -> bool:
        return (
            (self.red is not None and self.red != "default")
            or (self.green is not None and self.green != "default")
            or (self.blue is not None and self.blue != "default")
        )


class ResolveMpcValueTest(unittest.TestCase):
    def test_none_returns_none(self):
        result = _resolve_mpc_value(None)
        self.assertIsNone(result)

    def test_scalar_float(self):
        result = _resolve_mpc_value(3.5)
        self.assertEqual(result, 3.5)

    def test_scalar_int(self):
        result = _resolve_mpc_value(7)
        self.assertEqual(result, 7.0)

    def test_range_config_returns_within_bounds(self):
        np.random.seed(42)
        rc = _FakeRangeConfig(low=1.0, high=5.0)
        result = _resolve_mpc_value(rc)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result, 1.0)
        self.assertLessEqual(result, 5.0)

    def test_dict_converted_to_range(self):
        np.random.seed(42)
        result = _resolve_mpc_value({"low": 2.0, "high": 4.0})
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result, 2.0)
        self.assertLessEqual(result, 4.0)


class LogMpcResultsTest(unittest.TestCase):
    @patch("superdex.physics.viewer.unrealcv.unrealcv_actor_utils.logger")
    def test_all_succeeded(self, mock_logger):
        commands = [
            "vset /object/Actor1/mpc_scalar Hue 0.5",
            "vset /object/Actor1/mpc_scalar Noise 0.3",
        ]
        responses = ["ok", "ok"]
        _log_mpc_results(commands, responses, ["Actor1"])
        self.assertEqual(mock_logger.info.call_count, 3)
        mock_logger.warning.assert_not_called()

    @patch("superdex.physics.viewer.unrealcv.unrealcv_actor_utils.logger")
    def test_mixed_results(self, mock_logger):
        commands = [
            "vset /object/Actor1/mpc_scalar Hue 0.5",
            "vset /object/Actor2/mpc_scalar Noise 0.3",
        ]
        responses = ["ok", "error"]
        _log_mpc_results(commands, responses, ["Actor1", "Actor2"])
        self.assertGreaterEqual(mock_logger.info.call_count, 1)
        mock_logger.error.assert_called_once()

    @patch("superdex.physics.viewer.unrealcv.unrealcv_actor_utils.logger")
    def test_empty_commands(self, mock_logger):
        _log_mpc_results([], [], [])
        mock_logger.info.assert_called_once()
        mock_logger.warning.assert_not_called()


class ApplyMpcDrTest(unittest.TestCase):
    def _make_cfg(self, mpc_overrides=None):
        cfg = SimpleNamespace(mpc_overrides=mpc_overrides or {})
        return cfg

    def _make_client(self, responses=None):
        client = MagicMock()
        if responses is None:
            responses = ["ok"]
        client._request_batch.return_value = responses
        return client

    def test_no_overrides_returns_early(self):
        client = self._make_client()
        cfg = self._make_cfg(mpc_overrides={})
        apply_mpc_dr(client, cfg)
        client._request_batch.assert_not_called()

    def test_missing_attr_returns_early(self):
        client = self._make_client()
        cfg = SimpleNamespace()
        apply_mpc_dr(client, cfg)
        client._request_batch.assert_not_called()

    def test_scalar_values_generate_commands(self):
        client = self._make_client(responses=["ok", "ok"])
        cfg = self._make_cfg(
            mpc_overrides={
                "Actor1": {"Hue": 0.5, "Noise": 0.3},
            }
        )
        apply_mpc_dr(client, cfg)
        client._request_batch.assert_called_once()
        commands = client._request_batch.call_args[0][0]
        self.assertEqual(len(commands), 2)
        self.assertIn("mpc_scalar Hue 0.5", commands[0])
        self.assertIn("mpc_scalar Noise 0.3", commands[1])

    def test_none_values_skipped(self):
        client = self._make_client(responses=["ok"])
        cfg = self._make_cfg(
            mpc_overrides={
                "Actor1": {"Hue": 0.5, "Noise": None},
            }
        )
        apply_mpc_dr(client, cfg)
        commands = client._request_batch.call_args[0][0]
        self.assertEqual(len(commands), 1)
        self.assertIn("Hue", commands[0])

    def test_empty_actor_params_skipped(self):
        client = self._make_client(responses=["ok"])
        cfg = self._make_cfg(
            mpc_overrides={
                "Actor1": {},
                "Actor2": {"Param": 1.0},
            }
        )
        apply_mpc_dr(client, cfg)
        commands = client._request_batch.call_args[0][0]
        self.assertEqual(len(commands), 1)
        self.assertIn("Actor2", commands[0])

    def test_dict_range_resolved(self):
        np.random.seed(0)
        client = self._make_client(responses=["ok"])
        cfg = self._make_cfg(
            mpc_overrides={
                "Actor1": {"Hue": {"low": -30.0, "high": 30.0}},
            }
        )
        apply_mpc_dr(client, cfg)
        commands = client._request_batch.call_args[0][0]
        self.assertEqual(len(commands), 1)
        self.assertIn("mpc_scalar Hue", commands[0])

    def test_range_config_object_resolved(self):
        np.random.seed(0)
        client = self._make_client(responses=["ok"])
        cfg = self._make_cfg(
            mpc_overrides={
                "Actor1": {"Hue": _FakeRangeConfig(low=-30.0, high=30.0)},
            }
        )
        apply_mpc_dr(client, cfg)
        commands = client._request_batch.call_args[0][0]
        self.assertEqual(len(commands), 1)
        self.assertIn("mpc_scalar Hue", commands[0])

    def test_tuple_response_unwrapped(self):
        client = MagicMock()
        client._request_batch.return_value = (["ok"], 0.01)
        cfg = self._make_cfg(mpc_overrides={"Actor1": {"Hue": 1.0}})
        apply_mpc_dr(client, cfg)
        client._request_batch.assert_called_once()
        commands = client._request_batch.call_args[0][0]
        self.assertEqual(len(commands), 1)
        self.assertIn("mpc_scalar Hue", commands[0])


class ApplyHdriLightingDrTest(unittest.TestCase):
    _DOME = "BP_HDRI_Skydome_C_1"
    _SKYLIGHT = "SkyLight_0"

    def _make_override(self, **overrides):
        cubemaps = [f"/Game/HDRI/TC_HDRI_{i:02d}" for i in range(1, 11)]
        base = {
            "skylight_actor": self._SKYLIGHT,
            "dome_actor": self._DOME,
            "cubemap_asset_paths": cubemaps,
            "dome_cube_param": "EnvCube",
            "offset_yaw": 30.0,
            "dome_yaw_sign": 1,
        }
        base.update(overrides)
        return SimpleNamespace(hdri_lighting_override=base)

    def _make_client(self, responses=None):
        client = MagicMock()
        client._request_batch.return_value = responses or ["ok", "ok", "ok", "ok"]
        return client

    def test_none_override_returns_early(self):
        client = self._make_client()
        cfg = SimpleNamespace(hdri_lighting_override=None)
        apply_hdri_lighting_dr(client, cfg, {})
        client._request_batch.assert_not_called()

    def test_missing_attr_returns_early(self):
        client = self._make_client()
        apply_hdri_lighting_dr(client, SimpleNamespace(), {})
        client._request_batch.assert_not_called()

    def test_incomplete_config_returns_early(self):
        client = self._make_client()
        cfg = self._make_override(skylight_actor="")
        apply_hdri_lighting_dr(client, cfg, {self._DOME: (0.0, 0.0, 0.0)})
        client._request_batch.assert_not_called()

    @patch("numpy.random.randint", return_value=3)
    def test_drives_dome_and_skylight_in_sync(self, _mock_randint):
        client = self._make_client()
        cfg = self._make_override()
        # Pre-seed the dome baseline so no rotation-capture round trip happens.
        apply_hdri_lighting_dr(client, cfg, {self._DOME: (0.0, 100.0, 0.0)})

        client._request_batch.assert_called_once()
        commands = client._request_batch.call_args[0][0]
        # One index + one yaw drive all four commands.
        self.assertEqual(
            commands,
            [
                f"vset /object/{self._DOME}/texture_cube_param EnvCube /Game/HDRI/TC_HDRI_04",
                f"vset /object/{self._DOME}/rotation 0.0 130.0 0.0",
                f"vset /object/{self._SKYLIGHT}/skylight_cubemap /Game/HDRI/TC_HDRI_04",
                f"vset /object/{self._SKYLIGHT}/skylight_angle 30.0",
            ],
        )

    @patch("numpy.random.randint", return_value=3)
    def test_dome_yaw_sign_flips_dome_rotation_only(self, _mock_randint):
        client = self._make_client()
        cfg = self._make_override(dome_yaw_sign=-1)
        apply_hdri_lighting_dr(client, cfg, {self._DOME: (0.0, 100.0, 0.0)})

        commands = client._request_batch.call_args[0][0]
        # Dome yaw is base - yaw; skylight angle stays the unsigned yaw.
        self.assertIn(f"/object/{self._DOME}/rotation 0.0 70.0 0.0", commands[1])
        self.assertIn(f"/object/{self._SKYLIGHT}/skylight_angle 30.0", commands[3])

    @patch("numpy.random.randint", return_value=7)
    def test_dome_and_skylight_use_same_cube_asset(self, _mock_randint):
        client = self._make_client()
        cfg = self._make_override()
        apply_hdri_lighting_dr(client, cfg, {self._DOME: (0.0, 0.0, 0.0)})
        commands = client._request_batch.call_args[0][0]
        # The unified design drives BOTH the dome and the SkyLight from ONE
        # cube asset path.
        cube = "/Game/HDRI/TC_HDRI_08"
        self.assertIn(f"texture_cube_param EnvCube {cube}", commands[0])
        self.assertIn(f"skylight_cubemap {cube}", commands[2])

    @patch("numpy.random.randint", return_value=0)
    def test_new_fields_omitted_keeps_current_four_commands(self, _mock_randint):
        client = self._make_client()
        cfg = self._make_override()
        apply_hdri_lighting_dr(client, cfg, {self._DOME: (0.0, 0.0, 0.0)})
        commands = client._request_batch.call_args[0][0]
        self.assertEqual(len(commands), 4)

    @patch("numpy.random.randint", return_value=0)
    def test_skylight_intensity_fixed_emits_command(self, _mock_randint):
        client = self._make_client(responses=["ok"] * 5)
        cfg = self._make_override(skylight_intensity=0.4)
        apply_hdri_lighting_dr(client, cfg, {self._DOME: (0.0, 0.0, 0.0)})
        commands = client._request_batch.call_args[0][0]
        self.assertIn(f"vset /object/{self._SKYLIGHT}/skylight_intensity 0.4", commands)

    @patch("numpy.random.uniform", return_value=0.55)
    @patch("numpy.random.randint", return_value=0)
    def test_skylight_intensity_range_samples_per_rep(
        self, _mock_randint, _mock_uniform
    ):
        client = self._make_client(responses=["ok"] * 5)
        cfg = self._make_override(
            offset_yaw=None,
            skylight_intensity={"low": 0.3, "high": 0.9},
        )
        apply_hdri_lighting_dr(client, cfg, {self._DOME: (0.0, 0.0, 0.0)})
        commands = client._request_batch.call_args[0][0]
        self.assertIn(
            f"vset /object/{self._SKYLIGHT}/skylight_intensity 0.55", commands
        )

    @patch("numpy.random.randint", return_value=0)
    def test_skylight_color_emits_three_channel_command(self, _mock_randint):
        client = self._make_client(responses=["ok"] * 5)
        cfg = self._make_override(
            skylight_color={"red": 0.5, "green": 0.6, "blue": 1.0},
        )
        apply_hdri_lighting_dr(client, cfg, {self._DOME: (0.0, 0.0, 0.0)})
        commands = client._request_batch.call_args[0][0]
        self.assertIn(
            f"vset /object/{self._SKYLIGHT}/skylight_color 0.5 0.6 1.0", commands
        )

    @patch("numpy.random.randint", return_value=0)
    def test_skydome_brightness_emits_scalar_param_command(self, _mock_randint):
        client = self._make_client(responses=["ok"] * 5)
        cfg = self._make_override(skydome_brightness=0.5)
        apply_hdri_lighting_dr(client, cfg, {self._DOME: (0.0, 0.0, 0.0)})
        commands = client._request_batch.call_args[0][0]
        self.assertIn(
            f"vset /object/{self._DOME}/scalar_param Brightness 0.5", commands
        )


class ApplyCastShadowDrTest(unittest.TestCase):
    def _make_cfg(self, cast_shadow_overrides=None):
        return SimpleNamespace(cast_shadow_overrides=cast_shadow_overrides or {})

    def _make_client(self, responses=None):
        client = MagicMock()
        client._request_batch.return_value = responses or ["ok"]
        return client

    def test_no_overrides_returns_early(self):
        client = self._make_client()
        apply_cast_shadow_dr(client, self._make_cfg({}))
        client._request_batch.assert_not_called()

    def test_missing_attr_returns_early(self):
        client = self._make_client()
        apply_cast_shadow_dr(client, SimpleNamespace())
        client._request_batch.assert_not_called()

    def test_bool_toggles_map_to_one_and_zero(self):
        client = self._make_client(responses=["ok", "ok"])
        cfg = self._make_cfg({"WallA": False, "LightB": True})
        apply_cast_shadow_dr(client, cfg)
        commands = client._request_batch.call_args[0][0]
        self.assertEqual(
            commands,
            [
                "vset /object/WallA/cast_shadow 0",
                "vset /object/LightB/cast_shadow 1",
            ],
        )

    def test_tuple_response_unwrapped(self):
        client = MagicMock()
        client._request_batch.return_value = (["ok"], 0.01)
        cfg = self._make_cfg({"WallA": False})
        apply_cast_shadow_dr(client, cfg)
        client._request_batch.assert_called_once()
        commands = client._request_batch.call_args[0][0]
        self.assertEqual(commands, ["vset /object/WallA/cast_shadow 0"])


class ApplyScaleDrTest(unittest.TestCase):
    def _make_cfg(self, scale_overrides=None):
        return SimpleNamespace(scale_overrides=scale_overrides or {})

    def _make_client(self, responses=None):
        client = MagicMock()
        client._request_batch.return_value = responses or ["ok"]
        return client

    def _override(
        self, scale=None, scale_x=None, scale_y=None, scale_z=None, scale_group=None
    ):
        return SimpleNamespace(
            scale=scale,
            scale_x=scale_x,
            scale_y=scale_y,
            scale_z=scale_z,
            scale_group=scale_group,
        )

    def test_no_overrides_returns_early(self):
        client = self._make_client()
        apply_scale_dr(client, self._make_cfg({}))
        client._request_batch.assert_not_called()

    def test_missing_attr_returns_early(self):
        client = self._make_client()
        apply_scale_dr(client, SimpleNamespace())
        client._request_batch.assert_not_called()

    def test_none_override_skipped(self):
        client = self._make_client(responses=["ok"])
        cfg = self._make_cfg({"A": None, "B": self._override(scale=1.5)})
        apply_scale_dr(client, cfg)
        commands = client._request_batch.call_args[0][0]
        self.assertEqual(commands, ["vset /object/B/scale 1.5 1.5 1.5"])

    def test_per_axis_precedence_and_uniform_fallback(self):
        client = self._make_client(responses=["ok"])
        cfg = self._make_cfg({"A": self._override(scale=1.2, scale_z=2.0)})
        apply_scale_dr(client, cfg)
        commands = client._request_batch.call_args[0][0]
        # scale_x/y fall back to uniform 1.2; scale_z override wins.
        self.assertEqual(commands, ["vset /object/A/scale 1.2 1.2 2.0"])

    @patch("numpy.random.uniform", side_effect=[0.9, 1.05, 0.81, 1.11])
    def test_no_scale_group_samples_independently(self, mock_uniform):
        client = self._make_client(responses=["ok", "ok"])
        rc = _FakeRangeConfig
        cfg = self._make_cfg(
            {
                "Foam": self._override(scale_x=rc(0.8, 1.2), scale_y=rc(0.9, 1.1)),
                "Foam2": self._override(scale_x=rc(0.8, 1.2), scale_y=rc(0.9, 1.1)),
            }
        )
        apply_scale_dr(client, cfg)
        commands = client._request_batch.call_args[0][0]
        # Ungrouped: each foam draws its own values → different sizes.
        self.assertEqual(
            commands,
            [
                "vset /object/Foam/scale 0.9 1.05 1.0",
                "vset /object/Foam2/scale 0.81 1.11 1.0",
            ],
        )
        self.assertEqual(mock_uniform.call_count, 4)

    @patch("numpy.random.uniform", side_effect=[0.9, 1.05, 0.5, 0.5])
    def test_scale_group_shares_single_sample(self, mock_uniform):
        client = self._make_client(responses=["ok", "ok"])
        rc = _FakeRangeConfig
        cfg = self._make_cfg(
            {
                "Foam": self._override(
                    scale_x=rc(0.8, 1.2), scale_y=rc(0.9, 1.1), scale_group="foam"
                ),
                "Foam2": self._override(scale_group="foam"),
            }
        )
        apply_scale_dr(client, cfg)
        commands = client._request_batch.call_args[0][0]
        # Grouped: Foam samples once (0.9, 1.05); Foam2 reuses the SAME scale.
        self.assertEqual(
            commands,
            [
                "vset /object/Foam/scale 0.9 1.05 1.0",
                "vset /object/Foam2/scale 0.9 1.05 1.0",
            ],
        )
        # Only Foam's two axes were drawn — Foam2 reused, no extra sampling.
        self.assertEqual(mock_uniform.call_count, 2)


class BuildActorMaterialCommandsExclusiveTest(unittest.TestCase):
    """Exclusive per-rep hue-vs-texture mode in _build_actor_material_commands."""

    _ACTOR = "Board"

    def test_texture_probability_omitted_matches_legacy_behavior(self):
        # Regression guard: with texture_probability unset, hue AND texture are
        # applied independently exactly as before.
        override = _FakeMaterialOverride(
            hue=_FakeRangeConfig(0.0, 360.0),
            roughness=_FakeRangeConfig(0.4, 0.9),
            texture_index=_FakeRangeConfig(0.0, 1.0),
        )
        with patch("numpy.random.uniform", side_effect=[123.0, 0.55, 0.42]):
            commands = _build_actor_material_commands(self._ACTOR, override, {})
        self.assertEqual(
            commands,
            [
                f"vset /object/{self._ACTOR}/hsva 123.0 -1000000.0 -1000000.0 -1000000.0",
                f"vset /object/{self._ACTOR}/roughness 0.55",
                f"vset /object/{self._ACTOR}/texture_index 0.42",
            ],
        )

    def _exclusive_override(self) -> _FakeMaterialOverride:
        return _FakeMaterialOverride(
            hue=_FakeRangeConfig(0.0, 360.0),
            saturation=_FakeRangeConfig(0.2, 0.8),
            value=_FakeRangeConfig(0.4, 0.9),
            roughness=_FakeRangeConfig(0.4, 0.9),
            metallic=_FakeRangeConfig(0.0, 0.3),
            specular=None,
            texture_index=_FakeRangeConfig(0.0, 1.0),
            texture_probability=0.8,
        )

    def test_forced_texture_mode_emits_texture_and_neutral_tint(self):
        # uniform() = 0.1 < 0.8 → texture mode. Draws: mode, texture, rough, metallic.
        override = self._exclusive_override()
        with patch("numpy.random.uniform", side_effect=[0.1, 0.5, 0.6, 0.2]):
            commands = _build_actor_material_commands(self._ACTOR, override, {})
        # Sampled texture + neutral white tint reset; NO sampled hue.
        self.assertEqual(
            commands,
            [
                f"vset /object/{self._ACTOR}/texture_index 0.5",
                f"vset /object/{self._ACTOR}/hsva 0.0 0.0 1.0 1.0",
                f"vset /object/{self._ACTOR}/roughness 0.6",
                f"vset /object/{self._ACTOR}/metallic 0.2",
            ],
        )

    def test_forced_hue_mode_emits_hsva_and_flat_texture_slot(self):
        # uniform() = 0.9 >= 0.8 → hue mode. Draws: mode, hue, sat, val, rough, metallic.
        override = self._exclusive_override()
        with patch(
            "numpy.random.uniform", side_effect=[0.9, 200.0, 0.5, 0.7, 0.6, 0.2]
        ):
            commands = _build_actor_material_commands(self._ACTOR, override, {})
        # Sampled hue tint + flat-white texture slot (index 0); alpha kept.
        self.assertEqual(
            commands,
            [
                f"vset /object/{self._ACTOR}/hsva 200.0 0.5 0.7 -1000000.0",
                f"vset /object/{self._ACTOR}/texture_index 0.0",
                f"vset /object/{self._ACTOR}/roughness 0.6",
                f"vset /object/{self._ACTOR}/metallic 0.2",
            ],
        )

    def test_per_component_texture_mode_routes_with_component_token(self):
        # Per-child form routes every command through the component token.
        override = self._exclusive_override()
        with patch("numpy.random.uniform", side_effect=[0.1, 0.5, 0.6, 0.2]):
            commands = _build_actor_material_commands(
                self._ACTOR, override, {}, component=["NIST_Board_Render"]
            )
        self.assertEqual(
            commands,
            [
                f"vset /object/{self._ACTOR}/texture_index NIST_Board_Render 0.5",
                f"vset /object/{self._ACTOR}/hsva NIST_Board_Render 0.0 0.0 1.0 1.0",
                f"vset /object/{self._ACTOR}/roughness NIST_Board_Render 0.6",
                f"vset /object/{self._ACTOR}/metallic NIST_Board_Render 0.2",
            ],
        )


class AbsentActorDrFailuresTest(unittest.TestCase):
    """DR configs deliberately name actors from more than one map variant.

    Whichever name is not in the loaded map fails on every episode reset, so an
    intentional no-op was reported ~867 times at ERROR over a 200-rollout eval.
    """

    _GHOST = "Curtains_GEN_VARIABLE_Curtains_C_CAT_5"
    _ABSENT = "error Can not find object"

    def setUp(self) -> None:
        # The registry is process-scoped by design, and this target runs the
        # whole file in one interpreter, so isolation here is load-bearing.
        patcher = patch(
            "superdex.physics.viewer.unrealcv.unrealcv_actor_utils._ABSENT_ACTORS_REPORTED",
            None,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    @patch("superdex.physics.viewer.unrealcv.unrealcv_actor_utils.logger")
    def test_absent_actor_reported_once_across_resets(self, mock_logger):
        for _ in range(3):
            _log_visibility_results(
                [f"vset /object/{self._GHOST}/hide"], [self._ABSENT], [self._GHOST]
            )
        mock_logger.error.assert_called_once()
        self.assertIn(self._GHOST, mock_logger.error.call_args[0][3])

    @patch("superdex.physics.viewer.unrealcv.unrealcv_actor_utils.logger")
    def test_absent_actor_reported_once_across_dr_kinds(self, mock_logger):
        # Every property targeting a missing actor reports the same one fact,
        # so the key is the actor, not the command.
        _log_material_dr_results(
            [f"vset /object/{self._GHOST}/hsva 1 2 3 4"], [self._ABSENT], [self._GHOST]
        )
        _log_visibility_results(
            [f"vset /object/{self._GHOST}/hide"], [self._ABSENT], [self._GHOST]
        )
        mock_logger.error.assert_called_once()

    @patch("superdex.physics.viewer.unrealcv.unrealcv_actor_utils.logger")
    def test_each_absent_actor_gets_its_own_line(self, mock_logger):
        # A typo'd actor name must stay discoverable, not be folded into a peer.
        for actor in (self._GHOST, "BP_Typoed_Actor_C_0"):
            _log_visibility_results(
                [f"vset /object/{actor}/hide"], [self._ABSENT], [actor]
            )
        self.assertEqual(mock_logger.error.call_count, 2)

    @patch("superdex.physics.viewer.unrealcv.unrealcv_actor_utils.logger")
    def test_other_failures_are_never_suppressed(self, mock_logger):
        # A present actor that rejects a command, and a dropped batch (resp
        # None), are live faults -- they must report on every reset.
        for _ in range(3):
            _log_visibility_results(
                ["vset /object/RealActor_C_0/hide"], ["error something else"], ["a"]
            )
            _log_visibility_results(["vset /object/RealActor_C_0/hide"], [None], ["a"])
        self.assertEqual(mock_logger.error.call_count, 6)
