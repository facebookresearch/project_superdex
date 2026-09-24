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

"""Behavior tests for the hold-while-filling environment (``Fr3Revo2FillEnv``)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from superdex.lab.gym.envs.robots.fr3_revo2_env import FINGERS
from superdex.lab.gym.envs.robots.fr3_revo2_fill_env import (
    CUP_CAPACITY,
    WATER_DENSITY,
    Fr3Revo2FillEnv,
    Fr3Revo2FillEnvCfg,
)

_DEMO = Path(__file__).resolve().parents[2] / "apps" / "envs" / "run_fr3_revo2_fill.py"


def _load_demo():
    spec = importlib.util.spec_from_file_location("run_fr3_revo2_fill", _DEMO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def env():
    # The scripted grips command absolute targets.
    with Fr3Revo2FillEnv(Fr3Revo2FillEnvCfg(hand_action="absolute")) as env:
        yield env


@pytest.fixture(scope="module")
def demo():
    return _load_demo()


def _open(env):
    return env.to_action({"hand": -np.ones(6, np.float32)})


def test_spaces(env):
    assert env.action_space.shape == (6,)
    structure = env.get_observation_space_structure()
    assert structure["tactile"].shape == (len(FINGERS) * 5,)
    assert "fill" not in structure and "object_in_hand" not in structure
    with Fr3Revo2FillEnv(
        Fr3Revo2FillEnvCfg(observe_tactile=False, observe_fill=True)
    ) as ablation:
        structure = ablation.get_observation_space_structure()
        assert "tactile" not in structure
        assert structure["fill"].shape == (1,)


def test_random_delta_actions_explore_without_crushing_at_once():
    """With delta actions (the default), a random policy does not slam the hand shut:
    it survives the first 0.6 s (with absolute targets it crushes the cup in ~4 steps)."""
    with Fr3Revo2FillEnv(Fr3Revo2FillEnvCfg()) as env:
        env.action_space.seed(0)
        for seed in range(3):
            env.reset(seed=seed)
            for step in range(15):
                *_, terminated, _, info = env.step(env.action_space.sample())
                assert not terminated, (seed, step, info.get("terminated_reason"))


def test_cup_holds_about_410_ml():
    assert 0.40e-3 < CUP_CAPACITY < 0.42e-3


def test_hand_starts_clear_of_the_cup(env):
    """The hand is placed around the cup, not in it: no contact until it closes."""
    for seed in range(4):
        env.reset(seed=seed)
        for _ in range(3):
            *_, info = env.step(_open(env))
            assert info["squeeze"] == 0.0, seed
            assert max(info["tactile_normal_force"].values()) == 0.0, seed


def test_water_fills_and_moves_the_mass_not_the_cup(env):
    env.reset(seed=0)
    cfg = env._cfg
    water = env._water
    mass0 = env._object.get_mass()
    pose0 = np.asarray(env._object.get_root_transform().translation)
    # Adding water changes mass and center of mass; the cup itself must stay put.
    water.mass = 0.2
    env._apply_cup_properties()
    assert env._object.get_mass() == pytest.approx(mass0 + 0.2, rel=1e-5)
    np.testing.assert_allclose(
        np.asarray(env._object.get_root_transform().translation), pose0, atol=1e-7
    )
    # The pour's final amount lies in the configured fill range.
    lo, hi = cfg.fill_fraction
    full = WATER_DENSITY * CUP_CAPACITY
    assert lo * full <= water.target_mass <= hi * full


def test_firm_grip_crushes_the_empty_cup(env, demo):
    result = demo.run_episode(env, "firm", seed=0)
    assert result["outcome"] == "Crushed"
    assert result["water"] == 0.0


def test_light_grip_loses_the_filling_cup(env, demo):
    result = demo.run_episode(env, "light", seed=1)
    assert result["outcome"] in ("Dropped", "slid")
    assert result["hold_slip"] > env._cfg.success_slip


def test_tactile_grip_holds_the_filling_cup(env, demo):
    """Reading the load from the fingertips holds a 318 g pour without crushing it."""
    result = demo.run_episode(env, "tactile", seed=1)
    info = env._last_info
    assert result["outcome"] == "held"
    assert result["water"] > 0.3
    assert info["is_success"]
    assert env._last_reward["success"] > 0.0
    assert result["max_squeeze"] < env._cfg.crush_squeeze
