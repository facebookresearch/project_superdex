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

"""Tests for the environment auto-discovery mechanism itself.

These cover the rules discovery applies rather than the environments it finds: which JSON
files count as gym config variants, which variants are test-only and therefore hidden,
which usage recipe an entry resolves to, and what gets registered with Gymnasium.

The environments themselves are smoke-tested in ``test_envs``, which generates one
instantiate-and-step test per discovered entry.
"""

import inspect
import json
import unittest
from pathlib import Path

import gymnasium as gym
from superdex.lab.gym.utils.env_discovery import (
    _is_valid_variant_name,
    discover_envs,
    get_env_entries,
    get_env_short_names,
    load_entry_config,
    register_all_envs,
)

########################################################################################


class TestEnvDiscovery(unittest.TestCase):
    """Tests for the discovery rules in
    :mod:`superdex.lab.gym.utils.env_discovery`."""

    def test_base_ids_are_makeable(self) -> None:
        """``gym.make`` must work for base ids, not just config variants. Env
        constructors take a required ``cfg``, so registering a base id without one makes
        ``gym.make`` call the constructor with no config and raise ``TypeError``.
        ``make_env()`` bypasses Gymnasium, so this is the only coverage of that path."""
        for entry in register_all_envs():
            if entry.variant is not None:
                continue
            with self.subTest(env_id=entry.env_id):
                gym.make(entry.env_id).close()

    def test_variant_name_rejects_non_variant_configs(self) -> None:
        """The ``<module>_*.json`` glob over-matches, so only snake_case variant names with
        no reserved segment may pass. Chiefly this keeps a variant-scoped usage config out
        of gym registration: ``foo_env_hard.train.json`` yields ``hard.train`` because
        ``Path.stem`` strips only the final suffix, and registering it would produce
        ``FooHard.train-v0``."""
        for variant in ("hard", "soft", "v2", "soft_fingertips", "test_no_dynamics"):
            with self.subTest(variant=variant):
                self.assertTrue(_is_valid_variant_name(variant))

        for variant in (
            "hard.train",  # this module's usage config, scoped to a variant
            "extra_env.train",  # ditto, with a longer name
            "hard.",
            "",  # `<module>_.json`
            "Hard",  # not snake_case: would break the generated id
            "hard-mode",
            "hard mode",
            "_hard",
            "hard__mode",
        ):
            with self.subTest(variant=variant):
                self.assertFalse(_is_valid_variant_name(variant))

    def test_variant_name_rejects_reserved_segments(self) -> None:
        """``env`` is reserved because every env module ends in that segment, so a file
        belonging to a longer sibling (``foo_env_extra_env.json`` next to ``foo_env.py``)
        would otherwise read as variant ``extra_env`` of the shorter module. ``train`` and
        ``benchmark`` are reserved so that ``foo_env_train.json`` (a variant) cannot be
        mistaken for ``foo_env.train.json`` (a recipe)."""
        for variant in (
            "env",
            "extra_env",
            "env_extra",
            "extra_env_hard",
            "train",
            "long_train",
            "benchmark",
            "benchmark_base",
        ):
            with self.subTest(variant=variant):
                self.assertFalse(_is_valid_variant_name(variant))

    def test_test_segment_marks_variant_test_only(self) -> None:
        """A ``test`` segment in the variant name is the only marker for a test-only
        variant, so the mapping from name to :attr:`EnvEntry.test_only` must hold for every
        discovered entry."""
        entries = discover_envs()
        self.assertTrue(
            any(entry.test_only for entry in entries),
            "no test-only variant was discovered, so this test proves nothing",
        )
        for entry in entries:
            with self.subTest(env_id=entry.env_id):
                expected = entry.variant is not None and "test" in entry.variant.split(
                    "_"
                )
                self.assertEqual(entry.test_only, expected)

    def test_test_only_variants_are_hidden(self) -> None:
        """Test-only variants are degenerate configurations kept as crash checks, so they
        must be discovered (and hence smoke-tested) yet stay out of the Gymnasium registry
        -- which is also what keeps them out of the Ray Tune registry -- and out of the
        short-name listing the CLIs build their ``choices`` from."""
        register_all_envs()
        public_names = get_env_short_names()
        all_names = get_env_short_names(include_test_only=True)

        test_only = [entry for entry in get_env_entries() if entry.test_only]
        self.assertTrue(test_only, "no test-only variant was discovered")
        for entry in test_only:
            with self.subTest(env_id=entry.env_id):
                self.assertNotIn(entry.env_id, gym.registry)
                self.assertNotIn(entry.short_name, public_names)
                self.assertIn(entry.short_name, all_names)

    def test_variant_recipe_does_not_fall_back_to_base(self) -> None:
        """A usage recipe is written for one configuration, so a variant must not inherit
        the base env's recipe (nor the reverse): the Ant training recipe targets
        ``ant_no_contact``, and applying its stop criteria to the base Ant would train
        against a different observation space."""
        entries = {entry.short_name: entry for entry in discover_envs()}
        base, variant = entries.get("ant"), entries.get("ant_no_contact")
        if base is None or variant is None:
            self.skipTest("the Ant env is absent from this build")
        self.assertTrue(load_entry_config(variant, "train"))
        self.assertEqual(load_entry_config(base, "train"), {})

    def test_train_recipes_declare_no_env_config(self) -> None:
        """Env configuration belongs in a gym config variant, so that every configuration
        that gets trained is also nameable, runnable and smoke-tested. A recipe-only
        override is rejected at training time, but catch it here instead."""
        directories = {
            Path(inspect.getfile(entry.env_cls)).parent for entry in discover_envs()
        }
        recipes = sorted(
            path for directory in directories for path in directory.glob("*.train.json")
        )
        self.assertTrue(recipes, "no training recipes were found")
        for path in recipes:
            with self.subTest(recipe=path.name):
                self.assertNotIn("env_config", json.loads(path.read_text()))

    def test_registered_cfg_is_decoupled_from_cached_entry(self) -> None:
        """Entries are cached for the process lifetime, so the registry must not share
        their ``cfg_kwargs`` dict: a consumer mutating ``gym.spec(id).kwargs['cfg']``
        would otherwise silently reconfigure every later lookup of that env."""
        register_all_envs()
        for entry in get_env_entries():
            if entry.test_only:
                continue
            with self.subTest(env_id=entry.env_id):
                registered_cfg = gym.spec(entry.env_id).kwargs["cfg"]
                self.assertIsNot(registered_cfg, entry.cfg_kwargs)
                registered_cfg["_mutated_by_test"] = True
                self.assertNotIn("_mutated_by_test", entry.cfg_kwargs)
                del registered_cfg["_mutated_by_test"]


########################################################################################

if __name__ == "__main__":
    unittest.main()
