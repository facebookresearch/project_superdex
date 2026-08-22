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

"""Tests for bot-asset resolution in ``superdex.lab.gym.utils.mochi_helpers``."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from superdex.lab.gym.utils.mochi_helpers import (
    BOTS_ASSETS_PATH_ENV_VAR,
    LEGACY_BOTS_ASSETS_PATH_ENV_VAR,
    resolve_bot_asset,
)

########################################################################################

# A synthetic path: every test writes this file into a temporary root, so it never has to
# match a real asset. Only its shape matters, since that is what resolution walks.
_RELATIVE_BOT = "bots/hands/example_hand/right/example_hand_right.superdex_bot"


class TestResolveBotAsset(unittest.TestCase):
    """Bot lookup must be driven by explicit roots, never by the current directory."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.bot_path = self.root / _RELATIVE_BOT
        self.bot_path.parent.mkdir(parents=True)
        self.bot_path.write_text("{}")

    def _set_bot_assets_env(self, **overrides: str) -> None:
        """Makes ``overrides`` the only bot-assets env vars for this test."""
        environment = dict(os.environ)
        for name in (BOTS_ASSETS_PATH_ENV_VAR, LEGACY_BOTS_ASSETS_PATH_ENV_VAR):
            environment.pop(name, None)
        environment.update(overrides)
        patcher = patch.dict(os.environ, environment, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_env_override_resolves(self) -> None:
        self._set_bot_assets_env(**{BOTS_ASSETS_PATH_ENV_VAR: str(self.root)})

        assert resolve_bot_asset(_RELATIVE_BOT) == self.bot_path.resolve()

    def test_legacy_env_override_resolves(self) -> None:
        self._set_bot_assets_env(**{LEGACY_BOTS_ASSETS_PATH_ENV_VAR: str(self.root)})

        assert resolve_bot_asset(_RELATIVE_BOT) == self.bot_path.resolve()

    def test_canonical_env_var_wins_over_legacy(self) -> None:
        self._set_bot_assets_env(
            **{
                BOTS_ASSETS_PATH_ENV_VAR: str(self.root),
                LEGACY_BOTS_ASSETS_PATH_ENV_VAR: str(self.root / "does_not_exist"),
            }
        )

        assert resolve_bot_asset(_RELATIVE_BOT) == self.bot_path.resolve()

    def test_resolution_is_independent_of_cwd(self) -> None:
        # The previous implementation walked up from the cwd looking for a repo marker,
        # so it failed whenever the process ran from outside a checkout.
        self._set_bot_assets_env(**{BOTS_ASSETS_PATH_ENV_VAR: str(self.root)})
        original_cwd = Path.cwd()

        with tempfile.TemporaryDirectory() as elsewhere:
            try:
                os.chdir(elsewhere)
                assert resolve_bot_asset(_RELATIVE_BOT) == self.bot_path.resolve()
            finally:
                # Windows cannot remove a directory while it is the process cwd.
                os.chdir(original_cwd)

    def test_missing_asset_reports_searched_roots(self) -> None:
        self._set_bot_assets_env(**{BOTS_ASSETS_PATH_ENV_VAR: str(self.root)})

        with self.assertRaises(FileNotFoundError) as context:
            resolve_bot_asset("bots/hands/nonexistent/nonexistent.superdex_bot")

        message = str(context.exception)
        assert str(self.root) in message
        assert BOTS_ASSETS_PATH_ENV_VAR in message


########################################################################################

if __name__ == "__main__":
    unittest.main()
