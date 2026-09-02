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

import unittest
from typing import cast, TYPE_CHECKING

from superdex.physics.utils import render_model_registry

if TYPE_CHECKING:
    import superdex.physics as sdp

########################################################################################


class _FakeHandle:
    """Minimal stand-in for a mochi handle: only a ``.value`` is used by the registry."""

    __slots__ = ("value",)

    def __init__(self, value: int):
        self.value = value


class _FakeScene:
    """Minimal stand-in for a mochi Scene exposing only ``get_handle().value``."""

    __slots__ = ("_handle",)

    def __init__(self, handle_value: int):
        self._handle = _FakeHandle(handle_value)

    def get_handle(self) -> _FakeHandle:
        return self._handle


# The registry only touches ``get_handle().value`` / ``.value`` on scenes and handles,
# and stores ``local_transform`` / ``scale`` opaquely, so the tests drive it with these
# duck-typed stand-ins cast to the mochi types the registry API is annotated with.


def _scene(handle_value: int) -> "sdp.Scene":
    return cast("sdp.Scene", _FakeScene(handle_value))


def _handle(value: int) -> "sdp.ActorHandle":
    return cast("sdp.ActorHandle", _FakeHandle(value))


def _scene_handle(value: int) -> "sdp.SceneHandle":
    return cast("sdp.SceneHandle", _FakeHandle(value))


def _transform() -> "sdp.TransformRT":
    return cast("sdp.TransformRT", object())


def _scale() -> "sdp.Real3":
    return cast("sdp.Real3", object())


########################################################################################


class RenderModelRegistryTest(unittest.TestCase):
    """Tests for the neutral render-model registry.

    The registry stores ``local_transform`` and ``scale`` opaquely, so the tests use
    sentinel objects for them rather than real mochi types.
    """

    def setUp(self) -> None:
        # The registry is module-global; start each test from a clean slate.
        for handle_value in (1, 2):
            render_model_registry.clear_scene(_scene(handle_value))

    def tearDown(self) -> None:
        for handle_value in (1, 2):
            render_model_registry.clear_scene(_scene(handle_value))

    def test_register_and_get(self) -> None:
        transform = _transform()
        scale = _scale()

        render_model_registry.register(
            _scene(1), _handle(10), "hand.glb", transform, scale
        )

        entry = render_model_registry.get(_scene_handle(1), _handle(10))
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.glb_path, "hand.glb")
        self.assertIs(entry.local_transform, transform)
        self.assertIs(entry.scale, scale)

    def test_get_unknown_returns_none(self) -> None:
        render_model_registry.register(
            _scene(1), _handle(10), "hand.glb", _transform(), _scale()
        )

        # Unknown actor in a known scene, and unknown scene entirely.
        self.assertIsNone(render_model_registry.get(_scene_handle(1), _handle(999)))
        self.assertIsNone(render_model_registry.get(_scene_handle(999), _handle(10)))

    def test_scenes_are_isolated(self) -> None:
        # The same actor handle value in two different scenes must not collide.
        render_model_registry.register(
            _scene(1), _handle(10), "a.glb", _transform(), _scale()
        )
        render_model_registry.register(
            _scene(2), _handle(10), "b.glb", _transform(), _scale()
        )

        entry_1 = render_model_registry.get(_scene_handle(1), _handle(10))
        entry_2 = render_model_registry.get(_scene_handle(2), _handle(10))
        assert entry_1 is not None and entry_2 is not None
        self.assertEqual(entry_1.glb_path, "a.glb")
        self.assertEqual(entry_2.glb_path, "b.glb")

    def test_register_overwrites_existing(self) -> None:
        render_model_registry.register(
            _scene(1), _handle(10), "old.glb", _transform(), _scale()
        )
        render_model_registry.register(
            _scene(1), _handle(10), "new.glb", _transform(), _scale()
        )

        entry = render_model_registry.get(_scene_handle(1), _handle(10))
        assert entry is not None
        self.assertEqual(entry.glb_path, "new.glb")

    def test_unregister_actors(self) -> None:
        keep = _handle(10)
        drop = _handle(11)
        render_model_registry.register(
            _scene(1), keep, "keep.glb", _transform(), _scale()
        )
        render_model_registry.register(
            _scene(1), drop, "drop.glb", _transform(), _scale()
        )

        render_model_registry.unregister_actors(_scene(1), [drop])

        self.assertIsNone(render_model_registry.get(_scene_handle(1), _handle(11)))
        entry = render_model_registry.get(_scene_handle(1), _handle(10))
        assert entry is not None
        self.assertEqual(entry.glb_path, "keep.glb")

    def test_unregister_missing_actor_is_noop(self) -> None:
        render_model_registry.register(
            _scene(1), _handle(10), "keep.glb", _transform(), _scale()
        )

        # Unregistering an actor that was never registered must not raise.
        render_model_registry.unregister_actors(_scene(1), [_handle(999)])
        entry = render_model_registry.get(_scene_handle(1), _handle(10))
        assert entry is not None
        self.assertEqual(entry.glb_path, "keep.glb")

    def test_unregister_on_unknown_scene_is_noop(self) -> None:
        # Must not raise even if the scene has no registered entries.
        render_model_registry.unregister_actors(_scene(2), [_handle(10)])

    def test_clear_scene(self) -> None:
        render_model_registry.register(
            _scene(1), _handle(10), "a.glb", _transform(), _scale()
        )
        render_model_registry.register(
            _scene(1), _handle(11), "b.glb", _transform(), _scale()
        )

        render_model_registry.clear_scene(_scene(1))

        self.assertIsNone(render_model_registry.get(_scene_handle(1), _handle(10)))
        self.assertIsNone(render_model_registry.get(_scene_handle(1), _handle(11)))
