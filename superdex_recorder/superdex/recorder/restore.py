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

"""Restore a Mochi scene to the recorded frames of a SuperDex episode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import superdex.physics as sdp

from . import _extension
from .reader import EpisodeReader


@dataclass(frozen=True)
class RecordedActor:
    """A recorded actor and the scene actor that it is restored to."""

    path: str
    """Recorded data path, such as ``data/physics/actors/cube``."""
    actor: sdp.Actor
    is_articulated: bool
    is_static: bool


def _recorded_actors(scene: sdp.Scene, native_actors: list[Any]) -> list[RecordedActor]:
    actors = []
    for native in native_actors:
        actor = scene.get_actor(native.handle)
        assert actor is not None
        actors.append(
            RecordedActor(native.path, actor, native.is_articulated, native.is_static)
        )
    return actors


def _bind_actors(recording: EpisodeReader, scene: sdp.Scene) -> list[RecordedActor]:
    """Bind each recorded actor to its scene actor, as EpisodeRestorer does."""
    native_recording = _extension._EpisodeReader(recording.path)
    return _recorded_actors(
        scene, _extension._bind_recorded_actors(scene, native_recording)
    )


class EpisodeRestorer:
    """Restores a scene to the state recorded at any frame of an episode.

    The scene must contain the recorded actors, with static actors at their recorded
    poses. An actor recorded with a spawn name binds to the scene actor named
    ``<spawn name>/<runtime name>``, or else ``<spawn name>``, or else its runtime
    name, and other actors bind by runtime name. No two recorded actors may bind to
    the same scene actor, and scene actors that were not recorded must be static or
    nested, such as robot links. A scene that does not match raises ``sdp.Error``.

    Restoring a frame sets the scene clock and each dynamic actor's pose, velocity
    and pose-controller target through the public setters, then takes a zero step to
    refresh actor queries. A positive step then continues the episode. The integrator
    starts afresh from the restored state, as at the start of a simulation, so a
    resumed run stays close to the original but doesn't match it bit for bit.

    Restoring a frame also sets each recorded robot's external forces to the joint
    effort recorded for the step after the frame, and clears other dynamic actors'
    external forces. At frames without a recorded effort, such as the last frame, a
    robot keeps its latest earlier effort, because Mochi applies an external force
    until it is replaced. Before its first recorded effort, a robot has none.

    Actors that were spawned or destroyed during the episode are not
    supported.
    """

    def __init__(self, recording: EpisodeReader, scene: sdp.Scene) -> None:
        self._native = _extension._EpisodeRestorer(
            scene, _extension._EpisodeReader(recording.path)
        )
        self._frame_indices = {
            frame: index for index, frame in enumerate(self._native.frames)
        }
        self._actors = _recorded_actors(scene, self._native.actors())

    @property
    def frames(self) -> list[int]:
        """Recorded frame numbers, in recording order."""
        return list(self._native.frames)

    @property
    def actors(self) -> list[RecordedActor]:
        """Recorded actors, sorted by path."""
        return list(self._actors)

    def restore_frame(self, frame: int) -> None:
        """Restore the scene to the end of the step recorded at ``frame``.

        Recorded robots get the joint efforts recorded for the next step.
        """
        index = self._frame_indices.get(frame)
        if index is None:
            raise ValueError(f"Frame {frame} was not recorded.")
        self._native.restore_frame(index)
