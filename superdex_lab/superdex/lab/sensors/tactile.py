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

"""Taxel-array tactile sensor for fingertip pads, built on SuperDex contact points.

A ``TACTILE_PAD`` sensor is declared on a pad link in a ``.superdex_bot`` as
``{"type": "TACTILE_PAD", "name": "index_tactile", "params": "<json>"}``, where the params
string (or a path to a JSON file) decodes to :class:`TactilePadParams` fields, e.g.::

    {"rows": 8, "cols": 6, "size": [0.012, 0.018],
     "pad_from_sensor": {"rotation": [x, y, z, w], "translation": [x, y, z]}}

It becomes live once the type is registered on the robotics context *before*
``create_bot()`` (a build without the registration skips the sensor with a warning)::

    register_tactile_pad_sensor(robotics_context)
    bot = robotics.create_bot(scene, prefab, robotics_context)
    sensors = find_tactile_pad_sensors(bot, robotics_context)
    ...
    scene.step(dt)
    reading = sensors["index_tactile"].compute_signal(dt)

The sensor frame sits on the pad surface: +z is the outward pad normal, +y runs along the
finger toward the tip, and +x completes the right-handed frame. Every contact point on
the pad link is resolved into its compressive normal load (along the local contact
normal) and splatted bilinearly onto a ``rows x cols`` grid of taxels covering the pad
footprint in the sensor x-y plane. Row index grows along +y, column index along +x.

The model is geometric, not calibrated against a particular hardware sensor: it reports
what an ideal pressure array would see, with optional noise, dead band, saturation and a
first-order lag to approximate a real sensor's non-idealities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import superdex.physics as physics
import superdex.robotics as robotics
from scipy.spatial.transform import Rotation

TACTILE_PAD_SENSOR_TYPE = "TACTILE_PAD"
"""Registered sensor type name, as used in ``.superdex_bot`` files."""


@dataclass
class TactilePadParams:
    """Parameters of a :class:`TactilePadSensor`, parsed from the bot's ``params``."""

    rows: int = 8
    """Taxel rows, along the sensor +y axis (along the finger)."""
    cols: int = 6
    """Taxel columns, along the sensor +x axis (across the finger)."""
    size: tuple[float, float] = (0.012, 0.018)
    """Footprint of the taxel grid along sensor (x, y) [m], centered on the origin."""
    pad_from_sensor_rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Sensor frame orientation in the pad link frame, quaternion [x, y, z, w]."""
    pad_from_sensor_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Sensor frame origin in the pad link frame [m]."""
    max_incidence_deg: float = 85.0
    """Only contacts pressing on the sensing face count: a contact is sensed when its
    load direction is within this angle of the sensor's inward normal (-z). Contacts on
    the back or sides of the pad are ignored."""
    footprint_margin: float = 0.002
    """Contacts up to this far outside the footprint [m] land on the edge taxels; farther
    ones (e.g. on the pad's sides) still count toward the total force but not the map."""
    threshold: float = 0.0
    """Dead band [N]: taxel readings below this are reported as zero."""
    saturation: float = float("inf")
    """Per-taxel full-scale reading [N]."""
    noise_std: float = 0.0
    """Standard deviation of additive Gaussian noise on each taxel [N]."""
    time_constant: float = 0.0
    """First-order response time constant [s]; 0 reports the instantaneous load."""
    seed: int | None = None
    """Seed of the noise generator, re-applied on :meth:`TactilePadSensor.reset`."""

    @staticmethod
    def from_param_args(param_args: str) -> TactilePadParams:
        """Parse inline JSON or a path to a JSON file; empty gives the defaults."""
        text = param_args.strip()
        if not text:
            return TactilePadParams()
        raw = json.loads(text if text.startswith("{") else Path(text).read_text())
        params = TactilePadParams()
        frame = raw.pop("pad_from_sensor", {})
        if "rotation" in frame:
            params.pad_from_sensor_rotation = tuple(float(v) for v in frame["rotation"])
        if "translation" in frame:
            params.pad_from_sensor_translation = tuple(
                float(v) for v in frame["translation"]
            )
        for key, value in raw.items():
            if not hasattr(params, key):
                raise ValueError(f"unknown {TACTILE_PAD_SENSOR_TYPE} parameter '{key}'")
            setattr(params, key, tuple(value) if isinstance(value, list) else value)
        if params.rows < 1 or params.cols < 1:
            raise ValueError("rows and cols must be positive")
        if min(params.size) <= 0:
            raise ValueError("size must be positive")
        return params


@dataclass
class TactileReading:
    """One sample of a :class:`TactilePadSensor`. Forces act *on* the pad."""

    taxels: npt.NDArray[np.float32]
    """Compressive normal load per taxel [N], shape ``(rows, cols)``."""
    force: npt.NDArray[np.float32]
    """Net contact force on the pad in the sensor frame [N]; pressing gives -z."""
    normal_force: float
    """Compressive load along the pad normal, ``max(0, -force[2])`` [N]."""
    tangential_force: float
    """Magnitude of the in-plane (shear) force ``|force[:2]|`` [N]."""
    tangential_direction: float
    """Direction of the shear force in the sensor x-y plane [rad]."""
    center_of_pressure: npt.NDArray[np.float32]
    """Load-weighted contact location in the sensor x-y plane [m]; zeros without load."""
    num_contacts: int
    """Number of loaded contact points on the pad this step."""
    in_contact: bool = field(init=False)

    def __post_init__(self) -> None:
        self.in_contact = self.num_contacts > 0

    def to_vector(self) -> npt.NDArray[np.float32]:
        """Summary features ``[force(3), normal, tangential, cop(2)]`` as float32."""
        return np.concatenate(
            [
                self.force,
                [self.normal_force, self.tangential_force],
                self.center_of_pressure,
            ]
        ).astype(np.float32)


class TactilePadSensor:
    """Taxel-array pressure sensor on a rigid pad link. See the module docstring."""

    # The framework calls the factory with the link actor and the sensor's params.
    def __init__(self, actor: physics.Actor | None, param_args: str):
        if actor is None:
            raise ValueError(f"{TACTILE_PAD_SENSOR_TYPE} requires a pad link actor")
        self.actor = actor
        self.params = TactilePadParams.from_param_args(param_args)
        p = self.params
        self._pad_from_sensor_R = Rotation.from_quat(
            p.pad_from_sensor_rotation
        ).as_matrix()
        self._pad_from_sensor_t = np.asarray(
            p.pad_from_sensor_translation, dtype=np.float64
        )
        self._size = np.asarray(p.size, dtype=np.float64)
        self._pitch = self._size / np.array([p.cols, p.rows], dtype=np.float64)
        self._filtered: npt.NDArray[np.float64] | None = None
        self._rng = np.random.default_rng(p.seed)
        # Queries must be registered before the step whose results they report.
        actor.register_query(physics.QueryType.CONTACT_POINTS)

    # Required by the framework: clear per-episode state, keep params.
    def reset(self) -> None:
        self._filtered = None
        self._rng = np.random.default_rng(self.params.seed)

    @property
    def shape(self) -> tuple[int, int]:
        """Taxel grid shape ``(rows, cols)``."""
        return (self.params.rows, self.params.cols)

    def world_from_sensor(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Sensor pose in the world as ``(R, t)``."""
        world_from_pad = self.actor.get_root_transform()
        R_wp = Rotation.from_quat(list(world_from_pad.rotation)).as_matrix()
        t_wp = np.asarray(world_from_pad.translation, dtype=np.float64)
        return R_wp @ self._pad_from_sensor_R, R_wp @ self._pad_from_sensor_t + t_wp

    def compute_signal(self, dt: float | None = None) -> TactileReading:
        """Read the pad. Call after ``scene.step()``; ``dt`` drives the optional lag."""
        points, forces, normal_loads = self._gather_contacts()
        p = self.params
        rows, cols = p.rows, p.cols
        load_map = np.zeros((rows, cols), dtype=np.float64)

        loaded = normal_loads > 0.0
        num_contacts = int(np.count_nonzero(loaded))
        net_force = forces.sum(axis=0) if len(forces) else np.zeros(3)
        cop = np.zeros(2)
        if num_contacts:
            xy, fn = points[loaded, :2], normal_loads[loaded]
            cop = (xy * fn[:, None]).sum(axis=0) / fn.sum()
            self._splat(load_map, xy, fn)

        load_map = self._apply_response(load_map, dt)
        normal = max(0.0, -float(net_force[2]))
        shear = net_force[:2]
        return TactileReading(
            taxels=load_map.astype(np.float32),
            force=net_force.astype(np.float32),
            normal_force=normal,
            tangential_force=float(np.linalg.norm(shear)),
            tangential_direction=float(np.arctan2(shear[1], shear[0])),
            center_of_pressure=cop.astype(np.float32),
            num_contacts=num_contacts,
        )

    def _gather_contacts(self):
        """Positions and forces of the contacts on the pad's sensing face, in the sensor
        frame, plus each point's compressive load along its own contact normal."""
        # Hot loop: .tolist() and integer handle compares are several times faster than
        # converting each Real3 or comparing handle objects.
        handle = self.actor.get_handle().value
        pos, force, inward, sign = [], [], [], []
        for cp in self.actor.get_contact_points_world():
            is_a = cp.actor_a.value == handle
            if is_a == (cp.actor_b.value == handle):  # self-contact of the pad
                continue
            f = cp.force.tolist()
            if f[0] == 0.0 and f[1] == 0.0 and f[2] == 0.0:  # unloaded near-contact
                continue
            force.append(f)
            inward.append(cp.normal.tolist())  # points away from actor_b
            pos.append(cp.pos_a.tolist() if is_a else cp.pos_b.tolist())
            sign.append(1.0 if is_a else -1.0)
        if not pos:
            empty = np.zeros((0, 3))
            return empty, empty, np.zeros(0)
        # Forces and normals are reported for actor_a; flip them when the pad is actor_b
        # so both are "on the pad" and "into the pad".
        sign = np.asarray(sign)[:, None]
        force = np.asarray(force, dtype=np.float64) * sign
        inward = np.asarray(inward, dtype=np.float64) * sign
        R, t = self.world_from_sensor()
        pos = (np.asarray(pos, dtype=np.float64) - t) @ R
        # Keep contacts pressing on the sensing face (inward normal close to -z).
        facing = -(inward @ R[:, 2]) >= np.cos(
            np.radians(self.params.max_incidence_deg)
        )
        pos, force, inward = pos[facing], force[facing], inward[facing]
        normal_loads = np.maximum(0.0, np.einsum("ij,ij->i", force, inward))
        return pos, force @ R, normal_loads

    def _splat(self, load_map, xy, fn) -> None:
        """Distribute each point load bilinearly onto its four nearest taxels."""
        half = 0.5 * self._size
        on_pad = np.all(np.abs(xy) <= half + self.params.footprint_margin, axis=1)
        xy, fn = xy[on_pad], fn[on_pad]
        rows, cols = load_map.shape
        # Continuous taxel coordinates: taxel (r, c) is centered at (c, r).
        gc = np.clip((xy[:, 0] + half[0]) / self._pitch[0] - 0.5, 0.0, cols - 1)
        gr = np.clip((xy[:, 1] + half[1]) / self._pitch[1] - 0.5, 0.0, rows - 1)
        c0 = np.minimum(np.floor(gc).astype(int), max(cols - 2, 0))
        r0 = np.minimum(np.floor(gr).astype(int), max(rows - 2, 0))
        wc, wr = gc - c0, gr - r0
        c1, r1 = np.minimum(c0 + 1, cols - 1), np.minimum(r0 + 1, rows - 1)
        np.add.at(load_map, (r0, c0), fn * (1 - wr) * (1 - wc))
        np.add.at(load_map, (r0, c1), fn * (1 - wr) * wc)
        np.add.at(load_map, (r1, c0), fn * wr * (1 - wc))
        np.add.at(load_map, (r1, c1), fn * wr * wc)

    def _apply_response(self, load_map, dt):
        p = self.params
        if p.time_constant > 0.0 and dt is not None:
            if self._filtered is None:  # a reset sensor starts unloaded
                self._filtered = np.zeros_like(load_map)
            alpha = dt / (p.time_constant + dt)
            self._filtered = self._filtered + alpha * (load_map - self._filtered)
            load_map = self._filtered
        if p.noise_std > 0.0:
            load_map = load_map + self._rng.normal(0.0, p.noise_std, load_map.shape)
        load_map = np.clip(load_map, 0.0, p.saturation)
        load_map[load_map < p.threshold] = 0.0
        return load_map


def register_tactile_pad_sensor(robotics_context: robotics.RoboticsContext) -> None:
    """Register :class:`TactilePadSensor` as ``TACTILE_PAD``; call before ``create_bot``.

    Safe to call repeatedly: the context is process-global and keeps the first factory.
    """
    if not robotics_context.is_sensor_type_registered(TACTILE_PAD_SENSOR_TYPE):
        robotics.register_python_sensor(
            robotics_context, TACTILE_PAD_SENSOR_TYPE, TactilePadSensor
        )


def find_tactile_pad_sensors(
    bot: robotics.Bot, robotics_context: robotics.RoboticsContext
) -> dict[str, TactilePadSensor]:
    """All live ``TACTILE_PAD`` sensors of ``bot``, keyed by sensor name."""
    sensors = {}
    for handle in bot.find_sensors_by_type(TACTILE_PAD_SENSOR_TYPE):
        sensor = robotics.get_python_sensor(robotics_context, handle)
        if sensor is None:
            raise RuntimeError(
                f"a {TACTILE_PAD_SENSOR_TYPE} sensor is registered by another factory"
            )
        name = bot.get_sensor(handle).get_name()
        sensors[name] = sensor
    return sensors
