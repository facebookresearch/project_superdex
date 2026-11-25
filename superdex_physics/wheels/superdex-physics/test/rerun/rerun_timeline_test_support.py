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

"""Shared Rerun timeline contract checks for the mirrored Python loggers."""

from __future__ import annotations

import gc
import os
import tempfile
import time
import unittest
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import numpy as np


TransformRow = tuple[int, float, np.ndarray, np.ndarray]
ROOT_JOINT_OFFSET = (0.0, 0.2, 0.0)


def flush_and_load_rrd(save_path: str) -> Any:
    """Flush the active Rerun recording and load its RRD."""
    import rerun as rr

    rr.init("test_flush", recording_id="flush")
    gc.collect()
    time.sleep(0.5)
    return rr.experimental.RrdReader(save_path).stream().collect()


def _sim_time_seconds(value: int | float | timedelta) -> float:
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, int):
        return value / 1_000_000_000.0
    return float(value)


def _x_rotation(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float64,
    )


def _canonical_entity_path(path: str) -> str:
    return "/" + "/".join(part for part in path.split("/") if part)


def _read_transform_rows(
    recording: Any, root_path: str
) -> dict[str, list[TransformRow]]:
    rows_by_path: dict[str, list[TransformRow]] = {}
    for chunk in recording.stream().filter(has_timeline="frame"):
        entity_path = _canonical_entity_path(str(chunk.entity_path))
        if entity_path != root_path and not entity_path.startswith(f"{root_path}/"):
            continue
        batch = chunk.to_record_batch()
        translation_col = next(
            (name for name in batch.schema.names if "translation" in name.lower()),
            None,
        )
        matrix_col = next(
            (name for name in batch.schema.names if "mat3x3" in name.lower()), None
        )
        if translation_col is None or matrix_col is None:
            continue
        for frame, sim_time, translation, matrix in zip(
            batch.column("frame").to_pylist(),
            batch.column("sim_time").to_pylist(),
            batch.column(translation_col).to_pylist(),
            batch.column(matrix_col).to_pylist(),
        ):
            if translation is None or matrix is None:
                continue
            rows_by_path.setdefault(entity_path, []).append(
                (
                    frame,
                    _sim_time_seconds(sim_time),
                    np.asarray(translation, dtype=np.float64).reshape(3),
                    # Rerun stores TransformMat3x3 flat columns column-major.
                    np.asarray(matrix, dtype=np.float64).reshape(3, 3).T,
                )
            )
    return rows_by_path


def check_scene_snapshot_timeline(
    test_case: unittest.TestCase,
    *,
    create_logger: Callable[[str], Any],
    make_empty_scene: Callable[[], Any],
    add_rigid_cube: Callable[..., Any],
    physics_module: Any,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.rrd")
        with create_logger(path) as logger, make_empty_scene() as scene:
            actor = add_rigid_cube(
                scene,
                "Cube",
                position=physics_module.Real3(1.0, 2.0, 3.0),
            )
            solid_path = _canonical_entity_path(
                f"world/{actor.get_contact_layer()}/"
                f"{actor.get_name()}_h{actor.get_handle().value}"
            )
            logger.set_scene(scene)
            logger.log_scene_snapshot(frame_idx=-1, sim_time=-0.05)
            test_case.assertAlmostEqual(logger.get_sim_time(), -0.05)
            logger.log_frame()
            test_case.assertAlmostEqual(logger.get_sim_time(), 0.0)

        rows = _read_transform_rows(flush_and_load_rrd(path), solid_path)[solid_path]
        test_case.assertEqual(len(rows), 2)
        rows_by_frame = {row[0]: row for row in rows}
        test_case.assertEqual(set(rows_by_frame), {-1, 0})
        test_case.assertAlmostEqual(rows_by_frame[-1][1], -0.05)
        test_case.assertAlmostEqual(rows_by_frame[0][1], 0.0)
        test_case.assertGreater(np.linalg.norm(rows_by_frame[-1][2]), 0.0)
        np.testing.assert_allclose(rows_by_frame[-1][2], rows_by_frame[0][2])
        np.testing.assert_allclose(rows_by_frame[-1][3], rows_by_frame[0][3])


def check_default_overlay_timeline(
    test_case: unittest.TestCase,
    *,
    create_logger: Callable[[str], Any],
    make_empty_scene: Callable[[], Any],
    create_revolute_chain: Callable[..., Any],
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.rrd")
        with create_logger(path) as logger, make_empty_scene() as scene:
            create_revolute_chain(scene, num_joints=3)
            overlay = logger.create_overlay("ghost", scene=scene)
            root_path = overlay.get_entity_path("TestRobot")
            if root_path is None:
                test_case.fail("TestRobot overlay has no root entity path")
            root_path = _canonical_entity_path(root_path)

        recording = flush_and_load_rrd(path)
        index_names = {col.name for col in recording.schema().index_columns()}
        test_case.assertIn("frame", index_names, f"index columns: {index_names}")
        test_case.assertIn("sim_time", index_names, f"index columns: {index_names}")
        rows_by_path = _read_transform_rows(recording, root_path)
        test_case.assertTrue(rows_by_path)
        for rows in rows_by_path.values():
            test_case.assertEqual({row[0] for row in rows}, {0})
            for row in rows:
                test_case.assertAlmostEqual(row[1], 0.0)


def _check_preroll_timeline_rows(
    test_case: unittest.TestCase,
    rows_by_path: dict[str, list[TransformRow]],
    root_path: str,
    expected_preroll_only_path: str,
    expected_setup_sim_time: float,
    expected_entity_count: int,
) -> set[str]:
    test_case.assertEqual(len(rows_by_path), expected_entity_count)
    preroll_paths = {
        entity_path
        for entity_path, rows in rows_by_path.items()
        if any(frame == -1 for frame, *_ in rows)
    }
    frame_zero_paths = {
        entity_path
        for entity_path, rows in rows_by_path.items()
        if any(frame == 0 for frame, *_ in rows)
    }
    test_case.assertEqual(preroll_paths, set(rows_by_path))
    test_case.assertIn(root_path, frame_zero_paths)
    test_case.assertEqual(len(frame_zero_paths), expected_entity_count - 1)
    test_case.assertEqual(
        preroll_paths - frame_zero_paths,
        {expected_preroll_only_path},
    )

    for entity_path, rows in rows_by_path.items():
        setup_rows = [row for row in rows if row[0] == -1]
        source_rows = [row for row in rows if row[0] == 0]
        test_case.assertEqual(len(setup_rows), 1, entity_path)
        test_case.assertAlmostEqual(setup_rows[0][1], expected_setup_sim_time)
        test_case.assertEqual(
            len(source_rows), 1 if entity_path in frame_zero_paths else 0
        )
        if source_rows:
            test_case.assertAlmostEqual(source_rows[0][1], 0.0)
        test_case.assertEqual(
            {row[0] for row in rows}, {-1, 0} if source_rows else {-1}
        )
    return frame_zero_paths


def _check_preroll_source_values(
    rows_by_path: dict[str, list[TransformRow]],
    root_path: str,
    frame_zero_paths: set[str],
    source_anchor: np.ndarray,
    expected_root_setup_translation: np.ndarray,
) -> None:
    root_setup = next(row for row in rows_by_path[root_path] if row[0] == -1)
    root_source = next(row for row in rows_by_path[root_path] if row[0] == 0)
    np.testing.assert_allclose(root_setup[2], expected_root_setup_translation)
    np.testing.assert_allclose(root_source[2], source_anchor[:3, 3])
    np.testing.assert_allclose(root_source[3], np.eye(3))

    active_leaf_paths = frame_zero_paths - {root_path}
    for entity_path in active_leaf_paths:
        setup = next(row for row in rows_by_path[entity_path] if row[0] == -1)
        np.testing.assert_allclose(setup[3], np.eye(3), atol=1e-12)
    actual_leaf_rotations = [
        next(row for row in rows_by_path[entity_path] if row[0] == 0)[3]
        for entity_path in active_leaf_paths
    ]
    actual_leaf_rotations.sort(key=lambda matrix: float(matrix[2, 1]))
    expected_leaf_rotations = [_x_rotation(0.25), _x_rotation(0.5)]
    np.testing.assert_allclose(
        np.stack(actual_leaf_rotations), np.stack(expected_leaf_rotations)
    )


def check_overlay_preroll_timeline(
    test_case: unittest.TestCase,
    *,
    create_logger: Callable[[str], Any],
    make_empty_scene: Callable[[], Any],
    create_revolute_chain: Callable[..., Any],
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.rrd")
        with create_logger(path) as logger, make_empty_scene() as scene:
            num_joints = 3
            actor = create_revolute_chain(scene, num_joints=num_joints)
            logger.set_scene(scene)
            setup_sim_time = -0.05
            overlay = logger.create_overlay(
                "ghost",
                scene=scene,
                setup_frame=-1,
                setup_sim_time=setup_sim_time,
            )
            root_path = overlay.get_entity_path("TestRobot")
            if root_path is None:
                test_case.fail("TestRobot overlay has no root entity path")
            root_path = _canonical_entity_path(root_path)
            anchor_link_name = str(
                actor.get_articulated_shape_info().link_names[0]
            ).replace("/", "_")
            anchor_leaf_path = (
                f"{root_path}/{anchor_link_name}_mount/{anchor_link_name}"
            )

            source_joints = np.array([0.0, 0.25, 0.5], dtype=np.float64)
            test_case.assertEqual(actor.get_num_dofs(), len(source_joints))
            source_anchor = np.eye(4, dtype=np.float64)
            source_anchor[:3, 3] = [1.0, 2.0, 3.0]
            expected_root_setup_translation = np.asarray(
                ROOT_JOINT_OFFSET, dtype=np.float64
            )
            logger.log_frame(frame_idx=0)
            overlay.set_articulated_pose(
                "TestRobot",
                source_joints,
                anchor_transform=source_anchor,
            )

        recording = flush_and_load_rrd(path)
        index_names = {col.name for col in recording.schema().index_columns()}
        test_case.assertIn("sim_time", index_names)
        rows_by_path = _read_transform_rows(recording, root_path)
        frame_zero_paths = _check_preroll_timeline_rows(
            test_case,
            rows_by_path,
            root_path,
            anchor_leaf_path,
            setup_sim_time,
            expected_entity_count=num_joints + 1,
        )
        _check_preroll_source_values(
            rows_by_path,
            root_path,
            frame_zero_paths,
            source_anchor,
            expected_root_setup_translation,
        )
