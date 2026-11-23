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

import json
import tempfile
from pathlib import Path

from test.conftest import MochiTestBase, sdp, sdr


class DRTransportTest(MochiTestBase):
    def test_opaque_metadata_round_trip(self) -> None:
        payload = {
            "profile": {"name": "example", "values": [0.25, True, None]},
            "schema_version": "producer-owned label",
            "lineage": {"episode_id": str(2**64 - 1)},
        }
        manifest = sdr.DREpisodeList(
            schema_version=2,
            spec_version="test",
            master_seed="1",
            metadata=json.dumps(payload),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.json"
            sdr.save_dr_episode_list_to_file(manifest, str(path))
            wire = json.loads(path.read_text())
            self.assertEqual(wire["metadata"], payload)
            self.assertNotIn("profile", wire)
            loaded = sdr.load_dr_episode_list_from_file(str(path))
        self.assertEqual(json.loads(loaded.metadata), payload)

    def test_real_array_registration_reuses_physics_type(self) -> None:
        values = sdr.DREpisodeEntry(value=[1.0000000000000002]).value
        self.assertIs(type(values), sdp.DynamicArrayReal)
        expected = 1.0000000000000002 if sdp.uses_double_precision() else 1.0
        self.assertEqual(values[0], expected)

    def test_schema_two_signed_indices_and_application_evidence(self) -> None:
        for index in (0, 2**31, 2**53 + 1, 2**63 - 1):
            for applied in (None, [], ["dr/test/value"]):
                with self.subTest(index=index, applied=applied):
                    episode = sdr.DREpisode(
                        index=index,
                        episode_seed="2",
                        values=[
                            sdr.DREpisodeEntry(
                                key="dr/test/value",
                                scalar=True,
                                value=[1.0000000000000002],
                            )
                        ],
                    )
                    episode.applied_keys = applied
                    episode.index = index
                    manifest = sdr.DREpisodeList(
                        schema_version=2,
                        spec_version="test",
                        master_seed="1",
                        episodes=[episode],
                    )
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "episodes.json"
                        sdr.save_dr_episode_list_to_file(manifest, str(path))
                        wire = json.loads(path.read_text())
                        self.assertEqual(wire["episodes"][0]["index"], index)
                        self.assertEqual(
                            "applied_keys" in wire["episodes"][0], applied is not None
                        )
                        loaded = sdr.load_dr_episode_list_from_file(str(path))
                    self.assertEqual(loaded.episodes[0].index, index)
                    expected = (
                        1.0000000000000002 if sdp.uses_double_precision() else 1.0
                    )
                    self.assertEqual(loaded.episodes[0].values[0].value[0], expected)
                    self.assertEqual(
                        loaded.episodes[0].applied_keys is None, applied is None
                    )
                    if applied is not None:
                        self.assertEqual(
                            [str(key) for key in loaded.episodes[0].applied_keys],
                            applied,
                        )

    def test_binding_rejects_out_of_range_indices(self) -> None:
        for index in (-(2**63) - 1, 2**63):
            with self.subTest(index=index):
                with self.assertRaises((TypeError, OverflowError, RuntimeError)):
                    sdr.DREpisode(index=index)
                episode = sdr.DREpisode()
                with self.assertRaises((TypeError, OverflowError, RuntimeError)):
                    episode.index = index

    def test_negative_indices_fail_validation(self) -> None:
        episode = sdr.DREpisode(index=-1, episode_seed="2")
        manifest = sdr.DREpisodeList(
            schema_version=2, spec_version="test", master_seed="1", episodes=[episode]
        )
        with self.assertRaises(RuntimeError):
            sdr.validate_dr_episode_list(manifest)
