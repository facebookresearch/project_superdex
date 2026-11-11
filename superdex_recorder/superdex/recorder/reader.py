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

"""Typed access to the SuperDex components stored in an episode RRD."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import rerun.experimental as rr_experimental

# The Rerun wheel does not expose its experimental API to static type checkers.
_rr_experimental: Any = rr_experimental

SCHEMA_NAME = "superdex.episode"
SCHEMA_VERSION = "1.1"
SCHEMA_VERSION_MAJOR = 1


@dataclass(frozen=True)
class ComponentSample:
    """One component value and the Rerun timelines attached to its row.

    ``frame`` is the frame number and ``sim_time`` the simulation time in nanoseconds.
    ``wall_time``, when the episode recorded it, is the wall-clock time since the episode
    began in nanoseconds, from a monotonic clock.
    """

    entity_path: str
    timelines: dict[str, int]
    value: Any
    is_static: bool


class EpisodeReader:
    """Read and validate SuperDex episode components without exposing Rerun internals."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._reader = _rr_experimental.RrdReader(self.path)
        recordings = self._reader.recordings()
        if len(recordings) != 1:
            raise ValueError(
                f"Expected one recording store in {self.path}, found {len(recordings)}"
            )
        schema_name = self.read_static_value("superdex.episode.schema_name")
        schema_version = self.read_static_value("superdex.episode.schema_version")
        try:
            schema_major = int(schema_version.split(".", 1)[0])
        except (AttributeError, ValueError):
            schema_major = -1
        if schema_name != SCHEMA_NAME or schema_major != SCHEMA_VERSION_MAJOR:
            raise ValueError(
                f"Unsupported SuperDex schema {schema_name!r} version {schema_version!r}"
            )

    def component_samples(
        self,
        component: str,
        *,
        entity_path: str | None = None,
        frames: Collection[int] | None = None,
    ) -> list[ComponentSample]:
        """Return all rows containing ``component``, ordered by frame when present."""
        return self.component_samples_many(
            (component,), entity_path=entity_path, frames=frames
        )[component]

    def component_samples_many(
        self,
        components: Sequence[str],
        *,
        entity_path: str | None = None,
        frames: Collection[int] | None = None,
    ) -> dict[str, list[ComponentSample]]:
        """Return several components while decoding each matching chunk once.

        With ``frames``, return only rows at those frames. Static rows have no frame and
        are excluded. Other rows are dropped before they are converted to Python, which
        makes reading a few frames of a large component much faster.
        """
        stream = self._reader.stream().filter(
            content=entity_path,
            components=components,
        )
        frame_set = None if frames is None else pa.array(list(frames), type=pa.int64())
        samples: dict[str, list[ComponentSample]] = {
            component: [] for component in components
        }
        for chunk in stream:
            batch = chunk.to_record_batch()
            if frame_set is not None:
                frame_index = batch.schema.get_field_index("frame")
                if frame_index < 0:
                    continue
                batch = batch.filter(
                    pc.is_in(batch.column(frame_index), value_set=frame_set)
                )
            timeline_values = {}
            for timeline in chunk.timeline_names:
                if timeline not in {"frame", "sim_time", "wall_time"}:
                    continue
                timeline_index = batch.schema.get_field_index(timeline)
                if timeline_index < 0:
                    raise ValueError(
                        f"Chunk for {chunk.entity_path!r} is missing its "
                        f"{timeline!r} timeline column"
                    )
                column = batch.column(timeline_index)
                if pa.types.is_duration(column.type):
                    # A Python timedelta cannot hold nanoseconds.
                    column = column.cast(pa.duration("ns")).cast(pa.int64())
                timeline_values[timeline] = column.to_pylist()
            for component in components:
                component_index = batch.schema.get_field_index(component)
                if component_index < 0:
                    continue
                for row, value in enumerate(batch.column(component_index).to_pylist()):
                    if value is None:
                        continue
                    samples[component].append(
                        ComponentSample(
                            entity_path=chunk.entity_path,
                            timelines={
                                timeline: values[row]
                                for timeline, values in timeline_values.items()
                            },
                            value=value,
                            is_static=chunk.is_static,
                        )
                    )
        return {
            component: sorted(
                component_samples,
                key=lambda sample: (
                    sample.timelines.get("frame", -1),
                    sample.entity_path,
                ),
            )
            for component, component_samples in samples.items()
        }

    def _component_columns(
        self, components: Sequence[str]
    ) -> dict[tuple[str, str], pa.Table]:
        """Return each entity's rows of ``components`` as Arrow tables.

        Tables are keyed by (entity path, component), have ``frame`` and ``value``
        columns, are sorted by frame, and have no rows without a value. Rows at the
        same frame keep their recorded order. Unlike :meth:`component_samples_many`,
        nothing is converted to Python, which makes reading whole components much
        faster. Raises ValueError if a row has no frame, such as a static row.
        """
        tables: dict[tuple[str, str], list[pa.Table]] = {}
        for chunk in self._reader.stream().filter(components=components):
            batch = chunk.to_record_batch()
            frame_index = batch.schema.get_field_index("frame")
            for component in components:
                component_index = batch.schema.get_field_index(component)
                if component_index < 0:
                    continue
                values = batch.column(component_index)
                if values.null_count == len(values):
                    continue
                if frame_index < 0:
                    raise ValueError(
                        f"{chunk.entity_path!r} has a row without a frame."
                    )
                table = pa.table({"frame": batch.column(frame_index), "value": values})
                tables.setdefault((chunk.entity_path, component), []).append(
                    table.filter(values.is_valid())
                )
        return {
            key: pa.concat_tables(parts).sort_by("frame")
            for key, parts in tables.items()
        }

    def read_static_value(self, component: str) -> Any:
        """Read the sole static value for ``component``.

        Single-element values are returned unwrapped; longer arrays are returned as lists.
        """
        samples = self.component_samples(component)
        if len(samples) != 1 or not samples[0].is_static:
            raise ValueError(
                f"Expected exactly one static value for {component!r}; got {samples!r}"
            )
        value = samples[0].value
        if isinstance(value, list) and len(value) == 1:
            return value[0]
        return value
