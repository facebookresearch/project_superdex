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

"""Reads a superdex_recorder recording once and cuts it at recorded frames.

A cut holds the static chunks under /world and, for every component of every entity
under /world, its latest row at or before the frame, with that row's original time.
Rendered with the time cursor following the newest data, it shows what Rerun shows at
that frame, and its size does not grow with the frame number.
"""

from __future__ import annotations

import secrets
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import rerun.experimental as rre  # @manual

FRAME = "frame"
SIM_TIME = "sim_time"
FRAMES_ENTITY = "/data/physics/frames"
PROPERTIES_PREFIX = "/episode/properties/"
ROW_ID = "rerun.controls.RowId"
_NAMED_FRAME_COMPONENTS = (
    "CoordinateFrame:frame",
    "Transform3D:child_frame",
    "Transform3D:parent_frame",
)

# rerun.experimental ships no type information.
_experimental: Any = rre


class RecordingError(ValueError):
    pass


@dataclass(frozen=True)
class Clock:
    """Recorded frame ids and their sim_time, both ascending."""

    frames: np.ndarray
    times_ns: np.ndarray


@dataclass(frozen=True)
class TemporalChunk:
    """A chunk on the frame timeline, with its frame ids."""

    chunk: Any
    batch: pa.RecordBatch
    frames: np.ndarray

    @property
    def entity_path(self) -> str:
        return self.chunk.entity_path


@dataclass(frozen=True)
class _Latest:
    """Rows where one component of one entity holds a value, ascending by frame."""

    component: str
    frames: np.ndarray
    chunks: np.ndarray
    rows: np.ndarray


def renderable(path: str) -> bool:
    """Entities a 3D view of /world draws, plus the frame clock."""
    return path in ("/", "/world", FRAMES_ENTITY) or path.startswith("/world/")


class Recording:
    """All chunks of one recording on the frame timeline, read once.

    Data on other timelines never shows on the frame timeline the videos use.
    """

    def __init__(self, path: Path) -> None:
        reader, stores = _open(path)
        if len(stores) != 1:
            raise RecordingError(f"{path} holds {len(stores)} recordings, expected 1")
        self.application_id: str = stores[0].application_id
        self.recording_id: str = stores[0].recording_id
        self.statics: list[Any] = []
        self.temporal: list[TemporalChunk] = []
        for chunk in _stream(path, reader):
            if chunk.is_static:
                self.statics.append(chunk)
                continue
            batch = chunk.to_record_batch()
            if FRAME in batch.schema.names:
                self.temporal.append(
                    TemporalChunk(chunk, batch, _int64(batch.column(FRAME)))
                )
        _check_recorder_schema(path, self.statics, self.temporal)
        self.clock: Clock = _clock(path, self.temporal)
        self.properties: dict[str, str] = _properties(self.statics)
        self._latest: list[_Latest] = _latest_index(self.temporal)

    def chunks_at(self, frame: int) -> list[Any]:
        """Renderable statics plus each renderable component's latest row at `frame`."""
        picked: dict[tuple[int, int], list[str]] = {}
        for latest in self._latest:
            index = int(np.searchsorted(latest.frames, frame, side="right")) - 1
            if index >= 0:
                source = (int(latest.chunks[index]), int(latest.rows[index]))
                picked.setdefault(source, []).append(latest.component)
        chunks = [c for c in self.statics if renderable(c.entity_path)]
        for (chunk_index, row), components in sorted(picked.items()):
            temporal = self.temporal[chunk_index]
            batch = temporal.batch.slice(row, 1).select(
                _index_columns(temporal.batch) + components
            )
            chunks.append(_new_chunk(batch, temporal.entity_path))
        return chunks

    def write_at(self, frame: int, path: Path) -> None:
        store = _experimental.ChunkStore.from_chunks(self.chunks_at(frame))
        store.write_rrd(
            path,
            application_id=self.application_id,
            recording_id=f"{self.recording_id}-frame-{frame}",
        )


def _open(path: Path) -> tuple[Any, list[Any]]:
    """The reader and its recordings; refuses files without a footer, which every closed
    recording has, so that a truncated file is never rendered."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            reader = _experimental.RrdReader(path)
            stores = reader.recordings()
        except (OSError, RuntimeError, ValueError) as e:
            raise RecordingError(f"Cannot read {path}: {e}") from e
    if any("no footer" in str(w.message) for w in caught):
        raise RecordingError(
            f"{path} has no footer: the file is truncated or its recording was not closed"
        )
    return reader, stores


def _stream(path: Path, reader: Any) -> list[Any]:
    try:
        return list(reader.stream())
    except (OSError, RuntimeError, ValueError) as e:
        raise RecordingError(f"Cannot read {path}: {e}") from e


def _int64(column: pa.Array) -> np.ndarray:
    return np.asarray(column.cast(pa.int64()).to_numpy(zero_copy_only=False))


def _kind(field: pa.Field) -> bytes | None:
    return (field.metadata or {}).get(b"rerun:kind")


def _index_columns(batch: pa.RecordBatch) -> list[str]:
    return [f.name for f in batch.schema if _kind(f) in (b"control", b"index")]


def _data_columns(batch: pa.RecordBatch) -> list[str]:
    return [f.name for f in batch.schema if _kind(f) == b"data"]


def _new_chunk(batch: pa.RecordBatch, entity_path: str) -> Any:
    """A chunk with a fresh id: a store keeps only one chunk per id."""
    metadata = dict(batch.schema.metadata or {})
    metadata[b"rerun:id"] = (
        f"chunk_{time.time_ns():016X}{secrets.randbits(64):016x}".encode()
    )
    chunk = _experimental.Chunk.from_record_batch(
        batch.replace_schema_metadata(metadata)
    )
    return chunk.with_entity_path(entity_path)


def _row_order(batch: pa.RecordBatch) -> tuple[np.ndarray, np.ndarray]:
    """Row ids as two big-endian words, which order rows logged at the same frame."""
    column = batch.column(ROW_ID)
    raw = np.frombuffer(column.buffers()[1], dtype=np.uint8)
    words = raw[column.offset * 16 : (column.offset + len(column)) * 16].view(">u8")
    words = words.reshape(-1, 2).astype(np.uint64)
    return words[:, 0], words[:, 1]


def _latest_index(temporal: list[TemporalChunk]) -> list[_Latest]:
    parts: dict[tuple[str, str], list[tuple[np.ndarray, ...]]] = {}
    for chunk_index, chunk in enumerate(temporal):
        if not renderable(chunk.entity_path):
            continue
        high, low = _row_order(chunk.batch)
        for name in _data_columns(chunk.batch):
            valid = chunk.batch.column(name).is_valid().to_numpy(zero_copy_only=False)
            rows = np.flatnonzero(valid)
            columns = (
                chunk.frames[rows],
                high[rows],
                low[rows],
                np.full(len(rows), chunk_index),
                rows,
            )
            parts.setdefault((chunk.entity_path, name), []).append(columns)
    index = []
    for (_, name), pieces in sorted(parts.items()):
        frames, high, low, chunks, rows = (np.concatenate(c) for c in zip(*pieces))
        order = np.lexsort((low, high, frames))
        index.append(_Latest(name, frames[order], chunks[order], rows[order]))
    return index


def _check_recorder_schema(
    path: Path, statics: list[Any], temporal: list[TemporalChunk]
) -> None:
    """Only superdex_recorder episodes: their components hold values until overwritten."""
    if static_text(statics, "/episode", "superdex.episode.schema_name") is None:
        raise RecordingError(
            f"{path} has no /episode schema: not a superdex_recorder recording"
        )
    batches = [(c.entity_path, c.to_record_batch()) for c in statics]
    batches += [(t.entity_path, t.batch) for t in temporal]
    for entity_path, batch in batches:
        names = batch.schema.names
        if renderable(entity_path) and any(n.startswith("Clear:") for n in names):
            raise RecordingError(
                f"{path}: Clear events at {entity_path} are not supported"
            )
        # A cut keeps only /world, where named transform frames could be resolved wrongly.
        if any(n in _NAMED_FRAME_COMPONENTS for n in names):
            raise RecordingError(
                f"{path}: named transform frames at {entity_path} are not supported"
            )


def _clock(path: Path, temporal: list[TemporalChunk]) -> Clock:
    """Frame clock from /data/physics/frames, or from every frame/sim_time pair."""
    pairs = [t for t in temporal if SIM_TIME in t.batch.schema.names]
    selected = [t for t in pairs if t.entity_path == FRAMES_ENTITY] or pairs
    if not selected:
        raise RecordingError(
            f"{path} has no frame and sim_time timelines: not a superdex_recorder recording"
        )
    frames = np.concatenate([t.frames for t in selected])
    times = np.concatenate([_int64(t.batch.column(SIM_TIME)) for t in selected])
    order = np.lexsort((times, frames))
    frames, times = frames[order], times[order]
    unique, first = np.unique(frames, return_index=True)
    last = np.append(first[1:] - 1, len(frames) - 1)
    if np.any(times[first] != times[last]):
        bad = int(unique[np.flatnonzero(times[first] != times[last])[0]])
        raise RecordingError(f"{path}: frame {bad} has conflicting sim_time values")
    unique_times = times[first]
    if np.any(np.diff(unique_times) < 0):
        raise RecordingError(f"{path}: sim_time decreases as frames increase")
    return Clock(unique.astype(np.int64), unique_times.astype(np.int64))


def _properties(statics: list[Any]) -> dict[str, str]:
    """Episode properties written by superdex_recorder EpisodeMetadata."""
    properties = {}
    for chunk in statics:
        if not chunk.entity_path.startswith(PROPERTIES_PREFIX):
            continue
        batch = chunk.to_record_batch()
        if "superdex.episode.property_key" not in batch.schema.names:
            continue
        key = _first_text(batch.column("superdex.episode.property_key"))
        value = _first_text(batch.column("superdex.episode.property_value"))
        if key is not None and value is not None:
            properties[key] = value
    return properties


def _first_text(column: pa.Array) -> str | None:
    value = column[0].as_py() if len(column) else None
    while isinstance(value, list):
        value = value[0] if value else None
    return None if value is None else str(value)


def static_text(statics: list[Any], entity_path: str, component: str) -> str | None:
    """The first text value of a static component, or None."""
    for chunk in statics:
        if chunk.entity_path != entity_path:
            continue
        batch = chunk.to_record_batch()
        if component in batch.schema.names:
            return _first_text(batch.column(component))
    return None
