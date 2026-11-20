/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "chunk_writer.h"

#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/log.h>

#include <mutex>
#include <string_view>
#include <utility>

namespace superdex::recorder::detail {
namespace {

constexpr std::string_view kFrameTimeline = "frame";
constexpr std::string_view kSimulationTimeTimeline = "sim_time";
constexpr std::string_view kWallTimeTimeline = "wall_time";
constexpr size_t kWarnBehindBytes = 1'000'000'000; // 1 GB

void SetRerunError(rerun::Error const& rerunError, mochi::Error& error, char const* description) {
  if (rerunError.is_ok()) {
    return;
  }
  MOCHI_LOG_ERROR("%s: %s", description, rerunError.description.c_str());
  error.SetFirstError(description, __FILE__, __LINE__);
}

void WriteChunk(rerun::RecordingStream& stream, SeriesChunk& chunk, mochi::Error& error) {
  std::vector<rerun::ComponentColumn> componentColumns;
  chunk.rows->BuildColumns(componentColumns, error);
  MOCHI_ERROR_RETURN(error);

  std::vector<rerun::TimeColumn> timeColumns;
  timeColumns.reserve(3);
  timeColumns.emplace_back(
      rerun::TimeColumn::from_sequence(
          std::string{kFrameTimeline}, std::move(chunk.frames), rerun::SortingStatus::Sorted));
  timeColumns.emplace_back(
      rerun::TimeColumn::from_duration_secs(
          std::string{kSimulationTimeTimeline},
          std::move(chunk.simulationTimesSec),
          rerun::SortingStatus::Sorted));
  if (!chunk.wallTimesSec.empty()) {
    timeColumns.emplace_back(
        rerun::TimeColumn::from_duration_secs(
            std::string{kWallTimeTimeline},
            std::move(chunk.wallTimesSec),
            rerun::SortingStatus::Sorted));
  }
  SetRerunError(
      stream.try_send_columns(
          chunk.entityPath,
          rerun::Collection<rerun::TimeColumn>::take_ownership(std::move(timeColumns)),
          rerun::Collection<rerun::ComponentColumn>::take_ownership(std::move(componentColumns))),
      error,
      "Failed to write sampled series.");
}

} // namespace

ChunkWriter::ChunkWriter(
    rerun::RecordingStream& stream,
    size_t chunkMaxBytes,
    double chunkMaxDurationSec)
    : _stream{stream},
      _chunkMaxBytes{chunkMaxBytes},
      _chunkMaxDurationSec{chunkMaxDurationSec},
      _thread{[this] { Run(); }} {}

ChunkWriter::~ChunkWriter() {
  Stop(/*discard=*/false);
}

bool ChunkWriter::IsChunkFull(size_t numBytes, double durationSec) const {
  return (_chunkMaxBytes != 0 && numBytes >= _chunkMaxBytes) ||
      (_chunkMaxDurationSec > 0.0 && durationSec >= _chunkMaxDurationSec);
}

void ChunkWriter::Submit(SeriesChunk chunk) {
  size_t pendingBytes = 0;
  bool warn = false;
  _state.Mutate([&](State& state) {
    MOCHI_ASSERT_VERBOSE(!state.stopping);
    state.pendingBytes += chunk.numBytes;
    state.pending.push_back(std::move(chunk));
    pendingBytes = state.pendingBytes;
    warn = !state.warnedBehind && pendingBytes > kWarnBehindBytes;
    state.warnedBehind = state.warnedBehind || warn;
  });
  _wake.notify_one();
  if (warn) {
    MOCHI_LOG_WARNING(
        "The recorder's writer thread is %zu MB behind; the disk may be too slow.",
        pendingBytes / 1'000'000);
  }
}

void ChunkWriter::CheckError(mochi::Error& error) const {
  if (!_failed.load(std::memory_order_acquire)) {
    return;
  }
  _state.Read([&](State const& state) {
    error.SetFirstError(state.error.GetDescription(), state.error.GetFile(), state.error.GetLine());
  });
}

void ChunkWriter::Finish(mochi::Error& error) {
  Stop(/*discard=*/false);
  CheckError(error);
}

void ChunkWriter::Discard() {
  Stop(/*discard=*/true);
}

void ChunkWriter::Stop(bool discard) {
  _state.Mutate([&](State& state) {
    state.stopping = true;
    state.discarding = state.discarding || discard;
  });
  _wake.notify_one();
  if (_thread.joinable()) {
    _thread.join();
  }
}

void ChunkWriter::Run() {
  while (true) {
    std::vector<SeriesChunk> chunks;
    _state.MutateWithLock([&](std::unique_lock<std::mutex>& lock, State& state) {
      _wake.wait(lock, [&] { return state.stopping || !state.pending.empty(); });
      if (state.discarding) {
        state.pending.clear();
        state.pendingBytes = 0;
      }
      chunks = std::exchange(state.pending, {});
    });
    if (chunks.empty()) {
      break; // Stopping with every chunk written or discarded.
    }
    for (auto& chunk : chunks) {
      mochi::Error error;
      if (!_failed.load(std::memory_order_relaxed)) {
        WriteChunk(_stream, chunk, error);
      }
      bool const discarding = _state.Mutate([&](State& state) {
        state.pendingBytes -= chunk.numBytes;
        if (!error.IsOK() && state.error.IsOK()) {
          state.error = error.Copy();
          _failed.store(true, std::memory_order_release);
        }
        return state.discarding;
      });
      chunk = {};
      if (discarding) {
        break;
      }
    }
  }
}

} // namespace superdex::recorder::detail
