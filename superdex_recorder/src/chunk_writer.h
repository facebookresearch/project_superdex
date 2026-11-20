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

#pragma once

#include <superdex_recorder/series.h>

#include <mochi_core/utils/error.h>
#include <mochi_core/utils/guarded.h>

#include <rerun.hpp>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace superdex::recorder::detail {

/* Rows of one series, written as one RRD chunk. */
struct SeriesChunk {
  std::string entityPath;
  std::vector<int64_t> frames;
  std::vector<double> simulationTimesSec;
  std::vector<double> wallTimesSec; // Empty when the samples have no wall time.
  std::unique_ptr<SeriesRows> rows;
  size_t numBytes = 0;
};

/* Writes series chunks to an episode's Rerun stream on its own thread, in submission order. */
class ChunkWriter final {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(ChunkWriter)

 public:
  ChunkWriter(rerun::RecordingStream& stream, size_t chunkMaxBytes, double chunkMaxDurationSec);
  /* Writes the remaining chunks, then stops the thread. */
  ~ChunkWriter();

  /* True once a series has buffered enough samples to submit them as a chunk. */
  [[nodiscard]] bool IsChunkFull(size_t numBytes, double durationSec) const;
  void Submit(SeriesChunk chunk);
  /* Sets the first failure to write a chunk, if any. */
  void CheckError(mochi::Error& error) const;
  /* Writes the remaining chunks, then stops the thread. */
  void Finish(mochi::Error& error);
  /* Drops the remaining chunks, then stops the thread. */
  void Discard();

 private:
  struct State {
    std::vector<SeriesChunk> pending;
    size_t pendingBytes = 0;
    bool stopping = false;
    bool discarding = false;
    bool warnedBehind = false;
    mochi::Error error;
  };

  void Run();
  void Stop(bool discard);

  rerun::RecordingStream& _stream;
  size_t const _chunkMaxBytes = 0;
  double const _chunkMaxDurationSec = 0.0;
  mochi::Guarded<State> _state;
  std::condition_variable _wake;
  // Lets CheckError, which runs on every append, skip the lock until a write fails.
  std::atomic<bool> _failed = false;
  std::thread _thread;
};

} // namespace superdex::recorder::detail
