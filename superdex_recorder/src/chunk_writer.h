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
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
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

/* Emptied chunks, kept by entity path so that the series with that path, in the same or a later
 * episode of a recorder, continue in grown buffers instead of growing new ones. A series fills one
 * chunk while the writer writes the other, so it keeps two per path. */
class ChunkPool final {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(ChunkPool)

 public:
  static constexpr size_t kMaxChunksPerPath = 2;

  ChunkPool() = default;
  ~ChunkPool() = default;

  /* The oldest kept chunk for the entity path, or an empty chunk without rows if none is kept. */
  [[nodiscard]] SeriesChunk Take(std::string const& entityPath);
  /* Keeps an emptied chunk, unless its entity path already has kMaxChunksPerPath. */
  void Give(SeriesChunk chunk);
  /* Frees the chunks of entity paths that no series took and no writer gave since the previous
   * episode began. */
  void BeginEpisode();

 private:
  struct Entry {
    std::vector<SeriesChunk> chunks;
    uint64_t lastUsedEpisode = 0; // numEpisodes when a chunk of the path was last taken or given.
  };
  struct State {
    std::unordered_map<std::string, Entry> entries;
    uint64_t numEpisodes = 0; // Episodes begun.
  };

  mochi::Guarded<State> _state;
};

/* Writes series chunks to an episode's Rerun stream on its own thread, in submission order, then
 * gives their emptied buffers to the pool. Each episode owns one writer, internal to it, whose
 * thread starts with the writer. Destroying the writer waits for that thread; after FinishAsync,
 * until finish has returned. */
class ChunkWriter final {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(ChunkWriter)

 public:
  ChunkWriter(
      rerun::RecordingStream& stream,
      size_t chunkMaxBytes,
      double chunkMaxDurationSec,
      std::shared_ptr<ChunkPool> pool);
  /* Writes the remaining chunks, then stops the thread. */
  ~ChunkWriter();

  /* True once a series has buffered enough samples to submit them as a chunk. */
  [[nodiscard]] bool IsChunkFull(size_t numBytes, double durationSec) const;
  /* Queues a chunk to write, or recycles it if it has no samples. */
  void Submit(SeriesChunk chunk);
  /* Empty buffers for the next chunk of the series with this entity path. */
  [[nodiscard]] SeriesChunk TakeEmptyChunk(std::string const& entityPath);
  /* Empties a chunk that is written or no longer needed, and keeps its buffers for the next chunk
   * of its entity path. */
  void Recycle(SeriesChunk chunk);
  /* Sets the first failure to write a chunk, if any. */
  void CheckError(mochi::Error& error) const;
  /* Writes the remaining chunks, then stops the thread. */
  void Finish(mochi::Error& error);
  /* Returns at once. The thread writes the remaining chunks, calls finish with the first failure,
   * if any, then stops. finish runs on the thread, so it must not destroy the writer. */
  void FinishAsync(std::function<void(mochi::Error&)> finish);
  /* True once the thread has stopped: after Finish or Discard, or once FinishAsync's finish has
   * returned. */
  [[nodiscard]] bool IsFinished() const;
  /* Waits for the thread to stop after FinishAsync. */
  void Wait();
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
    std::function<void(mochi::Error&)> finish;
  };

  void Run();
  void Stop(bool discard);

  rerun::RecordingStream& _stream;
  std::shared_ptr<ChunkPool> const _pool;
  size_t const _chunkMaxBytes = 0;
  double const _chunkMaxDurationSec = 0.0;
  mochi::Guarded<State> _state;
  std::condition_variable _wake;
  // Lets CheckError, which runs on every append, skip the lock until a write fails.
  std::atomic<bool> _failed = false;
  std::atomic<bool> _finished = false;
  std::thread _thread;
};

} // namespace superdex::recorder::detail
