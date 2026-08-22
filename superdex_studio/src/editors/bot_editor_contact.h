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

#include <superdex_robotics/superdex_robotics.h>

#include <atomic>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <utility>

namespace superdex::studio {

// An unordered pair of link indices into a bot's @ref superdex::robotics::BotPrefab::links.
struct BotLinkPair {
  int linkA = superdex::robotics::kIndexNone;
  int linkB = superdex::robotics::kIndexNone;
};

// Editing model for a bot's contact-filter overrides. Encapsulates the rule the
// contact-filter UI applies: a pair's default state is "enabled" unless Mochi
// implicitly disables it, and an override is stored only when it differs from
// that default. Caches the implicit-disable mask for the prefab's topology.
struct BotContactFilterBuilder {
  // Computes the implicit-disable mask for the prefab's current topology (O(N^2)).
  // The builder binds to `prefab` for its lifetime; all queries and edits operate
  // on it and assume its topology (link count and order) does not change. Do not
  // outlive the referenced prefab.
  explicit BotContactFilterBuilder(superdex::robotics::BotPrefab& prefab);

  // True if Mochi implicitly disables contact between the two links.
  [[nodiscard]] bool IsImplicitlyDisabled(int linkA, int linkB) const;
  // The default checkbox state for a pair (enabled unless implicitly disabled).
  [[nodiscard]] bool DefaultEnabled(int linkA, int linkB) const;
  // True if the bound prefab has an explicit override for the pair.
  [[nodiscard]] bool HasOverride(int linkA, int linkB) const;
  // Effective enabled state for the pair (override if present, else default).
  [[nodiscard]] bool IsEnabled(int linkA, int linkB) const;

  // Make contact between the pair `enable`d. Stores an override only when it
  // differs from the default; otherwise removes any redundant existing override.
  void SetFilter(int linkA, int linkB, bool enable);
  // Clear all overrides and write the minimal set so every pair is `enable`d.
  void SetAll(bool enable);

  int numLinks = 0;
  mochi::DynamicArray<bool> implicitDisabledMask; // row-major numLinks*numLinks

 private:
  [[nodiscard]] int FindFilterIndex(int linkA, int linkB) const;

  superdex::robotics::BotPrefab& _prefab;
};

// Reusable collision probe for a fixed bot. Building the scene, articulated actor,
// and per-link contact queries is expensive and pose-independent, so the constructor
// does it once and DetectCollidingLinks only re-poses + steps. Reuse one probe across
// many poses (e.g. the estimator) to amortize that setup. Owns its scene.
struct BotContactProbe {
  mochi::Context* context = nullptr;
  mochi::Scene* scene = nullptr;
  mochi::Actor* actor = nullptr;
  superdex::robotics::BotPrefab prefab; // copy used to convert bot pose -> actor pose per query
  mochi::DynamicArray<mochi::ActorHandle> linkActors; // nested link actors, by link index
  mochi::DynamicArray<mochi::QueryHandle> queries; // per-link ContactPoints query (may be invalid)
  std::unordered_map<mochi::Handle::ValueType, int> handleToIndex; // link actor handle -> index

  // Creates a dedicated scene on `context` and adds the bot once (its contactOverrides
  // are applied). Sets `error` and leaves IsValid() false on failure.
  BotContactProbe(
      superdex::robotics::BotPrefab const& botPrefab,
      mochi::Context* context,
      superdex::robotics::IBotLoader const& loader,
      mochi::Error& error);
  ~BotContactProbe();
  // True if construction succeeded and the probe can be queried.
  bool IsValid() const;
  // Re-pose the bot to `pose`, step once, and return the deduplicated colliding link
  // pairs (linkA < linkB). Cheap relative to construction.
  mochi::DynamicArray<BotLinkPair> DetectCollidingLinks(
      mochi::Span<mochi::real const> pose,
      mochi::Error& error);
};

// Background worker that samples bot poses to find link pairs that never collide.
// Runs a @ref BotContactProbe over many poses on its own thread; the UI polls
// IsRunning / IsFinished / GetProgress while it runs and applies TakeResult() when
// it finishes. Owns the worker thread.
struct BotContactEstimator {
  BotContactEstimator() = default;
  // Joins the worker via RAII so the owned thread is never left joinable.
  ~BotContactEstimator() {
    Stop();
  }
  // Owns a worker thread; non-copyable and non-movable.
  BotContactEstimator(BotContactEstimator const&) = delete;
  BotContactEstimator& operator=(BotContactEstimator const&) = delete;
  BotContactEstimator(BotContactEstimator&&) = delete;
  BotContactEstimator& operator=(BotContactEstimator&&) = delete;

  // Spawns the worker; joins any previous run first.
  void Start(
      mochi::Context* context,
      superdex::robotics::IBotLoader const& loader,
      superdex::robotics::BotPrefab const& prefab,
      int iterations);
  // Cancels and joins the worker if it is running.
  void Stop();

  [[nodiscard]] bool IsRunning() const {
    return _running.load();
  }
  [[nodiscard]] bool IsFinished() const {
    return _finished.load();
  }
  // Request the worker to stop at the next sample (does not join).
  void RequestCancel() {
    _cancel.store(true);
  }

  struct Progress {
    int completed = 0;
    int total = 0;
  };
  [[nodiscard]] Progress GetProgress() const {
    return {_completed.load(), _total.load()};
  }

  // Snapshot of the worker's most recently sampled pose (empty if none yet).
  [[nodiscard]] mochi::DynamicArray<mochi::real> GetLatestPose() const;

  struct Result {
    bool cancelled = false;
    std::set<std::pair<int, int>> survivors; // pairs that never collided
    mochi::DynamicArray<mochi::real> originalPose; // pose to restore after the run
  };
  // Extract a finished run's result and join the worker. Call after IsFinished().
  [[nodiscard]] Result TakeResult();

 private:
  std::thread _thread;
  mutable std::mutex _resultMutex;
  std::atomic<bool> _running{false};
  std::atomic<bool> _cancel{false};
  std::atomic<bool> _finished{false};
  std::atomic<int> _completed{0};
  std::atomic<int> _total{0};
  // Guarded by _resultMutex:
  mochi::DynamicArray<mochi::real> _lastPose;
  std::set<std::pair<int, int>> _candidates;
  mochi::DynamicArray<mochi::real> _originalPose;
};

} // namespace superdex::studio
