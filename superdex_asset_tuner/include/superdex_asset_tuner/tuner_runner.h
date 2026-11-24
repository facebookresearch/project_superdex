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

#include "superdex_asset_tuner/run_writer.h"
#include "superdex_asset_tuner/tuner_session.h"
#include "superdex_asset_tuner/tuner_spec.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace superdex::robotics {
class BotScene;
class RoboticsContext;
} // namespace superdex::robotics

namespace superdex::tuner {

struct RunOptions {
  // Run directories are created under this folder. Empty: no files are written.
  std::filesystem::path outDir;
  std::string tag;
  // Called on every post-step sample, after it is written.
  std::function<void(Span<DynamicString const> channelNames, Sample const& sample)> onSample;
};

// Name of the ground plane actor the runner adds when RunSpec::groundPlane is set.
inline constexpr std::string_view kGroundPlaneName = "TunerGroundPlane";

// Sets default gravity. Call before adding the prefab so prefab scene settings win; then call
// AddGroundPlane, ApplyRunSpecOverrides and ApplySolverOverrides.
void PrepareScene(mochi::Scene& scene);

// Lowest point (world Z) of `actors`' geometry, including articulated actors' links, ignoring
// unbounded geometry such as planes; 0 if none is bounded. This is the automatic ground height.
[[nodiscard]] real ComputeAutoGroundHeight(Span<mochi::Actor* const> actors);

// Adds the RunSpec ground plane (if enabled) at RunSpec::groundHeight, or at the automatic height
// of `prefabActors` when unset. Returns the height used, or nullopt when no plane was added.
std::optional<real>
AddGroundPlane(mochi::Scene& scene, RunSpec const& run, Span<mochi::Actor* const> prefabActors);

// Applies RunSpec values that override the prefab (currently gravity).
void ApplyRunSpecOverrides(mochi::Scene& scene, RunSpec const& run);

// Applies `overrides` (TunerSpec::solverOverrides) to the scene's solver parameters. False if one
// does not fit SolverParams.
bool ApplySolverOverrides(
    mochi::Scene& scene,
    Span<PrefabOverride const> overrides,
    std::string& outError);

// A scene tuner's robots and props: the bot scene's bots and controllers, which last as long as
// this does.
class TunerBots {
 public:
  TunerBots() = default;
  ~TunerBots();
  TunerBots(TunerBots const&) = delete;
  TunerBots& operator=(TunerBots const&) = delete;
  TunerBots(TunerBots&&) = delete;
  TunerBots& operator=(TunerBots&&) = delete;

  // Builds the spec's bot scene into `scene`; nothing without one. With a task, spawns its reset
  // state and starts deciding its criteria. The bots belong to `robotics`, which must outlive
  // them, or to a robotics context of their own without one.
  bool Load(
      TunerSpec const& spec,
      std::filesystem::path const& tunerFilePath,
      mochi::Context* context,
      mochi::Scene& scene,
      std::string& outError,
      robotics::RoboticsContext* robotics = nullptr);

  // Destroys the bots and their controllers; call while their scene is alive.
  void Reset();

  // Null without a bot scene.
  [[nodiscard]] robotics::BotScene* Get() {
    return _bots.get();
  }

  // The task's time limit [s]; 0 without one.
  [[nodiscard]] double GetTaskTimeLimit() const {
    return _taskTimeLimit;
  }

 private:
  // The robotics context this owns, if any.
  robotics::RoboticsContext* _ownedRobotics = nullptr;
  std::unique_ptr<robotics::BotScene> _bots;
  // The task's time limit [s]; 0: none.
  double _taskTimeLimit = 0.0;
};

// Sets up `scene` as a tuner run sees it: default gravity, the bot scene (with the task's objects)
// or the prefab, the ground plane, then the run and solver overrides. Returns the ground height
// used, if any, in `outGroundHeight`, and the prefab's or the task's actors in `outPrefabActors`.
bool BuildTunerScene(
    TunerSpec const& spec,
    std::filesystem::path const& tunerFilePath,
    mochi::Context* context,
    mochi::Scene& scene,
    TunerBots& outBots,
    std::optional<real>& outGroundHeight,
    DynamicArray<mochi::Actor*>& outPrefabActors,
    std::string& outError);

// A run's length: `spec`'s duration, or its task's time limit (TunerBots::GetTaskTimeLimit) if
// sooner.
[[nodiscard]] double GetRunDuration(TunerSpec const& spec, double taskTimeLimit);

// Sets `summary`'s status and message from the run's task, for its `outcome` at the run's last
// sample, at simulation time `time`: "succeeded", or "failed" (also when it had not succeeded by
// the run's end, or when a stop condition, `stopMessage`, ended the run first). A run without a
// task that decides it (TaskOutcome::None) keeps its status.
void JudgeTask(TaskOutcome outcome, double time, std::string_view stopMessage, RunSummary& summary);

// Loads the tuner's scene into a fresh headless scene and steps it for `spec.run.duration`.
// Returns false only for setup errors; a run that diverges or loses a controller returns true with
// that status.
bool RunTuner(
    TunerSpec const& spec,
    std::filesystem::path const& tunerFilePath,
    RunOptions const& options,
    RunSummary& outSummary,
    std::string& outError);

} // namespace superdex::tuner
