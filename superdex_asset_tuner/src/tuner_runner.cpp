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

#include "superdex_asset_tuner/tuner_runner.h"

#include "superdex_asset_tuner/json_path.h"
#include "superdex_asset_tuner/sim_recording.h"

#include "scene_utils.h"

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/defer.h>
#include <mochi_physics/utils/mochi_prefab.h>
#include <superdex_robotics/core/context.h>
#include <superdex_robotics/scene/bot_scene.h>
#include <superdex_robotics/superdex_robotics.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>

namespace superdex::tuner {

namespace {

// Adds why sim.rrd was not written to the summary's message, keeping what it already says.
void NoteRecordingFailure(RunSummary& summary, std::string_view reason) {
  summary.simRecording = false;
  std::string message(std::string_view(summary.message));
  message += message.empty() ? "sim.rrd not written: " : " sim.rrd not written: ";
  message += reason;
  summary.message = DynamicString(message);
}

bool AllFinite(Span<real const> values) {
  for (real const v : values) {
    if (!std::isfinite(v)) {
      return false;
    }
  }
  return true;
}

bool ActorsFinite(Span<mochi::Actor* const> actors) {
  for (mochi::Actor const* actor : actors) {
    Real3 const t = actor->GetRootTransform().GetTranslation();
    if (!std::isfinite(t[0]) || !std::isfinite(t[1]) || !std::isfinite(t[2])) {
      return false;
    }
  }
  return true;
}

} // namespace

void PrepareScene(mochi::Scene& scene) {
  scene.SetGravity(kDefaultGravity);
}

real ComputeAutoGroundHeight(Span<mochi::Actor* const> actors) {
  // Planes (and anything else unbounded) do not rest on the ground, so they must not drag it down.
  std::optional<real> lowest;
  for (mochi::Actor const* actor : actors) {
    if (std::optional<mochi::Aabb> const bounds = details::WorldBounds(*actor)) {
      real const minZ = bounds->GetMin()[2];
      lowest = lowest ? std::min(*lowest, minZ) : minZ;
    }
  }
  return lowest.value_or(real{0});
}

std::optional<real>
AddGroundPlane(mochi::Scene& scene, RunSpec const& run, Span<mochi::Actor* const> prefabActors) {
  if (!run.groundPlane) {
    return std::nullopt;
  }
  real const height = run.groundHeight.value_or(ComputeAutoGroundHeight(prefabActors));
  mochi::ErrorLog error;
  mochi::RigidActorParams ground;
  ground.isStatic = true;
  ground.name = DynamicString(kGroundPlaneName);
  ground.contact = run.groundContact.value_or(mochi::ContactParams{});
  ground.shape = scene.GetContext()->CreatePlaneShape({0.0f, 0.0f, 1.0f}, height, error);
  scene.CreateRigidActor(ground, error);
  return height;
}

void ApplyRunSpecOverrides(mochi::Scene& scene, RunSpec const& run) {
  if (run.gravity) {
    scene.SetGravity(*run.gravity);
  }
}

bool ApplySolverOverrides(
    mochi::Scene& scene,
    Span<PrefabOverride const> overrides,
    std::string& outError) {
  if (overrides.empty()) {
    return true;
  }
  picojson::value params;
  if (!ParseJson(SReflect::ToJsonString(scene.GetSolverParams()), params, outError)) {
    return false;
  }
  for (PrefabOverride const& entry : overrides) {
    if (!SetJsonPathFromText(params, entry.path, entry.value, outError)) {
      outError = "Solver override: " + outError;
      return false;
    }
  }
  mochi::SolverParams solver;
  if (!SReflect::FromJsonString(
          solver, params.serialize(), SReflect::DeserializeFlags::WarnIfExtraneousFields)) {
    outError = "solverOverrides do not match SolverParams (unknown field or wrong type; see log).";
    return false;
  }
  mochi::Error error;
  scene.SetSolverParams(solver, error);
  if (!error.IsOK()) {
    outError = std::string("Solver override: ") + error.GetDescription();
    return false;
  }
  return true;
}

TunerBots::~TunerBots() {
  // The bots and controllers go first: they belong to the robotics context.
  Reset();
  if (_ownedRobotics != nullptr) {
    robotics::DestroyRoboticsContext(_ownedRobotics);
  }
}

void TunerBots::Reset() {
  _bots.reset();
}

bool TunerBots::Load(
    TunerSpec const& spec,
    std::filesystem::path const& tunerFilePath,
    mochi::Context* context,
    mochi::Scene& scene,
    std::string& outError,
    robotics::RoboticsContext* robotics) {
  Reset();
  std::filesystem::path const path = ResolveScenePath(spec, tunerFilePath);
  if (path.empty()) {
    return true;
  }
  if (robotics == nullptr) {
    if (_ownedRobotics == nullptr) {
      _ownedRobotics = robotics::CreateRoboticsContext();
    }
    robotics = _ownedRobotics;
  }
  DynamicArray<std::string_view> names;
  for (DynamicString const& bot : spec.bots) {
    names.push_back(std::string_view(bot));
  }
  mochi::Error error;
  _bots = std::make_unique<robotics::BotScene>(
      names.empty()
          ? robotics::LoadBotScene(&scene, path.string(), context, robotics, error)
          : robotics::LoadBotScene(
                &scene, path.string(), mochi::MakeConstSpan(names), context, robotics, error));
  if (!error.IsOK()) {
    _bots.reset();
    outError = "Failed to load bot scene '" + path.string() + "': " + error.GetDescription();
    return false;
  }
  return true;
}

bool BuildTunerScene(
    TunerSpec const& spec,
    std::filesystem::path const& tunerFilePath,
    mochi::Context* context,
    mochi::Scene& scene,
    TunerBots& outBots,
    std::optional<real>& outGroundHeight,
    DynamicArray<mochi::Actor*>& outPrefabActors,
    std::string& outError) {
  if (!CheckPrefabOrScene(spec, outError)) {
    return false;
  }
  PrepareScene(scene);
  if (!outBots.Load(spec, tunerFilePath, context, scene, outError)) {
    return false;
  }
  outPrefabActors.clear();
  if (!spec.prefab.empty()) {
    mochi::prefab::ScenePrefab prefab;
    if (!LoadTunerPrefab(spec, tunerFilePath, context, prefab, outError)) {
      return false;
    }
    mochi::Error error;
    mochi::prefab::AddToSceneResult const added = mochi::prefab::AddToScene(prefab, &scene, error);
    if (!error.IsOK()) {
      outError = std::string("Failed to add prefab to scene: ") + error.GetDescription();
      return false;
    }
    outPrefabActors.assign(added.actors.begin(), added.actors.end());
  }
  outGroundHeight = AddGroundPlane(scene, spec.run, mochi::MakeConstSpan(outPrefabActors));
  ApplyRunSpecOverrides(scene, spec.run);
  return ApplySolverOverrides(scene, mochi::MakeConstSpan(spec.solverOverrides), outError);
}

bool RunTuner(
    TunerSpec const& spec,
    std::filesystem::path const& tunerFilePath,
    RunOptions const& options,
    RunSummary& outSummary,
    std::string& outError) {
  if (!(spec.run.dt > 0.0) || !(spec.run.duration >= 0.0)) {
    outError = "run.dt must be > 0 and run.duration >= 0.";
    return false;
  }
  auto const wallStart = std::chrono::steady_clock::now();
  outSummary = {};

  mochi::Context* context = mochi::CreateContext(spec.run.singleThreaded ? 0 : -1);
  MOCHI_DEFER(mochi::DestroyContext(context));
  mochi::Scene* scene = context->CreateScene("superdex_asset_tuner");
  MOCHI_DEFER(context->DestroyScene(scene));

  TunerBots bots;
  std::optional<real> groundHeight;
  DynamicArray<mochi::Actor*> prefabActors;
  if (!BuildTunerScene(
          spec, tunerFilePath, context, *scene, bots, groundHeight, prefabActors, outError)) {
    return false;
  }
  // The resolved spec records the ground height actually used, so rerunning it reproduces this run
  // even if the automatic height would change (e.g. after the prefab is edited).
  TunerSpec resolved = spec;
  if (groundHeight) {
    resolved.run.groundHeight = *groundHeight;
  }

  TunerSession session;
  if (!session.Bind(*scene, spec, outError, bots.Get())) {
    return false;
  }
  MOCHI_DEFER(session.Unbind(*scene));
  // The actors whose state shows a divergence: the prefab's, and the bots' links (an articulated
  // actor's own transform is where it was placed).
  DynamicArray<mochi::Actor*> watched = prefabActors;
  if (robotics::BotScene* const botScene = bots.Get()) {
    mochi::ErrorLog error;
    for (robotics::BotEntry const& entry : botScene->GetBotScenePrefab().bots) {
      robotics::Bot const* const bot = botScene->GetBot(entry.name);
      mochi::Actor const* const robot = bot != nullptr ? bot->GetArticulatedActor() : nullptr;
      if (robot == nullptr) {
        continue;
      }
      for (mochi::ActorHandle const link : robot->GetNestedLinkActors(error)) {
        if (mochi::Actor* const linkActor = scene->GetActor(link)) {
          watched.push_back(linkActor);
        }
      }
    }
  }

  RunWriter writer;
  SimRecording simRecording;
  // Why sim.rrd is not written, added to the summary's message once the run has set it.
  std::string recordError;
  if (!options.outDir.empty()) {
    std::string const tunerName = tunerFilePath.stem().string();
    std::filesystem::path const prefabPath = ResolvePrefabPath(spec, tunerFilePath);
    if (!writer.Open(options.outDir, tunerName, options.tag, resolved, prefabPath, outError)) {
      return false;
    }
    writer.WriteHeader(session.GetChannelNames());
    if (spec.run.record.enabled && SimRecording::IsSupported()) {
      outSummary.simRecording =
          simRecording.Start(writer.GetSimRecordingPath(), *scene, session, resolved, recordError);
    }
  }

  Sample sample;
  bool diverged = false;
  int stopCondition = -1;
  DynamicString stopMessage;
  for (auto const& name : session.GetChannelNames()) {
    ChannelSummary channel;
    channel.name = name;
    outSummary.channels.push_back(std::move(channel));
  }
  mochi::CallbackHandle const preStep = scene->RegisterPreStepCallback(
      "superdex_asset_tuner::Loads", [&](mochi::StepInfo const& info) { session.OnPreStep(info); });
  mochi::CallbackHandle const postStep = scene->RegisterPostStepCallback(
      "superdex_asset_tuner::Probes", [&](mochi::StepInfo const& info) {
        session.OnPostStep(info, sample);
        UpdateChannelSummaries(
            sample.values, outSummary.steps == 0, mochi::MakeSpan(outSummary.channels));
        if (writer.IsOpen()) {
          writer.WriteSample(sample);
        }
        simRecording.Record(sample);
        if (options.onSample) {
          options.onSample(session.GetChannelNames(), sample);
        }
        diverged = !AllFinite(sample.values) || !ActorsFinite(watched);
        if (sample.stopCondition >= 0) {
          stopCondition = sample.stopCondition;
          stopMessage = sample.stopMessage;
        }
      });

  // Round so accumulated float error in duration/dt does not add or drop a step.
  auto const numSteps = static_cast<int>(std::llround(spec.run.duration / spec.run.dt));
  while (outSummary.steps < numSteps && !diverged && stopCondition < 0 &&
         session.GetFailure().empty()) {
    scene->Step(spec.run.dt);
    ++outSummary.steps;
  }
  scene->CancelCallback(preStep);
  scene->CancelCallback(postStep);
  simRecording.StopSampling();

  outSummary.simTime = session.GetTime();
  if (diverged) {
    outSummary.status = "diverged";
    outSummary.message = "Non-finite state at t = " + std::to_string(outSummary.simTime) + " s.";
  } else if (!session.GetFailure().empty()) {
    outSummary.status = "error";
    outSummary.message = DynamicString(session.GetFailure());
  } else if (stopCondition >= 0) {
    outSummary.status = "stopped";
    if (static_cast<size_t>(stopCondition) < spec.stopConditions.size()) {
      outSummary.stopCondition = spec.stopConditions[static_cast<size_t>(stopCondition)].name;
    }
    outSummary.message = stopMessage;
  }
  outSummary.wallTimeSec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - wallStart).count();
  if (simRecording.IsActive()) {
    // summary.json says whether sim.rrd was written, so the recording is finished first.
    outSummary.runDir = DynamicString(writer.GetRunDir().string());
    simRecording.Finish(outSummary, recordError);
  }
  if (!recordError.empty()) {
    NoteRecordingFailure(outSummary, recordError);
  }
  if (writer.IsOpen()) {
    return writer.Close(outSummary, outError);
  }
  return true;
}

} // namespace superdex::tuner
