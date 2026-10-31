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

#include <superdex_recorder/episode_restorer.h>

#include <superdex_recorder/formatted_error.h>

#include <mochi_physics/mochi_physics_experimental.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <numeric>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace superdex::recorder {
namespace {

// A static actor must be this close to its recorded pose, in quaternion and translation [m].
constexpr double kStaticPoseTolerance = 1e-5;

mochi::Real3 GetXyz(std::vector<mochi::real> const& values, size_t frameIndex) {
  size_t const offset = 3 * frameIndex;
  return {values[offset], values[offset + 1], values[offset + 2]};
}

mochi::Quaternion GetXyzw(std::vector<mochi::real> const& values, size_t frameIndex) {
  size_t const offset = 4 * frameIndex;
  return mochi::Quaternion{
      values[offset], values[offset + 1], values[offset + 2], values[offset + 3]};
}

mochi::Span<mochi::real const>
GetFrameValues(std::vector<mochi::real> const& values, size_t frameIndex, size_t width) {
  return {values.data() + frameIndex * width, width};
}

mochi::TransformRT GetWorldFromCom(RestoredActor const& actor, size_t frameIndex) {
  return mochi::TransformRT{
      GetXyzw(actor.rootRotations, frameIndex), GetXyz(actor.comTranslations, frameIndex)};
}

mochi::TransformRT GetWorldFromRoot(RestoredActor const& actor, size_t frameIndex) {
  return mochi::TransformRT{
      GetXyzw(actor.rootRotations, frameIndex), GetXyz(actor.rootTranslations, frameIndex)};
}

bool HasFrames(std::vector<mochi::real> const& values, size_t width, size_t numFrames) {
  return values.size() == width * numFrames;
}

/* Reads a static component's single value on each entity that has it. */
template <typename Value>
std::map<std::string, Value>
ReadStaticValues(EpisodeReader const& recording, std::string_view component, mochi::Error& error) {
  std::map<std::string, Value> values;
  for (auto const& stored : recording.GetComponents()) {
    if (!stored.isStatic || stored.component != component) {
      continue;
    }
    std::vector<Value> read;
    if constexpr (std::is_same_v<Value, std::string>) {
      read = recording.ReadStaticTexts(stored.entityPath, component, error);
    } else {
      read = recording.ReadStaticScalars(stored.entityPath, component, error);
    }
    MOCHI_ERROR_RETURN(error, {});
    if (read.size() != 1) {
      SUPERDEX_ERROR_SET(
          error,
          "Expected one value of '%.*s' on '%s', but found %zu.",
          static_cast<int>(component.size()),
          component.data(),
          stored.entityPath.c_str(),
          read.size());
      return {};
    }
    values.emplace(stored.entityPath, std::move(read.front()));
  }
  return values;
}

template <typename A, typename B>
bool HaveSameKeys(std::map<std::string, A> const& a, std::map<std::string, B> const& b) {
  return std::ranges::equal(a, b, [](auto const& x, auto const& y) { return x.first == y.first; });
}

/* Calls visit(name, actor) for each scene actor that is not nested in another. */
template <typename Visit>
void ForEachTopLevelActor(mochi::Scene& scene, Visit visit) {
  scene.ForEachActor([&](mochi::Actor* actor) {
    if (actor != nullptr && !actor->IsNestedLinkActor() && !actor->IsNestedSoftActor()) {
      char const* const name = actor->GetName();
      visit(std::string{name == nullptr ? "" : name}, *actor);
    }
  });
}

void ValidateUnrecordedActors(
    mochi::Scene& scene,
    std::vector<RecordedActor> const& recorded,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  std::string unrecorded;
  ForEachTopLevelActor(scene, [&](std::string const& name, mochi::Actor const& actor) {
    bool const isRecorded = std::ranges::any_of(
        recorded, [&](RecordedActor const& r) { return r.handle == actor.GetHandle(); });
    if (!actor.IsStatic() && !isRecorded) {
      unrecorded += (unrecorded.empty() ? "'" : ", '") + name + "'";
    }
  });
  if (!unrecorded.empty()) {
    SUPERDEX_ERROR_SET(error, "Scene has unrecorded dynamic actors %s.", unrecorded.c_str());
  }
}

/* Returns the index of the last row at or before each frame, or -1 before the first row. */
std::vector<ptrdiff_t> FindLatestRows(
    std::vector<int64_t> const& rowFrames,
    std::vector<int64_t> const& frames) {
  std::vector<ptrdiff_t> latest;
  latest.reserve(frames.size());
  for (int64_t const frame : frames) {
    latest.push_back(std::ranges::upper_bound(rowFrames, frame) - rowFrames.begin() - 1);
  }
  return latest;
}

void AppendRow(ScalarRows const& rows, ptrdiff_t row, std::vector<mochi::real>& output) {
  auto const values = rows.GetRow(static_cast<size_t>(row));
  std::ranges::transform(values, std::back_inserter(output), [](double value) {
    return static_cast<mochi::real>(value);
  });
}

/* Reads an actor's state at each frame, from the last row recorded at that frame. */
std::vector<mochi::real> ReadStates(
    EpisodeReader const& recording,
    RecordedActor const& actor,
    std::string_view state,
    std::vector<int64_t> const& frames,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  std::string const component = "superdex.physics." + std::string{state};
  ScalarRows const rows = recording.HasComponent(actor.path, component, false)
      ? recording.ReadScalars(actor.path, component, error)
      : ScalarRows{};
  std::vector<ptrdiff_t> const latest = FindLatestRows(rows.frames, frames);
  std::vector<mochi::real> values;
  for (size_t i = 0; i < frames.size() && error.IsOK(); ++i) {
    if (latest[i] < 0 || rows.frames[latest[i]] != frames[i]) {
      SUPERDEX_ERROR_SET(
          error,
          "Actor '%s' has no '%.*s' at frame %lld.",
          actor.runtimeName.c_str(),
          static_cast<int>(state.size()),
          state.data(),
          static_cast<long long>(frames[i]));
    } else {
      AppendRow(rows, latest[i], values);
    }
  }
  return values;
}

/* Reads each recorded robot's joint efforts at each frame, by path. Mochi applies an external
 * force until it is replaced, so an effort holds until the next one; before the first, there is
 * none. */
std::map<std::string, std::vector<mochi::real>> ReadEfforts(
    EpisodeReader const& recording,
    std::vector<RecordedActor> const& recorded,
    std::vector<int64_t> const& frames,
    mochi::Error& error) {
  auto const robotIds =
      ReadStaticValues<std::string>(recording, "superdex.robotics.robot_id", error);
  std::map<std::string, std::vector<mochi::real>> efforts;
  for (auto const& stored : recording.GetComponents()) {
    if (stored.isStatic || stored.component != "superdex.robotics.applied_joint_effort") {
      continue;
    }
    MOCHI_ERROR_RETURN(error, {});
    // A robot records its efforts on a child of its own entity.
    auto const robotId = robotIds.find(stored.entityPath.substr(0, stored.entityPath.rfind('/')));
    auto const isRobot = [&](RecordedActor const& actor) {
      return robotId != robotIds.end() && actor.isArticulated &&
          actor.runtimeName == robotId->second;
    };
    if (std::ranges::count_if(recorded, isRobot) != 1) {
      SUPERDEX_ERROR_SET(error, "'%s' has no unique recorded robot.", stored.entityPath.c_str());
      return {};
    }
    ScalarRows const rows = recording.ReadScalars(stored.entityPath, stored.component, error);
    size_t const width = rows.GetNumRows() == 0 ? 0 : rows.values.size() / rows.GetNumRows();
    for (size_t row = 0; row < rows.GetNumRows() && error.IsOK(); ++row) {
      if (!std::ranges::binary_search(frames, rows.frames[row])) {
        SUPERDEX_ERROR_SET(
            error,
            "'%s' has an effort at unrecorded frame %lld.",
            stored.entityPath.c_str(),
            static_cast<long long>(rows.frames[row]));
      } else if (rows.GetRow(row).size() != width) {
        SUPERDEX_ERROR_SET(
            error,
            "'%s' changes size at frame %lld.",
            stored.entityPath.c_str(),
            static_cast<long long>(rows.frames[row]));
      }
    }
    MOCHI_ERROR_RETURN(error, {});
    std::vector<mochi::real>& held = efforts[std::ranges::find_if(recorded, isRobot)->path];
    for (ptrdiff_t const row : FindLatestRows(rows.frames, frames)) {
      if (row < 0) {
        held.insert(held.end(), width, mochi::real{});
      } else {
        AppendRow(rows, row, held);
      }
    }
  }
  return efforts;
}

void ValidateStaticPose(
    EpisodeReader const& recording,
    RecordedActor const& actor,
    mochi::Actor const& sceneActor,
    int64_t firstFrame,
    mochi::Error& error) {
  auto const rotation = ReadStates(recording, actor, "root_quaternion_xyzw", {firstFrame}, error);
  auto const translation =
      ReadStates(recording, actor, "root_translation_xyz", {firstFrame}, error);
  MOCHI_ERROR_RETURN(error);
  if (rotation.size() != 4 || translation.size() != 3) {
    SUPERDEX_ERROR_SET(
        error, "Static actor '%s' has a malformed recorded pose.", actor.runtimeName.c_str());
    return;
  }
  auto const pose = sceneActor.GetRootTransform();
  auto const square = [](double value) { return value * value; };
  double rotationError = 0.0;
  double flippedRotationError = 0.0;
  double translationError = 0.0;
  for (int i = 0; i < 4; ++i) {
    double const sceneValue = pose.GetRotation().data[i];
    double const recordedValue = rotation[static_cast<size_t>(i)];
    rotationError += square(sceneValue - recordedValue);
    flippedRotationError += square(sceneValue + recordedValue);
  }
  for (size_t i = 0; i < 3; ++i) {
    translationError += square(double{pose.GetTranslation()[i]} - double{translation[i]});
  }
  // q and -q are the same rotation.
  double const poseError =
      std::sqrt(std::max(std::min(rotationError, flippedRotationError), translationError));
  if (poseError > kStaticPoseTolerance) {
    SUPERDEX_ERROR_SET(
        error, "Static actor '%s' differs from its recorded pose.", actor.runtimeName.c_str());
  }
}

RestoredActor ReadRestoredActor(
    EpisodeReader const& recording,
    RecordedActor const& actor,
    std::vector<int64_t> const& frames,
    std::map<std::string, std::vector<mochi::real>>& efforts,
    mochi::Error& error) {
  auto const read = [&](std::string_view state) {
    return ReadStates(recording, actor, state, frames, error);
  };
  RestoredActor restored;
  restored.handle = actor.handle;
  restored.isArticulated = actor.isArticulated;
  restored.rootRotations = read("root_quaternion_xyzw");
  if (!actor.isArticulated) {
    restored.comTranslations = read("com_translation_xyz");
    restored.linearVelocities = read("linear_velocity_xyz");
    restored.angularVelocities = read("angular_velocity_xyz");
    return restored;
  }
  restored.rootTranslations = read("root_translation_xyz");
  restored.jointPositions = read("joint_positions");
  restored.jointVelocities = read("joint_velocities");
  // Targets are recorded at every frame or at none; ReadStates rejects gaps.
  if (recording.HasComponent(actor.path, "superdex.physics.controller_target_positions", false)) {
    restored.controllerTargets = read("controller_target_positions");
  }
  restored.jointEfforts = std::move(efforts[actor.path]);
  return restored;
}

bool HasStatePerFrame(
    RestoredActor const& actor,
    mochi::Actor const& sceneActor,
    size_t numFrames) {
  if (!actor.isArticulated) {
    return HasFrames(actor.rootRotations, 4, numFrames) &&
        HasFrames(actor.comTranslations, 3, numFrames) &&
        HasFrames(actor.linearVelocities, 3, numFrames) &&
        HasFrames(actor.angularVelocities, 3, numFrames);
  }
  auto const numDofs = static_cast<size_t>(sceneActor.GetNumDofs());
  return HasFrames(actor.rootRotations, 4, numFrames) &&
      HasFrames(actor.rootTranslations, 3, numFrames) &&
      HasFrames(actor.jointPositions, numDofs, numFrames) &&
      HasFrames(actor.jointVelocities, numDofs, numFrames) &&
      (actor.controllerTargets.empty() || HasFrames(actor.controllerTargets, numDofs, numFrames)) &&
      (actor.jointEfforts.empty() || HasFrames(actor.jointEfforts, numDofs, numFrames));
}

} // namespace

std::vector<RecordedActor>
BindRecordedActors(mochi::Scene& scene, EpisodeReader const& recording, mochi::Error& error) {
  auto const runtimeNames =
      ReadStaticValues<std::string>(recording, "superdex.physics.runtime_name", error);
  auto const spawnNames =
      ReadStaticValues<std::string>(recording, "superdex.physics.spawn_name", error);
  auto const actorTypes =
      ReadStaticValues<std::string>(recording, "superdex.physics.actor_type", error);
  auto const staticFlags = ReadStaticValues<double>(recording, "superdex.physics.is_static", error);
  MOCHI_ERROR_IF_NOT(
      HaveSameKeys(runtimeNames, actorTypes) && HaveSameKeys(runtimeNames, staticFlags),
      error,
      "Recorded actor metadata is incomplete.");
  std::map<std::string, mochi::Actor*, std::less<>> sceneActors;
  ForEachTopLevelActor(scene, [&](std::string const& name, mochi::Actor& actor) {
    if (!sceneActors.emplace(name, &actor).second) {
      SUPERDEX_ERROR_SET(error, "Restoring requires unique actor names: '%s'.", name.c_str());
    }
  });
  MOCHI_ERROR_RETURN(error, {});

  std::vector<RecordedActor> recorded;
  std::map<mochi::Actor const*, std::string> boundPaths;
  for (auto const& [path, runtimeName] : runtimeNames) {
    // Loading a task names a spawn's bodies "<spawn name>/<name>", and a spawned bot after its
    // spawn.
    std::vector<std::string> names{runtimeName};
    if (auto const spawnName = spawnNames.find(path); spawnName != spawnNames.end()) {
      names = {spawnName->second + "/" + runtimeName, spawnName->second, runtimeName};
    }
    auto const name = std::ranges::find_if(
        names, [&](std::string const& candidate) { return sceneActors.contains(candidate); });
    if (name == names.end()) {
      SUPERDEX_ERROR_SET(error, "Scene is missing recorded actor '%s'.", names.front().c_str());
      return {};
    }
    mochi::Actor const& actor = *sceneActors.find(*name)->second;
    if (auto const [other, inserted] = boundPaths.emplace(&actor, path); !inserted) {
      SUPERDEX_ERROR_SET(
          error,
          "Recorded actors '%s' and '%s' both bind to scene actor '%s'.",
          other->second.c_str(),
          path.c_str(),
          name->c_str());
      return {};
    }
    std::string const& type = actorTypes.at(path);
    if ((type != "rigid" || actor.GetType() != mochi::ActorType::Rigid) &&
        (type != "articulated" || actor.GetType() != mochi::ActorType::Articulated)) {
      SUPERDEX_ERROR_SET(
          error,
          "Actor '%s' was recorded as %s, but the scene actor is not.",
          runtimeName.c_str(),
          type.c_str());
      return {};
    }
    recorded.push_back({
        .path = path,
        .runtimeName = runtimeName,
        .handle = actor.GetHandle(),
        .isArticulated = type == "articulated",
        .isStatic = staticFlags.at(path) != 0.0,
    });
  }
  return recorded;
}

std::unique_ptr<EpisodeRestorer>
EpisodeRestorer::Create(mochi::Scene& scene, EpisodeReader const& recording, mochi::Error& error) {
  MOCHI_ERROR_RETURN(error, nullptr);
  std::vector<RecordedActor> recorded = BindRecordedActors(scene, recording, error);
  ValidateUnrecordedActors(scene, recorded, error);
  ScalarRows const frames =
      recording.ReadScalars("data/physics/frames", "superdex.physics.step_dt_sec", error);
  MOCHI_ERROR_IF(
      error.IsOK() && (frames.GetNumRows() == 0 || frames.values.size() != frames.GetNumRows()),
      error,
      "The recording has no valid physics frames.");
  auto efforts = ReadEfforts(recording, recorded, frames.frames, error);
  MOCHI_ERROR_RETURN(error, nullptr);

  std::unique_ptr<EpisodeRestorer> restorer{new EpisodeRestorer{scene}};
  restorer->_frames = frames.frames;
  restorer->_stepDtsSec = frames.values;
  for (int64_t const timeNs : frames.simulationTimesNs) {
    restorer->_simulationTimesSec.push_back(static_cast<double>(timeNs) / 1e9);
  }
  for (auto const& actor : recorded) {
    mochi::Actor const* const sceneActor = scene.GetActor(actor.handle);
    if (sceneActor == nullptr) {
      MOCHI_ERROR_SET(error, "A recorded actor was destroyed.");
      return nullptr;
    }
    if (actor.isStatic) {
      ValidateStaticPose(recording, actor, *sceneActor, frames.frames.front(), error);
      continue;
    }
    restorer->_actors.push_back(ReadRestoredActor(recording, actor, frames.frames, efforts, error));
    if (error.IsOK() &&
        !HasStatePerFrame(restorer->_actors.back(), *sceneActor, frames.GetNumRows())) {
      SUPERDEX_ERROR_SET(
          error,
          "Recorded states of '%s' do not match the degrees of freedom of scene actor '%s'.",
          actor.path.c_str(),
          sceneActor->GetName());
    }
  }
  MOCHI_ERROR_RETURN(error, nullptr);
  restorer->_recordedActors = std::move(recorded);
  return restorer;
}

EpisodeRestorer::EpisodeRestorer(mochi::Scene& scene) : _scene{scene} {}

EpisodeRestorer::~EpisodeRestorer() = default;

std::vector<int64_t> const& EpisodeRestorer::GetFrames() const {
  return _frames;
}

std::vector<RecordedActor> const& EpisodeRestorer::GetActors() const {
  return _recordedActors;
}

void EpisodeRestorer::RestoreFrame(size_t frameIndex, mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(frameIndex >= _frames.size(), error, "Frame index is out of range.");
  MOCHI_ERROR_RETURN(error);
  mochi::experimental::SetSimulationTime(
      &_scene, _simulationTimesSec[frameIndex], _stepDtsSec[frameIndex], error);
  for (auto const& actor : _actors) {
    mochi::Actor* const sceneActor = _scene.GetActor(actor.handle);
    MOCHI_ERROR_IF(sceneActor == nullptr, error, "A restored actor was destroyed.");
    MOCHI_ERROR_RETURN(error);
    // External forces persist until replaced, so clear any left from before the restore.
    sceneActor->ClearExternalForces();
    if (!actor.isArticulated) {
      sceneActor->SetCenterOfMassTransform(GetWorldFromCom(actor, frameIndex), error);
      sceneActor->SetVelocity(
          GetXyz(actor.linearVelocities, frameIndex),
          GetXyz(actor.angularVelocities, frameIndex),
          error);
      continue;
    }

    auto const numDofs = static_cast<size_t>(sceneActor->GetNumDofs());
    sceneActor->SetRootTransform(GetWorldFromRoot(actor, frameIndex), error);
    sceneActor->SetArticulatedPoseFromJoints(
        GetFrameValues(actor.jointPositions, frameIndex, numDofs), error);
    sceneActor->SetArticulatedJointVelocities(
        GetFrameValues(actor.jointVelocities, frameIndex, numDofs), error);
    // The pose setter resets the target, so set it afterwards.
    if (!actor.controllerTargets.empty()) {
      sceneActor->SetArticulatedTargetPose(
          GetFrameValues(actor.controllerTargets, frameIndex, numDofs), error);
    }
    if (!actor.jointEfforts.empty()) {
      std::vector<int> dofIndices(numDofs);
      std::iota(dofIndices.begin(), dofIndices.end(), 0);
      sceneActor->SetExternalForcesOnDofs(
          dofIndices, GetFrameValues(actor.jointEfforts, frameIndex, numDofs), error);
    }
  }
  MOCHI_ERROR_RETURN(error);
  _scene.Step(0.0);
}

} // namespace superdex::recorder
