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

#include "simulation/ik_recording_player.h"
#include <superdex_recorder/episode_reader.h>

#include <algorithm>
#include <bit>
#include <cctype>
#include <cmath>
#include <limits>
#include <map>
#include <set>

namespace superdex::studio {
namespace {
using namespace superdex::recorder;
using namespace mochi;
using superdex::robotics::BotPrefab;
constexpr std::string_view kActors = "data/physics/actors/";
constexpr std::string_view kRobots = "data/robotics/robots/";
constexpr char kPositions[] = "superdex.physics.joint_positions";
constexpr char kTranslation[] = "superdex.physics.root_translation_xyz";
constexpr char kRotation[] = "superdex.physics.root_quaternion_xyzw";
constexpr uintmax_t kFileBytes = 1024ULL * 1024 * 1024;
constexpr size_t kPoseBudget = 256 * 1024 * 1024;
constexpr size_t kMaxLayoutSize = 4096;

uint64_t Distance(int64_t first, int64_t last) {
  return static_cast<uint64_t>(last) - static_cast<uint64_t>(first);
}
int64_t AtOffset(int64_t first, uint64_t offset) {
  return std::bit_cast<int64_t>(static_cast<uint64_t>(first) + offset);
}
bool Finite(double value) {
  return std::isfinite(value) && std::abs(value) <= std::numeric_limits<real>::max();
}
bool FiniteArray(Span<double const> values, size_t width) {
  return values.size() == width && std::all_of(values.begin(), values.end(), Finite);
}
bool Integer(double value, size_t maximum) {
  return std::isfinite(value) && value >= 0 && value <= static_cast<double>(maximum) &&
      value == std::floor(value);
}
bool UnitRotation(Quaternion const& rotation) {
  return IsFinite(rotation.ToReal4()) && std::abs(Norm(rotation.ToReal4()) - real(1)) <= real(1e-4);
}
std::string NativeActorId(std::string name) {
  std::replace(name.begin(), name.end(), '/', '.');
  return name;
}

// Metadata must be static, rather than a time-varying reinterpretation of the same coordinates.
bool Text(
    EpisodeReader const& reader,
    std::string_view path,
    std::string_view field,
    std::string& value,
    std::string& diagnostic) {
  Error error;
  auto values = reader.ReadStaticTexts(path, field, error);
  if (!error.IsOK() || reader.HasComponent(path, field, false) || values.size() != 1 ||
      values[0].empty()) {
    diagnostic = "Required single static text " + std::string{field} + " on " + std::string{path};
    if (!error.IsOK())
      diagnostic += std::string{": "} + error.GetDescription();
    return false;
  }
  value = std::move(values[0]);
  return true;
}
bool Number(
    EpisodeReader const& reader,
    std::string_view path,
    std::string_view field,
    double& value,
    std::string& diagnostic) {
  Error error;
  auto values = reader.ReadStaticScalars(path, field, error);
  if (!error.IsOK() || reader.HasComponent(path, field, false) || !FiniteArray(values, 1)) {
    diagnostic =
        "Required single finite static scalar " + std::string{field} + " on " + std::string{path};
    if (!error.IsOK())
      diagnostic += std::string{": "} + error.GetDescription();
    return false;
  }
  value = values[0];
  return true;
}

struct JointKind {
  std::string_view name;
  size_t translations;
  size_t rotations;
};
JointKind Kind(ArticulatedJointType type) {
  switch (type) {
    case ArticulatedJointType::Free:
      return {"free", 3, 3};
    case ArticulatedJointType::Prismatic:
      return {"prismatic", 1, 0};
    case ArticulatedJointType::Revolute:
      return {"revolute", 0, 1};
    case ArticulatedJointType::Spherical:
      return {"spherical", 0, 3};
    case ArticulatedJointType::Hard:
      return {"hard", 0, 0};
    default:
      return {"unsupported", 0, 0};
  }
}
bool ValidateInputBot(BotPrefab const& inputBot, std::string& diagnostic) {
  if (inputBot.name.empty() || inputBot.joints.empty() || inputBot.joints.size() > kMaxLayoutSize ||
      inputBot.links.size() != inputBot.joints.size() || inputBot._numDofs <= 0 ||
      inputBot._numDofs > static_cast<int>(kMaxLayoutSize) ||
      inputBot._dofIndices.size() != static_cast<size_t>(inputBot._numDofs)) {
    diagnostic = "Input bot requires a native model name and rebuilt nonempty joint/DOF layout";
    return false;
  }
  // The explicit root placement is the root-link frame in Studio. Nonidentity root rest frames
  // would need recorded rest metadata, which PhysicsRecorder does not provide.
  if ((inputBot.joints[0].type != ArticulatedJointType::Free &&
       inputBot.joints[0].type != ArticulatedJointType::Hard) ||
      !NearEqual(inputBot.joints[0].parentLinkFromJoint, TransformRT{}, real(1e-4)) ||
      !NearEqual(inputBot.links[0].parentJointFromLink, TransformRT{}, real(1e-4))) {
    diagnostic = "Input bot root must be free or hard with identity rest frames";
    return false;
  }
  std::set<std::string> names;
  size_t cursor = 0;
  for (size_t i = 0; i < inputBot.joints.size(); ++i) {
    auto const& joint = inputBot.joints[i];
    auto const kind = Kind(joint.type);
    if (joint.name.empty() || !names.emplace(joint.name).second || kind.name == "unsupported" ||
        (i != 0 && joint.type == ArticulatedJointType::Free) || !IsFinite(joint.axis) ||
        !IsFinite(joint.parentLinkFromJoint) || !IsFinite(inputBot.links[i].parentJointFromLink) ||
        !UnitRotation(joint.parentLinkFromJoint.GetRotation()) ||
        !UnitRotation(inputBot.links[i].parentJointFromLink.GetRotation()) ||
        (i != 0 &&
         (inputBot.links[i].parentLink < 0 ||
          inputBot.links[i].parentLink >= static_cast<int>(i)))) {
      diagnostic = "Input bot joint names, types, topology, axes or rest frames are invalid";
      return false;
    }
    if (i == 0)
      continue;
    for (size_t j = 0; j < kind.translations + kind.rotations; ++j) {
      if (cursor >= inputBot._dofIndices.size() ||
          inputBot._dofIndices[cursor++] != static_cast<int>(i)) {
        diagnostic = "Input bot joint/DOF layout is stale; rebuild it before loading";
        return false;
      }
    }
  }
  if (cursor != inputBot._dofIndices.size()) {
    diagnostic = "Input bot joint/DOF layout is stale; rebuild it before loading";
    return false;
  }
  return true;
}
struct ActorLayout {
  std::string path;
  std::string identity;
  size_t dofs = 0;
  size_t rootOffset = 0;
  bool freeRoot = false;
  std::vector<size_t> coordinates;
};

bool Layout(
    EpisodeReader const& reader,
    BotPrefab const& inputBot,
    ActorLayout& actor,
    std::string& diagnostic) {
  double count = 0;
  if (!Number(reader, actor.path, "superdex.physics.num_dofs", count, diagnostic) ||
      !Integer(count, kMaxLayoutSize) || count == 0) {
    diagnostic = "Invalid native actor num_dofs: " + diagnostic;
    return false;
  }
  double isStatic = 0;
  if (!Number(reader, actor.path, "superdex.physics.is_static", isStatic, diagnostic) ||
      isStatic != 0) {
    diagnostic = "Native articulation playback requires is_static=0: " + diagnostic;
    return false;
  }
  actor.dofs = static_cast<size_t>(count);
  auto const path = actor.path + "/joint_layout";
  constexpr std::string_view fields[] = {
      "superdex.physics.joint_names",
      "superdex.physics.joint_types",
      "superdex.physics.joint_dof_offsets",
      "superdex.physics.joint_translation_dof_counts",
      "superdex.physics.joint_rotation_dof_counts"};
  for (auto field : fields) {
    if (reader.HasComponent(path, field, false)) {
      diagnostic = "Joint layout must be static: " + std::string{field};
      return false;
    }
  }
  Error error;
  auto names = reader.ReadStaticTexts(path, fields[0], error);
  auto types = reader.ReadStaticTexts(path, fields[1], error);
  auto offsets = reader.ReadStaticScalars(path, fields[2], error);
  auto translations = reader.ReadStaticScalars(path, fields[3], error);
  auto rotations = reader.ReadStaticScalars(path, fields[4], error);
  if (!error.IsOK() || names.size() != inputBot.joints.size() || types.size() != names.size() ||
      offsets.size() != names.size() || translations.size() != names.size() ||
      rotations.size() != names.size()) {
    diagnostic = "Native joint layout fields are missing, mistyped or incompatible with input bot";
    if (!error.IsOK())
      diagnostic += std::string{": "} + error.GetDescription();
    return false;
  }
  std::map<std::string, size_t, std::less<>> indices;
  std::vector<bool> used(actor.dofs, false);
  for (size_t i = 0; i < names.size(); ++i) {
    if (names[i].empty() || !indices.emplace(names[i], i).second ||
        !Integer(offsets[i], actor.dofs) || !Integer(translations[i], 3) ||
        !Integer(rotations[i], 3) || translations[i] + rotations[i] > actor.dofs - offsets[i]) {
      diagnostic = "Native joint layout has duplicate names or invalid DOF offsets/counts";
      return false;
    }
    auto const begin = static_cast<size_t>(offsets[i]);
    auto const width = static_cast<size_t>(translations[i] + rotations[i]);
    for (size_t j = begin; j < begin + width; ++j) {
      if (used[j]) {
        diagnostic = "Native joint layout DOF spans overlap";
        return false;
      }
      used[j] = true;
    }
  }
  if (std::find(used.begin(), used.end(), false) != used.end()) {
    diagnostic = "Native joint layout does not cover num_dofs exactly";
    return false;
  }
  for (size_t joint = 0; joint < inputBot.joints.size(); ++joint) {
    auto const found = indices.find(std::string_view{inputBot.joints[joint].name});
    if (found == indices.end()) {
      diagnostic =
          "Native joint names do not match input bot: " + std::string{inputBot.joints[joint].name};
      return false;
    }
    size_t const i = found->second;
    auto expected = Kind(inputBot.joints[joint].type);
    if (joint == 0) {
      // A free recorded root and a hard input root both have explicit Studio root placement.
      if (i != 0 || (types[i] != "free" && types[i] != "hard")) {
        diagnostic = "Native root joint must be the named free or hard root";
        return false;
      }
      actor.freeRoot = types[i] == "free";
      actor.rootOffset = static_cast<size_t>(offsets[i]);
      expected = Kind(actor.freeRoot ? ArticulatedJointType::Free : ArticulatedJointType::Hard);
    }
    if (types[i] != expected.name || translations[i] != expected.translations ||
        rotations[i] != expected.rotations) {
      diagnostic = "Native joint types or translation/rotation DOF counts differ from input bot: " +
          names[i];
      return false;
    }
    if (joint != 0) {
      for (size_t j = 0; j < expected.translations + expected.rotations; ++j) {
        actor.coordinates.push_back(static_cast<size_t>(offsets[i]) + j);
      }
    }
  }
  return actor.coordinates.size() == static_cast<size_t>(inputBot._numDofs);
}

bool SelectActor(
    EpisodeReader const& reader,
    BotPrefab const& inputBot,
    ActorLayout& selected,
    std::string& diagnostic) {
  std::vector<ActorLayout> matches;
  std::string failures;
  for (auto const& component : reader.GetComponents()) {
    if (!component.isStatic || component.component != "superdex.physics.actor_type" ||
        !component.entityPath.starts_with(kActors))
      continue;
    auto const& path = component.entityPath;
    std::string type, id, runtime;
    std::string reason;
    if (!Text(reader, path, component.component, type, reason) || type != "articulated")
      continue;
    if (!Text(reader, path, "superdex.physics.actor_id", id, reason) ||
        !Text(reader, path, "superdex.physics.runtime_name", runtime, reason)) {
      failures += "\n" + path + ": " + reason;
      continue;
    }
    // RoboticsRecorder writes robot_id from the ORIGINAL runtime name, not PhysicsRecorder's
    // unique/path-normalized actor id. Never select by a handedness substring or suffix.
    std::string prefab;
    bool identified = false;
    bool identityValid = true;
    for (auto const& robot : reader.GetComponents()) {
      if (!robot.isStatic || robot.component != "superdex.robotics.robot_id" ||
          !robot.entityPath.starts_with(kRobots))
        continue;
      std::string robotId;
      if (!Text(reader, robot.entityPath, robot.component, robotId, reason) || robotId != runtime)
        continue;
      if (identified ||
          !Text(reader, robot.entityPath, "superdex.robotics.robot_prefab_name", prefab, reason)) {
        identityValid = false;
        break;
      }
      identified = true;
    }
    if (!identityValid) {
      failures += "\n" + path + ": ambiguous or malformed native robotics model identity";
      continue;
    }
    auto const inputBotName = std::string_view{inputBot.name};
    std::string identity;
    if (identified) {
      if (prefab != inputBotName) {
        failures += "\n" + path + ": native robot_prefab_name=" + prefab + " does not match";
        continue; // Authoritative model identity overrides display names.
      }
      identity = "native robot_prefab_name=" + prefab;
    } else {
      std::string spawn;
      if (reader.HasComponent(path, "superdex.physics.spawn_name", true) &&
          !Text(reader, path, "superdex.physics.spawn_name", spawn, reason)) {
        failures += "\n" + path + ": " + reason;
        continue;
      }
      if (runtime != inputBotName && spawn != inputBotName &&
          id != NativeActorId(std::string{inputBotName}))
        continue;
      identity = "exact native actor identity=" + id;
    }
    ActorLayout actor{.path = path, .identity = std::move(identity)};
    if (!Layout(reader, inputBot, actor, reason)) {
      failures += "\n" + path + ": " + reason;
      continue;
    }
    matches.push_back(std::move(actor));
  }
  if (matches.size() != 1) {
    diagnostic = matches.empty() ? "No compatible native physics articulation for input bot '" +
            std::string{inputBot.name} + "'" + failures
                                 : "Ambiguous native physics articulations for input bot '" +
            std::string{inputBot.name} + "':";
    if (!matches.empty())
      for (auto const& actor : matches)
        diagnostic += "\n" + actor.path;
    return false;
  }
  selected = std::move(matches[0]);
  return true;
}

bool Rows(ScalarRows const& rows, size_t width, std::string& diagnostic) {
  auto const count = rows.GetNumRows();
  if (count == 0 || rows.simulationTimesNs.size() != count || rows.offsets.size() != count + 1 ||
      rows.offsets.front() != 0 || rows.offsets.back() != rows.values.size() ||
      !std::is_sorted(rows.offsets.begin(), rows.offsets.end())) {
    diagnostic = "Native state has empty or inconsistent rows/timelines/offsets";
    return false;
  }
  for (size_t i = 0; i < count; ++i) {
    if (rows.frames[i] < 0 || rows.simulationTimesNs[i] < 0 ||
        (i &&
         (rows.frames[i] <= rows.frames[i - 1] ||
          rows.simulationTimesNs[i] < rows.simulationTimesNs[i - 1]))) {
      diagnostic =
          "Native snapshot frames must be unique/nonnegative and sim_time nonnegative/nondecreasing";
      return false;
    }
    if (!FiniteArray(rows.GetRow(i), width)) {
      diagnostic = "Malformed native state row " + std::to_string(i) + ": expected " +
          std::to_string(width) + " finite scalars";
      return false;
    }
  }
  return true;
}
Real3 Xyz(Span<double const> values, size_t offset = 0) {
  return {
      static_cast<real>(values[offset]),
      static_cast<real>(values[offset + 1]),
      static_cast<real>(values[offset + 2])};
}
} // namespace

std::shared_ptr<IKRecordingClip const> IKRecordingClip::Load(
    std::string const& path,
    BotPrefab const& inputBot,
    std::string& diagnostic,
    std::atomic_bool const* cancelled) {
  diagnostic.clear();
  auto const isCancelled = [&] {
    if (cancelled && cancelled->load()) {
      diagnostic = "Recording load cancelled";
      return true;
    }
    return false;
  };
  if (isCancelled())
    return {};
  auto const file = std::filesystem::u8path(path);
  auto extension = file.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (extension != ".rrd") {
    diagnostic = "Unsupported recording format; expected native RRD (.rrd), not MCAP";
    return {};
  }
  std::error_code fileError;
  auto const fileBytes = std::filesystem::file_size(file, fileError);
  if (fileError || fileBytes > kFileBytes) {
    diagnostic = fileError ? "Cannot read recording file: " + fileError.message()
                           : "Recording exceeds its 1 GiB file limit";
    return {};
  }
  if (!ValidateInputBot(inputBot, diagnostic))
    return {};
  Error error;
  // Public Open/ReadScalars eagerly decompress/decode. Their internal allocations and work cannot
  // be bounded or interrupted here; the file/result budgets and construction checks are separate.
  auto reader = EpisodeReader::Open(file, error);
  if (isCancelled())
    return {};
  if (!reader || !error.IsOK()) {
    diagnostic = error.GetDescription();
    return {};
  }
  std::string space;
  double units = 0;
  if (!Text(*reader, "episode", "superdex.episode.coordinate_space", space, diagnostic) ||
      space != "FLU" ||
      !Number(*reader, "episode", "superdex.episode.units_per_meter", units, diagnostic) ||
      units != 1) {
    diagnostic =
        "Native physics playback requires static episode FLU coordinates and units_per_meter=1: " +
        diagnostic;
    return {};
  }
  ActorLayout actor;
  if (!SelectActor(*reader, inputBot, actor, diagnostic) || isCancelled())
    return {};
  for (auto field : {kPositions, kTranslation, kRotation}) {
    if (reader->HasComponent(actor.path, field, true)) {
      diagnostic =
          "Native articulation state must be temporal complete snapshots, not static constants: " +
          std::string{field};
      return {};
    }
  }
  auto positions = reader->ReadScalars(actor.path, kPositions, error);
  if (isCancelled())
    return {};
  auto translations = reader->ReadScalars(actor.path, kTranslation, error);
  if (isCancelled())
    return {};
  auto rotations = reader->ReadScalars(actor.path, kRotation, error);
  if (isCancelled())
    return {};
  if (!error.IsOK()) {
    diagnostic = actor.path + ": " + error.GetDescription();
    return {};
  }
  size_t const count = positions.GetNumRows();
  size_t const bytesPerSample = sizeof(IKRecordingSample) + sizeof(int64_t) +
      actor.coordinates.size() * sizeof(real) + (actor.dofs + 7) * sizeof(double) +
      3 * (2 * sizeof(int64_t) + sizeof(size_t));
  if (count > kPoseBudget / bytesPerSample ||
      positions.values.size() > kPoseBudget / sizeof(double) ||
      translations.values.size() > kPoseBudget / sizeof(double) ||
      rotations.values.size() > kPoseBudget / sizeof(double)) {
    diagnostic = "Decoded native articulation clip exceeds its 256 MiB result/pose budget";
    return {};
  }
  if (!Rows(positions, actor.dofs, diagnostic) || !Rows(translations, 3, diagnostic) ||
      !Rows(rotations, 4, diagnostic))
    return {};
  // Each native component is independently chunked/sorted by EpisodeReader. Zip only after exact
  // equality of UNIQUE frame and integer sim_time keys. Never interpolate/hold a missing field.
  if (positions.frames != translations.frames || positions.frames != rotations.frames ||
      positions.simulationTimesNs != translations.simulationTimesNs ||
      positions.simulationTimesNs != rotations.simulationTimesNs) {
    diagnostic =
        "Native joint_positions/root translation/quaternion snapshot frame and sim_time alignment mismatch";
    return {};
  }
  auto clip = std::shared_ptr<IKRecordingClip>(new IKRecordingClip);
  clip->_compatibility = "Physics-scene articulation poses: " + actor.path + " (" + actor.identity +
      "); named joint types/DOF spans validated. Input asset supplies axes/rest frames; geometry/native scale not compared; no raw tracking confidence.";
  clip->_samples.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    if (isCancelled())
      return {};
    auto const q = positions.GetRow(i);
    auto const rotation = rotations.GetRow(i);
    Quaternion rootRotation{
        static_cast<real>(rotation[0]),
        static_cast<real>(rotation[1]),
        static_cast<real>(rotation[2]),
        static_cast<real>(rotation[3])};
    if (!UnitRotation(rootRotation)) {
      diagnostic = "Malformed native state row " + std::to_string(i) +
          ": root quaternion must be a unit rotation";
      return {};
    }
    IKRecordingSample sample;
    sample.frame = static_cast<uint64_t>(positions.frames[i]);
    sample.time = positions.simulationTimesNs[i];
    sample.recordingOrder = i;
    sample.recordedWorldFrameFromInputBaseFrame =
        TransformRT{Normalize(rootRotation), Xyz(translations.GetRow(i))};
    if (actor.freeRoot) {
      // Mochi's free joint is xyz + rotation vector, NOT xyz + Euler angles. The recorded actor
      // root is an anchor; free-root DoFs may carry ALL simulated wrist motion (as in UE captures).
      auto const freeRotation = Quaternion::FromRotationVector(Xyz(q, actor.rootOffset + 3));
      if (!UnitRotation(freeRotation)) {
        diagnostic = "Malformed native state row " + std::to_string(i) +
            ": invalid free-root rotation vector";
        return {};
      }
      sample.recordedWorldFrameFromInputBaseFrame = sample.recordedWorldFrameFromInputBaseFrame *
          TransformRT{freeRotation, Xyz(q, actor.rootOffset)};
    }
    if (!IsFinite(sample.recordedWorldFrameFromInputBaseFrame) ||
        !UnitRotation(sample.recordedWorldFrameFromInputBaseFrame.GetRotation())) {
      diagnostic =
          "Malformed native state row " + std::to_string(i) + ": nonfinite simulated root pose";
      return {};
    }
    for (auto coordinate : actor.coordinates)
      sample.inputBotConfiguration.push_back(static_cast<real>(q[coordinate]));
    sample.hasPose = true;
    sample.tracked =
        true; // Availability only; PhysicsRecorder has no tracking-loss/confidence flag.
    clip->_samples.push_back(std::move(sample));
    if (clip->_times.empty() || clip->_times.back() != positions.simulationTimesNs[i]) {
      clip->_times.push_back(positions.simulationTimesNs[i]);
    }
  }
  clip->_first = clip->_times.front();
  clip->_last = clip->_times.back();
  // SampleAt chooses the final complete snapshot at a tied sim_time; anchor that same pose.
  clip->_firstRecordedWorldFrameFromInputBaseFrame =
      clip->SampleAt(clip->_first)->recordedWorldFrameFromInputBaseFrame;
  if (isCancelled())
    return {};
  return clip;
}

IKRecordingSample const* IKRecordingClip::SampleAt(int64_t time) const {
  auto const next = std::upper_bound(
      _samples.begin(), _samples.end(), time, [](int64_t t, IKRecordingSample const& sample) {
        return t < sample.time;
      });
  return next == _samples.begin() ? nullptr : &*std::prev(next);
}
mochi::TransformRT IKRecordingClip::InputBotRoot(
    IKRecordingSample const& sample,
    mochi::TransformRT const& authoredWorldFrameFromInputBaseFrame,
    bool fixedBase) const {
  return fixedBase || !sample.hasPose
      ? authoredWorldFrameFromInputBaseFrame
      : authoredWorldFrameFromInputBaseFrame * Invert(_firstRecordedWorldFrameFromInputBaseFrame) *
          sample.recordedWorldFrameFromInputBaseFrame;
}
double IKRecordingClip::GetDurationSeconds() const {
  return static_cast<double>(Distance(_first, _last)) / 1e9;
}

void IKRecordingClock::Configure(IKRecordingClip const& clip) {
  Configure(clip.GetFirstTime(), clip.GetLastTime(), clip.GetTimes());
}
void IKRecordingClock::Configure(int64_t first, int64_t last, std::vector<int64_t> times) {
  _first = first;
  _last = std::max(first, last);
  _times = std::move(times);
  std::erase_if(_times, [&](auto time) { return time < _first || time > _last; });
  std::sort(_times.begin(), _times.end());
  _times.erase(std::unique(_times.begin(), _times.end()), _times.end());
  Stop();
}
void IKRecordingClock::Play(Clock::time_point now) {
  if (_playing) {
    Tick(now);
    return;
  }
  _lastTick = now;
  _playing = true;
}
void IKRecordingClock::Pause(Clock::time_point now) {
  Tick(now);
  _playing = false;
}
void IKRecordingClock::Stop() {
  _playing = false;
  _cursor = _first;
}
void IKRecordingClock::Step(Clock::time_point now) {
  Pause(now);
  auto const next = std::upper_bound(_times.begin(), _times.end(), _cursor);
  _cursor = next == _times.end() ? _first : *next;
}
int64_t IKRecordingClock::Tick(Clock::time_point now) {
  if (!_playing || now <= _lastTick)
    return _cursor;
  auto const elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now - _lastTick).count();
  _lastTick = now;
  auto const duration = Distance(_first, _last);
  if (duration == 0 || elapsed <= 0)
    return _cursor;
  auto offset = Distance(_first, _cursor) % duration;
  auto const advance = static_cast<uint64_t>(elapsed) % duration;
  offset = advance >= duration - offset ? advance - (duration - offset) : offset + advance;
  _cursor = AtOffset(_first, offset);
  return _cursor;
}
} // namespace superdex::studio
