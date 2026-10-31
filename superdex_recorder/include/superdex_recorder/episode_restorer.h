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

#include <superdex_recorder/episode_reader.h>
#include <superdex_recorder/export.h>

#include <mochi_core/utils/error.h>
#include <mochi_physics/cpp_api/mochi_scene.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace superdex::recorder {

/* @brief A recorded actor and the scene actor that it binds to. */
struct RecordedActor {
  /* Recorded data path, such as "data/physics/actors/cube". */
  std::string path;
  /* The actor's name when it was recorded. */
  std::string runtimeName;
  mochi::ActorHandle handle;
  bool isArticulated = false;
  bool isStatic = false;
};

/* @brief Finds the scene actor of each recorded actor. Returns them sorted by path.
 * An actor recorded with a spawn name binds to the scene actor named "<spawn name>/<runtime name>",
 * or else "<spawn name>", or else its runtime name; other actors bind by runtime name. Fails if a
 * recorded actor has no scene actor or one of another type, or if two bind to one scene actor. */
std::vector<RecordedActor>
BindRecordedActors(mochi::Scene& scene, EpisodeReader const& recording, mochi::Error& error);

/* @brief Recorded state of one dynamic actor at every frame, concatenated in frame order. */
struct RestoredActor {
  mochi::ActorHandle handle;
  bool isArticulated = false;
  /* xyzw per frame. */
  std::vector<mochi::real> rootRotations;
  /* Rigid actors: center-of-mass position and velocities, xyz per frame. */
  std::vector<mochi::real> comTranslations;
  std::vector<mochi::real> linearVelocities;
  std::vector<mochi::real> angularVelocities;
  /* Articulated actors: root position (xyz per frame) and one value per DoF per frame. */
  std::vector<mochi::real> rootTranslations;
  std::vector<mochi::real> jointPositions;
  std::vector<mochi::real> jointVelocities;
  /* Empty for actors without a pose controller. */
  std::vector<mochi::real> controllerTargets;
  /* External forces during the step after each frame, one value per DoF per frame. Empty for
   * actors without recorded joint efforts. */
  std::vector<mochi::real> jointEfforts;
};

/* @brief Restores a scene to the end of any recorded step.
 * Each restore sets the scene clock, each actor's state through the public setters, pose-controller
 * targets, and external forces, then takes a zero step. The integrator starts afresh from the
 * restored state. The scene must outlive the restorer. */
class SUPERDEX_RECORDER_API EpisodeRestorer final {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(EpisodeRestorer)

 public:
  /* @brief Binds a recorded episode to a scene, as BindRecordedActors does.
   * Static scene actors must be at their recorded poses, and every dynamic actor that is not
   * nested must have been recorded. Restoring a frame applies each robot's joint effort recorded
   * for the step after it, which holds until the next recorded effort. Actors that were spawned or
   * destroyed during the episode are not supported. */
  static std::unique_ptr<EpisodeRestorer>
  Create(mochi::Scene& scene, EpisodeReader const& recording, mochi::Error& error);
  ~EpisodeRestorer();

  /* Recorded frame numbers, by frame index. */
  [[nodiscard]] std::vector<int64_t> const& GetFrames() const;
  /* Sorted by path, including static actors. */
  [[nodiscard]] std::vector<RecordedActor> const& GetActors() const;

  void RestoreFrame(size_t frameIndex, mochi::Error& error);

 private:
  explicit EpisodeRestorer(mochi::Scene& scene);

  mochi::Scene& _scene;
  std::vector<int64_t> _frames;
  std::vector<double> _simulationTimesSec;
  std::vector<double> _stepDtsSec;
  std::vector<RecordedActor> _recordedActors;
  std::vector<RestoredActor> _actors;
};

} // namespace superdex::recorder
