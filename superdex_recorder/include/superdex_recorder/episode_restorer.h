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
#include <superdex_recorder/frame_bank.h>

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
  mochi::ActorType type = mochi::ActorType::None;
  bool isStatic = false;
};

/* @brief Finds the scene actor of each recorded actor. Returns them sorted by path.
 * An actor recorded with a spawn name binds to the scene actor named "<spawn name>/<runtime name>",
 * or else "<spawn name>", or else its runtime name; other actors bind by runtime name. Fails if a
 * recorded actor has no scene actor or one of another type, or if two bind to one scene actor. */
std::vector<RecordedActor>
BindRecordedActors(mochi::Scene& scene, EpisodeReader const& recording, mochi::Error& error);

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
  /* @brief Binds a bank's actors once, as BindRecordedActors does. Every episode must list the same
   * actors. A restore places each static actor at its episode's pose. Dynamic scene actors that
   * the bank does not hold keep their state, though a restore may wake them. */
  static std::unique_ptr<EpisodeRestorer>
  Create(mochi::Scene& scene, std::shared_ptr<FrameBank const> bank, mochi::Error& error);
  ~EpisodeRestorer();

  /* Recorded frame numbers of the first episode, by frame index. */
  [[nodiscard]] std::vector<int64_t> const& GetFrames() const;
  /* Sorted by path, including static actors. */
  [[nodiscard]] std::vector<RecordedActor> const& GetActors() const;

  /* Restores a frame of the first episode. */
  void RestoreFrame(size_t frameIndex, mochi::Error& error);
  void RestoreFrame(size_t episode, size_t frameIndex, mochi::Error& error);

 private:
  EpisodeRestorer(
      mochi::Scene& scene,
      std::shared_ptr<FrameBank const> bank,
      std::vector<RecordedActor> actors);

  mochi::Scene& _scene;
  std::shared_ptr<FrameBank const> _bank;
  std::vector<int64_t> _frames;
  std::vector<RecordedActor> _recordedActors;
  /* DOF indices 0, 1, ..., for every restore. */
  std::vector<int> _dofIndices;
};

} // namespace superdex::recorder
