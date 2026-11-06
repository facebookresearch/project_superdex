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

/* PhysicsRecorder records a Mochi scene's actors at every Record, so that EpisodeRestorer can
 * restore the scene to any recorded frame:
 * - Rigid actors: root pose and velocities, and the center of mass of dynamic ones. Static actors
 *   are recorded once.
 * - Articulated actors: root pose, joint positions and velocities, pose-controller targets, and
 *   link transforms.
 * - Soft actors: root pose, displacements and node velocities, and surface vertices for viewing.
 *
 * Not supported:
 * - Shell and rod actors are skipped with a warning, and the episode cannot be restored.
 * - Nested soft actors, such as soft pads on an articulated actor, move with their articulated
 *   actor; their own deformation is not recorded.
 * - Actors created after the recorder are not recorded.
 * - Contact recording supports only rigid and articulated actors, and fails on a scene with others.
 */

#pragma once

#include <superdex_recorder/contact_collector.h>
#include <superdex_recorder/link_kinematics.h>
#include <superdex_recorder/recorder.h>

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/transform_rt.h>
#include <mochi_physics/cpp_api/mochi_actor.h>
#include <mochi_physics/cpp_api/mochi_scene.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace superdex::recorder {

/* @brief Optional physics channels recorded alongside kinematics. */
struct PhysicsRecorderConfig {
  /* @brief Record contact-point forces, in the layout of ContactColumns, and Rerun force arrows. */
  bool recordContactForces = false;
  /* @brief Arrow length per newton [m/N]. */
  mochi::real contactForceVisualizationScaleMetersPerNewton = static_cast<mochi::real>(0.002);
  /* @brief The task spawn name of each actor that a scene task spawned.
   *
   * Each named actor records its spawn name, which a rebuilt scene uses to find the matching body.
   * Like the legacy teleop recorder, a spawn's only body is recorded under the spawn name. Bodies
   * of a spawn with several are recorded as "<spawn name>.<Mochi name>". A `/` in a spawn or Mochi
   * name becomes `.` in the recorded actor ID; `spawn_name` and `runtime_name` keep the originals.
   */
  std::vector<std::pair<mochi::ActorHandle, std::string>> spawnNames;
};

/* Records all supported actors of a Mochi scene, as described at the top of this file.
 * The episode and scene must outlive this recorder. Record fails if a recorded actor is
 * destroyed. */
class SUPERDEX_RECORDER_API PhysicsRecorder final {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(PhysicsRecorder)

 private:
  struct Link {
    Link(std::string name, TransformSeries& transformSeries);

    std::string name;
    TransformSeries& transformSeries;
  };

  struct Actor {
    Actor(
        std::string id,
        std::string dataPath,
        mochi::ActorHandle handle,
        ScalarSeries& stateSeries,
        TransformSeries& transformSeries);

    std::string id;
    std::string dataPath;
    mochi::ActorHandle handle;
    ScalarSeries& stateSeries;
    TransformSeries& transformSeries;
    std::vector<Link> links;
    mochi::DynamicArray<mochi::real> positions;
    mochi::DynamicArray<mochi::real> velocities;
    mochi::DynamicArray<mochi::real> controllerTargets;
    bool hasPoseController = false;
    std::optional<detail::LinkKinematics> linkKinematics;
    /* Soft actors. */
    DeformableMeshSeries* surfaceSeries = nullptr;
    /* Mochi queries this recorder registered on the actor; the destructor cancels them. */
    std::vector<mochi::QueryHandle> queries;
  };

 public:
  PhysicsRecorder(Episode& episode, mochi::Scene& scene, mochi::Error& error);
  PhysicsRecorder(
      Episode& episode,
      mochi::Scene& scene,
      PhysicsRecorderConfig const& config,
      mochi::Error& error);
  ~PhysicsRecorder();

  void Record(SampleTime time, mochi::Error& error);

 private:
  void RegisterActor(std::string id, mochi::Actor& actor, mochi::Error& error);
  void RegisterContactBody(std::string path, mochi::Actor& actor);
  void RecordContacts(SampleTime time, mochi::Error& error);
  void WriteGeometry(std::string_view entityPath, mochi::Actor const& actor, mochi::Error& error);
  void
  WritePhysicalParameters(std::string_view path, mochi::Actor const& actor, mochi::Error& error);

  Episode& _episode;
  mochi::Scene& _scene;
  PhysicsRecorderConfig _config;
  double _contactRegistrationTimeSec = 0.0;
  ScalarSeries* _frameSeries = nullptr;
  ScalarSeries* _contactSeries = nullptr;
  Arrow3DSeries* _contactArrowSeries = nullptr;
  std::vector<Actor> _actors;
  std::vector<ContactBody> _contactBodies;
  std::unique_ptr<ContactCollector> _contactCollector;
  ContactColumns _contactColumns;
};

} // namespace superdex::recorder
