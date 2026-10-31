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

#include <superdex_recorder/episode_reader.h>
#include <superdex_recorder/episode_restorer.h>
#include <superdex_recorder/physics_recorder.h>
#include <superdex_recorder/recorder.h>
#include <superdex_recorder/robot_shadow.h>
#include <superdex_recorder/robotics_recorder.h>
#include <superdex_recorder/skeleton_shadow.h>

#include <mochi_physics/pybind/core/pybind_core.h>
#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/file_utils.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

#if MOCHI_USE_DOUBLE_PRECISION
#define SUPERDEX_RECORDER_MODULE_NAME superdex_recorder_double
#else
#define SUPERDEX_RECORDER_MODULE_NAME superdex_recorder
#endif

namespace {

template <typename Function>
void CallWithError(Function&& function) {
  mochi::Error error;
  function(error);
  if (!error.IsOK()) {
    throw mochi::MochiErrorException{error};
  }
}

template <typename Function>
auto CallWithErrorResult(Function&& function) {
  mochi::Error error;
  auto result = function(error);
  if (!error.IsOK()) {
    throw mochi::MochiErrorException{error};
  }
  return result;
}

using RealArray = nb::ndarray<mochi::real const, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

mochi::Span<mochi::real const> AsSpan(RealArray const& values) {
  return {values.data(), values.size()};
}

superdex::recorder::SampleTime Sample(uint64_t frame, double simulationTimeSec) {
  return {.frame = frame, .simulationTimeSec = simulationTimeSec};
}

/* Binds an EpisodeReader method that reads one component of one entity. */
template <auto Read>
auto ReadComponent() {
  return [](superdex::recorder::EpisodeReader const& self,
            std::string const& entityPath,
            std::string const& component) {
    return CallWithErrorResult(
        [&](mochi::Error& error) { return (self.*Read)(entityPath, component, error); });
  };
}

} // namespace

// NOLINTNEXTLINE(performance-unnecessary-value-param)
NB_MODULE(SUPERDEX_RECORDER_MODULE_NAME, module) {
  nb::set_leak_warnings(false);
  nb::module_::import_("superdex.physics");
  nb::module_::import_("superdex.robotics");

  using namespace superdex::recorder;

  nb::class_<RecorderConfig>(module, "RecorderConfig")
      .def(
          nb::new_([](std::filesystem::path outputDirectory) {
            return RecorderConfig{std::move(outputDirectory)};
          }),
          nb::arg("output_directory"))
      .def_rw("output_directory", &RecorderConfig::outputDirectory);

  nb::class_<EpisodeMetadata>(module, "EpisodeMetadata")
      .def(
          nb::new_([](std::map<std::string, std::string> properties) {
            return EpisodeMetadata{std::move(properties)};
          }),
          nb::arg("properties") = std::map<std::string, std::string>{});

  nb::class_<Episode>(module, "Episode")
      .def(
          "commit",
          [](Episode& self) { CallWithError([&](mochi::Error& error) { self.Commit(error); }); })
      .def(
          "abort",
          [](Episode& self, std::string const& reason) {
            CallWithError([&](mochi::Error& error) { self.Abort(reason, error); });
          },
          nb::arg("reason"))
      .def_prop_ro("episode_id", &Episode::GetEpisodeId)
      .def_prop_ro("output_path", [](Episode const& self) { return self.GetOutputPath(); })
      .def_prop_ro("is_active", &Episode::IsActive)
      .def(
          "__enter__", [](Episode& self) -> Episode& { return self; }, nb::rv_policy::reference)
      .def(
          "__exit__",
          [](Episode& self, nb::handle, nb::handle exception, nb::handle) {
            if (!self.IsActive()) {
              return false;
            }

            mochi::Error error;
            self.Abort("episode context exited without commit", error);
            if (!error.IsOK()) {
              if (exception.is_none()) {
                throw mochi::MochiErrorException{error};
              }
              exception.attr("add_note")("Episode cleanup also failed: " + error.ToString());
            }
            return false;
          },
          nb::arg("exception_type").none(),
          nb::arg("exception").none(),
          nb::arg("traceback").none());

  nb::class_<Recorder>(module, "Recorder")
      .def(nb::init<RecorderConfig>(), nb::arg("config"))
      .def(
          "begin_episode",
          [](Recorder const& self, EpisodeMetadata const& metadata) {
            return CallWithErrorResult(
                [&](mochi::Error& error) { return self.BeginEpisode(metadata, error); });
          },
          nb::arg("metadata") = EpisodeMetadata{});

  nb::class_<PhysicsRecorderConfig>(module, "PhysicsRecorderConfig")
      .def(nb::init<>())
      .def_rw("record_contact_forces", &PhysicsRecorderConfig::recordContactForces)
      .def_rw(
          "contact_force_visualization_scale_meters_per_newton",
          &PhysicsRecorderConfig::contactForceVisualizationScaleMetersPerNewton)
      .def_rw("spawn_names", &PhysicsRecorderConfig::spawnNames);

  nb::class_<PhysicsRecorder>(module, "PhysicsRecorder")
      .def(
          nb::new_([](Episode& episode, mochi::Scene& scene, PhysicsRecorderConfig const& config) {
            return CallWithErrorResult([&](mochi::Error& error) {
              return std::make_unique<PhysicsRecorder>(episode, scene, config, error);
            });
          }),
          nb::arg("episode"),
          nb::arg("scene"),
          nb::arg("config") = PhysicsRecorderConfig{},
          nb::keep_alive<0, 2>(),
          nb::keep_alive<0, 3>())
      .def(
          "record",
          [](PhysicsRecorder& self, uint64_t frame, double simulationTimeSec) {
            CallWithError(
                [&](mochi::Error& error) { self.Record(Sample(frame, simulationTimeSec), error); });
          },
          nb::arg("frame"),
          nb::arg("simulation_time_sec"));

  // Native episode reader, which the parity tests compare with Rerun's own reader.
  nb::class_<EpisodeComponent>(module, "_EpisodeComponent")
      .def_ro("entity_path", &EpisodeComponent::entityPath)
      .def_ro("component", &EpisodeComponent::component)
      .def_ro("is_static", &EpisodeComponent::isStatic);
  nb::class_<ScalarRows>(module, "_ScalarRows")
      .def_ro("frames", &ScalarRows::frames)
      .def_ro("simulation_times_ns", &ScalarRows::simulationTimesNs)
      .def_ro("offsets", &ScalarRows::offsets)
      .def_ro("values", &ScalarRows::values);
  nb::class_<EpisodeReader>(module, "_EpisodeReader")
      .def(
          nb::new_([](std::filesystem::path const& path) {
            return CallWithErrorResult(
                [&](mochi::Error& error) { return EpisodeReader::Open(path, error); });
          }),
          nb::arg("path"))
      .def("components", &EpisodeReader::GetComponents)
      .def(
          "read_static_texts",
          ReadComponent<&EpisodeReader::ReadStaticTexts>(),
          nb::arg("entity_path"),
          nb::arg("component"))
      .def(
          "read_static_scalars",
          ReadComponent<&EpisodeReader::ReadStaticScalars>(),
          nb::arg("entity_path"),
          nb::arg("component"))
      .def(
          "read_scalars",
          ReadComponent<&EpisodeReader::ReadScalars>(),
          nb::arg("entity_path"),
          nb::arg("component"));

  // Native core of the Python EpisodeRestorer.
  nb::class_<RecordedActor>(module, "_RecordedActor")
      .def_ro("path", &RecordedActor::path)
      .def_ro("handle", &RecordedActor::handle)
      .def_ro("is_articulated", &RecordedActor::isArticulated)
      .def_ro("is_static", &RecordedActor::isStatic);
  module.def(
      "_bind_recorded_actors",
      [](mochi::Scene& scene, EpisodeReader const& recording) {
        return CallWithErrorResult(
            [&](mochi::Error& error) { return BindRecordedActors(scene, recording, error); });
      },
      nb::arg("scene"),
      nb::arg("recording"));
  nb::class_<EpisodeRestorer>(module, "_EpisodeRestorer")
      .def(
          nb::new_([](mochi::Scene& scene, EpisodeReader const& recording) {
            return CallWithErrorResult([&](mochi::Error& error) {
              return EpisodeRestorer::Create(scene, recording, error);
            });
          }),
          nb::arg("scene"),
          nb::arg("recording"),
          nb::keep_alive<0, 1>())
      .def_prop_ro("frames", &EpisodeRestorer::GetFrames)
      .def("actors", &EpisodeRestorer::GetActors)
      .def(
          "restore_frame",
          [](EpisodeRestorer& self, size_t frameIndex) {
            CallWithError([&](mochi::Error& error) { self.RestoreFrame(frameIndex, error); });
          },
          nb::arg("frame_index"));

  // Content hash that the Unreal recorder stores for the scene archive and task files.
  module.def(
      "_hash_file",
      [](std::string const& path) {
        auto const hash = CallWithErrorResult(
            [&](mochi::Error& error) { return superdex::robotics::HashGenericFile(path, error); });
        return std::string{hash.c_str(), hash.size()};
      },
      nb::arg("path"));

  nb::class_<RoboticsRecorder>(module, "RoboticsRecorder")
      .def(
          nb::new_([](Episode& episode, superdex::robotics::Bot& robot) {
            return CallWithErrorResult([&](mochi::Error& error) {
              return std::make_unique<RoboticsRecorder>(episode, robot, error);
            });
          }),
          nb::arg("episode"),
          nb::arg("robot"),
          nb::keep_alive<0, 2>(),
          nb::keep_alive<0, 3>())
      .def(
          "record",
          [](RoboticsRecorder& self, uint64_t frame, double simulationTimeSec) {
            CallWithError(
                [&](mochi::Error& error) { self.Record(Sample(frame, simulationTimeSec), error); });
          },
          nb::arg("frame"),
          nb::arg("simulation_time_sec"));

  nb::class_<RobotShadow>(module, "RobotShadow")
      .def(
          nb::new_([](Episode& episode,
                      superdex::robotics::Bot& robot,
                      std::string id,
                      std::optional<size_t> subtreeRootLinkIndex,
                      std::array<uint8_t, 4> color) {
            return CallWithErrorResult([&](mochi::Error& error) {
              return std::make_unique<RobotShadow>(
                  episode,
                  robot,
                  RobotShadowConfig{std::move(id), subtreeRootLinkIndex, color},
                  error);
            });
          }),
          nb::arg("episode"),
          nb::arg("robot"),
          nb::arg("id"),
          nb::arg("subtree_root_link_index") = nb::none(),
          nb::arg("color_rgba") = RobotShadowConfig{}.colorRgba,
          nb::keep_alive<0, 2>(),
          nb::keep_alive<0, 3>())
      .def(
          "record",
          [](RobotShadow& self,
             uint64_t frame,
             double simulationTimeSec,
             mochi::TransformRT const& worldFromAnchor,
             RealArray const& jointPositions) {
            CallWithError([&](mochi::Error& error) {
              self.Record(
                  Sample(frame, simulationTimeSec), worldFromAnchor, AsSpan(jointPositions), error);
            });
          },
          nb::arg("frame"),
          nb::arg("simulation_time_sec"),
          nb::arg("world_from_anchor"),
          nb::arg("joint_positions"));

  nb::class_<SkeletonShadow>(module, "SkeletonShadow")
      .def(
          nb::new_([](Episode& episode,
                      superdex::robotics::Bot& robot,
                      std::string id,
                      std::vector<std::string> keypointNames,
                      std::vector<int32_t> parentIndices,
                      std::string label,
                      std::array<uint8_t, 4> color,
                      float keypointRadiusMeters) {
            return CallWithErrorResult([&](mochi::Error& error) {
              return std::make_unique<SkeletonShadow>(
                  episode,
                  robot,
                  SkeletonShadowConfig{
                      .id = std::move(id),
                      .label = std::move(label),
                      .keypointNames = std::move(keypointNames),
                      .parentIndices = std::move(parentIndices),
                      .colorRgba = color,
                      .keypointRadiusMeters = keypointRadiusMeters},
                  error);
            });
          }),
          nb::arg("episode"),
          nb::arg("robot"),
          nb::arg("id"),
          nb::arg("keypoint_names"),
          nb::arg("parent_indices"),
          nb::arg("label") = "",
          nb::arg("color_rgba") = SkeletonShadowConfig{}.colorRgba,
          nb::arg("keypoint_radius_meters") = SkeletonShadowConfig{}.keypointRadiusMeters,
          nb::keep_alive<0, 2>())
      .def(
          "record",
          [](SkeletonShadow& self,
             uint64_t frame,
             double simulationTimeSec,
             nb::ndarray<mochi::real const, nb::c_contig, nb::device::cpu> const&
                 worldKeypointsXyz) {
            CallWithError([&](mochi::Error& error) {
              MOCHI_ERROR_IF_NOT(
                  (worldKeypointsXyz.ndim() == 1 && worldKeypointsXyz.size() % 3 == 0) ||
                      (worldKeypointsXyz.ndim() == 2 && worldKeypointsXyz.shape(1) == 3),
                  error,
                  "Skeleton shadow keypoints must be XYZ triples, as a flat or an (N, 3) array.");
              MOCHI_ERROR_RETURN(error);
              mochi::Span<mochi::real const> const xyz{
                  worldKeypointsXyz.data(), worldKeypointsXyz.size()};
              std::vector<mochi::Real3> keypoints(xyz.size() / 3);
              for (size_t i = 0; i < keypoints.size(); ++i) {
                keypoints[i] = {xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]};
              }
              self.Record(Sample(frame, simulationTimeSec), keypoints, error);
            });
          },
          nb::arg("frame"),
          nb::arg("simulation_time_sec"),
          nb::arg("world_keypoints_xyz"))
      .def(
          "record_untracked",
          [](SkeletonShadow& self, uint64_t frame, double simulationTimeSec) {
            CallWithError([&](mochi::Error& error) {
              self.RecordUntracked(Sample(frame, simulationTimeSec), error);
            });
          },
          nb::arg("frame"),
          nb::arg("simulation_time_sec"));
}
