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

#include <superdex_physics.h>
#include <superdex_robotics/sensors/sensor_base.h>

#include <string_view>

namespace superdex::robotics {

/* @brief Row identity and spatial meaning of an actor-wide articulated-joint wrench tensor. */
struct ArticulatedJointWrenchMeta {
  Scene const* scene = nullptr;
  ActorHandle articulatedActor;
  DynamicString coordinateFrame;
  DynamicString momentReference;
  Span<int32_t const> rowJointIndices;
  Span<int32_t const> parentLinkIndices;
  Span<int32_t const> omittedStaticLinkIndices;
};

template <>
struct SensorMetadataKey<ArticulatedJointWrenchMeta> {
  static constexpr std::string_view Key = "wrench.articulated_joint_list.v1";
};

/* @brief Return true for each static link: its own joint and every ancestor joint are Hard. */
[[nodiscard]] MOCHI_API DynamicArray<bool> ClassifyStaticArticulatedLinks(
    Span<ArticulatedJointType const> jointTypes,
    Span<int32_t const> parents,
    superdex::Error& error);

/* @brief Actor-wide source of every solver-backed articulated-joint wrench that Mochi can measure.
 *
 * The output has shape [N, 6], where rows follow articulated-link index order and components are
 * force xyz followed by torque xyz. Wrenches are the proximal side acting on the distal side, in
 * world frame, with torque about the child-side joint origin. A row is omitted only when its child
 * link is static: its own joint and every ancestor joint up to the root are Hard, so the link is
 * welded to the world and Mochi computes no contact forces for it. Every joint with a dynamic child
 * is reported, even when all its ancestors are static, and its wrench is complete: a dynamic link
 * has no static descendants, so every force on the distal subtree is included. The stable
 * row-to-joint mapping and omitted indices are carried in typed metadata.
 *
 * This source is mounted on any nested link only to identify its owning articulated actor. A
 * @ref SensorBotaV1 mounted at a particular distal link selects one row declaratively. */
// NOLINTNEXTLINE(cppcoreguidelines-virtual-class-destructor)
class MOCHI_API SensorArticulatedJointWrench : public SensorBase {
 public:
  static constexpr std::string_view TypeName() {
    return "SENSOR_ARTICULATED_JOINT_WRENCH";
  }

  [[nodiscard]] std::string_view GetTypeName() const override {
    return TypeName();
  }

  SensorArticulatedJointWrench(
      Actor* linkActor,
      std::string_view paramArgs,
      superdex::Error& error);

  ~SensorArticulatedJointWrench() override;

  void Shutdown();

  [[nodiscard]] SensorStepResult Step(StepTime const& now, superdex::Error& error) override;

  [[nodiscard]] Span<std::byte const> TryGetMetadata(std::string_view key) const override;

  void ConfigureRecording(SensorRecordingSink& sink, superdex::Error& error) const override;

  [[nodiscard]] Span<int32_t const> GetJointIndices() const {
    return _layout.rowJointIndices;
  }

 private:
  void ResetSensor() override {}

  [[nodiscard]] SensorFormat InitSensor(SensorScope const& scope, superdex::Error& error) override;

  void Initialize(Actor* linkActor, std::string_view paramArgs, superdex::Error& error);
  void ReadQuery(superdex::Error& error);

  bool _initialized = false;
  ActorHandle _articulatedHandle = {};
  QueryHandle _query = {};
  bool _queryRegistered = false;
  DynamicArray<int32_t> _rowJointIndices;
  DynamicArray<int32_t> _parentLinkIndices;
  DynamicArray<int32_t> _omittedStaticLinkIndices;
  ArticulatedJointWrenchMeta _layout;
  DynamicArray<float> _queryValues;
};

} // namespace superdex::robotics
