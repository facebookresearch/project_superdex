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

#include <superdex_robotics/sensors/interaction_wrench_metadata.h>
#include <superdex_robotics/sensors/sensor_base.h>
#include <superdex_robotics/sensors/tactile_observation.h>

namespace superdex::robotics {

/* @brief Ordered spatial meaning of a vertex-list tensor. */
struct VertexListMeta {
  DynamicString coordinateFrame;
  SensorDimension vertexAxis = -1;
  Span<Real3 const> positionsMeters;
  Span<Real3 const> normals;
};

template <>
struct SensorMetadataKey<VertexListMeta> {
  static constexpr std::string_view Key = "tactile.vertex_list.v1";
};

// NOLINTNEXTLINE(cppcoreguidelines-virtual-class-destructor)
class MOCHI_API SensorLinkVertexForce : public SensorBase {
 public:
  static constexpr std::string_view TypeName() {
    return "SENSOR_LINK_VERTEX_FORCE";
  }

  [[nodiscard]] std::string_view GetTypeName() const override {
    return TypeName();
  }

  SensorLinkVertexForce(Actor* linkActor, std::string_view paramArgs, superdex::Error& error);
  ~SensorLinkVertexForce() override;

  MOCHI_DECLARE_NO_COPY_NO_MOVE(SensorLinkVertexForce)

  [[nodiscard]] SensorStepResult Step(StepTime const& now, superdex::Error& error) override;

  [[nodiscard]] Span<std::byte const> TryGetMetadata(std::string_view key) const override;

  void ConfigureRecording(SensorRecordingSink& sink, superdex::Error& error) const override;

  [[nodiscard]] TactileTaxelForces ComputeSignal(superdex::Error& error);

  [[nodiscard]] TactileTaxelLayout const& GetLayout() const {
    return _layout;
  }

 private:
  void ResetSensor() override {}

  [[nodiscard]] SensorFormat InitSensor(SensorScope const& scope, superdex::Error& error) override;

  void ResolveLayout(superdex::Error& error);

  TactileTaxelLayout _layout;
  VertexListMeta _metadata;
  InteractionWrenchMeta _interactionMetadata;
  TactileTaxelForces _graphForces;
  DynamicArray<float> _graphOutput;
  QueryHandle _nodeForceQuery;
  bool _queryRegistered = false;
};

} // namespace superdex::robotics
