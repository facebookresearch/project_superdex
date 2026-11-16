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

#include <superdex_robotics/sensors/tactile_observation.h>

#include <cstdint>

namespace superdex::robotics {

/* @brief Spatial meaning of an image-like tactile force map. */
struct RectMapMeta {
  SensorDimension rowAxis = -1;
  SensorDimension columnAxis = -1;
  SensorDimension componentAxis = -1;
  Real3 centerMeters = {};
  Real3 tipDirection = {};
  Real3 zeroLongitudeDirection = {};
  real maxPolarLatitudeRadians = 0_r;
  Span<int32_t const> cellToSourceVertex;
};

template <>
struct SensorMetadataKey<RectMapMeta> {
  static constexpr std::string_view Key = "tactile.rect_map.v1";
};

/* @brief Fixed geometry and execution mode for a SEED force RectMap sensor. */
struct SensorSeedRectMapParams {
  Real3 center = {};
  Real3 tipDirection{0_r, 0_r, 1_r};
  Real3 zeroLongitudeDirection{1_r, 0_r, 0_r};
  int32_t width = 256;
  int32_t height = 256;
  real maxPolarLatitude = 1.5707963267948966_r;
  TactileRectProjectionType projection = TactileRectProjectionType::AzimuthalEqualArea;
  TactileRectRasterizationType rasterization = TactileRectRasterizationType::ConservativeVoronoi;
  int32_t voronoiOversampling = 4;
  TactileRectOutsidePolicy outsidePolicy = TactileRectOutsidePolicy::Reject;

  MOCHI_STRUCT_BEGIN(superdex::robotics::SensorSeedRectMapParams)
  MOCHI_FIELD(center) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(tipDirection);
  MOCHI_FIELD(zeroLongitudeDirection);
  MOCHI_FIELD(width);
  MOCHI_FIELD(height);
  MOCHI_FIELD(maxPolarLatitude) MOCHI_ATTRIBUTE(Units("rad"));
  MOCHI_FIELD(projection);
  MOCHI_FIELD(rasterization);
  MOCHI_FIELD(voronoiOversampling);
  MOCHI_FIELD(outsidePolicy);
  MOCHI_STRUCT_END()
};

/* @brief Image-like force map derived from one vertex-force point-cloud sensor. */
class MOCHI_API SensorSeedRectMap final : public SensorBase {
 public:
  using Params = SensorSeedRectMapParams;

  static constexpr std::string_view TypeName() {
    return "SENSOR_SEED_RECT_MAP";
  }

  SensorSeedRectMap(Actor* actor, std::string_view paramArgs, superdex::Error& error);
  SensorSeedRectMap(Actor* actor, Params const& params, superdex::Error& error);
  explicit SensorSeedRectMap(Params const& params, superdex::Error& error);
  ~SensorSeedRectMap() override;

  [[nodiscard]] std::string_view GetTypeName() const override {
    return TypeName();
  }

  [[nodiscard]] SensorStepResult Step(StepTime const& now, superdex::Error& error) override;

  [[nodiscard]] Span<std::byte const> TryGetMetadata(std::string_view key) const override;

  [[nodiscard]] Params const& GetParams() const {
    return _params;
  }

  [[nodiscard]] TactileRectProjectionMap const& GetProjection() const {
    return _projection;
  }

 private:
  [[nodiscard]] SensorFormat InitSensor(SensorScope const& scope, superdex::Error& error) override;
  void ResetSensor() override {}

  void Initialize(Params const& params, superdex::Error& error);
  [[nodiscard]] TactileHemisphereRectParams ProjectionParams() const;
  void ReadInput(SensorSample const& source, superdex::Error& error);
  void ComputeInto(float* output, superdex::Error& error);

  Params _params;
  TactileTaxelLayout _sourceLayout;
  TactileRectProjectionMap _projection;
  TactileTaxelForces _inputForces;
  TactileForceGrid _grid;
  DynamicArray<float> _syncOutput;
  DynamicArray<int32_t> _cellToSourceVertex;
  RectMapMeta _rectMeta;
};

} // namespace superdex::robotics
