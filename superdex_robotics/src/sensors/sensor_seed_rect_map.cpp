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

#include <superdex_robotics/sensors/sensor_link_vertex_force.h>
#include <superdex_robotics/sensors/sensor_seed_rect_map.h>

#include <superdex_robotics/utils/file_utils.h>

#include <cmath>
#include <numbers>

using namespace mochi;
using namespace superdex::robotics;

namespace {

DynamicArray<int32_t> MakeCellToSourceVertex(TactileRectProjectionMap const& projection) {
  std::size_t const cells =
      static_cast<std::size_t>(projection.width) * static_cast<std::size_t>(projection.height);
  DynamicArray<int32_t> result(cells, -1);
  DynamicArray<real> bestWeight(cells, -1_r);
  MOCHI_ASSERT(
      projection.taxelContributionOffsets.size() == projection.taxelCells.size() + 1,
      "A tactile projection must use N+1 CSR contribution offsets");
  for (std::size_t vertex = 0; vertex < projection.taxelCells.size(); ++vertex) {
    int32_t const begin = projection.taxelContributionOffsets[vertex];
    int32_t const end = projection.taxelContributionOffsets[vertex + 1];
    for (int32_t contribution = begin; contribution < end; ++contribution) {
      auto const contributionIndex = static_cast<std::size_t>(contribution);
      int32_t const cell = projection.contributionCells[contributionIndex];
      auto const cellIndex = static_cast<std::size_t>(cell);
      real const weight = projection.contributionWeights[contributionIndex];
      if (weight > bestWeight[cellIndex]) {
        bestWeight[cellIndex] = weight;
        result[cellIndex] = static_cast<int32_t>(vertex);
      }
    }
  }
  return result;
}

} // namespace

SensorSeedRectMap::SensorSeedRectMap(Actor* actor, std::string_view paramArgs, Error& error)
    : SensorBase(actor, error) {
  Params const params = paramArgs.empty()
      ? Params{}
      : LoadParamsFromPathOrJson<Params>(paramArgs, error, /*allowPartial=*/false);
  Initialize(params, error);
}

SensorSeedRectMap::SensorSeedRectMap(Actor* actor, Params const& params, Error& error)
    : SensorBase(actor, error) {
  Initialize(params, error);
}

SensorSeedRectMap::SensorSeedRectMap(Params const& params, Error& error)
    : SensorSeedRectMap(nullptr, params, error) {}

SensorSeedRectMap::~SensorSeedRectMap() = default;

void SensorSeedRectMap::Initialize(Params const& params, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      params.width < 2 || params.height < 2,
      error,
      "SensorSeedRectMap requires width and height of at least two");
  MOCHI_ERROR_RETURN(error);
  int64_t const cells = static_cast<int64_t>(params.width) * static_cast<int64_t>(params.height);
  MOCHI_ERROR_IF(
      cells > std::numeric_limits<int32_t>::max(), error, "SensorSeedRectMap dimensions overflow");
  MOCHI_ERROR_RETURN(error);
  constexpr real kPi = std::numbers::pi_v<real>;
  MOCHI_ERROR_IF(
      !std::isfinite(params.maxPolarLatitude) || params.maxPolarLatitude <= 0_r ||
          params.maxPolarLatitude >= kPi,
      error,
      "SensorSeedRectMap maxPolarLatitude must be in (0, pi)");
  MOCHI_ERROR_RETURN(error);
  _params = params;
}

TactileHemisphereRectParams SensorSeedRectMap::ProjectionParams() const {
  TactileHemisphereRectParams params;
  params.center = _params.center;
  params.tipDirection = _params.tipDirection;
  params.zeroLongitudeDirection = _params.zeroLongitudeDirection;
  params.width = _params.width;
  params.height = _params.height;
  params.maxPolarLatitude = _params.maxPolarLatitude;
  params.projection = _params.projection;
  params.rasterization = _params.rasterization;
  params.voronoiOversampling = _params.voronoiOversampling;
  params.outsidePolicy = _params.outsidePolicy;
  return params;
}

SensorFormat SensorSeedRectMap::InitSensor(SensorScope const& /*scope*/, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  Span<SensorBase const* const> const inputs = Inputs();
  MOCHI_ERROR_IF_NOT(inputs.size() == 1, error, "SensorSeedRectMap requires one input");
  MOCHI_ERROR_RETURN(error, {});
  SensorFormat const& source = inputs[0]->GetFormat();
  MOCHI_ERROR_IF_NOT(
      source.kind == kSensorFormatKindForce3D && source.shape.size() == 2 && source.shape[0] > 0 &&
          source.shape[1] == 3,
      error,
      "SensorSeedRectMap requires a Force3D [N, 3] input");
  MOCHI_ERROR_RETURN(error, {});
  auto const* const sourceMeta = TryGetSensorMetadata<VertexListMeta>(*inputs[0]);
  if (sourceMeta == nullptr) {
    MOCHI_ERROR_SET(error, "SensorSeedRectMap requires VertexListMeta");
    return {};
  }
  MOCHI_ERROR_IF_NOT(
      sourceMeta->vertexAxis == 0 &&
          sourceMeta->positionsMeters.size() == static_cast<std::size_t>(source.shape[0]),
      error,
      "SensorSeedRectMap vertex metadata does not match its input format");
  MOCHI_ERROR_RETURN(error, {});

  _sourceLayout = MakeTactileTaxelLayout(sourceMeta->positionsMeters, error);
  MOCHI_ERROR_RETURN(error, {});
  _projection = BuildTactileRectProjection(_sourceLayout, ProjectionParams(), error);
  MOCHI_ERROR_RETURN(error, {});

  _inputForces.forces.resize(_sourceLayout.positions.size());
  std::size_t const cells =
      static_cast<std::size_t>(_params.width) * static_cast<std::size_t>(_params.height);
  _grid.forces.resize(cells);
  _grid.taxelCounts.resize(cells);
  _syncOutput.resize(3 * cells);

  SensorFormat format{
      .kind = DynamicString{kSensorFormatKindForce3D},
      .shape = {_params.height, _params.width, 3},
      .axes =
          {
              {.name = "row", .labels = {}, .units = {}},
              {.name = "column", .labels = {}, .units = {}},
              MakeForceComponentAxis(),
          },
  };

  _cellToSourceVertex = MakeCellToSourceVertex(_projection);
  _rectMeta.rowAxis = 0;
  _rectMeta.columnAxis = 1;
  _rectMeta.componentAxis = 2;
  _rectMeta.centerMeters = _params.center;
  _rectMeta.tipDirection = _params.tipDirection;
  _rectMeta.zeroLongitudeDirection = _params.zeroLongitudeDirection;
  _rectMeta.maxPolarLatitudeRadians = _params.maxPolarLatitude;
  _rectMeta.cellToSourceVertex = MakeConstSpan(_cellToSourceVertex);
  return format;
}

void SensorSeedRectMap::ReadInput(SensorSample const& source, Error& error) {
  MOCHI_ERROR_RETURN(error);
  Span<float const> const values = source.Floats();
  MOCHI_ERROR_IF_NOT(
      values.size() == 3 * _inputForces.forces.size(),
      error,
      "SensorSeedRectMap input size changed after graph build");
  MOCHI_ERROR_RETURN(error);
  for (std::size_t vertex = 0; vertex < _inputForces.forces.size(); ++vertex) {
    for (std::size_t component = 0; component < 3; ++component) {
      _inputForces.forces[vertex][component] = values[3 * vertex + component];
    }
  }
}

void SensorSeedRectMap::ComputeInto(float* output, Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (output == nullptr) {
    MOCHI_ERROR_SET(error, "SensorSeedRectMap output is null");
    return;
  }
  RasterizeTactileRectProjectionInto(_projection, _inputForces, _grid, error);
  MOCHI_ERROR_RETURN(error);
  for (std::size_t cell = 0; cell < _grid.forces.size(); ++cell) {
    for (std::size_t component = 0; component < 3; ++component) {
      output[3 * cell + component] = static_cast<float>(_grid.forces[cell][component]);
    }
  }
}

SensorStepResult SensorSeedRectMap::Step(StepTime const& now, Error& error) {
  Span<SensorBase const* const> const inputs = Inputs();
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  MOCHI_ERROR_IF_NOT(inputs.size() == 1, error, "SensorSeedRectMap requires one input");
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  SensorSample const source = inputs[0]->Latest();
  if (source.captureStep < 0 || source.captureStep == Latest().captureStep) {
    return SensorStepResult::NoUpdate;
  }
  ReadInput(source, error);
  ComputeInto(_syncOutput.data(), error);
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  PublishSample(now, Span<float const>{_syncOutput}, error, &source);
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  return SensorStepResult::Published;
}

Span<std::byte const> SensorSeedRectMap::TryGetMetadata(std::string_view key) const {
  if (_sourceLayout.positions.empty()) {
    return {};
  }
  if (key == SensorMetadataKey<RectMapMeta>::Key) {
    return {reinterpret_cast<std::byte const*>(&_rectMeta), sizeof(_rectMeta)};
  }
  Span<SensorBase const* const> const inputs = Inputs();
  return inputs.size() == 1 ? inputs[0]->TryGetMetadata(key) : Span<std::byte const>{};
}
