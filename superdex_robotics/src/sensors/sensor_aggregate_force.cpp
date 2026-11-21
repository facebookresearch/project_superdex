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

#include <superdex_robotics/sensors/interaction_wrench_metadata.h>
#include <superdex_robotics/sensors/sensor_aggregate_force.h>

#include <superdex_robotics/utils/file_utils.h>

#include <array>
#include <cmath>

using namespace mochi;
using namespace superdex::robotics;

SensorAggregateForce::SensorAggregateForce(Actor* actor, std::string_view paramArgs, Error& error)
    : SensorBase(actor, error) {
  auto const params = paramArgs.empty()
      ? Params{}
      : LoadParamsFromPathOrJson<Params>(paramArgs, error, /*allowPartial=*/false);
  Initialize(params, error);
}

SensorAggregateForce::SensorAggregateForce(Params const& params, Error& error) : SensorBase(error) {
  Initialize(params, error);
}

void SensorAggregateForce::Initialize(Params const& params, Error& error) {
  MOCHI_ERROR_RETURN(error);
  _params = params;
}

SensorFormat SensorAggregateForce::InitSensor(SensorScope const& /*scope*/, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  Span<SensorBase const* const> const inputs = Inputs();
  MOCHI_ERROR_IF_NOT(inputs.size() == 1, error, "SensorAggregateForce requires one input");
  MOCHI_ERROR_RETURN(error, {});
  SensorFormat const& source = inputs[0]->GetFormat();
  MOCHI_ERROR_IF_NOT(
      source.kind == kSensorFormatKindForce3D && source.dtype == SensorDType::Float32 &&
          source.shape.size() == 2 && source.shape[0] > 0 && source.shape[1] == 3,
      error,
      "SensorAggregateForce requires a Force3D [N, 3] input");
  MOCHI_ERROR_RETURN(error, {});
  auto const* const interaction = TryGetSensorMetadata<InteractionWrenchMeta>(*inputs[0]);
  MOCHI_ERROR_IF(
      interaction == nullptr, error, "SensorAggregateForce requires InteractionWrenchMeta");
  MOCHI_ERROR_RETURN(error, {});
  _interactionMetadata = *interaction;
  return {
      .kind = DynamicString{kSensorFormatKindForce3D},
      .shape = {3},
      .axes = {MakeForceComponentAxis()},
  };
}

Span<std::byte const> SensorAggregateForce::TryGetMetadata(std::string_view key) const {
  if (key == SensorMetadataKey<InteractionWrenchMeta>::Key) {
    return {
        reinterpret_cast<std::byte const*>(&_interactionMetadata), sizeof(_interactionMetadata)};
  }
  return SensorBase::TryGetMetadata(key);
}

SensorStepResult SensorAggregateForce::Step(StepTime const& now, Error& error) {
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  Span<SensorBase const* const> const inputs = Inputs();
  SensorSample const source = inputs[0]->Latest();
  if (source.captureStep < 0 || source.captureStep == Latest().captureStep) {
    return SensorStepResult::NoUpdate;
  }
  Span<float const> const values = source.Floats();
  MOCHI_ERROR_IF_NOT(
      values.size() == 3 * static_cast<std::size_t>(inputs[0]->GetFormat().shape[0]),
      error,
      "SensorAggregateForce input size changed after graph build");
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  Real3 force{};
  for (std::size_t vertex = 0; vertex < values.size(); vertex += 3) {
    for (std::size_t component = 0; component < 3; ++component) {
      MOCHI_ERROR_IF(
          !std::isfinite(values[vertex + component]), error, "Input forces must be finite");
      MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
      force[component] += values[vertex + component];
    }
  }
  MOCHI_ERROR_IF(!IsFinite(force), error, "Aggregate forces must be finite");
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  std::array<float, 3> const output = {
      static_cast<float>(force[0]),
      static_cast<float>(force[1]),
      static_cast<float>(force[2]),
  };
  PublishSample(now, output, error, &source);
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  return SensorStepResult::Published;
}
