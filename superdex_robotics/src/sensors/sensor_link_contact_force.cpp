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

#include <superdex_robotics/sensors/sensor_link_contact_force.h>

#include <sensors/sensor_timing.h>
#include <superdex_robotics/sensors/tactile_observation.h>
#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/integration/integration_utils.h>

#include <array>
#include <cmath>

using namespace mochi;
using namespace superdex::robotics;

SensorLinkContactForce::SensorLinkContactForce(
    Actor* linkActor,
    std::string_view paramArgs,
    Error& error)
    : SensorBase(linkActor, error) {
  Params const params = paramArgs.empty()
      ? Params{}
      : LoadParamsFromPathOrJson<Params>(paramArgs, error, /*allowPartial=*/false);
  Initialize(params, error);
}

SensorLinkContactForce::SensorLinkContactForce(Actor* linkActor, Params const& params, Error& error)
    : SensorBase(linkActor, error) {
  Initialize(params, error);
}

void SensorLinkContactForce::Initialize(Params const& params, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      !std::isfinite(params.sampleRateHz) || params.sampleRateHz < 0_r,
      error,
      "SensorLinkContactForce sampleRateHz must be finite and nonnegative");
  MOCHI_ERROR_RETURN(error);
  if (GetActor() == nullptr) {
    MOCHI_ERROR_SET(error, "SensorLinkContactForce requires a link actor");
    return;
  }
  _params = params;
}

SensorLinkContactForce::~SensorLinkContactForce() {
  Actor* const actor = GetActor();
  if (actor != nullptr && _queryRegistered) {
    actor->CancelQuery(_query);
  }
}

SensorFormat SensorLinkContactForce::InitSensor(SensorScope const& /*scope*/, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF_NOT(Inputs().empty(), error, "SensorLinkContactForce does not accept inputs");
  MOCHI_ERROR_RETURN(error, {});
  Actor* const actor = GetActor();
  if (actor == nullptr || !actor->IsQuerySupported(QueryType::TotalContactForce)) {
    MOCHI_ERROR_SET(error, "SensorLinkContactForce requires a link with total contact-force query");
    return {};
  }
  if (!_queryRegistered) {
    _query = actor->RegisterQuery(QueryType::TotalContactForce, error);
    MOCHI_ERROR_RETURN(error, {});
    _queryRegistered = true;
  }
  return {
      .kind = DynamicString{kSensorFormatKindForce3D},
      .shape = {3},
      .axes = {MakeForceComponentAxis()},
  };
}

void SensorLinkContactForce::ResetSensor() {
  _lastSamplePeriod.reset();
  _sampleEpochTime.reset();
}

SensorStepResult SensorLinkContactForce::Step(StepTime const& now, Error& error) {
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  auto const samplePeriod =
      detail::SensorCapturePeriod(now, _params.sampleRateHz, _lastSamplePeriod, _sampleEpochTime);
  if (!samplePeriod) {
    return SensorStepResult::NoUpdate;
  }
  MOCHI_ERROR_IF_NOT(
      EndsAtLastStage(GetActor()->GetScene()->GetSolverParams().integrationMethod),
      error,
      "SensorLinkContactForce requires an integration method whose step ends at its last stage");
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  Real3 const worldForce = GetActor()->GetContactForceWorld(error);
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  Real3 const force = GetWorldTransform().TransformDirectionInverse(worldForce);
  std::array<float, 3> const output = {
      static_cast<float>(force[0]),
      static_cast<float>(force[1]),
      static_cast<float>(force[2]),
  };
  if (!PublishSample(now, Span<float const>{output}, error)) {
    return SensorStepResult::NoUpdate;
  }
  _lastSamplePeriod = samplePeriod;
  return SensorStepResult::Published;
}

Span<std::byte const> SensorLinkContactForce::TryGetMetadata(std::string_view key) const {
  if (key == SensorMetadataKey<InteractionWrenchMeta>::Key) {
    return {
        reinterpret_cast<std::byte const*>(&_interactionMetadata), sizeof(_interactionMetadata)};
  }
  return SensorBase::TryGetMetadata(key);
}
