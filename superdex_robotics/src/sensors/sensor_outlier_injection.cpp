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

#include <superdex_robotics/sensors/sensor_outlier_injection.h>

#include <superdex_robotics/domain_randomization/derive.h>
#include <superdex_robotics/domain_randomization/random.h>
#include <superdex_robotics/utils/file_utils.h>

#include <mochi_core/utils/quaternion_utils.h>

#include <array>
#include <cmath>

using namespace mochi;
using namespace superdex::robotics;

namespace {

/* Rows 2 and 3 of a uniformly random rotation matrix, row-major. Any two rows of a uniformly
 * random rotation are distributed alike. */
void WriteRandomRotationRows(uint64_t& state, Span<float> out) {
  std::array<double, 4> const q = UniformQuat(state);
  Matrix3x3r const m = ToMatrix3x3(Quaternion(
      static_cast<real>(q[0]),
      static_cast<real>(q[1]),
      static_cast<real>(q[2]),
      static_cast<real>(q[3])));
  for (int r = 1; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      out[3 * (r - 1) + c] = static_cast<float>(m[r][c]);
    }
  }
}

} // namespace

SensorOutlierInjection::SensorOutlierInjection(
    Actor* actor,
    std::string_view paramArgs,
    Error& error)
    : SensorBase(actor, error) {
  MOCHI_ERROR_RETURN(error);
  Params const params = LoadParamsFromPathOrJson<Params>(paramArgs, error, /*allowPartial=*/false);
  SetParams(params, error);
}

SensorOutlierInjection::SensorOutlierInjection(Params const& params, Error& error)
    : SensorBase(error) {
  SetParams(params, error);
}

void SensorOutlierInjection::SetParams(Params const& params, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF_NOT(
      params.probability >= 0.0 && params.probability <= 1.0 && std::isfinite(params.range) &&
          params.range >= 0.0,
      error,
      "SensorOutlierInjection needs a probability in [0, 1] and a finite nonnegative range");
  MOCHI_ERROR_IF(
      params.random.processKey.empty(),
      error,
      "SensorOutlierInjection requires a nonempty processKey");
  MOCHI_ERROR_IF(
      params.rotation && !_outlier.empty() && _outlier.size() != 6,
      error,
      "SensorOutlierInjection replaces rotations of 6 values only");
  MOCHI_ERROR_RETURN(error);
  _params = params;
}

SensorFormat SensorOutlierInjection::InitSensor(SensorScope const& /*scope*/, Error& error) {
  MOCHI_ERROR_IF_NOT(
      Inputs().size() == 1 && Inputs()[0]->GetFormat().dtype == SensorDType::Float32,
      error,
      "SensorOutlierInjection requires one Float32 input");
  MOCHI_ERROR_RETURN(error, {});
  auto const count = static_cast<std::size_t>(Inputs()[0]->Latest().elementCount);
  MOCHI_ERROR_IF(
      _params.rotation && count != 6,
      error,
      "SensorOutlierInjection replaces rotations of 6 values only");
  MOCHI_ERROR_RETURN(error, {});
  _outlier.resize(count);
  ResetSensor();
  return Inputs()[0]->GetFormat();
}

SensorStepResult SensorOutlierInjection::Step(StepTime const& now, Error& error) {
  MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
  SensorSample const source = Inputs()[0]->Latest();
  if (source.captureStep < 0 || source.captureStep == _lastCaptureStep) {
    return SensorStepResult::NoUpdate;
  }
  uint64_t state = Derive(
      _params.random.episodeSeed,
      _params.random.processKey,
      static_cast<uint64_t>(source.captureStep));
  bool const outlier = Uniform(state, 0.0, 1.0) < _params.probability;
  if (outlier) {
    if (_params.rotation) {
      WriteRandomRotationRows(state, MakeSpan(_outlier));
    } else {
      for (float& value : _outlier) {
        value = static_cast<float>(Uniform(state, -_params.range, _params.range));
      }
    }
  }
  if (!PublishSample(now, outlier ? MakeConstSpan(_outlier) : source.Floats(), error, &source)) {
    return SensorStepResult::NoUpdate;
  }
  _lastCaptureStep = source.captureStep;
  return SensorStepResult::Published;
}

void SensorOutlierInjection::ResetSensor() {
  _lastCaptureStep = -1;
}

Span<std::byte const> SensorOutlierInjection::TryGetMetadata(std::string_view key) const {
  return Inputs().size() == 1 ? Inputs()[0]->TryGetMetadata(key) : Span<std::byte const>{};
}
