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

#include <mochi_core/utils/pose_stabilizer.h>

#include <algorithm>

namespace mochi {

static void CopyPose(Span<TransformRT const> pose, Span<TransformRT> outPose) {
  MOCHI_ASSERT_VERBOSE(pose.size() == outPose.size());
  std::ranges::copy(pose, outPose.begin());
}

LastValidPoseStabilizer::LastValidPoseStabilizer(
    real confidenceThreshold,
    Span<TransformRT const> initialPose)
    : _confidenceThreshold(confidenceThreshold),
      _lastValidPose(initialPose.begin(), initialPose.end()) {}

void LastValidPoseStabilizer::Process(
    real confidence,
    Span<TransformRT const> pose,
    Span<TransformRT> outPose) {
  if (confidence < _confidenceThreshold) {
    CopyPose(_lastValidPose, outPose);
  } else {
    CopyPose(pose, outPose);
    CopyPose(pose, _lastValidPose);
  }
}

void LastValidPoseStabilizer::UpdateSettings(PoseStabilizerParams const& params) {
  _confidenceThreshold = params.confidenceThreshold;
}

LinearInterpolationPoseStabilizer::LinearInterpolationPoseStabilizer(
    real confidenceThreshold,
    int interpolationSteps,
    Span<TransformRT const> initialPose)
    : _confidenceThreshold(confidenceThreshold),
      _interpolationSteps(interpolationSteps),
      _lastValidPose(initialPose.begin(), initialPose.end()),
      _lastReferencePose(initialPose.begin(), initialPose.end()) {}

void LinearInterpolationPoseStabilizer::Process(
    real confidence,
    Span<TransformRT const> pose,
    Span<TransformRT> outPose) {
  UpdateInternalState(confidence);

  switch (_stabilizerState) {
    case Unstable:
      CopyPose(_lastValidPose, outPose);
      break;
    case Interpolating:
      ComputeInterpolatedPose(pose, outPose);
      _interpolationCount++;
      CopyPose(outPose, _lastValidPose);
      break;
    case Stable:
      CopyPose(pose, outPose);
      CopyPose(outPose, _lastValidPose);
      break;
    default:
      MOCHI_ASSERT_VERBOSE(false, "Invalid stabilizer state");
      break;
  }
}

void LinearInterpolationPoseStabilizer::UpdateSettings(PoseStabilizerParams const& params) {
  _confidenceThreshold = params.confidenceThreshold;
  _interpolationSteps = params.numInterpolationSteps;
}

void LinearInterpolationPoseStabilizer::UpdateInternalState(real confidence) {
  switch (_stabilizerState) {
    case Unstable:
      if (confidence > _confidenceThreshold) {
        _stabilizerState = Interpolating;
        _interpolationCount = 0;
        UpdateInterpolationReference(_lastValidPose);
      }
      break;
    case Interpolating:
      if (confidence < _confidenceThreshold) {
        _stabilizerState = Unstable;
        _interpolationCount = 0;
      } else if (_interpolationCount >= _interpolationSteps) {
        _stabilizerState = Stable;
      }
      break;
    case Stable:
      if (confidence < _confidenceThreshold) {
        _stabilizerState = Unstable;
      }
      break;
    default:
      MOCHI_ASSERT_VERBOSE(false, "Invalid stabilizer state")
      break;
  }
}

void LinearInterpolationPoseStabilizer::UpdateInterpolationReference(Span<TransformRT const> pose) {
  CopyPose(pose, _lastReferencePose);
}

void LinearInterpolationPoseStabilizer::ComputeInterpolatedPose(
    Span<TransformRT const> pose,
    Span<TransformRT> outPose) const {
  real alpha = real(_interpolationCount + 1) / _interpolationSteps;
  for (auto i = 0; i < pose.size(); ++i) {
    outPose[i] = Interpolate(_lastReferencePose[i], pose[i], alpha);
  }
}

} // namespace mochi
