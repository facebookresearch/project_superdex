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

#include <mochi_core/utils/span.h>
#include <mochi_core/utils/transform_rt.h>

#include <algorithm>

namespace superdex::robotics {

/* The floats of a ControllerInputKind::Pose input: x y z, qx qy qz qw. */
constexpr int kPoseInputSize = 7;

/* @brief Writes the tool pose root^-1 * @p eeLink * @p eeLinkFromEE of @p robot as a pose input,
 * its rotation as stored (no sign change). */
inline void WriteToolPoseInput(
    Actor* robot,
    Actor* eeLink,
    TransformRT const& eeLinkFromEE,
    Span<real> input,
    superdex::Error& error) {
  MOCHI_ERROR_IF(isize(input) != kPoseInputSize, error, "A pose input has 7 values");
  MOCHI_ERROR_IF(
      robot == nullptr || eeLink == nullptr,
      error,
      "The robot or its end-effector link has been destroyed");
  MOCHI_ERROR_RETURN(error);
  TransformRT const rootFromTool =
      Invert(robot->GetRootTransform()) * eeLink->GetRootTransform() * eeLinkFromEE;
  Real3 const translation = rootFromTool.GetTranslation();
  Quaternion const& rotation = rootFromTool.GetRotation();
  for (int i = 0; i < 3; ++i) {
    input[i] = translation[i];
  }
  for (int i = 0; i < 4; ++i) {
    input[3 + i] = rotation.data[i];
  }
}

/* @brief The typed path of an OSC for a pose input: the live observations, the input as given
 * (not normalized) as Target::rootFromTargetEE, then ComputeOutput, copied into @p efforts. */
template <typename Osc>
void ComputeOscOutputFromPoseInput(
    Osc& osc,
    Span<real const> input,
    Span<real> efforts,
    superdex::Error& error) {
  MOCHI_ERROR_IF(isize(input) != kPoseInputSize, error, "A pose input has 7 values");
  MOCHI_ERROR_RETURN(error);
  typename Osc::Obsv const obsv = osc.GetCurrentObservationsFromMochi(error);
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      isize(efforts) != isize(obsv.dofPositions), error, "Efforts take one value per actor DOF");
  MOCHI_ERROR_RETURN(error);
  typename Osc::Target target;
  target.rootFromTargetEE = TransformRT{
      Quaternion(input[3], input[4], input[5], input[6]), Real3{input[0], input[1], input[2]}};
  Span<real const> const output = osc.ComputeOutput(obsv, target, error);
  MOCHI_ERROR_RETURN(error);
  std::copy(output.begin(), output.end(), efforts.begin());
}

} // namespace superdex::robotics
