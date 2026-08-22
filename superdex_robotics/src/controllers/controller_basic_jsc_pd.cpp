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

#include <superdex_robotics/controllers/controller_basic_jsc_pd.h>
#include <superdex_robotics/utils/file_utils.h>

using namespace mochi;
using namespace superdex::robotics;

/* Implementation */

ControllerBasicJscPd::ControllerBasicJscPd(BotPrefab const* prefab, Actor* robot, Error& error)
    : ControllerBase(prefab, robot, error) {
  MOCHI_ERROR_RETURN(error);

  /* Actor-dependent setup runs only when a live sim Actor is provided; this controller can also be
   * constructed without one (e.g. real-robot use with externally pushed observations). */
  if (Actor* const actor = GetActor()) {
    _efforts.Resize(actor->GetNumDofs());
    _efforts.SetZero();
  }
}

ControllerBasicJscPd::Params const& ControllerBasicJscPd::GetParams() const {
  return _params;
}

void ControllerBasicJscPd::CheckParams(ControllerBasicJscPd::Params const& params, Error& error) {
  Actor* const actor = GetActor();
  /* With a live actor the robot's DOF count is authoritative. Without one (real-robot use, where
   * observations are pushed in externally) there is no actor to size against, so fall back to the
   * params' own length -- this still validates that Kp/Kd/saturation/deadband are mutually
   * consistent. */
  int const numDofs = actor != nullptr ? actor->GetNumDofs() : static_cast<int>(params.Kp.size());
  MOCHI_ERROR_IF(
      params.Kp.size() != numDofs,
      error,
      "ControllerBasicJscPd params.Kp does not have the same number of DOFs as the robot");
  MOCHI_ERROR_IF(
      params.Kd.size() != numDofs,
      error,
      "ControllerBasicJscPd params.Kd does not have the same number of DOFs as the robot");
  MOCHI_ERROR_IF(
      params.saturation.size() != numDofs,
      error,
      "ControllerBasicJscPd params.saturation does not have the same number of DOFs as the robot");
  MOCHI_ERROR_IF(
      params.deadband.size() != numDofs,
      error,
      "ControllerBasicJscPd params.deadband does not have the same number of DOFs as the robot");
  MOCHI_ERROR_RETURN(error);
}

void ControllerBasicJscPd::SetParams(ControllerBasicJscPd::Params const& params, Error& error) {
  CheckParams(params, error);
  MOCHI_ERROR_RETURN(error);
  _params = params;
}

void ControllerBasicJscPd::ConfigureFromSceneEntry(
    std::string_view paramArgs,
    std::string_view /*initArgs*/,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  // Joint-space PD needs no init args; just load params if provided.
  if (!paramArgs.empty()) {
    auto const params = LoadParamsFromPathOrJson<Params>(paramArgs, error);
    MOCHI_ERROR_RETURN(error);
    SetParams(params, error);
    MOCHI_ERROR_RETURN(error);
  }
}

Span<real const> ControllerBasicJscPd::ComputeOutput(
    ControllerBasicJscPd::Obsv const& obsv,
    ControllerBasicJscPd::Target const& target,
    Error& error) {
  /* Size against the live actor when present; otherwise (real-robot use) fall back to the params'
   * DOF count, which CheckParams below validates. */
  Actor* const actor = GetActor();
  int const numDofs = actor != nullptr ? actor->GetNumDofs() : static_cast<int>(_params.Kp.size());
  MOCHI_ERROR_IF(
      obsv.dofPositions.size() != numDofs,
      error,
      "ControllerBasicJscPd Obsv positions size does not match robot DOF count");
  MOCHI_ERROR_IF(
      obsv.dofVelocities.size() != numDofs,
      error,
      "ControllerBasicJscPd Obsv velocities size does not match robot DOF count");
  MOCHI_ERROR_IF(
      target.targetPose.size() != numDofs,
      error,
      "ControllerBasicJscPd targetPose does not have the same number of DOFs as the robot");
  MOCHI_ERROR_RETURN(error, {});

  CheckParams(_params, error);
  MOCHI_ERROR_RETURN(error, {});

  // if we have not initialized _prevTargetPose yet, set it equal to targetPose
  if (!_prevTargetPoseInitialized) {
    _prevTargetPose = target.targetPose;
    _prevTargetPoseInitialized = true;
  }

  _efforts.SetZero();
  for (int iDof = 0; iDof < numDofs; ++iDof) {
    real targetVelocity =
        (obsv.dt > 0_r) ? (target.targetPose[iDof] - _prevTargetPose[iDof]) / obsv.dt : 0_r;
    real poseError = target.targetPose[iDof] - obsv.dofPositions[iDof];
    // apply deadband
    real errorSign = Sign(poseError);
    poseError = errorSign * Max(0_r, Abs(poseError) - _params.deadband[iDof]);
    // compute controller efforts
    _efforts(iDof) = _params.Kp[iDof] * poseError +
        _params.Kd[iDof] * (targetVelocity - obsv.dofVelocities[iDof]);
    // apply saturation (negative values mean no saturation / infinity)
    if (_params.saturation[iDof] >= 0) {
      _efforts(iDof) = Clamp(_efforts(iDof), -_params.saturation[iDof], _params.saturation[iDof]);
    }
  }

  // copy out our targetPose so we can compute a delta
  _prevTargetPose = target.targetPose;

  return _efforts;
}

ControllerBasicJscPd::Obsv ControllerBasicJscPd::GetCurrentObservationsFromMochi(Error& error) {
  /* This path reads state from the live mochi actor, so unlike ComputeOutput it cannot fall back to
   * externally-pushed observations: fail cleanly if the actor is gone (e.g. its scene was already
   * destroyed) rather than dereferencing null. */
  Actor* const actor = GetActor();
  MOCHI_ERROR_IF(
      actor == nullptr,
      error,
      "ControllerBasicJscPd::GetCurrentObservationsFromMochi requires a live mochi actor");
  MOCHI_ERROR_RETURN(error, {});
  ControllerBasicJscPd::Obsv obsv;
  obsv.dofPositions.resize(actor->GetNumDofs());
  obsv.dofVelocities.resize(actor->GetNumDofs());
  actor->GetArticulatedPose(obsv.dofPositions, error);
  actor->GetArticulatedJointVelocities(obsv.dofVelocities, error);
  MOCHI_ERROR_RETURN(error, {});
  return obsv;
}

ControllerBasicJscPdParams ControllerBasicJscPdParams::LoadFromFile(
    std::string_view path,
    Error& error) {
  return LoadParamsFromFile<ControllerBasicJscPdParams>(path, error);
}

void ControllerBasicJscPdParams::SaveToFile(std::string_view path, Error& error) const {
  SaveParamsToFile(*this, path, error);
}

void ControllerBasicJscPd::Reset() {
  _prevTargetPose.clear();
  _prevTargetPoseInitialized = false;
}
