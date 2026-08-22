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

#include <superdex_robotics/controllers/controller_basic_osc_pd.h>
#include <superdex_robotics/utils/bot_utils.h>
#include <superdex_robotics/utils/file_utils.h>
#include "mochi_core/linear_algebra/matrix_operations.h"

using namespace mochi;
using namespace superdex::robotics;

namespace {
// Error deadband thresholds to filter out sensor noise and numerical errors
constexpr real kPositionErrorDeadband = 1e-4_r; // [m] - Position tracking noise floor
constexpr real kRotationErrorDeadband = 5e-3_r; // [rad] - Rotation tracking noise floor

// A Free root joint contributes this many actor DOFs (3 translation + 3 rotation). Bot space
// always excludes them, so it is the offset between an actor-space and a bot-space DOF index.
// Any other root joint type contributes none.
constexpr int kFreeRootNumDofs = 6;
} // namespace

/* Implementation */

ControllerBasicOscPd::ControllerBasicOscPd(BotPrefab const* prefab, Actor* robot, Error& error)
    : ControllerBase(prefab, robot, error) {
  MOCHI_ERROR_RETURN(error);

  /* Actor-dependent setup runs only when a live sim Actor is provided; this controller can also be
   * constructed without one (e.g. real-robot use with externally pushed observations). */
  if (Actor* const actor = GetActor()) {
    _efforts.Resize(actor->GetNumDofs());
    _efforts.SetZero();
  }
}

void ControllerBasicOscPd::Initialize(
    std::string_view baseLinkName,
    std::string_view eeLinkName,
    Error& error) {
  Actor* const actor = GetActor();
  MOCHI_ERROR_IF(
      actor == nullptr,
      error,
      "ControllerBasicOscPd::Initialize: the actor's scene has been destroyed");
  MOCHI_ERROR_RETURN(error);

  Span<ActorHandle const> const& actors = actor->GetNestedLinkActors(error);
  MOCHI_ERROR_RETURN(error);

  Scene* const scene = GetScene();

  /* Find the base link and EE link by name */
  int BaseLinkIndex = -1;
  int EELinkIndex = -1;
  for (int i = 0; i < isize(actors); ++i) {
    Actor* link = scene->GetActor(actors[i]);
    if (link != nullptr) {
      char const* linkName = link->GetName();
      if (linkName != nullptr) {
        if (baseLinkName == linkName) {
          BaseLinkIndex = i;
        }
        if (eeLinkName == linkName) {
          EELinkIndex = i;
        }
      }
    }
  }

  /* Log all link names on failure for diagnostics */
  if (BaseLinkIndex < 0 || EELinkIndex < 0) {
    MOCHI_LOG_WARNING(
        "Failed to find BaseLink [%.*s] or EELink [%.*s]. Listing all link indices and names:",
        static_cast<int>(baseLinkName.size()),
        baseLinkName.data(),
        static_cast<int>(eeLinkName.size()),
        eeLinkName.data());

    MOCHI_LOG_WARNING("Total nested link actors: %d", isize(actors));
    for (int i = 0; i < isize(actors); ++i) {
      Actor* link = scene->GetActor(actors[i]);
      char const* name = (link != nullptr) ? link->GetName() : "<null>";
      MOCHI_LOG_WARNING("  Link[%d]: %s", i, name ? name : "<unnamed>");
    }
  }

  MOCHI_ERROR_IF(BaseLinkIndex < 0, error, "Could not find base link by name");
  MOCHI_ERROR_IF(EELinkIndex < 0, error, "Could not find EE link by name");
  MOCHI_ERROR_RETURN(error);

  _baseLinkIndex = BaseLinkIndex;
  _baseLinkHandle = actors[BaseLinkIndex];
  _eeLinkIndex = EELinkIndex;
  _eeLinkHandle = actors[EELinkIndex];

  /* Get the shape info for this articulation */
  auto const shapeInfo = actor->GetArticulatedShapeInfo(error);
  MOCHI_ERROR_RETURN(error);

  /* Walk from EELink back through parent chain to verify BaseLink is an ancestor.
   * Ancestor links should always have a lower index value than descendant links. */
  _linkIndices.clear();
  _linkIndices.reserve(actors.size());
  int currentLinkIndex = _eeLinkIndex;
  bool foundBaseLink = false;
  while (currentLinkIndex >= 0) {
    _linkIndices.emplace_back(currentLinkIndex);
    if (currentLinkIndex == _baseLinkIndex) {
      foundBaseLink = true;
      break;
    }
    currentLinkIndex = shapeInfo.parents[currentLinkIndex];
  }
  MOCHI_ERROR_IF(!foundBaseLink, error, "BaseLink must be a direct ancestor of EELink");
  MOCHI_ERROR_RETURN(error);

  /* Reverse so that link indices are in ascending order */
  std::reverse(_linkIndices.begin(), _linkIndices.end());

  /* Get the DOF indices associated with the link indices */
  _dofIndices.clear();
  _dofIndices.reserve(actor->GetNumDofs());
  int iCheck = 0;
  int totalDofCount = 0;
  for (int iLink = 0; iLink <= EELinkIndex; ++iLink) {
    int dofSize = shapeInfo.dofInfo[iLink].GetSize();
    if (iLink == _linkIndices[iCheck]) {
      for (int i = 0; i < dofSize; ++i) {
        _dofIndices.emplace_back(i + totalDofCount);
      }
      iCheck++;
    }
    totalDofCount += dofSize;
  }

  /* Pre-allocate ComputeOutput temporaries now that numArm is known */
  int const numArm = isize(_dofIndices);
  _tmpJArmRoot.Resize(6, numArm);
  _effortsArm.Resize(numArm);

  /* Harvest per-controlled-DOF effort limits from the bot model (if one was provided). These feed
   * the bApplyMaxOSCTorqueNormalization path in ComputeOutput. The controller stays sim-agnostic: a
   * real-robot integrator supplies a BotPrefab built from the robot's URDF.
   * GetEffortLimitsPerDof returns limits in bot space and _dofIndices is actor-space. Bot space
   * always excludes the root joint's DOFs, so the two share an order and differ only by the
   * leading base-DOF offset: 6 for a Free root, 0 for a Hard one. Subtract it before indexing --
   * with a fixed base the offset is zero and this is a no-op, but on a floating base every lookup
   * would otherwise be shifted by 6. */
  _effortLimits.clear();
  if (_prefab != nullptr) {
    DynamicArray<real> const allLimits = GetEffortLimitsPerDof(*_prefab, error);
    MOCHI_ERROR_RETURN(error);
    int const numBaseDofs =
        (!_prefab->joints.empty() && _prefab->joints[0].type == ArticulatedJointType::Free)
        ? kFreeRootNumDofs
        : 0;
    _effortLimits.reserve(numArm);
    for (int i = 0; i < numArm; ++i) {
      int const botDof = _dofIndices[i] - numBaseDofs;
      MOCHI_ERROR_IF(
          botDof < 0 || botDof >= isize(allLimits),
          error,
          "ControllerBasicOscPd: controlled DOF index out of range of the bot's per-DOF effort limits (bot-space vs actor-space DOF mismatch?)");
      MOCHI_ERROR_RETURN(error);
      _effortLimits.emplace_back(allLimits[botDof]);
    }
  }
}

ControllerBasicOscPd::Params const& ControllerBasicOscPd::GetParams() const {
  return _params;
}

void ControllerBasicOscPd::CheckParams(ControllerBasicOscPd::Params const& params, Error& error) {
  if (params.bApplyMaxErrorMagnitudeNormalization) {
    MOCHI_ERROR_IF(
        params.maxTranslationError <= 0_r || !IsFinite(params.maxTranslationError),
        error,
        "ControllerBasicOscPd params.maxTranslationError must be positive and finite when max error magnitude normalization is enabled");
    MOCHI_ERROR_IF(
        params.maxRotationError <= 0_r || !IsFinite(params.maxRotationError),
        error,
        "ControllerBasicOscPd params.maxRotationError must be positive and finite when max error magnitude normalization is enabled");
  }
  /* Effort limits for bApplyMaxOSCTorqueNormalization are harvested from the BotPrefab in
   * Initialize (into _effortLimits) rather than supplied via params, and validated at use in
   * ComputeOutput. */
  MOCHI_ERROR_RETURN(error);
}

void ControllerBasicOscPd::SetParams(ControllerBasicOscPd::Params const& params, Error& error) {
  CheckParams(params, error);
  MOCHI_ERROR_RETURN(error);
  _params = params;
}

void ControllerBasicOscPd::ConfigureFromSceneEntry(
    std::string_view paramArgs,
    std::string_view initArgs,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  auto args = LoadParamsFromPathOrJson<ControllerBasicOscPdInitArgs>(
      initArgs, error, /*allowPartial=*/true);
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(args.baseLinkName.empty(), error, "ControllerBasicOscPd requires a baseLinkName");
  MOCHI_ERROR_IF(args.eeLinkName.empty(), error, "ControllerBasicOscPd requires an eeLinkName");
  MOCHI_ERROR_RETURN(error);
  Initialize(
      QualifiedLinkName(std::string_view(args.baseLinkName.c_str(), args.baseLinkName.size())),
      QualifiedLinkName(std::string_view(args.eeLinkName.c_str(), args.eeLinkName.size())),
      error);
  MOCHI_ERROR_RETURN(error);
  if (!paramArgs.empty()) {
    auto const params = LoadParamsFromPathOrJson<Params>(paramArgs, error);
    MOCHI_ERROR_RETURN(error);
    SetParams(params, error);
    MOCHI_ERROR_RETURN(error);
  }
}

void ControllerBasicOscPd::Reset() {}

Span<int const> ControllerBasicOscPd::GetControlledDofs() {
  return _dofIndices;
}

Span<int const> ControllerBasicOscPd::GetControlledLinks() {
  return _linkIndices;
}

int ControllerBasicOscPd::GetEELinkIndex() const {
  return _eeLinkIndex;
}

Span<real const> ControllerBasicOscPd::ComputeOutput(
    ControllerBasicOscPd::Obsv const& obsv,
    ControllerBasicOscPd::Target const& target,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(_dofIndices.empty(), error, "Initialize() must be called before ComputeOutput");
  MOCHI_ERROR_RETURN(error, {});

  Actor* const actor = GetActor();
  MOCHI_ERROR_IF(
      actor == nullptr,
      error,
      "ControllerBasicOscPd::ComputeOutput: the actor's scene has been destroyed");
  MOCHI_ERROR_RETURN(error, {});

  int const numDofs = actor->GetNumDofs();
  MOCHI_ERROR_IF(
      obsv.dofPositions.size() != numDofs,
      error,
      "ControllerBasicOscPd Obsv positions size does not match robot DOF count");
  MOCHI_ERROR_IF(
      obsv.dofVelocities.size() != numDofs,
      error,
      "ControllerBasicOscPd Obsv velocities size does not match robot DOF count");
  MOCHI_ERROR_IF(
      obsv.eeJacobian.size() != 6 * numDofs,
      error,
      "ControllerBasicOscPd Obsv eeJacobian size does not match 6 * robot DOF count");
  MOCHI_ERROR_RETURN(error, {});

  CheckParams(_params, error);
  MOCHI_ERROR_RETURN(error, {});

  // Extract controlled-DOF positions and velocities from the full-articulation obsv
  _positionsArm.Resize(isize(_dofIndices));
  _velocitiesArm.Resize(isize(_dofIndices));
  for (int i = 0; i < isize(_dofIndices); ++i) {
    _positionsArm(i) = obsv.dofPositions[_dofIndices[i]];
    _velocitiesArm(i) = obsv.dofVelocities[_dofIndices[i]];
  }

  // Extract the 6×numControlledDofs arm Jacobian from the full 6×numDofs EE Jacobian
  _jArm.Resize(6, isize(_dofIndices));
  for (int row = 0; row < 6; ++row) {
    for (int col = 0; col < isize(_dofIndices); ++col) {
      _jArm(row, col) = obsv.eeJacobian[row * numDofs + _dofIndices[col]];
    }
  }

  // Rotate arm Jacobian from world space to root space.
  // The Jacobian rows are defined in world frame; we need them in root frame.
  Matrix3x3r worldFromRootRot = GetRotationMatrix(obsv.worldFromRoot);
  Matrix3x3r rootFromWorldRot = Transpose(worldFromRootRot);

  Matrix<real, 6, 6> rotationBlock6x6;
  rotationBlock6x6.SetZero();
  rotationBlock6x6.Block(0, 0, 3, 3) = AsConstView<real, 3, 3>(rootFromWorldRot);
  rotationBlock6x6.Block(3, 3, 3, 3) = AsConstView<real, 3, 3>(rootFromWorldRot);

  _tmpJArmRoot = rotationBlock6x6 * _jArm;
  _jArm = _tmpJArmRoot;

  _effortsArm.SetZero();

  //******************************************************************************************************************/
  // Apply operational space impedance control to drive the end-effector toward the target.
  // Uses transpose Jacobian method: tau = J^T * F, where F = Kp*error + Kd*vel_error.
  // This maps task-space forces to joint torques without requiring Jacobian inversion.

  TransformRT RootFromEE = Invert(obsv.worldFromRoot) * obsv.worldFromEELink * _params.EELinkFromEE;

  Real3 error_pos = target.rootFromTargetEE.GetTranslation() - RootFromEE.GetTranslation();

  Quaternion quat_error =
      target.rootFromTargetEE.GetRotation() * RootFromEE.GetRotation().GetConjugate();
  // Mochi ToRotationVector() ensures this represents the positive normalized rotations
  Real3 error_rotvec = quat_error.ToRotationVector();

  // Velocity expressed in root frame for proper force transformation
  ColumnVector<real, 6> vel_error = -_jArm * _velocitiesArm;

  if (_params.bApplyMaxErrorMagnitudeNormalization) {
    real const posErrorLimitScalar = Norm(error_pos) / _params.maxTranslationError;
    real const rotErrorLimitScalar = Norm(error_rotvec) / _params.maxRotationError;

    // Renormalize errors to prevent position and rotation from arriving at different times.
    // This maintains a consistent force direction and improves tracking behavior.
    real const limitScalarMax = Max(posErrorLimitScalar, rotErrorLimitScalar);
    if (limitScalarMax > 1_r) {
      MOCHI_LOG_VERBOSE(
          "Max error clamped: [pos: %.4f > %.4f (%.4f)] [rot: %.4f > %.4f (%.4f)]",
          static_cast<float>(Norm(error_pos)),
          static_cast<float>(_params.maxTranslationError),
          static_cast<float>(posErrorLimitScalar),
          static_cast<float>(Norm(error_rotvec)),
          static_cast<float>(_params.maxRotationError),
          static_cast<float>(rotErrorLimitScalar));
      error_pos *= 1_r / limitScalarMax;
      error_rotvec *= 1_r / limitScalarMax;
    }
  }

  // Zero out small errors (deadband) to prevent jitter from sensor noise
  for (int i = 0; i < 3; ++i) {
    if (Abs(error_pos[i]) < kPositionErrorDeadband) {
      error_pos[i] = 0_r;
    }
    if (Abs(error_rotvec[i]) < kRotationErrorDeadband) {
      error_rotvec[i] = 0_r;
    }
  }

  // Combine position and orientation errors into 6D vector
  ColumnVector<real, 6> error_6d;
  for (int i = 0; i < 3; ++i) {
    error_6d(i) = error_pos[i];
    error_6d(i + 3) = error_rotvec[i];
  }

  // Build 6x6 diagonal stiffness and damping matrices
  Matrix<real, 6, 6> k_p;
  Matrix<real, 6, 6> k_d;
  k_p.SetZero();
  k_d.SetZero();
  for (int i = 0; i < 3; ++i) {
    k_p(i, i) = static_cast<real>(_params.Kp_p);
    k_p(i + 3, i + 3) = static_cast<real>(_params.Kp_r);
    k_d(i, i) = static_cast<real>(_params.Kd_p);
    k_d(i + 3, i + 3) = static_cast<real>(_params.Kd_r);
  }

  _effortsArm += _jArm.Transpose() * (k_p * error_6d + k_d * vel_error);

  //******************************************************************************************************************/
  // Scale efforts uniformly so no controlled DOF exceeds its per-joint effort limit (harvested
  // from the BotPrefab into _effortLimits in Initialize), preserving the task-space force
  // direction.

  if (_params.bApplyMaxOSCTorqueNormalization) {
    MOCHI_ERROR_IF(
        isize(_effortLimits) != isize(_dofIndices),
        error,
        "ControllerBasicOscPd: bApplyMaxOSCTorqueNormalization requires the controller to be constructed with a BotPrefab");
    MOCHI_ERROR_RETURN(error, {});
    real maxScale = 1_r;
    for (int i = 0; i < _effortsArm.Rows(); ++i) {
      // Only finite limits (> 0) constrain the scale. Skip unbounded (< 0) and non-actuated (0)
      // DOFs: they impose no finite cap, and an unbounded DOF would contribute a ~0 ratio anyway.
      if (_effortLimits[i] <= 0_r) {
        continue;
      }
      real const scale = Abs(_effortsArm(i) / _effortLimits[i]);
      if (scale > maxScale) {
        maxScale = scale;
      }
    }
    if (maxScale > 1_r) {
      _effortsArm *= 1_r / maxScale;
      MOCHI_LOG_VERBOSE(
          "Effort scaled by %.3f to stay within effort limits\n",
          static_cast<float>(1_r / maxScale));
    }
  }

  //******************************************************************************************************************/
  // Scatter the controlled-DOF efforts back into the full-articulation effort vector.

  _efforts.SetZero();
  for (int iDof = 0; iDof < isize(_dofIndices); ++iDof) {
    _efforts(_dofIndices[iDof]) = _effortsArm(iDof);
  }
  return _efforts;
}

ControllerBasicOscPdParams ControllerBasicOscPdParams::LoadFromFile(
    std::string_view path,
    Error& error) {
  return LoadParamsFromFile<ControllerBasicOscPdParams>(path, error);
}

void ControllerBasicOscPdParams::SaveToFile(std::string_view path, Error& error) const {
  SaveParamsToFile(*this, path, error);
}

ControllerBasicOscPd::Obsv ControllerBasicOscPd::GetCurrentObservationsFromMochi(Error& error) {
  MOCHI_ERROR_IF(
      _eeLinkIndex < 0,
      error,
      "ControllerBasicOscPd::GetCurrentObservationsFromMochi: Initialize() must be called first");
  MOCHI_ERROR_RETURN(error, {});

  Actor* const actor = GetActor();
  MOCHI_ERROR_IF(
      actor == nullptr,
      error,
      "ControllerBasicOscPd::GetCurrentObservationsFromMochi: the actor's scene has been destroyed");
  MOCHI_ERROR_RETURN(error, {});
  Actor* const eeLink = GetScene()->GetActor(_eeLinkHandle);
  MOCHI_ERROR_IF(
      eeLink == nullptr,
      error,
      "ControllerBasicOscPd::GetCurrentObservationsFromMochi: end-effector link actor has been destroyed");
  MOCHI_ERROR_RETURN(error, {});

  Span<real const> eeJacobian = eeLink->GetArticulatedJacobian(error);
  MOCHI_ERROR_RETURN(error, {});

  ControllerBasicOscPd::Obsv obsv;
  obsv.dofPositions.resize(actor->GetNumDofs());
  obsv.dofVelocities.resize(actor->GetNumDofs());
  actor->GetArticulatedPose(obsv.dofPositions, error);
  actor->GetArticulatedJointVelocities(obsv.dofVelocities, error);
  MOCHI_ERROR_RETURN(error, {});
  obsv.worldFromRoot = actor->GetRootTransform();
  obsv.worldFromEELink = eeLink->GetRootTransform();
  obsv.eeJacobian = eeJacobian;
  return obsv;
}
