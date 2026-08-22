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

#include <superdex_robotics/controllers/controller_mochi_articulated_pose.h>
#include <superdex_robotics/utils/file_utils.h>

#include <mochi_physics/utils/mochi_physics_reflection.generated.h>

using namespace mochi;
using namespace superdex::robotics;

/* Implementation */

ControllerMochiArticulatedPose::~ControllerMochiArticulatedPose() {
  // If the mochi scene (or actor) was destroyed before this controller, GetActor() resolves to
  // nullptr, so this guard skips the pose-controller removal instead of dereferencing a dangling
  // actor.
  Actor* const actor = GetActor();
  if (_poseControllerAdded && actor != nullptr) {
    Error error;
    if (actor->HasArticulatedPoseController(error)) {
      actor->RemoveArticulatedPoseController(error);
    }
    if (!error.IsOK()) {
      MOCHI_LOG_WARNING(
          "ControllerMochiArticulatedPose: failed to remove ArticulatedPoseController during destruction");
    }
  }
}

ControllerMochiArticulatedPose::ControllerMochiArticulatedPose(
    BotPrefab const* prefab,
    Actor* robot,
    Error& error)
    : ControllerBase(prefab, robot, error) {
  MOCHI_ERROR_RETURN(error);
  Actor* const actor = GetActor();
  MOCHI_ERROR_IF(
      actor == nullptr,
      error,
      "ControllerMochiArticulatedPose requires a robot Actor: it drives a live Mochi simulation and cannot run on a real robot");
  MOCHI_ERROR_RETURN(error);

  ArticulatedShapeInfo const shapeInfo = actor->GetArticulatedShapeInfo(error);
  MOCHI_ERROR_IF(
      shapeInfo.dofInfo.empty(),
      error,
      "ControllerMochiArticulatedPose: actor has no articulated joints");
  MOCHI_ERROR_RETURN(error);

  _numLinks = isize(shapeInfo.parents);
  _parentIndices.resize(_numLinks);
  for (int i = 0; i < _numLinks; ++i) {
    _parentIndices[i] = shapeInfo.parents[i];
  }

  _worldFromLinks.resize(_numLinks);

  auto const& rootDofs = shapeInfo.dofInfo[0];
  _rootTransDofsCount = rootDofs.transSize;
  _rootRotDofsCount = rootDofs.rotSize;
  _totalDofCount = actor->GetNumDofs();
  _combinedDofs.resize(_totalDofCount);
}

void ControllerMochiArticulatedPose::Initialize(bool removeExisting, Error& error) {
  /* This controller drives an internal mochi ArticulatedPoseController and cannot run without a
   * live actor (enforced at construction). GetActor() resolves on demand and goes null if the scene
   * (or actor) was destroyed, so guard and fail cleanly rather than dereferencing a dangling actor.
   */
  Actor* const actor = GetActor();
  MOCHI_ERROR_IF(
      actor == nullptr, error, "ControllerMochiArticulatedPose::Initialize requires a live actor");
  MOCHI_ERROR_RETURN(error);
  bool const hasExisting = actor->HasArticulatedPoseController(error);
  MOCHI_ERROR_RETURN(error);
  if (hasExisting) {
    MOCHI_ERROR_IF(
        !removeExisting,
        error,
        "ControllerMochiArticulatedPose::Initialize: actor already has an ArticulatedPoseController; pass removeExisting=true to replace it");
    MOCHI_ERROR_RETURN(error);
    actor->RemoveArticulatedPoseController(error);
    MOCHI_ERROR_RETURN(error);
  }

  // Use the user-supplied params if SetParams was called; otherwise fall back to a
  // minimal default (track each link's position and rotation). We track this with an
  // explicit flag rather than inferring intent from empty tracking arrays, so params
  // whose non-tracking fields (gains, limits, ...) were configured are not discarded.
  if (_hasUserParams) {
    actor->AddArticulatedPoseController(_params.poseControllerParams, error);
  } else {
    PoseControllerParams defaults;
    defaults.linkPosTracking = {PoseTrackingParams{}};
    defaults.linkRotTracking = {PoseTrackingParams{}};
    actor->AddArticulatedPoseController(defaults, error);
  }
  MOCHI_ERROR_RETURN(error);
  _poseControllerAdded = true;

  // AddArticulatedPoseController sets target pose and velocity from the actor's current state.
  _needsReset = false;
}

ControllerMochiArticulatedPose::Params const& ControllerMochiArticulatedPose::GetParams() const {
  return _params;
}

void ControllerMochiArticulatedPose::SetParams(
    ControllerMochiArticulatedPose::Params const& params,
    Error& error) {
  _params = params;
  _hasUserParams = true;

  if (_poseControllerAdded) {
    /* A pose controller was added, so an actor existed; it may still have gone away if its scene
     * was destroyed. Guard and fail cleanly rather than dereferencing a dangling actor. */
    Actor* const actor = GetActor();
    MOCHI_ERROR_IF(
        actor == nullptr, error, "ControllerMochiArticulatedPose::SetParams requires a live actor");
    MOCHI_ERROR_RETURN(error);
    actor->SetArticulatedPoseControllerParams(_params.poseControllerParams, error);
    MOCHI_ERROR_RETURN(error);
  }
}

void ControllerMochiArticulatedPose::ConfigureFromSceneEntry(
    std::string_view paramArgs,
    std::string_view /*initArgs*/,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  // Articulated-pose needs no init args. Load params if provided, then add the pose controller to
  // the actor so the controller is ready to receive targets.
  if (!paramArgs.empty()) {
    auto const params = LoadParamsFromPathOrJson<Params>(paramArgs, error);
    MOCHI_ERROR_RETURN(error);
    SetParams(params, error);
    MOCHI_ERROR_RETURN(error);
  }
  Initialize(/*removeExisting=*/true, error);
  MOCHI_ERROR_RETURN(error);
}

Span<real const> ControllerMochiArticulatedPose::ComputeOutput(
    ControllerMochiArticulatedPose::Obsv const& /*obsv*/,
    ControllerMochiArticulatedPose::Target const& target,
    Error& error) {
  MOCHI_ERROR_IF(
      !_poseControllerAdded,
      error,
      "ControllerMochiArticulatedPose: Initialize() must be called before ComputeOutput");
  MOCHI_ERROR_RETURN(error, {});

  Actor* const actor = GetActor();
  MOCHI_ERROR_IF(
      actor == nullptr,
      error,
      "ControllerMochiArticulatedPose::ComputeOutput requires a live actor");
  MOCHI_ERROR_RETURN(error, {});
  bool const hasTransforms = !target.localToParentTransforms.empty();
  bool const hasDofs = !target.poseDofs.empty();
  MOCHI_ERROR_IF(
      hasTransforms == hasDofs,
      error,
      "ControllerMochiArticulatedPose: exactly one of localToParentTransforms or poseDofs must be provided");
  MOCHI_ERROR_RETURN(error, {});

  if (hasTransforms) {
    MOCHI_ERROR_IF(
        isize(target.localToParentTransforms) != _numLinks,
        error,
        "ControllerMochiArticulatedPose localToParentTransforms size does not match link count");
    MOCHI_ERROR_RETURN(error, {});

    for (int i = 0; i < _numLinks; ++i) {
      if (_parentIndices[i] < 0) {
        _worldFromLinks[i] = target.worldFromRoot * target.localToParentTransforms[i];
      } else {
        _worldFromLinks[i] = _worldFromLinks[_parentIndices[i]] * target.localToParentTransforms[i];
      }
    }

    if (_needsReset) {
      actor->ResetArticulatedTargetLinkTransforms(_worldFromLinks, error);
      MOCHI_ERROR_RETURN(error, {});
      _needsReset = false;
    } else {
      actor->SetArticulatedTargetLinkTransforms(_worldFromLinks, error);
    }
  } else {
    int rootDofCount = _rootTransDofsCount + _rootRotDofsCount;
    MOCHI_ERROR_IF(
        isize(target.poseDofs) != _totalDofCount - rootDofCount,
        error,
        "ControllerMochiArticulatedPose poseDofs size does not match non-root DOF count");
    MOCHI_ERROR_RETURN(error, {});
    MOCHI_ERROR_IF(
        (_rootTransDofsCount != 0 && _rootTransDofsCount != 3) ||
            (_rootRotDofsCount != 0 && _rootRotDofsCount != 3),
        error,
        "ControllerMochiArticulatedPose: poseDofs path requires a Free, Spherical, or Hard root joint");
    MOCHI_ERROR_RETURN(error, {});

    int offset = 0;
    if (_rootTransDofsCount == 3) {
      Real3 t = target.worldFromRoot.GetTranslation();
      _combinedDofs[offset++] = t[0];
      _combinedDofs[offset++] = t[1];
      _combinedDofs[offset++] = t[2];
    }
    if (_rootRotDofsCount == 3) {
      Real3 r = target.worldFromRoot.GetRotation().ToRotationVector();
      _combinedDofs[offset++] = r[0];
      _combinedDofs[offset++] = r[1];
      _combinedDofs[offset++] = r[2];
    }
    for (int i = 0; i < isize(target.poseDofs); ++i) {
      _combinedDofs[offset + i] = target.poseDofs[i];
    }

    if (_needsReset) {
      actor->ResetArticulatedTargetPose(_combinedDofs, error);
      MOCHI_ERROR_RETURN(error, {});
      _needsReset = false;
    } else {
      actor->SetArticulatedTargetPose(_combinedDofs, error);
    }
  }
  MOCHI_ERROR_RETURN(error, {});

  return {};
}

ControllerMochiArticulatedPoseParams ControllerMochiArticulatedPoseParams::LoadFromFile(
    std::string_view path,
    Error& error) {
  return LoadParamsFromFile<ControllerMochiArticulatedPoseParams>(
      path, error, /*allowPartial=*/false);
}

void ControllerMochiArticulatedPoseParams::SaveToFile(std::string_view path, Error& error) const {
  SaveParamsToFile(*this, path, error);
}

void ControllerMochiArticulatedPose::Reset() {
  _needsReset = true;
}
