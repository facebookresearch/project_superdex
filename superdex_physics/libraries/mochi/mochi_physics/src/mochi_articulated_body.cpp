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

#include "mochi_articulated_body.h"

#include "mochi_actor_convergence.h"
#include "mochi_blended.h"
#include "mochi_capture.h"
#include "mochi_compound.h"
#include "mochi_constraint.h"
#include "mochi_contact.h"
#include "mochi_contact_filter.h"
#include "mochi_contact_partition.h"
#include "mochi_deformable.h"
#include "mochi_discretization_functions.h"
#include "mochi_ecs_utils.h"
#include "mochi_group.h"
#include "mochi_integration.h"
#include "mochi_island.h"
#include "mochi_pose_controller.h"
#include "mochi_simulation.h"
#include "mochi_skinning.h"
#include "mochi_soft_skinned.h"

#include <mochi_core/contact/contact_partition.h>
#include <mochi_core/contact/dmap.h>
#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/rigid_body_assembly.h>
#include <mochi_core/utils/single_dof_assembly.h>

#include <algorithm>
#include <array>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

using namespace mochi;
using namespace mochi::articulated;
using namespace mochi::dmap;

static constexpr real kDResRegularizationCoefficient = 1e-6_r;

/*
 * Pipeline to resolve current skinning displacements for all nodes, including inactive nodes when
 * subsampling is enabled.
 */
static void ResolveAllNodeSkinningDisplacementsPipeline(
    entt::registry& reg,
    Span<entt::entity const> entities) {
  MOCHI_PROFILE_SCOPE();
  ecs::InvokeForEach(
      &articulated::compound::ResolveSkinning<TimeStep::Current, /* kForceUseAllNodes */ true>,
      reg,
      entities);
}

// Set full pose from internal rigid actors' state
static void SetFullPoseFromBones(
    ecs::PartialRegistry<CRigidState<TimeStep::Current> const> reg,
    CGroupMembers const& groupMembers,
    CArticulatedFullPose& fullPose) {
  MOCHI_PROFILE_SCOPE();
  int offset = 0;
  for (auto actor : groupMembers.actors) {
    auto const& rbState = reg.get<CRigidState<TimeStep::Current> const>(actor).value;
    TransformToRawPose(rbState, fullPose.value.Slice<RigidSize::kAll>(offset, RigidSize::kAll));
    offset += RigidSize::kAll;
  }
}

// Set reduced pose from full pose
static void SetReducedPoseFromFullPose(
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CArticulatedRestTransforms const& restTransforms,
    CRootTransform const& rootTransform,
    CArticulatedParents const& parents,
    CArticulatedFullPose const& fullPose,
    CArticulatedJointTransforms<TimeStep::Current>& outJointTransforms,
    CArticulatedLinkTransforms<TimeStep::Current>& outLinkTransforms,
    CArticulatedReducedPose<TimeStep::Current>& outCurrReducedPose,
    CArticulatedReducedPose<TimeStep::Previous>& outPrevReducedPose) {
  MOCHI_PROFILE_SCOPE();
  auto const* joints = bodyShape.shape->GetJointsData();
  ComputeReducedPose(
      joints->jointTypes,
      joints->jointAxes,
      poseInfo,
      parents,
      restTransforms,
      rootTransform.worldFromLocal,
      fullPose.value,
      outJointTransforms,
      outLinkTransforms,
      outCurrReducedPose.value);
  outPrevReducedPose.value = outCurrReducedPose.value; // Deep copy
}

static void SetFullPoseFromReducedPose(
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CArticulatedParents const& parents,
    CArticulatedRestTransforms const& restTransforms,
    CArticulatedReducedPose<TimeStep::Current> const& reducedPose,
    CRootTransform const& rootTransform,
    CArticulatedJointTransforms<TimeStep::Current>& outJointTransforms,
    CArticulatedLinkTransforms<TimeStep::Current>& outLinkTransforms,
    CArticulatedFullPose& outFullPose) {
  auto const* joints = bodyShape.shape->GetJointsData();
  ComputeFullPose(
      joints->jointTypes,
      joints->jointAxes,
      poseInfo,
      parents,
      restTransforms,
      rootTransform.worldFromLocal,
      reducedPose.value,
      outJointTransforms,
      outLinkTransforms,
      outFullPose.value);
}

// Recompute state derived from the reduced-pose except skinning: full/joint/link transforms,
// (optional) the internal rigid links' state and root transforms, and the Jacobian. Assumes
// CArticulatedReducedPose<Current> is already set. Shared by all external state-setting paths
// (set-pose, set-link-transforms, init, and the post-restore fixup) so they update exactly the same
// derived state. The post-restore fixup does not update links as they are state-captured.
template <bool kUpdateLinks>
static void UpdateDerivedStateFromPose(
    entt::registry& reg,
    entt::entity e,
    ecs::RequiredTag<TagArticulatedActor>,
    [[maybe_unused]] CGroupMembers const& members) {
  ecs::InvokeOnEntity(&SetFullPoseFromReducedPose, reg, e);
  if constexpr (kUpdateLinks) {
    ecs::InvokeForEach(&articulated::rigid::EntitySetSolution, reg, members.actors);
    ecs::InvokeForEach(&mochi::rigid::ComputeRootTransformCurrent, reg, members.actors);
  }
  ecs::InvokeOnEntity(&articulated::compound::UpdateJacobianState<TimeStep::Current>, reg, e);
}

static void SynchronizeAfterExternalPoseChange(entt::registry& reg, entt::entity e) {
  MOCHI_ASSERT_VERBOSE(
      reg.all_of<TagArticulatedActor>(e),
      "SynchronizeAfterExternalPoseChange requires an articulated actor.");

  ecs::InvokeOnEntity<ecs::policy::AllowFullRegistryAccess>(
      UpdateDerivedStateFromPose</*kUpdateLinks*/ true>, reg, e);
  ResolveAllNodeSkinningDisplacementsPipeline(reg, MakeSingletonConstSpan(e));

  if (auto const* composition = reg.try_get<CBlendedComposition const>(e)) {
    skinned::ResolveAllNodeSkinningDisplacementsPipeline(reg, MakeConstSpan(composition->soft));
    blended::ResolveAllNodeBlendingDisplacementsPipeline(reg, MakeSingletonConstSpan(e));

    for (auto const soft : composition->soft) {
      InvalidateActorStepHistory(reg, soft);
    }
  }

  for (auto const link : reg.get<CGroupMembers const>(e).actors) {
    InvalidateActorStepHistory(reg, link);
  }
  InvalidateActorStepHistory(reg, e);
}

void articulated::compound::SetArticulatedPoseFromLinks(entt::registry& reg, entt::entity e) {
  MOCHI_PROFILE_SCOPE();
  // Project the internal rigid links' state onto the reduced pose. This may be lossy if the links
  // are not exactly representable by the joint model.
  ecs::InvokeOnEntity(&SetFullPoseFromBones, reg, e);
  ecs::InvokeOnEntity(&SetReducedPoseFromFullPose, reg, e);

  SynchronizeAfterExternalPoseChange(reg, e);
}

void articulated::compound::SetArticulatedBodyPose(
    entt::registry& reg,
    entt::entity e,
    Span<real const> pose,
    Error& error) {
  MOCHI_ERROR_RETURN(error);

  auto& current = reg.get<CArticulatedReducedPose<TimeStep::Current>>(e).value;
  MOCHI_ERROR_IF(pose.size() != current.size(), error, "Size of pose must be num dofs.");
  MOCHI_ERROR_IF_NOT(IsFinite(pose), error, "Articulated pose values must be finite.");
  MOCHI_ERROR_RETURN(error);

  // Copy the pose to the actor's component
  current = AsConstView(pose);

  SynchronizeAfterExternalPoseChange(reg, e);
}

void articulated::compound::SetArticulatedRootTransform(
    entt::registry& reg,
    entt::entity e,
    TransformRT const& worldFromRoot) {
  // The link world transforms depend on the root transform via forward kinematics.
  reg.get<CRootTransform>(e).worldFromLocal = worldFromRoot;

  SynchronizeAfterExternalPoseChange(reg, e);
}

void articulated::compound::GetLinkTransforms(
    entt::registry& reg,
    entt::entity e,
    Span<TransformRT> outWorldFromLink,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  auto const& members = reg.get<CGroupMembers const>(e);
  MOCHI_ERROR_IF(
      outWorldFromLink.size() != members.actors.size(), error, "Incorrect number of transforms");
  MOCHI_ERROR_RETURN(error);
  for (int i = 0; i < outWorldFromLink.size(); ++i) {
    outWorldFromLink[i] = reg.get<CRootTransform const>(members.actors[i]).worldFromLocal;
  }
}

// Set the velocity of rigid actors
static void SetArticulatedRigidVelocity(
    ColumnVectorView<real const> fullDofsVel,
    ecs::RequiredTag<TagArticulatedLinkActor>,
    CDofOffset const& rigidDofOffset,
    CRigidVel<TimeStep::Current>& outCurrVel) {
  auto vel = fullDofsVel.template MiddleRows<RigidSize::kDAll>(
      rigidDofOffset.dofsOffset, RigidSize::kDAll);
  outCurrVel.value.SetVCom(Load<RigidSize::kDTrans, Vec4r>(&vel[0]));
  outCurrVel.value.SetOmega(Load<RigidSize::kDRot, Vec4r>(&vel[RigidSize::kDTrans]));
}

void articulated::compound::SetArticulatedJointVelocities(
    entt::registry& reg,
    entt::entity e,
    Span<real const> vel,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  auto const& props = reg.get<CArticulatedProps const>(e);
  MOCHI_ERROR_IF(vel.size() != props.reducedDofsDim, error, "Size of vel must be num dofs.");
  MOCHI_ERROR_IF_NOT(IsFinite(vel), error, "Articulated joint velocities must be finite.");
  MOCHI_ERROR_RETURN(error);

  // Set the current velocity of active joints
  auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
  auto& jointVels = reg.get<CArticulatedJointVels<TimeStep::Current>>(e).value;
  for (int i = 0; i < props.numLinks; ++i) {
    auto const& dofInfo = joints->dofInfo[i];
    auto& jointVel = jointVels[i].value;
    switch (joints->jointTypes[i]) {
      case ArticulatedJointType::Free:
        jointVel.SetVCom(Load<3, Vec4r>(&vel[dofInfo.GetTransOffset()]));
        [[fallthrough]];
      case ArticulatedJointType::Spherical: {
        jointVel.SetOmega(Load<3, Vec4r>(&vel[dofInfo.GetRotOffset()]));
      } break;
      case ArticulatedJointType::Revolute: {
        jointVel.SetOmega(vel[dofInfo.GetRotOffset()] * ToSimd(joints->jointAxes[i]));
      } break;
      case ArticulatedJointType::Prismatic:
        jointVel.SetVCom(vel[dofInfo.GetTransOffset()] * ToSimd(joints->jointAxes[i]));
        break;
      default:
        AssertJointTypeCount<6>();
        break;
    }
  }

  // Compute the full-dof velocity. The Jacobian is up to date.
  auto const& jacobian = reg.get<CArticulatedJacobian const>(e);
  ColumnVector<real> velFull = jacobian.value * AsConstView(vel);

  // Update the velocity of the rigid actors within the articulated body.
  ecs::InvokeForEach(
      &SetArticulatedRigidVelocity,
      reg,
      reg.get<CGroupMembers const>(e).actors,
      AsConstView(velFull));
  ecs::InvokeForEach(
      &mochi::rigid::UpdateRigidVelocity_Dynamic, reg, reg.get<CGroupMembers const>(e).actors);

  // External state changes invalidate step history.
  InvalidateActorStepHistory(reg, e);
  for (auto const& link : reg.get<CGroupMembers const>(e).actors) {
    InvalidateActorStepHistory(reg, link);
  }
}

static void RootLinkTransformsToCoMLinkTransforms(
    entt::registry& reg,
    entt::entity e,
    Span<TransformRT const> transformsRoot,
    Span<TransformRT> outTransformsCom) {
  auto const& links = reg.get<CGroupMembers const>(e).actors;
  for (auto i = 0; i < links.size(); ++i) {
    auto comLocal = ToReal3(reg.get<CRigidBodyInertia const>(links[i]).GetCenterOfMassLocal());
    outTransformsCom[i] = TransformRT{
        transformsRoot[i].GetRotation(),
        transformsRoot[i].GetTranslation() + transformsRoot[i].GetRotation() * comLocal};
  }
}

static void ComLinkTransformsToRootLinkTransforms(
    entt::registry const& reg,
    entt::entity e,
    Span<TransformRT const> transformsCom,
    Span<TransformRT> outTransformsRoot) {
  auto const& links = reg.get<CGroupMembers const>(e).actors;
  for (auto i = 0; i < links.size(); ++i) {
    auto comLocal = ToReal3(reg.get<CRigidBodyInertia const>(links[i]).GetCenterOfMassLocal());
    outTransformsRoot[i] = TransformRT{
        transformsCom[i].GetRotation(),
        transformsCom[i].GetTranslation() - transformsCom[i].GetRotation() * comLocal};
  }
}

static void GetPoseFromLinkTransformsCom(
    entt::registry& reg,
    entt::entity e,
    Span<TransformRT const> linkTransformsCom,
    Span<real> outPose) {
  auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
  auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(e);
  auto const& parents = reg.get<CArticulatedParents const>(e);
  auto const& restTransforms = reg.get<CArticulatedRestTransforms const>(e);
  auto const& worldFromRoot = reg.get<CRootTransform const>(e).worldFromLocal;
  auto const& props = reg.get<CArticulatedProps const>(e);
  auto const& links = reg.get<CGroupMembers const>(e).actors;
  MOCHI_ASSERT(linkTransformsCom.size() == links.size(), "Invalid number of link transforms");
  MOCHI_ASSERT(outPose.size() == props.reducedPoseDim, "Invalid pose size");

  // Compute reduced pose (joints) from full pose (links).
  MOCHI_FILO_STACK_ALLOCATOR(allocator, sizeof(TransformRT) * 128); // Allocate 128 transforms
  DynamicArray<TransformRT> jointTransforms(&allocator);
  jointTransforms.resize_noinit(props.numLinks);
  ComputeReducedPoseFromTransforms(
      joints->jointTypes,
      joints->jointAxes,
      poseInfo,
      parents,
      restTransforms,
      worldFromRoot,
      linkTransformsCom,
      jointTransforms,
      AsView(outPose));
}

void articulated::compound::GetLinkTransformsComFromPose(
    entt::registry const& reg,
    entt::entity e,
    Span<real const> pose,
    Span<TransformRT> outLinkTransformsCom) {
  auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
  auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(e);
  auto const& parents = reg.get<CArticulatedParents const>(e);
  auto const& restTransforms = reg.get<CArticulatedRestTransforms const>(e);
  auto const& props = reg.get<CArticulatedProps const>(e);
  auto const& worldFromRoot = reg.get<CRootTransform const>(e).worldFromLocal;
  MOCHI_ASSERT(isize(pose) == props.reducedPoseDim, "Invalid pose size");
  MOCHI_ASSERT(outLinkTransformsCom.size() == parents.size(), "Invalid number of transforms");

  // Compute internal link transforms (with CoM translation) from reduced pose.
  MOCHI_FILO_STACK_ALLOCATOR(allocator, sizeof(TransformRT) * 128); // Allocate 128 transforms
  DynamicArray<TransformRT> jointTransforms(&allocator);
  jointTransforms.resize_noinit(props.numLinks);
  ComputeTransformsFromReducedPose(
      joints->jointTypes,
      joints->jointAxes,
      poseInfo,
      parents,
      restTransforms,
      worldFromRoot,
      AsConstView(pose),
      jointTransforms,
      outLinkTransformsCom);
}

// Copy the current velocity of active joints.
static void GetArticulatedJointVelocitiesImpl(
    CArticulatedProps const& props,
    Span<ArticulatedJointType const> jointTypes,
    Span<Real3 const> jointAxes,
    Span<ArticulatedDofInfo const> jointDofInfo,
    CArticulatedJointVels<TimeStep::Current> const& jointVels,
    ColumnVectorView<real> outVel) {
  for (int i = 0; i < props.numLinks; ++i) {
    auto const& jointVel = jointVels.value[i].value;
    auto const& dofInfo = jointDofInfo[i];
    switch (jointTypes[i]) {
      case ArticulatedJointType::Free:
        Store(&outVel[dofInfo.GetTransOffset()], jointVel.GetVCom());
        [[fallthrough]];
      case ArticulatedJointType::Spherical: {
        Store<RigidSize::kDRot>(&outVel[dofInfo.GetRotOffset()], jointVel.GetOmegaAndVSym().first);
      } break;
      case ArticulatedJointType::Revolute: {
        outVel[dofInfo.GetRotOffset()] =
            Dot(ToReal3(jointVel.GetOmegaAndVSym().first), jointAxes[i]);
      } break;
      case ArticulatedJointType::Prismatic:
        outVel[dofInfo.GetTransOffset()] = Dot(ToReal3(jointVel.GetVCom()), jointAxes[i]);
        break;
      default:
        AssertJointTypeCount<6>();
        break;
    }
  }
}

void articulated::compound::GetArticulatedJointVelocities(
    entt::registry const& reg,
    entt::entity e,
    Span<real> outVel,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  auto const& props = reg.get<CArticulatedProps const>(e);
  MOCHI_ERROR_IF(outVel.size() != props.reducedDofsDim, error, "Size of outVel must be num dofs.");
  MOCHI_ERROR_RETURN(error);

  auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
  auto const& jointVels = reg.get<CArticulatedJointVels<TimeStep::Current> const>(e);
  GetArticulatedJointVelocitiesImpl(
      props, joints->jointTypes, joints->jointAxes, joints->dofInfo, jointVels, AsView(outVel));
}

static void AddConstraints(
    Span<PoseConstraintInfo const> info,
    Span<PoseConstraintImpl const> impl,
    std::vector<PoseConstraintInfo>& outInfo,
    std::vector<PoseConstraintImpl>& outImpl) {
  outInfo.reserve(outInfo.size() + info.size());
  std::copy(info.begin(), info.end(), std::back_inserter(outInfo));
  outImpl.reserve(outImpl.size() + impl.size());
  std::copy(impl.begin(), impl.end(), std::back_inserter(outImpl));
}

static void
ValidatePoseTrackingParams(Span<PoseTrackingParams const> params, int numLinks, Error& error) {
  MOCHI_ERROR_IF(
      !params.empty() && isize(params) != 1 && isize(params) != numLinks,
      error,
      "Invalid number of tracking params");
  for (auto const& p : params) {
    MOCHI_ERROR_IF(
        p.stiffness < 0_r || !IsFinite(p.stiffness),
        error,
        "Tracking stiffness must be non-negative and finite.");
    MOCHI_ERROR_IF(
        p.damping < 0_r || !IsFinite(p.damping),
        error,
        "Tracking damping must be non-negative and finite.");
    MOCHI_ERROR_IF_NOT(IsFinite(p.saturation), error, "Tracking saturation must be finite.");
    MOCHI_ERROR_IF(
        p.saturation == 0_r,
        error,
        "Tracking saturation must not be zero. Use any negative value "
        "to disable saturation or a positive value to enable it.");
  }
}

// Warn when a full per-link jointTracking array assigns non-default gains to a link whose joint is
// Hard. Hard joints get no controller, so those gains are silently ignored. Any other unsupported
// joint configuration raises an error in JointConstraint::Create instead.
// Broadcast (size 1) and empty inputs intentionally apply only to the joints that exist, so they
// are not checked.
static void WarnOnIgnoredJointTrackingParams(
    Span<ArticulatedJointType const> jointTypes,
    DynamicArray<PoseTrackingParams> const& jointTracking,
    int numLinks) {
  if (isize(jointTracking) != numLinks) {
    return;
  }

  PoseTrackingParams const kDefault{};
  for (int link = 0; link < numLinks; ++link) {
    auto const& tp = jointTracking[link];
    if (jointTypes[link] == ArticulatedJointType::Hard && tp != kDefault) {
      MOCHI_LOG_WARNING(
          "jointTracking[%d] sets non-default gains (stiffness=%g, damping=%g, saturation=%g) for a "
          "joint that has no controller; these values are ignored.",
          link,
          static_cast<double>(tp.stiffness),
          static_cast<double>(tp.damping),
          static_cast<double>(tp.saturation));
    }
  }
}

void articulated::compound::AddPoseController(
    entt::registry& reg,
    entt::entity e,
    Scene* scene,
    PoseControllerParams const& params,
    Error& error) {
  MOCHI_ERROR_IF(reg.all_of<CControllerConstraints>(e), error, "Controller already exists.")
  MOCHI_ERROR_RETURN(error);

  // Validate params
  auto const& actors = reg.get<CGroupMembers const>(e).actors;
  int const numLinks = isize(actors);
  ValidatePoseTrackingParams(params.jointTracking, numLinks, error);
  ValidatePoseTrackingParams(params.linkPosTracking, numLinks, error);
  ValidatePoseTrackingParams(params.linkRotTracking, numLinks, error);
  MOCHI_ERROR_RETURN(error);

  // Fetch necessary data
  SceneHandle sceneHandle = reg.ctx<CSceneHandle const>().value;
  auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
  auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(e);
  auto const& parents = reg.get<CArticulatedParents const>(e);
  auto const& props = reg.get<CArticulatedProps const>(e);
  std::vector<ActorHandle> links(actors.size());
  std::transform(actors.begin(), actors.end(), links.begin(), [&](auto e) {
    return GetActorHandle(e, sceneHandle);
  });
  auto const& linkTransformsCom = reg.get<CArticulatedLinkTransforms<TimeStep::Current> const>(e);
  auto const& pose = reg.get<CArticulatedReducedPose<TimeStep::Current> const>(e).value;

  // Container to gather all constraints
  std::vector<PoseConstraintInfo> info;
  std::vector<PoseConstraintImpl> impl;

  // Emplace components to control pose slices
  ActorHandle art = GetActorHandle(e, sceneHandle);
  {
    auto& slice = reg.emplace<CLinkPosController>(
        e,
        scene,
        art,
        links,
        joints->jointTypes,
        joints->jointAxes,
        joints->dofInfo,
        poseInfo,
        parents,
        pose,
        linkTransformsCom,
        params.linkPosTracking);
    AddConstraints(slice.info, slice.impl, info, impl);
  }
  {
    auto& slice = reg.emplace<CLinkRotController>(
        e,
        scene,
        art,
        links,
        joints->jointTypes,
        joints->jointAxes,
        joints->dofInfo,
        poseInfo,
        parents,
        pose,
        linkTransformsCom,
        params.linkRotTracking);
    AddConstraints(slice.info, slice.impl, info, impl);
  }
  {
    auto& slice = reg.emplace<CJointController>(
        e,
        scene,
        art,
        links,
        joints->jointTypes,
        joints->jointAxes,
        joints->dofInfo,
        poseInfo,
        parents,
        pose,
        linkTransformsCom,
        params.jointTracking);
    AddConstraints(slice.info, slice.impl, info, impl);
    WarnOnIgnoredJointTrackingParams(joints->jointTypes, params.jointTracking, isize(links));
  }

  // Emplace component with all pose constraints
  reg.emplace<CControllerConstraints>(e, std::move(info), std::move(impl));

  // Emplace and initialize components to store pose targets
  auto& target =
      reg.emplace<CControllerTarget<TimeStep::Current>>(e, props.reducedPoseDim, props.numLinks);
  target.Set(pose, linkTransformsCom);
  auto& targetOld =
      reg.emplace<CControllerTarget<TimeStep::Previous>>(e, props.reducedPoseDim, props.numLinks);
  targetOld.Set(pose, linkTransformsCom);

  // Emplace component to store velocity target
  reg.emplace<CControllerTargetVelocity>(e, props.reducedPoseDim);

  // If the scene is differentiable, add other components
  if (reg.try_ctx<TagDifferentiableScene>()) {
    InitializeDifferentiablePoseController(reg, e);
  }

  // Initialize the target velocity
  std::vector<real> vel(props.reducedDofsDim);
  GetArticulatedJointVelocities(reg, e, vel, ErrorAssert{});
  SetTargetJointVelocities(reg, e, vel, ErrorAssert{});
}

void articulated::compound::InitializeDifferentiablePoseController(
    entt::registry& reg,
    entt::entity e) {
  // Component to store gradients wrt pose targets.
  auto const& props = reg.get<CArticulatedProps const>(e);
  reg.emplace<CDiffTargetPoseGrad>(e, props.reducedDofsDim);

  // Component to track which forward function owns each controller-target gradient.
  reg.emplace<CTargetOwners>(e);

  // Components for differentiable-input vector indexing
  auto& diffInputInfo = reg.get<CActorDiffInputInfo>(e);
  diffInputInfo.dofsSize = props.reducedDofsDim;
  for (auto link : reg.get<CGroupMembers const>(e).actors) {
    auto& diffInputInfoLink = reg.emplace<CActorDiffInputInfo>(link);
    diffInputInfoLink.dofsSize = RigidSize::kDAll;
    reg.emplace<CDiffInputOffset>(link);
  }

  // Components for the constraints
  for (auto const& c : reg.get<CControllerConstraints>(e).impl) {
    auto constraintEntity = GetEntity(reg, c.constraint->GetHandle(), ErrorAssert{});
    EmplaceConstraintDifferentiabilityComponents(reg, constraintEntity);
  }
}

void articulated::compound::RemoveDifferentiablePoseController(
    entt::registry& reg,
    entt::entity e) {
  if (!reg.try_ctx<TagDifferentiableScene>()) {
    return;
  }

  // Remove/resize components of the articulated actor
  reg.get<CActorDiffInputInfo>(e).dofsSize = 0;
  reg.remove<CDiffTargetPoseGrad, CTargetOwners>(e);

  // Remove components of the links
  for (auto link : reg.get<CGroupMembers const>(e).actors) {
    reg.remove<CActorDiffInputInfo, CDiffInputOffset>(link);
  }

  // Nothing to do for the constraints, as they are fully deleted
}

static void SetControllerTargets(
    entt::registry& reg,
    entt::entity e,
    Span<real const> pose,
    Span<TransformRT const> linkTransformsCom) {
  auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(e);
  if (auto* linkPosController = reg.try_get<CLinkPosController>(e)) {
    controller::SetTargets(pose, linkTransformsCom, poseInfo, *linkPosController);
  }
  if (auto* linkRotController = reg.try_get<CLinkRotController>(e)) {
    controller::SetTargets(pose, linkTransformsCom, poseInfo, *linkRotController);
  }
  if (auto* jointController = reg.try_get<CJointController>(e)) {
    controller::SetTargets(pose, linkTransformsCom, poseInfo, *jointController);
  }
}

static void SetTargetPoseImpl(
    entt::registry& reg,
    entt::entity e,
    Span<real const> pose,
    Span<TransformRT const> linkTransformsCom) {
  // Update the target pose on the constraints
  SetControllerTargets(reg, e, pose, linkTransformsCom);

  // Update the local target pose and link transforms
  auto& target = reg.get<CControllerTarget<TimeStep::Current>>(e);
  target.Set(AsConstView(pose), linkTransformsCom);
}

static void InitializeOldTargetImpl(entt::registry& reg, entt::entity e) {
  // Set the old target pose on the constraints using the stored links
  auto const& targetOld = reg.get<CControllerTarget<TimeStep::Previous> const>(e);
  SetControllerTargets(reg, e, targetOld.JointPose(), targetOld.LinkTransformsCom());

  // Update the internal old target of the constraints
  auto& constraints = reg.get<CControllerConstraints>(e).impl;
  for (auto& c : constraints) {
    c.constraint->UpdateOldTarget(ErrorAssert{});
  }
}

void articulated::compound::GetTargetLinkTransforms(
    entt::registry const& reg,
    entt::entity e,
    Span<TransformRT> outWorldFromTarget,
    Error& error) {
  auto const& props = reg.get<CArticulatedProps const>(e);
  MOCHI_ERROR_IF(
      outWorldFromTarget.size() != props.numLinks, error, "Invalid number of transforms");
  MOCHI_ERROR_RETURN(error);

  // Get the stored target link transforms (CoM frame)
  auto const linksCom = reg.get<CControllerTarget<TimeStep::Current> const>(e).LinkTransformsCom();

  // Convert internal link transforms (with CoM translation) to external link transforms (with root
  // translation).
  ComLinkTransformsToRootLinkTransforms(reg, e, linksCom, outWorldFromTarget);
}

void articulated::compound::SetTargetLinkTransforms(
    entt::registry& reg,
    entt::entity e,
    Span<TransformRT const> worldFromTarget,
    Error& error) {
  auto const& props = reg.get<CArticulatedProps const>(e);
  MOCHI_ERROR_IF(worldFromTarget.size() != props.numLinks, error, "Invalid number of transforms");
  MOCHI_ERROR_RETURN(error);

  // Reserve stack memory for 128 transforms and pose (up to 256 joints with 4 values per joint).
  MOCHI_FILO_STACK_ALLOCATOR(allocator, sizeof(TransformRT) * 128 + sizeof(real) * 4 * 256);

  // Convert external link transforms (with root translation) to internal link transforms (with CoM
  // translation).
  DynamicArray<TransformRT> linkTransformsCom(&allocator);
  linkTransformsCom.resize_noinit(worldFromTarget.size());
  RootLinkTransformsToCoMLinkTransforms(reg, e, worldFromTarget, linkTransformsCom);

  // Obtain the pose from the link transforms
  DynamicArray<real> pose(&allocator);
  pose.resize_noinit(props.reducedPoseDim);
  GetPoseFromLinkTransformsCom(reg, e, linkTransformsCom, pose);

  SetTargetPoseImpl(reg, e, pose, linkTransformsCom);
}

void articulated::compound::CombinePoseAndLinkTargets(
    entt::registry& reg,
    entt::entity e,
    Span<real const> pose,
    Span<bool const> isLinkPosControlled,
    Span<bool const> isLinkRotControlled,
    Span<TransformRT> outLinkTransforms) {
  int const numLinks = isize(outLinkTransforms);
  auto const& parents = reg.get<CArticulatedParents const>(e);
  auto const& restTransforms = reg.get<CArticulatedRestTransforms const>(e);
  auto const& worldFromRoot = reg.get<CRootTransform const>(e).worldFromLocal;
  auto const& links = reg.get<CGroupMembers const>(e).actors;
  auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
  auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(e);

  // If necessary, convert dofs to pose
  auto const& dofInfo = reg.get<CActorDofInfo const>(e);
  MOCHI_FILO_STACK_ALLOCATOR(alloc, sizeof(real) * 4 * 256); // 4 values × 256 joints
  DynamicArray<real> poseContainer(&alloc);
  Span<real const> poseSpan = pose;
  if (dofInfo.poseSize != dofInfo.dofsSize) {
    poseContainer.resize_noinit(dofInfo.poseSize);
    articulated::ConvertDofsToPose(
        joints->jointTypes, joints->dofInfo, poseInfo, AsConstView(pose), AsView(poseContainer));
    poseSpan = poseContainer;
  }

  // Replace each processed entry with its final CoM-frame transform so descendants compose against
  // any merged parent link target. During the loop outLinkTransforms is mixed-frame: entries before
  // i hold final CoM-frame transforms, entries from i onward still hold the caller's root-frame
  // targets. Reading the parent below therefore requires links to be ordered parent-before-child.
  for (int i = 0; i < numLinks; ++i) {
    TransformRT const jointTransform = articulated::ComputeJointTransform(
        AsConstView(poseSpan), joints->jointTypes[i], joints->jointAxes[i], poseInfo[i]);
    TransformRT const parentFromBone =
        restTransforms[i].parentFromOuter * jointTransform * restTransforms[i].innerFromBone;

    int const parentIdx = parents[i];
    MOCHI_ASSERT_VERBOSE(parentIdx < i, "Links must be ordered parent-before-child.");
    TransformRT const& worldFromParent =
        parentIdx == -1 ? worldFromRoot : outLinkTransforms[parentIdx];
    TransformRT const fkTransformCom = NormalizeRotation(worldFromParent * parentFromBone);

    if (!isLinkPosControlled[i] && !isLinkRotControlled[i]) {
      outLinkTransforms[i] = fkTransformCom;
      continue;
    }

    TransformRT const targetTransformRoot = outLinkTransforms[i];
    Real3 const comLocal =
        ToReal3(reg.get<CRigidBodyInertia const>(links[i]).GetCenterOfMassLocal());
    Real3 const fkTranslationRoot =
        fkTransformCom.GetTranslation() - fkTransformCom.GetRotation() * comLocal;

    // Merge partial link controls in root frame. In CoM frame, a nonzero CoM offset would couple a
    // position target to the selected rotation.
    Quaternion const mergedRotation = Normalize(
        isLinkRotControlled[i] ? targetTransformRoot.GetRotation() : fkTransformCom.GetRotation());
    Real3 const mergedTranslationRoot =
        isLinkPosControlled[i] ? targetTransformRoot.GetTranslation() : fkTranslationRoot;
    outLinkTransforms[i] =
        TransformRT{mergedRotation, mergedTranslationRoot + mergedRotation * comLocal};
  }

  // Convert back to root frame
  ComLinkTransformsToRootLinkTransforms(reg, e, outLinkTransforms, outLinkTransforms);
}

void articulated::compound::SetTargetPose(
    entt::registry& reg,
    entt::entity e,
    Span<real const> pose,
    Error& error) {
  auto const& props = reg.get<CArticulatedProps const>(e);
  MOCHI_ERROR_IF(pose.size() != props.reducedPoseDim, error, "Invalid pose size");
  MOCHI_ERROR_RETURN(error);

  // Reserve stack memory for the transforms (up to 256 links).
  MOCHI_FILO_STACK_ALLOCATOR(allocator, sizeof(TransformRT) * 256);
  DynamicArray<TransformRT> linkTransformsCom(&allocator);
  linkTransformsCom.resize_noinit(props.numLinks);
  GetLinkTransformsComFromPose(reg, e, pose, linkTransformsCom);

  SetTargetPoseImpl(reg, e, pose, linkTransformsCom);
}

static void SetZeroTargetJointVelocity(entt::registry& reg, entt::entity e, Error& error) {
  auto const& props = reg.get<CArticulatedProps const>(e);
  // Reserve stack memory for the velocity (up to 256 joint DoFs).
  MOCHI_FILO_STACK_ALLOCATOR(allocator, sizeof(real) * 256);
  ColumnVector<real> velocity(props.reducedDofsDim, &allocator);
  velocity.SetZero();
  articulated::compound::SetTargetJointVelocities(reg, e, velocity, error);
}

void articulated::compound::ResetTargetLinkTransforms(
    entt::registry& reg,
    entt::entity e,
    Span<TransformRT const> worldFromTarget,
    Error& error) {
  articulated::compound::SetTargetLinkTransforms(reg, e, worldFromTarget, error);
  MOCHI_ERROR_RETURN(error);
  SetZeroTargetJointVelocity(reg, e, error);
}

void articulated::compound::ResetTargetPose(
    entt::registry& reg,
    entt::entity e,
    Span<real const> pose,
    Error& error) {
  articulated::compound::SetTargetPose(reg, e, pose, error);
  MOCHI_ERROR_RETURN(error);
  SetZeroTargetJointVelocity(reg, e, error);
}

void articulated::compound::SetTargetJointVelocities(
    entt::registry& reg,
    entt::entity e,
    Span<real const> vel,
    Error& error) {
  auto const& props = reg.get<CArticulatedProps const>(e);
  MOCHI_ERROR_IF(vel.size() != props.reducedDofsDim, error, "Invalid velocity size");
  MOCHI_ERROR_RETURN(error);

  // Store the target velocity for a deferred update once the time step size is known
  auto& target = reg.get<CControllerTargetVelocity>(e);
  target.value.Reset(AsConstView(vel));
  target.use = true;
}

void articulated::compound::SetOldControllerTargets(
    entt::registry& reg,
    entt::entity e,
    ecs::CtxGlobal<CSceneTime const> time,
    CControllerTarget<TimeStep::Current> const& target,
    CControllerTargetVelocity& vel,
    CControllerTarget<TimeStep::Previous>& outTargetOld) {
  if (!vel.use) {
    return; // Nothing to do
  }

  auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
  auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(e);

  // Reserve stack memory for the delta and the result pose
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 2 * 256 * sizeof(real) + 2 * 50 * sizeof(TransformRT));
  ColumnVector<real> deltaJointsLie(vel.value.size(), &allocator);
  ColumnVector<real> jointPoseOld(target.JointPose().size(), &allocator);
  DynamicArray<TransformRT> linkTransforms(&allocator);
  linkTransforms.resize_noinit(target.LinkTransformsCom().size());
  DynamicArray<TransformRT> linkTransformsOld(&allocator);
  linkTransformsOld.resize_noinit(target.LinkTransformsCom().size());

  // Recompute old target as current target - timestep * velocity.
  // For links, apply the FK-derived delta (old / current) to the stored target link transforms.
  deltaJointsLie = (-time->DeltaTime()) * vel.value;
  AddLieDeltaToReducedPose(
      joints->jointTypes,
      joints->dofInfo,
      poseInfo,
      target.JointPose(),
      deltaJointsLie,
      jointPoseOld);
  GetLinkTransformsComFromPose(reg, e, target.JointPose(), linkTransforms);
  GetLinkTransformsComFromPose(reg, e, jointPoseOld, linkTransformsOld);
  for (int i = 0; i < linkTransforms.size(); ++i) {
    linkTransformsOld[i] =
        linkTransformsOld[i] * Invert(linkTransforms[i]) * target.LinkTransformsCom()[i];
  }
  outTargetOld.Set(jointPoseOld, linkTransformsOld);

  // Initialize slices with the old target
  InitializeOldTargetImpl(reg, e);

  // Update the target pose on the constraints
  SetControllerTargets(reg, e, target.JointPose(), target.LinkTransformsCom());

  // Clear velocity target
  vel.use = false;
}

void articulated::compound::GetPoseControllerParams(
    entt::registry& reg,
    entt::entity e,
    PoseControllerParams& outParams,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  auto const& constraints = reg.get<CControllerConstraints const>(e);
  auto const& props = reg.get<CArticulatedProps const>(e);
  int const numLinks = static_cast<int>(props.numLinks);

  // All three constraint types are always present (see AddPoseController) and link-indexed. The
  // caller must pre-size every array to numLinks; we never resize the arrays since they may be
  // owned by another module/DLL.
  MOCHI_ERROR_IF(
      isize(outParams.linkPosTracking) != numLinks,
      error,
      "outParams.linkPosTracking size must equal numLinks");
  MOCHI_ERROR_IF(
      isize(outParams.linkRotTracking) != numLinks,
      error,
      "outParams.linkRotTracking size must equal numLinks");
  MOCHI_ERROR_IF(
      isize(outParams.jointTracking) != numLinks,
      error,
      "outParams.jointTracking size must equal numLinks");
  MOCHI_ERROR_RETURN(error);

  // Links without a controllable joint have no joint constraint, so default their slots to zero
  // gains. Link translation/rotation have a constraint per link and are fully overwritten below.
  std::fill(outParams.jointTracking.begin(), outParams.jointTracking.end(), PoseTrackingParams{});

  for (int i = 0; i < isize(constraints.info); ++i) {
    PoseTrackingParams const tp{
        .stiffness = constraints.impl[i].constraint->GetStiffness(),
        .damping = constraints.impl[i].constraint->GetDamping(),
        .saturation = constraints.impl[i].constraint->GetSaturation(),
    };
    int const link = constraints.info[i].link;
    if (constraints.info[i].type == PoseConstraintType::LinkTranslation) {
      outParams.linkPosTracking[link] = tp;
    } else if (constraints.info[i].type == PoseConstraintType::LinkRotation) {
      outParams.linkRotTracking[link] = tp;
    } else if (constraints.info[i].type == PoseConstraintType::Joint) {
      outParams.jointTracking[link] = tp;
    }
  }
}

// TODO: Currently uses absolute stiffness and damping values. A better approach would be natural
// frequency (ω_n) and damping ratio (ζ), or equivalently ω_n² and 2ζω_n.
void articulated::compound::SetPoseControllerParams(
    entt::registry& reg,
    entt::entity e,
    PoseControllerParams const& params,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  auto& constraints = reg.get<CControllerConstraints>(e);
  auto const& props = reg.get<CArticulatedProps const>(e);
  int const numLinks = static_cast<int>(props.numLinks);

  // All three constraint types are always present (see AddPoseController). An empty array is
  // treated as a single default-constructed PoseTrackingParams (zero gains) broadcast to all
  // links, matching the behavior at controller creation.
  PoseTrackingParams const paramsZero{};
  auto const linkPos = params.linkPosTracking.empty() ? MakeSingletonConstSpan(paramsZero)
                                                      : MakeConstSpan(params.linkPosTracking);
  auto const linkRot = params.linkRotTracking.empty() ? MakeSingletonConstSpan(paramsZero)
                                                      : MakeConstSpan(params.linkRotTracking);
  auto const joint = params.jointTracking.empty() ? MakeSingletonConstSpan(paramsZero)
                                                  : MakeConstSpan(params.jointTracking);

  MOCHI_ERROR_IF(
      isize(linkPos) != 1 && isize(linkPos) != numLinks,
      error,
      "linkPosTracking size must be empty, 1, or numLinks");
  MOCHI_ERROR_IF(
      isize(linkRot) != 1 && isize(linkRot) != numLinks,
      error,
      "linkRotTracking size must be empty, 1, or numLinks");
  MOCHI_ERROR_IF(
      isize(joint) != 1 && isize(joint) != numLinks,
      error,
      "jointTracking size must be empty, 1, or numLinks");
  MOCHI_ERROR_RETURN(error);

  auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
  WarnOnIgnoredJointTrackingParams(joints->jointTypes, params.jointTracking, numLinks);

  for (int i = 0; i < isize(constraints.info); ++i) {
    Span<PoseTrackingParams const> source;
    if (constraints.info[i].type == PoseConstraintType::LinkTranslation) {
      source = linkPos;
    } else if (constraints.info[i].type == PoseConstraintType::LinkRotation) {
      source = linkRot;
    } else if (constraints.info[i].type == PoseConstraintType::Joint) {
      source = joint;
    }

    int const idx = isize(source) == 1 ? 0 : constraints.info[i].link;
    constraints.impl[i].constraint->SetStiffness(source[idx].stiffness, error);
    constraints.impl[i].constraint->SetDamping(source[idx].damping, error);
    constraints.impl[i].constraint->SetSaturation(source[idx].saturation, error);
    MOCHI_ERROR_RETURN(error);
  }
}

Span<real const> articulated::compound::GetPoseControllerForce(
    entt::registry& reg,
    entt::entity e,
    Scene const* scene,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  // TODO: This query should be computed during the simulation step, but it has to be computed
  //       on-demand because the implementation requires the public virtual Scene and Actor
  //       interfaces, which are only available during public API calls. This is an antipattern. The
  //       internals of mochi_physics should use ECS entities and not reach out to the public API
  //       layer.

  auto* query = reg.try_get<CQueryArticulatedControllerForce>(e);
  MOCHI_ERROR_IF(
      !query,
      error,
      "You must register QueryType::ArticulatedControllerForce and step the scene at least once before this data will be available.");
  MOCHI_ERROR_RETURN(error, {});

  // Resize the first time
  if (query->force.empty()) {
    int numDofs = reg.get<CActorDofInfo const>(e).dofsSize;
    query->force.Resize(numDofs);
  }

  SceneHandle sceneHandle = reg.ctx<CSceneHandle const>().value;
  auto const& actors = reg.get<CGroupMembers const>(e).actors;
  std::vector<Actor const*> links(actors.size());
  std::transform(actors.begin(), actors.end(), links.begin(), [&](auto e) {
    return scene->GetActor(GetActorHandle(e, sceneHandle));
  });
  auto const linksSpan = MakeConstSpan(links);

  query->force.SetZero();
  ecs::TryInvokeOnEntity<ecs::policy::AllowMutableExternalParams>(
      &controller::AddForces<LinkPosConstraint>,
      reg,
      e,
      linksSpan,
      AsView(query->force),
      std::ref(error));
  ecs::TryInvokeOnEntity<ecs::policy::AllowMutableExternalParams>(
      &controller::AddForces<LinkRotConstraint>,
      reg,
      e,
      linksSpan,
      AsView(query->force),
      std::ref(error));
  ecs::TryInvokeOnEntity<ecs::policy::AllowMutableExternalParams>(
      &controller::AddForces<JointConstraint>,
      reg,
      e,
      linksSpan,
      AsView(query->force),
      std::ref(error));

  MOCHI_ERROR_RETURN(error, {});
  return query->force;
}

static ArticulatedSkinningData CreateArticulatedSkinningData(
    Span<Real3 const> restCoordsSpan,
    SkinningData const& skinningData,
    SkinningParams const& skinningParams,
    int numReducedDofs) {
  MOCHI_PROFILE_SCOPE();
  int numSkinnedNodes = isize(skinningData.weights) / skinningData.weightsPerNode;

  // Consistency checks for input data
  MOCHI_ASSERT(
      isize(skinningData.weights) == isize(skinningData.indices),
      "Indices and weights have different sizes");

  // Create components for resolution of skinning coordinates
  ColumnVector<real> restCoords =
      ColumnVectorView<real const>(Flatten(restCoordsSpan).data(), numSkinnedNodes * 3);
  Error error;
  auto skinningTransform = skinning::CreateSkinningTransform(skinningData, skinningParams, error);
  MOCHI_ASSERT(error.IsOK(), "Error creating skinning transform");

  // Create components for resolution of skinning Jacobians
  auto jacobianDJoints = CreateJacobianStorage(restCoords.Rows(), numReducedDofs);
  auto jacobianDBones = skinningTransform.CreateDBones();

  // Create and return skinning data component
  SkinningData skinningDataCopy(skinningData);
  return ArticulatedSkinningData{
      .restCoords = std::move(restCoords),
      .skinningData = std::move(skinningDataCopy),
      .skinningTransform = std::move(skinningTransform),
      .jacobianDBones = std::move(jacobianDBones),
      .jacobianDJoints = std::move(jacobianDJoints)};
}

void mochi::InitSkinnedMesh(
    entt::registry& reg,
    entt::entity entity,
    entt::entity articulated,
    std::shared_ptr<Shape const> shape,
    Error& error) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ERROR_RETURN(error);

  MOCHI_ASSERT_VERBOSE(shape, "Skin shape pointer is null.");

  // Resolve the polymorphic skin mesh. Callers (CreateArticulatedActorImpl and
  // InitSoftSkinnedMesh) validate the shape kind upstream, so this is a precondition.
  auto const* tetShape = dynamic_cast<TetrahedralMeshShape const*>(shape.get());
  auto const* triShape = dynamic_cast<TriangularMeshShape const*>(shape.get());
  MOCHI_ASSERT_VERBOSE(
      tetShape || triShape, "Skin shape must be a TetrahedralMeshShape or TriangularMeshShape.");

  std::shared_ptr<SimplicialMesh const> mesh;
  std::shared_ptr<TriangularMesh const> surfaceMesh;
  if (tetShape) {
    mesh = tetShape->GetMesh();
    MOCHI_ERROR_IF_NOT(mesh, error, "Mesh not available");
    MOCHI_ERROR_RETURN(error);
    surfaceMesh = tetShape->GetMesh()->GetBoundaryMesh();
  } else {
    mesh = triShape->GetMesh();
    MOCHI_ERROR_IF_NOT(mesh, error, "Mesh not available");
    MOCHI_ERROR_RETURN(error);
    surfaceMesh = triShape->GetMesh();
  }
  auto const& skinningData = shape->GetMeshSkinning();
  MOCHI_ERROR_IF_NOT(skinningData, error, "Skinning data not available");
  MOCHI_ERROR_RETURN(error);

  // Create mesh and basic transform components if not present
  if (!reg.try_get<CSimplicialMesh>(entity)) {
    reg.emplace<CSimplicialMesh>(entity, mesh);
    reg.emplace<CSurfaceMesh>(entity, surfaceMesh);
    reg.emplace<CShape>(entity, shape);

    std::shared_ptr<TriangularMesh const> visualMesh;
    std::shared_ptr<MeshEmbedding const> visualEmbedding;
    if (tetShape) {
      // Components relevant only for volumetric skin meshes.
      reg.emplace<CTetrahedralMesh>(entity, tetShape->GetMesh());

      visualMesh = tetShape->GetVisualMesh();
      visualEmbedding = tetShape->GetVisualEmbedding();
    } else {
      // Components relevant only for surface skin meshes.
      reg.emplace<CTriangularMesh>(entity, triShape->GetMesh());

      visualMesh = triShape->GetVisualMesh();
      visualEmbedding = triShape->GetVisualEmbedding();
    }

    // Add visual mesh if available
    if (visualMesh && visualEmbedding) {
      reg.emplace<CVisualMesh>(entity, visualMesh, visualEmbedding);
    }
  }

  // Create components to store current and stage-start mesh displacements.
  auto& dispCurr =
      reg.emplace<CDisplacementSlice<real, TimeStep::Current, DisplacementLayer::Skinned>>(
          entity, 3 * mesh->GetNumNodes());
  auto& dispStart =
      reg.emplace<CDisplacementSlice<real, TimeStep::StageStart, DisplacementLayer::Skinned>>(
          entity, 3 * mesh->GetNumNodes());

  // Set the displacement-slice references
  reg.emplace_or_replace<CFinalDisplacementRef<TimeStep::Current>>(entity, dispCurr.value);
  reg.emplace_or_replace<CFinalDisplacementRef<TimeStep::StageStart>>(entity, dispStart.value);

  // Emplace skinning data component. Some data is fetched from the articulated entity, some data
  // from this entity (possibly different).
  SkinningParams skinningParams =
      articulated::compound::CreateSkinningParams(reg, articulated, true, error);
  MOCHI_ERROR_RETURN(error);
  auto const restCoords = mesh->GetNodeCoordinates();
  int numReducedDofs = reg.get<CArticulatedProps const>(articulated).reducedDofsDim;
  reg.emplace<CArticulatedSkinningData>(
      entity,
      CreateArticulatedSkinningData(restCoords, *skinningData, skinningParams, numReducedDofs));
}

int mochi::InitDiscretizationSkinMesh(
    entt::registry& reg,
    entt::entity e,
    Shape const& shape,
    std::optional<BoundarySubsamplingParams> boundarySubsampling,
    ActorBoundaryElementType elementType,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  auto const* tetShape = dynamic_cast<TetrahedralMeshShape const*>(&shape);
  auto const* triShape = dynamic_cast<TriangularMeshShape const*>(&shape);
  MOCHI_ASSERT_VERBOSE(
      tetShape || triShape, "Skin shape must be a TetrahedralMeshShape or TriangularMeshShape.");

  // Validate boundary subsampling parameters once (shared by both mesh-type paths).
  if (boundarySubsampling) {
    MOCHI_ERROR_IF_NOT(
        boundarySubsampling->subsamplingDensity >= 0_r &&
            boundarySubsampling->subsamplingDensity <= 1_r,
        error,
        "Boundary subsampling density must be between 0 and 1.");
    MOCHI_ERROR_IF(
        boundarySubsampling->strategy == BoundarySubsamplingStrategy::Count,
        error,
        "Invalid boundary subsampling strategy.");
    MOCHI_ERROR_RETURN(error, {});
  }

  // Branch on the runtime skin-mesh type.
  if (tetShape) {
    // Emplace low vol discretization.
    // IMPORTANT: It MUST be emplaced because CFemBoundaryDiscretization references it.
    auto const& tetMesh = tetShape->GetMesh();
    auto& femVolDisc = reg.emplace<CFemVolumeDiscretizationP1Q1>(e);
    femVolDisc.femElements.reserve(tetMesh->GetNumElements());
    for (int i = 0; i < tetMesh->GetNumElements(); ++i) {
      femVolDisc.femElements.emplace_back(
          i,
          tetMesh->GetNodeCoordinates(),
          tetMesh->GetElementConnectivity(),
          tetrahedral::kTetrahedralQuadrature1);
    }

    // Emplace the boundary discretization.
    auto& femBoundaryDisc = reg.emplace<CFemBoundaryDiscretization>(
        e, CFemBoundaryDiscretization::Create(*tetMesh, femVolDisc, elementType));

    // If applicable, emplace active boundary faces.
    if (boundarySubsampling && boundarySubsampling->subsamplingDensity < 1_r) {
      auto const* activeBoundaryFaces = &reg.emplace<CActiveBoundaryFaces>(
          e, CreateActiveBoundaryFaces(*boundarySubsampling, *tetShape, femBoundaryDisc));
      // Figure out the active unique nodes
      CActiveVolumeElements activeVolElems(
          tetMesh, Span<int const>{}); // Dummy empty volume elements
      reg.emplace<CActiveUniqueNodes>(e, tetMesh, activeVolElems, *activeBoundaryFaces);
    }

    return femBoundaryDisc.GetNumQuadPoints();
  } else {
    // Tri skin: emplace a CFemSurfaceDiscretization directly. No volumetric components.
    auto const& triMesh = triShape->GetMesh();
    auto& femSurfaceDisc = reg.emplace<CFemSurfaceDiscretization>(
        e, CFemSurfaceDiscretization::Create(elementType, *triMesh));
    // Also emplace the "lite" surface discretization for queries that read it.
    reg.emplace<CFemSurfaceDiscretizationLite>(
        e, CFemSurfaceDiscretizationLite::Create(elementType, *triMesh));

    // If applicable, emplace active boundary faces.
    if (boundarySubsampling && boundarySubsampling->subsamplingDensity < 1_r) {
      reg.emplace<CActiveBoundaryFaces>(
          e, CreateActiveBoundaryFaces(*boundarySubsampling, triMesh, femSurfaceDisc));
    }

    return femSurfaceDisc.GetNumQuadPoints();
  }
}

void mochi::InitCollidingSkinMesh(
    entt::registry& reg,
    entt::entity e,
    entt::entity articulated,
    Shape const& shape,
    Span<ContactPartitionStrategy const> strategies,
    ContactParams const& contactParams,
    int numCollidingSamples) {
  reg.emplace_or_replace<CContactParams>(e, contactParams);

  // Needed for skinned contact assembly.
  reg.emplace<CSkinnedContactSnle>(e);
  reg.emplace<TagSkinnedContact>(e);

  reg.emplace_or_replace<CBoundingVolume<TimeStep::Current>>(
      e, shape.GetBoundingVolume(ErrorAssert{}));
  reg.emplace_or_replace<CBoundingVolume<TimeStep::Previous>>(
      e, shape.GetBoundingVolume(ErrorAssert{}));

  // Other components for collision detection (as colliding object, not as collider)
  deformable::EmplaceContactComponents(reg, e, numCollidingSamples);
  reg.emplace_or_replace<CConservativeStepBounds>(e);

  // Initialize contact partitions
  reg.emplace<CContactPartitions>(e, InitializeContactPartitions(reg, e, articulated, strategies));
}

template <TimeStep kStep>
void articulated::compound::UpdateBounds(
    ecs::RequiredTag<TagCompoundActor>,
    CTriangularMesh const& meshComponent,
    CFinalDisplacementRef<kStep> const& solComponent,
    CBoundingVolume<TimeStep::Current>& outBounds) {
  static_assert(kStep == TimeStep::Current || kStep == TimeStep::StageStart);
  MOCHI_PROFILE_SCOPE();
  auto const& sol = solComponent.value;
  auto nodeCoordinates = meshComponent.mesh->GetNodeCoordinates();
  auto nodeDisplacements = Unflatten<Real3 const>(sol.GetConstSpan());
  outBounds.localShape = GetObb(CalcAabbWithDisplacements(nodeCoordinates, nodeDisplacements));
}

template void articulated::compound::UpdateBounds<TimeStep::Current>(
    ecs::RequiredTag<TagCompoundActor>,
    CTriangularMesh const&,
    CFinalDisplacementRef<TimeStep::Current> const&,
    CBoundingVolume<TimeStep::Current>&);
template void articulated::compound::UpdateBounds<TimeStep::StageStart>(
    ecs::RequiredTag<TagCompoundActor>,
    CTriangularMesh const&,
    CFinalDisplacementRef<TimeStep::StageStart> const&,
    CBoundingVolume<TimeStep::Current>&);

static void InitFullSparsityPattern(
    entt::registry const& reg,
    entt::entity compound,
    CFullSparsityPattern& outFullSparsity,
    CCompoundConstraintSnle* outConstraintSnle) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(reg.all_of<TagCompoundActor>(compound));
  auto const& members = reg.get<CGroupMembers const>(compound);

  // Find the non-zero coordinates from actors
  std::vector<NdArray<int, 2>> totalEntries;
  totalEntries.reserve(4096); // Reduce re-allocation
  for (entt::entity a : members.actors) {
    int actorDofOffset = reg.get<CDofOffset const>(a).dofsOffset;
    if (auto const* reducedSparsity = reg.try_get<CReducedSparsityPattern const>(a)) {
      AppendNonZeroCoordinates(totalEntries, reducedSparsity->graph, actorDofOffset);
    } else {
      auto const& fullSparsity = reg.get<CFullSparsityPattern const>(a);
      AppendNonZeroCoordinates(totalEntries, fullSparsity.graph, actorDofOffset);
    }
  }

  // Add the non-zero coordinates from constraints and update CCompoundConstraintSnle
  if (!members.constraints.empty()) {
    MOCHI_ASSERT(
        outConstraintSnle != nullptr,
        "A compound with constraints should have CCompoundConstraintSnle");

    std::vector<NdArray<int, 2>> constraintEntries;
    constraintEntries.reserve(512); // Reduce reallocation
    for (entt::entity c : members.constraints) {
      auto const& globalResIndices = reg.get<CConstraintGlobalSparsityCache const>(c).resIndices;
      int const numGlobalDofs = isize(globalResIndices);
      constraintEntries.reserve(constraintEntries.size() + Sqr(numGlobalDofs));
      for (int d0 = 0; d0 < numGlobalDofs; ++d0) {
        for (int d1 = 0; d1 < numGlobalDofs; ++d1) {
          constraintEntries.emplace_back(globalResIndices[d0], globalResIndices[d1]);
        }
      }
    }

    Append(totalEntries, constraintEntries);

    auto constraintSparsity = MakeSparsityGraph(std::move(constraintEntries));
    outConstraintSnle->residuals.clear();
    outConstraintSnle->dresiduals.clear();
    outConstraintSnle->residuals.emplace_back(
        /*offset*/ 0, ColumnVector<real>::Zero(isize(constraintSparsity)));
    outConstraintSnle->dresiduals.emplace_back(
        /*rowOffset*/ 0,
        /*colOffset*/ 0,
        SparseMatrix<real>{std::move(constraintSparsity)},
        /*symmetricPair*/ std::nullopt);
  }

  // Store the compound's global sparsity
  outFullSparsity.graph = MakeSparsityGraph(std::move(totalEntries));
}

static void InitSkinMesh(
    entt::registry& reg,
    entt::entity e,
    ArticulatedSkinParams const& params,
    bool useContact,
    std::shared_ptr<Shape const> shape,
    Error& error) {
  MOCHI_ERROR_RETURN(error);

  InitSkinnedMesh(reg, e, e, shape, error);
  MOCHI_ERROR_RETURN(error);

  // If not colliding, we're done
  if (!useContact) {
    return;
  }

  int const numCollidingSamples = InitDiscretizationSkinMesh(
      reg, e, *shape, params.boundarySubsampling, params.boundaryElementType, error);
  MOCHI_ERROR_RETURN(error);

  auto strategy = ContactPartitionStrategy::SkinningDofGroups;
  InitCollidingSkinMesh(
      reg, e, e, *shape, MakeSingletonConstSpan(strategy), params.contact, numCollidingSamples);
}

void mochi::articulated::compound::InitFullDofProblem(entt::registry& reg, entt::entity e) {
  // Every compound has a CDofOffset. In regular compounds, this component refers to DoFs in the
  // island's SNLE problem. However, articulated compounds contain full DoF actors which do not
  // contribute to the island directly. Instead, these full DoF actors are assembled into a local
  // SNLE problem. Therefore, the actors in an articulated compound should start with a DoF offset
  // of zero (relative to the articulation).
  auto const& groupMembers = reg.get<CGroupMembers const>(e);
  mochi::compound::UpdateDofInfo(
      reg,
      e, // compound
      groupMembers,
      0, // Start at zero dofs offset
      0, // Start at zero pose offset
      0, // Start at zero derived state offset (unused by articulated links)
      0 // Start at zero diff input offset (unused by articulated links)
  );

  // An articulated compound can have constraints (e.g. the position & rotation tracking constraints
  // at the wrist of an articulated hand). Such constraints pertain to the full pose and do not get
  // used by the solver directly.
  auto* fullConstraintSnle = reg.try_get<CCompoundConstraintSnle>(e);
  if (fullConstraintSnle) {
    fullConstraintSnle->useInSolver = false;
  }

  // Initialize the full sparsity pattern
  auto& fullSparsity = reg.emplace_or_replace<CFullSparsityPattern>(e);
  InitFullSparsityPattern(reg, e, fullSparsity, fullConstraintSnle);

  // A compound with zero reduced dofs (e.g. an all-Hard/weld skeleton) has nothing to solve. Skip
  // the SNLE machinery entirely instead of running it as a chain of no-ops on a 0x0 problem.
  if (reg.get<CActorDofInfo const>(e).dofsSize == 0) {
    return;
  }

  // Storage for assembly of the compound's SNLE problem.
  if (auto* snle = reg.try_get<CActorSnle>(e)) {
    MOCHI_ASSERT(snle->UseReduced(), "Expected reduced SNLE to be enabled.");
    int const reducedSize = snle->reducedResidual.Rows();
    reg.replace<CActorSnle>(
        e, SparseMatrix<real>{fullSparsity.graph}, Matrix<real>::Zero(reducedSize, reducedSize));
  } else {
    reg.emplace<CActorSnle>(e, SparseMatrix<real>{fullSparsity.graph});
  }

  // Non-linear solver convergence weights (lazily initialized). Preserved across re-invocations of
  // InitFullDofProblem. Nothing this function does invalidates the convergence weights.
  if (!reg.any_of<CActorConvergenceWeights>(e)) {
    reg.emplace<CActorConvergenceWeights>(e);
  }
}

/**
 * Adds freshly-created rigid link actors to a newly-initialized articulated compound.
 *
 * Preconditions:
 * - `compound` was just initialized with `InitCompoundActor`.
 * - Every handle in `links` is valid and refers to a distinct, dynamic rigid actor created by
 *   `CreateArticulatedLinkActorsImpl`.
 * - None of the link actors already belongs to a compound.
 * - Must run before any link is tagged `TagStaticActor`.
 *
 * Violating these preconditions is an internal construction bug, so this helper uses
 * `ErrorAssert` instead of providing transactional rollback.
 */
static void AddLinkActorsToArticulatedCompound(
    entt::registry& reg,
    entt::entity compound,
    Span<ActorHandle const> links) {
  MOCHI_ASSERT(
      (reg.valid(compound) && reg.all_of<TagCompoundActor, CGroupMembers>(compound)),
      "Not a valid compound");

  auto& members = reg.get<CGroupMembers>(compound);
  members.actors.reserve(members.actors.size() + links.size());

  for (auto const link : links) {
    entt::entity actor = GetEntity(reg, link, ErrorAssert{});
    AddActorToCompound(reg, compound, actor, ErrorAssert{});
  }
}

void mochi::articulated::compound::InitArticulatedBodyActor(
    entt::registry& reg,
    entt::entity e,
    ArticulatedActorParams const& params,
    bool useContact,
    std::shared_ptr<ArticulatedBodyShape const> articulatedShapePtr,
    std::shared_ptr<Shape const> skinMeshShape,
    Span<ActorHandle const> bones,
    Error& error) {
  MOCHI_PROFILE_SCOPE();
  if (params.skin.has_value()) {
    ValidateContactParams(params.skin->contact, error);
    MOCHI_ERROR_RETURN(error);
  }

  // An articulated body is a type of compound. Initialize it as a compound first.
  InitCompoundActor(reg, e, error);
  MOCHI_ERROR_RETURN(error);

  // Tag before adopting links so partial-initialization cleanup recognizes the aggregate.
  reg.emplace<TagArticulatedActor>(e);

  AddLinkActorsToArticulatedCompound(reg, e, bones);

  // Update actor info component
  reg.emplace_or_replace<CActorInfo>(e, std::string(params.name), ActorType::Articulated);

  MOCHI_ASSERT(reg.all_of<CDofOffset>(e), "Expected InitCompoundActor to add CDofOffset already");

  // Add root transform
  reg.emplace<CRootTransform>(e, params.worldFromRoot);

  // Add time integrator state component
  reg.emplace<CTimeIntegratorState>(e);

  // Storage for boundary conditions on the articulated compound
  reg.emplace<CDirichletBC<real>>(e);

  // Storage for external forces
  reg.emplace<CExternalForces>(e);

  // Convergence control
  reg.emplace<CConvergenceStatus>(e);

  // Articulated body shape
  reg.emplace<CArticulatedBodyShape>(e, articulatedShapePtr);

  // Fixed transmissions component
  reg.emplace<CTransmissions>(e);

  // Extract articulated parameters from articulated body shape
  auto const* joints = articulatedShapePtr->GetJointsData();
  // Pose-space layout is stored on the actor; dof-space layout lives on the shape's JointsData.
  auto const& poseInfo =
      reg.emplace<CArticulatedJointPoseInfo>(e, articulated::SetupJointPose(joints->jointTypes));
  Span<ArticulatedDofInfo const> const dofInfo = joints->dofInfo;
  ParentIndexArray const& parents = *articulatedShapePtr->GetBoneParents();

  // Add articulated compound properties
  auto const& groupMembers = reg.get<CGroupMembers const>(e);
  auto numActors = isize(groupMembers.actors);
  auto fullPoseDim = numActors * RigidSize::kAll;
  auto fullDofsDim = numActors * RigidSize::kDAll;
  auto reducedPoseDim = GetReducedPoseSize(poseInfo);
  auto reducedDofsDim = GetReducedDofsSize(dofInfo);
  reg.emplace<CArticulatedProps>(
      e,
      ArticulatedProperties{
          .numLinks = uint32_t(numActors),
          .fullDofsDim = uint32_t(fullDofsDim),
          .fullPoseDim = uint32_t(fullPoseDim),
          .reducedDofsDim = uint32_t(reducedDofsDim),
          .reducedPoseDim = uint32_t(reducedPoseDim)});

  // Joint friction and inertia come from the active joints (index-aligned with params.joints); the
  // remaining cycle joints keep default (zero) values.
  int const numJointsTotal = isize(dofInfo);
  std::vector<ArticulatedJointFrictionParams> jointFriction(numJointsTotal);
  std::vector<real> inertiaCoeffs(numJointsTotal, 0_r);
  for (int i = 0; i < isize(params.joints); ++i) {
    jointFriction[i] = params.joints[i].friction;
    inertiaCoeffs[i] = params.joints[i].inertia.value_or(0_r);
  }
  reg.emplace<CArticulatedJointFrictionParams>(e, std::move(jointFriction));
  reg.emplace<CArticulatedInertiaParams>(e, std::move(inertiaCoeffs));

  // Hierarchy info (contains parent for each link and other hierarchical data)
  reg.emplace<CArticulatedParents>(e, parents);

  // Collect bone CoM locals
  DynamicArray<Real3> comLocals;
  comLocals.reserve(numActors);
  for (auto const actor : groupMembers.actors) {
    comLocals.emplace_back(ToReal3(reg.get<CRigidBodyInertia const>(actor).GetCenterOfMassLocal()));
  }

  // Transforms that define the state of the articulated compound. We use the Paired Joint
  // Coordinates from the book Physics-Based Animation, by K. Erleben et al., chapter 2.
  RestTransformArray restTransforms = CreateRestTransforms(
      comLocals,
      articulatedShapePtr->GetJointsData()->jointsChildLinks,
      articulatedShapePtr->GetJointsData()->jointsParentLinks,
      articulatedShapePtr->GetJointsData()->jointFromChildLink,
      articulatedShapePtr->GetJointsData()->parentLinkFromJoint);
  reg.emplace<CArticulatedRestTransforms>(e, std::move(restTransforms));

  // Create storage for reduced pose, full pose and Jacobian
  auto& fullPose = reg.emplace<CArticulatedFullPose>(e, fullPoseDim);
  reg.emplace<CArticulatedReducedPose<TimeStep::Current>>(e, reducedPoseDim);
  reg.emplace<CArticulatedReducedPose<TimeStep::Previous>>(e, reducedPoseDim);
  reg.emplace<CArticulatedReducedPose<TimeStep::StageStart>>(e, reducedPoseDim);
  reg.emplace<CIntegrationArticulatedReducedPose>(e, reducedPoseDim);
  reg.emplace<CArticulatedJacobian>(e, CreateJacobianStorage(fullDofsDim, reducedDofsDim));
  reg.emplace<CArticulatedJointTransforms<TimeStep::Current>>(e, numActors);
  reg.emplace<CArticulatedJointTransforms<TimeStep::StageStart>>(e, numActors);
  reg.emplace<CArticulatedJointVels<TimeStep::Current>>(e, numActors);
  reg.emplace<CArticulatedJointVels<TimeStep::Previous>>(e, numActors);
  reg.emplace<CArticulatedJointVels<TimeStep::StageStart>>(e, numActors);
  reg.emplace<CIntegrationArticulatedJointVels>(e, numActors);
  reg.emplace<CArticulatedLinkTransforms<TimeStep::Current>>(e, numActors);
  reg.emplace<CArticulatedLinkTransforms<TimeStep::StageStart>>(e, numActors);

  // CActorDofInfo needs to report the reduced pose and DOFs, as seen by the island.
  auto& actorDofInfo = reg.get<CActorDofInfo>(e);
  actorDofInfo.poseSize = reducedPoseDim;
  actorDofInfo.dofsSize = reducedDofsDim;

  // Initialize data related to the full DoF problem
  InitFullDofProblem(reg, e);

  // The articulated compound outputs a reduced space residual and dresidual, which has different
  // dimensions from the full space ones. Create a new sparsity pattern based on the reduced space
  // dofs and overwrite compound-level sparsity pattern
  reg.emplace<CReducedSparsityPattern>(e, MakeDenseSparsityGraph(reducedDofsDim, reducedDofsDim));

  // Lean assembly uses one actor residual & matrix for the articulated compound.
  // CActorSnle only exists (initialized in InitFullDofProblem) when the compound has reduced dofs.
  if (reducedDofsDim > 0) {
    reg.get<CActorSnle>(e).EnableReduced(Matrix<real>::Zero(reducedDofsDim, reducedDofsDim));
  } else {
    MOCHI_ASSERT(
        !reg.any_of<CActorSnle>(e), "Articulated actors with no DoFs should not have CActorSnle.");
  }

  // Create vector of internal rigid actors' handles
  SceneHandle sceneHandle = reg.ctx<CSceneHandle const>().value;
  auto& boneHandles = reg.emplace<CBoneHandles>(e);
  boneHandles.bones.reserve(groupMembers.actors.size());
  for (auto actor : groupMembers.actors) {
    boneHandles.bones.push_back(GetActorHandle(actor, sceneHandle));
  }

  // Create internal rigid actors' components
  // This method must be called after the compound has already been initialized
  ReducedDofsMap reducedDofsMap = CreateBonesToReducedDofsMap(parents, dofInfo);
  for (auto a = 0; a < numActors; ++a) {
    reg.emplace<TagArticulatedLinkActor>(groupMembers.actors[a]);
    reg.emplace<CArticulatedEntity>(groupMembers.actors[a], e);
    reg.emplace<CArticulatedFullPoseRef>(groupMembers.actors[a], AsConstView(fullPose.value));
    reg.emplace<CArticulatedRigidJacobian>(
        groupMembers.actors[a],
        CreateJacobianStorage(RigidSize::kDAll, isize(reducedDofsMap.dofs[a])),
        reducedDofsMap.dofs[a]);
    reg.get<CActorSnle>(groupMembers.actors[a]).useInSolver = false;

    // Rigid poses are algebraic variables that depend on the articulated body's DoFs. They are NOT
    // differential variables to be integrated. Remove CIntegrationRigidStates component to prevent
    // potential misuse.
    reg.remove<CIntegrationRigidStates>(groupMembers.actors[a]);
  }

  EmplaceContactLayer(
      reg, e, params.skin.has_value() ? std::string_view(params.skin->layer) : std::string_view{});

  // Set no collider
  auto& collider = reg.emplace<CColliderInfo>(e);
  collider.type = ColliderType::None;

  if (skinMeshShape) {
    MOCHI_ASSERT_VERBOSE(
        params.skin.has_value(), "A skin mesh shape requires skin params to be present.");
    InitSkinMesh(reg, e, *params.skin, useContact, skinMeshShape, error);
    MOCHI_ERROR_RETURN(error);
  }

  // Initialize the reduced pose to the rest configuration.
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 256 * sizeof(real));
  ColumnVector<real> zeroDofs(reducedDofsDim, &allocator);
  zeroDofs.SetZero();
  auto pose = AsView(reg.get<CArticulatedReducedPose<TimeStep::Current>>(e).value);
  articulated::ConvertDofsToPose(joints->jointTypes, joints->dofInfo, poseInfo, zeroDofs, pose);
  reg.get<CArticulatedReducedPose<TimeStep::Previous>>(e).value = pose;

  // Recompute all pose-derived state (transforms, link states, Jacobian) from the reduced pose.
  ecs::InvokeOnEntity<ecs::policy::AllowFullRegistryAccess>(
      UpdateDerivedStateFromPose</*kUpdateLinks*/ true>, reg, e);

  // Resolve skinning so the skinned displacements reflect the actual skeleton pose rather than the
  // rest mesh.
  if (skinMeshShape) {
    ResolveAllNodeSkinningDisplacementsPipeline(reg, MakeSingletonConstSpan(e));
  }

  // Initialize the velocity to specified values or zero otherwise.
  if (params.jointVelocities.has_value() && !params.jointVelocities->empty()) {
    SetArticulatedJointVelocities(reg, e, *params.jointVelocities, error);
  } else {
    std::vector<real> velocity(reducedDofsDim, 0_r);
    SetArticulatedJointVelocities(reg, e, velocity, error);
  }
}

void mochi::articulated::compound::ValidateDifferentiabilitySupport(
    entt::registry const& reg,
    entt::entity e,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ASSERT_VERBOSE(reg.all_of<TagArticulatedActor>(e), "Expected an articulated actor.");
  MOCHI_ERROR_IF(
      reg.any_of<CArticulatedSkinningData>(e),
      error,
      "Differentiable articulated actors do not support skinning.");
}

void mochi::articulated::compound::InitDifferentiableActor(
    entt::registry& reg,
    entt::entity e,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  ValidateDifferentiabilitySupport(reg, e, error);
  MOCHI_ERROR_RETURN(error);

  // The derived state is (conservatively) the union of joint dofs (for joint inertia) and link dofs
  // (for link inertia)
  auto const& props = reg.get<CArticulatedProps>(e);
  int const numDerivedStateDofs = props.reducedDofsDim + props.fullDofsDim;

  // Emplace general components needed for differentiability
  ecs::InvokeOnEntity<ecs::policy::AllowFullRegistryAccess>(
      &EmplaceDifferentiabilityComponents, reg, e, numDerivedStateDofs);

  // Emplace components needed for differentiable contact forces
  ecs::InvokeOnEntity<ecs::policy::AllowFullRegistryAccess>(
      &EmplaceDifferentiableContactComponents, reg, e);

  // If the actor has a pose controller, initialize it too
  if (reg.all_of<CControllerConstraints>(e)) {
    InitializeDifferentiablePoseController(reg, e);
  }
}

void articulated::compound::UpdateDerivedStatePipeline(
    entt::registry& reg,
    Span<entt::entity const> entities) {
  if (entities.empty()) {
    return;
  }
  MOCHI_PROFILE_SCOPE();
  ecs::InvokeForEach(&ResolveSkinning<TimeStep::Current>, reg, entities);
}

template <TimeStep kTimeStep>
void articulated::compound::UpdateJacobiansStatePipeline(
    entt::registry& reg,
    Span<entt::entity const> entities) {
  MOCHI_PROFILE_SCOPE();
  ecs::ParallelInvokeForEach(
      "Evaluate articulated Jacobian on state", &UpdateJacobianState<kTimeStep>, reg, entities);
  ecs::InvokeForEach(&ResolveSkinningJacobianDJoints, reg, entities);
}

#define MOCHI_SPECIALIZE_UPDATE_JACOBIANS_STATE_PIPELINE(kTimeStep)             \
  template void articulated::compound::UpdateJacobiansStatePipeline<kTimeStep>( \
      entt::registry&, Span<entt::entity const>);
MOCHI_SPECIALIZE_UPDATE_JACOBIANS_STATE_PIPELINE(TimeStep::Current);
MOCHI_SPECIALIZE_UPDATE_JACOBIANS_STATE_PIPELINE(TimeStep::StageStart);
#undef MOCHI_SPECIALIZE_UPDATE_JACOBIANS_STATE_PIPELINE

template <TimeStep kTimeStep>
static void UpdateJacobianInput(
    ecs::Included<TagArticulatedActor>,
    CArticulatedParents const& parents,
    CArticulatedRestTransforms const& restTransforms,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CRootTransform const& rootTransform,
    CControllerTarget<kTimeStep> const& controllerTarget,
    CArticulatedJacobian& outJacobian) {
  MOCHI_PROFILE_SCOPE();
  auto const* joints = bodyShape.shape->GetJointsData();

  // Get joint and link transforms from target pose
  MOCHI_FILO_STACK_ALLOCATOR(allocator, sizeof(TransformRT) * 256);
  DynamicArray<TransformRT> jointTransforms(&allocator);
  jointTransforms.resize_noinit(parents.size());
  DynamicArray<TransformRT> linkTransforms(&allocator);
  linkTransforms.resize_noinit(parents.size());
  ComputeTransformsFromReducedPose(
      joints->jointTypes,
      joints->jointAxes,
      poseInfo,
      parents,
      restTransforms,
      rootTransform.worldFromLocal,
      controllerTarget.JointPose(),
      jointTransforms,
      linkTransforms);

  // Compute Jacobian
  Jacobian(
      joints->jointTypes,
      parents,
      joints->jointAxes,
      joints->dofInfo,
      restTransforms,
      rootTransform.worldFromLocal,
      jointTransforms,
      linkTransforms,
      outJacobian.value);
}

template <TimeStep kTimeStep>
void articulated::compound::UpdateJacobiansInputPipeline(
    entt::registry& reg,
    Span<entt::entity const> entities) {
  MOCHI_PROFILE_SCOPE();
  ecs::ParallelInvokeForEach(
      "Evaluate articulated Jacobian on input", &UpdateJacobianInput<kTimeStep>, reg, entities);
}

#define MOCHI_SPECIALIZE_UPDATE_JACOBIANS_INPUT_PIPELINE(kTimeStep)             \
  template void articulated::compound::UpdateJacobiansInputPipeline<kTimeStep>( \
      entt::registry&, Span<entt::entity const>);
MOCHI_SPECIALIZE_UPDATE_JACOBIANS_INPUT_PIPELINE(TimeStep::Current);
MOCHI_SPECIALIZE_UPDATE_JACOBIANS_INPUT_PIPELINE(TimeStep::Previous);
#undef MOCHI_SPECIALIZE_UPDATE_JACOBIANS_INPUT_PIPELINE

// Generic implementation. Specializations follow.
template <TimeStep kStep, bool kForceUseAllNodes>
void articulated::compound::ResolveSkinning(
    CArticulatedLinkTransforms<kStep> const& linkTransforms,
    CArticulatedSkinningData const& skinningData,
    CActiveUniqueNodes const* activeNodes,
    CDisplacementSlice<real, kStep, DisplacementLayer::Skinned>& outDisplacements) {
  MOCHI_PROFILE_SCOPE();
  if (activeNodes && !kForceUseAllNodes) {
    auto rest3 = Unflatten<Real3 const>(MakeConstSpan(skinningData.restCoords));
    auto disp3 = Unflatten<Real3>(MakeSpan(outDisplacements.value));
    Span<int const> activeNodeSpan = activeNodes->ViewIds();
    skinningData.skinningTransform.Transform(
        linkTransforms, skinningData.restCoords, outDisplacements.value, activeNodeSpan);
    for (int iNode : activeNodeSpan) {
      disp3[iNode] -= rest3[iNode];
    }
  } else {
    skinningData.skinningTransform.Transform(
        linkTransforms, skinningData.restCoords, outDisplacements.value);
    outDisplacements.value -= skinningData.restCoords;
  }
}

// Explicit specializations
#define MOCHI_SPECIALIZE_RESOLVE_SKINNING(StepT)               \
  template void articulated::compound::ResolveSkinning<StepT>( \
      CArticulatedLinkTransforms<StepT> const& linkTransforms, \
      CArticulatedSkinningData const& skinningData,            \
      CActiveUniqueNodes const* activeNodes,                   \
      CDisplacementSlice<real, StepT, DisplacementLayer::Skinned>& outDisplacements);
MOCHI_SPECIALIZE_RESOLVE_SKINNING(TimeStep::Current);
MOCHI_SPECIALIZE_RESOLVE_SKINNING(TimeStep::StageStart);
#undef MOCHI_SPECIALIZE_RESOLVE_SKINNING

void articulated::compound::ResolveSkinningJacobianDBones(
    Span<TransformRT const> linkTransforms,
    ColumnVectorView<real const> unposedCoords,
    ArticulatedSkinningData& skinningData,
    CActiveUniqueNodes const* activeNodes) {
  MOCHI_PROFILE_SCOPE();
  // Compute Jacobian DBones
  Span<int const> activeNodesSpan = activeNodes ? activeNodes->ViewIds() : Span<int const>{};
  skinningData.skinningTransform.DTransformDBones(
      linkTransforms, unposedCoords, skinningData.jacobianDBones, activeNodesSpan);
}

void articulated::compound::ResolveSkinningJacobianDJoints(
    CArticulatedLinkTransforms<TimeStep::Current> const& linkTransforms,
    CArticulatedJacobian const& articulatedJacobian,
    CActiveUniqueNodes const* activeNodes,
    CArticulatedSkinningData& skinningData) {
  MOCHI_PROFILE_SCOPE();

  // Compute Jacobian DBones. Then Jacobian DJoints by chain rule with Jacobian DBones/DJoints
  ResolveSkinningJacobianDBones(linkTransforms, skinningData.restCoords, skinningData, activeNodes);
  static_assert(
      krylov::details::MatTraits<decltype(skinningData.jacobianDJoints)>::kMajorDir ==
      krylov::Direction::RowMajor); // Enforced for performance reasons.
  static_assert(
      krylov::details::MatTraits<decltype(articulatedJacobian.value)>::kMajorDir ==
      krylov::Direction::RowMajor); // Enforced for performance reasons.

  if (activeNodes) {
    // TODO: If the % of nodes that are active is large, it may be more efficient to compute the
    // full product like in the "else" branch.
    // TODO(@pabfer): Introduce SparseMatrix::ApplyToSubset to encapsulate this logic inside the
    // class.
    constexpr long long kMinFlopsPerTask = 100000; // ~20 μs @ 5 GFLOP/s
    auto const numTotalVertices = skinningData.jacobianDBones.Rows() / kSpaceDim3;
    auto const minVerticesPerTask = static_cast<int>(
        (kMinFlopsPerTask * numTotalVertices) /
        (2 * skinningData.jacobianDBones.NumNonZeros() * articulatedJacobian.value.Cols()));
    ParallelForEach(
        "DJoints Skinning Jacobian Product",
        activeNodes->ViewIds(),
        Max(1, minVerticesPerTask),
        [&](int vertexId) {
          skinningData.jacobianDBones.ApplyToRange(
              articulatedJacobian.value,
              skinningData.jacobianDJoints,
              vertexId * kSpaceDim3,
              vertexId * kSpaceDim3 + kSpaceDim3);
        });
  } else {
    skinningData.jacobianDJoints = skinningData.jacobianDBones * articulatedJacobian.value;
  }
}

template <typename DiscretizationT>
void articulated::compound::SetupCollidingJacobians(
    ecs::Included<TagArticulatedActor>,
    ecs::Excluded<TagBlendedActor>,
    DiscretizationT const& discretization,
    CDofOffset const& dofOffset,
    CArticulatedSkinningData const& skinningData,
    CContactPartitions const& contactPartitions,
    CCollJacs<CollRole::Colliding>& outJacobians) {
  MOCHI_PROFILE_SCOPE();

  // Reserve stack memory for up to 256 elements.
  MOCHI_FILO_STACK_ALLOCATOR(allocator, sizeof(JacData*) * 256);
  auto jacobiansActive = outJacobians.GetPtrsNonEmpty(&allocator);
  if (jacobiansActive.empty()) {
    return;
  }

  discretization.Visit([&](auto const& discretizationImpl) {
    using ElementT = typename std::decay_t<decltype(discretizationImpl)>::ElementT;
    using DQuad = DMapQuad<ElementT>;
    // Process all Jacobians in parallel
    ParallelForEach(
        "SetupCollidingJacobiansArticulatedCompound", jacobiansActive, 1, [&](JacData* jac) {
          // Create differentiable map
          auto const& dofsVariant =
              contactPartitions[jac->query->collidingPartitionId].GetDofDescriptors()[0];
          auto dofs = MakeConstSpan(std::get<DynamicArray<int>>(dofsVariant));
          DMapSkinNoInput dskinning(0, skinningData.jacobianDJoints, dofs, dofOffset.dofsOffset);
          DQuad dquad(discretizationImpl.femElements, jac->query->jacColliderFromWorld);
          DMap<DQuad, DMapSkinNoInput> dmap(&dquad, &dskinning);

          // Compute Jacobian
          auto& jacs = *jac->jacs;
          dmap.GetJac(jac->query->sampleIndices, jacs);
          jacs[0].CompressIndices();
        });
  });
}

#define MOCHI_ARTICULATED_COMPOUND_SETUP_COLLIDING_JACOBIANS_INST(DISCRETIZATION_TYPE) \
  template void articulated::compound::SetupCollidingJacobians<DISCRETIZATION_TYPE>(   \
      ecs::Included<TagArticulatedActor>,                                              \
      ecs::Excluded<TagBlendedActor>,                                                  \
      DISCRETIZATION_TYPE const&,                                                      \
      CDofOffset const&,                                                               \
      CArticulatedSkinningData const&,                                                 \
      CContactPartitions const&,                                                       \
      CCollJacs<CollRole::Colliding>&);
MOCHI_ARTICULATED_COMPOUND_SETUP_COLLIDING_JACOBIANS_INST(CFemBoundaryDiscretization);
MOCHI_ARTICULATED_COMPOUND_SETUP_COLLIDING_JACOBIANS_INST(CFemSurfaceDiscretization);
#undef MOCHI_ARTICULATED_COMPOUND_SETUP_COLLIDING_JACOBIANS_INST

static void AssembleJointForces(
    AssemblyParams const& params,
    Span<ArticulatedJointType const> jointTypes,
    Span<ArticulatedDofInfo const> jointDofInfo,
    std::function<bool(int, double*, RigidGradient*, RigidHessian*)> const& assemblerRigidBody,
    std::function<bool(int, double*, real*, real*)> const& assemblerSingleDof,
    CActorSnle& outCompoundSnle) {
  MOCHI_ASSERT_VERBOSE(outCompoundSnle.UseReduced(), "Reduced SNLE must be enabled.");
  if (!params.assemObj && !params.assemRes && !params.assemDRes) {
    return;
  }

  // Prepare call arguments
  double tempEnergy{};
  double* energy = params.assemObj ? &tempEnergy : nullptr;
  RigidGradient tempGradient6D;
  RigidGradient* gradient6D = params.assemRes ? &tempGradient6D : nullptr;
  real tempGradient1D{};
  real* gradient1D = params.assemRes ? &tempGradient1D : nullptr;
  RigidHessian tempHessian6D;
  RigidHessian* hessian6D = params.assemDRes ? &tempHessian6D : nullptr;
  real tempHessian1D{};
  real* hessian1D = params.assemDRes ? &tempHessian1D : nullptr;

  // Traverse joints, evaluate and assemble gradient and hessian
  for (int i = 0; i < isize(jointTypes); ++i) {
    auto const type = jointTypes[i];
    auto const& dofInfo = jointDofInfo[i];
    if (type == ArticulatedJointType::Free || type == ArticulatedJointType::Spherical) {
      if (gradient6D) {
        tempGradient6D = {};
      }
      if (hessian6D) {
        tempHessian6D = {};
      }

      if (!assemblerRigidBody(i, energy, gradient6D, hessian6D)) {
        continue;
      }

      auto const offsetDofsRot = dofInfo.GetRotOffset();
      auto const offsetDofsTrans = dofInfo.GetTransOffset();
      if (gradient6D) {
        if (type == ArticulatedJointType::Free) {
          outCompoundSnle.reducedResidual.Slice<RigidSize::kDTrans>(
              offsetDofsTrans, RigidSize::kDTrans) +=
              AsConstView(tempGradient6D).Slice<RigidSize::kDTrans>(0, RigidSize::kDTrans); // force
        }
        outCompoundSnle.reducedResidual.Slice<RigidSize::kDRot>(offsetDofsRot, RigidSize::kDRot) +=
            AsConstView(tempGradient6D)
                .Slice<RigidSize::kDRot>(RigidSize::kDTrans, RigidSize::kDRot); // torque
      }
      if (hessian6D) {
        auto hessianView = RowMatrixView<real const, RigidSize::kDAll, RigidSize::kDAll>(
            Flatten(tempHessian6D).begin());
        auto& dresidual = std::get<Matrix<real>>(outCompoundSnle.reducedDResidual);
        // Mixed hessians force-torque are zero for all existing joint forces.
        if (type == ArticulatedJointType::Free) {
          dresidual.Block<RigidSize::kDTrans, RigidSize::kDTrans>(
              offsetDofsTrans, offsetDofsTrans, RigidSize::kDTrans, RigidSize::kDTrans) +=
              hessianView.Block<RigidSize::kDTrans, RigidSize::kDTrans>(
                  0, 0, RigidSize::kDTrans, RigidSize::kDTrans);
        }
        dresidual.Block<RigidSize::kDRot, RigidSize::kDRot>(
            offsetDofsRot, offsetDofsRot, RigidSize::kDRot, RigidSize::kDRot) +=
            hessianView.Block<RigidSize::kDRot, RigidSize::kDRot>(
                RigidSize::kDTrans, RigidSize::kDTrans, RigidSize::kDRot, RigidSize::kDRot);
      }
    } else if (type == ArticulatedJointType::Revolute || type == ArticulatedJointType::Prismatic) {
      tempGradient1D = {};
      tempHessian1D = {};

      if (!assemblerSingleDof(i, energy, gradient1D, hessian1D)) {
        continue;
      }

      int const offsetDofs = dofInfo.offset;
      if (gradient1D) {
        outCompoundSnle.reducedResidual(offsetDofs) += tempGradient1D;
      }
      if (hessian1D) {
        std::get<Matrix<real>>(outCompoundSnle.reducedDResidual)(offsetDofs, offsetDofs) +=
            tempHessian1D;
      }
    } else {
      AssertJointTypeCount<6>();
    }
  }

  // Assemble energy
  if (energy) {
    outCompoundSnle.objective += tempEnergy;
  }
}

void articulated::compound::AssembleTransmissionForces(
    AssemblyParams const& params,
    CArticulatedLinkTransforms<TimeStep::Current> const& currLinkTxs,
    CArticulatedLinkTransforms<TimeStep::StageStart> const& stageStartLinkTxs,
    CArticulatedJacobian const& jacobian,
    CArticulatedReducedPose<TimeStep::Current> const& currPose,
    CArticulatedReducedPose<TimeStep::StageStart> const& stageStartPose,
    CTimeIntegratorState const& intState,
    CTransmissions const& transmissions,
    CActorSnle& outCompoundSnle) {
  for (auto const& transmission : transmissions.transmissions) {
    transmission->AddObjResDRes(
        MakeConstSpan(currLinkTxs),
        MakeConstSpan(stageStartLinkTxs),
        AsConstView(jacobian.value),
        currPose.value,
        stageStartPose.value,
        intState.dtStage,
        params,
        outCompoundSnle.objective,
        outCompoundSnle.reducedResidual,
        std::get<Matrix<real>>(outCompoundSnle.reducedDResidual));
  }
}

#define MOCHI_ASSERT_VERBOSE_FREE_OR_SPHERICAL(type)                                     \
  MOCHI_ASSERT_VERBOSE(                                                                  \
      (type) == ArticulatedJointType::Free || (type) == ArticulatedJointType::Spherical, \
      "Wrong joint type");
#define MOCHI_ASSERT_VERBOSE_REVOLUTE_OR_PRISMATIC(type)                                     \
  MOCHI_ASSERT_VERBOSE(                                                                      \
      (type) == ArticulatedJointType::Revolute || (type) == ArticulatedJointType::Prismatic, \
      "Wrong joint type");

void articulated::compound::AssembleDampingForces(
    AssemblyParams const& params,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CArticulatedJointFrictionParams const& frictionParams,
    CArticulatedJointTransforms<TimeStep::Current> const& currJointTxs,
    CArticulatedJointTransforms<TimeStep::StageStart> const& stageStartJointTxs,
    CArticulatedReducedPose<TimeStep::Current> const& currPose,
    CArticulatedReducedPose<TimeStep::StageStart> const& stageStartPose,
    CTimeIntegratorState const& intState,
    CActorSnle& outCompoundSnle) {
  auto const* joints = bodyShape.shape->GetJointsData();
  auto const jointTypes = joints->jointTypes;
  auto const dofInfo = joints->dofInfo;
  auto dampingFuncRigidBody = params.gradTarget == GradTarget::Current
      ? AddRigidBodyDamping<GradTarget::Current>
      : AddRigidBodyDamping<GradTarget::Previous>;
  auto assemblerRigidBody =
      [&](int i, double* energy, RigidGradient* gradient, RigidHessian* hessian) -> bool {
    MOCHI_ASSERT_VERBOSE_FREE_OR_SPHERICAL(jointTypes[i]);
    if (frictionParams[i].viscous == 0_r) {
      return false;
    }
    dampingFuncRigidBody(
        currJointTxs[i],
        frictionParams[i].viscous,
        stageStartJointTxs[i],
        intState.dtStage,
        energy,
        gradient,
        hessian);
    return true;
  };
  auto dampingFuncSingleDof = params.gradTarget == GradTarget::Current
      ? AddSingleDofDamping<GradTarget::Current>
      : AddSingleDofDamping<GradTarget::Previous>;
  auto assemblerSingleDof = [&](int i, double* energy, real* gradient, real* hessian) -> bool {
    MOCHI_ASSERT_VERBOSE_REVOLUTE_OR_PRISMATIC(jointTypes[i]);
    if (frictionParams[i].viscous == 0_r) {
      return false;
    }
    int const offset = poseInfo[i].offset;
    dampingFuncSingleDof(
        currPose.value[offset],
        frictionParams[i].viscous,
        stageStartPose.value[offset],
        intState.dtStage,
        energy,
        gradient,
        hessian);
    return true;
  };
  AssembleJointForces(
      params, jointTypes, dofInfo, assemblerRigidBody, assemblerSingleDof, outCompoundSnle);
}

void articulated::compound::AssembleInertiaForces(
    AssemblyParams const& params,
    ecs::OptionalTag<TagUseNewtonEulerInertia> useNewtonEulerInertia,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CArticulatedInertiaParams const& inertiaParams,
    CArticulatedJointTransforms<TimeStep::Current> const& currJointTxs,
    CArticulatedJointTransforms<TimeStep::StageStart> const& stageStartJointTxs,
    CArticulatedJointVels<TimeStep::StageStart> const& stageStartJointVels,
    CArticulatedReducedPose<TimeStep::Current> const& currPose,
    CArticulatedReducedPose<TimeStep::StageStart> const& stageStartPose,
    CTimeIntegratorState const& intState,
    CActorSnle& outCompoundSnle) {
  auto const* joints = bodyShape.shape->GetJointsData();
  auto const jointTypes = joints->jointTypes;
  auto const jointAxes = joints->jointAxes;
  auto const dofInfo = joints->dofInfo;
  auto inertiaFuncRigidBody = GetRigidInertiaFn<real>(params.gradTarget, useNewtonEulerInertia);
  auto assemblerRigidBody =
      [&](int i, double* energy, RigidGradient* gradient, RigidHessian* hessian) -> bool {
    MOCHI_ASSERT_VERBOSE_FREE_OR_SPHERICAL(jointTypes[i]);
    if (inertiaParams[i] == 0_r) {
      return false;
    }
    inertiaFuncRigidBody(
        inertiaParams[i] /*mass*/,
        inertiaParams[i] * (useNewtonEulerInertia ? 1_r : 0.5_r) /*inertia*/,
        intState.dtStage,
        currJointTxs[i],
        stageStartJointTxs[i],
        stageStartJointVels.value[i].value,
        energy,
        gradient,
        hessian);

    if (params.psdDRes && hessian) {
      VMatrix3x3r outHessianRot;
      LoadSubmatrix<3, 3, 6, 6>(outHessianRot, Int2{3, 3}, *hessian);
      ProjectSymPsd(outHessianRot, std::numeric_limits<real>::epsilon());
      StoreSubmatrix<3, 3, 6, 6>(*hessian, Int2{3, 3}, outHessianRot);
    }
    return true;
  };
  auto inertiaFuncSingleDof = AddSingleDofInertia<GradTarget::Current>;
  if (params.gradTarget == GradTarget::Previous) {
    inertiaFuncSingleDof = AddSingleDofInertia<GradTarget::Previous>;
  } else if (params.gradTarget == GradTarget::PreviousDelta) {
    inertiaFuncSingleDof = AddSingleDofInertia<GradTarget::PreviousDelta>;
  }
  auto assemblerSingleDof = [&](int i, double* energy, real* gradient, real* hessian) -> bool {
    MOCHI_ASSERT_VERBOSE_REVOLUTE_OR_PRISMATIC(jointTypes[i]);
    if (inertiaParams[i] == 0_r) {
      return false;
    }
    int const offset = poseInfo[i].offset;
    auto const stageStartJointVel = jointTypes[i] == ArticulatedJointType::Revolute
        ? stageStartJointVels.value[i].value.GetOmegaAndVSym().first
        : stageStartJointVels.value[i].value.GetVCom();
    real const stageStartVel = Dot<3>(ToSimd(jointAxes[i]), stageStartJointVel);
    inertiaFuncSingleDof(
        currPose.value[offset],
        inertiaParams[i],
        stageStartPose.value[offset],
        stageStartVel,
        intState.dtStage,
        energy,
        gradient,
        hessian);
    return true;
  };
  AssembleJointForces(
      params, jointTypes, dofInfo, assemblerRigidBody, assemblerSingleDof, outCompoundSnle);
}

void articulated::compound::AssembleFrictionForces(
    AssemblyParams const& params,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CArticulatedJointFrictionParams const& frictionParams,
    CArticulatedJointTransforms<TimeStep::Current> const& currJointTxs,
    CArticulatedJointTransforms<TimeStep::StageStart> const& stageStartJointTxs,
    CArticulatedReducedPose<TimeStep::Current> const& currPose,
    CArticulatedReducedPose<TimeStep::StageStart> const& stageStartPose,
    CTimeIntegratorState const& intState,
    CActorSnle& outCompoundSnle) {
  auto const* joints = bodyShape.shape->GetJointsData();
  auto const jointTypes = joints->jointTypes;
  auto const dofInfo = joints->dofInfo;
  auto frictionFuncRigidBody = params.gradTarget == GradTarget::Current
      ? AddRigidBodyFriction<GradTarget::Current>
      : AddRigidBodyFriction<GradTarget::Previous>;
  auto assemblerRigidBody =
      [&](int i, double* energy, RigidGradient* gradient, RigidHessian* hessian) -> bool {
    MOCHI_ASSERT_VERBOSE_FREE_OR_SPHERICAL(jointTypes[i]);
    auto const& fp = frictionParams[i];
    MOCHI_ASSERT_VERBOSE(
        jointTypes[i] != ArticulatedJointType::Free ||
            (fp.coulomb == 0_r && fp.stictionExtra == 0_r),
        "Joint friction must be zero for free joints");
    if (fp.coulomb + fp.stictionExtra == 0_r) {
      return false;
    }
    frictionFuncRigidBody(
        params.fittedSaturationHessian.jointFriction,
        params.psdDRes,
        fp,
        intState.dtStage,
        currJointTxs[i],
        stageStartJointTxs[i],
        energy,
        gradient,
        hessian);
    return true;
  };
  auto frictionFuncSingleDof = params.gradTarget == GradTarget::Current
      ? AddSingleDofFriction<GradTarget::Current>
      : AddSingleDofFriction<GradTarget::Previous>;
  auto assemblerSingleDof = [&](int i, double* energy, real* gradient, real* hessian) -> bool {
    MOCHI_ASSERT_VERBOSE_REVOLUTE_OR_PRISMATIC(jointTypes[i]);
    auto const& fp = frictionParams[i];
    if (fp.coulomb + fp.stictionExtra == 0_r) {
      return false;
    }
    int const offset = poseInfo[i].offset;
    frictionFuncSingleDof(
        params.fittedSaturationHessian.jointFriction,
        params.psdDRes,
        fp,
        intState.dtStage,
        currPose.value[offset],
        stageStartPose.value[offset],
        energy,
        gradient,
        hessian);
    return true;
  };
  AssembleJointForces(
      params, jointTypes, dofInfo, assemblerRigidBody, assemblerSingleDof, outCompoundSnle);
}

void articulated::compound::AssembleExternalForces(
    AssemblyParams const& params,
    CArticulatedProps const& props,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CExternalForces const& externalForces,
    CArticulatedJointTransforms<TimeStep::Current> const& currJointTxs,
    CArticulatedJointTransforms<TimeStep::StageStart> const& stageStartJointTxs,
    CArticulatedReducedPose<TimeStep::Current> const& currPose,
    CActorSnle& outCompoundSnle) {
  auto const* joints = bodyShape.shape->GetJointsData();
  auto const jointTypes = joints->jointTypes;
  auto const jointDofInfo = joints->dofInfo;
  // For convenience, fill an array with force values for all dofs, even if they're zero.
  MOCHI_FILO_STACK_ALLOCATOR(allocator, sizeof(real) * 256); // Up to 256 dofs
  DynamicArray<real> allForces(props.reducedDofsDim, 0_r, &allocator);
  for (int i = 0; i < externalForces.dofs.size(); ++i) {
    allForces[externalForces.dofs[i]] = externalForces.forces[i];
  }

  // Disable hessian assembly
  AssemblyParams paramsLocal = params;
  paramsLocal.assemDRes = false;

  auto assemblerRigidBody =
      [&](int i, double* energy, RigidGradient* gradient, RigidHessian* /* hessian */) -> bool {
    MOCHI_ASSERT_VERBOSE_FREE_OR_SPHERICAL(jointTypes[i]);
    // Get the external forces and dofs for this joint
    RigidGradient forces{};
    std::array<int, RigidSize::kDAll> constexpr kDofs = {0, 1, 2, 3, 4, 5};
    auto const& dofInfo = jointDofInfo[i];
    if (jointTypes[i] == ArticulatedJointType::Free) {
      auto force = Span(&allForces[dofInfo.GetTransOffset()], RigidSize::kDTrans);
      Unflatten<Real3>(forces)[0] = Unflatten<Real3 const>(force)[0];
    }
    auto torque = Span(&allForces[dofInfo.GetRotOffset()], RigidSize::kDRot);
    Unflatten<Real3>(forces)[1] = Unflatten<Real3 const>(torque)[0];
    if (forces == RigidGradient{}) {
      return false;
    }
    AddRigidBodyExternalForces(
        currJointTxs[i], stageStartJointTxs[i], kDofs, forces, energy, gradient);
    return true;
  };
  auto assemblerSingleDof =
      [&](int i, double* energy, real* gradient, real* /* hessian */) -> bool {
    MOCHI_ASSERT_VERBOSE_REVOLUTE_OR_PRISMATIC(jointTypes[i]);
    auto const& dofInfo = jointDofInfo[i];
    auto force = allForces[dofInfo.offset];
    if (force == 0_r) {
      return false;
    }
    AddSingleDofExternalForce(currPose.value[poseInfo[i].offset], force, energy, gradient);
    return true;
  };
  AssembleJointForces(
      paramsLocal,
      jointTypes,
      jointDofInfo,
      assemblerRigidBody,
      assemblerSingleDof,
      outCompoundSnle);
}

#undef MOCHI_ASSERT_VERBOSE_FREE_OR_SPHERICAL
#undef MOCHI_ASSERT_VERBOSE_REVOLUTE_OR_PRISMATIC

// Receives the raw solution (in reduced space). Copies it to the reduced pose and computes the
// full pose.
void articulated::compound::EntitySetSolution(
    ColumnVectorView<real const> solution,
    ecs::RequiredTag<TagArticulatedActor>,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CArticulatedParents const& parents,
    CArticulatedRestTransforms const& restTransforms,
    CRootTransform const& rootTransform,
    CArticulatedReducedPose<TimeStep::Current>& outReducedPose,
    CArticulatedJointTransforms<TimeStep::Current>& outJointTransforms,
    CArticulatedLinkTransforms<TimeStep::Current>& outLinkTransforms,
    CArticulatedFullPose& outFullPose) {
  MOCHI_PROFILE_SCOPE();
  outReducedPose.value = solution; // deep copy
  auto const* joints = bodyShape.shape->GetJointsData();
  ComputeFullPose(
      joints->jointTypes,
      joints->jointAxes,
      poseInfo,
      parents,
      restTransforms,
      rootTransform.worldFromLocal,
      outReducedPose.value,
      outJointTransforms,
      outLinkTransforms,
      outFullPose.value);
}

// Writes the reduced pose to the solution span
void articulated::compound::EntityGetSolution(
    ColumnVectorView<real> outSolution,
    ecs::RequiredTag<TagArticulatedActor>,
    CArticulatedReducedPose<TimeStep::Current> const& reducedPose) {
  MOCHI_PROFILE_SCOPE();
  AsView(outSolution) = reducedPose.value; // deep copy
}

// Calls the internal EntitySetSolution system
void articulated::compound::EntityPostNewSolution(
    ColumnVectorView<real const> solution,
    ecs::RequiredTag<TagArticulatedActor>,
    CDofOffset const& dofOffset,
    CActorDofInfo const& dofInfo,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CArticulatedParents const& parents,
    CArticulatedRestTransforms const& restTransforms,
    CRootTransform const& rootTransform,
    CArticulatedJointTransforms<TimeStep::Current>& jointTransforms,
    CArticulatedLinkTransforms<TimeStep::Current>& linkTransforms,
    CArticulatedFullPose& fullPose,
    CArticulatedReducedPose<TimeStep::Current>& reducedPose) {
  MOCHI_PROFILE_SCOPE();
  EntitySetSolution(
      solution.MiddleRows(dofOffset.poseOffset, dofInfo.poseSize),
      {},
      bodyShape,
      poseInfo,
      parents,
      restTransforms,
      rootTransform,
      reducedPose,
      jointTransforms,
      linkTransforms,
      fullPose);
}

void articulated::compound::EntityPostNewIncrement(
    ColumnVectorView<real const> reference,
    ColumnVectorView<real const> increment,
    ecs::RequiredTag<TagArticulatedActor>,
    CDofOffset const& dofOffset,
    CActorDofInfo const& dofInfo,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CArticulatedParents const& parents,
    CArticulatedRestTransforms const& restTransforms,
    CRootTransform const& rootTransform,
    CDirichletBC<real> const& dirichlet,
    CArticulatedJointTransforms<TimeStep::Current>& jointTransforms,
    CArticulatedLinkTransforms<TimeStep::Current>& linkTransforms,
    CArticulatedFullPose& fullPose,
    CArticulatedReducedPose<TimeStep::Current>& reducedPose) {
  MOCHI_PROFILE_SCOPE();
  auto const* joints = bodyShape.shape->GetJointsData();

  // Reserve stack memory for the solution (up to 256 joints with 3 dofs per joint).
  MOCHI_FILO_STACK_ALLOCATOR(allocator, sizeof(real) * 3 * 256);
  ColumnVector<real> solution(dofInfo.poseSize, &allocator);

  // Add the increment to the reference, accounting for Lie-algebra parameterizations.
  AddLieDeltaToReducedPose(
      joints->jointTypes,
      joints->dofInfo,
      poseInfo,
      reference.MiddleRows(dofOffset.poseOffset, dofInfo.poseSize),
      increment.MiddleRows(dofOffset.dofsOffset, dofInfo.dofsSize),
      solution);
  // Normalize quaternions to avoid drift
  NormalizeQuaternions(joints->jointTypes, poseInfo, solution);

  // Apply Dirichlet boundary conditions explicitly, because the rotation composition may add
  // noise.
  SetDirichletBCs(dirichlet, solution);

  EntitySetSolution(
      solution,
      {},
      bodyShape,
      poseInfo,
      parents,
      restTransforms,
      rootTransform,
      reducedPose,
      jointTransforms,
      linkTransforms,
      fullPose);
}

// Whether any joint has nonzero viscous damping, requiring damping-force assembly.
static bool HasViscousDamping(CArticulatedJointFrictionParams const& friction) {
  return std::any_of(
      friction.begin(), friction.end(), [](auto const& fp) { return fp.viscous != 0_r; });
}

// Whether any joint has nonzero dry friction, requiring friction-force assembly.
static bool HasDryFriction(CArticulatedJointFrictionParams const& friction) {
  return std::any_of(friction.begin(), friction.end(), [](auto const& fp) {
    return (fp.coulomb + fp.stictionExtra) != 0_r;
  });
}

// Whether any joint has nonzero joint inertia, requiring inertia-force assembly.
static bool HasJointInertia(CArticulatedInertiaParams const& inertia) {
  return std::any_of(inertia.begin(), inertia.end(), [](real value) { return value != 0_r; });
}

void articulated::compound::EntityAssemble(
    AssemblyParams const& params,
    ecs::RequiredTag<TagArticulatedActor>,
    ecs::PartialRegistry<CActorSnle const, CDofOffset const> reg,
    ecs::OptionalTag<TagUseNewtonEulerInertia> useNewtonEulerInertia,
    CArticulatedProps const& props,
    CGroupMembers const& groupMembers,
    CArticulatedJacobian const& jacobian,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CTimeIntegratorState const& intState,
    CArticulatedInertiaParams const& inertiaParams,
    CArticulatedJointFrictionParams const& frictionParams,
    CCompoundConstraintSnle const* constraintFullSnle,
    CExternalForces const& externalForces,
    CTransmissions const& transmissions,
    CArticulatedReducedPose<TimeStep::StageStart> const& stageStartPose,
    CArticulatedReducedPose<TimeStep::Current> const& currPose,
    CArticulatedJointTransforms<TimeStep::StageStart> const& stageStartJointTxs,
    CArticulatedJointTransforms<TimeStep::Current> const& currJointTxs,
    CArticulatedLinkTransforms<TimeStep::StageStart> const& stageStartLinkTxs,
    CArticulatedLinkTransforms<TimeStep::Current> const& currLinkTxs,
    CArticulatedJointVels<TimeStep::StageStart> const& stageStartJointVels,
    CActorDerivedStateInfo const* derivedStateInfo,
    CActorDiffInputInfo const* diffInputInfo,
    CActorSnle& outCompoundSnle) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT_VERBOSE(outCompoundSnle.UseReduced(), "Reduced SNLE must be enabled.");

  // If this is an input grad target and this actor has no actual differentiable input, all we need
  // to do is resize the residual to 0
  auto const gradTarget = params.gradTarget;
  if (gradTarget == GradTarget::CurrentInput || gradTarget == GradTarget::PreviousInput) {
    MOCHI_ASSERT_VERBOSE(
        !params.assemObj && params.assemRes && !params.assemDRes, "Invalid request");
    MOCHI_ASSERT_VERBOSE(diffInputInfo, "CActorDiffInputInfo not found");
    if (diffInputInfo->dofsSize == 0) {
      outCompoundSnle.reducedResidual.Resize(0);
      return;
    }
  }

  // This code currently assumes SparseMatrix format (TODO: Change this)
  MOCHI_ASSERT(std::holds_alternative<SparseMatrix<real>>(outCompoundSnle.fullDResidual));
  auto outCompoundFullDRes =
      std::get<SparseMatrixView<real>>(AsView(outCompoundSnle.fullDResidual));
  outCompoundSnle.SetFullToZero(params);

  // First, assemble the full DOF rigid bodies into this compound's CActorSnle.
  // TODO: We could skip this step if we had a way to go directly from the full DOF bodies to the
  // reduced DOF problem.
  if (IsAssemblyNeeded(StateDependency::SecondOrder, false /*inputDependency*/, gradTarget)) {
    for (auto bodyEntity : groupMembers.actors) {
      auto const& bodySnle = reg.get<CActorSnle const>(bodyEntity);
      auto const& bodyDofOffset = reg.get<CDofOffset const>(bodyEntity);
      if (params.assemObj) {
        outCompoundSnle.objective += bodySnle.objective;
      }
      if (params.assemRes) {
        MOCHI_ASSERT(bodySnle.fullResidual.size() == RigidSize::kDAll);
        MOCHI_ASSERT(
            bodyDofOffset.dofsOffset + RigidSize::kDAll <= outCompoundSnle.fullResidual.size());
        outCompoundSnle.fullResidual.MiddleRows(bodyDofOffset.dofsOffset, RigidSize::kDAll) =
            bodySnle.fullResidual;
      }
      if (params.assemDRes) {
        MOCHI_ASSERT(
            std::holds_alternative<Matrix<real>>(bodySnle.fullDResidual),
            "Expected dense storage.");
        auto const& bodyFullDRes = std::get<Matrix<real>>(bodySnle.fullDResidual);
        MOCHI_ASSERT(
            bodyFullDRes.Rows() == RigidSize::kDAll && bodyFullDRes.Cols() == RigidSize::kDAll);
        MOCHI_ASSERT(bodyDofOffset.dofsOffset + RigidSize::kDAll <= outCompoundFullDRes.Rows());
        // bodyFullDRes should be symmetric, so transpose in order to convert to row-matrix view
        IndexGroup inds{.src = 0, .dst = bodyDofOffset.dofsOffset, .count = RigidSize::kDAll};
        auto indsSpan = MakeSingletonConstSpan(inds);
        MatAddSubBlocks(outCompoundFullDRes, indsSpan, indsSpan, bodyFullDRes.Transpose());
      }
    }
  }

  // If there were any constraints affecting the full DOF rigid bodies (e.g. forming a closed
  // kinematic loop or link-based pose control), then add the constraint assembly values into this
  // compound's CActorSnle. Constraints are first-order; do the work only if needed
  if (constraintFullSnle &&
      IsAssemblyNeeded(StateDependency::FirstOrder, true /*inputDependency*/, gradTarget)) {
    if (params.assemObj) {
      outCompoundSnle.objective += constraintFullSnle->objective;
    }
    if (params.assemRes) {
      MOCHI_ASSERT(constraintFullSnle->residuals.size() == 1, "Expected 1 residual.");
      int constraintOffset = constraintFullSnle->residuals[0].first;
      auto const& constraintResidual = constraintFullSnle->residuals[0].second;
      MOCHI_ASSERT(
          constraintOffset + constraintResidual.Rows() <= outCompoundSnle.fullResidual.Rows(),
          "CCompoundConstraintSnle should have been resized to the number of full DOFs in this articulated compound");
      outCompoundSnle.fullResidual.MiddleRows(constraintOffset, constraintResidual.Rows()) +=
          constraintResidual;
    }
    if (params.assemDRes) {
      MOCHI_ASSERT(constraintFullSnle->dresiduals.size() == 1, "Expected 1 DResidual.");
      auto const& constraintDResidual = constraintFullSnle->dresiduals[0];
      MOCHI_ASSERT(
          constraintDResidual.rowOffset == 0 && constraintDResidual.colOffset == 0,
          "Expected zero row and col offsets.");
      MOCHI_ASSERT(
          GetNumRows(constraintDResidual.matrix) <= GetNumRows(outCompoundSnle.fullDResidual),
          "CCompoundConstraintSnle should have been resized to the number of full DOFs in this articulated compound");
      auto constraintFullDRes =
          std::get<SparseMatrixView<real const>>(AsConstView(constraintDResidual.matrix));
      outCompoundFullDRes += constraintFullDRes;
    }
  }

  // Reduced residual. It must be resized depending on the target (state, derived state or input).
  if (params.assemRes) {
    if (gradTarget == GradTarget::CurrentInput || gradTarget == GradTarget::PreviousInput) {
      MOCHI_ASSERT_VERBOSE(diffInputInfo, "CActorDiffInputInfo not found");
      auto const diffInputSize = diffInputInfo->dofsSize;
      MOCHI_ASSERT_VERBOSE(diffInputSize > 0, "Expected non-zero size");
      outCompoundSnle.reducedResidual.Resize(diffInputSize);
      outCompoundSnle.reducedResidual = jacobian.value.Transpose() * outCompoundSnle.fullResidual;
    } else if (gradTarget == GradTarget::PreviousDelta) {
      MOCHI_ASSERT_VERBOSE(derivedStateInfo, "CActorDerivedStateInfo not found");
      auto const derivedDofsSize = derivedStateInfo->dofsSize;
      outCompoundSnle.reducedResidual.Resize(derivedDofsSize);
      outCompoundSnle.reducedResidual.TopRows(props.reducedDofsDim).SetZero();
      outCompoundSnle.reducedResidual.BottomRows(props.fullDofsDim) = outCompoundSnle.fullResidual;
    } else {
      MOCHI_ASSERT_VERBOSE(
          gradTarget == GradTarget::Current || gradTarget == GradTarget::Previous,
          "Unexpected grad target");
      outCompoundSnle.reducedResidual.Resize(props.reducedDofsDim);
      outCompoundSnle.reducedResidual = jacobian.value.Transpose() * outCompoundSnle.fullResidual;
    }
  }

  // Reduced dresidual
  if (params.assemDRes) {
    MOCHI_ASSERT_VERBOSE(gradTarget == GradTarget::Current, "Unexpected grad target");

    // Temporary storage
    MOCHI_FILO_STACK_ALLOCATOR(tempAllocator, 32 * 1024);

    // Project the Jacobian
    Matrix<real> DJ(outCompoundFullDRes * jacobian.value, &tempAllocator);
    MOCHI_ASSERT(
        std::holds_alternative<Matrix<real>>(outCompoundSnle.reducedDResidual),
        "Expected dense storage.");
    auto& JtDJ = std::get<Matrix<real>>(outCompoundSnle.reducedDResidual);
    JtDJ = jacobian.value.Transpose() * DJ;

    // regularize the jacobian in case we want to do quasi-static optimization
    if (!IsFinite(intState.dtStage)) {
      real maxDiagonalElement = 0_r;
      for (int i = 0; i < JtDJ.Rows(); i++) {
        maxDiagonalElement = Max(maxDiagonalElement, JtDJ(i, i));
      }
      maxDiagonalElement *= kDResRegularizationCoefficient;
      for (int i = 0; i < JtDJ.Rows(); i++) {
        JtDJ(i, i) += maxDiagonalElement;
      }
    }
  }

  // Add damping forces on joints.
  if (HasViscousDamping(frictionParams) &&
      IsAssemblyNeeded(StateDependency::FirstOrder, false /*inputDependency*/, gradTarget)) {
    AssembleDampingForces(
        params,
        bodyShape,
        poseInfo,
        frictionParams,
        currJointTxs,
        stageStartJointTxs,
        currPose,
        stageStartPose,
        intState,
        outCompoundSnle);
  }

  // Add inertial forces on joints.
  if (HasJointInertia(inertiaParams) &&
      IsAssemblyNeeded(StateDependency::SecondOrder, false /*inputDependency*/, gradTarget)) {
    AssembleInertiaForces(
        params,
        useNewtonEulerInertia,
        bodyShape,
        poseInfo,
        inertiaParams,
        currJointTxs,
        stageStartJointTxs,
        stageStartJointVels,
        currPose,
        stageStartPose,
        intState,
        outCompoundSnle);
  }

  // Add friction forces on joints.
  if (HasDryFriction(frictionParams) &&
      IsAssemblyNeeded(StateDependency::FirstOrder, false /*inputDependency*/, gradTarget)) {
    AssembleFrictionForces(
        params,
        bodyShape,
        poseInfo,
        frictionParams,
        currJointTxs,
        stageStartJointTxs,
        currPose,
        stageStartPose,
        intState,
        outCompoundSnle);
  }

  // Add external forces on joints.
  if (!externalForces.Empty() &&
      IsAssemblyNeeded(StateDependency::ZeroOrder, true /*inputDependency*/, gradTarget)) {
    AssembleExternalForces(
        params,
        props,
        bodyShape,
        poseInfo,
        externalForces,
        currJointTxs,
        stageStartJointTxs,
        currPose,
        outCompoundSnle);
  }

  if (!transmissions.transmissions.empty()) {
    AssembleTransmissionForces(
        params,
        currLinkTxs,
        stageStartLinkTxs,
        jacobian,
        currPose,
        stageStartPose,
        intState,
        transmissions,
        outCompoundSnle);
  }
}

void articulated::compound::EntityPreFirstStage(
    ecs::Included<TagArticulatedActor>,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CTimeIntegratorState const& intState,
    CArticulatedReducedPose<TimeStep::Previous> const& prevPose,
    CArticulatedJointVels<TimeStep::Previous> const& prevJointVels,
    CIntegrationArticulatedReducedPose& outIntPose,
    CIntegrationArticulatedJointVels& outIntJointVels) {
  MOCHI_PROFILE_SCOPE();
  auto const* joints = bodyShape.shape->GetJointsData();
  integration::ArticulatedIntegrationMetadata const metadata{
      joints->jointTypes, joints->dofInfo, poseInfo};

  // Joint DoFs and joint velocities are differential variables. Use integration utilities to
  // compute their values at the beginning of the step.
  integration::ApplyTimeIntegrationStepStart(
      metadata, intState, outIntPose, prevPose, outIntPose.stepStart);
  for (int i = 0; i < prevJointVels.value.size(); ++i) {
    integration::ApplyTimeIntegrationStepStart(
        intState,
        outIntJointVels.value[i],
        prevJointVels.value[i],
        outIntJointVels.value[i].stepStart);
  }
}

template <TimeTarget kTargetTime, TimeStep kOutTime>
static void ComputeStateAndVelocity(
    ecs::Included<TagArticulatedActor>,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CArticulatedParents const& parents,
    CArticulatedRestTransforms const& restTransforms,
    CRootTransform const& rootTransform,
    CTimeIntegratorState const& intState,
    CArticulatedReducedPose<kOutTime>& outPose,
    CArticulatedJointVels<kOutTime>& outJointVels,
    CIntegrationArticulatedReducedPose& outIntDofs,
    CIntegrationArticulatedJointVels& outIntJointVels,
    CArticulatedJointTransforms<kOutTime>& outJointTransforms,
    CArticulatedLinkTransforms<kOutTime>& outLinkTransforms,
    CArticulatedFullPose& outFullPose) {
  auto const* joints = bodyShape.shape->GetJointsData();
  integration::ArticulatedIntegrationMetadata const metadata{
      joints->jointTypes, joints->dofInfo, poseInfo};
  // Joint DoFs are differential variables. Use integration utilities to compute their value.
  integration::ApplyTimeIntegration<kTargetTime>(metadata, intState, outIntDofs, outPose);

  // Joint and link transforms are algebraic variables. Compute them from joint DoFs.
  ComputeFullPose(
      joints->jointTypes,
      joints->jointAxes,
      poseInfo,
      parents,
      restTransforms,
      rootTransform.worldFromLocal,
      outPose.value,
      outJointTransforms,
      outLinkTransforms,
      outFullPose.value);

  // Joint velocities are differential variables. Use integration utilities to compute their
  // value.
  for (int i = 0; i < outIntJointVels.value.size(); ++i) {
    integration::ApplyTimeIntegration<kTargetTime>(
        intState, outIntJointVels.value[i], outJointVels.value[i]);
  }
}

static void PushCurrentStateAndVelocityToIntegrationStages(
    ecs::Included<TagArticulatedActor>,
    CTimeIntegratorState const& intState,
    CArticulatedReducedPose<TimeStep::Current> const& currDofs,
    CArticulatedJointVels<TimeStep::Current> const& currJointVels,
    CIntegrationArticulatedReducedPose& outIntDofs,
    CIntegrationArticulatedJointVels& outIntJointVels) {
  // Joint DoFs and velocities are differential variables. Push them to the vectors containing
  // their values at the end of each time integration stage.
  outIntDofs.stages[intState.currentStage].value = currDofs.value;
  for (int i = 0; i < isize(outIntJointVels.value); ++i) {
    outIntJointVels.value[i].stages[intState.currentStage].value = currJointVels.value[i].value;
  }
}

static void ComputeCurrentVelocity(
    ecs::RequiredTag<TagArticulatedActor>,
    CTimeIntegratorState const& intState,
    CArticulatedJointTransforms<TimeStep::StageStart> const& stageStartJointTransforms,
    CArticulatedJointTransforms<TimeStep::Current> const& currJointTransforms,
    CArticulatedJointVels<TimeStep::Current>& outCurrJointVels) {
  // Joint velocities are recovered via finite differences of the pose at the beginning and at the
  // end of the stage.
  for (int i = 0; i < isize(currJointTransforms); ++i) {
    outCurrJointVels.value[i].value.SetFromFiniteDifferencePose(
        stageStartJointTransforms[i], currJointTransforms[i], intState.dtStage);
  }
}

static void HandleSolverDivergence(
    ecs::Included<TagArticulatedActor>,
    CConvergenceStatus const& convergence,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CArticulatedParents const& parents,
    CArticulatedRestTransforms const& restTransforms,
    CRootTransform const& rootTransform,
    CArticulatedReducedPose<TimeStep::Previous> const& prevReducedPose,
    CArticulatedReducedPose<TimeStep::Current>& outCurrReducedPose,
    CArticulatedFullPose& outFullPose,
    CArticulatedJointTransforms<TimeStep::Current>& outJointTransforms,
    CArticulatedLinkTransforms<TimeStep::Current>& outLinkTransforms,
    CArticulatedJointVels<TimeStep::Current>& outCurrJointVels) {
  if (convergence.stageStatus == ConvergenceStatus::Diverged) {
    // If the solver diverged, reset the DoFs at the previous time step. Do not reset them at the
    // beginning of the stage since there is no guarantee that's a safe state.
    outCurrReducedPose.value = prevReducedPose.value;
    auto const* joints = bodyShape.shape->GetJointsData();
    ComputeFullPose(
        joints->jointTypes,
        joints->jointAxes,
        poseInfo,
        parents,
        restTransforms,
        rootTransform.worldFromLocal,
        outCurrReducedPose.value,
        outJointTransforms,
        outLinkTransforms,
        outFullPose.value);

    // Reset the velocity to zero.
    for (auto& jointVel : outCurrJointVels.value) {
      jointVel.value.SetZero();
    }
  }
}

static void CompoundEntityPreStep(
    ecs::Included<TagArticulatedActor>,
    CArticulatedReducedPose<TimeStep::Current> const& currState,
    CArticulatedJointVels<TimeStep::Current>& currJointVels,
    CArticulatedReducedPose<TimeStep::Previous>& prevState,
    CArticulatedJointVels<TimeStep::Previous>& prevJointVels) {
  // Shift joint DoFs from current to previous.
  prevState.value = currState.value;
  // Shift joint velocities from current to previous and reset current velocity.
  for (int i = 0; i < isize(currJointVels.value); ++i) {
    prevJointVels.value[i].value = currJointVels.value[i].value;
    currJointVels.value[i].value.SetZero();
  }
}

void articulated::compound::PreStepPipeline(entt::registry& reg) {
  ecs::InvokeForEachGlobal(&CompoundEntityPreStep, reg);
  ecs::InvokeForEachGlobal(&articulated::rigid::EntityPreStep, reg);
}

void articulated::compound::PreStagePipeline(
    entt::registry& reg,
    Span<entt::entity const> entities) {
  MOCHI_PROFILE_SCOPE();
  // First, compute the state and velocity of the articulated compound at the beginning of the
  // stage.
  ecs::InvokeForEach(
      &ComputeStateAndVelocity<TimeTarget::StageStart, TimeStep::StageStart>, reg, entities);

  // Then, update the rigid actors in the compound (if any).
  ecs::InvokeForEach(&articulated::rigid::EntityPreStage, reg, entities);

  ecs::InvokeForEach(&articulated::compound::ResolveSkinning<TimeStep::StageStart>, reg, entities);
}

void articulated::compound::PostStagePipeline(
    entt::registry& reg,
    Span<entt::entity const> entities) {
  MOCHI_PROFILE_SCOPE();
  // Compute the velocity
  ecs::InvokeForEach(&ComputeCurrentVelocity, reg, entities);

  // Handle potential solver divergence.
  ecs::InvokeForEach(&HandleSolverDivergence, reg, entities);

  // Then, push the state and velocity to the integration stages, update the rigid actors in the
  // compound (if any), and compute skinning on the visual mesh. These operations are independent
  // of each other.
  ecs::InvokeForEach(&PushCurrentStateAndVelocityToIntegrationStages, reg, entities);
  ecs::InvokeForEach(&articulated::rigid::EntityPostStage, reg, entities);
  ecs::InvokeForEach(&ResolveSkinning<TimeStep::Current>, reg, entities);
}

void articulated::compound::PostLastStagePipeline(
    entt::registry& reg,
    Span<entt::entity const> entities) {
  MOCHI_PROFILE_SCOPE();
  // First, compute the state and velocity of the articulated compound at the end of the time
  // step.
  ecs::InvokeForEach(
      &ComputeStateAndVelocity<TimeTarget::StepEnd, TimeStep::Current>, reg, entities);

  // Update the Jacobian to match the step-end pose.
  ecs::InvokeForEach(&articulated::compound::UpdateJacobianState<TimeStep::Current>, reg, entities);

  // Then, update the rigid actors in the compound (if any).
  ecs::InvokeForEach(&articulated::rigid::EntityPostLastStage, reg, entities);

  // Resolve skinning on the full mesh because this is where we need to do this
  // for, e.g., rendering purposes
  ecs::InvokeForEach(
      &ResolveSkinning<TimeStep::Current, /*kForceUseAllNodes = */ true>, reg, entities);
}

template <typename ContainerT, size_t... DIMS>
static void RecordDatasetFromContainers(
    Span<ContainerT const> allContainers,
    std::string_view name,
    NdArray<real, DIMS...> (*extractData)(ContainerT const&),
    Allocator* tempAllocator,
    CRecordingData& outData) {
  auto count = isize(allContainers);
  DynamicArray<NdArray<real, DIMS...>> data(tempAllocator);
  data.reserve(count);
  for (auto const& container : allContainers) {
    data.emplace_back(extractData(container));
  }
  int dims[2] = {count, sizeof(NdArray<real, DIMS...>) / sizeof(real)};
  RecordDataset(name, dims, Flatten(MakeConstSpan(data)), outData);
}

static Real3 ExtractTranslation(TransformRT const& t) {
  return t.GetTranslation();
}

static Real4 ExtractRotation(TransformRT const& t) {
  return t.GetRotation().ToReal4();
}

void articulated::compound::RecordTargetState(
    CArticulatedProps const& articulatedProps,
    CControllerTarget<TimeStep::Current> const& controllerTarget,
    CControllerConstraints const& controllerConstraints,
    entt::registry& reg,
    entt::entity e,
    CRecordingData& outData) {
  // Temporary storage
  MOCHI_FILO_STACK_ALLOCATOR(alloc, sizeof(real) * 512);

  // Record target link transforms as two 2D datasets
  {
    // Compute external link transforms (with root translation).
    DynamicArray<TransformRT> linkTransforms(&alloc);
    linkTransforms.resize_noinit(articulatedProps.numLinks);
    ComLinkTransformsToRootLinkTransforms(
        reg, e, controllerTarget.LinkTransformsCom(), linkTransforms);

    // Record datasets
    RecordDatasetFromContainers(
        MakeConstSpan(linkTransforms), "targetLinkRotations", ExtractRotation, &alloc, outData);
    RecordDatasetFromContainers(
        MakeConstSpan(linkTransforms),
        "targetLinkTranslations",
        ExtractTranslation,
        &alloc,
        outData);
  }

  // Record target joint pose (reduced pose) as a 1D dataset.
  RecordDataset("targetPose", controllerTarget.JointPose(), outData);

  // Record tracking parameters
  {
    auto numConstraints = isize(controllerConstraints.info);
    ColumnVector<real> stiffness(numConstraints, &alloc);
    ColumnVector<real> damping(numConstraints, &alloc);
    ColumnVector<real> saturation(numConstraints, &alloc);
    for (int i = 0; i < numConstraints; ++i) {
      stiffness[i] = controllerConstraints.impl[i].constraint->GetStiffness();
      damping[i] = controllerConstraints.impl[i].constraint->GetDamping();
      saturation[i] = controllerConstraints.impl[i].constraint->GetSaturation();
    }

    // Write tracking params as 1D datasets
    RecordDataset<real>("poseTrackingStiffness", AsConstView(stiffness), outData);
    RecordDataset<real>("poseTrackingDamping", AsConstView(damping), outData);
    RecordDataset<real>("poseTrackingSaturation", AsConstView(saturation), outData);
  }
}

static Real3 ExtractVcom(RigidBodyVel const& vel) {
  return ToReal3(vel.GetVCom());
}

static Real3 ExtractOmega(RigidBodyVel const& vel) {
  return ToReal3(vel.GetOmegaAndVSym().first);
}

static Matrix3x3r ExtractVsym(RigidBodyVel const& vel) {
  return ToNdArraySym3x3(vel.GetOmegaAndVSym().second);
}

void articulated::compound::RecordState(
    CArticulatedReducedPose<TimeStep::Current> const& reducedPose,
    CArticulatedJointVels<TimeStep::Current> const& jointVels,
    CControllerTarget<TimeStep::Previous> const* targetOld,
    CRecordingData& outData) {
  // Record current joint pose (reduced pose) as a 1D dataset.
  RecordDataset("pose", AsConstView(reducedPose.value), outData);

  // Record current joint velocities as three 2D datasets
  MOCHI_FILO_STACK_ALLOCATOR(alloc, sizeof(RigidBodyVel) * 256);
  DynamicArray<RigidBodyVel> jointVelsRaw(&alloc);
  jointVelsRaw.reserve(jointVels.value.size());
  for (auto const& jointVel : jointVels.value) {
    jointVelsRaw.emplace_back(jointVel.value);
  }
  RecordDatasetFromContainers(MakeConstSpan(jointVelsRaw), "vcom", ExtractVcom, &alloc, outData);
  RecordDatasetFromContainers(MakeConstSpan(jointVelsRaw), "omega", ExtractOmega, &alloc, outData);
  RecordDatasetFromContainers(MakeConstSpan(jointVelsRaw), "vsym", ExtractVsym, &alloc, outData);

  // If there is a pose controller, record the old target pose as a 1D dataset.
  if (targetOld) {
    RecordDataset("oldTargetPose", targetOld->JointPose(), outData);
  }
}

void articulated::compound::UpdateVSym(
    ecs::Included<TagArticulatedActor>,
    ecs::CtxGlobal<CSceneTime const> time,
    CArticulatedJointVels<TimeStep::Current>& outJointVels) {
  for (auto& jointVel : outJointVels.value) {
    jointVel.value.UpdateVSymIfDirty(static_cast<real>(time->DeltaTime()));
  }
}

template <typename ContainerT, size_t... DIMS>
static void DecodeDatasetToContainers(
    std::string_view name,
    GroupReader* reader,
    void (*writeData)(NdArray<real, DIMS...> const&, ContainerT&),
    Allocator* tempAllocator,
    Span<ContainerT> outAllContainers,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  auto dims = reader->GetDataSetDimensions(name, error);
  MOCHI_ERROR_RETURN(error);
  auto count = isize(outAllContainers);
  MOCHI_ERROR_IF(dims[0] != count, error, "Dataset size mismatch");
  MOCHI_ERROR_IF(
      dims[1] != sizeof(NdArray<real, DIMS...>) / sizeof(real), error, "Dataset size mismatch");
  MOCHI_ERROR_RETURN(error);
  DynamicArray<NdArray<real, DIMS...>> data(count, tempAllocator);
  reader->ReadDataSet(name, Flatten(MakeSpan(data)), error);
  MOCHI_ERROR_RETURN(error);
  for (int i = 0; i < count; ++i) {
    writeData(data[i], outAllContainers[i]);
  }
}

SkinningParams articulated::compound::CreateSkinningParams(
    entt::registry& reg,
    entt::entity e,
    bool allowUnusedBones,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  // Fetch reference bone transforms. This requires that the rigid actors are currently in their
  // reference pose. In practice, the reference bone transforms are translations between the root
  // and the center of mass of each bone.
  auto const* groupMembers = reg.try_get<CGroupMembers const>(e);
  MOCHI_ERROR_IF(!groupMembers, error, "Not a compound actor");
  MOCHI_ERROR_RETURN(error, {});
  auto numActors = groupMembers->actors.size();
  SkinningParams skinningParams;
  SkeletonReferenceFrames refFrames;
  refFrames.initialWorldFromBone.resize(numActors);
  refFrames.referenceRootFromBone.resize(numActors);
  TransformRT rootInverse =
      Invert(reg.get<CRootTransform const>(groupMembers->actors[0]).worldFromLocal);
  for (int i = 0; i < numActors; i++) {
    TransformRT const& transform =
        reg.get<CRigidState<TimeStep::Current> const>(groupMembers->actors[i]).value;
    refFrames.initialWorldFromBone[i] = transform;
    refFrames.referenceRootFromBone[i] = rootInverse * transform;
  }
  skinningParams.allowUnusedBones = allowUnusedBones;
  skinningParams.referenceFrames = refFrames;
  return skinningParams;
}

void articulated::compound::ProjectDerivedStateGradient(
    ecs::Included<TagArticulatedActor>,
    CArticulatedJacobian const& jacobian,
    CDiffContainerDerivedState const& derivedStateGrad,
    CDiffContainerState& outStateGrad) {
  // Copy joint-space gradient
  outStateGrad = derivedStateGrad.TopRows(outStateGrad.Rows());
  // Add link-space gradient
  outStateGrad += jacobian.value.Transpose() * derivedStateGrad.BottomRows(jacobian.value.Rows());
}

void articulated::compound::ShiftDerivedStateGradient(
    ecs::PartialRegistry<
        CRigidState<TimeStep::Current> const,
        CRigidState<TimeStep::Previous> const> reg,
    ecs::Included<TagArticulatedActor>,
    CArticulatedProps const& props,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CGroupMembers const& members,
    CArticulatedReducedPose<TimeStep::Current> const& currPose,
    CArticulatedReducedPose<TimeStep::Previous> const& prevPose,
    CDiffContainerDerivedState& outDerivedStateGrad) {
  auto const* joints = bodyShape.shape->GetJointsData();
  // Shift joint-space gradient.
  ChainArticulatedGradientDDeltaDOld(
      joints->jointTypes,
      joints->dofInfo,
      poseInfo,
      currPose.value,
      prevPose.value,
      outDerivedStateGrad.TopRows(props.reducedDofsDim));
  // Shift link-space gradient.
  for (int i = 0, offset = props.reducedDofsDim; i < isize(members.actors);
       ++i, offset += RigidSize::kDAll) {
    entt::entity const link = members.actors[i];
    ChainRigidGradientDDeltaDOld(
        reg.get<CRigidState<TimeStep::Current> const>(link).value,
        reg.get<CRigidState<TimeStep::Previous> const>(link).value,
        outDerivedStateGrad.MiddleRows<RigidSize::kDAll>(offset, RigidSize::kDAll));
  }
}

void articulated::compound::TransportGradient(
    ColumnVectorView<real const> delta,
    ColumnVectorView<real> outGradient,
    ecs::Included<TagArticulatedActor>,
    CArticulatedBodyShape const& bodyShape,
    CDofOffset const& dofOffset,
    CActorDofInfo const& dofInfo) {
  auto const* joints = bodyShape.shape->GetJointsData();
  TransportInputOfLieJacobian(
      joints->jointTypes,
      joints->dofInfo,
      delta.MiddleRows(dofOffset.dofsOffset, dofInfo.dofsSize),
      outGradient.MiddleRows(dofOffset.dofsOffset, dofInfo.dofsSize).Transpose());
}

template <GradTarget kGradTarget>
void articulated::compound::ProjectContactForceAdjoints(
    ecs::PartialRegistry<CDofOffset const, CDiffContactGrad<kGradTarget> const> reg,
    ecs::RequiredTag<TagArticulatedActor>,
    CGroupMembers const& groupMembers,
    CArticulatedJacobian const& jacobian,
    CDiffContactGrad<kGradTarget>& outGrad) {
  // Project the per-link adjoints
  for (auto link : groupMembers.actors) {
    if (auto const* linkGrad = reg.template try_get<CDiffContactGrad<kGradTarget> const>(link)) {
      int const offset = reg.template get<CDofOffset const>(link).dofsOffset;
      auto const jac = jacobian.value.MiddleRows<RigidSize::kDAll>(offset, RigidSize::kDAll);
      outGrad += jac.Transpose() * (*linkGrad);
    }
  }
}

#define MOCHI_SPECIALIZE_PROJECT_CONTACT_FORCE_ADJOINTS(kGradTarget)               \
  template void articulated::compound::ProjectContactForceAdjoints(                \
      ecs::PartialRegistry<CDofOffset const, CDiffContactGrad<kGradTarget> const>, \
      ecs::RequiredTag<TagArticulatedActor>,                                       \
      CGroupMembers const&,                                                        \
      CArticulatedJacobian const&,                                                 \
      CDiffContactGrad<kGradTarget>& outGrad);
MOCHI_SPECIALIZE_PROJECT_CONTACT_FORCE_ADJOINTS(GradTarget::Current);
MOCHI_SPECIALIZE_PROJECT_CONTACT_FORCE_ADJOINTS(GradTarget::Previous);
#undef MOCHI_SPECIALIZE_PROJECT_CONTACT_FORCE_ADJOINTS

void articulated::rigid::EntityUpdateJacobian(
    ecs::Included<TagArticulatedLinkActor>,
    ecs::PartialRegistry<CArticulatedJacobian const> reg,
    CArticulatedEntity const& entArticulated,
    CDofOffset const& dofOffset,
    CActorDofInfo const& numDofs,
    CArticulatedRigidJacobian& outJacobian) {
  MOCHI_ASSERT_VERBOSE(numDofs.dofsSize == RigidSize::kDAll);

  auto const& artJacobian = reg.get<CArticulatedJacobian const>(entArticulated.entity).value;
  auto artJacobianBlock = artJacobian.MiddleRows(dofOffset.dofsOffset, numDofs.dofsSize);
  for (int i = 0; i < isize(outJacobian.dofs); i++) {
    outJacobian.value.Col(i) = artJacobianBlock.Col(outJacobian.dofs[i]);
  }
}

// Take the full space DoFs from the articulated and distribute it into the rigid body states of
// the internal rigid actors.
void articulated::rigid::EntitySetSolution(
    ecs::RequiredTag<TagArticulatedLinkActor>,
    CDofOffset const& rigidDofOffset,
    CArticulatedFullPoseRef const& fullPoseRef,
    CRigidState<TimeStep::Current>& outCurrPose) {
  MOCHI_PROFILE_SCOPE();
  auto actorSol = fullPoseRef.value.MiddleRows(rigidDofOffset.poseOffset, RigidSize::kAll);
  mochi::rigid::EntitySetSolution(actorSol, {}, {}, outCurrPose);
}

MOCHI_API void articulated::rigid::EntityPreStep(
    ecs::RequiredTag<TagArticulatedLinkActor>,
    CRigidState<TimeStep::Current> const& currPose,
    CRigidVel<TimeStep::Current>& currVel,
    CRigidState<TimeStep::Previous>& prevPose,
    CRigidVel<TimeStep::Previous>& prevVel) {
  MOCHI_PROFILE_SCOPE();
  // Shift rigid state from current to previous.
  mochi::rigid::EntityIncrementStep({}, {}, currPose, currVel, prevPose, prevVel);
}

void articulated::rigid::EntityPreFirstStage(
    ecs::RequiredTag<TagArticulatedLinkActor>,
    CTimeIntegratorState const& intState,
    CRigidVel<TimeStep::Previous> const& prevVel,
    CIntegrationRigidVels& intVels) {
  MOCHI_PROFILE_SCOPE();
  // Compute differential variables (i.e. velocity) at the beginning of the step.
  mochi::rigid::ComputeVelocityAtStepStart(intState, prevVel, intVels);
}

void articulated::rigid::EntityPreStage(
    ecs::RequiredTag<TagArticulatedLinkActor>,
    ecs::OptionalTag<TagStaticActor> isStatic,
    CDofOffset const& rigidDofOffset,
    CRigidBodyInertia const& rigidInertia,
    CArticulatedFullPoseRef const& fullPoseRef,
    CTimeIntegratorState const& intState,
    CIntegrationRigidVels& intVels,
    CRigidState<TimeStep::StageStart>& stageStartPose,
    CRigidVel<TimeStep::StageStart>& stageStartVel,
    CRootTransform& rootTransform) {
  MOCHI_PROFILE_SCOPE();
  // Pose is an algebraic variable. Set its value at the beginning of the stage based on the
  // articulated full pose.
  auto actorSol = fullPoseRef.value.MiddleRows(rigidDofOffset.poseOffset, RigidSize::kAll);
  mochi::rigid::EntitySetSolution(actorSol, {}, {}, stageStartPose);

  // Root transform is an algebraic variable. Compute its value at the beginning of the stage based
  // on the rigid pose.
  if (!isStatic) {
    // Other islands may concurrently read a static link's CRootTransform, so it must not be written
    // during the island solve, even if the value would be unchanged.
    mochi::rigid::ComputeRootTransformAtStageStart(rigidInertia, stageStartPose, rootTransform);
  }

  // Velocity is a differential variable. Compute its value at the beginning of the stage using
  // integration utilities.
  mochi::rigid::ComputeVelocityAtStageStart(intState, intVels, stageStartVel);
}

MOCHI_API void articulated::rigid::EntityPostStage(
    ecs::RequiredTag<TagArticulatedLinkActor>,
    CConvergenceStatus const& convergence,
    CDofOffset const& rigidDofOffset,
    CTimeIntegratorState const& intState,
    CArticulatedFullPoseRef const& fullPoseRef,
    CRigidState<TimeStep::StageStart> const& stageStartPose,
    CRigidState<TimeStep::Current>& currPose,
    CRigidVel<TimeStep::Current>& currVel,
    CIntegrationRigidVels& intRigidVels) {
  MOCHI_PROFILE_SCOPE();
  if (convergence.stageStatus == ConvergenceStatus::Diverged) {
    // The state of the articulated actor and its internal rigid actors is always in sync due to
    // PostNewSolution systems. The exception is if the solver diverged, in which case the
    // articulated actor was reset to a safe state but its internal rigid actors were not.
    auto actorSol = fullPoseRef.value.MiddleRows(rigidDofOffset.poseOffset, RigidSize::kAll);
    mochi::rigid::EntitySetSolution(actorSol, {}, {}, currPose);
  }

  // Compute velocity (differential variable) at the end of the stage, and push it to the vector
  // containing its value at the end of each stage.
  mochi::rigid::ComputeVelocityAtStageEnd(intState, stageStartPose, currPose, currVel);
  intRigidVels.stages[intState.currentStage].value = currVel.value;
}

void articulated::rigid::EntityPostLastStage(
    ecs::RequiredTag<TagArticulatedLinkActor>,
    ecs::OptionalTag<TagStaticActor> isStatic,
    CRigidBodyInertia const& rigidInertia,
    CDofOffset const& rigidDofOffset,
    CTimeIntegratorState const& intState,
    CArticulatedFullPoseRef const& fullPoseRef,
    CRigidState<TimeStep::Current>& currPose,
    CRigidVel<TimeStep::Current>& currVel,
    CIntegrationRigidVels& intVels,
    CRootTransform& rootTransform) {
  MOCHI_PROFILE_SCOPE();
  // Compute the pose and root transform (algebraic variables) at the end of the time step based on
  // the articulated full pose.
  if (!isStatic) {
    // Other islands may concurrently read a static link's CRootTransform, so it must not be written
    // during the island solve, even if the value would be unchanged.
    mochi::rigid::EntityPostNewSolution(
        fullPoseRef.value, {}, {}, rigidDofOffset, rigidInertia, currPose, rootTransform);
  }

  // Compute velocity (differential variable) at the end of the time step.
  mochi::rigid::ComputeVelocityAtTimeStepEnd(intState, intVels, currVel);
}

void articulated::rigid::EntityPostNewSolution(
    ecs::RequiredTag<TagArticulatedLinkActor>,
    // Other islands may concurrently read a static link's CRootTransform, so it must not be written
    // during the island solve, even if the value would be unchanged.
    ecs::Excluded<TagStaticActor>,
    CRigidBodyInertia const& rigidInertia,
    CDofOffset const& rigidDofOffset,
    CArticulatedFullPoseRef& fullPoseRef,
    CRigidState<TimeStep::Current>& rigidPose,
    CRootTransform& rootTransform) {
  MOCHI_PROFILE_SCOPE();
  mochi::rigid::EntityPostNewSolution(
      fullPoseRef.value, {}, {}, rigidDofOffset, rigidInertia, rigidPose, rootTransform);
}

static void ApplyBoundaryConditions(
    ecs::RequiredTag<TagArticulatedActor>,
    CDofPositionsBC const* inWorldBC,
    CDirichletBC<real>& outWorldBC) {
  if (inWorldBC) {
    outWorldBC.poseIndices = inWorldBC->poseIndices;
    outWorldBC.dofIndices = inWorldBC->dofIndices;
    outWorldBC.poseValues = inWorldBC->poseValues;
    outWorldBC.colValueIndices = inWorldBC->colValueIndices;
  } else {
    outWorldBC.Clear();
  }
}

MOCHI_API void mochi::articulated::compound::PreStepArticulatedBodyActorAsync(
    entt::registry& reg,
    entt::entity e) {
  MOCHI_PROFILE_SCOPE();

  // Apply boundary conditions
  ecs::TryInvokeOnEntity(&ApplyBoundaryConditions, reg, e);
}

namespace mochi::articulated {

void InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<CArticulatedEntity>(reg);
  ecs::RegisterComponent<CArticulatedFullPose>(reg);
  ecs::RegisterComponent<CArticulatedFullPoseRef>(reg);
  ecs::RegisterComponent<CArticulatedInertiaParams>(reg);
  ecs::RegisterComponent<CArticulatedJointFrictionParams>(reg);
  ecs::RegisterComponent<CArticulatedParents>(reg);
  ecs::RegisterComponent<CArticulatedJacobian>(reg);
  ecs::RegisterComponent<CArticulatedJointLimits>(reg);
  ecs::RegisterComponent<CArticulatedCycleJoints>(reg);
  ecs::RegisterComponent<CArticulatedJointPoseInfo>(reg);
  ecs::RegisterComponent<CArticulatedJointTransforms<TimeStep::Current>>(reg);
  ecs::RegisterComponent<CArticulatedJointTransforms<TimeStep::Previous>>(reg);
  ecs::RegisterComponent<CArticulatedJointTransforms<TimeStep::StageStart>>(reg);
  ecs::RegisterComponent<CArticulatedJointVels<TimeStep::Current>>(reg);
  ecs::RegisterComponent<CArticulatedJointVels<TimeStep::Previous>>(reg);
  ecs::RegisterComponent<CArticulatedJointVels<TimeStep::StageStart>>(reg);
  ecs::RegisterComponent<CArticulatedLinkTransforms<TimeStep::Current>>(reg);
  ecs::RegisterComponent<CArticulatedLinkTransforms<TimeStep::Previous>>(reg);
  ecs::RegisterComponent<CArticulatedLinkTransforms<TimeStep::StageStart>>(reg);
  ecs::RegisterComponent<CArticulatedProps>(reg);
  ecs::RegisterComponent<CArticulatedReducedPose<TimeStep::Current>>(reg);
  ecs::RegisterComponent<CArticulatedReducedPose<TimeStep::Previous>>(reg);
  ecs::RegisterComponent<CArticulatedReducedPose<TimeStep::StageStart>>(reg);
  ecs::RegisterComponent<CArticulatedRestTransforms>(reg);
  ecs::RegisterComponent<CArticulatedRigidJacobian>(reg);
  ecs::RegisterComponent<CArticulatedSkinExportParams>(reg);
  ecs::RegisterComponent<CArticulatedSkinningData>(reg);
  ecs::RegisterComponent<CBoneHandles>(reg);
  ecs::RegisterComponent<CIntegrationArticulatedReducedPose>(reg);
  ecs::RegisterComponent<CIntegrationArticulatedJointVels>(reg);
  ecs::RegisterComponent<CTransmissions>(reg);

  // Post-restore fixup: update derived state but not links, as they are state-captured.
  capture::RegisterPostRestoreSystem<ecs::policy::AllowFullRegistryAccess>(
      &UpdateDerivedStateFromPose</*kUpdateLinks*/ false>, reg);
}

real GetActorMass(entt::registry const& reg, entt::entity actor) {
  MOCHI_ASSERT(reg.all_of<TagArticulatedActor>(actor), "Expected articulated actor.");
  auto const& groupMembers = reg.get<CGroupMembers>(actor);

  real totalMass = 0_r;
  for (auto const& link : groupMembers.actors) {
    totalMass += reg.get<CRigidBodyInertia>(link).GetMass();
  }
  return totalMass;
}
} // namespace mochi::articulated
