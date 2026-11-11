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

#include "mochi_common_components.h"
#include "mochi_ecs.h"

#include <mochi_core/articulated_body/articulated_body.h>
#include <mochi_core/utils/dynamic_array.h>

#include <concepts>
#include <utility>
#include <vector>

namespace mochi {

// Forwards
struct CArticulatedJacobian;

/***************************************************************************************************
  Implementation classes to control links and joints of an articulated body. They connect link and
  joint data to the actual implementation of soft constraints.
  * They are classified based on the "slice" of the pose they control (which can be link
  translation, link rotation, or joint transform) and the agent type.
  * A base class defines the API, which includes constraint initialization, target update, and force
  query.
*/

// Virtual class that defines the API for a pose slice controller.
class PoseConstraintAPI {
 protected:
  virtual ~PoseConstraintAPI() = default;

 public:
  PoseConstraintAPI() = default;
  MOCHI_DECLARE_COPY(PoseConstraintAPI);
  MOCHI_DECLARE_MOVE(PoseConstraintAPI);

  virtual Constraint* Create(
      Scene* scene,
      ActorHandle articulated,
      ActorHandle link,
      ArticulatedJointType type,
      ArticulatedDofInfo const& dofInfo,
      ArticulatedPoseInfo const& poseInfo,
      Span<real const> pose,
      TransformRT const& linkTransform,
      int id,
      PoseTrackingParams const& params) const = 0;

  virtual void SetTarget(
      Constraint* constraint,
      ArticulatedPoseInfo const& poseInfo,
      Span<real const> pose,
      TransformRT const& linkTransformCom) const = 0;

  virtual void AddForce(
      ArticulatedJacobian const& jacobian,
      int link,
      Constraint const* constraint,
      ColumnVectorView<real> outForce,
      Error& error) const = 0;
};

template <typename T>
concept PoseConstraintT = std::derived_from<T, PoseConstraintAPI>;

// Link position tracking
class LinkPosConstraint final : public PoseConstraintAPI {
 public:
  static constexpr PoseConstraintType kConstraintType = PoseConstraintType::LinkTranslation;

  using PoseConstraintAPI::PoseConstraintAPI;

  Constraint* Create(
      Scene* scene,
      ActorHandle articulated,
      ActorHandle link,
      ArticulatedJointType type,
      ArticulatedDofInfo const& dofInfo,
      ArticulatedPoseInfo const& poseInfo,
      Span<real const> pose,
      TransformRT const& linkTransform,
      int id,
      PoseTrackingParams const& params) const override;

  void SetTarget(
      Constraint* constraint,
      ArticulatedPoseInfo const& poseInfo,
      Span<real const> pose,
      TransformRT const& linkTransformCom) const override;

  void AddForce(
      ArticulatedJacobian const& jacobian,
      int link,
      Constraint const* constraint,
      ColumnVectorView<real> outForce,
      Error& error) const override;
};

// Link rotation tracking
class LinkRotConstraint final : public PoseConstraintAPI {
 public:
  static constexpr PoseConstraintType kConstraintType = PoseConstraintType::LinkRotation;

  using PoseConstraintAPI::PoseConstraintAPI;

  Constraint* Create(
      Scene* scene,
      ActorHandle articulated,
      ActorHandle link,
      ArticulatedJointType type,
      ArticulatedDofInfo const& dofInfo,
      ArticulatedPoseInfo const& poseInfo,
      Span<real const> pose,
      TransformRT const& linkTransform,
      int id,
      PoseTrackingParams const& params) const override;

  void SetTarget(
      Constraint* constraint,
      ArticulatedPoseInfo const& poseInfo,
      Span<real const> pose,
      TransformRT const& linkTransformCom) const override;

  void AddForce(
      ArticulatedJacobian const& jacobian,
      int link,
      Constraint const* constraint,
      ColumnVectorView<real> outForce,
      Error& error) const override;
};

// Joint tracking for articulated agents
class JointConstraint final : public PoseConstraintAPI {
 public:
  static constexpr PoseConstraintType kConstraintType = PoseConstraintType::Joint;

  using PoseConstraintAPI::PoseConstraintAPI;

  Constraint* Create(
      Scene* scene,
      ActorHandle articulated,
      ActorHandle link,
      ArticulatedJointType type,
      ArticulatedDofInfo const& dofInfo,
      ArticulatedPoseInfo const& poseInfo,
      Span<real const> pose,
      TransformRT const& linkTransform,
      int id,
      PoseTrackingParams const& params) const override;

  void SetTarget(
      Constraint* constraint,
      ArticulatedPoseInfo const& poseInfo,
      Span<real const> pose,
      TransformRT const& linkTransformCom) const override;

  void AddForce(
      ArticulatedJacobian const& jacobian,
      int link,
      Constraint const* constraint,
      ColumnVectorView<real> outForce,
      Error& error) const override;
};

/***************************************************************************************************
  Struct to store the implementation details of a constraint. It complements PoseConstraintInfo.
*/

struct PoseConstraintImpl {
  PoseConstraintImpl(Constraint* c) : constraint(c) {}
  Constraint* constraint;
};

/***************************************************************************************************
  Component to control a full slice of pose, templatized by the constraint type
  * It stores a vector of active constraints for the particular slice.
*/

template <PoseConstraintT C>
struct CPoseSliceController : public NoCopy {
  CPoseSliceController(
      Scene* scene,
      ActorHandle articulated,
      Span<ActorHandle const> links,
      Span<ArticulatedJointType const> jointTypes,
      Span<Real3 const> /* jointAxes */,
      Span<ArticulatedDofInfo const> dofInfo,
      Span<ArticulatedPoseInfo const> poseInfo,
      Span<int const> parents,
      Span<real const> pose,
      Span<TransformRT const> linkTxs,
      Span<PoseTrackingParams const> paramsIn) {
    // Create default params if paramsIn is empty
    PoseTrackingParams const paramsZero{};
    auto const params = paramsIn.empty() ? MakeSingletonConstSpan(paramsZero) : paramsIn;

    // Validate number of params
    int const numLinks = isize(links);
    int const numParams = isize(params);
    MOCHI_ASSERT_VERBOSE(numParams == 1 || numParams == numLinks, "Invalid number of params");
    int const stride = numParams == 1 ? 0 : 1;

    // Create the constraints
    info.reserve(numLinks);
    impl.reserve(numLinks);
    for (int i = 0; i < numLinks; ++i) {
      auto const& p = params[stride * i];
      auto* constraint = constraintType.Create(
          scene,
          articulated,
          links[i],
          jointTypes[i],
          dofInfo[i],
          poseInfo[i],
          pose,
          linkTxs[i],
          i,
          p);
      if (constraint) {
        info.push_back(
            PoseConstraintInfo{
                constraint->GetHandle(), constraintType.kConstraintType, i, parents[i]});
        impl.emplace_back(constraint);
      }
    }
  }

  C constraintType;
  std::vector<PoseConstraintInfo> info;
  std::vector<PoseConstraintImpl> impl;
};

using CLinkPosController = CPoseSliceController<LinkPosConstraint>;
using CLinkRotController = CPoseSliceController<LinkRotConstraint>;
using CJointController = CPoseSliceController<JointConstraint>;

/***************************************************************************************************
  Components to store joint-pose and link-transform targets for the controller. The two targets may
  not be consistent (i.e., the link transforms are not constrained to the feasible joint space),
  hence the need to store both.
*/

class ControllerTarget {
 protected:
  ColumnVector<real> _jointPose;
  DynamicArray<TransformRT> _linkTransformsCom;

 public:
  ControllerTarget() = default;
  explicit ControllerTarget(int numDofs, int numLinks) {
    _jointPose.Reset(numDofs);
    _linkTransformsCom.resize(numLinks);
  }

  // Accessors that prevent direct access to the underlying data.
  ColumnVectorView<real const> JointPose() const {
    return _jointPose;
  }
  Span<TransformRT const> LinkTransformsCom() const {
    return _linkTransformsCom;
  }

  // Both joint and link targets must be set together, making it explicit that they may be
  // inconsistent.
  void Set(ColumnVectorView<real const> jointPose, Span<TransformRT const> linkTransformsCom) {
    MOCHI_ASSERT_VERBOSE(_jointPose.size() == jointPose.size(), "Invalid joint pose size");
    MOCHI_ASSERT_VERBOSE(
        _linkTransformsCom.size() == linkTransformsCom.size(), "Invalid link size");
    _jointPose = jointPose;
    _linkTransformsCom = linkTransformsCom;
  }

  MOCHI_STRUCT_BEGIN(mochi::ControllerTarget);
  MOCHI_FIELD(_jointPose);
  MOCHI_FIELD(_linkTransformsCom);
  MOCHI_STRUCT_END();
};

template <TimeStep kStep>
class CControllerTarget : public ControllerTarget, public NoCopy {
 public:
  using ControllerTarget::ControllerTarget;

  // Copy the target state from another time step's target.
  template <TimeStep kOther>
  void CopyFrom(CControllerTarget<kOther> const& other) {
    static_cast<ControllerTarget&>(*this) = other;
  }

  // Swap the target state with another time step's target.
  template <TimeStep kOther>
  void SwapWith(CControllerTarget<kOther>& other) {
    std::swap(static_cast<ControllerTarget&>(*this), static_cast<ControllerTarget&>(other));
  }

  MOCHI_TEMPLATE_BEGIN(mochi::CControllerTarget, kStep);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_BASE_CLASS(ControllerTarget);
  MOCHI_TEMPLATE_END();
};

/***************************************************************************************************
  Component to store all the active constraints of a pose controller
*/

struct CControllerConstraints : public NoCopy {
  CControllerConstraints(
      std::vector<PoseConstraintInfo>&& infoIn,
      std::vector<PoseConstraintImpl>&& implIn)
      : info(std::move(infoIn)), impl(std::move(implIn)) {}
  std::vector<PoseConstraintInfo> info;
  std::vector<PoseConstraintImpl> impl;
};

/***************************************************************************************************
  Motion prescribed for the targets of a pose controller over the next step.

  Between steps, the Previous and Current targets hold the targets at the start and at the end of
  the last step. Setting the targets prescribes the end targets of the next step, even if they are
  unchanged: the first set since the last step sets the Previous targets to the end targets of the
  last step, and sets newTargetApplied. A target velocity
  (articulated::compound::SetTargetJointVelocities) prescribes the start targets of the next step.
  It is applied at the beginning of the step, once the time step is known: the start targets are the
  end targets moved back by the velocity over the time step.
*/

struct CControllerPrescribedMotion : NoCopy {
  CControllerPrescribedMotion() = default;
  explicit CControllerPrescribedMotion(int numDofs) {
    targetVelocity.Reset(numDofs);
  }

  ColumnVector<real> targetVelocity;
  bool newTargetApplied = false; // The targets were set since the last step
  bool hasVelocity = false; // A target velocity was set since the last step

  MOCHI_STRUCT_BEGIN(mochi::CControllerPrescribedMotion);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_FIELD(targetVelocity);
  MOCHI_FIELD(newTargetApplied);
  MOCHI_FIELD(hasVelocity);
  MOCHI_STRUCT_END();
};

/***************************************************************************************************
  Controller target gradient ownership.

  Tracks which forward function last set each controller target, so that backward functions only
  claim the gradient containers they own. This prevents double-counting when multiple forward
  functions set controller targets on the same entity.
*/

// Tracks which forward function last set a controller target.
enum class TargetOwner {
  ControllerInit, // AddPoseController / InitializeDifferentiablePoseController
  PoseFromJoints, // SetArticulatedPoseFromJoints
  PoseFromLinks, // SetArticulatedPoseFromLinks
  JointVelocities, // SetArticulatedJointVelocities
  ResetTargetPose, // ResetArticulatedTargetPose
  ResetTargetLinkTransforms, // ResetArticulatedTargetLinkTransforms
  TargetPose, // SetArticulatedTargetPose
  TargetLinkTransforms, // SetArticulatedTargetLinkTransforms
  TargetVelocity // SetArticulatedTargetVelocity
};

} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::TargetOwner);
MOCHI_ENUM_ITEM(ControllerInit);
MOCHI_ENUM_ITEM(PoseFromJoints);
MOCHI_ENUM_ITEM(PoseFromLinks);
MOCHI_ENUM_ITEM(JointVelocities);
MOCHI_ENUM_ITEM(ResetTargetPose);
MOCHI_ENUM_ITEM(ResetTargetLinkTransforms);
MOCHI_ENUM_ITEM(TargetPose);
MOCHI_ENUM_ITEM(TargetLinkTransforms);
MOCHI_ENUM_ITEM(TargetVelocity);
MOCHI_ENUM_END();

namespace mochi {

// ECS component tracking which forward function last set each controller target. The new pose
// target is controlled directly. The old pose target is controlled by the new pose target and the
// target velocity.
// Captured in state so that BackPropagate restores correct per-step ownership.
// The step fields record the simulation step at which each target was last set by a forward
// function. During back-propagation, comparing these against the current step counter determines
// whether gradients should propagate across steps (target inherited) or stay for a backward
// function to consume (target explicitly set).
struct CTargetOwners {
  TargetOwner newPoseOwner = TargetOwner::ControllerInit;
  TargetOwner oldPoseOwner = TargetOwner::ControllerInit;
  TargetOwner velOwner = TargetOwner::ControllerInit;
  uint64_t newPoseStep = 0;
  uint64_t oldPoseStep = 0;
  uint64_t velStep = 0;
  // velStep must be equal to or earlier than oldPoseStep. Differentiability does not allow setting
  // the velocity target without setting the pose target in the same step. A warning is issued if
  // the condition does not hold.

  MOCHI_STRUCT_BEGIN(mochi::CTargetOwners);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_FIELD(newPoseOwner);
  MOCHI_FIELD(oldPoseOwner);
  MOCHI_FIELD(velOwner);
  MOCHI_FIELD(newPoseStep);
  MOCHI_FIELD(oldPoseStep);
  MOCHI_FIELD(velStep);
  MOCHI_STRUCT_END();
};

/***************************************************************************************************
  Systems of the pose controller
*/

namespace controller {

// System to set the current targets of a pose controller, and report in outChanged whether they
// changed. The first set since the last step, even if unchanged, starts the next step at the end
// targets of the last step.
void SetCurrentTarget(
    Span<real const> pose,
    Span<TransformRT const> linkTransformsCom,
    bool& outChanged,
    CControllerTarget<TimeStep::Current>& outTarget,
    CControllerTarget<TimeStep::Previous>& outTargetOld,
    CControllerPrescribedMotion& outMotion);

// System to set the targets of a pose controller slice.
template <PoseConstraintT C>
void SetTargets(
    Span<real const> pose,
    Span<TransformRT const> linkTransformsCom,
    Span<ArticulatedPoseInfo const> poseInfo,
    CPoseSliceController<C>& outSlice);

// System to add the force of a pose controller slice.
template <PoseConstraintT C>
void AddForces(
    ColumnVectorView<real> outForce,
    Error& error,
    CArticulatedJacobian const& jacobian,
    CPoseSliceController<C> const& slice);

void InitializeOnce(entt::registry& reg);

} // namespace controller

} // namespace mochi
