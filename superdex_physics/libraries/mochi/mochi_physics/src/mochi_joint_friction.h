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

#include "mochi_articulated_body.h"
#include "mochi_common_components.h"
#include "mochi_differentiable.h"
#include "mochi_ecs.h"

#include <mochi_core/articulated_body/articulated_body_params.h>
#include <mochi_core/utils/span.h>

#include <algorithm>

// Elasto-plastic dry friction of the joints of articulated actors (see
// ArticulatedJointFrictionParams and mochi_core/articulated_body/joint_friction.h).

namespace mochi {

/**
 * @brief Friction anchors of the joints of an articulated actor, for their elasto-plastic dry
 * friction, as a reduced pose: the anchor of each joint has the layout of its pose
 * (CArticulatedJointPoseInfo). They are the anchors that the last step used.
 *
 * @details Empty means without friction preload: after creation, after the joint positions are set
 * (ClearJointFrictionAnchors), and while the actor has no joint dry friction. At the start of each
 * step, PrepareJointFrictionAnchors advances them by radial return to the step-start pose, the end
 * pose of the last step, or anchors the joints there if they are empty. They stay fixed during the
 * stages of a step. A diverged stage, which resets the actor to the step-start pose, anchors the
 * joints there (articulated HandleSolverDivergence), which releases the preload.
 *
 * @note They are captured: they reproduce the completed step's forces (solve-free assembly and
 * back-propagation), and resume forward simulation.
 */
struct CArticulatedJointFrictionAnchors : public ArticulatedPose, NoCopy {
  MOCHI_STRUCT_BEGIN(mochi::CArticulatedJointFrictionAnchors);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_BASE_CLASS(ArticulatedPose);
  MOCHI_STRUCT_END();
};

namespace articulated {

// Whether a joint has dry friction.
[[nodiscard]] inline bool HasDryFriction(ArticulatedJointFrictionParams const& friction) {
  return friction.coulomb + friction.stictionExtra != 0_r;
}

// Whether any joint of an articulated actor has dry friction.
[[nodiscard]] inline bool HasDryFriction(CArticulatedJointFrictionParams const& friction) {
  return std::ranges::any_of(friction, [](auto const& fp) { return HasDryFriction(fp); });
}

// Whether any of the entities is an articulated actor with joint dry friction. Only their assembly
// reads SaturationHessianParams::jointFriction and their joint friction anchors.
[[nodiscard]] bool HasJointDryFriction(
    entt::registry const& reg,
    Span<entt::entity const> entities);

} // namespace articulated

namespace articulated::compound {

// Stores the joint friction parameters of an articulated actor, without the deprecated falloffVel.
// Returns whether they changed. The parameters must be valid (see ValidateFriction).
bool StoreJointFrictionParams(
    Span<ArticulatedJointFrictionParams const> friction,
    CArticulatedJointFrictionParams& outParams);

// System to prepare the joint friction anchors of an articulated actor at the start of a step,
// after the previous state is set (see CArticulatedJointFrictionAnchors): anchors empty ones at the
// step-start pose, and advances the others by radial return to it, except in a solve-free assembly
// (TagAssemblyPreserveHistory), which keeps the anchors the last step used. Joints without dry
// friction keep theirs, and actors without joint dry friction keep none.
void PrepareJointFrictionAnchors(
    ecs::OptionalCtxGlobal<TagAssemblyPreserveHistory const> preserveHistory,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CArticulatedJointFrictionParams const& frictionParams,
    CArticulatedReducedPose<TimeStep::Previous> const& prevPose,
    CArticulatedJointFrictionAnchors& inOutAnchors);

// Completes the last step's advance of the friction anchors of an articulated actor whose friction
// parameters change, with the parameters that step used, if it keeps anchors: radial return to the
// current pose with the last presliding displacement, zero for joints without dry friction, whose
// anchors are not advanced (see PrepareJointFrictionAnchors). Each joint keeps its displacement
// from the anchor, and a joint whose dry friction is enabled starts without preload. Call it only
// when the parameters change, so that solve-free assemblies keep reproducing the last step.
void SettleJointFrictionAnchors(
    Span<ArticulatedJointFrictionParams const> lastParams,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CArticulatedReducedPose<TimeStep::Current> const& pose,
    CArticulatedJointFrictionAnchors& inOutAnchors);

// System to drop the joint friction preload of an articulated actor, whose joint positions are set:
// the next step anchors its joints at its start pose. Other changes of its history, such as its
// velocity or root transform, keep it: the anchors are joint positions.
void ClearJointFrictionAnchors(CArticulatedJointFrictionAnchors& outAnchors);

/*
 * Back-propagation of the joint friction anchors (see CArticulatedJointFrictionAnchors). Step k,
 * from q_{k-1} to q_k, reads the anchors a_{k-1} = R(q_{k-1}, a_{k-2}), advanced at its start
 * (PrepareJointFrictionAnchors) from those of step k - 1 to its start pose, or a_{k-1} = q_{k-1}
 * without them. CDiffJointFrictionAnchorGrad carries the adjoint of a_{k-1} from the
 * back-propagation of step k + 1. Back-propagating step k adds:
 * - Through the solve, after it (AccumulateJointFrictionAnchorSolveAdjoints): (dr_k/da_{k-1})^T z,
 *   the dependence of the solve on a_{k-1}, with z = -CDiffForceGrad.
 * - After restoring the old state (HandOffJointFrictionAnchorAdjoints): the adjoint of a_{k-1} back
 *   through the advance R, to the old pose q_{k-1} (CDiffContactGrad<Previous>) and to a_{k-2},
 *   which CDiffJointFrictionAnchorGrad carries to the back-propagation of step k - 1.
 */
void AccumulateJointFrictionAnchorSolveAdjoints(
    ecs::RequiredTag<TagArticulatedActor>,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CArticulatedJointFrictionParams const& frictionParams,
    CTimeIntegratorState const& intState,
    CArticulatedReducedPose<TimeStep::Current> const& currPose,
    CArticulatedJointFrictionAnchors const& anchors,
    CDiffForceGrad const& forceGrad,
    CDiffJointFrictionAnchorGrad& outAnchorGrad);

void HandOffJointFrictionAnchorAdjoints(
    ecs::RequiredTag<TagArticulatedActor>,
    CArticulatedBodyShape const& bodyShape,
    CArticulatedJointPoseInfo const& poseInfo,
    CArticulatedJointFrictionParams const& frictionParams,
    CArticulatedReducedPose<TimeStep::Current> const& currPose,
    CArticulatedJointFrictionAnchors const& anchors,
    CDiffJointFrictionAnchorGrad& inOutAnchorGrad,
    CDiffContactGrad<GradTarget::Previous>& outGradPrev);

} // namespace articulated::compound

} // namespace mochi
