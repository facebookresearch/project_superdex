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

#include <mochi_physics/diffsim/mochi_diffsim_types.h>
#include <mochi_physics/mochi_physics.h>

#include <mochi_core/articulated_body/articulated_body_params.h>
#include <mochi_core/utils/transform_rt.h>

namespace mochi::diffsim {

MOCHI_API void MakeSceneDifferentiable(Scene* scene, Error& error);

[[nodiscard]] MOCHI_API BackPropagationSolverParams
GetBackPropagationSolverParams(Scene const* scene, Error& error);

MOCHI_API void SetBackPropagationSolverParams(
    Scene* scene,
    BackPropagationSolverParams const& params,
    Error& error);

[[nodiscard]] MOCHI_API BackPropagationSceneStats
GetBackPropagationSceneStats(Scene const* scene, Error& error);

MOCHI_API void ResetBackPropagation(Scene* scene, Error& error);

MOCHI_API void
PrepareBackPropagate(Scene* scene, StateHandle stateNew, StateHandle stateOld, Error& error);

MOCHI_API void BackPropagate(Scene* scene, Error& error);

MOCHI_API void GetStepJacobian(
    Scene* scene,
    StateHandle stateNew,
    StateHandle stateCurr,
    StateHandle stateOld,
    Span<real> outJacCurr,
    Span<real> outJacOld,
    Error& error);

MOCHI_API void
ConvertRigidGradientLieToRotationVector(TransformRT const& state, Span<real> outGrad, Error& error);

MOCHI_API void ConvertRigidGradientLieToQuaternion(
    TransformRT const& state,
    Span<real const> inGrad,
    Span<real> outGrad,
    Error& error);

MOCHI_API void ConvertArticulatedGradientLieToRotationVector(
    Actor const* actor,
    Span<real const> pose,
    Span<real> outGrad,
    Error& error);

MOCHI_API void ConvertRigidGradientQuaternionToLie(
    TransformRT const& state,
    Span<real const> inGrad,
    Span<real> outGrad,
    Error& error);

MOCHI_API void ConvertArticulatedGradientRotationVectorToLie(
    Actor const* actor,
    Span<real const> pose,
    Span<real> outGrad,
    Error& error);

MOCHI_API void
GetCenterOfMassTransformBackward(Actor* actor, Span<real const> gradOutput, Error& error);

MOCHI_API void GetRootTransformBackward(Actor* actor, Span<real const> gradOutput, Error& error);

MOCHI_API void
GetContactForceWorldBackward(Actor* actor, Span<real const> gradOutput, Error& error);

MOCHI_API void GetContactForceFromActorWorldBackward(
    Actor* actor,
    Actor const* other,
    Span<real const> gradOutput,
    Error& error);

MOCHI_API void GetArticulatedPoseBackward(Actor* actor, Span<real const> gradOutput, Error& error);

MOCHI_API void
SetArticulatedTargetPoseBackward(Actor const* actor, Span<real> outGradTargetPose, Error& error);

MOCHI_API void
SetArticulatedPoseFromJointsBackward(Actor const* actor, Span<real> outGradPose, Error& error);

MOCHI_API void SetArticulatedTargetVelocityBackward(
    Actor const* actor,
    Span<real> outGradTargetVelocity,
    Error& error);

MOCHI_API void SetArticulatedJointVelocitiesBackward(
    Actor const* actor,
    Span<real> outGradVelocities,
    Error& error);

MOCHI_API void SetExternalForcesOnDofsBackward(
    Actor const* actor,
    Span<int const> dofIndices,
    Span<real> outGradForceValues,
    Error& error);

MOCHI_API void SetVelocityBackward(
    Actor const* actor,
    Span<real> outGradLinearVel,
    Span<real> outGradAngularVel,
    Error& error);

MOCHI_API void
SetCenterOfMassTransformBackward(Actor const* actor, Span<real> outGradTransform, Error& error);

} // namespace mochi::diffsim
