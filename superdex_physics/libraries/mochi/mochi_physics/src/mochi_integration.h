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
#include "mochi_ecs.h"
#include "mochi_rigid.h"
#include "mochi_rod_pose.h"

#include <mochi_core/articulated_body/articulated_body.h>
#include <mochi_core/element_operations/fem_rod.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/rodrigues_utils.h>

#include <algorithm>
#include <utility>

enum class TimeTarget { StepStart = 0, StageStart = 1, StepEnd = 2 };

/*
 * Utilities to evaluate differential variables at the beginning of the time step ('StepStart'), at
 * the beginning of each time integration stage ('StageStart') and at the end of the time step
 * ('StepEnd').
 */

namespace mochi::integration {

// Joint-layout metadata needed to time-integrate an articulated reduced pose. Bundles the spans
// that the articulated-pose @ref AddWeightedDifferences overload forwards to the mochi_core
// articulated helpers.
struct ArticulatedIntegrationMetadata {
  Span<ArticulatedJointType const> jointTypes;
  Span<ArticulatedDofInfo const> dofInfo;
  Span<ArticulatedPoseInfo const> poseInfo;
};

namespace details {

/// @brief Return a reference to the coefficients to compute the solution at the target time, i.e.
/// - 'alpha' for the solution at the beginning of the step.
/// - 'aTilde' for the solution at the beginning of the stage.
/// - 'bTilde' for the solution at the end of the step.
template <TimeTarget kTargetTime>
auto const& GetCoefficients(CTimeIntegratorState const& intState) {
  if constexpr (kTargetTime == TimeTarget::StepStart) {
    MOCHI_ASSERT_VERBOSE(intState.alpha.size() == intState.numSteps);
    return intState.alpha;
  } else if constexpr (kTargetTime == TimeTarget::StageStart) {
    MOCHI_ASSERT_VERBOSE(intState.aTilde.size() == intState.currentStage);
    return intState.aTilde;
  } else {
    static_assert(kTargetTime == TimeTarget::StepEnd, "Unexpected target time");
    MOCHI_ASSERT_VERBOSE(intState.bTilde.size() == intState.numStages);
    return intState.bTilde;
  }
}

/// @brief Resize the per-stage solution storage to the integrator's stage count, initializing any
/// new entries with the correct number of DoFs. Does not touch the previous-step history.
template <typename PreviousT, typename IntegrationT>
void ResizeIntegrationStages(
    CTimeIntegratorState const& intState,
    PreviousT const& previous,
    IntegrationT& integration) {
  while (integration.stages.size() < intState.numStages) {
    // Initialize with the correct number of DoFs.
    integration.stages.emplace_back(previous.value);
  }
  while (integration.stages.size() > intState.numStages) {
    integration.stages.pop_back();
  }
  MOCHI_ASSERT_VERBOSE(integration.stages.size() == intState.numStages);
}

/// @brief Perform auxiliary time-integration operations required for all actors at the beginning of
/// the time step.
template <typename PreviousT, typename IntegrationT>
void RunPreStep(
    CTimeIntegratorState const& intState,
    PreviousT const& previous,
    IntegrationT& integration) {
  MOCHI_ASSERT( // "+1" since "previous" is not in "prevSteps" yet.
      isize(integration.prevSteps) + 1 >= intState.numSteps,
      "Insufficient previous steps for the time integrator.");

  // Switch previous solutions one time step back.
  for (int i = isize(integration.prevSteps) - 1; i - 1 >= 0; --i) {
    std::swap(integration.prevSteps[i], integration.prevSteps[i - 1]);
  }

  // Insert solution from the previous time step.
  if (integration.prevSteps.empty()) {
    integration.prevSteps.emplace_back(previous.value);
  } else {
    // For cold start, push the oldest solution (now at the front) back.
    if (integration.prevSteps.size() < intState.numSteps) {
      auto copy = integration.prevSteps[0].value; // In case emplace_back reallocates
      integration.prevSteps.emplace_back(std::move(copy));
    }
    integration.prevSteps[0].value = previous.value;
  }

  // Remove unnecessary (and potentially obsolete) previous time steps.
  while (integration.prevSteps.size() > intState.numSteps) {
    integration.prevSteps.pop_back();
  }
  MOCHI_ASSERT_VERBOSE(integration.prevSteps.size() == intState.numSteps);
}

/// The following functions compute conceptually:
/// out = base + sumi coeffs[i] * (vals[i] - base).

/// @brief Implementation for Euclidean-space vectors.
template <typename CoeffsT, ValueContainer T>
void AddWeightedDifferences(
    std::monostate /* unused */,
    ColumnVectorView<real const> base,
    Span<T const> vals,
    CoeffsT const& coeffs,
    Int2 range,
    ColumnVectorView<real> out) {
  out = base;
  for (int s = range[0]; s < range[1]; ++s) {
    out += coeffs[s] * (vals[s].value - base);
  }
}

/// @brief Implementation for TransformRT.
template <typename CoeffsT>
void AddWeightedDifferences(
    std::monostate /* unused */,
    TransformRT const& base,
    Span<TransformRTContainer const> vals,
    CoeffsT const& coeffs,
    Int2 range,
    TransformRT& out) {
  Quaternion qBaseConj = base.GetRotation().GetConjugate();
  Vec4r deltaPos = {}; // Weighted-average delta pos as 3D vector.
  Vec4r deltaRot = {}; // Weighted-average delta rot as 3D rotation vector.
  for (int s = range[0]; s < range[1]; ++s) {
    deltaPos += coeffs[s] * (vals[s].value.VGetTranslation() - base.VGetTranslation());
    deltaRot += coeffs[s] * (vals[s].value.GetRotation() * qBaseConj).VToRotationVector();
  }
  out.SetTranslation(base.VGetTranslation() + deltaPos);
  out.SetRotation(Normalize(Quaternion::FromRotationVector(deltaRot) * base.GetRotation()));
}

/// @brief Implementation for RigidBodyVel.
template <typename CoeffsT>
void AddWeightedDifferences(
    std::monostate /* unused */,
    RigidBodyVel const& base,
    Span<RigidBodyVelContainer const> vals,
    CoeffsT const& coeffs,
    Int2 range,
    RigidBodyVel& out) {
  ColumnVector<real, RigidBodyVel::kRawSize> baseVector;
  base.ToRawValues(baseVector);
  ColumnVector<real, RigidBodyVel::kRawSize> deltaVector = {}; // Weighted-average delta.
  for (int s = range[0]; s < range[1]; ++s) {
    ColumnVector<real, RigidBodyVel::kRawSize> valVector;
    vals[s].value.ToRawValues(valVector);
    deltaVector += coeffs[s] * (valVector - baseVector);
  }
  // Add the delta to the base and store in the result.
  baseVector += deltaVector;
  out.FromRawValues(baseVector);
}

/// @brief Implementation for articulated pose.
template <typename CoeffsT>
void AddWeightedDifferences(
    ArticulatedIntegrationMetadata const& metadata,
    ColumnVectorView<real const> base,
    Span<ArticulatedPose const> vals,
    CoeffsT const& coeffs,
    Int2 range,
    ColumnVectorView<real> out) {
  // Stack memory for up to 500 reduced DoFs
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 2 * 500 * sizeof(real));
  int const reducedDofsSize = articulated::GetReducedDofsSize(metadata.dofInfo);
  ColumnVector<real> deltaAll(reducedDofsSize, &allocator);
  deltaAll.SetZero();
  ColumnVector<real> delta(reducedDofsSize, &allocator);
  for (int s = range[0]; s < range[1]; ++s) {
    articulated::ComputeLieDeltaReducedPose(
        metadata.jointTypes, metadata.dofInfo, metadata.poseInfo, base, vals[s].value, delta);
    deltaAll += coeffs[s] * delta;
  }
  articulated::AddLieDeltaToReducedPose(
      metadata.jointTypes, metadata.dofInfo, metadata.poseInfo, base, deltaAll, out);
  articulated::NormalizeQuaternions(metadata.jointTypes, metadata.poseInfo, out);
}

/// @brief Implementation for rod pose (displacement-twist + frame axes).
template <typename CoeffsT>
void AddWeightedDifferences(
    Span<Real3 const> meshNodes,
    RodPose const& base,
    Span<RodPoseContainer const> vals,
    CoeffsT const& coeffs,
    Int2 range,
    RodPose& out) {
  // Euclidean combination of displacement+twist DoFs.
  out.displacements = base.displacements;
  for (int s = range[0]; s < range[1]; ++s) {
    out.displacements += coeffs[s] * (vals[s].value.displacements - base.displacements);
  }

  // Parallel-transport and twisting of frame axes.
  int const numElements = isize(base.frameAxes);
  out.frameAxes.resize_noinit(numElements);
  for (int j = 0; j < numElements; ++j) {
    int const twistDofIndex = j * fem::kNumRodFields + (fem::kNumRodFields - 1);
    real const deltaTheta = out.displacements[twistDofIndex] - base.displacements[twistDofIndex];
    Real3 const baseTangent = rod::ComputeRodElementTangent(meshNodes, base.displacements, j);
    Real3 const outTangent = rod::ComputeRodElementTangent(meshNodes, out.displacements, j);
    out.frameAxes[j] = ToReal3(
        fem::TransportFrameAxis(
            ToSimd(baseTangent), ToSimd(outTangent), deltaTheta, ToSimd(base.frameAxes[j])));
  }
}

/// @brief Compute a differential variable at the beginning of the time step ('StepStart').
///
/// @details Opens the step: resizes the per-stage solution storage and advances the previous-step
/// history. The begin-of-step value is expressed as a linear combination of differences with
/// respect to @p previous, using the fact that HSum(alpha) must be 1 for the scheme to be
/// consistent (asserted in TimeIntegratorParams' constructor), and written to @p out.
template <typename MetadataT, ValueContainer T>
void ApplyTimeIntegrationStepStart(
    MetadataT const& metadata,
    CTimeIntegratorState const& intState,
    IntegrationBundle<T>& integration,
    T const& previous,
    T& out) {
  auto const& coeffs = GetCoefficients<TimeTarget::StepStart>(intState);
  MOCHI_ASSERT_VERBOSE(coeffs.size() == intState.numSteps);
  ResizeIntegrationStages(intState, previous, integration);
  RunPreStep(intState, previous, integration);
  if (intState.numSteps <= 1) {
    out.value = previous.value;
  } else {
    MOCHI_ASSERT_VERBOSE(
        isize(integration.prevSteps) >= intState.numSteps,
        "Multi-step time integration requires a complete previous-step history.");
    auto vals = MakeConstSpan(integration.prevSteps);
    AddWeightedDifferences(
        metadata, previous.value, vals, coeffs, {1, intState.numSteps}, out.value);
  }
}

/// @brief Compute a differential variable at the beginning of a stage ('StageStart') or the end of
/// the time step ('StepEnd').
///
/// @details Expresses the result as a linear combination of the completed integration stages with
/// respect to the begin-of-step reference @c integration.stepStart. Reads only the per-stage
/// solutions (never the previous-step history), so it takes no `previous` argument.
template <TimeTarget kTargetTime, typename MetadataT, ValueContainer T>
void ApplyTimeIntegration(
    MetadataT const& metadata,
    CTimeIntegratorState const& intState,
    IntegrationBundle<T>& integration,
    T& out) {
  static_assert(kTargetTime == TimeTarget::StageStart || kTargetTime == TimeTarget::StepEnd);
  auto const& coeffs = GetCoefficients<kTargetTime>(intState);
  MOCHI_ASSERT_VERBOSE(coeffs.size() <= integration.stages.size());
  if (isize(coeffs) == 0) {
    out.value = integration.stepStart.value;
  } else if (isize(coeffs) == 1 && coeffs[0] == 1_r) {
    out.value = integration.stages[0].value;
  } else {
    auto vals = MakeConstSpan(integration.stages);
    AddWeightedDifferences(
        metadata, integration.stepStart.value, vals, coeffs, {0, isize(coeffs)}, out.value);
  }
}

} // namespace details

// Concept to ensure PrevT can be static_cast to T const&
template <typename PrevT, typename T>
concept StaticCastableToConstRef = requires(PrevT const& prev) { static_cast<T const&>(prev); };

// Concept to ensure OutT can be static_cast to T&
template <typename OutT, typename T>
concept StaticCastableToRef = requires(OutT& out) { static_cast<T&>(out); };

// Combined concept for both requirements
template <typename PrevT, typename OutT, typename T>
concept TimeIntegrationCompatible =
    StaticCastableToConstRef<PrevT, T> && StaticCastableToRef<OutT, T>;

/// @brief Entry function to compute the begin-of-step value ('StepStart') with metadata.
template <typename MetadataT, ValueContainer T, typename PrevT, typename OutT>
  requires TimeIntegrationCompatible<PrevT, OutT, T>
void ApplyTimeIntegrationStepStart(
    MetadataT const& metadata,
    CTimeIntegratorState const& intState,
    IntegrationBundle<T>& integration,
    PrevT const& prev,
    OutT& out) {
  details::ApplyTimeIntegrationStepStart(
      metadata, intState, integration, static_cast<T const&>(prev), static_cast<T&>(out));
}

/// @brief Entry function to compute the begin-of-step value ('StepStart') without metadata.
template <ValueContainer T, typename PrevT, typename OutT>
  requires TimeIntegrationCompatible<PrevT, OutT, T>
void ApplyTimeIntegrationStepStart(
    CTimeIntegratorState const& intState,
    IntegrationBundle<T>& integration,
    PrevT const& prev,
    OutT& out) {
  details::ApplyTimeIntegrationStepStart(
      std::monostate{}, intState, integration, static_cast<T const&>(prev), static_cast<T&>(out));
}

/// @brief Entry function to compute a stage-start or step-end value with metadata.
template <TimeTarget kTargetTime, typename MetadataT, ValueContainer T, typename OutT>
  requires StaticCastableToRef<OutT, T>
void ApplyTimeIntegration(
    MetadataT const& metadata,
    CTimeIntegratorState const& intState,
    IntegrationBundle<T>& integration,
    OutT& out) {
  details::ApplyTimeIntegration<kTargetTime>(metadata, intState, integration, static_cast<T&>(out));
}

/// @brief Entry function to compute a stage-start or step-end value without metadata.
template <TimeTarget kTargetTime, ValueContainer T, typename OutT>
  requires StaticCastableToRef<OutT, T>
void ApplyTimeIntegration(
    CTimeIntegratorState const& intState,
    IntegrationBundle<T>& integration,
    OutT& out) {
  details::ApplyTimeIntegration<kTargetTime>(
      std::monostate{}, intState, integration, static_cast<T&>(out));
}

void ClearMultiStepIntegrationData(entt::registry& reg, entt::entity e);

} // namespace mochi::integration
