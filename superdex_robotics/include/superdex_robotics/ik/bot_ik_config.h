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

#include <mochi_core/utils/dynamic_string.h>
#include <mochi_core/utils/reflection.h>
#include <superdex_robotics/ik/bot_ik.h>

#include <string_view>

namespace superdex::robotics {

/* @brief Supported IK document schema version. */
inline constexpr int kIKConfigVersion = 9;
/* @brief Initial soft solve-time budget [s]; zero remains a legal unlimited setting. */
inline constexpr double kIKInitialSolveBudgetSeconds = 0.01;

/* @brief Initial translational goal stiffness [N/m]. */
inline constexpr mochi::real kIKInitialPositionStiffness = mochi::real(100);
/* @brief Initial rotational goal stiffness [N*m/rad]. */
inline constexpr mochi::real kIKInitialRotationStiffness = mochi::real(1);

/* @brief Constraint frames attached to output and input links, or the output base when unbound. */
struct BotIKGoalConfig {
  mochi::DynamicString name;
  bool enabled = true;
  mochi::DynamicString outputBotLink;
  /* @brief Output-side constraint coordinates mapped into the output link. */
  mochi::TransformRT linkFrameFromConstraintFrame = {};
  mochi::DynamicString inputBotLink;
  /* @brief Input-side constraint coordinates mapped into inputBotLink, or the output base. */
  mochi::TransformRT baseOrInputLinkFrameFromConstraintFrame = {};
  mochi::real positionStiffness = kIKInitialPositionStiffness;
  mochi::real rotationStiffness = kIKInitialRotationStiffness;
};

/* @brief Document soft penalty qB ~= ratio*qA, with zero offset.
 * Active pairs bind distinct nonroot single-DOF joints of the same kind by exact-case name.
 * Both joints may move; the penalty is bilateral, not a guaranteed mechanical linkage.
 */
struct BotIKJointCouplingConfig {
  mochi::DynamicString jointA;
  mochi::DynamicString jointB;
  mochi::real ratio = mochi::real(1);
  mochi::real stiffness = mochi::real(1);
  bool enabled = true;
#if MOCHI_LANGUAGE_CPP20
  bool operator==(BotIKJointCouplingConfig const&) const = default;
#endif
};

/* @brief Authored document; solved poses and preview state are not persisted. */
struct BotIKConfig {
  int version = kIKConfigVersion;
  mochi::DynamicString outputBot;
  /* @brief Output joint positions; empty uses the output bot's authored defaultPose. */
  mochi::DynamicArray<mochi::real> outputBotConfiguration;
  mochi::DynamicString inputBot;
  /* @brief Input-base coordinates mapped into output-base coordinates. */
  mochi::TransformRT outputBaseFrameFromInputBaseFrame = {};
  /* @brief Positive finite uniform retarget scale, separate from native BotPrefab::scale. */
  mochi::real inputBotScale = mochi::real(1);
  mochi::DynamicArray<mochi::real> inputBotConfiguration;
  superdex::robotics::BotIKCreateParams createParams = [] {
    superdex::robotics::BotIKCreateParams params;
    params.fixedBase = true;
    params.solverParams.maxElapsedTimeSeconds = kIKInitialSolveBudgetSeconds;
    return params;
  }();
  mochi::DynamicArray<BotIKGoalConfig> goals;
  mochi::DynamicArray<BotIKJointCouplingConfig> jointCouplings;
};

/* @brief Failed parsing/loading preserves the destination. Duplicate bindings load with warnings.
 */
bool MOCHI_API LoadIKConfig(std::string_view, BotIKConfig&, mochi::DynamicString& diagnostic);
/* @brief Validate and write to an existing parent directory; invalid values are not written. */
bool MOCHI_API SaveIKConfig(std::string_view, BotIKConfig const&, mochi::DynamicString& diagnostic);
/* @brief Serialize every authored field without validation, for document snapshots. */
mochi::DynamicString MOCHI_API SerializeIKConfig(BotIKConfig const&);
/* @brief Strict JSON parsing; v1-v8 migrate to v9, dropping v5/v6 recording strings with warnings.
 * Failure preserves output; duplicate bindings also succeed with diagnostics. v9 accepts only
 * canonical Input/Output fields; legacy versions accept only their historical field names.
 */
bool MOCHI_API ParseIKConfig(std::string_view, BotIKConfig&, mochi::DynamicString& diagnostic);
/* @brief Validates values and unique goal names/link bindings; does not resolve the bot. */
bool MOCHI_API ValidateIKConfig(BotIKConfig const&, mochi::DynamicString& diagnostic);
/* @brief Validate all document coupling values/pairs, then bind active same-kind pairs.
 * Disabled/zero-stiffness stale names are allowed. Duplicate unordered pairs are rejected.
 */
bool MOCHI_API ValidateIKJointCouplings(
    BotPrefab const&,
    mochi::Span<BotIKJointCouplingConfig const>,
    mochi::DynamicString& diagnostic);
/* @brief Convert document couplings to creation-time runtime parameters, with zero offsets. */
mochi::DynamicArray<BotIKJointCouplingParams> MOCHI_API
    ToBotIKJointCouplingParams(mochi::Span<BotIKJointCouplingConfig const>);

/* @brief Inspectable duplicate diagnostics, independent of structural schema validation. */
bool MOCHI_API ValidateIKGoalBindings(BotIKConfig const&, mochi::DynamicString& diagnostic);
/* @brief Resolve a bot reference against the document directory, not the working directory. */
mochi::DynamicString MOCHI_API ResolveIKBotPath(std::string_view doc, std::string_view ref);
/* @brief Make a lexical document-relative reference; retain absolute paths across roots. */
mochi::DynamicString MOCHI_API MakeIKBotReference(std::string_view doc, std::string_view bot);
/* @brief Replace goals with all authored FK leaves, including hard joints; preserve on failure.
 */
bool MOCHI_API InitializeIKGoals(
    superdex::robotics::BotPrefab const& preparedOutputBot,
    BotIKConfig&,
    mochi::DynamicString& diagnostic);
/* @brief Return the goal frame in its authored reference (input-bot link or IK base). */
mochi::TransformRT MOCHI_API IKGoalConstraintTransform(BotIKGoalConfig const&);
/* @brief Store a goal frame; callers must supply a finite, normalized rotation. */
void MOCHI_API SetIKGoalConstraintTransform(BotIKGoalConfig&, mochi::TransformRT const&);

/* @brief Convert a resolved world goal to its authored reference frame.
 * Callers must validate finite unit-rotation frames and positive finite inputBotScale, and
 * supply the resolved input-link anchor when bound or the output-base anchor when unbound.
 * Only bound local translations are divided by retarget scale; native bot scale is not applied.
 */
mochi::TransformRT MOCHI_API IKGoalLocalConstraintTransform(
    BotIKGoalConfig const& goal,
    mochi::TransformRT const& anchor,
    mochi::TransformRT const& worldFrameFromConstraintFrame,
    mochi::real inputBotScale);

/* @brief FK at an explicit root placement; empty positions use the authored default pose.
 * Invalid input preserves the output. Positions exclude root DOFs. The bot must be prepared
 * (@ref RebuildBotData) and valid (@ref Validate); it isn't rechecked, as this runs every frame.
 */
void MOCHI_API ComputeIKInputBotTransforms(
    BotPrefab const&,
    mochi::Span<mochi::real const> positions,
    mochi::TransformRT const& worldFrameFromInputBaseFrame,
    mochi::DynamicArray<mochi::TransformRT>& worldFromLinks,
    mochi::Error&);
/* @brief Scale root-relative FK translations by native bot scale and retarget scale.
 * Root placement, rotations, and input joint values are unchanged. Invalid input preserves output.
 */
void MOCHI_API ComputeIKInputBotTransforms(
    BotPrefab const&,
    mochi::Span<mochi::real const> positions,
    mochi::TransformRT const& worldFrameFromInputBaseFrame,
    mochi::real inputBotScale,
    mochi::DynamicArray<mochi::TransformRT>& worldFromLinks,
    mochi::Error&);
/* @brief Resolve an authored constraint frame through an input-bot link or fixed output base. */
void MOCHI_API ComputeIKGoalWorldConstraintTransform(
    BotIKGoalConfig const&,
    mochi::TransformRT const& worldFrameFromOutputBaseFrame,
    BotPrefab const* inputBot,
    mochi::Span<mochi::TransformRT const> worldFrameFromInputLinks,
    mochi::TransformRT& worldFrameFromConstraintFrame,
    mochi::Error&);
/* @brief Resolve a constraint frame, scaling only bound input-side translations by retarget scale.
 * Unbound goals and all rotations are unchanged. Invalid input preserves output.
 */
void MOCHI_API ComputeIKGoalWorldConstraintTransform(
    BotIKGoalConfig const&,
    mochi::TransformRT const& worldFrameFromOutputBaseFrame,
    BotPrefab const* inputBot,
    mochi::Span<mochi::TransformRT const> worldFrameFromInputLinks,
    mochi::real inputBotScale,
    mochi::TransformRT& worldFrameFromConstraintFrame,
    mochi::Error&);

} // namespace superdex::robotics

// Canonical external reflection: do not duplicate these specializations in editor headers.
MOCHI_STRUCT_BEGIN_EX(mochi::experimental::IKSolverParams)
MOCHI_FIELD(maxIter)
MOCHI_FIELD(verbosity)
MOCHI_FIELD(absTol)
MOCHI_FIELD(relTol)
MOCHI_FIELD(positionErrorThres) MOCHI_ATTRIBUTE(Units("m"));
MOCHI_FIELD(rotationErrorThres) MOCHI_ATTRIBUTE(Units("rad"));
MOCHI_FIELD(lineSearchMaxIter)
MOCHI_FIELD(maxElapsedTimeSeconds) MOCHI_ATTRIBUTE(Units("s"));
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(superdex::robotics::BotIKCreateParams)
MOCHI_FIELD(fixedBase)
MOCHI_FIELD(excludeCollisionShapes)
MOCHI_FIELD(solverParams)
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(superdex::robotics::BotIKGoalConfig)
MOCHI_FIELD(name);
MOCHI_FIELD(enabled);
MOCHI_FIELD(outputBotLink);
MOCHI_FIELD(linkFrameFromConstraintFrame);
MOCHI_FIELD(inputBotLink);
MOCHI_FIELD(baseOrInputLinkFrameFromConstraintFrame);
MOCHI_FIELD(positionStiffness) MOCHI_ATTRIBUTE(Units("N/m"));
MOCHI_FIELD(rotationStiffness) MOCHI_ATTRIBUTE(Units("N*m/rad"));
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(superdex::robotics::BotIKJointCouplingConfig)
MOCHI_FIELD(jointA);
MOCHI_FIELD(jointB);
MOCHI_FIELD(ratio);
MOCHI_FIELD(stiffness);
MOCHI_FIELD(enabled);
MOCHI_STRUCT_END_EX()

MOCHI_STRUCT_BEGIN_EX(superdex::robotics::BotIKConfig)
MOCHI_FIELD(version);
MOCHI_FIELD(outputBot);
MOCHI_FIELD(outputBotConfiguration);
MOCHI_FIELD(inputBot);
MOCHI_FIELD(outputBaseFrameFromInputBaseFrame);
MOCHI_FIELD(inputBotScale);
MOCHI_FIELD(inputBotConfiguration);
MOCHI_FIELD(createParams);
MOCHI_FIELD(goals);
MOCHI_FIELD(jointCouplings);
MOCHI_STRUCT_END_EX()
