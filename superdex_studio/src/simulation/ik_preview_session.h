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

#include <superdex_robotics/superdex_robotics.h>
#include "assets/ik_config.h"

#include <memory>
#include <optional>
#include <unordered_map>

namespace superdex::studio {

/* @brief Borrowed transient input-bot pose; it never replaces the authored output-bot
 * configuration. */
struct IKInputBotPose {
  mochi::Span<mochi::real const> inputBotConfiguration;
  mochi::TransformRT worldFrameFromInputBaseFrame;
};

/* @brief Serialized, synchronous IK preview. Destroy/Stop before the borrowed context dies. */
class IKPreviewSession {
 public:
  /* @brief Copy a prepared bot and capture its immutable authored anchor and initial FK seed.
   * Success releases the previous solver; failure preserves the previous generation.
   */
  bool Prepare(superdex::robotics::BotPrefab const&, std::string& diagnostic);
  /* @brief Replace the preview seed and display transactionally; empty uses authored defaultPose.
   * Success releases the solver. Failure preserves the seed, solver, and display.
   */
  bool SetOutputBotConfiguration(mochi::Span<mochi::real const> positions, std::string& diagnostic);
  bool SetInputBot(superdex::robotics::BotPrefab const*, std::string& diagnostic);
  superdex::robotics::BotPrefab const* GetInputBot() const {
    return _inputBot ? &*_inputBot : nullptr;
  }
  /* @brief Warm-start one synchronous solve, recreating from the seed on creation-option changes.
   * False means an error; best-effort and no-active-goal ticks succeed. Errors retain the last
   * finite display, except that explicit recreation first restores the configured seed.
   */
  bool Step(
      mochi::Context*,
      superdex::robotics::IBotLoader const&,
      BotIKConfig const&,
      std::string& diagnostic,
      IKInputBotPose const* inputBotPose = nullptr);
  /* @brief Release the solver and restore the configured seed, leaving document goals untouched.
   */
  void Stop();
  bool HasSolver() const {
    return _solver != nullptr;
  }
  /* @brief Authored anchor, unchanged by floating-base solves. */
  mochi::TransformRT const& GetWorldFrameFromOutputBaseFrame() const {
    return _worldFrameFromOutputBaseFrame;
  }
  superdex::robotics::BotPrefab const& GetOutputBot() const {
    return _outputBot;
  }
  /* @brief Owned world-space snapshots in prepared link order, including solved root placement. */
  mochi::DynamicArray<mochi::TransformRT> const& GetOutputBotWorldFromLinks() const {
    return _worldFrameFromOutputLinkFrames;
  }
  mochi::DynamicArray<mochi::real> const& GetOutputBotJointPositions() const {
    return _outputBotJointPositions;
  }
  mochi::DynamicArray<std::string> const& GetOutputBotLinkNames() const {
    return _outputBotLinkNames;
  }
  std::string const& GetStatus() const {
    return _status;
  }
  /* @brief Last solver-call duration [ms], excluding creation; zero on ticks that do not solve. */
  double GetLastSolveMilliseconds() const {
    return _lastSolveMilliseconds;
  }
  /* @brief Validate the document and all named bindings; empty/disabled goal sets are legal. */
  bool CanSolve(
      BotIKConfig const&,
      std::string& diagnostic,
      IKInputBotPose const* inputBotPose = nullptr) const;

 private:
  void ComputeInputBotLinks(
      BotIKConfig const&,
      IKInputBotPose const*,
      mochi::DynamicArray<mochi::TransformRT>&,
      mochi::Error&) const;
  std::unique_ptr<superdex::robotics::BotIK, decltype(&superdex::robotics::BotIK::Destroy)> _solver{
      nullptr,
      &superdex::robotics::BotIK::Destroy};
  superdex::robotics::BotPrefab _outputBot;
  std::optional<superdex::robotics::BotPrefab> _inputBot;
  mochi::TransformRT _worldFrameFromOutputBaseFrame;
  mochi::DynamicArray<mochi::TransformRT> _seedWorldFrameFromOutputLinkFrames;
  mochi::DynamicArray<mochi::TransformRT> _worldFrameFromOutputLinkFrames;
  mochi::DynamicArray<mochi::real> _seedOutputBotJointPositions;
  mochi::DynamicArray<mochi::real> _outputBotJointPositions;
  mochi::DynamicArray<std::string> _outputBotLinkNames;
  std::unordered_map<std::string, int> _outputBotLinkIndices;
  mochi::DynamicArray<bool> _positionGoals;
  mochi::DynamicArray<bool> _rotationGoals;
  // Couplings are creation-time output-bot state; unchanged ticks retain the solver warm start.
  superdex::robotics::BotIKCreateParams _createdParams;
  mochi::DynamicArray<superdex::robotics::BotIKJointCouplingConfig> _createdCouplings;
  mochi::Context* _context = nullptr;
  bool _prepared = false;
  std::string _status = "Not Prepared";
  double _lastSolveMilliseconds = 0;
};

/* @brief Map composed skin bone identities into prepared link order, preserving output on failure.
 * Empty bone identities use the ordinary link order. No renderer or bind-frame mutation is needed.
 */
bool MapIKSkinTransforms(
    superdex::robotics::BotPrefab const&,
    mochi::Span<mochi::TransformRT const> worldFrameFromLinks,
    mochi::DynamicArray<mochi::TransformRT>& worldFrameFromBones,
    std::string& diagnostic);

} // namespace superdex::studio
