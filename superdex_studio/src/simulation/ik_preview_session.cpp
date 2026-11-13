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

#include "simulation/ik_preview_session.h"

#include <mochi_core/utils/math_utils.h>
#include <superdex_robotics/utils/bot_utils.h>

#include <algorithm>
#include <chrono>
#include <string_view>
#include <utility>

using namespace mochi;
using namespace superdex::robotics;

namespace superdex::studio {
namespace {

bool ValidatePrepared(BotPrefab const& bot, std::string& diagnostic) {
  Error error;
  Validate(bot, nullptr, error);
  if (!error.IsOK()) {
    diagnostic = error.GetDescription();
    return false;
  }
  DynamicArray<int> dofs;
  DynamicArray<DynamicArray<int>> children(bot.links.size());
  DynamicArray<TransformRT> rootFromLinks(bot.links.size());
  for (int i = 0; i < isize(bot.links); ++i) {
    auto const& link = bot.links[i];
    if (link._index != i || (i > 0 && (link.parentLink < 0 || link.parentLink >= i))) {
      diagnostic = "Prepared topology must contain indexed, parent-before-child links";
      return false;
    }
    TransformRT const parentFromLink = i == 0 ? TransformRT{} : bot.joints[i].parentLinkFromJoint;
    if (i > 0) {
      children[link.parentLink].push_back(i);
      rootFromLinks[i] = rootFromLinks[link.parentLink] * parentFromLink;
    }
    if (link._parentFromLink != parentFromLink || link._rootFromLink != rootFromLinks[i]) {
      diagnostic = "Prepared link transforms are stale";
      return false;
    }
    auto const type = bot.joints[i].type;
    int const count = type == ArticulatedJointType::Spherical
        ? 3
        : (type == ArticulatedJointType::Prismatic || type == ArticulatedJointType::Revolute ? 1
                                                                                             : 0);
    for (int j = 0; j < count; ++j) {
      dofs.push_back(i);
    }
  }
  for (int i = 0; i < isize(bot.links); ++i) {
    if (bot.links[i]._childrenIndices != children[i]) {
      diagnostic = "Prepared child topology is stale";
      return false;
    }
  }
  if (dofs != bot._dofIndices || isize(dofs) != bot._numDofs ||
      isize(bot.defaultPose) != bot._numDofs || !IsFinite(bot.worldFromRoot)) {
    diagnostic = "Prepared DOF data, default pose, or base transform is invalid";
    return false;
  }
  for (real const position : bot.defaultPose) {
    if (!IsFinite(position)) {
      diagnostic = "Default joint positions must be finite";
      return false;
    }
  }
  return true;
}

bool SamePositions(Span<real const> a, Span<real const> b) {
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

Span<real const> OutputBotPositions(BotPrefab const& outputBot, Span<real const> positions) {
  return positions.empty() ? Span<real const>{outputBot.defaultPose} : positions;
}

bool ComputeOutputBotLinks(
    BotPrefab const& outputBot,
    Span<real const> positions,
    DynamicArray<TransformRT>& links,
    std::string& diagnostic) {
  if (isize(positions) != outputBot._numDofs) {
    diagnostic = "Output joint position count does not match prepared DOFs";
    return false;
  }
  for (real const position : positions) {
    if (!IsFinite(position)) {
      diagnostic = "Output joint positions must be finite";
      return false;
    }
  }
  Error error;
  ComputeLinkTransformsFromPose(
      outputBot, positions, links, LinkTransformSpace::RootFromParent, error);
  if (!error.IsOK()) {
    diagnostic = error.GetDescription();
    return false;
  }
  for (auto& link : links) {
    link.SetTranslation(link.GetTranslation() * outputBot.scale);
    link = outputBot.worldFromRoot * link;
    if (!IsFinite(link)) {
      diagnostic = "Non-finite output link pose";
      return false;
    }
  }
  return true;
}

} // namespace

bool BuildIKInputPoseSnapshot(
    BotIKConfig const& config,
    BotPrefab const* inputBot,
    IKInputPoseSnapshot const& displayedPose,
    TransformRT const& worldFrameFromOutputBaseFrame,
    BotIKConfig& candidate,
    std::string& diagnostic) {
  diagnostic.clear();
  if (!inputBot || config.inputBot.empty() || inputBot->links.empty() || inputBot->_numDofs < 0 ||
      inputBot->joints.size() != inputBot->links.size() ||
      isize(inputBot->_dofIndices) != inputBot->_numDofs) {
    diagnostic = "Snapshot requires a prepared input bot";
    return false;
  }
  if (!ValidatePrepared(*inputBot, diagnostic)) {
    return false;
  }
  if (isize(displayedPose.inputBotConfiguration) != inputBot->_numDofs) {
    diagnostic = "Displayed input pose does not contain the complete joint configuration";
    return false;
  }
  auto const validFrame = [](TransformRT const& frame) {
    return IsFinite(frame) && NearEqual(Norm(frame.GetRotation().ToReal4()), real(1), real(1e-4));
  };
  if (!validFrame(worldFrameFromOutputBaseFrame) ||
      !validFrame(displayedPose.worldFrameFromInputBaseFrame)) {
    diagnostic = "Snapshot requires finite unit-rotation input and authored output frames";
    return false;
  }
  DynamicArray<TransformRT> links;
  Error error;
  ComputeIKInputBotTransforms(
      *inputBot,
      displayedPose.inputBotConfiguration,
      displayedPose.worldFrameFromInputBaseFrame,
      config.inputBotScale,
      links,
      error);
  if (!error.IsOK() || links.size() != inputBot->links.size()) {
    diagnostic = error.IsOK() ? "Displayed input pose has incomplete forward kinematics"
                              : error.GetDescription();
    return false;
  }
  BotIKConfig next = config;
  next.inputBotConfiguration = displayedPose.inputBotConfiguration;
  next.outputBaseFrameFromInputBaseFrame = NormalizeRotation(
      Invert(worldFrameFromOutputBaseFrame) * displayedPose.worldFrameFromInputBaseFrame);
  // A repeated capture should not create an undo entry from floating-point anchor roundtrips.
  if (NearEqual(
          next.outputBaseFrameFromInputBaseFrame,
          config.outputBaseFrameFromInputBaseFrame,
          real(1e-6))) {
    next.outputBaseFrameFromInputBaseFrame = config.outputBaseFrameFromInputBaseFrame;
  }
  if (!ValidateIKConfig(next, diagnostic)) {
    return false;
  }
  candidate = std::move(next);
  return true;
}

bool IKPreviewSession::Prepare(BotPrefab const& outputBot, std::string& diagnostic) {
  diagnostic.clear();
  if (!ValidatePrepared(outputBot, diagnostic)) {
    return false;
  }
  DynamicArray<TransformRT> links;
  Error error;
  ComputeLinkTransformsFromPose(
      outputBot, outputBot.defaultPose, links, LinkTransformSpace::RootFromParent, error);
  if (!error.IsOK()) {
    diagnostic = error.GetDescription();
    return false;
  }
  std::unordered_map<std::string, int> indices;
  DynamicArray<std::string> names;
  for (int i = 0; i < isize(outputBot.links); ++i) {
    std::string const name{std::string_view(outputBot.links[i].name)};
    if (name.empty() || !indices.emplace(name, i).second) {
      diagnostic = "Prepared link names must be nonempty and unique: " + name;
      return false;
    }
    names.push_back(name);
    links[i] = outputBot.worldFromRoot * links[i];
    if (!IsFinite(links[i])) {
      diagnostic = "Non-finite default link pose";
      return false;
    }
  }
  _solver.reset();
  _createdCouplings.clear();
  _context = nullptr;
  _outputBot = outputBot;
  _worldFrameFromOutputBaseFrame = outputBot.worldFromRoot;
  _seedWorldFrameFromOutputLinkFrames = std::move(links);
  _worldFrameFromOutputLinkFrames = _seedWorldFrameFromOutputLinkFrames;
  _seedOutputBotJointPositions = outputBot.defaultPose;
  _outputBotJointPositions = _seedOutputBotJointPositions;
  _outputBotLinkNames = std::move(names);
  _outputBotLinkIndices = std::move(indices);
  _positionGoals = DynamicArray<bool>(outputBot.links.size());
  _rotationGoals = DynamicArray<bool>(outputBot.links.size());
  _prepared = true;
  _status = "Prepared";
  _lastSolveMilliseconds = 0;
  _lastSolveIterations = 0;
  _lastSolveConvergence = ConvergenceStatus::None;
  return true;
}

bool IKPreviewSession::SetOutputBotConfiguration(
    Span<real const> positions,
    std::string& diagnostic) {
  diagnostic.clear();
  if (!_prepared) {
    diagnostic = "No prepared output bot";
    return false;
  }
  DynamicArray<real> candidate{OutputBotPositions(_outputBot, positions)};
  DynamicArray<TransformRT> links;
  if (!ComputeOutputBotLinks(_outputBot, candidate, links, diagnostic)) {
    return false;
  }
  Stop();
  _seedOutputBotJointPositions = std::move(candidate);
  _seedWorldFrameFromOutputLinkFrames = std::move(links);
  _outputBotJointPositions = _seedOutputBotJointPositions;
  _worldFrameFromOutputLinkFrames = _seedWorldFrameFromOutputLinkFrames;
  _status = "Prepared";
  return true;
}

bool IKPreviewSession::SetInputBot(BotPrefab const* inputBot, std::string& diagnostic) {
  diagnostic.clear();
  if (inputBot && !ValidatePrepared(*inputBot, diagnostic)) {
    return false;
  }
  _inputBot = inputBot ? std::optional<BotPrefab>{*inputBot} : std::nullopt;
  return true;
}

void IKPreviewSession::ComputeInputBotLinks(
    BotIKConfig const& spec,
    IKInputBotPose const* inputBotPose,
    DynamicArray<TransformRT>& worldFrameFromInputLinkFrames,
    Error& error) const {
  if (spec.inputBot.empty()) {
    if (inputBotPose) {
      MOCHI_ERROR_SET(error, "Transient input-bot pose requires a input bot");
    }
    return;
  }
  if (!_inputBot) {
    MOCHI_ERROR_SET(error, "Input bot could not be resolved");
    return;
  }
  if (inputBotPose && isize(inputBotPose->inputBotConfiguration) != _inputBot->_numDofs) {
    MOCHI_ERROR_SET(error, "Transient input-bot configuration does not match input-bot DOFs");
    return;
  }
  ComputeIKInputBotTransforms(
      *_inputBot,
      inputBotPose ? inputBotPose->inputBotConfiguration
                   : Span<real const>{spec.inputBotConfiguration},
      inputBotPose ? inputBotPose->worldFrameFromInputBaseFrame
                   : _worldFrameFromOutputBaseFrame * spec.outputBaseFrameFromInputBaseFrame,
      spec.inputBotScale,
      worldFrameFromInputLinkFrames,
      error);
}

bool IKPreviewSession::CanSolve(
    BotIKConfig const& spec,
    std::string& diagnostic,
    IKInputBotPose const* inputBotPose) const {
  diagnostic.clear();
  if (!_prepared) {
    diagnostic = "No prepared output bot";
    return false;
  }
  if (!ValidateIKConfig(spec, diagnostic)) {
    return false;
  }
  mochi::DynamicString couplingDiagnostic;
  if (!robotics::ValidateIKJointCouplings(_outputBot, spec.jointCouplings, couplingDiagnostic)) {
    diagnostic.assign(couplingDiagnostic.begin(), couplingDiagnostic.end());
    return false;
  }
  auto const outputBotConfiguration = OutputBotPositions(_outputBot, spec.outputBotConfiguration);
  if (!SamePositions(outputBotConfiguration, _seedOutputBotJointPositions)) {
    DynamicArray<TransformRT> worldFrameFromOutputLinkFrames;
    if (!ComputeOutputBotLinks(
            _outputBot, outputBotConfiguration, worldFrameFromOutputLinkFrames, diagnostic)) {
      return false;
    }
  }
  DynamicArray<TransformRT> worldFrameFromInputLinkFrames;
  Error error;
  ComputeInputBotLinks(spec, inputBotPose, worldFrameFromInputLinkFrames, error);
  if (!error.IsOK()) {
    diagnostic = error.GetDescription();
    return false;
  }
  for (auto const& goal : spec.goals) {
    if (_outputBotLinkIndices.find(std::string(goal.outputBotLink)) ==
        _outputBotLinkIndices.end()) {
      diagnostic = "Unknown goal link: " + std::string(goal.outputBotLink);
      return false;
    }
    TransformRT worldFrameFromConstraintFrame;
    ComputeIKGoalWorldConstraintTransform(
        goal,
        _worldFrameFromOutputBaseFrame,
        GetInputBot(),
        worldFrameFromInputLinkFrames,
        spec.inputBotScale,
        worldFrameFromConstraintFrame,
        error);
  }
  if (!error.IsOK()) {
    diagnostic = error.GetDescription();
    return false;
  }
  return true;
}

void IKPreviewSession::Stop() {
  _solver.reset();
  _createdCouplings.clear();
  _context = nullptr;
  _worldFrameFromOutputLinkFrames = _seedWorldFrameFromOutputLinkFrames;
  _outputBotJointPositions = _seedOutputBotJointPositions;
  std::fill(_positionGoals.begin(), _positionGoals.end(), false);
  std::fill(_rotationGoals.begin(), _rotationGoals.end(), false);
  _lastSolveMilliseconds = 0;
  _lastSolveIterations = 0;
  _lastSolveConvergence = ConvergenceStatus::None;
  _status = _prepared ? "Stopped" : "Not Prepared";
}

bool IKPreviewSession::Step(
    Context* context,
    IBotLoader const& loader,
    BotIKConfig const& spec,
    std::string& diagnostic,
    IKInputBotPose const* inputBotPose) {
  _lastSolveMilliseconds = 0;
  _lastSolveIterations = 0;
  _lastSolveConvergence = ConvergenceStatus::None;
  auto fail = [&] {
    _status = diagnostic;
    return false;
  };
  if (!CanSolve(spec, diagnostic, inputBotPose)) {
    return fail();
  }
  if (context == nullptr) {
    diagnostic = "Mochi context is null";
    return fail();
  }
  auto const outputBotConfiguration = OutputBotPositions(_outputBot, spec.outputBotConfiguration);
  if (!SamePositions(outputBotConfiguration, _seedOutputBotJointPositions) &&
      !SetOutputBotConfiguration(outputBotConfiguration, diagnostic)) {
    return fail();
  }
  bool const recreate = _solver &&
      (_context != context || _createdParams.fixedBase != spec.createParams.fixedBase ||
       _createdParams.excludeCollisionShapes != spec.createParams.excludeCollisionShapes ||
       _createdCouplings != spec.jointCouplings);
  if (recreate) {
    Stop();
  }
  Error error;
  if (!_solver) {
    // Seed only on creation; subsequent ticks retain the solver's joint and root warm starts.
    BotPrefab solverBot = _outputBot;
    solverBot.defaultPose = _seedOutputBotJointPositions;
    auto const couplings = robotics::ToBotIKJointCouplingParams(spec.jointCouplings);
    _solver.reset(BotIK::Create(context, solverBot, spec.createParams, couplings, loader, error));
    if (!error.IsOK() || !_solver) {
      diagnostic = error.IsOK() ? "Failed to create IK solver" : error.GetDescription();
      return fail();
    }
    _context = context;
    _createdParams = spec.createParams;
    _createdCouplings = spec.jointCouplings;
  }
  _solver->SetSolverParams(spec.createParams.solverParams);
  DynamicArray<TransformRT> worldFrameFromInputLinkFrames;
  ComputeInputBotLinks(spec, inputBotPose, worldFrameFromInputLinkFrames, error);
  DynamicArray<BotIKGoalConfig const*> goals(_outputBotLinkNames.size());
  for (auto const& goal : spec.goals) {
    goals[_outputBotLinkIndices.at(std::string(goal.outputBotLink))] = &goal;
  }
  bool active = false;
  for (int i = 0; i < isize(goals); ++i) {
    auto const* goal = goals[i];
    bool const position = goal && goal->enabled && goal->positionStiffness > 0_r;
    bool const rotation = goal && goal->enabled && goal->rotationStiffness > 0_r;
    if (position || rotation) {
      TransformRT worldFrameFromConstraintFrame;
      ComputeIKGoalWorldConstraintTransform(
          *goal,
          _worldFrameFromOutputBaseFrame,
          GetInputBot(),
          worldFrameFromInputLinkFrames,
          spec.inputBotScale,
          worldFrameFromConstraintFrame,
          error);
      if (position) {
        _solver->SetPositionGoal(
            i,
            goal->linkFrameFromConstraintFrame.GetTranslation(),
            worldFrameFromConstraintFrame.GetTranslation(),
            goal->positionStiffness,
            error);
      }
      if (rotation) {
        _solver->SetRotationGoal(
            i,
            goal->linkFrameFromConstraintFrame.GetRotation(),
            worldFrameFromConstraintFrame.GetRotation(),
            goal->rotationStiffness,
            error);
      }
    }
    if (!position && _positionGoals[i]) {
      _solver->ClearPositionGoal(i, error);
    }
    if (!rotation && _rotationGoals[i]) {
      _solver->ClearRotationGoal(i, error);
    }
    if (!error.IsOK()) {
      diagnostic = error.GetDescription();
      return fail();
    }
    _positionGoals[i] = position;
    _rotationGoals[i] = rotation;
    active |= position || rotation;
  }
  if (!active) {
    _status = "No Active Goals";
    return true;
  }
  auto const start = std::chrono::steady_clock::now();
  bool const reached = _solver->SolveIK(error);
  _lastSolveMilliseconds =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  if (!error.IsOK()) {
    diagnostic = error.GetDescription();
    return fail();
  }
  auto const solverStats = _solver->GetLastSolverStats();
  _lastSolveIterations = solverStats.maxNonLinearIters;
  _lastSolveConvergence = solverStats.convergenceStatus;
  // Copy both views before any subsequent mutation. Only publish a fully finite snapshot.
  DynamicArray<TransformRT> links{_solver->GetWorldFromLinks()};
  DynamicArray<real> positions{_solver->GetJointPositions()};
  for (auto const& link : links) {
    if (!IsFinite(link)) {
      diagnostic = "IK returned a non-finite link pose";
      _solver.reset();
      return fail();
    }
  }
  for (real const position : positions) {
    if (!IsFinite(position)) {
      diagnostic = "IK returned a non-finite joint position";
      _solver.reset();
      return fail();
    }
  }
  _worldFrameFromOutputLinkFrames = std::move(links);
  _outputBotJointPositions = std::move(positions);
  _status = reached ? "Goals Reached" : "Best Effort: Goals Unreached";
  ++_solveCount;
  return true;
}

bool MapIKSkinTransforms(
    BotPrefab const& bot,
    Span<TransformRT const> worldFrameFromLinks,
    DynamicArray<TransformRT>& worldFrameFromBones,
    std::string& diagnostic) {
  diagnostic.clear();
  if (worldFrameFromLinks.size() != bot.links.size()) {
    diagnostic = "Skin mapping requires one transform per prepared link";
    return false;
  }
  std::unordered_map<std::string_view, int> indices;
  indices.reserve(bot.links.size());
  for (int i = 0; i < isize(bot.links); ++i) {
    std::string_view const name{bot.links[i].name};
    if (!indices.emplace(name, i).second) {
      diagnostic = "Duplicate skin link: " + std::string{name};
      return false;
    }
    if (!IsFinite(worldFrameFromLinks[i])) {
      diagnostic = "Non-finite skin link transform";
      return false;
    }
  }
  DynamicArray<TransformRT> mapped;
  if (bot._skinBoneLinks.empty()) {
    mapped = DynamicArray<TransformRT>{worldFrameFromLinks};
  } else {
    for (auto const& bone : bot._skinBoneLinks) {
      auto const it = indices.find(std::string_view{bone});
      if (it == indices.end()) {
        diagnostic = "Unknown skin bone link: " + std::string{std::string_view(bone)};
        return false;
      }
      mapped.push_back(worldFrameFromLinks[it->second]);
    }
  }
  worldFrameFromBones = std::move(mapped);
  return true;
}

} // namespace superdex::studio
