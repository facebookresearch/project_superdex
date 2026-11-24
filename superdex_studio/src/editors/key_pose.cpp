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

#include "editors/key_pose.h"

#include <superdex_robotics/utils/bot_utils.h>

#include <mochi_core/utils/basic_utils.h>

namespace superdex::studio {

namespace {

// Whether the bot's derived data (RebuildBotData) fits it, and `pose` too if given.
bool Fits(superdex::robotics::BotPrefab const& bot, mochi::Span<mochi::real const> pose) {
  if (mochi::isize(pose) != bot._numDofs || mochi::isize(bot._dofIndices) != bot._numDofs ||
      bot.joints.size() != bot.links.size()) {
    return false;
  }
  for (int const joint : bot._dofIndices) {
    if (joint < 0 || joint >= mochi::isize(bot.joints)) {
      return false;
    }
  }
  for (int i = 0; i < mochi::isize(bot.links); ++i) {
    if (bot.links[i].parentLink >= i) {
      return false;
    }
  }
  return true;
}

} // namespace

bool SetJointPositions(
    superdex::robotics::BotPrefab const& bot,
    mochi::Span<std::string_view const> joints,
    mochi::Span<mochi::real const> values,
    std::vector<mochi::real>& pose) {
  if (joints.size() != values.size() || !Fits(bot, mochi::MakeConstSpan(pose))) {
    return false;
  }
  std::vector<mochi::real> set = pose;
  for (size_t k = 0; k < joints.size(); ++k) {
    // A spherical joint's DOFs all name it, so a name must match exactly one DOF.
    int dof = -1;
    for (int d = 0; d < bot._numDofs; ++d) {
      if (std::string_view(bot.joints[static_cast<size_t>(bot._dofIndices[d])].name) == joints[k]) {
        if (dof >= 0) {
          return false;
        }
        dof = d;
      }
    }
    if (dof < 0) {
      return false;
    }
    set[static_cast<size_t>(dof)] = values[k];
  }
  pose = std::move(set);
  return true;
}

std::vector<bool> ComputeMovedLinks(
    superdex::robotics::BotPrefab const& bot,
    mochi::Span<mochi::real const> from,
    mochi::Span<mochi::real const> to,
    mochi::real tolerance) {
  if (from.size() != to.size() || !Fits(bot, from)) {
    return {};
  }
  std::vector<bool> moved(bot.links.size(), false);
  for (int d = 0; d < bot._numDofs; ++d) {
    if (mochi::Abs(from[d] - to[d]) > tolerance) {
      // Joint i connects link i to its parent.
      moved[static_cast<size_t>(bot._dofIndices[d])] = true;
    }
  }
  // Parents come first.
  for (size_t i = 0; i < bot.links.size(); ++i) {
    int const parent = bot.links[i].parentLink;
    if (parent >= 0 && moved[static_cast<size_t>(parent)]) {
      moved[i] = true;
    }
  }
  return moved;
}

ToolLink FindToolLink(
    superdex::robotics::BotPrefab const& bot,
    mochi::Span<mochi::real const> pose,
    mochi::TransformRT const& rootFromTool,
    superdex::Error& error) {
  std::vector<mochi::TransformRT> rootFromLinks;
  ComputeLinkWorldTransforms(bot, mochi::TransformRT{}, pose, rootFromLinks, error);
  MOCHI_ERROR_RETURN(error, {});
  // Nearest first; among links at one spot (a flange and the tool link on it), the one turned least
  // from the tool.
  ToolLink found;
  mochi::real bestDistance = 0;
  mochi::real bestAngle = 0;
  constexpr mochi::real kSameSpot = 1e-6f; // [m]
  for (int i = 0; i < mochi::isize(rootFromLinks); ++i) {
    mochi::TransformRT const linkFromTool =
        mochi::Invert(rootFromLinks[static_cast<size_t>(i)]) * rootFromTool;
    mochi::real const distance = mochi::Norm(linkFromTool.GetTranslation());
    mochi::real const angle = mochi::Norm(linkFromTool.GetRotation().ToRotationVector());
    bool const better = found.link < 0 || distance < bestDistance - kSameSpot ||
        (distance < bestDistance + kSameSpot && angle < bestAngle);
    if (better) {
      found = ToolLink{.link = i, .linkFromTool = linkFromTool};
      bestDistance = distance;
      bestAngle = angle;
    }
  }
  return found;
}

void ComputeLinkWorldTransforms(
    superdex::robotics::BotPrefab const& bot,
    mochi::TransformRT const& worldFromRoot,
    mochi::Span<mochi::real const> pose,
    std::vector<mochi::TransformRT>& out,
    superdex::Error& error) {
  out.clear();
  mochi::DynamicArray<mochi::TransformRT> rootFromLinks;
  superdex::robotics::ComputeLinkTransformsFromPose(
      bot, pose, rootFromLinks, superdex::robotics::LinkTransformSpace::RootFromParent, error);
  if (!error.IsOK()) {
    return;
  }
  out.reserve(rootFromLinks.size());
  for (mochi::TransformRT const& rootFromLink : rootFromLinks) {
    out.push_back(
        worldFromRoot *
        mochi::TransformRT{rootFromLink.GetRotation(), rootFromLink.GetTranslation() * bot.scale});
  }
}

void ComputePoseFromLinkWorldTransforms(
    superdex::robotics::BotPrefab const& bot,
    mochi::Span<mochi::TransformRT const> worldFromLinks,
    mochi::Span<mochi::real const> referencePose,
    std::vector<mochi::real>& out,
    superdex::Error& error) {
  out.clear();
  MOCHI_ERROR_IF_NOT(
      !worldFromLinks.empty() && worldFromLinks.size() == bot.links.size(),
      error,
      "Expected one transform per link.");
  MOCHI_ERROR_RETURN(error);
  // Nominal bot space: rooted at the root link, unscaled.
  mochi::TransformRT const rootFromWorld = mochi::Invert(worldFromLinks[0]);
  mochi::real const inverseScale = 1 / bot.scale;
  mochi::DynamicArray<mochi::TransformRT> rootFromLinks;
  rootFromLinks.reserve(worldFromLinks.size());
  for (mochi::TransformRT const& worldFromLink : worldFromLinks) {
    mochi::TransformRT const rootFromLink = rootFromWorld * worldFromLink;
    rootFromLinks.push_back(
        mochi::TransformRT{
            rootFromLink.GetRotation(), rootFromLink.GetTranslation() * inverseScale});
  }
  mochi::DynamicArray<mochi::real> pose;
  superdex::robotics::ComputePoseFromLinkTransforms(
      bot,
      mochi::MakeConstSpan(rootFromLinks),
      pose,
      superdex::robotics::LinkTransformSpace::RootFromParent,
      error,
      referencePose);
  MOCHI_ERROR_RETURN(error);
  out.assign(pose.begin(), pose.end());
}

} // namespace superdex::studio
