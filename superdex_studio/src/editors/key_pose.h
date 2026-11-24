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

#include <mochi_core/utils/span.h>
#include <mochi_core/utils/transform_rt.h>

#include <string_view>
#include <vector>

namespace superdex::studio {

// The pose a joint key asks of a bot, and the links it moves.
struct KeyPose {
  // One entry per bot-space DOF.
  std::vector<mochi::real> pose;
  // Per link: whether a keyed joint moves it, its own or an ancestor's.
  std::vector<bool> movedLinks;

  bool operator==(KeyPose const&) const = default;
};

// `now` (one entry per bot-space DOF) with each of `joints` set to its entry in `values`. False if
// the sizes disagree, a name is not one of the bot's one-DOF joints (revolute or prismatic), or the
// bot's derived data (RebuildBotData) does not fit it.
[[nodiscard]] bool ComputeKeyPose(
    superdex::robotics::BotPrefab const& bot,
    mochi::Span<mochi::real const> now,
    mochi::Span<std::string_view const> joints,
    mochi::Span<mochi::real const> values,
    KeyPose& out);

// The world transforms of `bot`'s links at `pose`, its root at `worldFromRoot`, scaled by its
// `scale`.
void ComputeLinkWorldTransforms(
    superdex::robotics::BotPrefab const& bot,
    mochi::TransformRT const& worldFromRoot,
    mochi::Span<mochi::real const> pose,
    std::vector<mochi::TransformRT>& out,
    superdex::Error& error);

// The pose (one entry per bot-space DOF) of `bot`'s links at `worldFromLinks`, placed as
// ComputeLinkWorldTransforms places them, with the root at the root link: its inverse. Revolute
// angles take the turn nearest `referencePose`'s.
void ComputePoseFromLinkWorldTransforms(
    superdex::robotics::BotPrefab const& bot,
    mochi::Span<mochi::TransformRT const> worldFromLinks,
    mochi::Span<mochi::real const> referencePose,
    std::vector<mochi::real>& out,
    superdex::Error& error);

} // namespace superdex::studio
