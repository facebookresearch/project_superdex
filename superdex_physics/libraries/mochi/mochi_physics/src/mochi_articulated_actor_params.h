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

#include "mochi_shape.h"

#include <mochi_physics/mochi_physics.h>

#include <memory>
#include <string_view>
#include <utility>

namespace mochi {

[[nodiscard]] inline bool HasInvalidNestedActorNameCharacter(std::string_view name) {
  return name.find('/') != std::string_view::npos || name.find('\\') != std::string_view::npos ||
      name.find('\0') != std::string_view::npos;
}

[[nodiscard]] inline DynamicString GetNestedActorLocalName(std::string_view actorName) {
  auto const slash = actorName.find_last_of('/');
  return DynamicString(slash == std::string_view::npos ? actorName : actorName.substr(slash + 1));
}

// Build the articulated body shape from the articulated actor params.
std::shared_ptr<ArticulatedBodyShape> GetArticulatedShape(
    ArticulatedActorParams const& params,
    Error& error);

// Apply minor fixes to the joint parameters
void AutoCorrect(ArticulatedJointParams& joint, Error& error);

// Apply minor fixes to the cycle joint parameters
void AutoCorrect(ArticulatedCycleJointParams& cycle, Error& error);

// Apply minor fixes to the link parameters
void AutoCorrect(ArticulatedLinkParams& link, Error& error);

// Apply minor fixes to the actor parameters
void AutoCorrect(ArticulatedActorParams& params, Error& error);

// Check for errors in ArticulatedActorParams
void Validate(ArticulatedActorParams const& params, Error& error);

// Check for errors in a single joint's friction parameters (finite + non-negative).
void ValidateFriction(ArticulatedJointFrictionParams const& friction, Error& error);

// Check for errors in a single joint's inertia coefficient (finite + non-negative).
void ValidateInertia(real inertia, Error& error);

} // namespace mochi
