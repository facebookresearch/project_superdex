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

#include "mochi_ecs.h"

#include <mochi_core/utils/reflection.h>

namespace mochi::attribute {

/**
 * @brief [Struct Attribute] Indicates that the struct's data should be recorded by
 * @ref Scene::CaptureState and restored by @ref Scene::RestoreState.
 *
 * @note Optionally specify other components/tags that must be present for the data to be captured.
 * @note If it is a global context component, then use @ref CaptureStateCtx instead.
 *
 * @example MOCHI_ATTRIBUTE(CaptureState) // Capture for all actors
 * @example MOCHI_ATTRIBUTE(CaptureState(ecs::RequiredTag<TagSoftActor>{})) // Soft actors only
 */
struct CaptureState : Attribute {
  CaptureState() = default;

  template <class ComponentT>
  explicit CaptureState(ecs::Included<ComponentT>)
      : onlyCaptureWith(entt::type_id<ComponentT>().hash()) {}

  entt::id_type onlyCaptureWith = 0; // Only capture if the entity also has this (if non-zero).
  MOCHI_STRUCT_WITH_BASE(mochi::attribute::CaptureState, mochi::Attribute);
};

/**
 * @brief [Struct Attribute] Use in place of @ref attribute::CaptureState for components that will
 * be set as global context, rather than being emplaced on an entity.
 */
struct CaptureStateCtx : Attribute {
  MOCHI_STRUCT_WITH_BASE(mochi::attribute::CaptureStateCtx, mochi::Attribute);
};

/**
 * @brief [Struct Attribute] Marks a component as carrying *adjoint* state (back-propagation
 * gradients that accumulate across @ref BackPropagate calls).
 *
 * @details This is a discriminator that is consulted by internal restore paths inside @ref
 * BackPropagate to skip restoring adjoint components, so that the live, accumulating adjoint state
 * is not clobbered by the per-step forward-state restores.
 */
struct HasAdjoint : Attribute {
  MOCHI_STRUCT_WITH_BASE(mochi::attribute::HasAdjoint, mochi::Attribute);
};

/**
 * @brief [Struct Attribute] Marks a component as carrying old constraint targets.
 *
 * @details This is a discriminator that is consulted by internal restore paths inside @ref
 * RestoreStatePair to skip restoring old constraint targets.
 */
struct HasOldTarget : Attribute {
  MOCHI_STRUCT_WITH_BASE(mochi::attribute::HasOldTarget, mochi::Attribute);
};
} // namespace mochi::attribute
