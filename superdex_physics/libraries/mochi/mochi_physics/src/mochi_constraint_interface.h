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

#include <mochi_physics/mochi_physics.h>

#include <memory>

namespace mochi {

class SceneImpl;

// Unlike Actor, the ConstraintInterface class has a public destructor so std::unique_ptr can
// destroy it.
class ConstraintInterface : public Constraint {};
using ConstraintInterfacePtr = std::unique_ptr<ConstraintInterface>;

// Create an object that implements mochi::Constraint to satisfy the public interface.
ConstraintInterfacePtr
CreateConstraintInterface(entt::registry& reg, entt::entity e, SceneImpl* scene);

} // namespace mochi
