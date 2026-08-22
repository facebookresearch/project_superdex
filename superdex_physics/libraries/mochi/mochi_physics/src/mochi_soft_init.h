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

#include "mochi_soft.h"

#include <memory>

namespace mochi {

// Forwards:
namespace experimental {
struct ExperimentalSoftActorParams;
}

// Initializes a soft actor from parameters with the given tetrahedral mesh and deep flow shape.
void InitSoftActor(
    entt::registry& reg,
    entt::entity e,
    SoftActorParams const& params,
    experimental::ExperimentalSoftActorParams const& experimentalParams,
    bool useContact,
    bool isSkinned,
    std::shared_ptr<TetrahedralMeshShape const> shapePtr,
    std::shared_ptr<DeepFlowShape const> flow,
    Error& error);

} // namespace mochi
