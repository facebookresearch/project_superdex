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

#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/utils/mochi_physics_macros.h>

#include <string_view>

namespace mochi {

// Temporary bridge that loads an articulation's joint/link topology from a
// `.mochi.h5` file so existing samples and tests can move off the deleted Context
// shape-loading APIs. The caller fills in the rigid/joint properties and root transform. New code
// should build the params by hand or load a prefab; this utility (and its `.h5` parsing)
// will be removed once callers migrate.
MOCHI_API ArticulatedActorParams LoadArticulatedActorParams(
    Context* context,
    std::string_view articulatedShapePath,
    Span<ShapeHandle const> linkShapes,
    Error& error);

} // namespace mochi
