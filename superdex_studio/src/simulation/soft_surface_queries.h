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

#include "rendering/soft_surface.h"

#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_physics/mochi_physics.h>

#include <utility>
#include <vector>

namespace superdex::studio {

// Registers the node position and normal queries of one of @p actor's meshes.
[[nodiscard]] std::pair<mochi::QueryHandle, mochi::QueryHandle>
RegisterSoftSurfaceQueries(mochi::Actor& actor, SoftSurface surface, mochi::Error& error);

// Reads the deformed nodes of one of @p actor's meshes, from the queries RegisterSoftSurfaceQueries
// registered, into flat renderer-space buffers (resized in place).
void ReadSoftSurfaceNodes(
    mochi::Actor const& actor,
    SoftSurface surface,
    mochi::CoordinateSpaceConverter const& converter,
    std::vector<float>& outPositions,
    std::vector<float>& outNormals,
    mochi::Error& error);

} // namespace superdex::studio
