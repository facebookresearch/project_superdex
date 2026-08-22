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

#include <mochi_core/utils/coordinate_space.h>

namespace mochi_renderer {

/**
 * @brief The coordinate space this renderer's scene contents are expressed in.
 *
 * @details This is FUR (X-forward, Y-up, Z-right) rather than
 * @ref mochi::CoordinateSpace::Filament (RUB). It is an asset-compatibility constraint, not a
 * preference: the checked-in `.glb` render meshes were baked through an earlier converter that
 * mislabelled Filament's axes, so they sit 90 degrees yawed from the glTF convention. gltfio loads
 * them straight into this space without passing through a
 * @ref mochi::CoordinateSpaceConverter, so this basis must match how they were authored.
 */
[[nodiscard]] inline mochi::CoordinateSpace RenderSpace() {
  return {mochi::CoordinateSpaceAxes::FUR, mochi::real{1}};
}

} // namespace mochi_renderer
