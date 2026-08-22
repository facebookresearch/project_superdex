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

#include <mochi_renderer/types.h>

namespace filament {
class ColorGrading;
class Engine;
class View;
} // namespace filament

namespace mochi_renderer {

// Applies the Filament View configuration shared by Scene (interactive viewport) and
// ObservationCamera (offscreen render) from @p settings: post-processing, MSAA, bloom, GTAO ambient
// occlusion, vignette, shadows, and color grading / tone mapping. @p colorGrading is destroyed (if
// non-null) and rebuilt in place, then assigned to @p view. Skybox visibility is intentionally not
// handled here (only Scene has a skybox), so callers apply it themselves.
void ApplyViewSettingsToView(
    filament::Engine* engine,
    filament::View* view,
    filament::ColorGrading*& colorGrading,
    SceneViewSettings const& settings);

} // namespace mochi_renderer
