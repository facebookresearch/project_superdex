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

// "Source from CAD Model" modifier: sources geometry from a CAD file, either the editor's slotted
// CAD model ("From Model Viewer") or the modifier's own slot ("From File"). A STEP (.step/.stp) is
// tessellated (with adjustable options); an STL (.stl) is read directly. This header exposes only
// the registry entry.

#include "meshing/processing_modifiers/processing_modifier.h"

namespace superdex::studio {

ModifierRegistryEntry MakeCadModelSourceEntry();

} // namespace superdex::studio
