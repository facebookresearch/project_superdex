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

// "Wrap Mesh" modifier: builds a fresh closed envelope around the input (shrink wrap, convex hull,
// alpha wrap). Each is a method; this header exposes only the registry entry.

#include "meshing/processing_modifiers/processing_modifier.h"

namespace superdex::studio {

ModifierRegistryEntry MakeWrapMeshEntry();

} // namespace superdex::studio
