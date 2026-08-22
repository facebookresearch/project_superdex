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

// Lossless passthrough placeholder for an unrecognized modifier/method encountered while loading a
// pipeline JSON (e.g. written by a newer Studio). It captures the original modifier name, method
// name, and raw serialized properties; renders a warning; passes its input through unchanged; and
// re-emits the captured data verbatim on save so nothing is lost on a round-trip. Created only by
// the loader -- never offered in the Add menu.

#include "meshing/processing_modifiers/processing_modifier.h"

#include <memory>
#include <string>

namespace superdex::studio {

std::unique_ptr<MeshProcessingModifier> MakeUnknownPlaceholderModifier(
    std::string modifierName,
    std::string methodName,
    std::string propertiesJson);

} // namespace superdex::studio
