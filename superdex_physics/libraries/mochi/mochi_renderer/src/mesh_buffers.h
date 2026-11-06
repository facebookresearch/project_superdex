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

#include <filament/Box.h>

#include <mochi_core/utils/span.h>

#include <cstddef>

namespace filament {
class Engine;
class IndexBuffer;
} // namespace filament

namespace mochi_renderer {

// Builds a UINT IndexBuffer holding a copy of @p indices.
filament::IndexBuffer* CreateIndexBuffer(filament::Engine& engine, mochi::Span<int const> indices);

// The axis-aligned bounds of @p vertexCount packed xyz positions at @p posData.
filament::Box ComputeBoundingBox(float const* posData, size_t vertexCount);

} // namespace mochi_renderer
