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

#include <mochi_core/geometry/model_data.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/span.h>

#include <string_view>

namespace mochi::model {

// Loads an OBJ surface mesh into @p outData using tiny_obj_loader. Non-triangle faces are
// triangulated on load.
void LoadObjFromFile(ModelData& outData, std::string_view path, Error& error);
void LoadObjFromBytes(ModelData& outData, Span<char const> data, Error& error);

// Loads an STL surface mesh into @p outData (binary or ASCII, auto-detected). STL stores each
// triangle with independent vertices, so coincident vertices are welded on load.
void LoadStlFromFile(ModelData& outData, std::string_view path, Error& error);
void LoadStlFromBytes(ModelData& outData, Span<char const> data, Error& error);

// Loads a PLY surface mesh into @p outData using happly (little-endian binary or ASCII). Polygon
// faces are fan-triangulated and coincident vertices are welded on load.
void LoadPlyFromFile(ModelData& outData, std::string_view path, Error& error);
void LoadPlyFromBytes(ModelData& outData, Span<char const> data, Error& error);

// Loads an OFF surface mesh into @p outData (ASCII). Polygon faces are fan-triangulated. OFF stores
// shared vertices, so no welding is performed.
void LoadOffFromFile(ModelData& outData, std::string_view path, Error& error);
void LoadOffFromBytes(ModelData& outData, Span<char const> data, Error& error);

} // namespace mochi::model
