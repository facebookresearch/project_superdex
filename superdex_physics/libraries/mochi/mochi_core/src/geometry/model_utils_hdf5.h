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

// Implements model::LoadFromFileUnchecked for FileFormat::H5.
void LoadFromFileUncheckedH5(ModelData& outData, std::string_view filePath, Error& error);

// Implements model::LoadFromBytesUnchecked for FileFormat::H5.
void LoadFromBytesUncheckedH5(ModelData& outData, Span<char const> fileData, Error& error);

// Implements model::SaveToFile for FileFormat::H5.
void SaveToFileH5(ModelDataView const& data, std::string_view filePath, Error& error);

// This can be called just before overwriting an existing file. If the existing file contained
// experimental data that cannot be represented using the ModelData struct, then we log a warning to
// alert the user.
void WarnIfOverwritingExperimentalDataH5(std::string_view filePath);

} // namespace mochi::model
