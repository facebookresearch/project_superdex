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

#include <mochi_renderer/windows_compat.h> // Must be first — cleans up Windows macros before Filament headers

#include <mochi_renderer/material.h>

namespace mochi_renderer {

MaterialInstance::MaterialInstance(
    filament::Engine* engine,
    filament::MaterialInstance* materialInstance)
    : _engine(engine), _materialInstance(materialInstance) {}

MaterialInstance::~MaterialInstance() {
  if (_materialInstance && _engine) {
    _engine->destroy(_materialInstance);
    _materialInstance = nullptr;
  }
  _engine = nullptr;
}

filament::MaterialInstance* MaterialInstance::Get() const {
  return _materialInstance;
}

} // namespace mochi_renderer
