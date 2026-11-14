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

#include <superdex_robotics/superdex_robotics.h>

#include <string_view>

namespace mochi::prefab {
struct ScenePrefab;
} // namespace mochi::prefab

namespace superdex::robotics {

/* Sets the fields of `edit`, a DiffPrefab result, in `prefab` and leaves the rest. Fails for an
   element the prefab lacks, a null field, or a value that does not fit its field; the prefab may
   then be partly set. */
void ApplyPrefabEdit(std::string_view edit, BotPrefab& prefab, superdex::Error& error);
void ApplyPrefabEdit(
    std::string_view edit,
    mochi::prefab::ScenePrefab& prefab,
    superdex::Error& error);

} // namespace superdex::robotics
