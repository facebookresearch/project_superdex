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

#include <string>
#include <string_view>

namespace superdex::robotics {
struct TerminationCriteria;
} // namespace superdex::robotics

namespace superdex::studio {

// Writes a new .superdex_task at @p path with no spawns and its metadata name set to @p name.
bool CreateBotTaskFile(std::string const& path, std::string_view name, std::string& error);

// Sets the metadata name of the .superdex_task at @p path, keeping the rest of the task.
bool SetBotTaskName(std::string const& path, std::string_view name, std::string& error);

// Points references to task object @p from, including sub-actor addresses "<from>.<sub>", at @p to.
// An empty @p from, or one containing '.', matches no reference.
void RenameTaskObjectReferences(
    superdex::robotics::TerminationCriteria& criteria,
    std::string_view from,
    std::string_view to);

} // namespace superdex::studio
