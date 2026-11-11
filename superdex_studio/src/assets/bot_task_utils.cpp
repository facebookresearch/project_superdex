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

#include "assets/bot_task_utils.h"

#include <superdex_robotics/scene/bot_task.h>
#include <superdex_robotics/utils/file_utils.h>

namespace superdex::studio {

namespace {

bool SaveTask(
    superdex::robotics::BotTaskPrefab const& task,
    std::string const& path,
    std::string& error) {
  superdex::Error saveError;
  superdex::robotics::SaveParamsToFile(task, path, saveError);
  if (!saveError.IsOK()) {
    error = saveError.GetDescription();
    return false;
  }
  return true;
}

} // namespace

bool CreateBotTaskFile(std::string const& path, std::string_view name, std::string& error) {
  superdex::robotics::BotTaskPrefab task;
  task.metadata.name = mochi::DynamicString{std::string(name)};
  return SaveTask(task, path, error);
}

bool SetBotTaskName(std::string const& path, std::string_view name, std::string& error) {
  superdex::Error loadError;
  auto task = superdex::robotics::LoadBotTaskPrefabFromFile(path, loadError);
  if (!loadError.IsOK()) {
    error = loadError.GetDescription();
    return false;
  }
  task.metadata.name = mochi::DynamicString{std::string(name)};
  return SaveTask(task, path, error);
}

} // namespace superdex::studio
