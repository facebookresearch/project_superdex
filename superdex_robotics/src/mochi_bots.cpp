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

#include <superdex_robotics/core/loader.h>
#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/archive_utils.h>
#include <superdex_robotics/utils/bot_utils.h>

#if MOCHI_USE_REFLECTION
#include <simple_reflection/simple_reflection.h>
#endif

using namespace mochi;
using namespace superdex::robotics;

BotPrefab superdex::robotics::LoadBotPrefabFromFile(std::string_view path, Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  constexpr bool validate = true;
  if (IsBotArchivePath(path) && std::filesystem::is_regular_file(std::string(path))) {
    auto extractedDir = ExtractBotArchiveToCache(path, error);
    auto targetPath = GetExtractedBotArchiveTarget(extractedDir, error);
    return LoadBotPrefab(targetPath, FileBotLoader{}, validate, error);
  }
  return LoadBotPrefab(path, FileBotLoader{}, validate, error);
}

Bot* superdex::robotics::CreateBot(
    Scene* scene,
    BotPrefab const& botPrefab,
    RoboticsContext* botsContext,
    Error& error) {
  MOCHI_ERROR_RETURN(error, nullptr);
  MOCHI_ERROR_IF(botsContext == nullptr, error, "RoboticsContext is null");
  MOCHI_ERROR_RETURN(error, nullptr);
  return botsContext->CreateBot(scene, botPrefab, FileBotLoader{}, error);
}

void superdex::robotics::DestroyBot(Scene* scene, Bot* bot) {
  if (bot == nullptr) {
    return;
  }
  if (auto* botsContext = bot->GetBotContext(); botsContext != nullptr) {
    botsContext->DestroyBot(scene, bot->GetHandle());
  }
}
