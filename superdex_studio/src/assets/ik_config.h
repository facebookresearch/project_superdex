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

#include <mochi_renderer/path.h>
#include <superdex_robotics/ik/bot_ik_config.h>

namespace superdex::studio {

using robotics::BotIKConfig;
using robotics::BotIKGoalConfig;
using robotics::IKGoalConstraintTransform;
using robotics::kIKConfigVersion;
using robotics::kIKInitialPositionStiffness;
using robotics::kIKInitialRotationStiffness;
using robotics::kIKInitialSolveBudgetSeconds;
using robotics::SetIKGoalConstraintTransform;

inline std::string SerializeStudioIKConfig(BotIKConfig const& config) {
  return std::string(robotics::SerializeIKConfig(config));
}
inline mochi::Path ResolveIKBotPath(mochi::Path const& doc, std::string_view ref) {
  return mochi::Path(std::string(robotics::ResolveIKBotPath(doc.ToString(), ref)));
}
inline std::string MakeIKBotReference(mochi::Path const& doc, mochi::Path const& bot) {
  return std::string(robotics::MakeIKBotReference(doc.ToString(), bot.ToString()));
}
inline bool LoadIKConfig(mochi::Path const& path, BotIKConfig& config, std::string& diagnostic) {
  mochi::DynamicString message;
  bool const result = robotics::LoadIKConfig(path.ToString(), config, message);
  diagnostic.assign(message.begin(), message.end());
  return result;
}
inline bool
SaveIKConfig(mochi::Path const& path, BotIKConfig const& config, std::string& diagnostic) {
  mochi::DynamicString message;
  bool const result = robotics::SaveIKConfig(path.ToString(), config, message);
  diagnostic.assign(message.begin(), message.end());
  return result;
}
inline bool ParseIKConfig(std::string_view text, BotIKConfig& config, std::string& diagnostic) {
  mochi::DynamicString message;
  bool const result = robotics::ParseIKConfig(text, config, message);
  diagnostic.assign(message.begin(), message.end());
  return result;
}
inline bool ValidateIKConfig(BotIKConfig const& config, std::string& diagnostic) {
  mochi::DynamicString message;
  bool const result = robotics::ValidateIKConfig(config, message);
  diagnostic.assign(message.begin(), message.end());
  return result;
}
inline bool ValidateIKGoalBindings(BotIKConfig const& config, std::string& diagnostic) {
  mochi::DynamicString message;
  bool const result = robotics::ValidateIKGoalBindings(config, message);
  diagnostic.assign(message.begin(), message.end());
  return result;
}
inline bool InitializeIKGoals(
    robotics::BotPrefab const& outputBot,
    BotIKConfig& config,
    std::string& diagnostic) {
  mochi::DynamicString message;
  bool const result = robotics::InitializeIKGoals(outputBot, config, message);
  diagnostic.assign(message.begin(), message.end());
  return result;
}

} // namespace superdex::studio
