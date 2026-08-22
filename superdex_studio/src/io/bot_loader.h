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

#include "superdex_robotics/core/loader.h"

namespace superdex::studio {

// Forwards
class AssetReferenceManager;
class AssetManager;

struct SuperDexStudioBotLoader : superdex::robotics::IBotLoader {
  SuperDexStudioBotLoader(AssetManager* manager) : _manager(manager) {}
  ~SuperDexStudioBotLoader() override = default;
  superdex::robotics::BotFileType GetBotFileType(std::string_view path, mochi::Error& error)
      const override;
  superdex::robotics::BotPrefab LoadBotPrefab(std::string_view path, mochi::Error& error)
      const override;
  superdex::robotics::ModBotPrefab LoadModBotPrefab(std::string_view path, mochi::Error& error)
      const override;
  mochi::ShapeHandle LoadShape(
      std::string_view path,
      mochi::Real3 const& bakeScale,
      mochi::TransformRT const& bakeTransform,
      mochi::Context* context,
      mochi::Error& error) const override;
  AssetManager* _manager = nullptr;
};

} // namespace superdex::studio
