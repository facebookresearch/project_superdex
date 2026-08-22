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

#include "io/bot_loader.h"
#include "app/app.h"
#include "assets/asset_manager.h"
#include "assets/bot_asset.h"
#include "assets/mochi_model_asset.h"

#include <mochi_renderer/resource_manager.h>

namespace superdex::studio {

superdex::robotics::BotFileType SuperDexStudioBotLoader::GetBotFileType(
    std::string_view path,
    mochi::Error& error) const {
  MOCHI_ERROR_RETURN(error, {});
  if (auto* asset = _manager->FindAssetByPath<BotAsset>(path)) {
    return asset->GetBotFileType();
  }
  MOCHI_ERROR_SET(error, "Path is not a Mochi Bot");
  return {};
}

superdex::robotics::BotPrefab SuperDexStudioBotLoader::LoadBotPrefab(
    std::string_view path,
    mochi::Error& error) const {
  MOCHI_ERROR_RETURN(error, {});
  auto* asset = _manager->FindAssetByPath<BotAsset>(path);
  if (asset && asset->GetBotFileType() == superdex::robotics::BotFileType::BotPrefab) {
    asset->Rebuild(*this);
    return asset->GetBotPrefab();
  }
  MOCHI_ERROR_SET(error, "Path is not a superdex::robotics::BotPrefab");
  return {};
}

superdex::robotics::ModBotPrefab SuperDexStudioBotLoader::LoadModBotPrefab(
    std::string_view path,
    mochi::Error& error) const {
  MOCHI_ERROR_RETURN(error, {});
  auto* asset = _manager->FindAssetByPath<BotAsset>(path);
  if (asset && asset->GetBotFileType() == superdex::robotics::BotFileType::ModBotPrefab) {
    return asset->GetModBotPrefab();
  }
  MOCHI_ERROR_SET(error, "Path is not a superdex::robotics::ModBotPrefab");
  return {};
}

mochi::ShapeHandle SuperDexStudioBotLoader::LoadShape(
    std::string_view path,
    mochi::Real3 const& bakeScale,
    mochi::TransformRT const& bakeTransform,
    mochi::Context* context,
    mochi::Error& error) const {
  MOCHI_ERROR_RETURN(error, {});
  // The in-memory asset path builds shapes on the studio's context, so the caller must be using the
  // same context. Otherwise the returned shape would belong to a different context than expected.
  MOCHI_ASSERT_VERBOSE(
      context == _manager->GetStudio()->GetMochiContext(),
      "LoadShape context must match the studio's Mochi context.");
  if (auto* model = _manager->FindAssetByPath<MochiModelAsset>(mochi::Path{std::string(path)})) {
    auto shape = model->GetShape(bakeScale, bakeTransform, error);
    if (shape.IsValid()) {
      return shape;
    }
  }
  // Warn because this really shouldn't happen.
  MOCHI_LOG_WARNING(
      "Bot loader failed to load shape from MochiModelAsset at path: %s",
      std::string(path).c_str());
  error = {};
  superdex::robotics::FileBotLoader loader;
  return loader.LoadShape(path, bakeScale, bakeTransform, context, error);
}

} // namespace superdex::studio
