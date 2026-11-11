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

#include "CoreMinimal.h"

#include <superdex_robotics/core/loader.h>

class USuperDexBot;

/// Custom IBotLoader that loads BotPrefab from USuperDexBot assets in the UE Content Browser.
///
/// Instead of reading from disk, this loader resolves a UE asset path to a USuperDexBot,
/// then deserializes the stored JSON. This allows the superdex_robotics API
/// (e.g., ComputeLinkTransformsFromPose) to operate on imported bot data.
///
/// The path argument to LoadBotPrefab/LoadModBotPrefab/GetBotFileType should be
/// the UE package path of the USuperDexBot asset (e.g., "/Game/Bots/my_bot/my_bot").
/// For convenience, an empty path can be provided if the loader is constructed with
/// a default USuperDexBot, which will be used as a fallback.
class SUPERDEXROBOTICS_API FSuperDexBotLoader : public superdex::robotics::IBotLoader {
 public:
  /// Construct with a default bot. If a path is provided to the loader methods and
  /// resolves to a USuperDexBot, that bot is used. Otherwise, falls back to DefaultBot.
  explicit FSuperDexBotLoader();
  ~FSuperDexBotLoader() override = default;

  superdex::robotics::BotFileType GetBotFileType(std::string_view path, mochi::Error& error)
      const override;

  superdex::robotics::BotPrefab LoadBotPrefab(std::string_view path, mochi::Error& error)
      const override;

  /// Convenience overload: loads BotPrefab directly from a USuperDexBot pointer.
  superdex::robotics::BotPrefab LoadBotPrefab(const USuperDexBot* Bot, mochi::Error& error) const;

  superdex::robotics::ModBotPrefab LoadModBotPrefab(std::string_view path, mochi::Error& error)
      const override;

  mochi::ShapeHandle LoadShape(
      std::string_view path,
      mochi::Real3 const& bakeScale,
      mochi::TransformRT const& bakeTransform,
      mochi::Context* context,
      mochi::Error& error) const override;

  /// Load raw model data (mesh + skinning) for a skin whose baked bone indices must be remapped
  /// onto a composed skeleton (the non-identity skin path in BuildArticulatedActorParams). Resolves
  /// the UE asset path stored in BotPrefab to a UMochiModel, same as LoadShape.
  mochi::ModelData LoadModelData(std::string_view path, mochi::Error& error) const override;

 private:
  /// Resolve a path to a USuperDexBot. Tries to load as a UE asset path first,
  /// falls back to DefaultBot if the path is empty or resolution fails.
  const USuperDexBot* ResolveBot(std::string_view path, mochi::Error& error) const;
};
