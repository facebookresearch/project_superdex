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

#include "SuperDexBotLoader.h"

#include "MochiConversions.h"
#include "MochiModel.h"
#include "MochiPhysicsModule.h"
#include "SuperDexBot.h"

#include <mochi_core/geometry/model_data.h>
#include <superdex_robotics/utils/bot_utils.h>
#include <superdex_robotics/utils/file_utils.h>

FSuperDexBotLoader::FSuperDexBotLoader() {}

const USuperDexBot* FSuperDexBotLoader::ResolveBot(std::string_view Path, mochi::Error& Error)
    const {
  // If a non-empty path is provided, try to load the USuperDexBot asset at that UE path.
  if (!Path.empty()) {
    FString UEPath = UTF8_TO_TCHAR(std::string(Path).c_str());
    USuperDexBot* LoadedBot =
        Cast<USuperDexBot>(StaticLoadObject(USuperDexBot::StaticClass(), nullptr, *UEPath));
    if (LoadedBot) {
      return LoadedBot;
    }
    // If loading by path failed, maybe the path is a package path without the object name.
    // Try appending the short name (last component of the path).
    FString ShortName = FPackageName::GetShortName(UEPath);
    FString FullPath = UEPath + TEXT(".") + ShortName;
    LoadedBot =
        Cast<USuperDexBot>(StaticLoadObject(USuperDexBot::StaticClass(), nullptr, *FullPath));
    if (LoadedBot) {
      return LoadedBot;
    }
  }
  MOCHI_ERROR_SET(Error, "FUnrealBotLoader: could not resolve USuperDexBot from path");
  return nullptr;
}

superdex::robotics::BotFileType FSuperDexBotLoader::GetBotFileType(
    std::string_view Path,
    mochi::Error& Error) const {
  // NB: The stored JSON is always the fully-resolved BotPrefab (even for ModBotPrefab bots).
  // Update this should UE ever not flatten all imported .superdex_robotics
  return superdex::robotics::BotFileType::BotPrefab;
}

superdex::robotics::BotPrefab FSuperDexBotLoader::LoadBotPrefab(
    std::string_view Path,
    mochi::Error& Error) const {
  MOCHI_ERROR_RETURN(Error, {});

  const USuperDexBot* Bot = ResolveBot(Path, Error);
  MOCHI_ERROR_RETURN(Error, {});

  return LoadBotPrefab(Bot, Error);
}

superdex::robotics::BotPrefab FSuperDexBotLoader::LoadBotPrefab(
    const USuperDexBot* Bot,
    mochi::Error& Error) const {
  MOCHI_ERROR_RETURN(Error, {});

  if (!Bot) {
    MOCHI_ERROR_SET(Error, "FUnrealBotLoader::LoadBotPrefab: Bot is null");
    return {};
  }

  FString const& Json = Bot->GetBotPrefabJson();
  if (Json.IsEmpty()) {
    MOCHI_ERROR_SET(Error, "USuperDexBot has no resolved BotPrefab JSON");
    return {};
  }

  superdex::robotics::BotPrefab Params =
      superdex::robotics::ReadFromJsonString(TCHAR_TO_UTF8(*Json), Error);
  MOCHI_ERROR_RETURN(Error, {});

  // Sensor params are read straight off the filesystem by superdex_robotics, so the files the
  // import captured have to exist on disk before this prefab is handed to CreateBot.
  if (!Bot->MaterializeSensorParamFiles(Params)) {
    MOCHI_ERROR_SET(Error, "Failed to materialize the bot's captured sensor params files");
    return {};
  }

  return Params;
}

superdex::robotics::ModBotPrefab FSuperDexBotLoader::LoadModBotPrefab(
    std::string_view Path,
    mochi::Error& Error) const {
  MOCHI_ERROR_SET(Error, "FUnrealBotLoader does not support LoadModBotPrefab");
  return {};
}

mochi::ShapeHandle FSuperDexBotLoader::LoadShape(
    std::string_view Path,
    mochi::Real3 const& BakeScale,
    mochi::TransformRT const& BakeTransform,
    mochi::Context* Context,
    mochi::Error& Error) const {
  MOCHI_ERROR_RETURN(Error, {});

  // Resolve the UMochiModel asset from the UE package path stored in BotPrefab.
  // These paths were remapped from absolute disk paths to UE asset paths during import.
  FString UEPath = UTF8_TO_TCHAR(std::string(Path).c_str());
  UMochiModel* Model = LoadObject<UMochiModel>(nullptr, *UEPath);
  if (!Model) {
    MOCHI_ERROR_SET(Error, "FUnrealBotLoader::LoadShape: could not load UMochiModel at path");
    return {};
  }

  // BakeScale and BakeTransform arrive in ROS space (RH, meters) from the .superdex_bot JSON.
  // UMochiModel::GetMochiShape expects them in UE space (LH, centimeters).
  //
  // GetMochiShape applies: finalMatrix = spaceMatrix * bakeMatrix * importMatrix
  // The importMatrix (set during import to Scale(1,-1,1)) already flips geometry from ROS→UE.
  // The bakeMatrix is applied AFTER import, so it operates on UE-space geometry.
  //
  // BakeScale: Scale magnitudes are axis-agnostic, so they pass through unchanged.
  // BakeTransform: Rotation and translation need ROS→UE conversion (Y-negate + m→cm).
  TMochiToUnrealConverter<double> Converter(SuperDexSpace, UnrealCentimetersSpace);
  FVector UEBakeScale(BakeScale[0], BakeScale[1], BakeScale[2]);
  FTransform UEBakeTransform = Converter.TransformToUnreal(BakeTransform);

  return Model->GetMochiShape(UEBakeScale, UEBakeTransform);
}

mochi::ModelData FSuperDexBotLoader::LoadModelData(std::string_view Path, mochi::Error& Error)
    const {
  MOCHI_ERROR_RETURN(Error, {});

  // Resolve the UMochiModel asset from the UE package path stored in BotPrefab (remapped from an
  // absolute disk path to a UE asset path during import), mirroring LoadShape. Returns owning
  // CPU-side model data so BuildArticulatedActorParams can remap a composed skin's baked bone
  // indices before creating its shape.
  FString UEPath = UTF8_TO_TCHAR(std::string(Path).c_str());
  UMochiModel* Model = LoadObject<UMochiModel>(nullptr, *UEPath);
  if (!Model) {
    MOCHI_ERROR_SET(Error, "FSuperDexBotLoader::LoadModelData: could not load UMochiModel at path");
    return {};
  }
  return Model->GetMochiModelData(Error);
}
