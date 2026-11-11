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

#include "SuperDexBotIKConfigFactory.h"

#include <superdex_robotics/ik/bot_ik_config.h>
#include <limits>
#include "AssetRegistry/AssetRegistryModule.h"
#include "Dom/JsonObject.h"
#include "EditorFramework/AssetImportData.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "SuperDexBot.h"
#include "SuperDexBotIKConfig.h"

DEFINE_LOG_CATEGORY_STATIC(LogSuperDexBotIKConfigFactory, Log, All);

namespace {

// Native parsing owns schema/version/type validation. Check the original doubles as well,
// before narrowing can silently disable a nonzero stiffness or erase a supplied coordinate.
bool ValidateJsonNativePrecision(const TSharedPtr<FJsonValue>& Value, const FString& Path) {
  if (Value->Type == EJson::Number &&
      Path != TEXT("createParams.solverParams.maxElapsedTimeSeconds")) {
    const double Number = Value->AsNumber();
    if (!FMath::IsFinite(Number) ||
        FMath::Abs(Number) > static_cast<double>(std::numeric_limits<mochi::real>::max()) ||
        (Number != 0.0 && static_cast<mochi::real>(Number) == 0)) {
      UE_LOG(
          LogSuperDexBotIKConfigFactory,
          Warning,
          TEXT(
              "IK config '%s' is not representable in native precision (including nonzero underflow)."),
          *Path);
      return false;
    }
  } else if (Value->Type == EJson::Object) {
    for (const auto& Entry : Value->AsObject()->Values) {
      const FString ChildPath = Path.IsEmpty() ? FString(Entry.Key) : Path + TEXT(".") + Entry.Key;
      if (!ValidateJsonNativePrecision(Entry.Value, ChildPath)) {
        return false;
      }
    }
  } else if (Value->Type == EJson::Array) {
    const auto& Values = Value->AsArray();
    for (int32 Index = 0; Index < Values.Num(); ++Index) {
      if (!ValidateJsonNativePrecision(
              Values[Index], FString::Printf(TEXT("%s[%d]"), *Path, Index))) {
        return false;
      }
    }
  }
  return true;
}

} // namespace

FString USuperDexBotIKConfigFactory::NormalizeSourcePath(const FString& Path) {
  if (Path.IsEmpty()) {
    return {};
  }
  FString Result = FPaths::ConvertRelativePathToFull(Path);
  FPaths::NormalizeFilename(Result);
  FPaths::CollapseRelativeDirectories(Result);
  return Result;
}

USuperDexBot* USuperDexBotIKConfigFactory::ResolveBotAsset(const FString& SourcePath) {
  if (SourcePath.IsEmpty()) {
    return nullptr;
  }
  IAssetRegistry& Registry =
      FModuleManager::LoadModuleChecked<FAssetRegistryModule>("AssetRegistry").Get();
  Registry.SearchAllAssets(true);
  TArray<FAssetData> Assets;
  Registry.GetAssetsByClass(USuperDexBot::StaticClass()->GetClassPathName(), Assets, true);
  TArray<FAssetData> Matches;
  for (const auto& Data : Assets) {
    FString SourceJson;
    if (!Data.GetTagValue(USuperDexBot::SourceFileTagName(), SourceJson)) {
      continue;
    }
    const auto SourceInfo = FAssetImportInfo::FromJson(SourceJson);
    if (!SourceInfo.IsSet() || SourceInfo->SourceFiles.IsEmpty()) {
      continue;
    }
    const FString PackageName = Data.PackageName.ToString();
    const FString ImportedPath = UAssetImportData::ResolveImportFilename(
        FStringView(SourceInfo->SourceFiles[0].RelativeFilename), FStringView(PackageName));
    if (FPaths::IsSamePath(NormalizeSourcePath(ImportedPath), NormalizeSourcePath(SourcePath))) {
      Matches.Add(Data);
    }
  }
  if (Matches.Num() != 1) {
    UE_LOG(
        LogSuperDexBotIKConfigFactory,
        Warning,
        TEXT(
            "IK config bot reference '%s' matches %d imported bots. Select the output/input bot in Details; no match is selected automatically."),
        *SourcePath,
        Matches.Num());
    return nullptr;
  }
  USuperDexBot* Bot = Cast<USuperDexBot>(Matches[0].GetAsset());
  if (!Bot) {
    UE_LOG(
        LogSuperDexBotIKConfigFactory,
        Warning,
        TEXT("Could not load the matched bot for '%s'. Select a bot in Details."),
        *SourcePath);
  }
  return Bot;
}

USuperDexBotIKConfigFactory::USuperDexBotIKConfigFactory() {
  SupportedClass = USuperDexBotIKConfig::StaticClass();
  bEditorImport = true;
  Formats.Add(TEXT("superdex_ik;SuperDex Bot IK Configuration"));
}

bool USuperDexBotIKConfigFactory::FactoryCanImport(const FString& Filename) {
  return FPaths::GetExtension(Filename).Equals(TEXT("superdex_ik"), ESearchCase::IgnoreCase);
}

bool USuperDexBotIKConfigFactory::ImportConfig(
    USuperDexBotIKConfig* Asset,
    const FString& Filename,
    bool bReimport) {
  if (!IsInGameThread() || !Asset) {
    return false;
  }
  FString SourceText;
  if (!FFileHelper::LoadFileToString(SourceText, *Filename)) {
    UE_LOG(LogSuperDexBotIKConfigFactory, Warning, TEXT("Could not load IK config: %s"), *Filename);
    return false;
  }
  superdex::robotics::BotIKConfig Parsed;
  mochi::DynamicString Diagnostic;
  if (!superdex::robotics::ParseIKConfig(TCHAR_TO_UTF8(*SourceText), Parsed, Diagnostic)) {
    UE_LOG(
        LogSuperDexBotIKConfigFactory,
        Warning,
        TEXT("Could not parse IK config: %hs"),
        Diagnostic.c_str());
    return false;
  }
  if (!Diagnostic.empty()) {
    UE_LOG(LogSuperDexBotIKConfigFactory, Warning, TEXT("IK config: %hs"), Diagnostic.c_str());
  }
  // Parse and validate the entire document before touching the destination asset. Keep the
  // native parser's duplicate-binding warning behavior; runtime Validate still rejects duplicates.
  TSharedPtr<FJsonValue> Json;
  if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(SourceText), Json) ||
      !Json.IsValid() || !ValidateJsonNativePrecision(Json, TEXT(""))) {
    return false;
  }
  const FString AbsoluteFilename = NormalizeSourcePath(Filename);
  const auto NativeOutputBotPath =
      superdex::robotics::ResolveIKBotPath(TCHAR_TO_UTF8(*AbsoluteFilename), Parsed.outputBot);
  const auto NativeInputBotPath =
      superdex::robotics::ResolveIKBotPath(TCHAR_TO_UTF8(*AbsoluteFilename), Parsed.inputBot);
  const FString OutputBotPath = NormalizeSourcePath(UTF8_TO_TCHAR(NativeOutputBotPath.c_str()));
  const FString InputBotPath = NormalizeSourcePath(UTF8_TO_TCHAR(NativeInputBotPath.c_str()));
  const bool bPreserveOutputBot = bReimport && Asset->OutputBot &&
      !Asset->SourceOutputBotPath.IsEmpty() &&
      FPaths::IsSamePath(NormalizeSourcePath(Asset->SourceOutputBotPath), OutputBotPath);
  const bool bPreserveInputBot = bReimport && Asset->InputBot &&
      !Asset->SourceInputBotPath.IsEmpty() &&
      FPaths::IsSamePath(NormalizeSourcePath(Asset->SourceInputBotPath), InputBotPath);
  USuperDexBot* OutputBot =
      bPreserveOutputBot ? Asset->OutputBot.Get() : ResolveBotAsset(OutputBotPath);
  USuperDexBot* InputBot =
      bPreserveInputBot ? Asset->InputBot.Get() : ResolveBotAsset(InputBotPath);
  // Missing or ambiguous assets are editable unresolved references, not a reason to drop bindings.
  Asset->Modify();
  Asset->FromNative(Parsed);
  Asset->OutputBot = OutputBot;
  Asset->InputBot = InputBot;
  Asset->SourceOutputBotPath = OutputBotPath;
  Asset->SourceInputBotPath = InputBotPath;
  if (!Asset->AssetImportData) {
    Asset->AssetImportData = NewObject<UAssetImportData>(Asset, TEXT("AssetImportData"));
  }
  Asset->AssetImportData->Update(AbsoluteFilename);
  Asset->MarkPackageDirty();
  return true;
}

UObject* USuperDexBotIKConfigFactory::FactoryCreateFile(
    UClass* InClass,
    UObject* InParent,
    FName InName,
    EObjectFlags Flags,
    const FString& Filename,
    const TCHAR* Parms,
    FFeedbackContext* Warn,
    bool& bOutOperationCanceled) {
  bOutOperationCanceled = false;
  auto* Asset = NewObject<USuperDexBotIKConfig>(InParent, InClass, InName, Flags);
  if (!ImportConfig(Asset, Filename, false)) {
    Asset->MarkAsGarbage();
    return nullptr;
  }
  return Asset;
}

USuperDexBotIKConfigFactoryNew::USuperDexBotIKConfigFactoryNew() {
  SupportedClass = USuperDexBotIKConfig::StaticClass();
  bCreateNew = true;
  bEditAfterNew = true;
}

UObject* USuperDexBotIKConfigFactoryNew::FactoryCreateNew(
    UClass* InClass,
    UObject* InParent,
    FName InName,
    EObjectFlags Flags,
    UObject* Context,
    FFeedbackContext* Warn) {
  return NewObject<USuperDexBotIKConfig>(InParent, InClass, InName, Flags);
}

UReimportSuperDexBotIKConfigFactory::UReimportSuperDexBotIKConfigFactory() {
  bEditorImport = false;
}

bool UReimportSuperDexBotIKConfigFactory::CanReimport(UObject* Obj, TArray<FString>& OutFilenames) {
  const auto* Asset = Cast<USuperDexBotIKConfig>(Obj);
  if (!Asset || !Asset->AssetImportData || Asset->AssetImportData->GetFirstFilename().IsEmpty()) {
    return false;
  }
  Asset->AssetImportData->ExtractFilenames(OutFilenames);
  return true;
}

void UReimportSuperDexBotIKConfigFactory::SetReimportPaths(
    UObject* Obj,
    const TArray<FString>& Paths) {
  if (auto* Asset = Cast<USuperDexBotIKConfig>(Obj);
      Asset && Asset->AssetImportData && Paths.Num() == 1) {
    Asset->AssetImportData->UpdateFilenameOnly(Paths[0]);
  }
}

EReimportResult::Type UReimportSuperDexBotIKConfigFactory::Reimport(UObject* Obj) {
  auto* Asset = Cast<USuperDexBotIKConfig>(Obj);
  if (!Asset || !Asset->AssetImportData ||
      !ImportConfig(Asset, Asset->AssetImportData->GetFirstFilename(), true)) {
    return EReimportResult::Failed;
  }
  return EReimportResult::Succeeded;
}

int32 UReimportSuperDexBotIKConfigFactory::GetPriority() const {
  return ImportPriority;
}
