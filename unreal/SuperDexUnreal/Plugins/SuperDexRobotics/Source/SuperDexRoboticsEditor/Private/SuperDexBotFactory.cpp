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

#include "SuperDexBotFactory.h"

#include "AssetRegistry/AssetRegistryModule.h"
#include "Dom/JsonObject.h"
#include "EditorFramework/AssetImportData.h"
#include "EditorReimportHandler.h"
#include "Engine/SkeletalMesh.h"
#include "Engine/StaticMesh.h"
#include "HAL/FileManager.h"
#include "Materials/MaterialInterface.h"
#include "Math/TransformCalculus3D.h"
#include "Misc/FileHelper.h"
#include "Misc/ScopedSlowTask.h"
#include "MochiArticulationAssetUtils.h"
#include "MochiConversions.h"
#include "MochiModel.h"
#include "MochiPhysicsModule.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "SkeletalMeshAttributes.h"
#include "SuperDexImportUtils.h"
#include "SuperDexRuntimeRenderUtils.h"

#include <superdex_robotics/core/loader.h>
#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/archive_utils.h>
#include <superdex_robotics/utils/bot_utils.h>
#include <superdex_robotics/utils/file_utils.h>

DEFINE_LOG_CATEGORY_STATIC(LogSuperDexBotFactory, Log, All);

static constexpr bool kCleanupStaticMeshes = false;

static TMochiToUnrealConverter<double> GetRosToUnrealConverter() {
  return TMochiToUnrealConverter<double>(kSuperDexSpace, kUnrealCentimetersSpace);
}

// Transform of a render model (or skin render model) relative to the frame it is attached to.
static FTransform MakeRenderModelTransform(
    const mochi::Quaternion& Rotation,
    const mochi::Real3& Translation,
    const mochi::Real3& Scale) {
  FTransform Transform =
      GetRosToUnrealConverter().TransformToUnreal(mochi::TransformRT(Rotation, Translation));
  Transform.SetScale3D(FVector(Scale[0], Scale[1], Scale[2]));
  return Transform;
}

// Map the bot's skin onto its links, in BotPrefab.links order: OutSkinBoneToLink receives the link
// each skin bone binds to (see superdex::robotics::ComputeSkinBoneLinkIndices), OutHiddenLinks the
// links whose render models the skin's render model stands in for (see
// superdex::robotics::ComputeSkinHiddenLinks).
static void ComputeSkinLinkMapping(
    const superdex::robotics::BotPrefab& BotPrefab,
    TArray<int32>& OutSkinBoneToLink,
    TArray<bool>& OutHiddenLinks) {
  mochi::DynamicArray<mochi::DynamicString> LinkNames;
  for (const auto& Link : BotPrefab.links) {
    LinkNames.push_back(Link.name);
  }
  const mochi::DynamicArray<int> BoneLinks =
      superdex::robotics::ComputeSkinBoneLinkIndices(BotPrefab, mochi::MakeConstSpan(LinkNames));
  OutSkinBoneToLink = TArray<int32>(BoneLinks.data(), mochi::isize(BoneLinks));
  for (int32 Bone = 0; Bone < OutSkinBoneToLink.Num(); ++Bone) {
    if (OutSkinBoneToLink[Bone] == superdex::robotics::kIndexNone) {
      UE_LOG(
          LogSuperDexBotFactory,
          Warning,
          TEXT(
              "Skin bone references link '%hs', which is not in the bot; its influence is dropped."),
          BotPrefab._skinBoneLinks[Bone].c_str());
    }
  }
  const mochi::DynamicArray<bool> HiddenLinks =
      superdex::robotics::ComputeSkinHiddenLinks(BotPrefab, mochi::MakeConstSpan(LinkNames));
  OutHiddenLinks = TArray<bool>(HiddenLinks.data(), mochi::isize(HiddenLinks));
}

// The render model of the bot's skin as a part skinned over Links (in BotPrefab.links order),
// placed as Studio's ExportSkeletalGlb places it: through the rest transform of the skin's root
// link (skin bone 0) and the skin's render model offset. SkinStaticMesh, the same file imported as
// a static mesh, supplies the materials.
static bool MakeSkinnedPart(
    const superdex::robotics::BotPrefab& BotPrefab,
    const TArray<int32>& SkinBoneToLink,
    const TArray<FMochiArticulationSkeletonLink>& Links,
    const UStaticMesh* SkinStaticMesh,
    FMochiArticulationSkinnedPart& OutPart) {
  const auto& Skin = *BotPrefab.skin;
  TArray<FString> MaterialNames;
  if (Links.IsEmpty() ||
      !SuperDexRuntimeRender::ReadSkinnedGlb(
          UTF8_TO_TCHAR(Skin.renderModelFile.c_str()),
          SkinBoneToLink,
          OutPart.Mesh,
          MaterialNames)) {
    return false;
  }
  const int32 RootLink = SkinBoneToLink.IsValidIndex(0) && Links.IsValidIndex(SkinBoneToLink[0])
      ? SkinBoneToLink[0]
      : 0;
  OutPart.MeshToRoot =
      MakeRenderModelTransform(
          Skin.renderModelRotation, Skin.renderModelTranslation, Skin.renderModelScale) *
      Links[RootLink].LinkToRoot;

  OutPart.Materials.Reset();
  for (const FString& MaterialName : MaterialNames) {
    OutPart.Materials.Add(
        SkinStaticMesh ? SuperDexRuntimeRender::FindMaterialForSlot(*SkinStaticMesh, MaterialName)
                       : nullptr);
  }
  return true;
}

// Grow each link's visual bounds (in the link's frame) to contain every skin vertex it influences.
// A skinned vertex is a weighted average of its rest position carried by each influencing link, so
// in any pose it lies in the box enclosing those links' posed bounds, which is what
// USuperDexBot::ComputeVisualBoundsFromLinkTransforms builds. A single link's bounds need not
// contain it.
static void AddSkinToLinkBounds(
    const FMochiArticulationSkinnedPart& Part,
    const TArray<FMochiArticulationSkeletonLink>& Links,
    TArray<FBoxSphereBounds>& InOutLinkBounds) {
  check(InOutLinkBounds.Num() == Links.Num());
  TArray<FBox> SkinBoxes;
  SkinBoxes.Init(FBox(ForceInit), Links.Num());
  FSkeletalMeshConstAttributes Attributes(Part.Mesh);
  TVertexAttributesConstRef<FVector3f> Positions = Attributes.GetVertexPositions();
  FSkinWeightsVertexAttributesConstRef SkinWeights = Attributes.GetVertexSkinWeights();
  for (const FVertexID VertexID : Part.Mesh.Vertices().GetElementIDs()) {
    const FVector RootPosition = Part.MeshToRoot.TransformPosition(FVector(Positions[VertexID]));
    for (const UE::AnimationCore::FBoneWeight BoneWeight : SkinWeights.Get(VertexID)) {
      const int32 Bone = BoneWeight.GetBoneIndex();
      if (BoneWeight.GetWeight() > 0.f && Links.IsValidIndex(Bone)) {
        SkinBoxes[Bone] += Links[Bone].LinkToRoot.InverseTransformPosition(RootPosition);
      }
    }
  }
  for (int32 i = 0; i < Links.Num(); ++i) {
    if (!SkinBoxes[i].IsValid) {
      continue;
    }
    FBox Box = SkinBoxes[i];
    if (InOutLinkBounds[i].SphereRadius > 0) {
      Box += InOutLinkBounds[i].GetBox();
    }
    InOutLinkBounds[i] = FBoxSphereBounds(Box);
  }
}

// Every string value in a JSON value. Keys are not collected -- only values can name a companion
// file. Use an explicit stack so nesting depth is limited by available memory rather than the C++
// call stack.
static void CollectJsonStringValues(
    const TSharedPtr<FJsonValue>& RootValue,
    TArray<FString>& OutStrings) {
  TArray<TSharedPtr<FJsonValue>> PendingValues;
  PendingValues.Add(RootValue);
  while (!PendingValues.IsEmpty()) {
    const TSharedPtr<FJsonValue> Value = PendingValues.Pop();
    if (!Value.IsValid()) {
      continue;
    }
    if (Value->Type == EJson::String) {
      OutStrings.Add(Value->AsString());
    } else if (Value->Type == EJson::Array) {
      PendingValues.Append(Value->AsArray());
    } else if (Value->Type == EJson::Object) {
      if (const TSharedPtr<FJsonObject> Object = Value->AsObject()) {
        for (const TPair<FString, TSharedPtr<FJsonValue>>& Pair : Object->Values) {
          PendingValues.Add(Pair.Value);
        }
      }
    }
  }
}

// Read the sensor params file at ParamsPath, plus every companion file it names, into OutFiles
// under KeyPrefix; OutParamsKey receives the key of the params file itself.
//
// A string value in the params JSON counts as a companion when it resolves to a real file
// beneath the params file's own directory -- the same rule superdex_robotics applies when it
// decides what a .superdex_bot_archive must carry (CollectSensorParamsReferences), and the same
// directory a sensor resolves its own relative references against at load time (e.g. the
// weightsPath of a learned sensor). Keeping the captured layout relative to the params file
// therefore keeps those references resolving once the files are written back out.
//
// A missing params file can belong to a sensor type omitted from this build. Companion discovery
// is also best-effort because the type-agnostic JSON walk cannot distinguish file fields from
// incidental strings. Both cases are reported to the caller without blocking the import.
static bool CaptureSensorParamsFiles(
    const FString& ParamsPath,
    const FString& KeyPrefix,
    TMap<FString, FSuperDexBotDataFile>& OutFiles,
    FString& OutParamsKey) {
  TArray<uint8> ParamsBytes;
  if (!FFileHelper::LoadFileToArray(ParamsBytes, *ParamsPath)) {
    return false;
  }
  TMap<FString, FSuperDexBotDataFile> Captured;
  const FString ParamsKey = KeyPrefix / FPaths::GetCleanFilename(ParamsPath);
  FSuperDexBotDataFile ParamsFile;
  ParamsFile.Contents = MoveTemp(ParamsBytes);
  Captured.Add(ParamsKey, MoveTemp(ParamsFile));

  // Read the text separately rather than decoding the bytes above: LoadFileToString handles a
  // BOM and a non-UTF-8 encoding, and the params file is a few hundred bytes. A failure here
  // only costs us companion discovery, so it is not fatal -- the params file is still captured.
  FString ParamsText;
  TSharedPtr<FJsonObject> Root;
  if (FFileHelper::LoadFileToString(ParamsText, *ParamsPath)) {
    const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(ParamsText);
    if (!FJsonSerializer::Deserialize(Reader, Root)) {
      Root.Reset();
    }
  }
  if (!Root.IsValid()) {
    // Unreadable, unparseable, or not a JSON object: no companions we can discover. Capturing
    // the params file alone is still strictly better than leaving a temp-dir path behind.
    OutFiles.Append(MoveTemp(Captured));
    OutParamsKey = ParamsKey;
    return true;
  }

  TArray<FString> Candidates;
  for (const TPair<FString, TSharedPtr<FJsonValue>>& Pair : Root->Values) {
    CollectJsonStringValues(Pair.Value, Candidates);
  }

  const FString ParamsDir =
      FPaths::ConvertRelativePathToFull(FPaths::GetPath(ParamsPath)) / TEXT("");
  for (const FString& Candidate : Candidates) {
    if (Candidate.IsEmpty()) {
      continue;
    }
    if (!FPaths::IsRelative(Candidate)) {
      // An absolute companion cannot be captured usefully: the sensor would keep using the
      // absolute string at load time, so rehoming the bytes would not help.
      if (IFileManager::Get().FileExists(*Candidate)) {
        UE_LOG(
            LogSuperDexBotFactory,
            Warning,
            TEXT(
                "Sensor params '%s' references '%s' by absolute path; it cannot be captured "
                "into the asset."),
            *ParamsPath,
            *Candidate);
      }
      continue;
    }
    const FString Resolved =
        FPaths::ConvertRelativePathToFull(FPaths::Combine(ParamsDir, Candidate));
    if (!IFileManager::Get().FileExists(*Resolved)) {
      continue;
    }
    // The captured key has to stay beneath the params file in the scratch directory.
    FString RelKey = Resolved;
    if (!FPaths::MakePathRelativeTo(RelKey, *ParamsDir) || RelKey.StartsWith(TEXT(".."))) {
      UE_LOG(
          LogSuperDexBotFactory,
          Warning,
          TEXT(
              "Companion '%s' referenced by sensor params '%s' is outside its directory "
              "and cannot be captured"),
          *Resolved,
          *ParamsPath);
      continue;
    }
    const FString Key = KeyPrefix / RelKey;
    if (Captured.Contains(Key)) {
      continue;
    }
    TArray<uint8> Bytes;
    if (!FFileHelper::LoadFileToArray(Bytes, *Resolved)) {
      UE_LOG(
          LogSuperDexBotFactory,
          Warning,
          TEXT("Could not capture companion '%s' referenced by sensor params '%s'"),
          *Resolved,
          *ParamsPath);
      continue;
    }
    FSuperDexBotDataFile CompanionFile;
    CompanionFile.Contents = MoveTemp(Bytes);
    Captured.Add(Key, MoveTemp(CompanionFile));
  }

  OutFiles.Append(MoveTemp(Captured));
  OutParamsKey = ParamsKey;
  return true;
}

static bool CaptureBotSensorParamsFiles(
    superdex::robotics::BotPrefab& BotPrefab,
    TMap<FString, FSuperDexBotDataFile>& OutFiles) {
  TMap<FString, FString> CapturedParamsKeys;
  TSet<FString> UnavailableParamsPaths;
  for (int32 i = 0; i < static_cast<int32>(BotPrefab.links.size()); ++i) {
    auto& Link = BotPrefab.links[i];
    const FString LinkKey =
        SuperDexImportUtils::SanitizeForFileName(UTF8_TO_TCHAR(std::string(Link.name).c_str()));
    for (int32 s = 0; s < static_cast<int32>(Link.sensors.size()); ++s) {
      auto& Sensor = Link.sensors[s];
      const std::string_view Params(Sensor.params.c_str(), Sensor.params.size());
      const ESuperDexBotComponentParamsKind ParamsKind = ClassifyBotComponentParams(Params);
      if (ParamsKind == ESuperDexBotComponentParamsKind::Empty ||
          ParamsKind == ESuperDexBotComponentParamsKind::InlineJson) {
        continue;
      }
      if (ParamsKind == ESuperDexBotComponentParamsKind::CapturedFile) {
        const std::string CapturedPrefix(kSuperDexBotDataPrefix);
        UE_LOG(
            LogSuperDexBotFactory,
            Error,
            TEXT("Sensor params on link '%s' use the reserved '%hs' marker"),
            *LinkKey,
            CapturedPrefix.c_str());
        return false;
      }
      if (ParamsKind == ESuperDexBotComponentParamsKind::LoaderPath) {
        UE_LOG(
            LogSuperDexBotFactory,
            Error,
            TEXT("Sensor params on link '%s' retained an unresolved loader path '%hs'"),
            *LinkKey,
            Sensor.params.c_str());
        return false;
      }
      const FString ParamsPath = UTF8_TO_TCHAR(Sensor.params.c_str());
      if (FPaths::IsRelative(ParamsPath)) {
        UE_LOG(
            LogSuperDexBotFactory,
            Error,
            TEXT("Sensor params on link '%s' remained relative after loading: '%s'"),
            *LinkKey,
            *ParamsPath);
        return false;
      }
      FString NormalizedParamsPath = FPaths::ConvertRelativePathToFull(ParamsPath);
      FPaths::NormalizeFilename(NormalizedParamsPath);
      FString ParamsKey;
      if (const FString* ExistingKey = CapturedParamsKeys.Find(NormalizedParamsPath)) {
        ParamsKey = *ExistingKey;
      } else if (UnavailableParamsPaths.Contains(NormalizedParamsPath)) {
        continue;
      } else {
        if (!CaptureSensorParamsFiles(
                NormalizedParamsPath, FString::Printf(TEXT("%d/%d"), i, s), OutFiles, ParamsKey)) {
          UE_LOG(
              LogSuperDexBotFactory,
              Warning,
              TEXT("Could not capture sensor params '%s' on link '%s'; leaving it unchanged"),
              *NormalizedParamsPath,
              *LinkKey);
          UnavailableParamsPaths.Add(NormalizedParamsPath);
          continue;
        }
        CapturedParamsKeys.Add(NormalizedParamsPath, ParamsKey);
      }
      std::string Marker(kSuperDexBotDataPrefix);
      Marker += TCHAR_TO_UTF8(*ParamsKey);
      Sensor.params = mochi::DynamicString(Marker);
    }
  }
  return true;
}

///////////////////////////////////////////////////////////////////////////////
// USuperDexBotFactory
///////////////////////////////////////////////////////////////////////////////

USuperDexBotFactory::USuperDexBotFactory(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer) {
  Formats.Add(FString(TEXT("superdex_bot;SuperDex Bot")));
  Formats.Add(FString(TEXT("superdex_bot_archive;SuperDex Bot Archive")));
  SupportedClass = USuperDexBot::StaticClass();
  bCreateNew = false;
  bEditorImport = true;
}

UObject* USuperDexBotFactory::FactoryCreateFile(
    UClass* InClass,
    UObject* InParent,
    FName InName,
    EObjectFlags Flags,
    const FString& Filename,
    const TCHAR* Parms,
    FFeedbackContext* Warn,
    bool& bOutOperationCanceled) {
  bOutOperationCanceled = false;
  return ImportBot(
      Filename,
      FPackageName::GetLongPackagePath(InParent->GetPathName()),
      InName.ToString(),
      Flags);
}

USuperDexBot* USuperDexBotFactory::ImportBot(
    const FString& Filename,
    const FString& DestPackagePath,
    const FString& FallbackBotName,
    EObjectFlags Flags) {
  // Only the .superdex_bot/.superdex_bot_archive extensions are offered for import, and the shared
  // file_utils helpers may accept more, so check the extension here.
  const std::string FilenameUtf8(TCHAR_TO_UTF8(*Filename));
  const FString Extension = FPaths::GetExtension(Filename).ToLower();
  if ((Extension != TEXT("superdex_bot") && Extension != TEXT("superdex_bot_archive")) ||
      (!superdex::robotics::IsBotPath(FilenameUtf8) &&
       !superdex::robotics::IsBotArchivePath(FilenameUtf8))) {
    UE_LOG(
        LogSuperDexBotFactory,
        Error,
        TEXT("File is not a .superdex_bot/.superdex_bot_archive file: %s"),
        *Filename);
    return nullptr;
  }

  // We need the bot name to check for re-drag and create the asset.
  mochi::ErrorLog MochiError;
  auto ResolvedParams =
      superdex::robotics::LoadBotPrefabFromFile(TCHAR_TO_UTF8(*Filename), MochiError);
  if (!MochiError.IsOK()) {
    UE_LOG(LogSuperDexBotFactory, Error, TEXT("Failed to load BotPrefab from %s"), *Filename);
    return nullptr;
  }

  FString BotName = SuperDexImportUtils::SanitizeForPackagePath(
      UTF8_TO_TCHAR(std::string(ResolvedParams.name).c_str()));
  if (BotName.IsEmpty()) {
    BotName = FallbackBotName;
  }

  // Re-drag detection: if a USuperDexBot already exists at the destination, reimport it from
  // Filename.
  FString BotAssetName = FString::Printf(TEXT("SDB_%s"), *BotName);
  FString BotAssetPath = FPaths::Combine(DestPackagePath, BotAssetName);
  if (USuperDexBot* ExistingBot = LoadObject<USuperDexBot>(
          nullptr, *(BotAssetPath + TEXT(".") + BotAssetName), nullptr, LOAD_NoWarn | LOAD_Quiet)) {
    UE_LOG(
        LogSuperDexBotFactory,
        Display,
        TEXT("SuperDexBot '%s' already exists at %s, triggering reimport"),
        *BotName,
        *BotAssetPath);
    if (ExistingBot->AssetImportData) {
      ExistingBot->AssetImportData->UpdateFilenameOnly(Filename);
    }
    return FReimportManager::Instance()->Reimport(ExistingBot, false, true) ? ExistingBot : nullptr;
  }

  UE_LOG(
      LogSuperDexBotFactory,
      Display,
      TEXT("Importing SuperDexBot '%s' from %s to %s"),
      *BotName,
      *Filename,
      *DestPackagePath);

  // Create the USuperDexBot asset and run the core import pipeline.
  UPackage* BotPackage = CreatePackage(*BotAssetPath);
  USuperDexBot* Bot = NewObject<USuperDexBot>(
      BotPackage,
      USuperDexBot::StaticClass(),
      FName(*BotAssetName),
      Flags | RF_Public | RF_Standalone);

  if (!ImportBotFromFile(Bot, Filename, DestPackagePath, Flags)) {
    return nullptr;
  }

  FAssetRegistryModule::AssetCreated(Bot);
  Bot->MarkPackageDirty();
  return Bot;
}

bool USuperDexBotFactory::ImportBotFromFile(
    USuperDexBot* Bot,
    const FString& SourcePath,
    const FString& DestPackagePath,
    EObjectFlags Flags) {
  CleanUp();

  mochi::ErrorLog MochiError;
  auto BotPrefab =
      superdex::robotics::LoadBotPrefabFromFile(TCHAR_TO_UTF8(*SourcePath), MochiError);
  if (!MochiError.IsOK()) {
    UE_LOG(LogSuperDexBotFactory, Error, TEXT("Failed to load BotPrefab from %s"), *SourcePath);
    return false;
  }

  FString BotName = SuperDexImportUtils::SanitizeForPackagePath(
      UTF8_TO_TCHAR(std::string(BotPrefab.name).c_str()));
  if (BotName.IsEmpty()) {
    BotName = Bot->BotName;
  }

  // superdex_robotics opens sensor params through the filesystem, from deep inside CreateBot,
  // with no loader hook to intercept. LoadBotPrefabFromFile resolves those paths, which for an
  // archive points into this machine's extraction temp directory. Capture the files before
  // creating any generated assets, so the stored bot is portable and a capture failure leaves
  // the existing asset untouched instead of failing after a partial mesh import.
  TMap<FString, FSuperDexBotDataFile> CapturedSensorParamFiles;
  if (!CaptureBotSensorParamsFiles(BotPrefab, CapturedSensorParamFiles)) {
    return false;
  }

  // Import render meshes and MochiModels.
  // Arrays are parallel to ResolvedParams.links (size N).
  int32 const NumLinks = static_cast<int32>(BotPrefab.links.size());
  TArray<UStaticMesh*> ImportedRenderMeshes;
  TArray<UMochiModel*> ImportedMochiModels;
  ImportedRenderMeshes.SetNum(NumLinks);
  ImportedMochiModels.SetNum(NumLinks);

  TArray<FString> NewRenderModelHashes;
  TArray<FString> NewMochiModelHashes;
  NewRenderModelHashes.SetNum(NumLinks);
  NewMochiModelHashes.SetNum(NumLinks);

  FString RenderPath = FPaths::Combine(DestPackagePath, TEXT("render"));
  FString MochiPath = FPaths::Combine(DestPackagePath, TEXT("collision"));

  FScopedSlowTask SlowTask(
      NumLinks + 1,
      FText::FromString(FString::Printf(TEXT("Importing SuperDexBot '%s'..."), *BotName)));
  SlowTask.MakeDialog();

  for (int32 i = 0; i < NumLinks; ++i) {
    auto const& Link = BotPrefab.links[i];
    FString LinkName =
        SuperDexImportUtils::SanitizeForPackagePath(UTF8_TO_TCHAR(std::string(Link.name).c_str()));

    SlowTask.EnterProgressFrame(
        1.0f,
        FText::FromString(
            FString::Printf(TEXT("Importing link '%s' (%d/%d)"), *LinkName, i + 1, NumLinks)));

    if (!Link.renderModelFile.empty()) {
      FString GlbPath = UTF8_TO_TCHAR(Link.renderModelFile.c_str());
      FString GlbName =
          SuperDexImportUtils::SanitizeForPackagePath(FPaths::GetBaseFilename(GlbPath));
      // Hash the render mesh file and compare against stored hash to skip unchanged meshes.
      mochi::ErrorLog HashErr;
      FString GlbHash(UTF8_TO_TCHAR(
          superdex::robotics::HashGenericFile(Link.renderModelFile.c_str(), HashErr).c_str()));
      NewRenderModelHashes[i] = GlbHash;
      FString ExpectedAssetPath =
          FPaths::Combine(RenderPath, FString::Printf(TEXT("SM_%s"), *GlbName));
      bool bCanReuse = !GlbHash.IsEmpty() && i < Bot->AssetImportData->RenderModelHashes.Num() &&
          GlbHash == Bot->AssetImportData->RenderModelHashes[i];
      if (bCanReuse) {
        ImportedRenderMeshes[i] = LoadObject<UStaticMesh>(nullptr, *ExpectedAssetPath);
      }
      if (ImportedRenderMeshes[i]) {
        UE_LOG(
            LogSuperDexBotFactory,
            Display,
            TEXT("Skipping reimport of render mesh '%s' (hash unchanged)"),
            *GlbName);
      } else {
        ImportedRenderMeshes[i] = ImportSession.ImportGlbToStaticMesh(GlbPath, RenderPath, GlbName);
      }
    }

    if (!Link.shapeFile.empty()) {
      FString H5Path = UTF8_TO_TCHAR(Link.shapeFile.c_str());
      FString H5Name = SuperDexImportUtils::GetMochiModelBaseName(H5Path);
      // Hash the shape mesh file and compare against stored hash to skip unchanged meshes.
      mochi::ErrorLog HashErr;
      FString H5Hash(UTF8_TO_TCHAR(
          superdex::robotics::HashGenericFile(Link.shapeFile.c_str(), HashErr).c_str()));
      NewMochiModelHashes[i] = H5Hash;
      FString ExpectedAssetPath =
          FPaths::Combine(MochiPath, FString::Printf(TEXT("MM_%s"), *H5Name));
      bool bCanReuse = !H5Hash.IsEmpty() && i < Bot->AssetImportData->MochiModelHashes.Num() &&
          H5Hash == Bot->AssetImportData->MochiModelHashes[i];
      if (bCanReuse) {
        ImportedMochiModels[i] = LoadObject<UMochiModel>(nullptr, *ExpectedAssetPath);
      }
      if (ImportedMochiModels[i]) {
        UE_LOG(
            LogSuperDexBotFactory,
            Display,
            TEXT("Skipping reimport of MochiModel '%s' (hash unchanged)"),
            *H5Name);
      } else {
        ImportedMochiModels[i] = ImportSession.ImportH5ToMochiModel(H5Path, MochiPath, H5Name);
      }
    }
  }

  // Convert to UE link arrays and generate skeletal mesh.
  TArray<FMochiArticulationSkeletonLink> Links;
  ConvertBotPrefabToLinks(BotPrefab, ImportedRenderMeshes, Links);

  // A skin with a render model stands in for the links it hides: it is skinned over the skeleton
  // and their rigid render models are left out of the skeletal mesh. The skin GLB is also imported
  // as a static mesh, for its materials. If the skin cannot be read, the hidden links keep their
  // rigid render models.
  UStaticMesh* ImportedSkinRenderMesh = nullptr;
  FString NewSkinRenderModelHash;
  TArray<FMochiArticulationSkinnedPart> SkinnedParts;
  if (BotPrefab.skin.has_value() && !BotPrefab.skin->renderModelFile.empty()) {
    FString SkinGlbPath = UTF8_TO_TCHAR(BotPrefab.skin->renderModelFile.c_str());
    FString SkinGlbName =
        SuperDexImportUtils::SanitizeForPackagePath(FPaths::GetBaseFilename(SkinGlbPath));
    mochi::ErrorLog HashErr;
    NewSkinRenderModelHash = FString(UTF8_TO_TCHAR(
        superdex::robotics::HashGenericFile(BotPrefab.skin->renderModelFile.c_str(), HashErr)
            .c_str()));
    if (!NewSkinRenderModelHash.IsEmpty() &&
        NewSkinRenderModelHash == Bot->AssetImportData->SkinRenderModelHash) {
      ImportedSkinRenderMesh = LoadObject<UStaticMesh>(
          nullptr, *FPaths::Combine(RenderPath, FString::Printf(TEXT("SM_%s"), *SkinGlbName)));
    }
    if (ImportedSkinRenderMesh) {
      UE_LOG(
          LogSuperDexBotFactory,
          Display,
          TEXT("Skipping reimport of skin render mesh '%s' (hash unchanged)"),
          *SkinGlbName);
    } else {
      ImportedSkinRenderMesh =
          ImportSession.ImportGlbToStaticMesh(SkinGlbPath, RenderPath, SkinGlbName);
    }

    TArray<int32> SkinBoneToLink;
    TArray<bool> SkinHiddenLinks;
    ComputeSkinLinkMapping(BotPrefab, SkinBoneToLink, SkinHiddenLinks);
    FMochiArticulationSkinnedPart Part;
    if (MakeSkinnedPart(BotPrefab, SkinBoneToLink, Links, ImportedSkinRenderMesh, Part)) {
      for (int32 i = 0; i < SkinHiddenLinks.Num(); ++i) {
        if (SkinHiddenLinks[i]) {
          Links[i].RenderModel = nullptr;
        }
      }
      SkinnedParts.Add(MoveTemp(Part));
    } else {
      UE_LOG(
          LogSuperDexBotFactory,
          Warning,
          TEXT("Could not use the skin of bot '%s'; showing the links it hides instead."),
          *BotName);
    }
  }

  // Import the optional collision skin (skin.shape). This is a single deformable collision
  // mesh skinned over the articulation's links; it is collision-only and is not part of the
  // per-link skeletal-mesh generation.
  UMochiModel* ImportedSkinModel = nullptr;
  FString NewSkinMochiModelHash;
  if (BotPrefab.skin.has_value() && !BotPrefab.skin->shapeFile.empty()) {
    FString SkinH5Path = UTF8_TO_TCHAR(BotPrefab.skin->shapeFile.c_str());
    FString SkinH5Name = SuperDexImportUtils::GetMochiModelBaseName(SkinH5Path);
    mochi::ErrorLog HashErr;
    NewSkinMochiModelHash = FString(UTF8_TO_TCHAR(
        superdex::robotics::HashGenericFile(BotPrefab.skin->shapeFile.c_str(), HashErr).c_str()));
    FString ExpectedAssetPath =
        FPaths::Combine(MochiPath, FString::Printf(TEXT("MM_%s"), *SkinH5Name));
    bool bCanReuse = !NewSkinMochiModelHash.IsEmpty() &&
        NewSkinMochiModelHash == Bot->AssetImportData->SkinMochiModelHash;
    if (bCanReuse) {
      ImportedSkinModel = LoadObject<UMochiModel>(nullptr, *ExpectedAssetPath);
    }
    if (ImportedSkinModel) {
      UE_LOG(
          LogSuperDexBotFactory,
          Display,
          TEXT("Skipping reimport of skin collision model '%s' (hash unchanged)"),
          *SkinH5Name);
    } else {
      ImportedSkinModel = ImportSession.ImportH5ToMochiModel(SkinH5Path, MochiPath, SkinH5Name);
    }
  }

  SlowTask.EnterProgressFrame(1.0f, FText::FromString(TEXT("Generating skeletal mesh...")));

  USkeletalMesh* SkelMesh = nullptr;
  if (Links.Num() > 0) {
    USkeleton* Skeleton = FMochiArticulationAssetUtils::GenerateSkeleton(
        Links, RenderPath, TEXT("SKEL_") + BotName, Flags);
    if (Skeleton) {
      SkelMesh = FMochiArticulationAssetUtils::GenerateSkeletalMesh(
          Links, Skeleton, RenderPath, TEXT("SKM_") + BotName, Flags, SkinnedParts);
    }
  }

  // Cache per-link visual mesh bounds before cleaning up static meshes.
  Bot->LinkBounds.SetNum(NumLinks);
  for (int32 i = 0; i < NumLinks; ++i) {
    if (ImportedRenderMeshes[i]) {
      // In the link's frame: the stored prefab drops render models, and with them their offsets.
      Bot->LinkBounds[i] =
          ImportedRenderMeshes[i]->GetBounds().TransformBy(Links[i].RenderModelToLink);
    } else {
      Bot->LinkBounds[i] = {};
    }
  }
  for (const FMochiArticulationSkinnedPart& Part : SkinnedParts) {
    AddSkinToLinkBounds(Part, Links, Bot->LinkBounds);
  }

  // Record the relative paths of every asset this bot generated so a later reimport
  // can verify they still exist (see UReimportSuperDexBotFactory::AreImportedAssetsPresent).
  // We capture only THIS bot's outputs -- its render/collision meshes, skeleton,
  // skeletal mesh, and the materials they reference -- not the whole render/collision
  // folders, which may also hold assets generated by other bots sharing the folder.
  // Paths are stored relative to the bot's package folder so the folder can be moved
  // and still reimport correctly.
  TArray<FString> NewGeneratedAssetPaths;
  {
    const FString BotFolderPrefix = DestPackagePath + TEXT("/");
    auto RecordAsset = [&](const UObject* Asset) {
      if (!Asset) {
        return;
      }
      const FString ObjectPath = Asset->GetPathName();
      // Only track assets under this bot's folder; engine defaults and assets owned
      // by other bots elsewhere are not ours to regenerate.
      if (!ObjectPath.StartsWith(BotFolderPrefix)) {
        return;
      }
      NewGeneratedAssetPaths.AddUnique(ObjectPath.RightChop(BotFolderPrefix.Len()));
    };
    auto RecordMeshMaterials = [&](const auto& Materials) {
      for (const auto& Material : Materials) {
        RecordAsset(Material.MaterialInterface);
      }
    };

    if (SkelMesh) {
      RecordAsset(SkelMesh);
      RecordAsset(SkelMesh->GetSkeleton());
      RecordMeshMaterials(SkelMesh->GetMaterials());
    }
    for (int32 i = 0; i < NumLinks; ++i) {
      if (ImportedMochiModels[i]) {
        RecordAsset(ImportedMochiModels[i]);
      }
      // The render static meshes (and their materials) persist on disk but the bot
      // holds no direct reference to them, so they must be tracked explicitly.
      if constexpr (!kCleanupStaticMeshes) {
        if (ImportedRenderMeshes[i]) {
          RecordAsset(ImportedRenderMeshes[i]);
          RecordMeshMaterials(ImportedRenderMeshes[i]->GetStaticMaterials());
        }
      }
    }
    if (ImportedSkinModel) {
      RecordAsset(ImportedSkinModel);
    }
    if constexpr (!kCleanupStaticMeshes) {
      if (ImportedSkinRenderMesh && !SkinnedParts.IsEmpty()) {
        RecordAsset(ImportedSkinRenderMesh);
        RecordMeshMaterials(ImportedSkinRenderMesh->GetStaticMaterials());
      }
    }
  }

  // Optionally clean up intermediate static meshes after skeletal mesh generation.
  if constexpr (kCleanupStaticMeshes) {
    for (int32 i = 0; i < NumLinks; ++i) {
      if (ImportedRenderMeshes[i]) {
        ImportedRenderMeshes[i]->ClearFlags(RF_Public | RF_Standalone);
        ImportedRenderMeshes[i]->MarkAsGarbage();
      }
    }
    if (ImportedSkinRenderMesh) {
      ImportedSkinRenderMesh->ClearFlags(RF_Public | RF_Standalone);
      ImportedSkinRenderMesh->MarkAsGarbage();
    }
  }

  // Populate USuperDexBot fields.
  Bot->BotName = BotName;

  // Remap model paths in the resolved params from absolute disk paths to UE asset paths
  // so that FUnrealBotLoader can resolve them to UE assets at runtime.
  for (int32 i = 0; i < NumLinks; ++i) {
    auto& Link = BotPrefab.links[i];
    if (!Link.shapeFile.empty() && ImportedMochiModels[i]) {
      FString UEPath = ImportedMochiModels[i]->GetPathName();
      Link.shapeFile = mochi::DynamicString(TCHAR_TO_UTF8(*UEPath));
    }
    if (!Link.renderModelFile.empty()) {
      // Clear render model paths - static meshes are cleaned up after skeletal mesh
      // generation, and rendering uses the skeletal mesh instead.
      Link.renderModelFile = mochi::DynamicString();
    }
  }

  // Store the sensor files captured before asset generation.
  Bot->SensorParamFiles = MoveTemp(CapturedSensorParamFiles);

  // Remap the skin collision path to its UE asset path so FSuperDexBotLoader::LoadShape
  // resolves it at runtime, and clear the render skin path: like the links' render models, the
  // skin renders through the skeletal mesh.
  if (BotPrefab.skin.has_value()) {
    if (ImportedSkinModel && !BotPrefab.skin->shapeFile.empty()) {
      FString UEPath = ImportedSkinModel->GetPathName();
      BotPrefab.skin->shapeFile = mochi::DynamicString(TCHAR_TO_UTF8(*UEPath));
    }
    BotPrefab.skin->renderModelFile = mochi::DynamicString();
  }

  // Serialize the resolved params with remapped UE asset paths.
  {
    mochi::ErrorLog JsonErr;
    auto Json = superdex::robotics::WriteToJsonString(BotPrefab, JsonErr);
    Bot->BotPrefabJson = UTF8_TO_TCHAR(Json.c_str());
  }

  Bot->SkeletalMesh = SkelMesh;
  Bot->MochiModels.Empty();
  for (int32 i = 0; i < NumLinks; ++i) {
    if (ImportedMochiModels[i]) {
      Bot->MochiModels.Add(ImportedMochiModels[i]);
    }
  }
  if (ImportedSkinModel) {
    Bot->MochiModels.Add(ImportedSkinModel);
  }

  if (Bot->AssetImportData) {
    Bot->AssetImportData->Update(SourcePath);
    Bot->AssetImportData->RenderModelHashes = MoveTemp(NewRenderModelHashes);
    Bot->AssetImportData->MochiModelHashes = MoveTemp(NewMochiModelHashes);
    Bot->AssetImportData->SkinMochiModelHash = MoveTemp(NewSkinMochiModelHash);
    Bot->AssetImportData->SkinRenderModelHash = MoveTemp(NewSkinRenderModelHash);
    Bot->AssetImportData->SourceFileHash = Bot->ComputeSourceHash();
    Bot->AssetImportData->GeneratedAssetPaths = MoveTemp(NewGeneratedAssetPaths);
  }

  // NOTE: We intentionally do not delete unreferenced ("stale") MochiModel assets
  // from the destination folder. Existing assets are updated in place (see
  // FSuperDexImportSession::ImportH5ToMochiModel) so references are preserved, but the
  // folder itself is left user-managed: multiple bots may share it, and users may add
  // their own assets.
  // Cruft can accumulate, but that is preferable to clobbering shared or user-owned
  // assets. To purge cruft, a user deletes the unwanted files and reimports the bots.

  UE_LOG(
      LogSuperDexBotFactory,
      Display,
      TEXT("Successfully imported SuperDexBot '%s' (%d links, %d mochi models)"),
      *BotName,
      NumLinks,
      Bot->MochiModels.Num());

  return true;
}

void USuperDexBotFactory::CleanUp() {
  Super::CleanUp();
  ImportSession.Reset();
}

void USuperDexBotFactory::ConvertBotPrefabToLinks(
    const superdex::robotics::BotPrefab& Params,
    const TArray<UStaticMesh*>& ImportedRenderMeshes,
    TArray<FMochiArticulationSkeletonLink>& OutLinks) {
  auto Converter = GetRosToUnrealConverter();

  int32 const NumLinks = static_cast<int32>(Params.links.size());
  OutLinks.SetNum(NumLinks);

  for (int32 i = 0; i < NumLinks; ++i) {
    auto const& BotLink = Params.links[i];
    FMochiArticulationSkeletonLink& Link = OutLinks[i];

    Link.Name = FName(UTF8_TO_TCHAR(std::string(BotLink.name).c_str()));
    Link.ParentIndex = BotLink.parentLink;
    Link.LinkToParent = Converter.TransformToUnreal(BotLink._parentFromLink);
    Link.LinkToRoot = Converter.TransformToUnreal(BotLink._rootFromLink);

    if (i < ImportedRenderMeshes.Num()) {
      Link.RenderModel = ImportedRenderMeshes[i];
      if (Link.RenderModel && !BotLink.renderModelFile.empty()) {
        Link.RenderModelToLink = MakeRenderModelTransform(
            BotLink.renderModelRotation, BotLink.renderModelTranslation, BotLink.renderModelScale);
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// UReimportSuperDexBotFactory
///////////////////////////////////////////////////////////////////////////////

UReimportSuperDexBotFactory::UReimportSuperDexBotFactory() {
  ImportPriority = DefaultImportPriority - 1;
}

bool UReimportSuperDexBotFactory::CanReimport(UObject* Obj, TArray<FString>& OutFilenames) {
  USuperDexBot* Bot = Cast<USuperDexBot>(Obj);
  if (!Bot) {
    return false;
  }
  if (Bot->AssetImportData) {
    Bot->AssetImportData->ExtractFilenames(OutFilenames);
  }
  return true;
}

void UReimportSuperDexBotFactory::SetReimportPaths(
    UObject* Obj,
    const TArray<FString>& NewReimportPaths) {
  USuperDexBot* Bot = Cast<USuperDexBot>(Obj);
  if (!Bot || NewReimportPaths.Num() == 0) {
    return;
  }
  if (Bot->AssetImportData) {
    Bot->AssetImportData->UpdateFilenameOnly(NewReimportPaths[0]);
  }
}

// Returns true if every asset this bot generated on its last import is still
// present. ComputeSourceHash() only hashes the on-disk source files, so a hash match
// alone does not prove the generated UE assets still exist; if the user deleted any
// of them (a collider, a render mesh, a material) we must reimport to recreate them.
// Rather than walking asset dependency chains, we check the list of generated asset
// paths recorded at import time. Paths are relative to the bot's package folder and
// resolved against its current folder, so moving the folder still resolves.
bool UReimportSuperDexBotFactory::AreImportedAssetsPresent(const USuperDexBot* Bot) {
  if (!Bot->GetSkeletalMesh()) {
    return false;
  }
  // An empty MochiModels array is correct for a bot with no colliders, but if the
  // prefab defines collider shapes yet no models are present, a prior import likely
  // failed partway -- force a reimport instead of treating the bot as fully present.
  if (Bot->GetMochiModels().IsEmpty()) {
    for (const auto& Link : Bot->GetBotPrefab().links) {
      if (!Link.shapeFile.empty()) {
        return false;
      }
    }
  }
  for (const UMochiModel* Model : Bot->GetMochiModels()) {
    if (!Model) {
      return false;
    }
  }
  // An empty cache means either the bot has no generated assets (unexpected -- it
  // always has at least a skeletal mesh) or it was imported before this cache
  // existed. Either way, force a reimport so the cache gets populated and future
  // deletion checks are accurate.
  if (!Bot->AssetImportData || Bot->AssetImportData->GeneratedAssetPaths.IsEmpty()) {
    return false;
  }
  // Resolve each cached relative path against the bot's current package folder and
  // confirm the asset still loads.
  const FString BotFolder = FPackageName::GetLongPackagePath(Bot->GetOutermost()->GetName());
  for (const FString& RelPath : Bot->AssetImportData->GeneratedAssetPaths) {
    const FString ObjectPath = FPaths::Combine(BotFolder, RelPath);
    if (!LoadObject<UObject>(nullptr, *ObjectPath)) {
      UE_LOG(
          LogSuperDexBotFactory,
          Display,
          TEXT("Generated asset missing (%s), forcing reimport"),
          *ObjectPath);
      return false;
    }
  }
  return true;
}

// True when the bot's stored prefab still names a sensor params file by path, i.e. the
// import-time capture into USuperDexBot::SensorParamFiles has not been applied to this asset.
//
// Such an asset is out of date even though its *source* is unchanged: what changed is the
// importer, and ComputeSourceHash only covers the source files on disk. Without this, the
// hash-match early-out in Reimport reports success and does nothing, leaving every bot
// imported before the capture existed holding a params path that resolves only on the machine
// that first imported it -- with no way for the user to tell the reimport actually no-opped.
static bool NeedsSensorParamsCapture(const USuperDexBot* Bot) {
  const FString& Json = Bot->GetBotPrefabJson();
  if (Json.IsEmpty()) {
    return false;
  }
  // Read the stored JSON rather than GetBotPrefab(): the cached prefab has already had its
  // `botdata:` markers rewritten to absolute scratch paths by MaterializeSensorParamFiles, so a
  // captured sensor would be indistinguishable from an uncaptured one there.
  mochi::ErrorLog JsonErr;
  const auto Prefab = superdex::robotics::ReadFromJsonString(TCHAR_TO_UTF8(*Json), JsonErr);
  if (!JsonErr.IsOK()) {
    const char* ErrorDescription = JsonErr._error.GetDescription();
    if (ErrorDescription == nullptr) {
      ErrorDescription = "Unknown JSON parse failure";
    }
    UE_LOG(
        LogSuperDexBotFactory,
        Warning,
        TEXT("SuperDexBot '%s' has invalid stored prefab JSON; forcing reimport: %hs"),
        *Bot->GetBotName(),
        ErrorDescription);
    return true;
  }
  for (const auto& Link : Prefab.links) {
    for (const auto& Sensor : Link.sensors) {
      const std::string_view Params(Sensor.params.c_str(), Sensor.params.size());
      const ESuperDexBotComponentParamsKind ParamsKind = ClassifyBotComponentParams(Params);
      if (ParamsKind == ESuperDexBotComponentParamsKind::FilePath ||
          ParamsKind == ESuperDexBotComponentParamsKind::LoaderPath) {
        return true;
      }
    }
  }
  return false;
}

EReimportResult::Type UReimportSuperDexBotFactory::Reimport(UObject* Obj) {
  USuperDexBot* Bot = Cast<USuperDexBot>(Obj);
  if (!Bot) {
    return EReimportResult::Failed;
  }

  FString SourcePath;
  if (Bot->AssetImportData) {
    SourcePath = Bot->AssetImportData->GetFirstFilename();
  }
  if (SourcePath.IsEmpty()) {
    UE_LOG(LogSuperDexBotFactory, Error, TEXT("No source file path for reimport"));
    return EReimportResult::Failed;
  }

  if (!FPaths::FileExists(SourcePath)) {
    UE_LOG(LogSuperDexBotFactory, Error, TEXT("Source file not found: %s"), *SourcePath);
    return EReimportResult::Failed;
  }

  // Hash check: skip reimport only if the source is unchanged AND all previously
  // imported assets are still present AND the importer has nothing new to bake in.
  // ComputeSourceHash() hashes the on-disk source files (.superdex_bot, .glb, .mochi.h5),
  // not the imported UE assets, so a hash match alone does not guarantee the outputs
  // still exist. If the user deleted generated assets (e.g. the collision folder),
  // reimport to recreate them.
  const FString CurrentHash = Bot->ComputeSourceHash();
  const bool bSourceUnchanged =
      !CurrentHash.IsEmpty() && CurrentHash == Bot->AssetImportData->SourceFileHash;
  const bool bAssetsPresent = bSourceUnchanged && AreImportedAssetsPresent(Bot);
  const bool bNeedsSensorParamsCapture = bAssetsPresent && NeedsSensorParamsCapture(Bot);
  if (bAssetsPresent && !bNeedsSensorParamsCapture) {
    UE_LOG(
        LogSuperDexBotFactory,
        Display,
        TEXT("SuperDexBot '%s' is already up to date (hash match), skipping reimport"),
        *Bot->BotName);
    return EReimportResult::Succeeded;
  }
  if (bNeedsSensorParamsCapture) {
    UE_LOG(
        LogSuperDexBotFactory,
        Display,
        TEXT(
            "SuperDexBot '%s' still references its sensor params by path; reimporting to "
            "capture them into the asset even though the source is unchanged."),
        *Bot->BotName);
  }

  UE_LOG(
      LogSuperDexBotFactory,
      Display,
      TEXT("Reimporting SuperDexBot '%s' from %s"),
      *Bot->BotName,
      *SourcePath);

  // Generated assets live in the bot's folder, wherever its first import (e.g. a SuperDex scene)
  // put it.
  const FString DestPackagePath = FPackageName::GetLongPackagePath(Bot->GetOutermost()->GetName());
  if (!ImportBotFromFile(Bot, SourcePath, DestPackagePath, RF_Public | RF_Standalone)) {
    return EReimportResult::Failed;
  }

  Bot->EnsureBotPrefabCached(true);
  if (Bot->OnSkeletalMeshRegenerated && Bot->SkeletalMesh) {
    Bot->OnSkeletalMeshRegenerated(Bot->SkeletalMesh);
  }
  Bot->MarkPackageDirty();
  return EReimportResult::Succeeded;
}

int32 UReimportSuperDexBotFactory::GetPriority() const {
  return ImportPriority;
}
