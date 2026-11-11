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

#include "SuperDexBot.h"

#include "HAL/FileManager.h"
#include "HAL/PlatformTime.h"
#include "Misc/FileHelper.h"
#include "Misc/Guid.h"
#include "Misc/Paths.h"
#include "Misc/ScopeLock.h"
#include "Misc/SecureHash.h"
#include "MochiConversions.h"
#include "MochiModel.h"
#include "MochiPhysicsModule.h"
#include "MochiTypes.h"
#include "SuperDexBotLoader.h"

#include "CanvasItem.h"
#include "CanvasTypes.h"
#include "SceneManagement.h"
#include "SceneView.h"
#include "UnrealClient.h"

#if WITH_EDITOR
#include "EditorReimportHandler.h"
#endif

#include "superdex_robotics/utils/file_utils.h"

#include <superdex_robotics/superdex_robotics.h>

#include <superdex_robotics/utils/bot_utils.h>

#if WITH_EDITOR

void USuperDexBot::PostInitProperties() {
  Super::PostInitProperties();
  if (!HasAnyFlags(RF_ClassDefaultObject)) {
    AssetImportData = NewObject<USuperDexBotImportData>(this, TEXT("AssetImportData"));
  }
}

void USuperDexBot::PostLoad() {
  Super::PostLoad();
  // Upgrade: replace plain UAssetImportData with USuperDexBotImportData.
  if (AssetImportData && !AssetImportData->IsA<USuperDexBotImportData>()) {
    UAssetImportData* OldData = AssetImportData;
    auto* NewData = NewObject<USuperDexBotImportData>(this, NAME_None, RF_NoFlags, nullptr);
    // Migrate source file info from the old subobject.
    TArray<FAssetImportInfo::FSourceFile> files = OldData->GetSourceData().SourceFiles;
    NewData->SetSourceFiles(MoveTemp(files));
    AssetImportData = NewData;
    // Mark dirty so the upgrade is saved.
    MarkPackageDirty();
  }
}

void USuperDexBot::GetAssetRegistryTags(FAssetRegistryTagsContext Context) const {
  if (AssetImportData) {
    Context.AddTag(FAssetRegistryTag(
        SourceFileTagName(),
        AssetImportData->GetSourceData().ToJson(),
        FAssetRegistryTag::TT_Hidden));
  }
  Super::GetAssetRegistryTags(Context);
}

void USuperDexBot::Reimport() {
  FReimportManager::Instance()->Reimport(this, false, true);
}

FString USuperDexBot::GetSourceFilePath() const {
  if (AssetImportData) {
    return AssetImportData->GetFirstFilename();
  }
  return FString();
}

FString USuperDexBot::ComputeSourceHash() const {
  FString SourcePath = GetSourceFilePath();
  if (SourcePath.IsEmpty()) {
    return FString();
  }
  mochi::ErrorLog Error;
  auto Hash = superdex::robotics::HashBotFile(TCHAR_TO_UTF8(*SourcePath), Error);
  if (!Error.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("[USuperDexBot] Failed to hash source file: %s"),
        *SourcePath);
    return FString();
  }
  return FString(UTF8_TO_TCHAR(Hash.c_str()));
}

bool USuperDexBot::IsOutOfSync() const {
  FString SourcePath = GetSourceFilePath();
  if (SourcePath.IsEmpty() || AssetImportData->SourceFileHash.IsEmpty()) {
    return true;
  }
  if (!IFileManager::Get().FileExists(*SourcePath)) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("[USuperDexBot] Source file no longer exists: %s"),
        *SourcePath);
    return false;
  }
  return ComputeSourceHash() != AssetImportData->SourceFileHash;
}

#endif // WITH_EDITOR

const FString& USuperDexBot::GetBotName() const {
  return BotName;
}

USkeletalMesh* USuperDexBot::GetSkeletalMesh() const {
  return SkeletalMesh;
}

const TArray<UMochiModel*>& USuperDexBot::GetMochiModels() const {
  return MochiModels;
}

const FString& USuperDexBot::GetBotPrefabJson() const {
  return BotPrefabJson;
}

const TArray<FBoxSphereBounds>& USuperDexBot::GetLinkBounds() const {
  return LinkBounds;
}

// --- UMochiArticulationBase overrides ---

const FString& USuperDexBot::GetArticulationName() const {
  return BotName;
}

int USuperDexBot::GetNumDofs() const {
  if (EnsureBotPrefabCached()) {
    return CachedBotPrefab._numDofs;
  }
  UE_LOG(
      LogMochiPhysics,
      Error,
      TEXT("[USuperDexBot] GetNumDofs failed bot prefab could not be ensured!"));
  return 0;
}

const TArray<double>& USuperDexBot::GetDefaultJointPositions() const {
  if (EnsureBotPrefabCached()) {
    CachedDefaultJointPositions.SetNum(CachedBotPrefab._numDofs);
    for (int32 i = 0; i < CachedBotPrefab._numDofs; ++i) {
      CachedDefaultJointPositions[i] = static_cast<double>(CachedBotPrefab.defaultPose[i]);
    }
    return CachedDefaultJointPositions;
  }
  UE_LOG(
      LogMochiPhysics,
      Error,
      TEXT("[USuperDexBot] GetDefaultJointPositions failed bot prefab could not be ensured!"));
  return CachedDefaultJointPositions;
}

TArray<FName> USuperDexBot::GetLinkNames() const {
  if (EnsureBotPrefabCached()) {
    TArray<FName> Names;
    Names.SetNum(static_cast<int32>(CachedBotPrefab.links.size()));
    for (int32 i = 0; i < Names.Num(); ++i) {
      Names[i] = FName(UTF8_TO_TCHAR(std::string(CachedBotPrefab.links[i].name).c_str()));
    }
    return Names;
  }
  UE_LOG(
      LogMochiPhysics,
      Error,
      TEXT("[USuperDexBot] GetLinkNames failed bot prefab could not be ensured!"));
  return {};
}

TArray<FName> USuperDexBot::GetJointNames() const {
  if (EnsureBotPrefabCached()) {
    TArray<FName> Names;
    Names.SetNum(static_cast<int32>(CachedBotPrefab.joints.size()));
    for (int32 i = 0; i < Names.Num(); ++i) {
      Names[i] = FName(UTF8_TO_TCHAR(std::string(CachedBotPrefab.joints[i].name).c_str()));
    }
    return Names;
  }
  UE_LOG(
      LogMochiPhysics,
      Error,
      TEXT("[USuperDexBot] GetJointNames failed bot prefab could not be ensured!"));
  return {};
}

int32 USuperDexBot::GetDofIndex(FName JointName) const {
  if (!EnsureBotPrefabCached()) {
    return INDEX_NONE;
  }
  const auto& DofJoints = CachedBotPrefab._dofIndices;
  for (int32 Dof = 0; Dof < static_cast<int32>(DofJoints.size()); ++Dof) {
    const auto& Joint = CachedBotPrefab.joints[DofJoints[Dof]];
    if (FName(UTF8_TO_TCHAR(std::string(Joint.name).c_str())) == JointName) {
      return Dof;
    }
  }
  return INDEX_NONE;
}

FName USuperDexBot::GetDofLinkName(int32 DofIndex) const {
  if (!EnsureBotPrefabCached()) {
    return NAME_None;
  }
  const auto& DofJoints = CachedBotPrefab._dofIndices;
  if (DofIndex < 0 || DofIndex >= static_cast<int32>(DofJoints.size())) {
    return NAME_None;
  }
  // Joint i joins link i to its parent, so it moves link i.
  const int32 Link = DofJoints[DofIndex];
  if (Link < 0 || Link >= static_cast<int32>(CachedBotPrefab.links.size())) {
    return NAME_None;
  }
  return FName(UTF8_TO_TCHAR(std::string(CachedBotPrefab.links[Link].name).c_str()));
}

FString USuperDexBot::ToUrdfXml(const FString& VisualRelPath, const FString& CollisionRelPath)
    const {
  if (EnsureBotPrefabCached()) {
    auto str = superdex::robotics::SaveToUrdfString(CachedBotPrefab, mochi::ErrorLog{});
    return ANSI_TO_TCHAR(str.c_str());
  }
  UE_LOG(
      LogMochiPhysics,
      Error,
      TEXT("[USuperDexBot] ToUrdfXml failed bot prefab could not be ensured!"));
  return {};
}

bool USuperDexBot::MaterializeSensorParamFiles(superdex::robotics::BotPrefab& Prefab) const {
  const auto GetCapturedKey = [](const auto& Sensor, FString& OutKey) -> bool {
    const std::string_view Params(Sensor.params.c_str(), Sensor.params.size());
    if (!Params.starts_with(kSuperDexBotDataPrefix)) {
      return false;
    }
    const std::string Key(Params.substr(kSuperDexBotDataPrefix.size()));
    OutKey = UTF8_TO_TCHAR(Key.c_str());
    return true;
  };

  TArray<FString> ReferencedParamsKeys;
  TArray<FString> MissingParamsKeys;
  for (const auto& Link : Prefab.links) {
    for (const auto& Sensor : Link.sensors) {
      FString Key;
      if (!GetCapturedKey(Sensor, Key)) {
        continue;
      }
      if (!SensorParamFiles.Contains(Key)) {
        MissingParamsKeys.AddUnique(Key);
        continue;
      }
      ReferencedParamsKeys.AddUnique(Key);
    }
  }
  if (ReferencedParamsKeys.IsEmpty() && MissingParamsKeys.IsEmpty()) {
    return true;
  }

  FScopeLock MaterializationLock(&SensorParamFilesMaterializationMutex);
  const double Now = FPlatformTime::Seconds();
  if (NextSensorParamFilesMaterializationAttemptSeconds > Now) {
    UE_LOG(
        LogMochiPhysics,
        Verbose,
        TEXT("[%hs] Deferring another sensor params materialization attempt for bot '%s'"),
        __FUNCTION__,
        *BotName);
    return false;
  }
  if (!MissingParamsKeys.IsEmpty()) {
    NextSensorParamFilesMaterializationAttemptSeconds =
        FMath::Max(NextSensorParamFilesMaterializationAttemptSeconds, Now + 5.0);
  }
  if (!MissingParamsKeys.IsEmpty()) {
    for (const FString& Key : MissingParamsKeys) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT(
              "[%hs] Bot '%s' names captured sensor params '%s' that the asset does not "
              "carry; reimport the bot."),
          __FUNCTION__,
          *BotName,
          *Key);
    }
    return false;
  }

  //
  // Saved/ is writable in the editor and in a packaged build alike, and is disposable --
  // everything under here is reproduced from the asset on demand.
  //
  // Include both the package path and captured contents in the directory name. This lets two
  // processes holding different revisions of the same asset use different immutable directories.
  // Do not prune sibling versions here: another editor or packaged process may still use one.
  TArray<FString> SortedKeys;
  SortedKeys.Reserve(SensorParamFiles.Num());
  TArray<FString> ReferencedParamsPrefixes;
  ReferencedParamsPrefixes.Reserve(ReferencedParamsKeys.Num());
  for (const FString& ParamsKey : ReferencedParamsKeys) {
    const FString ParamsDir = FPaths::GetPath(ParamsKey);
    ReferencedParamsPrefixes.Add(ParamsDir.IsEmpty() ? FString() : ParamsDir + TEXT("/"));
  }
  // Companion files are captured beside each params file, under the params key's parent.
  for (const TPair<FString, FSuperDexBotDataFile>& Entry : SensorParamFiles) {
    for (int32 i = 0; i < ReferencedParamsKeys.Num(); ++i) {
      if (Entry.Key == ReferencedParamsKeys[i] ||
          (!ReferencedParamsPrefixes[i].IsEmpty() &&
           Entry.Key.StartsWith(ReferencedParamsPrefixes[i]))) {
        SortedKeys.Add(Entry.Key);
        break;
      }
    }
  }
  SortedKeys.Sort();

  FString MaterializedDir = CachedMaterializedSensorParamFilesDir;
  if (MaterializedDir.IsEmpty()) {
    FString ContentIdentity;
    int32 IdentityLength = 0;
    for (const FString& Key : SortedKeys) {
      IdentityLength += Key.Len() + 64;
    }
    ContentIdentity.Reserve(IdentityLength);
    for (const FString& Key : SortedKeys) {
      const FSuperDexBotDataFile& File = SensorParamFiles.FindChecked(Key);
      FSHAHash ContentHash;
      FSHA1::HashBuffer(File.Contents.GetData(), File.Contents.Num(), ContentHash.Hash);
      ContentIdentity.Appendf(TEXT("%d:%s:%s;"), Key.Len(), *Key, *ContentHash.ToString());
    }

    const FTCHARToUTF8 AssetPathUtf8(*GetPathName());
    FSHAHash AssetPathHash;
    FSHA1::HashBuffer(AssetPathUtf8.Get(), AssetPathUtf8.Length(), AssetPathHash.Hash);
    const FTCHARToUTF8 ContentIdentityUtf8(*ContentIdentity);
    FSHAHash ContentIdentityHash;
    FSHA1::HashBuffer(
        ContentIdentityUtf8.Get(), ContentIdentityUtf8.Length(), ContentIdentityHash.Hash);
    MaterializedDir = FPaths::ConvertRelativePathToFull(
        FPaths::Combine(
            FPaths::ProjectSavedDir(),
            TEXT("SDB"),
            FString::Printf(
                TEXT("%s_%s"),
                *AssetPathHash.ToString().Left(12),
                *ContentIdentityHash.ToString())));
    CachedMaterializedSensorParamFilesDir = MaterializedDir;
  }

  const auto ResolveCapturedPath = [&MaterializedDir](
                                       const FString& Key, FString& OutPath) -> bool {
    FString RelativeKey = Key;
    FPaths::NormalizeFilename(RelativeKey);
    OutPath = FPaths::ConvertRelativePathToFull(FPaths::Combine(MaterializedDir, RelativeKey));
    return !RelativeKey.IsEmpty() && FPaths::IsRelative(RelativeKey) &&
        FPaths::IsUnderDirectory(OutPath, MaterializedDir);
  };

  bool bOk = true;
  TMap<FString, FString> ResolvedCapturedPaths;
  ResolvedCapturedPaths.Reserve(SortedKeys.Num());
  for (const FString& Key : SortedKeys) {
    const FSuperDexBotDataFile& File = SensorParamFiles.FindChecked(Key);
    FString OutPath;
    if (!ResolveCapturedPath(Key, OutPath)) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[%hs] Bot '%s' has an invalid captured sensor params key '%s'"),
          __FUNCTION__,
          *BotName,
          *Key);
      bOk = false;
      continue;
    }
    ResolvedCapturedPaths.Add(Key, OutPath);

    // The directory names this exact key/content set and each file is published atomically.
    // A same-sized file here therefore came from this content version; avoid reading large
    // weights files back on the bot-load path. A later reload can recreate a missing file or
    // recover from a transient write failure.
    if (IFileManager::Get().FileSize(*OutPath) == File.Contents.Num()) {
      continue;
    }
    IFileManager::Get().MakeDirectory(*FPaths::GetPath(OutPath), /*Tree=*/true);
    const FString TempPath = FString::Printf(
        TEXT("%s.%s.tmp"), *OutPath, *FGuid::NewGuid().ToString(EGuidFormats::Digits));
    if (FFileHelper::SaveArrayToFile(File.Contents, *TempPath) &&
        IFileManager::Get().Move(*OutPath, *TempPath, /*Replace=*/true)) {
      continue;
    }
    IFileManager::Get().Delete(
        *TempPath, /*RequireExists=*/false, /*EvenReadOnly=*/false, /*Quiet=*/true);
    // A concurrent process materializing the same content may have won the rename race.
    if (IFileManager::Get().FileSize(*OutPath) == File.Contents.Num()) {
      continue;
    }
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[%hs] Failed to write captured sensor params file for '%s' to '%s'"),
        __FUNCTION__,
        *BotName,
        *OutPath);
    bOk = false;
  }

  for (auto& Link : Prefab.links) {
    for (auto& Sensor : Link.sensors) {
      FString Key;
      if (!GetCapturedKey(Sensor, Key)) {
        continue;
      }
      const FString* OutPath = ResolvedCapturedPaths.Find(Key);
      if (OutPath == nullptr) {
        bOk = false;
        continue;
      }
      Sensor.params = mochi::DynamicString(TCHAR_TO_UTF8(**OutPath));
    }
  }
  if (!bOk) {
    NextSensorParamFilesMaterializationAttemptSeconds = FMath::Max(
        NextSensorParamFilesMaterializationAttemptSeconds, FPlatformTime::Seconds() + 5.0);
  }
  return bOk;
}

bool USuperDexBot::EnsureBotPrefabCached(bool bForceReload) const {
  if (bForceReload) {
    bHasCachedBotPrefab = false;
    FScopeLock Lock(&SensorParamFilesMaterializationMutex);
    CachedMaterializedSensorParamFilesDir.Reset();
    NextSensorParamFilesMaterializationAttemptSeconds = 0.0;
  }
  if (bHasCachedBotPrefab) {
    return true;
  }
  if (BotPrefabJson.IsEmpty()) {
    bHasCachedBotPrefab = false;
    CachedBotPrefab = {};
    return false;
  }
  FSuperDexBotLoader Loader;
  mochi::ErrorLog e{};
  CachedBotPrefab = Loader.LoadBotPrefab(this, e);
  if (!e.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[%hs] Failed to load BotPrefab for '%s': %hs"),
        __FUNCTION__,
        *BotName,
        e._error.GetDescription());
    bHasCachedBotPrefab = false;
    CachedBotPrefab = {};
    return false;
  }
  // NB: FSuperDexBotLoader::LoadBotPrefab already ran MaterializeSensorParamFiles on this
  // prefab, and reported a failure through `e` above.
  bHasCachedBotPrefab = true;
  return true;
}

const superdex::robotics::BotPrefab& USuperDexBot::GetBotPrefab() const {
  EnsureBotPrefabCached();
  return CachedBotPrefab;
}

namespace {

// Whether each bot-space DOF (root-joint DOFs excluded, the pose layout of
// superdex::robotics::ComputeLinkTransformsFromPose) is prismatic. Joints are per link, so a DOF's
// index is not its joint's index.
TArray<bool> GetPrismaticDofs(const superdex::robotics::BotPrefab& BotPrefab) {
  TArray<bool> PrismaticDofs;
  for (size_t iLink = 1; iLink < BotPrefab.joints.size(); ++iLink) {
    switch (BotPrefab.joints[iLink].type) {
      case mochi::ArticulatedJointType::Revolute:
        PrismaticDofs.Add(false);
        break;
      case mochi::ArticulatedJointType::Prismatic:
        PrismaticDofs.Add(true);
        break;
      case mochi::ArticulatedJointType::Spherical:
        PrismaticDofs.Add(false);
        PrismaticDofs.Add(false);
        PrismaticDofs.Add(false);
        break;
      case mochi::ArticulatedJointType::Free:
      case mochi::ArticulatedJointType::Hard:
      case mochi::ArticulatedJointType::Cycle:
      case mochi::ArticulatedJointType::Count:
        break;
    }
  }
  return PrismaticDofs;
}

superdex::robotics::LinkTransformSpace ToBotLinkTransformSpace(
    EMochiArticulationLinkTransformSpace TransformSpace) {
  return TransformSpace == EMochiArticulationLinkTransformSpace::LinkToParent
      ? superdex::robotics::LinkTransformSpace::ParentFromLink
      : superdex::robotics::LinkTransformSpace::RootFromParent;
}

} // namespace

bool USuperDexBot::ComputeLinkTransformsFromJointPositions(
    const TArray<double>& JointPositions,
    TArray<FTransform>& OutLinkTransforms,
    EMochiArticulationLinkTransformSpace TransformSpace,
    double AngleScale,
    double LengthScale,
    int FinalLinkIndex) const {
  if (!EnsureBotPrefabCached()) {
    return false;
  }
  if (JointPositions.Num() != CachedBotPrefab._numDofs) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[%hs] JointPositions.Num() (%d) != NumDofs (%d) for '%s'"),
        __FUNCTION__,
        JointPositions.Num(),
        CachedBotPrefab._numDofs,
        *BotName);
    return false;
  }
  const TArray<bool> PrismaticDofs = GetPrismaticDofs(CachedBotPrefab);
  mochi::DynamicArray<mochi::real> Pose;
  Pose.resize_noinit(JointPositions.Num());
  for (int32 i = 0; i < JointPositions.Num(); ++i) {
    const bool isPrismatic = PrismaticDofs.IsValidIndex(i) && PrismaticDofs[i];
    mochi::real scale = isPrismatic ? LengthScale : AngleScale;
    Pose[i] = static_cast<mochi::real>(JointPositions[i] * scale);
  }
  mochi::DynamicArray<mochi::TransformRT> MochiTransforms;
  mochi::ErrorLog e{};
  superdex::robotics::ComputeLinkTransformsFromPose(
      CachedBotPrefab,
      Pose,
      MochiTransforms,
      ToBotLinkTransformSpace(TransformSpace),
      e,
      FinalLinkIndex);
  if (!e.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[%hs] ComputeLinkTransformsFromPose failed for '%s': %hs"),
        __FUNCTION__,
        *BotName,
        e._error.GetDescription());
    return false;
  }
  TMochiToUnrealConverter<double> Converter(kSuperDexSpace, kUnrealCentimetersSpace);
  int32 NumLinks = static_cast<int32>(MochiTransforms.size());
  OutLinkTransforms.SetNum(NumLinks);
  for (int32 i = 0; i < NumLinks; ++i) {
    OutLinkTransforms[i] = Converter.TransformToUnreal(MochiTransforms[i]);
  }
  return true;
}

bool USuperDexBot::ComputeJointPositionsFromLinkTransforms(
    const TArray<FTransform>& LinkTransforms,
    TArray<double>& OutJointPositions,
    EMochiArticulationLinkTransformSpace TransformSpace,
    double AngleScale,
    double LengthScale) const {
  if (!EnsureBotPrefabCached()) {
    return false;
  }
  const int32 NumLinks = static_cast<int32>(CachedBotPrefab.links.size());
  if (LinkTransforms.Num() != NumLinks) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[%hs] LinkTransforms.Num() (%d) != NumLinks (%d) for '%s'"),
        __FUNCTION__,
        LinkTransforms.Num(),
        NumLinks,
        *BotName);
    return false;
  }
  // (output space, input space)
  TUnrealToMochiConverter<double> Converter(kSuperDexSpace, kUnrealCentimetersSpace);
  mochi::DynamicArray<mochi::TransformRT> MochiTransforms;
  MochiTransforms.resize(NumLinks);
  for (int32 i = 0; i < NumLinks; ++i) {
    MochiTransforms[i] = Converter.TransformToMochi_RT(LinkTransforms[i]);
  }
  mochi::DynamicArray<mochi::real> Pose;
  mochi::ErrorLog e{};
  superdex::robotics::ComputePoseFromLinkTransforms(
      CachedBotPrefab, MochiTransforms, Pose, ToBotLinkTransformSpace(TransformSpace), e);
  if (!e.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[%hs] ComputePoseFromLinkTransforms failed for '%s': %hs"),
        __FUNCTION__,
        *BotName,
        e._error.GetDescription());
    return false;
  }
  const TArray<bool> PrismaticDofs = GetPrismaticDofs(CachedBotPrefab);
  const int32 NumDofs = static_cast<int32>(Pose.size());
  OutJointPositions.SetNum(NumDofs);
  for (int32 i = 0; i < NumDofs; ++i) {
    const bool isPrismatic = PrismaticDofs.IsValidIndex(i) && PrismaticDofs[i];
    OutJointPositions[i] = static_cast<double>(Pose[i]) / (isPrismatic ? LengthScale : AngleScale);
  }
  return true;
}

bool USuperDexBot::ComputeVisualBoundsFromLinkTransforms(
    const TArray<FTransform>& LinksToTransformSpace,
    EMochiArticulationLinkTransformSpace TransformSpace,
    FBoxSphereBounds& OutBounds) const {
  if (!EnsureBotPrefabCached()) {
    return false;
  }
  int32 NumLinks = FMath::Min(
      LinksToTransformSpace.Num(),
      FMath::Min(LinkBounds.Num(), static_cast<int32>(CachedBotPrefab.links.size())));
  if (NumLinks == 0) {
    return false;
  }
  TArray<FTransform> LinksToRoot;
  if (TransformSpace == EMochiArticulationLinkTransformSpace::LinkToParent) {
    LinksToRoot.SetNum(NumLinks);
    for (int32 i = 0; i < NumLinks; ++i) {
      int32 iParent = CachedBotPrefab.links[i].parentLink;
      if (iParent < 0 || iParent >= NumLinks) {
        LinksToRoot[i] = LinksToTransformSpace[i];
      } else {
        LinksToRoot[i] = LinksToTransformSpace[i] * LinksToRoot[iParent];
      }
    }
  }
  const TArray<FTransform>& LinkToRootTransforms =
      TransformSpace == EMochiArticulationLinkTransformSpace::LinkToRoot ? LinksToTransformSpace
                                                                         : LinksToRoot;
  TMochiToUnrealConverter<double> Converter(kSuperDexSpace, kUnrealCentimetersSpace);
  FBox CompositeBox(ForceInit);
  for (int32 i = 0; i < NumLinks; ++i) {
    if (LinkBounds[i].SphereRadius <= 0) {
      continue;
    }
    FTransform MeshToRoot = LinkToRootTransforms[i];
    if (!CachedBotPrefab.links[i].renderModelFile.empty()) {
      auto const& link = CachedBotPrefab.links[i];
      mochi::TransformRT RT(link.renderModelRotation, link.renderModelTranslation);
      FTransform VisualMeshTransform = Converter.TransformToUnreal(RT);
      VisualMeshTransform.SetScale3D(
          FVector(link.renderModelScale[0], link.renderModelScale[1], link.renderModelScale[2]));
      MeshToRoot = VisualMeshTransform * LinkToRootTransforms[i];
    }
    FBox TransformedBox = LinkBounds[i].GetBox().TransformBy(MeshToRoot);
    CompositeBox += TransformedBox;
  }
  if (!CompositeBox.IsValid) {
    OutBounds = FBoxSphereBounds(FVector::ZeroVector, FVector::ZeroVector, 0);
    return false;
  }
  OutBounds = FBoxSphereBounds(CompositeBox);
  return true;
}

void USuperDexBot::ComputeOverlayStats(double& OutMass, int& OutNumSurfaceFaces) const {
  OutMass = 0.0;
  OutNumSurfaceFaces = 0;
  auto const& params = GetBotPrefab();
  for (auto const& l : params.links) {
    if (l.mass.has_value()) {
      OutMass += *l.mass;
    }
  }
  for (auto const& m : GetMochiModels()) {
    if (m) {
      OutNumSurfaceFaces += m->NumSurfaceFaces;
    }
  }
}

void USuperDexBot::DrawCustom(
    const FSceneView* View,
    FPrimitiveDrawInterface* PDI,
    bool bShowLinkTransforms,
    bool bShowJoints,
    bool bShowBounds,
    FMaterialRenderProxy* JointMaterialRenderProxy) const {
  static constexpr FColor kPurpleColor = FColor(138, 43, 226, 255);
  TMochiToUnrealConverter<double> Conv(kSuperDexSpace, kUnrealCentimetersSpace);
  auto const& P = GetBotPrefab();
  mochi::DynamicArray<mochi::real> Pose(P.defaultPose.begin(), P.defaultPose.end());
  mochi::DynamicArray<mochi::TransformRT> RT;
  mochi::ErrorLog Err;
  superdex::robotics::ComputeLinkTransformsFromPose(
      P, Pose, RT, superdex::robotics::LinkTransformSpace::RootFromParent, Err);
  if (!Err.IsOK()) {
    return;
  }
  for (int32 i = 0; i < (int32)P.links.size(); ++i) {
    FTransform L2R = Conv.TransformToUnreal(RT[i]);
    FVector loc = L2R.GetLocation();
    if (bShowLinkTransforms) {
      FRotationMatrix R(L2R.GetRotation().Rotator());
      float s = 5.f;
      PDI->DrawLine(
          loc, loc + R.GetScaledAxis(EAxis::X) * s, FLinearColor::Red, SDPG_Foreground, 0.1f, 1.f);
      PDI->DrawLine(
          loc,
          loc + R.GetScaledAxis(EAxis::Y) * s,
          FLinearColor::Green,
          SDPG_Foreground,
          0.1f,
          1.f);
      PDI->DrawLine(
          loc, loc + R.GetScaledAxis(EAxis::Z) * s, FLinearColor::Blue, SDPG_Foreground, 0.1f, 1.f);
    }
    if (bShowJoints && i > 0) {
      auto const& jt = P.joints[i];
      if (jt.type == mochi::ArticulatedJointType::Revolute ||
          jt.type == mochi::ArticulatedJointType::Prismatic) {
        int sides = jt.type == mochi::ArticulatedJointType::Prismatic ? 3 : 16;
        FVector ax = L2R.TransformVector(Conv.DirectionToUnreal(jt.axis)).GetSafeNormal();
        FTransform t = L2R;
        t.SetRotation(FQuat::FindBetweenNormals(FVector(0, 0, 1), ax));
        DrawCylinder(
            PDI,
            loc,
            t.TransformVector({1, 0, 0}),
            t.TransformVector({0, 1, 0}),
            t.TransformVector({0, 0, 1}),
            0.4f,
            1.5f,
            sides,
            JointMaterialRenderProxy,
            SDPG_Foreground);
      } else if (jt.type == mochi::ArticulatedJointType::Hard) {
        PDI->DrawPoint(loc, kPurpleColor, 4, SDPG_Foreground);
      }
    }
  }
  if (bShowBounds) {
    FBoxSphereBounds bounds;
    if (ComputeVisualBoundsFromJointPositions(GetDefaultJointPositions(), bounds)) {
      DrawWireBox(PDI, bounds.GetBox(), kPurpleColor, SDPG_World);
    }
  }
}

void USuperDexBot::DrawCanvasCustom(
    FViewport& InViewport,
    FSceneView& View,
    FCanvas& Canvas,
    float DPIScale,
    bool bShowLinkTransforms,
    bool bShowJoints) const {
  TMochiToUnrealConverter<double> Conv(kSuperDexSpace, kUnrealCentimetersSpace);
  auto const& P = GetBotPrefab();
  mochi::DynamicArray<mochi::real> Pose(P.defaultPose.begin(), P.defaultPose.end());
  mochi::DynamicArray<mochi::TransformRT> RT;
  mochi::ErrorLog Err;
  superdex::robotics::ComputeLinkTransformsFromPose(
      P, Pose, RT, superdex::robotics::LinkTransformSpace::RootFromParent, Err);
  if (!Err.IsOK()) {
    return;
  }
  // Stack multiple labels
  // Stack multiple labels that land on the same screen pixel, mirroring
  // FMochiArticulationEditorViewportClient::DrawCanvasArticulation's local
  // StackedLabelDrawer helper (kept file-local there in MochiPhysicsEditor, so this small
  // amount of logic is duplicated here rather than shared across the Runtime/Editor and
  // MochiPhysics/SuperDexRobotics module boundaries).
  TMap<TPair<int32, int32>, TArray<FString>> LabelsByScreenPos;
  auto AddLabel = [&](const FString& Text, const FVector& WorldLocation) {
    if (!View.ViewFrustum.IntersectPoint(WorldLocation)) {
      return;
    }
    FVector2D ScreenLoc;
    if (!View.WorldToPixel(WorldLocation, ScreenLoc)) {
      return;
    }
    ScreenLoc /= DPIScale;
    ScreenLoc = ScreenLoc.RoundToVector();
    LabelsByScreenPos.FindOrAdd(TPair<int32, int32>((int32)ScreenLoc.X, (int32)ScreenLoc.Y))
        .AddUnique(Text);
  };
  for (int32 i = 0; i < (int32)P.links.size(); ++i) {
    FTransform L2R = Conv.TransformToUnreal(RT[i]);
    FVector loc = L2R.GetLocation();
    if (bShowLinkTransforms) {
      AddLabel(UTF8_TO_TCHAR(std::string(P.links[i].name).c_str()), loc);
    }
    if (bShowJoints && i > 0) {
      AddLabel(UTF8_TO_TCHAR(std::string(P.joints[i].name).c_str()), loc);
    }
  }
  for (auto& Pair : LabelsByScreenPos) {
    FString Text;
    for (int32 i = 0; i < Pair.Value.Num(); ++i) {
      Text += Pair.Value[i] + (i == Pair.Value.Num() - 1 ? TEXT("") : TEXT("\n"));
    }
    FCanvasTextItem TextItem(
        FVector2D(Pair.Key.Key, Pair.Key.Value),
        FText::FromString(Text),
        GEngine->GetSmallFont(),
        FLinearColor::White);
    TextItem.EnableShadow(FLinearColor::Black);
    Canvas.DrawItem(TextItem);
  }
}
