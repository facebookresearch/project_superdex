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

#include "SuperDexSplitBlueprintChildren.h"

#include "AssetRegistry/AssetRegistryModule.h"
#include "Components/ChildActorComponent.h"
#include "Components/SceneComponent.h"
#include "Dialog/SCustomDialog.h"
#include "Editor.h"
#include "EditorAssetLibrary.h"
#include "Engine/Blueprint.h"
#include "Engine/BlueprintGeneratedClass.h"
#include "Engine/Engine.h"
#include "Engine/SCS_Node.h"
#include "Engine/SimpleConstructionScript.h"
#include "HAL/PlatformFileManager.h"
#include "Kismet2/BlueprintEditorUtils.h"
#include "Kismet2/KismetEditorUtilities.h"
#include "Misc/FileHelper.h"
#include "Misc/MessageDialog.h"
#include "Misc/PackageName.h"
#include "Misc/Paths.h"
#include "MochiSpaceConverter.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "UObject/Package.h"

DEFINE_LOG_CATEGORY_STATIC(LogMochiSplit, Log, All);

#define LOCTEXT_NAMESPACE "SuperDexSplitBlueprintChildren"

namespace SuperDexSplitBlueprintChildren {
namespace {

// Any component class shipped from the MochiPhysics or SuperDexRobotics UE modules counts as
// "SuperDex-bearing." This avoids hard-coding the (growing) list of concrete component
// classes and keeps third-party Mochi subclasses in.
//
// NOTE: prior to the MochiPhysics/SuperDexRobotics plugin split, this only checked the
// "/Script/MochiPhysics" prefix, since every Mochi UCLASS lived in that one module. Bot
// components (e.g. USuperDexBotComponent) now live in "/Script/SuperDexRobotics" instead, so both
// prefixes must be checked or bot-bearing Blueprints would silently stop being detected
// as splittable.
bool IsMochiComponentClass(UClass const* Class) {
  if (Class == nullptr) {
    return false;
  }
  UPackage const* Pkg = Class->GetOutermost();
  if (Pkg == nullptr) {
    return false;
  }
  FString const& PkgName = Pkg->GetName();
  return PkgName.StartsWith(TEXT("/Script/SuperDexPhysics")) ||
      PkgName.StartsWith(TEXT("/Script/SuperDexRobotics"));
}

bool IsChildActorClass(UClass const* Class) {
  return Class != nullptr && Class->IsChildOf(UChildActorComponent::StaticClass());
}

// A "splittable" node is one that becomes its own spawn entry: either a
// SuperDex-bearing component (gets a new BP) or a ChildActorComponent (just
// references its existing ChildActorClass).
bool IsSplittableNode(USCS_Node const* Node) {
  if (Node == nullptr) {
    return false;
  }
  return IsMochiComponentClass(Node->ComponentClass) || IsChildActorClass(Node->ComponentClass);
}

bool TreeHasSplittable(USCS_Node const* Node) {
  if (Node == nullptr) {
    return false;
  }
  if (IsSplittableNode(Node)) {
    return true;
  }
  for (USCS_Node const* Child : Node->GetChildNodes()) {
    if (TreeHasSplittable(Child)) {
      return true;
    }
  }
  return false;
}

// Strip the trailing "_C" from a generated UClass name so it matches the
// underlying UBlueprint asset name (e.g. "BP_Foo_C" -> "BP_Foo").
FString PrefabNameForChildActorClass(UClass const* ChildActorClass) {
  if (ChildActorClass == nullptr) {
    return FString();
  }
  if (UBlueprint const* BP = Cast<UBlueprint>(ChildActorClass->ClassGeneratedBy)) {
    return BP->GetName();
  }
  FString Name = ChildActorClass->GetName();
  if (Name.EndsWith(TEXT("_C"))) {
    Name.LeftChopInline(2);
  }
  return Name;
}

void CopyComponentProperties(UActorComponent const* Src, UActorComponent* Dst) {
  if (Src == nullptr || Dst == nullptr) {
    return;
  }
  UEngine::FCopyPropertiesForUnrelatedObjectsParams Params;
  Params.bNotifyObjectReplacement = false;
  UEngine::CopyPropertiesForUnrelatedObjects(const_cast<UActorComponent*>(Src), Dst, Params);
}

// Recursively copy the cosmetic (non-splittable) descendants of SrcParent into
// DstParent within DstSCS. A splittable descendant terminates the walk on its
// branch - it will live in its own BP and be re-attached via the task file's
// parent field.
void CloneCosmeticDescendantsInto(
    USCS_Node const* SrcParent,
    USCS_Node* DstParent,
    USimpleConstructionScript* DstSCS) {
  for (USCS_Node* SrcChild : SrcParent->GetChildNodes()) {
    if (SrcChild == nullptr) {
      continue;
    }
    if (IsSplittableNode(SrcChild)) {
      continue;
    }
    USCS_Node* DstChild = DstSCS->CreateNode(SrcChild->ComponentClass, SrcChild->GetVariableName());
    CopyComponentProperties(SrcChild->ComponentTemplate, DstChild->ComponentTemplate);
    DstParent->AddChildNode(DstChild);
    CloneCosmeticDescendantsInto(SrcChild, DstChild, DstSCS);
  }
}

// Create a new BP at TargetPkgName whose root SCS payload is a copy of SrcUnit
// (re-rooted with identity local transform) plus its non-splittable
// descendants. Returns the new (or pre-existing) Blueprint. Returns nullptr on
// failure.
UBlueprint* CreateChildBlueprintForUnit(USCS_Node const* SrcUnit, FString const& TargetPkgName) {
  if (FPackageName::DoesPackageExist(TargetPkgName)) {
    UE_LOG(
        LogMochiSplit,
        Display,
        TEXT("[%hs] Skipping existing asset: %s"),
        __FUNCTION__,
        *TargetPkgName);
    return Cast<UBlueprint>(UEditorAssetLibrary::LoadAsset(TargetPkgName));
  }

  UPackage* Pkg = CreatePackage(*TargetPkgName);
  if (Pkg == nullptr) {
    UE_LOG(
        LogMochiSplit,
        Error,
        TEXT("[%hs] CreatePackage failed for %s"),
        __FUNCTION__,
        *TargetPkgName);
    return nullptr;
  }

  FName const NewBPName(*FPackageName::GetLongPackageAssetName(TargetPkgName));
  UBlueprint* NewBP = FKismetEditorUtilities::CreateBlueprint(
      AActor::StaticClass(),
      Pkg,
      NewBPName,
      EBlueprintType::BPTYPE_Normal,
      UBlueprint::StaticClass(),
      UBlueprintGeneratedClass::StaticClass(),
      FName("SuperDexSplitBlueprintChildren"));
  if (NewBP == nullptr || NewBP->SimpleConstructionScript == nullptr) {
    UE_LOG(
        LogMochiSplit,
        Error,
        TEXT("[%hs] CreateBlueprint failed for %s"),
        __FUNCTION__,
        *TargetPkgName);
    return nullptr;
  }

  USimpleConstructionScript* NewSCS = NewBP->SimpleConstructionScript;

  // Copy the unit's own component as the payload root, with identity local
  // transform - the BP's pivot lives at the unit's local origin. The unit's
  // placement under its original parent is recorded separately in the task
  // file.
  USCS_Node* UnitCopy = NewSCS->CreateNode(SrcUnit->ComponentClass, SrcUnit->GetVariableName());
  CopyComponentProperties(SrcUnit->ComponentTemplate, UnitCopy->ComponentTemplate);
  if (USceneComponent* SC = Cast<USceneComponent>(UnitCopy->ComponentTemplate)) {
    SC->SetRelativeTransform(FTransform::Identity);
  }
  CloneCosmeticDescendantsInto(SrcUnit, UnitCopy, NewSCS);

  // Attach under whatever root the new BP already has (DefaultSceneRoot for a
  // freshly-created AActor BP). If there's no root for some reason, promote
  // UnitCopy to a root node.
  TArray<USCS_Node*> Roots = NewSCS->GetRootNodes();
  if (Roots.Num() > 0 && Roots[0] != nullptr) {
    Roots[0]->AddChildNode(UnitCopy);
  } else {
    NewSCS->AddNode(UnitCopy);
  }

  FBlueprintEditorUtils::PostEditChangeBlueprintActors(NewBP);
  FBlueprintEditorUtils::MarkBlueprintAsStructurallyModified(NewBP);

  FAssetRegistryModule::AssetCreated(NewBP);
  Pkg->MarkPackageDirty();
  UEditorAssetLibrary::SaveLoadedAsset(NewBP, /*bOnlyIfIsDirty=*/false);

  return NewBP;
}

// One entry in the task file's spawns[] array.
struct FSpawnRecord {
  FString Name;
  FString PrefabName;
  FString ParentName; // empty => "task_root"
  FTransform LocalTransform; // Unreal-cm, relative to ParentName's frame
};

struct FCounters {
  int32 NewBPsCreated = 0;
  int32 NewBPsReused = 0;
  int32 ChildActorsReferenced = 0;
  int32 Failed = 0;
};

// Recursively walks the parent BP's SCS, emitting a spawn for every splittable
// node and a new BP for every Mochi node. SplitAncestorName is the name of
// the closest splittable ancestor in the walk so far (empty = task_root).
// XfFromSplitAncestor is this node's parent's transform relative to that
// ancestor's frame; child world = parent world * child local in UE, so the
// node's own transform-from-ancestor is XfFromSplitAncestor * NodeLocal.
void WalkAndSplit(
    USCS_Node* Node,
    FString const& SplitAncestorName,
    FTransform const& XfFromSplitAncestor,
    FString const& ChildrenPkgDir,
    TArray<FSpawnRecord>& OutSpawns,
    FCounters& Counts) {
  if (Node == nullptr) {
    return;
  }

  USceneComponent const* SceneTpl = Cast<USceneComponent>(Node->ComponentTemplate);
  FTransform const NodeLocal =
      SceneTpl != nullptr ? SceneTpl->GetRelativeTransform() : FTransform::Identity;
  FTransform const NodeXfFromAncestor = XfFromSplitAncestor * NodeLocal;

  FString NextSplitAncestor = SplitAncestorName;
  FTransform NextXfFromAncestor = NodeXfFromAncestor;

  if (IsSplittableNode(Node)) {
    FString const NodeName = Node->GetVariableName().ToString();
    FSpawnRecord Rec;
    Rec.Name = NodeName;
    Rec.ParentName = SplitAncestorName;
    Rec.LocalTransform = NodeXfFromAncestor;

    if (IsChildActorClass(Node->ComponentClass)) {
      UChildActorComponent const* CAC = Cast<UChildActorComponent>(Node->ComponentTemplate);
      UClass const* CAClass = CAC != nullptr ? CAC->GetChildActorClass() : nullptr;
      Rec.PrefabName = PrefabNameForChildActorClass(CAClass);
      if (Rec.PrefabName.IsEmpty()) {
        UE_LOG(
            LogMochiSplit,
            Warning,
            TEXT("[%hs] ChildActorComponent '%s' has no ChildActorClass - skipped."),
            __FUNCTION__,
            *NodeName);
        ++Counts.Failed;
      } else {
        ++Counts.ChildActorsReferenced;
        OutSpawns.Add(Rec);
      }
    } else {
      // Mochi node - create its own BP.
      FString const NewBPPkgName = ChildrenPkgDir / NodeName;
      bool const bExisted = FPackageName::DoesPackageExist(NewBPPkgName);
      UBlueprint* NewBP = CreateChildBlueprintForUnit(Node, NewBPPkgName);
      if (NewBP != nullptr) {
        Rec.PrefabName = NodeName;
        if (bExisted) {
          ++Counts.NewBPsReused;
        } else {
          ++Counts.NewBPsCreated;
        }
        OutSpawns.Add(Rec);
      } else {
        UE_LOG(
            LogMochiSplit,
            Warning,
            TEXT("[%hs] Failed to create child BP for '%s' - skipped."),
            __FUNCTION__,
            *NodeName);
        ++Counts.Failed;
      }
    }

    // From here on, this node is the split ancestor for its descendants and
    // their transforms are expressed in this node's local frame.
    NextSplitAncestor = NodeName;
    NextXfFromAncestor = FTransform::Identity;
  }

  for (USCS_Node* Child : Node->GetChildNodes()) {
    WalkAndSplit(Child, NextSplitAncestor, NextXfFromAncestor, ChildrenPkgDir, OutSpawns, Counts);
  }
}

// Serialize spawns to a .superdex_task JSON. Translations are converted to
// meters in SuperDexRobotics space (+X, -Y, +Z), rotations are XYZW quaternions. The
// "parent" field is omitted when empty (superdex_task treats absent parent
// as "task_root").
FString BuildTaskJson(FString const& MetadataName, TArray<FSpawnRecord> const& Spawns) {
  FMochiSpaceConverter const Converter(kUnrealCentimetersSpace, kSuperDexSpace);

  FString Out;
  TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Out);
  Writer->WriteObjectStart();

  Writer->WriteObjectStart(TEXT("metadata"));
  Writer->WriteValue(TEXT("name"), MetadataName);
  Writer->WriteObjectEnd();

  Writer->WriteArrayStart(TEXT("spawns"));
  for (FSpawnRecord const& S : Spawns) {
    FTransform const Out_xf = Converter.TransformToOutput(S.LocalTransform);
    FVector const T = Out_xf.GetTranslation();
    FQuat const R = Out_xf.GetRotation();

    Writer->WriteObjectStart();
    Writer->WriteValue(TEXT("name"), S.Name);
    Writer->WriteValue(TEXT("prefabName"), S.PrefabName);
    if (!S.ParentName.IsEmpty()) {
      Writer->WriteValue(TEXT("parent"), S.ParentName);
    }

    Writer->WriteArrayStart(TEXT("translation"));
    Writer->WriteValue(T.X);
    Writer->WriteValue(T.Y);
    Writer->WriteValue(T.Z);
    Writer->WriteArrayEnd();

    Writer->WriteArrayStart(TEXT("rotation"));
    Writer->WriteValue(R.X);
    Writer->WriteValue(R.Y);
    Writer->WriteValue(R.Z);
    Writer->WriteValue(R.W);
    Writer->WriteArrayEnd();

    Writer->WriteObjectEnd();
  }
  Writer->WriteArrayEnd();

  Writer->WriteObjectEnd();
  Writer->Close();
  return Out;
}

} // namespace

bool IsBlueprintSplittable(UBlueprint const* ParentBP) {
  if (ParentBP == nullptr || ParentBP->SimpleConstructionScript == nullptr) {
    return false;
  }
  for (USCS_Node const* Root : ParentBP->SimpleConstructionScript->GetRootNodes()) {
    if (TreeHasSplittable(Root)) {
      return true;
    }
  }
  return false;
}

void ExecuteSplitBlueprintChildren(UBlueprint* ParentBP) {
  if (ParentBP == nullptr || ParentBP->SimpleConstructionScript == nullptr) {
    return;
  }

  FString const ParentPkgName = ParentBP->GetOutermost()->GetName(); // "/Game/Foo/BP_X"
  FString const ParentPkgDir = FPackageName::GetLongPackagePath(ParentPkgName); // "/Game/Foo"
  FString const ParentBPName = ParentBP->GetName(); // "BP_X"
  FString const ChildrenPkgDir =
      ParentPkgDir / (ParentBPName + TEXT("_Children")); // "/Game/Foo/BP_X_Children"

  TArray<FSpawnRecord> Spawns;
  FCounters Counts;
  for (USCS_Node* Root : ParentBP->SimpleConstructionScript->GetRootNodes()) {
    WalkAndSplit(
        Root,
        /*SplitAncestorName=*/FString(),
        /*XfFromSplitAncestor=*/FTransform::Identity,
        ChildrenPkgDir,
        Spawns,
        Counts);
  }

  // Write the .superdex_task. Each spawn's prefabName is the corresponding
  // asset's name; the user must ensure those names exist in the target
  // .superdex_scene's spawnablePrefabs[] - that binding is not visible from
  // the UE side.
  FString const TaskJson = BuildTaskJson(ParentBPName, Spawns);
  FString const ChildrenContentDir =
      FPackageName::LongPackageNameToFilename(ChildrenPkgDir / TEXT(""), /*Extension=*/FString());
  FString const TaskDiskPath =
      FPaths::Combine(ChildrenContentDir, ParentBPName + TEXT(".superdex_task"));

  IPlatformFile& Platform = FPlatformFileManager::Get().GetPlatformFile();
  Platform.CreateDirectoryTree(*FPaths::GetPath(TaskDiskPath));

  // UTF-8 without BOM: the C++ loader (LoadBotTaskPrefabFromFile, picojson + std::ifstream)
  // does not handle UTF-16 or a BOM; AutoDetect would emit UTF-16LE+BOM for any non-ASCII char.
  bool const bWrote = FFileHelper::SaveStringToFile(
      TaskJson, *TaskDiskPath, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);
  if (!bWrote) {
    UE_LOG(
        LogMochiSplit,
        Error,
        TEXT("[%hs] Failed to write task file: %s"),
        __FUNCTION__,
        *TaskDiskPath);
  }

  auto OnOpenLocation = [TaskDiskPath]() {
    FString Dir = FPaths::GetPath(TaskDiskPath);
    FPaths::NormalizeDirectoryName(Dir);
    FPaths::MakePlatformFilename(Dir);
    FPlatformProcess::ExploreFolder(*Dir);
  };

  FText const Body = FText::Format(
      LOCTEXT(
          "MochiSplitDoneBody",
          "Parent: {0}\n"
          "Total spawns in task: {1}\n"
          "  New child Blueprints created: {2}\n"
          "  Existing child Blueprints reused: {3}\n"
          "  ChildActorComponents referenced: {4}\n"
          "Failed: {5}\n\n"
          "Task file: {6}\n\n"
          "Note: each spawn's 'prefabName' is set to the child asset name. "
          "Make sure those names exist in the target .superdex_scene's "
          "spawnablePrefabs[]."),
      FText::FromString(ParentBPName),
      FText::AsNumber(Spawns.Num()),
      FText::AsNumber(Counts.NewBPsCreated),
      FText::AsNumber(Counts.NewBPsReused),
      FText::AsNumber(Counts.ChildActorsReferenced),
      FText::AsNumber(Counts.Failed),
      FText::FromString(TaskDiskPath));

  TSharedRef<SCustomDialog> Dlg =
      SNew(SCustomDialog)
          .Title(LOCTEXT("MochiSplitDoneTitle", "Split SuperDex Children - Done"))
          .Content()[SNew(STextBlock).Text(Body).AutoWrapText(true)]
          .Buttons(
              {SCustomDialog::FButton(
                   LOCTEXT("MochiSplitOpenLoc", "Open Location"),
                   FSimpleDelegate::CreateLambda(OnOpenLocation)),
               SCustomDialog::FButton(LOCTEXT("MochiSplitDone", "Done"))});
  Dlg->Show();
}

} // namespace SuperDexSplitBlueprintChildren

#undef LOCTEXT_NAMESPACE
