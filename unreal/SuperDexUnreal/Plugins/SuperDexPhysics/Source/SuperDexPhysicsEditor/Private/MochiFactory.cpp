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

#include "MochiFactory.h"

#include "MochiActors.h"
#include "MochiModel.h"
#include "MochiPhysicsEditorModule.h"

#define LOCTEXT_NAMESPACE "MochiModelEditor"

//------------------------------------------------------------------------------
// MOCHI H5
//------------------------------------------------------------------------------

UMochiH5Factory::UMochiH5Factory(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer) {
  Formats.Add(FString(TEXT("h5;Hierarchical Data Format 5")));
  Formats.Add(FString(TEXT("hdf5;Hierarchical Data Format 5")));
  SupportedClass = UMochiModel::StaticClass();
  bCreateNew = false;
  bEditorImport = true;
}

UObject* UMochiH5Factory::FactoryCreateFile(
    UClass* InClass,
    UObject* InParent,
    FName InName,
    EObjectFlags Flags,
    const FString& Filename,
    const TCHAR* Parms,
    FFeedbackContext* Warn,
    bool& bOutOperationCanceled) {
  bOutOperationCanceled = false;
  UMochiModel* model = NewObject<UMochiModel>(InParent, UMochiModel::StaticClass(), InName, Flags);
  if (Filename.EndsWith(TEXT(".mochi.h5")) || Filename.EndsWith(TEXT(".mochi.json")) ||
      Filename.EndsWith(TEXT(".mochi.hdf5"))) {
    // Mochi assets were created for a right-handed coordinate system with:
    //   +X = right
    //   +Y = up
    //   -Z = forward
    //
    // Unreal uses a left-handed coordinate system with:
    //   +Y = right
    //   +Z = up
    //   +X = forward
    //
    // Set a default import transform to accomplish this conversion.
    // In the editor, you will see euler angles of (-90.0, 0.0, 90.0) degrees.
    //
    model->SetImportTransform(
        FTransform{
            FRotator{UE::Math::TQuat<double>(0.5, 0.5, 0.5, 0.5)}, // rotation
            FVector{0.0, 0.0, 0.0}, // translation
            FVector{1.0, 1.0, -1.0} // scale
        });
  }
  if (model->ImportFromFilename(Filename)) {
    return model;
  }

  // ImportFromFilename rejects anything that is not a single Mochi shape/model (e.g. an articulated
  // shape, or a corrupt/unsupported file). The underlying Mochi error is already logged. Surface a
  // readable hint here so editor users understand why the import produced no asset.
  UE_LOG(
      LogMochiPhysicsEditor,
      Warning,
      TEXT(
          "[%hs] Failed to import [%s]; the file is not a supported Mochi model. It may be an articulated shape or otherwise unsupported. See the log above for details."),
      __FUNCTION__,
      *Filename);
  return nullptr;
}

UReimportMochiH5Factory::UReimportMochiH5Factory(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer) {
  // Reimport only: never chosen over UMochiH5Factory for a new import.
  ImportPriority = DefaultImportPriority - 1;
}

bool UReimportMochiH5Factory::CanReimport(UObject* Obj, TArray<FString>& OutFilenames) {
  UMochiModel* model = Cast<UMochiModel>(Obj);
  if (!model || !model->AssetImportData) {
    return false;
  }
  model->AssetImportData->ExtractFilenames(OutFilenames);
  return true;
}

void UReimportMochiH5Factory::SetReimportPaths(
    UObject* Obj,
    const TArray<FString>& NewReimportPaths) {
  UMochiModel* model = Cast<UMochiModel>(Obj);
  if (model && model->AssetImportData && NewReimportPaths.Num() > 0) {
    model->AssetImportData->UpdateFilenameOnly(NewReimportPaths[0]);
  }
}

EReimportResult::Type UReimportMochiH5Factory::Reimport(UObject* Obj) {
  UMochiModel* model = Cast<UMochiModel>(Obj);
  if (!model) {
    return EReimportResult::Failed;
  }
  const FString sourcePath = model->GetSourceFilePath();
  if (sourcePath.IsEmpty() || !FPaths::FileExists(sourcePath)) {
    UE_LOG(
        LogMochiPhysicsEditor,
        Error,
        TEXT("[%hs] Source file not found for %s: [%s]"),
        __FUNCTION__,
        *model->GetName(),
        *sourcePath);
    return EReimportResult::Failed;
  }
  if (!model->ImportFromFilename(sourcePath)) {
    return EReimportResult::Failed;
  }
  model->MarkPackageDirty();
  return EReimportResult::Succeeded;
}

int32 UReimportMochiH5Factory::GetPriority() const {
  return ImportPriority;
}

//------------------------------------------------------------------------------
// MOCHI SOFT MESH ACTOR FACTORY
//------------------------------------------------------------------------------

UMochiSoftMeshActorFactory::UMochiSoftMeshActorFactory(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer) {
  DisplayName = LOCTEXT("MochiModelDisplayName", "Mochi Model");
  NewActorClass = AMochiSoftMeshActor::StaticClass();
  bUseSurfaceOrientation = true;
}

bool UMochiSoftMeshActorFactory::CanCreateActorFrom(
    const FAssetData& AssetData,
    FText& OutErrorMsg) {
  if (!AssetData.IsValid() || !AssetData.IsInstanceOf(UMochiModel::StaticClass())) {
    OutErrorMsg =
        NSLOCTEXT("CanCreateActor", "NoMochiModel", "A valid Mochi model must be specified.");
    return false;
  }
  return true;
}

void UMochiSoftMeshActorFactory::PostSpawnActor(UObject* Asset, AActor* NewActor) {
  Super::PostSpawnActor(Asset, NewActor);

  UMochiModel* mochiModel = CastChecked<UMochiModel>(Asset);
  AMochiSoftMeshActor* mochiSoftMeshActor = CastChecked<AMochiSoftMeshActor>(NewActor);
  UMochiSoftMeshComponent* mochiSoftMeshComponent = mochiSoftMeshActor->GetMochiSoftMeshComponent();
  check(mochiSoftMeshComponent);
  mochiSoftMeshComponent->UnregisterComponent();
  mochiSoftMeshComponent->SetMochiModel(mochiModel);
  mochiSoftMeshComponent->RegisterComponent();
}

UObject* UMochiSoftMeshActorFactory::GetAssetFromActorInstance(AActor* Instance) {
  check(Instance->IsA(NewActorClass));
  AMochiSoftMeshActor* mochiSoftMeshActor = CastChecked<AMochiSoftMeshActor>(Instance);

  check(mochiSoftMeshActor->GetMochiSoftMeshComponent());
  return mochiSoftMeshActor->GetMochiSoftMeshComponent()->GetMochiModel();
}

#if UE_VERSION_OLDER_THAN(5, 4, 0)
void UMochiSoftMeshActorFactory::PostCreateBlueprint(UObject* Asset, AActor* CDO) {
  if (Asset != nullptr && CDO != nullptr) {
    UMochiModel* mochiModel = CastChecked<UMochiModel>(Asset);
    AMochiSoftMeshActor* mochiSoftMeshActor = CastChecked<AMochiSoftMeshActor>(CDO);
    UMochiSoftMeshComponent* mochiSoftMeshComponent =
        mochiSoftMeshActor->GetMochiSoftMeshComponent();

    mochiSoftMeshComponent->SetMochiModel(mochiModel);
  }
}
#endif

FQuat UMochiSoftMeshActorFactory::AlignObjectToSurfaceNormal(
    const FVector& InSurfaceNormal,
    const FQuat& ActorRotation) const {
  return FindActorAlignmentRotation(ActorRotation, FVector(0.f, 0.f, 1.f), InSurfaceNormal);
}

//------------------------------------------------------------------------------
// MOCHI ARTICULATION ACTOR FACTORY
//------------------------------------------------------------------------------

UMochiArticulationActorFactory::UMochiArticulationActorFactory(
    const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer) {
  DisplayName = LOCTEXT("MochiArticulationActorDisplayName", "Mochi Articulation Actor");
  NewActorClass = AMochiArticulationActor::StaticClass();
  bUseSurfaceOrientation = true;
}

bool UMochiArticulationActorFactory::CanCreateActorFrom(
    const FAssetData& AssetData,
    FText& OutErrorMsg) {
  if (!AssetData.IsValid() || !AssetData.IsInstanceOf(UMochiArticulation::StaticClass())) {
    OutErrorMsg = NSLOCTEXT(
        "CanCreateActor", "NoMochiArticulation", "A valid MochiArticulation must be specified.");
    return false;
  }
  return true;
}

void UMochiArticulationActorFactory::PostSpawnActor(UObject* Asset, AActor* NewActor) {
  Super::PostSpawnActor(Asset, NewActor);
  UMochiArticulation* articulation = CastChecked<UMochiArticulation>(Asset);
  AMochiArticulationActor* actor = CastChecked<AMochiArticulationActor>(NewActor);
  actor->SetArticulationAsset(articulation);
}

UObject* UMochiArticulationActorFactory::GetAssetFromActorInstance(AActor* Instance) {
  check(Instance->IsA(NewActorClass));
  AMochiArticulationActor* actor = CastChecked<AMochiArticulationActor>(Instance);
  return actor->GetArticulationAsset();
}

FQuat UMochiArticulationActorFactory::AlignObjectToSurfaceNormal(
    const FVector& InSurfaceNormal,
    const FQuat& ActorRotation) const {
  return FindActorAlignmentRotation(ActorRotation, FVector(0.f, 0.f, 1.f), InSurfaceNormal);
}

#undef LOCTEXT_NAMESPACE
