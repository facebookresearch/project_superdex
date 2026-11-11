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

#include "MochiAssetsEditor.h"
#include "MochiModel.h"
#include "MochiModelEditorToolkit.h"
#include "MochiPhysicsEditorModule.h"

#include "Interfaces/IPluginManager.h"

FSuperDexClassIconStyle::FSuperDexClassIconStyle(
    const FName& StyleSetName,
    const TCHAR* PluginName,
    TConstArrayView<const TCHAR*> ClassNames)
    : FSlateStyleSet(StyleSetName) {
  const FVector2D Icon16x16(16.f, 16.f);
  const FVector2D Icon64x64(64.f, 64.f);
  const FString baseDir = IPluginManager::Get().FindPlugin(PluginName)->GetBaseDir();
  SetContentRoot(FPaths::Combine(baseDir, TEXT("Resources")));

  for (const TCHAR* className : ClassNames) {
    Set(*FString::Printf(TEXT("ClassIcon.%s"), className),
        new FSlateVectorImageBrush(
            RootToContentDir(FString::Printf(TEXT("%s_16.svg"), className)), Icon16x16));
    Set(*FString::Printf(TEXT("ClassThumbnail.%s"), className),
        new FSlateVectorImageBrush(
            RootToContentDir(FString::Printf(TEXT("%s_64.svg"), className)), Icon64x64));
  }

  FSlateStyleRegistry::RegisterSlateStyle(*this);
}

FSuperDexClassIconStyle::~FSuperDexClassIconStyle() {
  FSlateStyleRegistry::UnRegisterSlateStyle(*this);
}

// Class names without their U prefix.
static const TCHAR* const kMochiPhysicsClassIcons[] = {
    TEXT("MochiModel"),
    TEXT("MochiMaterialParams"),
    TEXT("MochiShellMaterialParams"),
    TEXT("MochiRodMaterialParams"),
    TEXT("MochiContactParams"),
    TEXT("MochiContactMatrix"),
    TEXT("MochiSolverParams"),
    TEXT("MochiModelMaterialModifiers"),
    TEXT("MochiArticulation"),
    TEXT("SuperDexPrefab"),
};

FMochiPhysicsEditorStyle::FMochiPhysicsEditorStyle()
    : FSuperDexClassIconStyle(
          TEXT("MochiPhysicsEditorStyle"),
          TEXT("SuperDexPhysics"),
          kMochiPhysicsClassIcons) {}

//------------------------------------------------------------------------------
// MOCHI MODEL
//------------------------------------------------------------------------------

UMochiModelFactoryNew::UMochiModelFactoryNew(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer) {
  SupportedClass = UMochiModel::StaticClass();
  bCreateNew = true;
  bEditAfterNew = true;
}

UObject* UMochiModelFactoryNew::FactoryCreateNew(
    UClass* InClass,
    UObject* InParent,
    FName InName,
    EObjectFlags Flags,
    UObject* Context,
    FFeedbackContext* Warn) {
  return NewObject<UMochiModel>(InParent, InClass, InName, Flags);
}

void FMochiModelAssetActions::OpenAssetEditor(
    const TArray<UObject*>& InObjects,
    TSharedPtr<class IToolkitHost> EditWithinLevelEditor) {
  UE_LOG(LogMochiPhysicsEditor, Display, TEXT("[FMochiModelAssetActions] OpenAssetEditor()"));
  if (InObjects.Num() > 0) {
    if (UMochiModel* object = Cast<UMochiModel>(InObjects[0])) {
      FMochiModelEditorToolkit::CreateEditor(
          EToolkitMode::Standalone, EditWithinLevelEditor, object);
    }
  }
}

//------------------------------------------------------------------------------
// MOCHI MATERIAL PARAMS
//------------------------------------------------------------------------------

UMochiMaterialParamsFactoryNew::UMochiMaterialParamsFactoryNew(
    const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer) {
  SupportedClass = UMochiMaterialParams::StaticClass();
  bCreateNew = true;
  bEditAfterNew = true;
}

UObject* UMochiMaterialParamsFactoryNew::FactoryCreateNew(
    UClass* InClass,
    UObject* InParent,
    FName InName,
    EObjectFlags Flags,
    UObject* Context,
    FFeedbackContext* Warn) {
  return NewObject<UMochiMaterialParams>(InParent, InClass, InName, Flags);
}

//------------------------------------------------------------------------------
// MOCHI SHELL MATERIAL PARAMS
//------------------------------------------------------------------------------

UMochiShellMaterialParamsFactoryNew::UMochiShellMaterialParamsFactoryNew(
    const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer) {
  SupportedClass = UMochiShellMaterialParams::StaticClass();
  bCreateNew = true;
  bEditAfterNew = true;
}

UObject* UMochiShellMaterialParamsFactoryNew::FactoryCreateNew(
    UClass* InClass,
    UObject* InParent,
    FName InName,
    EObjectFlags Flags,
    UObject* Context,
    FFeedbackContext* Warn) {
  return NewObject<UMochiShellMaterialParams>(InParent, InClass, InName, Flags);
}

//------------------------------------------------------------------------------
// MOCHI ROD MATERIAL PARAMS
//------------------------------------------------------------------------------

UMochiRodMaterialParamsFactoryNew::UMochiRodMaterialParamsFactoryNew(
    const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer) {
  SupportedClass = UMochiRodMaterialParams::StaticClass();
  bCreateNew = true;
  bEditAfterNew = true;
}

UObject* UMochiRodMaterialParamsFactoryNew::FactoryCreateNew(
    UClass* InClass,
    UObject* InParent,
    FName InName,
    EObjectFlags Flags,
    UObject* Context,
    FFeedbackContext* Warn) {
  return NewObject<UMochiRodMaterialParams>(InParent, InClass, InName, Flags);
}

//------------------------------------------------------------------------------
// MOCHI CONTACT PARAMS
//------------------------------------------------------------------------------

UMochiContactParamsFactoryNew::UMochiContactParamsFactoryNew(
    const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer) {
  SupportedClass = UMochiContactParams::StaticClass();
  bCreateNew = true;
  bEditAfterNew = true;
}

UObject* UMochiContactParamsFactoryNew::FactoryCreateNew(
    UClass* InClass,
    UObject* InParent,
    FName InName,
    EObjectFlags Flags,
    UObject* Context,
    FFeedbackContext* Warn) {
  return NewObject<UMochiContactParams>(InParent, InClass, InName, Flags);
}

//------------------------------------------------------------------------------
// MOCHI SOLVER PARAMS
//------------------------------------------------------------------------------

UMochiSolverParamsFactoryNew::UMochiSolverParamsFactoryNew(
    const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer) {
  SupportedClass = UMochiSolverParams::StaticClass();
  bCreateNew = true;
  bEditAfterNew = true;
}

UObject* UMochiSolverParamsFactoryNew::FactoryCreateNew(
    UClass* InClass,
    UObject* InParent,
    FName InName,
    EObjectFlags Flags,
    UObject* Context,
    FFeedbackContext* Warn) {
  return NewObject<UMochiSolverParams>(InParent, InClass, InName, Flags);
}

//------------------------------------------------------------------------------

// MOCHI CONTACT MATRIX
//------------------------------------------------------------------------------

UMochiContactMatrixFactoryNew::UMochiContactMatrixFactoryNew(
    const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer) {
  SupportedClass = UMochiContactMatrix::StaticClass();
  bCreateNew = true;
  bEditAfterNew = true;
}

UObject* UMochiContactMatrixFactoryNew::FactoryCreateNew(
    UClass* InClass,
    UObject* InParent,
    FName InName,
    EObjectFlags Flags,
    UObject* Context,
    FFeedbackContext* Warn) {
  return NewObject<UMochiContactMatrix>(InParent, InClass, InName, Flags);
}

//------------------------------------------------------------------------------
// MOCHI MODEL MATERIAL MODIFIER
//------------------------------------------------------------------------------

UMochiModelMaterialModifiersFactoryNew::UMochiModelMaterialModifiersFactoryNew(
    const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer) {
  SupportedClass = UMochiModelMaterialModifiers::StaticClass();
  bCreateNew = true;
  // this asset is not meant to be edited directly
  bEditAfterNew = false;
}

UObject* UMochiModelMaterialModifiersFactoryNew::FactoryCreateNew(
    UClass* InClass,
    UObject* InParent,
    FName InName,
    EObjectFlags Flags,
    UObject* Context,
    FFeedbackContext* Warn) {
  return NewObject<UMochiModelMaterialModifiers>(InParent, InClass, InName, Flags);
}
