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

#include "AssetTypeActions_Base.h"
#include "CoreMinimal.h"
#include "Factories/Factory.h"
#include "MochiArticulation.h"
#include "MochiArticulationEditor.h"
#include "MochiAssets.h"
#include "MochiModel.h"
#include "Styling/SlateStyle.h"
#include "Styling/SlateStyleRegistry.h"
#include "SuperDexPrefab.h"
#include "Templates/SharedPointer.h"
#include "UObject/Object.h"
#include "UObject/ObjectMacros.h"

#include "MochiAssetsEditor.generated.h"

//------------------------------------------------------------------------------
// GENERAL
//------------------------------------------------------------------------------

// Content Browser type color of every SuperDex asset type: Meta blue (#0064E0). The asset icons in
// each plugin's Resources/ use the same blue as their accent.
static constexpr FColor kSuperdexAssetColor = FColor(0, 100, 224, 255);

template <typename TMochiAsset>
class FMochiAssetActions : public FAssetTypeActions_Base {
 public:
  FMochiAssetActions(EAssetTypeCategories::Type InAssetCategory) : AssetCategory(InAssetCategory) {}
  virtual uint32 GetCategories() override {
    return AssetCategory;
  }
  virtual FText GetName() const override {
    return TMochiAsset::StaticClass()->GetDisplayNameText();
  }
  virtual UClass* GetSupportedClass() const override {
    return TMochiAsset::StaticClass();
  }
  virtual FColor GetTypeColor() const override {
    return kSuperdexAssetColor;
  }

 private:
  const EAssetTypeCategories::Type AssetCategory;
};

// A Slate style registering a 16px "ClassIcon.<ClassName>" and a 64px "ClassThumbnail.<ClassName>"
// brush for each class name, loaded from <Plugin>/Resources/<ClassName>_16.svg and _64.svg.
// FSlateIconFinder searches every registered style, so these apply wherever Unreal shows a class
// icon (Content Browser, asset editors, pickers).
class SUPERDEXPHYSICSEDITOR_API FSuperDexClassIconStyle : public FSlateStyleSet {
 public:
  FSuperDexClassIconStyle(
      const FName& StyleSetName,
      const TCHAR* PluginName,
      TConstArrayView<const TCHAR*> ClassNames);
  virtual ~FSuperDexClassIconStyle() override;
};

class FMochiPhysicsEditorStyle final : public FSuperDexClassIconStyle {
 public:
  FMochiPhysicsEditorStyle();
};

//------------------------------------------------------------------------------
// MOCHI MODEL
//------------------------------------------------------------------------------

UCLASS(HideCategories = Object)
class SUPERDEXPHYSICSEDITOR_API UMochiModelFactoryNew : public UFactory {
  GENERATED_UCLASS_BODY()
 public:
  virtual UObject* FactoryCreateNew(
      UClass* InClass,
      UObject* InParent,
      FName InName,
      EObjectFlags Flags,
      UObject* Context,
      FFeedbackContext* Warn) override;
};

class FMochiModelAssetActions : public FMochiAssetActions<UMochiModel> {
 public:
  using FMochiAssetActions::FMochiAssetActions;

  virtual void OpenAssetEditor(
      const TArray<UObject*>& InObjects,
      TSharedPtr<class IToolkitHost> EditWithinLevelEditor = TSharedPtr<IToolkitHost>()) override;
};

//------------------------------------------------------------------------------
// MOCHI MATERIAL PARAMS
//------------------------------------------------------------------------------

UCLASS(HideCategories = Object)
class UMochiMaterialParamsFactoryNew : public UFactory {
  GENERATED_UCLASS_BODY()
 public:
  virtual UObject* FactoryCreateNew(
      UClass* InClass,
      UObject* InParent,
      FName InName,
      EObjectFlags Flags,
      UObject* Context,
      FFeedbackContext* Warn) override;
};

typedef FMochiAssetActions<UMochiMaterialParams> FMochiMaterialParamsAssetActions;

//------------------------------------------------------------------------------
// MOCHI SHELL MATERIAL PARAMS
//------------------------------------------------------------------------------

UCLASS(HideCategories = Object)
class UMochiShellMaterialParamsFactoryNew : public UFactory {
  GENERATED_UCLASS_BODY()
 public:
  virtual UObject* FactoryCreateNew(
      UClass* InClass,
      UObject* InParent,
      FName InName,
      EObjectFlags Flags,
      UObject* Context,
      FFeedbackContext* Warn) override;
};

typedef FMochiAssetActions<UMochiShellMaterialParams> FMochiShellMaterialParamsAssetActions;

//------------------------------------------------------------------------------
// MOCHI ROD MATERIAL PARAMS
//------------------------------------------------------------------------------

UCLASS(HideCategories = Object)
class UMochiRodMaterialParamsFactoryNew : public UFactory {
  GENERATED_UCLASS_BODY()
 public:
  virtual UObject* FactoryCreateNew(
      UClass* InClass,
      UObject* InParent,
      FName InName,
      EObjectFlags Flags,
      UObject* Context,
      FFeedbackContext* Warn) override;
};

typedef FMochiAssetActions<UMochiRodMaterialParams> FMochiRodMaterialParamsAssetActions;

//------------------------------------------------------------------------------
// MOCHI CONTACT PARAMS
//------------------------------------------------------------------------------

UCLASS(HideCategories = Object)
class UMochiContactParamsFactoryNew : public UFactory {
  GENERATED_UCLASS_BODY()
 public:
  virtual UObject* FactoryCreateNew(
      UClass* InClass,
      UObject* InParent,
      FName InName,
      EObjectFlags Flags,
      UObject* Context,
      FFeedbackContext* Warn) override;
};

typedef FMochiAssetActions<UMochiContactParams> FMochiContactParamsAssetActions;

//------------------------------------------------------------------------------
// MOCHI SOLVER PARAMS
//------------------------------------------------------------------------------

UCLASS(HideCategories = Object)
class UMochiSolverParamsFactoryNew : public UFactory {
  GENERATED_UCLASS_BODY()
 public:
  virtual UObject* FactoryCreateNew(
      UClass* InClass,
      UObject* InParent,
      FName InName,
      EObjectFlags Flags,
      UObject* Context,
      FFeedbackContext* Warn) override;
};

typedef FMochiAssetActions<UMochiSolverParams> FMochiSolverParamsAssetActions;

//------------------------------------------------------------------------------
// MOCHI CONTACT MATRIX
//------------------------------------------------------------------------------

UCLASS(HideCategories = Object)
class UMochiContactMatrixFactoryNew : public UFactory {
  GENERATED_UCLASS_BODY()
 public:
  virtual UObject* FactoryCreateNew(
      UClass* InClass,
      UObject* InParent,
      FName InName,
      EObjectFlags Flags,
      UObject* Context,
      FFeedbackContext* Warn) override;
};

typedef FMochiAssetActions<UMochiContactMatrix> FMochiContactMatrixAssetActions;

//------------------------------------------------------------------------------
// MOCHI MODEL MATERIAL MODIFIER
//------------------------------------------------------------------------------

UCLASS(HideCategories = Object)
class UMochiModelMaterialModifiersFactoryNew : public UFactory {
  GENERATED_UCLASS_BODY()
 public:
  virtual UObject* FactoryCreateNew(
      UClass* InClass,
      UObject* InParent,
      FName InName,
      EObjectFlags Flags,
      UObject* Context,
      FFeedbackContext* Warn) override;
};

typedef FMochiAssetActions<UMochiModelMaterialModifiers> FMochiModelMaterialModifiersAssetActions;

//------------------------------------------------------------------------------
// MOCHI ARTICULATION
//------------------------------------------------------------------------------

class FMochiArticulationAssetActions : public FAssetTypeActions_Base {
 public:
  FMochiArticulationAssetActions(EAssetTypeCategories::Type InAssetCategory)
      : AssetCategory(InAssetCategory) {}
  virtual uint32 GetCategories() override {
    return AssetCategory;
  }
  virtual FText GetName() const override {
    return UMochiArticulation::StaticClass()->GetDisplayNameText();
  }
  virtual UClass* GetSupportedClass() const override {
    return UMochiArticulation::StaticClass();
  }
  virtual FColor GetTypeColor() const override {
    return kSuperdexAssetColor;
  }

  virtual void OpenAssetEditor(
      const TArray<UObject*>& InObjects,
      TSharedPtr<IToolkitHost> EditWithinLevelEditor = TSharedPtr<IToolkitHost>()) override {
    if (InObjects.Num() > 0) {
      if (UMochiArticulation* object = Cast<UMochiArticulation>(InObjects[0])) {
        FMochiArticulationEditorToolkit::CreateEditor(
            EToolkitMode::Standalone, EditWithinLevelEditor, object);
      }
    }
  }

 private:
  const EAssetTypeCategories::Type AssetCategory;
};

//------------------------------------------------------------------------------
// SUPERDEX PREFAB
//------------------------------------------------------------------------------

class FSuperDexPrefabAssetActions : public FMochiAssetActions<USuperDexPrefab> {
 public:
  using FMochiAssetActions::FMochiAssetActions;

  virtual bool IsImportedAsset() const override {
    return true;
  }
};
