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

/**
 * Content Browser type color of every SuperDex asset type: Meta blue (#0064E0). The asset icons in
 * each plugin's Resources/ use the same blue as their accent.
 */
static constexpr FColor SuperdexAssetColor = FColor(0, 100, 224, 255);

/** Shared editor actions for physics asset types. */
template <typename TMochiAsset>
class FMochiAssetActions : public FAssetTypeActions_Base {
 public:
  /**
   * Initialize physics-asset actions for the specified editor category.
   *
   * @param InAssetCategory Editor category assigned to the asset actions.
   */
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
    return SuperdexAssetColor;
  }

 private:
  const EAssetTypeCategories::Type AssetCategory;
};

/**
 * A Slate style registering a 16px "ClassIcon.<ClassName>" and a 64px "ClassThumbnail.<ClassName>"
 * brush for each class name, loaded from <Plugin>/Resources/<ClassName>_16.svg and _64.svg.
 * FSlateIconFinder searches every registered style, so these apply wherever Unreal shows a class
 * icon (Content Browser, asset editors, pickers).
 */
class SUPERDEXPHYSICSEDITOR_API FSuperDexClassIconStyle : public FSlateStyleSet {
 public:
  /**
   * Register class icons from the plugin's editor resources.
   *
   * @param StyleSetName Name of the registered class-icon style.
   * @param PluginName Plugin supplying class-icon resources.
   * @param ClassNames Classes whose icons are registered.
   */
  FSuperDexClassIconStyle(
      const FName& StyleSetName,
      const TCHAR* PluginName,
      TConstArrayView<const TCHAR*> ClassNames);
  virtual ~FSuperDexClassIconStyle() override;
};

/** Class-icon style registered by the physics editor module. */
class FMochiPhysicsEditorStyle final : public FSuperDexClassIconStyle {
 public:
  /** Initialize the physics editor's class-icon style. */
  FMochiPhysicsEditorStyle();
};

//------------------------------------------------------------------------------
// MOCHI MODEL
//------------------------------------------------------------------------------

/** Factory creating model assets in the Content Browser. */
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

/** Content Browser actions opening the model asset editor. */
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

/** Factory creating soft-material parameter assets. */
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

/** Factory creating shell-material parameter assets. */
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

/** Factory creating rod-material parameter assets. */
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

/** Factory creating contact-parameter assets. */
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

/** Factory creating solver-parameter assets. */
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

/** Factory creating layer-contact matrix assets. */
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

/** Factory creating element material-modifier assets. */
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

/** Content Browser actions opening the articulation asset editor. */
class FMochiArticulationAssetActions : public FAssetTypeActions_Base {
 public:
  /**
   * Initialize articulation-asset actions for the specified editor category.
   *
   * @param InAssetCategory Editor category assigned to the asset actions.
   */
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
    return SuperdexAssetColor;
  }

  virtual void OpenAssetEditor(
      const TArray<UObject*>& InObjects,
      TSharedPtr<IToolkitHost> EditWithinLevelEditor = TSharedPtr<IToolkitHost>()) override {
    if (InObjects.Num() > 0) {
      if (UMochiArticulation* Object = Cast<UMochiArticulation>(InObjects[0])) {
        FMochiArticulationEditorToolkit::CreateEditor(
            EToolkitMode::Standalone, EditWithinLevelEditor, Object);
      }
    }
  }

 private:
  const EAssetTypeCategories::Type AssetCategory;
};

//------------------------------------------------------------------------------
// SUPERDEX PREFAB
//------------------------------------------------------------------------------

/** Content Browser actions for imported prefab assets. */
class FSuperDexPrefabAssetActions : public FMochiAssetActions<USuperDexPrefab> {
 public:
  using FMochiAssetActions::FMochiAssetActions;

  virtual bool IsImportedAsset() const override {
    return true;
  }
};
