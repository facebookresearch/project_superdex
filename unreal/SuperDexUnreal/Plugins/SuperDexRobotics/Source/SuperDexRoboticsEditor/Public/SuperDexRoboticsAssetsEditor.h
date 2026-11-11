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
#include "MochiArticulationEditor.h"
#include "MochiAssetsEditor.h"
#include "Questionnaire/SuperDexStudyQuestionData.h"
#include "SuperDexBot.h"
#include "SuperDexBotIKConfig.h"
#include "SuperDexPoseTypes.h"
#include "SuperDexScene.h"
#include "Templates/SharedPointer.h"
#include "Toolkits/SimpleAssetEditor.h"

//------------------------------------------------------------------------------
// SUPERDEX BOT
//------------------------------------------------------------------------------

class FSuperDexBotAssetActions : public FAssetTypeActions_Base {
 public:
  FSuperDexBotAssetActions(EAssetTypeCategories::Type InAssetCategory)
      : AssetCategory(InAssetCategory) {}
  virtual uint32 GetCategories() override {
    return AssetCategory;
  }
  virtual FText GetName() const override {
    return USuperDexBot::StaticClass()->GetDisplayNameText();
  }
  virtual UClass* GetSupportedClass() const override {
    return USuperDexBot::StaticClass();
  }
  virtual FColor GetTypeColor() const override {
    return kSuperdexAssetColor;
  }
  virtual bool IsImportedAsset() const override {
    return true;
  }
  virtual void OpenAssetEditor(
      const TArray<UObject*>& InObjects,
      TSharedPtr<IToolkitHost> EditWithinLevelEditor = TSharedPtr<IToolkitHost>()) override {
    if (InObjects.Num() > 0) {
      if (USuperDexBot* object = Cast<USuperDexBot>(InObjects[0])) {
        FMochiArticulationEditorToolkit::CreateEditor(
            EToolkitMode::Standalone, EditWithinLevelEditor, object);
      }
    }
  }

 private:
  const EAssetTypeCategories::Type AssetCategory;
};

//------------------------------------------------------------------------------
// SUPERDEX SCENE
//------------------------------------------------------------------------------

class FSuperDexSceneAssetActions : public FMochiAssetActions<USuperDexScene> {
 public:
  using FMochiAssetActions::FMochiAssetActions;

  virtual bool IsImportedAsset() const override {
    return true;
  }
};

class FSuperDexBotIKConfigAssetActions : public FMochiAssetActions<USuperDexBotIKConfig> {
 public:
  using FMochiAssetActions::FMochiAssetActions;

  virtual bool IsImportedAsset() const override {
    return true;
  }

  virtual void OpenAssetEditor(
      const TArray<UObject*>& Objects,
      TSharedPtr<IToolkitHost> Host = TSharedPtr<IToolkitHost>()) override {
    FSimpleAssetEditor::CreateEditor(EToolkitMode::Standalone, Host, Objects);
  }
};

//------------------------------------------------------------------------------
// DATA ASSETS
//------------------------------------------------------------------------------

// Data assets are created through the generic Data Asset factory; these actions only give them the
// SuperDex type color and category.
using FSuperDexPoseAssetActions = FMochiAssetActions<USuperDexPoseAsset>;
using FSuperDexStudyQuestionDataAssetActions = FMochiAssetActions<USuperDexStudyQuestionData>;
