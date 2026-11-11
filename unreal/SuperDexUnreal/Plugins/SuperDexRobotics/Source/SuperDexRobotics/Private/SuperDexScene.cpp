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

#include "SuperDexScene.h"

#include "SuperDexBot.h"

#if WITH_EDITOR

void USuperDexScene::PostInitProperties() {
  Super::PostInitProperties();
  if (!HasAnyFlags(RF_ClassDefaultObject)) {
    AssetImportData = NewObject<USuperDexPrefabImportData>(this, TEXT("AssetImportData"));
  }
}

void USuperDexScene::GetAssetRegistryTags(FAssetRegistryTagsContext Context) const {
  if (AssetImportData) {
    Context.AddTag(FAssetRegistryTag(
        SourceFileTagName(),
        AssetImportData->GetSourceData().ToJson(),
        FAssetRegistryTag::TT_Hidden));
  }
  Super::GetAssetRegistryTags(Context);
}

FString USuperDexScene::GetSourceFilePath() const {
  return AssetImportData ? AssetImportData->GetFirstFilename() : FString();
}

#endif // WITH_EDITOR
