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

#include "MochiModelPreviewScene.h"
#include "GameFramework/WorldSettings.h"

FMochiModelPreviewScene::FMochiModelPreviewScene(
    ConstructionValues CVS,
    const TSharedRef<FMochiModelEditorToolkit>& EditorToolkit)
    : FAdvancedPreviewScene(CVS), EditorToolkitPtr(EditorToolkit) {
  // prevents killing actors outside world bounds
  AWorldSettings* WorldSettings = GetWorld()->GetWorldSettings(true);
  WorldSettings->bEnableWorldBoundsChecks = false;

  SetFloorVisibility(false, false);
}

FMochiModelPreviewScene::~FMochiModelPreviewScene() {}

void FMochiModelPreviewScene::Tick(float DeltaTime) {
  FAdvancedPreviewScene::Tick(DeltaTime);
}
