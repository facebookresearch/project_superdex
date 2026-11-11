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

#include "AdvancedPreviewScene.h"
#include "CoreMinimal.h"
#include "MochiModelEditorToolkit.h"

// This allows us to customize the scene even further
// gives us access to Tick-level execution
// we can customize the skybox or floor, for instance.
class FMochiModelPreviewScene : public FAdvancedPreviewScene {
 public:
  FMochiModelPreviewScene(
      ConstructionValues CVS,
      const TSharedRef<FMochiModelEditorToolkit>& EditorToolkit);
  virtual ~FMochiModelPreviewScene() override;

  virtual void Tick(float DeltaTime) override;

  TSharedRef<FMochiModelEditorToolkit> GetEditor() const {
    return EditorToolkitPtr.Pin().ToSharedRef();
  }

 private:
  TWeakPtr<FMochiModelEditorToolkit> EditorToolkitPtr;
};
