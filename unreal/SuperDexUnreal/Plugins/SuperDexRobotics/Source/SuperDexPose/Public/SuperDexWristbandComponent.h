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

#include "Components/StaticMeshComponent.h"
#include "Components/TextRenderComponent.h"
#include "CoreMinimal.h"

#include "SuperDexWristbandComponent.generated.h"

UCLASS()
class SUPERDEXPOSE_API USuperDexWristbandComponent : public UStaticMeshComponent {
  GENERATED_BODY()

 public:
  /** Text render component for poseable name display. */
  UPROPERTY(BlueprintReadOnly, Category = "SuperDex")
  TObjectPtr<class UTextRenderComponent> NameTextRender = nullptr;

  /** Text render component for source name display. */
  UPROPERTY(BlueprintReadOnly, Category = "SuperDex")
  TObjectPtr<class UTextRenderComponent> SourceTextRender = nullptr;

  /** Relative transform from the wristband mesh root to poseable name text render component. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  FTransform WristbandToNameText;

  /** Relative transform from the wristband mesh root to source name text render component. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  FTransform WristbandToSourceText;

  /** Material to use for text rendering on both the poseable name and source text renders. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex")
  TObjectPtr<class UMaterialInterface> TextMaterial;

 public:
  USuperDexWristbandComponent();

 protected:
  virtual void OnComponentDestroyed(bool bDestroyingHierarchy) override;
  virtual void OnRegister() override;
  virtual void TickComponent(
      float DeltaTime,
      enum ELevelTick TickType,
      FActorComponentTickFunction* ThisTickFunction) override;

 private:
  /** Creates a text render component, with given relative transform, and sets up default values. */
  UTextRenderComponent* CreateTextRenderComponent(FTransform RelativeTransform);
};
