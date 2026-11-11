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

#include "Components/ActorComponent.h"
#include "CoreMinimal.h"
#include "Drawing/LineSetComponent.h"

#include "MochiForceVisualizerComponent.generated.h"

UENUM(BlueprintType)
enum class EMochiForceVisualizerColorMode : uint8 {
  // Use two user-defined colors with linear interpolation
  TwoTone = 0,
  // Use the 'turbo' color map, with pressure mapped to hue
  Rainbow = 1
};

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class SUPERDEXPHYSICS_API UMochiForceVisualizerComponent : public ULineSetComponent {
  GENERATED_BODY()
 public:
  // Enable/disable rendering visualization.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  bool bEnabled = true;

  // Used to determine min/max force [N] for interpolating color and line length.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector2D ForceMinMax = FVector2D(0.0f, 1.0f);

  // Use to determine the line length [cm] given ForceMinMax.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  FVector2D LineLengthMinMax = FVector2D(0.0f, 10.0f);

  // How to map forces to colors
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  EMochiForceVisualizerColorMode ColorMode = EMochiForceVisualizerColorMode::TwoTone;

  // The number of discrete color bands to which color should be quantized
  // (set to zero to disable)
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  int NumColorBands = 0;

  // Color representing low force (for the TwoTone color mode).
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta =
          (EditCondition = "ColorMode == EMochiForceVisualizerColorMode::TwoTone",
           EditConditionHides),
      Category = "Mochi")
  FLinearColor LowColor = FColor::Blue;

  // Color representing high force (for the TwoTone color mode).
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta =
          (EditCondition = "ColorMode == EMochiForceVisualizerColorMode::TwoTone",
           EditConditionHides),
      Category = "Mochi")
  FLinearColor HighColor = FColor::Red;

  // The range of the turbo color map used by the Rainbow color mode
  // Minimum force maps to RainbowRange.X
  // Maximum force maps to RainbowRange.Y
  // Intermediate values are linearly interpolated between these extremes
  // X may be greater than Y
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta =
          (EditCondition = "ColorMode == EMochiForceVisualizerColorMode::Rainbow",
           EditConditionHides),
      Category = "Mochi")
  FVector2D RainbowRange = FVector2D(0.0f, 1.0f);

  // A distinct color value to use when force is exactly equal to ForceMinMax.X
  // (e.g so you can identify non-inflated bubbles)
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta =
          (EditCondition = "ColorMode == EMochiForceVisualizerColorMode::Rainbow",
           EditConditionHides),
      Category = "Mochi")
  FColor MinColor = FColor::Black;

  // A distinct color value to use when force is exactly equal to ForceMinMax.Y
  // (e.g so you can identify fully-inflated bubbles)
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta =
          (EditCondition = "ColorMode == EMochiForceVisualizerColorMode::Rainbow",
           EditConditionHides),
      Category = "Mochi")
  FColor MaxColor = FColor::White;

 public:
  UMochiForceVisualizerComponent();

 protected:
  virtual void BeginPlay() override;
  virtual void TickComponent(
      float DeltaTime,
      ELevelTick TickType,
      FActorComponentTickFunction* ThisTickFunction) override;

 private:
  UFUNCTION(Category = "Mochi")
  void OnMochiComponentBeginContact(
      USceneComponent* MochiComponent,
      AActor* OtherMochiActor,
      USceneComponent* OtherMochiComponent,
      const TArray<FMochiContactPoint>& Contacts);

  UFUNCTION(Category = "Mochi")
  void OnMochiComponentEndContact(
      USceneComponent* MochiComponent,
      AActor* OtherMochiActor,
      USceneComponent* OtherMochiComponent);

  UFUNCTION(Category = "Mochi")
  void OnMochiComponentContact(
      USceneComponent* MochiComponent,
      const TArray<FMochiContactPoint>& Contacts);

 public:
  FColor GetColorFromForce(float Force) const;

 protected:
  TMap<USceneComponent*, TArray<FMochiContactPoint>> ContactsMap;
};
