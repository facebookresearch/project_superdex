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

#include "CoreMinimal.h"
#include "Engine/DeveloperSettings.h"

#include "SuperDexOpenXRPoseSourceSettings.generated.h"

/** Project settings of the SuperDex OpenXR (hand tracking) pose source. */
UCLASS(Config = Game, DefaultConfig, meta = (DisplayName = "SuperDex OpenXR Pose Source"))
class SUPERDEXOPENXRPOSESOURCE_API USuperDexOpenXRPoseSourceSettings : public UDeveloperSettings {
  GENERATED_BODY()

 public:
  /**
   * Account for the Meta (Quest) hand tracking runtime's thumb in pinches. The runtime's hand is
   * the Meta hand skeleton, whose thumb metacarpal origin moves with a trapezium (CMC) joint that
   * the OpenXR bot lacks, so the bot's thumb falls short of the tracked one and tracked pinches
   * open. When true, the thumb is refit to the tracked thumb distal frame and nearby fingers are
   * blended into the pinch (ApplyOpenXRPinchCorrection). When false, the pose holds the tracked
   * joint rotations only. Read every frame; the console variable
   * superdex.OpenXRPinchCorrection.Enable (0 or 1) overrides it at runtime, -1 (its default) defers
   * to this setting.
   */
  UPROPERTY(Config, EditAnywhere, BlueprintReadOnly, Category = "Hand Tracking")
  bool bAccountForQuestHandTrackingPinch = true;

  virtual FName GetCategoryName() const override {
    return TEXT("Plugins");
  }
};
