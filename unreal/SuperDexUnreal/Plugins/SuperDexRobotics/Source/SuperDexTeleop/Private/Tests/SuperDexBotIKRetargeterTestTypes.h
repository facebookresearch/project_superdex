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

#include "Retargeters/SuperDexBotIKRetargeter.h"
#include "SuperDexPoseableComponent.h"

#include "SuperDexBotIKRetargeterTestTypes.generated.h"

// Reflected callback fixtures; instantiated only by dev automation tests.
UCLASS()
class USuperDexBotIKTestPoseSource : public USuperDexPoseableComponent {
  GENERATED_BODY()
 public:
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Transient, Category = "Test")
  FSuperDexPose TestPose;
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Transient, Category = "Test")
  bool bTestValid = true;
  bool bReenter = false;
  bool bNestedReturned = true;
  UPROPERTY()
  TObjectPtr<USuperDexBotIKRetargeter> Retargeter;
  virtual bool GetPose(FSuperDexPose& Out) override {
    if (bReenter && Retargeter) {
      FSuperDexPose Nested;
      bNestedReturned = Retargeter->GetPose(Nested);
    }
    Out = TestPose;
    return bTestValid;
  }
};
