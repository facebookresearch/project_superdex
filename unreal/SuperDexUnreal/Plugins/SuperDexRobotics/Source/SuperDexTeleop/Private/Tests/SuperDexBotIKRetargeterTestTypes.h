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

/** Pose source fixture; instantiated only by dev automation tests. */
UCLASS()
class USuperDexBotIKTestPoseSource : public USuperDexPoseableComponent {
  GENERATED_BODY()
 public:
  /** The pose GetPose returns. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Transient, Category = "SuperDex Robotics|Test")
  FSuperDexPose TestPose;

  /** The value GetPose returns. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Transient, Category = "SuperDex Robotics|Test")
  bool bTestValid = true;
  virtual bool GetPose(FSuperDexPose& Out) override {
    Out = TestPose;
    return bTestValid;
  }
};
