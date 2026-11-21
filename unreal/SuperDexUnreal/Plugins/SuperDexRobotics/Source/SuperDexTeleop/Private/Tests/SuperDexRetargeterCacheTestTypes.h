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

#include "Retargeters/SuperDexRetargeter.h"
#include "SuperDexPoseableComponent.h"

#include "SuperDexRetargeterCacheTestTypes.generated.h"

// Reflected fixtures; instantiated only by dev automation tests.

// Supplies a fixed pose.
UCLASS()
class USuperDexRetargeterCacheTestSource : public USuperDexPoseableComponent {
  GENERATED_BODY()
 public:
  UPROPERTY(Transient)
  FSuperDexPose TestPose;
  virtual bool GetPose(FSuperDexPose& Out) override {
    Out = TestPose;
    return true;
  }
};

// Reads its own pose cache while it computes its pose, as Blueprint and C++ overrides may.
UCLASS()
class USuperDexRetargeterCacheTestReader : public USuperDexRetargeter {
  GENERATED_BODY()
 public:
  FName LinkName;
  int32 Calls = 0;
  int32 CachedPoseReads = 0;
  int32 CachedLinkReads = 0;
  virtual bool GetTransformedPose_Implementation(
      const FSuperDexPose& SourcePose,
      bool bSourceIsValid,
      FSuperDexPose& TransformedPose) override {
    ++Calls;
    FSuperDexPose Cached;
    CachedPoseReads += GetCachedPose(Cached) ? 1 : 0;
    FTransform LinkToWorld;
    CachedLinkReads += GetCachedPoseLinkTransform(LinkName, LinkToWorld) ? 1 : 0;
    GetVrRootToWorld();
    GetFingerTipTransform(ESuperDexFinger::Index);
    TransformedPose = SourcePose;
    return bSourceIsValid;
  }
};
