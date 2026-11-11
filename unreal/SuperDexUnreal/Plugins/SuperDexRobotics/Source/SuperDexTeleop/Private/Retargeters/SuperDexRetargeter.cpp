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

#include "Retargeters/SuperDexRetargeter.h"

#include "SuperDexPoseModule.h"

#include "Kismet/GameplayStatics.h"
#include "SuperDexPoseSubsystem.h"

#include "SuperDexPlayerController.h"
#include "SuperDexPoseSource.h"
#include "SuperDexPoseableActor.h"

// #pragma optimize("", off)

// Duplicate function to the one found in the USuperDexPoseSubsystem.
// Wrapped in a named namespace to avoid redefinition in UE5 unity builds.
namespace SuperDexRetargeterInternal {
inline double GetAccurateRealTime() {
  int Seconds;
  double PartialSeconds;
  UGameplayStatics::GetAccurateRealTime(Seconds, PartialSeconds);
  double CurrentTime = Seconds + PartialSeconds;
  return CurrentTime;
}
} // namespace SuperDexRetargeterInternal

// Duplicate function to the one found in the USuperDexPoseSubsystem
uint32 USuperDexRetargeter::GetHighestPriorityLastCalledWithin(
    double CurrentTime,
    double WithinDeltaSecondsAgo) {
  uint32 HighestPriority = 0;
  for (auto kv : CachePriorityToLastRequestTimeMap) {
    if (kv.Key > HighestPriority) {
      if (CurrentTime - kv.Value < WithinDeltaSecondsAgo) {
        HighestPriority = kv.Key;
      }
    }
  }
  return HighestPriority;
}

bool USuperDexRetargeter::GetPose(FSuperDexPose& OutPose) {
  if (bEnableGetPose == false) {
    return false;
  }

  // don't update pose unless we're in a game world
  check(IsValid(GetWorld()));
  if (!GetWorld()->IsGameWorld())
    return false;

  uint32 ThreadId = FPlatformTLS::GetCurrentThreadId();
  USuperDexPoseSubsystem* Hpss = GetWorld()->GetSubsystem<USuperDexPoseSubsystem>();
  ThreadInfoForCaching const* Info = Hpss->GetThreadInfoForCaching().Find(ThreadId);
  double CurrentTime = SuperDexRetargeterInternal::GetAccurateRealTime();

  // Check for reentrant calls and exit or the CacheLock will deadlock itself
  bool infiniteLoopDetected = CheckIfInInfiniteLoop(OutPose);
  if (infiniteLoopDetected) {
    return false;
  }

  FWriteScopeLock Lock1(CacheLock);
  // Get cache and thread priorities
  uint32 PriorityNeededToUpdateCache =
      GetHighestPriorityLastCalledWithin(CurrentTime, kPriorityCacheHoldDurationSeconds);
  // Unregistered threads will be error reported by the USuperDexPoseSubsystem so only check
  // for null
  uint32 ThisThreadPriority = 0;
  if (Info != nullptr) {
    ThisThreadPriority = Info->Priority;
    double& lastAccessTime = CachePriorityToLastRequestTimeMap.FindOrAdd(ThisThreadPriority);
    lastAccessTime = CurrentTime;
  }

  // Update cache if needed based on thread priorities
  if (ThisThreadPriority >= PriorityNeededToUpdateCache) {
    // Ensure the key is different so we default to update the cache
    double CurrentCacheKey = CacheKey - 1;
    if (Info != nullptr) {
      CurrentCacheKey = Info->GetCacheKeyCallback();
    }
    // If the keys are different the cache is invalid and should be updated
    if (CurrentCacheKey != CacheKey) {
      FSuperDexPose SourcePose;
      bool bHavePose = GetInputPoseFromSource(SourcePose);
      // update the cache and OutPose to reflect infinite loops so we catch all of them
      // the cache may have been initialized by the class using a different method
      OutPose.bIsInInfiniteLoop |= SourcePose.bIsInInfiniteLoop;
      if ((bHavePose == false) && (bGetTransformedPoseDoesNotNeedValidSourcePose == false)) {
        return false;
      }
      FSuperDexPose TransformedPose;
      bHavePose = GetTransformedPose(SourcePose, bHavePose, TransformedPose);
      // update the cache and OutPose to reflect infinite loops so we catch all of them
      // the cache may have been initialized by the class using a different method
      OutPose.bIsInInfiniteLoop |= TransformedPose.bIsInInfiniteLoop;
      if (bHavePose == false) {
        return false;
      }
      if (bOverrideOutputPoseScale) {
        TransformedPose.SkeletonRootToVrRoot.SetScale3D(FVector(OutputPoseScaleOverride));
      }
      CachedOutputPose = TransformedPose;
      CacheKey = CurrentCacheKey;
      CacheInitialized = true;
    }
  }
  // Get pose from cache
  if (CacheInitialized) {
    // no need to preserve the entry OutPose.bIsInInfiniteLoop because all combinations of it in
    // OutPose and CachedOutputPose result in the correct bool value
    bool bIsInInfiniteLoop = OutPose.bIsInInfiniteLoop;
    OutPose = CachedOutputPose;
    OutPose.bIsInInfiniteLoop = bIsInInfiniteLoop;
  }
  return CacheInitialized;
}

void USuperDexRetargeter::OnTickManageCache(FSuperDexPose& Pose, bool isValid) {
  // The cache is updated as part of GetPose() so dont do the normal cache management
}

bool USuperDexRetargeter::GetTransformedPose_Implementation(
    FSuperDexPose const& SourcePose,
    bool bSourceIsValid,
    FSuperDexPose& TransformedPose) {
  TransformedPose = SourcePose;
  return bSourceIsValid;
}
