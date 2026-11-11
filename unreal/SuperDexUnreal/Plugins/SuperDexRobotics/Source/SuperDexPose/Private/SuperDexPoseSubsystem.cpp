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

#include "SuperDexPoseSubsystem.h"
#include "SuperDexPoseModule.h"

#include "Kismet/GameplayStatics.h"
#include "Misc/ScopeLock.h"

USuperDexPoseSubsystem::USuperDexPoseSubsystem() {}

void USuperDexPoseSubsystem::RegisterThisThreadForCaching(
    uint32 Priority,
    std::function<double()> GetCacheKeyCallback) {
  uint32 ThreadId = FPlatformTLS::GetCurrentThreadId();
  if (ThreadIdToRegisteredThreadInfoMap.Contains(ThreadId)) {
    UE_LOG(
        LogSuperDexPose,
        Error,
        TEXT("RegisterThisThreadForCaching: thread %u is already registered; ignoring."),
        ThreadId);
    return;
  }
  ThreadInfoForCaching& Info = ThreadIdToRegisteredThreadInfoMap.FindOrAdd(ThreadId);
  Info.ThreadId = ThreadId;
  Info.Priority = Priority;
  Info.GetCacheKeyCallback = GetCacheKeyCallback;
  {
    FScopeLock Lock1(&UnregisteredThreadsLock);
    int32 NumRemoved = UnregisteredThreadIds.Remove(ThreadId);
    if (NumRemoved > 0) {
      UE_LOG(
          LogSuperDexPose,
          Error,
          TEXT(
              "GetPose: Previously unregistered thread [%d] has been registered."
              " To properly make use of the system please register a thread before calling GetPose() on it."),
          ThreadId);
    }
  }
}

uint32 USuperDexPoseSubsystem::GetHighestPriorityLastCalledWithin(
    USuperDexPoseSourceAndCache* SourceAndCache,
    double CurrentTime,
    double WithinDeltaSecondsAgo) {
  uint32 HighestPriority = 0;
  for (auto kv : SourceAndCache->PriorityToLastRequestTimeMap) {
    if (kv.Key > HighestPriority) {
      if (CurrentTime - kv.Value < WithinDeltaSecondsAgo) {
        HighestPriority = kv.Key;
      }
    }
  }
  return HighestPriority;
}

namespace SuperDexPoseSubsystemInternal {
inline double GetAccurateRealTime() {
  int Seconds;
  double PartialSeconds;
  UGameplayStatics::GetAccurateRealTime(Seconds, PartialSeconds);
  double CurrentTime = Seconds + PartialSeconds;
  return CurrentTime;
}
} // namespace SuperDexPoseSubsystemInternal

bool USuperDexPoseSubsystem::GetPose(
    TSubclassOf<USuperDexPoseSource> PoseSourceClass,
    ESuperDexHandedness Handedness,
    FSuperDexPose& OutPose) {
  if (PoseSourceClass == nullptr) {
    return false;
  }
  uint8 HandednessInt = static_cast<uint8>(Handedness);
  if ((HandednessInt < 0) || (Handedness >= ESuperDexHandedness::Count)) {
    return false; // false if invalid chirality used
  }

  // Get access to the cache for this hand pose
  USuperDexPoseSourceAndCache* SourceAndCache = nullptr;
  {
    FScopeLock Lock1(&MapLock); // limited scope: find the pointer for minimal lock time
    USuperDexPoseSourceAndCache** SourceAndCachePtr = PoseSourceAndCacheMap.Find(PoseSourceClass);
    if (SourceAndCachePtr != nullptr) {
      SourceAndCache = *SourceAndCachePtr;
    }
  }
  if (SourceAndCache == nullptr) {
    return false;
  }

  // Get cache and thread priorities
  uint32 ThreadId = FPlatformTLS::GetCurrentThreadId();
  bool bHasNecessaryThread = (SourceAndCache->Source->GetNecessaryThreadId() != kInvalidThreadId);
  bool bIsNecessaryThread = (SourceAndCache->Source->GetNecessaryThreadId() == ThreadId);
  double CurrentTime = SuperDexPoseSubsystemInternal::GetAccurateRealTime();
  uint32 PriorityNeededToUpdateCache = bHasNecessaryThread
      ? kSourceNecessaryThreadPriority
      : GetHighestPriorityLastCalledWithin(
            SourceAndCache, CurrentTime, kPriorityCacheHoldDurationSeconds);

  ThreadInfoForCaching const* Info = ThreadIdToRegisteredThreadInfoMap.Find(ThreadId);
  uint32 ThisThreadPriority = bIsNecessaryThread ? kSourceNecessaryThreadPriority : 0;
  if (Info != nullptr) {
    if (!bIsNecessaryThread) {
      ThisThreadPriority = Info->Priority;
    }
    double& lastAccessTime =
        SourceAndCache->PriorityToLastRequestTimeMap.FindOrAdd(ThisThreadPriority);
    lastAccessTime = CurrentTime;
  } else {
    FScopeLock Lock2(&UnregisteredThreadsLock);
    if (!UnregisteredThreadIds.Contains(ThreadId)) {
      UE_LOG(
          LogSuperDexPose,
          Error,
          TEXT(
              "GetPose: Called from unregistered thread [%d]."
              " If you are seeing this message you should register your thread to enable caching and avoid slowdowns."
              " This message is only printed once for each thread."),
          ThreadId);
      UnregisteredThreadIds.Add(ThreadId);
    }
  }
  // Lock source in case it needs to update cache. Slightly better than grabbing 2 locks for write
  // then read, but worse if collisions are high (which we dont expect).
  FScopeLock Lock3(&SourceAndCache->SourceLock);
  // Update cache if needed based on thread priorities
  if (ThisThreadPriority >= PriorityNeededToUpdateCache) {
    // Ensure the key is different so we default to update the cache
    double currentCacheKey = SourceAndCache->CacheKey - 1;
    if (Info != nullptr) {
      currentCacheKey = Info->GetCacheKeyCallback();
    }
    // If the keys are different the cache is invalid and both hands should be updated
    if (currentCacheKey != SourceAndCache->CacheKey) {
      for (int iChirality = 0; iChirality < static_cast<int>(ESuperDexHandedness::Count);
           iChirality++) {
        bool haveValidPose = SourceAndCache->Source->GetPose(
            static_cast<ESuperDexHandedness>(iChirality), SourceAndCache->CachedPose[iChirality]);
        if (haveValidPose == true) {
          SourceAndCache->CacheKey = currentCacheKey;
          SourceAndCache->bIsInitialized[iChirality] = true;
        }
      }
    }
  }
  // Get pose from cache
  if (SourceAndCache->bIsInitialized[HandednessInt]) {
    OutPose = SourceAndCache->CachedPose[HandednessInt];
    return true;
  }
  return false;
}

void USuperDexPoseSubsystem::Initialize(FSubsystemCollectionBase& Collection) {
  Super::Initialize(Collection);

  // initialize all hand pose sources: every non-abstract USuperDexPoseSource class, in any loaded
  // module (pose source plugins included), that wants to exist in this world
  // TODO: Consider the module-watcher initialization USubsystemCollection uses, or loading pose
  // source modules late, so that no pose source class is missed (needs testing first).
  TArray<UClass*> HandPoseSourceClasses;
  GetDerivedClasses(USuperDexPoseSource::StaticClass(), HandPoseSourceClasses, true);

  for (UClass* PoseSourceClass : HandPoseSourceClasses) {
    if (PoseSourceClass->HasAllClassFlags(CLASS_Abstract)) {
      continue;
    }
    if (!GetDefault<USuperDexPoseSource>(PoseSourceClass)->ShouldCreatePoseSource(GetWorld())) {
      UE_LOG(
          LogSuperDexPose,
          Log,
          TEXT("Not creating pose source %s: it is not supported here."),
          *PoseSourceClass->GetName());
      continue;
    }
    USuperDexPoseSourceAndCache* HandPoseSourceAndCache =
        NewObject<USuperDexPoseSourceAndCache>(this);
    HandPoseSourceAndCache->Source = NewObject<USuperDexPoseSource>(this, PoseSourceClass);

    PoseSourceAndCacheMap.Add(PoseSourceClass, HandPoseSourceAndCache);
  }

  // Register the UE main thread for caching with our system. Assign it a priority higher than the
  // default and a callback function. GetRealTimeSeconds() will return the current game time value
  // and is only updated once per frame so it should work well. Initialize() is called from the same
  // thread as BeginPlay() and Tick().
  RegisterThisThreadForCaching(
      kUEMainThreadCachePriority, [this]() { return GetWorld()->GetRealTimeSeconds(); });
}

void USuperDexPoseSubsystem::Deinitialize() {
  Super::Deinitialize();

  // clean up hand pose sources
  for (auto Iter = PoseSourceAndCacheMap.CreateIterator(); Iter; ++Iter) {
    USuperDexPoseSourceAndCache* HandPoseSourceAndCache = Iter.Value();
    USuperDexPoseSource* HandPoseSource = HandPoseSourceAndCache->Source;

    if (HandPoseSource) {
      // HandPoseSource->Deinitialize();
    }
  }

  PoseSourceAndCacheMap.Empty();
}
