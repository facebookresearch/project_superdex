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
#include "Subsystems/WorldSubsystem.h"
#include "SuperDexPoseSource.h"

#include <functional>

#include "SuperDexPoseSubsystem.generated.h"

/** Default thread cache priority constant. Lowest by default. */
constexpr uint32 kDefaultThreadCachePriority = 0;

/** UE main thread cache priority constant. */
constexpr uint32 kUEMainThreadCachePriority = 10;

/** Mochi thread cache priority constant. Highest by default. */
constexpr uint32 kMochiThreadCachePriority = 100;

/** Priority cache hold duration constant, in seconds. */
constexpr double kPriorityCacheHoldDurationSeconds = 1.0;

/** Priority assigned to specific thread IDs that sources have flagged as necessary. */
constexpr uint32 kSourceNecessaryThreadPriority = std::numeric_limits<uint32>::max();

/** Thread information used by the caching system to manage caches. */
struct ThreadInfoForCaching {
  /** Thread ID. */
  uint32 ThreadId = 0;

  /** Priority of the thread. Higher priorities take ownership. */
  uint32 Priority = kDefaultThreadCachePriority;

  /** Callback function to get cache key used for checking whether cache needs updating. */
  std::function<double()> GetCacheKeyCallback;
};

/** A pose source and its associated cache. */
UCLASS()
class USuperDexPoseSourceAndCache : public UObject {
  GENERATED_BODY()

 public:
  /** Flag indicating whether the cache has ever been assigned a valid pose.
   * One for each handedness.
   */
  TStaticArray<bool, static_cast<uint8>(ESuperDexHandedness::Count)> bIsInitialized = false;

  /** Critical section to prevent torn data when getting or updating info from the cache. */
  mutable FCriticalSection SourceLock;

  /** Array of poses, one for each handedness. */
  TStaticArray<FSuperDexPose, static_cast<uint8>(ESuperDexHandedness::Count)> CachedPose;

  /** Key as of the last cache update. If a new key does not match it means the cache is invalid.
   * Each update refreshes each handedness so that the cached poses stay in sync. */
  double CacheKey = -1;

  /**
   * Map to keep track of what priorities were called on the inputs for the source to determine if
   * the cache is allowed to update.
   */
  TMap<uint32, double> PriorityToLastRequestTimeMap;

  /** Pose source associated with this cache. */
  UPROPERTY()
  USuperDexPoseSource* Source;
};

/**
 * Subsystem for getting (hand) poses with SuperDexPose.
 *
 * Connects to several, usually external, sources for hand pose - making them available to UE.
 * Manages a cache to ensure pose isn't retrieved from sources too often, and that higher priority
 * threads control the cache. Has the same lifetime as UWorld, connections begin at begin play.
 */
UCLASS()
class SUPERDEXPOSE_API USuperDexPoseSubsystem : public UWorldSubsystem {
  GENERATED_BODY()

 private:
  /** Critical section to protect access to the pose source and cache map. */
  mutable FCriticalSection MapLock;

  /** Critical section to protect access to the unregistered threads set. */
  mutable FCriticalSection UnregisteredThreadsLock;

  /**
   * Map of pose sources to their associated caches. The map value is a pointer to ensure that the
   * cache remains valid even if the map is reallocated.
   */
  UPROPERTY()
  TMap<UClass*, USuperDexPoseSourceAndCache*> PoseSourceAndCacheMap;

  /** Map of thread IDs to their corresponding thread information structs. */
  TMap<uint32, ThreadInfoForCaching> ThreadIdToRegisteredThreadInfoMap;

  /** Set of thread IDs that have called GetPose without registering. */
  TSet<uint32> UnregisteredThreadIds;

 public:
  USuperDexPoseSubsystem();

  /**
   * Get pose from given pose source class for given handedness.
   *
   * @param PoseSourceClass The pose source class to get pose from.
   * @param Handedness The handedness to get pose for.
   * @param OutPose The output pose.
   * @return true if successful, false otherwise.
   */
  UFUNCTION(BlueprintCallable, Category = "SuperDexPoseSubsystem")
  bool GetPose(
      TSubclassOf<USuperDexPoseSource> PoseSourceClass,
      ESuperDexHandedness Handedness,
      FSuperDexPose& OutPose);

  /**
   * Register a thread to have a priority and cache key callback function.
   *
   * @param Priority The priority of the thread. Highest takes ownership.
   * @param GetCacheKeyCallback The callback function to get cache key used for checking whether
   * cache needs updating.
   */
  void RegisterThisThreadForCaching(uint32 Priority, std::function<double()> GetCacheKeyCallback);

  /**
   * Get the thread information for caching.
   *
   * @return A map of thread IDs to their corresponding thread information.
   */
  TMap<uint32, ThreadInfoForCaching> const& GetThreadInfoForCaching() {
    return ThreadIdToRegisteredThreadInfoMap;
  }

 protected:
  virtual void Initialize(FSubsystemCollectionBase& Collection) override;
  virtual void Deinitialize() override;

  /**
   * Get the highest priority last called within a certain time frame.
   *
   * @param SourceAndCache The pose source and cache to check.
   * @param CurrentTime The current time.
   * @param WithinDeltaSecondsAgo The time frame to check within.
   * @return The highest priority last called within
   */
  uint32 GetHighestPriorityLastCalledWithin(
      USuperDexPoseSourceAndCache* SourceAndCache,
      double CurrentTime,
      double WithinDeltaSecondsAgo);
};
