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

#include "Skeletons/SuperDexSkeletonBots.h"

#include "ISuperDexRoboticsModule.h"
#include "Skeletons/SuperDexSkeletonBotJson.h"
#include "Skeletons/SuperDexSkeletonBotsLifetime.h"
#include "Skeletons/SuperDexSkeleton_OculusXR.h"
#include "Skeletons/SuperDexSkeleton_OpenXR.h"
#include "SuperDexBot.h"
#include "SuperDexPoseModule.h"

#include "UObject/GCObject.h"
#include "UObject/Package.h"

namespace {

enum class ESkeleton : uint8 { OpenXR, OculusXR, Count };

constexpr int32 kNumSkeletons = static_cast<int32>(ESkeleton::Count);
constexpr int32 kNumHands = static_cast<int32>(ESuperDexHandedness::Count);

/* Pruned BotPrefab JSON of each skeleton bot (SuperDexSkeletonBotJson.h), by ESkeleton and hand. */
const char* const kSkeletonBotJson[kNumSkeletons][kNumHands] = {
    {kSkeletonBotJson_OpenXR_Left, kSkeletonBotJson_OpenXR_Right},
    {kSkeletonBotJson_OculusXR_Left, kSkeletonBotJson_OculusXR_Right},
};

/* Bone names of each skeleton, which must be its bot's link names. */
TArray<FName> GetBoneNames(ESkeleton Skeleton) {
  switch (Skeleton) {
    case ESkeleton::OpenXR:
      return TArray<FName>(kSuperDexBoneNames_OXR, UE_ARRAY_COUNT(kSuperDexBoneNames_OXR));
    case ESkeleton::OculusXR:
      return TArray<FName>(
          kSuperDexBoneNames_OculusXR, UE_ARRAY_COUNT(kSuperDexBoneNames_OculusXR));
    case ESkeleton::Count:
      break;
  }
  return {};
}

/* Keeps the skeleton bots alive from CreateSkeletonBots to DestroySkeletonBots. */
class FSkeletonBots : public FGCObject {
 public:
  TObjectPtr<USuperDexBot> Bots[kNumSkeletons][kNumHands] = {};

  virtual void AddReferencedObjects(FReferenceCollector& Collector) override {
    for (TObjectPtr<USuperDexBot>(&Hands)[kNumHands] : Bots) {
      for (TObjectPtr<USuperDexBot>& Bot : Hands) {
        Collector.AddReferencedObject(Bot);
      }
    }
  }

  virtual FString GetReferencerName() const override {
    return TEXT("SuperDexSkeletonBots");
  }
};

TUniquePtr<FSkeletonBots> GSkeletonBots;

UMochiArticulationBase* GetSkeletonBot(ESkeleton Skeleton, ESuperDexHandedness Handedness) {
  if (!GSkeletonBots.IsValid() ||
      (Handedness != ESuperDexHandedness::Left && Handedness != ESuperDexHandedness::Right)) {
    return nullptr;
  }
  return GSkeletonBots->Bots[static_cast<int32>(Skeleton)][static_cast<int32>(Handedness)];
}

} // namespace

void CreateSkeletonBots() {
  check(IsInGameThread());
  // SuperDexRobotics loads the superdex_robotics library that reads the bots.
  FModuleManager::LoadModuleChecked<ISuperDexRoboticsModule>("SuperDexRobotics");
  GSkeletonBots = MakeUnique<FSkeletonBots>();
  for (int32 Skeleton = 0; Skeleton < kNumSkeletons; ++Skeleton) {
    for (int32 Hand = 0; Hand < kNumHands; ++Hand) {
      USuperDexBot* Bot = USuperDexBot::CreateTransient(
          GetTransientPackage(), FString(UTF8_TO_TCHAR(kSkeletonBotJson[Skeleton][Hand])));
      if (Bot == nullptr) {
        UE_LOG(
            LogSuperDexPose,
            Error,
            TEXT("Pose skeleton bot %d of hand %d does not load: no poses on that skeleton."),
            Skeleton,
            Hand);
        continue;
      }
      if (!ensureMsgf(
              Bot->GetLinkNames() == GetBoneNames(static_cast<ESkeleton>(Skeleton)),
              TEXT("The links of pose skeleton bot %s are not the bones of its skeleton."),
              *Bot->GetBotName())) {
        continue;
      }
      GSkeletonBots->Bots[Skeleton][Hand] = Bot;
    }
  }
}

void DestroySkeletonBots() {
  GSkeletonBots.Reset();
}

UMochiArticulationBase* GetSkeletonBot_OpenXR(ESuperDexHandedness Handedness) {
  return GetSkeletonBot(ESkeleton::OpenXR, Handedness);
}

UMochiArticulationBase* GetSkeletonBot_OculusXR(ESuperDexHandedness Handedness) {
  return GetSkeletonBot(ESkeleton::OculusXR, Handedness);
}
