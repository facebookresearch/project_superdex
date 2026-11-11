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

#include "Engine/DataAsset.h"
#include "Misc/EnumRange.h"
#include "SuperDexHandTypes.h"

#include "SuperDexSkeleton_OpenXR.generated.h"

/**
 * OpenXR (XR_EXT_hand_tracking) hand skeleton, as used by the openxr superdex hand bots
 * (assets/bots/hands/openxr in the SuperDex repository).
 *
 * Bones are in the bots' link order: XrHandJointEXT order with WRIST and PALM swapped so that the
 * wrist is the root. Every bone frame is the OpenXR joint frame (+Z along the bone away from the
 * finger tip, +Y dorsal) carried into Unreal by the superdex importer, so an FSuperDexPose on this
 * skeleton drives the imported bots with no per-bone correction.
 */
static const FName kSkeletonName_OpenXR = "OpenXR";

UENUM(BlueprintType)
enum class ESuperDexBone_OXR : uint8 {
  Wrist = 0,
  Palm = 1,
  ThumbMetacarpal = 2,
  ThumbProximal = 3,
  ThumbDistal = 4,
  ThumbTip = 5,
  IndexMetacarpal = 6,
  IndexProximal = 7,
  IndexIntermediate = 8,
  IndexDistal = 9,
  IndexTip = 10,
  MiddleMetacarpal = 11,
  MiddleProximal = 12,
  MiddleIntermediate = 13,
  MiddleDistal = 14,
  MiddleTip = 15,
  RingMetacarpal = 16,
  RingProximal = 17,
  RingIntermediate = 18,
  RingDistal = 19,
  RingTip = 20,
  LittleMetacarpal = 21,
  LittleProximal = 22,
  LittleIntermediate = 23,
  LittleDistal = 24,
  LittleTip = 25,
  Count = 26 UMETA(Hidden),
  Invalid = 255 UMETA(Hidden)
};
ENUM_RANGE_BY_COUNT(ESuperDexBone_OXR, ESuperDexBone_OXR::Count);

/** Bone (link) names of the openxr superdex hand bots, indexed by ESuperDexBone_OXR. */
static const FName kSuperDexBoneNames_OXR[] = {
    "xr_01_wrist",
    "xr_00_palm",
    "xr_02_thumb_metacarpal",
    "xr_03_thumb_proximal",
    "xr_04_thumb_distal",
    "xr_05_thumb_tip",
    "xr_06_index_metacarpal",
    "xr_07_index_proximal",
    "xr_08_index_intermediate",
    "xr_09_index_distal",
    "xr_10_index_tip",
    "xr_11_middle_metacarpal",
    "xr_12_middle_proximal",
    "xr_13_middle_intermediate",
    "xr_14_middle_distal",
    "xr_15_middle_tip",
    "xr_16_ring_metacarpal",
    "xr_17_ring_proximal",
    "xr_18_ring_intermediate",
    "xr_19_ring_distal",
    "xr_20_ring_tip",
    "xr_21_little_metacarpal",
    "xr_22_little_proximal",
    "xr_23_little_intermediate",
    "xr_24_little_distal",
    "xr_25_little_tip"};
static_assert(UE_ARRAY_COUNT(kSuperDexBoneNames_OXR) == static_cast<int>(ESuperDexBone_OXR::Count));

/** Parent bone of each ESuperDexBone_OXR, or Invalid for the root. */
static constexpr ESuperDexBone_OXR kSuperDexBoneParents_OXR[] = {
    ESuperDexBone_OXR::Invalid,
    ESuperDexBone_OXR::Wrist,
    ESuperDexBone_OXR::Wrist,
    ESuperDexBone_OXR::ThumbMetacarpal,
    ESuperDexBone_OXR::ThumbProximal,
    ESuperDexBone_OXR::ThumbDistal,
    ESuperDexBone_OXR::Wrist,
    ESuperDexBone_OXR::IndexMetacarpal,
    ESuperDexBone_OXR::IndexProximal,
    ESuperDexBone_OXR::IndexIntermediate,
    ESuperDexBone_OXR::IndexDistal,
    ESuperDexBone_OXR::Wrist,
    ESuperDexBone_OXR::MiddleMetacarpal,
    ESuperDexBone_OXR::MiddleProximal,
    ESuperDexBone_OXR::MiddleIntermediate,
    ESuperDexBone_OXR::MiddleDistal,
    ESuperDexBone_OXR::Wrist,
    ESuperDexBone_OXR::RingMetacarpal,
    ESuperDexBone_OXR::RingProximal,
    ESuperDexBone_OXR::RingIntermediate,
    ESuperDexBone_OXR::RingDistal,
    ESuperDexBone_OXR::Wrist,
    ESuperDexBone_OXR::LittleMetacarpal,
    ESuperDexBone_OXR::LittleProximal,
    ESuperDexBone_OXR::LittleIntermediate,
    ESuperDexBone_OXR::LittleDistal};
static_assert(
    UE_ARRAY_COUNT(kSuperDexBoneParents_OXR) == static_cast<int>(ESuperDexBone_OXR::Count));

/** XrHandJointEXT (and EHandKeypoint) value of an ESuperDexBone_OXR. */
constexpr int32 GetXrHandJointOfBone_OXR(ESuperDexBone_OXR Bone) {
  const int32 Index = static_cast<int32>(Bone);
  return Index == 0 ? 1 : (Index == 1 ? 0 : Index);
}

/** Tip bone of a finger on the OpenXR skeleton (its name: kSuperDexBoneNames_OXR[Bone]). */
constexpr ESuperDexBone_OXR GetFingerTipBone_OXR(ESuperDexFinger Finger) {
  switch (Finger) {
    case ESuperDexFinger::Thumb:
      return ESuperDexBone_OXR::ThumbTip;
    case ESuperDexFinger::Index:
      return ESuperDexBone_OXR::IndexTip;
    case ESuperDexFinger::Middle:
      return ESuperDexBone_OXR::MiddleTip;
    case ESuperDexFinger::Ring:
      return ESuperDexBone_OXR::RingTip;
    case ESuperDexFinger::Pinky:
      return ESuperDexBone_OXR::LittleTip;
    case ESuperDexFinger::Count:
    case ESuperDexFinger::Invalid:
      break;
  }
  return ESuperDexBone_OXR::Invalid;
}
