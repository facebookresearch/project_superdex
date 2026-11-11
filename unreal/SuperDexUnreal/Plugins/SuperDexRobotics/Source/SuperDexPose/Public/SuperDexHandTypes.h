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

#include "Misc/EnumRange.h"

#include "SuperDexHandTypes.generated.h"

// Hand types shared by every skeleton. Skeleton-specific bones, joints and tables live in the
// Skeletons/SuperDexSkeleton_*.h headers.

UENUM(BlueprintType)
enum class ESuperDexHandedness : uint8 {
  Left = 0, // Left Hand
  Right = 1, // Right Hand
  Count = 2 UMETA(Hidden),
  None = 255 UMETA(Hidden) // No Hand
};
ENUM_RANGE_BY_COUNT(ESuperDexHandedness, ESuperDexHandedness::Count);

UENUM(BlueprintType)
enum class ESuperDexFinger : uint8 {
  Thumb = 0,
  Index = 1,
  Middle = 2,
  Ring = 3,
  Pinky = 4,
  Count UMETA(Hidden),
  Invalid = 255 UMETA(Hidden)
};
ENUM_RANGE_BY_COUNT(ESuperDexFinger, ESuperDexFinger::Count);

UENUM(BlueprintType)
enum class ESuperDexFingerSegment : uint8 {
  Metacarpal = 0,
  Proximal,
  Intermediate,
  Distal,
  Count UMETA(Hidden),
  Invalid = 255 UMETA(Hidden),
};
ENUM_RANGE_BY_COUNT(ESuperDexFingerSegment, ESuperDexFingerSegment::Count);
