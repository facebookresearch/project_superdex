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
#include "SuperDexPoseSource.h"
#include "SuperDexPoseTypes.h"
#include "Tickable.h"

#include "SuperDexOpenXRPoseSource.generated.h"

/**
 * Pose source for OpenXR (XR_EXT_hand_tracking) hand tracking through the active XR system, e.g.
 * Epic's OpenXR plugin with OpenXRHandTracking enabled. Produces poses on kSkeletonName_OpenXR,
 * which drive the openxr superdex hand bots directly.
 *
 * Hand tracking state is read on the game thread in Tick and cached.
 *
 * OpenXR and OpenXRHandTracking are optional dependencies of this plugin: where they are missing
 * (an engine without them, or a project that disables them) the pose subsystem does not create
 * this source (ShouldCreatePoseSource).
 *
 * Frame contract: OpenXR reports right-handed joint frames (+Y dorsal, +Z along the bone away from
 * the finger tip). UE's OpenXR plugin re-expresses them in Unreal space with the basis change
 * B: (x, y, z) -> (-z, x, y). The superdex importer re-expresses the openxr bot's link frames (the
 * same OpenXR joint frames) with the Y-flip S = diag(1, -1, 1). The bone frame of joint j is
 * therefore G_j = Q_j * C with C = B * S, for the UE joint rotation Q_j, on both hands. Its local
 * rotations are exactly the importer-converted OpenXR local rotations.
 */
UCLASS()
class SUPERDEXOPENXRPOSESOURCE_API USuperDexOpenXRPoseSource : public USuperDexPoseSource,
                                                               public FTickableGameObject {
  GENERATED_BODY()

 public:
  virtual bool ShouldCreatePoseSource(UWorld* World) const override;

  virtual bool GetPose(const ESuperDexHandedness Handedness, FSuperDexPose& OutPose) override;

  /**
   * Builds an FSuperDexPose on kSkeletonName_OpenXR from OpenXR joint poses as UE's OpenXR plugin
   * reports them.
   *
   * @param KeypointRotations Joint rotations in EHandKeypoint (XrHandJointEXT) order, all in the
   *   space the pose's root is expressed in.
   * @param WristLocation Wrist joint location in that space.
   * @param OutPose Receives BoneRotations (local to parent, ESuperDexBone_OXR order) and
   *   SkeletonRootToVrRoot (the wrist bone, unit scale; the pose source sets the hand scale, see
   *   EstimateHandScale). BoneLocations is left empty so consumers use the bot's rest offsets.
   */
  static void ConvertHandKeypointsToPose(
      TConstArrayView<FQuat> KeypointRotations,
      const FVector& WristLocation,
      FSuperDexPose& OutPose);

  /**
   * Estimates the uniform scale of a tracked hand relative to the OpenXR bot's nominal hand.
   *
   * The OpenXR runtime places the finger joints by scaling one unit hand skeleton (the one the
   * openxr bots are built from) by the user's hand scale, so the phalanx lengths are exactly the
   * nominal ones times that scale whatever the pose.
   *
   * The scale is relative to the bots as Unreal shows them, at 1 cm per Unreal unit (Mochi's
   * Unreal space) whatever the world's WorldToMeters, so a bot at this scale matches the tracked
   * hand in the world. With WorldToMeters = 100 it is the user's physical hand scale.
   *
   * @param KeypointLocations Joint locations in EHandKeypoint (XrHandJointEXT) order, in tracking
   *   space [Unreal units].
   * @return The hand scale, or 1 if the joints are degenerate.
   */
  static double EstimateHandScale(TConstArrayView<FVector> KeypointLocations);

  /** FTickableGameObject interface */
  virtual UWorld* GetTickableGameObjectWorld() const override;
  virtual ETickableTickType GetTickableTickType() const override;
  virtual bool IsTickable() const override;
  virtual bool IsTickableWhenPaused() const override;
  virtual bool IsTickableInEditor() const override;
  virtual void Tick(float DeltaTime) override;
  virtual TStatId GetStatId() const override;

 private:
  void CachePose(const ESuperDexHandedness Handedness);

  mutable FCriticalSection StateLock;

  FSuperDexPose LeftPoseCache;

  FSuperDexPose RightPoseCache;
};
