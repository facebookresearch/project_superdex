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
#include "Skeletons/SuperDexSkeleton_OpenXR.h"
#include "SuperDexHandTypes.h"

/*
 * Pinch correction for poses on the OpenXR skeleton bot (GetSkeletonBot_OpenXR).
 *
 * Hand trackers built on the Meta hand skeleton (the OculusXR skeleton; the Meta OpenXR runtime's
 * XR_EXT_hand_tracking joints are that skeleton's joints) move the thumb metacarpal's origin with a
 * trapezium (CMC) joint. The OpenXR bot keeps it at a fixed pivot,
 * so a pose transferred by bone rotations leaves its thumb short of the tracked one and opens
 * tracked pinches. FitOpenXRThumbToDistalFrame refits the thumb to the tracked thumb distal frame;
 * BlendOpenXRFingersIntoPinch then moves the fingers near the thumb by what the thumb still cannot
 * reach. Both are fixed functions of the pose (no state carries across frames).
 */

/**
 * Refits the OpenXR skeleton bot's thumb to a thumb-distal frame: the distal link takes the frame's
 * orientation exactly and comes as close to its origin as the fixed metacarpal pivot and the MCP
 * and IP limits allow. The MCP angle sets the pivot-to-distal distance; the IP angle turns the
 * thumb toward the origin; the metacarpal takes whatever orientation keeps the distal orientation.
 * Of two MCP angles giving that distance, the one nearer the current MCP rotation is used.
 *
 * @param DistalToWrist The target ThumbDistal frame in the OpenXR wrist link's frame (bot units).
 * @param InOutLinkToParent Link-to-parent rotations of every ESuperDexBone_OXR. The
 * ThumbMetacarpal, ThumbProximal and ThumbDistal rotations are replaced.
 * @return false, leaving InOutLinkToParent unchanged, if the bot's thumb is not a spherical pivot
 * on the wrist followed by two revolute joints, or the rotation count is wrong.
 */
SUPERDEXPOSE_API bool FitOpenXRThumbToDistalFrame(
    ESuperDexHandedness Handedness,
    const FTransform& DistalToWrist,
    TArray<FQuat>& InOutLinkToParent);

/**
 * The pad point BlendOpenXRFingersIntoPinch moves on a finger's distal link, in that link's frame:
 * half the fingertip radius (GetFingerTipRadius_OXR) from the tip bone toward the palm (+Y).
 * @param TipInDistal The finger's tip bone in its distal link frame (bot units).
 */
SUPERDEXPOSE_API FVector
GetOpenXRFingerPadInDistal(ESuperDexFinger Finger, const FVector& TipInDistal);

/**
 * The far end of a finger's distal centerline, which BlendOpenXRFingersIntoPinch measures thumb
 * distances to (from the distal link origin): the fingertip surface, one fingertip radius beyond
 * the tip bone along the bone (-Z), in the distal link frame.
 */
SUPERDEXPOSE_API FVector
GetOpenXRFingerEndInDistal(ESuperDexFinger Finger, const FVector& TipInDistal);

/**
 * Parameters of BlendOpenXRFingersIntoPinch. The defaults are the tuned values every pose source
 * uses; tests vary them. Distances are between the tracked thumb and finger distal centerlines
 * (each distal origin to its fingertip surface, GetOpenXRFingerEndInDistal), in bot units (cm).
 */
struct FOpenXRPinchBlend {
  // A finger is fully corrected this close to the thumb.
  double FullWeightDistance = 0.9;
  // A finger is uncorrected this far from the thumb.
  double ZeroWeightDistance = 3.5;
  // Index, middle, ring and little flex allowances in [0, 1]: how far each finger's joints go from
  // the knuckle turn toward the least-squares pad move. The defaults follow which looked right
  // against the real hand in recorded right-hand pinches.
  double FlexAllowance[4] = {1.0, 0.0, 0.0, 1.0};
};

/**
 * Moves each OpenXR finger near the thumb to carry its pad (GetOpenXRFingerPadInDistal) along the
 * thumb distal link's shortfall from the tracked thumb: the OpenXR thumb cannot always reach the
 * tracked distal origin (FitOpenXRThumbToDistalFrame keeps its orientation), so its distal link is
 * offset from the tracked one; moving the finger pads near it by the same offset keeps a tracked
 * pinch closed. Each finger's pad goal is its pad plus Weight * Shortfall, where Weight is a
 * smoothstep of the finger's tracked distal centerline distance to the tracked thumb's, so the
 * correction is continuous and free-space fingers far from the thumb are untouched.
 *
 * Each finger first takes the smallest turn about its knuckle that points the pad at its goal (only
 * the proximal rotation changes). Its joints then go from that turn toward two damped least-squares
 * steps over the proximal (3 DOF), intermediate and distal joints (flexion damped 4x, within the
 * hinge limits), which also follow the pad's motion along the knuckle-to-pad direction, by the
 * finger's flex allowance. The allowance fades to zero when the thumb is nearer the finger's
 * intermediate segment than its distal one: bending the distal joint does not help a contact there.
 *
 * @param TrackedThumbDistal The tracked ThumbDistal origin in the OpenXR wrist link's frame (bot
 * units), as passed to FitOpenXRThumbToDistalFrame.
 * @param InOutLinkToParent Link-to-parent rotations of every ESuperDexBone_OXR after the thumb fit,
 * with the fingers tracked. The finger proximal, intermediate and distal rotations are replaced.
 * @return false, leaving InOutLinkToParent unchanged, if the bot's layout is not the expected one
 * or the rotation count is wrong.
 */
SUPERDEXPOSE_API bool BlendOpenXRFingersIntoPinch(
    ESuperDexHandedness Handedness,
    const FVector& TrackedThumbDistal,
    TArray<FQuat>& InOutLinkToParent,
    const FOpenXRPinchBlend& Blend = FOpenXRPinchBlend());

/**
 * Whether a pose source applies ApplyOpenXRPinchCorrection: its project setting, unless the console
 * variable superdex.OpenXRPinchCorrection.Enable overrides it (-1, the default: follow the setting;
 * 0: off; 1: on) for every pose source at once.
 */
SUPERDEXPOSE_API bool ShouldApplyOpenXRPinchCorrection(bool bProjectSetting);

/**
 * FitOpenXRThumbToDistalFrame, then BlendOpenXRFingersIntoPinch with the default parameters. A
 * fixed function of the pose: no state, filtering or hysteresis across frames.
 *
 * @param TrackedThumbDistalToWrist The tracked ThumbDistal frame in the OpenXR wrist link's frame
 * (bot units, at the bot's nominal scale).
 * @param InOutLinkToParent Link-to-parent rotations of every ESuperDexBone_OXR, with the fingers
 * tracked.
 * @return false, leaving InOutLinkToParent unchanged, if the thumb fit fails.
 */
SUPERDEXPOSE_API bool ApplyOpenXRPinchCorrection(
    ESuperDexHandedness Handedness,
    const FTransform& TrackedThumbDistalToWrist,
    TArray<FQuat>& InOutLinkToParent);
