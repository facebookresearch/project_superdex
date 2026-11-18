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

#include "MochiArticulationBase.h"

bool UMochiArticulationBase::ComputeLinkToParentTransforms(
    const TArray<double>& JointPositions,
    TArray<FTransform>& OutLinkTransforms,
    double AngleScale,
    double LengthScale,
    int FinalLinkIndex) const {
  return ComputeLinkTransformsInternal(
      JointPositions, OutLinkTransforms, false, AngleScale, LengthScale, FinalLinkIndex);
}

bool UMochiArticulationBase::ComputeLinkToArticulationRootTransforms(
    const TArray<double>& JointPositions,
    TArray<FTransform>& OutLinkTransforms,
    double AngleScale,
    double LengthScale,
    int FinalLinkIndex) const {
  return ComputeLinkTransformsInternal(
      JointPositions, OutLinkTransforms, true, AngleScale, LengthScale, FinalLinkIndex);
}

bool UMochiArticulationBase::ComputeJointPositionsFromLinkToParentTransforms(
    const TArray<FTransform>& LinkTransforms,
    TArray<double>& OutJointPositions,
    double AngleScale,
    double LengthScale) const {
  return ComputeJointPositionsInternal(
      LinkTransforms, OutJointPositions, false, AngleScale, LengthScale);
}

bool UMochiArticulationBase::ComputeJointPositionsFromLinkToArticulationRootTransforms(
    const TArray<FTransform>& LinkTransforms,
    TArray<double>& OutJointPositions,
    double AngleScale,
    double LengthScale) const {
  return ComputeJointPositionsInternal(
      LinkTransforms, OutJointPositions, true, AngleScale, LengthScale);
}

bool UMochiArticulationBase::ComputeVisualBoundsFromLinkToParentTransforms(
    const TArray<FTransform>& LinksToParent,
    FBoxSphereBounds& OutBounds) const {
  return ComputeVisualBoundsInternal(LinksToParent, false, OutBounds);
}

bool UMochiArticulationBase::ComputeVisualBoundsFromLinkToArticulationRootTransforms(
    const TArray<FTransform>& LinksToArticulationRoot,
    FBoxSphereBounds& OutBounds) const {
  return ComputeVisualBoundsInternal(LinksToArticulationRoot, true, OutBounds);
}

bool UMochiArticulationBase::ComputeVisualBoundsFromJointPositions(
    const TArray<double>& JointPositions,
    FBoxSphereBounds& OutBounds,
    double AngleScale,
    double LengthScale) const {
  thread_local TArray<FTransform> sLinksToRoot;
  if (ComputeLinkToArticulationRootTransforms(
          JointPositions, sLinksToRoot, AngleScale, LengthScale)) {
    return ComputeVisualBoundsFromLinkToArticulationRootTransforms(sLinksToRoot, OutBounds);
  }
  return false;
}
