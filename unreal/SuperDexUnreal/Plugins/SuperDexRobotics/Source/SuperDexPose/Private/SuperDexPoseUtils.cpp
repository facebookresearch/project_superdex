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

#include "SuperDexPoseUtils.h"

FTransform USuperDexPoseUtilsFunctionLibrary::GetRelativeTransform_DescendentToAncestor(
    USceneComponent* ancestorComp,
    USceneComponent* descendentComp) {
  if (!IsValid(descendentComp)) {
    return FTransform::Identity;
  }
  USceneComponent* parent = descendentComp->GetAttachParent();
  if (parent != nullptr) {
    if (parent != ancestorComp) {
      FTransform compToParent = descendentComp->GetRelativeTransform();
      FTransform descendentToAncestor =
          compToParent * GetRelativeTransform_DescendentToAncestor(ancestorComp, parent);
      return descendentToAncestor;
    }
  }
  // GetRelativeTransform() is inconsistent if called on the root component, so return identity if
  // there is no parent
  return FTransform::Identity;
}

FTransform USuperDexPoseUtilsFunctionLibrary::GetRelativeTransform_CompToRoot(
    USceneComponent* comp) {
  if (!IsValid(comp)) {
    return FTransform::Identity;
  }
  AActor* owner = comp->GetOwner();
  if (!IsValid(owner)) {
    return FTransform::Identity;
  }
  return GetRelativeTransform_DescendentToAncestor(owner->GetRootComponent(), comp);
}
