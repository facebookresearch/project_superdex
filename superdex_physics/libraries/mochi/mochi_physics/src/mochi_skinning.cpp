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

#include "mochi_skinning.h"

#include <algorithm>
#include <iterator>
#include <vector>

using namespace mochi;
using namespace mochi::skinning;

std::vector<TransformRT> mochi::skinning::CreateInitialSkinningPose(SkinningParams const& params) {
  std::vector<TransformRT> result;
  auto const& init = params.referenceFrames.initialWorldFromBone;
  std::transform(init.begin(), init.end(), std::back_inserter(result), [](auto const& t) {
    return TransformRT(t);
  });
  return result;
}

DTransformParameterizationCollection mochi::skinning::CreateSkinningParameterization(
    SkinningParams const& params) {
  auto const& referenceFrames = params.referenceFrames;
  // Reference frames provided
  MOCHI_ASSERT(
      referenceFrames.initialWorldFromBone.size() == referenceFrames.referenceRootFromBone.size());

  // Parameterize using the parameterization expected by hand tracking
  return DTransformParameterizationCollection::FromRootFromBone(
      referenceFrames.referenceRootFromBone, referenceFrames.scale);
}

DSkinningTransform mochi::skinning::CreateSkinningTransform(
    SkinningData const& skinning,
    SkinningParams const& params,
    Error& error) {
  // If specified, use all bones in the parameterization
  auto numBones = params.referenceFrames.initialWorldFromBone.size();
  SkinningWeightsByBone fullWeights(
      MakeSpan(skinning.indices), MakeSpan(skinning.weights), skinning.weightsPerNode, numBones);

  if (!params.allowUnusedBones) {
    MOCHI_ERROR_IF(fullWeights.HasUnusedBones(), error, "Mesh skeleton has unused bones.");
    MOCHI_ERROR_RETURN(error, DSkinningTransform{});
  }

  MOCHI_ERROR_IF(
      fullWeights.GetBoneCount() != params.referenceFrames.referenceRootFromBone.size(),
      error,
      "referenceFrames bone count doesn't match skinning data bone count.");
  MOCHI_ERROR_RETURN(error, DSkinningTransform{});

  return DSkinningTransform(fullWeights, CreateSkinningParameterization(params));
}
