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

#include "mochi_discretization_components.h"
#include "mochi_shape.h"

#include <mochi_core/utils/dskinning.h>
#include <mochi_core/utils/dtransform.h>
#include <mochi_core/utils/transform_rt.h>

#include <optional>
#include <vector>

namespace mochi {

struct SkeletonReferenceFrames {
  real scale = 1.0_r;
  std::vector<TransformRT> referenceRootFromBone;
  std::vector<TransformRT> initialWorldFromBone;

  MOCHI_STRUCT_BEGIN(mochi::SkeletonReferenceFrames)
  MOCHI_FIELD(scale)
  MOCHI_FIELD(referenceRootFromBone)
  MOCHI_FIELD(initialWorldFromBone)
  MOCHI_STRUCT_END()
};

struct SkinningParams {
  // Parameters that have to do with the implementation of skinning
  SkeletonReferenceFrames referenceFrames = {};

  // Bypass an error that comes from having unused bones. If this is enabled, the bone
  // should be constrained.
  bool allowUnusedBones = false;

  MOCHI_STRUCT_BEGIN(mochi::SkinningParams)
  MOCHI_FIELD(referenceFrames)
  MOCHI_FIELD(allowUnusedBones)
  MOCHI_STRUCT_END()
};

namespace skinning {

// Create the initial skinning pose of a skeleton from input parameters
std::vector<TransformRT> CreateInitialSkinningPose(SkinningParams const& params);

// Create a differentiable skinning parameterization from skeleton layer parameters
DTransformParameterizationCollection CreateSkinningParameterization(SkinningParams const& params);

// Create a differentiable skinning transformation from skinning data and skeleton layer parameters
DSkinningTransform
CreateSkinningTransform(SkinningData const& skinning, SkinningParams const& params, Error& error);

} // namespace skinning
} // namespace mochi
