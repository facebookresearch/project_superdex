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

/*
    A utility file for performing differentiable skinning operations. These utilities are
    used for articulated model reduction.
*/

/*
    A differentiable skinning transform takes the following form:

    y(i) = sum_{k = 0}^N w(k, i) R_{j_{i,k}}(x(i)),            eq. (1)

    where:
        x(i) is the input vector (in reference configuration),
        y(i) is the output skinned vector,
        w(k, i) are the skinning weights. Every vertex has only N nonzero
            weights for nearby bones.
        j_{i,1}, ..., j_{i,N} denote the indices of those bones for which
            vertex i has nonzero weight.
        R_{j} denote the bone transforms.

    /////////////////////////////////////////////////////////////
    //  Input derivatives
    /////////////////////////////////////////////////////////////

    If x is a function of a reduced parameter set z (i.e., x = x(z)), then the derivative of dy/dz
    has the form:

        dy/dz = sum_{k = 0}^N w(k, i) R_{j_{i,k}}(dx/dz)

    To compute the derivative dy/dz from dx/dz is handled with the DTransform function and the
    process is essentially the same as the Transform operation.

    /////////////////////////////////////////////////////////////
    //  Bone derivatives
    /////////////////////////////////////////////////////////////

    Suppose that theta(i) is the vector of bone transform parameters. That is, theta is a block
    vector where each block theta_j(i) holds the parameters of the rigid transform of R_{j_{i,k}}
    (i.e., quaternion and translation). Then the derivative of y with respect to theta is

        dy(i)/dtheta = sum_{k = 0}^N w(k, i) dR_{j_{i, k}}/dtheta (x(i))

    In particular, this means dy/dtheta is a sparse matrix. Moreover, the (i, j) block of dy/dtheta,
    corresponding to the i-th vertex and j-th bone is zero if j is not in the set of skinning bones
    for vertex i. Therefore, we only need to compute the blocks

        dy(i)/dtheta_{j_{i, k}} = w(j_{i, k}, i) dR_{j_{i, k}}/dtheta_j (x(i))

    All other blocks for vertex i will be zero.
*/

#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/utils/dtransform.h>
#include <mochi_core/utils/transform_srt.h>

#include <algorithm>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

namespace mochi {

/*
    Determines the parameterization of a specific transform. i.e., this might be a parameterization
    with respect to a specific pivot (i.e., post and pre transforms are translations). This could
    also be a parameterization with respect to a boneFromRoot transform.
*/
struct DTransformParameterization {
  // The transforms that is applied before the the joint transforms in skinning
  // This will usually be a scaled boneFromRoot transform
  TransformSRT preTransform;

  // The transforms that are applied after the joint transforms in skinning
  // This will usually just be a pure scale to undo hand scaling
  TransformSRT postTransform;
};

struct DTransformParameterizationCollection : std::vector<DTransformParameterization> {
  static DTransformParameterizationCollection FromRootFromBone(
      std::vector<TransformSRT> const& referenceRootFromBone,
      real scale);
  static DTransformParameterizationCollection FromRootFromBone(
      std::vector<TransformRT> const& referenceRootFromBone,
      real scale);
};

constexpr int kDSkinningDofsPerVertex = 3;
/*
    A pair containing a vertex and a bone together with the skinning weight of that pair.
*/
template <typename weight_t>
struct VertexBonePair {
  int boneId;
  int vertexId;
  weight_t weight;
};

/*
    A table that contains all of the nonzero vertex-bone pairs of a skinned model, but stored in
    order by bone so that the user can quickly query all the vertices that have been skinned to a
    particular bone.
*/
struct SkinningWeightsByBone {
 private:
  SkinningWeightsByBone() = default;

 public:
  // A list of vertex bone pairs, stored in sorted order by bone id.
  std::vector<VertexBonePair<real>> boneVertexPairsByBone;
  /*
      Denotes the boundaries of the particular ranges of the above bone vertex pairs,
      corresponding to a single bone. i.e., the range corresponding to bone i is
      boneRanges[i] to boneRanges[i + 1] (range is inclusive-exclusive).
  */
  std::vector<long long> boneRanges;
  int vertexCount = 0;

  using iterator_t = typename std::vector<VertexBonePair<real>>::iterator;
  using const_iterator_t = typename std::vector<VertexBonePair<real>>::const_iterator;

  int GetVertexCount() const {
    return vertexCount;
  }

  int GetBoneCount() const {
    return isize(boneRanges) - 1;
  }

  iterator_t BeginBone(int boneId) {
    MOCHI_ASSERT(boneId < GetBoneCount(), "Index out of range!");
    return boneVertexPairsByBone.begin() + boneRanges[boneId];
  }

  iterator_t EndBone(int boneId) {
    MOCHI_ASSERT(boneId < GetBoneCount(), "Index out of range!");
    return boneVertexPairsByBone.begin() + boneRanges[boneId + 1];
  }

  const_iterator_t BeginBone(int boneId) const {
    MOCHI_ASSERT(boneId < GetBoneCount(), "Index out of range!");
    return boneVertexPairsByBone.begin() + boneRanges[boneId];
  }

  const_iterator_t EndBone(int boneId) const {
    MOCHI_ASSERT(boneId < GetBoneCount(), "Index out of range!");
    return boneVertexPairsByBone.begin() + boneRanges[boneId + 1];
  }

  // Returns true if there are any bones with no paired vertices.
  bool HasUnusedBones() const;

  /*
    Constructs a skinning table from an array of skinning indices and weights.
    The input is assumed to be grouped into groups of size weightsPerNode such that
    weights of vertex i are given by skinWeight[i * weightsPerNode] to
    skinWeight[(i + 1) * weightsPerNode] (inclusive-exclusive). Likewise for vertices.
  */
  SkinningWeightsByBone(
      Span<int const> skinIdx,
      Span<real const> skinWeight,
      int weightsPerNode,
      int numBones);
};

// A differentiable skinning transform
// See notes at top of file for complete explanation.
struct DSkinningTransform {
  using VertexBones = std::vector<std::pair<int, real>>;

  std::vector<VertexBones> perVertexBones = {};
  DTransformParameterizationCollection transformParameterizations;
  int boneCount = 0;
  int totalPairs = 0;

  static std::vector<VertexBones> BuildPerVertexBones(SkinningWeightsByBone const& weights);

  DSkinningTransform() = default;
  explicit DSkinningTransform(
      SkinningWeightsByBone const& weights,
      std::optional<DTransformParameterizationCollection> transforms = std::nullopt)
      : perVertexBones(BuildPerVertexBones(weights)), boneCount(weights.GetBoneCount()) {
    if (transforms) {
      MOCHI_ASSERT(isize(*transforms) == boneCount);
      transformParameterizations = *transforms;
    } else {
      transformParameterizations.resize(boneCount);
    }
    totalPairs = 0;
    for (auto const& pairs : perVertexBones) {
      totalPairs += isize(pairs);
    }
  }
  MOCHI_DECLARE_MOVE_ONLY(DSkinningTransform);

  // Compute the forward map
  inline void Transform(
      Span<TransformRT const> worldFromBoneTransforms,
      ColumnVectorView<real const> input,
      ColumnVectorView<real> output,
      Span<int const> activeVertices = {}) const;

  // Compute the derivative of the forward map with respect to the inputs
  inline void DTransform(
      Span<TransformRT const> worldFromBoneTransforms,
      RowMatrixView<real const, krylov::kDynamic, RigidSize::kDim> input,
      RowMatrixView<real, krylov::kDynamic, RigidSize::kDim> output,
      Span<int const> activeVertices = {}) const;

  // Compute the derivative of the forward map with respect to the bone parameters.
  // Use CreateDBones() to create storage for the output parameter.
  inline void DTransformDBones(
      Span<TransformRT const> worldFromBoneTransforms,
      ColumnVectorView<real const, krylov::kDynamic> input,
      SparseMatrix<real>& outputDBones,
      Span<int const> activeVertices = {}) const;

  // Creates storage for DTransformDBones
  inline SparseMatrix<real> CreateDBones() const;

  TransformSRT const& GetBonePreTransform(int boneId) const {
    MOCHI_ASSERT_VERBOSE(boneId >= 0 && boneId < boneCount, "Invalid bone index.");
    return transformParameterizations[boneId].preTransform;
  }
  TransformSRT const& GetBonePostTransform(int boneId) const {
    MOCHI_ASSERT_VERBOSE(boneId >= 0 && boneId < boneCount, "Invalid bone index.");
    return transformParameterizations[boneId].postTransform;
  }
  DTransformParameterizationCollection const& GetParameterizations() const {
    return transformParameterizations;
  }

  int GetNumVertices() const {
    return isize(perVertexBones);
  }
  int GetBoneCount() const {
    return boneCount;
  }
};

} // namespace mochi

#include "dskinning_inl.h"
