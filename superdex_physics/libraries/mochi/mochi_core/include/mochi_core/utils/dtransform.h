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

#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/lie.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/rigid_body_utils.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/transform_rt.h>
#include <mochi_core/utils/transform_srt.h>
#include <mochi_core/utils/vmatrix.h>

namespace mochi {

/*
  Transform an input array of Real3 as a batch operation. The input and output arrays should be
  3*n dimensional vectors where n is the batch size.

  'nodesSubset' is an optional parameter specifying a subset of nodes to transform:
  - If nodesSubset is empty, the transform is applied to all entries in the input array.
  - If non-empty, the transform is applied to DoFs corresponding to the nodes in nodesSubset.
  When provided, nodesSubset must contain node indices that are consistent with the input and
  output vectors. For example, if the input is a 3*n vector of displacements, nodesSubset must
  specify a valid subset of node indices whose displacements are represented in the input.
  The entries in nodesSubset must be unique.
*/
inline void TransformBatch(
    TransformRT const& transform,
    ColumnVectorView<real const> input,
    ColumnVectorView<real> output,
    TransformSRT const& preTransform = {},
    TransformSRT const& postTransform = {},
    Span<int const> nodesSubset = {}) {
  MOCHI_ASSERT_VERBOSE(input.Rows() == output.Rows());
  MOCHI_ASSERT_VERBOSE(input.Rows() % RigidSize::kDim == 0);
  MOCHI_ASSERT_VERBOSE(IsUnique(nodesSubset));
  if (input.Rows() == 0)
    MOCHI_UNLIKELY {
      return;
    }

  // Full transform (postTransform * transform * preTransform)^T
  auto preTransformT = ToVMatrix4x4Transpose(preTransform);
  auto transformT = ToVMatrix4x4Transpose(transform);
  auto postTransformT = ToVMatrix4x4Transpose(postTransform);
  auto transformPrePostT = Dot4x4(Dot4x4(preTransformT, transformT), postTransformT);

  if (nodesSubset) {
    // FRIZZI: The indices in nodesSubset are not necessarily sorted, so we must use a partial SIMD
    // loads/stores. This could be improved by either making sure the nodesSubset is sorted and
    // tailor the loop accordingly similarly to the else branch below, or checking within the loop
    // if the node ID is the last node. Need to profile to see if it's worth it.
    for (int nodeIndex : nodesSubset) {
      int const rOffset = nodeIndex * RigidSize::kDim;
      MOCHI_ASSERT_VERBOSE(rOffset >= 0 && (rOffset + RigidSize::kDim <= input.size()));
      auto inPoint = Load<RigidSize::kDim, Vec4r>(&input(rOffset));
      Store<RigidSize::kDim>(
          &output(rOffset), DotVecMat4x4(ToSimdPoint(inPoint), transformPrePostT));
    }
  } else {
    // Use partial SIMD load and store for the last point, and full SIMD loads and stores (faster)
    // for all other points.
    int i = 0;
    for (; i + RigidSize::kDim < input.Rows(); i += RigidSize::kDim) {
      auto inPoint = Load<Vec4r>(&input(i));
      Store(&output(i), DotVecMat4x4(ToSimdPoint(inPoint), transformPrePostT));
    }
    auto inPoint = Load<RigidSize::kDim, Vec4r>(&input(i));
    Store<RigidSize::kDim>(&output(i), DotVecMat4x4(ToSimdPoint(inPoint), transformPrePostT));
  }
}

MOCHI_FORCE_INLINE void TransformBatch(
    TransformRT const& transform,
    ColumnVectorView<real const> input,
    ColumnVectorView<real> output,
    Real3 const& pivot,
    Span<int const> nodesSubset = {}) {
  TransformBatch(
      transform,
      input,
      output,
      TranslateSRT(ToSimd(-pivot)),
      TranslateSRT(ToSimd(pivot)),
      nodesSubset);
}

/*
  Applies the derivative of the transform (i.e., the rotation) to an input. The input
  is another derivative matrix. This function therefore allows the user to apply
  the chain rule without having to store the full batch derivative in memory.

  This function is for computing the derivative of the composition x = R(g(z)) where R is the
  batch transform operation applied to g(z). This function outputs the matrix dx/dy given
  the matrix dg/dz.

  [input] is dg/dz. Should be a 3*n x m dimensional derivate matrix where n is the
      batch size and m is the size of z.
  [output] is dx/dz. Should also be a 3*n x m dimensional matrix.

  'nodesSubset' is an optional parameter specifying a subset of nodes to transform.
  See TransformBatch for details.
*/
template <
    int kColsIn,
    int kColsOut,
    krylov::Direction kMajorIn,
    krylov::Direction kMajorOut,
    int kLeadingDimIn,
    int kLeadingDimOut>
void DTransformBatch(
    TransformRT const& transform,
    MatrixView<real const, krylov::kDynamic, kColsIn, kMajorIn, kLeadingDimIn> input,
    MatrixView<real, krylov::kDynamic, kColsOut, kMajorOut, kLeadingDimOut> output,
    TransformSRT const& preTransform = {},
    TransformSRT const& postTransform = {},
    Span<int const> nodesSubset = {}) {
  MOCHI_ASSERT_VERBOSE(output.Cols() == input.Cols());
  MOCHI_ASSERT_VERBOSE(output.Rows() == input.Rows());
  MOCHI_ASSERT_VERBOSE(input.Rows() % RigidSize::kDim == 0);
  MOCHI_ASSERT_VERBOSE(IsUnique(nodesSubset));

  auto preJ = preTransform.Jacobian3x3();
  auto rotMatrix = GetRotationMatrix(transform);
  auto postJ = postTransform.Jacobian3x3();

  // The full transform is stored with the same storage direction as the input to improve
  // performance of the batch of matrix-matrix products.
  Matrix<real, RigidSize::kDim, RigidSize::kDim, kMajorIn> fullTransform =
      AsConstView<real, RigidSize::kDim, RigidSize::kDim>(Dot(postJ, Dot(rotMatrix, preJ)));

  // TODO: If the input and output are column-major, the batch of matrix-matrix products could be
  // performed in a single matrix-matrix product.

  // If provided, nodesSubset is a list of node indices to transform so the loop must be modified to
  // only transform the DoFs corresponding to the nodes in nodesSubset.
  int const loopCount = nodesSubset.empty() ? input.Rows() : isize(nodesSubset);
  int const loopStride = nodesSubset.empty() ? RigidSize::kDim : 1;
  for (int i = 0; i < loopCount; i += loopStride) {
    int const rOffset = nodesSubset.empty() ? i : (nodesSubset[i] * RigidSize::kDim);
    MOCHI_ASSERT_VERBOSE(rOffset >= 0 && (rOffset + RigidSize::kDim <= input.Rows()));
    auto outBlock = output.template MiddleRows<RigidSize::kDim>(rOffset, RigidSize::kDim);
    auto inBlock = input.template MiddleRows<RigidSize::kDim>(rOffset, RigidSize::kDim);
    outBlock = fullTransform * inBlock;
  }
}

/*
  Applies the derivative of the transform (i.e., the rotation) with respect to the
  transform rotation and translation.

  This function is for computing the derivative of the composition x = R(y, p) where R is the
  batch transform operation applied to y and p are the rotation and translation of the rigid
  transform.

  [input] is y. Should be a 3*n dimensional vector where n is the batch size.
  [output] is dx/dp. Should also be a 3*n x kRawSize dimensional matrix. The kRawSize degrees of
      freedom have the same ordering used by the function `TransformToRawPose`.

  'nodesSubset' is an optional parameter specifying a subset of nodes to transform.
  See TransformBatch for details.
*/
template <int kRows, int kCols, krylov::Direction kMajor, int kLeadingDim>
void DTransformDParametersBatch(
    TransformRT const& transform,
    ColumnVectorView<real const> input,
    MatrixView<real, kRows, kCols, kMajor, kLeadingDim> output,
    TransformSRT const& preTransform = {},
    TransformSRT const& postTransform = {},
    Span<int const> nodesSubset = {}) {
  MOCHI_ASSERT_VERBOSE(input.Rows() == output.Rows() && input.Rows() % RigidSize::kDim == 0);
  MOCHI_ASSERT_VERBOSE(output.Cols() == RigidSize::kDAll);
  MOCHI_ASSERT_VERBOSE(IsUnique(nodesSubset));
  VMatrix3x3r postMat = ToSimdMatrix(postTransform.Jacobian3x3()); // postMat = spost * Rpost
  VMatrix4x4r preMatT = Dot4x4(
      ToVMatrix4x4Transpose(preTransform),
      ToVMatrix4x4Transpose(
          TransformRT{transform.GetRotation()})); // preMatT = (R * (spre Rpre, tpre))^T

  // If provided, nodesSubset is a list of node indices to transform so the loop must be modified to
  // only transform the DoFs corresponding to the nodes in nodesSubset.
  // Important: The indices in nodesSubset are not necessarily sorted, so we must use a partial SIMD
  // loads.
  int const loopCount = nodesSubset.empty() ? input.Rows() : isize(nodesSubset);
  int const loopStride = nodesSubset.empty() ? RigidSize::kDim : 1;
  for (int i = 0; i < loopCount; i += loopStride) {
    int const rOffset = nodesSubset.empty() ? i : (nodesSubset[i] * RigidSize::kDim);
    MOCHI_ASSERT_VERBOSE(rOffset >= 0 && (rOffset + RigidSize::kDim <= input.size()));
    auto inPoint = ToSimdPoint(Load<RigidSize::kDim, Vec4r>(&input(rOffset)));

    // Translation parameters derivatives are just the post-transform.
    auto outPointDerivative = output.template Block<RigidSize::kDim, RigidSize::kDAll>(
        rOffset, 0, RigidSize::kDim, RigidSize::kDAll);
    outPointDerivative.template LeftCols<RigidSize::kDTrans>(RigidSize::kDTrans) =
        AsMatrixView(postMat);

    // Rotation parameter derivatives require Lie derivatives.
    outPointDerivative.template MiddleCols<RigidSize::kDRot>(RigidSize::kDTrans, RigidSize::kDRot) =
        AsMatrixView(lie::DMultMatRotVecDRot(postMat, DotVecMat4x4(inPoint, preMatT)));
  }
}
template <int kRows, int kCols, krylov::Direction kMajor, int kLeadingDim>
void DTransformDParametersBatch(
    TransformRT const& transform,
    ColumnVectorView<real const> input,
    MatrixView<real, kRows, kCols, kMajor, kLeadingDim> output,
    Real3 const& pivot,
    Span<int const> nodesSubset = {}) {
  DTransformDParametersBatch(
      transform,
      input,
      output,
      TranslateSRT(ToSimd(-pivot)),
      TranslateSRT(ToSimd(pivot)),
      nodesSubset);
}

} // namespace mochi
