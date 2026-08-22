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

#include <mochi_core/rom/rom.h>
#include <mochi_core/utils/dtransform.h>

#include <optional>

namespace mochi::rom::pivoted {

/*
    Used to convert actor state to a raw pose vector.
    This function extends an existing ToRawFunc function by considering the
    first chunk of the vector as pose corresponding to the pivot transform.
    The baseFunc function (if not nullptr) is called on the remains of the vector.
*/
inline void ToRawPose(
    ToRawFunc const& baseFunc,
    TransformRT const& handleTransform,
    ColumnVectorView<real> outRawPose) {
  // First RigidSize::kAll pose elements are rigid transform, rest are amplitudes
  TransformToRawPose(handleTransform, outRawPose.TopRows<RigidSize::kAll>(RigidSize::kAll));

  if (baseFunc) {
    auto remainingPose = outRawPose.BottomRows(outRawPose.Rows() - RigidSize::kAll);
    baseFunc(remainingPose);
  }
}

/*
    Inverse of ToRawPose.
*/
inline void FromRawPose(
    ColumnVectorView<real const> rawPose,
    FromRawFunc const& baseFunc,
    TransformRT& outHandleTransform) {
  // First RigidSize::kAll pose elements are rigid transform, rest are amplitudes
  outHandleTransform = TransformFromRawPose(rawPose.TopRows<RigidSize::kAll>(RigidSize::kAll));

  if (baseFunc) {
    auto remainingPose = rawPose.BottomRows(rawPose.Rows() - RigidSize::kAll);
    baseFunc(remainingPose);
  }
}

/*
    Convert a pose (using quaternion) to dofs (using rotation vector).
*/
inline void ConvertPoseToDofs(Span<real const> pose, Span<real> outDofs) {
  int const numAmplitudes = isize(pose) - RigidSize::kAll;
  MOCHI_ASSERT_VERBOSE(isize(outDofs) - RigidSize::kDAll == numAmplitudes, "Invalid sizes");
  // Convert the rigid transform layer
  auto transform =
      TransformFromRawPose(AsConstView(pose).TopRows<RigidSize::kAll>(RigidSize::kAll));
  TransformToRawDofs(transform, AsView(outDofs).TopRows<RigidSize::kDAll>(RigidSize::kDAll));
  // Copy the ROM amplitudes
  AsView(outDofs).BottomRows(numAmplitudes) = AsConstView(pose).BottomRows(numAmplitudes);
}

/*
    Used to convert reference degrees of freedom plus an increment in local tangent space to actor
   state. This function extends an existing FromIncrement function by considering the first chunk of
   each vector as dofs corresponding to the pivot transform. The baseFunc function (if not nullptr)
   is called on the remains of the vectors.
*/
inline void FromIncrement(
    ColumnVectorView<real const> reference,
    ColumnVectorView<real const> increment,
    FromIncrementFunc const& baseFunc,
    TransformRT& outHandleTransform) {
  static_assert(RigidSize::kTrans == RigidSize::kDTrans, "Revisit rigid sizes");
  int constexpr kRotOffset = RigidSize::kTrans;

  // First degrees of freedom are rigid transform, rest are amplitudes
  auto posRef = Load<Vec4r>(reference.data());
  auto rotRef = Quaternion(Load<RigidSize::kRot, Vec4r>(&reference[kRotOffset]));
  auto posInc = Load<Vec4r>(increment.data());
  auto rotInc =
      Quaternion::FromRotationVector(Load<RigidSize::kDRot, Vec4r>(&increment[kRotOffset]));
  // Normalize the new quaternion to avoid drift
  outHandleTransform = TransformRT(Normalize(rotInc * rotRef), posRef + posInc);

  if (baseFunc) {
    auto remainingRef = reference.BottomRows(reference.Rows() - RigidSize::kAll);
    auto remainingInc = increment.BottomRows(increment.Rows() - RigidSize::kDAll);
    baseFunc(remainingRef, remainingInc);
  }
}

/*
    Computes the jacobian of the full ROM model after the pivot transform layer is applied.
    Requires the input positions to the transform layer and the jacobian of the base ROM.
*/
template <krylov::Direction kMajorDirIn, krylov::Direction kMajorDirOut>
void JacobianFromPositions(
    MatrixView<real const, krylov::kDynamic, krylov::kDynamic, kMajorDirIn> const& inJacobian,
    TransformRT const& handleTransform,
    Real3 const& meshPivot,
    ColumnVectorView<real const> inPositions,
    MatrixView<real, krylov::kDynamic, krylov::kDynamic, kMajorDirOut> outJacobian,
    Span<int const> nodeSubset) {
  // First RigidSize::kAll columns of jacobian are reserved for rigid parameters, the rest
  // are amplitudes.
  auto dOutputDRigid = outJacobian.template LeftCols<RigidSize::kDAll>(RigidSize::kDAll);

  // Get derivative of the output with respect to rigid parameters
  DTransformDParametersBatch(handleTransform, inPositions, dOutputDRigid, meshPivot, nodeSubset);

  // Get derivative of output with respect to the amplitudes
  auto dOutputDAmplitudes = outJacobian.RightCols(outJacobian.Cols() - RigidSize::kDAll);
  DTransformBatch(
      handleTransform,
      inJacobian,
      dOutputDAmplitudes,
      /*preTransform*/ {},
      /*postTransform*/ {},
      nodeSubset);
}

} // namespace mochi::rom::pivoted
