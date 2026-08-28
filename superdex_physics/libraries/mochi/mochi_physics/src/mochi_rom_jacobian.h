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

#include "mochi_common_components.h"
#include "mochi_soft.h"

#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/utils/dtransform.h>
#include <mochi_core/utils/overload_visitor.h>

#include <utility>

namespace mochi::rom {
struct CRomJacobianTypes {
  using DenseT = RowMatrix<real>;
  using DenseViewT = RowMatrixView<real>;
  using DenseViewConstT = RowMatrixView<real const>;
};

// Used to store the Jacobian of a ROM at the current iteration location
struct CRomJacobian : public CVariant<CRomJacobianTypes::DenseT>,
                      public CRomJacobianTypes,
                      public NoCopy {
  MOCHI_DECLARE_MOVE(CRomJacobian);

  explicit CRomJacobian(DenseT&& dense) : CVariant(std::move(dense)) {}

  int Cols() const;
  void ApplyTranspose(ColumnVectorView<real const> input, ColumnVectorView<real> output) const;
  void Apply(ColumnVectorView<real const> input, ColumnVectorView<real> output) const;

  /**
   * @brief Conjugate the ROM Jacobian with the given input matrix: output = J^T * input * J
   *
   * @warning This method is NOT thread-safe due to internal caching. It cannot be invoked in
   * parallel from different threads on the same CRomJacobian instance.
   */
  template <krylov::Direction kOutDir>
  void Conjugate(
      BlockSparseMatrixView<real const, 3> input,
      MatrixView<real, krylov::kDynamic, krylov::kDynamic, kOutDir> output);

 protected:
  /** @brief Cache temporary matrix to avoid dynamic memory allocation in conjugate operations. */
  DenseT _DJ;
};

struct CIntermediateRomJacobian : public CRomJacobian {
  explicit CIntermediateRomJacobian(DenseT&& dense) : CRomJacobian(std::move(dense)) {}
};

template <krylov::Direction kOutDir>
void mochi::rom::CRomJacobian::Conjugate(
    BlockSparseMatrixView<real const, 3> input,
    MatrixView<real, krylov::kDynamic, krylov::kDynamic, kOutDir> output) {
  Visit(OverloadVisitor{[&](DenseT const& dense) {
    _DJ.Resize(input.Rows(), dense.Cols());
    _DJ = input * dense;

    // Parallelize the dense matrix-matrix product across the contraction direction to exploit that
    // the number of FOM DoFs is much greater than the number of ROM DoFs.
    ParallelMatMatAlongK(dense.Transpose(), _DJ, output);
  }});
}

// Compute the contact Jacobians as colliding actor
void SetupCollidingJacobians(
    ecs::Included<TagRomActor>,
    ecs::Excluded<TagNestedSoftActor>,
    CFemBoundaryDiscretization const& discretization,
    CRootTransform const& transform,
    CDofOffset const& dofOffset,
    CRomJacobian const& jacobianRom,
    CCollJacs<CollRole::Colliding>& outJacobians);

namespace jacobian {
void InitializeOnce(entt::registry& reg);
}

} // namespace mochi::rom
