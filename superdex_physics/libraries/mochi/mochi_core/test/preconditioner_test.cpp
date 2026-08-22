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

#include <mochi_core/linear_algebra/krylov/amg/amg_prec.h>
#include <mochi_core/linear_algebra/krylov/block_jacobi_prec.h>
#include <mochi_core/linear_algebra/krylov/block_ssor_prec.h>
#include <mochi_core/linear_algebra/krylov/colored_ssor_prec.h>
#include <mochi_core/linear_algebra/krylov/identity_prec.h>
#include <mochi_core/linear_algebra/krylov/incomplete_cholesky_prec.h>
#include <mochi_core/linear_algebra/krylov/ldlt_prec.h>
#include <mochi_core/linear_algebra/krylov/lu_prec.h>
#include <mochi_core/linear_algebra/krylov/preconditioner_utils.h>
#include <mochi_core/linear_algebra/krylov/relaxed_ilu_prec.h>
#include <mochi_core/linear_algebra/krylov/ssor_prec.h>
#include <mochi_core/linear_algebra/krylov/sym_inverse_prec.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/solvers/island_operators.h>

#include <gtest/gtest.h>

using namespace mochi;
using namespace mochi::krylov;

// Use real instead of float or double to reduce build time. Both are checked by CI.
using Scalar = real;

TEST(Preconditioner, Type) {
  using Dense = Matrix<Scalar>;
  using Sp = SparseMatrix<Scalar>;
  using BSp1 = BlockSparseMatrix<Scalar, 1>;
  using BSp3 = BlockSparseMatrix<Scalar, 3>;
  static_assert(IdentityPrec<Scalar>::kType == PreconditionerType::None);
  static_assert(JacobiPrec<Scalar>::kType == PreconditionerType::Jacobi);
  static_assert(BlockJacobiPrec<Scalar, 1>::kType == PreconditionerType::Jacobi);
  static_assert(BlockJacobiPrec<Scalar, 3>::kType == PreconditionerType::BlockJacobi);
  static_assert(SSORPrec<Scalar, Dense>::kType == PreconditionerType::SSOR);
  static_assert(SSORPrec<Scalar, Sp>::kType == PreconditionerType::SSOR);
  static_assert(SSORPrec<Scalar, BSp1>::kType == PreconditionerType::SSOR);
  static_assert(SSORPrec<Scalar, BSp3>::kType == PreconditionerType::SSOR);
  static_assert(BlockSSORPrec<Scalar, 1, Dense>::kType == PreconditionerType::BlockSSOR);
  static_assert(BlockSSORPrec<Scalar, 1, Sp>::kType == PreconditionerType::BlockSSOR);
  static_assert(BlockSSORPrec<Scalar, 1, BSp1>::kType == PreconditionerType::BlockSSOR);
  static_assert(BlockSSORPrec<Scalar, 1, BSp3>::kType == PreconditionerType::BlockSSOR);
  static_assert(BlockSSORPrec<Scalar, 3, Dense>::kType == PreconditionerType::BlockSSOR);
  static_assert(BlockSSORPrec<Scalar, 3, Sp>::kType == PreconditionerType::BlockSSOR);
  static_assert(BlockSSORPrec<Scalar, 3, BSp1>::kType == PreconditionerType::BlockSSOR);
  static_assert(BlockSSORPrec<Scalar, 3, BSp3>::kType == PreconditionerType::BlockSSOR);
  static_assert(ColoredSSORPrec<Dense>::kType == PreconditionerType::ColoredSSOR);
  static_assert(ColoredSSORPrec<Sp>::kType == PreconditionerType::ColoredSSOR);
  static_assert(ColoredSSORPrec<BSp1>::kType == PreconditionerType::ColoredSSOR);
  static_assert(ColoredSSORPrec<BSp3>::kType == PreconditionerType::ColoredSSOR);
  static_assert(AMGPrec<Scalar, 1>::kType == PreconditionerType::AMG);
  static_assert(AMGPrec<Scalar, 3>::kType == PreconditionerType::AMG);
  static_assert(RelaxedILUPrec<Dense>::kType == PreconditionerType::ILU0);
  static_assert(RelaxedILUPrec<Sp>::kType == PreconditionerType::ILU0);
  static_assert(RelaxedILUPrec<BSp1>::kType == PreconditionerType::ILU0);
  static_assert(RelaxedILUPrec<BSp3>::kType == PreconditionerType::ILU0);
  static_assert(IncompleteCholeskyPrec<Dense>::kType == PreconditionerType::IC0);
  static_assert(IncompleteCholeskyPrec<Sp>::kType == PreconditionerType::IC0);
  static_assert(IncompleteCholeskyPrec<BSp1>::kType == PreconditionerType::IC0);
  static_assert(IncompleteCholeskyPrec<BSp3>::kType == PreconditionerType::IC0);
  static_assert(LUPrec<Scalar>::kType == PreconditionerType::LU);
  static_assert(LDLtPrec<Scalar>::kType == PreconditionerType::LDLT);
  static_assert(SymInversePrec<Scalar>::kType == PreconditionerType::SymInverse);
  static_assert(PerActorPrec<Scalar>::kType == PreconditionerType::PerActor);
}

TEST(PreconditionerUtils, StoresInputView) {
  static_assert(!mochi::details::PreconditionerStoresInputView(PreconditionerType::None));
  static_assert(!mochi::details::PreconditionerStoresInputView(PreconditionerType::Jacobi));
  static_assert(!mochi::details::PreconditionerStoresInputView(PreconditionerType::BlockJacobi));
  static_assert(mochi::details::PreconditionerStoresInputView(PreconditionerType::SSOR));
  static_assert(mochi::details::PreconditionerStoresInputView(PreconditionerType::BlockSSOR));
  static_assert(!mochi::details::PreconditionerStoresInputView(PreconditionerType::ColoredSSOR));
  static_assert(!mochi::details::PreconditionerStoresInputView(PreconditionerType::ILU0));
  static_assert(!mochi::details::PreconditionerStoresInputView(PreconditionerType::IC0));
  static_assert(!mochi::details::PreconditionerStoresInputView(PreconditionerType::SymInverse));
  static_assert(!mochi::details::PreconditionerStoresInputView(PreconditionerType::LDLT));
  static_assert(!mochi::details::PreconditionerStoresInputView(PreconditionerType::LU));
  static_assert(mochi::details::PreconditionerStoresInputView(PreconditionerType::AMG));
  static_assert(!mochi::details::PreconditionerStoresInputView(PreconditionerType::PerActor));
  static_assert(
      static_cast<int>(PreconditionerType::Count) == 13,
      "Please update unit test if PreconditionerType enumerator changes");
}

TEST(PreconditionerUtils, IsReusable) {
  static_assert(mochi::details::PreconditionerIsReusable(PreconditionerType::None));
  static_assert(mochi::details::PreconditionerIsReusable(PreconditionerType::Jacobi));
  static_assert(mochi::details::PreconditionerIsReusable(PreconditionerType::BlockJacobi));
  static_assert(!mochi::details::PreconditionerIsReusable(PreconditionerType::SSOR));
  static_assert(!mochi::details::PreconditionerIsReusable(PreconditionerType::BlockSSOR));
  static_assert(mochi::details::PreconditionerIsReusable(PreconditionerType::ColoredSSOR));
  static_assert(mochi::details::PreconditionerIsReusable(PreconditionerType::ILU0));
  static_assert(mochi::details::PreconditionerIsReusable(PreconditionerType::IC0));
  static_assert(mochi::details::PreconditionerIsReusable(PreconditionerType::SymInverse));
  static_assert(mochi::details::PreconditionerIsReusable(PreconditionerType::LDLT));
  static_assert(mochi::details::PreconditionerIsReusable(PreconditionerType::LU));
  static_assert(!mochi::details::PreconditionerIsReusable(PreconditionerType::AMG));
  static_assert(mochi::details::PreconditionerIsReusable(PreconditionerType::PerActor));
  static_assert(
      static_cast<int>(PreconditionerType::Count) == 13,
      "Please update unit test if PreconditionerType enumerator changes");
}
