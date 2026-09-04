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

#include <mochi_core/linear_algebra/krylov/pcg.h>
#include <mochi_core/linear_algebra/krylov/recycling_bin.h>
#include <mochi_core/linear_algebra/krylov/stopping_criterion.h>
#include <mochi_core/linear_algebra/krylov/tools/tensor_functions.h>
#include <mochi_core/linear_algebra/krylov/tools/tensor_traits.h>
#include <mochi_core/linear_algebra/ldlt.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/solvers/krylov_solver.h>
#include <mochi_core/utils/basic_utils.h>

#include <type_traits>

namespace mochi::krylov {

/** @brief Solve a linear system of equations using an augmented preconditioned CG method.
 *
 * @tparam MatrixType Type of the matrix.
 * @tparam RhsType Type of the right-hand side vector.
 * @tparam SolType Type of the solution vector.
 * @tparam PrecType Type of the preconditioner.
 * @tparam RecyclingStatusType Type of the recycling status.
 * @tparam Dot Type of the dot operation object/functor.
 * @tparam StopCriterion Type of the stop criteria checker.
 * @tparam VectorFactory Type of the vector factory.
 *
 * @param[in] A The matrix of the linear system.
 * @param[in] b The right-hand side vector of \f$ A x = b\f$.
 * @param[in,out] x Vector containing the initial guess at input and the solution at output.
 * @param[in] prec The preconditioner application functor.
 * @param[in] maxIter Maximum number of iterations.
 * @param[in,out] stopCriterion A functor called every iteration to check the stop criteria. The
 * norm used in the stop criteria is determined by this object.
 * @param[in] recyclingParams The recycling parameters.
 * @param[in,out] recyclingStatus The recycling status. It contains the recycling subspace from
 * previous linear solves at input and the updated recycling subspace at output. It's the
 * responsibility of the caller to set the 'hasOperatorChanged' flag before every solve depending on
 * whether the linear operator has changed since the previous solve.
 * @param[in] abortIfNotSpd Boolean to abort the solve if the matrix is detected not to be symmetric
 * positive definite. Default is false.
 * @param[in] verbosity Verbosity level for logging output.
 * @param[in] initialGuessHint Indicates whether @p x is known to be zero. The zero hint skips the
 * initial matrix-vector product when no recycling subspace is available and requires @p x to be
 * exactly zero.
 * @param[in] projectEveryIteration Controls the projection strategy in the augmented
 * preconditioner. When true, projects the residual at every iteration, maintaining consistent
 * preconditioner behavior. When false, alternates between projecting and not projecting
 * every other iteration to reduce computational cost, but creates a varying preconditioner that
 * may degrade stability. The use of Polak-Ribière formula helps balance this variation. Default is
 * true.
 * @param[in] dot The dot operator. Must also handle matrix-vector operations.
 * @param[in] vectorFactory Factory to create vectors of a given type.
 *
 * @return Linear solver status. Contains the convergence status, number of iterations, and achieved
 * absolute and relative residuals.
 *
 * @details It implements Algorithm 3.6 in Y. Saad, M. Yeung, J. Erhel, and F. Guyomarc'H, "A
 * deflated version of the conjugate gradient algorithm", SISC, 21(5), pp. 1909-1926 (2000).
 * Extracted in September 2023 from https://inria.hal.science/inria-00523686/document
 * @details When projectEveryIteration is false, the preconditioner changes between iterations since
 * the residual is projected every other iteration. The use of Polak-Ribière formula helps balance
 * this variation.
 *
 * @note The input matrix must be a supported matrix or linear operator type. Matrix application
 * functors are NOT supported.
 * @note CUDA matrices are not supported.
 * @note It uses left preconditioning.
 * @note The norm used in the stop criteria is specified by the object 'stopCriterion'.
 * @note Complex arithmetic is not supported.
 */
template <
    typename MatrixType,
    typename RhsType,
    typename SolType,
    typename PrecType,
    typename RecyclingStatusType,
    typename Dot = UsualDot,
    typename StopCriterion = StatusResidualL2<Dot, typename MatrixType::NonConstScalar>,
    typename VectorFactory = MatrixFactoryType<SolType>>
LinearSolverStatus AugmentedPCG(
    MatrixType const& A,
    RhsType const& b,
    SolType& x,
    PrecType const& P,
    int maxIter,
    StopCriterion& stopCriterion,
    RecyclingParams const& recyclingParams,
    RecyclingStatusType& recyclingStatus,
    bool abortIfNotSpd = false,
    VerbosityLevel verbosity = VerbosityLevel::Warning,
    InitialGuessHint initialGuessHint = InitialGuessHint::Unknown,
    bool projectEveryIteration = true,
    Dot dot = {},
    VectorFactory vectorFactory = {}) {
  MOCHI_PROFILE_SCOPE();
  using Scalar = typename MatrixType::NonConstScalar;
  static_assert(IsLinearOperator<MatrixType>, "Matrix type not supported with augmented PCG");
  static_assert(!mochi::IsCuda<MatrixType>, "CUDA matrices not supported with augmented PCG");
  static_assert(
      std::is_same_v<StopCriterion, StatusResidualL2<Dot, Scalar>>,
      "Stop criterion not supported with augmented PCG");
  static_assert(
      std::is_same_v<typename RecyclingStatusType::Scalar, Scalar>, "Inconsistent scalar types");
  MOCHI_ASSERT_VERBOSE(A.Rows() == A.Cols(), "Input matrix must be square.");
  MOCHI_ASSERT_VERBOSE(
      (A.Cols() == GetNumRows(x)) && (GetNumRows(x) == GetNumRows(b)), "Inconsistent sizes.");

  int recyclingSubspaceSize = recyclingStatus.subspaceSize;
  int const targetNumColsV = recyclingParams.maxSubspaceSize + recyclingParams.incrDirections;
  int const targetNumColsAV = targetNumColsV;
  MOCHI_ASSERT_VERBOSE(recyclingSubspaceSize >= 0, "Recycling subspace size must not be negative.");

  LinearSolverStatus status;
  if (recyclingSubspaceSize == 0) {
    recyclingStatus.V.Resize(A.Cols(), targetNumColsV);
    recyclingStatus.AV.Resize(A.Rows(), targetNumColsAV);
    status =
        PCG(A,
            b,
            x,
            P,
            maxIter,
            stopCriterion,
            abortIfNotSpd,
            verbosity,
            /*usePolakRibiere*/ true,
            initialGuessHint,
            dot,
            vectorFactory);
  } else {
    MOCHI_ASSERT_VERBOSE(
        initialGuessHint != InitialGuessHint::Zero || dot(x, x) == 0,
        "InitialGuessHint::Zero requires an exactly zero initial guess.");
    MOCHI_ASSERT_VERBOSE(
        (recyclingStatus.V.Rows() == A.Cols()) && (recyclingStatus.AV.Rows() == A.Rows()) &&
        (recyclingStatus.V.Cols() >= recyclingSubspaceSize) &&
        (recyclingStatus.AV.Cols() >= recyclingSubspaceSize));
    if (recyclingStatus.hasOperatorChanged) {
      //--- Update AV.
      auto const Q = recyclingStatus.V.LeftCols(recyclingSubspaceSize);
      auto AQ = recyclingStatus.AV.LeftCols(recyclingSubspaceSize);
      Apply(A, Q, AQ); // AQ = A * Q
    }
    //--- Update QtAQ.
    Matrix<Scalar> QtAQ(recyclingSubspaceSize, recyclingSubspaceSize);
    auto const Q = recyclingStatus.V.LeftCols(recyclingSubspaceSize);
    auto const AQ = recyclingStatus.AV.LeftCols(recyclingSubspaceSize);
    QtAQ = Q.Transpose() * AQ;
    Matrix<Scalar> QtAAQ(AQ.Cols(), AQ.Cols());
    QtAAQ = AQ.Transpose() * AQ;
    //--- Solve the eigenproblem.
    recyclingSubspaceSize = Min(recyclingSubspaceSize, recyclingParams.maxSubspaceSize);
    Matrix<Scalar> Z;
    ColumnVector<Scalar> D;
    GeneralizedSelfAdjointEigenDecomposition(QtAAQ, QtAQ, Z, D);
    auto Zn = Z.LeftCols(recyclingSubspaceSize);
    //--- Resize and update V and AV. Resizing is needed if the recycling parameters change between
    //--- solves.
    Matrix<Scalar> tmp = Q * Zn;
    if (recyclingStatus.V.Cols() != targetNumColsV) {
      recyclingStatus.V.Resize(A.Cols(), targetNumColsV);
    }
    recyclingStatus.V.LeftCols(recyclingSubspaceSize) = tmp;
    tmp = AQ * Zn;
    if (recyclingStatus.AV.Cols() != targetNumColsAV) {
      recyclingStatus.AV.Resize(A.Rows(), targetNumColsAV);
    }
    recyclingStatus.AV.LeftCols(recyclingSubspaceSize) = tmp;
    //--- Update QtAQ (use QtAAQ memory space as workspace).
    auto work = QtAAQ.LeftCols(recyclingSubspaceSize);
    work = QtAQ * Zn;
    QtAQ.Block(0, 0, recyclingSubspaceSize, recyclingSubspaceSize) = Zn.Transpose() * work;
    //--- Invert QntAQn.
    int info = 0;
    auto const Qn = recyclingStatus.V.LeftCols(recyclingSubspaceSize);
    auto const AQn = recyclingStatus.AV.LeftCols(recyclingSubspaceSize);
    auto const QntAQn = QtAQ.Block(0, 0, recyclingSubspaceSize, recyclingSubspaceSize);
    LDLt<Scalar> invQtAQ(QntAQn, info);
    //--- Update the initial guess.
    ColumnVector<Scalar> Qdot(Qn.Cols(), 1);
    Qdot = (Qn.Transpose() * b);
    if (initialGuessHint != InitialGuessHint::Zero) {
      Qdot -= (AQn.Transpose() * x);
    }
    invQtAQ.LeftSolveInPlace(Qdot);
    x += Qn * Qdot;
    //--- Invert QntQn.
    Matrix<Scalar> QntQn = Qn.Transpose() * Qn;
    LDLt<Scalar> invQtQ(QntQn, info);
    //--- Compute the augmented preconditioner.
    bool doWtr = true;
    ColumnVector<Scalar> xNew(Qn.Rows());
    auto augP = [&](auto const& x, auto& Px) {
      if (doWtr) {
        Qdot = Qn.Transpose() * x;
        invQtQ.LeftSolveInPlace(Qdot);
        xNew = x - Qn * Qdot;
        P(xNew, Px);
      } else {
        P(x, Px);
      }
      if (!projectEveryIteration) {
        doWtr = !doWtr;
      }
      //--- Compute inner product.
      Qdot = AQn.Transpose() * Px;
      invQtAQ.LeftSolveInPlace(Qdot);
      //--- Update Px.
      Px -= Qn * Qdot;
    };
    status =
        PCG(A,
            b,
            x,
            augP,
            maxIter,
            stopCriterion,
            abortIfNotSpd,
            verbosity,
            /*usePolakRibiere*/ true,
            // Projection can make x nonzero even if it was initially zero.
            InitialGuessHint::Unknown,
            dot,
            vectorFactory);
  }

  //--- Copy the retained directions and update the sizes.
  if (status.numIterDone > 0) {
    auto Vnew = stopCriterion.GetRetainedDirections();
    auto AVnew = stopCriterion.GetRetainedMappedDirections();
    MOCHI_ASSERT_VERBOSE(Vnew.Cols() == AVnew.Cols());
    if (recyclingParams.algorithm == RecyclingAlgorithm::LiFo) {
      recyclingStatus.V.MiddleCols(recyclingSubspaceSize, Vnew.Cols()) = Vnew;
      recyclingStatus.AV.MiddleCols(recyclingSubspaceSize, AVnew.Cols()) = AVnew;
      recyclingStatus.subspaceSize = recyclingSubspaceSize + Vnew.Cols();
    } else
      MOCHI_UNLIKELY {
        MOCHI_ASSERT(false, "Unsupported recycling algorithm.");
      }
  }

  return status;
}

} // namespace mochi::krylov
