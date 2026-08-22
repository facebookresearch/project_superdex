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

#include <mochi_core/linear_algebra/any_matrix.h>
#include <mochi_core/linear_algebra/cuda/cuda_block_jacobi_prec.h>
#include <mochi_core/linear_algebra/cuda/cuda_matrix.h>
#include <mochi_core/linear_algebra/krylov/amg/amg_prec.h>
#include <mochi_core/linear_algebra/krylov/block_jacobi_prec.h>
#include <mochi_core/linear_algebra/krylov/block_ssor_prec.h>
#include <mochi_core/linear_algebra/krylov/colored_ssor_prec.h>
#include <mochi_core/linear_algebra/krylov/identity_prec.h>
#include <mochi_core/linear_algebra/krylov/incomplete_cholesky_prec.h>
#include <mochi_core/linear_algebra/krylov/ldlt_prec.h>
#include <mochi_core/linear_algebra/krylov/lu_prec.h>
#include <mochi_core/linear_algebra/krylov/preconditioner.h>
#include <mochi_core/linear_algebra/krylov/relaxed_ilu_prec.h>
#include <mochi_core/linear_algebra/krylov/ssor_prec.h>
#include <mochi_core/linear_algebra/krylov/sym_inverse_prec.h>
#include <mochi_core/linear_algebra/low_rank_augmented_matrix.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/solvers/island_operators.h>
#include <mochi_core/solvers/linear_solver_params.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/no_copy.h>

#include <functional>
#include <memory>
#include <utility>
#include <variant>

namespace mochi {

namespace details {

template <typename T, int kPrecBlockSize, typename MatrixType>
std::function<void(CudaVectorView<T> const& x, CudaVectorView<T> Px)> CreateCudaPreconditioner(
    MatrixType const& A,
    PreconditionerType const& preconType) {
  using InputVectorType = CudaVectorView<T>;
  using OutputVectorType = CudaVectorView<T>;
  if (preconType == PreconditionerType::None) {
    return [](InputVectorType const& x, OutputVectorType Px) { Px = x; };
  } else if (preconType == PreconditionerType::Jacobi) {
    krylov::CudaJacobiPrec<T> prec(A);
    return [P = std::move(prec)](InputVectorType const& x, OutputVectorType Px) { P(x, Px); };
  } else if (preconType == PreconditionerType::BlockJacobi) {
    if (A.Rows() % kPrecBlockSize == 0) {
      krylov::CudaBlockJacobiPrec<T, kPrecBlockSize> prec(A);
      return [P = std::move(prec)](InputVectorType const& x, OutputVectorType Px) { P(x, Px); };
    } else {
      MOCHI_LOG_WARNING_ONCE(
          "The requested preconditioner block size (%i) is incompatible with the matrix dimension (%i). Will look for a compatible block size.",
          kPrecBlockSize,
          A.Rows());
      if (A.Rows() % 5 == 0) {
        krylov::CudaBlockJacobiPrec<T, 5> prec(A);
        return [P = std::move(prec)](InputVectorType const& x, OutputVectorType Px) { P(x, Px); };
      } else if (A.Rows() % 4 == 0) {
        krylov::CudaBlockJacobiPrec<T, 4> prec(A);
        return [P = std::move(prec)](InputVectorType const& x, OutputVectorType Px) { P(x, Px); };
      } else if (A.Rows() % 3 == 0) {
        krylov::CudaBlockJacobiPrec<T, 3> prec(A);
        return [P = std::move(prec)](InputVectorType const& x, OutputVectorType Px) { P(x, Px); };
      } else if (A.Rows() % 2 == 0) {
        krylov::CudaBlockJacobiPrec<T, 2> prec(A);
        return [P = std::move(prec)](InputVectorType const& x, OutputVectorType Px) { P(x, Px); };
      } else {
        krylov::CudaJacobiPrec<T> prec(A);
        return [P = std::move(prec)](InputVectorType const& x, OutputVectorType Px) { P(x, Px); };
      }
    }
  } else {
    MOCHI_LOG_ERROR_ONCE(
        "Preconditioner type (%i) not supported with CUDA. No preconditioner will be used.",
        static_cast<int>(preconType));
    return [](InputVectorType const& x, OutputVectorType Px) { Px = x; };
  }
}

/// @brief Whether a preconditioner type stores a reference or view into its input matrix.
[[nodiscard]] inline constexpr bool PreconditionerStoresInputView(
    PreconditionerType const& preconType) {
  static_assert(
      static_cast<int>(PreconditionerType::Count) == 13,
      "Please update this function if PreconditionerType enumerator changes");
  switch (preconType) {
    case PreconditionerType::SSOR:
    case PreconditionerType::BlockSSOR:
    case PreconditionerType::AMG:
      return true;
    case PreconditionerType::None:
    case PreconditionerType::Jacobi:
    case PreconditionerType::BlockJacobi:
    case PreconditionerType::ColoredSSOR:
    case PreconditionerType::ILU0:
    case PreconditionerType::IC0:
    case PreconditionerType::SymInverse:
    case PreconditionerType::LDLT:
    case PreconditionerType::LU:
    case PreconditionerType::PerActor:
      return false;
    default:
      MOCHI_ASSERT(false, "Unexpected preconditioner type.");
      return false;
  }
}

template <typename T, int kPrecBlockSize, typename MatrixType>
std::unique_ptr<Preconditioner<T>> CreatePreconditioner(
    MatrixType const& A,
    PreconditionerType const& preconType) {
  if (preconType == PreconditionerType::None) {
    // Early exit to avoid expensive work, e.g. condensing to full global matrix.
    return std::make_unique<krylov::IdentityPrec<T>>(A);
  }

  if (((preconType == PreconditionerType::BlockJacobi) ||
       (preconType == PreconditionerType::BlockSSOR) || (preconType == PreconditionerType::AMG)) &&
      (GetNumRows(A) % kPrecBlockSize != 0)) {
    MOCHI_LOG_WARNING_ONCE(
        "The requested preconditioner block size (%i) is incompatible with the matrix dimension (%i). No preconditioner will be used.",
        kPrecBlockSize,
        GetNumRows(A));
    return std::make_unique<krylov::IdentityPrec<T>>(A);
  }

  if constexpr (IsAnyMatrix<MatrixType>) {
    switch (preconType) {
      case PreconditionerType::Jacobi:
        return std::make_unique<krylov::JacobiPrec<T>>(A);
      case PreconditionerType::BlockJacobi:
        return std::make_unique<krylov::BlockJacobiPrec<T, kPrecBlockSize>>(A);
      case PreconditionerType::SSOR:
        return std::make_unique<krylov::SSORPrec<T, MatrixType>>(A);
      case PreconditionerType::BlockSSOR:
        return std::make_unique<krylov::BlockSSORPrec<T, kPrecBlockSize, MatrixType>>(A);
      case PreconditionerType::ColoredSSOR:
        return std::make_unique<krylov::ColoredSSORPrec<MatrixType>>(A);
      case PreconditionerType::ILU0:
        return std::make_unique<krylov::RelaxedILUPrec<MatrixType>>(
            A, /*fillInLevel*/ 0, /*alphaRelax*/ T{0});
      case PreconditionerType::IC0:
        return std::make_unique<krylov::IncompleteCholeskyPrec<MatrixType>>(
            A, /*fillInLevel*/ 0, /*alphaShift*/ T{0});
      case PreconditionerType::SymInverse:
        return std::make_unique<krylov::SymInversePrec<T>>(A);
      case PreconditionerType::LDLT:
        return std::make_unique<krylov::LDLtPrec<T>>(A);
      case PreconditionerType::LU:
        return std::make_unique<krylov::LUPrec<T>>(A);
      case PreconditionerType::AMG:
        return std::make_unique<krylov::AMGPrec<T, kPrecBlockSize>>(A);
      default: {
        MOCHI_LOG_ERROR_ONCE(
            "Preconditioner type (%i) not supported. No preconditioner will be used.",
            static_cast<int>(preconType));
        return std::make_unique<krylov::IdentityPrec<T>>(A);
      }
    }
  } else if constexpr (IsIslandOperators<MatrixType>) {
    if (preconType == PreconditionerType::PerActor) {
      return std::make_unique<PerActorPrec<T>>(std::move(A.MakePerActorPrec()));
    } else {
      MOCHI_ASSERT(
          !PreconditionerStoresInputView(preconType),
          "Preconditioner type (%d) stores a reference/view into its input matrix and cannot be safely created from an IslandOperators.",
          static_cast<int>(preconType));
      // Global preconditioners require conversion to global matrix. TODO: Some global
      // preconditioners don't require the full global matrix. If needed, their construction could
      // be optimized.
      auto globalMatrix = A.CondenseFullMatrix();
      return std::visit(
          [&](auto const& mat) { return CreatePreconditioner<T, kPrecBlockSize>(mat, preconType); },
          globalMatrix);
    }
  } else {
    static_assert(IsLowRankAugmentedMatrix<MatrixType>, "Unsupported matrix type");
    // TODO(T189246854): Introduce augmented preconditioners with low-rank updates.
    return CreatePreconditioner<T, kPrecBlockSize>(A.GetUnaugmentedMatrix(), preconType);
  }
  static_assert(
      static_cast<int>(PreconditionerType::Count) == 13,
      "Please update this function if PreconditionerType enumerator changes");
}

/// @brief Update the preconditioner. Returns true if the preconditioner was successfully updated
/// and false otherwise.
template <typename T, int kPrecBlockSize, typename MatrixType>
[[nodiscard]] bool UpdatePreconditioner(
    MatrixType const& A,
    PreconditionerType const& preconType,
    std::unique_ptr<Preconditioner<T>>& outPrec) {
  auto Update = [&]<typename PrecType>() -> bool {
    if (auto* precPt = dynamic_cast<PrecType*>(outPrec.get())) {
      precPt->Update(A);
      return true;
    } else {
      return false;
    }
  };

  if (!outPrec) {
    return false;
  } else if (preconType == PreconditionerType::None) {
    // Early exit to avoid expensive work, e.g. condensing to full global matrix.
    return Update.template operator()<krylov::IdentityPrec<T>>();
  }

  if constexpr (IsAnyMatrix<MatrixType>) {
    switch (preconType) {
      case PreconditionerType::Jacobi:
        return Update.template operator()<krylov::JacobiPrec<T>>();
      case PreconditionerType::BlockJacobi:
        return Update.template operator()<krylov::BlockJacobiPrec<T, kPrecBlockSize>>();
      case PreconditionerType::SSOR:
        return Update.template operator()<krylov::SSORPrec<T, MatrixType>>();
      case PreconditionerType::BlockSSOR:
        return Update.template operator()<krylov::BlockSSORPrec<T, kPrecBlockSize, MatrixType>>();
      case PreconditionerType::SymInverse:
        return Update.template operator()<krylov::SymInversePrec<T>>();
      case PreconditionerType::ColoredSSOR: // 'Update' assumes the sparsity pattern is unchanged.
      case PreconditionerType::AMG: // 'Update' method assumes the sparsity pattern is unchanged.
      case PreconditionerType::ILU0: // 'Update' method assumes the sparsity pattern is unchanged.
      case PreconditionerType::IC0: // 'Update' method assumes the sparsity pattern is unchanged.
      case PreconditionerType::LDLT: // 'Update' method is not supported.
      case PreconditionerType::LU: // 'Update' method is not supported.
        return false;
      default: {
        MOCHI_LOG_ERROR_ONCE(
            "Preconditioner type (%i) not supported.", static_cast<int>(preconType));
        return false;
      }
    }
  } else if constexpr (IsIslandOperators<MatrixType>) {
    if (preconType == PreconditionerType::PerActor) {
      return Update.template operator()<PerActorPrec<T>>();
    } else {
      MOCHI_ASSERT(
          !PreconditionerStoresInputView(preconType),
          "Preconditioner type (%d) stores a reference/view into its input matrix and cannot be safely updated from an IslandOperators.",
          static_cast<int>(preconType));
      // Global preconditioners require conversion to global matrix. TODO: Some global
      // preconditioners don't require the full global matrix. If needed, their construction could
      // be optimized.
      auto globalMatrix = A.CondenseFullMatrix();
      return std::visit(
          [&](auto const& mat) {
            return UpdatePreconditioner<T, kPrecBlockSize>(mat, preconType, outPrec);
          },
          globalMatrix);
    }
  } else {
    static_assert(IsLowRankAugmentedMatrix<MatrixType>, "Unsupported matrix type");
    // TODO(T189246854): Introduce augmented preconditioners with low-rank updates.
    return UpdatePreconditioner<T, kPrecBlockSize>(A.GetUnaugmentedMatrix(), preconType, outPrec);
  }
  static_assert(
      static_cast<int>(PreconditionerType::Count) == 13,
      "Please update this function if PreconditionerType enumerator changes");
}

/// @brief Whether a preconditioner type can be reused across solves with different matrices of the
/// same size. A common reason a preconditioner cannot be reused is if it stores a reference or view
/// to the matrix, which may not outlive the preconditioner.
[[nodiscard]] inline constexpr bool PreconditionerIsReusable(PreconditionerType const& preconType) {
  static_assert(
      static_cast<int>(PreconditionerType::Count) == 13,
      "Please confirm that 'reusable across solves' is still the exact complement of 'stores an input view' for "
      "the new preconditioner type(s). If it's not, please update this function accordingly.");
  return !PreconditionerStoresInputView(preconType);
}

} // namespace details

/// @brief Preconditioner recycling manager. Tracks the current preconditioner and its usage count
/// to determine when it should be updated based on its configured lifespan.
template <typename T>
class PreconditionerRecyclingManager : NoCopy {
 public:
  auto const& GetPreconditioner() const {
    return _preconditioner;
  }

  /// @brief Set up the preconditioner before a new solve.
  /// @param A The matrix/operator for which to create/update the preconditioner.
  /// @param hasMatrixChanged Whether the matrix/operator has changed since the last solve.
  /// @param preconType The type of preconditioner to use.
  /// @param preconditionerLifespan Maximum number of solves before updating the preconditioner.
  ///
  /// @details The recycling mechanism works as follows:
  /// - When `hasMatrixChanged=true`, `_solveCount` is incremented.
  /// - If no preconditioner exists or the preconditioner type has changed, a new one is created.
  /// - If `_solveCount` exceeds `preconditionerLifespan`, the preconditioner is updated.
  /// - Otherwise, the existing preconditioner is reused.
  template <int kPrecBlockSize, typename MatrixType>
  void SetupPreconditioner(
      MatrixType const& A,
      bool hasMatrixChanged,
      PreconditionerType const& preconType,
      int preconditionerLifespan) {
    _solveCount += static_cast<int>(hasMatrixChanged);

    if (!_preconditioner || (_preconditioner->GetType() != preconType)) {
      // Create a new preconditioner.
      // NOTE: The preconditioner type check ignores whether the requested block size matches the
      // current block size.
      _preconditioner = details::CreatePreconditioner<T, kPrecBlockSize>(A, preconType);
      _solveCount = 1;
    } else if (
        (_solveCount <= preconditionerLifespan) && details::PreconditionerIsReusable(preconType)) {
      // Reuse the existing preconditioner.
    } else {
      // Update the existing preconditioner.
      bool const success =
          details::UpdatePreconditioner<T, kPrecBlockSize>(A, preconType, _preconditioner);
      _solveCount = 1;
      if (!success) {
        _preconditioner = details::CreatePreconditioner<T, kPrecBlockSize>(A, preconType);
      }
    }
  }

 private:
  /// @brief Current preconditioner (null if none exists).
  std::unique_ptr<Preconditioner<T>> _preconditioner = nullptr;

  /// @brief Number of solves performed with the current preconditioner.
  /// Used to determine when the preconditioner should be updated based on its configured lifespan.
  int _solveCount = 0;
};

} // namespace mochi
