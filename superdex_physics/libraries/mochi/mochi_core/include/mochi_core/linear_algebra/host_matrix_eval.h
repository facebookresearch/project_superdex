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

#include <mochi_core/linear_algebra/accessor_based_products.h>
#include <mochi_core/linear_algebra/base_tools.h>
#include <mochi_core/linear_algebra/matrix_accessors.h>
#include <mochi_core/linear_algebra/matrix_expressions.h>

#include <type_traits>
#include <utility>

namespace mochi::details {

template <typename Scalar>
struct Multiplier {
  Scalar factor = {};
  template <typename S2>
  MOCHI_ANY MOCHI_FORCE_INLINE auto Apply(S2 v) {
    return factor * v;
  }

  template <typename T, int N>
  MOCHI_ANY MOCHI_FORCE_INLINE auto Apply(Simd<T, N> v) {
    return T(factor) * v;
  }

  template <typename S2>
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr auto Value() const {
    return S2{factor};
  }
};

template <>
struct Multiplier<void> {
  template <typename S2>
  MOCHI_ANY MOCHI_FORCE_INLINE auto Apply(S2 v) {
    return v;
  }
  template <typename S2>
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr auto Value() const {
    return S2{1};
  }
};

struct SetTag {};

template <typename Accessor>
inline constexpr bool kEvalDidSetBase = false;

template <>
inline constexpr bool kEvalDidSetBase<bool> = true;

template <>
inline constexpr bool kEvalDidSetBase<SetTag> = true;

template <typename T>
inline constexpr bool kEvalDidSetBase<AccessorWithSet<T>> = true;

template <typename T>
inline constexpr bool kEvalDidSet = kEvalDidSetBase<std::decay_t<T>>;

template <typename T>
inline constexpr bool kIsMatrixProduct = false;

template <typename LHS, typename RHS>
inline constexpr bool kIsMatrixProduct<BinOp<LHS, RHS, ops::Mul>> =
    IsMatrixLike<LHS> && IsMatrixLike<RHS>;

template <typename T>
  requires std::is_arithmetic_v<T>
MOCHI_ANY auto GetSimpleAccessor(T v) {
  return ScalarAccessor<T>{v};
}

template <typename Mat>
  requires requires(Mat A) { GetAccessor(A); }
MOCHI_ANY auto GetSimpleAccessor(Mat const& mat) {
  return GetAccessor(mat);
}

template <typename Left, typename Right>
MOCHI_ANY auto GetSimpleAccessor(BinOp<Left, Right, ops::Mul> const& expr) {
  static_assert(
      !kIsMatrixProduct<std::decay_t<decltype(expr)>>,
      "Matrix matrix product has no GetSimpleAccessor");
  return BinaryOpAcc{GetSimpleAccessor(expr.lhs), GetSimpleAccessor(expr.rhs), MultOp{}};
}

template <typename T, typename DestAccessor, int kRows, int kCols>
  requires std::is_arithmetic_v<T>
MOCHI_ANY auto Eval(
    T v,
    [[maybe_unused]] DestAccessor&& destAccessor,
    [[maybe_unused]] IntOrEmpty<kRows> rows,
    [[maybe_unused]] IntOrEmpty<kCols> cols) {
  return ScalarAccessor<T>{v};
}

template <
    typename Mat,
    typename DestAccessor,
    int kRows,
    int kCols,
    typename = decltype(GetAccessor(std::declval<Mat>()))>
MOCHI_ANY auto Eval(
    Mat const& mat,
    [[maybe_unused]] DestAccessor&& destAccessor,
    [[maybe_unused]] IntOrEmpty<kRows> rows,
    [[maybe_unused]] IntOrEmpty<kCols> cols) {
  return GetAccessor(mat);
}

/**
 * @brief Adapt the destination accessor to the effect of the evaluation of the lhs.
 *
 * @details The destination accessor can be one that overwrites the destination or does an update.
 * For a simple assignment (A = B + C), the evaluation of B into an accessor does not modify A
 * and the destination for the evaluation is C must be unchanged. When an expression includes
 * a product, (A = B*D + C), the product writes its result into the destination (A) and the
 * accessor for the evaluation of C must be a destination accessor with increment rather than
 * overwrite. The sign associated with the assignment must also be taken into account.
 * @tparam kNegate Whether the sign associated with the next evaluation is negative.
 * @tparam DA The underlying accessor type for the destination.
 * @tparam kOp The operation of the original DestinationAccessor.
 * @tparam Previous The type of Eval(Rhs). It lets us know if that performed an assignment.
 * @param da The initial destination accessor.
 * @return An accessor accounting for existing data in the destination.
 */
template <bool kNegate = false, typename DA, DestOp kOp, typename Previous>
MOCHI_ANY auto After(DestinationAccessor<kOp, DA> da, Previous const& /*unused*/) {
  if constexpr (kEvalDidSet<Previous>) {
    return DestinationAccessor<AfterSet(kOp, kNegate), DA>{da.accessor};
  } else if constexpr (kNegate) {
    return DestinationAccessor<-kOp, DA>{da.accessor};
  } else {
    return da;
  }
}

template <typename DA, DestOp kOp>
MOCHI_ANY auto Negate(DestinationAccessor<kOp, DA> da) {
  return DestinationAccessor<-kOp, DA>{da.accessor};
}

template <typename A, typename T = std::remove_reference_t<A>>
MOCHI_ANY decltype(auto) ForwardStripped(T& t) {
  using Y = std::remove_const_t<T>;
  if constexpr (kIsAccessorWithSet<Y>) {
    return static_cast<typename T::type&&>(t.accessor);
  } else {
    return static_cast<T&&>(t);
  }
}

template <typename A, typename B, typename Op>
MOCHI_ANY auto MakeBinOpAccessor(A&& a, B&& b, Op op) {
  constexpr bool kWithSet = kEvalDidSet<A> || kEvalDidSet<B>;
  if constexpr (std::is_same_v<A, SetTag> && std::is_same_v<B, SetTag>) {
    return SetTag{};
  } else if constexpr (std::is_same_v<A, SetTag>) {
    if constexpr (std::is_same_v<Op, SumOp>) {
      return AccessorWithSet{ForwardStripped<B>(b)};
    } else {
      return AccessorWithSet{NegateAccessor{ForwardStripped<B>(b)}};
    }
  } else if constexpr (std::is_same_v<B, SetTag>) {
    return AccessorWithSet{ForwardStripped<A>(a)};
  } else {
    if constexpr (kWithSet) {
      return AccessorWithSet{BinaryOpAcc{ForwardStripped<A>(a), ForwardStripped<B>(b), op}};
    } else {
      return BinaryOpAcc{std::forward<A>(a), std::forward<B>(b), op};
    }
  }
}

template <typename Scalar, typename Expr, typename DestAccessor, int kRows, int kCols>
MOCHI_ANY auto Eval(
    ScaledExpr<Scalar, Expr> const& expr,
    DestAccessor&& destAccessor,
    IntOrEmpty<kRows> rows,
    IntOrEmpty<kCols> cols) {
  if constexpr (kIsMatrixProduct<Expr>) {
    return EvalProduct(expr.matExpr, destAccessor, rows, cols, Multiplier<Scalar>{expr.scale});
  } else {
    return BinaryOpAcc{
        ScalarAccessor<Scalar>{expr.scale}, Eval(expr.matExpr, destAccessor, rows, cols), MultOp{}};
  }
}

template <typename Expr, typename DestAccessor, int kRows, int kCols>
MOCHI_ANY auto Eval(
    UnaryNeg<Expr> const& expr,
    DestAccessor&& destAccessor,
    IntOrEmpty<kRows> rows,
    IntOrEmpty<kCols> cols) {
  using Scalar = expr_scalar_type<Expr>;
  // The expression destination must account for the negation.
  if constexpr (kIsMatrixProduct<Expr>) {
    return EvalProduct(expr.arg, destAccessor, rows, cols, Multiplier<Scalar>{-1});
  } else {
    auto negDestAccessor = Negate(destAccessor);
    auto exprAccessor = Eval(expr.arg, negDestAccessor, rows, cols);
    if constexpr (kEvalDidSet<decltype(exprAccessor)>) {
      return AccessorWithSet{
          BinaryOpAcc{ScalarAccessor<Scalar>{-1}, std::move(exprAccessor.accessor), MultOp{}}};
    } else {
      return BinaryOpAcc{ScalarAccessor<Scalar>{-1}, std::move(exprAccessor), MultOp{}};
    }
  }
}

template <typename Left, typename Right, typename DestAccessor, int kRows, int kCols>
MOCHI_ANY auto Eval(
    BinOp<Left, Right, ops::Add> const& expr,
    DestAccessor&& destAccessor,
    IntOrEmpty<kRows> rows,
    IntOrEmpty<kCols> cols) {
  // future optimization
  //  if constexpr (kIsMatrixProduct<Left> || kIsMatrixProduct<Right>) {
  //    return SetTag{};
  //  } else {
  auto lhs = Eval(expr.lhs, destAccessor, rows, cols);
  auto rhsDestAccessor = After(destAccessor, lhs);
  auto rhs = Eval(expr.rhs, rhsDestAccessor, rows, cols);
  return MakeBinOpAccessor(std::move(lhs), std::move(rhs), SumOp{});
  //  }
}

template <
    DestOp kOpMode = DestOp::Set,
    typename Left,
    typename Right,
    typename DestAccessor,
    int kRows,
    int kCols>
MOCHI_ANY auto Eval(
    BinOp<Left, Right, ops::Sub> const& expr,
    DestAccessor&& destAccessor,
    IntOrEmpty<kRows> rows,
    IntOrEmpty<kCols> cols) {
  // future optimization
  //  if constexpr (kIsMatrixProduct<Left> || kIsMatrixProduct<Right>) {
  //    return false;
  //  } else {
  auto lhs = Eval(expr.lhs, destAccessor, rows, cols);
  auto rhsDestAccessor = After<true>(destAccessor, lhs);
  auto rhs = Eval(expr.rhs, rhsDestAccessor, rows, cols);
  return MakeBinOpAccessor(std::move(lhs), std::move(rhs), SubtractOp{});
  //  }
}

template <typename T>
inline constexpr bool IsSimple = !IsMatrixExpr<std::decay_t<T>>;

template <typename L, typename R>
inline constexpr bool IsSimple<BinOp<L, R, ops::Mul>> = !IsMatrixLike<L> || !IsMatrixLike<R>;

static_assert(IsSimple<Matrix<float>> && IsSimple<Matrix<double>>);
static_assert(!IsSimple<BinOp<Matrix<float>&, Matrix<float>&, MultOp>>);
static_assert(!IsSimple<BinOp<Matrix<double>&, Matrix<double>&, MultOp>>);

template <typename Expr>
MOCHI_ANY auto ProductArgAccessor(Expr const& expr);

/** @brief Fully evaluate the product of matrices.
 *
 * @tparam Left
 * @tparam Right
 * @tparam DestAccessor
 * @tparam kRows
 * @tparam kCols
 * @param expr
 * @param destAccessor
 * @param rows
 * @param cols
 * @return
 */
template <
    typename Left,
    typename Right,
    typename DestAccessor,
    int kRows,
    int kCols,
    typename Factor = Multiplier<void>>
MOCHI_ANY auto EvalProduct(
    BinOp<Left, Right, ops::Mul> const& expr,
    DestAccessor&& destAccessor,
    IntOrEmpty<kRows> rows,
    IntOrEmpty<kCols> cols,
    Factor factor = Multiplier<void>{}) {
  using Scalar = expr_scalar_type<Left>;
  auto l = ProductArgAccessor(expr.lhs);
  auto r = ProductArgAccessor(expr.rhs);
  details::DoProduct<Scalar>(destAccessor, l, r, rows, cols, expr.lhs.CECols(), factor);
  return SetTag{};
}

template <typename Left, typename Right, typename DestAccessor, int kRows, int kCols>
MOCHI_ANY auto Eval(
    BinOp<Left, Right, ops::Mul> const& expr,
    DestAccessor&& destAccessor,
    IntOrEmpty<kRows> rows,
    IntOrEmpty<kCols> cols) {
  // Determine if we create a temporary for either or both sides or we can
  // directly compute the result into destAccessor.
  // We accept at most a scalar*matrix or -scalar*matrix
  if constexpr (kIsMatrixProduct<std::decay_t<decltype(expr)>>) {
    // If any side of the operator contains itself
    return EvalProduct(expr, destAccessor, rows, cols);
  } else if constexpr (IsBlockSparseMatrix<Left> || IsSparseMatrix<Left>) {
    // Check if the RHS will write to the destAccessor we pass it.
    using EvalType = decltype(Eval(expr.rhs, destAccessor, IntOrEmpty<-1>{1}, IntOrEmpty<-1>{1}));
    static_assert(!kEvalDidSet<EvalType>, "Currently cannot deal with this case");
    auto rhsAcc = Eval(expr.rhs, destAccessor, IntOrEmpty<-1>{1}, IntOrEmpty<-1>{1});
    expr.lhs.AccessorApplyToRange(
        rhsAcc, destAccessor, 0, expr.lhs.Rows(), expr.rhs.CECols().iVal());
    return SetTag{};
  } else {
    return BinaryOpAcc{
        Eval(expr.lhs, destAccessor, rows, cols),
        Eval(expr.rhs, destAccessor, rows, cols),
        MultOp{}};
  }
}

/** @brief Perform the assignment using left and right-hand-side Accessors
 * @details The assignment chooses the most effective SIMD orientation.
 * @tparam LHS
 * @tparam RHS
 * @param lhs A write accessor to the left-hand-side.
 * @param rhs A read accessor to the right-hand-side.
 * @param rows Row count of the result.
 * @param cols Column count of the result.
 * @return
 */
template <typename Scalar, typename LHS, typename RHS, int kRows, int kCols>
MOCHI_ANY void
Assign(LHS const& lhs_, RHS const& rhs_, IntOrEmpty<kRows> rows, IntOrEmpty<kCols> cols) {
  using VType = Simd<std::remove_const_t<Scalar>>; // Native SIMD size.
  using VTypeHalf = Simd<std::remove_const_t<Scalar>, VType::kSize / 2>; // Half the size of VType.
  constexpr auto kVecSize = VType::kSize;
  constexpr auto kVecSizeHalf = VTypeHalf::kSize;
  constexpr bool kUseSimd = VType::kIsSupported; // TODO(T158480383): Introduce minimum SIMD size
                                                 // to favor the SIMD implementation.

  // A SetTag accessor has already done full work.
  if constexpr (std::is_same_v<RHS, SetTag>) {
    return;
  } else {
    // If the rhs has partially filled the lhs, the destination accessor must account for it.
    auto const& lhs = After(lhs_, rhs_);
    // Get the underlying accessor if the rhs is wrapped in an AccessorWithSet
    auto const& rhs = [](auto const& r) -> decltype(auto) {
      if constexpr (kEvalDidSet<RHS>) {
        return r.accessor;
      } else {
        return r;
      }
    }(rhs_);

    constexpr auto kLhsCosts = LHS::RowColCosts();
    constexpr auto kRhsCosts = RHS::RowColCosts();
    if constexpr (
        kUseSimd && (kLhsCosts.first + kRhsCosts.first < kLhsCosts.second + kRhsCosts.second)) {
      // Column vectors are cheaper
      for (int c = 0; c < cols; ++c) {
        int r = 0;
        if constexpr (kRows == krylov::kDynamic || kRows >= kVecSize) {
          for (; r + kVecSize <= rows; r += kVecSize) {
            lhs.StoreColVector(r, c, rhs.template ColVector<VType>(r, c));
          }
        }
        if constexpr (
            VTypeHalf::kIsSupported &&
            (kRows == krylov::kDynamic || kRows % kVecSize >= kVecSizeHalf)) {
          for (; r + kVecSizeHalf <= rows; r += kVecSizeHalf) {
            lhs.StoreColVector(r, c, rhs.template ColVector<VTypeHalf>(r, c));
          }
        }
        if constexpr (
            kRows == krylov::kDynamic || (VTypeHalf::kIsSupported && kRows % kVecSizeHalf > 0) ||
            (!VTypeHalf::kIsSupported && kRows % kVecSize > 0)) {
          for (; r < rows; ++r) {
            lhs.Store(r, c, rhs(r, c));
          }
        }
      }
    } else if constexpr (
        kUseSimd && (kLhsCosts.first + kRhsCosts.first > kLhsCosts.second + kRhsCosts.second)) {
      // Row vectors are cheaper
      for (int r = 0; r < rows; ++r) {
        int c = 0;
        if constexpr (kCols == krylov::kDynamic || kCols >= kVecSize) {
          for (; c + kVecSize <= cols; c += kVecSize) {
            lhs.StoreRowVector(r, c, rhs.template RowVector<VType>(r, c));
          }
        }
        if constexpr (
            VTypeHalf::kIsSupported &&
            (kCols == krylov::kDynamic || kCols % kVecSize >= kVecSizeHalf)) {
          for (; c + kVecSizeHalf <= cols; c += kVecSizeHalf) {
            lhs.StoreRowVector(r, c, rhs.template RowVector<VTypeHalf>(r, c));
          }
        }
        if constexpr (
            kCols == krylov::kDynamic || (VTypeHalf::kIsSupported && kCols % kVecSizeHalf > 0) ||
            (!VTypeHalf::kIsSupported && kCols % kVecSize > 0)) {
          for (; c < cols; ++c) {
            lhs.Store(r, c, rhs(r, c));
          }
        }
      }
    } else {
      // Non-SIMD implementation. Also used if the type has SIMD support but rows and columns have
      // the same cost in order to let the compiler optimize.
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          lhs.Store(r, c, rhs(r, c));
        }
      }
    }
  }
}

template <typename Expr>
MOCHI_ANY auto ProductArgAccessor(Expr const& expr) {
  if constexpr (!IsMatrixLike<Expr> || IsSimple<Expr>) {
    return GetSimpleAccessor(expr);
  } else {
    // For more complex expressions, a temporary matrix must be created.
    // TODO: Some expressions currently handled in this branch don't require a temporary.
    using Scalar = expr_scalar_type<Expr>;
    auto rows = expr.CERows();
    auto cols = expr.CECols();
    constexpr int kRowsAtCT = kValueAtCompileTime<decltype(rows)>;
    constexpr int kColsAtCT = kValueAtCompileTime<decltype(cols)>;
    TempAccessor<typename Expr::scalar_type, kRowsAtCT, kColsAtCT> tmpAccessor(
        rows.iVal(), cols.iVal());

    auto dest =
        details::SetDest(Accessor<Scalar, 1, kRowsAtCT>{tmpAccessor.ptr(0, 0), 1, rows.iVal()});

    auto res = details::Eval(expr, dest, rows, cols);
    // Compute the result.
    Assign<Scalar>(dest, res, rows, cols);
    //
    return std::move(tmpAccessor);
  }
}

} // namespace mochi::details
