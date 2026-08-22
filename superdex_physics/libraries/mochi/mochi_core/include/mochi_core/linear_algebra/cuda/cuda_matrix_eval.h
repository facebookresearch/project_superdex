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

#include <mochi_core/linear_algebra/cuda/cuda_api.h>

#include <mochi_core/linear_algebra/base_tools.h>
#include <mochi_core/linear_algebra/cuda/cuda_matrix_product.h>
#include <mochi_core/linear_algebra/host_matrix_eval.h>
#include <mochi_core/linear_algebra/matrix_accessors.h>
#include <mochi_core/linear_algebra/matrix_expressions.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

namespace mochi::details {

template <typename Scalar, int kRowStrideAtCompileTime, int kColStrideAtCompileTime>
using CudaAccessor = BasicAccessor<Scalar, kRowStrideAtCompileTime, kColStrideAtCompileTime>;

template <typename Mat, typename DestAccessor, int kRows, int kCols>
auto CudaEval(
    Mat const& mat,
    [[maybe_unused]] DestAccessor&& destAccessor,
    [[maybe_unused]] IntOrEmpty<kRows> rows,
    [[maybe_unused]] IntOrEmpty<kCols> cols) {
  return GetAccessor(mat);
}

template <typename Scalar, typename Expr>
struct ScaledAccessor {
  ScaledAccessor(Scalar v, Expr e) : scale(v), expr(std::move(e)) {}
  Scalar scale;
  Expr expr;

  auto ScaledData() const {
    auto sd = expr.ScaledData();
    sd.scale *= scale;
    return sd;
  }
};

template <typename Scalar, typename Expr, typename DestAccessor, int kRows, int kCols>
auto CudaEval(
    ScaledExpr<Scalar, Expr> const& expr,
    DestAccessor&& destAccessor,
    IntOrEmpty<kRows> rows,
    IntOrEmpty<kCols> cols) {
  if constexpr (kIsMatrixProduct<Expr>) {
    return CudaEvalProduct(
        expr.matExpr, destAccessor, rows.iVal(), cols.iVal(), Multiplier<Scalar>{expr.scale});
  } else {
    auto intermediate = CudaEval(expr.matExpr, destAccessor, rows, cols);
    static_assert(!kEvalDidSet<decltype(intermediate)>, "Scaled expression cannot do a set");
    return ScaledAccessor{expr.scale, std::move(intermediate)};
  }
}

template <typename Positives, typename Negatives>
struct CudaSum {
  CudaSum(Positives positives, Negatives negatives)
      : positiveTerms(std::move(positives)), negativeTerms(std::move(negatives)) {}
  Positives positiveTerms;
  Negatives negativeTerms;
};

template <typename T>
auto PositiveTuple(T&& t) {
  return std::tuple{std::forward<T>(t)};
}

/// @brief Obtain the negative part of a CUDA expression.
/// @details The negated expression is returned as a tuple.
/// For anything but a CudaSum, the negated expression is an empty tuple.
template <typename T>
auto NegativeTuple([[maybe_unused]] T&& t) {
  return std::tuple{};
}

template <typename... Positives, typename Negative>
auto PositiveTuple(CudaSum<std::tuple<Positives...>, Negative>&& term) {
  return std::move(term.positiveTerms);
}

template <typename Positive, typename... Negatives>
auto NegativeTuple(CudaSum<Positive, std::tuple<Negatives...>>&& term) {
  return std::move(term.negativeTerms);
}

template <typename Positive, typename Negative>
auto Negate(CudaSum<Positive, Negative>&& term) {
  return CudaSum{std::move(term.negativeTerms), std::move(term.positiveTerms)};
}

template <typename T>
auto Negate(T&& t) {
  // Using std::move similarly to `GetExpr` in `matrix_expressions.h`
  return NegateAccessor{std::forward<T>(t)};
}

/// @brief Treat the expression from the combination A +/- B
template <bool kIsNegative, typename A, typename B>
auto MakeSum(A&& a, B&& b) {
  constexpr bool kWithSet = kEvalDidSet<A> || kEvalDidSet<B>;
  if constexpr (std::is_same_v<A, SetTag> && std::is_same_v<B, SetTag>) {
    // Both expressions have been "inserted" in the result memory space
    return SetTag{};
  } else if constexpr (std::is_same_v<B, SetTag>) {
    // Only expression B has been completely "treated".
    return AccessorWithSet{ForwardStripped<A>(a)};
  } else if constexpr (std::is_same_v<A, SetTag>) {
    // Only expression A has been completely "treated".
    // We need to propagate the potential "-" sign
    if constexpr (kIsNegative) {
      return AccessorWithSet{Negate(ForwardStripped<B>(b))};
    } else {
      return AccessorWithSet{ForwardStripped<B>(b)};
    }
  } else {
    //--- Create a combined expression from A +/- B
    if constexpr (kIsNegative) {
      auto tempExpr = CudaSum{
          std::tuple_cat(
              PositiveTuple(ForwardStripped<A>(a)), NegativeTuple(ForwardStripped<B>(b))),
          std::tuple_cat(
              NegativeTuple(ForwardStripped<A>(a)), PositiveTuple(ForwardStripped<B>(b)))};
      if constexpr (kWithSet) {
        return AccessorWithSet{std::move(tempExpr)};
      } else {
        return tempExpr;
      }
    } else {
      auto tempExpr = CudaSum{
          std::tuple_cat(
              PositiveTuple(ForwardStripped<A>(a)), PositiveTuple(ForwardStripped<B>(b))),
          std::tuple_cat(
              NegativeTuple(ForwardStripped<A>(a)), NegativeTuple(ForwardStripped<B>(b)))};
      if constexpr (kWithSet) {
        return AccessorWithSet{std::move(tempExpr)};
      } else {
        return tempExpr;
      }
    }
  }
}

template <typename Expr>
auto CudaProductArgAccessor(Expr const& expr);

/**
 * @brief Fully evaluate the product of matrices.
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
template <typename Left, typename Right, typename DestAccessor, typename Factor = Multiplier<void>>
auto CudaEvalProduct(
    BinOp<Left, Right, ops::Mul> const& expr,
    DestAccessor&& destAccessor,
    int rows,
    int cols,
    Factor factor = Multiplier<void>{}) {
  // Determine if we create a temporary for either or both sides or we can
  // directly compute the result into destAccessor.
  // We accept at most a scalar*matrix or -scalar*matrix
  using Scalar = expr_scalar_type<Left>;
  auto l = CudaProductArgAccessor(expr.lhs);
  auto r = CudaProductArgAccessor(expr.rhs);
  details::CudaProduct<Scalar>(destAccessor, l, r, rows, cols, expr.lhs.CECols().iVal(), factor);
  return SetTag{};
}

template <typename Left, typename Right, typename DestAccessor, int kRows, int kCols>
auto CudaEval(
    BinOp<Left, Right, ops::Add> const& expr,
    DestAccessor&& destAccessor,
    IntOrEmpty<kRows> rows,
    IntOrEmpty<kCols> cols) {
  auto lhs = CudaEval(expr.lhs, destAccessor, rows, cols);
  auto rhsDestAccessor = After(destAccessor, lhs);
  auto rhs = CudaEval(expr.rhs, rhsDestAccessor, rows, cols);
  return MakeSum<false>(std::move(lhs), std::move(rhs));
}

template <typename Left, typename Right, typename DestAccessor, int kRows, int kCols>
auto CudaEval(
    BinOp<Left, Right, ops::Sub> const& expr,
    DestAccessor&& destAccessor,
    IntOrEmpty<kRows> rows,
    IntOrEmpty<kCols> cols) {
  auto lhs = CudaEval(expr.lhs, destAccessor, rows, cols);
  auto rhsDestAccessor = After<true>(destAccessor, lhs);
  auto rhs = CudaEval(expr.rhs, rhsDestAccessor, rows, cols);
  return MakeSum<true>(std::move(lhs), std::move(rhs));
}

template <typename Left, typename Right, typename DestAccessor, int kRows, int kCols>
auto CudaEval(
    BinOp<Left, Right, ops::Mul> const& expr,
    DestAccessor&& destAccessor,
    IntOrEmpty<kRows> rows,
    IntOrEmpty<kCols> cols) {
  if constexpr (kIsMatrixProduct<std::decay_t<decltype(expr)>>) {
    return CudaEvalProduct(expr, destAccessor, rows.iVal(), cols.iVal());
  } else {
    static_assert(kIsMatrixProduct<std::decay_t<decltype(expr)>>, "This case is not expected.");
  }
}

template <typename Scalar, typename T, size_t... I>
auto GetScaledArray(T const& terms, std::index_sequence<I...> /*unused*/) {
  static_assert(
      sizeof...(I) <= kMaxNumTermsInExpression,
      "Expression side has more terms than kMaxNumTermsInExpression.");
  return std::array<ScaledMatData<Scalar const>, kMaxNumTermsInExpression>{
      std::get<I>(terms).ScaledData()...};
}

template <typename Scalar, typename... T>
auto GetScaledTerms(std::tuple<T...> const& terms) {
  return GetScaledArray<Scalar const>(terms, std::make_index_sequence<sizeof...(T)>());
}

template <typename Scalar>
struct MatDataDest;

void ExecuteCudaAssign(
    MatDataDest<double>&& dest,
    int nPos,
    ScaledMatData<double const> const* posSrc,
    int nNeg,
    ScaledMatData<double const> const* negSrc,
    int rows,
    int cols);

void ExecuteCudaAssign(
    MatDataDest<float>&& dest,
    int nPos,
    ScaledMatData<float const> const* posSrc,
    int nNeg,
    ScaledMatData<float const> const* negSrc,
    int rows,
    int cols);

template <
    typename Scalar,
    DestOp op,
    int kRowStrideAtCompileTime,
    int kColStrideAtCompileTime,
    typename RHS>
void Assign(
    DestinationAccessor<
        op,
        CudaAccessor<Scalar, kRowStrideAtCompileTime, kColStrideAtCompileTime>> const& lhs,
    RHS const& rhs_,
    int rows,
    int cols) {
  RHS& rhs = const_cast<RHS&>(rhs_);
  auto posTuple = PositiveTuple(std::move(rhs));
  auto negTuple = NegativeTuple(std::move(rhs));

  constexpr size_t kNumPos = std::tuple_size_v<decltype(posTuple)>;
  constexpr size_t kNumNeg = std::tuple_size_v<decltype(negTuple)>;
  constexpr size_t kAppendsLhs = (op == DestOp::Add || op == DestOp::Sub) ? 1 : 0;

  //--- Total term count, including the appended LHS for Add/Sub, must fit in the CUDA kernel's
  //--- fixed-size buffer.
  static_assert(
      kNumPos + kNumNeg + kAppendsLhs <= kMaxNumTermsInExpression,
      "Expression has too many terms for CUDA evaluation.");

  size_t numPos = kNumPos;
  size_t numNeg = kNumNeg;
  auto positives = GetScaledTerms<Scalar>(posTuple);
  auto negatives = GetScaledTerms<Scalar>(negTuple);

  if constexpr (op == DestOp::NegSet || op == DestOp::Sub) {
    std::swap(positives, negatives);
    std::swap(numPos, numNeg);
  }
  if constexpr (op == DestOp::Add || op == DestOp::Sub) {
    positives[numPos++] = lhs.accessor.ScaledData();
  }
  ExecuteCudaAssign(
      lhs.accessor.DestData(),
      static_cast<int>(numPos),
      positives.data(),
      static_cast<int>(numNeg),
      negatives.data(),
      rows,
      cols);
}

/**
 * @brief Perform the assignment using left and right-hand-side Accessors
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
void CudaAssign(LHS const& lhs_, RHS const& rhs_, IntOrEmpty<kRows> rows, IntOrEmpty<kCols> cols) {
  if constexpr (std::is_same_v<RHS, SetTag>) {
    // The expression `rhs_` has already been "assigned" to the result memory space.
    return;
  } else {
    // If the rhs has partially filled the lhs must do a += type operation.
    auto const& lhs = After(lhs_, rhs_);
    // If the rhs has partially filled the lhs, the new rhs must be the sum
    auto const& rhs = [](auto const& r) {
      if constexpr (kEvalDidSet<RHS>) {
        return r.accessor;
      } else {
        return r;
      }
    }(rhs_);
    Assign(lhs, rhs, rows.iVal(), cols.iVal());
  }
}

template <typename Mat>
auto CudaGetSimpleAccessor(Mat const& mat) {
  return GetAccessor(mat);
}

template <typename Left, typename Right>
auto CudaGetSimpleAccessor(BinOp<Left, Right, ops::Mul> const& expr) {
  static_assert(
      !kIsMatrixProduct<std::decay_t<decltype(expr)>>,
      "Matrix matrix product has no GetSimpleAccessor");
  return BinaryOpAcc{CudaGetSimpleAccessor(expr.lhs), CudaGetSimpleAccessor(expr.rhs), MultOp{}};
}

template <typename Expr>
auto CudaProductArgAccessor(Expr const& expr) {
  if constexpr (!IsMatrixLike<Expr> || IsSimple<Expr>) {
    return CudaGetSimpleAccessor(expr);
  } else { // For more complex expressions, a temporary matrix must be created.
    using Scalar = expr_scalar_type<Expr>;
    auto rows = expr.CERows();
    auto cols = expr.CECols();
    constexpr int kRowsAtCT = kValueAtCompileTime<decltype(rows)>;
    constexpr int kColsAtCT = kValueAtCompileTime<decltype(cols)>;
    TempAccessor<typename Expr::scalar_type, kRowsAtCT, kColsAtCT, krylov::Ownership::Cuda>
        tmpAccessor(rows.iVal(), cols.iVal());
    auto res = details::CudaEval<DestOp::Set>(expr, tmpAccessor, rows, cols);
    // Compute the result.
    CudaAssign<Scalar>(tmpAccessor, res, rows, cols);
    //
    return std::move(tmpAccessor);
  }
}

} // namespace mochi::details
