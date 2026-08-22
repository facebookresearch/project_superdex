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

#include <mochi_core/linear_algebra/matrix_fwd.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>

#include <type_traits>
#include <utility>

/**
 * @page expression_templates Expression Templates for Matrix Algebra
 * @section et_intro Introduction
 * The goal of Expression Templates (ET) for Matrix Algebra is to allow
 * programmers to write code that is as close to algebra notation
 * as possible, simplifying both code writing and understanding by future
 * readers and maintainers of the code.
 * For example, a mathematical formula such as \f$ y = \alpha z + \beta A x \f$
 * should be written as:
 * \code{.cpp}
   Matrix<float> A;
   Vector<float> x, y, z;
   // ..... code initializing A, x, z
   y = alpha * z + beta * A * x;
 * @endcode
 * The resulting generated code should be as efficient as possible, using SIMD
 * to the full possible extent.
 *
 * @section et_architecture Architecture
 *
 * The way an expression such as `y = alpha * z + beta * A * x;` is implemented
 * is a two step process:
 *   - First, an expression tree of the right hand side is formed
 *   - Secondly, the assignment operator `=` evaluates the tree and assigns
 *   the result of the evaluation for each entry of `y`
 *
 * @subsection et_assign Evaluation and Assignment
 * The mechanism involved in performing the assignment from an expression tree
 * starts with forming an evaluation object of the expression. This evaluation
 * is expected to return an object that conforms to the concept of `Accessor`.
 * An `Accessor` is an object that is equipped with three read operations modes.
 * The first one is to access a single entry in a matrix-like object. The second is to obtain
 * a column SIMD vector and the last one is to obtain a row SIMD vector
 * starting at a given position in the object.
 *
 * For expressions that do not involve any matrix-matrix multiplication, the
 * `Accessor` mechanism is sufficient to fully cover any type of expression.
 * The complete assignment results in the execution of code roughly of this type:
 \code{.cpp}
 auto rightAccessor = eval(expressionTree, ...); // see below for extra arguments
 for (int r = 0; r < m; ++r)
   for (int c = 0; c < n; ++c)
     leftAccessor(r,c) = rightAccessor(r,c);
 @endcode
 In the full implementation, three versions of the nested loops exist:
   - The above purely element-by-element version
   - A row-SIMD based where columns are accessed in groups of N (4 or 8)
   - A column-SIMD based, inverting the row and column loop order and accessing
     row indices in groups of N.
 The choice between the 3 loop versions is based on a constexpr test of the
 estimated cost for the retrieval of SIMD vectors in row or column mode.

 @subsection et_matmatprod Matrix-Matrix Products
 For expressions involving matrix products, the call to `eval` may either create
 a temporary with the result of the expression and return an evaluator owning the
 temporary's memory or write a contribution directly to the destination matrix.
 Since `eval` may need to write directly to the result matrix, three arguments
 are added with the expression, carrying a Destination Accessor (DA) and the row and
 column count of the result matrix.

 The Destination Accessor carries some additional information, as the destination
 may be uninitialized or may already contain a partial result from a previously
 evaluated sub-expression. The sign associated with the product sub-expression within
 the full expression is also carried within the destination accessor. For example,
 in the expression:
 \code{.cpp}
  Y = A * B - C * D;
 @endcode
 The call to `eval` for `A * B` can write to `Y`'s memory with a positive sign,
 overwriting the uninitialized values while the call to `eval` for `C * D` must
 subtract its result from the values already found in `Y`'s memory.

 The down-passing information is carried in a `DestinationAccessor<op, T>` object
 while the up-going information, notifying that the destination contains partial
 data is carried by `AccessorWithSet<T>`.

 One important detail to note, to check the correctness of the code is that
 down-going (in the call sequence) destination accessors are never owning. They
 reference memory owned by an ancestor caller. Thus `DestinationAccessor<op, T>`
 can carry either a reference or a cheap-to-copy object containing a reference.
 However up-going result accessors
 may return a memory owning accessor, such as when a temporary matrix had to be
 created and some values stored into the temporary.
 Thus correct use of move constructors for the `AccessorWithSet<T>` is crucial.

 @subsection et_optim Optimizing for Fixed Sizes

 The complete list of arguments of `eval` is an expression, a destination accessor
 and the number of rows and columns of the result. In order to help the compiler
 generate fully optimal code when the number of rows and columns is known at compile
 time, the type of the row and column count argument is the `IntOrEmpty<value>`
 so that for fixed sizes it is know as a constexpr.

 */

namespace mochi {

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
class SparseMatrix;

template <
    typename Scalar,
    int kBlockSize_,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
class BlockSparseMatrix;

} // namespace mochi

namespace mochi::details {

/// @brief Utility function declaration for trait use to determine
/// the scalar type of the elements of an expression object.
template <typename T, typename S = typename T::scalar_type>
MOCHI_ANY S get_scalar_for(T const*);

MOCHI_ANY float get_scalar_for(float const*);
MOCHI_ANY double get_scalar_for(double const*);

template <typename T>
  requires requires { typename T::Scalar; }
MOCHI_ANY auto get_scalar_for(T const*) -> typename T::Scalar;

/// @brief Trait giving the type of the elements of binary operations between
/// expression objects.
template <typename L, typename R>
using op_scalar_type =
    decltype(details::get_scalar_for((std::decay_t<L>*)nullptr) + details::get_scalar_for((std::decay_t<R>*)nullptr));

template <typename T>
using expr_scalar_type = decltype(get_scalar_for((std::decay_t<T>*)nullptr));

} // namespace mochi::details

namespace mochi::ops {
/// @brief Addition operator tag-class in expressions.
struct Add {};
/// @brief Subtraction operator tag-class in expressions.
struct Sub {};
/// @brief Product operator tag-class in expressions.
struct Mul {};
} // namespace mochi::ops

/// @brief Binary expression.
namespace mochi::details {

/** Trait type to get the domain of an expression or matrix. */
template <typename... T>
struct DomainFor {
  static constexpr ExprDomain domain = ExprDomain::Unknown;
};

template <typename T>
concept HasExprDomain = requires { T::domain; };

template <HasExprDomain T>
struct DomainFor<T> {
  static constexpr ExprDomain domain = std::decay_t<T>::domain;
};

template <HasExprDomain L, HasExprDomain R>
struct DomainFor<L, R> {
  static constexpr ExprDomain domain = std::decay_t<L>::domain & std::decay_t<R>::domain;
};

template <IsArithmetic L, HasExprDomain R>
struct DomainFor<L, R> {
  static constexpr ExprDomain domain = std::decay_t<R>::domain;
};

template <HasExprDomain L, IsArithmetic R>
struct DomainFor<L, R> {
  static constexpr ExprDomain domain = std::decay_t<L>::domain;
};

/**
 * @brief Binary expression combining two sub-expressions.
 *
 * @tparam LHS Type of the Left Hand Side.
 * @tparam RHS Type of the Right Hand Side.
 * @tparam Op Tag type for the binary operator.
 */
template <typename LHS, typename RHS, typename Op>
struct BinOp : public mochi::NewMatrix, public mochi::NewMatrixExpr {
  using scalar_type = op_scalar_type<LHS, RHS>;
  static constexpr ExprDomain domain = DomainFor<LHS, RHS>::domain;

  template <typename L, typename R, typename O>
  MOCHI_ANY BinOp(L&& l, R&& r, O&& o) : lhs(std::forward<L>(l)), rhs(std::forward<R>(r)), op(o) {
    if constexpr (IsAnyMatrix<LHS> && IsAnyMatrix<RHS>) {
      if constexpr (std::is_same_v<Op, ops::Add> || std::is_same_v<Op, ops::Sub>) {
        MOCHI_ASSERT_VERBOSE(
            lhs.CERows().iVal() == rhs.CERows().iVal() &&
                lhs.CECols().iVal() == rhs.CECols().iVal(),
            "Inconsistent matrix sizes");
      } else if constexpr (std::is_same_v<Op, ops::Mul>) {
        MOCHI_ASSERT_VERBOSE(
            lhs.CECols().iVal() == rhs.CERows().iVal(), "Inconsistent matrix sizes");
      }
    }
  }

  [[nodiscard]] MOCHI_ANY constexpr auto CERows() const {
    if constexpr (IsAnyMatrix<LHS>) {
      return lhs.CERows();
    } else {
      static_assert(std::is_same_v<LHS const, scalar_type const>, "Case not supported yet");
      return rhs.CERows();
    }
  }

  [[nodiscard]] MOCHI_ANY constexpr auto CECols() const {
    if constexpr (IsAnyMatrix<RHS>) {
      return rhs.CECols();
    } else {
      static_assert(std::is_same_v<RHS const, scalar_type const>, "Case not supported yet");
      return lhs.CECols();
    }
  }

  LHS lhs;
  RHS rhs;
  Op op;
};

template <typename L, typename R, typename O>
BinOp(L&& l, R&& r, O&& o) -> BinOp<L, R, std::remove_cv_t<O>>;

/**
 * @brief Expression with a factor, separating both.
 * @details Scaled expression collects the product of all factors in
 * a sequence of products such as alpha*A*beta*B, transforming it
 * into scale=(alpha*beta) and Expr the expression for A*B
 * @tparam Scalar
 * @tparam Expr
 */
template <typename Scalar, typename Expr>
struct ScaledExpr : NewMatrix, NewMatrixExpr {
  using scalar_type = op_scalar_type<Scalar, Expr>;
  static constexpr ExprDomain domain = DomainFor<Expr>::domain;

  template <typename E>
  MOCHI_ANY ScaledExpr(Scalar s, E&& expr) : scale(s), matExpr(std::forward<E>(expr)) {}

  [[nodiscard]] MOCHI_ANY constexpr auto CERows() const {
    return matExpr.CERows();
  }
  [[nodiscard]] MOCHI_ANY constexpr auto CECols() const {
    return matExpr.CECols();
  }
  Scalar scale;
  Expr matExpr;
};

template <typename Scalar, typename E>
ScaledExpr(Scalar s, E&& expr) -> ScaledExpr<Scalar, E>;

template <typename Scalar, typename T>
MOCHI_ANY Scalar GetExprScale(T const& /*unused*/) {
  return Scalar(1);
}

template <typename T, typename Scalar, typename Expr>
MOCHI_ANY T GetExprScale(details::ScaledExpr<Scalar, Expr> const& scaledExpr) {
  return scaledExpr.scale;
}

/// @brief Result of an expression of type `-A`
template <typename ARG>
struct UnaryNeg : public NewMatrix, public mochi::NewMatrixExpr {
  using scalar_type = expr_scalar_type<ARG>;

  template <typename A>
  MOCHI_ANY UnaryNeg(A&& a) : arg(std::forward<A>(a)) {}
  [[nodiscard]] MOCHI_ANY constexpr auto CERows() const {
    return arg.CERows();
  }
  [[nodiscard]] MOCHI_ANY constexpr auto CECols() const {
    return arg.CECols();
  }
  ARG arg;
};

template <typename Scalar, typename T>
MOCHI_ANY Scalar GetExprScale(details::UnaryNeg<T> const& /*unused*/) {
  return Scalar{-1};
}

} // namespace mochi::details

namespace mochi {
template <typename LHS, typename RHS>
  requires(IsMatrixLike<LHS> || IsMatrixLike<RHS>)
auto operator+(LHS&& l, RHS&& r) {
  // Universal references will give a reference T & for lvalues of type T and T
  // itself for a rvalue. This ensures that the final expression does hold a
  // copy of temporaries, such as in expressions of the type `auto expression =
  // A + f();` LHS will be Matrix & while RHS will be Matrix
  return details::BinOp<LHS, RHS, ops::Add>{std::forward<LHS>(l), std::forward<RHS>(r), ops::Add{}};
}

template <typename LHS, typename RHS>
  requires(IsMatrixLike<LHS> || IsMatrixLike<RHS>)
MOCHI_ANY auto operator-(LHS&& l, RHS&& r) {
  return details::BinOp<LHS, RHS, ops::Sub>{std::forward<LHS>(l), std::forward<RHS>(r), ops::Sub{}};
}

template <typename ScalarType, typename MatrixType>
MOCHI_ANY auto BuildScalarMatProduct(ScalarType a, MatrixType&& M) {
  return details::ScaledExpr<ScalarType, MatrixType>{a, std::forward<MatrixType>(M)};
}

template <typename ScalarType, typename Sc2, typename ExprType>
MOCHI_ANY auto BuildScalarMatProduct(ScalarType a, details::ScaledExpr<Sc2, ExprType> P) {
  using ResultScalar = std::common_type_t<ScalarType, Sc2>;

  if constexpr (std::is_reference_v<ExprType>) {
    return details::ScaledExpr<ResultScalar, ExprType>{a * P.scale, P.matExpr};
  } else {
    return details::ScaledExpr<ResultScalar, ExprType>{a * P.scale, std::move(P.matExpr)};
  }
}

template <typename T>
constexpr bool IsScaledExprBase = false;

template <typename Scalar, typename Expr>
constexpr bool IsScaledExprBase<details::ScaledExpr<Scalar, Expr>> = true;

template <typename T>
constexpr bool IsUnaryNegBase = false;

template <typename T>
constexpr bool IsUnaryNegBase<details::UnaryNeg<T>> = true;

template <typename T>
constexpr bool IsScaledExpr = IsScaledExprBase<std::decay_t<T>> || IsUnaryNegBase<std::decay_t<T>>;

template <typename T>
  requires(!IsScaledExpr<T>)
MOCHI_ANY T&& GetExpr(T&& t) {
  return static_cast<T&&>(t);
}

template <typename Scalar, typename Expr>
MOCHI_ANY Expr&& GetExpr(details::ScaledExpr<Scalar, Expr>&& scaledExpr) {
  // Reference collapsing rule states that if matExpr was a reference,
  // the result is a reference. If it was an object, the result is
  // a right value reference (movable object).
  return static_cast<Expr&&>(scaledExpr.matExpr);
}

template <typename Scalar, typename Expr>
MOCHI_ANY Expr const& GetExpr(details::ScaledExpr<Scalar, Expr> const& scaledExpr) {
  return scaledExpr.matExpr;
}

template <typename Expr>
MOCHI_ANY Expr&& GetExpr(details::UnaryNeg<Expr>&& negExpr) {
  // Reference collapsing rule states that if matExpr was a reference,
  // the result is a reference. If it was an object, the result is
  // a right value reference (movable object).
  return static_cast<Expr&&>(negExpr.arg);
}

template <typename Expr>
MOCHI_ANY Expr const& GetExpr(details::UnaryNeg<Expr> const& negExpr) {
  return negExpr.arg;
}

template <typename LHS, typename RHS>
  requires(IsMatrixLike<LHS> || IsMatrixLike<RHS>)
MOCHI_ANY auto operator*(LHS&& l, RHS&& r) {
  if constexpr (std::is_arithmetic_v<std::decay_t<LHS>>) {
    return BuildScalarMatProduct(l, std::forward<RHS>(r));
  } else if constexpr (std::is_arithmetic_v<std::decay_t<RHS>>) {
    return BuildScalarMatProduct(r, std::forward<LHS>(l));
  } else if constexpr (IsScaledExpr<LHS> || IsScaledExpr<RHS>) {
    using Scalar = details::op_scalar_type<LHS, RHS>;
    return details::ScaledExpr{
        details::GetExprScale<Scalar>(l) * details::GetExprScale<Scalar>(r),
        details::BinOp{GetExpr(std::forward<LHS>(l)), GetExpr(std::forward<RHS>(r)), ops::Mul{}}};
  } else {
    return details::BinOp<LHS, RHS, ops::Mul>{
        std::forward<LHS>(l), std::forward<RHS>(r), ops::Mul{}};
  }
}

/// Unary negation operator for matrices and matrix expressions.
template <IsMatrixLike LHS>
MOCHI_ANY auto operator-(LHS&& l) {
  return details::UnaryNeg<LHS>{std::forward<LHS>(l)};
}

namespace details {

template <typename T>
MOCHI_ANY constexpr auto GetDomainFor(T const&) {
  if constexpr (IsStridedMatrix<T>) {
    return DomainType<ExprDomain::Strided>{};
  } else {
    return DomainType<ExprDomain::Unknown>{};
  }
}

template <typename S, typename T>
MOCHI_ANY constexpr auto GetDomainFor(ScaledExpr<S, T> const& e) {
  return GetDomainFor(e.matExpr);
}

template <typename L, typename R, typename O>
MOCHI_ANY constexpr auto GetDomainFor(BinOp<L, R, O> const& e) {
  return GetDomainFor(e.lhs) & GetDomainFor(e.rhs);
}

} // namespace details

} // namespace mochi
