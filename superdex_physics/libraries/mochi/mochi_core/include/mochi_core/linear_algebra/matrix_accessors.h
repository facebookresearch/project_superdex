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

#include <mochi_core/linear_algebra/base_tools.h>
#include <mochi_core/linear_algebra/matrix_accessors_fwd.h>
#include <mochi_core/linear_algebra/matrix_fwd.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/simd.h>

#include <type_traits>
#include <utility>

namespace mochi::details {

using namespace krylov::details;

/// std::pair is not cleanly supported by device code on CUDA.
struct CostPair {
  int first = 0;
  int second = 0;
};

template <typename Scalar, int kRowStrideAtCompileTime, int kColStrideAtCompileTime>
struct BasicAccessor {
  using scalar_type = Scalar;

  static constexpr int kGivenRowStride = kRowStrideAtCompileTime;
  static constexpr int kGivenColStride = kColStrideAtCompileTime;

  /// Delete default constructor
  BasicAccessor() = delete;

  /// @brief Constructor for compile time strides in both directions.
  /// @param v Pointer to the element at (0,0).
  explicit BasicAccessor(Scalar* v) : _v(v) {
    static_assert(
        kGivenRowStride > 0 && kGivenColStride > 0,
        "Use of incorrect constructor for dynamic size accessor");
  }
  /// @brief Constructor when the stride in at least one direction is a runtime value.
  /// @param v Pointer to the element at (0,0).
  /// @param rowStride Row stride.
  /// @param colStride Column stride.
  BasicAccessor(Scalar* v, int rowStride, int colStride)
      : _v(v), _rowStride(rowStride), _colStride(colStride) {}

  Scalar const* Data() const {
    return _v;
  }

  ScaledMatData<Scalar const> ScaledData() const {
    Scalar one(1);
    auto* ptr = Data();
    int rs = _rowStride.iVal();
    int cs = _colStride.iVal();
    return {one, ptr, rs, cs};
  }

  MatDataDest<Scalar> DestData() const {
    return {const_cast<Scalar*>(_v), _rowStride.iVal(), _colStride.iVal()};
  }

 private:
  Scalar* _v = nullptr;

  IntOrEmpty<kGivenRowStride> _rowStride;
  IntOrEmpty<kGivenColStride> _colStride;
};

/// @brief Load N values into an SIMD vector given a pointer and the stride between values.
/// @tparam VType SIMD vector type
/// @tparam N0 Number of values to load. Must be -1, or between 1 and the SIMD vector size. -1
/// indicates to load as many values as the SIMD vector size.
/// @tparam kStride Stride or -1 if not a compile-time constant.
/// @param v Pointer to the first source element.
/// @param s Stride value.
template <typename VType, int N0 = -1, int kStride>
MOCHI_ANY auto GetSimd(typename VType::Scalar const* v, IntOrEmpty<kStride> s) {
  constexpr int N = (N0 == -1) ? VType::kSize : N0;
  static_assert(N > 0 && N <= VType::kSize, "Inconsistent N");
  if constexpr (kStride == 1) {
    return Load<N, VType>(v);
  } else {
    auto stride = s.iVal();
    if (stride == 1) {
      // Load N consecutive values.
      return Load<N, VType>(v);
    } else {
      if constexpr (N == 1) {
        // Load 1 value, followed by (VType::kSize - 1) zeros.
        using Scalar = typename VType::Scalar;
        return VType{v[0], Scalar(0)};
      } else {
        // Load N values with a stride, followed by (VType::kSize - N) zeros.
        auto eval = [=]<size_t... I>(std::index_sequence<I...>) {
          return VType{(v[I * stride])...};
        };
        return eval(std::make_index_sequence<N>{});
      }
    }
  }
}

/// @brief Overload for runtime N. Unlike the compile-time overload, N = -1 is illegal.
template <typename VType, int kStride>
MOCHI_ANY auto GetSimd(typename VType::Scalar const* v, int N, IntOrEmpty<kStride> s) {
  using Scalar = typename VType::Scalar;
  MOCHI_ASSERT_VERBOSE(N >= 0 && N <= VType::kSize, "Unsupported N");
  constexpr auto kVecSize = VType::kSize;
  if constexpr (kStride == 1) {
    return Load<VType>(v, N);
  } else {
    auto stride = s.iVal();
    if (stride == 1) {
      return Load<VType>(v, N);
    } else {
      // Load N values with a stride.
      auto eval = [=]<size_t... I>(std::index_sequence<I...>) {
        return VType{(N > I ? v[I * stride] : Scalar(0))...};
      };
      return eval(std::make_index_sequence<kVecSize>{});
    }
  }
}

/// @brief Store N values in an SIMD vector given a pointer and the stride between values.
/// @tparam N0 Number of values to store. Must be -1, or between 1 and the SIMD vector size. -1
/// indicates to store as many values as the SIMD vector size.
/// @tparam VType SIMD vector type
/// @tparam kStride Stride or -1 if not a compile-time constant.
/// @param d Pointer to the first destination element.
/// @param v SIMD vector with the source data.
/// @param s Stride value.
template <int N0 = -1, typename VType, int kStride>
MOCHI_ANY auto StoreSimd(typename VType::Scalar* d, VType v, IntOrEmpty<kStride> s) {
  constexpr int N = (N0 == -1) ? VType::kSize : N0;
  static_assert(N > 0 && N <= VType::kSize, "Inconsistent N");
  using Scalar = typename VType::Scalar;
  if constexpr (kStride == 1) {
    return Store<N>(d, v);
  } else {
    auto stride = s.iVal();
    if (stride == 1) {
      return Store<N>(d, v);
    }
    Scalar vtmp[VType::kSize];
    Store(&vtmp[0], v);
    for (int i = 0; i < N; ++i) {
      d[i * stride] = vtmp[i];
    }
    return;
  }
}

/// @brief Overload for runtime N. Unlike the compile-time overload, N = -1 is illegal.
template <typename VType, int kStride>
MOCHI_ANY auto StoreSimd(typename VType::Scalar* d, VType v, int N, IntOrEmpty<kStride> s) {
  using Scalar = typename VType::Scalar;
  MOCHI_ASSERT_VERBOSE(N >= 0 && N <= VType::kSize, "Unsupported N");
  if constexpr (kStride == 1) {
    return Store(d, v, N);
  } else {
    auto stride = s.iVal();
    if (stride == 1) {
      return Store(d, v, N);
    }
    if (N > 0) {
      Scalar vtmp[VType::kSize];
      Store(&vtmp[0], v);
      for (int i = 0; i < N; ++i) {
        d[i * stride] = vtmp[i];
      }
    }
    return;
  }
}

/// @brief Accessor object for retrieving/storing data from/to a matrix with given strides between
/// elements.
/// @tparam Scalar The scalar type of the matrix.
/// @tparam kRowStrideAtCompileTime Row stride at compile time or -1 if runtime value.
/// @tparam kColStrideAtCompileTime Column stride at compile time or -1 if runtime value.
template <typename Scalar, int kRowStrideAtCompileTime, int kColStrideAtCompileTime>
struct Accessor {
  using scalar_type = Scalar;
  static constexpr int kGivenRowStride = kRowStrideAtCompileTime;
  static constexpr int kGivenColStride = kColStrideAtCompileTime;

  /// @brief Constructor for compile time strides in both directions.
  /// @param v Pointer to the element at (0,0)
  explicit Accessor(Scalar* v) : _v(v) {
    static_assert(
        kGivenRowStride > 0 && kGivenColStride > 0,
        "Use of incorrect constructor for dynamic size accessor");
  }

  /// @brief Constructor when the stride in at least one direction is a runtime value.
  /// @param v Pointer to the element at (0,0)
  /// @param rowStride Row stride.
  /// @param colStride Column stride.
  Accessor(Scalar* v, int rowStride, int colStride)
      : _v(v), _rowStride(rowStride), _colStride(colStride) {}

  Accessor(
      Scalar* v,
      IntOrEmpty<kRowStrideAtCompileTime> rowStride,
      IntOrEmpty<kColStrideAtCompileTime> colStride) noexcept
      : Accessor(v, rowStride.iVal(), colStride.iVal()) {}

  friend auto Transpose(Accessor const& a) {
    return Accessor<Scalar, kColStrideAtCompileTime, kRowStrideAtCompileTime>(
        a._v, a._colStride, a._rowStride);
  }

  inline Scalar* Data() {
    return _v;
  }

  inline constexpr Scalar const* Data() const {
    return _v;
  }

  inline Scalar* ptr(int r, int c) const {
    return _v + static_cast<size_t>(r) * _rowStride.sVal() +
        static_cast<size_t>(c) * _colStride.sVal();
  }

  inline constexpr Scalar const& operator()(int r, int c) const {
    return _v[r * _rowStride.sVal() + c * _colStride.sVal()];
  }

  inline constexpr void Store(int r, int c, Scalar const& value) const {
    *ptr(r, c) = value;
  }

  template <typename VType, int N = -1>
  inline auto RowVector(int r, int c) const {
    static_assert(
        std::is_same_v<std::remove_const_t<Scalar>, typename VType::Scalar>,
        "Inconsistent scalar types");
    return GetSimd<VType, N>(ptr(r, c), _colStride);
  }

  template <typename VType>
  inline auto RowVector(int r, int c, int N) const {
    static_assert(
        std::is_same_v<std::remove_const_t<Scalar>, typename VType::Scalar>,
        "Inconsistent scalar types");
    return GetSimd<VType>(ptr(r, c), N, _colStride);
  }

  template <typename VType, int N = -1>
  inline auto ColVector(int r, int c) const {
    static_assert(
        std::is_same_v<std::remove_const_t<Scalar>, typename VType::Scalar>,
        "Inconsistent scalar types");
    return GetSimd<VType, N>(ptr(r, c), _rowStride);
  }

  template <typename VType>
  inline auto ColVector(int r, int c, int N) const {
    static_assert(
        std::is_same_v<std::remove_const_t<Scalar>, typename VType::Scalar>,
        "Inconsistent scalar types");
    return GetSimd<VType>(ptr(r, c), N, _rowStride);
  }

  template <int N = -1, typename VType>
  inline auto StoreRowVector(int r, int c, VType vec) const {
    static_assert(std::is_same_v<Scalar, typename VType::Scalar>, "Inconsistent scalar types");
    return StoreSimd<N>(ptr(r, c), vec, _colStride);
  }

  template <typename VType>
  inline auto StoreRowVector(int r, int c, VType vec, int N) const {
    static_assert(std::is_same_v<Scalar, typename VType::Scalar>, "Inconsistent scalar types");
    return StoreSimd(ptr(r, c), vec, N, _colStride);
  }

  template <int N = -1, typename VType>
  inline auto StoreColVector(int r, int c, VType vec) const {
    static_assert(std::is_same_v<Scalar, typename VType::Scalar>, "Inconsistent scalar types");
    return StoreSimd<N>(ptr(r, c), vec, _rowStride);
  }

  template <typename VType>
  inline auto StoreColVector(int r, int c, VType vec, int N) const {
    static_assert(std::is_same_v<Scalar, typename VType::Scalar>, "Inconsistent scalar types");
    return StoreSimd(ptr(r, c), vec, N, _rowStride);
  }

  constexpr static CostPair RowColCosts() {
    return {kRowStrideAtCompileTime == 1 ? 1 : 4, kColStrideAtCompileTime == 1 ? 1 : 4};
  }

  [[nodiscard]] int RowStride() const {
    return _rowStride.iVal();
  }

  [[nodiscard]] int ColStride() const {
    return _colStride.iVal();
  }

 private:
  Scalar* _v;
  IntOrEmpty<kGivenRowStride> _rowStride;
  IntOrEmpty<kGivenColStride> _colStride;
};

template <typename A>
constexpr bool IsMemoryAccessor = false;

template <typename Scalar, int kR, int kC>
constexpr bool IsMemoryAccessor<Accessor<Scalar, kR, kC>> = true;

/// @brief Get a read/write SIMD accessor for a Matrix.
template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDim>
auto GetAccessor(
    Matrix<
        Scalar,
        kRowsAtCompileTime,
        kColsAtCompileTime,
        kMajorDirection,
        kOwnership,
        kLeadingDim>& A) {
  // NOLINTBEGIN(bugprone-branch-clone) - Both if and else results may be the same. That's OK.
  constexpr int kLeadDimACT = kLeadingDim == krylov::kAutomaticLeadDim
      ? (kMajorDirection == krylov::Direction::ColMajor ? kRowsAtCompileTime : kColsAtCompileTime)
      : kLeadingDim;
  // NOLINTEND(bugprone-branch-clone)
  constexpr int kRowStrideACT = kMajorDirection == krylov::Direction::ColMajor ? 1 : kLeadDimACT;
  constexpr int kColStrideACT = kMajorDirection == krylov::Direction::RowMajor ? 1 : kLeadDimACT;
  int rowStride = kRowStrideACT == 1 ? 1 : A.LeadDim();
  int colStride = kColStrideACT == 1 ? 1 : A.LeadDim();
  if constexpr (krylov::IsCuda(kOwnership)) {
    // Cuda ownership can not use SIMD and does not accept entry-wise access (i.e. A(r,c))
    // The `BasicAccessor` provides access only to the memory pointer.
    return BasicAccessor<Scalar, kRowStrideACT, kColStrideACT>(A.Data(), rowStride, colStride);
  } else {
    return Accessor<Scalar, kRowStrideACT, kColStrideACT>(A.Data(), rowStride, colStride);
  }
}

/// @brief Get a read-only SIMD accessor for a Matrix.
template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDim>
auto GetAccessor(
    Matrix<
        Scalar,
        kRowsAtCompileTime,
        kColsAtCompileTime,
        kMajorDirection,
        kOwnership,
        kLeadingDim> const& A) {
  // NOLINTBEGIN(bugprone-branch-clone) - Both if and else results may be the same. That's OK.
  constexpr int kLeadDimACT = kLeadingDim == krylov::kAutomaticLeadDim
      ? (kMajorDirection == krylov::Direction::ColMajor ? kRowsAtCompileTime : kColsAtCompileTime)
      : kLeadingDim;
  // NOLINTEND(bugprone-branch-clone)
  constexpr int kRowStrideACT = kMajorDirection == krylov::Direction::ColMajor ? 1 : kLeadDimACT;
  constexpr int kColStrideACT = kMajorDirection == krylov::Direction::RowMajor ? 1 : kLeadDimACT;
  int rowStride = kRowStrideACT == 1 ? 1 : A.LeadDim();
  int colStride = kColStrideACT == 1 ? 1 : A.LeadDim();
  if constexpr (krylov::IsCuda(kOwnership)) {
    // Cuda ownership can not use SIMD and does not accept entry-wise access (i.e. A(r,c))
    // The `BasicAccessor` provides access only to the memory pointer.
    return BasicAccessor<Scalar const, kRowStrideACT, kColStrideACT>(
        A.Data(), rowStride, colStride);
  } else {
    return Accessor<Scalar const, kRowStrideACT, kColStrideACT>(A.Data(), rowStride, colStride);
  }
}

/// @brief Accessor for scalar values. Can create SIMD vectors with the same value repeated in all
/// SIMD elements.
template <typename Scalar>
struct ScalarAccessor {
  Scalar v;

  MOCHI_ANY MOCHI_FORCE_INLINE constexpr Scalar operator()(int /*unused*/, int /*unused*/) const {
    return v;
  }

  template <typename VTypeOut, int N0 = -1>
  MOCHI_ANY MOCHI_FORCE_INLINE auto RowVector(int /*r*/, int /*c*/) const {
    constexpr int N = (N0 == -1) ? VTypeOut::kSize : N0;
    static_assert(N == VTypeOut::kSize, "Unsupported case"); // Generic implementation: P890637484
    using ScalarOut = typename VTypeOut::Scalar;
    return VTypeOut(ScalarOut(v));
  }

  template <typename VTypeOut, int N0 = -1>
  MOCHI_ANY MOCHI_FORCE_INLINE auto ColVector(int r, int c) const {
    return RowVector<VTypeOut, N0>(r, c);
  }

  MOCHI_ANY MOCHI_FORCE_INLINE constexpr static CostPair RowColCosts() {
    return {0, 0};
  }

  MOCHI_ANY MOCHI_FORCE_INLINE friend auto Transpose(ScalarAccessor& sa) {
    return sa;
  }
};

/// @brief Accessor returning the negated value of a given data.
/// @tparam Accessor The accessor to the values to be negated.
template <typename Accessor>
struct NegateAccessor {
  Accessor a;

  MOCHI_ANY NegateAccessor(Accessor t) : a(std::move(t)) {}

  MOCHI_ANY MOCHI_FORCE_INLINE constexpr auto const* Data() const {
    return a.Data();
  }

  MOCHI_ANY MOCHI_FORCE_INLINE constexpr auto operator()(int r, int c) const {
    return -a(r, c);
  }

  template <typename VType, int N = -1>
  MOCHI_ANY MOCHI_FORCE_INLINE auto RowVector(int r, int c) const {
    return -(a.template RowVector<VType, N>(r, c));
  }

  template <typename VType>
  MOCHI_ANY MOCHI_FORCE_INLINE auto RowVector(int r, int c, int N) const {
    return -(a.template RowVector<VType>(r, c, N));
  }

  template <typename VType, int N = -1>
  MOCHI_ANY MOCHI_FORCE_INLINE auto ColVector(int r, int c) const {
    return -(a.template ColVector<VType, N>(r, c));
  }

  template <typename VType>
  MOCHI_ANY MOCHI_FORCE_INLINE auto ColVector(int r, int c, int N) const {
    return -(a.template ColVector<VType>(r, c, N));
  }

  MOCHI_ANY MOCHI_FORCE_INLINE constexpr static CostPair RowColCosts() {
    return Accessor::RowColCosts();
  }

  /// @details Transpose will contain references, it is the responsibility of the caller
  /// to guarantee object lifetime.
  MOCHI_ANY MOCHI_FORCE_INLINE friend auto Transpose(NegateAccessor const& neg) {
    using TType = decltype(Transpose(neg.a));
    return NegateAccessor<TType>{Transpose(neg.a)};
  }
};

/// @brief Accessor for a binary element-by-element expression.
/// @tparam LHS Left Hand Side Accessor type
/// @tparam RHS Right Hand Side Accessor type
/// @tparam Op Operator type
template <typename LHS, typename RHS, typename Op>
struct BinaryOpAcc {
  LHS lhs;
  RHS rhs;
  Op op;

  MOCHI_ANY BinaryOpAcc(LHS lhs, RHS rhs, Op op)
      : lhs(std::move(lhs)), rhs(std::move(rhs)), op(std::move(op)) {}

  MOCHI_ANY MOCHI_FORCE_INLINE constexpr auto operator()(int r, int c) const {
    return op(lhs(r, c), rhs(r, c));
  }

  template <typename VTypeOut, int N = -1>
  MOCHI_ANY MOCHI_FORCE_INLINE auto RowVector(int r, int c) const {
    return op(lhs.template RowVector<VTypeOut, N>(r, c), rhs.template RowVector<VTypeOut, N>(r, c));
  }

  template <typename VTypeOut>
  MOCHI_ANY MOCHI_FORCE_INLINE auto RowVector(int r, int c, int N) const {
    return op(lhs.template RowVector<VTypeOut>(r, c, N), rhs.template RowVector<VTypeOut>(r, c, N));
  }

  template <typename VTypeOut, int N = -1>
  MOCHI_ANY MOCHI_FORCE_INLINE auto ColVector(int r, int c) const {
    return op(lhs.template ColVector<VTypeOut, N>(r, c), rhs.template ColVector<VTypeOut, N>(r, c));
  }

  template <typename VTypeOut>
  MOCHI_ANY MOCHI_FORCE_INLINE auto ColVector(int r, int c, int N) const {
    return op(lhs.template ColVector<VTypeOut>(r, c, N), rhs.template ColVector<VTypeOut>(r, c, N));
  }

  MOCHI_ANY MOCHI_FORCE_INLINE constexpr static CostPair RowColCosts() {
    auto leftCosts = std::decay_t<LHS>::RowColCosts();
    auto rightCosts = std::decay_t<RHS>::RowColCosts();
    return {leftCosts.first + rightCosts.first, leftCosts.second + rightCosts.second};
  }
  /// @details There is no BinaryOpAcc with two matrix accessors, so no need to worry about
  /// operand order.
  MOCHI_ANY MOCHI_FORCE_INLINE friend auto Transpose(BinaryOpAcc const& boa) {
    return BinaryOpAcc(Transpose(boa.lhs), Transpose(boa.rhs), boa.op);
  }
};

/// @brief Operator for element summation.
struct SumOp {
  template <typename L, typename R>
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr auto operator()(L const& l, R const& r) const {
    return l + r;
  }
};

/// @brief Operator for element subtraction.
struct SubtractOp {
  template <typename L, typename R>
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr auto operator()(L const& l, R const& r) const {
    return l - r;
  }
};

/// @brief Operator for element multiplication
struct MultOp {
  template <typename L, typename R>
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr auto operator()(L const& l, R const& r) const {
    return l * r;
  }
};

/// @brief Memory owning Accessor for a temporary result, such as when matrix-matrix products must
/// be evaluated.
/// @tparam Scalar
/// @tparam kRowsAtCompileTime
/// @tparam kColsAtCompileTime
/// @tparam kOwnership
template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Ownership kOwnership = krylov::Ownership::Owner>
struct TempAccessor {
  // For simplicity, the temporary object will be column major.
  // A different orientation would require an additional template parameter.
  static constexpr int kGivenRowStride = 1;
  static constexpr int kGivenColStride = kRowsAtCompileTime;

  TempAccessor(int rows, int cols) : _rowStride(kGivenRowStride), _colStride(rows) {
    if constexpr (kRowsAtCompileTime < 0 || kColsAtCompileTime < 0) {
      size_t const len = static_cast<size_t>(rows) * static_cast<size_t>(cols);
      _values.Resize(len);
    }
  }

  MOCHI_FORCE_INLINE constexpr Scalar const* Data() const {
    return _values.Data();
  }

  /// @brief Provide entry-wise access to the temporary storage
  MOCHI_FORCE_INLINE Scalar* ptr(int r, int c) const {
    static_assert(!krylov::IsCuda(kOwnership), "Cuda ownership does not allow entry-wise access");
    return const_cast<Scalar*>(_values.data()) + static_cast<size_t>(r) * _rowStride.sVal() +
        static_cast<size_t>(c) * _colStride.sVal();
  }

  /// @brief Provide entry-wise read access to the temporary storage
  MOCHI_FORCE_INLINE constexpr Scalar const& operator()(int r, int c) const {
    static_assert(!krylov::IsCuda(kOwnership), "Cuda ownership does not allow entry-wise access");
    return const_cast<Scalar*>(_values.data())[r * _rowStride.sVal() + c * _colStride.sVal()];
  }

  /// @brief Update entry in to the temporary storage
  MOCHI_FORCE_INLINE constexpr void Store(int r, int c, Scalar const& v) const {
    static_assert(!krylov::IsCuda(kOwnership), "Cuda ownership does not allow entry-wise access");
    *ptr(r, c) = v;
  }

  /// @brief Provide (sub) row vector of length N (<= SIMD size) starting at (r, c)
  template <typename VType, int N = -1>
  MOCHI_FORCE_INLINE auto RowVector(int r, int c) const {
    static_assert(!krylov::IsCuda(kOwnership), "Cuda ownership does not allow entry-wise access");
    static_assert(
        std::is_same_v<std::remove_const_t<Scalar>, typename VType::Scalar>,
        "Inconsistent scalar types");
    return GetSimd<VType, N>(ptr(r, c), _colStride);
  }

  template <typename VType>
  MOCHI_FORCE_INLINE auto RowVector(int r, int c, int N) const {
    static_assert(!krylov::IsCuda(kOwnership), "Cuda ownership does not allow entry-wise access");
    static_assert(
        std::is_same_v<std::remove_const_t<Scalar>, typename VType::Scalar>,
        "Inconsistent scalar types");
    return GetSimd<VType>(ptr(r, c), N, _colStride);
  }

  /// @brief Provide (sub) column vector of length N (<= SIMD size) starting at (r, c)
  template <typename VType, int N = -1>
  MOCHI_FORCE_INLINE auto ColVector(int r, int c) const {
    static_assert(!krylov::IsCuda(kOwnership), "Cuda ownership does not allow entry-wise access");
    static_assert(
        std::is_same_v<std::remove_const_t<Scalar>, typename VType::Scalar>,
        "Inconsistent scalar types");
    return GetSimd<VType, N>(ptr(r, c), _rowStride);
  }

  template <typename VType>
  MOCHI_FORCE_INLINE auto ColVector(int r, int c, int N) const {
    static_assert(!krylov::IsCuda(kOwnership), "Cuda ownership does not allow entry-wise access");
    static_assert(
        std::is_same_v<std::remove_const_t<Scalar>, typename VType::Scalar>,
        "Inconsistent scalar types");
    return GetSimd<VType>(ptr(r, c), N, _rowStride);
  }

  /// @brief Store the first N values of a SIMD vector in a (sub) row vector starting at (r, c).
  /// N = -1 indicates to store all the values in the SIMD vector.
  /// @note Cuda ownership does not accept entry-wise access (i.e. A(r,c)).
  /// This function is de-activated with Ownership::Cuda.
  template <int N = -1, typename VType>
  MOCHI_FORCE_INLINE auto StoreRowVector(int r, int c, VType vec) const {
    static_assert(!krylov::IsCuda(kOwnership), "Cuda ownership does not allow entry-wise access");
    static_assert(std::is_same_v<Scalar, typename VType::Scalar>, "Inconsistent scalar types");
    return StoreSimd<N>(ptr(r, c), vec, _colStride);
  }

  /// @brief Overload for runtime N. Unlike the compile-time overload, N = -1 is illegal.
  template <typename VType>
  MOCHI_FORCE_INLINE auto StoreRowVector(int r, int c, VType vec, int N) const {
    static_assert(!krylov::IsCuda(kOwnership), "Cuda ownership does not allow entry-wise access");
    static_assert(std::is_same_v<Scalar, typename VType::Scalar>, "Inconsistent scalar types");
    return StoreSimd(ptr(r, c), vec, N, _colStride);
  }

  /// @brief Store the first N values of a SIMD vector in a (sub) column vector starting at (r, c).
  /// N = -1 indicates to store all the values in the SIMD vector.
  /// @note Cuda ownership does not accept entry-wise access (i.e. A(r,c)).
  /// This function is de-activated with Ownership::Cuda.
  template <int N = -1, typename VType>
  MOCHI_FORCE_INLINE auto StoreColVector(int r, int c, VType vec) const {
    static_assert(!krylov::IsCuda(kOwnership), "Cuda ownership does not allow entry-wise access");
    static_assert(std::is_same_v<Scalar, typename VType::Scalar>, "Inconsistent scalar types");
    return StoreSimd<N>(ptr(r, c), vec, _rowStride);
  }

  /// @brief Overload for runtime N. Unlike the compile-time overload, N = -1 is illegal.
  template <typename VType>
  MOCHI_FORCE_INLINE auto StoreColVector(int r, int c, VType vec, int N) const {
    static_assert(!krylov::IsCuda(kOwnership), "Cuda ownership does not allow entry-wise access");
    static_assert(std::is_same_v<Scalar, typename VType::Scalar>, "Inconsistent scalar types");
    return StoreSimd(ptr(r, c), vec, N, _rowStride);
  }

  /// @brief Get the cost of accessing data with the underlying row and column stride.
  MOCHI_FORCE_INLINE constexpr static CostPair RowColCosts() {
    if constexpr (krylov::IsCuda(kOwnership)) {
      return {kGivenRowStride, kGivenColStride};
    } else {
      // Recall that the temporary accessor is column-major
      return {kGivenRowStride == 1 ? 1 : 4, kGivenColStride == 1 ? 1 : 4};
    }
  }

  [[nodiscard]] MOCHI_FORCE_INLINE int RowStride() const {
    return _rowStride.iVal();
  }

  [[nodiscard]] MOCHI_FORCE_INLINE int ColStride() const {
    return _colStride.iVal();
  }

  MOCHI_FORCE_INLINE friend auto Transpose(TempAccessor const& ta) {
    return Accessor(ta._values.Data(), ta._colStride, ta._rowStride);
  }

 private:
  static constexpr int kSize = ((kRowsAtCompileTime >= 0) && (kColsAtCompileTime >= 0))
      ? kRowsAtCompileTime * kColsAtCompileTime
      : krylov::kDynamic;
  details::BaseStorage<Scalar, kSize, kOwnership> _values{};
  IntOrEmpty<kGivenRowStride> _rowStride; ///!< @brief Stride of data between rows
  IntOrEmpty<kGivenColStride> _colStride; ///!< @brief Stride of data between columns
};

template <typename Scalar, int kR, int kC, krylov::Ownership kOwnership>
constexpr bool IsMemoryAccessor<TempAccessor<Scalar, kR, kC, kOwnership>> = true;

/**
 * @brief Destination Accessor object carrying destination update operation.
 * @details The destination matrix can either be uninitialized, in which case
 * an evaluation of an expression should set its value, or already have part
 * of the result of a series of sum or subtractions of sub-expressions.
 * In this case, any modification of the destination should be done via
 * an addition or subtraction. For the set case,
 * @tparam kOp_
 * @tparam T
 */
template <DestOp kOp_, typename T>
struct DestinationAccessor {
  using scalar_type = typename T::scalar_type;
  static constexpr DestOp kOp = kOp_;

  MOCHI_ANY MOCHI_FORCE_INLINE DestinationAccessor(T t) : accessor(std::move(t)) {}
  T accessor; /// @brief The underlying raw accessor

  MOCHI_ANY MOCHI_FORCE_INLINE auto* Data() {
    return accessor.Data();
  }

  MOCHI_ANY MOCHI_FORCE_INLINE constexpr auto const* Data() const {
    return accessor.Data();
  }

  template <typename V>
  MOCHI_ANY MOCHI_FORCE_INLINE void Store(int row, int col, V v_) const {
    auto v = static_cast<scalar_type>(v_);
    if constexpr (kOp_ == DestOp::Set) {
      accessor.Store(row, col, v);
    } else if constexpr (kOp_ == DestOp::NegSet) { // A - B*C will test this
      accessor.Store(row, col, -v);
    } else if constexpr (kOp_ == DestOp::Add) {
      accessor.Store(row, col, accessor(row, col) + v);
    } else {
      static_assert(kOp_ == DestOp::Sub, "Unsupported DestOp");
      accessor.Store(row, col, accessor(row, col) - v);
    }
  }

  template <int N = -1, typename VType>
  MOCHI_ANY MOCHI_FORCE_INLINE auto StoreRowVector(int r, int c, VType vec) const {
    static_assert(std::is_same_v<scalar_type, typename VType::Scalar>, "Inconsistent scalar types");
    if constexpr (kOp_ == DestOp::Set) {
      return accessor.template StoreRowVector<N>(r, c, vec);
    } else if constexpr (kOp_ == DestOp::NegSet) { // A - (B*C) will test this
      return accessor.template StoreRowVector<N>(r, c, -vec);
    } else if constexpr (kOp_ == DestOp::Add) {
      return accessor.template StoreRowVector<N>(
          r, c, SumOp{}(accessor.template RowVector<VType, N>(r, c), vec));
    } else {
      static_assert(kOp_ == DestOp::Sub, "Unsupported DestOp");
      return accessor.template StoreRowVector<N>(
          r, c, SubtractOp{}(accessor.template RowVector<VType, N>(r, c), vec));
    }
  }

  template <typename VType>
  MOCHI_ANY MOCHI_FORCE_INLINE auto StoreRowVector(int r, int c, VType vec, int N) const {
    static_assert(std::is_same_v<scalar_type, typename VType::Scalar>, "Inconsistent scalar types");
    if constexpr (kOp_ == DestOp::Set) {
      return accessor.StoreRowVector(r, c, vec, N);
    } else if constexpr (kOp_ == DestOp::NegSet) {
      return accessor.StoreRowVector(r, c, -vec, N);
    } else if constexpr (kOp_ == DestOp::Add) {
      return accessor.StoreRowVector(
          r, c, SumOp{}(accessor.template RowVector<VType>(r, c, N), vec), N);
    } else {
      static_assert(kOp_ == DestOp::Sub, "Unsupported DestOp");
      return accessor.StoreRowVector(
          r, c, SubtractOp{}(accessor.template RowVector<VType>(r, c, N), vec), N);
    }
  }

  template <int N = -1, typename VType>
  MOCHI_ANY MOCHI_FORCE_INLINE auto StoreColVector(int r, int c, VType vec) const {
    static_assert(std::is_same_v<scalar_type, typename VType::Scalar>, "Inconsistent scalar types");
    if constexpr (kOp_ == DestOp::Set) {
      return accessor.template StoreColVector<N>(r, c, vec);
    } else if constexpr (kOp_ == DestOp::NegSet) { // A - (B*C) will test this
      return accessor.template StoreColVector<N>(r, c, -vec);
    } else if constexpr (kOp_ == DestOp::Add) {
      return accessor.template StoreColVector<N>(
          r, c, SumOp{}(accessor.template ColVector<VType, N>(r, c), vec));
    } else {
      static_assert(kOp_ == DestOp::Sub, "Unsupported DestOp");
      return accessor.template StoreColVector<N>(
          r, c, SubtractOp{}(accessor.template ColVector<VType, N>(r, c), vec));
    }
  }

  template <typename VType>
  MOCHI_ANY MOCHI_FORCE_INLINE auto StoreColVector(int r, int c, VType vec, int N) const {
    static_assert(std::is_same_v<scalar_type, typename VType::Scalar>, "Inconsistent scalar types");
    if constexpr (kOp_ == DestOp::Set) {
      return accessor.StoreColVector(r, c, vec, N);
    } else if constexpr (kOp_ == DestOp::NegSet) {
      return accessor.StoreColVector(r, c, -vec, N);
    } else if constexpr (kOp_ == DestOp::Add) {
      return accessor.StoreColVector(
          r, c, SumOp{}(accessor.template ColVector<VType>(r, c, N), vec), N);
    } else {
      static_assert(kOp_ == DestOp::Sub, "Unsupported DestOp");
      return accessor.StoreColVector(
          r, c, SubtractOp{}(accessor.template ColVector<VType>(r, c, N), vec), N);
    }
  }

  MOCHI_ANY MOCHI_FORCE_INLINE constexpr static auto RowColCosts() {
    return T::RowColCosts();
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE int RowStride() const {
    return accessor.RowStride();
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE int ColStride() const {
    return accessor.ColStride();
  }

  MOCHI_ANY MOCHI_FORCE_INLINE friend auto Transpose(DestinationAccessor const& da) {
    using AT = decltype(Transpose(da.accessor));
    return DestinationAccessor<kOp_, AT>{Transpose(da.accessor)};
  }
};

/// @brief Build an DestinationAccessor with the DestOp::Set tag.
/// @tparam T Underlying Accessor type.
/// @param rawAccessor The underlying Accessor.
/// @return A DestinationAccessor to the underlying data with a DestOp::Set tag.
template <typename T>
MOCHI_ANY MOCHI_FORCE_INLINE DestinationAccessor<DestOp::Set, T> SetDest(T&& rawAccessor) {
  return {std::forward<T>(rawAccessor)};
}

/// @brief Build an DestinationAccessor with the DestOp::Set tag.
/// @tparam T Underlying Accessor type.
/// @param rawAccessor The underlying Accessor.
/// @return A DestinationAccessor to the underlying data with a DestOp::Set tag.
template <typename T>
MOCHI_ANY MOCHI_FORCE_INLINE DestinationAccessor<DestOp::NegSet, T> NegSetDest(T&& rawAccessor) {
  return {std::forward<T>(rawAccessor)};
}

/// @brief Build an DestinationAccessor with the DestOp::Add tag.
/// @tparam T Underlying Accessor type.
/// @param rawAccessor The underlying Accessor.
/// @return A DestinationAccessor to the underlying data with a DestOp::Add tag.
template <typename T>
MOCHI_ANY MOCHI_FORCE_INLINE DestinationAccessor<DestOp::Add, T> SumDest(T&& rawAccessor) {
  return {std::forward<T>(rawAccessor)};
}

/// @brief Build an DestinationAccessor with the DestOp::Sub tag.
/// @tparam T Underlying Accessor type.
/// @param rawAccessor The underlying Accessor.
/// @return A DestinationAccessor to the underlying data with a DestOp::Sub tag.
template <typename T>
MOCHI_ANY MOCHI_FORCE_INLINE DestinationAccessor<DestOp::Sub, T> SubtractDest(T&& rawAccessor) {
  return {std::forward<T>(rawAccessor)};
}

/// @brief Build the negation of a DestinationAccessor
template <typename T, DestOp kOp>
MOCHI_ANY MOCHI_FORCE_INLINE auto operator-(DestinationAccessor<kOp, T> accessor) {
  return DestinationAccessor<-kOp, T>(std::forward<T>(accessor.accessor));
}

/// @brief An evaluation accessor indicating that the `eval` that return this has already set some
/// data in the destination.
template <typename T>
struct AccessorWithSet {
  using type = T;
  MOCHI_ANY explicit AccessorWithSet(T t) : accessor(std::move(t)) {}
  MOCHI_ANY constexpr static auto RowColCosts() {
    return T::RowColCosts();
  }
  T accessor; /// @brief The underlying raw accessor
};

template <typename T>
constexpr bool kIsAccessorWithSet = false;

template <typename T>
constexpr bool kIsAccessorWithSet<AccessorWithSet<T>> = true;

} // namespace mochi::details
