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

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_accessors.h>

#include <concepts>
#include <initializer_list>
#include <type_traits>

namespace mochi {
using Direction = krylov::Direction;
using Ownership = krylov::Ownership;

/** @brief Loose concept of a matrix.
    @details Accepts both Matrix<...> and StridedMatrix<...> */
template <typename T>
concept IsLooselyMatrix = requires(T m) {
  { m.Rows() } -> std::convertible_to<int>;
  { m.Cols() } -> std::convertible_to<int>;
  { m(1, 3) } -> std::convertible_to<double>;
};

template <typename T>
concept IsStridedExpr = IsStridedMatrix<T> || requires(T t) {
  { details::GetDomainFor(t) } -> std::same_as<details::DomainType<details::ExprDomain::Strided>>;
};

template <typename T, size_t size, Ownership kOwnership>
class StridedMatStorage {
 public:
  MOCHI_ANY constexpr StridedMatStorage() = default;
  MOCHI_ANY constexpr StridedMatStorage(StridedMatStorage const& other) {
    for (size_t i = 0; i < size; i++) {
      _data[i] = other._data[i];
    }
  }

  MOCHI_ANY constexpr StridedMatStorage(T v0, std::same_as<T> auto... v) : _data{v0, v...} {}
  MOCHI_ANY constexpr StridedMatStorage(
      std::initializer_list<std::initializer_list<T>> const& init) {
    size_t offset = 0;
    for (auto const& rowOrColumn : init) {
      for (auto const v : rowOrColumn) {
        _data[offset++] = v;
      }
    }
  }

  MOCHI_ANY constexpr T* Data() {
    return _data;
  }
  MOCHI_ANY constexpr T const* Data() const {
    return _data;
  }

 private:
  T _data[size];
};

template <typename T, size_t size>
class StridedMatStorage<T, size, Ownership::View> {
 public:
  MOCHI_ANY explicit StridedMatStorage(T* data) : _data(data) {}
  MOCHI_ANY constexpr T* Data() {
    return _data;
  }
  MOCHI_ANY constexpr T const* Data() const {
    return _data;
  }

 private:
  T* _data;
};

template <typename T, int kRows, int kCols, int kStride, Ownership kOwnership, int kLeadingDim>
struct StridedMatData {
  MOCHI_ANY constexpr StridedMatData() = default;
  MOCHI_ANY explicit constexpr StridedMatData(T* data, int = kRows, int = kCols, int = kLeadingDim)
    requires(kOwnership == Ownership::View)
      : storage(data) {}
  MOCHI_ANY constexpr StridedMatData(T v0, std::same_as<T> auto... v)
    requires(kStride == 1)
      : storage{v0, v...} {}
  MOCHI_ANY constexpr StridedMatData(std::initializer_list<std::initializer_list<T>> const& init)
    requires(kStride == 1)
      : storage(init) {}
  StridedMatStorage<
      T,
      static_cast<size_t>(kRows) * static_cast<size_t>(kCols) * static_cast<size_t>(kStride),
      kOwnership>
      storage;
};

template <typename T, int kCols, int kStride, int kLeadingDim>
  requires(kLeadingDim != krylov::kDynamic)
struct StridedMatData<T, krylov::kDynamic, kCols, kStride, Ownership::View, kLeadingDim> {
  MOCHI_ANY constexpr StridedMatData(T* data, int rows, int /*cols*/, int = kLeadingDim)
      : storage{data}, rows(rows) {}
  StridedMatStorage<T, static_cast<size_t>(-1), Ownership::View> storage;
  int rows;
};

template <typename T, int kRows, int kStride, int kLeadingDim>
  requires(kLeadingDim != krylov::kDynamic)
struct StridedMatData<T, kRows, krylov::kDynamic, kStride, Ownership::View, kLeadingDim> {
  MOCHI_ANY constexpr StridedMatData(T* data, int /*rows*/, int cols, int = kLeadingDim)
      : storage{data}, cols(cols) {}
  StridedMatStorage<T, static_cast<size_t>(-1), Ownership::View> storage;
  int cols;
};

template <typename T, int kStride, int kLeadingDim>
  requires(kLeadingDim != krylov::kDynamic)
struct StridedMatData<
    T,
    krylov::kDynamic,
    krylov::kDynamic,
    kStride,
    Ownership::View,
    kLeadingDim> {
  MOCHI_ANY constexpr StridedMatData(T* data, int rows, int cols, int = kLeadingDim)
      : storage{data}, rows(rows), cols(cols) {}
  StridedMatStorage<T, static_cast<size_t>(-1), Ownership::View> storage;
  int rows;
  int cols;
};

template <typename T, int kRows, int kCols, int kStride>
struct StridedMatData<T, kRows, kCols, kStride, Ownership::View, krylov::kDynamic> {
  MOCHI_ANY constexpr StridedMatData(T* data, int /*rows*/, int /*cols*/, int leadingDim)
      : storage{data}, leadingDim(leadingDim) {}
  StridedMatStorage<T, static_cast<size_t>(-1), Ownership::View> storage;
  int leadingDim;
};

template <typename T, int kCols, int kStride>
struct StridedMatData<T, krylov::kDynamic, kCols, kStride, Ownership::View, krylov::kDynamic> {
  MOCHI_ANY constexpr StridedMatData(T* data, int rows, int /*cols*/, int leadingDim)
      : storage{data}, rows(rows), leadingDim(leadingDim) {}
  StridedMatStorage<T, static_cast<size_t>(-1), Ownership::View> storage;
  int rows;
  int leadingDim;
};

template <typename T, int kRows, int kStride>
struct StridedMatData<T, kRows, krylov::kDynamic, kStride, Ownership::View, krylov::kDynamic> {
  MOCHI_ANY constexpr StridedMatData(T* data, int /*rows*/, int cols, int leadingDim)
      : storage{data}, cols(cols), leadingDim(leadingDim) {}
  StridedMatStorage<T, static_cast<size_t>(-1), Ownership::View> storage;
  int cols;
  int leadingDim;
};

template <typename T, int kStride>
struct StridedMatData<
    T,
    krylov::kDynamic,
    krylov::kDynamic,
    kStride,
    Ownership::View,
    krylov::kDynamic> {
  MOCHI_ANY constexpr StridedMatData(T* data, int rows, int cols, int leadingDim)
      : storage{data}, rows(rows), cols(cols), leadingDim(leadingDim) {}
  StridedMatStorage<T, static_cast<size_t>(-1), Ownership::View> storage;
  int rows;
  int cols;
  int leadingDim;
};

template <
    typename T,
    int kRows,
    int kCols,
    int kStride = 1,
    Direction kMajorDirection = Direction::ColMajor,
    Ownership kOwnership = Ownership::Owner,
    int kLeadingDim = krylov::kAutomaticLeadDim>
class MOCHI_EMPTY_BASE StridedMatrix : NewMatrix {
  static_assert(
      kOwnership == Ownership::Owner || kOwnership == Ownership::View,
      "Currently only Owner or View are supported ownerships.");
  static_assert(kStride >= 1, "Stride must be positive"); // Dynamic strides not supported yet

 public:
  using Scalar = T;
  static constexpr details::ExprDomain domain = details::ExprDomain::Strided;

  MOCHI_ANY constexpr StridedMatrix() = default;
  MOCHI_ANY constexpr explicit StridedMatrix(T* data)
    requires(kOwnership == Ownership::View)
      : _data{data} {}
  MOCHI_ANY constexpr explicit StridedMatrix(T* data, int rows, int cols)
    requires(kOwnership == Ownership::View)
      : _data{data, rows, cols} {}
  MOCHI_ANY constexpr explicit StridedMatrix(T* data, int rows, int cols, int ld)
    requires(kOwnership == Ownership::View)
      : _data{data, rows, cols, ld} {}
  MOCHI_ANY constexpr StridedMatrix(T v0, std::same_as<T> auto... v) : _data{v0, v...} {
    static_assert((sizeof...(v)) == kRows * kCols - 1);
  }
  MOCHI_ANY constexpr StridedMatrix(std::initializer_list<std::initializer_list<T>> const& init)
      : _data(init) {}

  MOCHI_ANY constexpr StridedMatrix(StridedMatrix const& other) = default;
  MOCHI_ANY constexpr StridedMatrix(StridedMatrix&& other) noexcept = default;

  template <IsStridedExpr M>
  MOCHI_ANY constexpr StridedMatrix(M const& expr) {
    *this = expr;
  }

  MOCHI_ANY constexpr StridedMatrix& operator=(StridedMatrix const& other);

  template <IsStridedExpr M>
  MOCHI_ANY constexpr StridedMatrix& operator=(M const& expr);

  template <IsLooselyMatrix M>
    requires(!IsStridedExpr<M>)
  MOCHI_ANY constexpr StridedMatrix& operator=(M const& A);

  template <IsStridedExpr M>
  MOCHI_ANY constexpr StridedMatrix& operator+=(M const& expr);

  template <IsStridedExpr M>
  MOCHI_ANY constexpr StridedMatrix& operator-=(M const& expr);

  MOCHI_ANY constexpr StridedMatrix& operator*=(T alpha);

  MOCHI_ANY constexpr T const* Data() const {
    return _data.storage.Data();
  }
  MOCHI_ANY constexpr T* Data() {
    return _data.storage.Data();
  }

  MOCHI_ANY constexpr T& operator()(int i, int j) {
    return _data.storage.Data()[Offset(i, j)];
  }

  MOCHI_ANY constexpr T operator()(int i, int j) const {
    return _data.storage.Data()[Offset(i, j)];
  }

  /// @brief Extract the block starting at (ir, jc) with p rows and q columns.
  ///
  /// @param ir Starting row index
  /// @param jc Starting column index
  /// @param numRows  Number of rows to extract
  /// @param numCols  Number of columns to extract
  ///
  /// @return Matrix view of the extracted block
  template <int kRowsBlockAtCompile = krylov::kDynamic, int kColsBlockAtCompile = krylov::kDynamic>
  [[nodiscard]] MOCHI_ANY auto
  Block(int r, int c, int numRows = kRowsBlockAtCompile, int numCols = kColsBlockAtCompile) {
    auto* d = this->Data() + Offset(r, c);
    constexpr int kMyLeadDim = (kLeadingDim > 0) ? kLeadingDim : krylov::kDynamic;
    return StridedMatrix<
        Scalar,
        kRowsBlockAtCompile,
        kColsBlockAtCompile,
        kStride,
        kMajorDirection,
        krylov::Viewed<kOwnership>,
        kMyLeadDim>{d, numRows, numCols, this->LeadDim()};
  }

  /** @copydoc Block() */
  template <int kRowsBlockAtCompile = krylov::kDynamic, int kColsBlockAtCompile = krylov::kDynamic>
  [[nodiscard]] MOCHI_ANY auto
  Block(int r, int c, int numRows = kRowsBlockAtCompile, int numCols = kColsBlockAtCompile) const {
    auto* d = this->Data() + Offset(r, c);
    constexpr int kMyLeadDim = (kLeadingDim > 0) ? kLeadingDim : krylov::kDynamic;
    return StridedMatrix<
        Scalar const,
        kRowsBlockAtCompile,
        kColsBlockAtCompile,
        kStride,
        kMajorDirection,
        krylov::Viewed<kOwnership>,
        kMyLeadDim>{d, numRows, numCols, this->LeadDim()};
  }

  /// @brief Extract a single column from the matrix.
  ///
  /// @param c Index of the column to extract
  /// @return Vector-view of extracted column
  [[nodiscard]] MOCHI_ANY auto Col(int c) {
    MOCHI_ASSERT_VERBOSE((c >= 0) && (c < this->Cols()), "Out-of-range column index");
    auto* d = this->Data() + Offset(0, c);
    // Col-major storage inherits the leading dimension
    // Row-major becomes dynamic, as the number of column of the result is
    // not the same as the starting matrix.
    constexpr int kColLeadDim =
        kMajorDirection == Direction::ColMajor ? kLeadingDim : krylov::kDynamic;
    return StridedMatrix<
        Scalar,
        kRows,
        1,
        kStride,
        kMajorDirection,
        krylov::Viewed<kOwnership>,
        kColLeadDim>{d, this->Rows(), 1, this->LeadDim()};
  }

  /// @brief Extract a single column from the matrix.
  ///
  /// @param c Index of the column to extract
  /// @return Vector-view of extracted column
  [[nodiscard]] MOCHI_ANY auto Col(int c) const {
    MOCHI_ASSERT_VERBOSE((c >= 0) && (c < this->Cols()), "Out-of-range column index");
    auto* d = this->Data() + Offset(0, c);
    // Col-major storage inherits the leading dimension
    // Row-major becomes dynamic, as the number of column of the result is
    // not the same as the starting matrix.
    constexpr int kColLeadDim =
        kMajorDirection == Direction::ColMajor ? kLeadingDim : krylov::kDynamic;
    return StridedMatrix<
        Scalar const,
        kRows,
        1,
        kStride,
        kMajorDirection,
        krylov::Viewed<kOwnership>,
        kColLeadDim>{d, this->Rows(), 1, this->LeadDim()};
  }

  [[nodiscard]] MOCHI_ANY constexpr size_t Offset(int r, int c) const {
    // The treatment of kStride == 1 as a special case is to avoid the
    // cost of multiplication when optimization is turned off or the compiler
    // cannot optimize it out for some obscure rule.
    if constexpr (kStride == 1) {
      if constexpr (kMajorDirection == Direction::ColMajor) {
        return c * LeadDim() + r;
      } else {
        return r * LeadDim() + c;
      }
    } else {
      if constexpr (kMajorDirection == Direction::ColMajor) {
        return kStride * (c * LeadDim() + r);
      } else {
        return kStride * (r * LeadDim() + c);
      }
    }
  }

  [[nodiscard]] MOCHI_ANY constexpr auto CERows() const
    requires(kRows != krylov::kDynamic)
  {
    return details::IntOrEmpty<kRows>{};
  }

  [[nodiscard]] MOCHI_ANY constexpr auto CERows() const
    requires(kRows == krylov::kDynamic)
  {
    return details::IntOrEmpty<kRows>{_data.rows};
  }

  [[nodiscard]] MOCHI_ANY constexpr auto Rows() const
    requires(kRows != krylov::kDynamic)
  {
    return kRows;
  }

  [[nodiscard]] MOCHI_ANY constexpr auto Rows() const
    requires(kRows == krylov::kDynamic)
  {
    return _data.rows;
  }

  [[nodiscard]] MOCHI_ANY constexpr auto CECols() const
    requires(kCols != krylov::kDynamic)
  {
    return details::IntOrEmpty<kCols>{};
  }

  [[nodiscard]] MOCHI_ANY constexpr auto CECols() const
    requires(kCols == krylov::kDynamic)
  {
    return details::IntOrEmpty<kCols>{_data.cols};
  }

  [[nodiscard]] MOCHI_ANY constexpr auto Cols() const
    requires(kCols != krylov::kDynamic)
  {
    return kCols;
  }

  [[nodiscard]] MOCHI_ANY constexpr auto Cols() const
    requires(kCols == krylov::kDynamic)
  {
    return _data.cols;
  }

  [[nodiscard]] MOCHI_ANY constexpr auto LeadDim() const
    requires(kLeadingDim != krylov::kDynamic)
  {
    if constexpr (kLeadingDim == krylov::kAutomaticLeadDim) {
      if constexpr (kMajorDirection == Direction::ColMajor) {
        return Rows();
      } else {
        return Cols();
      }
    } else {
      return kLeadingDim;
    }
  }

  [[nodiscard]] MOCHI_ANY constexpr auto LeadDim() const
    requires(kLeadingDim == krylov::kDynamic)
  {
    return _data.leadingDim;
  }

  [[nodiscard]] MOCHI_ANY constexpr auto Transpose() const {
    if constexpr (kRows > 0 && kCols > 0 && kLeadingDim == krylov::kAutomaticLeadDim) {
      return StridedMatrix<
          T const,
          kCols,
          kRows,
          kStride,
          ~kMajorDirection,
          Ownership::View,
          kLeadingDim>(Data());
    } else {
      return StridedMatrix<
          T const,
          kCols,
          kRows,
          kStride,
          ~kMajorDirection,
          Ownership::View,
          kLeadingDim>(Data(), Cols(), Rows(), LeadDim());
    }
  }

  [[nodiscard]] MOCHI_ANY constexpr auto Transpose() {
    if constexpr (kRows > 0 && kCols > 0 && kLeadingDim == krylov::kAutomaticLeadDim) {
      return StridedMatrix<
          T,
          kCols,
          kRows,
          kStride,
          ~kMajorDirection,
          Ownership::View,
          kLeadingDim>(Data());
    } else {
      return StridedMatrix<
          T,
          kCols,
          kRows,
          kStride,
          ~kMajorDirection,
          Ownership::View,
          kLeadingDim>(Data(), Cols(), Rows(), LeadDim());
    }
  }

  MOCHI_ANY StridedMatrix& SetZero() {
    for (int r = 0; r < Rows(); ++r) {
      for (int c = 0; c < Cols(); ++c) {
        (*this)(r, c) = 0;
      }
    }
    return *this;
  }

 private:
  StridedMatData<T, kRows, kCols, kStride, kOwnership, kLeadingDim> _data;
};

template <
    typename T,
    int kRows,
    int kCols,
    int kStride,
    Direction kMajorDirection,
    Ownership kOwnership,
    int kLeadingDim>
MOCHI_ANY constexpr StridedMatrix<
    T,
    kRows,
    kCols,
    kStride,
    kMajorDirection,
    kOwnership,
    kLeadingDim>&
StridedMatrix<T, kRows, kCols, kStride, kMajorDirection, kOwnership, kLeadingDim>::operator=(
    StridedMatrix const& other) {
  for (int r = 0; r < Rows(); ++r) {
    for (int c = 0; c < Cols(); ++c) {
      (*this)(r, c) = other(r, c);
    }
  }
  return *this;
}

template <
    typename T,
    int kRows,
    int kCols,
    int kStride,
    Direction kMajorDirection,
    Ownership kOwnership,
    int kLeadingDim>
MOCHI_ANY constexpr StridedMatrix<
    T,
    kRows,
    kCols,
    kStride,
    kMajorDirection,
    kOwnership,
    kLeadingDim>&
StridedMatrix<T, kRows, kCols, kStride, kMajorDirection, kOwnership, kLeadingDim>::operator*=(
    T alpha) {
  for (int r = 0; r < Rows(); ++r) {
    for (int c = 0; c < Cols(); ++c) {
      (*this)(r, c) *= alpha;
    }
  }
  return *this;
}

template <
    typename T,
    int kRows,
    int kCols,
    int kStride = 1,
    Direction kMajorDirection = Direction::ColMajor,
    int kLeadingDim = krylov::kAutomaticLeadDim>
using StridedMatrixView =
    StridedMatrix<T, kRows, kCols, kStride, kMajorDirection, Ownership::View, kLeadingDim>;

template <
    typename T,
    int kRows,
    int kStride = 1,
    Direction kMajorDirection = Direction::ColMajor,
    Ownership kOwnership = Ownership::Owner,
    int kLeadingDim = krylov::kAutomaticLeadDim>
using StridedVector = StridedMatrix<T, kRows, 1, kStride, kMajorDirection, kOwnership, kLeadingDim>;

template <
    typename T,
    int kRows,
    int kStride = 1,
    Direction kMajorDirection = Direction::ColMajor,
    int kLeadingDim = krylov::kAutomaticLeadDim>
using StridedVectorView =
    StridedMatrix<T, kRows, 1, kStride, kMajorDirection, Ownership::View, kLeadingDim>;

namespace details {

template <
    typename T,
    int kRows,
    int kCols,
    int kStride,
    Direction kMajorDirection,
    Ownership kOwnership,
    int kLeadingDim>
constexpr bool IsStridedMatrixDef<
    StridedMatrix<T, kRows, kCols, kStride, kMajorDirection, kOwnership, kLeadingDim>> = true;

template <
    typename T,
    int kRows,
    int kCols,
    int kStride,
    Direction kMajorDirection,
    Ownership kOwnership,
    int kLeadingDim>
constexpr bool IsCudaDef<
    StridedMatrix<T, kRows, kCols, kStride, kMajorDirection, kOwnership, kLeadingDim>,
    void> = []() {
  // TODO: Resolve the issue below and remove this IsCudaDef specialization.
  static_assert(
      sizeof(T) == 0,
      "IsCuda<StridedMatrix<...>> is not supported. StridedMatrix memory location is context-dependent and cannot be determined at compile-time.");
  return false;
}();

template <typename Scalar, int kRowStrideAtCompileTime, int kColStrideAtCompileTime>
struct StridedAccessor {
  using scalar_type = Scalar;
  static constexpr int kGivenRowStride = kRowStrideAtCompileTime;
  static constexpr int kGivenColStride = kColStrideAtCompileTime;

  MOCHI_ANY StridedAccessor(Scalar* v, int rowStride, int colStride)
      : _v(v), _rowStride(rowStride), _colStride(colStride) {}

  MOCHI_ANY friend auto Transpose(StridedAccessor const& a) {
    return StridedAccessor<Scalar, kColStrideAtCompileTime, kRowStrideAtCompileTime>(
        a._v, a._colStride.iVal(), a._rowStride.iVal());
  }

  MOCHI_ANY Scalar* Data() {
    return _v;
  }

  MOCHI_ANY constexpr Scalar const* Data() const {
    return _v;
  }

  MOCHI_ANY Scalar* ptr(int r, int c) const {
    return _v + static_cast<size_t>(r) * _rowStride.sVal() +
        static_cast<size_t>(c) * _colStride.sVal();
  }

  MOCHI_ANY constexpr Scalar const& operator()(int r, int c) const {
    return _v
        [static_cast<size_t>(r) * _rowStride.sVal() + static_cast<size_t>(c) * _colStride.sVal()];
  }

  MOCHI_ANY constexpr void Store(int r, int c, Scalar const& value) const {
    *ptr(r, c) = value;
  }

  template <typename VType, int N = -1>
  MOCHI_ANY auto RowVector(int r, int c) const {
    static_assert(
        std::is_same_v<std::remove_const_t<Scalar>, typename VType::Scalar>,
        "Inconsistent scalar types");
    return GetSimd<VType, N>(ptr(r, c), _colStride);
  }

  template <typename VType>
  MOCHI_ANY auto RowVector(int r, int c, int N) const {
    static_assert(
        std::is_same_v<std::remove_const_t<Scalar>, typename VType::Scalar>,
        "Inconsistent scalar types");
    return GetSimd<VType>(ptr(r, c), N, _colStride);
  }

  template <typename VType, int N = -1>
  MOCHI_ANY auto ColVector(int r, int c) const {
    static_assert(
        std::is_same_v<std::remove_const_t<Scalar>, typename VType::Scalar>,
        "Inconsistent scalar types");
    return GetSimd<VType, N>(ptr(r, c), _rowStride);
  }

  template <typename VType>
  MOCHI_ANY auto ColVector(int r, int c, int N) const {
    static_assert(
        std::is_same_v<std::remove_const_t<Scalar>, typename VType::Scalar>,
        "Inconsistent scalar types");
    return GetSimd<VType>(ptr(r, c), N, _rowStride);
  }

  template <int N = -1, typename VType>
  MOCHI_ANY auto StoreRowVector(int r, int c, VType vec) const {
    static_assert(std::is_same_v<Scalar, typename VType::Scalar>, "Inconsistent scalar types");
    return StoreSimd<N>(ptr(r, c), vec, _colStride);
  }

  template <typename VType>
  MOCHI_ANY auto StoreRowVector(int r, int c, VType vec, int N) const {
    static_assert(std::is_same_v<Scalar, typename VType::Scalar>, "Inconsistent scalar types");
    return StoreSimd(ptr(r, c), vec, N, _colStride);
  }

  template <int N = -1, typename VType>
  MOCHI_ANY auto StoreColVector(int r, int c, VType vec) const {
    static_assert(std::is_same_v<Scalar, typename VType::Scalar>, "Inconsistent scalar types");
    return StoreSimd<N>(ptr(r, c), vec, _rowStride);
  }

  template <typename VType>
  MOCHI_ANY auto StoreColVector(int r, int c, VType vec, int N) const {
    static_assert(std::is_same_v<Scalar, typename VType::Scalar>, "Inconsistent scalar types");
    return StoreSimd(ptr(r, c), vec, N, _rowStride);
  }

  MOCHI_ANY constexpr static CostPair RowColCosts() {
    return {kRowStrideAtCompileTime == 1 ? 1 : 4, kColStrideAtCompileTime == 1 ? 1 : 4};
  }

  [[nodiscard]] MOCHI_ANY int RowStride() const {
    return _rowStride.iVal();
  }

  [[nodiscard]] MOCHI_ANY int ColStride() const {
    return _colStride.iVal();
  }

 private:
  Scalar* _v;
  IntOrEmpty<kGivenRowStride> _rowStride;
  IntOrEmpty<kGivenColStride> _colStride;
};
} // namespace details

template <
    typename T,
    int kRows,
    int kCols,
    int kStride,
    Direction kMajorDirection,
    Ownership kOwnership,
    int kLeadingDim>
MOCHI_ANY auto GetAccessor(
    StridedMatrix<T, kRows, kCols, kStride, kMajorDirection, kOwnership, kLeadingDim>& A) {
  constexpr auto strided = [](int k) {
    return (k == krylov::kDynamic) ? krylov::kDynamic : kStride * k;
  };
  constexpr int kLeadDimACT = kLeadingDim == krylov::kAutomaticLeadDim
      ? (kMajorDirection == Direction::ColMajor ? strided(kRows) : strided(kCols))
      : strided(kLeadingDim);
  constexpr int kRowStrideACT = kMajorDirection == Direction::ColMajor ? kStride : kLeadDimACT;
  constexpr int kColStrideACT = kMajorDirection == Direction::RowMajor ? kStride : kLeadDimACT;
  int rowStride = (kRowStrideACT == krylov::kDynamic) ? strided(A.LeadDim()) : kRowStrideACT;
  int colStride = (kColStrideACT == krylov::kDynamic) ? strided(A.LeadDim()) : kColStrideACT;
  return details::StridedAccessor<T, kRowStrideACT, kColStrideACT>(A.Data(), rowStride, colStride);
}

template <
    typename T,
    int kRows,
    int kCols,
    int kStride,
    Direction kMajorDirection,
    Ownership kOwnership,
    int kLeadingDim>
MOCHI_ANY auto GetAccessor(
    StridedMatrix<T, kRows, kCols, kStride, kMajorDirection, kOwnership, kLeadingDim> const& A) {
  constexpr auto strided = [](int k) {
    return (k == krylov::kDynamic) ? krylov::kDynamic : kStride * k;
  };
  constexpr int kLeadDimACT = kLeadingDim == krylov::kAutomaticLeadDim
      ? (kMajorDirection == Direction::ColMajor ? strided(kRows) : strided(kCols))
      : strided(kLeadingDim);
  constexpr int kRowStrideACT = kMajorDirection == Direction::ColMajor ? kStride : kLeadDimACT;
  constexpr int kColStrideACT = kMajorDirection == Direction::RowMajor ? kStride : kLeadDimACT;
  int rowStride = (kRowStrideACT == krylov::kDynamic) ? strided(A.LeadDim()) : kRowStrideACT;
  int colStride = (kColStrideACT == krylov::kDynamic) ? strided(A.LeadDim()) : kColStrideACT;
  return details::StridedAccessor<T const, kRowStrideACT, kColStrideACT>(
      A.Data(), rowStride, colStride);
}

template <
    typename T,
    int kRows,
    int kCols,
    int kStride,
    Direction kMajorDirection,
    Ownership kOwnership,
    int kLeadingDim>
template <IsStridedExpr M>
MOCHI_ANY constexpr StridedMatrix<
    T,
    kRows,
    kCols,
    kStride,
    kMajorDirection,
    kOwnership,
    kLeadingDim>&
StridedMatrix<T, kRows, kCols, kStride, kMajorDirection, kOwnership, kLeadingDim>::operator=(
    M const& expr) {
  auto dest = details::SetDest(GetAccessor(*this));
  auto res = details::Eval(expr, dest, this->CERows(), this->CECols());
  details::Assign<Scalar>(dest, res, this->CERows(), this->CECols());
  return *this;
}

template <
    typename T,
    int kRows,
    int kCols,
    int kStride,
    Direction kMajorDirection,
    Ownership kOwnership,
    int kLeadingDim>
template <IsLooselyMatrix M>
  requires(!IsStridedExpr<M>)
MOCHI_ANY constexpr StridedMatrix<
    T,
    kRows,
    kCols,
    kStride,
    kMajorDirection,
    kOwnership,
    kLeadingDim>&
StridedMatrix<T, kRows, kCols, kStride, kMajorDirection, kOwnership, kLeadingDim>::operator=(
    M const& A) {
  int nRows = A.Rows();
  int nCols = A.Cols();
  for (int r = 0; r < nRows; r++) {
    for (int c = 0; c < nCols; c++) {
      (*this)(r, c) = A(r, c);
    }
  }
  return *this;
}

template <
    typename T,
    int kRows,
    int kCols,
    int kStride,
    Direction kMajorDirection,
    Ownership kOwnership,
    int kLeadingDim>
template <IsStridedExpr M>
MOCHI_ANY constexpr StridedMatrix<
    T,
    kRows,
    kCols,
    kStride,
    kMajorDirection,
    kOwnership,
    kLeadingDim>&
StridedMatrix<T, kRows, kCols, kStride, kMajorDirection, kOwnership, kLeadingDim>::operator+=(
    M const& expr) {
  auto dest = details::SumDest(GetAccessor(*this));
  auto res = details::Eval(expr, dest, this->CERows(), this->CECols());
  details::Assign<Scalar>(dest, res, this->CERows(), this->CECols());
  return *this;
}

template <
    typename T,
    int kRows,
    int kCols,
    int kStride,
    Direction kMajorDirection,
    Ownership kOwnership,
    int kLeadingDim>
template <IsStridedExpr M>
MOCHI_ANY constexpr StridedMatrix<
    T,
    kRows,
    kCols,
    kStride,
    kMajorDirection,
    kOwnership,
    kLeadingDim>&
StridedMatrix<T, kRows, kCols, kStride, kMajorDirection, kOwnership, kLeadingDim>::operator-=(
    M const& expr) {
  auto dest = details::SubtractDest(GetAccessor(*this));
  auto res = details::Eval(expr, dest, this->CERows(), this->CECols());
  details::Assign<Scalar>(dest, res, this->CERows(), this->CECols());
  return *this;
}

template <
    typename T,
    int kRows,
    int kCols,
    int kStride,
    Direction kMajorDirection = Direction::ColMajor>
  requires(kStride > 0 && ((kStride & (kStride - 1)) == 0))
struct StridedView {
  MOCHI_ANY StridedView(int rowsIn, int colsIn, Span<T> vIn) : rows(rowsIn), cols(colsIn), v(vIn) {
    MOCHI_ASSERT_VERBOSE(
        v.size() % (kStride * rows.sVal() * cols.sVal()) == 0,
        "Span size must be a multiple of kStride * rows * cols.");
  }

  MOCHI_ANY int Size() const {
    return static_cast<int>(v.size() / (rows.sVal() * cols.sVal()));
  }

  MOCHI_ANY bool Has(int n) const {
    MOCHI_ASSERT_VERBOSE(n >= 0, "Invalid index.");
    return (n + 1) * rows.sVal() * cols.sVal() <= v.size();
  }

  MOCHI_ANY auto Get(int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0, "Invalid index.");
    int block = n / kStride;
    int rem = n % kStride;
    auto r = rows.iVal();
    auto c = cols.iVal();
    auto ptr = v.data() + static_cast<size_t>(block) * kStride * r * c + rem;
    return StridedMatrixView<T, kRows, kCols, kStride, kMajorDirection, krylov::kAutomaticLeadDim>(
        ptr, r, c);
  }

  MOCHI_ANY auto Get(int n) const {
    MOCHI_ASSERT_VERBOSE(n >= 0, "Invalid index.");
    int block = n / kStride;
    int rem = n % kStride;
    auto r = rows.iVal();
    auto c = cols.iVal();
    auto ptr = v.data() + static_cast<size_t>(block) * kStride * r * c + rem;
    return StridedMatrixView<
        T const,
        kRows,
        kCols,
        kStride,
        kMajorDirection,
        krylov::kAutomaticLeadDim>(ptr, r, c);
  }

  MOCHI_ANY auto operator[](int n) {
    return Get(n);
  }

  details::IntOrEmpty<kRows> rows;
  details::IntOrEmpty<kCols> cols;
  Span<T> v;
};

} // namespace mochi
