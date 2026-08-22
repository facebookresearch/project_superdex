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

#include <mochi_core/linear_algebra/krylov/tools/tensor_functions.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph.h>
#include <mochi_core/utils/graph_views.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/span_utils.h>
#include <mochi_core/utils/task_scheduler.h>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace mochi {

/// @brief Sparse matrix with compressed sparse row (CSR) format.
///
/// @tparam Scalar_ Scalar type for entries
/// @tparam CRIdx Signed integral type for the column indices
/// @tparam Ptr Signed integral type for the pointer in the column index array
/// @tparam Storage Container pointing to the data (either owning or viewing). Default value is
/// `DynamicArray` (owning).
template <
    typename Scalar_,
    typename CRIdx = int,
    typename Ptr = int,
    template <typename, typename...> typename Storage = DynamicArray>
class SparseMatrix {
 public:
  using Scalar = Scalar_;
  using NonConstScalar = std::remove_const_t<Scalar>;
  using NonConstIdx = std::remove_const_t<CRIdx>;
  using NonConstPtr = std::remove_const_t<Ptr>;

  static_assert(
      std::signed_integral<CRIdx> && std::signed_integral<Ptr>,
      "SparseMatrix requires signed integer types.");

  SparseMatrix() = default;
  SparseMatrix(SparseMatrix&&) noexcept = default;
  SparseMatrix(SparseMatrix const&) = default;

  SparseMatrix& operator=(SparseMatrix&& rhs) noexcept {
    // Polymorphic_allocator does not propagate on container copy assignment, move assignment, or
    // swap. As a result, move assignment of a polymorphic_allocator-using container can throw, and
    // swapping two polymorphic_allocator-using containers whose allocators do not compare equal
    // results in undefined behavior.
    if (std::addressof(rhs) != this) {
      this->Reset(std::move(rhs));
    }
    return *this;
  }

  SparseMatrix& operator=(SparseMatrix const& rhs) = default;

  // Construct by copying a const SparseMatrix of compatible type.
  // Performs a shallow or deep copy of the values, depending on the Storage type.
  template <
      typename OScal,
      typename OIdx,
      typename OPtr,
      template <typename, typename...> typename OStorage>
  SparseMatrix(SparseMatrix<OScal, OIdx, OPtr, OStorage> const& other)
      : _nCol(other.Cols()),
        _ptr(other.Pointers().data(), other.Pointers().data() + other.Pointers().size()),
        _idx(other.Indices().data(), other.Indices().data() + other.Indices().size()),
        _v(other.Values().data(), other.Values().data() + other.Values().size()) {}

  // Construct by copying a non-const SparseMatrix of compatible type.
  // Performs a shallow or deep copy of the values, depending on the Storage type.
  template <
      typename OScal,
      typename OIdx,
      typename OPtr,
      template <typename, typename...> typename OStorage>
  SparseMatrix(SparseMatrix<OScal, OIdx, OPtr, OStorage>& other)
      : _nCol(other.Cols()),
        _ptr(other.Pointers().data(), other.Pointers().data() + other.Pointers().size()),
        _idx(other.Indices().data(), other.Indices().data() + other.Indices().size()),
        _v(other.Values().data(), other.Values().data() + other.Values().size()) {}

  // Construct by moving the input data
  SparseMatrix(CRIdx nCol, Storage<Ptr> ptr, Storage<CRIdx> idx, Storage<Scalar> v)
      : _nCol(nCol), _ptr(std::move(ptr)), _idx(std::move(idx)), _v(std::move(v)) {
    MOCHI_ASSERT(isize(_idx) == isize(_v), "Size mismatch");
    MOCHI_ASSERT(_ptr.empty() || (_ptr.back() == isize(_v)), "Invalid pointer offset.");
  }

  // Construct a SparseMatrix by copying the sparsity pattern from a Graph
  template <
      typename CRIdxG,
      typename PtrG,
      template <typename, typename...> typename StorageG,
      typename... ExtraArgs>
  SparseMatrix(CRIdx nCol, Graph<CRIdxG, PtrG, StorageG> const& graph, ExtraArgs const&... rest)
      : _nCol(nCol),
        _ptr(graph.GetPointers().begin(), graph.GetPointers().end()),
        _idx(graph.GetTargets().begin(), graph.GetTargets().end()),
        _v(graph.NumTargets(), Scalar(0), rest...) {
    static_assert(std::is_same_v<CRIdx const, CRIdxG const>, "Inconsistent row index types");
    static_assert(std::is_same_v<Ptr const, PtrG const>, "Inconsistent pointer index types");
    MOCHI_ASSERT(isize(_idx) == isize(_v), "Size mismatch");
    MOCHI_ASSERT(_ptr.empty() || (_ptr.back() == isize(_v)), "Invalid pointer offset.");
  }

  // Construct a SQUARE SparseMatrix by moving the sparsity pattern from a Graph.
  // The number of rows and columns is determined by the size of the graph.
  template <
      typename CRIdxG,
      typename PtrG,
      template <typename, typename...> typename StorageG,
      typename... ExtraArgs>
  explicit SparseMatrix(Graph<CRIdxG, PtrG, StorageG> const& graph, ExtraArgs const&... rest)
      : SparseMatrix(graph.size(), graph, rest...) {}

  // Construct a SparseMatrix by moving the sparsity pattern from a Graph
  template <
      typename CRIdxG,
      typename PtrG,
      template <typename, typename...> typename StorageG,
      typename... ExtraArgs>
  SparseMatrix(CRIdx nCol, Graph<CRIdxG, PtrG, StorageG>&& graph, ExtraArgs const&... rest)
      : _nCol(nCol),
        _ptr(std::move(graph.GetMovablePointers())),
        _idx(std::move(graph.GetMovableTargets())),
        _v(_idx.size(), Scalar(0), rest...) {
    static_assert(std::is_same_v<CRIdx const, CRIdxG const>, "Inconsistent row index types");
    static_assert(std::is_same_v<Ptr const, PtrG const>, "Inconsistent pointer index types");
    MOCHI_ASSERT(isize(_idx) == isize(_v), "Size mismatch");
    MOCHI_ASSERT(_ptr.empty() || (_ptr.back() == isize(_v)), "Invalid pointer offset.");
  }

  // Construct a SQUARE SparseMatrix by moving the sparsity pattern from a Graph.
  // The number of rows and columns is determined by the size of the graph.
  template <
      typename CRIdxG,
      typename PtrG,
      template <typename, typename...> typename StorageG,
      typename... ExtraArgs>
  explicit SparseMatrix(Graph<CRIdxG, PtrG, StorageG>&& graph, ExtraArgs const&... rest)
      : SparseMatrix(graph.size(), std::move(graph), rest...) {}

  // Conversion operator
  template <typename ToScalar, typename ToIdx = int, typename ToPtr = int>
  explicit operator SparseMatrix<ToScalar, ToIdx, ToPtr>() const {
    static_assert(!std::is_const_v<ToIdx>, "Destination index type must be non-const");
    static_assert(!std::is_const_v<ToPtr>, "Destination pointer type must be non-const");
    static_assert(!std::is_const_v<ToScalar>, "Destination scalar type must be non-const");
    //
    DynamicArray<ToIdx> newIdx;
    newIdx.resize_noinit(_idx.size());
    StaticCast<ToIdx>(MakeSpan(_idx), MakeSpan(newIdx));
    //
    DynamicArray<ToPtr> newPtr;
    newPtr.resize_noinit(_ptr.size());
    StaticCast<ToPtr>(MakeSpan(_ptr), MakeSpan(newPtr));
    //
    DynamicArray<ToScalar> newValues;
    newValues.resize_noinit(_v.size());
    StaticCast<ToScalar>(MakeSpan(_v), MakeSpan(newValues));
    //
    return SparseMatrix<ToScalar, ToIdx, ToPtr>(
        _nCol, std::move(newPtr), std::move(newIdx), std::move(newValues));
  }

  // Reset this SparseMatrix using the arguments for any of its constructors.
  template <typename... Args>
  SparseMatrix& Reset(Args&&... args) {
    this->~SparseMatrix();
    new (this) SparseMatrix(std::forward<Args>(args)...);
    return *this;
  }

  void SetZero() {
    memset(_v.data(), 0, _v.size() * sizeof(Scalar));
  }

  void SetConstant(Scalar value) {
    std::fill(_v.begin(), _v.end(), value);
  }

  /// @brief Set a single value.
  /// @note Asserts if trying to set a non-zero value outside the sparsity pattern.
  void SetValue(CRIdx row, CRIdx col, Scalar value) {
    auto index = FindEntry(row, col);
    if (index != _idx.size()) {
      _v[index] = value;
    } else {
      MOCHI_ASSERT(
          value == 0,
          "Attempting to set a non-zero entry that is not within the sparsity pattern of the matrix");
    }
  }

  CRIdx Rows() const {
    return _ptr.empty() ? 0 : (static_cast<CRIdx>(_ptr.size()) - 1);
  }

  [[nodiscard]] constexpr auto CERows() const {
    static_assert(
        std::is_same_v<NonConstIdx, int>,
        "SparseMatrix can only be used in matrix expressions if CRIdx = int");
    return details::IntOrEmpty<-1>{Rows()};
  }

  CRIdx Cols() const {
    return _nCol;
  }

  [[nodiscard]] constexpr auto CECols() const {
    static_assert(
        std::is_same_v<NonConstIdx, int>,
        "SparseMatrix can only be used in matrix expressions if CRIdx = int");
    return details::IntOrEmpty<-1>{Cols()};
  }

  Ptr NumNonZeros() const {
    return static_cast<Ptr>(_idx.size());
  }

  // @brief Number of non-zeros in the row range [rBegin, rEnd).
  // @note The range end is NOT inclusive.
  Ptr NumNonZerosInRowRange(CRIdx rBegin, CRIdx rEnd) const {
    MOCHI_ASSERT_VERBOSE(rBegin >= 0 && rEnd <= Rows() && rBegin <= rEnd, "Invalid row range.");
    return (_ptr[rEnd] - _ptr[rBegin]);
  }

  // Return true if there are zero rows or columns (see default constructor).
  [[nodiscard]] bool empty() const {
    return Rows() == 0 || Cols() == 0;
  }

  bool IsRowEmpty(CRIdx r) const {
    MOCHI_ASSERT_VERBOSE(r >= 0 && r < Rows(), "Out of range row index.");
    return (_ptr[r] == _ptr[r + 1]);
  }

  // Return true if not empty (i.e. it was initialized with non-zero rows & columns).
  explicit operator bool() const {
    return !empty();
  }

  /// @brief Returns the list of column indices for the row r
  /// @param[in] r Row index
  /// @returns Span of column indices
  ///
  auto Indices(CRIdx r) {
    MOCHI_ASSERT_VERBOSE(r >= 0 && r < Rows(), "Out of range row index.");
    return Span{_idx.data() + _ptr[r], static_cast<size_t>(NumNonZerosInRowRange(r, r + 1))};
  }

  /// @brief Returns the list of column indices for the row r
  /// @param[in] r Row index
  /// @returns Span of column indices
  ///
  auto Indices(CRIdx r) const {
    MOCHI_ASSERT_VERBOSE(r >= 0 && r < Rows(), "Out of range row index.");
    return Span{_idx.data() + _ptr[r], static_cast<size_t>(NumNonZerosInRowRange(r, r + 1))};
  }

  /// @brief Returns the list of numerical values for the row r
  /// @param[in] r Row index
  /// @returns Span of numerical values
  ///
  auto Values(CRIdx r) {
    MOCHI_ASSERT_VERBOSE(r >= 0 && r < Rows(), "Out of range row index.");
    return Span{_v.data() + _ptr[r], static_cast<size_t>(NumNonZerosInRowRange(r, r + 1))};
  }

  /// @brief Returns the list of numerical values for the row r
  /// @param[in] r Row index
  /// @returns Span of numerical values
  ///
  auto Values(CRIdx r) const {
    MOCHI_ASSERT_VERBOSE(r >= 0 && r < Rows(), "Out of range row index.");
    return Span{_v.data() + _ptr[r], static_cast<size_t>(NumNonZerosInRowRange(r, r + 1))};
  }

  /// @brief Application of the sparse matrix on a dense matrix, including a column vector.
  /// @note If the number of FLOPs is sufficiently large and a TaskScheduler is available and in
  /// multi-threaded mode (both global and thread-local), the computation is performed in parallel.
  template <typename MatrixIn, typename MatrixOut>
  void Apply(MatrixIn const& X, MatrixOut&& AX) const;

  /// @brief Application of the row subset [rowBegin, rowEnd) of the sparse matrix on a dense
  /// matrix, including a column vector.
  /// @note If the number of FLOPs is sufficiently large and a TaskScheduler is available and in
  /// multi-threaded mode (both global and thread-local), the computation is performed in parallel.
  template <typename MatrixIn, typename MatrixOut>
  void ApplyToRange(MatrixIn const& X, MatrixOut&& AX, CRIdx rowBegin, CRIdx rowEnd) const;

  /// @brief Accessor-based application of the row subset [rowBegin, rowEnd) of the sparse matrix on
  /// a dense matrix, including a column vector.
  /// @note If the number of FLOPs is sufficiently large and a TaskScheduler is available and in
  /// multi-threaded mode (both global and thread-local), the computation is performed in parallel.
  template <typename AccessorIn, typename AccessorOut>
  void AccessorApplyToRange(
      AccessorIn const& X,
      AccessorOut&& AX,
      CRIdx rowBegin,
      CRIdx rowEnd,
      int numColsX) const;

  template <typename MatrixIn, typename MatrixOut>
  void TransposeApply(MatrixIn const& x, MatrixOut&& Atx) const;

  Span<Scalar> Values() {
    return _v;
  }

  Span<Scalar const> Values() const {
    return _v;
  }

  Span<CRIdx const> Indices() const {
    return _idx;
  }

  Span<Ptr const> Pointers() const {
    return _ptr;
  }

  // Return a copy of this SparseMatrix
  auto Duplicate() const {
    return SparseMatrix<NonConstScalar, NonConstIdx, NonConstPtr>{*this};
  }

  // Square of the Frobenius norm
  Scalar NormSqr() const {
    ColumnVectorView<Scalar const> values{_v.data(), isize(_v)};
    return values.NormSqr();
  }

  // Frobenius norm
  Scalar Norm() const {
    return std::sqrt(NormSqr());
  }

  // Add a single value. Asserts if the coordinates are not within the sparsity pattern.
  void AddValue(CRIdx row, CRIdx col, Scalar value) {
    auto index = FindEntry(row, col);
    MOCHI_ASSERT_VERBOSE(
        index < _idx.size(),
        "Attempting to set an entry that is not within the sparsity pattern of the matrix");
    _v[index] += value;
  }

  /// @brief Read-access to matrix entry (r, c). If to write a value, use the 'Set' function.
  /// @returns a copy of the value or 0. It is const so users can't write: spmat(r, c) = value;
  Scalar operator()(CRIdx r, CRIdx c) const {
    auto index = FindEntry(r, c);
    return (index < _idx.size()) ? _v[index] : Scalar(0);
  }

  /// @brief  Add another SparseMatrix into this one.
  ///
  /// @note All non-zero values in the other matrix must be included in the non-zero pattern of this
  /// matrix.
  /// @note (Performance) If you know that the two matrices have the same sparsity pattern,
  /// then it would be better to add the values arrays directly.
  /// @note This operation assumes that the LHS sparse matrix has sorted column indices per row.
  template <
      typename OScalar,
      typename OIdx,
      typename OPtr,
      template <typename, typename...> typename OStorage>
  SparseMatrix& operator+=(SparseMatrix<OScalar, OIdx, OPtr, OStorage> const& other) {
    MOCHI_PROFILE_SCOPE();
    MOCHI_ASSERT(other.Rows() <= this->Rows());
    MOCHI_ASSERT(other.Cols() <= this->Cols());
    auto const numRows = other.Rows();
    for (std::remove_const_t<OIdx> r = 0; r < numRows; ++r) {
      auto otherIndices = other.Indices(r);
      if (!otherIndices.empty()) {
        auto myIndices = this->Indices(r);
        MOCHI_ASSERT_VERBOSE(
            std::is_sorted(myIndices.begin(), myIndices.end()), "Row entries are not sorted");
        auto myIndicesItr = myIndices.begin();
        auto myValues = this->Values(r);
        auto otherValues = other.Values(r);
        for (int otherIdx = 0; otherIdx < isize(otherIndices); ++otherIdx) {
          // Find the matching value in myIndices
          auto bc = otherIndices[otherIdx];
          // TODO Explore whether a version without `lower_bound` is useful
          auto next = std::lower_bound(myIndicesItr, myIndices.end(), bc);
          MOCHI_ASSERT_VERBOSE(
              (next != myIndices.end()) && (*next == bc),
              "Attempting to add values that are not supported by the sparsity pattern of this matrix.");
          // Add value to this entry
          auto myIdx = (next - myIndices.begin());
          myValues[myIdx] += otherValues[otherIdx];
          myIndicesItr = next + 1;
        }
      }
    }
    return *this;
  }

  /// @brief Find a specific entry (r, c) in the sparsity storage.
  /// @return Offset for the corresponding value in the array from `Values()`.
  /// It returns the number of non-zeros when it is not found.
  NonConstPtr FindEntry(CRIdx r, CRIdx c) const {
    MOCHI_ASSERT_VERBOSE((r >= 0) && (r < Rows()), "Out of range row index");
    MOCHI_ASSERT_VERBOSE((c >= 0) && (c < Cols()), "Out of range column index");
    auto const inds = Indices(r);
    //-- First - assume that the entries are sorted
    auto const dist =
        static_cast<int>(std::lower_bound(inds.begin(), inds.end(), c) - inds.begin());
    if ((dist != isize(inds)) && (inds[dist] == c)) {
      return static_cast<NonConstPtr>(_ptr[r] + dist);
    } else {
      //--- Try with find as backup
      auto const dist2 = static_cast<int>(std::find(inds.begin(), inds.end(), c) - inds.begin());
      if (dist2 != isize(inds)) {
        return static_cast<NonConstPtr>(_ptr[r] + dist2);
      }
    }
    return static_cast<NonConstPtr>(_idx.size()); // not in sparsity pattern
  }

 protected:
  NonConstIdx _nCol = 0;
  Storage<Ptr> _ptr;
  Storage<CRIdx> _idx;
  Storage<Scalar> _v;
};

template <typename Scalar, typename CRIdx = int, typename Ptr = int>
struct SparseMatrixViewType_ {
  template <typename T>
  using Storage = Span<T, CRIdx>;
  using type = SparseMatrix<Scalar, CRIdx const, Ptr const, Storage>;
};

template <typename Scalar, typename CRIdx = int, typename Ptr = int>
using SparseMatrixView = typename SparseMatrixViewType_<Scalar, CRIdx, Ptr>::type;

} // namespace mochi

namespace mochi::details {
template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
constexpr bool IsSparseMatrixDef<SparseMatrix<Scalar, CRIdx, Ptr, Storage>> = true;
} // namespace mochi::details

namespace mochi {

// Create a SparseMatrixView pointing to the data owned by a SparseMatrix
template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
auto AsView(SparseMatrix<Scalar, CRIdx, Ptr, Storage>& mat) {
  return SparseMatrixView<Scalar, CRIdx, Ptr>{
      mat.Cols(), Span{mat.Pointers()}, Span{mat.Indices()}, Span{mat.Values()}};
}

// Create a const SparseMatrixView pointing to the data owned by a SparseMatrix
template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
auto AsConstView(SparseMatrix<Scalar, CRIdx, Ptr, Storage> const& mat) {
  return SparseMatrixView<Scalar const, CRIdx const, Ptr const>{
      mat.Cols(), Span{mat.Pointers()}, Span{mat.Indices()}, Span{mat.Values()}};
}

} // namespace mochi

//////////////////////////////
//
//////////////////////////////

namespace mochi::details {

/// @brief Routine to compute result += A(row, :) * x[:], where A is a sparse matrix.
///
/// @tparam Scalar
/// @tparam Idx
/// @param[in] colIdx Indices for the non-zero values of the sparse row.
/// @param[in] values Non-zero values for the sparse row.
/// @param[in] x Vector with one column (column-major orientation).
/// @param[out] result Scalar where the product is added.
template <typename Scalar, typename Idx>
MOCHI_FORCE_INLINE void AddRowSparseTimesColumnVector(
    Span<Idx const> colIdx,
    Span<Scalar const> values,
    Scalar const* x,
    std::remove_const_t<Scalar>& result) {
  // Performance notes:
  // - This kernel could be implemented using 'LoadIndexed'. On x64, using 'LoadIndexed'
  //   is substantially slower, both on AMD and Intel. On ARM, using 'LoadIndexed' should compile
  //   to the same as the current implementation, but the latter is faster on some machines.
  // - Even if the native SIMD size is 8, there doesn't seem to be a performance improvement using
  //   SIMD size of 8 instead of 4 (and if the number of non-zeros per row is small, using SIMD size
  //   of 4 is faster).
  // - Looping over entries in batches of 2 SIMD vectors at a time is faster on some machines than
  //   looping 1 SIMD vector at a time.
  // - On most architectures, using a scalar (non-vectorized) implementation for the (up to 3)
  //   leftover entries is faster than using partial SIMD loads and stores.
  // - For some architectures and compilers, the cost of creating the indices and values spans on
  //   the caller end could be reduced by using moving pointers, but the overall performance
  //   improvement is usually <10%.
  using NonConstIdx = std::remove_const_t<Idx>;
  constexpr int kSimdSize = 4;
  using V4 = Simd<std::remove_const_t<Scalar>, kSimdSize>;
  NonConstIdx j = 0;
  if constexpr (V4::kIsSupported) {
    V4 vResult = {}; // Initializes to zero.
    for (; j + 2 * kSimdSize <= colIdx.size(); j += 2 * kSimdSize) {
      vResult += Load<V4>(values.data() + j) *
              V4{x[colIdx[j + 0]], x[colIdx[j + 1]], x[colIdx[j + 2]], x[colIdx[j + 3]]} +
          Load<V4>(values.data() + j + kSimdSize) *
              V4{x[colIdx[j + 4]], x[colIdx[j + 5]], x[colIdx[j + 6]], x[colIdx[j + 7]]};
    }
    if (j + kSimdSize <= colIdx.size()) {
      vResult += Load<V4>(values.data() + j) *
          V4{x[colIdx[j + 0]], x[colIdx[j + 1]], x[colIdx[j + 2]], x[colIdx[j + 3]]};
      j += kSimdSize;
    }
    result += HSum(vResult);
  }
  for (; j < colIdx.size(); ++j) {
    result += values[j] * x[colIdx[j]];
  }
}

/// @brief Routine to compute Y(i, :) = A(i, :) * X(:, :), where A is a sparse matrix, X is a
/// row-major accessor, and Y is a row-major or col-major accessor.
///
/// @tparam Scalar
/// @tparam Idx
/// @tparam AccessorIn
/// @tparam AccessorOut
/// @param[in] i Row index.
/// @param[in] colIdx Indices of the non-zero columns in the sparse row.
/// @param[in] values Non-zero values in the sparse row.
/// @param[in] X Input accessor with row-major storage.
/// @param[out] Y Output accessor with row-major or col-major storage.
///
/// @details Most performant if 'Y' is also stored row-major.
/// @details Most performant if the number of columns in 'X' and 'Y' is a multiple of the SIMD
/// vector size.
template <typename Scalar, typename Idx, typename AccessorIn, typename AccessorOut>
MOCHI_FORCE_INLINE void RowSparseTimesRowMajorDense(
    Idx i,
    Span<Idx const> colIdx,
    Span<Scalar const> values,
    AccessorIn const& X,
    AccessorOut&& Y,
    int numColsX) {
  static_assert(AccessorIn::RowColCosts().second == 1, "Input accessor must be row-major");
  using NonConstScalar = std::remove_const_t<Scalar>;
  using NonConstIdx = std::remove_const_t<Idx>;
  using VType = Simd<NonConstScalar>; // Native SIMD size.
  using VType2x =
      Simd<NonConstScalar, 2 * VType::kSize>; // 2x the native SIMD size to hide latency.
  static_assert( // Non-SIMD fallback was deprecated in D63399208.
      VType::kIsSupported && VType2x::kIsSupported,
      "Implementation requires a scalar type with SIMD support");
  //--- Loop over columns in the input and output matrices (outer loops) and non-zeros in the
  //--- sparse row (inner loops) using SIMD instructions along the columns in the input and output
  //--- matrices.
  constexpr auto kIncr = static_cast<Idx>(VType::kSize);
  NonConstIdx k = 0;
  if (numColsX >= kIncr) {
    for (; k + VType2x::kSize <= numColsX; k += VType2x::kSize) {
      VType2x yData = {};
      for (NonConstIdx j = 0; j < colIdx.size(); ++j) {
        yData += values[j] * X.template RowVector<VType2x>(colIdx[j], k);
      }
      Y.StoreRowVector(i, k, yData);
    }
    if (k + kIncr <= numColsX) {
      VType yData = {};
      for (NonConstIdx j = 0; j < colIdx.size(); ++j) {
        yData += values[j] * X.template RowVector<VType>(colIdx[j], k);
      }
      Y.StoreRowVector(i, k, yData);
      k += kIncr;
    }
  }
  //--- Leftover columns. TODO: Implementation without partial SIMD loads and stores may be faster
  //--- on some architectures. Reference implementation: P871406589
  Idx const numLeftoverCols = numColsX - k;
  if (numLeftoverCols > 1) {
    MOCHI_ASSERT_VERBOSE(numLeftoverCols < kIncr, "Unexpected number of leftover columns.");
    VType yData = {};
    for (NonConstIdx j = 0; j < colIdx.size(); ++j) {
      yData += values[j] * X.template RowVector<VType>(colIdx[j], k, numLeftoverCols);
    }
    Y.StoreRowVector(i, k, yData, numLeftoverCols);
  } else if (numLeftoverCols == 1) {
    NonConstScalar yTmp = 0;
    for (NonConstIdx j = 0; j < colIdx.size(); ++j) {
      yTmp += values[j] * X(colIdx[j], k);
    }
    Y.Store(i, k, yTmp);
  }
}

/// @brief Routine to compute result += A(row, :) * x(:, 0), where A is a sparse matrix.
///
/// @tparam Scalar
/// @tparam Idx
/// @tparam Input
/// @param[in] colIdx Indices for the non-zero values of the sparse row
/// @param[in] values Non-zero values for the sparse row
/// @param[in] x Vector with one column (no assumption on storage orientation)
/// @param[out] result Scalar where the product is added
///
/// @note This implementation is "plain vanilla" and not optimized.
template <typename Scalar, typename Idx, typename Input>
MOCHI_FORCE_INLINE void AddRowSparseTimesVector(
    Span<Idx const> colIdx,
    Span<Scalar const> values,
    Input const& x,
    std::remove_const_t<Scalar>& result) {
  using NonConstIdx = std::remove_const_t<Idx>;
  for (NonConstIdx j = 0; j < colIdx.size(); j += 1) {
    result += values[j] * x(colIdx[j], 0);
  }
}

} // namespace mochi::details

namespace mochi {

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <typename MatrixIn, typename MatrixOut>
void SparseMatrix<Scalar, CRIdx, Ptr, Storage>::Apply(MatrixIn const& X, MatrixOut&& AX) const {
  static_assert(
      std::is_same_v<Scalar const, typename details::MatTraits<MatrixIn>::Scalar const> &&
          std::is_same_v<NonConstScalar, typename details::MatTraits<MatrixOut>::Scalar>,
      "Inconsistent scalar types");
  MOCHI_ASSERT_VERBOSE(
      (X.Cols() == AX.Cols()) && (X.Rows() == this->Cols()) && (AX.Rows() == this->Rows()),
      "Inconsistent dimensions");
  AccessorApplyToRange(details::GetAccessor(X), details::GetAccessor(AX), 0, Rows(), X.Cols());
}

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <typename MatrixIn, typename MatrixOut>
void SparseMatrix<Scalar, CRIdx, Ptr, Storage>::ApplyToRange(
    MatrixIn const& X,
    MatrixOut&& AX,
    CRIdx rowBegin,
    CRIdx rowEnd) const {
  static_assert(
      std::is_same_v<Scalar const, typename details::MatTraits<MatrixIn>::Scalar const> &&
          std::is_same_v<NonConstScalar, typename details::MatTraits<MatrixOut>::Scalar>,
      "Inconsistent scalar types");
  MOCHI_ASSERT_VERBOSE(
      (X.Cols() == AX.Cols()) && (X.Rows() == this->Cols()) && (AX.Rows() == this->Rows()),
      "Inconsistent dimensions");
  AccessorApplyToRange(
      details::GetAccessor(X), details::GetAccessor(AX), rowBegin, rowEnd, X.Cols());
}

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <typename AccessorIn, typename AccessorOut>
void SparseMatrix<Scalar, CRIdx, Ptr, Storage>::AccessorApplyToRange(
    AccessorIn const& X,
    AccessorOut&& AX,
    CRIdx rowBegin,
    CRIdx rowEnd,
    int numColsX) const {
  MOCHI_ASSERT_VERBOSE(numColsX >= 0, "Invalid number of columns.");
  MOCHI_ASSERT_VERBOSE(
      rowBegin >= 0 && rowBegin <= rowEnd && rowEnd <= this->Rows(), "Invalid row range.");
  if (rowBegin == rowEnd) {
    return;
  }

  // If 'X' and 'AX' have multiple columns, they count as half since the incremental overhead is
  // smaller.
  constexpr long long kMinFlopsPerTask = 50000; // ~5 μs @ 10 GFLOP/s
  auto const numRows = rowEnd - rowBegin;
  auto const minRowsPerTask = Clamp<CRIdx>(
      static_cast<CRIdx>(
          kMinFlopsPerTask * numRows /
          Max<Ptr>(1, 2 * NumNonZerosInRowRange(rowBegin, rowEnd) * ((numColsX + 1) / 2))),
      1,
      numRows);
  ParallelForRange(
      "SparseMatrixProduct",
      rowBegin,
      rowEnd,
      minRowsPerTask,
      numRows,
      [&](CRIdx rowBeginTask, CRIdx rowEndTask) {
        constexpr auto kCostsX = AccessorIn::RowColCosts();
        if constexpr (kCostsX.first == 1) {
          //--- X is col-major
          for (NonConstIdx c = 0; c < numColsX; ++c) {
            for (NonConstIdx r = rowBeginTask; r < rowEndTask; ++r) {
              NonConstScalar result = 0;
              if (!IsRowEmpty(r)) {
                auto const rowIndices = Indices(r);
                auto const valRow = Values(r);
                details::AddRowSparseTimesColumnVector<NonConstScalar, NonConstIdx>(
                    rowIndices, valRow, &X(0, c), result);
              }
              AX.Store(r, c, result);
            }
          }
        } else {
          //--- X is row-major
          static_assert(kCostsX.second == 1, "Unexpected cost");
          for (NonConstIdx r = rowBeginTask; r < rowEndTask; ++r) {
            if (!IsRowEmpty(r)) {
              auto const rowIndices = Indices(r);
              auto const rowValues = Values(r);
              details::RowSparseTimesRowMajorDense<Scalar, CRIdx>(
                  r, rowIndices, rowValues, X, AX, numColsX);
            } else {
              for (int c = 0; c < numColsX; ++c) {
                AX.Store(r, c, 0);
              }
            }
          }
        }
      });
}

template <
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <typename MatrixIn, typename MatrixOut>
void SparseMatrix<Scalar, CRIdx, Ptr, Storage>::TransposeApply(MatrixIn const& X, MatrixOut&& AtX)
    const {
  static_assert(
      std::is_same_v<
          NonConstScalar,
          std::remove_const_t<typename details::MatTraits<MatrixIn>::Scalar>>,
      "Scalar types must be compatible");
  static_assert(
      std::is_same_v<NonConstScalar, typename details::MatTraits<MatrixOut>::Scalar>,
      "Scalar types must be compatible");
  MOCHI_ASSERT_VERBOSE(
      (X.Cols() == AtX.Cols()) && (X.Rows() == this->Rows()) && (AtX.Rows() == this->Cols()),
      "Dimensions do not match");
  AtX.SetZero();
  if (X.Cols() == 1) {
    for (NonConstIdx i = 0; i < Rows(); ++i) {
      auto idxTmp = Indices(i);
      auto vrow = Values(i);
      auto const xi = X(i, 0);
      NonConstPtr j = 0;
      using SimdT = Simd<NonConstScalar>;
      if constexpr (SimdT::kIsSupported) {
        auto const xval = Broadcast<SimdT>(xi);
        for (; j + SimdT::kSize <= idxTmp.size(); j += SimdT::kSize) {
          auto const aval = Load<SimdT>(&vrow[j]);
          auto const rhs = aval * xval;
          for (int k = 0; k < SimdT::kSize; ++k) {
            AtX(idxTmp[j + k], 0) += Get(rhs, k);
          }
        }
      }
      for (; j < idxTmp.size(); ++j) {
        AtX(idxTmp[j], 0) += vrow[j] * xi;
      }
    }
    return;
  }
  //
  // Case with multiple columns
  //
  // TODO Optimize or parallelize
  for (NonConstIdx i = 0; i < Rows(); ++i) {
    auto idxTmp = Indices(i);
    auto vrow = Values(i);
    auto const Xi = X.Row(i);
    for (NonConstIdx j = 0; j < idxTmp.size(); ++j) {
      AtX.Row(idxTmp[j]) += vrow[j] * Xi;
    }
  }
}

} // namespace mochi
