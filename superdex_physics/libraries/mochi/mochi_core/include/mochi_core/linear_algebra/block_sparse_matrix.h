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

#include <mochi_core/linear_algebra/block_one_d_view.h>
#include <mochi_core/linear_algebra/block_view_vector.h>
#include <mochi_core/linear_algebra/krylov/tools/tensor_functions.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_accessors.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/graph.h>
#include <mochi_core/utils/math_utils.h>
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

/// @brief Block sparse matrix with block sparse row (BSR) format.
///
/// @tparam Scalar_ Scalar type for entries
/// @tparam CRIdx Signed integral type for the block column indices
/// @tparam Ptr Signed integral type for the pointer in the block column index array
/// @tparam Storage Container pointing to the data (either owning or viewing). Default value is
/// `DynamicArray` (owning).
template <
    typename Scalar_,
    int kBlockSize_,
    typename CRIdx = int,
    typename Ptr = int,
    template <typename, typename...> typename Storage = DynamicArray>
class BlockSparseMatrix {
 public:
  using Scalar = Scalar_;
  using NonConstScalar = std::remove_const_t<Scalar>;
  using NonConstIdx = std::remove_const_t<CRIdx>;
  using NonConstPtr = std::remove_const_t<Ptr>;
  static constexpr int kBlockSize = kBlockSize_;

  static_assert(kBlockSize > 0, "Block size must be positive");
  static_assert(
      std::signed_integral<CRIdx> && std::signed_integral<Ptr>,
      "BlockSparseMatrix requires signed integer types.");

  BlockSparseMatrix() = default;

  BlockSparseMatrix(BlockSparseMatrix&& rhs) noexcept = default;
  BlockSparseMatrix(BlockSparseMatrix const& rhs) = default;

  // TODO: Should these copy operators be change?
  // Their behavior is inconsistent with Matrix<Scalar> in two ways:
  //   1) Assignment can change the dimensions of this matrix.
  //   2) Assignment will NOT copy values if this is a BlockSparseMatrixView (unlike MatrixView).
  BlockSparseMatrix& operator=(BlockSparseMatrix&& rhs) noexcept {
    // Polymorphic_allocator does not propagate on container copy assignment, move assignment, or
    // swap. As a result, move assignment of a polymorphic_allocator-using container can throw, and
    // swapping two polymorphic_allocator-using containers whose allocators do not compare equal
    // results in undefined behavior.
    if (std::addressof(rhs) != this) {
      this->Reset(std::move(rhs));
    }
    return *this;
  }

  BlockSparseMatrix& operator=(BlockSparseMatrix const& rhs) = default;

  // Construct by copying a const BlockSparseMatrix of compatible type.
  // Performs a shallow or deep copy of the values, depending on the Storage type.
  template <
      typename OScal,
      typename OIdx,
      typename OPtr,
      template <typename, typename...> typename OStorage>
  BlockSparseMatrix(BlockSparseMatrix<OScal, kBlockSize, OIdx, OPtr, OStorage> const& other)
      : _nBlockCols(other.BlockCols()),
        _ptr(other._ptr.data(), other._ptr.data() + other._ptr.size()),
        _idx(other._idx.data(), other._idx.data() + other._idx.size()),
        _v(other._v.data(), other._v.data() + other._v.size()),
        _maxNnzPerRow(other._maxNnzPerRow) {}

  // Construct by copying a non-const BlockSparseMatrix of compatible type.
  // Performs a shallow or deep copy of the values, depending on the Storage type.
  template <
      typename OScal,
      typename OIdx,
      typename OPtr,
      template <typename, typename...> typename OStorage>
  BlockSparseMatrix(BlockSparseMatrix<OScal, kBlockSize, OIdx, OPtr, OStorage>& other)
      : _nBlockCols(other.BlockCols()),
        _ptr(other._ptr.data(), other._ptr.data() + other._ptr.size()),
        _idx(other._idx.data(), other._idx.data() + other._idx.size()),
        _v(other._v.data(), other._v.data() + other._v.size()),
        _maxNnzPerRow(other._maxNnzPerRow) {}

  BlockSparseMatrix(CRIdx nBlockCol_, Storage<Ptr> ptr_, Storage<CRIdx> idx_, Storage<Scalar> v_)
      : _nBlockCols(nBlockCol_),
        _ptr(std::move(ptr_)),
        _idx(std::move(idx_)),
        _v(std::move(v_)),
        _maxNnzPerRow(0) {
    MOCHI_ASSERT_VERBOSE(isize(_idx) * kBlockSize * kBlockSize == isize(_v), "Size mismatch");
    GetMaxNnzPerRow();
  }

  template <
      typename CRIdxG,
      typename PtrG,
      template <typename, typename...> typename StorageG,
      typename... ExtraArgs>
  BlockSparseMatrix(
      CRIdx nBlockCol,
      Graph<CRIdxG, PtrG, StorageG> const& graph,
      ExtraArgs const&... rest)
      : _nBlockCols(nBlockCol),
        _ptr(graph.GetPointers().begin(), graph.GetPointers().end()),
        _idx(graph.GetTargets().begin(), graph.GetTargets().end()),
        _v(kBlockSize * kBlockSize * graph.NumTargets(), Scalar(0), rest...),
        _maxNnzPerRow(0) {
    static_assert(std::is_same_v<CRIdx const, CRIdxG const>, "Inconsistent row index types");
    static_assert(std::is_same_v<Ptr const, PtrG const>, "Inconsistent pointer index types");
    if constexpr (sizeof...(ExtraArgs) > 0) {
      static_assert(std::is_same_v<Storage<Scalar>, DynamicArray<Scalar>>);
    }
    GetMaxNnzPerRow();
  }

  template <
      typename CRIdxG,
      typename PtrG,
      template <typename, typename...> typename StorageG,
      typename... ExtraArgs>
  BlockSparseMatrix(
      CRIdx nBlockCol,
      Graph<CRIdxG, PtrG, StorageG>&& graph,
      ExtraArgs const&... rest)
      : _nBlockCols(nBlockCol),
        _ptr(std::move(graph.GetMovablePointers())),
        _idx(std::move(graph.GetMovableTargets())),
        _v(kBlockSize * kBlockSize * _idx.size(), Scalar(0), rest...),
        _maxNnzPerRow(0) {
    static_assert(std::is_same_v<CRIdx const, CRIdxG const>, "Inconsistent row index types");
    static_assert(std::is_same_v<Ptr const, PtrG const>, "Inconsistent pointer index types");
    if constexpr (sizeof...(ExtraArgs) > 0) {
      static_assert(std::is_same_v<Storage<Scalar>, DynamicArray<Scalar>>);
    }
    GetMaxNnzPerRow();
  }

  // Construct a SQUARE BlockSparseMatrix by moving the sparsity pattern from a Graph.
  // The number of block rows and block columns is determined by the size of the graph.
  template <
      typename CRIdxG,
      typename PtrG,
      template <typename, typename...> typename StorageG,
      typename... ExtraArgs>
  explicit BlockSparseMatrix(Graph<CRIdxG, PtrG, StorageG>&& graph, ExtraArgs const&... rest)
      : BlockSparseMatrix(graph.size(), std::move(graph), rest...) {}

  // Conversion operator
  template <typename ToScalar, typename ToIdx = int, typename ToPtr = int>
  explicit operator BlockSparseMatrix<ToScalar, kBlockSize, ToIdx, ToPtr>() const {
    static_assert(!std::is_const_v<ToIdx>, "Destination index type must be non-const");
    static_assert(!std::is_const_v<ToPtr>, "Destination pointer type must be non-const");
    static_assert(!std::is_const_v<ToScalar>, "Destination scalar type must be non-const");

    DynamicArray<ToIdx> newIdx;
    newIdx.resize_noinit(_idx.size());
    StaticCast<ToIdx>(MakeSpan(_idx), MakeSpan(newIdx));

    DynamicArray<ToPtr> newPtr;
    newPtr.resize_noinit(_ptr.size());
    StaticCast<ToPtr>(MakeSpan(_ptr), MakeSpan(newPtr));

    DynamicArray<ToScalar> newValues;
    newValues.resize_noinit(_v.size());
    StaticCast<ToScalar>(MakeSpan(_v), MakeSpan(newValues));

    return BlockSparseMatrix<ToScalar, kBlockSize, ToIdx, ToPtr>(
        _nBlockCols, std::move(newPtr), std::move(newIdx), std::move(newValues));
  }

  // Reset this BlockSparseMatrix using the arguments for any of its constructors.
  template <typename... Args>
  BlockSparseMatrix& Reset(Args&&... args) {
    this->~BlockSparseMatrix();
    new (this) BlockSparseMatrix(std::forward<Args>(args)...);
    return *this;
  }

  BlockSparseMatrix& SetZero() {
    if (!_v.empty())
      MOCHI_LIKELY {
        memset(_v.data(), 0, _v.size() * sizeof(Scalar));
      }
    return *this;
  }

  // Set a single value. Asserts if trying to set a non-zero value outside the sparsity pattern.
  void SetValue(CRIdx r, CRIdx c, Scalar value) {
    auto vIndex = FindValueIndex(r, c);
    if (vIndex >= 0) {
      _v[vIndex] = value;
    } else {
      MOCHI_ASSERT(
          value == 0,
          "Attempting to set a non-zero entry that is not within the sparsity pattern of the matrix.");
    }
  }

  /// @brief Returns the maximum number of non-zeros per row.
  CRIdx MaxNnzPerRow() const {
    return _maxNnzPerRow;
  }

  // @brief Returns the total number of rows in the matrix.
  // It is a multiple of kBlockSize.
  CRIdx Rows() const {
    return BlockRows() * static_cast<CRIdx>(kBlockSize);
  }

  // @brief Returns the number of blocks in the row direction.
  // Note that kBlockSize * BlockRows() is equal to rows()
  CRIdx BlockRows() const {
    return _ptr.empty() ? 0 : static_cast<CRIdx>(_ptr.size() - 1);
  }

  [[nodiscard]] constexpr auto CERows() const {
    static_assert(
        std::is_same_v<NonConstIdx, int>,
        "BlockSparseMatrix can only be used in matrix expressions if CRIdx = int");
    return details::IntOrEmpty<-1>{Rows()};
  }

  // @brief Returns the total number of columns in the matrix.
  // It is a multiple of kBlockSize.
  CRIdx Cols() const {
    return static_cast<CRIdx>(kBlockSize) * _nBlockCols;
  }

  // @brief Returns the number of blocks in the column direction.
  // Note that kBlockSize * BlockCols() is equal to cols()
  CRIdx BlockCols() const {
    return _nBlockCols;
  }

  [[nodiscard]] constexpr auto CECols() const {
    static_assert(
        std::is_same_v<NonConstIdx, int>,
        "BlockSparseMatrix can only be used in matrix expressions if CRIdx = int");
    return details::IntOrEmpty<-1>{Cols()};
  }

  Ptr NumNonZeros() const {
    return static_cast<Ptr>(_idx.size()) * kBlockSize * kBlockSize;
  }

  // @brief Number of non-zero blocks in the block row range [brBegin, brEnd).
  // @note The range end is NOT inclusive.
  Ptr NumNonZeroBlocksInBlockRowRange(CRIdx brBegin, CRIdx brEnd) const {
    MOCHI_ASSERT_VERBOSE(
        brBegin >= 0 && brEnd <= BlockRows() && brBegin <= brEnd, "Invalid range.");
    return (_ptr[brEnd] - _ptr[brBegin]);
  }

  Ptr NumNonZeroBlocks() const {
    return static_cast<Ptr>(_idx.size());
  }

  auto Indices(CRIdx r_c) const {
    return Span<CRIdx const>{
        _idx.data() + _ptr[r_c],
        static_cast<typename Span<CRIdx const>::size_type>(_ptr[r_c + 1] - _ptr[r_c])};
  }

  auto Values(CRIdx r_c) {
    // TODO Revise the pointer offset when implementing aligned starts of row
    // data.
    auto const numBlocks = _ptr[r_c + 1] - _ptr[r_c];
    return BlockRowView<Scalar, kBlockSize, CRIdx>{
        _v.data() + kBlockSize * kBlockSize * _ptr[r_c],
        static_cast<CRIdx>(kBlockSize * numBlocks),
        static_cast<CRIdx>(numBlocks)};
  }

  auto Values(CRIdx r_c) const {
    // TODO Revise the pointer offset when implementing aligned starts of row
    // data.
    auto const numBlocks = _ptr[r_c + 1] - _ptr[r_c];
    return BlockRowView<Scalar const, kBlockSize, CRIdx>{
        _v.data() + kBlockSize * kBlockSize * _ptr[r_c],
        static_cast<CRIdx>(kBlockSize * numBlocks),
        static_cast<CRIdx>(numBlocks)};
  }

  /// @brief Application of the block sparse matrix on a dense matrix, including a column vector.
  /// @note If the number of FLOPs is sufficiently large and a TaskScheduler is available and in
  /// multi-threaded mode (both global and thread-local), the computation is performed in parallel.
  template <typename VectorIn, typename VectorOut>
  void Apply(VectorIn const& x, VectorOut&& Ax) const;

  /// @brief Application of the row subset [rowBegin, rowEnd) of the block sparse matrix on a dense
  /// matrix, including a column vector.
  /// @note If the number of FLOPs is sufficiently large and a TaskScheduler is available and in
  /// multi-threaded mode (both global and thread-local), the computation is performed in parallel.
  template <typename VectorIn, typename VectorOut>
  void ApplyToRange(VectorIn const& x, VectorOut&& Ax, CRIdx rowBegin, CRIdx rowEnd) const;

  /// @brief Accessor-based application of the row subset [rowBegin, rowEnd) of the block sparse
  /// matrix on a dense matrix, including a column vector.
  /// @note If the number of FLOPs is sufficiently large and a TaskScheduler is available and in
  /// multi-threaded mode (both global and thread-local), the computation is performed in parallel.
  template <typename AccessorIn, typename AccessorOut>
  void AccessorApplyToRange(
      AccessorIn const& x,
      AccessorOut&& Ax,
      CRIdx rowBegin,
      CRIdx rowEnd,
      int nCols) const;

  template <typename VectorIn, typename VectorOut>
  void TransposeApply(VectorIn const& xv, VectorOut&& Axv) const;

  /// @brief Add another BlockSparseMatrix into this one.
  ///
  /// @note All non-zero values in the other matrix must be included in the non-zero pattern of this
  /// matrix.
  /// @note (Performance) If you know that the two matrices have the same sparsity pattern,
  /// then it would be better to add the values arrays directly.
  /// @note This operation assumes that the LHS block-sparse matrix has sorted block-column indices
  /// per block-row.
  template <
      typename OScalar,
      typename OIdx,
      typename OPtr,
      template <typename, typename...> typename OStorage>
  BlockSparseMatrix& operator+=(
      BlockSparseMatrix<OScalar, kBlockSize, OIdx, OPtr, OStorage> const& other) {
    MOCHI_PROFILE_SCOPE();
    MOCHI_ASSERT(other.BlockRows() <= this->BlockRows());
    MOCHI_ASSERT(other.BlockCols() <= this->BlockCols());
    auto const numBlockRows = other.BlockRows();
    for (std::remove_const_t<OIdx> br = 0; br < numBlockRows; ++br) {
      auto otherIndices = other.Indices(br);
      if (!otherIndices.empty()) {
        auto myIndices = this->Indices(br);
        MOCHI_ASSERT_VERBOSE(
            std::is_sorted(myIndices.begin(), myIndices.end()), "Row entries are not sorted");
        auto myIndicesItr = myIndices.begin();
        auto myValues = this->Values(br);
        auto otherValues = other.Values(br);
        for (int otherIdx = 0; otherIdx < isize(otherIndices); ++otherIdx) {
          // Find the matching block in myIndices
          auto bc = otherIndices[otherIdx];
          // TODO Explore whether a version without `lower_bound` is useful
          auto next = std::lower_bound(myIndicesItr, myIndices.end(), bc);
          MOCHI_ASSERT_VERBOSE(
              (next != myIndices.end()) && (*next == bc),
              "Attempting to add values that are not supported by the sparsity pattern of this matrix.");
          // Add values in this block
          auto myIdx = (next - myIndices.begin());
          myValues[myIdx] += otherValues[otherIdx];
          myIndicesItr = next + 1;
        }
      }
    }
    return *this;
  }

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

  // Return a copy of this BlockSparseMatrix
  auto Duplicate() const {
    return BlockSparseMatrix<NonConstScalar, kBlockSize, NonConstIdx, NonConstPtr>{*this};
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

  // Return true if there are zero rows or columns (see default constructor).
  bool empty() const {
    return Rows() == 0 || Cols() == 0;
  }

  bool IsBlockRowEmpty(CRIdx br) const {
    MOCHI_ASSERT_VERBOSE(br >= 0 && br < BlockRows(), "Out of range block row index.");
    return (_ptr[br] == _ptr[br + 1]);
  }

  // Return true if not empty (i.e. it was initialized with non-zero rows & columns).
  explicit operator bool() const {
    return !empty();
  }

  /// @brief Read-only access to matrix entry (r, c)
  auto operator()(CRIdx r, CRIdx c) const;

 protected:
  /// @brief Count the maximum number of non-zero entries per row
  void GetMaxNnzPerRow() {
    for (NonConstIdx i = 0; i + 1 < _ptr.size(); ++i) {
      _maxNnzPerRow = std::max<NonConstIdx>(_maxNnzPerRow, _ptr[i + 1] - _ptr[i]);
    }
    _maxNnzPerRow *= kBlockSize;
  }

  /// @brief Find a row and column in the sparsity pattern. Returns the offset of the corresponding
  /// value in _v, or -1 if not found.
  auto FindValueIndex(CRIdx r, CRIdx c) const {
    MOCHI_ASSERT_VERBOSE((r >= 0) && (r < Rows()), "Out of range row index");
    MOCHI_ASSERT_VERBOSE((c >= 0) && (c < Cols()), "Out of range column index");
    using SignedPtr = std::make_signed_t<Ptr>;
    auto const br = r / kBlockSize;
    auto const bc = c / kBlockSize;
    auto const bInds = Indices(br);
    auto const dist = static_cast<int>(std::find(bInds.begin(), bInds.end(), bc) - bInds.begin());
    if (dist != isize(bInds)) {
      auto const lr = r % kBlockSize;
      auto const lc = c % kBlockSize;
      return static_cast<SignedPtr>(
          kBlockSize * kBlockSize * _ptr[br] + kBlockSize * (lr * isize(bInds) + dist) + lc);
    }
    return static_cast<SignedPtr>(-1); // Not in the sparsity pattern
  }

 protected:
  NonConstIdx _nBlockCols = 0;
  Storage<Ptr> _ptr;
  Storage<CRIdx> _idx;
  Storage<Scalar> _v;
  NonConstIdx _maxNnzPerRow = 0;

  template <
      typename OScal,
      int kOBS,
      typename OIdx,
      typename OPtr,
      template <typename, typename...> typename OStorage>
  friend class BlockSparseMatrix;
};

} // namespace mochi

namespace mochi::details {
template <
    typename Scalar,
    int kBlockSize_,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
constexpr bool IsBlockSparseMatrixDef<BlockSparseMatrix<Scalar, kBlockSize_, CRIdx, Ptr, Storage>> =
    true;

template <typename Scalar, int kBlockSize, typename CRIdx, typename Ptr>
struct BlockSparseMatrixViewType_ {
  template <typename T>
  using Storage = Span<T, Ptr>;
  using Type = BlockSparseMatrix<Scalar, kBlockSize, CRIdx const, Ptr const, Storage>;
  static_assert(
      std::numeric_limits<CRIdx>::max() <= std::numeric_limits<Ptr>::max(),
      "Integral combination not implemented");
};
} // namespace mochi::details

namespace mochi {

template <typename Scalar, int kBlockSize, typename CRIdx = int, typename Ptr = int>
using BlockSparseMatrixView =
    typename details::BlockSparseMatrixViewType_<Scalar, kBlockSize, CRIdx, Ptr>::Type;

/// @brief Function checking whether the sparse CSR matrix has a blocked structure
///     with the specified block size.
///
/// @returns True or false whether the input matrix can be converted
///       into a `BlockSparseMatrix` object.
///
/// @note The function tests whether:
/// - for each row at kBlockSize * jb + {0, 1, ... kBlockSize},
///    the column indices in the input sparse matrix are identical
/// - for each row, the column indices in the input sparse matrix are "block-complete"
///     (i.e. A has entries at kBlockSize * jb + {0, 1, ... kBlockSize} )
/// - for each row, the column indices in the input sparse matrix are sorted per block
///     (i.e. A has sorted column indices at kBlockSize * jb + {0, 1, ... kBlockSize} )
template <
    int kBlockSize,
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
bool IsBlockable(SparseMatrix<Scalar, CRIdx, Ptr, Storage> const& A);

/// Overload for dense matrices. A dense matrix is considered blockable if and only if the number of
/// rows and the number of columns are a multiple of the block size. From this definition, an empty
/// matrix is also blockable.
template <
    int kBlockSize,
    typename Scalar,
    int kRowsAtCT,
    int kColsAtCT,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDimension>
bool IsBlockable(
    Matrix<Scalar, kRowsAtCT, kColsAtCT, kMajorDirection, kOwnership, kLeadingDimension> const& A);

/// @brief Structure for the graph of a `BlockSparseMatrix`
template <int kBlockSize, typename CRIdx = int, typename Ptr = int>
struct BlockViewStructure {
  std::remove_const_t<CRIdx> nBlockCols;
  DynamicArray<std::remove_const_t<Ptr>> ptr;
  DynamicArray<std::remove_const_t<CRIdx>> ndIndices;

  template <typename Scalar>
  BlockSparseMatrixView<Scalar, kBlockSize, CRIdx const, Ptr const> View(Scalar* data) {
    return {
        nBlockCols,
        {ptr.data(), static_cast<Ptr>(ptr.size())},
        {ndIndices.data(), static_cast<Ptr>(ndIndices.size())},
        {data, static_cast<Ptr>(kBlockSize * kBlockSize * ptr.back())}};
  }

  template <typename Scalar>
  BlockSparseMatrixView<Scalar const, kBlockSize, CRIdx const, Ptr const> ConstView(
      Scalar const* data) const {
    return {
        nBlockCols,
        {ptr.data(), static_cast<Ptr>(ptr.size())},
        {ndIndices.data(), static_cast<Ptr>(ndIndices.size())},
        {data, static_cast<Ptr>(kBlockSize * kBlockSize * ptr.back())}};
  }
};

/// @brief Extract a BlockViewStructure from a blockable sparsity graph.
/// @tparam kBlockSize Target block size.
/// @tparam CRIdx Column index integer type.
/// @tparam Ptr Pointer integer type.
/// @param numCols[in] Number of columns of the matrix.
/// @param pointers[in] Pointers of the sparsity graph.
/// @param indices[in] Column indices of the sparsity graph.
/// @return BlockViewStructure
template <int kBlockSize, typename CRIdx, typename Ptr>
auto BlockedStructure(CRIdx numCols, Span<Ptr const> pointers, Span<CRIdx const> indices) {
  MOCHI_PROFILE_SCOPE();
  CRIdx const numRows = pointers.empty() ? 0 : pointers.size() - 1;
  MOCHI_ASSERT_VERBOSE(
      (numRows % kBlockSize == 0) && (numCols % kBlockSize == 0),
      "Dimension and block size mismatch.");
  CRIdx const nBlockRows = (numRows / kBlockSize);
  CRIdx const nBlockCols = (numCols / kBlockSize);
  //
  using NonConstPtr = std::remove_const_t<Ptr>;
  using NonConstIdx = std::remove_const_t<CRIdx>;
  if ((numRows == 0) || (numCols == 0)) {
    MOCHI_ASSERT_VERBOSE(indices.empty());
    DynamicArray<NonConstPtr> blockPointers(nBlockRows + 1, 0);
    DynamicArray<NonConstIdx> blockIndices;
    return BlockViewStructure<kBlockSize, NonConstIdx, NonConstPtr>{
        nBlockCols, std::move(blockPointers), std::move(blockIndices)};
  }
  //
  DynamicArray<NonConstPtr> blockPointers;
  blockPointers.reserve(nBlockRows + 1);
  for (int i = 0; i <= nBlockRows; ++i) {
    blockPointers.push_back(pointers[kBlockSize * i] / (kBlockSize * kBlockSize));
  }
  //
  DynamicArray<NonConstIdx> blockIndices;
  blockIndices.reserve(blockPointers[nBlockRows]);
  for (NonConstIdx ii = 0; ii < numRows; ii += kBlockSize) {
    MOCHI_ASSERT_VERBOSE(
        pointers[ii] >= 0 && pointers[ii] <= pointers[ii + 1] && pointers[ii + 1] <= indices.size(),
        "Inconsistent sparsity pattern.")
    Span<CRIdx const, Ptr> rowIndices{
        indices.data() + pointers[ii], pointers[ii + 1] - pointers[ii]};
    MOCHI_ASSERT_VERBOSE(rowIndices.size() % kBlockSize == 0, "Sparsity pattern is not blockable.");
    for (NonConstIdx jj = 0; jj < rowIndices.size(); jj += kBlockSize) {
#if MOCHI_ASSERT_VERBOSE_ENABLED
      MOCHI_ASSERT_VERBOSE(rowIndices[jj] % kBlockSize == 0, "Sparsity pattern is not blockable.");
      for (int kk = 1; kk < kBlockSize; ++kk) {
        MOCHI_ASSERT_VERBOSE(
            rowIndices[jj + kk] == rowIndices[jj] + kk, "Sparsity pattern is not blockable.");
      }
      MOCHI_ASSERT_VERBOSE(
          rowIndices[jj] >= 0 && rowIndices[jj] + kBlockSize <= numCols,
          "Column index out of range.");
#endif // MOCHI_ASSERT_VERBOSE_ENABLED
      blockIndices.push_back(rowIndices[jj] / kBlockSize);
    }
  }
  return BlockViewStructure<kBlockSize, NonConstIdx, NonConstPtr>{
      nBlockCols, std::move(blockPointers), std::move(blockIndices)};
}

/// @brief Extract a BlockViewStructure from a blockable sparse matrix.
/// @tparam kBlockSize Target block size.
/// @tparam Scalar Scalar type of input sparse matrix.
/// @tparam CRIdx Column index integer type of input sparse matrix.
/// @tparam Ptr Pointer integer type of the input sparse matrix.
/// @tparam Storage Storage type of the input sparse matrix.
/// @param A[in] Input sparse matrix.
/// @return BlockViewStructure
template <
    int kBlockSize,
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage = DynamicArray>
auto BlockedStructure(SparseMatrix<Scalar, CRIdx, Ptr, Storage> const& A) {
  return BlockedStructure<kBlockSize>(A.Cols(), A.Pointers(), A.Indices());
}

} // namespace mochi

//
// Implementation details
//

namespace mochi::details {

template <typename Scalar, int kBlockSize>
struct RowMultiplier {
  using NonConstScalar = std::remove_const_t<Scalar>;
  using VType = Simd<NonConstScalar>; // Native SIMD type.

  /// @brief Workspace size to guarantee safe memory writes. Only needed for ApplyToColVector.
  static int GetWorkspaceSize(int nnz) {
    return nnz + VType::kSize; // Enough for the native SIMD type.
  }

  /// @brief Product between a block row and a row-major matrix.
  /// @remarks
  /// - Requires a scalar type with SIMD support.
  /// - Most performant if the number of columns in 'x' and 'Ax' is a multiple of the SIMD vector
  ///   size.
  /// - Most performant if 'Ax' is also stored row-major.
  template <typename Indices, typename Values, typename AccessorIn, typename AccessorOut>
  MOCHI_FORCE_INLINE static void ApplyToRowMajor(
      Indices const& rowIndices,
      Values const& rowValues,
      AccessorIn const& x,
      AccessorOut&& Ax,
      int blkRowIdx,
      int nCols) {
    static_assert(VType::kIsSupported, "The kernel requires a scalar type with SIMD support");
    static_assert(AccessorIn::RowColCosts().second == 1, "The kernel requires row-major x");
    constexpr auto kVecSize = VType::kSize;
    //--- Loop over columns in batches of twice the SIMD size.
    int c = 0;
    for (; c + 2 * kVecSize <= nCols; c += 2 * kVecSize) {
      VType result[2 * kBlockSize] = {}; // Initializes to zero.
      for (int j = 0; j < isize(rowIndices); ++j) {
        for (int jj = 0; jj < kBlockSize; ++jj) {
          auto const xVec1 = x.template RowVector<VType>(jj + kBlockSize * rowIndices[j], c);
          auto const xVec2 =
              x.template RowVector<VType>(jj + kBlockSize * rowIndices[j], c + kVecSize);
          for (int r = 0; r < kBlockSize; ++r) {
            auto const aCoef = Broadcast<VType>(rowValues(r, jj + j * kBlockSize));
            result[r] = MulAdd(aCoef, xVec1, result[r]);
            result[r + kBlockSize] = MulAdd(aCoef, xVec2, result[r + kBlockSize]);
          }
        }
      }
      for (int r = 0; r < kBlockSize; ++r) {
        Ax.StoreRowVector(r + blkRowIdx * kBlockSize, c, result[r]);
        Ax.StoreRowVector(r + blkRowIdx * kBlockSize, c + kVecSize, result[r + kBlockSize]);
      }
    }

    //--- Additional batch of the same number of columns as the SIMD size, if possible.
    if (c + kVecSize <= nCols) {
      VType result[kBlockSize] = {}; // Initializes to zero.
      for (int j = 0; j < isize(rowIndices); ++j) {
        for (int jj = 0; jj < kBlockSize; ++jj) {
          auto const xVec = x.template RowVector<VType>(jj + kBlockSize * rowIndices[j], c);
          for (int r = 0; r < kBlockSize; ++r) {
            VType const aCoef = rowValues(r, jj + j * kBlockSize);
            result[r] = MulAdd(aCoef, xVec, result[r]);
          }
        }
      }
      for (int r = 0; r < kBlockSize; ++r) {
        Ax.StoreRowVector(r + blkRowIdx * kBlockSize, c, result[r]);
      }
      c += kVecSize;
    }

    //--- Leftover columns.
    int const leftoverCols = nCols - c;
    MOCHI_ASSERT_VERBOSE(
        leftoverCols >= 0 && leftoverCols < kVecSize, "Inconsistent number of leftover columns.");
    if (leftoverCols > 0) {
      VType result[kBlockSize] = {}; // Initializes to zero.
      for (int j = 0; j < isize(rowIndices); ++j) {
        for (int jj = 0; jj < kBlockSize; ++jj) {
          auto const xVec =
              x.template RowVector<VType>(jj + kBlockSize * rowIndices[j], c, leftoverCols);
          for (int r = 0; r < kBlockSize; ++r) {
            auto const aCoef = Broadcast<VType>(rowValues(r, jj + j * kBlockSize));
            result[r] = MulAdd(aCoef, xVec, result[r]);
          }
        }
      }
      for (int r = 0; r < kBlockSize; ++r) {
        Ax.StoreRowVector(r + blkRowIdx * kBlockSize, c, result[r], leftoverCols);
      }
    }
  }

  /// @brief Product between a block row and a column vector. The column vector may be part of a
  /// column-major matrix.
  /// @remarks
  /// - 'xTmpRequiredSize' must be greater than or equal to the largest number of non-zeros in a
  ///   row, rounded up to the closest multiple of the SIMD vector size. It is **strongly**
  ///   recommended to call the method `GetWorkspaceSize` to get the required size.
  /// - Requires a scalar type with SIMD support.
  /// - Most performant if 'Ax' is also stored column-major.
  /// - Most performant if 'x' is a column vector itself and NOT part of a column-major matrix with
  ///   multiple columns. If the latter use-case is important, performance could be optimized
  ///   further, e.g. by moving the loop over columns from outside to inside the kernel to improve
  ///   cache efficiency.
  template <
      typename Indices,
      typename Values,
      typename AccessorIn,
      typename AccessorOut,
      typename Idx>
  static void ApplyToColVector(
      Indices const& rowIndices,
      Values const& rowValues,
      int numBlocks,
      int rowLeadDim,
      AccessorIn const& x,
      AccessorOut&& Ax,
      int br,
      int c,
      ColumnVector<NonConstScalar>& xTmp,
      Idx xTmpRequiredSize) {
    static_assert(VType::kIsSupported, "The kernel requires a scalar type with SIMD support");
    static_assert(AccessorIn::RowColCosts().first == 1, "The kernel requires column-major x");

    if constexpr (kBlockSize == 3) {
      static_assert(
          Simd<NonConstScalar, 4>::kIsSupported && Simd<NonConstScalar, 8>::kIsSupported,
          "Multiples of the native SIMD size are expected to be supported");
      if constexpr (!Simd<NonConstScalar, 8>::kIsComposite) {
        // Fastest kernel for block size of 3, except if Simd<T, 8> is composite. If it is,
        // ApplyToColVector3x3Simd4 is in general faster, even if Simd<T, 4> is also composite.
        ApplyToColVector3x3Simd8(rowIndices, rowValues, numBlocks, rowLeadDim, x, Ax, br, c);
      } else {
        // Faster than the fallback kernel for block size of 3, even if Simd<T, 4> is composite.
        ApplyToColVector3x3Simd4(rowIndices, rowValues, numBlocks, rowLeadDim, x, Ax, br, c);
      }
    } else if constexpr (kBlockSize == 4) {
      ApplyToColVector4x4(rowIndices, rowValues, numBlocks, rowLeadDim, x, Ax, br, c);
    } else {
      using VTypeHalf = Simd<NonConstScalar, VType::kSize / 2>; // Half the size of VType.
      constexpr auto kIncr = static_cast<size_t>(VType::kSize);
      auto const len = static_cast<size_t>(kBlockSize * numBlocks);
      MOCHI_ASSERT_VERBOSE(
          xTmpRequiredSize >= mochi::RoundUp(len, kIncr),
          "Insufficient buffer size for SIMD writes.");
      if (xTmp.Rows() < xTmpRequiredSize) {
        xTmp.Resize(xTmpRequiredSize);
      }

      //--- Store the entries of 'x' into contiguous memory space.
      if constexpr (kBlockSize <= VType::kSize) {
        //--- The smallest SIMD size that is at least as large as the block size is most efficient.
        //--- The pointer provided in Store moves by increments of `kBlockSize`. Some values may be
        //--- read and then overwritten.
        //--- A read mask is used for safety. No mask is needed for the write ('xTmp' owns enough
        //--- memory).
        using VTypeBlockCopy = std::conditional_t<
            VTypeHalf::kIsSupported && VTypeHalf::kSize >= kBlockSize,
            VTypeHalf,
            VType>;
        int j = 0;
        // Batches of 2 to hide latency and reduce loop overhead.
        for (; j + 2 <= numBlocks; j += 2) {
          auto v0 =
              x.template ColVector<VTypeBlockCopy, kBlockSize>(kBlockSize * rowIndices[j + 0], c);
          auto v1 =
              x.template ColVector<VTypeBlockCopy, kBlockSize>(kBlockSize * rowIndices[j + 1], c);
          Store(xTmp.Data() + kBlockSize * (j + 0), v0);
          Store(xTmp.Data() + kBlockSize * (j + 1), v1);
        }
        if (j < numBlocks) {
          auto v = x.template ColVector<VTypeBlockCopy, kBlockSize>(kBlockSize * rowIndices[j], c);
          Store(xTmp.Data() + kBlockSize * j, v);
        }
      } else {
        for (int j = 0; j < numBlocks; ++j) {
          int k = 0;
          auto const shift1 = rowIndices[j] * kBlockSize;
          auto const shift2 = j * kBlockSize;
          for (; k + VType::kSize <= kBlockSize; k += VType::kSize) {
            auto v = x.template ColVector<VType>(k + shift1, c);
            Store(xTmp.Data() + k + shift2, v);
          }
          if (k < kBlockSize) {
            auto v = x.template ColVector<VType>(k + shift1, c, kBlockSize - k);
            Store(xTmp.Data() + k + shift2, v);
          }
        }
      }

      //--- Perform the block row multiplication.
      int j = 0;
      VType entries[kBlockSize] = {};
      for (; j + kIncr <= len; j += kIncr) {
        auto const xData = Load<VType>(xTmp.Data() + j);
        for (int k = 0; k < kBlockSize; ++k) {
          auto const matData = Load<VType>(&rowValues[k * rowLeadDim + j]);
          entries[k] = MulAdd(matData, xData, entries[k]);
        }
      }
      if (len - j > 0) {
        auto const xData = Load<VType>(xTmp.Data() + j, len - j);
        for (int k = 0; k < kBlockSize; ++k) {
          auto const matData = Load<VType>(&rowValues[k * rowLeadDim + j], len - j);
          entries[k] = MulAdd(matData, xData, entries[k]);
        }
      }

      //--- Store the result.
      for (int k = 0; k < kBlockSize; ++k) {
        Ax.Store(k + br * kBlockSize, c, HSum(entries[k]));
      }
    }
  }

 private:
  /// @brief Dedicated kernel for the product between a block row with block size of 3 and a column
  /// vector. The column vector may be part of a column-major matrix.
  /// @remarks
  /// - Requires a scalar type with SIMD support of sizes 4 and 8.
  /// - Most performant if 'Ax' is also stored column-major.
  /// - Most performant if 'x' is a column vector itself and NOT part of a column-major matrix with
  ///   multiple columns. If the latter use-case is important, performance could be optimized
  ///   further, e.g. by moving the loop over columns from outside to inside the kernel to improve
  ///   cache efficiency.
  template <typename Indices, typename Values, typename AccessorIn, typename AccessorOut>
  static void ApplyToColVector3x3Simd8(
      Indices const& rowIndices,
      Values const& rowValues,
      int numBlocks,
      int rowLeadDim,
      AccessorIn const& x,
      AccessorOut&& Ax,
      int br,
      int c) {
    // Performance note: Using LoadIndexed to load the blocks in 'x' (reference implementation:
    // P1015747415) is substantially slower on AMD and slightly slower on Intel than the current
    // implementation.
    using V4 = Simd<NonConstScalar, 4>;
    using V8 = Simd<NonConstScalar, 8>;
    static_assert(
        (kBlockSize == 3) && V4::kIsSupported && V8::kIsSupported, "Configuration not supported");
    static_assert(AccessorIn::RowColCosts().first == 1, "The kernel requires column-major x");
    int j = 0;
    int jb = 0;
    V8 results[kBlockSize] = {}; // Initializes to zero.
    V8 matData, vecData;
    V4 v0, v1, v2, v3, v4, v5, v6, v7;
    //--- Batches of 8 blocks.
    for (; jb + 8 <= numBlocks; jb += 8, j += 8 * kBlockSize) {
      v0 = x.template ColVector<V4, kBlockSize>(kBlockSize * rowIndices[jb + 0], c);
      v1 = Shuffle<1, 2, 3, 0>(
          x.template ColVector<V4, kBlockSize>(kBlockSize * rowIndices[jb + 1], c));
      v2 = Shuffle<2, 3, 0, 1>(
          x.template ColVector<V4, kBlockSize>(kBlockSize * rowIndices[jb + 2], c));
      vecData = {Blend<0, 0, 0, 1>(v0, v1), Blend<0, 0, 1, 1>(v1, v2)};
      for (int k = 0; k < kBlockSize; ++k) {
        matData = Load<V8>(&rowValues[k * rowLeadDim + j]);
        results[k] = MulAdd(matData, vecData, results[k]);
      }

      v3 = Shuffle<3, 0, 1, 2>(
          x.template ColVector<V4, kBlockSize>(kBlockSize * rowIndices[jb + 3], c));
      v4 = x.template ColVector<V4, kBlockSize>(kBlockSize * rowIndices[jb + 4], c);
      v5 = Shuffle<1, 2, 3, 0>(
          x.template ColVector<V4, kBlockSize>(kBlockSize * rowIndices[jb + 5], c));
      vecData = {Blend<0, 1, 1, 1>(v2, v3), Blend<0, 0, 0, 1>(v4, v5)};
      for (int k = 0; k < kBlockSize; ++k) {
        matData = Load<V8>(&rowValues[k * rowLeadDim + j + 8]);
        results[k] = MulAdd(matData, vecData, results[k]);
      }

      v6 = Shuffle<2, 3, 0, 1>(
          x.template ColVector<V4, kBlockSize>(kBlockSize * rowIndices[jb + 6], c));
      v7 = Shuffle<3, 0, 1, 2>(
          x.template ColVector<V4, kBlockSize>(kBlockSize * rowIndices[jb + 7], c));
      vecData = {Blend<0, 0, 1, 1>(v5, v6), Blend<0, 1, 1, 1>(v6, v7)};
      for (int k = 0; k < kBlockSize; ++k) {
        matData = Load<V8>(&rowValues[k * rowLeadDim + j + 16]);
        results[k] = MulAdd(matData, vecData, results[k]);
      }
    }
    //--- Additional batches of 2 blocks.
    for (; jb + 2 <= numBlocks; jb += 2, j += 2 * kBlockSize) {
      v0 = x.template ColVector<V4, kBlockSize>(kBlockSize * rowIndices[jb + 0], c);
      v1 = Shuffle<1, 2, 3, 0>(
          x.template ColVector<V4, kBlockSize>(kBlockSize * rowIndices[jb + 1], c));
      vecData = {Blend<0, 0, 0, 1>(v0, v1), v1};
      for (int k = 0; k < kBlockSize; ++k) {
        matData = Load<6, V8>(&rowValues[k * rowLeadDim + j]);
        results[k] = MulAdd(matData, vecData, results[k]);
      }
    }
    //--- Last block.
    if (jb < numBlocks) {
      vecData = x.template ColVector<V8, kBlockSize>(kBlockSize * rowIndices[jb], c);
      for (int k = 0; k < kBlockSize; ++k) {
        matData = Load<3, V8>(&rowValues[k * rowLeadDim + j]);
        results[k] = MulAdd(matData, vecData, results[k]);
      }
    }
    //--- Store result.
    for (int k = 0; k < kBlockSize; ++k) {
      Ax.Store(k + br * kBlockSize, c, HSum(results[k]));
    }
  }

  /// @brief Dedicated kernel for the product between a block row with block size of 3 and a column
  /// vector. The column vector may be part of a column-major matrix.
  /// @remarks
  /// - Requires a scalar type with SIMD support of size 4.
  /// - Most performant if 'Ax' is also stored column-major.
  /// - Most performant if 'x' is a column vector itself and NOT part of a column-major matrix with
  ///   multiple columns. If the latter use-case is important, performance could be optimized
  ///   further, e.g. by moving the loop over columns from outside to inside the kernel to improve
  ///   cache efficiency.
  template <typename Indices, typename Values, typename AccessorIn, typename AccessorOut>
  static void ApplyToColVector3x3Simd4(
      Indices const& rowIndices,
      Values const& rowValues,
      int numBlocks,
      int rowLeadDim,
      AccessorIn const& x,
      AccessorOut&& Ax,
      int br,
      int c) {
    // Performance note: Using LoadIndexed to load the blocks in 'x' (reference implementation:
    // P1105598263) is slightly slower on ARM than the current implementation.
    using V4 = Simd<NonConstScalar, 4>;
    static_assert((kBlockSize == 3) && V4::kIsSupported, "Configuration not supported");
    static_assert(AccessorIn::RowColCosts().first == 1, "The kernel requires column-major x");
    int j = 0;
    int jb = 0;
    V4 v0, v1, v2, v3;
    V4 results[kBlockSize] = {}; // Initializes to zero.
    //--- Batches of 4 blocks.
    for (; jb + 4 <= numBlocks; jb += 4, j += 4 * kBlockSize) {
      v0 = x.template ColVector<V4, kBlockSize>(kBlockSize * rowIndices[jb + 0], c);
      v1 = Shuffle<1, 2, 3, 0>(
          x.template ColVector<V4, kBlockSize>(kBlockSize * rowIndices[jb + 1], c));
      for (int k = 0; k < kBlockSize; ++k) {
        results[k] += Load<V4>(&rowValues[k * rowLeadDim + j + 0]) * Blend<0, 0, 0, 1>(v0, v1);
      }

      v2 = Shuffle<2, 3, 0, 1>(
          x.template ColVector<V4, kBlockSize>(kBlockSize * rowIndices[jb + 2], c));
      for (int k = 0; k < kBlockSize; ++k) {
        results[k] += Load<V4>(&rowValues[k * rowLeadDim + j + 4]) * Blend<0, 0, 1, 1>(v1, v2);
      }

      v3 = Shuffle<3, 0, 1, 2>(
          x.template ColVector<V4, kBlockSize>(kBlockSize * rowIndices[jb + 3], c));
      for (int k = 0; k < kBlockSize; ++k) {
        results[k] += Load<V4>(&rowValues[k * rowLeadDim + j + 8]) * Blend<0, 1, 1, 1>(v2, v3);
      }
    }
    //--- Additional batches of 2 blocks.
    for (; jb + 2 <= numBlocks; jb += 2, j += 2 * kBlockSize) {
      v0 = x.template ColVector<V4, kBlockSize>(kBlockSize * rowIndices[jb + 0], c);
      v1 = Shuffle<1, 2, 3, 0>(
          x.template ColVector<V4, kBlockSize>(kBlockSize * rowIndices[jb + 1], c));
      for (int k = 0; k < kBlockSize; ++k) {
        results[k] += Load<V4>(&rowValues[k * rowLeadDim + j + 0]) * Blend<0, 0, 0, 1>(v0, v1);
      }
      for (int k = 0; k < kBlockSize; ++k) {
        results[k] += Load<2, V4>(&rowValues[k * rowLeadDim + j + 4]) * v1;
      }
    }
    //--- Last block.
    if (jb < numBlocks) {
      v0 = x.template ColVector<V4, kBlockSize>(kBlockSize * rowIndices[jb], c);
      for (int k = 0; k < kBlockSize; ++k) {
        results[k] += Load<kBlockSize, V4>(&rowValues[k * rowLeadDim + j]) * v0;
      }
    }
    //--- Store result.
    for (int k = 0; k < kBlockSize; ++k) {
      Ax.Store(k + br * kBlockSize, c, HSum(results[k]));
    }
  }

  /// @brief Kernel for the product between a block row with block size of 4 and a column vector.
  /// The column vector may be part of a column-major matrix.
  /// @remarks
  /// - Requires a scalar type with SIMD support of sizes 4 and 8.
  /// - Most performant if 'Ax' is also stored column-major.
  /// - Most performant if 'x' is a column vector itself and not part of a column-major matrix with
  ///   multiple columns. If the latter use-case is important, performance could be optimized
  ///   further, e.g. by moving the loop over columns from outside to inside the kernel to improve
  ///   cache efficiency.
  template <typename Indices, typename Values, typename AccessorIn, typename AccessorOut>
  static void ApplyToColVector4x4(
      Indices const& rowIndices,
      Values const& rowValues,
      int numBlocks,
      int rowLeadDim,
      AccessorIn const& x,
      AccessorOut&& Ax,
      int br,
      int c) {
    using V4 = Simd<NonConstScalar, 4>;
    using V8 = Simd<NonConstScalar, 8>;
    static_assert(
        (kBlockSize == 4) && V4::kIsSupported && V8::kIsSupported, "Configuration not supported");
    static_assert(AccessorIn::RowColCosts().first == 1, "ApplyToColVector4x4 requires col-major x");
    int jb = 0;
    int j = 0;
    V4 results[kBlockSize] = {}; // One accumulator per output row

    if constexpr (!V8::kIsComposite) {
      // If V8 is the native size, compute the product in batches of 2 blocks at a time (8 elements
      // = 2 blocks of 4) for better throughput
      V8 results8[kBlockSize] = {};
      for (; jb + 2 <= numBlocks; jb += 2, j += 2 * kBlockSize) {
        V8 vecData = {
            x.template ColVector<V4>(kBlockSize * rowIndices[jb + 0], c),
            x.template ColVector<V4>(kBlockSize * rowIndices[jb + 1], c)};
        for (int k = 0; k < kBlockSize; ++k) {
          V8 matData = Load<V8>(&rowValues[k * rowLeadDim + j]);
          results8[k] += matData * vecData;
        }
      }

      // Reduce V8 results to V4
      for (int k = 0; k < kBlockSize; ++k) {
        results[k] = GetHalf<0>(results8[k]) + GetHalf<1>(results8[k]);
      }
    }

    // Process remaining blocks
    for (; jb < numBlocks; ++jb, j += kBlockSize) {
      V4 vecData = x.template ColVector<V4>(kBlockSize * rowIndices[jb], c);
      for (int k = 0; k < kBlockSize; ++k) {
        V4 matData = Load<V4>(&rowValues[k * rowLeadDim + j]);
        results[k] += matData * vecData;
      }
    }

    // Store result
    for (int k = 0; k < kBlockSize; ++k) {
      Ax.Store(k + br * kBlockSize, c, HSum(results[k]));
    }
  }
};

template <int kBlockSize, typename Dest>
MOCHI_FORCE_INLINE void SetBlockRowToZero(Dest&& dest, int br, int nCols) {
  for (int c = 0; c < nCols; ++c) {
    for (int k = 0; k < kBlockSize; ++k) {
      dest.Store(br * kBlockSize + k, c, 0);
    }
  }
}

} // namespace mochi::details

namespace mochi {

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <typename VectorIn, typename VectorOut>
void BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>::Apply(
    VectorIn const& x,
    VectorOut&& Ax) const {
  static_assert(
      std::is_same_v<Scalar const, typename details::MatTraits<VectorIn>::Scalar const> &&
          std::is_same_v<Scalar const, typename details::MatTraits<VectorOut>::Scalar const>,
      "Inconsistent scalar types");
  MOCHI_ASSERT_VERBOSE(
      (x.Cols() == Ax.Cols()) && (x.Rows() == this->Cols()) && (Ax.Rows() == this->Rows()),
      "Inconsistent dimensions.");
  AccessorApplyToRange(details::GetAccessor(x), details::GetAccessor(Ax), 0, Rows(), x.Cols());
}

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <typename VectorIn, typename VectorOut>
void BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>::ApplyToRange(
    VectorIn const& x,
    VectorOut&& Ax,
    CRIdx rowBegin,
    CRIdx rowEnd) const {
  static_assert(
      std::is_same_v<Scalar const, typename details::MatTraits<VectorIn>::Scalar const> &&
          std::is_same_v<Scalar const, typename details::MatTraits<VectorOut>::Scalar const>,
      "Inconsistent scalar types");
  MOCHI_ASSERT_VERBOSE(
      (x.Cols() == Ax.Cols()) && (x.Rows() == this->Cols()) && (Ax.Rows() == this->Rows()),
      "Inconsistent dimensions.");
  AccessorApplyToRange(
      details::GetAccessor(x), details::GetAccessor(Ax), rowBegin, rowEnd, x.Cols());
}

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <typename AccessorIn, typename AccessorOut>
void BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>::AccessorApplyToRange(
    AccessorIn const& x,
    AccessorOut&& Ax,
    CRIdx rowBegin,
    CRIdx rowEnd,
    int nCols) const {
  // TODO(T158480383): Introduce minimum SIMD size to favor the kernels.
  // TODO(@pabfer): Support row ranges that are not a multiple of the block size.
  static_assert( // Non-SIMD fallback was deprecated in D63392409.
      Simd<NonConstScalar>::kIsSupported,
      "Implementation requires a scalar type with SIMD support");
  using Idx = NonConstIdx;
  MOCHI_ASSERT_VERBOSE(nCols >= 0, "Invalid number of columns.");
  MOCHI_ASSERT_VERBOSE(
      rowBegin >= 0 && rowBegin <= rowEnd && rowEnd <= this->Rows(), "Invalid row range.");
  MOCHI_ASSERT(
      rowBegin % kBlockSize == 0 && rowEnd % kBlockSize == 0,
      "Row ranges must be a multiple of the block size.");
  if (rowBegin == rowEnd) {
    return;
  }

  auto const blkRowBegin = rowBegin / kBlockSize;
  auto const blkRowEnd = rowEnd / kBlockSize;

  // If 'x' and 'Ax' have multiple columns, they count as half since the incremental overhead is
  // smaller.
  constexpr long long kMinFlopsPerTask = 50000; // ~5 μs @ 10 GFLOP/s
  auto const numBlockRows = blkRowEnd - blkRowBegin;
  auto const minBlockRowsPerTask = Clamp<CRIdx>(
      static_cast<CRIdx>(
          kMinFlopsPerTask * numBlockRows /
          Max<Ptr>(
              1,
              2 * kBlockSize * kBlockSize *
                  NumNonZeroBlocksInBlockRowRange(blkRowBegin, blkRowEnd) * ((nCols + 1) / 2))),
      1,
      numBlockRows);
  ParallelForRange(
      "BlockSparseMatrixProduct",
      blkRowBegin,
      blkRowEnd,
      minBlockRowsPerTask,
      numBlockRows,
      [&](CRIdx blkRowBeginTask, CRIdx blkRowEndTask) {
        constexpr auto kCostsX = AccessorIn::RowColCosts();
        if constexpr (kCostsX.first == 1) {
          //--- Col-major X.
          //--- Temporary vector for contiguous memory access to 'x'.
          ColumnVector<NonConstScalar> xTmp;
          auto const xTmpRequiredSize = static_cast<Idx>(
              details::RowMultiplier<NonConstScalar, kBlockSize>::GetWorkspaceSize(_maxNnzPerRow));
          for (int c = 0; c < nCols; ++c) {
            //--- Compute the product looping over block rows. Using moving pointers to indices and
            //--- values instead of creating views for every block row improves performance for some
            //--- compilers and architectures.
            auto const* idxPtr = Indices(blkRowBeginTask).data();
            auto const* vPtr = Values(blkRowBeginTask).data();
            for (Idx br = blkRowBeginTask; br < blkRowEndTask; ++br) {
              auto const numBlocksInRow = static_cast<int>(_ptr[br + 1] - _ptr[br]);
              if (numBlocksInRow > 0) {
                auto const rowLeadDim = kBlockSize * numBlocksInRow;
                details::RowMultiplier<NonConstScalar, kBlockSize>::ApplyToColVector(
                    idxPtr, vPtr, numBlocksInRow, rowLeadDim, x, Ax, br, c, xTmp, xTmpRequiredSize);
                idxPtr += numBlocksInRow;
                vPtr += kBlockSize * rowLeadDim;
              } else {
                MOCHI_ASSERT_VERBOSE(numBlocksInRow == 0, "Number of blocks must not be negative.");
                details::SetBlockRowToZero<kBlockSize>(Ax, br, nCols);
              }
            }
          }
        } else {
          //--- Row-major X.
          static_assert(kCostsX.second == 1, "Unexpected case");
          for (Idx br = blkRowBeginTask; br < blkRowEndTask; ++br) {
            if (!IsBlockRowEmpty(br)) {
              details::RowMultiplier<NonConstScalar, kBlockSize>::ApplyToRowMajor(
                  Indices(br), Values(br), x, Ax, br, nCols);
            } else {
              details::SetBlockRowToZero<kBlockSize>(Ax, br, nCols);
            }
          }
        }
      });
}

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
auto BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>::operator()(CRIdx r, CRIdx c)
    const {
  auto const vIndex = FindValueIndex(r, c);
  return vIndex >= 0 ? NonConstScalar(_v[vIndex]) : NonConstScalar(0);
}

template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
template <typename VectorIn, typename VectorOut>
void BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>::TransposeApply(
    VectorIn const& xv,
    VectorOut&& Axv) const {
  MOCHI_ASSERT_VERBOSE(
      (xv.Cols() == Axv.Cols()) && (xv.Rows() == this->Rows()) && (Axv.Rows() == this->Cols()),
      "Dimensions do not match.");
  using Sx = std::remove_const_t<typename details::MatTraits<VectorIn>::Scalar>;
  using Sy = std::remove_const_t<typename details::MatTraits<VectorOut>::Scalar>;
  MatrixView<
      Sx const,
      details::MatTraits<VectorIn>::kNumRows,
      details::MatTraits<VectorIn>::kNumCols,
      details::MatTraits<VectorIn>::kMajorDir,
      krylov::kDynamic>
      x(xv.data(), xv.Rows(), xv.Cols(), xv.LeadDim());
  MatrixView<
      Sy,
      details::MatTraits<VectorOut>::kNumRows,
      details::MatTraits<VectorOut>::kNumCols,
      details::MatTraits<VectorOut>::kMajorDir,
      krylov::kDynamic>
      y(Axv.data(), Axv.Rows(), Axv.Cols(), Axv.LeadDim());
  y.SetZero();
  if (Axv.Cols() == 1) {
    if constexpr (
        (details::MatTraits<VectorIn>::kMajorDir == krylov::Direction::ColMajor) &&
        (details::MatTraits<VectorOut>::kMajorDir == krylov::Direction::ColMajor)) {
      BlockViewVector<Sx const, kBlockSize, CRIdx> bx{xv.Data(), BlockRows()};
      BlockViewVector<Sy, kBlockSize, CRIdx> bAx{Axv.Data(), BlockCols()};
      for (NonConstIdx i = 0; i < BlockRows(); ++i) {
        auto idxTmp = Indices(i);
        auto vtmp = Values(i);
        for (NonConstIdx j = 0; j < idxTmp.size(); ++j) {
          bAx[idxTmp[j]] += vtmp[j].Transpose() * bx[i];
        }
      }
    } else {
      for (NonConstIdx i = 0; i < BlockRows(); ++i) {
        auto idxTmp = Indices(i);
        auto vtmp = Values(i);
        auto const xi = x.template Block<kBlockSize, 1>(i * kBlockSize, 0, kBlockSize, 1);
        for (NonConstIdx j = 0; j < idxTmp.size(); ++j) {
          auto ylhs = y.template Block<kBlockSize, 1>(idxTmp[j] * kBlockSize, 0, kBlockSize, 1);
          ylhs += vtmp[j].Transpose() * xi;
        }
      }
    }
    return;
  }
  //
  // Case with multiple columns
  //
  for (NonConstIdx i = 0; i < BlockRows(); ++i) {
    auto idxTmp = Indices(i);
    auto vtmp = Values(i);
    auto const xi = x.template Block<kBlockSize>(i * kBlockSize, 0, kBlockSize, y.Cols());
    for (NonConstIdx j = 0; j < idxTmp.size(); ++j) {
      auto ylhs = y.template Block<kBlockSize>(idxTmp[j] * kBlockSize, 0, kBlockSize, y.Cols());
      ylhs += vtmp[j].Transpose() * xi;
    }
  }
}

template <
    int kBlockSize,
    typename Scalar,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
bool IsBlockable(SparseMatrix<Scalar, CRIdx, Ptr, Storage> const& A) {
  MOCHI_PROFILE_SCOPE();
  if ((A.Rows() % kBlockSize != 0) || (A.Cols() % kBlockSize != 0)) {
    return false;
  }
  using RowIdx = std::remove_const_t<CRIdx>;
  for (RowIdx r = 0; r < A.Rows(); r += kBlockSize) {
    auto&& rowIndices = A.Indices(r);
    if (rowIndices.size() % kBlockSize != 0) {
      return false;
    }
    using Idx = std::remove_const_t<decltype(rowIndices.size())>;
    for (Idx j = 0; j < rowIndices.size(); j += kBlockSize) {
      if (rowIndices[j] % kBlockSize != 0) {
        return false;
      }
      for (int k = 1; k < kBlockSize; ++k) {
        if (rowIndices[j + k] != rowIndices[j] + k) {
          return false;
        }
      }
    }
    for (int k = r + 1; k < r + kBlockSize; ++k) {
      auto kRowIndices = A.Indices(k);
      if (kRowIndices.size() != rowIndices.size()) {
        return false;
      }
      if (std::equal(rowIndices.begin(), rowIndices.end(), kRowIndices.begin()) == false) {
        return false;
      }
    }
  }
  return true;
}

template <
    int kBlockSize,
    typename Scalar,
    int kRowsAtCT,
    int kColsAtCT,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDimension>
bool IsBlockable(
    Matrix<Scalar, kRowsAtCT, kColsAtCT, kMajorDirection, kOwnership, kLeadingDimension> const& A) {
  return (A.Rows() % kBlockSize == 0) && (A.Cols() % kBlockSize == 0);
}

// Create a BlockSparseMatrixView pointing to the data owned by a BlockSparseMatrix
template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
auto AsView(BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage>& mat) {
  return BlockSparseMatrixView<Scalar, kBlockSize, CRIdx, Ptr>{
      mat.BlockCols(), Span{mat.Pointers()}, Span{mat.Indices()}, Span{mat.Values()}};
}

// Create a const BlockSparseMatrixView pointing to the data owned by a BlockSparseMatrix
template <
    typename Scalar,
    int kBlockSize,
    typename CRIdx,
    typename Ptr,
    template <typename, typename...> typename Storage>
auto AsConstView(BlockSparseMatrix<Scalar, kBlockSize, CRIdx, Ptr, Storage> const& mat) {
  return BlockSparseMatrixView<Scalar const, kBlockSize, CRIdx const, Ptr const>{
      mat.BlockCols(), Span{mat.Pointers()}, Span{mat.Indices()}, Span{mat.Values()}};
}

} // namespace mochi
