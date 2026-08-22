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
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>
#include <mochi_core/memory/allocator.h>
#include <mochi_core/utils/rand_utils.h>
#include <mochi_core/utils/simd.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace mochi {

/// @brief Matrix class
///
/// @tparam Scalar
/// @tparam kRowsAtCompileTime
/// @tparam kColsAtCompileTime
/// @tparam kMajorDirection
/// @tparam kOwnership
/// @tparam kLeadingDim
///
template <
    typename Scalar,
    int kRowsAtCompileTime = krylov::kDynamic,
    int kColsAtCompileTime = krylov::kDynamic,
    krylov::Direction kMajorDirection = krylov::Direction::ColMajor,
    krylov::Ownership kOwnership = krylov::Ownership::Owner,
    int kLeadingDim = krylov::kAutomaticLeadDim>
class Matrix;

} // namespace mochi

namespace mochi::krylov::details {
template <
    typename Scalar_,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDim>
struct MatTraitsDef<Matrix<
    Scalar_,
    kRowsAtCompileTime,
    kColsAtCompileTime,
    kMajorDirection,
    kOwnership,
    kLeadingDim>> {
  using Scalar = Scalar_;
  static constexpr int kNumRows = kRowsAtCompileTime;
  static constexpr int kNumCols = kColsAtCompileTime;
  static constexpr Direction kMajorDir = kMajorDirection;
  static constexpr Ownership kOwner = kOwnership;
  static constexpr int kLeadDim = kLeadingDim;
  static constexpr bool kIsCuda =
      kOwnership == Ownership::Cuda || kOwnership == Ownership::CudaView;
};
} // namespace mochi::krylov::details

namespace mochi::krylov {

/// @brief Flag for indicating that the template parameters are kept.
constexpr int kKeep = -3;

} // namespace mochi::krylov

namespace mochi {
namespace {
using krylov::details::MatTraits;

/// @brief Types that are the same with const.
template <typename T, typename G>
concept is_same_const_v = std::is_same_v<T const, G const>;
} // namespace

template <
    typename Scalar_,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDim>
class MOCHI_EMPTY_BASE Matrix : public krylov::BaseMatrix<
                                    Scalar_,
                                    kRowsAtCompileTime,
                                    kColsAtCompileTime,
                                    kMajorDirection,
                                    kOwnership,
                                    kLeadingDim>,
                                public NewMatrix {
  using BaseMat = krylov::BaseMatrix<
      Scalar_,
      kRowsAtCompileTime,
      kColsAtCompileTime,
      kMajorDirection,
      kOwnership,
      kLeadingDim>;

 public:
  using Scalar = Scalar_;
  using BaseMat::kIsColMajor;
  using BaseMat::kIsRowMajor;

  using BaseMat::kIsCuda;
  using BaseMat::kIsDenseHostVector;
  using BaseMat::kIsDenseVector;
  using BaseMat::kIsDynamic;
  using BaseMat::kIsHostVector;
  using BaseMat::kIsOwner;
  using BaseMat::kIsVector;
  using BaseMat::kIsView;

  using NonConstScalar = std::remove_const_t<Scalar>;

  /// @brief Default constructor when the matrix has compile-time dimensions.
  constexpr Matrix() : BaseMat{} {}

  /// @brief Constructor when kRowsAtCompileTime (or kColsAtCompileTime) is 1.
  ///
  /// @param[in] len Number of entries
  explicit Matrix(int len)
    requires kIsDenseVector
      : BaseMat{(kRowsAtCompileTime == 1) ? 1 : len, (kColsAtCompileTime == 1) ? 1 : len} {
    if constexpr ((kRowsAtCompileTime == 1) && (kColsAtCompileTime == 1)) {
      MOCHI_ASSERT_VERBOSE(len == 1, "Inconsistent size");
    }
  }

  /// @brief Constructor when kRowsAtCompileTime (or kColsAtCompileTime) is 1.
  ///
  /// @param[in] len Number of entries
  /// @param[in] allocator Pointer for the custom memory resource
  /// When allocator is nullptr, the default new/delete combination is used.
  Matrix(int len, Allocator* allocator)
    requires kIsDenseVector
      : BaseMat{
            (kRowsAtCompileTime == 1) ? 1 : len,
            (kColsAtCompileTime == 1) ? 1 : len,
            allocator} {
    if constexpr ((kRowsAtCompileTime == 1) && (kColsAtCompileTime == 1)) {
      MOCHI_ASSERT_VERBOSE(len == 1, "Inconsistent size");
    }
  }

  /// @brief Constructor
  /// @param[in] rows_ Number of rows of the matrix
  /// @param[in] cols_ Number of columns of the matrix
  Matrix(int rows_, int cols_) : BaseMat{rows_, cols_} {}

  /// @brief Constructor
  /// @param[in] rows Number of rows of the matrix
  /// @param[in] cols Number of columns of the matrix
  /// @param[in] allocator Pointer for the custom memory resource
  /// When allocator is nullptr, the default new/delete combination is used.
  Matrix(int rows, int cols, Allocator* allocator) : BaseMat{rows, cols, allocator} {}

  /// @brief Constructor
  /// @param[in] rows_ Number of rows of the matrix
  /// @param[in] cols_ Number of columns of the matrix
  /// @param[in] ld_   Leading dimension
  Matrix(int rows_, int cols_, int ld_) : BaseMat{rows_, cols_, ld_} {}

  /// @brief Constructor
  /// @param[in] rows Number of rows of the matrix
  /// @param[in] cols Number of columns of the matrix
  /// @param[in] ld   Leading dimension
  /// @param[in] allocator Pointer for the custom memory resource
  /// When allocator is nullptr, the default new/delete combination is used.
  Matrix(int rows, int cols, int ld, Allocator* allocator) : BaseMat{rows, cols, ld, allocator} {}

  /// @brief Constructor for list initialization for fixed-size matrix
  /// @note The flat list has to be consistent with the major direction.
  /// @note The function checks at compile time that the list has the correct size.
  /// (i.e. kRowsAtCompileTime * kColsAtCompileTime)
  constexpr Matrix(Scalar v0, std::same_as<Scalar> auto... v) : BaseMat{v0, v...} {}

  /// @brief Constructor for list initialization for fixed-size matrix.
  /// @note The function checks at runtime that the list has the correct size.
  /// (i.e. kRowsAtCompileTime * kColsAtCompileTime)
  constexpr Matrix(std::initializer_list<std::initializer_list<Scalar>> const& init)
      : BaseMat(init) {}

  // Attempting to construct a Matrix from nullptr is generally a mistake.
  Matrix(std::nullptr_t) = delete;

  explicit Matrix(Scalar* s) : BaseMat{s} {}

  Matrix(Scalar* s, int len_)
    requires(kIsVector)
      : BaseMat{s, (kRowsAtCompileTime == 1) ? 1 : len_, (kColsAtCompileTime == 1) ? 1 : len_} {
    if constexpr ((kRowsAtCompileTime == 1) && (kColsAtCompileTime == 1)) {
      MOCHI_ASSERT_VERBOSE(len_ == 1, "Inconsistent size");
    }
  }

  explicit Matrix(Span<Scalar> valueSpan) : Matrix(valueSpan.data(), valueSpan.size()) {}

  Matrix(Scalar* s, int rows_, int cols_) : BaseMat{s, rows_, cols_} {}

  Matrix(Scalar* s, int rows, int cols, int ld) : BaseMat{s, rows, cols, ld} {}

  /// @brief Copy constructor.
  ///
  /// @param[in] A Matrix with the same characteristics.
  ///
  constexpr Matrix(Matrix const& A) : BaseMat(A) {}

  /// @brief Copy constructor.
  ///
  /// @param[in] A Matrix with the same characteristics.
  ///
  template <
      is_same_const_v<Scalar> OtherScalar,
      int kOtherRowsAtCompileTime,
      int kOtherColsAtCompileTime,
      krylov::Direction kOtherMajorDirection,
      krylov::Ownership kOtherOwnership,
      int kOtherLeadDim>
    requires kIsOwner
  Matrix(
      Matrix<
          OtherScalar,
          kOtherRowsAtCompileTime,
          kOtherColsAtCompileTime,
          kOtherMajorDirection,
          kOtherOwnership,
          kOtherLeadDim> const& A)
      : BaseMat(
            A.Rows(),
            A.Cols(),
            (kLeadingDim > 0) ? kLeadingDim : (BaseMat::kIsColMajor ? A.Rows() : A.Cols())) {
    static_assert(
        std::is_same_v<NonConstScalar, std::remove_const_t<OtherScalar>>,
        "Inconsistent scalar types");
    if constexpr ((kRowsAtCompileTime >= 0) && (kOtherRowsAtCompileTime >= 0)) {
      static_assert(kRowsAtCompileTime == kOtherRowsAtCompileTime, "Inconsistent number of rows");
    } else {
      MOCHI_ASSERT(this->Rows() == A.Rows(), "Inconsistent number of rows");
    }
    if constexpr ((kColsAtCompileTime >= 0) && (kOtherColsAtCompileTime >= 0)) {
      static_assert(
          kColsAtCompileTime == kOtherColsAtCompileTime, "Inconsistent number of columns");
    } else {
      MOCHI_ASSERT(this->Cols() == A.Cols(), "Inconsistent number of columns");
    }
    this->operator=(A);
  }

  /// @brief Move constructor.
  Matrix(Matrix&& A) noexcept : BaseMat(static_cast<BaseMat&&>(A)) {}

  /// @brief Move constructor with allocator. Memory will be moved if allocators are compatible.
  Matrix(Matrix&& A, Allocator* allocator) : BaseMat(static_cast<BaseMat&&>(A), allocator) {}

  template <
      typename OtherScalar,
      int kOtherRowsAtCompileTime,
      int kOtherColsAtCompileTime,
      krylov::Ownership kOtherOwnership,
      int kOtherLeadDim>
    requires(
        kOwnership == krylov::Viewed<kOtherOwnership> &&
        (!std::is_same_v<Scalar, OtherScalar> || (kOtherRowsAtCompileTime != kRowsAtCompileTime) ||
         (kOtherColsAtCompileTime != kColsAtCompileTime) || (kOtherOwnership != kOwnership) ||
         (kLeadingDim != kOtherLeadDim)) && // Not a pure copy constructor
        (std::is_same_v<Scalar, OtherScalar> || std::is_same_v<Scalar, OtherScalar const>) &&
        ((kLeadingDim == kOtherLeadDim) || (kLeadingDim == krylov::kDynamic)) &&
        ((kRowsAtCompileTime == kOtherRowsAtCompileTime) ||
         (kRowsAtCompileTime == krylov::kDynamic)) &&
        ((kColsAtCompileTime == kOtherColsAtCompileTime) ||
         (kColsAtCompileTime == krylov::kDynamic)))
  Matrix(
      Matrix<
          OtherScalar,
          kOtherRowsAtCompileTime,
          kOtherColsAtCompileTime,
          kMajorDirection,
          kOtherOwnership,
          kOtherLeadDim> const& other)
      : Matrix(
            const_cast<NonConstScalar*>(other.Data()),
            other.Rows(),
            other.Cols(),
            other.LeadDim()) {}

  Matrix& operator=(Matrix&& rhs) noexcept {
    if (std::addressof(rhs) != this) {
      if constexpr (kIsOwner && kIsDynamic && !kIsCuda) {
        auto* myAlloc = this->GetAllocator();
        auto* rhsAlloc = rhs.GetAllocator();
        MOCHI_ASSERT_VERBOSE(
            myAlloc && rhsAlloc, "Dynamic matrices should always have an allocator");
        if ((myAlloc == rhsAlloc) || (myAlloc->is_equal(*rhsAlloc))) {
          // These two matrices have the same allocator, or the allocators are compatible ("equal").
          // Therefore, we can move the pointer while keeping our original allocator.
          this->~Matrix(); // Deallocate previous memory
          new (this)
              Matrix(std::move(rhs), myAlloc); // Move pointers from rhs, but keep my allocator
        } else {
          // We do not allow assignment to change our allocator and our allocator is not compatible
          // with theirs. Therefore, we cannot move ownership of the memory.
          this->Resize(rhs.Rows(), rhs.Cols()); // Noop if the dimensions are the same.
          *this = static_cast<Matrix const&>(rhs); // Copy values
        }
      } else if constexpr (kIsOwner && kIsDynamic && kIsCuda) {
        // Cuda matrices don't have custom allocators. We can always move the pointer.
        this->~Matrix();
        new (this) Matrix(std::move(rhs));
      } else {
        // Other cases cannot move the pointer and are not allowed to change the dimensions.
        // Simply copy the values.
        *this = static_cast<Matrix const&>(rhs);
      }
    }
    return *this;
  }

  /// @brief Copy assignment.
  ///
  /// @param[in] rhs Right-hand side with the same characteristics.
  ///
  /// @note Resizing is not allowed.
  constexpr Matrix& operator=(Matrix const& rhs) {
    MOCHI_ASSERT_VERBOSE(
        (this->Rows() == rhs.Rows()) && (this->Cols() == rhs.Cols()), "Dimensions do not match");
    if constexpr (krylov::IsCuda(kOwnership)) {
      this->operator= <Matrix>(rhs);
    } else if constexpr (kIsRowMajor) {
      if ((this->LeadDim() == this->Cols()) && (rhs.LeadDim() == rhs.Cols())) {
        std::copy(rhs.Data(), rhs.Data() + this->StorageSize(), this->Data());
      } else {
        for (int ir = 0; ir < this->Rows(); ++ir) {
          std::copy(&rhs(ir, 0), &rhs(ir, 0) + this->Cols(), &(*this)(ir, 0));
        }
      }
    } else {
      if ((this->LeadDim() == this->Rows()) && (rhs.LeadDim() == rhs.Rows())) {
        std::copy(rhs.Data(), rhs.Data() + this->StorageSize(), this->Data());
      } else {
        for (int ic = 0; ic < this->Cols(); ++ic) {
          std::copy(&rhs(0, ic), &rhs(0, ic) + this->Rows(), &(*this)(0, ic));
        }
      }
    }
    return *this;
  }

  /// @brief Construct a matrix with all entries initialized to zero.
  [[nodiscard]] static auto Zero(int rows, int cols) {
    static_assert(kIsOwner, "Zero is only supported for owning matrices");
    Matrix mat(rows, cols);
    mat.SetZero();
    return mat;
  }

  /// @brief Construct a row or column vector with all entries initialized to zero.
  [[nodiscard]] static auto Zero(int n) {
    static_assert(
        kIsOwner && (kRowsAtCompileTime == 1 || kColsAtCompileTime == 1),
        "Specialization only supported for owning vectors");
    Matrix mat(n);
    mat.SetZero();
    return mat;
  }

  /// @brief Construct a fixed-size matrix with all entries initialized to zero.
  [[nodiscard]] static auto Zero() {
    static_assert(
        kIsOwner && kRowsAtCompileTime >= 0 && kColsAtCompileTime >= 0,
        "Specialization only supported for owning fixed-size matrices");
    return Matrix::Zero(kRowsAtCompileTime, kColsAtCompileTime);
  }

  /// @brief Construct a matrix with all entries initialized to one.
  [[nodiscard]] static auto Ones(int rows, int cols) {
    static_assert(kIsOwner, "Ones is only supported for owning matrices");
    Matrix mat(rows, cols);
    mat.SetConstant(1);
    return mat;
  }

  /// @brief Construct a row or column vector with all entries initialized to one.
  [[nodiscard]] static auto Ones(int n) {
    static_assert(
        kIsOwner && (kRowsAtCompileTime == 1 || kColsAtCompileTime == 1),
        "Specialization only supported for owning vectors");
    Matrix mat(n);
    mat.SetConstant(1);
    return mat;
  }

  /// @brief Construct a fixed-size matrix with all entries initialized to one.
  [[nodiscard]] static auto Ones() {
    static_assert(
        kIsOwner && kRowsAtCompileTime >= 0 && kColsAtCompileTime >= 0,
        "Specialization only supported for owning fixed-size matrices");
    return Matrix::Ones(kRowsAtCompileTime, kColsAtCompileTime);
  }

  /// @brief Construct a fixed-size matrix with all entries initialized to random values.
  static auto Random(
      unsigned int seed = mochi::GetRandomSeed(),
      Scalar sMin = Scalar(0),
      Scalar sMax = Scalar(1)) {
    static_assert(
        kIsOwner && kRowsAtCompileTime >= 0 && kColsAtCompileTime >= 0,
        "Random constructor is only supported for owning fixed-size matrices");
    Matrix mat;
    mat.SetRandom(seed, sMin, sMax);
    return mat;
  }

  // Reset this Matrix using the arguments for any of its constructors.
  template <typename... Args>
  Matrix& Reset(Args&&... args) {
    this->~Matrix();
    new (this) Matrix(std::forward<Args>(args)...);
    return *this;
  }

  /// @brief True if the matrix is not empty (i.e. Rows() > 0 && Cols() > 0).
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE explicit operator bool() const {
    return !BaseMat::empty();
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE Span<Scalar const> GetConstSpan() const
    requires(kIsDenseVector)
  {
    return {
        this->Data(),
        static_cast<size_t>(
            kMajorDirection == krylov::Direction::ColMajor ? this->Rows() : this->Cols())};
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE Span<Scalar> GetSpan()
    requires(kIsDenseVector)
  {
    return {
        this->Data(),
        static_cast<size_t>(
            kMajorDirection == krylov::Direction::ColMajor ? this->Rows() : this->Cols())};
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE Span<Scalar const> GetSpan() const
    requires(kIsDenseVector)
  {
    return GetConstSpan();
  }

  // Construct from a matrix expression
  Matrix(IsMatrixExpr auto const& rhs) : Matrix(rhs.CERows().iVal(), rhs.CECols().iVal()) {
    this->operator=(rhs);
  }

  // Construct from a matrix or matrix expression, using a custom allocator (cannot be nullptr)
  Matrix(IsMatrixLike auto const& rhs, Allocator* allocator)
      : Matrix(rhs.CERows().iVal(), rhs.CECols().iVal(), allocator) {
    this->operator=(rhs);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE static constexpr bool IsResizable() {
    if constexpr (kRowsAtCompileTime > 0 && kColsAtCompileTime > 0) {
      return false;
    }
    if (krylov::IsView(kOwnership)) {
      return false;
    }
    return true;
  }

  template <IsMatrixLike RHS>
  Matrix& operator=(RHS const& rhs);

  template <IsMatrixLike RHS>
  Matrix& operator+=(RHS const& rhs);

  template <IsMatrixLike RHS>
  Matrix& operator-=(RHS const& rhs);

  Matrix& operator*=(Scalar alpha);

  Matrix& operator/=(Scalar alpha);

  /// @brief Extract the block starting at (ir, jc) with p rows and q columns.
  ///
  /// @param ir Starting row index
  /// @param jc Starting column index
  /// @param numRows  Number of rows to extract
  /// @param numCols  Number of columns to extract
  ///
  /// @return Matrix view of the extracted block
  template <int kRowsBlockAtCompile = krylov::kDynamic, int kColsBlockAtCompile = krylov::kDynamic>
  [[nodiscard]] auto Block(int ir, int jc, int numRows, int numCols) {
    MOCHI_ASSERT_VERBOSE(numRows >= 0, "Inappropriate number of rows");
    MOCHI_ASSERT_VERBOSE((ir >= 0) && (ir + numRows <= this->Rows()), "Out-of-range row bounds");
    if constexpr (kRowsBlockAtCompile > 0) {
      MOCHI_ASSERT_VERBOSE(numRows == kRowsBlockAtCompile, "Inconsistent number of rows");
    }
    MOCHI_ASSERT_VERBOSE(numCols >= 0, "Inappropriate number of columns");
    MOCHI_ASSERT_VERBOSE((jc >= 0) && (jc + numCols <= this->Cols()), "Out-of-range column bounds");
    if constexpr (kColsBlockAtCompile > 0) {
      MOCHI_ASSERT_VERBOSE(numCols == kColsBlockAtCompile, "Inconsistent number of columns");
    }
    Scalar* d = this->Data();
    constexpr int kMyLeadDim = (kLeadingDim > 0) ? kLeadingDim : krylov::kDynamic;
    return Matrix<
        Scalar,
        kRowsBlockAtCompile,
        kColsBlockAtCompile,
        kMajorDirection,
        krylov::Viewed<kOwnership>,
        kMyLeadDim>{d + this->GetOffset(ir, jc), numRows, numCols, this->LeadDim()};
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
  [[nodiscard]] auto Block(int ir, int jc, int numRows, int numCols) const {
    MOCHI_ASSERT_VERBOSE(numRows >= 0, "Inappropriate number of rows");
    MOCHI_ASSERT_VERBOSE((ir >= 0) && (ir + numRows <= this->Rows()), "Out-of-range row bounds");
    if constexpr (kRowsBlockAtCompile > 0) {
      MOCHI_ASSERT_VERBOSE(numRows == kRowsBlockAtCompile, "Inconsistent number of rows");
    }
    MOCHI_ASSERT_VERBOSE(numCols >= 0, "Inappropriate number of columns");
    MOCHI_ASSERT_VERBOSE((jc >= 0) && (jc + numCols <= this->Cols()), "Out-of-range column bounds");
    if constexpr (kColsBlockAtCompile > 0) {
      MOCHI_ASSERT_VERBOSE(numCols == kColsBlockAtCompile, "Inconsistent number of columns");
    }
    Scalar const* d = this->Data();
    constexpr int kMyLeadDim = (kLeadingDim > 0) ? kLeadingDim : krylov::kDynamic;
    return Matrix<
        Scalar const,
        kRowsBlockAtCompile,
        kColsBlockAtCompile,
        kMajorDirection,
        krylov::Viewed<kOwnership>,
        kMyLeadDim>{d + this->GetOffset(ir, jc), numRows, numCols, this->LeadDim()};
  }

  template <int kRowsBlockAtCompile = krylov::kDynamic>
  [[nodiscard]] auto MiddleRows(int start, int size) {
    Scalar* d = this->Data();
    constexpr bool kIsColVector =
        kColsAtCompileTime == 1 && kMajorDirection == krylov::Direction::ColMajor;
    if constexpr (
        kIsColVector || (BaseMat::kIsRowMajor && kLeadingDim == krylov::kAutomaticLeadDim)) {
      return Matrix<
          Scalar,
          kRowsBlockAtCompile,
          kColsAtCompileTime,
          kMajorDirection,
          krylov::Viewed<kOwnership>,
          krylov::kAutomaticLeadDim>(d + this->GetOffset(start, 0), size, this->Cols());
    } else {
      return Matrix<
          Scalar,
          kRowsBlockAtCompile,
          kColsAtCompileTime,
          kMajorDirection,
          krylov::Viewed<kOwnership>,
          krylov::kDynamic>(d + this->GetOffset(start, 0), size, this->Cols(), this->LeadDim());
    }
  }

  template <int kRowsBlockAtCompile = krylov::kDynamic>
  [[nodiscard]] auto MiddleRows(int start, int size) const {
    Scalar const* d = this->Data();
    constexpr bool kIsColVector =
        kColsAtCompileTime == 1 && kMajorDirection == krylov::Direction::ColMajor;
    if constexpr (
        kIsColVector || (BaseMat::kIsRowMajor && kLeadingDim == krylov::kAutomaticLeadDim)) {
      return Matrix<
          Scalar const,
          kRowsBlockAtCompile,
          kColsAtCompileTime,
          kMajorDirection,
          krylov::Viewed<kOwnership>,
          krylov::kAutomaticLeadDim>(d + this->GetOffset(start, 0), size, this->Cols());
    } else {
      return Matrix<
          Scalar const,
          kRowsBlockAtCompile,
          kColsAtCompileTime,
          kMajorDirection,
          krylov::Viewed<kOwnership>,
          krylov::kDynamic>(d + this->GetOffset(start, 0), size, this->Cols(), this->LeadDim());
    }
  }

  template <int kColsBlockAtCompile = krylov::kDynamic>
  [[nodiscard]] auto MiddleCols(int start, int size) {
    Scalar* d = this->Data();
    constexpr bool kIsRowVector =
        kRowsAtCompileTime == 1 && kMajorDirection == krylov::Direction::RowMajor;
    if constexpr (
        kIsRowVector || (BaseMat::kIsColMajor && kLeadingDim == krylov::kAutomaticLeadDim)) {
      return Matrix<
          Scalar,
          kRowsAtCompileTime,
          kColsBlockAtCompile,
          kMajorDirection,
          krylov::Viewed<kOwnership>,
          krylov::kAutomaticLeadDim>(d + this->GetOffset(0, start), this->Rows(), size);
    } else {
      return Matrix<
          Scalar,
          kRowsAtCompileTime,
          kColsBlockAtCompile,
          kMajorDirection,
          krylov::Viewed<kOwnership>,
          krylov::kDynamic>(d + this->GetOffset(0, start), this->Rows(), size, this->LeadDim());
    }
  }

  template <int kColsBlockAtCompile = krylov::kDynamic>
  [[nodiscard]] auto MiddleCols(int start, int size) const {
    Scalar const* d = this->Data();
    constexpr bool kIsRowVector =
        kRowsAtCompileTime == 1 && kMajorDirection == krylov::Direction::RowMajor;
    if constexpr (
        kIsRowVector || (BaseMat::kIsColMajor && kLeadingDim == krylov::kAutomaticLeadDim)) {
      return Matrix<
          Scalar const,
          kRowsAtCompileTime,
          kColsBlockAtCompile,
          kMajorDirection,
          krylov::Viewed<kOwnership>,
          krylov::kAutomaticLeadDim>(d + this->GetOffset(0, start), this->Rows(), size);
    } else {
      return Matrix<
          Scalar const,
          kRowsAtCompileTime,
          kColsBlockAtCompile,
          kMajorDirection,
          krylov::Viewed<kOwnership>,
          krylov::kDynamic>(d + this->GetOffset(0, start), this->Rows(), size, this->LeadDim());
    }
  }

  /// @brief Extract a single row from the matrix.
  ///
  /// @param r Index of the row to extract
  /// @return Vector-view of extracted row
  [[nodiscard]] auto Row(int r) {
    MOCHI_ASSERT_VERBOSE((r >= 0) && (r < this->Rows()), "Out-of-range row index");
    auto* d = this->Data() + this->GetOffset(r, 0);
    // Row-major storage inherits the leading dimension
    // Col-major becomes dynamic, as the number of rows of the result is
    // not the same as the starting matrix.
    constexpr int kRowLeadDim = (BaseMat::kIsRowMajor) ? kLeadingDim : krylov::kDynamic;
    return Matrix<
        Scalar,
        1,
        kColsAtCompileTime,
        kMajorDirection,
        krylov::Viewed<kOwnership>,
        kRowLeadDim>{d, 1, this->Cols(), this->LeadDim()};
  }

  /// @brief Extract a single row from the matrix.
  ///
  /// @param r Index of the row to extract
  /// @return Vector-view of extracted row
  [[nodiscard]] auto Row(int r) const {
    MOCHI_ASSERT_VERBOSE((r >= 0) && (r < this->Rows()), "Out-of-range row index");
    auto* d = this->Data() + this->GetOffset(r, 0);
    // Row-major storage inherits the leading dimension
    // Col-major becomes dynamic, as the number of rows of the result is
    // not the same as the starting matrix.
    constexpr int kRowLeadDim = (BaseMat::kIsRowMajor) ? kLeadingDim : krylov::kDynamic;
    return Matrix<
        Scalar const,
        1,
        kColsAtCompileTime,
        kMajorDirection,
        krylov::Viewed<kOwnership>,
        kRowLeadDim>{d, 1, this->Cols(), this->LeadDim()};
  }

  /// @brief Extract a single column from the matrix.
  ///
  /// @param c Index of the column to extract
  /// @return Vector-view of extracted column
  [[nodiscard]] auto Col(int c) {
    MOCHI_ASSERT_VERBOSE((c >= 0) && (c < this->Cols()), "Out-of-range column index");
    auto* d = this->Data() + this->GetOffset(0, c);
    // Col-major storage inherits the leading dimension
    // Row-major becomes dynamic, as the number of column of the result is
    // not the same as the starting matrix.
    constexpr int kColLeadDim = (BaseMat::kIsColMajor) ? kLeadingDim : krylov::kDynamic;
    return Matrix<
        Scalar,
        kRowsAtCompileTime,
        1,
        kMajorDirection,
        krylov::Viewed<kOwnership>,
        kColLeadDim>{d, this->Rows(), 1, this->LeadDim()};
  }

  /// @brief Extract a single column from the matrix.
  ///
  /// @param c Index of the column to extract
  /// @return Vector-view of extracted column
  [[nodiscard]] auto Col(int c) const {
    MOCHI_ASSERT_VERBOSE((c >= 0) && (c < this->Cols()), "Out-of-range column index");
    auto* d = this->Data() + this->GetOffset(0, c);
    // Col-major storage inherits the leading dimension
    // Row-major becomes dynamic, as the number of column of the result is
    // not the same as the starting matrix.
    constexpr int kColLeadDim = (BaseMat::kIsColMajor) ? kLeadingDim : krylov::kDynamic;
    return Matrix<
        Scalar const,
        kRowsAtCompileTime,
        1,
        kMajorDirection,
        krylov::Viewed<kOwnership>,
        kColLeadDim>{d, this->Rows(), 1, this->LeadDim()};
  }

  /// @brief Extract the leftmost columns from the matrix
  ///
  /// @param numCols Number of columns to extract
  /// @return Matrix view of the extracted columns
  template <int kNumCols = krylov::kDynamic>
  [[nodiscard]] auto LeftCols(int numCols) {
    MOCHI_ASSERT_VERBOSE(
        (numCols >= 0) && (numCols <= this->Cols()), "Out-of-range number of columns");
    if constexpr (kNumCols >= 0) {
      MOCHI_ASSERT_VERBOSE(numCols == kNumCols, "Inconsistent number of columns");
    }
    // Col-major storage inherits the leading dimension
    // Row-major becomes dynamic, as the number of column of the result is
    // not the same as the starting matrix.
    constexpr int kColLeadDim = (BaseMat::kIsColMajor) ? kLeadingDim : krylov::kDynamic;
    return Matrix<
        Scalar,
        kRowsAtCompileTime,
        kNumCols,
        kMajorDirection,
        krylov::Viewed<kOwnership>,
        kColLeadDim>{this->Data(), this->Rows(), numCols, this->LeadDim()};
  }

  /// @brief Extract the leftmost columns from the matrix
  ///
  /// @param numCols Number of columns to extract
  /// @return Matrix view of the extracted columns
  template <int kNumCols = krylov::kDynamic>
  [[nodiscard]] auto LeftCols(int numCols) const {
    MOCHI_ASSERT_VERBOSE(
        (numCols >= 0) && (numCols <= this->Cols()), "Out-of-range number of columns");
    if constexpr (kNumCols >= 0) {
      MOCHI_ASSERT_VERBOSE(numCols == kNumCols, "Inconsistent number of columns");
    }
    // Col-major storage inherits the leading dimension
    // Row-major becomes dynamic, as the number of column of the result is
    // not the same as the starting matrix.
    constexpr int kColLeadDim = (BaseMat::kIsColMajor) ? kLeadingDim : krylov::kDynamic;
    return Matrix<
        Scalar const,
        kRowsAtCompileTime,
        kNumCols,
        kMajorDirection,
        krylov::Viewed<kOwnership>,
        kColLeadDim>{this->data(), this->Rows(), numCols, this->LeadDim()};
  }

  /// @brief Extract the rightmost columns from the matrix
  ///
  /// @param numCols Number of columns to extract
  /// @return Matrix view of the extracted columns
  template <int kNumCols = krylov::kDynamic>
  [[nodiscard]] auto RightCols(int numCols) {
    MOCHI_ASSERT_VERBOSE(
        (numCols >= 0) && (numCols <= this->Cols()), "Out-of-range number of columns");
    if constexpr (kNumCols >= 0) {
      MOCHI_ASSERT_VERBOSE(numCols == kNumCols, "Inconsistent number of columns");
    }
    Scalar* d = this->Data();
    // Col-major storage inherits the leading dimension
    // Row-major becomes dynamic, as the number of column of the result is
    // not the same as the starting matrix.
    constexpr int kColLeadDim = (BaseMat::kIsColMajor) ? kLeadingDim : krylov::kDynamic;
    return Matrix<
        Scalar,
        kRowsAtCompileTime,
        kNumCols,
        kMajorDirection,
        krylov::Viewed<kOwnership>,
        kColLeadDim>{
        d + this->GetOffset(0, this->Cols() - numCols), this->Rows(), numCols, this->LeadDim()};
  }

  /// @brief Extract the rightmost columns from the matrix
  ///
  /// @param numCols Number of columns to extract
  /// @return Matrix view of the extracted columns
  template <int kNumCols = krylov::kDynamic>
  [[nodiscard]] auto RightCols(int numCols) const {
    MOCHI_ASSERT_VERBOSE(
        (numCols >= 0) && (numCols <= this->Cols()), "Out-of-range number of columns");
    if constexpr (kNumCols >= 0) {
      MOCHI_ASSERT_VERBOSE(numCols == kNumCols, "Inconsistent number of columns");
    }
    Scalar const* d = this->Data();
    // Col-major storage inherits the leading dimension
    // Row-major becomes dynamic, as the number of column of the result is
    // not the same as the starting matrix.
    constexpr int kColLeadDim = (BaseMat::kIsColMajor) ? kLeadingDim : krylov::kDynamic;
    return Matrix<
        Scalar const,
        kRowsAtCompileTime,
        kNumCols,
        kMajorDirection,
        krylov::Viewed<kOwnership>,
        kColLeadDim>{
        d + this->GetOffset(0, this->Cols() - numCols), this->Rows(), numCols, this->LeadDim()};
  }

  /// @brief Extract the slice starting at row ir with p rows
  ///
  /// @param ir Starting row index
  /// @param p  Number of rows to extract
  ///
  /// @return View of extracted slice
  template <int kRowsSliceAtCompile = krylov::kDynamic, int kDummyColNumber = kColsAtCompileTime>
    requires(
        kDummyColNumber == kColsAtCompileTime && kColsAtCompileTime == 1 &&
        kMajorDirection == krylov::Direction::ColMajor)
  [[nodiscard]] auto Slice(int ir, int p) {
    MOCHI_ASSERT_VERBOSE((ir >= 0) && (ir < this->Rows()), "Out-of-range row index");
    MOCHI_ASSERT_VERBOSE(p >= 0, "Inappropriate number of rows");
    auto const numRows = mochi::Min(p, this->Rows() - ir);
    MOCHI_ASSERT_VERBOSE(numRows >= p, "Requested number of rows larger than available");
    if constexpr (kRowsSliceAtCompile > 0) {
      MOCHI_ASSERT_VERBOSE(numRows == kRowsSliceAtCompile, "Inconsistent number of rows");
    }
    return Matrix<
        Scalar,
        kRowsSliceAtCompile,
        1,
        krylov::Direction::ColMajor,
        krylov::Viewed<kOwnership>,
        krylov::kAutomaticLeadDim>{this->Data() + ir, numRows};
  }

  /// @brief Extract the slice starting at row ir with p rows
  ///
  /// @param ir Starting row index
  /// @param p  Number of rows to extract
  ///
  /// @return View of extracted slice
  template <int kRowsSliceAtCompile = krylov::kDynamic, int kDummyColNumber = kColsAtCompileTime>
    requires(
        kDummyColNumber == kColsAtCompileTime && kColsAtCompileTime == 1 &&
        kMajorDirection == krylov::Direction::ColMajor)
  [[nodiscard]] auto Slice(int ir, int p) const {
    MOCHI_ASSERT_VERBOSE((ir >= 0) && (ir < this->Rows()), "Out-of-range row index");
    MOCHI_ASSERT_VERBOSE(p >= 0, "Inappropriate number of rows");
    auto const numRows = mochi::Min(p, this->Rows() - ir);
    MOCHI_ASSERT_VERBOSE(numRows >= p, "Requested number of rows larger than available");
    if constexpr (kRowsSliceAtCompile > 0) {
      MOCHI_ASSERT_VERBOSE(numRows == kRowsSliceAtCompile, "Inconsistent number of rows");
    }
    return Matrix<
        Scalar const,
        kRowsSliceAtCompile,
        1,
        krylov::Direction::ColMajor,
        krylov::Viewed<kOwnership>,
        krylov::kAutomaticLeadDim>{this->Data() + ir, numRows};
  }

  /// @brief Get a transposed view of a matrix.
  Matrix<
      Scalar,
      kColsAtCompileTime,
      kRowsAtCompileTime,
      ~kMajorDirection,
      krylov::Viewed<kOwnership>,
      kLeadingDim>
  Transpose() {
    return {this->Data(), this->Cols(), this->Rows(), this->LeadDim()};
  }
  /// @brief Get a read-only transposed view of a matrix.
  Matrix<
      Scalar const,
      kColsAtCompileTime,
      kRowsAtCompileTime,
      ~kMajorDirection,
      krylov::Viewed<kOwnership>,
      kLeadingDim>
  Transpose() const {
    return {this->Data(), this->Cols(), this->Rows(), this->LeadDim()};
  }

  /// @brief Extract the top rows from the matrix
  ///
  /// @param numRows Number of rows to extract
  /// @return Matrix view of the extracted rows
  template <int kNumRows = krylov::kDynamic>
  [[nodiscard]] auto TopRows(int numRows) {
    MOCHI_ASSERT_VERBOSE(
        (numRows >= 0) && (numRows <= this->Rows()), "Out-of-range number of rows");
    if constexpr (kNumRows >= 0) {
      MOCHI_ASSERT_VERBOSE(numRows == kNumRows, "Inconsistent number of rows");
    }
    constexpr bool kIsColVector =
        kColsAtCompileTime == 1 && kMajorDirection == krylov::Direction::ColMajor;
    constexpr int kMyLeadDim = (BaseMat::kIsRowMajor || kIsColVector)
        ? kLeadingDim
        : ((kLeadingDim == krylov::kAutomaticLeadDim) ? krylov::kDynamic : kLeadingDim);
    return Matrix<
        Scalar,
        kNumRows,
        kColsAtCompileTime,
        kMajorDirection,
        krylov::Viewed<kOwnership>,
        kMyLeadDim>{this->Data(), numRows, this->Cols(), this->LeadDim()};
  }

  /// @brief Extract the top rows from the matrix
  ///
  /// @param numRows Number of rows to extract
  /// @return Matrix view of the extracted rows
  template <int kNumRows = krylov::kDynamic>
  [[nodiscard]] auto TopRows(int numRows) const {
    MOCHI_ASSERT_VERBOSE(
        (numRows >= 0) && (numRows <= this->Rows()), "Out-of-range number of rows");
    if constexpr (kNumRows >= 0) {
      MOCHI_ASSERT_VERBOSE(numRows == kNumRows, "Inconsistent number of rows");
    }
    constexpr bool kIsColVector =
        kColsAtCompileTime == 1 && kMajorDirection == krylov::Direction::ColMajor;
    constexpr int kMyLeadDim = (BaseMat::kIsRowMajor || kIsColVector)
        ? kLeadingDim
        : ((kLeadingDim == krylov::kAutomaticLeadDim) ? krylov::kDynamic : kLeadingDim);
    return Matrix<
        Scalar const,
        kNumRows,
        kColsAtCompileTime,
        kMajorDirection,
        krylov::Viewed<kOwnership>,
        kMyLeadDim>{this->data(), numRows, this->Cols(), this->LeadDim()};
  }

  /// @brief Extract the bottom rows from the matrix
  ///
  /// @param numRows Number of rows to extract
  /// @return Matrix view of the extracted rows
  template <int kNumRows = krylov::kDynamic>
  [[nodiscard]] auto BottomRows(int numRows) {
    MOCHI_ASSERT_VERBOSE(
        (numRows >= 0) && (numRows <= this->Rows()), "Out-of-range number of rows");
    if constexpr (kNumRows >= 0) {
      MOCHI_ASSERT_VERBOSE(numRows == kNumRows, "Inconsistent number of rows");
    }
    Scalar* d = this->Data();
    constexpr bool kIsColVector =
        kColsAtCompileTime == 1 && kMajorDirection == krylov::Direction::ColMajor;
    constexpr int kMyLeadDim = (BaseMat::kIsRowMajor || kIsColVector)
        ? kLeadingDim
        : ((kLeadingDim == krylov::kAutomaticLeadDim) ? krylov::kDynamic : kLeadingDim);
    return Matrix<
        Scalar,
        kNumRows,
        kColsAtCompileTime,
        kMajorDirection,
        krylov::Viewed<kOwnership>,
        kMyLeadDim>{
        d + this->GetOffset(this->Rows() - numRows, 0), numRows, this->Cols(), this->LeadDim()};
  }

  /// @brief Extract the bottom rows from the matrix
  ///
  /// @param numRows Number of rows to extract
  /// @return Matrix view of the extracted rows
  template <int kNumRows = krylov::kDynamic>
  [[nodiscard]] auto BottomRows(int numRows) const {
    MOCHI_ASSERT_VERBOSE(
        (numRows >= 0) && (numRows <= this->Rows()), "Out-of-range number of rows");
    if constexpr (kNumRows >= 0) {
      MOCHI_ASSERT_VERBOSE(numRows == kNumRows, "Inconsistent number of rows");
    }
    constexpr bool kIsColVector =
        kColsAtCompileTime == 1 && kMajorDirection == krylov::Direction::ColMajor;
    constexpr int kMyLeadDim = (BaseMat::kIsRowMajor || kIsColVector)
        ? kLeadingDim
        : ((kLeadingDim == krylov::kAutomaticLeadDim) ? krylov::kDynamic : kLeadingDim);
    return Matrix<
        Scalar const,
        kNumRows,
        kColsAtCompileTime,
        kMajorDirection,
        krylov::Viewed<kOwnership>,
        kMyLeadDim>{
        this->data() + this->GetOffset(this->Rows() - numRows, 0),
        numRows,
        this->Cols(),
        this->LeadDim()};
  }

  /// @brief Extracts the diagonal of the matrix
  /// @return Column vector view of the diagonal
  [[nodiscard]] auto Diagonal() {
    static_assert(!kIsCuda, "Diagonal view not supported on CUDA");
    auto constexpr kEntries = (kRowsAtCompileTime >= 0 && kColsAtCompileTime >= 0)
        ? Min(kRowsAtCompileTime, kColsAtCompileTime)
        : krylov::kDynamic;
    auto constexpr kLeadDimOut = (kLeadingDim > 0) ? kLeadingDim + 1 : krylov::kDynamic;

    return Matrix<
        Scalar,
        kEntries,
        1,
        krylov::Direction::RowMajor,
        krylov::Ownership::View,
        kLeadDimOut>(this->Data(), Min(this->Rows(), this->Cols()), 1, this->LeadDim() + 1);
  }

  /// @brief Extracts the diagonal of the matrix
  /// @return Column vector view of the diagonal
  [[nodiscard]] auto Diagonal() const {
    static_assert(!kIsCuda, "Diagonal view not supported on CUDA");
    auto constexpr kEntries = (kRowsAtCompileTime >= 0 && kColsAtCompileTime >= 0)
        ? Min(kRowsAtCompileTime, kColsAtCompileTime)
        : krylov::kDynamic;
    auto constexpr kLeadDimOut = (kLeadingDim > 0) ? kLeadingDim + 1 : krylov::kDynamic;
    return Matrix<
        Scalar const,
        kEntries,
        1,
        krylov::Direction::RowMajor,
        krylov::Ownership::View,
        kLeadDimOut>(this->Data(), Min(this->Rows(), this->Cols()), 1, this->LeadDim() + 1);
  }

  Matrix& Normalize() &
    requires(kIsVector && std::is_floating_point_v<Scalar>)
  {
    *this *= Scalar{1} / Max(this->Norm(), std::numeric_limits<Scalar>::min());
    return *this;
  }

  Matrix Normalize() &&
    requires(kIsVector && std::is_floating_point_v<Scalar>)
  {
    this->Normalize();
    return *this;
  }

  // Return a matrix that owns a copy of the values.
  [[nodiscard]] auto Duplicate() const {
    return Matrix<
        std::remove_const_t<Scalar>,
        kRowsAtCompileTime,
        kColsAtCompileTime,
        kMajorDirection,
        krylov::Owning<kOwnership>>{*this};
  }

  // Entry-wise access functions

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr Scalar& operator()(int r, int c)
    requires(!kIsCuda)
  {
    MOCHI_ASSERT_VERBOSE(
        (r >= 0) && (r < this->Rows()) && (c >= 0) && (c < this->Cols()), "Index out-of-bounds");
    return this->Data()[this->GetOffset(r, c)];
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr Scalar const& operator()(int r, int c) const
    requires(!kIsCuda)
  {
    MOCHI_ASSERT_VERBOSE(
        (r >= 0) && (r < this->Rows()) && (c >= 0) && (c < this->Cols()), "Index out-of-bounds");
    return this->Data()[this->GetOffset(r, c)];
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr Scalar& operator()(int i)
    requires(kIsHostVector)
  {
    if constexpr (kRowsAtCompileTime == 1) {
      MOCHI_ASSERT_VERBOSE((i >= 0) && (i < this->Cols()), "Index out-of-bounds");
      return this->Data()[this->GetOffset(0, i)];
    } else {
      MOCHI_ASSERT_VERBOSE((i >= 0) && (i < this->Rows()), "Index out-of-bounds");
      return this->Data()[this->GetOffset(i, 0)];
    }
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr Scalar const& operator()(int i) const
    requires(kIsHostVector)
  {
    if constexpr (kRowsAtCompileTime == 1) {
      MOCHI_ASSERT_VERBOSE((i >= 0) && (i < this->Cols()), "Index out-of-bounds");
      return this->Data()[this->GetOffset(0, i)];
    } else {
      MOCHI_ASSERT_VERBOSE((i >= 0) && (i < this->Rows()), "Index out-of-bounds");
      return this->Data()[this->GetOffset(i, 0)];
    }
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr Scalar& operator[](int i)
    requires(kIsHostVector)
  {
    return operator()(i);
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr Scalar const& operator[](int i) const
    requires(kIsHostVector)
  {
    return operator()(i);
  }

  template <IsCompatibleMatrixClass<Matrix> YMatrix>
  [[nodiscard]] NonConstScalar Dot(YMatrix const& ymat) const {
    auto const xrows = this->Rows();
    MOCHI_ASSERT_VERBOSE(
        (xrows == ymat.Rows()) && (this->Cols() == ymat.Cols()), "Dimensions do not match.");
    MOCHI_ASSERT_VERBOSE(this->Cols() == 1, "Inner product works only for column vectors.");

    NonConstScalar result = 0;
    if constexpr ((kIsCuda) || (YMatrix::kIsCuda)) {
      static_assert(kIsCuda == YMatrix::kIsCuda, "Mismatched memory ownership");
      static_assert(
          kIsColMajor && YMatrix::kIsColMajor,
          "CUDA dot product only supported for col-major vectors");
      result = mochi::details::CudaDot(this->Rows(), this->Data(), ymat.Data());
    } else if constexpr (kUseSimd && kIsColMajor && YMatrix::kIsColMajor) {
      auto const* xData = this->Data();
      auto const* yData = ymat.Data();
      int pos = 0;
      auto partResult = SimdZero<VType>();
      constexpr auto kShift = 3 * VType::kSize; // Value chosen empirically chosen value
      constexpr int kValuesAtCT =
          Max(kRowsAtCompileTime, krylov::details::MatTraits<YMatrix>::kNumRows);
      if constexpr (kValuesAtCT == krylov::kDynamic || kValuesAtCT >= kShift) {
        for (; pos + kShift <= xrows; pos += kShift) {
          // PERFORMANCE NOTE:
          // This is at least 2X faster than simply using MulAdd for one SIMD vector at a time (see
          // "Dot" in mochi_benchmark for comparison of similar algorithms). The reason has to do
          // with the latency of the SIMD operations. It is better to do multiple multiplications,
          // which do not depend on the output of the previous instructions.
          auto const* px = xData + pos;
          auto const* py = yData + pos;
          VType xy0 = Load<VType>(px + 0 * VType::kSize) * Load<VType>(py + 0 * VType::kSize);
          VType xy1 = Load<VType>(px + 1 * VType::kSize) * Load<VType>(py + 1 * VType::kSize);
          VType xy2 = Load<VType>(px + 2 * VType::kSize) * Load<VType>(py + 2 * VType::kSize);
          partResult += xy0 + xy1 + xy2;
        }
      }
      if constexpr (kValuesAtCT == krylov::kDynamic || kValuesAtCT % kShift >= VType::kSize) {
        for (; pos + VType::kSize <= xrows; pos += VType::kSize) {
          partResult += Load<VType>(xData + pos) * Load<VType>(yData + pos);
        }
      }
      MOCHI_ASSERT_VERBOSE(xrows - pos < VType::kSize, "Unexpected number of leftover entries.");
      partResult += Load<VType>(xData + pos, xrows - pos) * Load<VType>(yData + pos, xrows - pos);
      result = HSum(partResult);
    } else {
      for (int ii = 0; ii < xrows; ++ii) {
        result += (*this)(ii, 0) * ymat(ii, 0);
      }
    }
    return result;
  }

  /// @brief Computes the Frobenius norm.
  ///
  /// @return Frobenius norm
  [[nodiscard]] NonConstScalar Norm() const {
    return Sqrt(NormSqr());
  }

  /// @brief Computes the square of the Frobenius norm.
  ///
  /// @return Square of the Frobenius norm
  [[nodiscard]] NonConstScalar NormSqr() const;

  /// @brief Sets random entries in the matrix
  ///
  /// @param[in] seed Seed for the random generator
  /// The default value is based on the system clock.
  /// @param[in] smin Lower bound for the random values (default = 0)
  /// @param[in] smax Upper bound for the random values (default = 1)
  void SetRandom(
      unsigned int seed = mochi::GetRandomSeed(),
      Scalar smin = Scalar(0),
      Scalar smax = Scalar(1)) {
    if constexpr (krylov::IsCuda(kOwnership)) {
      static_assert(!krylov::IsCuda(kOwnership), "Feature not implemented for CUDA matrices");
    } else {
      auto generator = mochi::RandomGenerator(seed);
      this->UpdateValue([&generator, &smin, &smax](Scalar& v) {
        v = mochi::RandomUniformValue(generator, smin, smax);
      });
    }
  }

  /// @brief Sets matrix entries to 0.
  MOCHI_ANY void SetZero() {
    if constexpr (krylov::IsCuda(kOwnership)) {
      mochi::details::CudaMemSetZero(this->Data(), sizeof(Scalar) * this->StorageSize());
    } else {
      if (this->Data())
        MOCHI_LIKELY {
          if constexpr (kLeadingDim == krylov::kAutomaticLeadDim) {
            std::memset(this->Data(), 0, sizeof(Scalar) * this->Rows() * this->Cols());
          } else if constexpr (kMajorDirection == krylov::Direction::RowMajor) {
            int const numRows = this->Rows();
            int const numCols = this->Cols();
            for (int ir = 0; ir < numRows; ++ir) {
              std::memset(this->Data() + this->GetOffset(ir, 0), 0, sizeof(Scalar) * numCols);
            }
          } else {
            static_assert(kMajorDirection == krylov::Direction::ColMajor);
            int const numRows = this->Rows();
            int const numCols = this->Cols();
            for (int ic = 0; ic < numCols; ++ic) {
              std::memset(this->Data() + this->GetOffset(0, ic), 0, sizeof(Scalar) * numRows);
            }
          }
        }
      else {
        MOCHI_ASSERT_VERBOSE(this->empty(), "Null pointer in non-empty matrix.");
      }
    }
  }

  /// @brief Sets constant entries in the matrix.
  void SetConstant(Scalar alpha) {
    if constexpr (krylov::IsCuda(kOwnership)) {
      static_assert(!krylov::IsCuda(kOwnership), "Feature not implemented for CUDA matrices");
    } else {
      this->UpdateValue([alpha](Scalar& v) { v = alpha; });
    }
  }

  /// @brief Sets matrix to identity. Only valid for square matrices.
  void SetIdentity() {
    static_assert(!krylov::IsCuda(kOwnership), "Feature not implemented for CUDA matrices");
    if constexpr (kRowsAtCompileTime >= 0 && kColsAtCompileTime >= 0) {
      static_assert(kRowsAtCompileTime == kColsAtCompileTime, "Matrix is not square");
    }
    auto const numRows = this->Rows();
    MOCHI_ASSERT_VERBOSE(numRows == this->Cols(), "Matrix is not square.");
    this->SetZero();
    for (int i = 0; i < numRows; i++) {
      (*this)(i, i) = Scalar(1);
    }
  }

 protected:
  // Use the native SIMD size as long as it isn't larger than the matrix size. If it's larger, use
  // half the native size.
  using VType = std::conditional_t<
      kRowsAtCompileTime >= 1 && kRowsAtCompileTime <= Simd<NonConstScalar>::kSize - 1 &&
          kColsAtCompileTime >= 1 && kColsAtCompileTime <= Simd<NonConstScalar>::kSize - 1 &&
          Simd<NonConstScalar, Simd<NonConstScalar>::kSize / 2>::kIsSupported,
      Simd<NonConstScalar, Simd<NonConstScalar>::kSize / 2>,
      Simd<NonConstScalar>>;
  static constexpr bool kUseSimd = VType::kIsSupported; // TODO(T158480383)

  void UpdateValue(auto unaryOp);

  /// @brief Provide the position in the storage array for pair (r, c)
  /// @note This is a utility function to compute an offset. It is agnostic to how the offset will
  /// be used. No out-of-bound checks should be performed. For example, it can be used to compute an
  /// `end()` or `rend()` iterator, in which case `r` or `c` can be outside of the matrix.
  /// @param r Row index (starting from 0)
  /// @param c Column index (starting from 0)
  /// @return  Offset
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr std::size_t GetOffset(int r, int c) const {
    // Warning: Do not move the casts around the full products. This would allow overflow.
    if constexpr (kMajorDirection == krylov::Direction::RowMajor) {
      return static_cast<size_t>(r) * this->LeadDim() + c;
    } else {
      return r + static_cast<size_t>(c) * this->LeadDim();
    }
  }
};

/// @brief Utility function to obtain a transposed view a matrix
[[nodiscard]] auto Transpose(IsMatrix auto&& M) {
  return M.Transpose();
}

/// @brief Utility function to obtain a read-only transposed view a matrix
[[nodiscard]] auto Transpose(IsMatrix auto const& M) {
  return M.Transpose();
}

/// @brief Get a normalized copy of a vector.
template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDim>
[[nodiscard]] auto Normalized(
    Matrix<
        Scalar,
        kRowsAtCompileTime,
        kColsAtCompileTime,
        kMajorDirection,
        kOwnership,
        kLeadingDim> const& M) {
  return Matrix<
             Scalar,
             kRowsAtCompileTime,
             kColsAtCompileTime,
             kMajorDirection,
             krylov::Owning<kOwnership>,
             kLeadingDim>{M}
      .Normalize();
}

/// @brief Compute the cross product of dimension 3 vectors as a Column Vector.
template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership_x,
    krylov::Ownership kOwnership_y,
    int kLeadingDim>
  requires((kRowsAtCompileTime == 1) || (kColsAtCompileTime == 1))
[[nodiscard]] auto Cross(
    Matrix<
        Scalar,
        kRowsAtCompileTime,
        kColsAtCompileTime,
        kMajorDirection,
        kOwnership_x,
        kLeadingDim> const& x,
    Matrix<
        Scalar,
        kRowsAtCompileTime,
        kColsAtCompileTime,
        kMajorDirection,
        kOwnership_y,
        kLeadingDim> const& y) {
  MOCHI_ASSERT_VERBOSE(
      (x.Rows() * x.Cols() == 3) && (y.Rows() * y.Cols() == 3),
      "Cross product only applies to vectors of size 3.");
  using NonConstScalar = std::remove_const_t<Scalar>;
  return Matrix<NonConstScalar, 3, 1, krylov::Direction::ColMajor, krylov::Ownership::Owner>{
      x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]};
}

template <
    typename Scalar,
    int kRowsAtCompileTime = krylov::kDynamic,
    int kColsAtCompileTime = krylov::kDynamic,
    krylov::Direction kMajorDirection = krylov::Direction::ColMajor,
    int kLeadingDim = krylov::kAutomaticLeadDim>
using MatrixView = Matrix<
    Scalar,
    kRowsAtCompileTime,
    kColsAtCompileTime,
    kMajorDirection,
    krylov::Ownership::View,
    kLeadingDim>;

template <
    typename Scalar,
    int kRowsAtCompileTime = krylov::kDynamic,
    int kColsAtCompileTime = krylov::kDynamic,
    krylov::Ownership kOwnership = krylov::Ownership::Owner,
    int kLeadingDim = krylov::kAutomaticLeadDim>
using RowMatrix = Matrix<
    Scalar,
    kRowsAtCompileTime,
    kColsAtCompileTime,
    krylov::Direction::RowMajor,
    kOwnership,
    kLeadingDim>;

template <
    typename Scalar,
    int kRowsAtCompileTime = krylov::kDynamic,
    int kColsAtCompileTime = krylov::kDynamic,
    int kLeadingDim = krylov::kAutomaticLeadDim>
using RowMatrixView = Matrix<
    Scalar,
    kRowsAtCompileTime,
    kColsAtCompileTime,
    krylov::Direction::RowMajor,
    krylov::Ownership::View,
    kLeadingDim>;

template <typename Scalar, int kSizeAtCompileTime = krylov::kDynamic>
using ColumnVector = Matrix<Scalar, kSizeAtCompileTime, 1, krylov::Direction::ColMajor>;

template <typename Scalar, int kSizeAtCompileTime = krylov::kDynamic>
using ColumnVectorView =
    Matrix<Scalar, kSizeAtCompileTime, 1, krylov::Direction::ColMajor, krylov::Ownership::View>;

template <typename Scalar, int kSizeAtCompileTime = krylov::kDynamic>
using RowVector = Matrix<Scalar, 1, kSizeAtCompileTime, krylov::Direction::RowMajor>;

template <typename Scalar, int kSizeAtCompileTime = krylov::kDynamic>
using RowVectorView =
    Matrix<Scalar, 1, kSizeAtCompileTime, krylov::Direction::RowMajor, krylov::Ownership::View>;

/// @brief Utility function to convert a matrix to a view object
template <
    int kOutRowSize = krylov::kKeep,
    int kOutColSize = krylov::kKeep,
    int kOutLeadDim = krylov::kKeep,
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDim>
[[nodiscard]] auto AsView(
    Matrix<
        Scalar,
        kRowsAtCompileTime,
        kColsAtCompileTime,
        kMajorDirection,
        kOwnership,
        kLeadingDim>& M);

/// @brief Utility function to convert a matrix to a const view object
template <
    int kOutRowSize = krylov::kKeep,
    int kOutColSize = krylov::kKeep,
    int kOutLeadDim = krylov::kKeep,
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDim>
[[nodiscard]] auto AsConstView(
    Matrix<
        Scalar,
        kRowsAtCompileTime,
        kColsAtCompileTime,
        kMajorDirection,
        kOwnership,
        kLeadingDim> const& M);

} // namespace mochi

//
// Implementation of functions
//

namespace mochi {

template <
    typename Scalar_,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDim>
std::remove_const_t<Scalar_>
Matrix<Scalar_, kRowsAtCompileTime, kColsAtCompileTime, kMajorDirection, kOwnership, kLeadingDim>::
    NormSqr() const {
  auto const mrows = this->Rows();
  auto const mcols = this->Cols();
  NonConstScalar frobeniusNorm2 = 0;
  int outerLoops = 0, innerLoops = 0, unmaskRow = 0, unmaskCol = 0;
  if constexpr (kIsRowMajor) {
    outerLoops = mrows;
    innerLoops = mcols;
    unmaskRow = 1;
    unmaskCol = 0;
  } else {
    outerLoops = mcols;
    innerLoops = mrows;
    unmaskRow = 0;
    unmaskCol = 1;
  }
  if constexpr (kLeadingDim == krylov::kAutomaticLeadDim) {
    outerLoops = 1;
    innerLoops = mrows * mcols;
  }
  constexpr int kInnerLoops = (kLeadingDim == krylov::kAutomaticLeadDim)
      ? (kIsDynamic ? krylov::kDynamic : kRowsAtCompileTime * kColsAtCompileTime)
      : (kIsRowMajor ? kColsAtCompileTime : kRowsAtCompileTime);
  MOCHI_WARNING_PUSH()
  MOCHI_WARNING_IGNORE_MSVC(4127) // conditional expression is constant
  MOCHI_ASSERT_VERBOSE(kInnerLoops == krylov::kDynamic || kInnerLoops == innerLoops);
  MOCHI_WARNING_POP()

  if constexpr (kIsCuda) {
    for (int io = 0; io < outerLoops; ++io) {
      auto const* ptr = this->Data() + this->GetOffset(io * unmaskRow, io * unmaskCol);
      frobeniusNorm2 += mochi::details::CudaDot(innerLoops, ptr, ptr);
    }
  } else if constexpr (kUseSimd) {
    VType vNorm = {};
    for (int io = 0; io < outerLoops; ++io) {
      auto* ptr = this->Data() + this->GetOffset(io * unmaskRow, io * unmaskCol);
      int pos = 0;
      constexpr auto kShift = 3 * VType::kSize; // Value chosen empirically
      static_assert(kShift % VType::kSize == 0); // Assumed by constexpr branches below
      if constexpr (kInnerLoops == krylov::kDynamic || kInnerLoops >= kShift) {
        for (; pos + kShift <= innerLoops; pos += kShift) {
          // PERFORMANCE NOTE:
          // This is at least 2X faster than simply using MulAdd for one SIMD vector at a time (see
          // "Dot" in mochi_benchmark for comparison of similar algorithms). The reason has to do
          // with the latency of the SIMD operations. It is better to do multiple multiplications,
          // which do not depend on the output of the previous instructions.
          auto v0 = Load<VType>(ptr + pos + 0 * VType::kSize);
          auto v1 = Load<VType>(ptr + pos + 1 * VType::kSize);
          auto v2 = Load<VType>(ptr + pos + 2 * VType::kSize);
          vNorm += (v0 * v0) + (v1 * v1) + (v2 * v2);
        }
      }
      if constexpr (kInnerLoops == krylov::kDynamic || kInnerLoops % kShift >= VType::kSize) {
        for (; pos + VType::kSize <= innerLoops; pos += VType::kSize) {
          auto v0 = Load<VType>(ptr + pos);
          vNorm += v0 * v0;
        }
      }
      MOCHI_ASSERT_VERBOSE(
          innerLoops - pos < VType::kSize, "Unexpected number of leftover entries.");
      if constexpr (kInnerLoops == krylov::kDynamic || kInnerLoops % VType::kSize > 0) {
        if (innerLoops - pos > 0) {
          auto v0 = Load<VType>(ptr + pos, innerLoops - pos);
          vNorm += v0 * v0;
        }
      }
    }
    frobeniusNorm2 = HSum(vNorm);

  } else { // Non-SIMD fallback.
    auto op = [&frobeniusNorm2](Scalar v) { frobeniusNorm2 += v * v; };
    for (int io = 0; io < outerLoops; ++io) {
      auto* ptr = this->Data() + this->GetOffset(io * unmaskRow, io * unmaskCol);
      std::for_each(ptr, ptr + innerLoops, op);
    }
  }
  return frobeniusNorm2;
}

template <
    typename Scalar_,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDim>
void Matrix<
    Scalar_,
    kRowsAtCompileTime,
    kColsAtCompileTime,
    kMajorDirection,
    kOwnership,
    kLeadingDim>::UpdateValue(auto unaryOp) {
  int const mrows = this->Rows();
  int const mcols = this->Cols();
  if constexpr (kLeadingDim == krylov::kAutomaticLeadDim) {
    auto* ptr = this->Data();
    std::for_each(ptr, ptr + mrows * mcols, unaryOp);
  } else if constexpr (kMajorDirection == krylov::Direction::RowMajor) {
    for (int ir = 0; ir < mrows; ++ir) {
      auto* ptr = this->Data() + GetOffset(ir, 0);
      std::for_each(ptr, ptr + mcols, unaryOp);
    }
  } else {
    for (int ic = 0; ic < mcols; ++ic) {
      auto* ptr = this->Data() + GetOffset(0, ic);
      std::for_each(ptr, ptr + mrows, unaryOp);
    }
  }
}

/// @brief Utility function to convert a matrix to a view object
template <
    int kOutRowSize,
    int kOutColSize,
    int kOutLeadDim,
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDim>
auto AsView(
    Matrix<
        Scalar,
        kRowsAtCompileTime,
        kColsAtCompileTime,
        kMajorDirection,
        kOwnership,
        kLeadingDim>& M) {
  //--- Verify template parameter values
  if constexpr (kOutRowSize <= 0) {
    static_assert(
        (kOutRowSize == krylov::kKeep) || (kOutRowSize == krylov::kDynamic),
        "Inappropriate parameter value");
  } else {
    static_assert(kOutRowSize == kRowsAtCompileTime, "Inconsistent parameter value");
  }
  //
  if constexpr (kOutColSize <= 0) {
    static_assert(
        (kOutColSize == krylov::kKeep) || (kOutColSize == krylov::kDynamic),
        "Inappropriate parameter value");
  } else {
    static_assert(kOutColSize == kColsAtCompileTime, "Inconsistent parameter value");
  }
  //
  if constexpr (kOutLeadDim <= 0) {
    static_assert(
        (kOutLeadDim == krylov::kKeep) || (kOutLeadDim == krylov::kDynamic),
        "Inappropriate parameter value");
  } else {
    static_assert(kOutLeadDim == kLeadingDim, "Inconsistent parameter value");
  }
  //
  constexpr int kRows =
      ((kOutRowSize == krylov::kKeep) || (kOutRowSize > 0)) ? kRowsAtCompileTime : krylov::kDynamic;
  constexpr int kCols =
      ((kOutColSize == krylov::kKeep) || (kOutColSize > 0)) ? kColsAtCompileTime : krylov::kDynamic;
  constexpr int kLeadDim =
      ((kOutLeadDim == krylov::kKeep) || (kOutLeadDim > 0)) ? kLeadingDim : krylov::kDynamic;
  return Matrix<Scalar, kRows, kCols, kMajorDirection, krylov::Viewed<kOwnership>, kLeadDim>(
      M.Data(), M.Rows(), M.Cols(), M.LeadDim());
}

/// @brief Utility function to convert a matrix to a const view object
template <
    int kOutRowSize,
    int kOutColSize,
    int kOutLeadDim,
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    krylov::Direction kMajorDirection,
    krylov::Ownership kOwnership,
    int kLeadingDim>
auto AsConstView(
    Matrix<
        Scalar,
        kRowsAtCompileTime,
        kColsAtCompileTime,
        kMajorDirection,
        kOwnership,
        kLeadingDim> const& M) {
  //--- Verify template parameter values
  if constexpr (kOutRowSize <= 0) {
    static_assert(
        (kOutRowSize == krylov::kKeep) || (kOutRowSize == krylov::kDynamic),
        "Inappropriate parameter value");
  } else {
    static_assert(kOutRowSize == kRowsAtCompileTime, "Inconsistent parameter value");
  }
  //
  if constexpr (kOutColSize <= 0) {
    static_assert(
        (kOutColSize == krylov::kKeep) || (kOutColSize == krylov::kDynamic),
        "Inappropriate parameter value");
  } else {
    static_assert(kOutColSize == kColsAtCompileTime, "Inconsistent parameter value");
  }
  //
  if constexpr (kOutLeadDim <= 0) {
    static_assert(
        (kOutLeadDim == krylov::kKeep) || (kOutLeadDim == krylov::kDynamic),
        "Inappropriate parameter value");
  } else {
    static_assert(kOutLeadDim == kLeadingDim, "Inconsistent parameter value");
  }
  //
  constexpr int kRows =
      ((kOutRowSize == krylov::kKeep) || (kOutRowSize > 0)) ? kRowsAtCompileTime : krylov::kDynamic;
  constexpr int kCols =
      ((kOutColSize == krylov::kKeep) || (kOutColSize > 0)) ? kColsAtCompileTime : krylov::kDynamic;
  constexpr int kLeadDim =
      ((kOutLeadDim == krylov::kKeep) || (kOutLeadDim > 0)) ? kLeadingDim : krylov::kDynamic;
  return Matrix<Scalar const, kRows, kCols, kMajorDirection, krylov::Viewed<kOwnership>, kLeadDim>(
      M.Data(), M.Rows(), M.Cols(), M.LeadDim());
}

} // namespace mochi
