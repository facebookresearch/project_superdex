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

#include <mochi_core/linear_algebra/base_enums.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>
#include <mochi_core/memory/allocator.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/concepts.h>
#include <mochi_core/utils/debug.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <type_traits>
#include <utility>

namespace mochi::krylov {

/// @brief Flag for indicating a run-time matrix dimension.
constexpr int kDynamic = -1;

/// @brief Flag for selecting automatically the leading dimension.
/// It will be the number of columns when the storage is row-major.
/// It will be the number of rows when the storage is column-major.
constexpr int kAutomaticLeadDim = 0;

} // namespace mochi::krylov

namespace mochi::details {

template <int kVal>
struct IntOrEmpty {
  static_assert(kVal >= 0, "Invalid value."); // This version should only instantiate fixed sizes.

  MOCHI_ANY MOCHI_FORCE_INLINE IntOrEmpty() = default;

  MOCHI_ANY MOCHI_FORCE_INLINE explicit IntOrEmpty([[maybe_unused]] int val_) {
    MOCHI_ASSERT_VERBOSE(val_ == kVal);
  }

  MOCHI_ANY MOCHI_FORCE_INLINE IntOrEmpty(IntOrEmpty const&) = default;

  MOCHI_ANY MOCHI_FORCE_INLINE IntOrEmpty& operator=([[maybe_unused]] int val_) {
    MOCHI_ASSERT_VERBOSE(val_ == kVal);
    return *this;
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr int iVal() const {
    return kVal;
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr size_t sVal() const {
    return static_cast<size_t>(kVal);
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE friend constexpr bool operator<=(
      int a,
      IntOrEmpty const& /*unused*/) {
    return a <= kVal;
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE friend constexpr bool operator<(
      int a,
      IntOrEmpty const& /*unused*/) {
    return a < kVal;
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE friend constexpr bool operator<=(
      IntOrEmpty const& /*unused*/,
      int a) {
    return kVal <= a;
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE friend constexpr bool operator<(
      IntOrEmpty const& /*unused*/,
      int a) {
    return kVal < a;
  }
};

template <typename T>
inline constexpr int kValueAtCompileTime = 0;

template <int kVal>
inline constexpr int kValueAtCompileTime<IntOrEmpty<kVal>> = kVal;

template <>
struct IntOrEmpty<krylov::kDynamic> {
  int val; // Must be non-negative.

  IntOrEmpty() = delete;

  MOCHI_ANY MOCHI_FORCE_INLINE explicit IntOrEmpty(int val_) : val(val_) {
    MOCHI_ASSERT_VERBOSE(val >= 0, "Value must not be negative.");
  }

  MOCHI_ANY MOCHI_FORCE_INLINE IntOrEmpty(IntOrEmpty const&) = default;

  MOCHI_ANY MOCHI_FORCE_INLINE IntOrEmpty& operator=(int val_) {
    MOCHI_ASSERT_VERBOSE(val_ >= 0, "Value must not be negative.");
    val = val_;
    return *this;
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr int iVal() const {
    return val;
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr size_t sVal() const {
    return static_cast<size_t>(val);
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE friend constexpr bool operator<(
      int a,
      IntOrEmpty const& b) {
    return a < b.val;
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE friend constexpr bool operator<=(
      int a,
      IntOrEmpty const& b) {
    return a <= b.val;
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE friend constexpr bool operator<=(
      IntOrEmpty const& b,
      int a) {
    return b.val <= a;
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE friend constexpr bool operator<(
      IntOrEmpty const& b,
      int a) {
    return b.val < a;
  }
};

template <int kRowsAtCompileTime, int kColsAtCompileTime, int kLeadingDim>
struct Sizes {
  constexpr Sizes() = default;
  constexpr Sizes(int, int = kColsAtCompileTime, int = kLeadingDim) {}
};

template <int kRowsAtCompileTime, int kColsAtCompileTime, int kLeadingDim>
  requires(kRowsAtCompileTime < 0 && kColsAtCompileTime >= 0 && kLeadingDim >= 0)
struct Sizes<kRowsAtCompileTime, kColsAtCompileTime, kLeadingDim> {
  Sizes() = default;
  Sizes(int nr, int /*nc*/, int /*ld*/ = 0) : _rows(nr) {}
  int _rows = 0;
};

template <int kRowsAtCompileTime, int kColsAtCompileTime, int kLeadingDim>
  requires(kRowsAtCompileTime < 0 && kColsAtCompileTime < 0 && kLeadingDim >= 0)
struct Sizes<kRowsAtCompileTime, kColsAtCompileTime, kLeadingDim> {
  Sizes() = default;
  Sizes(int nr, int nc, int /*ld*/ = 0) : _rows(nr), _cols(nc) {}
  int _rows = 0;
  int _cols = 0;
};

template <int kRowsAtCompileTime, int kColsAtCompileTime, int kLeadingDim>
  requires(kRowsAtCompileTime >= 0 && kColsAtCompileTime < 0 && kLeadingDim >= 0)
struct Sizes<kRowsAtCompileTime, kColsAtCompileTime, kLeadingDim> {
  Sizes() = default;
  Sizes(int /*nr*/, int nc, int /*ld*/ = 0) : _cols(nc) {}
  int _cols = 0;
};

template <int kRowsAtCompileTime, int kColsAtCompileTime, int kLeadingDim>
  requires(kRowsAtCompileTime < 0 && kColsAtCompileTime >= 0 && kLeadingDim < 0)
struct Sizes<kRowsAtCompileTime, kColsAtCompileTime, kLeadingDim> {
  Sizes() = default;
  Sizes(int nr, int /*nc*/, int ld) : _rows(nr), _ld(ld) {}
  int _rows = 0;
  int _ld = 0;
};

template <int kRowsAtCompileTime, int kColsAtCompileTime, int kLeadingDim>
  requires(kRowsAtCompileTime < 0 && kColsAtCompileTime < 0 && kLeadingDim < 0)
struct Sizes<kRowsAtCompileTime, kColsAtCompileTime, kLeadingDim> {
  // Constructors with 0, 1, 2 and 3 arguments are implicitly defined.
  int _rows = 0;
  int _cols = 0;
  int _ld = 0;
};

template <int kRowsAtCompileTime, int kColsAtCompileTime, int kLeadingDim>
  requires(kRowsAtCompileTime >= 0 && kColsAtCompileTime < 0 && kLeadingDim < 0)
struct Sizes<kRowsAtCompileTime, kColsAtCompileTime, kLeadingDim> {
  Sizes() = default;
  Sizes(int /*nr*/, int nc, int ld) : _cols(nc), _ld(ld) {}
  int _cols = 0;
  int _ld = 0;
};

template <int kRowsAtCompileTime, int kColsAtCompileTime, int kLeadingDim>
  requires(kRowsAtCompileTime >= 0 && kColsAtCompileTime >= 0 && kLeadingDim < 0)
struct Sizes<kRowsAtCompileTime, kColsAtCompileTime, kLeadingDim> {
  Sizes() = default;
  Sizes(int /*nr*/, int /*nc*/, int ld) : _ld(ld) {}
  int _ld = 0;
};

} // namespace mochi::details

namespace mochi::krylov::details {

template <typename F, typename... T>
constexpr bool all_same = (std::is_same_v<std::decay_t<F>, std::decay_t<T>> && ...);

template <typename Scalar, int kSize, Ownership kOwnership>
struct BaseStorage {
  BaseStorage() = delete;
};

/// @brief Specialization for dynamic sizes.
template <typename Scalar>
struct BaseStorage<Scalar, krylov::kDynamic, Ownership::Owner> {
  BaseStorage() = default;

  /// @brief Constructor
  /// @param[in] allocator Pointer to the custom memory resource
  /// When allocator is nullptr, the allocation uses the default resource
  explicit BaseStorage(Allocator* allocator)
      : _data(nullptr), _allocator((allocator) ? allocator : GetDefaultAllocator()) {}

  /// @brief Constructor
  /// @param[in] n Size requested
  /// @param[in] allocator Pointer to the custom memory resource
  /// When allocator is nullptr, the allocation uses the default resource allocator
  explicit BaseStorage(size_t n, Allocator* allocator = GetDefaultAllocator())
      : _data(nullptr), _n(n), _allocator((allocator) ? allocator : GetDefaultAllocator()) {
    AllocateValues();
  }

  BaseStorage(BaseStorage&& rhs, Allocator* allocator) : _n(rhs._n), _allocator(allocator) {
    MOCHI_ASSERT_VERBOSE(allocator && rhs._allocator);
    if ((allocator == rhs._allocator) || (allocator->is_equal(*rhs._allocator))) {
      // Move ownership of the memory
      _data = rhs._data;
      rhs._data = nullptr;
    } else {
      // Allocate and copy the values
      AllocateValues();
      std::copy(rhs._data, rhs._data + _n, _data);
    }
  }

  BaseStorage(BaseStorage&& b) noexcept : _data(b._data), _n(b._n), _allocator(b._allocator) {
    b._data = nullptr;
    b._n = 0;
  }

  MOCHI_DECLARE_NO_COPY(BaseStorage);

  /// @brief Destructor
  ~BaseStorage() {
    if (_data) {
      _allocator->deallocate(_data, _n * sizeof(Scalar), alignof(Scalar));
    }
  }

  /// @brief Return raw pointer to the memory (overload for compatibility with std containers)
  MOCHI_ANY MOCHI_FORCE_INLINE Scalar* data() {
    return _data;
  }

  /// @brief Return raw pointer to the memory (overload for compatibility with std containers)
  MOCHI_ANY MOCHI_FORCE_INLINE Scalar const* data() const {
    return _data;
  }

  /// @brief Return raw pointer to the memory.
  MOCHI_ANY MOCHI_FORCE_INLINE Scalar* Data() {
    return _data;
  }

  /// @brief Return raw pointer to the memory.
  MOCHI_ANY MOCHI_FORCE_INLINE Scalar const* Data() const {
    return _data;
  }

  /// @brief Resize the storage space to a specific size
  ///
  /// @param[in] n Size of the storage space
  void Resize(size_t n) {
    if (_n == n) {
      return;
    }
    if (_data) {
      _allocator->deallocate(_data, _n * sizeof(Scalar), alignof(Scalar));
    }
    _n = n;
    AllocateValues();
  }

  /// @brief Returns a similar storage space
  ///
  /// @param[in] n Size of the storage space
  /// @returns BaseStorage object of the specific size
  ///
  BaseStorage GetSimilar(size_t n) const {
    return BaseStorage(n, _allocator);
  }

  MOCHI_FORCE_INLINE Allocator* GetAllocator() const {
    return _allocator;
  }

 protected:
  Scalar* _data = nullptr;
  size_t _n = 0;
  Allocator* _allocator = GetDefaultAllocator();

 private:
  void AllocateValues() {
    _data = static_cast<Scalar*>(_allocator->allocate(_n * sizeof(Scalar), alignof(Scalar)));
  }
};

/// @brief Specialization for sizes known at compile time.
template <typename Scalar, int kSize>
struct BaseStorage<Scalar, kSize, Ownership::Owner> {
  static_assert(kSize >= 0, "Incompatible size");

  constexpr BaseStorage() = default;

  template <typename... Args>
  constexpr BaseStorage(Args&&... args) : _data{std::forward<Args>(args)...} {
    static_assert(sizeof...(Args) == kSize, "Invalid number of values for initialization");
  }

  /// @brief Returns a similar storage space
  ///
  /// @param[in] n Size
  /// @returns BaseStorage object of the specific size
  ///
  /// @note For this class, the variable `n` is unused as the dimension is obtained
  /// from the template parameters.
  constexpr BaseStorage GetSimilar([[maybe_unused]] size_t n) const {
    MOCHI_ASSERT_VERBOSE(n == kSize, "Inconsistent sizes.");
    return BaseStorage{};
  }

  /// @brief Return raw pointer to the memory.
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr Scalar* Data() {
    return _data;
  }

  /// @brief Return raw pointer to the memory.
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr Scalar const* Data() const {
    return _data;
  }

  /// @brief Return raw pointer to the memory.
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr Scalar* data() {
    return _data;
  }

  /// @brief Return raw pointer to the memory.
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr Scalar const* data() const {
    return _data;
  }

 private:
  Scalar _data[kSize];
};

template <typename Scalar, int kSize>
struct BaseStorage<Scalar, kSize, Ownership::View> {
  static_assert((kSize == krylov::kDynamic) || (kSize >= 0), "Incompatible size");

  BaseStorage() = default;

  constexpr explicit BaseStorage(Scalar* data) : _data(data) {}

  /// @brief Return raw pointer to the memory (overload for compatibility with std containers)
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr Scalar* data() {
    return _data;
  }

  /// @brief Return raw pointer to the memory (overload for compatibility with std containers)
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr Scalar const* data() const {
    return _data;
  }

  /// @brief Return raw pointer to the memory.
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr Scalar* Data() {
    return _data;
  }

  /// @brief Return raw pointer to the memory.
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr Scalar const* Data() const {
    return _data;
  }

  /// @brief Returns a storage space of similar type with a specific size
  ///
  /// @param[in] n Size of the storage space
  /// @returns BaseStorage object of the specific size
  ///
  /// @note When the ownership is View, the variable `n` is unused
  /// as the dimension is not stored in the class.
  constexpr BaseStorage GetSimilar([[maybe_unused]] size_t n) const {
    return BaseStorage(_data);
  }

  ~BaseStorage() = default;

 protected:
  Scalar* _data = nullptr;
};

template <typename Scalar, int kSize, Ownership kOwnership>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr auto GetBaseStorage(
    [[maybe_unused]] Allocator* allocator) {
  static_assert(IsOwner(kOwnership), "Unsupported ownership with allocator");
  if constexpr (kSize >= 0) {
    return BaseStorage<Scalar, kSize, kOwnership>();
  } else {
    return BaseStorage<Scalar, kSize, kOwnership>(allocator);
  }
}

template <typename Scalar, int kSize, Ownership kOwnership>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr auto MoveBaseStorage(
    BaseStorage<Scalar, kSize, kOwnership>&& other,
    [[maybe_unused]] Allocator* allocator) {
  static_assert(IsOwner(kOwnership), "Unsupported ownership with allocator");
  if constexpr (kSize >= 0) {
    return BaseStorage<Scalar, kSize, kOwnership>(std::move(other));
  } else {
    return BaseStorage<Scalar, kSize, kOwnership>(std::move(other), allocator);
  }
}

} // namespace mochi::krylov::details

namespace mochi::krylov {

// clang-format off
///
/// @brief BaseMatrix Class to manage the storage of the dense values
///
/// @tparam ScalarIn  Scalar type for the values
/// @tparam kRowsAtCompileTime  Integer flag for the number of rows:
///               = kDynamic : indicates a dynamic number of rows specified at construction (default)
///               > 0 indicates a static number of rows (known at compile time)
/// @tparam kColsAtCompileTime  Integer flag for the number of columns:
///               = kDynamic : indicates a dynamic number of rows specified at construction (default)
///               > 0 indicates a static number of rows (known at compile time)
/// @tparam kMajorDirection  Storage direction for the matrix
///                         `ColMajor` indicates a column-based storage (like in Fortran)
///                         `RowMajor` indicates a row-based storage (like in C)
/// @tparam kOwnership  Flag to indicate how the memory is managed
/// @tparam kLeadingDim  Flag controlling the leading dimension along the storage direction
///                   kLeadingDim = kAutomaticLeadDim (= default): same as the number of rows or columns (according to storage
///                   direction)
///                   kLeadingDim = kDynamic: dynamic value set at construction
///                   kLeadingDim = n > 0: the leading dimension is set to n (at compile time)
///                   LAPACK uses the expression "leading dimension".
///                   Other fields may use the word "stride".
///
// clang-format on
template <
    typename Scalar_,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    Direction kMajorDirection,
    Ownership kOwnership = Ownership::Owner,
    int kLeadingDim = kAutomaticLeadDim>
class MOCHI_EMPTY_BASE BaseMatrix
    : public mochi::details::Sizes<kRowsAtCompileTime, kColsAtCompileTime, kLeadingDim> {
 public:
  static constexpr bool kNumRowsIsDynamic = kRowsAtCompileTime < 0;
  static constexpr bool kNumColsIsDynamic = kColsAtCompileTime < 0;

  using Sizes = mochi::details::Sizes<kRowsAtCompileTime, kColsAtCompileTime, kLeadingDim>;
  using Scalar = Scalar_;
  static constexpr bool kIsDynamic = kRowsAtCompileTime < 0 || kColsAtCompileTime < 0;

  static constexpr bool kIsColMajor = (kMajorDirection == Direction::ColMajor);
  static constexpr bool kIsRowMajor = (kMajorDirection == Direction::RowMajor);

  static constexpr Direction kMajorDir = kMajorDirection;

  static constexpr bool kIsCuda = IsCuda(kOwnership);

  static constexpr bool kIsView = IsView(kOwnership);
  static constexpr bool kIsOwner = IsOwner(kOwnership);
  /// @brief Marker of a vector at compile time.
  /// @details Warning: stride may not be 1 such as in the result of Col(i) or Row(i)
  static constexpr bool kIsVector = kColsAtCompileTime == 1 || kRowsAtCompileTime == 1;
  /// @brief Marker of a vector with stride 1 at compile time.
  static constexpr bool kIsDenseVector =
      (kMajorDirection == Direction::ColMajor && kColsAtCompileTime == 1) ||
      (kMajorDirection == Direction::RowMajor && kRowsAtCompileTime == 1);
  static constexpr bool kIsHostVector = !kIsCuda && kIsVector;
  static constexpr bool kIsDenseHostVector = !kIsCuda && kIsDenseVector;

  // NOLINTBEGIN(bugprone-branch-clone) - Both if and else cases may have the same result. That's
  // OK.
  static_assert(
      kLeadingDim <= 0 || (kIsColMajor ? kRowsAtCompileTime : kColsAtCompileTime) != kDynamic,
      "Illegal case: Compile-time leading dimension and dynamic size along the storage direction");
  static_assert(
      !kIsOwner || kIsDynamic || kLeadingDim == kAutomaticLeadDim ||
          kLeadingDim == (kIsColMajor ? kRowsAtCompileTime : kColsAtCompileTime),
      "Illegal case: Owning matrix with fully compile-time size must use kAutomaticLeadDim "
      "or a leading dimension equal to the size along the storage direction");
  // NOLINTEND(bugprone-branch-clone)

  static_assert(
      (MOCHI_USE_CUDA) || !kIsCuda,
      "CUDA matrices require building with CUDA. To enable CUDA, add the CUDA dependencies to your build configuration and define MOCHI_USE_CUDA=1");

  /// @brief Return raw pointer to the memory (overload for compatibility with std containers)
  MOCHI_ANY MOCHI_FORCE_INLINE Scalar* data() {
    return _values.data();
  }

  /// @brief Return raw pointer to the memory (overload for compatibility with std containers)
  MOCHI_ANY MOCHI_FORCE_INLINE Scalar const* data() const {
    return _values.data();
  }

  /// @brief Return raw pointer to the memory
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr Scalar* Data() {
    return _values.data();
  }

  /// @brief Return raw pointer to the memory
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr Scalar const* Data() const {
    return _values.data();
  }

  /// @brief Return the number of values stored
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr size_t size() const {
    auto size = static_cast<size_t>(Rows()) * static_cast<size_t>(Cols());
    MOCHI_ASSERT_VERBOSE(
        size == StorageSize(),
        "The values in this matrix are not stored contiguously in memory. "
        "Please do not use the 'size' method to avoid confusion, in this case.");
    return size;
  }

  /// @brief Return true if (size() == 0)
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr bool empty() const {
    return (Rows() * Cols()) == 0;
  }

  /// @brief Destructive resize of a dynamic size matrix. Nothing happens if dimensions are
  /// unchanged.
  void Resize(int rows, int cols)
    requires(kIsDynamic && kIsOwner)
  {
    MOCHI_ASSERT_VERBOSE(rows >= 0, "Invalid number of rows.");
    MOCHI_ASSERT_VERBOSE(cols >= 0, "Invalid number of columns.");
    if constexpr (!kNumRowsIsDynamic) {
      MOCHI_ASSERT_VERBOSE(rows == kRowsAtCompileTime, "Inconsistent compile-time number of rows.");
    }
    if constexpr (!kNumColsIsDynamic) {
      MOCHI_ASSERT_VERBOSE(
          cols == kColsAtCompileTime, "Inconsistent compile-time number of columns.");
    }
    if (rows != Rows() || cols != Cols()) {
      if constexpr (requires { this->_rows; }) {
        this->_rows = rows;
      }
      if constexpr (requires { this->_cols; }) {
        this->_cols = cols;
      }
      if constexpr (requires { this->_ld; }) {
        this->_ld = (kMajorDirection == Direction::ColMajor) ? rows : cols;
      }
      AllocateValues();
    }
  }

  /// @brief Destructive resize a dynamic size vector. Nothing happens if size is unchanged.
  void Resize(int s)
    requires(kIsDenseVector && kIsDynamic && kIsOwner)
  {
    MOCHI_ASSERT_VERBOSE(s >= 0, "Invalid size.");
    if (s != StorageSize()) {
      if constexpr (requires { this->_rows; }) {
        this->_rows = kMajorDirection == Direction::RowMajor ? 1 : s;
      }
      if constexpr (requires { this->_cols; }) {
        this->_cols = kMajorDirection == Direction::RowMajor ? s : 1;
      }
      if constexpr (requires { this->_ld; }) {
        this->_ld = s;
      }
      AllocateValues();
    }
  }

  MOCHI_ANY MOCHI_FORCE_INLINE constexpr auto CERows() const {
    if constexpr (requires { this->_rows; }) {
      return mochi::details::IntOrEmpty<kRowsAtCompileTime>{this->_rows};
    } else {
      return mochi::details::IntOrEmpty<kRowsAtCompileTime>{};
    }
  }

  MOCHI_ANY MOCHI_FORCE_INLINE constexpr auto CECols() const {
    if constexpr (requires { this->_cols; }) {
      return mochi::details::IntOrEmpty<kColsAtCompileTime>{this->_cols};
    } else {
      return mochi::details::IntOrEmpty<kColsAtCompileTime>{};
    }
  }

  MOCHI_ANY MOCHI_FORCE_INLINE constexpr auto CELDim() const {
    if constexpr (requires { this->_ld; }) {
      return mochi::details::IntOrEmpty<kLeadingDim>{this->_ld};
    } else {
      return mochi::details::IntOrEmpty<kLeadingDim>{};
    }
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr int Rows() const {
    return this->CERows().iVal();
  }
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr int Cols() const {
    return this->CECols().iVal();
  }
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr int LeadDim() const {
    if constexpr (kLeadingDim > 0) {
      return kLeadingDim;
    } else if constexpr (kLeadingDim == kDynamic) {
      return this->CELDim().iVal();
    } else {
      if constexpr (kMajorDir == Direction::ColMajor) {
        return this->Rows();
      } else {
        return this->Cols();
      }
    }
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr auto RowStride() const {
    if constexpr (kMajorDir == Direction::ColMajor) {
      return mochi::details::IntOrEmpty<1>{};
    } else {
      if constexpr (kLeadingDim > 0 || kLeadingDim == kDynamic) {
        return this->CELDim();
      } else {
        return mochi::details::IntOrEmpty<kColsAtCompileTime>{Cols()};
      }
    }
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr auto ColStride() const {
    if constexpr (kMajorDir == Direction::RowMajor) {
      return mochi::details::IntOrEmpty<1>{};
    } else {
      if constexpr (kLeadingDim > 0 || kLeadingDim == kDynamic) {
        return this->CELDim();
      } else {
        return mochi::details::IntOrEmpty<kRowsAtCompileTime>{Rows()};
      }
    }
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr size_t StorageSize() const {
    MOCHI_ASSERT_VERBOSE(this->LeadDim() >= 0, "Leading dimension should be positive");
    // Warning: Both casts are necessary to avoid overflow.
    // Do not replace with a cast of the product.
    if constexpr (kMajorDir == Direction::RowMajor) {
      MOCHI_ASSERT_VERBOSE(this->Rows() >= 0, "Number of rows should be positive");
      return static_cast<size_t>(this->Rows()) * static_cast<size_t>(this->LeadDim());
    } else {
      MOCHI_ASSERT_VERBOSE(this->Cols() >= 0, "Number of columns should be positive");
      return static_cast<size_t>(this->LeadDim()) * static_cast<size_t>(this->Cols());
    }
  }

  /// @brief Get a pointer to the dynamic memory allocator (if any)
  MOCHI_FORCE_INLINE Allocator* GetAllocator() const {
    if constexpr (kIsDynamic && kIsOwner && !kIsCuda) {
      return _values.GetAllocator();
    } else {
      return nullptr;
    }
  }

 protected:
  /// @brief Default constructor
  MOCHI_ANY constexpr BaseMatrix()
    requires(kIsView || IsNonConst<Scalar>) && (kLeadingDim != kDynamic)
      : _values{} {}

  /// @brief Constructor for a matrix of dimension nr x nc
  /// @param[in] nr Number of rows
  /// @param[in] nc Number of columns
  MOCHI_ANY constexpr BaseMatrix(int nr, int nc)
    requires kIsOwner && (kLeadingDim != kDynamic) && IsNonConst<Scalar>
      : Sizes(nr, nc), _values() {
    if constexpr (!kNumRowsIsDynamic) {
      MOCHI_ASSERT_VERBOSE(nr == kRowsAtCompileTime, "Inconsistent compile-time number of rows");
    }
    //
    if constexpr (!kNumColsIsDynamic) {
      MOCHI_ASSERT_VERBOSE(nc == kColsAtCompileTime, "Inconsistent compile-time number of columns");
    }
    AllocateValues();
  }

  /// @brief Constructor for a matrix of dimension nr x nc
  /// @param[in] nr Number of rows
  /// @param[in] nc Number of columns
  /// @param[in] allocator Pointer for the custom memory resource
  /// @note When the allocator is nullptr, the default resource allocator is used.
  /// @note This constructor is also enabled for fixed-size matrices to allow writing generic
  /// templated code. In that case, the allocator is unused.
  MOCHI_ANY BaseMatrix(int nr, int nc, Allocator* allocator)
    requires kIsOwner && IsNonConst<Scalar> && (kLeadingDim != kDynamic)
      : Sizes(nr, nc), _values(details::GetBaseStorage<Scalar, kSize, kOwnership>(allocator)) {
    if constexpr (!kNumRowsIsDynamic) {
      MOCHI_ASSERT_VERBOSE(nr == kRowsAtCompileTime, "Inconsistent compile-time number of rows");
    }
    if constexpr (!kNumColsIsDynamic) {
      MOCHI_ASSERT_VERBOSE(nc == kColsAtCompileTime, "Inconsistent compile-time number of columns");
    }
    AllocateValues();
  }

  /// @brief Constructor for a matrix of dimension nr x nc and a leading dimension
  /// @param[in] nr Number of rows
  /// @param[in] nc Number of columns
  /// @param[in] leadDim Leading dimension
  MOCHI_ANY BaseMatrix(int nr, int nc, int leadDim)
    requires kIsOwner && IsNonConst<Scalar>
      : Sizes{nr, nc, leadDim}, _values{} {
    MOCHI_ASSERT_VERBOSE(leadDim >= 0, "Leading dimension can not be negative");
    if constexpr (!kNumRowsIsDynamic) {
      MOCHI_ASSERT_VERBOSE(nr == kRowsAtCompileTime, "Inconsistent compile-time number of rows");
    }
    if constexpr (!kNumColsIsDynamic) {
      MOCHI_ASSERT_VERBOSE(nc == kColsAtCompileTime, "Inconsistent compile-time number of columns");
    }
    if constexpr (kLeadingDim == kAutomaticLeadDim || !kIsDynamic) {
      MOCHI_ASSERT_VERBOSE(
          leadDim == (kIsColMajor ? nr : nc),
          "Inconsistent leading dimension and size along the storage direction.");
    } else {
      MOCHI_ASSERT_VERBOSE(
          leadDim >= (kIsColMajor ? nr : nc),
          "Inconsistent leading dimension and size along the storage direction.");
    }
    AllocateValues();
  }

  /// @brief Constructor for a matrix of dimension nr x nc and a leading dimension
  /// @param[in] nr Number of rows
  /// @param[in] nc Number of columns
  /// @param[in] leadDim Leading dimension
  /// @param[in] allocator Pointer for the custom memory resource
  /// @note This constructor is also enabled for fixed-size matrices to allow writing generic
  /// templated code. In that case, the allocator is unused.
  MOCHI_ANY BaseMatrix(int nr, int nc, int leadDim, Allocator* allocator)
    requires kIsOwner && IsNonConst<Scalar>
      : Sizes{nr, nc, leadDim},
        _values(details::GetBaseStorage<Scalar, kSize, kOwnership>(allocator)) {
    MOCHI_ASSERT_VERBOSE(leadDim >= 0, "Leading dimension can not be negative");
    if constexpr (!kNumRowsIsDynamic) {
      MOCHI_ASSERT_VERBOSE(nr == kRowsAtCompileTime, "Inconsistent compile-time number of rows");
    }
    if constexpr (!kNumColsIsDynamic) {
      MOCHI_ASSERT_VERBOSE(nc == kColsAtCompileTime, "Inconsistent compile-time number of columns");
    }
    if constexpr (kLeadingDim == kAutomaticLeadDim || !kIsDynamic) {
      MOCHI_ASSERT_VERBOSE(
          leadDim == (kIsColMajor ? nr : nc),
          "Inconsistent leading dimension and size along the storage direction.");
    } else {
      MOCHI_ASSERT_VERBOSE(
          leadDim >= (kIsColMajor ? nr : nc),
          "Inconsistent leading dimension and size along the storage direction.");
    }
    AllocateValues();
  }

  MOCHI_WARNING_PUSH()
  MOCHI_WARNING_IGNORE_GCC_CLANG(GCC diagnostic ignored "-Wmissing-braces")

  /// @brief Constructor for doing a list initialization
  MOCHI_ANY constexpr BaseMatrix(Scalar v0, std::same_as<Scalar> auto... v)
    requires kIsOwner && (kRowsAtCompileTime >= 0) && (kColsAtCompileTime >= 0)
      : _values{v0, v...} {
    static_assert(
        sizeof...(v) + 1 <= kRowsAtCompileTime * kColsAtCompileTime, "Too many entries specified");
    static_assert(
        sizeof...(v) + 1 >= kRowsAtCompileTime * kColsAtCompileTime,
        "Not enough entries specified");
  }

  MOCHI_WARNING_POP()

  /// @brief Constructor for list initialization.
  /// @code
  ///   |1|       ColumnVector<real> col = {{1,2,3}};
  ///   |2|
  ///   |3]
  ///
  ///   |123]     RowVector<real> rv = {{1,2,3}};
  ///
  ///   |147|     Matrix<real> mat = {{1,2,3}, {4,5,6}, {7,8,9}};
  ///   |258|
  ///   |369|
  ///
  ///   |123|     RowMatrix<real> mat = {{1,2,3}, {4,5,6}, {7,8,9}};
  ///   |456|
  ///   |789|
  /// @endcode
  MOCHI_ANY constexpr BaseMatrix(std::initializer_list<std::initializer_list<Scalar>> const& list)
    requires kIsOwner && (!kIsCuda)
      : BaseMatrix(
            kIsColMajor ? isize(*list.begin()) : isize(list), // rows
            kIsColMajor ? isize(list) : isize(*list.begin()) // cols
        ) {
    if constexpr (kIsColMajor) {
      int iCol = 0;
      for (auto& column : list) {
        MOCHI_ASSERT(column.size() == this->Rows(), "Inconsistent number of rows");
        std::copy(column.begin(), column.end(), _values.Data() + iCol * this->LeadDim());
        iCol += 1;
      }
    } else {
      int iRow = 0;
      for (auto& row : list) {
        MOCHI_ASSERT(row.size() == this->Cols(), "Inconsistent number of columns");
        std::copy(row.begin(), row.end(), _values.Data() + iRow * this->LeadDim());
        iRow += 1;
      }
    }
  }

  MOCHI_ANY explicit BaseMatrix(Scalar* s)
    requires kIsView && (!kIsDynamic) && (kLeadingDim != kDynamic)
      : _values{s} {
    MOCHI_ASSERT_VERBOSE((s != nullptr) || this->empty(), "Data pointer cannot be null");
  }

  MOCHI_ANY BaseMatrix(Scalar* s, int nr, int nc)
    requires kIsView && (kLeadingDim != kDynamic)
      : Sizes{nr, nc}, _values{s} {
    MOCHI_ASSERT_VERBOSE((s != nullptr) || this->empty(), "Data pointer cannot be null");
    if constexpr (!kNumRowsIsDynamic) {
      MOCHI_ASSERT_VERBOSE(nr == kRowsAtCompileTime, "Inconsistent compile-time number of rows");
    }
    //
    if constexpr (!kNumColsIsDynamic) {
      MOCHI_ASSERT_VERBOSE(nc == kColsAtCompileTime, "Inconsistent compile-time number of columns");
    }
  }

  MOCHI_ANY BaseMatrix(Scalar* s, int nr, int nc, int leadDim)
    requires kIsView
      : Sizes{nr, nc, leadDim}, _values{s} {
    MOCHI_ASSERT_VERBOSE((s != nullptr) || this->empty(), "Data pointer cannot be null");
    if constexpr (kIsRowMajor) {
      MOCHI_ASSERT_VERBOSE(
          leadDim >= nc,
          "Inconsistent leading dimension (%d) with number of columns (%d)",
          leadDim,
          nc);
    } else {
      MOCHI_ASSERT_VERBOSE(
          leadDim >= nr,
          "Inconsistent leading dimension (%d) with number of rows (%d)",
          leadDim,
          nr);
    }
  }

  /// @brief Copy constructor
  /// @param bm BaseMatrix to copy
  MOCHI_ANY constexpr BaseMatrix(BaseMatrix const& bm);

  /// @brief Move constructor
  MOCHI_ANY constexpr BaseMatrix(BaseMatrix&& b) noexcept = default;

  /// @brief Move constructor with allocator. If the allocators are compatible, then memory will be
  /// moved. Else, new memory will be allocated and the values will be copied.
  /// @note This constructor is also enabled for fixed-size matrices to allow writing generic
  /// templated code. In that case, the allocator is unused.
  template <typename RHS>
    requires(std::is_same_v<RHS, BaseMatrix> && kIsOwner && !kIsCuda)
  MOCHI_ANY constexpr BaseMatrix(RHS&& b, Allocator* allocator)
      : Sizes(b),
        _values(
            details::MoveBaseStorage<Scalar, kSize, kOwnership>(std::move(b._values), allocator)) {}

  /// @brief Allocate memory
  constexpr void AllocateValues();

  //
  // Variables
  //

 protected:
  static constexpr int kSize =
      kIsDynamic ? krylov::kDynamic : kRowsAtCompileTime * kColsAtCompileTime;

  // TODO: Clang MSVC accepts neither [[no_unique_address]] nor [[msvc::no_unique_address]]. If
  // issue can be resolved, enable the properties below and deprecate Sizes.
  // MOCHI_NO_UNIQUE_ADDRESS mochi::details::IntOrEmpty<kRowsAtCompileTime> _rows{0};
  // MOCHI_NO_UNIQUE_ADDRESS mochi::details::IntOrEmpty<kColsAtCompileTime> _cols{0};
  // MOCHI_NO_UNIQUE_ADDRESS mochi::details::IntOrEmpty<kLeadingDim> _ld{0};
  details::BaseStorage<Scalar, kSize, kOwnership> _values;
};

static_assert(
    sizeof(BaseMatrix<real, 3, 3, Direction::ColMajor>) ==
    sizeof(details::BaseStorage<real, 9, Ownership::Owner>));

} // namespace mochi::krylov

#include <mochi_core/linear_algebra/cuda/cuda_api.h>
#include <mochi_core/linear_algebra/cuda/cuda_base_tools_inl.h>

//
// Member functions definition
//

namespace mochi::krylov::details {

/// @brief Copy data between two compatible storage
///
/// @param[in] src Storage where data is copied from
/// @param[in] len Size of data to copy
/// @param[out] dest Storage where data is copied to
///
/// @note The routine does not check that both storages are compatible with the length 'len'
///     The function should be used cautiously.
template <typename Scalar, int kSize, Ownership kOwnership>
constexpr void Copy(
    BaseStorage<Scalar, kSize, kOwnership> const& src,
    size_t len,
    BaseStorage<Scalar, kSize, kOwnership>& dest) {
  if constexpr (kOwnership == Ownership::Cuda) {
    if constexpr (
        std::is_same_v<Scalar const, float const> || std::is_same_v<Scalar const, double const>) {
      // CudaDeviceCopy uses cuBLAS routine
      // which allows to run a specific stream.
      mochi::details::CudaDeviceCopy(dest.data(), src.data(), len);
    } else {
      mochi::details::CudaCopy<Scalar>(dest.data(), src.data(), len);
    }
  } else if constexpr (kOwnership == Ownership::Owner) {
    std::copy(src.data(), src.data() + len, dest.data());
  } else {
    static_assert(
        (kOwnership == Ownership::Cuda) || (kOwnership == Ownership::Owner),
        "Copy is not implemented for 'Ownership::View' or 'Ownership::CudaView'");
  }
}

} // namespace mochi::krylov::details

namespace mochi::krylov {

template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    Direction kMajorDirection,
    Ownership kOwnership,
    int kLeadingDim>
constexpr BaseMatrix<
    Scalar,
    kRowsAtCompileTime,
    kColsAtCompileTime,
    kMajorDirection,
    kOwnership,
    kLeadingDim>::BaseMatrix(BaseMatrix const& bm)
    : Sizes(bm), _values{bm._values.GetSimilar(this->StorageSize())} {
  if constexpr (!kIsView) {
    auto const len = this->StorageSize();
    details::Copy(bm._values, len, _values);
  }
}

template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    Direction kMajorDirection,
    Ownership kOwnership,
    int kLeadingDim>
constexpr void BaseMatrix<
    Scalar,
    kRowsAtCompileTime,
    kColsAtCompileTime,
    kMajorDirection,
    kOwnership,
    kLeadingDim>::AllocateValues() {
  if constexpr (kIsDynamic) {
    _values.Resize(this->StorageSize());
  }
}

} // namespace mochi::krylov
