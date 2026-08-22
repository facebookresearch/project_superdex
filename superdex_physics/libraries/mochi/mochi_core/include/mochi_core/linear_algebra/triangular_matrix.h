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

namespace mochi::krylov {

/// @brief Upper triangular view of a matrix
template <typename T, Direction kDirection>
class UpperTriangularMatrixView {
 public:
  using Scalar = T;

  /// @brief Constructor for the upper triangular view
  /// @param[in] v Pointer to the memory storage
  /// @param[in] nr Number of rows<param name="v_"></param>
  /// @param[in] nc Number of columns
  /// @param[in] ld Leading dimension in the orientation (i.e. row-major or column-major)
  ///
  /// @remark The memory v should be of size nr * ld for a row-major matrix
  /// or of size ld * nc for a column-major matrix.
  /// @remark The input matrix can be rectangular.
  explicit UpperTriangularMatrixView(Scalar* v, int nr, int nc, int ld)
      : _nr(nr), _nc(nc), _ld(ld), _entries(v) {
    MOCHI_ASSERT_VERBOSE(
        (kDirection == Direction::RowMajor && _ld >= _nc) ||
        (kDirection == Direction::ColMajor && _ld >= _nr));
  }

  [[nodiscard]] inline int Rows() const {
    return _nr;
  }

  [[nodiscard]] inline int Cols() const {
    return _nc;
  }

  Scalar* Data() {
    return _entries;
  }

  Scalar const* Data() const {
    return _entries;
  }

  /// @brief Return raw pointer to the memory (overload for compatibility with std containers)
  Scalar* data() {
    return _entries;
  }

  /// @brief Return raw pointer to the memory (overload for compatibility with std containers)
  Scalar const* data() const {
    return _entries;
  }

  /// @brief Returns a reference to the value
  /// @param[in] r Row index (0-based)
  /// @param[in] c Column index (0-based)</param>
  /// @returns Reference to the value at (r, c)
  ///
  /// @remark A precondition is that this function is not called with r > c.
  Scalar& operator()(int r, int c) {
    MOCHI_ASSERT_VERBOSE((r >= 0) && (r < _nr) && (c >= 0) && (c < _nc), "Out of range indices");
    MOCHI_ASSERT_VERBOSE(c >= r, "Scalar reference is not available");
    return _entries[_GetOffset(r, c)];
  }

  /// @brief Returns the value
  /// @param[in] r Row index (0-based)
  /// @param[in] c Column index (0-based)</param>
  /// @returns Value at (r, c)
  ///
  /// @remark When r > c, the returned value is equal to 0 as the view is upper triangular.
  /// @remark A precondition is that this function is not called with r > c.
  Scalar operator()(int r, int c) const {
    MOCHI_ASSERT_VERBOSE((r >= 0) && (r < _nr) && (c >= 0) && (c < _nc), "Out of range indices");
    return _entries[_GetOffset(r, c)];
  }

  template <
      int kRowsAtCompileTimeX,
      int kColsAtCompileTimeX,
      Direction kMajorDirectionX,
      Ownership kOwnershipX,
      int kLeadDimX,
      int kRowsAtCompileTimeY,
      int kColsAtCompileTimeY,
      Direction kMajorDirectionY,
      Ownership kOwnershipY,
      int kLeadDimY>
  void Apply(
      Matrix<
          Scalar,
          kRowsAtCompileTimeX,
          kColsAtCompileTimeX,
          kMajorDirectionX,
          kOwnershipX,
          kLeadDimX> const& x,
      Matrix<
          Scalar,
          kRowsAtCompileTimeY,
          kColsAtCompileTimeY,
          kMajorDirectionY,
          kOwnershipY,
          kLeadDimY>& y) const {
    MOCHI_ASSERT_VERBOSE(
        (x.Rows() == _nc) && (y.Rows() == _nr) && (y.Cols() == x.Cols()),
        "Dimensions do not match");
    for (int kc = 0; kc < x.Cols(); ++kc) {
      for (int ir = 0; ir < _nr; ++ir) {
        Scalar result{};
        for (int jj = ir; jj < _nc; ++jj) {
          result += _entries[_GetOffset(ir, jj)] * x(jj, kc);
        }
        y(ir, kc) = result;
      }
    }
  }

  template <
      int kRowsAtCompileTimeX,
      int kColsAtCompileTimeX,
      Direction kMajorDirectionX,
      Ownership kOwnershipX,
      int kLeadDimX,
      typename YT,
      int kRowsAtCompileTimeY,
      int kColsAtCompileTimeY,
      Direction kMajorDirectionY,
      Ownership kOwnershipY,
      int kLeadDimY>
  void Solve(
      Matrix<
          Scalar,
          kRowsAtCompileTimeX,
          kColsAtCompileTimeX,
          kMajorDirectionX,
          kOwnershipX,
          kLeadDimX> const& x,
      Matrix<
          YT,
          kRowsAtCompileTimeY,
          kColsAtCompileTimeY,
          kMajorDirectionY,
          kOwnershipY,
          kLeadDimY>& y) const {
    MOCHI_ASSERT_VERBOSE(_nr == _nc, "Input matrix is not square");
    MOCHI_ASSERT_VERBOSE(
        (x.Rows() == _nr) && (y.Rows() == _nr) && (y.Cols() == x.Cols()),
        "Dimensions do not match");
    for (int kc = 0; kc < x.Cols(); ++kc) {
      for (int ir = _nr - 1; ir >= 0; --ir) {
        YT result{};
        for (int jj = ir + 1; jj < _nc; ++jj) {
          result += y(jj, kc) * _entries[_GetOffset(ir, jj)];
        }
        y(ir, kc) = (x(ir, kc) - result) / _entries[_GetOffset(ir, ir)];
      }
    }
  }

  template <
      typename InputT,
      int kRowsAtCompileTimeX,
      int kColsAtCompileTimeX,
      Direction kMajorDirectionX,
      Ownership kOwnershipX,
      int kLeadDimX>
  void SolveInPlace(
      Matrix<
          InputT,
          kRowsAtCompileTimeX,
          kColsAtCompileTimeX,
          kMajorDirectionX,
          kOwnershipX,
          kLeadDimX>& x) const {
    MOCHI_ASSERT_VERBOSE(_nr == _nc, "Input matrix is not square");
    MOCHI_ASSERT_VERBOSE(x.Rows() == _nc, "Dimensions do not match");
    for (int kc = 0; kc < x.Cols(); ++kc) {
      for (int ir = _nr - 1; ir >= 0; --ir) {
        InputT result{};
        for (int jj = ir + 1; jj < _nc; ++jj) {
          result += x(jj, kc) * _entries[_GetOffset(ir, jj)];
        }
        x(ir, kc) = (x(ir, kc) - result) / _entries[_GetOffset(ir, ir)];
      }
    }
  }

 protected:
  int _nr = 0, _nc = 0, _ld = 1;
  Scalar* _entries;

  /// @brief Returns the entry position in the array
  ///
  /// @param r Row index
  /// @param c Column index
  /// @return Shift
  inline size_t _GetOffset(int r, int c) const {
    auto const rs = static_cast<size_t>(r);
    auto const cs = static_cast<size_t>(c);
    auto const lds = static_cast<size_t>(_ld);
    if constexpr (kDirection == Direction::RowMajor) {
      return rs * lds + cs;
    } else {
      return rs + cs * lds;
    }
  }
};

template <IsMatrix M>
auto UpperTriangularView(M const& mat) {
  constexpr auto const kResDir = details::MatTraits<M>::kMajorDir;
  return UpperTriangularMatrixView<typename M::Scalar const, kResDir>{
      mat.Data(), mat.Rows(), mat.Cols(), mat.LeadDim()};
}

} // namespace mochi::krylov
