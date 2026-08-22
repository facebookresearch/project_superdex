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
#include <mochi_core/utils/reflection.h>

#if MOCHI_USE_REFLECTION

namespace mochi {

/**
 * @brief Implements SReflect::MatrixTypeInfo to expose our matrix type to reflection (for
 * serialization, tools, etc...)
 */
template <class MatrixT>
class MatrixTypeInfo final : public SReflect::MatrixTypeInfo {
 public:
  using Layout = SReflect::MatrixTypeInfo::Layout;

  [[nodiscard]] void* GetData(void* obj) const final {
    return static_cast<MatrixT*>(obj)->data();
  };

  [[nodiscard]] bool IsMemCopySafe() const final {
    constexpr bool kIsMemCopySafe = MatrixT::kIsOwner && !MatrixT::kIsDynamic && !MatrixT::kIsCuda;
    return kIsMemCopySafe;
  }

  /// @brief Resize the matrix if possible. Else return false.
  [[nodiscard]] bool TryResize(void* obj, size_t numRows, size_t numColumns) const final {
    auto* mat = static_cast<MatrixT*>(obj);
    bool hasCorrectRows = (mat->Rows() == numRows);
    bool hasCorrectCols = (mat->Cols() == numColumns);
    if constexpr (MatrixT::kIsDynamic && MatrixT::kIsOwner) {
      if ((MatrixT::kNumRowsIsDynamic || hasCorrectRows) &&
          (MatrixT::kNumColsIsDynamic || hasCorrectCols)) {
        mat->Resize(static_cast<int>(numRows), static_cast<int>(numColumns));
        return true;
      }
    }
    return hasCorrectRows && hasCorrectCols;
  }

 private:
  Layout GetLayoutImpl(void const* obj) const final {
    auto* mat = static_cast<MatrixT const*>(obj);
    return Layout{size_t(mat->Rows()), size_t(mat->Cols()), size_t(mat->LeadDim())};
  }
};

namespace details {

// Implemented in the cpp to reduce bloat
void InitMatrixTypeInfo(
    size_t size,
    size_t alignment,
    SReflect::TypeInfo const& innerTypeInfo,
    int rowsAtCompileTime,
    int colsAtCompileTime,
    mochi::krylov::Direction majorDirection,
    mochi::krylov::Ownership ownership,
    int leadingDim,
    SReflect::MatrixTypeInfo* outTypeInfo);

} // namespace details
} // namespace mochi

/**  @brief Specialization of SReflectTypeTraits for mochi::Matrix */
template <
    typename Scalar,
    int kRowsAtCompileTime,
    int kColsAtCompileTime,
    mochi::krylov::Direction kMajorDirection,
    mochi::krylov::Ownership kOwnership,
    int kLeadingDim>
struct SReflectTypeTraits<mochi::Matrix<
    Scalar,
    kRowsAtCompileTime,
    kColsAtCompileTime,
    kMajorDirection,
    kOwnership,
    kLeadingDim>> {
  using MatrixT = mochi::Matrix<
      Scalar,
      kRowsAtCompileTime,
      kColsAtCompileTime,
      kMajorDirection,
      kOwnership,
      kLeadingDim>;
  static constexpr SReflect::CoreType coreType = SReflect::CoreType::CT_matrix;
  static SReflect::MatrixTypeInfo const& GetTypeInfo() {
    static auto const* typeInfo = []() {
      auto* ti = new mochi::MatrixTypeInfo<MatrixT>;
      mochi::details::InitMatrixTypeInfo(
          sizeof(MatrixT),
          alignof(MatrixT),
          SReflect::GetTypeInfo<Scalar>(),
          kRowsAtCompileTime,
          kColsAtCompileTime,
          kMajorDirection,
          kOwnership,
          kLeadingDim,
          ti);
      // Only owning matrices can be factor created/cloned
      if constexpr (mochi::krylov::IsOwner(kOwnership)) {
        ti->_constructInPlace = [](void* dst) { new (dst) MatrixT{}; };
        ti->_constructInPlaceByCopy = [](void* dst, void const* src) {
          new (dst) MatrixT{*static_cast<MatrixT const*>(src)};
        };
        ti->_destructInPlace = [](void* p) { static_cast<MatrixT*>(p)->~MatrixT(); };
      }
      return ti;
    }();
    return *typeInfo;
  }
};

#endif // MOCHI_USE_REFLECTION
