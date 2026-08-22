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

#include <mochi_core/linear_algebra/any_matrix.h>
#include <mochi_core/linear_algebra/utils/matrix_concepts.h>

namespace mochi {

/// @brief Cast-copy AnyMatrix with scalar entries to AnyMatrix with a different scalar
/// type.
///
/// @tparam ToAnyMatrix The target type for the conversion (must be an AnyMatrix instantiation).
/// @tparam Scalar The source scalar type from the input AnyMatrix object.
/// @param anyMat The input AnyMatrix to be converted.
/// @return ToAnyMatrix with the converted scalar type.
template <typename ToAnyMatrix, typename Scalar, MOCHI_CONCEPT(IsAnyMatrixVariant<ToAnyMatrix>)>
[[nodiscard]] ToAnyMatrix StaticCast(AnyMatrix<Scalar> const& anyMat);

/// @brief Cast-copy AnyMatrixView with scalar entries to AnyMatrix with a different scalar
/// type.
///
/// @tparam ToAnyMatrix The target type for the conversion (must be an AnyMatrix instantiation).
/// @tparam Scalar The source scalar type from the input AnyMatrixView object.
/// @param anyMat The input AnyMatrixView to be converted.
/// @return ToAnyMatrix with the converted scalar type.
template <typename ToAnyMatrix, typename Scalar, MOCHI_CONCEPT(IsAnyMatrixVariant<ToAnyMatrix>)>
[[nodiscard]] ToAnyMatrix StaticCast(AnyMatrixView<Scalar> const& anyMat);

} // namespace mochi

#include "any_matrix_utils_inl.h"
