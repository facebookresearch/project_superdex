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

#include <mochi_core/materials/material_params.h>
#include <mochi_core/utils/error.h>

#include <limits>

namespace mochi {

/**
 * @brief Validate @ref SoftMaterialParams for the active material model.
 *
 * @details Validates @ref SoftMaterialParams::type, @ref SoftMaterialParams::density, and the
 * params sub-struct corresponding to @ref SoftMaterialParams::type. Other sub-structs are ignored.
 *
 * @param[in] params Material parameters to check.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
void ValidateSoftMaterialParams(SoftMaterialParams const& params, Error& error);

/**
 * @brief Validate @ref PerElementSoftMaterialDataView for the active material model.
 *
 * @details Validates only the arrays used by @ref PerElementSoftMaterialDataView::type. Used arrays
 * must have size @p numElements, except @ref PerElementSoftMaterialDataView::shapeTargetTensor
 * which must have size 6x @p numElements.
 *
 * @param[in] data Per-element material parameters to check.
 * @param[in] numElements Number of elements in the soft actor to which this material will be
 * applied.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 */
void ValidateSoftMaterialParams(
    PerElementSoftMaterialDataView const& data,
    int numElements,
    Error& error);

} // namespace mochi

namespace mochi::materials {

/**
 * @brief Minimum eigenvalue (ε) used when projecting the Hessian to be positive semi-definite.
 *
 * @note Applies to any deformable element Hessian projection, not only material response — e.g.
 * the geometric tangents of codimensional elements (shells, rods).
 *
 * @see MaterialPsdStrategy::Projection, MaterialPsdStrategy::AbsEigenProjection
 */
inline constexpr real kMinProjectedEigenvalue = std::numeric_limits<real>::epsilon();

} // namespace mochi::materials
