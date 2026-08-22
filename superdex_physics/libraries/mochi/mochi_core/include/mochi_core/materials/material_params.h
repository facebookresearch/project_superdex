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

// PLEASE DO NOT ADD OTHER INCLUDES HERE. This header is included in the mochi_physics public API.
#include <mochi_core/materials/active_neo_hookean_params.h>
#include <mochi_core/materials/active_shape_targeting_arap_params.h>
#include <mochi_core/materials/arap_params.h>
#include <mochi_core/materials/linear_elastic_params.h>
#include <mochi_core/materials/material_types.h>
#include <mochi_core/materials/smith_neo_hookean_params.h>
#include <mochi_core/materials/st_venant_kirchhoff_params.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/span.h>

namespace mochi {

// Public API alias.
using NeoHookeanMaterialParams = SmithNeoHookeanMaterialParams;

/** @brief Default material density [kg/m³]. */
inline constexpr real kDefaultDensity = 1000_r;

/**
 * @brief Material parameters for soft body simulation.
 *
 * @details Contains the material type discriminant, one instance of each supported material model's
 * parameters, and shared properties (density). The @ref type field selects which material model's
 * parameters are active.
 *
 * @note Mochi uses the International System of Units (SI) by default. Other units can be used, but
 * require overwriting all dimensional default parameters (material, contact, constraints, solver,
 * etc.) to ensure consistency.
 *
 * @see SoftMaterialType
 */
struct SoftMaterialParams {
  /** @brief Material constitutive model. */
  SoftMaterialType type = SoftMaterialType::NeoHookean;

  /** @brief Parameters for the Neo-Hookean material model. */
  NeoHookeanMaterialParams neoHookean = {};

  /** @brief Parameters for the St. Venant-Kirchhoff material model. */
  StVenantKirchhoffMaterialParams stVenantKirchhoff = {};

  /** @brief Parameters for the Linear Elastic material model. */
  LinearElasticMaterialParams linearElastic = {};

  /** @brief Parameters for the Active Neo-Hookean material model. */
  ActiveNeoHookeanMaterialParams activeNeoHookean = {};

  /** @brief Parameters for the Active Shape Targeting ARAP material model. */
  ActiveShapeTargetingArapMaterialParams activeShapeTargetingArap = {};

  /** @brief Parameters for the ARAP material model. */
  ArapMaterialParams arap = {};

  /** @brief Material density in the undeformed configuration [kg/m³]. Must be positive. */
  real density = kDefaultDensity;

  /**
   * @brief Mass damping coefficient [1/s]. Applies a velocity-proportional force `α·M·v`.
   *
   * @details Must be non-negative. Active only when the actor has inertia. A nonzero value with no
   * inertia is valid but inactive (logged as a warning at actor initialization).
   */
  real massDampingCoefficient = 0_r;

  /**
   * @brief Stiffness damping coefficient `β` [s]. Adds a strain-rate-proportional viscous stress.
   *
   * @details Must be non-negative. Active only when the actor has stress. A nonzero value with no
   * stress is valid but inactive (logged as a warning at actor initialization).
   *
   * @details This is a total-Lagrangian variant of Kelvin–Voigt damping: it adds a viscous second
   * Piola–Kirchhoff stress
   * @code
   *   S_visc = β · C₀ : Ė
   * @endcode
   * where `Ė` is the material time derivative of the Green–Lagrange strain, and `C₀ = ∂²Ψ/∂E²` is
   * the Lagrangian material stiffness evaluated at zero deformation.
   *
   * @note While the parameterization in terms of a timescale β is similar, this differs from
   * standard Rayleigh damping using the global assembled stiffness matrix; it depends only on
   * strain rate, therefore dissipating energy only during deformation.
   */
  real stiffnessDampingCoefficient = 0_r;

  /**
   * @brief [Experimental] Include the geometric term in the stiffness-damping
   * tangent. Defaults to `false`.
   *
   * @details Only affects the Jacobian for Newton iteration, not the energy, residual, or converged
   * solution. When `false` (default), the viscous tangent uses a cheaper quasi-Newton
   * approximation. When `true`, the Newton Jacobian includes a geometric term, which is
   * proportional to the per-stage strain increment and may improve nonlinear solver convergence
   * in certain scenarios.
   *
   * @warning This is an experimental feature. It may be changed or removed in the future. Use at
   * your own risk.
   */
  bool stiffnessDampingIncludeGeometricTerm = false;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(SoftMaterialParams const&) const = default;
#endif

  // clang-format off
  MOCHI_STRUCT_BEGIN(mochi::SoftMaterialParams)
  MOCHI_FIELD(type)
  MOCHI_FIELD(neoHookean) MOCHI_ATTRIBUTE(PreviouslyKnownAs("stableNeoHookean"));
  MOCHI_FIELD(stVenantKirchhoff)
  MOCHI_FIELD(linearElastic)
  MOCHI_FIELD(activeNeoHookean) MOCHI_ATTRIBUTE(PreviouslyKnownAs("activeStableNeoHookean"));
  MOCHI_FIELD(activeShapeTargetingArap)
  MOCHI_FIELD(arap)
  MOCHI_FIELD(density) MOCHI_ATTRIBUTE(Units("kg/m^3"))
  MOCHI_FIELD(massDampingCoefficient) MOCHI_ATTRIBUTE(Units("1/s"))
  MOCHI_FIELD(stiffnessDampingCoefficient) MOCHI_ATTRIBUTE(Units("s"))
  MOCHI_FIELD(stiffnessDampingIncludeGeometricTerm)
  MOCHI_STRUCT_END()
  // clang-format on
};

// Forward
struct PerElementSoftMaterialDataView;

/**
 * @brief Per-element material parameters for a soft body in struct-of-arrays format.
 *
 * @see SoftMaterialParams, PerElementSoftMaterialDataView
 */
struct PerElementSoftMaterialData {
  PerElementSoftMaterialData() = default;

  /**
   * @brief Copy from @ref PerElementSoftMaterialDataView.
   *
   * @param[in] other Source data.
   */
  PerElementSoftMaterialData(PerElementSoftMaterialDataView const& other);

  /** @brief Material constitutive model. */
  SoftMaterialType type = SoftMaterialType::NeoHookean;

  /** @brief Strategy for ensuring positive semi-definite Hessian matrices. */
  MaterialPsdStrategy psdStrategy = MaterialPsdStrategy::MaterialDefault;
  DynamicArray<real> youngsModulus; ///< Young's modulus [Pa]. 1 per element.
  DynamicArray<real> poissonRatio; ///< Poisson's ratio (dimensionless). 1 per element.
  DynamicArray<real> anisoAlpha; ///< Anisotropic stiffness (α) [Pa]. 1 per element.
  DynamicArray<real> anisoLength; ///< Anisotropic reference length (dimensionless). 1 per element.
  DynamicArray<real> anisoTheta; ///< Fiber azimuthal angle (θ) [rad]. 1 per element.
  DynamicArray<real> anisoPhi; ///< Fiber elevation angle (φ) [rad]. 1 per element.
  DynamicArray<real> arapStiffness; ///< ARAP stiffness (μ) [Pa]. 1 per element.
  /// Shape target tensor offset in flat symmetric upper-triangle layout. 6 values per element.
  /// Defines S_t = I + [[s0, s1, s2], [s1, s3, s4], [s2, s4, s5]].
  DynamicArray<real> shapeTargetTensor;

  /**
   * @brief Number of elements described by this per-element material field.
   *
   * @details Derived from the per-element array appropriate to @ref type "type": @ref
   * arapStiffness for the ARAP-family materials (@ref SoftMaterialType::Arap and @ref
   * SoftMaterialType::ActiveShapeTargetingArap), and @ref poissonRatio for the Lame-based
   * materials.
   *
   * @return Number of elements.
   */
  [[nodiscard]] int GetNumElements() const;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(PerElementSoftMaterialData const& other) const = default;
  bool operator!=(PerElementSoftMaterialData const& other) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::PerElementSoftMaterialData)
  MOCHI_FIELD(type)
  MOCHI_FIELD(psdStrategy)
  MOCHI_FIELD(youngsModulus) MOCHI_ATTRIBUTE(Units("Pa"));
  MOCHI_FIELD(poissonRatio)
  MOCHI_FIELD(anisoAlpha) MOCHI_ATTRIBUTE(Units("Pa"));
  MOCHI_FIELD(anisoLength)
  MOCHI_FIELD(anisoTheta) MOCHI_ATTRIBUTE(Units("rad"));
  MOCHI_FIELD(anisoPhi) MOCHI_ATTRIBUTE(Units("rad"));
  MOCHI_FIELD(arapStiffness) MOCHI_ATTRIBUTE(Units("Pa"));
  MOCHI_FIELD(shapeTargetTensor)
  MOCHI_STRUCT_END()
};

/**
 * @brief Non-owning view of per-element material parameters for a soft body, in struct-of-arrays
 * format.
 *
 * @see SoftMaterialParams, PerElementSoftMaterialData
 */
struct PerElementSoftMaterialDataView {
  PerElementSoftMaterialDataView() = default;

  /**
   * @brief Implicit conversion from @ref PerElementSoftMaterialData.
   *
   * @param[in] other Source data.
   */
  PerElementSoftMaterialDataView(PerElementSoftMaterialData const& other);

  /** @brief Material constitutive model. */
  SoftMaterialType type = SoftMaterialType::NeoHookean;

  /** @brief Strategy for ensuring positive semi-definite Hessian matrices. */
  MaterialPsdStrategy psdStrategy = MaterialPsdStrategy::MaterialDefault;
  Span<real const> youngsModulus; ///< Young's modulus [Pa]. 1 per element.
  Span<real const> poissonRatio; ///< Poisson's ratio (dimensionless). 1 per element.
  Span<real const> anisoAlpha; ///< Anisotropic stiffness (α) [Pa]. 1 per element.
  Span<real const> anisoLength; ///< Anisotropic reference length (dimensionless). 1 per element.
  Span<real const> anisoTheta; ///< Fiber azimuthal angle (θ) [rad]. 1 per element.
  Span<real const> anisoPhi; ///< Fiber elevation angle (φ) [rad]. 1 per element.
  Span<real const> arapStiffness; ///< ARAP stiffness (μ) [Pa]. 1 per element.
  /// Shape target tensor offset in flat symmetric upper-triangle layout. 6 values per element.
  /// Defines S_t = I + [[s0, s1, s2], [s1, s3, s4], [s2, s4, s5]].
  Span<real const> shapeTargetTensor;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(PerElementSoftMaterialDataView const& other) const = default;
  bool operator!=(PerElementSoftMaterialDataView const& other) const = default;
#endif
};

} // namespace mochi

#include "material_params_inl.h"
