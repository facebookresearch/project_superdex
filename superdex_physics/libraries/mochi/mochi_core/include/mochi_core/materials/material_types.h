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
#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/reflection.h>

namespace mochi {
// Default material property values.

/** @brief Default Young's modulus [Pa]. */
inline constexpr real kDefaultYoungsModulus = 1e5_r;

/** @brief Default Poisson's ratio (dimensionless). */
inline constexpr real kDefaultPoissonRatio = 0.45_r;

/** @brief Default ARAP stiffness (μ) [Pa]. */
inline constexpr real kDefaultArapStiffness = 1000_r;

/** @brief Default anisotropic stiffness (α) [Pa]. */
inline constexpr real kDefaultAnisoAlpha = 1000_r;

/** @brief Default anisotropic reference length (dimensionless). */
inline constexpr real kDefaultAnisoLength = 1_r;

/**
 * @brief Strategy for ensuring positive semi-definite (PSD) Hessian matrices in material models.
 *
 * @details The material Hessian (∂P/∂F, where P is the first Piola-Kirchhoff stress and F is the
 * deformation gradient) should be PSD for stable simulation. This enum defines strategies to
 * enforce this property, balancing computational cost against stability and convergence quality.
 *
 * @note Different materials support different strategies. Check material-specific documentation.
 *
 * @see SoftMaterialType
 */
enum struct MaterialPsdStrategy {
  /**
   * @brief Use the material's default PSD strategy.
   *
   * @note Supported for all material models.
   * @note Each material model defines its own recommended default strategy.
   */
  MaterialDefault,

  /**
   * @brief No PSD enforcement. Negative eigenvalues may arise.
   *
   * @details Fastest strategy but may substantially degrade stability of most material models.
   *
   * @note Supported for all material models.
   */
  None,

  /**
   * @brief Eigenvalue clamping: U * Max(Λ, ε) * U^T.
   *
   * @details Performs eigendecomposition ∂P/∂F = U * Λ * U^T and clamps negative eigenvalues to a
   * small positive constant ε. Slower than most other methods, but typically ensures best
   * convergence.
   *
   * @note Supported for all material models.
   *
   * @see materials::kMinProjectedEigenvalue
   */
  Projection,

  /**
   * @brief Fast PSD enforcement by dropping problematic terms.
   *
   * @details Faster than @ref MaterialPsdStrategy::Projection but may degrade non-linear solver
   * convergence.
   *
   * @note Only supported for @ref SoftMaterialType::StVenantKirchhoff and @ref
   * SoftMaterialType::NeoHookean materials.
   *
   * @warning For @ref SoftMaterialType::StVenantKirchhoff, does not guarantee a PSD Hessian when
   * the Poisson ratio is negative (auxetic materials). Use @ref MaterialPsdStrategy::Projection or
   * @ref MaterialPsdStrategy::AbsEigenProjection instead.
   */
  Fast,

  /**
   * @brief Absolute eigenvalue projection: U * Max(|Λ|, ε) * U^T.
   *
   * @details Takes absolute value of eigenvalues instead of clamping. Similar cost to @ref
   * MaterialPsdStrategy::Projection but may improve stability, especially for highly deformed
   * configurations.
   *
   * @note Supported for all material models.
   *
   * @see [Stabler Neo-Hookean Simulation: Absolute Eigenvalue Filtering for Projected Newton (Chen
   * et al., 2024)](https://www.cs.columbia.edu/cg/abs-psd/paper.pdf)
   * @see materials::kMinProjectedEigenvalue
   */
  AbsEigenProjection,

  /**
   * @brief Per-term projection for composite Hessians.
   *
   * @details Projects each summand of the Hessian separately. For example, if H = H1 + H2, projects
   * both H1 and H2 independently.
   *
   * @note Only supported for @ref SoftMaterialType::ActiveShapeTargetingArap material.
   */
  PerTermProjection,

  /** @brief Number of unique enum values. */
  Count,
};
} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::MaterialPsdStrategy)
MOCHI_ENUM_ITEM(MaterialDefault)
MOCHI_ENUM_ITEM(None)
MOCHI_ENUM_ITEM(Projection)
MOCHI_ENUM_ITEM(Fast)
MOCHI_ENUM_ITEM(AbsEigenProjection)
MOCHI_ENUM_ITEM(PerTermProjection)
MOCHI_ENUM_COUNT(Count)
MOCHI_ENUM_END()

namespace mochi {
/**
 * @brief Material constitutive models for soft bodies.
 *
 * @details Defines the constitutive response for deformable materials (strain energy and
 * stress-strain relationship). Each material has different mechanical properties, computational
 * cost, and stability characteristics.
 *
 * @see SoftMaterialParams, MaterialPsdStrategy
 */
enum struct SoftMaterialType {
  /**
   * @brief Neo-Hookean hyperelastic material.
   *
   * @details Non-linear hyperelastic material with log-barrier stabilization to handle extreme
   * deformations and element inversions robustly. Recommended for most soft body simulations
   * requiring large deformations.
   *
   * @see [Stable Neo-Hookean Flesh Simulation (Smith et al.,
   * 2018)](https://www.tkim.graphics/NEO/StableNeoHookean2018.pdf)
   * @see NeoHookeanMaterialParams
   */
  NeoHookean = 0,

  /**
   * @brief Saint Venant-Kirchhoff (StVK) hyperelastic material.
   *
   * @details Classical non-linear hyperelastic material. Simpler than neo-Hookean but may exhibit
   * instabilities under large deformations, particularly under large compression.
   *
   * @see StVenantKirchhoffMaterialParams
   */
  StVenantKirchhoff = 1,

  /**
   * @brief Linear elastic material (Hookean).
   *
   * @details Simplest material model with linear stress-strain relationship. Computationally fast
   * but only accurate for small deformations.
   *
   * @note Not suitable for large deformations.
   * @note No PSD enforcement is needed. The Hessian matrix is positive definite regardless of the
   * PSD strategy.
   *
   * @see LinearElasticMaterialParams
   */
  LinearElastic = 2,

  /**
   * @brief Composite material: passive isotropic elasticity + active anisotropic fiber actuation.
   *
   * @details Combines a passive isotropic hyperelastic component (Neo-Hookean) with an active
   * anisotropic fiber contraction component (Aniso ARAP). Used to model materials with directional
   * actuation like muscle tissue.
   *
   * @see [Stable Neo-Hookean Flesh Simulation (Smith et al.,
   * 2018)](https://www.tkim.graphics/NEO/StableNeoHookean2018.pdf)
   * @see [Anisotropic Elasticity for Inversion-Safety and Element Rehabilitation (Kim et al.,
   * 2019)](http://tkim.graphics/ANISOTROPY/AnisotropyAndRehab.pdf)
   * @see ActiveNeoHookeanMaterialParams
   */
  ActiveNeoHookean = 3,

  /**
   * @brief Active As-Rigid-As-Possible (ARAP) material with Shape Targeting.
   *
   * @details Active material model based on the concept of "shape targeting". Unlike passive models
   * that only react to external loads, it introduces an internal actuation mechanism.
   *
   * The actuation is defined by the shape target tensor, S_t, which specifies the desired local
   * shape of the object. The model then resists deformation from this target shape while allowing
   * rigid rotation.
   *
   * This approach is effective for simulating biological phenomena like muscle contraction.
   *
   * @note The "activeness" is controlled externally by setting the shape target tensor S_t. The
   * model itself does not have an intrinsic, self-actuating mechanism without this input.
   *
   * @see [Shape Targeting: A Versatile Active Elasticity Constitutive Model (Klár et al.,
   * 2020)](https://par.nsf.gov/servlets/purl/10230451)
   * @see ActiveShapeTargetingArapMaterialParams
   */
  ActiveShapeTargetingArap = 4,

  /**
   * @brief As-Rigid-As-Possible (ARAP) material.
   *
   * @details ARAP's primary goal is to deform the object such that each local region undergoes a
   * transformation that is as close to a rigid rotation as possible.
   *
   * Because it preserves local shape details well and produces intuitive, natural-looking results,
   * ARAP is widely used in computer graphics for applications like interactive mesh editing and
   * character posing.
   *
   * @see [Dynamic Deformables (Kim and Eberle,
   * 2022)](https://www.tkim.graphics/DYNAMIC_DEFORMABLES/DynamicDeformables.pdf)
   * @see ArapMaterialParams
   */
  Arap = 5,

  /** @brief Number of unique material type enum values. */
  Count = 6
};
} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::SoftMaterialType);
MOCHI_ENUM_ITEM(NeoHookean) MOCHI_ATTRIBUTE(PreviouslyKnownAs("StableNeoHookean"));
MOCHI_ENUM_ITEM(StVenantKirchhoff);
MOCHI_ENUM_ITEM(LinearElastic);
MOCHI_ENUM_ITEM(ActiveNeoHookean) MOCHI_ATTRIBUTE(PreviouslyKnownAs("ActiveStableNeoHookean"));
MOCHI_ENUM_ITEM(ActiveShapeTargetingArap);
MOCHI_ENUM_ITEM(Arap);
MOCHI_ENUM_COUNT(Count);
MOCHI_ENUM_END();

namespace mochi::materials {

/// @brief Opt-in trait: a material params type declares this @c true iff its reference (@c F=I)
/// tangent is isotropic — i.e. invariant under rotation, so it is fully described by the two Lamé
/// scalars (λ, μ) with no directional coupling.
///
/// @details The primary template is @c false. A material specializes it to @c true only when this
/// holds by construction, independently of the specific parameter values. Consumers may optionally
/// use it to take faster, isotropy-specific code paths where applicable; leaving it @c false is
/// always correct and simply forgoes those paths.
template <class ParamsT>
inline constexpr bool kIsotropicReferenceStiffness = false;

} // namespace mochi::materials
