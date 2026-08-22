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

/**************************************************************************************************
  EXPERIMENTAL APIs

  WARNING: These features are primarily intended for internal use by the Mochi team.
           These features may not be fully stable, and they may change significantly
           or be removed without warning in the future. Use at your own risk.

***************************************************************************************************/

#include <mochi_physics/mochi_physics.h>

#include <mochi_core/ai/compute_type.h>
#include <mochi_core/articulated_body/articulated_body_params.h>
#include <mochi_core/contact/contact_params.h>
#include <mochi_core/rom/rom_hyper_reduction_params.h>
#include <mochi_core/solvers/nonlinear_solver_params.h>
#include <mochi_core/utils/color.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/dynamic_string.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/transform_rt.h>
#include <mochi_core/utils/verbosity_params.h>

#include <any>
#include <functional>
#include <optional>
#include <variant>

namespace mochi {
// Forwards
struct QPSolverParams;
struct NewtonSolverParams;

/**
 * @brief Default finite-difference epsilon used by @ref BackPropagationSolverParams::epsFiniteDiff.
 */
inline constexpr real kDefaultBackPropagationEpsFiniteDiff =
    MOCHI_USE_DOUBLE_PRECISION ? 1e-7_r : 1e-4_r;
} // namespace mochi

namespace mochi::experimental {

// -------------------------------------------------------------------
// Strategies for hyper-reduction
// -------------------------------------------------------------------
//
// List of valid strategies for doing hyper-reduction.
// A strategy can have arbitrary parameters/setting, but it MUST define
// its starting point, namely the params to use to create the first sample
// mesh at initialization time via the "initialSubsamplingParams" field (see below).
//
// IMPORTANT: If you add a new strategy:
// (1) it must have a "initialSubsamplingParams" inside
// (2) you must update the type alias "hypred_strategies_varian_t" (see below)
// (3) to actually use it, you must also modify the following functions:
//      - inside mochi_hyper_reduction:  SampleMeshNeedsToBeUpdatedAndHow
// (4) you need to add inside mochi_hyper_reduction.{h,cpp} a corresponding system
//     that defines the logic for triggering a sample mesh update

/**
 * @brief Initializes hyper-reduction from a sample mesh stored in the actor shape.
 *
 * @note The loaded sample mesh must provide one weight per volume and boundary element ID. Volume
 * and boundary element ID datasets must not contain duplicate IDs. Weights must be finite.
 */
struct SampleMeshInitFromFile {
  DynamicString source = "default";

#if MOCHI_LANGUAGE_CPP20
  bool operator==(SampleMeshInitFromFile const&) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::experimental::SampleMeshInitFromFile)
  MOCHI_FIELD(source)
  MOCHI_STRUCT_END()
};

struct SampleMeshInitRandomSampling {
  int stepSizeForBoundaryElementsSelection = rom::hyper::kAllElements;
  int stepSizeForInteriorElementsSelection = rom::hyper::kAllElements;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(SampleMeshInitRandomSampling const&) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::experimental::SampleMeshInitRandomSampling)
  MOCHI_FIELD(stepSizeForBoundaryElementsSelection)
  MOCHI_FIELD(stepSizeForInteriorElementsSelection)
  MOCHI_STRUCT_END()
};

struct SampleMeshInitFromSpecificVolumeElements {
  DynamicArray<int> ids = {};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(SampleMeshInitFromSpecificVolumeElements const&) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::experimental::SampleMeshInitFromSpecificVolumeElements)
  MOCHI_FIELD(ids)
  MOCHI_STRUCT_END()
};

using SampleMeshInitStrategy = std::variant<
    SampleMeshInitFromFile,
    SampleMeshInitRandomSampling,
    SampleMeshInitFromSpecificVolumeElements>;

// Uses dynamic boundary sampling from a BSH. Only supported with ActorBoundaryElementType::P1Q1.
// TODO: Deprecate maxColliderVelocity and use an upper bound of velocity at a given step through
// similar heuristics as in island splitting.
struct DynamicSampleMeshBsh {
  // The group under /bsh from which to load the BSH that will drive dynamic sample mesh updates.
  DynamicString source = "default";

  // Maximum distance at which the ROM will query the SDF from a collider. Higher values mean more
  // computation, but with a more stable sample mesh.
  real maxSdfCullDistance = 1.0_r;

  // Distance threshold to activate sample points. Sample points whose SDF lower bound is below
  // (above) the threshold are active (inactive). If the number of sample points with SDF lower
  // bound under the threshold exceeds the maximum subsampling density, the active sample points are
  // selected randomly among them.
  real sampleActivationDistance = 0.1_r;

  // Maximum subsampling density. Must be between 0 and 1. 0 means no sample point will be active. 1
  // means all sample points will be active.
  real maxSubsamplingDensity = 0.2_r;

  // Velocity that the BSH uses for the lower bound computation.
  real maxColliderVelocity = 0.0_r;

  // The set of anchor points over which to construct the SDF lower bound. 'Self' gives the loosest
  // lower bound and usually the fastest performance, while 'AncestorSibling' gives the tightest
  // lower bound and usually the slowest performance.
  rom::hyper::SdfLowerBoundAnchorSelection anchorSelectionMode =
      rom::hyper::SdfLowerBoundAnchorSelection::AncestorSibling;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(DynamicSampleMeshBsh const&) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::experimental::DynamicSampleMeshBsh)
  MOCHI_FIELD(source)
  MOCHI_FIELD(maxSdfCullDistance) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(sampleActivationDistance) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(maxSubsamplingDensity);
  MOCHI_FIELD(maxColliderVelocity) MOCHI_ATTRIBUTE(Units("m/s"));
  MOCHI_FIELD(anchorSelectionMode)
  MOCHI_STRUCT_END()
};

using DynamicSampleMeshStrategy = std::variant<DynamicSampleMeshBsh>;

struct HyperReductionParams {
  // Specifies how the sample mesh for hyper-reduction should be initialized.
  SampleMeshInitStrategy initializationStrategy = SampleMeshInitRandomSampling{};

  // Tells Mochi how it should update the sample mesh after it has been initialized.
  // By default, we use static sample meshes.
  std::optional<DynamicSampleMeshStrategy> dynamicStrategy = std::nullopt;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(HyperReductionParams const&) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::experimental::HyperReductionParams)
  // MOCHI_FIELD(initializationStrategy) // REFLECTION TODO
  // MOCHI_FIELD(dynamicStrategy) // REFLECTION TODO
  MOCHI_STRUCT_END()
};

// -------------------------------------------------------------------
// Strategies for ROM model adaptivity
// -------------------------------------------------------------------
// List of valid strategies for doing model adaptivity.

/*
Strategy to select the adaptive basis using contact force information.

IMPORTANT:
- Only supported for linear ROMs, including polynomial CROMs.
- Not supported with deep flow.
- Not supported with recentering.

HOW IT WORKS:

Step 1:
    At the end of each time step, we compute:
            Rvec = Phi_candidate^T * node_force_vec
    Where:
    - Phi_candidate is the candidate basis loaded from 'candidateSource', which must be a valid HDF5
      field in the target asset file or a string defining a polynomial CROM.
    - node_force_vec represents contact forces computed in CQueryNodeContactForces.

Step 2:
    We construct the final basis as:
            Phi = Phi_0 + Phi_adaptive
    Where:
    - Phi_0 is a fixed set of 12 modes (the 1st-order polynomial basis) capturing translation and
      rotation. These modes are only used if the ROM does NOT have a rigid transform.
    - Phi_adaptive consists of numAdaptiveBasis modes selected from Phi_candidate, based on the
      largest values in |Rvec|, i.e. using ArgSort(-Abs(Rvec)).

    As a result, the final adaptive basis has 12 + numAdaptiveBasis modes (if the ROM does not have
    a rigid transform) or numAdaptiveBasis modes (if it has a rigid transform).

NOTE:
- For polynomial CROMs, the 'candidateSource' must be 'polynomial_crom_order_X' with X >= 2 to
  ensure only higher-order modes are selected adaptively.
- If there is no contact, the initial (non-adaptive) basis is used.
- CQueryNodeContactForces is automatically enabled with this strategy.
*/
struct ContactForceInformedRomAdaptivityParams {
  int numAdaptiveBasis = 5;
  DynamicString candidateSource = {};

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ContactForceInformedRomAdaptivityParams const&) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::experimental::ContactForceInformedRomAdaptivityParams)
  MOCHI_FIELD(numAdaptiveBasis)
  MOCHI_FIELD(candidateSource)
  MOCHI_STRUCT_END()
};

/*
Strategy to select the adaptive basis using neural network gradients.

IMPORTANT: Not supported with deep flow.

HOW IT WORKS:

Step 1:
    At the end of each time step, we run the encoder to get reduced coordinates:
            p = g(u)
    Where:
    - g is the encoder from the neural CROM model specified by 'neuralCromSource'
    - u represents the current displacements
    - p are the reduced coordinates in the neural CROM latent space

    Alternatively, we can solve the optimization problem:
            min_p Σ||u - h(p, x_i)||²
    Where h is the decoder function.

Step 2:
    We compute the gradient of the decoder for all mesh nodes:
            ∂h(p, x_i)/∂p for all i = 1, ..., N_nodes
    Where:
    - Each gradient is a 3×n matrix corresponding to rows 3·i, 3·i+1, 3·i+2 of the basis Φ
    - The resulting basis Φ is used for the next time step

NOTE:
- The neural CROM model must be available in the asset file under the specified 'neuralCromSource'
*/

enum struct NeuralAffineRomMethod { CromEncoder, CromProjection, Interpolation };

} // namespace mochi::experimental

MOCHI_ENUM_BEGIN(mochi::experimental::NeuralAffineRomMethod)
MOCHI_ENUM_ITEM(CromEncoder)
MOCHI_ENUM_ITEM(CromProjection)
MOCHI_ENUM_ITEM(Interpolation)
MOCHI_ENUM_END()

namespace mochi::experimental {

struct NeuralAffineRomParams {
  DynamicString neuralCromSource = {}; // Source neural CROM model
  NeuralAffineRomMethod method = NeuralAffineRomMethod::CromEncoder;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(NeuralAffineRomParams const&) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::experimental::NeuralAffineRomParams)
  MOCHI_FIELD(neuralCromSource)
  MOCHI_FIELD(method)
  MOCHI_STRUCT_END()
};

using RomAdaptivityParams =
    std::variant<ContactForceInformedRomAdaptivityParams, NeuralAffineRomParams>;

// -------------------------------------------------------------------
// Strategies for hybrid ROM/FOM switching
// -------------------------------------------------------------------

/**
 * @brief Switch from ROM to FOM, and later from FOM to ROM, after a given number of steps.
 * @note This strategy is only useful for testing purposes.
 * @note fomToRomSwappingStep must be larger than romToFomSwappingStep, i.e. the actor must start as
 * ROM.
 */
struct RomFomSwitchingTestOnlyParams {
  int romToFomSwappingStep = 5; ///< After this many steps, switch to FOM.
  int fomToRomSwappingStep = 9; ///< After this many steps, switch back to ROM.

#if MOCHI_LANGUAGE_CPP20
  bool operator==(RomFomSwitchingTestOnlyParams const&) const = default;
#endif
};

/** @brief Switch between ROM and FOM based on the number of active collision points. */
struct RomFomSwitchingContactInformedParams {
  /** @brief Use ROM if the number of active contact points is smaller than this threshold. Use FOM
   * otherwise. */
  int numCollisionPtsThreshold = 5;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(RomFomSwitchingContactInformedParams const&) const = default;
#endif
};

using RomFomSwitchingParams =
    std::variant<RomFomSwitchingTestOnlyParams, RomFomSwitchingContactInformedParams>;

// -------------------------------------------------------------------
// ROM projection strategy
// -------------------------------------------------------------------

/**
 * @brief Strategy to project the residual and dresidual from FOM space to ROM space.
 *
 * @note Applies only to terms assembled in the actor residual and actor dresidual (e.g., volume
 * terms and async contact). It doesn't apply to terms assembled in the interaction residual and
 * interaction dresidual (e.g., sync contact).
 */
enum struct RomProjectionStrategy {
  /** @brief Assemble the FOM residual and dresidual first, then project them to ROM space. */
  ActorLevelProjection,

  /**
   * @brief Perform projection to ROM space element-by-element while assembling the FOM actor,
   * without ever forming the FOM residual and dresidual.
   */
  ElementLevelProjection,

  /** @brief Number of ROM projection strategy enum values. */
  Count,

  /** @brief Default ROM projection strategy. */
  Default = ActorLevelProjection,
};

} // namespace mochi::experimental

MOCHI_ENUM_BEGIN(mochi::experimental::RomProjectionStrategy)
MOCHI_ENUM_ITEM(ActorLevelProjection)
MOCHI_ENUM_ITEM(ElementLevelProjection)
MOCHI_ENUM_COUNT(Count)
MOCHI_ENUM_END()

namespace mochi::experimental {

// -------------------------------------------------------------------
// Parameters defining a ROM actor
// -------------------------------------------------------------------
struct RomParams {
  /**
   * @brief Specifies a ROM inside the 'rom' group of the .mochi.h5 file.
   * @note The exception is polynomial CROMs, which can be used for any asset without the need to
   * embed them into the H5 file. The source schema for polynomial CROMs is
   * 'polynomial_crom_order_X', where X is the polynomial order.
   */
  DynamicString source;

  /** @brief Hyper-reduction parameters, or empty if there is no hyper-reduction. */
  std::optional<HyperReductionParams> hyperReduction = std::nullopt;

  /**
   * @brief Adaptivity parameters, or empty if there is no adaptivity.
   * @note Adaptivity is not supported with deep flow.
   */
  std::optional<RomAdaptivityParams> adaptivity = std::nullopt;

  /**
   * @brief ROM/FOM switching parameters, or empty if there is no ROM/FOM switching.
   * @note ROM/FOM switching is not supported with deep flow.
   */
  std::optional<RomFomSwitchingParams> romFomSwitching = std::nullopt;

  /** @brief Mesh color used in the debug draw. */
  std::optional<Color> surfaceMeshColor = std::nullopt;

  /** @brief Strategy to project the residual and dresidual from FOM space to ROM space. */
  RomProjectionStrategy romProjectionStrategy = RomProjectionStrategy::Default;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(RomParams const&) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::experimental::RomParams)
  MOCHI_FIELD(source)
  MOCHI_FIELD(hyperReduction)
  // MOCHI_FIELD(adaptivity) // REFLECTION TODO
  // MOCHI_FIELD(romFomSwitching) // REFLECTION TODO
  // MOCHI_FIELD(surfaceMeshColor) // REFLECTION TODO
  MOCHI_FIELD(romProjectionStrategy)
  MOCHI_STRUCT_END()
};

struct DeepModelParams {
  // Path of the file containing the deep model.
  DynamicString deepModelPath;
  // The deep model is trained with a shift and scale with respect to the object's shape. World
  // and local coordinates are related as: world = scale * local + shift.
  real shiftX = 0_r;
  real shiftY = 0_r;
  real shiftZ = 0_r;
  real scale = 1_r;
  // Size of the kinematic code of the model.
  int numDof = 0;
  // Error bound for distances evaluated using the deep model, necessary for conservative
  // computation of object bounds. It must be scaled by the object's scale.
  // TODO: This value should be minimized to speed up collision detection.
  real errorBound = 0.1_r;

  MOCHI_STRUCT_BEGIN(mochi::experimental::DeepModelParams)
  MOCHI_FIELD(deepModelPath)
  MOCHI_FIELD(shiftX) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(shiftY) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(shiftZ) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_FIELD(scale)
  MOCHI_FIELD(numDof)
  MOCHI_FIELD(errorBound) MOCHI_ATTRIBUTE(Units("m"));
  MOCHI_STRUCT_END()
};

// TODO: Incorporate this into the public API struct analogous to SoftMaterialParams defined in
// (material_params.h) as shell modeling matures.  For now, it is a stub to ensure analogous code
// structure elsewhere between shell and soft actors.
struct ShellMaterialParams {
  // The membrane Lame parameters, lambda and mu, for plane-stress linear elasticity.  These should
  // be thought of as "2D" parameters, where the conversion from 3D material properties through the
  // membrane thickness has already been applied.  The units are [Pa*m] = [N/m], i.e., force per
  // unit length.
  real membraneLambda = 300_r; // [Pa*m]
  real membraneMu = 400_r; // [Pa*m]
  // These define a basic isotropic bending stiffness, which can be combined with any in-plane
  // membrane model, as is common in cloth modeling for graphics (as opposed to engineering shell
  // theories where bending and membrane responses are simultaneously derived from a 3D constitutive
  // model and thickness.)
  real bendingAlpha = 0.00002_r; // [Pa*m^3]
  real bendingBeta = 0.00007_r; // [Pa*m^3]
  // NOTE: This is interpreted as density per unit area, not per unit volume.
  real density = 1.0_r; // [kg/m^2]
  // Damping coefficient for velocity-proportional forces from the mass term.
  real massDampingCoefficient = 0_r; // [1/s]
  // Stiffness-proportional damping coefficient.
  real stiffnessDampingCoefficient = 0_r; // [s]

  // clang-format off
  MOCHI_STRUCT_BEGIN(mochi::experimental::ShellMaterialParams)
  MOCHI_FIELD(membraneLambda) MOCHI_ATTRIBUTE(Units("Pa*m"))
  MOCHI_FIELD(membraneMu) MOCHI_ATTRIBUTE(Units("Pa*m"))
  MOCHI_FIELD(bendingAlpha) MOCHI_ATTRIBUTE(Units("Pa*m^3"))
  MOCHI_FIELD(bendingBeta) MOCHI_ATTRIBUTE(Units("Pa*m^3"))
  MOCHI_FIELD(density) MOCHI_ATTRIBUTE(Units("kg/m^2"))
  MOCHI_FIELD(massDampingCoefficient) MOCHI_ATTRIBUTE(Units("1/s"))
  MOCHI_FIELD(stiffnessDampingCoefficient) MOCHI_ATTRIBUTE(Units("s"))
  MOCHI_STRUCT_END()
  // clang-format on
};

struct PointCloudColliderParams {
  /**
   * @brief Range of interaction of the contact penalty potential evaluated between pairs of
   * points [m].
   *
   * @details If the contact sample points (used as a quadrature for the contact penalty) are
   * spaced significantly further apart than this, then nodes will be able to slip between them.
   * This can be addressed without refining the mesh by increasing the number of contact samples
   * per element, whose spacing scales roughly like (element size) / (quadrature points per
   * element)^(1/d), where `d` is the parametric dimension of the collider geometry (`d = 2` for
   * surfaces such as shell, `d = 1` for curves such as rods).
   *
   * @note This value also serves as the length scale for dimensional correction of the contact
   * penalty coefficient on the collider side. The effective penalty stiffness is scaled by
   * `pow(radius, -integralDim)`, where `integralDim` is the dimension of the collider-side
   * contact integral.
   */
  real radius = 0.01_r;

  /**
   * @brief Contact samples whose reference-configuration positions are within a radius of
   * (radius * selfContactExclusionRatio) + (additional padding) of a node are excluded from
   * self-contact force computation for that node, where "additional padding" refers to any extra
   * padding associated with the contact penalty formulation.
   *
   * @note Must be > 1.
   */
  real selfContactExclusionRatio = 1.5_r;

  /**
   * @brief Load factor for the spatial hash table used in collision detection. Should be << 1.
   */
  real spatialHashLoadFactor = 0.0625_r;

  /**
   * @brief Whether to enable self-contact for this actor.
   *
   * @note False by default because it is still highly experimental, not optimized yet, and may
   * severely degrade performance in scenarios where it isn't needed. Has no effect if the actor's
   * colliderType is set to @ref ColliderType::None.
   */
  bool selfContact = false;

  /**
   * @brief Triangle element type for collider discretization on a surface.
   *
   * @details When nullopt, uses nodal collider placement (one collider point per mesh node,
   * weighted by assembled nodal measure — cheapest option). When set, uses quadrature-based
   * collider placement with the specified element type (denser and more accurate, but more
   * expensive). Only used for surface collider geometries (e.g., shell).
   */
  std::optional<ActorBoundaryElementType> colliderTriangleElementType = std::nullopt;

  /**
   * @brief Segment element type for collider discretization on a curve.
   *
   * @details When nullopt, uses nodal collider placement. When set, uses quadrature-based
   * placement with the specified segment element type. Only used for curve collider geometries
   * (e.g., rods).
   */
  std::optional<ActorSegmentElementType> colliderSegmentElementType = std::nullopt;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(PointCloudColliderParams const&) const = default;
#endif

  // clang-format off
  MOCHI_STRUCT_BEGIN(mochi::experimental::PointCloudColliderParams)
  MOCHI_FIELD(radius) MOCHI_ATTRIBUTE(Units("m"))
  MOCHI_FIELD(selfContactExclusionRatio)
  MOCHI_FIELD(spatialHashLoadFactor)
  MOCHI_FIELD(selfContact)
  MOCHI_FIELD(colliderTriangleElementType)
  MOCHI_FIELD(colliderSegmentElementType)
  MOCHI_STRUCT_END()
  // clang-format on
};

// Parameters for creating a shell actor.
struct ShellActorParams {
  DynamicString name;
  DynamicString layer;
  TransformRT worldFromLocal = TransformRT::Identity();
  ShapeHandle shape;
  ShellMaterialParams material = {};
  ColliderType colliderType = ColliderType::PointCloud;
  ContactParams contact;
  PointCloudColliderParams pointCloudCollider = {};
  bool hasGravity = true;
  ActorBoundaryElementType contactElementType = ActorBoundaryElementType::Default;

  MOCHI_STRUCT_BEGIN(mochi::experimental::ShellActorParams)
  MOCHI_FIELD(name)
  MOCHI_FIELD(layer)
  MOCHI_FIELD(worldFromLocal)
  // MOCHI_FIELD(shape) // REFLECTION TODO
  MOCHI_FIELD(material)
  MOCHI_FIELD(colliderType)
  MOCHI_FIELD(contact)
  MOCHI_FIELD(pointCloudCollider)
  MOCHI_FIELD(hasGravity)
  MOCHI_FIELD(contactElementType)
  MOCHI_STRUCT_END()
};

struct RodMaterialParams {
  // Mass per unit undeformed length in a rod. For a rod made of homogeneous material, this would be
  // the 3D mass density (per unit volume) times the cross-sectional area. This must be positive.
  real linearDensity = 1e-1_r; // [kg/m]
  // Rotational inertia about the rod's axis, per unit undeformed length. For a rod made of a
  // homogeneous material, the value would be the 3D density (per unit volume) times the cross
  // section's polar area moment of inertia. NOTE: This must be positive if there are no Dirichlet
  // boundary conditions or other constraints on torsional degrees of freedom and/or if the
  // torsional stiffness is zero.
  real linearRotationalInertia = 1e-6_r; // [kg m^2 / m] = [kg m]
  // Axial stiffness of a rod. This would be Young's modulus times cross-section area if the rod was
  // made of a homogeneous elastic material. This must be positive.
  real axialStiffness = 1e2_r; // [N]
  // Torsional stiffness of a rod. For rods made of homogenous elastic material, this would be the
  // shear modulus times a "torsion constant" that is the polar area moment of inertia for circular
  // cross sections, and depends on solving for a Prandtl stress function in the general case (with
  // values for common shapes available in engineering handbooks). NOTE: If rotational inertia is
  // zero (which may be well posed when some twisting DoFs are constrained), this must be positive
  // to transmit the effects of constrained twisting DoFs to the rest of the rod.
  real torsionalStiffness = 5e-4_r; // [N m^2]
  // The first value is flexural stiffness about the material frame axes explicitly provided, and
  // the second value is flexural stiffness about the perpendicular axis. For a beam made of
  // homogeneous elastic material, these would be Young's modulus times the cross section's area
  // moments of inertia about the respective axes. Both values must be non-negative.
  Real2 flexuralStiffness = {5e-4_r, 5e-4_r}; // [N m^2]
  // Damping coefficient for velocity-proportional forces from the mass term.
  real massDampingCoefficient = 0_r; // [1/s]
  // Stiffness-proportional damping coefficient.
  real stiffnessDampingCoefficient = 0_r; // [s]

#if MOCHI_LANGUAGE_CPP20
  bool operator==(RodMaterialParams const&) const = default;
#endif

  // clang-format off
  MOCHI_STRUCT_BEGIN(mochi::experimental::RodMaterialParams)
  MOCHI_FIELD(linearDensity) MOCHI_ATTRIBUTE(Units("kg/m"))
  MOCHI_FIELD(linearRotationalInertia) MOCHI_ATTRIBUTE(Units("kg*m"))
  MOCHI_FIELD(axialStiffness) MOCHI_ATTRIBUTE(Units("N"))
  MOCHI_FIELD(torsionalStiffness) MOCHI_ATTRIBUTE(Units("N*m^2"))
  MOCHI_FIELD(flexuralStiffness) MOCHI_ATTRIBUTE(Units("N*m^2"))
  MOCHI_FIELD(massDampingCoefficient) MOCHI_ATTRIBUTE(Units("1/s"))
  MOCHI_FIELD(stiffnessDampingCoefficient) MOCHI_ATTRIBUTE(Units("s"))
  MOCHI_STRUCT_END()
  // clang-format on
};

struct RodActorParams {
  DynamicString name;
  DynamicString layer;
  TransformRT worldFromLocal = TransformRT::Identity();
  ShapeHandle shape;
  ContactParams contact;
  ActorSegmentElementType contactElementType = ActorSegmentElementType::Default;
  RodMaterialParams material = {};
  ColliderType colliderType = ColliderType::None;
  PointCloudColliderParams pointCloudCollider = {};
  bool hasGravity = true;
  bool useVisualMeshContact = false;
  ActorBoundaryElementType visualMeshContactElementType = ActorBoundaryElementType::Default;

  MOCHI_STRUCT_BEGIN(mochi::experimental::RodActorParams)
  MOCHI_FIELD(name)
  MOCHI_FIELD(layer)
  MOCHI_FIELD(worldFromLocal)
  // MOCHI_FIELD(shape) // REFLECTION TODO
  MOCHI_FIELD(contact)
  MOCHI_FIELD(contactElementType)
  MOCHI_FIELD(material)
  MOCHI_FIELD(colliderType)
  MOCHI_FIELD(pointCloudCollider)
  MOCHI_FIELD(hasGravity)
  MOCHI_FIELD(useVisualMeshContact)
  MOCHI_FIELD(visualMeshContactElementType)
  MOCHI_STRUCT_END()
};

enum class ControlType {
  SingleDof,

  LinkPos,

  LinkRot,

  Force,

  Count,
};

} // namespace mochi::experimental

MOCHI_ENUM_BEGIN(mochi::experimental::ControlType)
MOCHI_ENUM_ITEM(SingleDof)
MOCHI_ENUM_ITEM(LinkPos)
MOCHI_ENUM_ITEM(LinkRot)
MOCHI_ENUM_ITEM(Force)
MOCHI_ENUM_COUNT(Count)
MOCHI_ENUM_END()

namespace mochi::experimental {

// These define a linear transmission, without specifying an actuator, different types of which
// can be added later. A "linear transmission" maps a subset of articulated joint DoFs to a single
// generalized coordinate (called the transmission's "displacement" from the rest pose) via a
// fixed weighted sum, and exposes an optional actuator that produces an energy-conjugate
// generalized force.
struct LinearTransmissionParams {
  DynamicArray<int> jointIndices;
  // Entries of the transmission's Jacobian dq_trans/dq_joint. Sign encodes direction: positive if
  // the transmission displacement increases with the joint DoF, negative if it decreases. For a
  // tendon, units are [m/(joint DoF units)] (i.e. the moment arm / radius); for a gearbox, units
  // are dimensionless (the gear ratio).
  DynamicArray<real> jointCoefficients;
};

// These define a spatial tendon, without specifying an actuator, different types of which can be
// added later. A spatial tendon is routed through an ordered list of elements (see
// @ref RoutingElement): waypoint elements (points fixed in a link's local frame — the same frame
// in which the link's mesh and geometry are authored — with straight-line segments between
// waypoint elements adjacent in the list) and linear-joint elements (a constant moment arm
// contributing coefficient * jointCoordinate).
struct SpatialTendonParams {
  // Ordered routing elements. Must contain at least one element, and every waypoint must be
  // adjacent to another waypoint (an isolated waypoint forms no segment and is rejected).
  DynamicArray<RoutingElement> routingElements;
};
struct DisplacementControlActuatorParams {
  // Initial value of the target displacement (state variable)
  real targetDisplacement = 0_r; // [generalized coordinate units]
  // Stiffness in the penalty used to control displacement
  real stiffness = 1e9_r; // [generalized force / generalized coordinate units]
  // Damping coefficient to reduce oscillations
  real damping = 0_r; // [generalized force / (generalized coordinate / s) units]
  // Set true if this actuator is attached to a transmission that can transmit force in both
  // directions (e.g. gearbox or rigid linkage); leave false (default) for rope-like
  // transmissions that go slack under compression.
  bool allowCompressiveForce = false;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(DisplacementControlActuatorParams const&) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::experimental::DisplacementControlActuatorParams)
  MOCHI_FIELD(targetDisplacement) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(stiffness) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(damping) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_FIELD(allowCompressiveForce) MOCHI_ATTRIBUTE(NoSerializeDefaults);
  MOCHI_STRUCT_END()
};
struct ForceControlActuatorParams {
  // Initial value of the applied generalized force (state variable)
  real force = 0_r; // [generalized force units]
  // Set true if this actuator is attached to a transmission that can transmit force in both
  // directions (e.g. gearbox or rigid linkage); leave false (default) for rope-like
  // transmissions, in which case negative forces are rejected with an error.
  bool allowCompressiveForce = false;
};
struct McKibbenActuatorParams {
  // Initial value of current pressure (state variable)
  real pressure = 0_r; // [Pa]
  // Minimum pressure for inner tube to contact outer braid
  real minimumPressure = 0_r; // [Pa]
  // Thread length
  real threadLength = 0_r; // [m]
  // Number of wraps
  real numberOfWraps = 0_r; // dimensionless
  // Muscle stiffness below minimum pressure
  real deflatedStiffness = 0_r; // [N/m]
  // Equillibrium muscle length below minimum pressure
  real deflatedEquilibriumLength = 0_r; // [m]
};

/**************************************************************************************************
  Experimental API Methods:
*/

/**
 * @brief Set force/torque and target pose on specific DoFs or links with various control modes.
 *
 * @details This function supports 4 control modes:
 * - **SingleDof**: PD-target on a single DoF
 * - **LinkPos**: PD-target on 3D link position
 * - **LinkRot**: PD-target on 3D link rotation (rotation vector)
 * - **Force**: Force/torque on a single DoF
 *
 * Target values are consumed sequentially: single-DoF modes consume 1 value,
 * 3D modes consume 3 values.
 *
 * @param[in] actor Articulated actor to control.
 * @param[in] target Target values. Size must equal sum of value counts for all control types.
 * @param[in] controlTypes Control mode for each controlled DoF/link.
 * @param[in] dofOrLinkIndices DoF index (single-DoF modes) or link index (3D modes) for each
 * control type.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 *
 * @see ControlType
 */
MOCHI_API void SetArticulatedForceAndTargetPose(
    Actor* actor,
    Span<real const> target,
    Span<ControlType const> controlTypes,
    Span<int const> dofOrLinkIndices,
    Error& error);

// [Articulated] Add a linear transmission to the actor and return the new transmission's index,
// which can be used to attach an actuator or query/update it in other ways.
MOCHI_API int
AddLinearTransmission(Actor* actor, LinearTransmissionParams const& params, Error& error);

// [Articulated] Add a spatial tendon to the actor and return the new transmission's index, which
// can be used to attach an actuator or query/update it in other ways.
MOCHI_API int AddSpatialTendon(Actor* actor, SpatialTendonParams const& params, Error& error);

// [Articulated] Attach actuators of different types to transmissions specified by the index
// returned by AddLinearTransmission or AddSpatialTendon.
MOCHI_API void AttachDisplacementControlActuator(
    Actor* actor,
    int transmissionIndex,
    DisplacementControlActuatorParams const& params,
    Error& error);

MOCHI_API void AttachForceControlActuator(
    Actor* actor,
    int transmissionIndex,
    ForceControlActuatorParams const& params,
    Error& error);

MOCHI_API void AttachMcKibbenActuator(
    Actor* actor,
    int transmissionIndex,
    McKibbenActuatorParams const& params,
    Error& error);

// [Articulated] Set (or update) the state variables for the actuator attached to the transmission
// with the given index, or set an error if the transmission or actuator is invalid.
MOCHI_API void SetTransmissionActuatorStateVariables(
    Actor* actor,
    int transmissionIndex,
    Span<real const> stateVariables,
    Error& error);

// [Articulated] Get the generalized coordinate ("displacement") of a transmission at the actor's
// current pose, relative to its value at the rest pose (all-zero joint DoFs).
[[nodiscard]] MOCHI_API real
GetTransmissionDisplacement(Actor const* actor, int transmissionIndex, Error& error);

// [Articulated] Get the Jacobian of a transmission's displacement with respect to the actor's
// reduced DoFs, evaluated at the current pose. `outJacobian` size must equal GetNumDofs().
MOCHI_API void GetTransmissionDisplacementJacobian(
    Actor const* actor,
    int transmissionIndex,
    Span<real> outJacobian,
    Error& error);

// [Articulated] Get the state variables for the actuator attached to the transmission with the
// given index, or set an error if the transmission or actuator is invalid.
MOCHI_API void GetTransmissionActuatorStateVariables(
    Actor const* actor,
    int transmissionIndex,
    Span<real> outStateVariables,
    Error& error);

// [Articulated] Get the number of state variables for the actuator attached to the transmission
// with the given index, or set an error if the transmission or actuator is invalid.
[[nodiscard]] MOCHI_API int
GetNumTransmissionActuatorStateVariables(Actor const* actor, int transmissionIndex, Error& error);
// [Soft, Shell] Fix nodes at their reference positions transformed into world space if those
// world-space positions and/or their corresponding nodal indices satisfy some given condition,
// passed as a callback.
//
// FIXME: This cannot be migrated to the stable mochi_physics API because it requires a callback,
// which is not portable to Python. This should be moved to a shared utility library that is
// accessible to both samples and unit tests.
MOCHI_API void ConstrainNodesByPosition(
    Actor* actor,
    std::function<bool(int, Real3 const&)> callback,
    Error& error);

// Create a shape defined by a flow map, where the map is approximated by a neural network.
// WARNING: Requires MOCHI_ENABLE_DEEP_FLOW_ACTORS=1.
// WARNING: Requires libtorch (see MOCHI_USE_TORCH in mochi_config.h) if the compute type is Torch.
[[nodiscard]] MOCHI_API ShapeHandle CreateDeepFlowShape(
    Context* context,
    DeepModelParams const& params, // Parameters of a deep model
    NeuralComputeType computeType, // Compute type, e.g. MochiCpu, TorchCpu, TorchGpu
    int preallocMemSize, // Amount of preallocated GPU memory. Only used if computeType is
                         // TorchGpu
    Error& error);

/**
 * @brief [Experimental] Additional parameters for creating a soft actor.
 *
 * @warning These are experimental features. They may be changed or removed in the future. Use at
 * your own risk.
 *
 * @see SoftActorParams
 */
struct ExperimentalSoftActorParams {
  /**
   * @brief [Experimental] Collision detection geometry.
   *
   * @warning The use of @ref ColliderType for a soft actor is experimental. It may be changed or
   * removed in the future. Use at your own risk.
   *
   * @note Determines how OTHER actors detect contact with this actor. It does not affect how this
   * actor detects contact with other actors.
   *
   * @see ColliderType
   */
  ColliderType colliderType = ColliderType::None;

  /**
   * @brief [Experimental] Parameters used to construct a grid-based Signed Distance Field (SDF) if
   * the shape doesn't already have one.
   *
   * @warning The use of @ref GridSdfParams for a soft actor is experimental. It may be changed or
   * removed in the future. Use at your own risk.
   *
   * @note Ignored if @ref colliderType is not @ref ColliderType::Sdf.
   * @note Ignored if the shape already has a grid-based SDF.
   */
  GridSdfParams sdf;

  /**
   * @brief [Experimental] Optional deep flow shape handle for collision detection.
   *
   * @warning This is an experimental feature. It may be changed or removed in the future. Use at
   * your own risk.
   * @warning Deep Flow requires `MOCHI_ENABLE_DEEP_FLOW_ACTORS=1`. If disabled, @ref
   * CreateDeepFlowShape fails, and actor creation fails when this handle is valid.
   *
   * @note Leave empty (default) to use standard tetrahedral mapping.
   * @note Scenes that contain actors with deep flow force a single simulation island, which may
   * degrade performance.
   */
  ShapeHandle flow;

  /**
   * @brief [Experimental] Optional Reduced-Order Model (ROM) parameters.
   *
   * @warning ROMs are an experimental feature and may change or be removed. Use at your own risk.
   * @warning ROM actor creation requires `MOCHI_ENABLE_ROM_ACTORS=1`. If disabled, @ref
   * CreateModelShapeWithLinearRom fails, and actor creation fails when this field is set.
   */
  std::optional<RomParams> rom;

  /**
   * @brief Enable automatic recentering of the local coordinate system.
   *
   * @warning This is an experimental feature. It may be changed or removed in the future. Use at
   * your own risk.
   *
   * @note Recentering updates the root transform as the actor moves, keeping local-space
   * displacements small. This improves numerical stability for actors that move far from their
   * initial position.
   * @note Ignored by soft skinned actors, which do not support recentering.
   */
  bool useRecentering = true;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ExperimentalSoftActorParams const&) const = default;
#endif
};

// Create a soft actor with experimental parameters included.
// WARNING: ROM actor creation requires `MOCHI_ENABLE_ROM_ACTORS=1`. Deep Flow actor creation
// requires `MOCHI_ENABLE_DEEP_FLOW_ACTORS=1`.
MOCHI_API Actor* CreateSoftActor(
    Scene* scene,
    SoftActorParams const& params,
    ExperimentalSoftActorParams const& experimentalParams,
    Error& error);

/**
 * @brief [Experimental] Additional parameters for creating a soft-skinned actor.
 *
 * @warning These are experimental features. They may be changed or removed in the future. Use at
 * your own risk.
 *
 * @see SoftSkinnedActorParams
 */
struct ExperimentalSoftSkinnedActorParams {
  /**
   * @brief [Experimental] Parameters for each soft actor in the soft-skinned actor.
   *
   * @warning This is an experimental feature. It may be changed or removed in the future. Use at
   * your own risk.
   *
   * @note Must be either empty or the same size as @ref SoftSkinnedActorParams::softParams.
   */
  DynamicArray<ExperimentalSoftActorParams> softParams;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ExperimentalSoftSkinnedActorParams const&) const = default;
#endif
};

// Create a soft-skinned actor with experimental parameters included.
// WARNING: ROM actor creation requires `MOCHI_ENABLE_ROM_ACTORS=1`. Deep Flow actor creation
// requires `MOCHI_ENABLE_DEEP_FLOW_ACTORS=1`.
MOCHI_API Actor* CreateSoftSkinnedActor(
    Scene* scene,
    SoftSkinnedActorParams const& params,
    ExperimentalSoftSkinnedActorParams const& experimentalParams,
    Error& error);

MOCHI_API Actor* CreateShellActor(Scene* scene, ShellActorParams const& params, Error& error);

MOCHI_API Actor* CreateRodActor(Scene* scene, RodActorParams const& params, Error& error);

// Like Context::CreateModelShape, but with additional data for a linear ROM.
// WARNING: Requires MOCHI_ENABLE_ROM_ACTORS=1.
[[nodiscard]] MOCHI_API ShapeHandle CreateModelShapeWithLinearRom(
    Context* context,
    ModelDataView const& model,
    std::string_view linearRomName,
    Span<real const> linearRomBasis,
    Error& error);

// Create a polyline shape for use with rod actors.
// @param[in] context The context to create the shape in.
// @param[in] nodes Node positions [m] defining the polyline centerline.
// @param[in] elementFrameAxes Unit vectors perpendicular to each element's tangent direction.
//            If empty, these will be auto-generated using parallel transport.
// @param[in] isClosedLoop If true, the last node connects back to the first, forming a closed loop.
// @param[in,out] error Error status. Check @ref Error::IsOK for success.
// @return Handle to the created shape.
[[nodiscard]] MOCHI_API ShapeHandle CreatePolylineShape(
    Context* context,
    Span<Real3 const> nodes,
    Span<Real3 const> elementFrameAxes,
    bool isClosedLoop,
    Error& error);

// Generate a @ref ModelData with a polyline simulation mesh and a tubular visual mesh for use
// with rod actors. The returned @ref ModelData can be passed to @ref Context::CreateModelShape.
// @param[in] nodes Node positions [m] defining the polyline centerline. Must have at least 2.
// @param[in] elementFrameAxes Unit vectors perpendicular to each element's tangent direction.
//            If empty, these will be auto-generated using parallel transport.
// @param[in] radius Radius [m] of the tubular visual mesh cross-section. Must be positive.
// @param[in] numCrossSectionSegments Number of segments around the tube circumference (>= 3).
// @param[in] isClosedLoop If true, the polyline forms a closed loop.
// @param[in,out] error Error status. Check @ref Error::IsOK for success.
// @return A @ref ModelData suitable for @ref Context::CreateModelShape.
[[nodiscard]] MOCHI_API ModelData GenerateTubularRodModelData(
    Span<Real3 const> nodes,
    Span<Real3 const> elementFrameAxes,
    real radius,
    int numCrossSectionSegments,
    bool isClosedLoop,
    Error& error);

// Convenience function to create an isotropic material for shell from 3D isotropic
// elasticity parameters and a thickness.
// Requires youngsModulus3d > 0, density3d > 0, thickness > 0, and poissonsRatio3d in (-1, 0.5);
// on error returns a default-constructed @ref ShellMaterialParams.
MOCHI_API ShellMaterialParams ShellMaterialParamsFrom3dIsotropic(
    real youngsModulus3d,
    real poissonsRatio3d,
    real density3d,
    real thickness,
    Error& error);

[[nodiscard]] MOCHI_API real
CalibrateNormalViscousDampingCoefficient(real cor, real impactVelocity, Error& error);

[[nodiscard]] MOCHI_API real EffectiveCoefficientOfRestitution(
    real normalViscousDampingCoefficient,
    real impactVelocity,
    Error& error);

// [Rigid and articulated] Enable/disable Newton-Euler inertia. Default is false.
MOCHI_API void EnableNewtonEulerInertia(Actor* actor, bool enable, Error& error);

// [Rigid and articulated] True with Newton-Euler inertia; false with merit-based inertia.
[[nodiscard]] MOCHI_API bool IsNewtonEulerInertiaEnabled(Actor const* actor, Error& error);

/**
 * @brief Set the material parameters for a specific element.
 *
 * @param[in,out] actor Actor to modify.
 * @param[in] params Material parameters to set.
 * @param[in] elementIndex Index of element to modify.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 *
 * @note Only supported for soft actors.
 * @note The requested element material type and PSD strategy must be consistent with the actor's
 * material type and PSD strategy.
 * @note Only the material sub-struct selected by @ref SoftMaterialParams::type is applied.
 * @ref SoftMaterialParams::density is actor-wide and ignored.
 * @note Callers modifying a subset of the parameters should use @ref GetSoftMaterialParamsField,
 * change the desired values, and pass the result to this function.
 *
 * @see SetSoftMaterialParams, GetSoftMaterialParamsField
 */
MOCHI_API void SetSoftMaterialParamsField(
    Actor* actor,
    SoftMaterialParams const& params,
    int elementIndex,
    Error& error);

/**
 * @brief Get the material parameters for a specific element.
 *
 * @param[in] actor Actor to query.
 * @param[in] elementIndex Index of element to query.
 * @param[in,out] error Error status. Check @ref Error::IsOK for success.
 * @return Material parameters of the specified element.
 *
 * @note Only supported for soft actors.
 * @note If the actor has homogeneous material parameters, returns the homogeneous material
 * parameters for any valid element. If the actor has per-element material parameters, returns the
 * parameters for the requested element.
 * @note Returned PSD strategies are the values used for simulation. For example,
 * @ref MaterialPsdStrategy::MaterialDefault is resolved to the material model's default strategy.
 *
 * @see GetSoftMaterialParams, SetSoftMaterialParamsField
 */
[[nodiscard]] MOCHI_API SoftMaterialParams
GetSoftMaterialParamsField(Actor const* actor, int elementIndex, Error& error);

// [Experimental] Function to restore the state of a scene using data from another scene. The
// function receives pointers to both scenes and the handle of the state to be copied. The function
// will return an error if the actor types and components in both scenes do not match, but it will
// not check if the actors and their properties match.
MOCHI_API void
RestoreStateFromScene(Scene* sceneTo, Scene const* sceneFrom, StateHandle handleFrom, Error& error);

// [Experimental] Apply scene settings that maximize convergence of the solver, by (1) ensuring that
// the residual is the true gradient of the merit, and (2) enabling merit-based line search. These
// settings are recommended together with double precision; single precision may not provide the
// necessary accuracy. Some of these settings may hurt stability or degrade performance. The
// function does not change solver tolerances or settings other than the line-search type.
// ContactParams.frictionWithColliderNormal = true. (default)
// EnableNewtonEulerInertia(false). (default)
// SolverParams.experimentalEval.implicitNormalForceForDissipation = false. (default)
// SolverParams.experimentalEval.explicitNormals = true. (non-default)
// SolverParams.nonlinearSolver.lineSearchType = LineSearchType::Armijo. (non-default)
MOCHI_API void ApplyImprovedConvergenceSettings(Scene* scene, Error& error);

// [Batch API] Function for testing performance of batched actor queries across the Python-C++
// boundary. It returns the total contact force (in N, in world frame) produced by 'colliders' on
// 'actors', computed as the sum of norms of forces. All 'actors' must have enabled
// QueryType::TotalContactForce.
[[nodiscard]] MOCHI_API real GetContactForceWorldBatch(
    Span<Actor const* const> actors,
    Span<Actor const* const> colliders,
    Error& error);

//----------------------------------------------------------------------------------------------
// Inverse Kinematics APIs
//----------------------------------------------------------------------------------------------

struct IKSolverParams {
  int maxIter = 20;
  VerbosityLevel verbosity = VerbosityLevel::Warning;
  real absTol = 1e-2_r;
  real relTol = 1e-8_r;
  real positionErrorThres = 1e-2_r;
  real rotationErrorThres = 1e-2_r;
  int lineSearchMaxIter = 10;
  double maxElapsedTimeSeconds = 0;
};

class IKSolver {
 public:
  virtual void SetSolverParams(IKSolverParams const& params) = 0;

  [[nodiscard]] virtual IKSolverParams GetSolverParams() const = 0;

  virtual Constraint* CreatePositionTarget(
      ActorHandle actor,
      Real3 localPosition,
      Real3 targetPosition,
      real weight,
      Error& error) = 0;

  virtual void ClearPositionTarget(ActorHandle actor, Error& error) = 0;

  virtual Constraint* CreateRotationTarget(
      ActorHandle actor,
      Real3 localRotation,
      Real3 targetRotation,
      real weight,
      Error& error) = 0;

  virtual void ClearRotationTarget(ActorHandle actor, Error& error) = 0;

  virtual bool SolveIK(Error& error) = 0;

 protected:
  /// Don't delete, call @ref experimental::DestroyIKSolver instead.
  virtual ~IKSolver() = default;
};

MOCHI_API IKSolver* CreateIKSolver(Scene* scene, Context* context, Error& error);

MOCHI_API void DestroyIKSolver(IKSolver* solver, Context* context, Error& error);

[[nodiscard]] MOCHI_API bool
IsValidIKSolver(IKSolver const* solver, Context* context, Error& error);

#if MOCHI_USE_OSC

/**************************************************************************************************
    Mochi Operational Space Controller. The controller works by 1) linearizing the robot dynamics,
   2) minimizing the distance between the current end-effector position and the target position in
   the task space. User would need to provide the current robot pose and velocity as the estimation
   of real robot or the state fetched from a simulated robot. The OSC would then computes the
   torque/force on the controlled degrees of freedom to achieve the user-desired task-space goal.

   User can also add hard constraints on the robot's velocity and applied force/torque. We then use
   an underlying quadratic program (QP) to solve for the best force/torque that achieves the goal of
   task-space motion while satisfying the constraints, which is otherwise known as QP-based OSC.

   Note: QP-based OSC is not supported under single precision,
   because the QP solver is not guaranteed to converge at reasonably high precision.
*/
struct OperationalSpaceControllerParams {
  // The position target
  Real3 posTarget = {};
  Real3 posVelocityTarget = {};
  real positionKp = 0_r; // The position gain
  real positionKd = 0_r; // The damping gain
  // We limit the position error to avoid excessively large change in control or joint
  real maxPositionError = 1e6_r;
  // Maximal position velocity constraint, whose elements must be zero or positive
  // If this value is positive, QP-OSC is used
  real maxPosVelocity = 0_r;

  // The rotation target
  Quaternion rotTarget = {};
  Real3 rotVelocityTarget = {};
  real rotationKp = 0_r; // The position gain (for rotation)
  real rotationKd = 0_r; // The damping gain (for rotation)
  // We limit the rotation error to avoid excessively large change in control or joint
  real maxRotationError = 1e6_r;
  // Maximal rotation velocity constraint, whose elements must be zero or positive
  // If this value is positive, QP-OSC is used
  real maxRotVelocity = 0_r;

  // joint and control space regularization
  real lambdaTau = 1e-6_r;
  real lambdaJoint = 1e-3_r;

  // Controlled degrees of freedom
  DynamicArray<int> controlledDofs;
  // Maximal force limits constraint, whose elements must be positive
  // If controlLimits is not empty, QP-OSC is used
  // The size of controlLimits can be 0, 1, or controlledDofs.size()
  // If the size of controlLimits is 1, the limit applies to all controlled DoFs
  DynamicArray<real> controlLimits;

  // Implicit OSC parameters
  int maxIter = 1; // Default to explicit OSC
  // Implicit OSC converges when:
  //   1) torque update is smaller than torqueRelTol
  //   2) residual norm is smaller than resAbsTol
  real torqueRelTol = 1e-2_r; // Convergence criteria using relative torque tolerance
  real resAbsTol = 1e-2_r; // Convergence criteria using absolute residual norm
  VerbosityLevel verbosity = VerbosityLevel::Warning;
};

/**
 * @brief Operational Space Controller (OSC) for articulated bodies.
 *
 * @details Compute the torque to be applied to the robot joints so that the end-effector position
 * and rotation can achieve a user-specified target. The OSC uses a one-step linearization of the
 * dynamic system to compute various force terms, such as the gravitational force, and the Coriolis
 * and centrifugal forces. These force terms are compensated in the output torque.
 *
 * @note Users need to create the associated articulated body before creating the OSC, and destroy
 * the OSC before the articulated body.
 *
 * @see experimental::CreateOperationalSpaceController
 * @see experimental::DestroyOperationalSpaceController
 */
class OperationalSpaceController {
 public:
  /**
   * @brief Get the OSC parameters, which includes the position and rotation target (position and
   * velocity), the PD-gains, and the weight of the regularization function in least-square solve.
   *
   * @return OSC parameters
   */
  virtual OperationalSpaceControllerParams const& GetParams() const = 0;

  /**
   * @brief Set the OSC parameters. All the PD-gain coefficients must be zero or positive, and at
   * least one of them must be strictly positive. Otherwise, an error is returned.
   *
   * @param[in,out] error Error status. Check Error::IsOK() for success
   *
   * @see OperationalSpaceController::GetParams
   */
  virtual void SetParams(OperationalSpaceControllerParams const& params, Error& error) = 0;

  /**
   * @brief Get the Newton solver parameters used by the internal QP solver.
   *
   * @return The Newton solver parameters.
   *
   * @see SetNewtonSolverParams, GetQPSolverParams
   */
  [[nodiscard]] virtual NewtonSolverParams const& GetNewtonSolverParams() = 0;

  /**
   * @brief Set the Newton solver parameters used by the internal QP solver.
   *
   * @param[in] params The Newton solver parameters to set.
   *
   * @note Requires using @ref NewtonSolverParams::convergenceMode = @ref
   * NonLinearSolverConvergenceMode::Global.
   *
   * @see GetNewtonSolverParams, SetQPSolverParams
   */
  virtual void SetNewtonSolverParams(NewtonSolverParams const& params) = 0;

  /**
   * @brief Get the QP solver parameters used by the internal QP solver.
   *
   * @return The QP solver parameters
   *
   * @see SetQPSolverParams, GetNewtonSolverParams
   */
  [[nodiscard]] virtual QPSolverParams const& GetQPSolverParams() = 0;

  /**
   * @brief Set the QP solver parameters used by the internal QP solver.
   *
   * @param[in] params The QP solver parameters to set
   *
   * @see GetQPSolverParams, SetNewtonSolverParams
   */
  virtual void SetQPSolverParams(QPSolverParams const& params) = 0;

  /**
   * @brief This is a helper function for initialization. We read the current position target of the
   * robot's end-effector and return it, so user can set it to the parameters. This function is
   * useful to initialize the OSC, letting the robot stay at the current position.
   *
   * @return The current end-effector position
   *
   * @see OperationalSpaceController::SetParams
   */
  virtual Real3 UseCurrentPositionAsTarget() = 0;

  /**
   * @brief This is a helper function for initialization. We read the current rotation target of the
   * robot's end-effector and return it, so user can set it to the parameters. This function is
   * useful to initialize the OSC, letting the robot stay at the current orientation.
   *
   * @return The current end-effector orientation
   *
   * @see OperationalSpaceController::SetParams,
   * OperationalSpaceController::UseCurrentPositionAsTarget
   */
  virtual Quaternion UseCurrentRotationAsTarget() = 0;

  /**
   * @brief This is a helper function for debugging. We output the difference between the current
   * robot end-effector and the target specified by the SetParams function. The first 3 outputs are
   * the position difference and the last 3 outputs are the rotation difference.
   *
   * @param[in] dt the timestep size used for simulation or control signal update
   * @return The difference between robot end-effector and the target
   *
   * @see OperationalSpaceController::SetParams
   */
  virtual Real6 ComputeDeltaXTarget(real dt) const = 0;

  /**
   * @brief This is a helper function for debugging. We assume the robot does not have any dynamics.
   * We update the robot pose by solving the following least square problem:
   *      argmin_{robot_joint_pose}
   *        kp * ||ee_position - delta_position||² + kd * ||ee_velocity - delta_velocity||²
   * where ee_position is the robot's end-effector position (and rotation), and
   * ee_velocity is the robot's end-effector velocity (and angular velocity). By repeatedly calling
   * this function, user can create a kinematic trajectory governed by the PD-gain parameters. This
   * function helps users to visualize the behavior of PD-gains. Note that after solving for the
   * robot_joint_pose, we also compute robot_joint_velocity by finite difference. This is not the
   * true velocity, but create a sense of motion for visualization.
   *
   * @note Error is set when the size of dofs and vel is incorrect (does not match the total number
   * of joint dofs in the robot).
   *
   * @param[in] dt the timestep size used for simulation or control signal update
   * @param[in] dofs the current joint pose
   * @param[in] vel the current joint velocity
   * @param[out] outDofs the updated joint pose
   * @param[out] outVel the updated joint velocity
   * @param[out] error Error status. Check Error::IsOK() for success
   */
  virtual void ComputeUpdatedKinematics(
      real dt,
      Span<real const> dofs,
      Span<real const> vel,
      Span<real> outDofs,
      Span<real> outVel,
      Error& error) = 0;

  /**
   * @brief This is a helper function for debugging. OSC is based on linearized dynamics, in order
   * to compute the torque. However, Mochi is nonlinear as a simulator. In the standard usage, user
   * can take the computed torque and apply it to the nonlinear dynamics. However, in this function,
   * we allow user to directly adopt the linearized dynamics (discard nonlinearity). This function
   * assumes the ComputeTorque function has been called, and it directly use the linearized dynamics
   * to update the robot state. This function matches the behavior of conventional OSC setting,
   * where both the simulator and the controller relies on linearization.
   *
   * @note Error is set when any of the size of outPose, outVel, or outForce is incorrect.
   * Specifically, outPose and outVel should be resized to match the robot DOF. The size of outForce
   * should match the size of controlledDofs in OperationalSpaceControllerParams.
   *
   * @param[out] outPose the updated joint pose
   * @param[out] outVel the updated joint velocity
   * @param[out] outForce the updated torque
   * @param[out] error Error status. Check Error::IsOK() for success
   *
   * @see OperationalSpaceController::SetParams, OperationalSpaceController::GetParams,
   * OperationalSpaceController::ComputeTorque
   */
  virtual void ImitateLinearizedDynamics(
      Span<real> outPose,
      Span<real> outVel,
      Span<real> outForce,
      Error& error) = 0;

  /**
   * @brief The main function of OSC, which takes the current robot pose and velocity, and then
   * computes the torque to bring the end-effector to the target.
   *
   * @note Error is set when any of the size of dofs or vel is incorrect. They should be resized to
   * match the robot DOF.
   *
   * @param[in] dt the timestep size used for simulation or control signal update
   * @param[in] dofs the current joint pose
   * @param[in] vel the current joint velocity
   * @return The output torque, with size matching controlledDofs in
   * OperationalSpaceControllerParams.
   */
  virtual Span<real const>
  ComputeTorque(real dt, Span<real const> dofs, Span<real const> vel, Error& error) = 0;

  /**
   * @brief A helper function for debugging. This function assumes that the following two functions
   * have been called in order:
   *    ComputeTorque()
   *    scene->Step()
   * At this point, calling the following function will print the amount of linearization error on
   * the equation of motion.
   *
   * @note Error is set when any of the size of currPose is incorrect. It should be resized to
   * match the robot DOF.
   *
   * @param[in] currPose the current pose of the robot
   * @param[out] error Error status. Check Error::IsOK() for success
   * @return Both the absolution and relative linearization error
   */
  virtual Real2 GetLinearizationError(Span<real const> currPose, Error& error) = 0;

 protected:
  virtual ~OperationalSpaceController() = default;
};

// Create a new OperationalSpaceController (OSC). The OSC always clones the robot into its own
// internal scene, so the caller's scene may contain any number of other actors. Our goal is to
// control the center of mass transform of the robot's link indexed by linkId.
MOCHI_API OperationalSpaceController*
CreateOperationalSpaceController(Actor* robot, int linkId, Context* context, Error& error);

// Immediately destroy the OSC. If the OSC created an internal scene, that scene is also destroyed.
MOCHI_API void DestroyOperationalSpaceController(
    OperationalSpaceController* controller,
    Context* context,
    Error& error);

// Check if a OSC is valid (not destroyed), and belongs to this Context.
MOCHI_API bool IsValidOperationalSpaceController(
    OperationalSpaceController const* controller,
    Context* context,
    Error& error);

#endif // MOCHI_USE_OSC

//----------------------------------------------------------------------------------------------
// Newton-Euler Comparison APIs
//----------------------------------------------------------------------------------------------

/// @brief Computes the terms of the Newton-Euler equation of motion for an articulated body.
///
/// @details Evaluates the mass matrix, Coriolis/centrifugal forces, Jacobian, and gravity-induced
/// generalized forces that appear in the Euler-Lagrange equation:
///
///     M(q) * ddq + C(q, dq) - J(q)^T * f = tau
///
/// where q is the joint configuration, dq the joint velocity, ddq the joint acceleration,
/// f the external (gravitational) forces, and tau the joint torques.
///
/// These terms are derived from Mochi's implicit formulation, which uses a discrete-then-optimize
/// paradigm rather than the classical Newton-Euler recursive approach. As a result, the extracted
/// M and C are approximate and depend on the time step @p dt.
///
/// Create via @ref CreateNewtonEulerTerms. The object clones the robot into its own internal
/// scene. The internal scene is destroyed when the object is destroyed.
///
/// @see CreateNewtonEulerTerms, DestroyNewtonEulerTerms, EnableNewtonEulerInertia
class MOCHI_API NewtonEulerTerms {
 public:
  /// @brief Compute the Newton-Euler terms for the given joint state.
  ///
  /// @param[in]  dt     Time step [s] used to extract the mass matrix and Coriolis term.
  /// @param[in]  q      Joint positions of size numDofs.
  /// @param[in]  dq     Joint velocities of size numDofs.
  /// @param[out] outM   Mass matrix M(q), stored column-major with size numDofs * numDofs.
  /// @param[out] outC   Coriolis and centrifugal term C(q, dq) of size numDofs.
  /// @param[out] outJ   Jacobian matrix J(q) of size numLinks * numDofs * 6. Pass an empty span to
  ///                     skip the Jacobian computation.
  /// @param[out] outJtF Gravity-induced generalized force J(q)^T * f of size numDofs, where f
  ///                     is the gravitational wrench acting on each link.
  /// @param[out] error  Error status. Set if any span has an incorrect size.
  virtual void Compute(
      real dt,
      Span<real const> q,
      Span<real const> dq,
      Span<real> outM,
      Span<real> outC,
      Span<real> outJ,
      Span<real> outJtF,
      Error& error) = 0;

 protected:
  virtual ~NewtonEulerTerms() = default;
};

/// @brief Create a NewtonEulerTerms object for the given robot.
///
/// @details The robot is cloned into an internal scene. The caller must destroy the returned object
/// via @ref DestroyNewtonEulerTerms.
///
/// @param[in]  robot                    Pointer to the articulated body @ref Actor. Must not be
/// null.
/// @param[in]  context                  The @ref Context that owns this object.
/// @param[out] error                    Error status.
/// @return Pointer to the newly created @ref NewtonEulerTerms, or null on error.
MOCHI_API NewtonEulerTerms* CreateNewtonEulerTerms(Actor* robot, Context* context, Error& error);

/// @brief Immediately destroy a @ref NewtonEulerTerms object.
///
/// If the object created an internal scene, that scene is also destroyed.
MOCHI_API void
DestroyNewtonEulerTerms(NewtonEulerTerms* newtonEulerTerms, Context* context, Error& error);

/// @brief Check if a @ref NewtonEulerTerms object is valid (not destroyed) and belongs to this
/// @ref Context.
MOCHI_API bool
IsValidNewtonEulerTerms(NewtonEulerTerms const* newtonEulerTerms, Context* context, Error& error);

// -------------------------------------------------------------------
// Debugging tools
// -------------------------------------------------------------------

struct DebugStats {
  real maxResidualNormRelativeError = 0.0;
};

[[nodiscard]] MOCHI_API DebugStats GetDebugStats(Scene const* scene, Error& error);

} // namespace mochi::experimental
