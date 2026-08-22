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

/**
 * @brief Parameters for contact mechanics simulation.
 *
 * @note In contact between a colliding actor and a collider, the collider's contact parameters
 * (not the colliding actor's) are used. The exceptions are:
 *   - For friction and dissipation coefficients (viscousFrictionCoefficient,
 *     coulombFrictionCoefficient, normalViscousDampingCoefficient), the geometric mean of the
 *     colliding and collider's coefficients is used. This disables friction/dissipation if either
 *     of them does.
 *   - For penalty coefficient (penaltyCoefficient) and friction velocity threshold
 *     (frictionFalloffVel), the geometric mean of the colliding and collider's values is used,
 *     except if the collider is static in which case the colliding's values are used.
 */
struct ContactParams {
  ContactParams() = default;

  /**
   * @brief Construct ContactParams with all parameters explicitly specified.
   *
   * @param[in] penaltyCoefficient See @ref penaltyCoefficient.
   * @param[in] penaltySmoothingHalfDistance See @ref penaltySmoothingHalfDistance.
   * @param[in] penaltyThresholdDefault See @ref penaltyThresholdDefault.
   * @param[in] penaltyThresholdExtraPadding See @ref penaltyThresholdExtraPadding.
   * @param[in] frictionWithColliderNormal See @ref frictionWithColliderNormal.
   * @param[in] maxAlignmentNormals See @ref maxAlignmentNormals.
   * @param[in] viscousFrictionCoefficient See @ref viscousFrictionCoefficient.
   * @param[in] coulombFrictionCoefficient See @ref coulombFrictionCoefficient.
   * @param[in] frictionFalloffVel See @ref frictionFalloffVel.
   * @param[in] normalViscousDampingCoefficient See @ref normalViscousDampingCoefficient.
   * @param[in] distanceErrorBound See @ref distanceErrorBound.
   * @param[in] objScale See @ref objScale.
   * @param[in] collidingPenaltyLengthScale See @ref collidingPenaltyLengthScale.
   *
   * @note Must take every parameter in order.
   * @note Used when converting types to/from the C API.
   */
  ContactParams(
      real penaltyCoefficient,
      real penaltySmoothingHalfDistance,
      real penaltyThresholdDefault,
      real penaltyThresholdExtraPadding,
      bool frictionWithColliderNormal,
      real maxAlignmentNormals,
      real viscousFrictionCoefficient,
      real coulombFrictionCoefficient,
      real frictionFalloffVel,
      real normalViscousDampingCoefficient,
      real distanceErrorBound,
      real objScale,
      real collidingPenaltyLengthScale);

  // Parameters for penalty contact.

  /**
   * @brief Stiffness of the contact penalty force [Pa/m].
   *
   * @note Must be strictly positive.
   * @note Higher penalties create stiffer contacts and reduce penetration.
   * @note Arbitrarily large penalties may degrade stability.
   * @note The default penalty is appropriate for actors with default density. For actors with much
   * higher/lower density than the default density, the penalty coefficient may need to be
   * increased/decreased accordingly.
   * @note The penalty coefficient used in a collision is the geometric mean of the colliding and
   * collider's coefficients. The exception is if the collider is static, in which case the
   * colliding's penalty is used.
   * @note The penalty coefficient is additionally scaled by length-scale corrections when the
   * colliding or collider integrates contact over a non-2D manifold (e.g., rod, shell). See @ref
   * collidingPenaltyLengthScale.
   */
  real penaltyCoefficient = 1e9_r;

  /**
   * @brief PolyReLU smoothing half-width [m].
   *
   * @details The penalty force transitions from 0 to linear over the decreasing distance range
   * (penaltyThreshold, penaltyThreshold - 2 * @ref penaltySmoothingHalfDistance).
   *
   * @note Must not be negative.
   * @note Larger smoothing distances improve stability but may increase penetration.
   * @note Smoothing distance is expected to be small relative to the collider geometry.
   */
  real penaltySmoothingHalfDistance = 0.005_r;

  /**
   * @brief Default contact detection threshold [m].
   *
   * @details The penalty force transitions from 0 to linear over the decreasing distance range
   * (penaltyThreshold, penaltyThreshold - 2 * @ref penaltySmoothingHalfDistance).
   *
   * @note Negative values are legal.
   * @note If the colliding actor has @ref ColliderType::None or @ref ColliderType::PointCloud,
   * penaltyThreshold = penaltyThresholdDefault + @ref penaltyThresholdExtraPadding. Otherwise,
   * penaltyThreshold = penaltyThresholdDefault.
   *
   * @see penaltyThresholdExtraPadding, GetPenaltyThresholdDist
   */
  real penaltyThresholdDefault = 0.001_r;

  /**
   * @brief Extra padding [m] added to the default contact detection threshold if the colliding
   * actor has @ref ColliderType::None or @ref ColliderType::PointCloud.
   *
   * @note Must not be negative.
   * @note Extra padding is useful to avoid tunneling through thin actors when the other actor has
   * @ref ColliderType::None or @ref ColliderType::PointCloud.
   *
   * @see penaltyThresholdDefault, GetPenaltyThresholdDist
   */
  real penaltyThresholdExtraPadding = 0_r;

  // Parameters for friction

  /**
   * @brief Use collider's normal (normalized SDF gradient) for friction direction if true, or the
   * colliding's surface normal at the sample point if false.
   *
   * @note For co-dimensional colliding actors with ambiguous normals, the collider's normal is
   * always used regardless of frictionWithColliderNormal.
   */
  bool frictionWithColliderNormal = true;

  /**
   * @brief Maximum normal alignment threshold.
   *
   * @details Normal alignment is defined as the dot product between colliding and collider normals.
   * Contact is disabled for sample points whose normal alignment exceeds this threshold. This
   * prevents sample points from being trapped inside the collider when penetration is large.
   *
   * @note Valid range is [-1, 1]. -1 allows contact only for perfectly opposing normals, 1 allows
   * all contacts.
   * @note For co-dimensional colliding actors with ambiguous normals, contact is not disabled
   * regardless of maxAlignmentNormals (normal alignment cannot be computed).
   */
  real maxAlignmentNormals = 0_r;

  /**
   * @brief Viscous friction coefficient [s/m].
   *
   * @note Friction force is proportional to contact force and tangential velocity.
   * @note Must not be negative.
   * @note Both viscousFrictionCoefficient and @ref coulombFrictionCoefficient can be >0.
   * @note The viscous friction coefficient used in a collision is the geometric mean of the
   * colliding and collider's coefficients. This disables viscous friction if either of them does.
   */
  real viscousFrictionCoefficient = 0_r;

  /**
   * @brief Coulomb friction coefficient (dimensionless).
   *
   * @note Must not be negative.
   * @note Both @ref viscousFrictionCoefficient and coulombFrictionCoefficient can be >0.
   * @note The Coulomb friction coefficient used in a collision is the geometric mean of the
   * colliding and collider's coefficients. This disables Coulomb friction if either of them does.
   */
  real coulombFrictionCoefficient = 0.5_r;

  /**
   * @brief Velocity threshold for Coulomb friction smoothing [m/s].
   *
   * @details For C1Regularized, the Coulomb friction force smoothly transitions from 0 to full
   * strength as tangential velocity increases from 0 to frictionFalloffVel (compact support). For
   * CinfRegularized, the force asymptotically approaches full strength with no compact support
   * boundary; frictionFalloffVel controls the regularization scale.
   *
   * @note Must not be negative. For CinfRegularized, a value of zero is clamped internally to
   * avoid numerical issues.
   * @note Smaller velocity thresholds improve physical accuracy but may degrade stability.
   * @note The velocity threshold used in a collision is the geometric mean of the colliding and
   * collider's thresholds. The exception is if the collider is static, in which case the
   * colliding's threshold is used.
   */
  real frictionFalloffVel = 0.01_r;

  /**
   * @brief Normal viscous damping coefficient [s/m].
   *
   * @details Damping force in the normal direction, proportional to the elastic normal contact
   * force and the normal velocity. Analogous to @ref viscousFrictionCoefficient but acting in the
   * normal direction instead of tangentially.
   *
   * @warning The calibration diverges as CoR approaches zero, which may cause numerical problems
   * when approaching fully-inelastic collisions.
   *
   * @note Must not be negative.
   * @note The normal viscous damping coefficient used in a collision is the geometric mean of the
   * colliding and collider's coefficients. This disables normal damping if either of them does.
   * @note The resulting coefficient of restitution (CoR) is velocity-dependent. For a
   * characteristic impact velocity,
   * @c experimental::CalibrateNormalViscousDampingCoefficient computes the coefficient that
   * approximates a target CoR, while @c experimental::EffectiveCoefficientOfRestitution recovers
   * the CoR produced by a coefficient.
   * @note Because the damping force depends on the normal velocity at each contact point, it also
   * introduces rolling resistance: a body rolling on a surface dissipates energy through the
   * differing normal velocities across its contacting region.
   */
  real normalViscousDampingCoefficient = 0_r;

  // Experimental parameters

  /**
   * @brief [Experimental] Padding [m] added to the collider's bounding volume for collision
   * culling.
   *
   * @warning This is an experimental feature. It may be changed or removed in the future. Use at
   * your own risk.
   *
   * @note Useful, for example, with approximate SDFs (e.g., deep flow map) to compensate for
   * potentially overestimating the true distance.
   */
  real distanceErrorBound = 0_r;

  /**
   * @brief [Experimental] Object scale relative to default size (dimensionless). Used by deep flow
   * only.
   *
   * @warning Deep flow is an experimental feature. It may be changed or removed in the future. Use
   * at your own risk.
   */
  real objScale = 1_r;

  /**
   * @brief [Experimental] Length scale [m] used to correct the penalty coefficient when the
   * colliding body integrates contact traction over a lower-than-two-dimensional manifold, such as
   * a rod or point mass. E.g., the penalty is scaled by this value if the colliding body lumps
   * contact tractions on a line, or this value squared if lumping contact forces on a point.
   *
   * @warning Contact with lower-dimensional bodies is an experimental feature. It may be changed or
   * removed in the future. Use at your own risk.
   *
   * @note This value is not used in the most common case, where contact traction is integrated over
   * a two-dimensional surface.
   * @note The colliding body's value is always used in a contact pair, because the colliding body
   * determines the dimension of the contact traction integral.
   */
  real collidingPenaltyLengthScale = 1_r;

  /**
   * @brief Get the total contact detection threshold.
   *
   * @param addPadding If true, includes @ref penaltyThresholdExtraPadding.
   *
   * @return Total contact detection threshold [m].
   */
  real GetPenaltyThresholdDist(bool addPadding) const;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ContactParams const&) const = default;
#endif

  // clang-format off
  MOCHI_STRUCT_BEGIN(mochi::ContactParams)
  MOCHI_FIELD(penaltyCoefficient) MOCHI_ATTRIBUTE(Units("Pa/m"))
  MOCHI_FIELD(penaltySmoothingHalfDistance) MOCHI_ATTRIBUTE(Units("m"))
  MOCHI_FIELD(penaltyThresholdDefault) MOCHI_ATTRIBUTE(Units("m"))
  MOCHI_FIELD(penaltyThresholdExtraPadding) MOCHI_ATTRIBUTE(Units("m"))
  MOCHI_FIELD(frictionWithColliderNormal)
  MOCHI_FIELD(maxAlignmentNormals)
  MOCHI_FIELD(viscousFrictionCoefficient) MOCHI_ATTRIBUTE(Units("s/m"))
  MOCHI_FIELD(coulombFrictionCoefficient)
  MOCHI_FIELD(frictionFalloffVel) MOCHI_ATTRIBUTE(Units("m/s"))
  MOCHI_FIELD(normalViscousDampingCoefficient) MOCHI_ATTRIBUTE(Units("s/m"))
  MOCHI_FIELD(distanceErrorBound) MOCHI_ATTRIBUTE(Units("m"))
  MOCHI_FIELD(objScale)
  MOCHI_FIELD(collidingPenaltyLengthScale) MOCHI_ATTRIBUTE(Units("m"))
  MOCHI_STRUCT_END()
  // clang-format on
};

/**
 * @brief Selects which Coulomb friction smoothing model to use.
 *
 * @see ContactParams::frictionFalloffVel
 */
enum class CoulombFrictionModel {
  /**
   * @brief C1-smoothed Coulomb friction with compact support: friction transitions linearly from 0
   * to full strength over [0, @ref ContactParams::frictionFalloffVel].
   */
  C1Regularized,

  /**
   * @brief C-infinity regularized Coulomb friction with no compact support: friction asymptotically
   * approaches full strength. @ref ContactParams::frictionFalloffVel controls the regularization
   * scale.
   */
  CinfRegularized,

  /** @brief Number of friction model enum values. */
  Count,

  /** @brief Default Coulomb friction model. */
  Default = C1Regularized
};

} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::CoulombFrictionModel)
MOCHI_ENUM_ITEM(C1Regularized)
MOCHI_ENUM_ITEM(CinfRegularized)
MOCHI_ENUM_COUNT(Count)
MOCHI_ENUM_END()

namespace mochi {

/**
 * @brief Collision detection geometry for an @ref Actor.
 *
 * @note This setting controls how OTHER actors detect contact with this actor. It does not affect
 * how this actor detects contact with other actors.
 */
enum class ColliderType {
  /**
   * @brief No collision representation. Other actors cannot detect contact with this actor.
   *
   * @note It does NOT prevent this actor from detecting contact with other actors.
   */
  None,

  /**
   * @brief Automatic collider type selection based on the actor's shape.
   *
   * @note Mochi will select the most appropriate collider type based on the actor's shape and type:
   * - Implicit sphere shapes: @ref ColliderType::Sphere
   * - Implicit plane shapes: @ref ColliderType::Plane
   * - Mesh shapes on volumetric actors (e.g., rigid, soft): @ref ColliderType::Sdf
   * - Shell and rod actors: @ref ColliderType::PointCloud
   *
   * @note If Auto is set on an actor, a valid collider type will be assigned even if the shape's
   * default collider type is @ref ColliderType::None. If you want @ref ColliderType::None, set
   * @ref ColliderType::None explicitly as your collider type.
   *
   * @note After initialization, @ref Actor::GetColliderType will return the selected type, not
   * @ref ColliderType::Auto.
   */
  Auto,

  /**
   * @brief Represent the actor by an approximating sphere derived from its bounding volume.
   *
   * @note For shapes with a sphere bounding volume, that sphere is used. For shapes with an
   * OBB/AABB bounding volume, the inscribed sphere is used. Otherwise, a true bounding sphere is
   * computed.
   * @note Fast but only accurate for spherical geometries.
   * @note Only supported for rigid actors and articulated links.
   */
  Sphere,

  /**
   * @brief Represent the actor by its axis-aligned bounding box (AABB), in the actor's local frame.
   *
   * @note Fast but only accurate for box geometries.
   * @note Only supported for rigid actors and articulated links.
   */
  Box,

  /**
   * @brief [Experimental] Represent the actor by its surface mesh.
   *
   * @warning This is an experimental feature. It may be changed or removed in the future. Use at
   * your own risk.
   *
   * @note Performance is slow.
   * @note Only supported for rigid actors and articulated links.
   */
  Mesh,

  /**
   * @brief Represent the actor as an infinite plane.
   *
   * @note Only supported for static rigid actors whose shape is an implicit plane.
   */
  Plane,

  /**
   * @brief Grid-based Signed Distance Field (SDF) collision representation.
   *
   * @note Uses a 3D grid storing the distance to the closest point in the actor's surface.
   * @note Accurate for closed non-intersecting geometries provided the grid resolution is
   * sufficient.
   * @note If the grid is not baked in the actor's Shape, it is computed at initialization
   * (expensive).
   * @note Also supported for soft actors via SDF mapping.
   *
   * @see GridSdfParams
   */
  Sdf,

  /**
   * @brief Point-cloud collision representation.
   *
   * @details Contacts are detected by proximity queries between collider-side sample points and
   * colliding-side sample points. Only other point-cloud colliders can collide with this type.
   */
  PointCloud,

  /** @brief Number of collider type enum values. */
  Count
};
} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::ColliderType)
MOCHI_ENUM_ITEM(None)
MOCHI_ENUM_ITEM(Auto)
MOCHI_ENUM_ITEM(Sphere)
MOCHI_ENUM_ITEM(Box)
MOCHI_ENUM_ITEM(Mesh)
MOCHI_ENUM_ITEM(Plane)
MOCHI_ENUM_ITEM(Sdf)
MOCHI_ENUM_ITEM(PointCloud)
MOCHI_ENUM_COUNT(Count)
MOCHI_ENUM_END()

#include "contact_params_inl.h"
