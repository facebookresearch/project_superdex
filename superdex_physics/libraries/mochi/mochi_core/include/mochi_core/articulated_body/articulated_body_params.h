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
#include <mochi_core/utils/transform_rt.h>

namespace mochi {

/**
 * @brief Degree-of-freedom layout information for a single joint in an articulated body.
 *
 * @details Describes the offset and sizes of translation and rotation DoFs for a joint within
 * the flattened DoF array of the articulated body.
 */
struct ArticulatedDofInfo {
  /** @brief Offset of this joint's DoFs in the flattened DoF array. */
  int offset = -1;
  /** @brief Number of translational DoFs for this joint. */
  int transSize = 0;
  /** @brief Number of rotational DoFs for this joint. */
  int rotSize = 0;

  /**
   * @brief Offset of this joint's translational DoFs in the flattened DoF array.
   *
   * @return The translational DoF offset (equal to @ref offset).
   */
  int GetTransOffset() const;

  /**
   * @brief Offset of this joint's rotational DoFs in the flattened DoF array.
   *
   * @return The rotational DoF offset (@ref offset + @ref transSize).
   */
  int GetRotOffset() const;

  /**
   * @brief Total number of DoFs for this joint.
   *
   * @return The total DoF count (@ref transSize + @ref rotSize).
   */
  int GetSize() const;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ArticulatedDofInfo const&) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::ArticulatedDofInfo)
  MOCHI_FIELD(offset)
  MOCHI_FIELD(transSize)
  MOCHI_FIELD(rotSize)
  MOCHI_STRUCT_END()
};

/**
 * @brief Types of joints in an articulated body.
 */
enum class ArticulatedJointType {
  Free, ///< 6-DoF joint. Unconstrained relative motion between links.
  Prismatic, ///< 1-DoF translational joint along a single axis.
  Revolute, ///< 1-DoF rotational joint around a single axis.
  Spherical, ///< 3-DoF rotational joint (ball-and-socket).
  Hard, ///< 0-DoF rigid (weld) joint. No relative motion between links.
  Cycle, ///< Cycle-closing joint in closed-loop topologies.
  Count,
  Invalid = Count,
};

} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::ArticulatedJointType)
MOCHI_ENUM_ITEM(Free)
MOCHI_ENUM_ITEM(Prismatic)
MOCHI_ENUM_ITEM(Revolute)
MOCHI_ENUM_ITEM(Spherical)
MOCHI_ENUM_ITEM(Hard)
MOCHI_ENUM_ITEM(Cycle)
MOCHI_ENUM_COUNT(Count)
MOCHI_ENUM_ITEM(Invalid)
MOCHI_ENUM_END()

namespace mochi {

/**
 * @brief Defines a cycle-closing joint in an articulated body.
 *
 * @details Cycle joints create closed loops in the kinematic chain, allowing more complex
 * topologies beyond simple tree structures.
 */
struct ArticulatedCycleJoint {
  /** @brief Child link index. */
  int child = 0;

  /** @brief Parent link index. */
  int parent = 0;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ArticulatedCycleJoint const&) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::ArticulatedCycleJoint)
  MOCHI_FIELD(child)
  MOCHI_FIELD(parent)
  MOCHI_STRUCT_END()
};

/**
 * @brief Per-joint friction parameters for articulated bodies.
 *
 * @details Viscous and dry friction force/torque contributions are added together.
 * - Viscous friction force/torque is proportional to velocity with coefficient @ref viscous.
 * - The static-friction peak/breakaway force/torque is given by @ref coulomb + @ref stictionExtra.
 *   Static friction is regularized to allow slight slippage at speeds of up to @ref falloffVel.
 * - Above the smoothing threshold |v| > @ref falloffVel, dynamic dry friction force/torque
 *   (excluding additional viscous friction) has magnitude
 *   @ref coulomb + @ref stictionExtra * exp(-pow((|v| - @ref falloffVel) / @ref stribeckVel, 2))
 *   when @ref stribeckVel > 0. If @ref stribeckVel = 0, the Stribeck term is omitted and the
 *   magnitude is @ref coulomb. Here |v| is the magnitude of the relative (linear or angular)
 *   velocity. This causes friction to decrease smoothly from the peak static value to @ref
 *   coulomb at high velocities.
 */
struct ArticulatedJointFrictionParams {
  /**
   * @brief Viscous friction coefficient [N·s/m or N·m·s/rad].
   *
   * @note Default is zero (no viscous friction).
   */
  real viscous = 0_r;

  /**
   * @brief Coulomb friction coefficient [N or N·m].
   *
   * @note Default is zero (no Coulomb friction).
   */
  real coulomb = 0_r;

  /**
   * @brief Velocity threshold for dry friction smoothing [m/s or rad/s].
   *
   * @details The dry friction force/torque smoothly transitions from 0 to the peak value as
   * (linear or angular) velocity increases from 0 to @ref falloffVel.
   *
   * @note Smaller velocity thresholds improve physical accuracy but may degrade stability.
   */
  real falloffVel = 1e-3_r;

  /**
   * @brief [Experimental] Extra stiction force/torque [N or N·m] representing the difference
   * between static and dynamic friction.
   *
   * @warning This is an experimental feature. It may be changed or removed in the future. Use at
   * your own risk.
   * @warning Nonzero values may harm convergence.
   * @warning Nonzero extra stiction with zero Stribeck velocity results in a nonsmooth force.
   *
   * @note Must not be negative. The @ref coulomb represents dynamic friction in the high-velocity
   * limit, and peak static friction is @ref coulomb + @ref stictionExtra.
   * @note Defaults to zero (no difference between static and dynamic friction).
   */
  real stictionExtra = 0_r;

  /**
   * @brief [Experimental] Stribeck velocity [m/s or rad/s] governing how sharply friction
   * transitions from static to dynamic coefficients as velocity increases.
   *
   * @warning This is an experimental feature. It may be changed or removed in the future. Use at
   * your own risk.
   * @warning Nonzero values may harm convergence.
   * @warning Nonzero extra stiction with zero Stribeck velocity results in a nonsmooth force.
   *
   * @note Must not be negative. A smaller value means a sharper transition.
   * @note Defaults to zero.
   * @note This has no effect if @ref stictionExtra is zero.
   */
  real stribeckVel = 0_r;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(ArticulatedJointFrictionParams const&) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::ArticulatedJointFrictionParams)
  MOCHI_FIELD(viscous)
  MOCHI_FIELD(coulomb)
  MOCHI_FIELD(falloffVel)
  MOCHI_FIELD(stictionExtra)
  MOCHI_FIELD(stribeckVel)
  MOCHI_STRUCT_END()
};

/**
 * @brief Selects how a @ref RoutingElement contributes to a spatial tendon's routing.
 */
enum class RoutingElementType {
  /// A point fixed in a link's local frame (the same frame in which the link's mesh and
  /// geometry are authored, i.e. the link's root-transform frame). Adjacent waypoints are joined
  /// by a straight segment whose length contributes to the tendon length.
  Waypoint,
  /// A constant moment arm contributing `coefficient * jointCoordinate`, as in a linear
  /// transmission. Carries no geometry and breaks the polyline between waypoints.
  LinearJoint,
};

} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::RoutingElementType)
MOCHI_ENUM_ITEM(Waypoint)
MOCHI_ENUM_ITEM(LinearJoint)
MOCHI_ENUM_END()

namespace mochi {

/**
 * @brief One ordered element in a spatial tendon's heterogeneous routing list.
 *
 * @details A @ref RoutingElementType::Waypoint uses @ref index (a link index) and @ref
 * localPosition; a @ref RoutingElementType::LinearJoint uses @ref index (a joint index) and @ref
 * coefficient. Order matters: a segment forms only between adjacent waypoints, so a linear-joint
 * element between two waypoints leaves a gap.
 */
struct RoutingElement {
  /// Selects the interpretation of the remaining fields.
  RoutingElementType type = RoutingElementType::Waypoint;

  /// Waypoint: the link index the waypoint is attached to. LinearJoint: the joint index.
  int index = 0;

  /// Waypoint only: the waypoint position in the link's local frame (the same frame in which
  /// the link's mesh and geometry are authored, i.e. the link's root-transform frame) [m].
  Real3 localPosition = {};

  /// LinearJoint only: the signed constant moment arm d(displacement)/d(joint DoF)
  /// [m / joint DoF]; its sign sets whether the tendon lengthens or shortens with the joint DoF.
  real coefficient = 0_r;

#if MOCHI_LANGUAGE_CPP20
  bool operator==(RoutingElement const&) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::RoutingElement)
  MOCHI_FIELD(type)
  MOCHI_FIELD(index)
  MOCHI_FIELD(localPosition)
  MOCHI_FIELD(coefficient)
  MOCHI_STRUCT_END()
};

} // namespace mochi

#include "articulated_body_params_inl.h"
