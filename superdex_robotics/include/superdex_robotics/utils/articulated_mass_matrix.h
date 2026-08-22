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

#include <mochi_physics/mochi_physics.h>
#include <superdex_physics.h>

#include <mochi_core/utils/dynamic_array.h>

namespace superdex::robotics {

/* @brief Per-link inertia data in the link's body frame.
 *
 * Constant in the joint configuration q -- build once at controller initialization with
 * @ref BuildArticulatedLinkInertias and reuse across calls. */
struct ArticulatedLinkInertia {
  /* Total link mass [kg]. */
  real mass = 0_r;
  /* Center of mass in the link's body frame [m]. */
  Real3 comLocal = {};
  /* 3x3 moment of inertia about the COM, in the link's body frame [kg.m^2].
   * Symmetric. Standard convention: diagonals are integral(y^2 + z^2)dm etc., off-diagonals are
   * negative products of inertia (-integral(xy)dm etc.). Matches URDF / Featherstone. */
  Matrix3x3r inertiaCom = {};
};

/* @brief Build per-link inertia data for an articulated actor.
 *
 * Walks @ref Actor::GetNestedLinkActors and reads mass, COM, and the 3x3 inertia tensor at COM
 * from each link. The resulting data is constant in q -- intended to be cached at controller
 * initialization, not recomputed per step.
 *
 * @param[in]     articulatedActor     Articulated actor to introspect (non-null,
 * ActorType::Articulated).
 * @param[out]    outLinkInertias      Cleared and populated with one entry per nested link.
 * @param[in,out] error                Error status. */
MOCHI_API void BuildArticulatedLinkInertias(
    Actor* articulatedActor,
    DynamicArray<ArticulatedLinkInertia>& outLinkInertias,
    superdex::Error& error);

/* @brief Compute the joint-space mass matrix M(q) for an articulated actor.
 *
 * @details Uses the kinetic-energy form of CRBA:
 *
 *     M(q) = sum over links i of  J_i^T * G_i^world * J_i
 *
 * where J_i is link i's 6 x numDofs spatial Jacobian (world frame, [linear; angular] row layout),
 * and G_i^world is the 6x6 spatial inertia of link i taken AT ITS COM expressed in world frame:
 *
 *     G_i^world = [ m_i * I_3              0          ]
 *                 [    0           R_i * I_C * R_i^T  ]
 *
 * with R_i = world rotation of link i and I_C = 3x3 inertia tensor about COM in link's body frame.
 *
 * Mochi's per-link Jacobian gives the linear velocity AT THE LINK'S COM, so the spatial inertia
 * must also be taken at the COM -- the off-diagonal blocks are zero and there is no parallel-axis
 * term. This holds structurally, not just empirically: @c CreateRestTransforms
 * (mochi_core/src/articulated_body/articulated_body.cpp) bakes each link's @c comLocal into the
 * rest transforms, so the articulation's internal "bone" frame that the Jacobian differentiates is
 * COM-centered (see the @c ArticulatedRestTransform docs, which describe it as the center-of-mass
 * frame). Mochi relies on the same property itself, computing joint-space gravity as
 * sum(J_linear^T * m_i * g) in @ref experimental::GetNewtonEulerTerms.
 *
 * The per-link Jacobian is fetched inside the loop via @ref Actor::GetArticulatedJacobian on each
 * nested link actor (no public accessor returns the full stacked body Jacobian).
 *
 * @param[in]     articulatedActor Articulated actor whose nested link actors expose per-link
 *                                Jacobians via @ref Actor::GetArticulatedJacobian. Must be the
 *                                same actor used to build @p linkInertias and produce
 *                                @p worldFromLinks.
 * @param[in]     linkInertias    Per-link inertia data from @ref BuildArticulatedLinkInertias.
 *                                Must contain numLinks entries.
 * @param[in]     worldFromLinks  Per-link world transforms (size numLinks) from
 *                                @ref Actor::GetArticulatedLinkTransforms. These are link-ORIGIN
 *                                transforms, which differ from the COM frame the Jacobian uses by
 *                                R * comLocal. Only their rotation is read, and the two frames
 *                                share it, so no conversion is needed or wanted here.
 * @param[in]     numDofs         Total articulated DOF count. M will be numDofs x numDofs.
 * @param[out]    outM            Row-major numDofs x numDofs mass matrix (size numDofs * numDofs).
 *                                Symmetric and positive-definite for non-degenerate inertias.
 * @param[in,out] error           Error status.
 *
 * @note Rigid-body inertia only. A joint's @ref ArticulatedJointParams::inertia (rotor/armature)
 * is stored separately and added to the diagonal by the solver, so it is absent from M here; add
 * it yourself if you need the diagonal the solver actually sees. */
MOCHI_API void ComputeArticulatedMassMatrix(
    Actor* articulatedActor,
    Span<ArticulatedLinkInertia const> linkInertias,
    Span<TransformRT const> worldFromLinks,
    int numDofs,
    Span<real> outM,
    superdex::Error& error);

} // namespace superdex::robotics
