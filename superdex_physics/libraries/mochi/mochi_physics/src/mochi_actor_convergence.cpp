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

#include "mochi_actor_convergence.h"

#include "mochi_articulated_body.h"
#include "mochi_common_components.h"
#include "mochi_discretization_components.h"
#include "mochi_group.h"
#include "mochi_rod.h"
#include "mochi_rom_jacobian.h"
#include "mochi_shell.h"
#include "mochi_soft.h"

#include <mochi_core/element_operations/fem_rod.h>
#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/decomposition_utils.h>
#include <mochi_core/utils/lie.h>
#include <mochi_core/utils/rigid_body_size.h>
#include <mochi_core/utils/rigid_body_utils.h>

#include <array>
#include <cstddef>

/**
 * @file mochi_actor_convergence.cpp
 * @brief Per-DoF convergence weights for per-actor weighted L2 residual norm.
 *
 * @details
 * ## Overview
 *
 * Convergence is assessed via a weighted L2 norm: ||r||_W = √(Σᵢ wᵢ·rᵢ²). An actor converges when
 * ||r||_W ≤ absTol or ||r||_W ≤ relTol·||r₀||_W.
 *
 * ## Design Goals
 *
 * 1. O(1) norm under characteristic loading: When an actor experiences uniform acceleration aRef
 *    (gravity magnitude, clamped ≥1 m/s²), the weighted residual norm is O(1) (or exactly 1 for
 *    some actors). This makes tolerances meaningful regardless of actor mass, size, or mesh
 *    resolution.
 *
 * 2. Dimensional bridging: Weights transform heterogeneous residual components (e.g., forces [N]
 *    vs. torques [N·m]) into a non-dimensional sum, enabling physically consistent convergence
 *    checks across mixed DoF types.
 *
 * 3. Zero-inertia DoFs (e.g., quasi-static problems) fall back to unit weights, providing
 *    convergence checking without physical normalization.
 *
 * ## Weight Formulation
 *
 * Weights have the general form: wᵢ = 1 / (Q_ref · Q_{c,i})
 *
 * - Q_{c,i}: Characteristic generalized force for DoF `i`, i.e., the load that produces aRef (force
 *   f_c for translational DoFs, torque τ_c for rotational DoFs).
 * - Q_ref: Actor-level reference generalized force (e.g., total mass × aRef for soft actors).
 *
 * The normalization by Q_ref ensures that under characteristic loading, ||r||_W is O(1) independent
 * of mesh resolution.
 *
 * See per-actor-type functions below for specific formulations.
 */

// TODO:
// - Improve the formulation for twist DoF weights of rod actors. Consider treating each rod element
//   as a rigid body in 3D space, computing its full inertia tensor (diagonal in element-adapted
//   frame), and using the maximum characteristic torque from that. This would provide a purely
//   inertial twist weight, eliminating the curvature-based branch.
// - Introduce stiffness-based weights for quasi-static simulation, defining characteristic forces
//   from elastic and geometric properties.

namespace mochi {

/**
 * @brief Principal matrix square root of a 3x3 symmetric PSD matrix, via analytical
 * eigendecomposition.
 */
static VMatrix3x3r SqrtSym3x3(VSymMatrix3x3r const& sym) {
  Vec4r lambda MOCHI_NO_INIT;
  VMatrix3x3r VT MOCHI_NO_INIT;
  AnalyticalEigendecompSym3x3(sym, lambda, &VT);
  auto const sqrtLambda = Sqrt(Max(Vec4r{}, lambda));
  return Dot3x3(
      Transpose3x3(VT),
      VMatrix3x3r{sqrtLambda[0] * VT[0], sqrtLambda[1] * VT[1], sqrtLambda[2] * VT[2]});
}

/**
 * @brief Compute per-DoF residual weights for a rigid actor.
 *
 * @details 6 DoFs: 3 translational + 3 rotational. No Q_ref normalization.
 * - Translational: w = 1/f²_c, where f_c = m·aRef.
 * - Rotational: w = 1/τ²_c, where τ_c = aRef·√m·(√I_world)_{ii}, I_world = R·I_local·R^T.
 * - ||r||_W ≈ 1 under characteristic loading.
 *
 * @note Rotational derivation: The characteristic torque induces linear acceleration aRef at the
 * radius of gyration k = √(I/m). From τ = I·α and α = aRef/k, it follows that τ = aRef·√(m·I). For
 * non-diagonal inertia, this generalizes to τ_c = aRef·√m·√I via the principal matrix square root.
 * The diagonal elements (√I_world)_{ii} give the per-axis characteristic torques. MOI must be in
 * the world frame to match the world-frame rotational residual DoFs.
 */
static void GetRigidActorResidualWeights(
    entt::registry const& reg,
    entt::entity actor,
    real aRef,
    ColumnVector<real>& outWeights) {
  MOCHI_ASSERT_VERBOSE(
      reg.all_of<TagRigidActor>(actor) && !reg.any_of<TagStaticActor>(actor),
      "Expected rigid dynamic actor.");
  MOCHI_ASSERT_VERBOSE(aRef > 0_r, "Characteristic acceleration must be positive.");

  auto const& inertia = reg.get<CRigidBodyInertia>(actor);
  real const mass = inertia.GetMass();
  auto const& rootTransform = reg.get<CRootTransform>(actor);
  auto const moiWorld =
      RotateInertia(inertia.GetMomentOfInertiaLocal(), rootTransform.worldFromLocal.GetRotation());

  outWeights.Resize(RigidSize::kDAll);

  // Translational DoFs: f_c = m·aRef.
  real const fRef = mass * aRef;
  real const forceWeight = (fRef > 0_r) ? (1_r / Sqr(fRef)) : 1_r;
  for (int i = 0; i < RigidSize::kDTrans; ++i) {
    outWeights(i) = forceWeight;
  }

  // Rotational DoFs: τ_c = aRef·√m·(√I_world)_{ii} via principal matrix square root.
  auto const sqrtI = SqrtSym3x3(SimdFullToSym(moiWorld));
  for (int i = 0; i < RigidSize::kDRot; ++i) {
    real const characteristicTorque = aRef * Sqrt(mass) * sqrtI[i][i];
    outWeights(RigidSize::kDTrans + i) =
        (characteristicTorque > 0_r) ? (1_r / Sqr(characteristicTorque)) : 1_r;
  }
}

/**
 * @brief Per-DoF residual weights from per-DoF masses: w_i = 1/(M·m_i·aRef²), or 1 where the scale
 * or mass is non-positive.
 */
[[nodiscard]] static ColumnVector<real>
ResidualWeightsFromMasses(Span<real const> masses, real totalMass, real aRef) {
  MOCHI_ASSERT_VERBOSE(aRef > 0_r, "Characteristic acceleration must be positive.");
  int const numDofs = isize(masses);
  real const coeff = (totalMass > 0_r) ? (1_r / (Sqr(aRef) * totalMass)) : 0_r;
  ColumnVector<real> weights(numDofs);
  for (int i = 0; i < numDofs; ++i) {
    weights(i) = (coeff > 0_r && masses[i] > 0_r) ? (coeff / masses[i]) : 1_r;
  }
  return weights;
}

/**
 * @brief Compute per-DoF residual weights for a ROM actor.
 *
 * @details Reduced DoFs via Jacobian J.
 * - Weight: w_j = 1/(M·M_j·aRef²), where M is the total mass and M_j = diag(Jᵀ·diag(m)·J)_j is
 *   the generalized mass of the j-th DoF.
 * - Approximately hyper-reduction invariant: cubature weight scaling cancels in M_j.
 * - ||r||_W ≈ 1 under characteristic loading.
 *
 * @note Hyper-reduction invariance: Non-active node rows of J are zero, while active node lumped
 * masses are upscaled by cubature weights (~1/α). The 1/α in the mass cancels the α from
 * subsampling, so M_j — and thus the weighted residual norm — is independent of subsampling ratio.
 */
static ColumnVector<real>
GetRomActorResidualWeights(entt::registry const& reg, entt::entity actor, real aRef) {
  MOCHI_ASSERT_VERBOSE((reg.all_of<TagSoftActor, TagRomActor>(actor)), "Expected ROM actor.");

  auto const& romJac = reg.get<rom::CRomJacobian const>(actor);
  auto const& lumpedMass = reg.get<CLumpedMassMatrix const>(actor);
  auto const* activeNodes = reg.try_get<CActiveUniqueNodes>(actor);

  auto const& J = romJac.Get<rom::CRomJacobianTypes::DenseT>();
  int const fullDofs = J.Rows();
  int const reducedDofs = J.Cols();
  MOCHI_ASSERT_VERBOSE(
      isize(lumpedMass.values) == fullDofs, "Inconsistent lumped mass and ROM Jacobian size.");

  // Generalized masses: M_j = Σᵢ mᵢ · J²_{ij}
  constexpr std::size_t kStackSizeBytes = 1024 * 4;
  MOCHI_FILO_STACK_ALLOCATOR(allocator, kStackSizeBytes);
  DynamicArray<real> generalizedMasses(reducedDofs, 0_r, &allocator);
  auto accumulateDof = [&](int i) {
    real const mi = lumpedMass.values[i];
    for (int j = 0; j < reducedDofs; ++j) {
      generalizedMasses[j] += mi * Sqr(J(i, j));
    }
  };
  if (activeNodes != nullptr) {
    // Hyper-reduction.
    constexpr int kDofsPerNode = 3;
    for (int const nodeId : activeNodes->ViewIds()) {
      int const dofStart = nodeId * kDofsPerNode;
      for (int d = 0; d < kDofsPerNode; ++d) {
        accumulateDof(dofStart + d);
      }
    }
  } else {
    for (int i = 0; i < fullDofs; ++i) {
      accumulateDof(i);
    }
  }

  return ResidualWeightsFromMasses(
      MakeConstSpan(generalizedMasses), soft::GetActorMass(reg, actor), aRef);
}

/**
 * @brief Compute per-DoF residual weights for a soft actor.
 *
 * @details 3 DoFs per node.
 * - Weight: w_i = 1/(M·m_i·aRef²), where M = total mass, m_i = lumped node mass.
 * - ||r||_W = 1 under characteristic loading.
 *
 * @note Delegates to @ref GetRomActorResidualWeights for ROMs.
 */
static ColumnVector<real>
GetSoftActorResidualWeights(entt::registry const& reg, entt::entity actor, real aRef) {
  MOCHI_ASSERT_VERBOSE(reg.all_of<TagSoftActor>(actor), "Expected soft actor.");

  if (reg.any_of<TagRomActor>(actor)) {
    return GetRomActorResidualWeights(reg, actor, aRef);
  }

  return ResidualWeightsFromMasses(
      MakeConstSpan(reg.get<CLumpedMassMatrix>(actor).values),
      soft::GetActorMass(reg, actor),
      aRef);
}

/**
 * @brief Compute per-DoF residual weights for a shell actor.
 *
 * @details 3 DoFs per node. Same formulation as soft actors.
 * - Weight: w_i = 1/(M·m_i·aRef²), where M = total mass, m_i = lumped node mass.
 * - ||r||_W = 1 under characteristic loading.
 */
static ColumnVector<real>
GetShellActorResidualWeights(entt::registry const& reg, entt::entity actor, real aRef) {
  MOCHI_ASSERT_VERBOSE(reg.all_of<TagShellActor>(actor), "Expected shell actor.");

  return ResidualWeightsFromMasses(
      MakeConstSpan(reg.get<CLumpedMassMatrix>(actor).values),
      shell::GetActorMass(reg, actor),
      aRef);
}

/**
 * @brief Compute per-DoF residual weights for a rod actor.
 *
 * @details 4 DoFs per node: 3 displacement + 1 twist.
 * - Displacement: w = 1/(M·m_i·aRef²), same as soft and shell actors.
 * - Twist: max of two criteria:
 *   (a) Curvature-based: w = κ²_eff/(M·m_elem·aRef²), where κ_eff is an effective curvature
 *   binormal magnitude. Measures how twist couples to lateral force. Dominates for highly curved
 *   rods where twist accuracy substantially affects position accuracy.
 *   (b) Inertia-based: w = 1/(I_total·I_i·(aRef/k)²), where I is rotational inertia and k is
 *   gyration radius. Measures direct twist solution accuracy. Dominates for straight rods under
 *   torsional load, where (a) gives zero weight.
 *   Taking the max ensures twist convergence is monitored regardless of configuration.
 * - ||r||_W ≈ 1 under characteristic loading.
 *
 * @note Topology: For open rods, numElements == numNodes - 1 and the last twist DoF is a dummy
 * (weight 0). For closed-loop (periodic) rods, numElements == numNodes, every twist DoF is real,
 * and element-wise quantities wrap around at the seam.
 */
static ColumnVector<real>
GetRodActorResidualWeights(entt::registry const& reg, entt::entity actor, real aRef) {
  MOCHI_ASSERT_VERBOSE(reg.all_of<TagRodActor>(actor), "Expected rod actor.");
  MOCHI_ASSERT_VERBOSE(aRef > 0_r, "Characteristic acceleration must be positive.");

  auto const& dofInfo = reg.get<CActorDofInfo>(actor);
  auto const& nodalMasses = reg.get<CNodalMasses>(actor);
  auto const& elementRotationalInertias = reg.get<CElementRotationalInertias>(actor);
  auto const& material = reg.get<CRodMaterialParams>(actor);
  auto const& polyline = reg.get<CPolylineMesh>(actor);

  int const numDofs = dofInfo.dofsSize;
  int const numNodes = isize(nodalMasses.values);
  bool const isClosedLoop = polyline.isClosedLoop;
  MOCHI_ASSERT_VERBOSE(
      numDofs == numNodes * fem::kNumRodFields, "Unexpected DoF count for rod actor.");
  MOCHI_ASSERT_VERBOSE(
      isize(elementRotationalInertias.values) == (isClosedLoop ? numNodes : numNodes - 1),
      "Inconsistent number of nodes and elements for rod actor.");
  MOCHI_ASSERT_VERBOSE(material.linearDensity > 0_r, "Linear density must be positive.");
  MOCHI_ASSERT_VERBOSE(
      material.linearRotationalInertia > 0_r, "Rotational inertia must be positive.");

  // --- Curvature-based twist weights ---

  auto const& refCurvatureBinormal = reg.get<CReferenceNodeCurvatureBinormal>(actor);
  auto const& currPose = reg.get<CRodPose<TimeStep::Current>>(actor);

  constexpr std::size_t kStackSizeBytes = 1024 * 8;
  MOCHI_FILO_STACK_ALLOCATOR(allocator, kStackSizeBytes);
  DynamicArray<Real3> currentCurvatureBinormal(numNodes, Real3{}, &allocator);
  real eiOverGj = 0_r;
  if (material.torsionalStiffness > 0_r) {
    DynamicArray<Real3> currentNodes(&allocator);
    currentNodes.resize_noinit(numNodes);
    for (int k = 0; k < numNodes; ++k) {
      Vec4r const p = Load<3, Vec4r>(polyline.nodes[k].data()) +
          Load<3, Vec4r>(&currPose.value.displacements[fem::kNumRodFields * k]);
      Store<3>(currentNodes[k].data(), p);
    }
    rod::ComputeRodNodeCurvatureBinormals(
        MakeConstSpan(currentNodes), isClosedLoop, MakeSpan(currentCurvatureBinormal));

    eiOverGj = Max(material.flexuralStiffness) / material.torsionalStiffness; // max(EI)/GJ
  }

  // --- Inertia-based twist weights ---

  real const totalMass = rod::GetActorMass(reg, actor);
  real const forceScaling = (totalMass > 0_r) ? (1_r / (Sqr(aRef) * totalMass)) : 0_r;
  real const gyrRadius = Sqrt(material.linearRotationalInertia / material.linearDensity);
  real const totalRotInertia = HSum(MakeConstSpan(elementRotationalInertias.values));
  real const rotScaling = 1_r / (Sqr(aRef / gyrRadius) * totalRotInertia);

  // --- Assemble weights ---
  ColumnVector<real> weights(numDofs);
  for (int nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx) {
    int const baseDof = nodeIdx * fem::kNumRodFields;

    // Displacement DoFs: same as soft/shell.
    real const nodalMass = nodalMasses.values[nodeIdx];
    real const forceWeight =
        (forceScaling > 0_r && nodalMass > 0_r) ? (forceScaling / nodalMass) : 1_r;
    weights(baseDof + 0) = forceWeight;
    weights(baseDof + 1) = forceWeight;
    weights(baseDof + 2) = forceWeight;

    // Twist DoF: max of curvature-based and inertia-based weights.
    // For open rods, the last node has no associated element. Its twist DoF is a dummy.
    // For closed-loop rods, every node owns the element that starts at it (wraparound).
    if (!isClosedLoop && nodeIdx + 1 == numNodes) {
      weights(baseDof + 3) = 0_r; // Dummy twist DoF at last node of open rod.
      continue;
    }

    int const nextNode = polyline.ElementNodes(nodeIdx)[1];

    // Effective curvature coupling factor: κ_eff = ||κ|| + (max(EI)/GJ) ||κ - κ_ref||
    real const elemMass = 0.5_r * (nodalMasses.values[nodeIdx] + nodalMasses.values[nextNode]);
    Real3 const kappaElem =
        0.5_r * (currentCurvatureBinormal[nodeIdx] + currentCurvatureBinormal[nextNode]);
    Real3 const kappaRefElem =
        0.5_r * (refCurvatureBinormal.values[nodeIdx] + refCurvatureBinormal.values[nextNode]);
    real const effectiveCurvature = Norm(kappaElem) + eiOverGj * Norm(kappaElem - kappaRefElem);

    real const curvatureCouplingWeight = (forceScaling > 0_r && elemMass > 0_r)
        ? (forceScaling * Sqr(effectiveCurvature) / elemMass)
        : 0_r;
    real const rotInertiaWeight = rotScaling / elementRotationalInertias.values[nodeIdx];
    MOCHI_ASSERT_VERBOSE(
        curvatureCouplingWeight >= 0_r && rotInertiaWeight > 0_r && IsFinite(rotInertiaWeight),
        "Curvature twist weight must be non-negative and inertia twist weight must be positive and "
        "finite.");

    weights(baseDof + 3) = Max(curvatureCouplingWeight, rotInertiaWeight);
  }

  return weights;
}

namespace {

/** @brief Inertia of a link's subtree: the link and its descendants. */
struct SubtreeInertia {
  real mass; // [kg]
  Vec4r comOffset; // From the link's center of mass to the subtree's, in world frame [m].
  VMatrix3x3r momentOfInertia; // About the subtree's center of mass, in world frame [kg·m²].
};

} // namespace

/**
 * @brief Compute the kBlockSize × kBlockSize diagonal block of M(q) = Jᵀ·H·J for kBlockSize DoFs of
 * a joint, from the inertia of the joint's subtree and the DoFs' columns [blockCol, blockCol +
 * kBlockSize) in the Jacobian block of the joint's child link.
 *
 * @details The Jacobian columns of these DoFs are zero for links outside the subtree, and the DoFs
 * move the subtree as one rigid body. The block is therefore that of a single body with the
 * subtree's inertia, moving with the joint's child link.
 */
template <int kBlockSize>
[[nodiscard]] static Matrix<real, kBlockSize, kBlockSize> ComputeMassBlock(
    SubtreeInertia const& subtree,
    RowMatrixView<real const, RigidSize::kDAll> childLinkBlock,
    int blockCol) {
  static_assert(kBlockSize == 1 || kBlockSize == 3, "Unsupported block size.");

  // Extract the kBlockSize columns of the child link's trans/rot rows.
  auto const JtBlock = childLinkBlock.template Block<RigidSize::kDTrans, kBlockSize>(
      0, blockCol, RigidSize::kDTrans, kBlockSize);
  auto const JrBlock = childLinkBlock.template Block<RigidSize::kDRot, kBlockSize>(
      RigidSize::kDTrans, blockCol, RigidSize::kDRot, kBlockSize);

  // Jacobian of the subtree's center of mass, propagated from the child link's as in
  // ArticulatedJacobian::Update.
  Matrix<real, RigidSize::kDTrans, kBlockSize> const JcomBlock =
      JtBlock + AsMatrixView(lie::DMultRotVecDRot(subtree.comOffset)) * JrBlock;

  // Translational: mass × J_comᵀ·J_com
  Matrix<real, kBlockSize, kBlockSize> block = subtree.mass * (JcomBlock.Transpose() * JcomBlock);

  // Rotational: J_rᵀ·MOI·J_r
  block += JrBlock.Transpose() * AsMatrixView(subtree.momentOfInertia) * JrBlock;

  return block;
}

/**
 * @brief Compute the inertia of each link's subtree.
 *
 * @note Assumes a tree topology (no closed loops). Cycles introduced by constraints are not
 * accounted for — the subtree inertia may undercount if additional links are coupled via
 * constraints. This is acceptable for the purpose of defining characteristic scales for actor
 * convergence criteria.
 */
static DynamicArray<SubtreeInertia> ComputeSubtreeInertias(
    entt::registry const& reg,
    CGroupMembers const& groupMembers,
    CArticulatedParents const& parents,
    CArticulatedLinkTransforms<TimeStep::Current> const& linkTransforms,
    FiloAllocator* allocator) {
  int const numLinks = isize(groupMembers.actors);
  DynamicArray<SubtreeInertia> subtrees(allocator);
  subtrees.resize_noinit(numLinks);
  for (int i = 0; i < numLinks; ++i) {
    auto const& inertia = reg.get<CRigidBodyInertia>(groupMembers.actors[i]);
    subtrees[i] = {
        .mass = inertia.GetMass(),
        .comOffset = Vec4r{},
        .momentOfInertia =
            RotateInertia(inertia.GetMomentOfInertiaLocal(), linkTransforms[i].GetRotation())};
  }

  // Merge each subtree into its parent's.
  for (int i = numLinks - 1; i >= 0; --i) {
    int const parent = parents[i];
    if (parent == -1) {
      continue;
    }
    MOCHI_ASSERT_VERBOSE(parent >= 0 && parent < i, "Invalid parent link.");
    auto const& child = subtrees[i];
    auto& merged = subtrees[parent];
    // From the merged center of mass to the child's.
    Vec4r const delta =
        (linkTransforms[i].VGetTranslation() - linkTransforms[parent].VGetTranslation()) +
        (child.comOffset - merged.comOffset);
    real const mass = merged.mass + child.mass;
    real const childFraction = (mass > 0_r) ? (child.mass / mass) : 0_r;
    merged.comOffset += childFraction * delta;
    // Parallel axis theorem, with the reduced mass of the two parts.
    merged.momentOfInertia += child.momentOfInertia +
        (merged.mass * childFraction) * SecondMomentToMomentOfInertia(Outer3(delta, delta));
    merged.mass = mass;
  }

  return subtrees;
}

/** @brief Compute the diagonal of the principal matrix square root of a symmetric PSD block. */
template <int kBlockSize>
[[nodiscard]] static std::array<real, kBlockSize> BlockSqrtDiagonal(
    MatrixView<real const, kBlockSize, kBlockSize> block) {
  static_assert(kBlockSize == 1 || kBlockSize == 3, "Unsupported block size.");
  if constexpr (kBlockSize == 1) {
    return {Sqrt(Max(0_r, block(0, 0)))};
  } else {
    VSymMatrix3x3r const sym{
        Vec4r{block(0, 0), block(1, 1), block(2, 2)}, Vec4r{block(1, 0), block(2, 0), block(2, 1)}};
    auto const sqrtM = SqrtSym3x3(sym);
    return {sqrtM[0][0], sqrtM[1][1], sqrtM[2][2]};
  }
}

/**
 * @brief Compute per-DoF residual weights for an articulated actor.
 *
 * @details Joint-space DoFs with configuration-dependent M(q) = Jᵀ·H·J.
 * - Weight: w = 1/(Q_type · Q_c), where Q_c = aRef·√m_subtree·(√M_block)_{ii} and Q_type = Σ Q_c
 *   over all DoFs of the same type (translational or rotational).
 * - Block-wise decomposition (trans/rot separately) avoids unit mixing in matrix sqrt.
 * - ||r||_W = O(1) under characteristic loading.
 *
 * Concept: Define a characteristic generalized force/torque for each DoF `i` that induces a
 * reference linear acceleration `aRef` at the characteristic distance of the subtree actuated by
 * that joint.
 *
 * Derivation & Rationale:
 * 1. Dimensional Challenge: The full generalized mass matrix M(q) mixes units (mass vs. inertia).
 *    A direct matrix square root M¹/² across the whole matrix is mathematically invalid because it
 *    sums quantities with different units.
 * 2. Block-Wise Unification: To capture off-diagonal inertial coupling and maintain consistency
 *    with the single rigid body formulation, we isolate uniform-unit diagonal blocks from M(q)
 *    (e.g., 3x3 in a rotational joint, 1x1 in a prismatic joint).
 * 3. Subtree Mass: Let m_i be the total mass of all descendant links for DoF `i`. Using subtree
 *    mass (rather than link mass) since a joint must accelerate its entire downstream kinematic
 *    chain. This maintains physical consistency with the generalized mass matrix, which
 *    intrinsically accounts for the effective inertia of that entire downstream payload, and
 *    enables a robust formulation even with massless dummy links.
 * 4. Characteristic Formula: We compute the principal matrix square root per block.
 *    Q_c,i = aRef * m_i¹/² * (M_block¹/²)_{ii}
 */
static void GetArticulatedActorResidualWeights(
    entt::registry const& reg,
    entt::entity actor,
    real aRef,
    ColumnVector<real>& outWeights) {
  MOCHI_ASSERT_VERBOSE(reg.all_of<TagArticulatedActor>(actor), "Expected articulated actor.");
  MOCHI_ASSERT_VERBOSE(aRef > 0_r, "Characteristic acceleration must be positive.");

  // Per-type normalization: w_j = 1/(Q_type · Q_c,j), yielding O(1) weighted norm.
  auto const& groupMembers = reg.get<CGroupMembers>(actor);
  auto const& parents = reg.get<CArticulatedParents>(actor);
  auto const* joints = reg.get<CArticulatedBodyShape const>(actor).shape->GetJointsData();
  auto const& jacobian = reg.get<CArticulatedJacobian>(actor).value;
  auto const& linkTransforms = reg.get<CArticulatedLinkTransforms<TimeStep::Current>>(actor);
  auto const& dofInfo = reg.get<CActorDofInfo>(actor);
  int const numDofs = dofInfo.dofsSize;

  // Stack allocator for temporaries (subtree inertias + Qc per DoF).
  constexpr std::size_t kStackSizeBytes = 16 * 1024;
  MOCHI_FILO_STACK_ALLOCATOR(allocator, kStackSizeBytes);

  auto const subtrees =
      ComputeSubtreeInertias(reg, groupMembers, parents, linkTransforms, &allocator);

  // Compute Qc per DoF and accumulate per-type totals. Only the joints of the links have DoFs.
  DynamicArray<real> Qc(&allocator);
  Qc.resize_noinit(numDofs);
  real Qtrans = 0_r;
  real Qrot = 0_r;
  for (int link = 0; link < jacobian.NumLinks(); ++link) {
    auto const& joint = joints->dofInfo[link];
    real const sqrtSubtreeMass = Sqrt(subtrees[link].mass);
    // The joint's own DoFs are the trailing columns of the link's block.
    auto const linkBlock = jacobian.Link(link).block;
    int const firstJointCol = linkBlock.Cols() - joint.GetSize();

    auto accumulateBlock = [&]<int kBlockSize>(int offset, real& Qtype) {
      auto const block = ComputeMassBlock<kBlockSize>(
          subtrees[link], linkBlock, firstJointCol + (offset - joint.offset));
      auto const sqrtDiag = BlockSqrtDiagonal<kBlockSize>(AsConstView(block));
      for (int d = 0; d < kBlockSize; ++d) {
        Qc[offset + d] = aRef * sqrtSubtreeMass * sqrtDiag[d];
        Qtype += Qc[offset + d];
      }
    };

    if (joint.transSize == 1) {
      accumulateBlock.template operator()<1>(joint.GetTransOffset(), Qtrans);
    } else if (joint.transSize == 3) {
      accumulateBlock.template operator()<3>(joint.GetTransOffset(), Qtrans);
    } else {
      MOCHI_ASSERT_VERBOSE(joint.transSize == 0, "Unexpected joint translation size.");
    }
    if (joint.rotSize == 1) {
      accumulateBlock.template operator()<1>(joint.GetRotOffset(), Qrot);
    } else if (joint.rotSize == 3) {
      accumulateBlock.template operator()<3>(joint.GetRotOffset(), Qrot);
    } else {
      MOCHI_ASSERT_VERBOSE(joint.rotSize == 0, "Unexpected joint rotation size.");
    }
  }

  // Compute weights.
  outWeights.Resize(numDofs);
  auto fillWeights = [&](int offset, int size, real Qtype) {
    for (int d = 0; d < size; ++d) {
      int const idx = offset + d;
      outWeights(idx) = (Qtype > 0_r && Qc[idx] > 0_r) ? (1_r / (Qtype * Qc[idx])) : 1_r;
    }
  };
  for (auto const& joint : joints->dofInfo) {
    fillWeights(joint.GetTransOffset(), joint.transSize, Qtrans);
    fillWeights(joint.GetRotOffset(), joint.rotSize, Qrot);
  }
}

void UpdateActorConvergenceWeights(
    entt::registry const& reg,
    entt::entity actor,
    CActorSnle const& actorSnle,
    CActorConvergenceWeights& outWeights) {
  static_assert(
      static_cast<int>(ActorType::Count) == 6,
      "Please update this function when adding new actor types");
  MOCHI_ASSERT(
      !outWeights.isValid || outWeights.values.empty() ||
          (outWeights.values.Rows() == reg.get<CActorDofInfo>(actor).dofsSize),
      "Convergence weights have stale DoF count. Please call InvalidateActorConvergenceWeights.");

  if (!actorSnle.useInSolver || outWeights.isValid) {
    return;
  }

  // Use gravity as characteristic acceleration (clamped to ≥1 m/s²).
  real const aRef = Max(1_r, Norm<3>(reg.ctx<CSceneGravity>().accel));

  if (reg.any_of<TagArticulatedActor>(actor)) {
    GetArticulatedActorResidualWeights(reg, actor, aRef, outWeights.values);
  } else if (reg.any_of<TagRigidActor>(actor)) {
    GetRigidActorResidualWeights(reg, actor, aRef, outWeights.values);
  } else if (reg.any_of<TagSoftActor>(actor)) {
    outWeights.values = GetSoftActorResidualWeights(reg, actor, aRef);
  } else if (reg.any_of<TagShellActor>(actor)) {
    outWeights.values = GetShellActorResidualWeights(reg, actor, aRef);
  } else if (reg.any_of<TagRodActor>(actor)) {
    outWeights.values = GetRodActorResidualWeights(reg, actor, aRef);
  } else MOCHI_UNLIKELY {
    MOCHI_ASSERT(false, "Unexpected actor type.");
  }
  MOCHI_ASSERT_VERBOSE(IsFinite(MakeConstSpan(outWeights.values)), "Weights must be finite.");

  outWeights.isValid = true;
}

} // namespace mochi
