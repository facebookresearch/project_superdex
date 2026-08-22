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

#include "mochi_common_components.h"
#include "mochi_discretization_components.h"
#include "mochi_ecs.h"

#include <mochi_core/integration/integration_utils.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/solvers/krylov_solver.h>
#include <mochi_core/solvers/newton_solver_params.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/sparsity_utils.h>
#include <mochi_physics/mochi_physics.h>

#include <utility>
#include <vector>

namespace mochi {

/**************************************************************************
  Common ECS components for simulated actors
*/

/*
  Groups all the parameters needed to time-step the scene.
*/
struct CSimulationParams final : public SolverParams, public NoCopy {};

/*
  Stores information on the number of DoFs of an actor. Note that the number of DoFs may differ from
  the size of the pose representation (aka position state). The struct is used by simulation islands
  to index full-island vectors and matrices with data of size #DoFs (step, residual, dresidual) and
  data of size #pose (solution).
*/
struct CActorDofInfo : NoCopy {
  int dofsSize = 0;
  int poseSize = 0;
};

/*
  Stores the DoF and pose offset within a single island. Same as CActorDofInfo, it is used by
  simulation islands to index full-island vectors and matrices.
*/
struct CDofOffset : NoCopy {
  int dofsOffset = 0;
  int poseOffset = 0;
};

// Indicates that an actor's CDofOffset or CActorDofInfo has changed.
// Cleared at the end of mochi::PreStepEcs.
struct TagGlobalDofsChanged {};

/*
  Stores the actor's mass matrix. This is pre-computed so that the per-step assembly doesn't have to
  assemble the inertia dresidual element-by-element every time. Instead, we can simply copy the
  values of this matrix and scale them by (1 / dt^2).
*/
struct CMassMatrix : public NoCopy {
  // Storing a SparseMatrix or BlockSparseMatrix here would require us to duplicate the indicies of
  // the actor's dresidual matrix. Instead, we store just the value array. It must match the number
  // of non-zeros stored in CActorSnle::fullDResidual.
  DynamicArray<real> values;
};

/**
 * @brief Stores the row-sum lumped mass matrix of an actor. This is a diagonal approximation of
 * the full mass matrix. Each entry represents the lumped mass associated with a single DoF.
 */
struct CLumpedMassMatrix : public NoCopy {
  DynamicArray<real> values;
};

/*
  The following component stores the per-element mass matrix.

  This is used in two places:
  - inside mochi::soft::UpdateSoftMass: it serves as auxiliary data
    that is computed/updated and then used to update the CMassMatrix

  - in the ROM code, when doing the per-element assembly and project,
    it is used directly to initialize the per-element DRes

  Note: templated on the fem volume discretization type to make the dependency explicit.
  This allows to set kNumEleDofs correctly for a generic discretization type rather
  than hardwiring it, even if currently in mochi the most likely type used for this is
  CFemVolumeDiscretizationP1Q4.
*/
template <typename CFemVolumeDiscretizationType>
struct CPerElementMassMatrix : public NoCopy {
  static int constexpr kNumElemDofs = CFemVolumeDiscretizationType::kNumEleDofs;
  DynamicArray<NdArray<real, kNumElemDofs, kNumElemDofs>> values;
};

/*
  Stores Dirichlet boundary conditions, represented as world-space pose values. Note that the pose
  representation (aka position state) and the DoFs may differ. The component stores both DoF and
  pose indices, and pose values.
*/
struct CDofPositionsBC : public NoCopy {
  struct PermanentRange {
    int dofBegin = 0;
    int dofCount = 0;
    int poseBegin = 0;
    int poseCount = 0;
    int colValueBegin = 0;
    int colValueCount = 0;
  };

  std::vector<int> dofIndices; // Indices of the constrained DoFs
  std::vector<int> poseIndices; // Constrained indices in the pose representation
  std::vector<real> dofValues; // Constraint values in the DoF representation (not used internally,
                               // only for API queries)
  std::vector<real> poseValues; // Constraint values in the pose representation

  /// @brief Cache for efficient column zeroing of sparse and block sparse actor matrices. Contains
  /// the indices into the actor matrix's value array that need to be zeroed when zeroing the DoF
  /// columns. Not used for dense actor matrices.
  DynamicArray<int> colValueIndices;

  /// @brief One entry per permanent BC call, each a half-open [begin, begin + count) slice into the
  /// dof, pose, and colValue arrays.
  DynamicArray<PermanentRange> permanentRanges;

  void Clear() {
    if (permanentRanges.empty()) {
      dofIndices.clear();
      poseIndices.clear();
      dofValues.clear();
      poseValues.clear();
      colValueIndices.clear();
      return;
    }

    int permanentDofCount = 0;
    int permanentPoseCount = 0;
    int permanentColValueCount = 0;
    for (auto const& range : permanentRanges) {
      permanentDofCount += range.dofCount;
      permanentPoseCount += range.poseCount;
      permanentColValueCount += range.colValueCount;
    }

    MOCHI_ASSERT_VERBOSE(
        isize(dofValues) == isize(dofIndices) && isize(poseValues) == isize(poseIndices),
        "BC value arrays must stay in lockstep with their index arrays.");
    if (permanentDofCount == isize(dofIndices) && permanentPoseCount == isize(poseIndices) &&
        permanentColValueCount == isize(colValueIndices)) {
      return;
    }

    std::vector<int> newDofIndices;
    std::vector<int> newPoseIndices;
    std::vector<real> newDofValues;
    std::vector<real> newPoseValues;
    DynamicArray<int> newColValueIndices;
    DynamicArray<PermanentRange> newPermanentRanges;
    newDofIndices.reserve(permanentDofCount);
    newPoseIndices.reserve(permanentPoseCount);
    newDofValues.reserve(permanentDofCount);
    newPoseValues.reserve(permanentPoseCount);
    newColValueIndices.reserve(permanentColValueCount);
    newPermanentRanges.reserve(permanentRanges.size());

    // Rebuild the arrays keeping only permanent ranges, compacting and remapping each range.
    for (auto const& range : permanentRanges) {
      PermanentRange newRange{
          .dofBegin = isize(newDofIndices),
          .dofCount = range.dofCount,
          .poseBegin = isize(newPoseIndices),
          .poseCount = range.poseCount,
          .colValueBegin = isize(newColValueIndices),
          .colValueCount = range.colValueCount,
      };

      Append(
          newDofIndices,
          dofIndices.begin() + range.dofBegin,
          dofIndices.begin() + range.dofBegin + range.dofCount);
      Append(
          newDofValues,
          dofValues.begin() + range.dofBegin,
          dofValues.begin() + range.dofBegin + range.dofCount);
      Append(
          newPoseIndices,
          poseIndices.begin() + range.poseBegin,
          poseIndices.begin() + range.poseBegin + range.poseCount);
      Append(
          newPoseValues,
          poseValues.begin() + range.poseBegin,
          poseValues.begin() + range.poseBegin + range.poseCount);
      Append(
          newColValueIndices,
          colValueIndices.begin() + range.colValueBegin,
          colValueIndices.begin() + range.colValueBegin + range.colValueCount);

      newPermanentRanges.push_back(newRange);
    }

    dofIndices = std::move(newDofIndices);
    poseIndices = std::move(newPoseIndices);
    dofValues = std::move(newDofValues);
    poseValues = std::move(newPoseValues);
    colValueIndices = std::move(newColValueIndices);
    permanentRanges = std::move(newPermanentRanges);
  }
};

/**
 * Component storing Dirichlet boundary conditions in the pose representation used by the system
 * solve. CDirichletBC differs from CDofPositionsBC for soft actors, as constraint values are
 * expressed in a local reference system.
 */
template <typename Scalar>
struct CDirichletBC : public NoCopy {
  [[nodiscard]] bool Empty() const {
    return dofIndices.empty();
  }
  void Clear() {
    dofIndices.clear();
    poseIndices.clear();
    poseValues.clear();
    colValueIndices.clear();
  }
  DynamicArray<int> dofIndices; // Indices of the constrained DoFs
  DynamicArray<int> poseIndices; // Constrained indices in the pose representation of the solve
  DynamicArray<Scalar> poseValues; // Constraint values
  DynamicArray<int> colValueIndices; // See CDofPositionsBC::colValueIndices
};

/**
 * Component storing external forces and the affected DoFs.
 */
struct CExternalForces : public NoCopy {
  [[nodiscard]] bool Empty() const {
    return dofs.empty();
  }
  void Clear() {
    dofs.clear();
    forces.clear();
  }
  std::vector<int> dofs;
  std::vector<real> forces;

  MOCHI_STRUCT_BEGIN(mochi::CExternalForces);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_FIELD(dofs);
  MOCHI_FIELD(forces);
  MOCHI_STRUCT_END();
};

/**************************************************************************
  Common ECS Tags for simulated actors
  WARNING: If tags for new forces are added in the future, please update the recording tools (e.g.
  AssembleAndRecordSnleTerm) that temporarily disable all the force tags in order to compute the
  individual contribution of each of them.
*/

// Indicates the inertia term must be included in ResDRes computations.
struct TagUseInertia {};

// Indicates the gravity term must be included in ResDRes computations.
struct TagUseGravity {};

// Indicates the stress term must be included in ResDRes computations.
struct TagUseStress {};

// Indicates the contact term must be included in ResDRes computations
struct TagUseContact {};

// Indicates that the inertia term of a rigid actor should be evaluated using Newton-Euler
// equations. This is an experimental feature, and the default is the Rigid IPC inertia model.
struct TagUseNewtonEulerInertia {};

/**************************************************************************
  Common ECS Utils for simulated actors
*/

// Get the Newton solver parameters for a simulation island.
void GetIslandNewtonParams(
    int numDofs,
    SolverParams const& simParams,
    NewtonSolverParams& outNewtonParams);

// Set Dirichlet BCs on the solution vector. The BC indices must index the solution vector.
void SetDirichletBCs(CDirichletBC<real> const& dirichlet, ColumnVectorView<real> outSolution);

// Call after externally overwriting actor state. Clears multi-step integration data and enables
// conservative-step-bounds relaxation for the next step.
void InvalidateActorStepHistory(entt::registry& reg, entt::entity actor);

namespace simulation {
void InitializeOnce(entt::registry& reg);
} // namespace simulation

} // namespace mochi
