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

/**
 * @brief Stores the mass matrix of an actor.
 *
 * @details Only soft actors with inertia but no stress store one. Soft-skinned actors also store
 * one if their inertia is posed and their stress unposed, or vice versa, because posed and unposed
 * terms are assembled separately. Without stress, the assembled dresidual is the mass matrix times
 * (1 / dt^2 + alpha / dt), with dt the stage time step and alpha the mass-damping coefficient, so
 * the full-order assembly copies the scaled values instead of adding the mass element by element.
 * When inertia and stress are assembled together, the mass is added in the element loop.
 */
struct CMassMatrix : public NoCopy {
  // Storing a SparseMatrix or BlockSparseMatrix here would require us to duplicate the indicies of
  // the actor's dresidual matrix. Instead, we store just the value array. It must match the number
  // of non-zeros stored in CActorSnle::fullDResidual.
  // TODO: Explore storing one value per 3x3 block. Each block of the consistent mass is m_ij·I₃:
  // its off-diagonal entries are zero and its diagonal entries are equal. The array, and the reads
  // of its copy into the dresidual, would be 9x smaller.
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

  // Removes all non-permanent boundary conditions. Returns true if any were removed.
  bool Clear() {
    if (permanentRanges.empty()) {
      bool const hadAny = !dofIndices.empty() || !poseIndices.empty();
      dofIndices.clear();
      poseIndices.clear();
      dofValues.clear();
      poseValues.clear();
      colValueIndices.clear();
      return hadAny;
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
      return false;
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
    return true;
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
    entt::registry const& reg,
    entt::entity island,
    NewtonSolverParams& outNewtonParams);

// Set Dirichlet BCs on the solution vector. The BC indices must index the solution vector.
void SetDirichletBCs(CDirichletBC<real> const& dirichlet, ColumnVectorView<real> outSolution);

// Call after externally overwriting actor state. Clears multi-step integration data and the
// convergence status, enables conservative-step-bounds relaxation for the next step, and requests
// clearing friction preload.
void InvalidateActorStepHistory(entt::registry& reg, entt::entity actor);

namespace simulation {
void InitializeOnce(entt::registry& reg);
} // namespace simulation

} // namespace mochi
