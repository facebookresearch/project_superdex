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

#include "mochi_ecs.h"

#include <mochi_core/linear_algebra/any_matrix.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/solvers/interaction_matrix_info.h>
#include <mochi_core/solvers/snle_problem.h>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace mochi {

/**************************************************************************
  ECS Components
*/

// Base class to store contributions to an SNLE problem.
struct BaseSnle : NoCopy {
  MOCHI_DECLARE_MOVE(BaseSnle);
  BaseSnle() = default;
  virtual ~BaseSnle() = default;

  virtual void SetZero(AssemblyParams const& params) = 0;

  // Zeroes the objective and the non-potential terms assembled with it.
  void ResetObjective() {
    objective = 0.0;
    nonPotentialEnergy = 0.0;
    hasNonPotentialResidual = false;
  }

  // Returns the non-potential residual, of numDofs entries, zeroed at the first call after
  // ResetObjective.
  [[nodiscard]] ColumnVectorView<real> NonPotentialResidual(int numDofs) {
    if (!hasNonPotentialResidual) {
      nonPotentialResidual.Resize(numDofs);
      nonPotentialResidual.SetZero();
      hasNonPotentialResidual = true;
    }
    MOCHI_ASSERT_VERBOSE(nonPotentialResidual.Rows() == numDofs);
    return nonPotentialResidual;
  }

  // Adds the energy and residual, on the DoFs from offset, of an active non-potential term.
  void AddNonPotentialTerms(
      double energy,
      ColumnVectorView<real const> residual,
      int offset,
      int numDofs) {
    nonPotentialEnergy += energy;
    NonPotentialResidual(numDofs).MiddleRows(offset, residual.Rows()) += residual;
  }

  double objective = 0.0;

  // If this SNLE data is not directly used by the solver, then set useInSolver to false.
  bool useInSolver = true;

  // Energy and residual of the non-potential terms (see SnleProblem::nonPotentialEnergy),
  // assembled with the objective. hasNonPotentialResidual says whether this SNLE has a
  // non-potential residual. A flag, rather than an empty residual, keeps the residual's allocation
  // across assemblies. An actor's residual spans its DoFs, and an interaction's the island's, at
  // offset 0.
  double nonPotentialEnergy = 0.0;
  ColumnVector<real> nonPotentialResidual = {};
  bool hasNonPotentialResidual = false;
};

/**
 * @brief Class storing the objective, residual and dresidual of an actor.
 * @details It stores both full and reduced representations of the residual and dresidual. The
 * reduced representation is optional. An actor that assembles its reduced dresidual directly may
 * leave the full dresidual empty as long as it stays reduced. Query UseReduced() to check whether
 * the reduced or full representation should be used in the nonlinear solver.
 */
struct ActorSnle : BaseSnle {
  ActorSnle(
      AnyMatrix<real>&& fullDRes,
      std::optional<PreconditionerType> fullPreconditionerTypeOverride = std::nullopt)
      : fullResidual(GetNumRows(fullDRes)),
        fullDResidual(std::move(fullDRes)),
        _fullPreconditionerTypeOverride(fullPreconditionerTypeOverride),
        _useReduced(false) {
    fullResidual.SetZero();
    mochi::SetZero(fullDResidual);
  }

  ActorSnle(
      AnyMatrix<real>&& reducedDRes,
      int fullSize,
      std::optional<PreconditionerType> reducedPreconditionerTypeOverride = std::nullopt)
      : fullResidual(fullSize),
        reducedResidual(GetNumRows(reducedDRes)),
        reducedDResidual(std::move(reducedDRes)),
        _reducedPreconditionerTypeOverride(reducedPreconditionerTypeOverride),
        _useReduced(true) {
    fullResidual.SetZero();
    reducedResidual.SetZero();
    mochi::SetZero(reducedDResidual);
  }

  void SetZero(AssemblyParams const& params) override {
    if (params.assemObj) {
      ResetObjective();
    }
    if (params.assemRes) {
      fullResidual.SetZero();
      reducedResidual.SetZero();
    }
    if (params.assemDRes) {
      mochi::SetZero(fullDResidual);
      mochi::SetZero(reducedDResidual);
    }
  }

  void SetFullToZero(AssemblyParams const& params) {
    if (params.assemObj) {
      ResetObjective();
    }
    if (params.assemRes) {
      fullResidual.SetZero();
    }
    if (params.assemDRes) {
      mochi::SetZero(fullDResidual);
    }
  }

  [[nodiscard]] bool UseReduced() const {
    return _useReduced;
  }

  [[nodiscard]] std::unique_ptr<ActorPreconditioner<real>>& GetActivePreconditioner() {
    return _useReduced ? _reducedPreconditioner : _fullPreconditioner;
  }

  // Preconditioner type set for the active representation at actor creation or by the user, if
  // any. This is not necessarily the type used in the island solve: use GetActorPreconditionerType
  // for that.
  [[nodiscard]] std::optional<PreconditionerType> GetActivePreconditionerTypeOverride() const {
    return _useReduced ? _reducedPreconditionerTypeOverride : _fullPreconditionerTypeOverride;
  }

  void SetActivePreconditionerTypeOverride(PreconditionerType type) {
    auto& typeOverride =
        _useReduced ? _reducedPreconditionerTypeOverride : _fullPreconditionerTypeOverride;
    typeOverride = type;
  }

  void EnableReduced(AnyMatrix<real>&& reducedDRes) {
    reducedResidual.Reset(ColumnVector<real>::Zero(GetNumRows(reducedDRes)));
    reducedDResidual = std::move(reducedDRes);
    _reducedPreconditioner.reset();
    _fullPreconditioner.reset();
    _useReduced = true;
  }

  void DisableReduced() {
    if (_useReduced) {
      _fullPreconditioner.reset();
    }
    reducedResidual = {};
    reducedDResidual = {};
    _reducedPreconditioner.reset();
    _useReduced = false;
  }

  ColumnVector<real> fullResidual;
  AnyMatrix<real> fullDResidual;

  ColumnVector<real> reducedResidual = {};
  AnyMatrix<real> reducedDResidual = {};

 private:
  std::unique_ptr<ActorPreconditioner<real>> _fullPreconditioner = nullptr;
  std::optional<PreconditionerType> _fullPreconditionerTypeOverride = std::nullopt;
  std::unique_ptr<ActorPreconditioner<real>> _reducedPreconditioner = nullptr;
  std::optional<PreconditionerType> _reducedPreconditionerTypeOverride = std::nullopt;
  bool _useReduced = false;
};

/**
  Component storing the objective, residual, and dresidual for a single actor, to be assembled in
  isolation.
*/
struct CActorSnle : ActorSnle {
  using ActorSnle::ActorSnle;
};

/**
  Component storing the objective, residual, and dresidual of a nested soft actor's posed body
  energy with respect to its skinned displacements, before the skinning Jacobians project them.
*/
struct CSoftSkinnedUnposedSnle : ActorSnle {
  using ActorSnle::ActorSnle;
};

/**
  Class storing the objective, residuals and dresiduals of the interaction between two or more
  actors.
*/
struct InteractionSnle : BaseSnle {
  void SetZero(AssemblyParams const& params) override {
    if (params.assemObj) {
      ResetObjective();
    }
    if (params.assemRes) {
      for (auto& [rOffset, res] : residuals) {
        res.SetZero();
      }
    }
    if (params.assemDRes) {
      for (auto& [rOffset, cOffset, dres, symmetricPair] : dresiduals) {
        mochi::SetZero(dres);
      }
    }
  }

  std::vector<std::pair</*offset*/ int, ColumnVector<real>>> residuals;
  std::vector<AnyInteractionMatrixInfo<real>> dresiduals;
};

/**
  Component storing the objective, residual, and dresidual for contact within an island. Used for
  sync contact within a regular island. TODO: If we stored the data for each pair of colliders
  separately, then we could assemble them all in parallel.
*/
struct CIslandContactSnle : InteractionSnle {
  using InteractionSnle::InteractionSnle;
};

/**
  Component storing the objective, residual, and dresidual for async contact of a skinned
  articulation. TODO: This could be assembled into the actor matrix for articulated bodies (not for
  soft-skinned actors).
*/
struct CSkinnedContactSnle : InteractionSnle {
  using InteractionSnle::InteractionSnle;
};

/**
  Component storing the objective, residual, and dresidual of a nested soft actor's posed body
  energy, after the skinning Jacobians project them. For performance reasons, only the off-diagonal
  sub-dresiduals (soft-articulated, articulated-soft) are stored in this component. The diagonal
  sub-dresiduals (soft-soft, articulated-articulated) are stored in the actor dresidual of the
  underlying soft and articulated actors. The residual holds the soft entry only. The articulated
  one is stored in link space in CArticulatedSkinnedBodyLinkResidual.
*/
struct CSkinnedInteractionSnle : InteractionSnle {
  using InteractionSnle::InteractionSnle;

  // Actor index in residual and dresidual vectors. For the dresidual, the index refers to the
  // off-diagonal submatrix whose rows (NOT columns) correspond to the actor. kArticulatedIdx only
  // applies to the dresidual.
  static constexpr int kSoftIdx = 0;
  static constexpr int kArticulatedIdx = 1;
};

/**
  Component storing the objective, residual, and dresidual for all static constraints within a
  compound.
*/
struct CCompoundConstraintSnle : InteractionSnle {
  using InteractionSnle::InteractionSnle;
};

namespace snle {
void InitializeOnce(entt::registry& reg);
} // namespace snle

} // namespace mochi
