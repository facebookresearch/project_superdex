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

  double objective = 0.0;

  // If this SNLE data is not directly used by the solver, then set useInSolver to false.
  bool useInSolver = true;
};

/**
 * @brief Class storing the objective, residual and dresidual of an actor.
 * @details It stores both full and reduced representations of the residual and dresidual. The
 * reduced representation is optional. Query UseReduced() to check whether the reduced or full
 * representation should be used in the nonlinear solver.
 */
struct ActorSnle : BaseSnle {
  ActorSnle(
      AnyMatrix<real>&& fullDRes,
      std::optional<PreconditionerType> fullPreconditionerType = std::nullopt)
      : fullResidual(GetNumRows(fullDRes)),
        fullDResidual(std::move(fullDRes)),
        fullPreconditionerType(fullPreconditionerType),
        _useReduced(false) {
    fullResidual.SetZero();
    mochi::SetZero(fullDResidual);
  }

  ActorSnle(
      AnyMatrix<real>&& fullDRes,
      AnyMatrix<real>&& reducedDRes,
      std::optional<PreconditionerType> fullPreconditionerType = std::nullopt,
      std::optional<PreconditionerType> reducedPreconditionerType = std::nullopt)
      : fullResidual(GetNumRows(fullDRes)),
        fullDResidual(std::move(fullDRes)),
        fullPreconditionerType(fullPreconditionerType),
        reducedResidual(GetNumRows(reducedDRes)),
        reducedDResidual(std::move(reducedDRes)),
        reducedPreconditionerType(reducedPreconditionerType),
        _useReduced(true) {
    fullResidual.SetZero();
    mochi::SetZero(fullDResidual);
    reducedResidual.SetZero();
    mochi::SetZero(reducedDResidual);
  }

  void SetZero(AssemblyParams const& params) override {
    if (params.assemObj) {
      objective = 0.0;
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
      objective = 0.0;
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

  void EnableReduced(AnyMatrix<real>&& reducedDRes) {
    reducedResidual.Reset(ColumnVector<real>::Zero(GetNumRows(reducedDRes)));
    reducedDResidual = std::move(reducedDRes);
    reducedPreconditioner.reset(nullptr);
    fullPreconditioner.reset(nullptr);
    _useReduced = true;
  }

  void DisableReduced() {
    if (_useReduced) {
      fullPreconditioner.reset(nullptr);
    }
    reducedResidual = {};
    reducedDResidual = {};
    reducedPreconditioner.reset(nullptr);
    _useReduced = false;
  }

  ColumnVector<real> fullResidual;
  AnyMatrix<real> fullDResidual;
  std::unique_ptr<ActorPreconditioner<real>> fullPreconditioner = nullptr;
  std::optional<PreconditionerType> fullPreconditionerType = std::nullopt;

  ColumnVector<real> reducedResidual = {};
  AnyMatrix<real> reducedDResidual = {};
  std::unique_ptr<ActorPreconditioner<real>> reducedPreconditioner = nullptr;
  std::optional<PreconditionerType> reducedPreconditionerType = std::nullopt;

 private:
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
    Component storing the unposed objective, residual, and dresidual for a nested soft actor.
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
      objective = 0.0;
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
  Component storing the objective, residual, and dresidual of the unposed interaction terms between
  the soft and articulated actors. For performance reasons, only the off-diagonal sub-dresiduals
  (soft-articulated, articulated-soft) are stored in this component. The diagonal sub-dresiduals
  (soft-soft, articulated-articulated) are stored in the actor dresidual of the underlying soft and
  articulated actors.
*/
struct CSkinnedInteractionSnle : InteractionSnle {
  using InteractionSnle::InteractionSnle;

  // Actor index in residual and dresidual vectors. For the dresidual, the index refers to the
  // off-diagonal submatrix whose rows (NOT columns) correspond to the actor.
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
