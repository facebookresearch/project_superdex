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

#include <mochi_core/solvers/interaction_matrix_info.h>
#include <mochi_core/solvers/island_operators.h>
#include <mochi_core/utils/assembly_params.h>

#include <functional>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace mochi {

// Forward
template <class T>
class SnleProblem;

/**
 * Struct storing user-defined functions for an SnleProblem.
 */
template <typename T>
struct SnleProblemFunctions {
  // Function to assemble problem at current solution
  std::function<void(SnleProblem<T>& problem, AssemblyParams const& params)> assemble;

  // Function to be executed after solution is updated
  std::function<void(SnleProblem<T>& problem)> onPostNewSolution;

  // Function to be executed after a solution increment is computed.
  std::function<void(SnleProblem<T>& problem)> onPostNewIncrement;
};

/**
 * Wraps a system of non-linear equations for Newton solvers.
 * Uses functions provided by the user.
 */
template <typename T>
class SnleProblem {
 public:
  SnleProblem() = default;
  SnleProblem(int dofsSize, int solutionSize, SnleProblemFunctions<T>&& functions);

  // Returns the number of degrees-of-freedom of the problem.
  int GetDofsSize() const;

  // Returns the size of the solution of the problem (possibly different from the dofs).
  int GetSolutionSize() const;

  // Returns the vector containing the current solution of the problem.
  ColumnVectorView<T const> GetSolution() const;

  // Returns the vector containing the current solution increment.
  ColumnVectorView<T const> GetIncrement() const;

  // Returns the current scalar value of the objective/merit function.
  double GetObjective() const;

  // Get the residual vector for the full problem.
  // Computed on-demand based on per-actor data, then cached internally.
  ColumnVectorView<T const> GetResidual() const;

  // Computes the dresidual matrix for the full problem.
  // Computed on-demand based on per-actor data, then cached internally.
  AnyMatrixView<T const> GetDResidual() const;

  // Returns the set of linear operators for the system's coupled equations.
  IslandOperators<T> GetOperators() const;

  // Scales the increment vector.
  void ScaleIncrement(T alpha);

  // Takes the current increment and modifies the problem solution. It also updates the position
  // state of the actors to be consistent with the new solution.
  void UpdateSolution();

  // Sets the solution vector to the given vector.
  // If 'invokePost' is true, it also updates the position state of the actors to be consistent with
  // the new solution.
  void SetSolution(ColumnVectorView<T const> val, bool invokePost = true);

  // Updates and returns the value of the objective merit function. Calls UpdateObjResDRes.
  void UpdateObjective();

  // Updates and returns the value of the current problem residual. Calls UpdateObjResDRes.
  void UpdateResidual();

  // Updates the problem residual derivative. Calls UpdateObjResDRes.
  void UpdateDResidual(bool psdDRes, SaturationHessianParams const& fittedSaturationHessian);

  // Updates the objective, residual, and/or residual derivative depending on parameters.
  void UpdateObjResDRes(AssemblyParams const& params = {});

  // Function to be called after the value of the solution is set externally.
  // It updates the position state of the actors to be consistent with the new solution.
  void OnPostNewSolution();

  // Function to be called after the value of the solution increment changes.
  // It updates the position state of the actors, and then updates the solution to be consistent.
  void OnPostNewIncrement();

  /**
   * @brief Utility method to perform consistency check of the residual and residual derivative
   *        against a finite difference approximation.
   * @param finDiffStep - Step size of the finite difference approximation.
   * @param numberLogEntries - Number of res/Dres entries to log.
   * @details It internally modifies the solution of the problem (to compute the finite difference
   *          approximation) but leaves it unmodified at output (same solution at output as at
   *          input).
   */
  void ConsistencyCheckResDRes(real finDiffStep, int numberLogEntries);

  /**
   * @brief Utility method to perform consistency check of the residual norm by finite difference
   * approximation.
   * @param finDiffStep - Step size of the finite difference approximation.
   * @param verbosity - If equal or above VerbosityLevel::Verbose, it prints the error.
   * @return Relative error of the residual norm.
   * @details It internally modifies the solution of the problem (to compute the finite difference
   *          approximation) but leaves it unmodified at output (same solution at output as at
   *          input).
   */
  real ConsistencyCheckResNorm(real finDiffStep, VerbosityLevel verbosity);

  // Return true if the solution has changed since the last assembly (objective, or res, or dres)
  bool HasSolutionChangedSinceLastAssembly() const;

  // Set all dirty flags so that previously cached data will not be used.
  void InvalidateCachedData();

  // Modify the assembly function.
  void SetAssemblyFunction(
      std::function<void(SnleProblem<T>& problem, AssemblyParams const& params)> assemble);

  // Solution of the global problem
  ColumnVector<T> solution;

  // Next solution increment. To be handled by actors, as it may not be in Euclidean space.
  ColumnVector<T> increment;

  // Objective value for the global problem.
  double objective = 0.0;

  // Collection of residual vectors for each actor. Paired with the actor's DOF offset.
  std::vector<std::pair</*offset*/ int, ColumnVector<T>*>> actorResiduals;

  // Collection of dresidual matrices for each actor. Paired with the actor's DOF offset.
  std::vector<std::pair</*offset*/ int, AnyMatrix<T>*>> actorMatrices;

  // Collection of preconditioners for each actor. Paired with the actor's DOF offset and an
  // optional preconditioner type hint. The actor preconditioners approximate the inverse of the
  // full actor matrices, including the contribution due to interaction matrices.
  // Tuple elements: (offset, preconditioner reference, optional type hint)
  std::vector<std::tuple<
      /*offset*/ int,
      std::reference_wrapper<std::unique_ptr<ActorPreconditioner<T>>>,
      std::optional<PreconditionerType>>>
      actorPreconditioners;

  // Collection of convergence weights for each actor. Paired with the actor's DOF offset, which
  // must match the corresponding actorResiduals offset. Must be populated by the assembly callback
  // whenever actorResiduals is populated. Consumed by the non-linear solver only when
  // convergenceMode == PerActorWeighted.
  std::vector<std::pair</*offset*/ int, ColumnVector<T> const*>> actorConvergenceWeights;

  // Contributions to the global residual from contact, constraints and other terms that involve 2
  // or more actors.
  std::vector<std::pair</*offset*/ int, ColumnVector<T>*>> interactionResiduals;

  // Contributions to the global dresidual from contact, constraints and other terms that involve 2
  // or more actors.
  std::vector<AnyInteractionMatrixPtrInfo<T>> interactionMatrices;

 protected:
  void SortActors();
  void ComputeFullResidual(ColumnVector<T>& outRes) const;

  SnleProblemFunctions<T> _functions;

  bool _dirtyAssembly = true; // True if the solution changed since the last objective, residual
                              // and/or per-actor dresidual assembly
  bool _dirtyObj = true; // True if solution changed since last objective assembly
  bool _dirtyRes = true; // True if solution changed since last residual assembly
  // NOTE: No dirty flag for per-actor dresidual assembly (at least for now). It's currently
  // unnecessary (dresidual assembly in never performed twice) and error-prone given the dresidual
  // can be computed with different configuration settings, e.g. PSD projection.

  mutable bool _dirtyFullDRes =
      true; // True if per-actor dresidual assembly has changed since last GetDResidual() call.

  mutable ColumnVector<T> _fullRes; // Up-to-date if !_dirtyRes
  mutable AnyMatrix<T> _fullDRes; // Up-to-date if !_dirtyFullDRes
};

} // namespace mochi

#include "snle_problem_inl.h"
