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

#include <mochi_physics/mochi_physics_experimental.h>

#include <mochi_core/solvers/nonlinear_solver_params.h>
#include <mochi_core/utils/verbosity_params.h>

namespace mochi::diffsim {

struct BackPropagationSceneStats {
  double totalDurationSec = 0.0;
  double solveDurationSec = 0.0;
  int maxOuterIters = 0;
  double residualNorm = 0.0;
  bool finiteDiffValid = true;
};

struct BackPropagationSolverParams {
  VerbosityLevel verbosity = NonLinearSolverParams{}.verbosity;
  bool useNewtonOuterSolver = false;
  int outerSolverMaxIter = 10;
  real outerSolverAbsTol = 1e-3_r;
  real outerSolverRelTol = 1e-10_r;
  NonLinearSolverConvergenceMode outerSolverConvergenceMode =
      NonLinearSolverConvergenceMode::Global;
  real innerSolverAbsTol = 1e-10_r;
  real epsFiniteDiff = kDefaultBackPropagationEpsFiniteDiff;
  bool validateFiniteDiff = false;
};

} // namespace mochi::diffsim
