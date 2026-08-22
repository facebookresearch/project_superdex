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

#include <mochi_core/solvers/snle_problem.h>

#include <mochi_core/utils/basic_utils.h>

#include <algorithm>
#include <cmath>
#include <limits>

// MSVC COMPILER BUG WORKAROUND:
// Code was moved into this cpp file from the header. This move was partly motivated by a bug in the
// VS2022 compiler. It was emitting 32 blocks of consecutive zero bytes into the machine code. This
// looks like "add byte ptr [rax],al" in the dissassembly view. Moving the code to the cpp file
// appears to prevent the problem. If the problem resurfaces, then I suggest using "#pragma
// optimize" to disable optimization for the affected function.

namespace mochi {

template <typename T>
void SnleProblem<T>::UpdateSolution() {
  MOCHI_PROFILE_SCOPE();
  OnPostNewIncrement();
}

template <typename T>
void SnleProblem<T>::SetSolution(ColumnVectorView<T const> val, bool invokePost) {
  MOCHI_PROFILE_SCOPE();
  this->solution = val;
  if (invokePost) {
    OnPostNewSolution();
  }
}

template <typename T>
void SnleProblem<T>::ComputeFullResidual(ColumnVector<T>& outRes) const {
// Validation
#if MOCHI_ASSERT_VERBOSE_ENABLED
  {
    int expectedOffset = 0;
    for (auto const& [offset, res] : actorResiduals) {
      MOCHI_ASSERT_VERBOSE(
          offset == expectedOffset,
          "Actor residuals should be sorted by offset and there should be no gaps or overlap between them.");
      expectedOffset += res->Rows();
    }
    int totalNumDofs = expectedOffset;
    for (auto const& [offset, res] : interactionResiduals) {
      MOCHI_ASSERT_VERBOSE(
          offset >= 0 && offset + res->Rows() <= totalNumDofs,
          "Interaction residual is out of range.");
    }
  }
#endif // MOCHI_ASSERT_VERBOSE_ENABLED

  // Compute num rows
  int numRows = 0;
  if (!actorResiduals.empty()) {
    auto const& [offset, lastRes] = actorResiduals.back();
    numRows = offset + lastRes->Rows();
  }

  // Allocate memory if size has changed
  outRes.Resize(numRows);

  // Fill values
  for (auto const& [offset, res] : actorResiduals) {
    outRes.MiddleRows(offset, res->Rows()) = *res;
  }
  for (auto const& [offset, res] : interactionResiduals) {
    outRes.MiddleRows(offset, res->Rows()) += *res;
  }
}

template <typename T>
AnyMatrixView<T const> SnleProblem<T>::GetDResidual() const {
  // The full dresidual is derived from the IslandOperators.
  if (_dirtyFullDRes) {
    _fullDRes = GetOperators().CondenseFullMatrix();
    _dirtyFullDRes = false;
  }
  return AsConstView(_fullDRes);
}

template <typename T>
void SnleProblem<T>::UpdateObjective() {
  MOCHI_PROFILE_SCOPE();
  UpdateObjResDRes({.assemObj = true, .assemRes = false, .assemDRes = false});
}

template <typename T>
void SnleProblem<T>::UpdateResidual() {
  MOCHI_PROFILE_SCOPE();
  UpdateObjResDRes({.assemObj = false, .assemRes = true, .assemDRes = false});
}

template <typename T>
void SnleProblem<T>::UpdateDResidual(
    bool psd,
    SaturationHessianParams const& fittedSaturationHessian) {
  MOCHI_PROFILE_SCOPE();
  UpdateObjResDRes(
      {.assemObj = false,
       .assemRes = false,
       .assemDRes = true,
       .psdDRes = psd,
       .fittedSaturationHessian = fittedSaturationHessian});
}

template <typename T>
void SnleProblem<T>::UpdateObjResDRes(AssemblyParams const& params) {
  MOCHI_PROFILE_SCOPE();
  // Params to avoid recomputing terms that are already up-to-date
  auto lazyParams(params);
  lazyParams.assemObj &= _dirtyObj;
  lazyParams.assemRes &= _dirtyRes;

  if (lazyParams.assemObj || lazyParams.assemRes || lazyParams.assemDRes) {
    _functions.assemble(*this, lazyParams);

    // If per-actor data changed, make sure they are properly sorted by DOF offset
    if (lazyParams.assemRes || lazyParams.assemDRes) {
      SortActors();
    }

    // Update dirty flags and island-wide assembly results
    _dirtyAssembly = false;
    if (lazyParams.assemObj) {
      _dirtyObj = false; // This was just updated
    }
    if (lazyParams.assemRes) {
      _dirtyRes = false; // This was just updated
      ComputeFullResidual(_fullRes);
    }
    if (lazyParams.assemDRes) {
      _dirtyFullDRes = true; // Full dresidual will be updated on-demand
    }
  }
}

template <typename T>
void SnleProblem<T>::OnPostNewSolution() {
  if (_functions.onPostNewSolution) {
    MOCHI_PROFILE_SCOPE();
    _functions.onPostNewSolution(*this);
  }

  // Assembly data is dirty since solution has changed. Note that the full dresidual is considered
  // up-to-date as long as it matches the per-actor dresiduals. Therefore _dirtyFullDRes is
  // unchanged.
  _dirtyAssembly = _dirtyObj = _dirtyRes = true;
}

template <typename T>
void SnleProblem<T>::OnPostNewIncrement() {
  if (_functions.onPostNewIncrement) {
    MOCHI_PROFILE_SCOPE();
    _functions.onPostNewIncrement(*this);
  }

  // Assembly data is dirty since solution has changed. Note that the full dresidual is considered
  // up-to-date as long as it matches the per-actor dresiduals. Therefore _dirtyFullDRes is
  // unchanged.
  _dirtyAssembly = _dirtyObj = _dirtyRes = true;
}

template <typename T>
void SnleProblem<T>::ConsistencyCheckResDRes(real const finDiffStep, int const numberLogEntries) {
  // Update residual.
  UpdateResidual();

  // Update Dresidual. For finite difference comparisons, make sure PSD projection and fitted
  // Hessian are not active.
  UpdateDResidual(false, SaturationHessianParams::All(false));

  // Define variables for code readability.
  auto sol = AsView(this->solution);
  auto inc = AsView(this->increment);
  auto res = GetResidual();
  auto Dres = GetDResidual();

  // Make copies of useful quantities.
  auto sol0 = sol.Duplicate();
  auto res0 = res.Duplicate();
  auto resForward = res0.Duplicate();
  auto resBackward = res0.Duplicate();
  auto DresDir = res0.Duplicate();

  //------------------------

  real errorRes = 0_r;
  real normRes = 0_r;

  real errorDRes = 0_r;
  real normDRes = 0_r;

  // Compute finite difference w.r.t. each DoF.
  int const s = isize(inc);
  for (int i = 0; i < s; i++) {
    // Forward step.
    inc.SetZero();
    inc[i] = finDiffStep / 2_r;
    UpdateSolution();
    UpdateObjective();
    double const meritForward = GetObjective();
    UpdateResidual();
    resForward = GetResidual();

    // Reset solution.
    SetSolution(sol0);

    // Backward step.
    inc.SetZero();
    inc[i] = -finDiffStep / 2_r;
    UpdateSolution();
    UpdateObjective();
    double const meritBackward = GetObjective();
    UpdateResidual();
    resBackward = GetResidual();

    // Reset solution.
    SetSolution(sol0);

    // Compute (centered) finite difference approximation.
    double const finDiffRes = (meritForward - meritBackward) / static_cast<double>(finDiffStep);
    resForward = (resForward - resBackward) * static_cast<real>(1.0 / finDiffStep);

    // Extract entries of Dresidual by multiplying it against a unit vector in the direction of the
    // DoF.
    inc.SetZero();
    inc[i] = 1_r;
    std::visit([&](auto const& A) { Apply(A, inc, DresDir); }, Dres);

    // Log analytical vs. finite difference for comparison.
    if (i < numberLogEntries) {
      MOCHI_LOG("res ( %i ) | Analytical: %.10e, Finite Difference: %.10e", i, res0[i], finDiffRes);

      for (int j = 0; j < Min(s, numberLogEntries); j++)
        MOCHI_LOG(
            "Dres ( %i,%i ) | Analytical: %.10e, Finite Difference: %.10e",
            j,
            i,
            DresDir[j],
            resForward[j]);
    }

    // Accumulate errors.
    real const res0_i = res0[i];
    real const finDiffRes_i = res0_i - static_cast<real>(finDiffRes);
    errorRes += (finDiffRes_i * finDiffRes_i);
    normRes += res0_i * res0_i;

    for (int j = 0; j < s; j++) {
      real const Dres_ji = DresDir[j];
      real const finDiffDRes_ji = resForward[j];
      real const diffDRes = Dres_ji - finDiffDRes_ji;
      errorDRes += (diffDRes * diffDRes);
      normDRes += Dres_ji * Dres_ji;
    }
  } // for (int i = 0; i < s; i++)

  // Compute and log error ratios.
  errorRes = Sqrt(errorRes);
  normRes = Sqrt(normRes);

  errorDRes = Sqrt(errorDRes);
  normDRes = Sqrt(normDRes);

  constexpr real kMin = std::numeric_limits<real>::min();
  real const ratioRes = errorRes / (normRes + kMin);
  real const ratioDRes = errorDRes / (normDRes + kMin);

  MOCHI_LOG(
      "res ratio: %.4e (%.4e / %.4e). Dres ratio: %.4e (%.4e / %.4e)",
      ratioRes,
      errorRes,
      normRes,
      ratioDRes,
      errorDRes,
      normDRes);
}

template <typename T>
real SnleProblem<T>::ConsistencyCheckResNorm(real const finDiffStep, VerbosityLevel verbosity) {
  // Define variables for code readability.
  auto sol0 = this->solution.Duplicate();
  auto inc = AsView(this->increment);

  // Get residual.
  UpdateResidual();
  auto res = GetResidual();
  real const resNorm = Max(std::numeric_limits<real>::epsilon(), res.Norm());

  // Compute scale for finite difference step.
  real const eps = finDiffStep;
  real const scale = eps / (2_r * resNorm);

  // Forward step in residual direction: x + (eps/2) * residual / ||residual||.
  inc = res;
  inc *= scale;
  UpdateSolution();
  UpdateObjective();
  double const meritForward = GetObjective();

  // Reset solution.
  SetSolution(sol0);

  // Backward step in residual direction: x - (eps/2) * residual / ||residual||.
  inc = -inc;
  UpdateSolution();
  UpdateObjective();
  double const meritBackward = GetObjective();

  // Reset solution
  SetSolution(sol0);

  // Compute centered finite difference approximation.
  // dmerit/dx in direction residual/||residual|| ≈ (meritForward - meritBackward) / eps.
  // This should equal ||residual|| if residual = ∇merit.
  real const finDiffResNorm = static_cast<real>((meritForward - meritBackward) / eps);

  // Compute error.
  real const error = Abs(resNorm - finDiffResNorm);
  real const ratio = error / resNorm;

  if (verbosity >= VerbosityLevel::Verbose) {
    MOCHI_LOG(
        "\nConsistencyCheckResNorm: ||res|| = %.10e, finite diff = %.10e, error = %.10e, ratio = %.4e\n",
        resNorm,
        finDiffResNorm,
        error,
        ratio);
  }

  return ratio;
}

template <typename T>
void SnleProblem<T>::SortActors() {
  // Sort by DOF offset
  std::sort(
      this->actorResiduals.begin(), this->actorResiduals.end(), [](auto const& a, auto const& b) {
        return a.first < b.first;
      });
  std::sort(
      this->actorMatrices.begin(), this->actorMatrices.end(), [](auto const& a, auto const& b) {
        return a.first < b.first;
      });
  std::sort(
      this->actorConvergenceWeights.begin(),
      this->actorConvergenceWeights.end(),
      [](auto const& a, auto const& b) { return a.first < b.first; });
  std::sort(
      this->actorPreconditioners.begin(),
      this->actorPreconditioners.end(),
      [](auto const& a, auto const& b) { return std::get<0>(a) < std::get<0>(b); });
}

template <typename T>
IslandOperators<T> SnleProblem<T>::GetOperators() const {
  std::vector<std::pair<int, AnyMatrixView<T const>>> actorMatrixViews;
  actorMatrixViews.reserve(this->actorMatrices.size());
  for (auto&& [offset, mat] : this->actorMatrices) {
    actorMatrixViews.emplace_back(offset, AsConstView(*mat));
  }

  std::vector<AnyInteractionMatrixViewInfo<T const>> interactionMatrixViews;
  interactionMatrixViews.reserve(this->interactionMatrices.size());
  for (auto const& [rowOffset, colOffset, matrix, symmetricPair] : this->interactionMatrices) {
    interactionMatrixViews.emplace_back(rowOffset, colOffset, AsConstView(*matrix), symmetricPair);
  }

  std::vector<std::reference_wrapper<std::unique_ptr<ActorPreconditioner<T>>>>
      actorPreconditionersRef;
  std::vector<std::optional<PreconditionerType>> actorPreconditionerTypeHints;
  actorPreconditionersRef.reserve(this->actorPreconditioners.size());
  actorPreconditionerTypeHints.reserve(this->actorPreconditioners.size());

  for (auto const& prec : this->actorPreconditioners) {
    actorPreconditionersRef.emplace_back(std::get<1>(prec));
    actorPreconditionerTypeHints.emplace_back(std::get<2>(prec));
  }

  return {
      std::move(actorMatrixViews),
      std::move(interactionMatrixViews),
      std::move(actorPreconditionersRef),
      std::move(actorPreconditionerTypeHints)};
}

template struct SnleProblemFunctions<real>;
template class SnleProblem<real>;

} // namespace mochi
