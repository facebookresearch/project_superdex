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

#include <mochi_core/linear_algebra/krylov/iteration_status.h>
#include <mochi_core/linear_algebra/krylov/recycling_bin.h>
#include <mochi_core/linear_algebra/krylov/tools/tensor_functions.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/task_scheduler.h>

#include <optional>

namespace mochi::krylov::details {

template <typename Scalar>
class StopTest {
 protected:
  /// @brief Absolute tolerance
  Scalar _absTol;

  /// @brief Relative tolerance
  Scalar _relTol;

  /// @brief Upper-limit to detect divergence
  Scalar _relDivTol;

  /// @brief Initial norm squared
  Scalar _startingNormSqr = {};

  /// @brief Latest residual norm squared
  Scalar _resNormSqr = {};

  /// @brief Threshold to declare convergence. The value is tested against the squared norm of the
  /// residual, i.e. resNormSqr <= convergenceThreshold
  Scalar _convergenceThreshold = {};

  /// @brief Threshold to declare divergence. The value is tested against the squared norm of the
  /// residual, i.e. resNormSqr > divergenceThreshold
  Scalar _divergenceThreshold = {};

 public:
  /// @brief Constructor
  /// @param relTol Relative tolerance for the convergence criterion ||r|| <= max( relTol ||b||,
  /// absTol )
  /// @param absTol Absolute tolerance for the convergence criterion (see above).
  /// @param relDivTol Relative tolerance for the divergence criterion ||r|| > relDivTol ||b||
  explicit StopTest(Scalar relTol, Scalar absTol, Scalar relDivTol)
      : _absTol(absTol), _relTol(relTol), _relDivTol(relDivTol) {}

  IterationStatus CheckStatus(int /*iter*/, Scalar latestNormSqr) {
    _resNormSqr = latestNormSqr;
    if (!IsFinite(_resNormSqr) || (_resNormSqr > _divergenceThreshold) ||
        (_resNormSqr < -_convergenceThreshold)) {
      return IterationStatus::DivergedRes;
    }
    if (_resNormSqr < Scalar(0)) {
      _resNormSqr = Scalar(0);
    }
    if (_resNormSqr <= _convergenceThreshold) {
      return (_resNormSqr <= _absTol * _absTol) ? IterationStatus::ConvergedAtol
                                                : IterationStatus::ConvergedRtol;
    }
    return IterationStatus::Active;
  }

  [[nodiscard]] Scalar GetLatestResidualNorm() const {
    return Sqrt(_resNormSqr);
  }

  [[nodiscard]] Scalar GetLatestRelativeResidualNorm() const {
    return _startingNormSqr == Scalar{} ? Scalar{} : Sqrt(_resNormSqr / _startingNormSqr);
  }

  [[nodiscard]] Scalar GetLatestResidualNormSqr() const {
    return _resNormSqr;
  }

  /// @brief Scaling for the relative tolerance and the divergence tolerance
  /// This scale is typically || b ||^2 (i.e. the square of norm of b)
  void SetScaling(Scalar scaleSqr) {
    _startingNormSqr = scaleSqr;
    //--- Update divergenceThreshold and convergenceThreshold
    _divergenceThreshold = (_startingNormSqr * _relDivTol) * _relDivTol;
    _convergenceThreshold = Max(_absTol * _absTol, (_startingNormSqr * _relTol) * _relTol);
  }
};

} // namespace mochi::krylov::details

namespace mochi::krylov {

/// @brief Structure to monitor convergence with the L2-norm of the residual: \f$\|r\|_2\f$.
/// This structure is helpful for PCG (where the residual is explicitly available).
template <typename Dot = UsualDot, typename Scalar = real>
class StatusResidualL2 {
 protected:
  details::StopTest<Scalar> _check;
  Dot _dot{};
  std::optional<krylov::RecyclingBin<Scalar>> _container;

 public:
  /// @brief Constructor
  /// @param relTol Relative tolerance for the convergence criterion ||r||_2 <= max( relTol ||b||_2,
  /// absTol )
  /// @param absTol Absolute tolerance for the convergence criterion (see above).
  /// @param relDivTol Relative tolerance for the divergence criterion ||r||_2 > relDivTol ||b||_2
  explicit StatusResidualL2(
      Scalar relTol,
      Scalar absTol,
      Scalar relDivTol,
      int retainedDirections = 0)
      : _check{relTol, absTol, relDivTol}, _container{std::nullopt} {
    if (retainedDirections > 0) {
      _container.emplace(retainedDirections);
    }
  }

  /// @brief Scaling for the relative tolerance and the divergence tolerance
  /// @note The scaling is ||b||_{2}
  template <typename VectorIn, typename Preconditioner, typename VectorOut>
  void SetScaling(VectorIn const& b, Preconditioner&& /*M*/, VectorOut& /*z*/) {
    auto bNormSqr = static_cast<Scalar>(_dot.NormSqr(b));
    _check.SetScaling(bNormSqr);
  }

  template <typename Vector>
  [[nodiscard]] IterationStatus
  CheckStatus(int iter, Vector const& r, Vector const& /*z*/, Vector const& p, Vector const& Ap) {
    auto resNormSqr = static_cast<Scalar>(_dot.NormSqr(r));
    if ((iter > 0) && (_container.has_value())) {
      _container->Insert(p, Ap);
    }
    return _check.CheckStatus(iter, resNormSqr);
  }

  template <typename Vector, typename Idx, typename DotScalar>
  [[nodiscard]] IterationStatus ParallelCheckStatus(
      int iter,
      Vector const& r,
      Vector const& /*z*/,
      Vector const& /*p*/,
      Vector const& /*Ap*/,
      Idx rowStart,
      Idx rowEnd,
      int workerIdx,
      ParallelDot<DotScalar> const& parDot) {
    MOCHI_ASSERT(!_container.has_value(), "Parallel status check does not support recycling.");
    auto rNormSqr = static_cast<Scalar>(parDot.Dot(_dot, r, r, rowStart, rowEnd, workerIdx));
    return _check.CheckStatus(iter, rNormSqr);
  }

  [[nodiscard]] Scalar GetLatestResidualNorm() const {
    return _check.GetLatestResidualNorm();
  }

  [[nodiscard]] Scalar GetLatestRelativeResidualNorm() const {
    return _check.GetLatestRelativeResidualNorm();
  }

  [[nodiscard]] Scalar GetLatestResidualNormSqr() const {
    return _check.GetLatestResidualNormSqr();
  }

  [[nodiscard]] auto GetRetainedDirections() {
    MOCHI_ASSERT(_container, "Recycling bin has not been set.");
    return _container->GetRetainedDirections();
  }

  [[nodiscard]] auto GetRetainedMappedDirections() {
    MOCHI_ASSERT(_container, "Recycling bin has not been set.");
    return _container->GetRetainedMappedDirections();
  }
};

/// @brief Structure to monitor convergence with the L2-norm of the preconditioned residual:
/// \f$\|z\|_2\f$ where \f$z = M^{-1}r\f$.
/// This structure is helpful for PCG (where the residual is explicitly available).
template <typename Dot = UsualDot, typename Scalar = real>
class StatusPreconditionedResidualL2 {
 protected:
  details::StopTest<Scalar> _check;
  Dot _dot{};
  std::optional<krylov::RecyclingBin<Scalar>> _container;

 public:
  /// @brief Constructor
  /// @param relTol Relative tolerance for the convergence criterion ||M^{-1} r||_2 <= max(relTol
  /// ||M^{-1} b||_2, absTol)
  /// @param absTol Absolute tolerance for the convergence criterion (see above).
  /// @param relDivTol Relative tolerance for the divergence criterion ||M^{-1} r||_2 > relDivTol
  /// ||M^{-1} b||_2
  explicit StatusPreconditionedResidualL2(
      Scalar relTol,
      Scalar absTol,
      Scalar relDivTol,
      int retainedDirections = 0)
      : _check{relTol, absTol, relDivTol}, _container{std::nullopt} {
    if (retainedDirections > 0) {
      _container.emplace(retainedDirections);
    }
  }

  template <typename Vector>
  [[nodiscard]] IterationStatus
  CheckStatus(int iter, Vector const& /*r*/, Vector const& z, Vector const& p, Vector const& Ap) {
    auto zNormSqr = static_cast<Scalar>(_dot.NormSqr(z));
    if ((iter > 0) && (_container.has_value())) {
      _container->Insert(p, Ap);
    }
    return _check.CheckStatus(iter, zNormSqr);
  }

  template <typename Vector, typename Idx, typename DotScalar>
  [[nodiscard]] IterationStatus ParallelCheckStatus(
      int iter,
      Vector const& /*r*/,
      Vector const& z,
      Vector const& /*p*/,
      Vector const& /*Ap*/,
      Idx rowStart,
      Idx rowEnd,
      int workerIdx,
      ParallelDot<DotScalar> const& parDot) {
    MOCHI_ASSERT(!_container.has_value(), "Parallel status check does not support recycling.");
    auto zNormSqr = static_cast<Scalar>(parDot.Dot(_dot, z, z, rowStart, rowEnd, workerIdx));
    return _check.CheckStatus(iter, zNormSqr);
  }

  [[nodiscard]] Scalar GetLatestResidualNorm() const {
    return _check.GetLatestResidualNorm();
  }

  [[nodiscard]] Scalar GetLatestRelativeResidualNorm() const {
    return _check.GetLatestRelativeResidualNorm();
  }

  [[nodiscard]] Scalar GetLatestResidualNormSqr() const {
    return _check.GetLatestResidualNormSqr();
  }

  [[nodiscard]] auto GetRetainedDirections() {
    MOCHI_ASSERT(_container.has_value(), "Recycling bin has not been set.");
    return _container->GetRetainedDirections();
  }

  [[nodiscard]] auto GetRetainedMappedDirections() {
    MOCHI_ASSERT(_container.has_value(), "Recycling bin has not been set.");
    return _container->GetRetainedMappedDirections();
  }

  /// @brief Scaling for the relative tolerance and the divergence tolerance
  /// @note The scaling is ||M^{-1} b||_{2}
  /// @note z is a workspace to store M^{-1} b
  template <typename VectorIn, typename Preconditioner, typename VectorOut>
  void SetScaling(VectorIn const& b, Preconditioner&& M, VectorOut& z) {
    Solve(M, b, z); // z = M^{-1} b
    auto zNormSqr = static_cast<Scalar>(_dot.NormSqr(z));
    _check.SetScaling(zNormSqr);
  }
};

/// @brief Structure to monitor convergence with the preconditioner-induced norm of the residual:
/// \f$\sqrt{\langle r, z \rangle} = \sqrt{r^T M^{-1} r}\f$.
template <typename Dot = UsualDot, typename Scalar = real>
class StatusResidualPreconditionerInduced {
 protected:
  details::StopTest<Scalar> _check;
  Dot _dot{};

 public:
  /// @brief Constructor
  /// @param relTol Relative tolerance for the convergence criterion sqrt(<r, M^{-1}r>) <=
  /// max(relTol * sqrt(<b, M^{-1}b>), absTol)
  /// @param absTol Absolute tolerance for the convergence criterion (see above).
  /// @param relDivTol Relative tolerance for the divergence criterion sqrt(<r, M^{-1}r>) >
  /// relDivTol * sqrt(<b, M^{-1}b>)
  explicit StatusResidualPreconditionerInduced(Scalar relTol, Scalar absTol, Scalar relDivTol)
      : _check{relTol, absTol, relDivTol} {}

  template <typename Vector>
  [[nodiscard]] IterationStatus CheckStatus(
      int iter,
      Vector const& r,
      Vector const& z,
      Vector const& /*p*/,
      Vector const& /*Ap*/) {
    auto rTz = static_cast<Scalar>(_dot(r, z));
    return _check.CheckStatus(iter, rTz);
  }

  template <typename Vector, typename Idx, typename DotScalar>
  [[nodiscard]] IterationStatus ParallelCheckStatus(
      int iter,
      Vector const& r,
      Vector const& z,
      Vector const& /*p*/,
      Vector const& /*Ap*/,
      Idx rowStart,
      Idx rowEnd,
      int workerIdx,
      ParallelDot<DotScalar> const& parDot) {
    auto rTz = static_cast<Scalar>(parDot.Dot(_dot, r, z, rowStart, rowEnd, workerIdx));
    return _check.CheckStatus(iter, rTz);
  }

  [[nodiscard]] Scalar GetLatestResidualNorm() const {
    return _check.GetLatestResidualNorm();
  }

  [[nodiscard]] Scalar GetLatestRelativeResidualNorm() const {
    return _check.GetLatestRelativeResidualNorm();
  }

  [[nodiscard]] Scalar GetLatestResidualNormSqr() const {
    return _check.GetLatestResidualNormSqr();
  }

  /// @brief Scaling for the relative tolerance and the divergence tolerance
  /// @note The scaling is <b, M^{-1}b>
  /// @note z is a workspace to store M^{-1} b
  template <typename VectorIn, typename Preconditioner, typename VectorOut>
  void SetScaling(VectorIn const& b, Preconditioner&& M, VectorOut& z) {
    Solve(M, b, z); // z = M^{-1} b
    auto bTz = static_cast<Scalar>(_dot(b, z)); // <b, M^{-1} b>
    _check.SetScaling(bTz);
  }
};

/// @brief Structure to monitor convergence
/// This structure is helpful for GMRes where the norm of the residual
/// is implicitly available.
template <typename Scalar>
class StatusImplicitResidualNorm {
 protected:
  details::StopTest<Scalar> _check;
  std::optional<krylov::RecyclingBin<Scalar>> _container;

 public:
  /// @brief Constructor
  /// @param relTol Relative tolerance for the convergence criterion ||r|| <= max( relTol ||b||,
  /// absTol )
  /// @param absTol Absolute tolerance for the convergence criterion (see above).
  /// @param relDivTol Relative tolerance for the divergence criterion ||r|| > relDivTol ||b||
  explicit StatusImplicitResidualNorm(
      Scalar relTol,
      Scalar absTol,
      Scalar relDivTol,
      int retainedDirections = 0)
      : _check{relTol, absTol, relDivTol}, _container{std::nullopt} {
    if (retainedDirections > 0) {
      _container.emplace(retainedDirections);
    }
  }

  template <typename VectorP, typename VectorAp>
  [[nodiscard]] IterationStatus
  CheckStatus(int iter, Scalar normEstimate, VectorP const& p, VectorAp const& Ap) {
    auto normSqr = Sqr(normEstimate);
    if ((iter > 0) && (_container.has_value())) {
      _container->Insert(p, Ap);
    }
    return _check.CheckStatus(iter, normSqr);
  }

  [[nodiscard]] Scalar GetLatestResidualNorm() const {
    return _check.GetLatestResidualNorm();
  }

  [[nodiscard]] Scalar GetLatestRelativeResidualNorm() const {
    return _check.GetLatestRelativeResidualNorm();
  }

  /// @brief Scaling for the relative tolerance and the divergence tolerance
  /// @param[in] initNorm Norm to scale the tolerance in the right hand side
  void SetScaling(Scalar initNorm) {
    auto normSqr = Sqr(initNorm);
    _check.SetScaling(normSqr);
  }
};

} // namespace mochi::krylov
