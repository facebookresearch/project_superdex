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

/// @brief Structure to monitor convergence with the norm of the residual induced by the
/// criterion-owned dot operation \f$D\f$: \f$\sqrt{D(r,r)}\f$, e.g. the L2-norm \f$\|r\|_2\f$ with
/// @ref UsualDot. This structure is helpful for PCG (where the residual is explicitly available).
template <typename Dot = UsualDot, typename Scalar = real>
class StatusResidualL2 {
 protected:
  details::StopTest<Scalar> _check;
  Dot _dot{};
  std::optional<krylov::RecyclingBin<Scalar>> _container;

 public:
  /// @brief Constructor
  /// @param relTol Relative tolerance for the convergence criterion ||r|| <= max( relTol ||b||,
  /// absTol )
  /// @param absTol Absolute tolerance for the convergence criterion (see above).
  /// @param relDivTol Relative tolerance for the divergence criterion ||r|| > relDivTol ||b||
  explicit StatusResidualL2(
      Scalar relTol,
      Scalar absTol,
      Scalar relDivTol,
      int retainedDirections = 0,
      Dot dot = Dot{})
      : _check{relTol, absTol, relDivTol}, _dot(dot), _container{std::nullopt} {
    if (retainedDirections > 0) {
      _container.emplace(retainedDirections);
    }
  }

  [[nodiscard]] Dot const& GetDot() const {
    return _dot;
  }

  /// @brief Sets the tolerance scaling to \f$D(b,b)\f$, where \f$D\f$ is the criterion-owned
  /// dot operation.
  ///
  /// @note The preconditioner and preconditioned right-hand side arguments are unused.
  template <typename RhsType, typename PreconditionerType, typename PreconditionedRhsType>
  void SetScaling(RhsType const& b, PreconditionerType&& /*M*/, PreconditionedRhsType& /*z*/) {
    auto const bNormSqr = static_cast<Scalar>(_dot.NormSqr(b));
    _check.SetScaling(bNormSqr);
  }

  /// @brief Collectively sets the tolerance scaling to \f$D(b,b)\f$, where \f$D\f$ is the
  /// criterion-owned dot operation.
  ///
  /// @pre Each participating worker calls this method once in the same collective phase with a
  /// unique worker index and an equivalently initialized criterion instance.
  /// @pre Each worker passes its own @ref ParallelDot copy. All copies share the same collective
  /// state.
  /// @pre The row ranges form a disjoint partition of @p b.
  /// @pre For a custom dot operation, summing its row-range products is mathematically equivalent
  /// to its full-vector squared-norm operation. Floating-point reduction order may differ.
  ///
  /// @note Performs one @ref ParallelDot reduction and does not apply the preconditioner. The
  /// @p z workspace is unused and need not be initialized.
  template <typename RhsType, typename PreconditionedRhsType, typename Idx, typename DotScalar>
  void SetConcurrentScaling(
      RhsType const& b,
      PreconditionedRhsType const& /*z*/,
      Idx rowStart,
      Idx rowEnd,
      int workerIdx,
      ParallelDot<DotScalar> const& parDot) {
    auto const bNormSqr = static_cast<Scalar>(parDot.Dot(_dot, b, b, rowStart, rowEnd, workerIdx));
    _check.SetScaling(bNormSqr);
  }

  template <typename Vector>
  [[nodiscard]] IterationStatus
  CheckStatus(int iter, Vector const& r, Vector const& /*z*/, Vector const& p, Vector const& Ap) {
    auto const resNormSqr = static_cast<Scalar>(_dot.NormSqr(r));
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
    auto const rNormSqr = static_cast<Scalar>(parDot.Dot(_dot, r, r, rowStart, rowEnd, workerIdx));
    return ParallelCheckStatus(iter, rNormSqr);
  }

  [[nodiscard]] IterationStatus ParallelCheckStatus(int iter, Scalar rNormSqr) {
    MOCHI_ASSERT(!_container.has_value(), "Parallel status check does not support recycling.");
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
    auto const zNormSqr = static_cast<Scalar>(_dot.NormSqr(z));
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
    auto const zNormSqr = static_cast<Scalar>(parDot.Dot(_dot, z, z, rowStart, rowEnd, workerIdx));
    return ParallelCheckStatus(iter, zNormSqr);
  }

  [[nodiscard]] IterationStatus ParallelCheckStatus(int iter, Scalar zNormSqr) {
    MOCHI_ASSERT(!_container.has_value(), "Parallel status check does not support recycling.");
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

  /// @brief Computes \f$z=M^{-1}b\f$ using @p M and sets the tolerance scaling to
  /// \f$D(z,z)\f$, where \f$D\f$ is the criterion-owned dot operation.
  template <typename RhsType, typename PreconditionerType, typename PreconditionedRhsType>
  void SetScaling(RhsType const& b, PreconditionerType&& M, PreconditionedRhsType& z) {
    Solve(M, b, z); // z = M^{-1} b
    auto const zNormSqr = static_cast<Scalar>(_dot.NormSqr(z));
    _check.SetScaling(zNormSqr);
  }

  /// @brief Collectively sets the tolerance scaling to \f$D(z,z)\f$, where \f$D\f$ is the
  /// criterion-owned dot operation.
  ///
  /// @pre On entry, the elements of @p z from @p rowStart through @p rowEnd (exclusive) equal the
  /// corresponding elements of \f$M^{-1}b\f$, are visible to this worker, and remain unchanged
  /// during the call.
  /// @pre Each participating worker calls this method once in the same collective phase with a
  /// unique worker index and an equivalently initialized criterion instance.
  /// @pre Each worker passes its own @ref ParallelDot copy. All copies share the same collective
  /// state.
  /// @pre The row ranges form a disjoint partition of @p z.
  /// @pre For a custom dot operation, summing its row-range products is mathematically equivalent
  /// to its full-vector squared-norm operation. Floating-point reduction order may differ.
  ///
  /// @note Performs one @ref ParallelDot reduction and does not apply the preconditioner. The
  /// caller supplies the precomputed @p z. The @p b argument is unused.
  template <typename RhsType, typename PreconditionedRhsType, typename Idx, typename DotScalar>
  void SetConcurrentScaling(
      RhsType const& /*b*/,
      PreconditionedRhsType const& z,
      Idx rowStart,
      Idx rowEnd,
      int workerIdx,
      ParallelDot<DotScalar> const& parDot) {
    auto const zNormSqr = static_cast<Scalar>(parDot.Dot(_dot, z, z, rowStart, rowEnd, workerIdx));
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
    auto const rTz = static_cast<Scalar>(_dot(r, z));
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
    auto const rTz = static_cast<Scalar>(parDot.Dot(_dot, r, z, rowStart, rowEnd, workerIdx));
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

  /// @brief Computes \f$z=M^{-1}b\f$ using @p M and sets the tolerance scaling to
  /// \f$D(b,z)\f$, where \f$D\f$ is the criterion-owned dot operation.
  template <typename RhsType, typename PreconditionerType, typename PreconditionedRhsType>
  void SetScaling(RhsType const& b, PreconditionerType&& M, PreconditionedRhsType& z) {
    Solve(M, b, z); // z = M^{-1} b
    auto const bTz = static_cast<Scalar>(_dot(b, z)); // <b, M^{-1} b>
    _check.SetScaling(bTz);
  }

  /// @brief Collectively sets the tolerance scaling to \f$D(b,z)\f$, where \f$D\f$ is the
  /// criterion-owned dot operation.
  ///
  /// @pre On entry, the elements of @p z from @p rowStart through @p rowEnd (exclusive) equal the
  /// corresponding elements of \f$M^{-1}b\f$, are visible to this worker, and remain unchanged
  /// during the call.
  /// @pre Each participating worker calls this method once in the same collective phase with a
  /// unique worker index and an equivalently initialized criterion instance.
  /// @pre Each worker passes its own @ref ParallelDot copy. All copies share the same collective
  /// state.
  /// @pre The row ranges form a disjoint partition of @p b and @p z.
  /// @pre The criterion-owned dot operation is mathematically additive across the worker row
  /// ranges. Floating-point reduction order may differ.
  ///
  /// @note Performs one @ref ParallelDot reduction and does not apply the preconditioner. The
  /// caller supplies the precomputed @p z.
  template <typename RhsType, typename PreconditionedRhsType, typename Idx, typename DotScalar>
  void SetConcurrentScaling(
      RhsType const& b,
      PreconditionedRhsType const& z,
      Idx rowStart,
      Idx rowEnd,
      int workerIdx,
      ParallelDot<DotScalar> const& parDot) {
    auto const bTz = static_cast<Scalar>(parDot.Dot(_dot, b, z, rowStart, rowEnd, workerIdx));
    _check.SetScaling(bTz);
  }
};

/// @brief Whether the stop criterion @p C, one of those that PCG supports, monitors the residual
/// itself (@ref StatusResidualL2 with scalar type @p Scalar and any dot operation) rather than a
/// norm that needs the preconditioned residual.
///
/// @note Undefined for other criteria, so that querying them doesn't compile.
template <typename C, typename Scalar>
struct IsResidualCriterion;

template <typename Dot, typename Scalar>
struct IsResidualCriterion<StatusResidualL2<Dot, Scalar>, Scalar> {
  static constexpr bool kValue = true;
};

template <typename Dot, typename Scalar>
struct IsResidualCriterion<StatusPreconditionedResidualL2<Dot, Scalar>, Scalar> {
  static constexpr bool kValue = false;
};

template <typename Dot, typename Scalar>
struct IsResidualCriterion<StatusResidualPreconditionerInduced<Dot, Scalar>, Scalar> {
  static constexpr bool kValue = false;
};

/// @brief Shorthand for @ref IsResidualCriterion.
template <typename C, typename Scalar>
inline constexpr bool kIsResidualCriterion = IsResidualCriterion<C, Scalar>::kValue;

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
