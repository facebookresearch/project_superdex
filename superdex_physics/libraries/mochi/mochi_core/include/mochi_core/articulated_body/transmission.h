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

#include <mochi_core/articulated_body/articulated_body.h>
#include <mochi_core/articulated_body/articulated_body_params.h>
#include <mochi_core/linear_algebra/matrix_fwd.h>
#include <mochi_core/utils/assembly_params.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/transform_rt.h>

namespace mochi {

// Abstraction for an actuator attached to a scalar transmission. The actuator produces a
// generalized (conjugate) force in response to the transmission's current generalized coordinate
// (its "displacement" from the rest pose). It also takes the previous displacement and time step
// to allow for rate-dependent actuator responses. The semantic interpretation of the displacement
// and conjugate force depends on the concrete transmission (e.g. for a tendon, displacement ==
// length change and the conjugate force is tension; for a gearbox, displacement == output-shaft
// angle and the conjugate force is the output-shaft torque).
class TransmissionActuator {
 public:
  virtual ~TransmissionActuator() = default;
  // The Error argument here can be used to enforce any sub-class specific constraints on the values
  // of state variables. Note that the number of state variables can be assumed to be correct and
  // enforced by the caller, since that is accessible through the transmission base interface.
  virtual void SetStateVariables(Span<real const> stateVariables, Error& error) = 0;
  virtual void GetStateVariables(Span<real> outStateVariables) const = 0;
  virtual int GetNumStateVariables() const = 0;

  virtual void EnergyGradientHessian(
      real displacement,
      real prevDisplacement,
      real timeStep,
      // (Incremental) potential minimized by the actuator. This is not strictly necessary, and may
      // not be possible to define in all cases, but can be used to improve performance via
      // merit-based line searches. Pass nullptr if not needed.
      real* outEnergy,
      // Derivative of energy w.r.t. displacement (if energy exists), or specified directly from the
      // actuator physics: the energy-conjugate generalized force on the transmission. Pass nullptr
      // if not needed.
      real* outGradient,
      // Derivative of the conjugate force w.r.t. displacement. Pass nullptr if not needed.
      real* outHessian) const = 0;
};

// Actuator that drives the transmission toward a target generalized coordinate (its internal state
// variable) using a linear spring + linear damper penalty. By default the actuator only produces
// non-negative forces (rope-like behavior); set `allowCompressiveForce = true` for transmissions
// that can transmit force in both directions (e.g. a gearbox or rigid linkage).
class DisplacementControlActuator : public TransmissionActuator {
 public:
  DisplacementControlActuator(
      real targetDisplacement,
      real stiffness,
      real damping,
      bool allowCompressiveForce = false)
      : _targetDisplacement(targetDisplacement),
        _stiffness(stiffness),
        _damping(damping),
        _allowCompressiveForce(allowCompressiveForce) {}
  int GetNumStateVariables() const override {
    return 1;
  }
  void SetStateVariables(Span<real const> stateVariables, Error& error) override {
    MOCHI_ERROR_RETURN(error);
    MOCHI_ASSERT_VERBOSE(isize(stateVariables) == 1, "Invalid number of state variables");
    _targetDisplacement = stateVariables[0];
  }
  void GetStateVariables(Span<real> outStateVariables) const override {
    MOCHI_ASSERT_VERBOSE(isize(outStateVariables) == 1, "Invalid number of state variables");
    outStateVariables[0] = _targetDisplacement;
  }

  void EnergyGradientHessian(
      real displacement,
      real previousDisplacement,
      real timeStep,
      real* outEnergy,
      real* outGradient,
      real* outHessian) const override {
    if (!outEnergy && !outGradient && !outHessian) {
      return;
    }
    real const dDisplacement = displacement - _targetDisplacement;
    real const velocity = (displacement - previousDisplacement) / timeStep;
    real const force = _stiffness * dDisplacement + _damping * velocity;
    if (!_allowCompressiveForce && force <= 0_r) {
      if (outEnergy) {
        *outEnergy = 0_r;
      }
      if (outGradient) {
        *outGradient = 0_r;
      }
      if (outHessian) {
        *outHessian = 0_r;
      }
      return;
    }
    if (outEnergy) {
      *outEnergy = 0.5_r * (_stiffness * Sqr(dDisplacement) + _damping * Sqr(velocity) * timeStep);
    }
    if (outGradient) {
      *outGradient = force;
    }
    if (outHessian) {
      *outHessian = _stiffness + _damping / timeStep;
    }
  }

 private:
  real _targetDisplacement = 0_r; // [generalized coordinate units]
  real _stiffness = 1e9_r; // [generalized force / generalized coordinate units]
  real _damping = 0_r; // [generalized force / (generalized coordinate / s) units]
  // If false (default), the produced force is clamped to be non-negative — appropriate for
  // rope-like transmissions (e.g. tendons) that go slack under compression. If true, the force is
  // returned as-is and can be negative — appropriate for transmissions that transmit force in
  // both directions (gearboxes, levers, rigid linkages).
  bool _allowCompressiveForce = false;
};

// Actuator that applies a fixed generalized force to the transmission, given by its internal state
// variable. By default the force is required to be non-negative (rope-like behavior); set
// `allowCompressiveForce = true` to allow negative forces for transmissions that transmit force
// in both directions.
class ForceControlActuator : public TransmissionActuator {
 public:
  ForceControlActuator(real force, Error& error, bool allowCompressiveForce = false)
      : _force(force), _allowCompressiveForce(allowCompressiveForce) {
    if (!_allowCompressiveForce) {
      EnforcePositiveForce(_force, error);
    }
  }
  int GetNumStateVariables() const override {
    return 1;
  }
  void SetStateVariables(Span<real const> stateVariables, Error& error) override {
    MOCHI_ERROR_RETURN(error);
    MOCHI_ASSERT_VERBOSE(isize(stateVariables) == 1, "Invalid number of state variables");
    if (!_allowCompressiveForce) {
      EnforcePositiveForce(stateVariables[0], error);
      MOCHI_ERROR_RETURN(error);
    }
    _force = stateVariables[0];
  }
  void GetStateVariables(Span<real> outStateVariables) const override {
    MOCHI_ASSERT_VERBOSE(isize(outStateVariables) == 1, "Invalid number of state variables");
    outStateVariables[0] = _force;
  }
  void EnergyGradientHessian(
      real displacement,
      real /* prevDisplacement */,
      real /* timeStep */,
      real* outEnergy,
      real* outGradient,
      real* outHessian) const override {
    if (outEnergy) {
      *outEnergy = _force * displacement;
    }
    if (outGradient) {
      *outGradient = _force;
    }
    if (outHessian) {
      *outHessian = 0_r;
    }
  }

 private:
  static void EnforcePositiveForce(real force, Error& error) {
    MOCHI_ERROR_IF(force < 0_r, error, "Actuator force must be positive");
  }
  real _force = 0_r; // [generalized force units]
  // See @ref DisplacementControlActuator::_allowCompressiveForce for semantics.
  bool _allowCompressiveForce = false;
};

// Model of a McKibben artificial muscle actuator, based on the model by Chou & Hannaford, given in
// this preprint: https://apps.dtic.mil/sti/pdfs/ADA299458.pdf
// It is controlled via a single pressure state variable. This concrete actuator subclass is only
// physically meaningful when the underlying transmission represents a tendon (length /
// tension). Its negative-force (compression) clamp is part of the pneumatic-muscle physical model
// and is therefore unconditional — unlike DisplacementControlActuator / ForceControlActuator there
// is no `allowCompressiveForce` opt-out.
class McKibbenActuator : public TransmissionActuator {
 public:
  McKibbenActuator(
      real pressure,
      real minimumPressure,
      real threadLength,
      real numberOfWraps,
      real deflatedStiffness,
      real deflatedEquilibriumLength)
      : _pressure(pressure),
        _minimumPressure(minimumPressure),
        _threadLengthSquared(threadLength * threadLength),
        _numberOfWrapsSquared(numberOfWraps * numberOfWraps),
        _deflatedStiffness(deflatedStiffness),
        _deflatedEquilibriumLength(deflatedEquilibriumLength) {
    // Full validity checks on parameters are done in the public API function calling this. Asserts
    // here are only to catch potential divide-by-zeros in debug.
    MOCHI_ASSERT_VERBOSE(_numberOfWrapsSquared > 0_r, "Must have nonzero number of wraps");
    MOCHI_ASSERT_VERBOSE(_threadLengthSquared > 0_r, "Must have nonzero thread length");
  }
  int GetNumStateVariables() const override {
    return 1;
  }
  void SetStateVariables(Span<real const> stateVariables, Error& error) override {
    MOCHI_ERROR_RETURN(error);
    MOCHI_ASSERT_VERBOSE(isize(stateVariables) == 1, "Invalid number of state variables");
    _pressure = stateVariables[0];
  }
  void GetStateVariables(Span<real> outStateVariables) const override {
    MOCHI_ASSERT_VERBOSE(isize(outStateVariables) == 1, "Invalid number of state variables");
    outStateVariables[0] = _pressure;
  }
  void EnergyGradientHessian(
      real displacement,
      real /* prevDisplacement */,
      real /* timeStep */,
      real* outEnergy,
      real* outGradient,
      real* outHessian) const override {
    real const L = _deflatedEquilibriumLength + displacement;
    real const pressureCoefficient = Max(0_r, _pressure - _minimumPressure) *
        (0.25_r * _threadLengthSquared / (_numberOfWrapsSquared * kPI));
    real const unclampedTension = pressureCoefficient * (3_r * L * L / _threadLengthSquared - 1_r) +
        _deflatedStiffness * displacement;
    if (unclampedTension < 0_r) {
      if (outEnergy) {
        *outEnergy = 0_r;
      }
      if (outGradient) {
        *outGradient = 0_r;
      }
      if (outHessian) {
        *outHessian = 0_r;
      }
      return;
    }
    if (outEnergy) {
      *outEnergy = pressureCoefficient * (L * L * L / _threadLengthSquared - displacement) +
          0.5_r * _deflatedStiffness * displacement * displacement;
    }
    if (outGradient) {
      *outGradient = unclampedTension;
    }
    if (outHessian) {
      *outHessian = pressureCoefficient * (6_r * L / _threadLengthSquared) + _deflatedStiffness;
    }
  }

 private:
  // Pressure of the inner tube, considered as an input control variable
  real _pressure = 0_r; // [Pa]
  // Minimum pressure for inner tube to contact outer braid
  real _minimumPressure = 0_r; // [Pa]
  // Constructor takes raw thread length, but it is always squared in the model evaluation
  real _threadLengthSquared = 0_r; // [m^2]
  // Number of wraps squared
  real _numberOfWrapsSquared = 0_r; // dimensionless
  // Muscle stiffness below minimum pressure
  real _deflatedStiffness = 0_r; // [N/m]
  // Equillibrium muscle length below minimum pressure
  real _deflatedEquilibriumLength = 0_r; // [m]
};

// Abstract scalar transmission: maps a subset of articulated joint DoFs to a single generalized
// coordinate ("displacement" from the rest pose), and exposes an optional actuator that produces
// an energy-conjugate generalized force on that coordinate. The classic example is a tendon
// (joint DoFs ↔ tendon length change), but the same abstraction applies to gearboxes, levers, and
// other coupled DoFs.
class Transmission {
 public:
  virtual ~Transmission() = default;
  Transmission(Transmission&&) = default;
  Transmission& operator=(Transmission&&) = default;
  Transmission(Transmission const&) = delete;
  Transmission& operator=(Transmission const&) = delete;
  Transmission() = default;

  Transmission(std::unique_ptr<TransmissionActuator>&& actuator)
      : _actuator(std::move(actuator)) {};

  // Need non-const accessor to update the owned actuator's state variables.
  TransmissionActuator* GetActuator() {
    return _actuator.get();
  }
  TransmissionActuator const* GetActuator() const {
    return _actuator.get();
  }
  void SetActuator(std::unique_ptr<TransmissionActuator>&& actuator) {
    _actuator = std::move(actuator);
  }

  bool HasActuator() const {
    return _actuator != nullptr;
  }

  // Generalized coordinate of the transmission at the given pose, relative to its value at the
  // rest pose (the rest pose is the all-zero pose for revolute / prismatic DoFs). For a
  // LinearTransmission with tendon-like radii, this equals the tendon length minus its rest length.
  //
  // `linkTransforms` are per-link world-from-CoM transforms evaluated at the given pose. Subclasses
  // whose displacement is a closed-form function of the reduced pose (e.g. LinearTransmission)
  // ignore them; pose-dependent transmissions (e.g. SpatialTendon) use them directly instead of
  // running forward kinematics.
  virtual real Displacement(Span<TransformRT const> linkTransforms, Span<real const> pose)
      const = 0;

  // Overwrites `outJacobian` with d(Displacement)/d(reducedDof) at the supplied configuration.
  // The scalar-output Jacobian is represented as one moment arm per reduced DoF, using the same
  // reduced tangent/Lie convention as `bodyJacobian`. Its size must equal bodyJacobian.Cols().
  virtual void DisplacementJacobian(
      Span<TransformRT const> linkTransforms,
      RowMatrixView<real const> bodyJacobian,
      Span<real> outJacobian) const = 0;

  // `currentLinkTransforms` / `stageStartLinkTransforms` are per-link world-from-CoM transforms at
  // the corresponding poses. The body Jacobian (d(link transforms)/d(reduced DoFs), Lie/angular
  // convention, with 6 rows per link as laid out by mochi::articulated::Jacobian) is at the current
  // pose. LinearTransmission ignores the link transforms and bodyJacobian.
  virtual void AddObjResDRes(
      Span<TransformRT const> currentLinkTransforms,
      Span<TransformRT const> stageStartLinkTransforms,
      RowMatrixView<real const> bodyJacobian,
      Span<real const> currentPose,
      Span<real const> stageStartPose,
      real timeStep,
      AssemblyParams const& params,
      double& outObjective,
      Span<real> outResidual,
      Matrix<real>& outDResidual) const = 0;

 protected:
  std::unique_ptr<TransmissionActuator> _actuator = nullptr;
};

// Concrete transmission whose generalized coordinate is a linear (fixed weighted) sum of
// single-DoF joint values: displacement = sum_i coefficient_i * pose[jointIndex_i]. The
// coefficient_i values are the entries of the (constant) Jacobian dq_trans/dq_joint. For a tendon,
// |coefficient_i| has units of length per joint DoF (i.e. the moment arm / radius), and its sign
// encodes whether the tendon length increases or decreases with the joint DoF.
class LinearTransmission : public Transmission {
 public:
  LinearTransmission(
      Span<int const> jointIndices,
      Span<real const> jointCoefficients,
      // Used at construction time to look up each joint's pose / dofs offsets, which are then
      // stored on the transmission. The articulated actor's DoF layout is fixed after creation, so
      // the precomputed offsets stay valid for the transmission's lifetime.
      Span<ArticulatedDofInfo const> dofInfo,
      Span<ArticulatedPoseInfo const> poseInfo,
      // Null actuator by default; the desired type can be attached later.
      std::unique_ptr<TransmissionActuator>&& actuator = std::unique_ptr<TransmissionActuator>());

  real Displacement(Span<TransformRT const> linkTransforms, Span<real const> pose) const override;

  void DisplacementJacobian(
      Span<TransformRT const> linkTransforms,
      RowMatrixView<real const> bodyJacobian,
      Span<real> outJacobian) const override;

  void AddObjResDRes(
      Span<TransformRT const> currentLinkTransforms,
      Span<TransformRT const> stageStartLinkTransforms,
      RowMatrixView<real const> bodyJacobian,
      Span<real const> currentPose,
      Span<real const> stageStartPose,
      real timeStep,
      AssemblyParams const& params,
      double& outObjective,
      Span<real> outResidual,
      Matrix<real>& outDResidual) const override;

  // Per-term precomputed data: the source joint index, the pose offset (where the joint's scalar
  // coordinate lives in the reduced pose), the dofs offset (where the joint's DoF lives in the
  // reduced residual / Jacobian matrices), and the signed coefficient (entry of the constant
  // Jacobian dq_trans/dq_joint).
  struct Term {
    int jointIndex;
    int poseOffset;
    int dofsOffset;
    real coefficient; // [generalized coordinate / joint DoF units]
  };

  // Non-owning view of the per-term data (joint index + coefficient + reduced-pose/DoF offsets),
  // valid for this transmission's lifetime.
  Span<Term const> GetTerms() const {
    return MakeConstSpan(_terms);
  }

 protected:
  DynamicArray<Term> _terms;
};

// Concrete transmission modeling a tendon routed through an ordered list of @ref RoutingElement
// (waypoints and linear-joint elements). Its generalized coordinate ("displacement") is the change
// in the combined routed-length-plus-linear-joint sum relative to the rest pose.
class SpatialTendon : public Transmission {
 public:
  // Builds a tendon from an ordered list of @ref RoutingElement (at least one; an all-linear-joint
  // list is allowed and degenerates to a LinearTransmission equivalent). The rest length is
  // computed once via forward kinematics at the all-zero-DoF rest pose, where linear-joint elements
  // contribute zero. The joint layout spans (`jointTypes`, `jointAxes`, `dofInfo`, `poseInfo`),
  // `parents`, `restTransforms`, and `comLocals` are used only at
  // construction time (rest-pose FK, linear-joint DoF / pose offset lookups, and the one-time
  // local-frame -> CoM-frame conversion of waypoint positions, which are then stored on the
  // tendon).
  //
  // Every Waypoint element must be adjacent to at least one other Waypoint in the routing list;
  // an isolated waypoint contributes no segment length and is therefore meaningless.
  //
  // Waypoint `localPosition` values in `routingElements` are interpreted in each link's local
  // frame — the same frame in which the link's mesh and geometry are authored (a.k.a. the link's
  // root-transform frame). The constructor converts them once to the link's CoM frame for
  // internal storage, using `comLocals[i]` — the CoM offset of link `i` in its local frame.
  // `comLocals` must have the same length as `parents`.
  SpatialTendon(
      Span<RoutingElement const> routingElements,
      Span<ArticulatedJointType const> jointTypes,
      Span<Real3 const> jointAxes,
      Span<ArticulatedDofInfo const> dofInfo,
      Span<ArticulatedPoseInfo const> poseInfo,
      Span<int const> parents,
      Span<ArticulatedRestTransform const> restTransforms,
      Span<Real3 const> comLocals,
      std::unique_ptr<TransmissionActuator>&& actuator = std::unique_ptr<TransmissionActuator>());

  real Displacement(Span<TransformRT const> linkTransforms, Span<real const> pose) const override;

  void DisplacementJacobian(
      Span<TransformRT const> linkTransforms,
      RowMatrixView<real const> bodyJacobian,
      Span<real> outJacobian) const override;

  void AddObjResDRes(
      Span<TransformRT const> currentLinkTransforms,
      Span<TransformRT const> stageStartLinkTransforms,
      RowMatrixView<real const> bodyJacobian,
      Span<real const> currentPose,
      Span<real const> stageStartPose,
      real timeStep,
      AssemblyParams const& params,
      double& outObjective,
      Span<real> outResidual,
      Matrix<real>& outDResidual) const override;

  // Non-owning view of the ordered routing elements, valid for this tendon's lifetime.
  //
  // Waypoint `localPosition` values are returned in each link's CoM frame (the internal
  // representation), suitable for use with CoM-frame `linkTransforms` from
  // `GetLinkTransformsComFromPose` / `ComputeTransformsFromReducedPose`.
  Span<RoutingElement const> GetRoutingElements() const;

  // Precomputed per-linear-joint-element offsets into the reduced pose and reduced DoFs (in the
  // same order as the corresponding LinearJoint entries of _routingElements). Avoids re-reading
  // the joint layout at each call to Displacement / AddObjResDRes.
  struct LinearJointOffsets {
    int poseOffset;
    int dofsOffset;
  };

 protected:
  DynamicArray<RoutingElement> _routingElements; // Ordered routing elements
  DynamicArray<LinearJointOffsets> _linearJointOffsets;
  real _restLength = 0_r; // Combined routed length + linear-joint sum at the rest pose [m]
};

} // namespace mochi
