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

#include <mochi_core/articulated_body/articulated_body_params.h>
#include <mochi_core/utils/activations.h>
#include <mochi_core/utils/differentiability.h>
#include <mochi_core/utils/dtransform.h>
#include <mochi_core/utils/lie.h>
#include <mochi_core/utils/rigid_body_utils.h>

namespace mochi {

// Energy and rotational gradient of a rigid body's terms without a potential, which kernels report
// besides their totals. rotGradient must match the rotational rows of RigidGradient, from
// RigidSize::kDTrans.
struct RigidNonPotentialTerms {
  double energy = 0.0;
  Real3 rotGradient{};
};

// Function to add rigid-body gravity terms. The energy is relative to the stage start.
inline void AddRigidBodyGravity(
    TransformRT const& state,
    TransformRT const& stageStartPos,
    RigidBodyInertia const& inertia,
    Real3 gravityAcc,
    double* outEnergy,
    RigidGradient* outGradient) {
  if (!outEnergy && !outGradient) {
    return;
  }

  // Compute common values
  Vec4r mg = ToSimd(gravityAcc * inertia.GetMass());

  if (outEnergy) {
    (*outEnergy) -= Get0(VDot(state.VGetTranslation() - stageStartPos.VGetTranslation(), mg));
  }

  if (outGradient) {
    // Load output
    auto outGradientCom = Load<Vec4r>(outGradient->data() + 0);

    outGradientCom -= mg;

    // Store output
    Store<3>(outGradient->data() + 0, outGradientCom);
  }
}

/**
 * @brief Adds rigid-body inertia terms derived from a merit function as presented in the Rigid IPC
 *paper.
 *
 * This function computes and accumulates energy, gradient, and Hessian contributions related to the
 * rigid-body inertia of a system given its state and inertia properties. The computation depends on
 * the gradient target type specified by the template parameter.
 *
 * @tparam kGradTarget The gradient target type. Effectively, given the merit as f( qk, qk-1, Dqk-1)
 *the template dictates whether we return derivatives wrt to qk (aka Current), qk-1 (aka Previous),
 *or Dqk-1 (aka PreviousDelta). Possible values: GradTarget::Previous, GradTarget::PreviousDelta,
 * GradTarget::Current (default).
 * @tparam InertiaT The type for the second moment of inertia: real (isotropic inertia), Vec4r
 * (anisotropic unrotated inertia), VMatrix3x3r (arbitrary second moment of inertia).
 *
 * @param mass The mass of the rigid body.
 * @param inertia The second moment of inertia of the rigid body.
 * @param dtStage The time step duration of the current stage.
 * @param state The current transform state of the rigid body (position and rotation).
 * @param stageStartPos The transform state at the start of the current stage.
 * @param stageStartVel The rigid body velocity at the start of the current stage.
 * @param[out] outEnergy Pointer to a double to accumulate energy contributions. Can be nullptr if
 *not needed.
 * @param[out] outGradient Pointer to a RigidGradient to accumulate gradient contributions. Can be
 *nullptr if not needed.
 * @param[out] outHessian Pointer to a RigidHessian to accumulate Hessian contributions. Can be
 *nullptr if not needed.
 * @param[out] outNonPotential Unused, as these terms have a potential. For dispatch with
 * @ref AddRigidBodyInertiaNewtonEuler.
 */
template <GradTarget kGradTarget, typename InertiaT>
inline void AddRigidBodyInertiaFromMerit(
    real mass,
    InertiaT const& inertia,
    real dtStage,
    TransformRT const& state,
    TransformRT const& stageStartPos,
    RigidBodyVel const& stageStartVel,
    double* outEnergy,
    RigidGradient* outGradient,
    RigidHessian* outHessian,
    RigidNonPotentialTerms* /*outNonPotential*/ = nullptr) {
  // Early exit if no outputs are requested
  if (!outEnergy && !outGradient && !outHessian) {
    return;
  }

  // Compute inverse squared time step for scaling
  real const dtStage2i = 1_r / (dtStage * dtStage);

  // Compute scaled inertia
  InertiaT const scaledInertia = dtStage2i * inertia;

  // Compute change in center of mass velocity over the stage
  Vec4r deltaVCom = (state.VGetTranslation() - stageStartPos.VGetTranslation()) / dtStage -
      stageStartVel.GetVCom();

  // Extract rotation matrices from current and start states
  VMatrix3x3r currRot = VGetRotationMatrix(state); // Rk
  VMatrix3x3r stageStartRot = VGetRotationMatrix(stageStartPos); // Rk-1

  // Compute explicit time-stepped rotation matrix from start velocity
  VMatrix3x3r explicitRot = stageStartVel.EvalTimeSteppedRotation(stageStartRot, dtStage); // ~Rk
  VMatrix3x3r oldRot = 2_r * stageStartRot - explicitRot; // Rk-2

  // The rotation energy can be expressed as the sum of two terms:
  // Psi = -1/h^2 tr(Rk * J * ~RkT) = 1/h^2 tr(Rk * J * Rk-2T) -2/h^2 tr(Rk * J * Rk-1T)
  // Note that ~Rk = 2 Rk-1 - Rk-2, and Rk-2 = DRk-1T * Rk-1
  // GradTarget::Current - We add both terms together
  // GradTarget::Previous - We use both terms, transposed
  // GradTarget::PreviousDelta - We only use the first term, transposed
  // Transposing the matrices is convenient for using the same Lie derivative functions.
  // With GradTarget::Previous and GradTarget::PreviousDelta, the first term requires an additional
  // chain rule from Rk-2 to the actual target.
  VMatrix3x3r energyRotMatrix MOCHI_NO_INIT;
  [[maybe_unused]] VMatrix3x3r energyRotMatrixAux MOCHI_NO_INIT;
  [[maybe_unused]] VMatrix3x3r chain MOCHI_NO_INIT;
  if constexpr (kGradTarget == GradTarget::Current) {
    energyRotMatrix = lie::WeightedRotationDifferenceMatrix(currRot, explicitRot, scaledInertia);
  } else {
    energyRotMatrix = lie::WeightedRotationDifferenceMatrix(oldRot, currRot, -scaledInertia);
    energyRotMatrixAux =
        lie::WeightedRotationDifferenceMatrix(stageStartRot, currRot, 2_r * scaledInertia);
    VMatrix3x3r stageStartRotDeltaT = Dot3x3(oldRot, Transpose3x3(stageStartRot)); // DRk-1T
    if constexpr (kGradTarget == GradTarget::Previous) {
      chain = lie::DMultRotaRotRotbDRot(stageStartRotDeltaT);
    } else {
      chain = lie::DMultRotaRotTRotbDRot(stageStartRotDeltaT);
    }
  }

  // Accumulate energy if requested
  if (outEnergy) {
    // Translational kinetic energy contribution
    (*outEnergy) += static_cast<double>(NormSqr<3>(deltaVCom) * mass * 0.5_r);

    // Rotational kinetic energy contribution. Decompose the merit into two terms that both use
    // rotation differences, for two reasons: (1) the minimum merit is zero, (2) the merit can be
    // the same for all targets.
    (*outEnergy) -=
        static_cast<double>(lie::WeightedRotationDifferenceMerit(currRot, oldRot, scaledInertia));
    (*outEnergy) += static_cast<double>(
        2_r * lie::WeightedRotationDifferenceMerit(currRot, stageStartRot, scaledInertia));
    // GradTarget::Current could use the more optimized expression below, but it does not yield zero
    // minimum merit, and it requires a different merit per target.
    // (*outEnergy) += static_cast<double>(
    //     lie::WeightedRotationDifferenceMerit(currRot, explicitRot, scaledInertia));
  }

  // Accumulate gradient if requested
  if (outGradient) {
    // Translation gradient contribution, sign depends on target type
    auto outGradientCom = Load<Vec4r>(outGradient->data() + 0);
    real constexpr kSign = kGradTarget == GradTarget::Current ? 1_r : -1_r;
    outGradientCom += (kSign * mass / dtStage) * deltaVCom;
    Store(outGradient->data() + 0, outGradientCom);

    // Rotation gradient contribution from derivative of trace with respect to rotation
    Vec4r gradientRot = lie::WeightedRotationDifferenceGradient(energyRotMatrix);
    if constexpr (kGradTarget != GradTarget::Current) {
      gradientRot = DotVecMat3x3(gradientRot, chain);
      if constexpr (kGradTarget == GradTarget::Previous) {
        gradientRot += lie::WeightedRotationDifferenceGradient(energyRotMatrixAux);
      }
    }
    Vec4r outGradientRot = Load<3, Vec4r>(outGradient->data() + 3);
    outGradientRot += gradientRot;
    Store<3>(outGradient->data() + 3, outGradientRot);
  }

  // Accumulate Hessian if requested
  if (outHessian) {
    // Translation Hessian block update, sign depends on target type
    VMatrix3x3r outHessianCom;
    LoadSubmatrix<3, 3, 6, 6>(outHessianCom, Int2{0, 0}, *outHessian);
    real constexpr kSign = kGradTarget == GradTarget::Current ? 1_r : -1_r;
    outHessianCom += VDiagonalMatrix<3>(kSign * dtStage2i * mass);
    StoreSubmatrix<3, 3, 6, 6>(*outHessian, Int2{0, 0}, outHessianCom);

    // Rotation Hessian block update from second derivative of trace term
    VMatrix3x3r outHessianRot;
    LoadSubmatrix<3, 3, 6, 6>(outHessianRot, Int2{3, 3}, *outHessian);
    if constexpr (kGradTarget == GradTarget::Current) {
      outHessianRot += lie::WeightedRotationDifferenceHessian(energyRotMatrix);
    } else {
      outHessianRot += Dot3x3(lie::WeightedRotationDifferenceHessianMixed(energyRotMatrix), chain);
      if constexpr (kGradTarget == GradTarget::Previous) {
        outHessianRot += lie::WeightedRotationDifferenceHessianMixed(energyRotMatrixAux);
      }
    }
    StoreSubmatrix<3, 3, 6, 6>(*outHessian, Int2{3, 3}, outHessianRot);
  }
}

/**
 * @brief Adds rigid-body inertia terms for Newton-Euler residual of the form M dw/dt + w x M w.
 *
 * This function computes and accumulates energy, gradient, and Hessian contributions related to the
 * rigid-body inertia of a system given its state and inertia properties. The Newton-Euler gradient
 * does not derive from a merit. The function computes an approximate merit and an approximate
 * Hessian. If a merit-based inertia is needed, use AddRigidBodyInertiaFromMerit().
 *
 * @tparam InertiaT The type for the first moment of inertia: real (isotropic inertia), Vec4r
 * (anisotropic unrotated inertia), VMatrix3x3r (arbitrary first moment of inertia).
 *
 * @param mass The mass of the rigid body.
 * @param inertia The first moment of inertia of the rigid body.
 * @param dtStage The time step duration of the current stage.
 * @param state The current transform state of the rigid body (position and rotation).
 * @param stageStartPos The transform state at the start of the current stage.
 * @param stageStartVel The rigid body velocity at the start of the current stage.
 * @param[out] outEnergy Pointer to a double to accumulate energy contributions. Can be nullptr if
 *not needed.
 * @param[out] outGradient Pointer to a RigidGradient to accumulate gradient contributions. Can be
 *nullptr if not needed.
 * @param[out] outHessian Pointer to a RigidHessian to accumulate Hessian contributions. Can be
 *nullptr if not needed.
 * @param[out] outNonPotential Pointer to also accumulate the rotational energy and gradient, as
 * these terms have no potential. Can be nullptr if not needed.
 */
template <typename InertiaT>
inline void AddRigidBodyInertiaNewtonEuler(
    real mass,
    InertiaT const& inertia,
    real dtStage,
    TransformRT const& state,
    TransformRT const& stageStartPos,
    RigidBodyVel const& stageStartVel,
    double* outEnergy,
    RigidGradient* outGradient,
    RigidHessian* outHessian,
    RigidNonPotentialTerms* outNonPotential = nullptr) {
  MOCHI_ASSERT_VERBOSE(
      !outNonPotential || outEnergy, "Non-potential terms must be reported with the energy.");
  if (!outEnergy && !outGradient && !outHessian) {
    return;
  }

  // Compute common values
  RigidBodyVel vel;
  vel.SetFromFiniteDifferencePose(stageStartPos, state, dtStage);
  Vec4r deltaVCom = vel.GetVCom() - stageStartVel.GetVCom();
  Vec4r omega = vel.GetOmegaAndVSym().first;
  Vec4r deltaOmega = omega - stageStartVel.GetOmegaAndVSym().first;
  VMatrix3x3r M = RotateInertia(inertia, state.GetRotation());

  // The rotation merit assumes:
  // 1. dM/dR = 0. This is correct if inertia is constant.
  // 2. dw/dR = 1/h eye. This is correct when currentRot = stageStartRot.
  double const rotEnergy = outEnergy
      ? static_cast<double>(0.5_r * Dot<3>(deltaOmega, DotVecMat3x3(deltaOmega, M)))
      : 0.0;
  Vec4r const rotGradient = (outGradient || outNonPotential)
      ? DotVecMat3x3(deltaOmega / dtStage, M) + Cross3(omega, DotVecMat3x3(omega, M))
      : Vec4r{};

  if (outEnergy) {
    (*outEnergy) += static_cast<double>(NormSqr<3>(deltaVCom) * mass * 0.5_r);
    (*outEnergy) += rotEnergy;
  }

  if (outGradient) {
    auto outGradientCom = Load<Vec4r>(outGradient->data() + 0);
    outGradientCom += (mass / dtStage) * deltaVCom;
    Store(outGradient->data() + 0, outGradientCom);

    Vec4r outGradientRot = Load<3, Vec4r>(outGradient->data() + 3);
    outGradientRot += rotGradient;
    Store<3>(outGradient->data() + 3, outGradientRot);
  }
  if (outNonPotential) {
    outNonPotential->energy += rotEnergy;
    outNonPotential->rotGradient += ToReal3(rotGradient);
  }

  if (outHessian) {
    real const dtStage2i = 1_r / (dtStage * dtStage);

    VMatrix3x3r outHessianCom;
    LoadSubmatrix<3, 3, 6, 6>(outHessianCom, Int2{0, 0}, *outHessian);
    outHessianCom += VDiagonalMatrix<3>(mass * dtStage2i);
    StoreSubmatrix<3, 3, 6, 6>(*outHessian, Int2{0, 0}, outHessianCom);

    // The rotation hessian is accurate only if the assumptions of the merit hold.
    VMatrix3x3r outHessianRot;
    LoadSubmatrix<3, 3, 6, 6>(outHessianRot, Int2{3, 3}, *outHessian);
    outHessianRot += M * dtStage2i;
    StoreSubmatrix<3, 3, 6, 6>(*outHessian, Int2{3, 3}, outHessianRot);
  }
}

// Helper function to dispatch the appropriate inertia assembly function
template <typename InertiaT>
using RigidInertiaFn = void (*)(
    real,
    InertiaT const&,
    real,
    TransformRT const&,
    TransformRT const&,
    RigidBodyVel const&,
    double*,
    RigidGradient*,
    RigidHessian*,
    RigidNonPotentialTerms*);
template <typename InertiaT>
inline RigidInertiaFn<InertiaT> GetRigidInertiaFn(GradTarget gradTarget, bool useNewtonEuler) {
  static_assert(
      static_cast<int>(GradTarget::Count) == 5,
      "Please update the switch statement below if GradTarget enum changes");

  if (useNewtonEuler) {
    MOCHI_ASSERT_VERBOSE(
        gradTarget == GradTarget::Current,
        "Newton-Euler inertia is not supported with differentiability");
    return AddRigidBodyInertiaNewtonEuler<InertiaT>;
  } else {
    // clang-format off
    switch (gradTarget) {
      case GradTarget::Current:
        return AddRigidBodyInertiaFromMerit<GradTarget::Current, InertiaT>;
      case GradTarget::Previous:
        return AddRigidBodyInertiaFromMerit<GradTarget::Previous, InertiaT>;
      case GradTarget::PreviousDelta:
        return AddRigidBodyInertiaFromMerit<GradTarget::PreviousDelta, InertiaT>;
      MOCHI_UNLIKELY default: MOCHI_ASSERT_VERBOSE(false, "Unexpected grad target");
        return nullptr;
    }
    // clang-format on
  }
}

// Function to add rigid-body damping terms.
template <GradTarget kGradTarget>
inline void AddRigidBodyDamping(
    TransformRT const& state,
    real damping,
    TransformRT const& stageStartPos,
    real dtStage,
    double* outEnergy,
    RigidGradient* outGradient,
    RigidHessian* outHessian) {
  static_assert(
      kGradTarget == GradTarget::Current || kGradTarget == GradTarget::Previous,
      "Unsupported target");
  if (!outEnergy && !outGradient && !outHessian) {
    return;
  }

  real dampingFactor = damping / dtStage;

  // Compute common values
  auto rot = ToVMatrix3x3(state.GetRotation());
  auto stageStartRot = ToVMatrix3x3(stageStartPos.GetRotation());
  VMatrix3x3r energyRotMatrix MOCHI_NO_INIT;
  Vec4r posDiff MOCHI_NO_INIT;
  if constexpr (kGradTarget == GradTarget::Current) {
    energyRotMatrix =
        lie::WeightedRotationDifferenceMatrix(rot, stageStartRot, 0.5_r * dampingFactor);
    posDiff = state.VGetTranslation() - stageStartPos.VGetTranslation();
  } else {
    energyRotMatrix =
        lie::WeightedRotationDifferenceMatrix(stageStartRot, rot, 0.5_r * dampingFactor);
    posDiff = stageStartPos.VGetTranslation() - state.VGetTranslation();
  }

  if (outEnergy) {
    // Translation component
    (*outEnergy) += static_cast<double>(NormSqr<3>(posDiff) * 0.5_r * dampingFactor);

    // Rotation component
    (*outEnergy) += static_cast<double>(
        lie::WeightedRotationDifferenceMerit(rot, stageStartRot, 0.5_r * dampingFactor));
  }

  if (outGradient) {
    // Load output
    auto outGradientCom = Load<Vec4r>(outGradient->data() + 0);
    Vec4r outGradientRot = Load<3, Vec4r>(outGradient->data() + 3);

    // Translation component
    outGradientCom += posDiff * dampingFactor;

    // Rotation component
    outGradientRot += lie::WeightedRotationDifferenceGradient(energyRotMatrix);

    // Store output
    Store(outGradient->data() + 0, outGradientCom);
    Store<3>(outGradient->data() + 3, outGradientRot);
  }

  if (outHessian) {
    // Load output
    VMatrix3x3r outHessianCom;
    VMatrix3x3r outHessianRot;
    LoadSubmatrix<3, 3, 6, 6>(outHessianCom, Int2{0, 0}, *outHessian);
    LoadSubmatrix<3, 3, 6, 6>(outHessianRot, Int2{3, 3}, *outHessian);

    // Translation component
    outHessianCom += VDiagonalMatrix<3>(dampingFactor);

    // Rotation component
    outHessianRot += lie::WeightedRotationDifferenceHessian(energyRotMatrix);

    // Store output
    StoreSubmatrix<3, 3, 6, 6>(*outHessian, Int2{0, 0}, outHessianCom);
    StoreSubmatrix<3, 3, 6, 6>(*outHessian, Int2{3, 3}, outHessianRot);
  }
}

/**
 * @brief Adds rigid-body friction terms separately for translational and rotational motion.
 *
 * This function computes and accumulates energy, gradient, and Hessian contributions related to
 * friction forces and torques acting on a rigid body. Static friction is modeled using the merit
 * function approximation of Coulomb friction from the Incremental Potential Contact paper. Dynamic
 * friction is attenuated with increasing velocity according to a Gaussian Stribeck model. The
 * computation depends on the gradient target type specified by the template parameter.
 *
 * @tparam kGradTarget The gradient target type. Determines whether derivatives are with respect to
 * the current state (Current) or the previous state (Previous). Possible values:
 * GradTarget::Previous, GradTarget::Current.
 *
 * @param useFittedHessian If true, approximates the Hessian along the relative velocity direction
 * by fitting a global quadratic function. This is more robust than the local Hessian, which cancels
 * out when friction saturates.
 * @param psdDRes If true, clamps negative Hessian contributions.
 * @param frictionParams Per-joint friction parameters including dynamic coefficient, falloff
 * velocity, extra stiction, and Stribeck velocity.
 * @param dtStage The time step duration of the current stage.
 * @param state The current transform state of the rigid body (position and rotation).
 * @param stageStartPos The transform state at the start of the current stage.
 * @param[out] outEnergy Pointer to a double to accumulate energy contributions. Can be nullptr if
 * not needed.
 * @param[out] outGradient Pointer to a RigidGradient to accumulate gradient contributions. Can be
 * nullptr if not needed.
 * @param[out] outHessian Pointer to a RigidHessian to accumulate Hessian contributions. Can be
 * nullptr if not needed.
 */
template <GradTarget kGradTarget>
inline void AddRigidBodyFriction(
    bool useFittedHessian,
    bool psdDRes,
    ArticulatedJointFrictionParams const& frictionParams,
    real dtStage,
    TransformRT const& state,
    TransformRT const& stageStartPos,
    double* outEnergy,
    RigidGradient* outGradient,
    RigidHessian* outHessian) {
  static_assert(
      kGradTarget == GradTarget::Current || kGradTarget == GradTarget::Previous,
      "Unsupported target");
  if (!outEnergy && !outGradient && !outHessian) {
    return;
  }

  real const coulombFriction = frictionParams.coulomb;
  real const falloffVel = frictionParams.falloffVel;
  real const stictionExtra = frictionParams.stictionExtra;
  real const stribeckVel = frictionParams.stribeckVel;

  real const staticFriction = coulombFriction + stictionExtra;
  MOCHI_ASSERT_VERBOSE(staticFriction > 0_r, "Static friction must be strictly positive.");
  real const dSmootherInfty = coulombFriction / staticFriction;

  // Lambda function for each slice (translation or rotation) of the velocity
  auto processSlice = [&](Vec4r const& vel,
                          double* outEnergySlice,
                          Vec4r* outGradientSlice,
                          VMatrix3x3r* outHessianSlice) {
    // Compute velocity norm and unit vector
    real velNorm = Norm<3>(vel);
    Vec4r velU = vel / velNorm;

    // Compute friction smoothing function f(velNorm) -> [0,1]
    real smoother{};
    real dSmoother{};
    real ddSmoother{};
    real dSmoother_velNorm;
    StribeckActivation(
        velNorm,
        falloffVel,
        dSmootherInfty,
        stribeckVel,
        smoother,
        dSmoother,
        ddSmoother,
        dSmoother_velNorm);

    if (outEnergySlice) {
      (*outEnergySlice) = static_cast<double>(staticFriction * dtStage * smoother);
    }

    if (outGradientSlice) {
      // grad = staticFriction * dSmoother * velU
      (*outGradientSlice) = (staticFriction * dSmoother_velNorm) * vel;
    }

    if (outHessianSlice) {
      real frictionFactor = staticFriction / dtStage;

      // hessian =
      //  a) staticFriction / dtStage * ddSmoother in the direction of velU
      //  b) staticFriction / dtStage * dSmoother / velNorm in the other directions
      // But if useFittedHessian = true, we use 'a' in all directions.
      (*outHessianSlice) = VDiagonalMatrix<3>(frictionFactor * dSmoother_velNorm);
      if (velNorm > 1e-9_r && !useFittedHessian) {
        real const ddSmootherClamped = psdDRes ? Max(ddSmoother, 0_r) : ddSmoother;
        (*outHessianSlice) +=
            Outer3((frictionFactor * (ddSmootherClamped - dSmoother_velNorm)) * velU, velU);
      }
    }
  };

  // Create containers for slice evaluations
  double energyTranslation{};
  double energyRotation{};
  Vec4r gradTranslation MOCHI_NO_INIT;
  Vec4r gradRotation MOCHI_NO_INIT;
  VMatrix3x3r hessianTranslation MOCHI_NO_INIT;
  VMatrix3x3r hessianRotation MOCHI_NO_INIT;

  // Compute velocities. For translation: finite-difference position. For rotation:
  // finite-difference the relative rotation vector.
  Vec4r velTranslation =
      (1_r / dtStage) * (state.VGetTranslation() - stageStartPos.VGetTranslation());
  VMatrix3x3r rotDelta =
      ToVMatrix3x3(state.GetRotation() * stageStartPos.GetRotation().GetConjugate());
  Vec4r velRotation = (1_r / dtStage) * InvRodrigues(rotDelta);

  // Process translation and rotation slices
  processSlice(
      velTranslation,
      outEnergy ? &energyTranslation : nullptr,
      outGradient ? &gradTranslation : nullptr,
      outHessian ? &hessianTranslation : nullptr);
  processSlice(
      velRotation,
      outEnergy ? &energyRotation : nullptr,
      outGradient ? &gradRotation : nullptr,
      outHessian ? &hessianRotation : nullptr);

  // Handle chain-rules.
  // 1. Log-map for the rotation. We ignore the 2nd derivative of the log-map for the Hessian.
  // 2. GradTarget::Previous: Sign-flip for the translation.
  // 3. GradTarget::Previous: dRotDeltadRotOld for the rotation.
  // In GradTarget::Current, dRotDeltadRot = eye, so nothing to do.
  if (outGradient || outHessian) {
    VMatrix3x3r chainRotation = DRotVectorDRotIncrement(dtStage * velRotation);
    if constexpr (kGradTarget == GradTarget::Previous) {
      if (outGradient) {
        gradTranslation = -gradTranslation;
      }
      chainRotation = Dot3x3(chainRotation, lie::DMultRotaRotTRotbDRot(rotDelta));
    }
    if (outGradient) {
      gradRotation = DotVecMat3x3(gradRotation, chainRotation);
    }
    if (outHessian) {
      hessianRotation = Dot3x3(Transpose3x3(chainRotation), Dot3x3(hessianRotation, chainRotation));
    }
  }

  // Store results
  if (outEnergy) {
    (*outEnergy) += energyTranslation + energyRotation;
  }

  if (outGradient) {
    // Load output
    auto outGradientCom = Load<Vec4r>(outGradient->data() + 0);
    Vec4r outGradientRot = Load<RigidSize::kDRot, Vec4r>(outGradient->data() + RigidSize::kDTrans);

    outGradientCom += gradTranslation;
    outGradientRot += gradRotation;

    // Store output
    Store(outGradient->data() + 0, outGradientCom);
    Store<3>(outGradient->data() + RigidSize::kDTrans, outGradientRot);
  }

  if (outHessian) {
    // Load output
    VMatrix3x3r outHessianCom;
    VMatrix3x3r outHessianRot;
    LoadSubmatrix<3, 3, 6, 6>(outHessianCom, Int2{0, 0}, *outHessian);
    LoadSubmatrix<3, 3, 6, 6>(outHessianRot, Int2{3, 3}, *outHessian);

    outHessianCom += hessianTranslation;
    outHessianRot += hessianRotation;

    // Store output
    StoreSubmatrix<3, 3, 6, 6>(*outHessian, Int2{0, 0}, outHessianCom);
    StoreSubmatrix<3, 3, 6, 6>(*outHessian, Int2{3, 3}, outHessianRot);
  }
}

// Function to add rigid-body external force terms. The energy is relative to the stage start. The
// torque terms, which have no potential, are also added to outNonPotential, if not null.
inline void AddRigidBodyExternalForces(
    TransformRT const& state,
    TransformRT const& stageStartPos,
    Span<int const> dofs,
    Span<real const> forces,
    double* outEnergy,
    RigidGradient* outGradient,
    RigidNonPotentialTerms* outNonPotential = nullptr) {
  MOCHI_ASSERT_VERBOSE(
      !outNonPotential || outEnergy, "Non-potential terms must be reported with the energy.");
  if (!outEnergy && !outGradient) {
    return;
  }

#if MOCHI_ASSERT_VERBOSE_ENABLED
  MOCHI_ASSERT_VERBOSE(dofs.size() == forces.size(), "Invalid number of external forces.");
  MOCHI_ASSERT_VERBOSE(dofs.size() <= RigidSize::kDAll, "Invalid number of external forces.");
  if (!dofs.empty()) {
    auto [min, max] = MinMax(dofs);
    MOCHI_ASSERT_VERBOSE(min >= 0 && max < RigidSize::kDAll, "Invalid external force DoF index.");
  }
#endif // MOCHI_ASSERT_VERBOSE_ENABLED

  // Compute common values
  Real3 force3 = {};
  Real3 torque3 = {};
  for (int i = 0; i < dofs.size(); i++) {
    int dof = dofs[i];
    if (dof < RigidSize::kDTrans) {
      force3[dof] = forces[i];
    } else if (dof < RigidSize::kDAll) {
      torque3[dof - RigidSize::kDTrans] = forces[i];
    }
  }
  Vec4r force = ToSimd(force3);
  Vec4r torque = ToSimd(torque3);

  // 0.5 tr(skew(torque) R(step)) = -2 w (torque·v) for the rotation step (v, w) from the stage
  // start. Its gradient at the stage start is -torque.
  double torqueEnergy = 0.0;
  if (outEnergy) {
    Quaternion const step = state.GetRotation() * stageStartPos.GetRotation().GetConjugate();
    real const w = Get<3>(step.data); // Scalar part
    Real3 const v = ToReal3(step.data); // Vector part
    torqueEnergy = -2.0 * w * Dot(torque3, v);
    (*outEnergy) -= Dot(state.VGetTranslation() - stageStartPos.VGetTranslation(), force);
    (*outEnergy) += torqueEnergy;
  }

  if (outGradient) {
    // Load output
    auto outGradientCom = Load<Vec4r>(outGradient->data() + 0);
    Vec4r outGradientRot = Load<RigidSize::kDRot, Vec4r>(outGradient->data() + RigidSize::kDTrans);

    outGradientCom -= force;
    outGradientRot -= torque;

    // Store output
    Store(outGradient->data() + 0, outGradientCom);
    Store<3>(outGradient->data() + RigidSize::kDTrans, outGradientRot);
  }
  if (outNonPotential) {
    outNonPotential->energy += torqueEnergy;
    outNonPotential->rotGradient -= torque3;
  }
}

} // namespace mochi
