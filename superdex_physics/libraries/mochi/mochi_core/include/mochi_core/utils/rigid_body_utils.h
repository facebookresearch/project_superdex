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

#include <mochi_core/elements/tetrahedral/finite_element.h>
#include <mochi_core/elements/triangular/finite_element.h>
#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/decomposition_utils.h>
#include <mochi_core/utils/lie.h>
#include <mochi_core/utils/quaternion_utils.h>
#include <mochi_core/utils/rigid_body_size.h>
#include <mochi_core/utils/rodrigues_utils.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/vmatrix.h>

#include <limits>

namespace mochi {

inline constexpr real kMomentOfInertiaValidationRelTol =
    64_r * std::numeric_limits<real>::epsilon();

// Convert between conventional moment of inertia to second moment (aka MomentOfInertia above)
// Moment of Inertia:
//     [ixx, ixy, ixz]   [ +∫(y^2 + z^2)dm, -∫(xy)dm,        -∫(xz)dm        ]
// I = [ixy, iyy, iyz] = [ -∫(xy)dm,        +∫(x^2 + z^2)dm, -∫(yz)dm        ]
//     [ixz, iyz, izz]   [ -∫(xz)dm,        -∫(yz)dm,        +∫(x^2 + y^2)dm ]
// Second Moment:
//     [mxx, mxy, mxz]   [ +∫(xx)dm, +∫(xy)dm, +∫(xz)dm ]
// M = [mxy, myy, myz] = [ +∫(xy)dm, +∫(yy)dm, +∫(yz)dm ]
//     [mxz, myz, mzz]   [ +∫(xz)dm, +∫(yz)dm, +∫(zz)dm ]
inline VMatrix3x3r MomentOfInertiaToSecondMoment(VMatrix3x3r const& I) {
  return VDiagonalMatrix<3>(0.5_r * Trace3x3(I)) - I;
}

inline VMatrix3x3r SecondMomentToMomentOfInertia(VMatrix3x3r const& M) {
  return VDiagonalMatrix<3>(Trace3x3(M)) - M;
}

/// @brief Checks whether a moment-of-inertia tensor is physically realizable.
///
/// @details Valid iff finite and its principal moments are non-negative and satisfy the triangle
/// inequalities (each principal moment is at most the sum of the other two).
///
/// @param[in] momentOfInertia Symmetric tensor upper triangle: xx, xy, xz, yy, yz, zz.
/// @param[in] relTol Relative tolerance applied to the non-negativity and triangle-inequality
/// checks, scaled by the largest absolute principal moment (or 1 if smaller). Defaults to
/// @ref kMomentOfInertiaValidationRelTol. Must be non-negative and finite.
///
/// @return True if the tensor is physically realizable within @p relTol, false otherwise.
[[nodiscard]] inline bool IsMomentOfInertiaValid(
    Real6 const& momentOfInertia,
    real relTol = kMomentOfInertiaValidationRelTol) {
  MOCHI_ASSERT_VERBOSE(
      IsFinite(relTol) && relTol >= 0_r, "relTol must be non-negative and finite.");
  if (!IsFinite(momentOfInertia)) {
    return false;
  }

  // Real6 stores the symmetric tensor's upper triangle: xx, xy, xz, yy, yz, zz.
  VSymMatrix3x3r const moiSym = {
      Vec4r{momentOfInertia[0], momentOfInertia[3], momentOfInertia[5]},
      Vec4r{momentOfInertia[1], momentOfInertia[2], momentOfInertia[4]}};
  Vec4r principalMoments;
  AnalyticalEigendecompSym3x3(moiSym, principalMoments);
  if (!IsFinite(principalMoments)) {
    return false;
  }

  real const absTol = relTol * Max(1_r, HMax<3>(Abs(principalMoments)));
  real const ix = principalMoments[0];
  real const iy = principalMoments[1];
  real const iz = principalMoments[2];

  // Physical MOI principal moments must be non-negative and satisfy the triangle inequalities.
  return HMin<3>(principalMoments) >= -absTol && ix <= iy + iz + absTol && iy <= ix + iz + absTol &&
      iz <= ix + iy + absTol;
}

// Rotate inertia from rest configuration to rotated configuration. Valid for both moment of inertia
// and second moment.
inline VMatrix3x3r RotateInertia(VMatrix3x3r const& M, Quaternion const& q) {
  auto R_RT = ToVMatrix3x3_WithTranspose(q);
  return Dot3x3(R_RT.first, Dot3x3(M, R_RT.second));
}

inline VMatrix3x3r RotateInertia(Vec4r const& M, Quaternion const& q) {
  auto R_RT = ToVMatrix3x3_WithTranspose(q);
  return Dot3x3(R_RT.first * M, R_RT.second);
}

inline VMatrix3x3r RotateInertia(real M, Quaternion const& /*q*/) {
  return VDiagonalMatrix<3>(M);
}

/**
 * @brief Inertia properties of a dynamic rigid body.
 *
 * @note Stores a reference state (_refMoi, _refDensity) anchored to the latest call to
 * @ref SetInertiaProperties. @ref SetDensity rescales mass and moment of inertia
 * proportionally about that reference; center of mass is never changed by @ref SetDensity.
 * @note After @ref SetDensity, calling SetDensity again with the current density reproduces all
 * properties bit-exactly (no floating-point roundtrip loss). After @ref SetInertiaProperties,
 * the reference state is changed and density is recomputed as mass / volume.
 * After @ref SetInertiaProperties, calling @ref SetInertiaProperties again with the same input
 * also reproduces all properties bit-exactly.
 */
class RigidBodyInertia {
 public:
  RigidBodyInertia() = default;

  RigidBodyInertia(Vec4r const& comLocal, real volume, VMatrix3x3r const& moi, real density)
      : _volume(volume), _refMoi(moi), _refDensity(density), _comLocal(comLocal) {
    MOCHI_ASSERT_VERBOSE(_volume > 0_r && IsFinite(_volume), "Volume must be positive and finite.");
    SetDensity(density);
  }

  [[nodiscard]] Vec4r const& GetCenterOfMassLocal() const {
    return _comLocal;
  }
  [[nodiscard]] real GetMass() const {
    return _mass;
  }
  [[nodiscard]] real GetDensity() const {
    return _density;
  }
  [[nodiscard]] VMatrix3x3r const& GetMomentOfInertiaLocal() const {
    return _moi;
  }
  [[nodiscard]] VMatrix3x3r const& GetSecondMomentLocal() const {
    return _mtwo;
  }

  /// @brief Update the density and rescale mass and moment of inertia about the current reference.
  /// Center of mass is not changed.
  void SetDensity(real density) {
    MOCHI_ASSERT_VERBOSE(
        _volume > 0_r, "SetDensity called on a default-constructed RigidBodyInertia.");
    MOCHI_ASSERT_VERBOSE(
        density > 0_r && IsFinite(density), "Density must be positive and finite.");
    _density = density;
    _mass = _volume * density;
    _moi = _refMoi * (density / _refDensity);
    _mtwo = MomentOfInertiaToSecondMoment(_moi);
  }

  /// @brief Directly set mass, center of mass, and moment of inertia. Re-anchors the density
  /// reference state so subsequent @ref SetDensity calls rescale mass and MoI proportionally about
  /// this new state.
  void SetInertiaProperties(real mass, Vec4r const& comLocal, VMatrix3x3r const& moi) {
    MOCHI_ASSERT_VERBOSE(
        _volume > 0_r, "SetInertiaProperties called on a default-constructed RigidBodyInertia.");
    MOCHI_ASSERT_VERBOSE(mass > 0_r && IsFinite(mass), "Mass must be positive and finite.");
    MOCHI_ASSERT_VERBOSE(IsFinite(moi), "MoI must be finite.");
    _comLocal = comLocal;
    _mass = mass;
    _density = _refDensity = mass / _volume;
    _refMoi = _moi = moi;
    _mtwo = MomentOfInertiaToSecondMoment(moi);
  }

 private:
  // Geometric reference quantity. Set once at construction, never mutated thereafter.
  real _volume = 0_r; ///< Volume [m^3].

  // Reference state for density-based rescaling.
  VMatrix3x3r _refMoi = {}; ///< Moment of inertia [kg m^2] at the reference density.
  real _refDensity = 0_r; ///< Reference density [kg/m^3].

  // Current state. Mutated by SetDensity and SetInertiaProperties.
  Vec4r _comLocal = {}; ///< Center of mass [m] in local coordinates.
  real _density = 0_r;
  real _mass = 0_r;
  VMatrix3x3r _mtwo = {}; ///< Second moment of inertia [kg m^2] in local coordinates.
  VMatrix3x3r _moi = {}; ///< Standard moment of inertia [kg m^2] in local coordinates.
};

// Class that stores rigid body velocity.
// - Translation velocity is stored as COM time-derivative (vcom).
// - Rotation velocity is stored as angular velocity (omega) plus the symmetric component of the
// rotation-matrix time derivative (vsym).
//
// Our merit-based inertial term (defined in the Rigid IPC paper) uses a finite-difference
// estimation of the rotation-matrix derivative, dRdt_approx = 1/dt (R - Rold). This is not the true
// rotation-matrix derivative, dRdt = skew(omega) R, as it includes a fictitious symmetric
// component. The full dRdt_approx can be reconstructed by storing the variable 'vsym' as
// follows.
// 1. Given rotation matrices of two consecutive steps, estimate dRdt_approx = 1/dt (R - Rold).
// 2. Store omega = inv_skew(dRdt_approx * R^T).
// 3. Store vsym = inv_sym(dRdt_approx * R^T).
// 4. Perform time integration on both omega and vsym.
// 5. Reconstruct dRdt_approx = (skew(omega) + sym(vsym)) * R.
//
// To integrate velocity and evaluate a new rotation, use the function EvalTimeSteppedRotation(),
// which uses both omega and vsym. Do not use omega alone.
//
// Velocity should be updated by calling the function SetFromFiniteDifferencePose(), which
// internally updates both omega and vsym.
//
// If omega is set externally, then vsym must be updated by calling the function
// UpdateVSymIfDirty() as soon as the time-step size is known. This ensures that omega and vsym
// together exactly reproduce a rotation when calling EvalTimeSteppedRotation().
class RigidBodyVel {
 public:
  static constexpr int kRawSize = 4 /* _vcom */ + 4 /* _omega */ + 8 /* _vsym */;

  Vec4r GetVCom() const {
    return _vcom;
  }

  std::pair<Vec4r const&, VSymMatrix3x3r const&> GetOmegaAndVSym() const {
    return {_omega, _vsym};
  }

  void SetVCom(Vec4r vcom) {
    _vcom = vcom;
  }

  void SetOmega(Vec4r omega) {
    _omega = omega;
    _isVSymDirty = true;
  }

  // Warning: omega and vsym can only reproduce a rotation if |omega| < 1/h. This should be checked
  // by the caller.
  void UpdateVSymIfDirty(real h) {
    if (!_isVSymDirty) {
      return; // Nothing to do
    }
    _isVSymDirty = false;

    // The symmetric part is set such that EvalTimeSteppedRotation(R, h), with R a rotation matrix,
    // produces another rotation matrix Q.
    // QT Q = eye, with Q = R + h * dR/dt and dR/dt = (sk(w) + sym) * R.
    // Then:
    // (eye + h sym + h sk(w))T (eye + h sym + h sk(w)) = eye
    // (eye + h sym - h sk(w)) (eye + h sym + h sk(w)) = eye
    // (eye + h sym)^2 - (h sk(w))^2 = eye
    // h^2 sym^2 + 2h sym - h^2 sk^2(w) = 0
    // h sym^2 + 2 sym - h sk^2(w) = 0
    // Apply SVD to sk^2(w) = U S UT, and sym = U X UT:
    // h X^2 + 2 X - h S = 0
    // The singular values are s1 = 0, and s2 = s3 = - |w|^2
    // The singular vectors are w/|w| and any two orthonormal vectors
    // For each entry:
    // x = (sqrt(h^2 s + 1) - 1) / h
    // x1 = 0; x2 = x3 = (sqrt(1 - h^2 |w|^2) - 1) / h
    // WARNING: We clamp the discriminant to zero, but valid omega should be checked by the caller.
    // Omega is valid if |w| < 1/h
    real normSqrOmega = NormSqr<3>(_omega);
    real disc = Max(0_r, 1_r - h * h * normSqrOmega);
    real x = (Sqrt(disc) - 1_r) / h;
    VMatrix3x3r X = VDiagonalMatrix<3>(Vec4r{0_r, x, x});
    Vec4r u1 =
        NearEqual<3>(_omega, Vec4r{0_r, 0_r, 0_r}) ? Vec4r{1_r, 0_r, 0_r} : Normalize<3>(_omega);
    Vec4r u2 = Abs(Get<1>(u1)) > 0.9_r ? Vec4r{0_r, 0_r, 1_r} : Vec4r{0_r, 1_r, 0_r};
    u2 = Normalize(u2 - Dot<3>(u1, u2) * u1);
    Vec4r u3 = Cross3(u1, u2);
    VMatrix3x3r UT{u1, u2, u3};
    VMatrix3x3r sym = Dot3x3(Transpose3x3(UT), Dot3x3(X, UT));
    _vsym = SimdFullToSym(sym);
  }

  void SetZero() {
    _vcom = {};
    _omega = {};
    _vsym = {};
    _isVSymDirty = false;
  }

  bool IsVSymDirty() const {
    return _isVSymDirty;
  }

  void FromRawValues(ColumnVectorView<real const, kRawSize> raw) {
    _vcom = Load<Vec4r>(raw.data());
    _omega = Load<Vec4r>(raw.data() + 4);
    _vsym[0] = Load<Vec4r>(raw.data() + 8);
    _vsym[1] = Load<Vec4r>(raw.data() + 12);
    _isVSymDirty = false;
  }

  void ToRawValues(ColumnVectorView<real, kRawSize> outRaw) const {
    MOCHI_ASSERT_VERBOSE(!_isVSymDirty, "vsym needs to be updated");
    Store(outRaw.data(), _vcom);
    Store(outRaw.data() + 4, _omega);
    Store(outRaw.data() + 8, _vsym[0]);
    Store(outRaw.data() + 12, _vsym[1]);
  }

  // Evaluate rigid velocity by finite differencing two poses
  void SetFromFiniteDifferencePose(
      TransformRT const& oldPose,
      TransformRT const& newPose,
      real dtStage) {
    // Finite-difference the translations
    _vcom = (newPose.VGetTranslation() - oldPose.VGetTranslation()) / dtStage;

    // Finite-difference the rotation matrices, and then convert to omega and vsym
    auto R_RT = ToVMatrix3x3_WithTranspose(newPose.GetRotation());
    auto dRdt = (R_RT.first - ToVMatrix3x3(oldPose.GetRotation())) / dtStage;
    auto full = Dot3x3(dRdt, R_RT.second);
    _omega = InvSkew3(full);
    _vsym = SimdFullToSym(0.5_r * (full + Transpose3x3(full)));
    _isVSymDirty = false;
  }

  // Evaluate rotation matrix by time-stepping the rotation velocity. The result is not really a
  // rotation if dtStage is the same value used for estimating vsym.
  VMatrix3x3r EvalTimeSteppedRotation(VMatrix3x3r const& R, real dtStage) const {
    MOCHI_ASSERT_VERBOSE(!_isVSymDirty, "vsym needs to be updated");
    // Convert the rotation velocity to rotation-matrix derivative, and then integrate
    auto dRdt = Dot3x3(Skew3(_omega) + SimdSymToFull(_vsym), R);
    return R + dtStage * dRdt;
  }

  MOCHI_STRUCT_BEGIN(mochi::RigidBodyVel)
  MOCHI_FIELD_NAME(_vcom, "vcom")
  MOCHI_FIELD_NAME(_omega, "omega")
  MOCHI_FIELD_NAME(_vsym, "vsym")
  MOCHI_FIELD_NAME(_isVSymDirty, "isVSymDirty")
  MOCHI_STRUCT_END()

 protected:
  // Center of mass velocity in world coordinates
  Vec4r _vcom = SimdZero();

  // Angular velocity in world coordinates
  Vec4r _omega = SimdZero();

  // Symmetric part of the rotation matrix derivative
  VSymMatrix3x3r _vsym = {};

  // Flag to indicate that _omega was set externally and _vsym has not been updated yet
  bool _isVSymDirty = false;
};

// Computes the center-of-mass of a rigid body based on a FEM discretization with uniform density
template <int kPolyOrder, int kQuadDegree>
inline void ComputeCenterOfMassFem(
    Span<tetrahedral::Pk3DElement<kPolyOrder, kQuadDegree> const> finiteElements,
    real density,
    Vec4r& outCenterOfMass,
    real& outMass) {
  using ElementT = tetrahedral::Pk3DElement<kPolyOrder, kQuadDegree>;
  constexpr int kNumQuad = ElementT::kNumQuadPoints;

  // Reset
  outMass = 0;
  outCenterOfMass = SimdZero();

  // Integrate
  for (auto element : finiteElements) {
    for (int q = 0; q < kNumQuad; ++q) {
      outCenterOfMass += ToSimd(element.mapEvaluated[q]) * element.quadWeights[q];
      outMass += element.quadWeights[q];
    }
  }

  outCenterOfMass /= outMass;
  outMass *= density;
}

// Computes the center-of-mass of a rigid body based on a FEM discretization with uniform density
template <int kPolyOrder, int kQuadDegree>
inline void ComputeCenterOfMassFem(
    Span<triangular::Pk2DElement<kPolyOrder, kQuadDegree> const> finiteElements,
    real density,
    Vec4r& outCenterOfMass,
    real& outMass) {
  using ElementT = triangular::Pk2DElement<kPolyOrder, kQuadDegree>;
  constexpr int kNumQuad = ElementT::kNumQuadPoints;

  // Reset
  outMass = 0;
  outCenterOfMass = {};

  // Integrate
  for (auto element : finiteElements) {
    for (int q = 0; q < kNumQuad; ++q) {
      outCenterOfMass += 0.25_r * ToSimd(element.mapEvaluated[q]) *
          Dot(element.mapEvaluated[q], element.normals[q]) * element.quadWeights[q];
      outMass +=
          1_r / 3_r * Dot(element.mapEvaluated[q], element.normals[q]) * element.quadWeights[q];
    }
  }

  outCenterOfMass /= outMass;
  outMass *= density;
}

// Computes the volume of a rigid body based on a FEM discretization
template <int kPolyOrder, int kQuadDegree>
inline real ComputeVolumeFem(
    Span<triangular::Pk2DElement<kPolyOrder, kQuadDegree> const> finiteElements) {
  // volume = mass if density = 1
  real volume{};
  Vec4r com{}; // unused
  ComputeCenterOfMassFem(finiteElements, 1_r, com, volume);
  return volume;
}

// Computes the second moment of inertia of a rigid body based on a FEM discretization with unifor
// density
template <int kPolyOrder, int kQuadDegree>
inline VMatrix3x3r ComputeSecondMomentOfInertiaFem(
    Span<tetrahedral::Pk3DElement<kPolyOrder, kQuadDegree> const> finiteElements,
    real density,
    Vec4r const& centerOfMass) {
  using ElementT = tetrahedral::Pk3DElement<kPolyOrder, kQuadDegree>;
  constexpr int kNumQuad = ElementT::kNumQuadPoints;

  // Reset
  VMatrix3x3r outInertia = {};
  // Integrate
  for (auto element : finiteElements) {
    for (int q = 0; q < kNumQuad; ++q) {
      Vec4r y = ToSimd(element.mapEvaluated[q]) - centerOfMass;
      outInertia += Outer3(y, y) * element.quadWeights[q];
    }
  }

  outInertia *= density;
  return outInertia;
}

// Computes the second moment of inertia of a rigid body based on a FEM discretization with uniform
// density
template <int kPolyOrder, int kQuadDegree>
inline VMatrix3x3r ComputeSecondMomentOfInertiaFem(
    Span<triangular::Pk2DElement<kPolyOrder, kQuadDegree> const> finiteElements,
    real density,
    Vec4r const& centerOfMass) {
  using ElementT = triangular::Pk2DElement<kPolyOrder, kQuadDegree>;
  constexpr int kNumQuad = ElementT::kNumQuadPoints;

  // Reset
  VMatrix3x3r outInertia = {};
  // Integrate
  for (auto element : finiteElements) {
    for (int q = 0; q < kNumQuad; ++q) {
      Vec4r normal = ToSimd(element.normals[q]);
      Vec4r y = ToSimd(element.mapEvaluated[q]) - centerOfMass;
      outInertia += Outer3(y, y) * Dot(y, normal) * element.quadWeights[q];
    }
  }

  outInertia *= density / 5.0_r;
  return outInertia;
}

// Computes the second moment of inertia of a sphere with given uniform density and radius
inline void ComputeSecondMomentOfInertiaSphere(real density, real radius, VMatrix3x3r& outInertia) {
  real radius2 = radius * radius;
  real mass = 4_r * density * kPI * radius * radius2 / 3_r;
  auto I = VDiagonalMatrix<3>(2_r * mass * radius2 / 5_r);
  outInertia = (VEye<3>() * 0.5_r * Trace3x3(I)) - I;
}

// Computes the second moment of inertia of a cuboid with given uniform density and dimensions
inline void
ComputeSecondMomentOfInertiaCuboid(real density, Vec4r const& sizes, VMatrix3x3r& outInertia) {
  Vec4r sizes2 = ToSimdDirection(sizes * sizes);
  real kMass = (density / 12_r) * HProd<3>(sizes);
  auto I = VDiagonalMatrix<3>(kMass * (HSum<3>(sizes2) - sizes2));
  outInertia = (VEye<3>() * 0.5_r * Trace3x3(I)) - I;
}

// Computes linear velocity of a pivot (e.g. the center of mass) and angular velocity in world
// coordinates. The pivot is defined in local coordinates. The velocities are obtained by
// finite-differencing a rigid transform.
inline void ComputeRigidVelocityWorldSpace(
    real dt,
    TransformRT const& worldFromLocal,
    TransformRT const& worldFromLocalPrev,
    Vec4r const& pivotLocal,
    Vec4r& outLinearVelocityWorld,
    Vec4r& outAngularVelocityWorld) {
  Vec4r pivotWorld = worldFromLocal.TransformPoint(pivotLocal);
  Vec4r pivotWorldPrev = worldFromLocalPrev.TransformPoint(pivotLocal);
  outLinearVelocityWorld = (pivotWorld - pivotWorldPrev) / dt;

  // For W(t) = (Wx(t), Wy(t), Wz(t), 0), angular velocity quaternion in world coordinates
  //    dq(t)/dt = (1/2) * W(t) * q(t) = (q(t) - qprev)/h
  // Therefore, W(t) = (2/h) * (q(t) - qprev) * Inv(q(t))
  auto const& qrot = worldFromLocal.GetRotation();
  auto const& qrotPrev = worldFromLocalPrev.GetRotation();
  Quaternion dqrotdt = (qrot - qrotPrev) * (2_r / dt);
  outAngularVelocityWorld = (dqrotdt * Conjugate(qrot)).data;
}

// Convert a pose (using quaternion) to dofs (using rotation vector)
inline void ConvertRigidPoseToDofs(Span<real const> pose, Span<real> outDofs) {
  MOCHI_ASSERT_VERBOSE(pose.size() == RigidSize::kAll, "Invalid pose size");
  MOCHI_ASSERT_VERBOSE(outDofs.size() == RigidSize::kDAll, "Invalid dofs size");
  auto transform =
      TransformFromRawPose(AsConstView(pose).TopRows<RigidSize::kAll>(RigidSize::kAll));
  TransformToRawDofs(transform, AsView(outDofs).TopRows<RigidSize::kDAll>(RigidSize::kDAll));
}

// Convert dofs (using rotation vector) to pose (using quaternion)
inline void ConvertRigidDofsToPose(Span<real const> dofs, Span<real> outPose) {
  MOCHI_ASSERT_VERBOSE(dofs.size() == RigidSize::kDAll, "Invalid dofs size");
  MOCHI_ASSERT_VERBOSE(outPose.size() == RigidSize::kAll, "Invalid pose size");
  auto transform =
      TransformFromRawDofs(AsConstView(dofs).TopRows<RigidSize::kDAll>(RigidSize::kDAll));
  TransformToRawPose(transform, AsView(outPose).TopRows<RigidSize::kAll>(RigidSize::kAll));
}

// Convert dof flags to pose flags
inline void ConvertRigidDofFlagsToPoseFlags(Span<bool const> dofs, Span<bool> outPose) {
  MOCHI_ASSERT_VERBOSE(dofs.size() == RigidSize::kDAll, "Invalid dofs size");
  MOCHI_ASSERT_VERBOSE(outPose.size() == RigidSize::kAll, "Invalid pose size");
  MOCHI_ASSERT_VERBOSE(
      dofs[RigidSize::kDTrans] == dofs[RigidSize::kDTrans + 1] &&
          dofs[RigidSize::kDTrans] == dofs[RigidSize::kDTrans + 2],
      "Rotation indices must be all or none");
  // Copy translation indices
  std::copy(dofs.begin(), dofs.begin() + RigidSize::kDTrans, outPose.begin());
  // Set rotation indices
  std::fill(outPose.begin() + RigidSize::kTrans, outPose.end(), dofs[RigidSize::kDTrans]);
}

// Given a Lie Jacobian dx/drot, with rot a rigid rotation, this function computes the
// Jacobian dx/drvec, with rvec expressing the rotation as a full rotation vector.
template <int kRowsAtCompileTime, krylov::Direction kMajorDirection, int kLeadingDim>
inline void TransportInputOfLieJacobian(
    Vec4r const& rvec,
    Matrix<
        real,
        kRowsAtCompileTime,
        RigidSize::kDRot,
        kMajorDirection,
        krylov::Ownership::View,
        kLeadingDim> outJacobian) {
  VMatrix3x3r transport = DRotIncrementDRotVector(rvec);
  MOCHI_FILO_STACK_ALLOCATOR(
      allocator, RigidSize::kDRot * 100 * sizeof(real)); // Stack memory for 100 rows
  Matrix<real, kRowsAtCompileTime, RigidSize::kDRot> aux(
      outJacobian.Rows(), RigidSize::kDRot, &allocator);
  aux = outJacobian * AsMatrixView(transport);
  outJacobian = aux;
}

// Given a Lie Jacobian dx/dq, with q = (trans, rot) a rigid state, this function computes the
// Jacobian dx/du, with u expressing the rotation rot as a full rotation vector rvec.
template <int kRowsAtCompileTime, krylov::Direction kMajorDirection, int kLeadingDim>
inline void TransportInputOfLieJacobian(
    ColumnVectorView<real const, RigidSize::kDAll> u,
    Matrix<
        real,
        kRowsAtCompileTime,
        RigidSize::kDAll,
        kMajorDirection,
        krylov::Ownership::View,
        kLeadingDim> outJacobian) {
  // We only need to transport the rotation Jacobian
  TransportInputOfLieJacobian(
      Load<RigidSize::kDRot, Vec4r>(u.data() + RigidSize::kDTrans),
      outJacobian.template RightCols<RigidSize::kDRot>(RigidSize::kDRot));
}

// Given a Lie Jacobian drot/dx, with rot a rigid rotation, this function computes the
// Jacobian du/dx, with u expressing the rotation rot as a full rotation vector rvec.
template <int kColsAtCompileTime, krylov::Direction kMajorDirection, int kLeadingDim>
inline void TransportOutputOfLieJacobian(
    Vec4r const& rvec,
    Matrix<
        real,
        RigidSize::kDRot,
        kColsAtCompileTime,
        kMajorDirection,
        krylov::Ownership::View,
        kLeadingDim> outJacobian) {
  VMatrix3x3r transport = DRotVectorDRotIncrement(rvec);
  MOCHI_FILO_STACK_ALLOCATOR(
      allocator, RigidSize::kDRot * 100 * sizeof(real)); // Stack memory for 100 columns
  Matrix<real, RigidSize::kDRot, kColsAtCompileTime> aux(
      RigidSize::kDRot, outJacobian.Cols(), &allocator);
  aux = AsMatrixView(transport) * outJacobian;
  outJacobian = aux;
}

// Given a Lie Jacobian dq/dx, with q = (trans, rot) a rigid state, this function computes the
// Jacobian du/dx, with u expressing the rotation rot as a full rotation vector rvec.
template <int kColsAtCompileTime, krylov::Direction kMajorDirection, int kLeadingDim>
inline void TransportOutputOfLieJacobian(
    ColumnVectorView<real const, RigidSize::kDAll> u,
    Matrix<
        real,
        RigidSize::kDAll,
        kColsAtCompileTime,
        kMajorDirection,
        krylov::Ownership::View,
        kLeadingDim> outJacobian) {
  // We only need to transport the rotation Jacobian
  TransportOutputOfLieJacobian(
      Load<RigidSize::kDRot, Vec4r>(u.data() + RigidSize::kDTrans),
      outJacobian.template BottomRows<RigidSize::kDRot>(RigidSize::kDRot));
}

// Given two rotations rotNew and rotOld, with relative rotation rotDelta = rotNew * rotOld^T, this
// operation implements the chain rule df/drotOld = df/drotDelta * drotDelta/drotOld.
inline void ChainRotationGradientDDeltaDOld(
    Quaternion const& rotNew,
    Quaternion const& rotOld,
    Vec4r& inOutGrad) {
  auto const rotDelta = rotNew * rotOld.GetConjugate();
  auto jac = lie::DMultRotaRotTRotbDRot(ToVMatrix3x3(rotDelta)); // rotDelta = rotNew * rotOld^T
  inOutGrad = DotVecMat3x3(inOutGrad, jac);
}

// Given two rigid transforms qNew and qOld, with relative transform qDelta, this operation
// implements the chain rule df/dqOld = df/dqDelta * dqDelta/dqOld.
inline void ChainRigidGradientDDeltaDOld(
    TransformRT const& qNew,
    TransformRT const& qOld,
    ColumnVectorView<real, RigidSize::kDAll> inOutGrad) {
  // For the translation: dxDelta/dxOld = - eye
  Store<RigidSize::kDTrans>(inOutGrad.data(), -Load<Vec4r>(inOutGrad.data()));
  // For the rotation: drDelta/drOld = - rDelta
  Vec4r rotGrad = Load<RigidSize::kDRot, Vec4r>(inOutGrad.data() + RigidSize::kDTrans);
  ChainRotationGradientDDeltaDOld(qNew.GetRotation(), qOld.GetRotation(), rotGrad);
  Store<RigidSize::kDRot>(inOutGrad.data() + RigidSize::kDTrans, rotGrad);
}

// Given two rotations rotNew and rotOld, relative rotation rotDelta = rotNew * rotOld^T, this
// operation implements the chain rule df/drotOld = df/drotNew * drotNew/drotOld.
inline void ChainRotationGradientDNewDOld(
    Quaternion const& rotNew,
    Quaternion const& rotOld,
    Vec4r& inOutGrad) {
  auto const rotDelta = rotNew * rotOld.GetConjugate();
  auto jac = lie::DMultRotaRotRotbDRot(ToVMatrix3x3(rotDelta)); // rotNew = rotDelta * rotOld
  inOutGrad = DotVecMat3x3(inOutGrad, jac);
}

} // namespace mochi
