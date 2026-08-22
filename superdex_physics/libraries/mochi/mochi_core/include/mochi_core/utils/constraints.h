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

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/transform_rt.h>

#include <optional>

namespace mochi {

// ==============================================================================================
// Standalone constraint evaluation functions
// ==============================================================================================

/**
 * @brief Evaluates a position constraint for a deformable node.
 *
 * Computes the constraint value and Jacobian for a deformable-node position constraint. The
 * constraint measures the difference between the world position of a local point and a target world
 * position.
 *
 * @param worldFromLocal A rigid transform to apply to the local position
 * @param posLocal The deformable-node position in local coordinates
 * @param posTargetWorld The target position in world coordinates
 * @param[out] outVal Output constraint value (3D vector). If nullptr, value computation is skipped
 * @param[out] outJac Output Jacobian matrix wrt the deformable-node local position (3x3). If
 * nullptr, Jacobian computation is skipped
 * @param[out] outJacTarget Output Jacobian matrix wrt the target position (3x3). If nullptr,
 * Jacobian computation is skipped
 */
void EvalDeformableNodeFixedConstraint(
    TransformRT const& worldFromLocal,
    Real3 const& posLocal,
    Real3 const& posTargetWorld,
    ColumnVector<real, 3>* outVal,
    RowMatrix<real, 3, 3>* outJac,
    RowMatrix<real, 3, 3>* outJacTarget);

/**
 * @brief Evaluates a position constraint between two deformable nodes.
 *
 * Computes the constraint value and Jacobian for coupling the positions of two deformable nodes.
 * The constraint measures the difference between the world positions of the points.
 *
 * @param worldFromLocalA Rigid transform to apply to point A
 * @param worldFromLocalB Rigid transform to apply to point B
 * @param posLocalA Deformable-node position A in local coordinates
 * @param posLocalB Deformable-node position B in local coordinates
 * @param[out] outVal Output constraint value (3D vector). If nullptr, value computation is skipped
 * @param[out] outJac Output Jacobian matrix wrt the deformable-node local positions (3x6). First 3
 * columns are wrt node A, last 3 columns are wrt node B. If nullptr, Jacobian computation is
 * skipped
 */
void EvalDeformableNodeToDeformableNodeConstraint(
    TransformRT const& worldFromLocalA,
    TransformRT const& worldFromLocalB,
    Real3 const& posLocalA,
    Real3 const& posLocalB,
    ColumnVector<real, 3>* outVal,
    RowMatrix<real, 3, 6>* outJac);

/**
 * @brief Evaluates a position constraint between a deformable node and a rigid body.
 *
 * Computes the constraint value and Jacobian for coupling a deformable node to a point on a rigid
 * body. The constraint measures the difference between the world position of the deformable node
 * and the world position of the point on the rigid body.
 *
 * @param worldFromLocalRigid Transform of the rigid body
 * @param worldFromLocalDeformable Rigid transform to apply to the deformable-node local position
 * @param posLocalDeformable Deformable-node position in local coordinates
 * @param posLocalRigid Local position of the attachment point on the rigid body
 * @param[out] outVal Output constraint value (3D vector). If nullptr, value computation is skipped
 * @param[out] outJac Output Jacobian matrix wrt the rigid transform and the deformable-node local
 * position (3x9). First 3 columns are wrt the translation of the rigid body, next 3 columns are wrt
 * the rotation of the rigid body, last 3 columns are wrt the deformable-node local position. If
 * nullptr, Jacobian computation is skipped
 */
void EvalDeformableNodeToRigidConstraint(
    TransformRT const& worldFromLocalRigid,
    TransformRT const& worldFromLocalDeformable,
    Real3 const& posLocalDeformable,
    Real3 const& posLocalRigid,
    ColumnVector<real, 3>* outVal,
    RowMatrix<real, 3, 9>* outJac);

/**
 * @brief Evaluates a position constraint for a point on a rigid body.
 *
 * Computes the constraint value and Jacobian for a position constraint on a point of a rigid body.
 * The constraint measures the difference between the world position of a local point and a target
 * world position.
 *
 * @param worldFromLocal Transform of the rigid body
 * @param posLocal Local position of the attachment point on the rigid body
 * @param posTargetWorld Target position in world coordinates
 * @param[out] outVal Output constraint value (3D vector). If nullptr, value computation is skipped
 * @param[out] outJac Output Jacobian matrix wrt the rigid transform (3x6). First 3 columns are wrt
 * the translation of the rigid body, last 3 columns are wrt the rotation of the rigid body. If
 * nullptr, Jacobian computation is skipped
 * @param[out] outJacTarget Output Jacobian matrix wrt the target position (3x3). If nullptr,
 * Jacobian computation is skipped
 */
void EvalRigidPositionFixedConstraint(
    TransformRT const& worldFromLocal,
    Real3 const& posLocal,
    Real3 const& posTargetWorld,
    ColumnVector<real, 3>* outVal,
    RowMatrix<real, 3, 6>* outJac,
    RowMatrix<real, 3, 3>* outJacTarget);

/**
 * @brief Evaluates a position constraint for a point on a rigid body, where the target is defined
 * as a transform of the same rigid body.
 *
 * Computes the constraint value and Jacobian for a position constraint on a point of a rigid body.
 * The constraint measures the difference between the world position of the local point on the body
 * and the world position of the same local point under the target transform.
 *
 * @param worldFromLocal Transform of the rigid body
 * @param worldFromLocalTarget Target transform of the rigid body
 * @param posLocal Local position of the attachment point on the rigid body
 * @param[out] outVal Output constraint value (3D vector). If nullptr, value computation is skipped
 * @param[out] outJac Output Jacobian matrix wrt the rigid transform (3x6). First 3 columns are wrt
 * the translation of the rigid body, last 3 columns are wrt the rotation of the rigid body. If
 * nullptr, Jacobian computation is skipped
 * @param[out] outJacTarget Output Jacobian matrix wrt the target's transform (3x6). First 3 columns
 * are wrt the target translation, last 3 columns are wrt the target rotation. If nullptr, Jacobian
 * computation is skipped
 */
void EvalRigidPositionToRigidTargetConstraint(
    TransformRT const& worldFromLocal,
    TransformRT const& worldFromLocalTarget,
    Real3 const& posLocal,
    ColumnVector<real, 3>* outVal,
    RowMatrix<real, 3, 6>* outJac,
    RowMatrix<real, 3, 6>* outJacTarget);

/**
 * @brief Evaluates a constraint on a rotation.
 *
 * Computes the constraint value and Jacobian for a rotation constraint on a local frame of a
 * rotating 3D frame. The constraint measures the relative rotation between the local frame rotated
 * to the world and a target rotation, expressed as a rotation vector.
 *
 * @param rotWorldFromLocal Rotation
 * @param rotLocal Local rotation frame
 * @param rotTargetWorld Target rotation in world coordinates
 * @param[out] outVal Output constraint value (3D rotation vector). If nullptr, value computation is
 * skipped
 * @param[out] outJac Output Jacobian matrix wrt the rigid rotation (3x3). If nullptr, Jacobian
 * computation is skipped
 * @param[out] outJacTarget Output Jacobian matrix wrt the target rotation (3x3). If nullptr,
 * Jacobian computation is skipped
 */
void EvalRotationFixedConstraint(
    Quaternion const& rotWorldFromLocal,
    Quaternion const& rotLocal,
    Quaternion const& rotTargetWorld,
    ColumnVector<real, 3>* outVal,
    RowMatrix<real, 3, 3>* outJac,
    RowMatrix<real, 3, 3>* outJacTarget);

/**
 * @brief Evaluates a spherical joint constraint between two rigid bodies.
 *
 * Computes the constraint value and Jacobian for a spherical (ball-and-socket) joint constraint.
 * The constraint measures the difference between the world positions of two local points.
 *
 * @param worldFromLocalA transform of rigid body A
 * @param worldFromLocalB transform of rigid body B
 * @param posLocalA Local position of the joint attachment on body A
 * @param posLocalB Local position of the joint attachment on body B
 * @param[out] outVal Output constraint value (3D vector). If nullptr, value computation is skipped
 * @param[out] outJac Output Jacobian matrix wrt the transforms of both bodies (3x12). First 3
 * columns are wrt the translation of body A, next 3 columns are wrt the rotation of body A, next 3
 * columns are wrt the translation of body B, last 3 columns are wrt the rotation of body B. If
 * nullptr, Jacobian computation is skipped
 */
void EvalRigidSphericalJointConstraint(
    TransformRT const& worldFromLocalA,
    TransformRT const& worldFromLocalB,
    Real3 const& posLocalA,
    Real3 const& posLocalB,
    ColumnVector<real, 3>* outVal,
    RowMatrix<real, 3, 12>* outJac);

/**
 * @brief Evaluates a prismatic joint constraint between two rigid bodies.
 *
 * Computes the constraint value and Jacobian for a prismatic (sliding) joint constraint. The
 * constraint measures the relative translation between two rigid bodies projected onto a local
 * reference frame defined on body A, allowing translation along a single axis while constraining
 * the two perpendicular directions. Optional min/max limits can be specified for the allowed
 * translation range.
 *
 * @note It does not constrain rotations. To constrain rotations and allow only sliding, it should
 * be combined with EvalJointRotationRangeConstraint.
 * @note However, the Jacobian acts on the rotation of body A because the local frame is expressed
 * in body A.
 *
 * @param worlFromLocalA Transform of rigid body A
 * @param posWorldFromLocalB Position of rigid body B
 * @param localFrame Local reference frame in body A defining the sliding axis (Z-axis is the free
 * axis)
 * @param tRef Reference translation vector in the local frame
 * @param max Optional maximum translation limit along the free axis
 * @param min Optional minimum translation limit along the free axis
 * @param[out] outVal Output constraint value (3D vector). If nullptr, value computation is skipped
 * @param[out] outJac Output Jacobian matrix wrt the transform of body A and the position of body B
 * (3x9). First 3 columns are wrt the translation of body A, next 3 columns are wrt the rotation of
 * body A, last 3 columns are wrt the translation of body B. If nullptr, Jacobian computation is
 * skipped
 */
void EvalRigidPrismaticJointConstraint(
    TransformRT const& worlFromLocalA,
    Real3 const& posWorldFromLocalB,
    Quaternion const& localFrame,
    Real3 const& tRef,
    std::optional<real> max,
    std::optional<real> min,
    ColumnVector<real, 3>* outVal,
    RowMatrix<real, 3, 9>* outJac);

/**
 * @brief Evaluates a constraint beteween the relative rotation of two rotations A and B, and a
 * target rotation. It models a constraint between a rotation joint and a target.
 *
 * Computes the constraint value and Jacobian for making the relative rotation between two rigid
 * bodies match a target rotation. The constraint measures the difference between the relative
 * rotation and the target rotation, expressed in a reference frame as a rotation vector.
 *
 * @param rotA Rotation A
 * @param rotB Rotation B
 * @param refFrame Reference frame for expressing the relative rotation, local to A
 * @param targetRot Target relative rotation
 * @param[out] outVal Output constraint value (3D rotation vector). If nullptr, value computation is
 * skipped
 * @param[out] outJac Output Jacobian matrix wrt both rotations (3x6). The first 3 columns are wrt
 * rotation A, the last 3 columns are wrt rotation B. If nullptr, Jacobian computation is skipped
 * @param[out] outJacTarget Output Jacobian matrix wrt the target rotation (3x3). If nullptr,
 * Jacobian computation is skipped
 */
void EvalJointRotationTargetConstraint(
    Quaternion const& rotA,
    Quaternion const& rotB,
    Quaternion const& refFrame,
    Quaternion const& targetRot,
    ColumnVector<real, 3>* outVal,
    RowMatrix<real, 3, 6>* outJac,
    RowMatrix<real, 3, 3>* outJacTarget);

/**
 * @brief Evaluates a constraint on the relative rotation between the rotation of a rigid body and
 * the rotation of a rod element, and a target rotation.
 *
 * @param rotRigid Rotation of the rigid body
 * @param worldFromLocalRod Transform of the rod actor
 * @param X0Rod Reference position of the first node of the rod element in the rod's local frame
 * @param X1Rod Reference position of the second node
 * @param frameAxisRod First cross-section axis of the rod element, which should be consistent with
 * the rod element DOFs.
 * @param rodElementDofs Current DOFs of the rod element (size must be 8)
 * @param refFrame Reference frame for expressing the relative rotation, local to the rigid body
 * @param targetRot Target relative rotation
 * @param[out] outVal Output constraint value (3D rotation vector); skipped if null
 * @param[out] outJac Output Jacobian matrix wrt the rigid rotation and rod element DoFs. For
 * indexing, the DoFs are ordered as: Rigid rotation DoFs (3), first rod node DoFs (4), second rod
 * node DoFs (4).
 * @param[out] outJacTarget Output Jacobian matrix wrt the target rotation (3x3). If nullptr,
 * Jacobian computation is skipped
 */
void EvalRodElementRotationToRigidConstraint(
    Quaternion const& rotRigid,
    TransformRT const& worldFromLocalRod,
    Vec4r const& X0Rod,
    Vec4r const& X1Rod,
    Vec4r const& frameAxisRod,
    Span<real const> rodElementDofs,
    Quaternion const& refFrame,
    Quaternion const& targetRot,
    ColumnVector<real, 3>* outVal,
    RowMatrix<real, 3, 11>* outJac,
    RowMatrix<real, 3, 3>* outJacTarget);

/**
 * @brief Evaluates a constraint on the range of relative rotation between two rotations A and B. It
 * models a constraint on the range of motion of a rotation joint.
 *
 * Computes the constraint value and Jacobian for limiting the relative rotation between two
 * rotations. The constraint measures the difference between the relative rotation of A and B
 * expressed in a reference frame (local to A) and a reference rotation, and each component of the
 * difference rotation vector is constrained to stay within specified min/max bounds. The function
 * returns whether some or no rotation component is actively constrained. A return value of false
 * signals that the output values are trivially zero.
 *
 * @param rotA Rotation A
 * @param rotB Rotation B
 * @param refFrame Reference frame for expressing the relative rotation, local to A
 * @param refRot Reference rotation, in refFrame
 * @param minRotVec Minimum allowed rotation vector components
 * @param maxRotVec Maximum allowed rotation vector components
 * @param[out] outVal Output constraint value (3D vector). If nullptr, value computation is skipped
 * @param[out] outJac Output Jacobian matrix wrt both rotations (3x6). The first 3 columns are wrt
 * rotation A, the last 3 columns are wrt rotation B. If nullptr, Jacobian computation is skipped
 * @param[out] outActive True if some component of the rotation is outside bounds (constraint is
 * active), false otherwise
 */
void EvalJointRotationRangeConstraint(
    Quaternion const& rotA,
    Quaternion const& rotB,
    Quaternion const& refFrame,
    Quaternion const& refRot,
    Real3 const& minRotVec,
    Real3 const& maxRotVec,
    ColumnVector<real, 3>* outVal,
    RowMatrix<real, 3, 6>* outJac,
    bool& outActive);

/**
 * @brief Evaluates a constraint on a single degree-of-freedom.
 *
 * Computes the constraint value and Jacobian for constraining a single scalar DOF to match a
 * target value. The constraint measures the difference between the current DOF value and the
 * target value.
 *
 * @param dof Current DOF value
 * @param target Target DOF value
 * @param[out] outVal Output constraint value. If nullptr, value computation is skipped
 * @param[out] outJac Output Jacobian wrt the DOF. If nullptr, Jacobian computation is skipped
 * @param[out] outJacTarget Output Jacobian wrt the target. If nullptr, Jacobian computation is
 * skipped
 */
inline void EvalSingleDofTargetConstraint(
    real dof,
    real target,
    real* outVal,
    real* outJac,
    real* outJacTarget) {
  if (outVal) {
    *outVal = dof - target;
  }

  if (outJac) {
    *outJac = 1_r;
  }

  if (outJacTarget) {
    *outJacTarget = -1_r;
  }
}

/**
 * @brief Evaluates a range constraint for a single degree-of-freedom.
 *
 * Computes the constraint value and Jacobian for constraining a single scalar DOF to stay within
 * specified bounds. The constraint measures the violation magnitude for values outside the bounds;
 * the value is zero when the DOF is within [min, max] (positive if above max, positive if below
 * min).
 *
 * @param dof Current DOF value
 * @param min Minimum allowed DOF value
 * @param max Maximum allowed DOF value
 * @param[out] outVal Output constraint value. If nullptr, value computation is skipped
 * @param[out] outJac Output Jacobian wrt the DOF. If nullptr, Jacobian computation is skipped
 * @param[out] outActive True if the dof is outside bounds (constraint is active), false otherwise
 */
inline void EvalSingleDofRangeConstraint(
    real dof,
    real min,
    real max,
    real* outVal,
    real* outJac,
    bool& outActive) {
  real topVal = dof - max;
  real botVal = min - dof;
  if (outVal) {
    *outVal = topVal > 0_r ? topVal : (botVal > 0_r ? botVal : 0_r);
  }
  if (outJac) {
    *outJac = topVal > 0_r ? 1_r : (botVal > 0_r ? -1_r : 0_r);
  }
  outActive = topVal > 0_r || botVal > 0_r;
}

/**
 * @brief Evaluates a constraint on the range of a rotation.
 *
 * Computes the constraint value and Jacobian for constraining each component of a rotation vector
 * to stay within specified min/max bounds. The constraint measures the violation magnitude for
 * each component outside its bounds.
 *
 * @note The constraint value is the same as applying EvalSingleDofRangeConstraint per component of
 * the rotation vector, but the Jacobian accounts for Lie derivatives
 *
 * @param rotVec Rotation expressed as a rotation vector
 * @param minRotVec Minimum allowed rotation vector components
 * @param maxRotVec Maximum allowed rotation vector components
 * @param[out] outVal Output constraint value (3D vector). If nullptr, value computation is skipped
 * @param[out] outJac Output Jacobian matrix wrt the rotation (3x3). If nullptr, Jacobian
 * computation is skipped
 * @param[out] outActive True if some component of the rotation is outside bounds (constraint is
 * active), false otherwise
 */
void EvalRotationRangeConstraint(
    Real3 const& rotVec,
    Real3 const& minRotVec,
    Real3 const& maxRotVec,
    ColumnVector<real, 3>* outVal,
    RowMatrix<real, 3, 3>* outJac,
    bool& outActive);
} // namespace mochi
