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

#include <gtest/gtest.h>

#include <mochi_core/articulated_body/articulated_body.h>
#include <mochi_core/articulated_body/articulated_body_hessian.h>
#include <mochi_core/test/linear_algebra_test_helpers.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/rigid_body_utils.h>
#include <vector>

using namespace mochi;
using namespace mochi::articulated;
using namespace mochi::test;

struct ArticulatedBodyBundle {
  ArticulatedProperties props;
  ParentIndexArray parents;
  DynamicArray<ArticulatedJointType> jointTypes;
  DynamicArray<Real3> jointAxes;
  DynamicArray<ArticulatedDofInfo> dofInfo;
  DynamicArray<ArticulatedPoseInfo> poseInfo;
  RestTransformArray restTransforms;
  TransformRT worldFromRoot;
  DynamicArray<TransformRT> jointTransforms;
  ColumnVector<real> reducedPose, fullPose;
  DynamicArray<TransformRT> linkTransforms;
  mochi_default_random_engine generator = RandomGenerator(42);

  ArticulatedBodyBundle() {
    props = {
        .numLinks = 0,
        .fullDofsDim = 0,
        .fullPoseDim = 0,
        .reducedDofsDim = 0,
        .reducedPoseDim = 0};
  }

  Real3 RandomVector() {
    Real3 v;
    SetRandom(generator, -1_r, 1_r, v);
    return v;
  }

  TransformRT RandomTransform() {
    Real4 q;
    SetRandom(generator, -1_r, 1_r, q);
    return TransformRT{Normalize(Quaternion(q)), RandomVector()};
  }

  void AddJoint(int parent, ArticulatedJointType type) {
    parents.push_back(parent);
    // create random axis
    Real3 axis = Normalize(RandomVector());
    // store joint type and axis (axis is only meaningful for prismatic/revolute joints)
    jointTypes.push_back(type);
    if (type == ArticulatedJointType::Prismatic || type == ArticulatedJointType::Revolute) {
      jointAxes.push_back(axis);
    } else {
      jointAxes.emplace_back();
    }
    dofInfo = articulated::SetupJointDofs(jointTypes);
    poseInfo = articulated::SetupJointPose(jointTypes);
    // Update props
    props.numLinks++;
    props.fullDofsDim += RigidSize::kDAll;
    props.fullPoseDim += RigidSize::kAll;
    props.reducedDofsDim += dofInfo.back().GetSize();
    props.reducedPoseDim += poseInfo.back().GetSize();
    // Create random rest and root transforms
    restTransforms.push_back({RandomTransform(), RandomTransform()});
    worldFromRoot = RandomTransform();
    jointTransforms.push_back(TransformRT::Identity());
    // Add reduced pose
    ColumnVector<real> dofs(props.reducedDofsDim);
    reducedPose.Resize(props.reducedPoseDim);
    fullPose.Resize(props.fullPoseDim);
    dofs.SetZero();
    ConvertDofsToPose(jointTypes, dofInfo, poseInfo, dofs, reducedPose);
    // Recompute all transforms
    linkTransforms.resize(props.numLinks);
    // Set to a random joint
    ApplyRandomDeltaPose();
  }

  ColumnVector<real> GetRandomDof() {
    ColumnVector<real> deltaReduced(props.reducedDofsDim);
    SetRandom(generator, -1_r, 1_r, deltaReduced.GetSpan());
    return deltaReduced;
  }

  void ApplyRandomDeltaPose() {
    ColumnVector<real> deltaReduced = GetRandomDof();
    ApplyRandomDeltaPose(deltaReduced);
  }

  void ApplyRandomDeltaPose(ColumnVectorView<real const> deltaReduced) {
    auto newPose = reducedPose.Duplicate();
    AddLieDeltaToReducedPose(jointTypes, dofInfo, poseInfo, reducedPose, deltaReduced, newPose);
    ComputeFullPose(
        jointTypes,
        jointAxes,
        poseInfo,
        parents,
        restTransforms,
        worldFromRoot,
        newPose,
        jointTransforms,
        linkTransforms,
        fullPose);
    reducedPose = newPose;
  }

  int HessianSize() const {
    return props.reducedDofsDim * props.reducedDofsDim * props.numLinks * RigidSize::kDAll;
  }
};

/*************************************************************************************************/
static void FlatVector(
    Span<ArticulatedJointType const> jointTypes,
    // Updated pose
    Span<TransformRT const> linkTransforms,
    // Reference pose
    Span<TransformRT const> linkTransformsRef,
    // Flat Vector
    ColumnVectorView<real> outFlatVector) {
  MOCHI_ASSERT(
      outFlatVector.size() == jointTypes.size() * RigidSize::kDAll, "Incorrect size of FlatVector")
  for (int link = 0; link < isize(jointTypes); link++) {
    Real3 t = linkTransforms[link].GetTranslation();
    auto r =
        linkTransforms[link].GetRotation() * linkTransformsRef[link].GetRotation().GetConjugate();
    outFlatVector.MiddleRows(link * RigidSize::kDAll, RigidSize::kDTrans) = AsView(t);
    outFlatVector.MiddleRows(link * RigidSize::kDAll + RigidSize::kDTrans, RigidSize::kDRot) =
        AsView(r.ToRotationVector());
  }
}

static void FlatJacobian(
    Span<ArticulatedJointType const> jointTypes,
    Span<Real3 const> jointAxes,
    Span<ArticulatedDofInfo const> dofInfo,
    Span<ArticulatedPoseInfo const> poseInfo,
    Span<int const> parents,
    Span<ArticulatedRestTransform const> restTransforms,
    TransformRT const& worldFromRoot,
    // Updated pose
    Span<TransformRT const> jointTransforms,
    Span<TransformRT const> linkTransforms,
    // Reference pose
    Span<TransformRT const> jointTransformsRef,
    Span<TransformRT const> linkTransformsRef,
    // Flat Jacobian
    RowMatrixView<real> outFlatJacobian) {
  Jacobian(
      jointTypes,
      parents,
      jointAxes,
      dofInfo,
      restTransforms,
      worldFromRoot,
      jointTransforms,
      linkTransforms,
      outFlatJacobian);
  ColumnVector<real> pose(GetReducedPoseSize(poseInfo));
  ColumnVector<real> poseRef(GetReducedPoseSize(poseInfo));
  ColumnVector<real> poseDelta(GetReducedDofsSize(dofInfo));
  internal::ComputeReducedPose(jointTypes, jointAxes, poseInfo, jointTransformsRef, poseRef);
  internal::ComputeReducedPose(jointTypes, jointAxes, poseInfo, jointTransforms, pose);
  ComputeLieDeltaReducedPose(jointTypes, dofInfo, poseInfo, poseRef, pose, poseDelta);
  TransportInputOfLieJacobian(jointTypes, dofInfo, poseDelta, outFlatJacobian);
  for (int i = 0; i < linkTransforms.size(); ++i) {
    ColumnVector<real, RigidSize::kDAll> linkDelta;
    TransformToRawDofs(linkTransforms[i] * Invert(linkTransformsRef[i]), linkDelta);
    TransportOutputOfLieJacobian(
        linkDelta,
        outFlatJacobian.MiddleRows<RigidSize::kDAll>(i * RigidSize::kDAll, RigidSize::kDAll));
  }
}

static void FlatJacobianFD(
    ArticulatedBodyBundle const& bundle,
    ColumnVectorView<real> deltaDof,
    RowMatrix<real>& flatJacobian,
    real eps = 1e-4_r) {
  flatJacobian.Resize(bundle.props.fullDofsDim, bundle.props.reducedDofsDim);

  ColumnVector<real> deltaDofp = deltaDof.Duplicate();
  ColumnVector<real> deltaDofn = deltaDof.Duplicate();
  ColumnVector<real> flatVectorp(bundle.props.fullDofsDim);
  ColumnVector<real> flatVectorn(bundle.props.fullDofsDim);

  for (uint32_t i = 0; i < bundle.props.reducedDofsDim; ++i) {
    // p
    ArticulatedBodyBundle bundlep = bundle;
    deltaDofp = deltaDof;
    deltaDofp[i] += eps;
    bundlep.ApplyRandomDeltaPose(deltaDofp);
    FlatVector(bundlep.jointTypes, bundlep.linkTransforms, bundle.linkTransforms, flatVectorp);
    // n
    ArticulatedBodyBundle bundlen = bundle;
    deltaDofn = deltaDof;
    deltaDofn[i] -= eps;
    bundlen.ApplyRandomDeltaPose(deltaDofn);
    FlatVector(bundlen.jointTypes, bundlen.linkTransforms, bundle.linkTransforms, flatVectorn);
    // fin-diff
    flatJacobian.Col(i) = flatVectorp - flatVectorn;
  }
  flatJacobian /= 2_r * eps;
}

static void
HessianFD(ArticulatedBodyBundle const& bundle, ArticulatedHessian& hessian, real eps = 1e-4_r) {
  ColumnVector<real> deltaDofp(bundle.props.reducedDofsDim);
  ColumnVector<real> deltaDofn(bundle.props.reducedDofsDim);
  RowMatrix<real> flatJacobianp(bundle.props.fullDofsDim, bundle.props.reducedDofsDim);
  RowMatrix<real> flatJacobiann(bundle.props.fullDofsDim, bundle.props.reducedDofsDim);

  for (uint32_t i = 0; i < bundle.props.reducedDofsDim; ++i) {
    // p
    ArticulatedBodyBundle bundlep = bundle;
    deltaDofp.SetZero();
    deltaDofp[i] += eps;
    bundlep.ApplyRandomDeltaPose(deltaDofp);
    FlatJacobian(
        bundle.jointTypes,
        bundle.jointAxes,
        bundle.dofInfo,
        bundle.poseInfo,
        bundle.parents,
        bundle.restTransforms,
        bundle.worldFromRoot,
        // bundlep is the new pose
        bundlep.jointTransforms,
        bundlep.linkTransforms,
        // bundle is the ref pose
        bundle.jointTransforms,
        bundle.linkTransforms,
        flatJacobianp);
    // n
    ArticulatedBodyBundle bundlen = bundle;
    deltaDofn.SetZero();
    deltaDofn[i] -= eps;
    bundlen.ApplyRandomDeltaPose(deltaDofn);
    FlatJacobian(
        bundle.jointTypes,
        bundle.jointAxes,
        bundle.dofInfo,
        bundle.poseInfo,
        bundle.parents,
        bundle.restTransforms,
        bundle.worldFromRoot,
        // bundlen is the new pose
        bundlen.jointTransforms,
        bundlen.linkTransforms,
        // bundle is the ref pose
        bundle.jointTransforms,
        bundle.linkTransforms,
        flatJacobiann);
    // fin-diff
    hessian[i] = flatJacobianp - flatJacobiann;
    hessian[i] /= 2_r * eps;
  }
}

static void CreateComplexBody(ArticulatedBodyBundle& bundle) {
  bundle.AddJoint(-1, ArticulatedJointType::Free);
  // First chain
  bundle.AddJoint(0, ArticulatedJointType::Spherical);
  bundle.AddJoint(1, ArticulatedJointType::Hard);
  bundle.AddJoint(2, ArticulatedJointType::Prismatic);
  bundle.AddJoint(3, ArticulatedJointType::Spherical);
  bundle.AddJoint(4, ArticulatedJointType::Revolute);
  // Second chain
  bundle.AddJoint(0, ArticulatedJointType::Spherical);
  bundle.AddJoint(6, ArticulatedJointType::Hard);
  bundle.AddJoint(7, ArticulatedJointType::Prismatic);
  bundle.AddJoint(8, ArticulatedJointType::Free);
  bundle.AddJoint(9, ArticulatedJointType::Revolute);
  bundle.ApplyRandomDeltaPose();
}

// Flat jacobian consistency check
TEST(ArticulatedBody, FlatJacobian_Consistency) {
  ArticulatedBodyBundle bundle;
  CreateComplexBody(bundle);

  ArticulatedBodyBundle bundleNew = bundle;
  ColumnVector<real> deltaDof = bundleNew.GetRandomDof();
  bundleNew.ApplyRandomDeltaPose(deltaDof);

  // Flat Jacobian
  RowMatrix<real> flatJacobian(bundle.props.fullDofsDim, bundle.props.reducedDofsDim);
  FlatJacobian(
      bundle.jointTypes,
      bundle.jointAxes,
      bundle.dofInfo,
      bundle.poseInfo,
      bundle.parents,
      bundle.restTransforms,
      bundle.worldFromRoot,
      // new pose
      bundleNew.jointTransforms,
      bundleNew.linkTransforms,
      // reference pose
      bundle.jointTransforms,
      bundle.linkTransforms,
      flatJacobian);

  // Flat Jacobian FD
  RowMatrix<real> flatJacobianFD;
  FlatJacobianFD(bundle, deltaDof, flatJacobianFD);

  // Compare
  real tol = sizeof(real) == sizeof(float) ? 1e-1_r : 1e-3_r;
  for (int i = 0; i < bundle.props.fullDofsDim; i++) {
    for (int j = 0; j < bundle.props.reducedDofsDim; j++) {
      EXPECT_NEAR_TOL(flatJacobian(i, j), flatJacobianFD(i, j), tol);
    }
  }
}

// Flat jacobian and normal jacobian should be the same at Zero
TEST(ArticulatedBody, FlatJacobian_Zero) {
  ArticulatedBodyBundle bundle;
  CreateComplexBody(bundle);

  // Flat Jacobian
  RowMatrix<real> flatJacobian(bundle.props.fullDofsDim, bundle.props.reducedDofsDim);
  FlatJacobian(
      bundle.jointTypes,
      bundle.jointAxes,
      bundle.dofInfo,
      bundle.poseInfo,
      bundle.parents,
      bundle.restTransforms,
      bundle.worldFromRoot,
      // new pose (at zero, they are just the same)
      bundle.jointTransforms,
      bundle.linkTransforms,
      // reference pose
      bundle.jointTransforms,
      bundle.linkTransforms,
      flatJacobian);

  // Normal Jacobian
  RowMatrix<real> jacobian(bundle.props.fullDofsDim, bundle.props.reducedDofsDim);
  Jacobian(
      bundle.jointTypes,
      bundle.parents,
      bundle.jointAxes,
      bundle.dofInfo,
      bundle.restTransforms,
      bundle.worldFromRoot,
      bundle.jointTransforms,
      bundle.linkTransforms,
      jacobian);

  // Compare
  real tol = sizeof(real) == sizeof(float) ? 1e-2_r : 1e-3_r;
  for (int i = 0; i < bundle.props.fullDofsDim; i++) {
    for (int j = 0; j < bundle.props.reducedDofsDim; j++) {
      EXPECT_NEAR_TOL(flatJacobian(i, j), jacobian(i, j), tol);
    }
  }
}

// Hessian is the finite difference of FlatJacobian, which should be symmetric in last two indices
// And the Hessian should match finite difference of FlatJacobian
TEST(ArticulatedBody, Hessian_Consistency) {
  ArticulatedBodyBundle bundle;
  CreateComplexBody(bundle);

  // HessianFD
  ArticulatedHessian hessianFD(
      bundle.props.reducedDofsDim,
      RowMatrix<real>(bundle.props.fullDofsDim, bundle.props.reducedDofsDim));
  HessianFD(bundle, hessianFD);

  // Hessian
  RowMatrix<real> jacobian(bundle.props.fullDofsDim, bundle.props.reducedDofsDim);
  ArticulatedHessian hessian(
      bundle.props.reducedDofsDim,
      RowMatrix<real>(bundle.props.fullDofsDim, bundle.props.reducedDofsDim));
  Jacobian(
      bundle.jointTypes,
      bundle.parents,
      bundle.jointAxes,
      bundle.dofInfo,
      bundle.restTransforms,
      bundle.worldFromRoot,
      bundle.jointTransforms,
      bundle.linkTransforms,
      jacobian);
  Hessian(
      bundle.dofInfo,
      bundle.jointAxes,
      bundle.parents,
      bundle.restTransforms,
      bundle.worldFromRoot,
      bundle.jointTransforms,
      bundle.linkTransforms,
      jacobian,
      hessian);

  real tol = sizeof(real) == sizeof(float) ? 1e-1_r : 1e-3_r;
  for (int d = 0; d < bundle.props.fullDofsDim; d++) {
    for (int i = 0; i < bundle.props.reducedDofsDim; i++) {
      for (int j = 0; j < i; j++) {
        EXPECT_NEAR_TOL(hessianFD[j](d, i), hessianFD[j](d, i), tol);
      }
    }
  }

  for (int d = 0; d < bundle.props.fullDofsDim; d++) {
    for (int i = 0; i < bundle.props.reducedDofsDim; i++) {
      for (int j = 0; j < bundle.props.reducedDofsDim; j++) {
        EXPECT_NEAR_TOL(hessianFD[j](d, i), hessian[j](d, i), tol);
      }
    }
  }
}

// Check the contracted Hessian using two methods
TEST(ArticulatedBody, Hessian_Contract) {
  ArticulatedBodyBundle bundle;
  CreateComplexBody(bundle);

  // Hessian
  RowMatrix<real> jacobian(bundle.props.fullDofsDim, bundle.props.reducedDofsDim);
  ArticulatedHessian hessian(
      bundle.props.reducedDofsDim,
      RowMatrix<real>(bundle.props.fullDofsDim, bundle.props.reducedDofsDim));
  Jacobian(
      bundle.jointTypes,
      bundle.parents,
      bundle.jointAxes,
      bundle.dofInfo,
      bundle.restTransforms,
      bundle.worldFromRoot,
      bundle.jointTransforms,
      bundle.linkTransforms,
      jacobian);
  Hessian(
      bundle.dofInfo,
      bundle.jointAxes,
      bundle.parents,
      bundle.restTransforms,
      bundle.worldFromRoot,
      bundle.jointTransforms,
      bundle.linkTransforms,
      jacobian,
      hessian);

  // Randomize contract data
  ColumnVector<real> contractedVector(bundle.props.numLinks * RigidSize::kDAll);
  RowMatrix<real> hessianContractedBF(bundle.props.reducedDofsDim, bundle.props.reducedDofsDim);
  contractedVector.SetRandom(123, -1_r, 1_r);
  hessianContractedBF.SetRandom(456, -1_r, 1_r);

  // Contract brute-force
  RowMatrix<real> hessianContracted = hessianContractedBF.Duplicate();
  for (int d = 0; d < bundle.props.numLinks * RigidSize::kDAll; d++) {
    for (int i = 0; i < bundle.props.reducedDofsDim; i++) {
      for (int j = 0; j < bundle.props.reducedDofsDim; j++) {
        hessianContractedBF(i, j) += hessian[j](d, i) * contractedVector[d];
      }
    }
  }

  // Contract using adjoint method
  HessianContract(
      bundle.dofInfo,
      bundle.jointAxes,
      bundle.parents,
      bundle.restTransforms,
      bundle.worldFromRoot,
      bundle.jointTransforms,
      bundle.linkTransforms,
      contractedVector,
      jacobian,
      hessianContracted,
      ColumnVectorView<real>());

  // Compare
  real tol = sizeof(real) == sizeof(float) ? 1e-1_r : 1e-3_r;
  for (int i = 0; i < bundle.props.reducedDofsDim; i++) {
    for (int j = 0; j < bundle.props.reducedDofsDim; j++) {
      EXPECT_NEAR_TOL(hessianContractedBF(i, j), hessianContracted(i, j), tol);
    }
  }
}

// Check the contracted Gradient using two methods
TEST(ArticulatedBody, Fast_Gradient_Contract) {
  ArticulatedBodyBundle bundle;
  CreateComplexBody(bundle);

  // Hessian
  RowMatrix<real> jacobian(bundle.props.fullDofsDim, bundle.props.reducedDofsDim);
  Jacobian(
      bundle.jointTypes,
      bundle.parents,
      bundle.jointAxes,
      bundle.dofInfo,
      bundle.restTransforms,
      bundle.worldFromRoot,
      bundle.jointTransforms,
      bundle.linkTransforms,
      jacobian);

  // Randomize contract data
  ColumnVector<real> contractedVector(bundle.props.numLinks * RigidSize::kDAll);
  ColumnVector<real> gradientContractedBF(bundle.props.reducedDofsDim);
  SetRandom(bundle.generator, -1_r, 1_r, contractedVector.GetSpan());
  SetRandom(bundle.generator, -1_r, 1_r, gradientContractedBF.GetSpan());

  // Contract brute-force
  ColumnVector<real> gradientContracted = gradientContractedBF.Duplicate();
  for (int d = 0; d < bundle.props.numLinks * RigidSize::kDAll; d++) {
    for (int i = 0; i < bundle.props.reducedDofsDim; i++) {
      gradientContractedBF[i] += jacobian(d, i) * contractedVector[d];
    }
  }

  // Contract using adjoint method
  HessianContract(
      bundle.dofInfo,
      bundle.jointAxes,
      bundle.parents,
      bundle.restTransforms,
      bundle.worldFromRoot,
      bundle.jointTransforms,
      bundle.linkTransforms,
      contractedVector,
      jacobian,
      RowMatrixView<real>(),
      gradientContracted);

  // Compare
  real tol = sizeof(real) == sizeof(float) ? 1e-1_r : 1e-3_r;
  for (int i = 0; i < bundle.props.reducedDofsDim; i++) {
    EXPECT_NEAR_TOL(gradientContractedBF[i], gradientContracted[i], tol);
  }
}
