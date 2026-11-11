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

#include <limits>

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

  ColumnVector<real> RandomColumnVector(int size) {
    ColumnVector<real> result(size);
    SetRandom(generator, -1_r, 1_r, result.GetSpan());
    return result;
  }

  ColumnVector<real> GetRandomDof() {
    return RandomColumnVector(props.reducedDofsDim);
  }

  ArticulatedJacobian ComputeCompactJacobian() const {
    ArticulatedJacobian result(parents, dofInfo);
    result.Update(jointAxes, restTransforms, worldFromRoot, jointTransforms, linkTransforms);
    return result;
  }

  // NaN-filled first, so that the tests check that ToDense overwrites every entry.
  RowMatrix<real> ComputeJacobian() const {
    RowMatrix<real> result(props.fullDofsDim, props.reducedDofsDim);
    result.SetConstant(std::numeric_limits<real>::quiet_NaN());
    ComputeCompactJacobian().ToDense(result);
    return result;
  }

  ArticulatedHessian ComputeHessian(RowMatrixView<real const> jacobian) const {
    ArticulatedHessian result(
        props.reducedDofsDim, RowMatrix<real>(props.fullDofsDim, props.reducedDofsDim));
    Hessian(
        dofInfo,
        jointAxes,
        parents,
        restTransforms,
        worldFromRoot,
        jointTransforms,
        linkTransforms,
        jacobian,
        result);
    return result;
  }

  RowMatrix<real> ContractHessian(
      Span<real const> contractedVector,
      RowMatrixView<real const> jacobian) const {
    RowMatrix<real> result = RowMatrix<real>::Zero(props.reducedDofsDim, props.reducedDofsDim);
    HessianContract(
        dofInfo,
        jointAxes,
        parents,
        restTransforms,
        worldFromRoot,
        jointTransforms,
        linkTransforms,
        contractedVector,
        jacobian,
        result);
    return result;
  }

  void AddHessianDoubleContract(
      Span<real const> contractedVector,
      RowMatrixView<real const> jacobian,
      Span<real const> reducedVector,
      ColumnVectorView<real> out) const {
    HessianDoubleContract(
        dofInfo,
        jointAxes,
        parents,
        restTransforms,
        worldFromRoot,
        jointTransforms,
        linkTransforms,
        contractedVector,
        jacobian,
        reducedVector,
        out);
  }

  void AddJacobianDerivativeDoubleContract(
      Span<real const> contractedVector,
      RowMatrixView<real const> jacobian,
      Span<real const> reducedVector,
      ColumnVectorView<real> out) const {
    JacobianDerivativeDoubleContract(
        dofInfo,
        jointAxes,
        parents,
        restTransforms,
        worldFromRoot,
        jointTransforms,
        linkTransforms,
        contractedVector,
        jacobian,
        reducedVector,
        out);
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

static void ExpectVectorsNear(
    ColumnVectorView<real const> expected,
    ColumnVectorView<real const> actual,
    real tolerance) {
  ASSERT_EQ(expected.Rows(), actual.Rows());
  for (int i = 0; i < expected.Rows(); ++i) {
    EXPECT_NEAR_TOL(expected[i], actual[i], tolerance);
  }
}

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
  ArticulatedJacobian jacobian(parents, dofInfo);
  jacobian.Update(jointAxes, restTransforms, worldFromRoot, jointTransforms, linkTransforms);
  jacobian.ToDense(outFlatJacobian);
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
  // Second root, without DoFs
  bundle.AddJoint(-1, ArticulatedJointType::Hard);
  bundle.AddJoint(11, ArticulatedJointType::Revolute);
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
  RowMatrix<real> jacobian = bundle.ComputeJacobian();

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

  RowMatrix<real> jacobian = bundle.ComputeJacobian();
  ArticulatedHessian hessian = bundle.ComputeHessian(jacobian);

  real tol = sizeof(real) == sizeof(float) ? 1e-1_r : 1e-3_r;
  for (int d = 0; d < bundle.props.fullDofsDim; d++) {
    for (int i = 0; i < bundle.props.reducedDofsDim; i++) {
      for (int j = 0; j < i; j++) {
        EXPECT_NEAR_TOL(hessianFD[j](d, i), hessianFD[i](d, j), tol);
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
  RowMatrix<real> jacobian = bundle.ComputeJacobian();
  ArticulatedHessian hessian = bundle.ComputeHessian(jacobian);

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
      hessianContracted);

  // Compare
  real tol = sizeof(real) == sizeof(float) ? 1e-1_r : 1e-3_r;
  for (int i = 0; i < bundle.props.reducedDofsDim; i++) {
    for (int j = 0; j < bundle.props.reducedDofsDim; j++) {
      EXPECT_NEAR_TOL(hessianContractedBF(i, j), hessianContracted(i, j), tol);
    }
  }
}

TEST(ArticulatedBody, Hessian_DoubleContract) {
  ArticulatedBodyBundle bundle;
  CreateComplexBody(bundle);

  RowMatrix<real> jacobian = bundle.ComputeJacobian();
  ArticulatedHessian hessian = bundle.ComputeHessian(jacobian);

  ColumnVector<real> contractedVector = bundle.RandomColumnVector(bundle.props.fullDofsDim);
  ColumnVector<real> reducedVector = bundle.RandomColumnVector(bundle.props.reducedDofsDim);
  ColumnVector<real> expected = bundle.RandomColumnVector(bundle.props.reducedDofsDim);
  ColumnVector<real> seed = expected.Duplicate();
  ColumnVector<real> result = seed.Duplicate();

  for (int d = 0; d < bundle.props.fullDofsDim; ++d) {
    for (int i = 0; i < bundle.props.reducedDofsDim; ++i) {
      for (int j = 0; j < bundle.props.reducedDofsDim; ++j) {
        expected[j] += contractedVector[d] * hessian[j](d, i) * reducedVector[i];
      }
    }
  }

  bundle.AddHessianDoubleContract(contractedVector, jacobian, reducedVector, result);

  RowMatrix<real> hessianContracted = bundle.ContractHessian(contractedVector, jacobian);
  ColumnVector<real> fromMatrix = hessianContracted.Transpose() * reducedVector;

  real const bruteForceTolerance = sizeof(real) == sizeof(float) ? 1e-1_r : 1e-3_r;
  real const sameKernelTolerance = sizeof(real) == sizeof(float) ? 1e-5_r : 1e-12_r;
  for (int i = 0; i < bundle.props.reducedDofsDim; ++i) {
    EXPECT_NEAR_TOL(expected[i], result[i], bruteForceTolerance);
    EXPECT_NEAR_RTOL(result[i] - seed[i], fromMatrix[i], sameKernelTolerance);
  }
}

TEST(ArticulatedBody, JacobianDerivative_DoubleContract) {
  ArticulatedBodyBundle bundle;
  CreateComplexBody(bundle);

  RowMatrix<real> jacobian = bundle.ComputeJacobian();

  ColumnVector<real> contractedVector = bundle.RandomColumnVector(bundle.props.fullDofsDim);
  ColumnVector<real> reducedVector = bundle.RandomColumnVector(bundle.props.reducedDofsDim);
  ColumnVector<real> expected = bundle.RandomColumnVector(bundle.props.reducedDofsDim);
  ColumnVector<real> result = expected.Duplicate();

  RowMatrix<real> hessianContracted = bundle.ContractHessian(contractedVector, jacobian);
  expected += hessianContracted.Transpose() * reducedVector;
  ColumnVector<real> reducedGradient = jacobian.Transpose() * contractedVector;

  ColumnVector<real> linkVelocity = jacobian * reducedVector;
  ColumnVector<real> outputTransportVector(bundle.props.fullDofsDim);
  outputTransportVector.SetZero();
  for (int link = 0; link < bundle.props.numLinks; ++link) {
    int const offset = link * RigidSize::kDAll + RigidSize::kDTrans;
    Real3 const omega{linkVelocity[offset], linkVelocity[offset + 1], linkVelocity[offset + 2]};
    Real3 const lambda{
        contractedVector[offset], contractedVector[offset + 1], contractedVector[offset + 2]};
    Real3 const correction = 0.5_r * Cross(omega, lambda);
    outputTransportVector.MiddleRows(offset, RigidSize::kDRot) = AsView(correction);
  }
  expected += jacobian.Transpose() * outputTransportVector;

  for (auto const& jointDofs : bundle.dofInfo) {
    if (jointDofs.rotSize != RigidSize::kDRot) {
      continue;
    }
    int const offset = jointDofs.GetRotOffset();
    Real3 const velocity{
        reducedVector[offset], reducedVector[offset + 1], reducedVector[offset + 2]};
    Real3 const gradient{
        reducedGradient[offset], reducedGradient[offset + 1], reducedGradient[offset + 2]};
    Real3 const correction = 0.5_r * Cross(velocity, gradient);
    expected.MiddleRows(offset, RigidSize::kDRot) -= AsView(correction);
  }

  bundle.AddJacobianDerivativeDoubleContract(contractedVector, jacobian, reducedVector, result);

  real const tol = sizeof(real) == sizeof(float) ? 1e-2_r : 1e-6_r;
  ExpectVectorsNear(expected, result, tol);
}

TEST(ArticulatedBody, JacobianDerivative_DoubleContract_FiniteDifference) {
  ArticulatedBodyBundle bundle;
  CreateComplexBody(bundle);

  RowMatrix<real> jacobian = bundle.ComputeJacobian();

  ColumnVector<real> contractedVector = bundle.RandomColumnVector(bundle.props.fullDofsDim);
  ColumnVector<real> reducedVector = bundle.RandomColumnVector(bundle.props.reducedDofsDim);
  ColumnVector<real> result = bundle.RandomColumnVector(bundle.props.reducedDofsDim);
  ColumnVector<real> expected = result.Duplicate();

  bundle.AddJacobianDerivativeDoubleContract(contractedVector, jacobian, reducedVector, result);

  real const eps = sizeof(real) == sizeof(float) ? 1e-3_r : 1e-5_r;
  for (int j = 0; j < bundle.props.reducedDofsDim; ++j) {
    ColumnVector<real> deltaPositive(bundle.props.reducedDofsDim);
    ColumnVector<real> deltaNegative(bundle.props.reducedDofsDim);
    deltaPositive.SetZero();
    deltaNegative.SetZero();
    deltaPositive[j] = eps;
    deltaNegative[j] = -eps;

    ArticulatedBodyBundle bundlePositive = bundle;
    ArticulatedBodyBundle bundleNegative = bundle;
    bundlePositive.ApplyRandomDeltaPose(deltaPositive);
    bundleNegative.ApplyRandomDeltaPose(deltaNegative);

    RowMatrix<real> jacobianPositive = bundlePositive.ComputeJacobian();
    RowMatrix<real> jacobianNegative = bundleNegative.ComputeJacobian();

    RowMatrix<real> jacobianDerivative = jacobianPositive - jacobianNegative;
    jacobianDerivative /= 2_r * eps;
    ColumnVector<real> velocityDerivative = jacobianDerivative * reducedVector;
    for (int d = 0; d < bundle.props.fullDofsDim; ++d) {
      expected[j] += contractedVector[d] * velocityDerivative[d];
    }
  }

  real const tol = sizeof(real) == sizeof(float) ? 1e-2_r : 1e-5_r;
  ExpectVectorsNear(expected, result, tol);
}

TEST(ArticulatedBody, Jacobian_TransposeContract) {
  ArticulatedBodyBundle bundle;
  CreateComplexBody(bundle);

  ArticulatedJacobian const compactJacobian = bundle.ComputeCompactJacobian();
  RowMatrix<real> jacobian = bundle.ComputeJacobian();

  // Randomize contract data
  ColumnVector<real> contractedVector = bundle.RandomColumnVector(bundle.props.fullDofsDim);
  ColumnVector<real> gradientContractedBF = bundle.RandomColumnVector(bundle.props.reducedDofsDim);

  // Contract brute-force
  ColumnVector<real> gradientContracted = gradientContractedBF.Duplicate();
  ColumnVector<real> gradientContractedByLink = gradientContractedBF.Duplicate();
  for (int d = 0; d < bundle.props.numLinks * RigidSize::kDAll; d++) {
    for (int i = 0; i < bundle.props.reducedDofsDim; i++) {
      gradientContractedBF[i] += jacobian(d, i) * contractedVector[d];
    }
  }

  // Contract using the articulated reverse pass, and link by link.
  compactJacobian.AddTransposeProduct(contractedVector, gradientContracted);
  for (int link = 0; link < bundle.props.numLinks; ++link) {
    compactJacobian.AddLinkTransposeProduct(
        link,
        contractedVector.Slice<RigidSize::kDAll>(RigidSize::kDAll * link, RigidSize::kDAll),
        gradientContractedByLink);
  }

  // Compare
  real tol = sizeof(real) == sizeof(float) ? 1e-1_r : 1e-3_r;
  ExpectVectorsNear(gradientContractedBF, gradientContracted, tol);
  ExpectVectorsNear(gradientContractedBF, gradientContractedByLink, tol);

  // Apply overwrites its output, which starts as NaN.
  ColumnVector<real> const reducedVector = bundle.RandomColumnVector(bundle.props.reducedDofsDim);
  ColumnVector<real> product(bundle.props.fullDofsDim);
  product.SetConstant(std::numeric_limits<real>::quiet_NaN());
  compactJacobian.Apply(reducedVector, product);
  ExpectVectorsNear(ColumnVector<real>(jacobian * reducedVector), product, tol);
}

TEST(ArticulatedBody, Jacobian_TransposeContract_SubtreeGradient) {
  ArticulatedBodyBundle bundle;
  CreateComplexBody(bundle);
  int const numLinks = bundle.props.numLinks;

  ArticulatedJacobian const jacobian = bundle.ComputeCompactJacobian();
  ColumnVector<real> fullGradient = bundle.RandomColumnVector(bundle.props.fullDofsDim);

  ColumnVector<real> expectedReduced(bundle.props.reducedDofsDim);
  expectedReduced.SetZero();
  jacobian.AddTransposeProduct(fullGradient, expectedReduced);

  ColumnVector<real> reduced(bundle.props.reducedDofsDim);
  reduced.SetZero();
  ColumnVector<real> subtreeGradient(bundle.props.fullDofsDim);
  jacobian.AddTransposeProduct(fullGradient, reduced, subtreeGradient);
  ExpectVectorsNear(expectedReduced, reduced, 0_r);

  // Brute force: sum of [force; torque] of every descendant, with torque moved to the link.
  auto isInSubtree = [&](int link, int root) {
    for (; link >= 0; link = bundle.parents[link]) {
      if (link == root) {
        return true;
      }
    }
    return false;
  };
  auto block = [&](ColumnVector<real> const& v, int link, int offset) {
    int const i = link * RigidSize::kDAll + offset;
    return Real3{v[i], v[i + 1], v[i + 2]};
  };
  ColumnVector<real> expectedSubtree(bundle.props.fullDofsDim);
  for (int root = 0; root < numLinks; ++root) {
    Real3 force{};
    Real3 torque{};
    Real3 const rootPosition = bundle.linkTransforms[root].GetTranslation();
    for (int link = 0; link < numLinks; ++link) {
      if (!isInSubtree(link, root)) {
        continue;
      }
      Real3 const linkForce = block(fullGradient, link, 0);
      Real3 const lever = bundle.linkTransforms[link].GetTranslation() - rootPosition;
      force += linkForce;
      torque += block(fullGradient, link, RigidSize::kDTrans) + Cross(lever, linkForce);
    }
    expectedSubtree.MiddleRows(root * RigidSize::kDAll, RigidSize::kDTrans) = AsView(force);
    expectedSubtree.MiddleRows(root * RigidSize::kDAll + RigidSize::kDTrans, RigidSize::kDRot) =
        AsView(torque);
  }

  real constexpr kTol = MOCHI_USE_DOUBLE_PRECISION ? 1e-10_r : 1e-3_r;
  ExpectVectorsNear(expectedSubtree, subtreeGradient, kTol);
}
