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

#include <mochi_core/articulated_body/articulated_body.h>
#include <mochi_core/articulated_body/articulated_body_hessian.h>

#include "config.h"

using namespace mochi;
using namespace mochi::articulated;

namespace {
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
  mochi_default_random_engine generator;

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
    Real3 t = RandomVector();
    Quaternion r = Quaternion(
        mochi::RandomUniformValue(generator, -1_r, 1_r),
        mochi::RandomUniformValue(generator, -1_r, 1_r),
        mochi::RandomUniformValue(generator, -1_r, 1_r),
        mochi::RandomUniformValue(generator, -1_r, 1_r));
    r = Normalize(r);
    return TransformRT{r, t};
  }

  void AddJoint(int parent, ArticulatedJointType type) {
    parents.push_back(parent);
    // create random axis
    Real3 axis = Normalize(RandomVector());
    // Store the per-joint type and axis (axis only meaningful for single-DoF joints).
    jointTypes.push_back(type);
    if (type == ArticulatedJointType::Prismatic || type == ArticulatedJointType::Revolute) {
      jointAxes.push_back(axis);
    } else {
      jointAxes.emplace_back();
    }
    // Recompute the dof/pose layout for the whole chain.
    dofInfo = SetupJointDofs(jointTypes);
    poseInfo = SetupJointPose(jointTypes);
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
};
} // namespace

namespace mochi_benchmark {

template <int n>
static void JacobianVectorContract(benchmark::State& state) {
  ArticulatedBodyBundle bundle;
  for (int i = 0; i < n; i++) {
    bundle.AddJoint(i - 1, ArticulatedJointType::Spherical);
  }

  // Compute Jacobian
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

  // Fast J^T*v
  ColumnVector<real> contractedVector(bundle.props.numLinks * RigidSize::kDAll);
  ColumnVector<real> gradientContracted(bundle.props.reducedDofsDim);
  for (auto x : state) {
    gradientContracted = Transpose(jacobian) * contractedVector;
    MOCHI_NO_DISCARD_IN_LOOP(gradientContracted);
  }
  benchmark::DoNotOptimize(gradientContracted);
}

template <int n>
static void JacobianVectorContractFast(benchmark::State& state) {
  ArticulatedBodyBundle bundle;
  for (int i = 0; i < n; i++) {
    bundle.AddJoint(i - 1, ArticulatedJointType::Spherical);
  }

  // Compute Jacobian
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

  // Fast J^T*v
  ColumnVector<real> contractedVector(bundle.props.numLinks * RigidSize::kDAll);
  ColumnVector<real> gradientContracted(bundle.props.reducedDofsDim);
  for (auto x : state) {
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
    MOCHI_NO_DISCARD_IN_LOOP(gradientContracted);
  }
  benchmark::DoNotOptimize(gradientContracted);
}

BENCHMARK_TEMPLATE(JacobianVectorContract, 10);
BENCHMARK_TEMPLATE(JacobianVectorContractFast, 10);
BENCHMARK_TEMPLATE(JacobianVectorContract, 20);
BENCHMARK_TEMPLATE(JacobianVectorContractFast, 20);
BENCHMARK_TEMPLATE(JacobianVectorContract, 30);
BENCHMARK_TEMPLATE(JacobianVectorContractFast, 30);
BENCHMARK_TEMPLATE(JacobianVectorContract, 40);
BENCHMARK_TEMPLATE(JacobianVectorContractFast, 40);
BENCHMARK_TEMPLATE(JacobianVectorContract, 50);
BENCHMARK_TEMPLATE(JacobianVectorContractFast, 50);
} // namespace mochi_benchmark
