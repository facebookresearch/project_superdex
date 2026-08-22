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

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dskinning.h>
#include <mochi_core/utils/dtransform.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <tuple>
#include <utility>
#include <vector>

using namespace mochi;
using namespace krylov;

static TransformRT SampleTransform1() {
  auto quat = Quaternion::FromAxisAngle(Real3{0.5_r, 0.5_r, 0.5_r}, kPI / 4.0);
  auto trans = Real3{0.382683456_r, 0_r, 0.923879504_r};
  return TransformRT(quat, trans);
}

static TransformRT SampleTransform2() {
  auto quat = Quaternion::FromAxisAngle(Real3{0.0_r, 0.5_r, 0.0_r}, kPI / 2.0);
  auto trans = Real3{-1.0_r, 0_r, 2.0_r};
  return TransformRT(quat, trans);
}

static TransformRT SampleTransform3() {
  auto quat = Quaternion::FromAxisAngle(Real3{0.0_r, 0.5_r, 0.5_r}, kPI / 3.0);
  auto trans = Real3{-1.0_r, 2.0_r, 2.0_r};
  return TransformRT(quat, trans);
}

// Creates an example with three bones and three vertices
// Each vertex is attached to only one bone
static std::tuple<SkinningWeightsByBone, std::vector<TransformRT>, ColumnVector<real>>
SampleOneBonePerVertex() {
  auto trans = std::vector<TransformRT>{SampleTransform1(), SampleTransform2(), SampleTransform3()};

  std::vector<int> skinningIdx = {0, 1, 2};
  std::vector<real> skinningWeights = {1.0_r, 1.0_r, 1.0_r};

  constexpr size_t kWeightsPerNode = 1;

  ColumnVector<real> inputVec(3 * 3);

  Real3 x1{1.0_r, 0.0_r, 0.0_r};
  Real3 x2{0.0_r, 1.0_r, 0.0_r};
  Real3 x3{0.0_r, 0.0_r, 1.0_r};

  inputVec.Slice(0, 3) = AsConstView(MakeSpan(x1));
  inputVec.Slice(3, 3) = AsConstView(MakeSpan(x2));
  inputVec.Slice(6, 3) = AsConstView(MakeSpan(x3));

  return {
      SkinningWeightsByBone(skinningIdx, skinningWeights, kWeightsPerNode, trans.size()),
      std::move(trans),
      inputVec};
}

// Creates an example with two vertices attached to two different bones.
// The skinning weights differ between the two vertices
static std::tuple<SkinningWeightsByBone, std::vector<TransformRT>, ColumnVector<real>>
SampleTwoVerticesTwoBones() {
  auto trans = std::vector<TransformRT>{SampleTransform1(), SampleTransform2()};

  std::vector<int> skinningIdx = {0, 1, 0, 1};
  std::vector<real> skinningWeights = {0.25_r, 0.75_r, 0.75_r, 0.25_r};

  constexpr size_t kWeightsPerNode = 2;
  constexpr int kDim = kDSkinningDofsPerVertex;

  ColumnVector<real> inputVec(kDim * 2);

  Real3 x1{1.0_r, 0.0_r, 0.0_r};
  Real3 x2{0.0_r, 1.0_r, 0.0_r};

  inputVec.Slice(0, 3) = AsConstView(MakeSpan(x1));
  inputVec.Slice(3, 3) = AsConstView(MakeSpan(x2));

  return {
      SkinningWeightsByBone(skinningIdx, skinningWeights, kWeightsPerNode, trans.size()),
      std::move(trans),
      inputVec};
}

// Tests to see if the output of the differentiable skinning transform matches
// the output of TransformRT.
TEST(DSkinning, CompareTransformToDRT) {
  auto [weights, trans, inputVec] = SampleOneBonePerVertex();

  auto dskinning = DSkinningTransform(weights);

  constexpr int kDim = kDSkinningDofsPerVertex;

  int const numDofs = isize(trans) * kDim;
  ColumnVector<real> outVecTest(numDofs);
  ColumnVector<real> outVecTruth(numDofs);

  RowMatrix<real, krylov::kDynamic, kDim> inJacobian(numDofs, kDim);
  for (int i = 0; i < kDim; ++i) {
    inJacobian.Col(i) = real(i + 1) * inputVec;
  }
  RowMatrix<real, krylov::kDynamic, kDim> outJacobianTest(numDofs, kDim);
  RowMatrix<real> outJacobianTruth(numDofs, kDim);
  RowMatrix<real> outDParamsTruth(numDofs, numDofs);

  dskinning.Transform(MakeConstSpan(trans), AsConstView(inputVec), AsView(outVecTest));
  dskinning.DTransform(MakeConstSpan(trans), AsConstView(inJacobian), AsView(outJacobianTest));
  auto dOutParamsTest = dskinning.CreateDBones();
  dskinning.DTransformDBones(MakeConstSpan(trans), AsConstView(inputVec), dOutParamsTest);

  for (int input = 0; input < trans.size(); ++input) {
    TransformBatch(
        trans[input], inputVec.Slice(input * kDim, kDim), outVecTruth.Slice(input * kDim, kDim));
    DTransformBatch(
        trans[input],
        AsConstView(inJacobian.MiddleRows(input * kDim, kDim)),
        outJacobianTruth.MiddleRows(input * kDim, kDim));
    DTransformDParametersBatch(
        trans[input],
        inputVec.Slice(input * kDim, kDim),
        outDParamsTruth.Block(input * kDim, 0, kDim, RigidSize::kDAll));
  }

  ColumnVector<real> diffTransform = outVecTest - outVecTruth;
  EXPECT_NEAR_EQ(diffTransform.Norm(), 0_r);

  RowMatrix<real> diffDTransform = outJacobianTest - outJacobianTruth;
  EXPECT_NEAR_EQ(diffDTransform.Norm(), 0_r);

  // Output dparams should be block diagonal
  for (int row = 0; row < dOutParamsTest.Rows(); ++row) {
    auto vals = dOutParamsTest.Values(row);
    auto idx = dOutParamsTest.Indices(row);

    EXPECT_EQ(vals.size(), RigidSize::kDAll);
    EXPECT_EQ(idx.size(), RigidSize::kDAll);

    for (int param = 0; param < RigidSize::kDAll; ++param) {
      int vertex = row / kDim;
      EXPECT_NEAR_EQ(vals[param], outDParamsTruth(row, param));
      EXPECT_EQ(idx[param], vertex * RigidSize::kDAll + param);
    }
  }
}

using dfunc_t = std::function<ColumnVector<real>(ColumnVectorView<real const>)>;

// Tests the derivative of a function in a specific direction
static void TestDerivative(
    dfunc_t const& func,
    ColumnVectorView<real const> x,
    ColumnVectorView<real const> dir,
    real h,
    ColumnVectorView<real const> derivativeInDir) {
  ColumnVector<real> test_mh = x - h * dir;
  ColumnVector<real> test_h = x + h * dir;

  auto ymh = func(AsConstView(test_mh));
  auto yh = func(AsConstView(test_h));

  ColumnVector<real> fd_dydh = (yh - ymh);
  fd_dydh /= (2.0_r * h);

  ColumnVector<real> diff = fd_dydh - derivativeInDir;
  EXPECT_TRUE(mochi::NearEqual(diff.Norm(), 0.0_r, 0.001_r * fd_dydh.Norm()));
}

// Tests the Lie derivative of a function in a specific direction
static void TestLieDerivative(
    dfunc_t const& func,
    ColumnVectorView<real const> x,
    ColumnVectorView<real const> dir,
    real h,
    ColumnVectorView<real const> derivativeInDir) {
  MOCHI_ASSERT_VERBOSE(dir.Rows() % RigidSize::kDAll == 0);

  auto addEps = [&](real h) {
    ColumnVector<real> xeval = x.Duplicate();
    for (int i = 0; i < dir.Rows() / RigidSize::kDAll; ++i) {
      int const offset = i * RigidSize::kDAll;
      Real3 trans = {x[offset], x[offset + 1], x[offset + 2]};
      Real3 rot = {x[offset + 3], x[offset + 4], x[offset + 5]};
      Real3 transDelta = h * Real3{dir[offset], dir[offset + 1], dir[offset + 2]};
      Real3 rotDelta = h * Real3{dir[offset + 3], dir[offset + 4], dir[offset + 5]};
      trans += transDelta;
      rot = (Quaternion::FromRotationVector(rotDelta) * Quaternion::FromRotationVector(rot))
                .ToRotationVector();
      xeval[offset] = trans[0];
      xeval[offset + 1] = trans[1];
      xeval[offset + 2] = trans[2];
      xeval[offset + 3] = rot[0];
      xeval[offset + 4] = rot[1];
      xeval[offset + 5] = rot[2];
    }
    return xeval;
  };
  ColumnVector<real> test_mh = addEps(-h);
  ColumnVector<real> test_h = addEps(h);

  auto ymh = func(AsConstView(test_mh));
  auto yh = func(AsConstView(test_h));

  ColumnVector<real> fd_dydh = (yh - ymh);
  fd_dydh /= (2.0_r * h);

  ColumnVector<real> diff = fd_dydh - derivativeInDir;
  EXPECT_TRUE(mochi::NearEqual(diff.Norm(), 0.0_r, 0.001_r * fd_dydh.Norm()));
}

// Checks if the derivative wrt the input of a skinning operation is consistent with
// a finite difference approximation.
TEST(DSkinning, ConsistencyTestDInput) {
  constexpr int kDim = kDSkinningDofsPerVertex;
  auto [weights, trans, inputVec] = SampleTwoVerticesTwoBones();

  auto transSpan = MakeConstSpan(trans);
  auto dskinning = DSkinningTransform(weights);

  dfunc_t func = [&](ColumnVectorView<real const> v) {
    ColumnVector<real> out(v.Rows());
    dskinning.Transform(transSpan, v, AsView(out));
    return out;
  };

  ColumnVector<real, 6> dirVec = {1_r, 2_r, 3_r, 4_r, 5_r, 6_r};
  RowMatrix<real, krylov::kDynamic, kDim> inJacobian(dirVec.Rows(), kDim);
  for (int i = 0; i < kDim; ++i) {
    inJacobian.Col(i) = real(i + 1) * dirVec;
  }
  RowMatrix<real, krylov::kDynamic, kDim> outJacobian(dirVec.Rows(), kDim);
  dskinning.DTransform(transSpan, AsConstView(inJacobian), AsView(outJacobian));
  for (int i = 0; i < kDim; ++i) {
    ColumnVector<real> outDerivTest = outJacobian.Col(i);
    ColumnVector<real> inDeriv = inJacobian.Col(i);
    TestDerivative(func, inputVec, inDeriv, 0.001_r, outDerivTest);
  }
}

// Checks that the sparse matrix of derivatives with respect to bone parameters
// is correct by manually checking the entries.
TEST(DSkinning, ValidateDBones) {
  auto [weights, trans, inputVec] = SampleTwoVerticesTwoBones();

  auto dskinning = DSkinningTransform(weights);

  auto dbones = dskinning.CreateDBones();

  dskinning.DTransformDBones(MakeConstSpan(trans), inputVec, dbones);

  constexpr int kDim = kDSkinningDofsPerVertex;
  for (int vertexId = 0; vertexId < dbones.Rows() / kDim; ++vertexId) {
    for (int bone = 0; bone < trans.size(); ++bone) {
      int start = bone * RigidSize::kDAll;
      int len = RigidSize::kDAll;

      auto vertexInput = inputVec.Slice(kDim * vertexId, kDim);
      RowMatrix<real> dboneTruth(kDim, RigidSize::kDAll);
      DTransformDParametersBatch(trans[bone], vertexInput, AsView(dboneTruth));

      for (int dim = 0; dim < kDim; ++dim) {
        int row = kDim * vertexId + dim;
        auto vals = dbones.Values(row);
        auto idx = dbones.Indices(row);

        EXPECT_EQ(vals.size(), RigidSize::kDAll * trans.size());
        EXPECT_EQ(idx.size(), RigidSize::kDAll * trans.size());

        auto valsSub = vals.subspan(start, len);
        auto idxSub = idx.subspan(start, len);

        auto beginBone = weights.BeginBone(bone);
        auto endBone = weights.EndBone(bone);

        auto boneVertexPair = std::find_if(
            beginBone, endBone, [&](auto const& pair) { return pair.vertexId == vertexId; });
        auto weight = boneVertexPair->weight;

        for (int param = 0; param < RigidSize::kDAll; ++param) {
          EXPECT_NEAR_EQ(valsSub[param], weight * dboneTruth(dim, param));
          EXPECT_EQ(idxSub[param], bone * RigidSize::kDAll + param);
        }
      }
    }
  }
}

// Checks that the derivative of a skinning operation with respect to bone transform parameters
// is consistent with finite different approximation.
TEST(DSkinning, ConsistencyTestDBones) {
  auto [weights, trans, inputVec] = SampleTwoVerticesTwoBones();

  DTransformParameterizationCollection parameterization;
  parameterization.resize(2);
  parameterization[0].preTransform = TransformSRT{
      0.5_r,
      Quaternion::FromRotationVector(Real3{0.5_r, 0.6_r, -1.1_r}),
      Real3{0.1_r, 0.2_r, 0.3_r}};
  parameterization[1].preTransform = TransformSRT{
      0.7_r,
      Quaternion::FromRotationVector(Real3{-0.8_r, -0.6_r, 0.6_r}),
      Real3{0.4_r, -0.6_r, -0.4_r}};
  parameterization[0].postTransform = TransformSRT{
      0.3_r,
      Quaternion::FromRotationVector(Real3{-0.5_r, 0.3_r, 1.1_r}),
      Real3{-0.1_r, 0.5_r, -0.3_r}};
  parameterization[1].postTransform = TransformSRT{
      0.8_r,
      Quaternion::FromRotationVector(Real3{-0.2_r, 0.9_r, -0.7_r}),
      Real3{-0.3_r, -0.8_r, -0.1_r}};

  auto dskinning = DSkinningTransform(weights, parameterization);

  ColumnVector<real> inputVecCopy = inputVec;
  auto transCount = trans.size();

  dfunc_t func = [&](ColumnVectorView<real const> params) {
    std::vector<TransformRT> inputTrans;

    inputTrans.reserve(transCount);
    for (int itrans = 0; itrans < transCount; ++itrans) {
      inputTrans.emplace_back(TransformFromRawDofs(
          params.Slice<RigidSize::kDAll>(itrans * RigidSize::kDAll, RigidSize::kDAll)));
    }

    ColumnVector<real> out(inputVecCopy.Rows());
    dskinning.Transform(MakeConstSpan(inputTrans), AsConstView(inputVecCopy), AsView(out));
    return out;
  };

  ColumnVector<real> initialParamsVec(static_cast<int>(RigidSize::kDAll * transCount));
  for (int itrans = 0, offset = 0; itrans < transCount; ++itrans, offset += RigidSize::kDAll) {
    auto paramsSlice = initialParamsVec.Slice<RigidSize::kDAll>(offset, RigidSize::kDAll);
    TransformToRawDofs(trans[itrans], paramsSlice);
  }

  ColumnVector<real, 2 * RigidSize::kDAll> dirVec{
      1.0_r, 2.0_r, 3.0_r, 4.0_r, 5.0_r, 6.0_r, 1.0_r, -0.5_r, 2.0_r, -3.0_r, 4.0_r, -2.0_r};

  auto dbones = dskinning.CreateDBones();
  dskinning.DTransformDBones(MakeConstSpan(trans), inputVec, dbones);

  ColumnVector<real> testVec(dbones.Rows());
  testVec = dbones * dirVec;

  TestLieDerivative(func, initialParamsVec, dirVec, 0.001_r, testVec);
}
