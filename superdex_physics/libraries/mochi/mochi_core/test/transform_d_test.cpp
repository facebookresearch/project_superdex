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

#include <mochi_core/test/linear_algebra_test_helpers.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dtransform.h>
#include <mochi_core/utils/nd_array_utils.h>

#include <gtest/gtest.h>

#include <array>
#include <functional>

using namespace mochi;
using namespace mochi::test;

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

static TransformSRT SampleTransformSRT1() {
  auto quat = Quaternion::FromAxisAngle(Real3{0.5_r, 0.5_r, 0.5_r}, kPI / 4.0);
  auto trans = Real3{0.382683456_r, 0_r, 0.923879504_r};
  auto scale = 0.5_r;
  return TransformSRT(scale, quat, trans);
}

static TransformSRT SampleTransformSRT2() {
  auto quat = Quaternion::FromAxisAngle(Real3{0.0_r, 0.5_r, 0.0_r}, kPI / 2.0);
  auto trans = Real3{-0.382683456_r, 0_r, 2.0_r};
  auto scale = 2.5_r;
  return TransformSRT(scale, quat, trans);
}

static Real3 SampleVector1() {
  return Real3{1_r, 2_r, 3_r};
}

static Real3 SampleVector2() {
  return Real3{3_r, 2_r, 1_r};
}

static Real3 TransformPrePost(
    TransformRT const& transform,
    Real3 input,
    TransformSRT const& preTransform,
    TransformSRT const& postTransform) {
  return postTransform.TransformPoint(transform.TransformPoint(preTransform.TransformPoint(input)));
}

static Real3 TransformPivot(TransformRT const& transform, Real3 const& input, Real3 const& pivot) {
  TransformSRT preTransform = TranslateSRT(-ToSimd(pivot));
  TransformSRT postTransform = TranslateSRT(ToSimd(pivot));
  return TransformPrePost(transform, input, preTransform, postTransform);
}

// Test to see if transforms agree on a batch input
TEST(DTransformRT, TestTransformBatch) {
  TransformRT rt(SampleTransform1());

  NdArray<real, 3, 3> vecs;
  vecs[0] = SampleVector1();
  vecs[1] = SampleVector2();
  vecs[2] = SampleVector1();
  NdArray<real, 3, 3> outDRT;

  TransformBatch(rt, AsView(Flatten(Span<Real3 const>(vecs))), AsView(Flatten(outDRT)));
  std::array<Real3, 3> outRT = {
      rt.TransformPoint(vecs[0]), rt.TransformPoint(vecs[1]), rt.TransformPoint(vecs[2])};

  EXPECT_NEAR_EQ(outDRT[0], outRT[0]);
  EXPECT_NEAR_EQ(outDRT[1], outRT[1]);
  EXPECT_NEAR_EQ(outDRT[2], outRT[2]);
}

// Compares a numerical directional derivative to a ground truth
template <size_t N, size_t M>
static void TestDerivative(
    std::function<NdArray<real, N>(NdArray<real, M>)> func,
    NdArray<real, M> const& x,
    NdArray<real, M> const& direction,
    real h,
    NdArray<real, N, M> const& derivative) {
  auto ymh = func(x - h * direction);
  auto yh = func(x + h * direction);
  auto fd_dydh = (yh - ymh) / (2.0_r * h);
  auto dydh = DotMatVec(derivative, direction);

  EXPECT_TRUE(mochi::NearEqual(fd_dydh, dydh, 0.001_r * Norm(fd_dydh)));
}

// Compares a numerical directional Lie derivative to a ground truth
template <size_t N, size_t M>
static void TestLieDerivative(
    std::function<NdArray<real, N>(NdArray<real, M>)> func,
    NdArray<real, M> const& x,
    NdArray<real, M> const& dir,
    real h,
    NdArray<real, N, M> const& derivative) {
  MOCHI_ASSERT_VERBOSE(M % RigidSize::kDAll == 0);

  auto addEps = [&](real h) {
    NdArray<real, M> xeval = x;
    for (int i = 0; i < M / RigidSize::kDAll; ++i) {
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

  auto ymh = func(addEps(-h));
  auto yh = func(addEps(h));
  auto fd_dydh = (yh - ymh) / (2.0_r * h);
  auto dydh = DotMatVec(derivative, dir);

  EXPECT_TRUE(mochi::NearEqual(fd_dydh, dydh, 0.001_r * Norm(fd_dydh)));
}

// Test TransformRT derivative in the input variables
static void TestDInputAt(
    TransformRT const& rt,
    Real3 input,
    TransformSRT preTransform = {},
    TransformSRT postTransform = {}) {
  real h = 0.001_r;
  Matrix<real> kry_doutput(RigidSize::kDim, RigidSize::kDim);
  auto kry_dinput = Matrix<real>::Zero(RigidSize::kDim, RigidSize::kDim);
  kry_dinput(0, 0) = 1.0_r;
  kry_dinput(1, 1) = 1.0_r;
  kry_dinput(2, 2) = 1.0_r;

  DTransformBatch(rt, AsConstView(kry_dinput), AsView(kry_doutput), preTransform, postTransform);
  auto doutput_dinput = ToNdArray<RigidSize::kDim, RigidSize::kDim>(kry_doutput);

  auto dirs = DiagonalMatrix<RigidSize::kDim, real>(1.0_r);

  std::function<Real3(Real3)> inputfunc = [rt, preTransform, postTransform](Real3 in) {
    return TransformPrePost(rt, in, preTransform, postTransform);
  };
  for (auto const& dir : dirs) {
    TestDerivative(inputfunc, input, dir, h, doutput_dinput);
  }
}

static void TestDInputAt(TransformRT const& rt, Real3 input, Real3 pivot) {
  real h = 0.001_r;
  Matrix<real> kry_doutput(RigidSize::kDim, RigidSize::kDim);
  auto kry_dinput = Matrix<real>::Zero(RigidSize::kDim, RigidSize::kDim);
  kry_dinput(0, 0) = 1.0_r;
  kry_dinput(1, 1) = 1.0_r;
  kry_dinput(2, 2) = 1.0_r;

  DTransformBatch(rt, AsConstView(kry_dinput), AsView(kry_doutput));
  auto doutput_dinput = ToNdArray<RigidSize::kDim, RigidSize::kDim>(kry_doutput);

  auto dirs = DiagonalMatrix<RigidSize::kDim, real>(1.0_r);

  std::function<Real3(Real3)> inputfunc = [rt, pivot](Real3 in) {
    return TransformPivot(rt, in, pivot);
  };
  for (auto const& dir : dirs) {
    TestDerivative(inputfunc, input, dir, h, doutput_dinput);
  }
}

// Test TransformRT derivative in the input variables
TEST(DTransformRT, TestDifferentialsDInput) {
  TransformRT rt1(SampleTransform1());
  TransformRT rt2(SampleTransform2());

  auto input1 = SampleVector1();
  auto input2 = SampleVector2();

  TestDInputAt(rt1, input1);
  TestDInputAt(rt1, input2);
  TestDInputAt(rt2, input1);
  TestDInputAt(rt2, input2);
}

// Test TransformRT derivative in the parameter variables
static void TestDParamsAt(
    TransformRT const& rt,
    Real3 input,
    TransformSRT preTransform = {},
    TransformSRT postTransform = {}) {
  real h = 0.001_r;
  Matrix<real> kry_doutput_dparams(RigidSize::kDim, RigidSize::kDAll);
  DTransformDParametersBatch(
      rt,
      AsView(Span<real const>(&input[0], 3)),
      AsView(kry_doutput_dparams),
      preTransform,
      postTransform);
  auto doutput_dparams = ToNdArray<RigidSize::kDim, RigidSize::kDAll>(kry_doutput_dparams);

  NdArray<real, RigidSize::kDAll> x;
  TransformToRawDofs(rt, AsView(x));

  auto dirs = DiagonalMatrix<RigidSize::kDAll, real>(1.0_r);

  using real_in_t = NdArray<real, RigidSize::kDAll>;
  std::function<Real3(real_in_t)> paramfunc = [&input, preTransform, postTransform](real_in_t in) {
    return TransformPrePost(
        TransformFromRawDofs(AsView(in).TopRows<RigidSize::kDAll>(RigidSize::kDAll)),
        input,
        preTransform,
        postTransform);
  };
  for (auto const& dir : dirs) {
    TestLieDerivative(paramfunc, x, dir, h, doutput_dparams);
  }
}

static void TestDParamsAt(TransformRT const& rt, Real3 input, Real3 pivot) {
  real h = 0.001_r;
  Matrix<real> kry_doutput_dparams(RigidSize::kDim, RigidSize::kDAll);
  DTransformDParametersBatch(
      rt, AsConstView(Span<real const>(&input[0], 3)), AsView(kry_doutput_dparams), pivot);
  auto doutput_dparams = ToNdArray<RigidSize::kDim, RigidSize::kDAll>(kry_doutput_dparams);

  NdArray<real, RigidSize::kDAll> x;
  TransformToRawDofs(rt, AsView(x));

  auto dirs = DiagonalMatrix<RigidSize::kDAll, real>(1.0_r);

  using real_in_t = NdArray<real, RigidSize::kDAll>;
  std::function<Real3(real_in_t)> paramfunc = [input, pivot](real_in_t in) {
    return TransformPivot(TransformFromRawDofs(AsView(in)), input, pivot);
  };
  for (auto const& dir : dirs) {
    TestLieDerivative(paramfunc, x, dir, h, doutput_dparams);
  }
}

// Test TransformRT derivative in the parameter variables
TEST(DTransformRT, TestDifferentialsDParams) {
  TransformRT rt1(SampleTransform1());
  TransformRT rt2(SampleTransform2());

  auto input1 = SampleVector1();
  auto input2 = SampleVector2();

  TestDParamsAt(rt1, input1);
  TestDParamsAt(rt1, input2);
  TestDParamsAt(rt2, input1);
  TestDParamsAt(rt2, input2);
}

// Test computing derivatives in batches
TEST(DTransformRT, TestDifferentialDInputBatch) {
  TransformRT rt(SampleTransform1());

  Matrix<real> outDRT(RigidSize::kDim * 3, RigidSize::kDim);
  auto inDRT = Matrix<real>::Zero(RigidSize::kDim * 3, RigidSize::kDim);
  for (int i = 0; i < RigidSize::kDim * 3; ++i) {
    for (int j = 0; j < RigidSize::kDim; ++j) {
      inDRT(i, j) = static_cast<mochi::real>(i * RigidSize::kDim + j);
    }
  }

  DTransformBatch(rt, AsConstView(inDRT), AsView(outDRT));

  Matrix<real> outCompare(RigidSize::kDim * 3, RigidSize::kDim);

  auto outCompare1 = outCompare.Block(0, 0, 3, 3);
  auto outCompare2 = outCompare.Block(3, 0, 3, 3);
  auto outCompare3 = outCompare.Block(6, 0, 3, 3);
  auto inCompare1 = inDRT.Block(0, 0, 3, 3);
  auto inCompare2 = inDRT.Block(3, 0, 3, 3);
  auto inCompare3 = inDRT.Block(6, 0, 3, 3);

  DTransformBatch(rt, AsConstView(inCompare1), outCompare1);
  DTransformBatch(rt, AsConstView(inCompare2), outCompare2);
  DTransformBatch(rt, AsConstView(inCompare3), outCompare3);

  auto ndOutDrt = ToNdArray<RigidSize::kDim * 3, RigidSize::kDim>(outDRT);
  auto ndOutCompare = ToNdArray<RigidSize::kDim * 3, RigidSize::kDim>(outCompare);

  EXPECT_NEAR_EQ(ndOutDrt, ndOutCompare);
}

// Test computing derivatives in batches
TEST(DTransformRT, TestDifferentialDParamsBatch) {
  TransformRT rt(SampleTransform1());

  NdArray<real, 3, RigidSize::kDim> vecs;
  vecs[0] = SampleVector1();
  vecs[1] = SampleVector2();
  vecs[2] = SampleVector1();
  Matrix<real> outDRT(RigidSize::kDim * 3, RigidSize::kDAll);
  ColumnVectorView<real> inDRT(&vecs[0][0], RigidSize::kDim * 3, 1);

  DTransformDParametersBatch(rt, AsConstView(inDRT), AsView(outDRT));

  RowMatrix<real> outCompare(RigidSize::kDim * 3, RigidSize::kDAll);

  auto outCompare1 = outCompare.Block(0, 0, 3, RigidSize::kDAll);
  auto outCompare2 = outCompare.Block(3, 0, 3, RigidSize::kDAll);
  auto outCompare3 = outCompare.Block(6, 0, 3, RigidSize::kDAll);
  auto inCompare1 = inDRT.Slice(0, 3);
  auto inCompare2 = inDRT.Slice(3, 3);
  auto inCompare3 = inDRT.Slice(6, 3);

  DTransformDParametersBatch(rt, AsConstView(inCompare1), AsView(outCompare1));
  DTransformDParametersBatch(rt, AsConstView(inCompare2), AsView(outCompare2));
  DTransformDParametersBatch(rt, AsConstView(inCompare3), AsView(outCompare3));

  auto ndOutDrt = ToNdArray<RigidSize::kDim * 3, RigidSize::kDAll>(outDRT);
  auto ndOutCompare = ToNdArray<RigidSize::kDim * 3, RigidSize::kDAll>(outCompare);

  EXPECT_NEAR_EQ(ndOutDrt, ndOutCompare);
}

TEST(DTransformRT, TestRepivoting) {
  TransformRT rt(SampleTransform1());

  NdArray<real, 3, RigidSize::kDim> vecs;
  vecs[0] = SampleVector1();
  vecs[1] = SampleVector2();
  vecs[2] = SampleVector1();

  ColumnVector<real> out1(RigidSize::kDim * 3);
  ColumnVector<real> out2(RigidSize::kDim * 3);
  ColumnVector<real> out3(RigidSize::kDim * 3);

  ColumnVectorView<real> in(&vecs[0][0], RigidSize::kDim * 3, 1);

  Real3 pivot = Real3{0.5_r, -0.5_r, 1.0_r};

  TransformRT rtRepivot = Repivot(rt, pivot);
  TransformRT rtUnpivot = Repivot(rtRepivot, -pivot);

  TransformBatch(rt, in, out1);
  TransformBatch(rtRepivot, in, out2, pivot);
  TransformBatch(rtUnpivot, in, out3);

  ColumnVector<real> diff = out1 - out2;
  EXPECT_NEAR_EQ(diff.Norm(), 0.0_r);

  diff = out1 - out3;
  EXPECT_NEAR_EQ(diff.Norm(), 0.0_r);
}

// Test TransformRT derivative in the input variables with a pivot
TEST(DTransformRT, TestTestDifferentialsDInputPivoted) {
  TransformRT rt1(SampleTransform1());
  TransformRT rt2(SampleTransform2());

  Real3 pivot = Real3{0.5_r, -0.5_r, 1.0_r};

  auto input1 = SampleVector1();
  auto input2 = SampleVector2();

  TestDInputAt(rt1, input1, pivot);
  TestDInputAt(rt1, input2, pivot);
  TestDInputAt(rt2, input1, pivot);
  TestDInputAt(rt2, input2, pivot);
}

// Test TransformRT derivative in the parameter variables with a pivot
TEST(DTransformRT, TestDifferentialsDParamsPivoted) {
  TransformRT rt1(SampleTransform1());
  TransformRT rt2(SampleTransform2());

  Real3 pivot = Real3{0.5_r, -0.5_r, 1.0_r};

  auto input1 = SampleVector1();
  auto input2 = SampleVector2();

  TestDParamsAt(rt1, input1, pivot);
  TestDParamsAt(rt1, input2, pivot);
  TestDParamsAt(rt2, input1, pivot);
  TestDParamsAt(rt2, input2, pivot);
}

// Test TransformRT derivative in the input variables with post/pre transform
TEST(DTransformRT, TestTestDifferentialsDInputPrePost) {
  TransformRT rt1(SampleTransform1());
  TransformRT rt2(SampleTransform2());

  auto input1 = SampleVector1();
  auto input2 = SampleVector2();

  auto srt1 = SampleTransformSRT1();
  auto srt2 = SampleTransformSRT2();

  TestDInputAt(rt1, input1, srt1, srt2);
  TestDInputAt(rt1, input2, srt2, srt1);
  TestDInputAt(rt2, input1, srt1, srt1);
  TestDInputAt(rt2, input2, srt2, srt2);
}

// Test TransformRT derivative in the parameter variables with a pivot
TEST(DTransformRT, TestDifferentialsDParamsPrePost) {
  TransformRT rt1(SampleTransform1());
  TransformRT rt2(SampleTransform2());

  auto input1 = SampleVector1();
  auto input2 = SampleVector2();

  auto srt1 = SampleTransformSRT1();
  auto srt2 = SampleTransformSRT2();

  TestDParamsAt(rt1, input1, srt1, srt2);
  TestDParamsAt(rt1, input2, srt2, srt1);
  TestDParamsAt(rt2, input1, srt1, srt1);
  TestDParamsAt(rt2, input2, srt2, srt2);
}
