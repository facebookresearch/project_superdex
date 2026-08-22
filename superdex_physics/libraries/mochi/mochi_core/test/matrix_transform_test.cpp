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
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/matrix_transform_rt.h>

using namespace mochi;

static_assert(alignof(MatrixTransformRT) == alignof(Vec4r), "Unexpected alignment");
static_assert(sizeof(MatrixTransformRT) == sizeof(real) * 16, "Unexpected padding");

TEST(MatrixTransformRT, TransformPoint) {
  // clang-format off
  Real3 trans{1_r, 2_r, 3_r};
  MatrixTransformRT transformX90 = ToMatrixTransformRT(Quaternion::RotationX(90_r * kRadiansPerDegree), trans);
  MatrixTransformRT transformY90 = ToMatrixTransformRT(Quaternion::RotationY(90_r * kRadiansPerDegree), trans);
  MatrixTransformRT transformZ90 = ToMatrixTransformRT(Quaternion::RotationZ(90_r * kRadiansPerDegree), trans);

  // Rotate point, then add translation
  EXPECT_NEAR_EQ(Vec4r(1_r, 2_r, 4_r, 1_r), transformX90.TransformPoint(Vec4r(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(1_r, 3_r, 3_r, 1_r), transformY90.TransformPoint(Vec4r(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(0_r, 2_r, 3_r, 1_r), transformZ90.TransformPoint(Vec4r(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Real3(1_r, 2_r, 4_r), transformX90.TransformPoint(Real3(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Real3(1_r, 3_r, 3_r), transformY90.TransformPoint(Real3(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Real3(0_r, 2_r, 3_r), transformZ90.TransformPoint(Real3(0_r, 1_r, 0_r)));

  // Subtract translation, then rotate in the opposite direction
  EXPECT_NEAR_EQ(Vec4r(-1_r, -3_r, 1_r, 1_r), transformX90.InverseTransformPoint(Vec4r(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(3_r, -1_r, -1_r, 1_r), transformY90.InverseTransformPoint(Vec4r(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(-1_r, 1_r, -3_r, 1_r), transformZ90.InverseTransformPoint(Vec4r(0_r, 1_r, 0_r)));

  EXPECT_NEAR_EQ(Real3(-1_r, -3_r, 1_r), transformX90.InverseTransformPoint(Real3(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Real3(3_r, -1_r, -1_r), transformY90.InverseTransformPoint(Real3(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Real3(-1_r, 1_r, -3_r), transformZ90.InverseTransformPoint(Real3(0_r, 1_r, 0_r)));
  // clang-format on
}

TEST(MatrixTransformRT, TransformDirection) {
  // clang-format off
  Real3 trans{1_r, 2_r, 3_r}; // not used for transforming direction vectors
  MatrixTransformRT transformX90 = ToMatrixTransformRT(Quaternion::RotationX(90_r * kRadiansPerDegree), trans);
  MatrixTransformRT transformY90 = ToMatrixTransformRT(Quaternion::RotationY(90_r * kRadiansPerDegree), trans);
  MatrixTransformRT transformZ90 = ToMatrixTransformRT(Quaternion::RotationZ(90_r * kRadiansPerDegree), trans);

  // Just rotate
  EXPECT_NEAR_EQ(Vec4r(0_r, 0_r, 1_r, 0_r), transformX90.TransformDirection(Vec4r(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(0_r, 1_r, 0_r, 0_r), transformY90.TransformDirection(Vec4r(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(-1_r, 0_r, 0_r, 0_r), transformZ90.TransformDirection(Vec4r(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Real3(0_r, 0_r, 1_r), transformX90.TransformDirection(Real3(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Real3(0_r, 1_r, 0_r), transformY90.TransformDirection(Real3(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Real3(-1_r, 0_r, 0_r), transformZ90.TransformDirection(Real3(0_r, 1_r, 0_r)));

  // Just rotate in opposite direction
  EXPECT_NEAR_EQ(Vec4r(0_r, 0_r, -1_r, 0_r), transformX90.InverseTransformDirection(Vec4r(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(0_r, 1_r, 0_r, 0_r), transformY90.InverseTransformDirection(Vec4r(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Vec4r(1_r, 0_r, 0_r, 0_r), transformZ90.InverseTransformDirection(Vec4r(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Real3(0_r, 0_r, -1_r), transformX90.InverseTransformDirection(Real3(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Real3(0_r, 1_r, 0_r), transformY90.InverseTransformDirection(Real3(0_r, 1_r, 0_r)));
  EXPECT_NEAR_EQ(Real3(1_r, 0_r, 0_r), transformZ90.InverseTransformDirection(Real3(0_r, 1_r, 0_r)));
  // clang-format on
}

TEST(MatrixTransformRT, MulTransform) {
  Quaternion aRot = Quaternion::RotationX(kPI / 2_r);
  Quaternion bRot = Quaternion::RotationY(-kPI);
  Real3 aTrans = Real3{0_r, 0_r, 1_r};
  Real3 bTrans = Real3{2_r, 0_r, 0_r};
  TransformRT a(aRot, aTrans);
  TransformRT b(bRot, bTrans);
  MatrixTransformRT ma = ToMatrixTransformRT(a);
  MatrixTransformRT mb = ToMatrixTransformRT(b);
  Real3 pt = {1_r, 2_r, 3_r};

  // Transform by a, then by b
  MatrixTransformRT mc = mb * ma;
  EXPECT_NEAR_EQ(ToMatrixTransformRT(b * a), mc); // equivalent to TransfromRT method
  EXPECT_NEAR_EQ(mb.TransformPoint(ma.TransformPoint(pt)), mc.TransformPoint(pt));

  // Transform by b, then by a
  mc = ma * mb;
  EXPECT_NEAR_EQ(ToMatrixTransformRT(a * b), mc); // equivalent to TransfromRT method
  EXPECT_NEAR_EQ(ma.TransformPoint(mb.TransformPoint(pt)), mc.TransformPoint(pt));

  // Identity
  EXPECT_NEAR_EQ(ma, ma * MatrixTransformRT());
  EXPECT_NEAR_EQ(ma, MatrixTransformRT() * ma);
}

TEST(MatrixTransformRT, ToVMatrix4x4) {
  Real3 const testAxes[] = {
      Real3{0.0_r, 0.0_r, 1.0_r},
      Real3{0.0_r, 1.0_r, 0.0_r},
      Real3{0.0_r, 1.0_r, 1.0_r},
      Real3{1.0_r, 0.0_r, 0.0_r},
      Real3{1.0_r, 0.0_r, 1.0_r},
      Real3{1.0_r, 1.0_r, 0.0_r},
      Real3{1.0_r, 1.0_r, 1.0_r}};
  Real3 const testTranslation = Real3{1.1f, 2.2f, 3.3f};
  real const testAnglesDeg[] = {0_r, 45_r, 90_r, 135_r, 180_r, -45_r, -90_r, -135_r, -180_r};
  for (Real3 const& axis : testAxes) {
    for (real deg : testAnglesDeg) {
      // Construct a TransformRT
      real angle = deg * kRadiansPerDegree;
      Quaternion rot = Quaternion::FromAxisAngle(Normalize(axis), angle);
      MatrixTransformRT matRT = ToMatrixTransformRT(rot, testTranslation);

      // Convert to matrix
      VMatrix4x4r mat4 = ToVMatrix4x4(matRT); // basis vectors as columns

      // The first 3 columns of the matrix should be the basis vectors
      VMatrix4x4r matTranspose4 = Transpose4x4(mat4);
      EXPECT_NEAR_EQ(rot * Vec4r(1_r, 0_r, 0_r), matTranspose4[0]);
      EXPECT_NEAR_EQ(rot * Vec4r(0_r, 1_r, 0_r), matTranspose4[1]);
      EXPECT_NEAR_EQ(rot * Vec4r(0_r, 0_r, 1_r), matTranspose4[2]);

      // Last column should be (x,y,z,1) translation
      EXPECT_NEAR_EQ(Vec4r(1.1_r, 2.2_r, 3.3_r, 1.0_r), matTranspose4[3]);

      // Convert to TransformRT and compare
      TransformRT rt2 = TransformRT::FromOrthoNormal(mat4);
      EXPECT_TRUE(EquivalentRotation(rot, rt2.GetRotation()));
      EXPECT_NEAR_EQ(testTranslation, rt2.GetTranslation());

      // DotMatVec4x4 should be equivalent to matRT.TransformPoint(pt)
      auto pt = Vec4r(2_r, 3_r, 4_r, 1_r);
      EXPECT_NEAR_EQ(DotMatVec4x4(mat4, pt), matRT.TransformPoint(pt));
      EXPECT_NEAR_EQ(DotVecMat4x4(pt, matTranspose4), matRT.TransformPoint(pt));
    }
  }
}

TEST(MatrixTransformRT, Invert) {
  // clang-format off
  Real3 trans{1_r, 2_r, 3_r}; // not used for transforming direction vectors
  MatrixTransformRT xPlus90 = ToMatrixTransformRT(Quaternion::RotationX(90_r * kRadiansPerDegree), trans);
  MatrixTransformRT yPlus90 = ToMatrixTransformRT(Quaternion::RotationY(90_r * kRadiansPerDegree), trans);
  MatrixTransformRT zPlus90 = ToMatrixTransformRT(Quaternion::RotationZ(90_r * kRadiansPerDegree), trans);

  // Test inverse directly
  EXPECT_NEAR_EQ(ToMatrixTransformRT(Quaternion::RotationX(-90_r * kRadiansPerDegree), Real3{-1_r, -3_r, 2_r}), Invert(xPlus90));
  EXPECT_NEAR_EQ(ToMatrixTransformRT(Quaternion::RotationY(-90_r * kRadiansPerDegree), Real3{3_r, -2_r, -1_r}), Invert(yPlus90));
  EXPECT_NEAR_EQ(ToMatrixTransformRT(Quaternion::RotationZ(-90_r * kRadiansPerDegree), Real3{-2_r, 1_r, -3_r}), Invert(zPlus90));

  // Test identity
  EXPECT_NEAR_EQ(MatrixTransformRT{}, xPlus90 * Invert(xPlus90));
  EXPECT_NEAR_EQ(MatrixTransformRT{}, yPlus90 * Invert(yPlus90));
  EXPECT_NEAR_EQ(MatrixTransformRT{}, zPlus90 * Invert(zPlus90));
  EXPECT_NEAR_EQ(MatrixTransformRT{}, Invert(xPlus90) * xPlus90);
  EXPECT_NEAR_EQ(MatrixTransformRT{}, Invert(yPlus90) * yPlus90);
  EXPECT_NEAR_EQ(MatrixTransformRT{}, Invert(zPlus90) * zPlus90);

  // clang-format on
}
