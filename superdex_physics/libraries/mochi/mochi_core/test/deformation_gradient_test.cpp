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

#include <mochi_core/element_operations/element_operation_utils.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/math_utils.h>

#include <gtest/gtest.h>

using namespace mochi;

TEST(DeformationGradient, TrianglePlaneStrainIdentity) {
  // Test case: Identity deformation (no displacement)

  // Create reference positions for a triangle in the xy-plane.
  NdArray<real, 3, 3> referencePositions = {
      Real3{0.0_r, 0.0_r, 0.0_r}, Real3{1.0_r, 0.0_r, 0.0_r}, Real3{0.0_r, 1.0_r, 0.0_r}};

  // Zero displacements:
  NdArray<Vec4r, 3> displacements = {
      Vec4r{0.0_r, 0.0_r, 0.0_r, 0.0_r},
      Vec4r{0.0_r, 0.0_r, 0.0_r, 0.0_r},
      Vec4r{0.0_r, 0.0_r, 0.0_r, 0.0_r}};

  // Basis function derivatives with respect to reference coordinates:
  NdArray<Vec4r, 3> dBasisEvaluated = {
      Vec4r{-1.0_r, -1.0_r, 0.0_r, 0.0_r},
      Vec4r{1.0_r, 0.0_r, 0.0_r, 0.0_r},
      Vec4r{0.0_r, 1.0_r, 0.0_r, 0.0_r}};

  // Expected result: Identity matrix for no deformation
  NdArray<real, 3, 3> expected = {
      Real3{1.0_r, 0.0_r, 0.0_r}, Real3{0.0_r, 1.0_r, 0.0_r}, Real3{0.0_r, 0.0_r, 1.0_r}};

  // Call the function.
  auto result =
      GetDeformationGrad(dBasisEvaluated, ToSimdMatrix(referencePositions), displacements);

  // Check the result.
  EXPECT_NEAR_EQ(ToSimdMatrix(expected), result);
}

TEST(DeformationGradient, TrianglePlaneStrainUniformScaling) {
  // Test case: Uniform scaling

  // Create reference positions for a triangle in the xy-plane
  NdArray<real, 3, 3> referencePositions = {
      Real3{0.0_r, 0.0_r, 0.0_r}, Real3{1.0_r, 0.0_r, 0.0_r}, Real3{0.0_r, 1.0_r, 0.0_r}};

  // Displacements for uniform scaling by factor of 2:
  NdArray<Vec4r, 3> displacements = {
      Vec4r{0.0_r, 0.0_r, 0.0_r, 0.0_r},
      Vec4r{1.0_r, 0.0_r, 0.0_r, 0.0_r},
      Vec4r{0.0_r, 1.0_r, 0.0_r, 0.0_r}};

  // Basis function derivatives with respect to reference coordinates:
  NdArray<Vec4r, 3> dBasisEvaluated = {
      Vec4r{-1.0_r, -1.0_r, 0.0_r, 0.0_r},
      Vec4r{1.0_r, 0.0_r, 0.0_r, 0.0_r},
      Vec4r{0.0_r, 1.0_r, 0.0_r, 0.0_r}};

  // Expected result: Diagonal matrix with scale factors
  NdArray<real, 3, 3> expected = {
      Real3{2.0_r, 0.0_r, 0.0_r}, Real3{0.0_r, 2.0_r, 0.0_r}, Real3{0.0_r, 0.0_r, 1.0_r}
      // No scaling in z-direction for plane strain
  };

  // Call the function.
  auto result =
      GetDeformationGrad(dBasisEvaluated, ToSimdMatrix(referencePositions), displacements);

  // Check the result.
  EXPECT_NEAR_EQ(ToSimdMatrix(expected), result);
}

TEST(DeformationGradient, TrianglePlaneStrainShear) {
  // Test case: Shear deformation

  // Create reference positions for a triangle in the xy-plane.
  NdArray<real, 3, 3> referencePositions = {
      Real3{0.0_r, 0.0_r, 0.0_r}, Real3{1.0_r, 0.0_r, 0.0_r}, Real3{0.0_r, 1.0_r, 0.0_r}};

  // Displacements for shear (move top vertex to the right):
  NdArray<Vec4r, 3> displacements = {
      Vec4r{0.0_r, 0.0_r, 0.0_r, 0.0_r},
      Vec4r{0.0_r, 0.0_r, 0.0_r, 0.0_r},
      Vec4r{0.5_r, 0.0_r, 0.0_r, 0.0_r}};

  // Basis function derivatives with respect to reference coordinates:
  NdArray<Vec4r, 3> dBasisEvaluated = {
      Vec4r{-1.0_r, -1.0_r, 0.0_r, 0.0_r},
      Vec4r{1.0_r, 0.0_r, 0.0_r, 0.0_r},
      Vec4r{0.0_r, 1.0_r, 0.0_r, 0.0_r}};

  // Expected result: Shear deformation matrix
  NdArray<real, 3, 3> expected = {
      Real3{1.0_r, 0.5_r, 0.0_r}, Real3{0.0_r, 1.0_r, 0.0_r}, Real3{0.0_r, 0.0_r, 1.0_r}};

  // Call the function.
  auto result =
      GetDeformationGrad(dBasisEvaluated, ToSimdMatrix(referencePositions), displacements);

  // Check the result.
  EXPECT_NEAR_EQ(ToSimdMatrix(expected), result);
}

TEST(DeformationGradient, TrianglePlaneStrainRotation) {
  // Test case: Rotation around z-axis

  // Create reference positions for a triangle in the xy-plane.
  NdArray<real, 3, 3> referencePositions = {
      Real3{0.0_r, 0.0_r, 0.0_r}, Real3{1.0_r, 0.0_r, 0.0_r}, Real3{0.0_r, 1.0_r, 0.0_r}};

  // Displacements for 90-degree rotation around z-axis:
  // New positions: (0,0,0), (0,1,0), (-1,0,0)
  NdArray<Vec4r, 3> displacements = {
      Vec4r{0.0_r, 0.0_r, 0.0_r, 0.0_r},
      Vec4r{-1.0_r, 1.0_r, 0.0_r, 0.0_r},
      Vec4r{-1.0_r, -1.0_r, 0.0_r, 0.0_r}};

  // Basis function derivatives with respect to reference coordinates:
  NdArray<Vec4r, 3> dBasisEvaluated = {
      Vec4r{-1.0_r, -1.0_r, 0.0_r, 0.0_r},
      Vec4r{1.0_r, 0.0_r, 0.0_r, 0.0_r},
      Vec4r{0.0_r, 1.0_r, 0.0_r, 0.0_r}};

  // Expected result: 90-degree rotation matrix around z-axis.
  NdArray<real, 3, 3> expected = {
      Real3{0.0_r, -1.0_r, 0.0_r}, Real3{1.0_r, 0.0_r, 0.0_r}, Real3{0.0_r, 0.0_r, 1.0_r}};

  // Call the function.
  auto result =
      GetDeformationGrad(dBasisEvaluated, ToSimdMatrix(referencePositions), displacements);

  // Check the result.
  EXPECT_NEAR_EQ(ToSimdMatrix(expected), result);
}

TEST(DeformationGradient, TrianglePlaneStrainOutOfPlaneRotation) {
  // Test case: Out-of-plane rotation (rotation around x-axis)

  // Create reference positions for a triangle in the xy-plane.
  NdArray<real, 3, 3> referencePositions = {
      Real3{0.0_r, 0.0_r, 0.0_r}, Real3{1.0_r, 0.0_r, 0.0_r}, Real3{0.0_r, 1.0_r, 0.0_r}};

  // Displacements for rotation around x-axis (CCW when viewed looking down the x-axis):
  real const theta = 0.12345_r; // Arbitrary angle
  real const sinTheta = std::sin(theta);
  real const cosTheta = std::cos(theta);

  NdArray<Vec4r, 3> displacements = {
      Vec4r{0.0_r, 0.0_r, 0.0_r, 0.0_r},
      Vec4r{0.0_r, 0.0_r, 0.0_r, 0.0_r},
      Vec4r{0.0_r, cosTheta - 1.0_r, sinTheta, 0.0_r}};

  // Basis function derivatives with respect to reference coordinates:
  NdArray<Vec4r, 3> dBasisEvaluated = {
      Vec4r{-1.0_r, -1.0_r, 0.0_r, 0.0_r},
      Vec4r{1.0_r, 0.0_r, 0.0_r, 0.0_r},
      Vec4r{0.0_r, 1.0_r, 0.0_r, 0.0_r}};

  // Expected result: Rotation matrix around x-axis
  NdArray<real, 3, 3> expected = {
      Real3{1.0_r, 0.0_r, 0.0_r},
      Real3{0.0_r, cosTheta, -sinTheta},
      Real3{0.0_r, sinTheta, cosTheta}};

  // Call the function.
  auto result =
      GetDeformationGrad(dBasisEvaluated, ToSimdMatrix(referencePositions), displacements);

  // Check the result.
  EXPECT_NEAR_EQ(ToSimdMatrix(expected), result);
}

TEST(DeformationGradient, TrianglePlaneStrainComplex) {
  // Test case: Complex deformation (combination of scaling, shear, and rotation)

  // Create reference positions for a triangle in the xy-plane.
  NdArray<real, 3, 3> referencePositions = {
      Real3{0.0_r, 0.0_r, 0.0_r}, Real3{1.0_r, 0.0_r, 0.0_r}, Real3{0.0_r, 1.0_r, 0.0_r}};

  // Displacements for a complex deformation:
  NdArray<Vec4r, 3> displacements = {
      Vec4r{0.1_r, 0.2_r, 0.3_r, 0.0_r},
      Vec4r{0.8_r, 0.4_r, 0.1_r, 0.0_r},
      Vec4r{0.3_r, 0.7_r, 0.5_r, 0.0_r}};

  // Basis function derivatives with respect to reference coordinates:
  NdArray<Vec4r, 3> dBasisEvaluated = {
      Vec4r{-1.0_r, -1.0_r, 0.0_r, 0.0_r},
      Vec4r{1.0_r, 0.0_r, 0.0_r, 0.0_r},
      Vec4r{0.0_r, 1.0_r, 0.0_r, 0.0_r}};

  // Call the function.
  auto result =
      GetDeformationGrad(dBasisEvaluated, ToSimdMatrix(referencePositions), displacements);

  // For complex deformations, we can verify certain properties rather than the exact matrix.

  // Verify that the function correctly computes the deformation gradient by checking
  // that it maps reference vectors to deformed vectors.

  // Reference edge vectors.
  Real3 refEdge1 = referencePositions[1] - referencePositions[0]; // (1,0,0)
  Real3 refEdge2 = referencePositions[2] - referencePositions[0]; // (0,1,0)

  // Deformed edge vectors.
  Real3 defPos0 = referencePositions[0] + ToReal3(displacements[0]);
  Real3 defPos1 = referencePositions[1] + ToReal3(displacements[1]);
  Real3 defPos2 = referencePositions[2] + ToReal3(displacements[2]);
  Real3 defEdge1 = defPos1 - defPos0;
  Real3 defEdge2 = defPos2 - defPos0;

  // Apply deformation gradient to reference edges.
  Real3 mappedEdge1 = {
      Dot(ToReal3(result[0]), refEdge1),
      Dot(ToReal3(result[1]), refEdge1),
      Dot(ToReal3(result[2]), refEdge1)};

  Real3 mappedEdge2 = {
      Dot(ToReal3(result[0]), refEdge2),
      Dot(ToReal3(result[1]), refEdge2),
      Dot(ToReal3(result[2]), refEdge2)};

  // Check that mapped edges match deformed edges.
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(defEdge1[i], mappedEdge1[i], 1e-5_r);
    EXPECT_NEAR(defEdge2[i], mappedEdge2[i], 1e-5_r);
  }

  // Verify that the pushforward of the reference normal matches the deformed normal obtained by
  // a direct normalized cross product of the deformed edges.  Note that we do not want to use
  // Nanson's formula here, as we are testing the explicit kinematic assumption that the current
  // normal is the pushforward of the reference normal, not general transformation of normal vectors
  // from 3D deformations.
  Real3 const refNormal = Normalize(Cross(refEdge1, refEdge2));
  Real3 const currentNormal = Normalize(Cross(defEdge1, defEdge2));
  Real3 const mappedNormal = {
      Dot(ToReal3(result[0]), refNormal),
      Dot(ToReal3(result[1]), refNormal),
      Dot(ToReal3(result[2]), refNormal)};
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(currentNormal[i], mappedNormal[i], 1e-5_r);
  }
}
