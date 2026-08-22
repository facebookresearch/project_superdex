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

#include <mochi_core/utils/matrix_utils.h>

#include <array>

// The data below is generated for few deformation gradients
// with data/generate_materials_test_data.py and, for a choice
// of deformation gradient spits out the strain energy, the
// first Piola-Kirchhoff stress and its tangent.
namespace mochi::materials::test {
constexpr int kSpaceDim = 3;
struct TestData {
  NdArray<real, kSpaceDim, kSpaceDim> deformationGradient;
  real strainEnergy;
  NdArray<real, kSpaceDim, kSpaceDim> piolaKirchhoffStress;
  NdArray<real, kSpaceDim, kSpaceDim, kSpaceDim, kSpaceDim> tangent;
};

namespace neo_hookean_test_data {
// The data was acquired for Youngs Modulus = 1.0 and poisson = 0.4
constexpr int kNumTests = 6;
constexpr std::array<TestData, kNumTests> kTestData{
    // Deformation gradient identity
    TestData{
        .deformationGradient =
            NdArray<real, 3, 3>{
                Real3{1.000000e+00_r, 0.000000e+00_r, 0.000000e+00_r},
                Real3{0.000000e+00_r, 1.000000e+00_r, 0.000000e+00_r},
                Real3{0.000000e+00_r, 0.000000e+00_r, 1.000000e+00_r}},
        // Strain Energy
        .strainEnergy = 0.00000e+00_r,
        // Material PK1 identity
        .piolaKirchhoffStress =
            NdArray<real, 3, 3>{
                Real3{0.000000e+00_r, 0.000000e+00_r, 0.000000e+00_r},
                Real3{0.000000e+00_r, 0.000000e+00_r, 0.000000e+00_r},
                Real3{0.000000e+00_r, 0.000000e+00_r, 0.000000e+00_r}},
        // Material Tangent identity
        .tangent =
            NdArray<real, 3, 3, 3, 3>{
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{2.142857e+00_r, 0.000000e+00_r, 0.000000e+00_r},
                        Real3{0.000000e+00_r, 1.428571e+00_r, 0.000000e+00_r},
                        Real3{0.000000e+00_r, 0.000000e+00_r, 1.428571e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{0.000000e+00_r, 3.571429e-01_r, 0.000000e+00_r},
                        Real3{3.571429e-01_r, 0.000000e+00_r, 0.000000e+00_r},
                        Real3{0.000000e+00_r, 0.000000e+00_r, 0.000000e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{0.000000e+00_r, 0.000000e+00_r, 3.571429e-01_r},
                        Real3{0.000000e+00_r, 0.000000e+00_r, 0.000000e+00_r},
                        Real3{3.571429e-01_r, 0.000000e+00_r, 0.000000e+00_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{0.000000e+00_r, 3.571429e-01_r, 0.000000e+00_r},
                        Real3{3.571429e-01_r, 0.000000e+00_r, 0.000000e+00_r},
                        Real3{0.000000e+00_r, 0.000000e+00_r, 0.000000e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{1.428571e+00_r, 0.000000e+00_r, 0.000000e+00_r},
                        Real3{0.000000e+00_r, 2.142857e+00_r, 0.000000e+00_r},
                        Real3{0.000000e+00_r, 0.000000e+00_r, 1.428571e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{0.000000e+00_r, 0.000000e+00_r, 0.000000e+00_r},
                        Real3{0.000000e+00_r, 0.000000e+00_r, 3.571429e-01_r},
                        Real3{0.000000e+00_r, 3.571429e-01_r, 0.000000e+00_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{0.000000e+00_r, 0.000000e+00_r, 3.571429e-01_r},
                        Real3{0.000000e+00_r, 0.000000e+00_r, 0.000000e+00_r},
                        Real3{3.571429e-01_r, 0.000000e+00_r, 0.000000e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{0.000000e+00_r, 0.000000e+00_r, 0.000000e+00_r},
                        Real3{0.000000e+00_r, 0.000000e+00_r, 3.571429e-01_r},
                        Real3{0.000000e+00_r, 3.571429e-01_r, 0.000000e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{1.428571e+00_r, 0.000000e+00_r, 0.000000e+00_r},
                        Real3{0.000000e+00_r, 1.428571e+00_r, 0.000000e+00_r},
                        Real3{0.000000e+00_r, 0.000000e+00_r, 2.142857e+00_r}}},
            }}, // End of TestData
    // Test : random
    // Deformation gradient random
    TestData{
        .deformationGradient =
            NdArray<real, 3, 3>{
                Real3{1.193431e+00_r, -3.102000e-01_r, 3.961964e-02_r},
                Real3{-2.149692e-01_r, 8.641365e-01_r, -1.112272e-01_r},
                Real3{1.199360e-02_r, 2.928228e-01_r, 7.080803e-01_r}},
        // Strain Energy
        .strainEnergy = 1.79858e-01_r,
        // Material PK1 random
        .piolaKirchhoffStress =
            NdArray<real, 3, 3>{
                Real3{-3.151828e-01_r, -2.843680e-01_r, 9.849201e-02_r},
                Real3{-3.428152e-01_r, -6.630194e-01_r, 3.665985e-01_r},
                Real3{3.977520e-03_r, -3.833571e-02_r, -8.568476e-01_r}},
        // Material Tangent random
        .tangent =
            NdArray<real, 3, 3, 3, 3>{
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{2.167723e+00_r, 4.239024e-01_r, -2.059705e-01_r},
                        Real3{6.496927e-01_r, 1.558210e+00_r, -6.553938e-01_r},
                        Real3{7.470563e-04_r, 2.210490e-01_r, 1.715911e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{4.239024e-01_r, 4.563891e-01_r, -4.822287e-02_r},
                        Real3{9.667238e-01_r, 5.555378e-01_r, -2.461142e-01_r},
                        Real3{1.281368e-01_r, 8.171223e-02_r, 3.871810e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{-2.059705e-01_r, -4.822287e-02_r, 3.805739e-01_r},
                        Real3{-4.107880e-01_r, -2.561329e-01_r, 1.128804e-01_r},
                        Real3{9.940651e-01_r, 2.076091e-01_r, -3.082948e-01_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{6.496927e-01_r, 9.667238e-01_r, -4.107880e-01_r},
                        Real3{5.902729e-01_r, 8.514434e-01_r, -3.560586e-01_r},
                        Real3{2.680671e-04_r, 7.965541e-02_r, 6.155833e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{1.558210e+00_r, 5.555378e-01_r, -2.561329e-01_r},
                        Real3{8.514434e-01_r, 3.466806e+00_r, -1.300406e+00_r},
                        Real3{4.655965e-02_r, 4.573901e-01_r, 2.178634e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{-6.553938e-01_r, -2.461142e-01_r, 1.128804e-01_r},
                        Real3{-3.560586e-01_r, -1.300406e+00_r, 9.009498e-01_r},
                        Real3{3.564617e-01_r, 1.181716e+00_r, -1.485226e+00_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{7.470563e-04_r, 1.281368e-01_r, 9.940651e-01_r},
                        Real3{2.680671e-04_r, 4.655965e-02_r, 3.564617e-01_r},
                        Real3{3.571432e-01_r, 1.440040e-04_r, 1.118187e-03_r}},
                    NdArray<real, 3, 3>{
                        Real3{2.210490e-01_r, 8.171223e-02_r, 2.076091e-01_r},
                        Real3{7.965541e-02_r, 4.573901e-01_r, 1.181716e+00_r},
                        Real3{1.440040e-04_r, 4.244189e-01_r, 5.223965e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{1.715911e+00_r, 3.871810e-01_r, -3.082948e-01_r},
                        Real3{6.155833e-01_r, 2.178634e+00_r, -1.485226e+00_r},
                        Real3{1.118187e-03_r, 5.223965e-01_r, 4.413538e+00_r}}},
            }}, // End of TestData
    // Test : isochoric
    // Deformation gradient isochoric
    TestData{
        .deformationGradient =
            NdArray<real, 3, 3>{
                Real3{1.163793e+00_r, 5.377804e-01_r, 2.805326e-01_r},
                Real3{1.739444e-01_r, 9.266909e-01_r, 5.663804e-01_r},
                Real3{1.193923e-01_r, 2.049263e-01_r, 1.136880e+00_r}},
        // Strain Energy
        .strainEnergy = 2.28726e-01_r,
        // Material PK1 isochoric
        .piolaKirchhoffStress =
            NdArray<real, 3, 3>{
                Real3{8.082960e-02_r, 2.385403e-01_r, 1.269738e-01_r},
                Real3{2.599456e-01_r, -1.296101e-01_r, 2.645233e-01_r},
                Real3{2.670398e-02_r, 2.911709e-01_r, 5.426693e-02_r}},
        // Material Tangent isochoric
        .tangent =
            NdArray<real, 3, 3, 3, 3>{
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{1.926518e+00_r, -2.178487e-01_r, -1.255440e-01_r},
                        Real3{-9.272636e-01_r, 1.752830e+00_r, -2.185743e-01_r},
                        Real3{7.469827e-02_r, -8.194838e-01_r, 1.317869e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{-2.178487e-01_r, 3.873829e-01_r, 1.742706e-02_r},
                        Real3{5.347442e-01_r, -2.996762e-01_r, -2.139911e-03_r},
                        Real3{-2.126477e-01_r, 1.418333e-01_r, -1.667549e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{-1.255440e-01_r, 1.742706e-02_r, 3.671859e-01_r},
                        Real3{9.895790e-04_r, -1.300603e-01_r, 2.333986e-02_r},
                        Real3{3.249855e-01_r, 1.961408e-02_r, -1.319001e-01_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{-9.272636e-01_r, 5.347442e-01_r, 9.895790e-04_r},
                        Real3{9.050156e-01_r, -1.275559e+00_r, 1.723874e-01_r},
                        Real3{-4.413539e-02_r, 5.035171e-01_r, -7.821448e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{1.752830e+00_r, -2.996762e-01_r, -1.300603e-01_r},
                        Real3{-1.275559e+00_r, 3.326904e+00_r, -4.013530e-01_r},
                        Real3{2.029464e-01_r, -1.405553e+00_r, 1.852517e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{-2.185743e-01_r, -2.139911e-03_r, 2.333986e-02_r},
                        Real3{1.723874e-01_r, -4.013530e-01_r, 4.113843e-01_r},
                        Real3{-2.059516e-01_r, 6.055960e-01_r, -3.065337e-01_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{7.469827e-02_r, -2.126477e-01_r, 3.249855e-01_r},
                        Real3{-4.413539e-02_r, 2.029464e-01_r, -2.059516e-01_r},
                        Real3{3.606983e-01_r, -4.863330e-02_r, 7.848010e-02_r}},
                    NdArray<real, 3, 3>{
                        Real3{-8.194838e-01_r, 1.418333e-01_r, 1.961408e-02_r},
                        Real3{5.035171e-01_r, -1.405553e+00_r, 6.055960e-01_r},
                        Real3{-4.863330e-02_r, 1.022375e+00_r, -1.073493e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{1.317869e+00_r, -1.667549e-01_r, -1.319001e-01_r},
                        Real3{-7.821448e-01_r, 1.852517e+00_r, -3.065337e-01_r},
                        Real3{7.848010e-02_r, -1.073493e+00_r, 2.089450e+00_r}}},
            }}, // End of TestData
    // Test : symmetric
    // Deformation gradient symmetric
    TestData{
        .deformationGradient =
            NdArray<real, 3, 3>{
                Real3{2.836028e+00_r, 2.672135e+00_r, 2.230825e+00_r},
                Real3{1.742152e+00_r, 2.972575e+00_r, 2.018759e+00_r},
                Real3{2.344743e+00_r, 2.911901e+00_r, 3.700514e+00_r}},
        // Strain Energy
        .strainEnergy = 1.24008e+01_r,
        // Material PK1 symmetric
        .piolaKirchhoffStress =
            NdArray<real, 3, 3>{
                Real3{2.924406e+00_r, 3.148488e-01_r, 8.872665e-02_r},
                Real3{-6.439179e-01_r, 3.026328e+00_r, -2.277062e-02_r},
                Real3{3.757623e-01_r, 3.536641e-01_r, 2.730566e+00_r}},
        // Material Tangent symmetric
        .tangent =
            NdArray<real, 3, 3, 3, 3>{
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{-2.085955e-01_r, 1.892617e-01_r, 2.095384e-01_r},
                        Real3{3.747189e-01_r, 7.996681e-01_r, -8.666833e-01_r},
                        Real3{1.366286e-01_r, -5.503416e-01_r, 6.924571e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{1.892617e-01_r, 2.938274e-01_r, -7.009883e-02_r},
                        Real3{-1.506496e+00_r, 1.945246e-01_r, 8.014861e-01_r},
                        Real3{7.077512e-01_r, -6.795070e-02_r, -5.107205e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{2.095384e-01_r, -7.009883e-02_r, 2.795340e-01_r},
                        Real3{9.480165e-01_r, -6.597604e-01_r, -8.152877e-02_r},
                        Real3{-1.160055e+00_r, 5.749905e-01_r, 1.544461e-01_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{3.747189e-01_r, -1.506496e+00_r, 9.480165e-01_r},
                        Real3{1.089464e-01_r, 3.851389e-01_r, -1.457985e-01_r},
                        Real3{-9.049645e-02_r, 6.980725e-01_r, -7.211203e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{7.996681e-01_r, 1.945246e-01_r, -6.597604e-01_r},
                        Real3{3.851389e-01_r, -2.404964e-01_r, 2.262428e-01_r},
                        Real3{-6.921801e-01_r, 2.087654e-01_r, 6.298981e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{-8.666833e-01_r, 8.014861e-01_r, -8.152877e-02_r},
                        Real3{-1.457985e-01_r, 2.262428e-01_r, 2.714962e-01_r},
                        Real3{9.441568e-01_r, -1.137517e+00_r, 1.622469e-01_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{1.366286e-01_r, 7.077512e-01_r, -1.160055e+00_r},
                        Real3{-9.049645e-02_r, -6.921801e-01_r, 9.441568e-01_r},
                        Real3{3.241464e-01_r, -4.905381e-02_r, 1.007059e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{-5.503416e-01_r, -6.795070e-02_r, 5.749905e-01_r},
                        Real3{6.980725e-01_r, 2.087654e-01_r, -1.137517e+00_r},
                        Real3{-4.905381e-02_r, 2.842176e-01_r, 1.497132e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{6.924571e-01_r, -5.107205e-01_r, 1.544461e-01_r},
                        Real3{-7.211203e-01_r, 6.298981e-01_r, 1.622469e-01_r},
                        Real3{1.007059e-01_r, 1.497132e-01_r, 4.978631e-02_r}}},
            }}, // End of TestData
    // Test : dilation
    // Deformation gradient dilation
    TestData{
        .deformationGradient =
            NdArray<real, 3, 3>{
                Real3{4.000000e+00_r, 8.326673e-16_r, 6.661338e-16_r},
                Real3{3.469447e-16_r, 4.000000e+00_r, 1.332268e-15_r},
                Real3{-5.551115e-17_r, 0.000000e+00_r, 4.000000e+00_r}},
        // Strain Energy
        .strainEnergy = 1.89049e+01_r,
        // Material PK1 dilation
        .piolaKirchhoffStress =
            NdArray<real, 3, 3>{
                Real3{2.824601e+00_r, 1.762949e-16_r, 2.572787e-16_r},
                Real3{-1.666982e-16_r, 2.824601e+00_r, 4.758099e-16_r},
                Real3{-2.523111e-16_r, -4.649713e-16_r, 2.824601e+00_r}},
        // Material Tangent dilation
        .tangent =
            NdArray<real, 3, 3, 3, 3>{
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{9.742115e-02_r, 2.252727e-17_r, -3.604363e-18_r},
                        Real3{5.406544e-17_r, 8.928571e-02_r, 7.503087e-34_r},
                        Real3{4.325235e-17_r, -2.973812e-17_r, 8.928571e-02_r}},
                    NdArray<real, 3, 3>{
                        Real3{2.252727e-17_r, 3.571429e-01_r, 3.126286e-34_r},
                        Real3{-3.490074e-01_r, 2.252727e-17_r, -4.843451e-18_r},
                        Real3{1.162428e-16_r, -7.503087e-33_r, -7.744301e-18_r}},
                    NdArray<real, 3, 3>{
                        Real3{-3.604363e-18_r, 3.126286e-34_r, 3.571429e-01_r},
                        Real3{7.503087e-34_r, 1.239088e-18_r, 1.041263e-50_r},
                        Real3{-3.490074e-01_r, 3.027157e-17_r, -3.604363e-18_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{5.406544e-17_r, -3.490074e-01_r, 7.503087e-34_r},
                        Real3{3.571429e-01_r, 5.406544e-17_r, -1.561894e-49_r},
                        Real3{-9.003705e-33_r, 5.812141e-17_r, -1.858632e-17_r}},
                    NdArray<real, 3, 3>{
                        Real3{8.928571e-02_r, 2.252727e-17_r, 1.239088e-18_r},
                        Real3{5.406544e-17_r, 9.742115e-02_r, 7.503087e-34_r},
                        Real3{-1.486906e-17_r, 8.650470e-17_r, 8.928571e-02_r}},
                    NdArray<real, 3, 3>{
                        Real3{7.503087e-34_r, -4.843451e-18_r, 1.041263e-50_r},
                        Real3{-1.561894e-49_r, 7.503087e-34_r, 3.571429e-01_r},
                        Real3{7.265176e-17_r, -3.490074e-01_r, 7.503087e-34_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{4.325235e-17_r, 1.162428e-16_r, -3.490074e-01_r},
                        Real3{-9.003705e-33_r, -1.486906e-17_r, 7.265176e-17_r},
                        Real3{3.571429e-01_r, -1.440593e-32_r, 4.325235e-17_r}},
                    NdArray<real, 3, 3>{
                        Real3{-2.973812e-17_r, -7.503087e-33_r, 3.027157e-17_r},
                        Real3{5.812141e-17_r, 8.650470e-17_r, -3.490074e-01_r},
                        Real3{-1.440593e-32_r, 3.571429e-01_r, 8.650470e-17_r}},
                    NdArray<real, 3, 3>{
                        Real3{8.928571e-02_r, -7.744301e-18_r, -3.604363e-18_r},
                        Real3{-1.858632e-17_r, 8.928571e-02_r, 7.503087e-34_r},
                        Real3{4.325235e-17_r, 8.650470e-17_r, 9.742115e-02_r}}},
            }}, // End of TestData
    // Test : shear
    // Deformation gradient shear
    TestData{
        .deformationGradient =
            NdArray<real, 3, 3>{
                Real3{1.000000e+00_r, 5.000000e-01_r, 2.914335e-16_r},
                Real3{5.000000e-01_r, 1.000000e+00_r, 5.000000e-01_r},
                Real3{2.500000e-01_r, 5.000000e-01_r, 1.000000e+00_r}},
        // Strain Energy
        .strainEnergy = 6.31679e-01_r,
        // Material PK1 shear
        .piolaKirchhoffStress =
            NdArray<real, 3, 3>{
                Real3{-1.214979e+00_r, 9.646325e-01_r, -1.227701e-17_r},
                Real3{1.226653e+00_r, -1.739020e+00_r, 9.646325e-01_r},
                Real3{-4.347550e-01_r, 1.226653e+00_r, -1.214979e+00_r}},
        // Material Tangent shear
        .tangent =
            NdArray<real, 3, 3, 3, 3>{
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{4.992988e+00_r, -2.317923e+00_r, 3.431215e-16_r},
                        Real3{-3.090564e+00_r, 4.084964e+00_r, -1.269841e+00_r},
                        Real3{1.545282e+00_r, -2.042482e+00_r, 2.539683e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{-2.317923e+00_r, 1.516104e+00_r, -1.715607e-16_r},
                        Real3{3.641445e+00_r, -3.090564e+00_r, 6.349206e-01_r},
                        Real3{-1.820722e+00_r, 1.545282e+00_r, -1.269841e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{3.431215e-16_r, -1.715607e-16_r, 3.571429e-01_r},
                        Real3{-1.048081e+00_r, 5.240407e-01_r, -1.715607e-16_r},
                        Real3{2.096163e+00_r, -1.048081e+00_r, 3.431215e-16_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{-3.090564e+00_r, 3.641445e+00_r, -1.048081e+00_r},
                        Real3{2.417519e+00_r, -4.120752e+00_r, 1.545282e+00_r},
                        Real3{-1.030188e+00_r, 2.060376e+00_r, -2.042482e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{4.084964e+00_r, -3.090564e+00_r, 5.240407e-01_r},
                        Real3{-4.120752e+00_r, 8.598646e+00_r, -3.090564e+00_r},
                        Real3{2.060376e+00_r, -4.120752e+00_r, 4.084964e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{-1.269841e+00_r, 6.349206e-01_r, -1.715607e-16_r},
                        Real3{1.545282e+00_r, -3.090564e+00_r, 1.516104e+00_r},
                        Real3{-1.820722e+00_r, 3.641445e+00_r, -2.317923e+00_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{1.545282e+00_r, -1.820722e+00_r, 2.096163e+00_r},
                        Real3{-1.030188e+00_r, 2.060376e+00_r, -1.820722e+00_r},
                        Real3{8.722368e-01_r, -1.030188e+00_r, 1.545282e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{-2.042482e+00_r, 1.545282e+00_r, -1.048081e+00_r},
                        Real3{2.060376e+00_r, -4.120752e+00_r, 3.641445e+00_r},
                        Real3{-1.030188e+00_r, 2.417519e+00_r, -3.090564e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{2.539683e+00_r, -1.269841e+00_r, 3.431215e-16_r},
                        Real3{-2.042482e+00_r, 4.084964e+00_r, -2.317923e+00_r},
                        Real3{1.545282e+00_r, -3.090564e+00_r, 4.992988e+00_r}}},
            }}, // End of TestData
}; // end of std::array<TestData>
} // namespace neo_hookean_test_data

namespace smith_neo_hookean_test_data {
// The data was acquired for Youngs Modulus = 1.0 and poisson = 0.4
constexpr int kNumTests = 5;
constexpr std::array<TestData, kNumTests> kTestData{
    // Test : random
    // Deformation gradient random
    TestData{
        .deformationGradient =
            NdArray<real, 3, 3>{
                Real3{1.439129e+00_r, -1.663222e-01_r, 4.140516e-01_r},
                Real3{-5.247711e-01_r, 9.343632e-01_r, 4.117087e-02_r},
                Real3{4.285113e-01_r, 5.002626e-02_r, 1.372375e+00_r}},
        // Strain Energy
        .strainEnergy = 2.45060e-01_r,
        // Material PK1 random
        .piolaKirchhoffStress =
            NdArray<real, 3, 3>{
                Real3{1.322600e+00_r, 3.611480e-01_r, -8.078288e-02_r},
                Real3{-6.691376e-02_r, 1.419487e+00_r, -6.654972e-02_r},
                Real3{-5.585820e-02_r, -1.403151e-01_r, 1.282452e+00_r}},
        // Material Tangent random
        .tangent =
            NdArray<real, 3, 3, 3, 3>{
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{3.278884e+00_r, 1.625130e+00_r, -9.293660e-01_r},
                        Real3{5.331295e-01_r, 4.799338e+00_r, -3.442959e-01_r},
                        Real3{-8.561549e-01_r, -6.333809e-01_r, 3.365595e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{1.625130e+00_r, 1.343154e+00_r, -5.449339e-01_r},
                        Real3{-4.772940e-01_r, 2.285954e+00_r, 6.603560e-02_r},
                        Real3{-4.791758e-01_r, -3.523872e-01_r, 1.900795e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{-9.293660e-01_r, -5.449339e-01_r, 7.208928e-01_r},
                        Real3{-1.592397e-01_r, -1.563764e+00_r, 1.058942e-01_r},
                        Real3{-2.482210e-01_r, -1.003916e-01_r, -9.131640e-01_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{5.331295e-01_r, -4.772940e-01_r, -1.592397e-01_r},
                        Real3{5.160424e-01_r, 7.614653e-01_r, -6.205954e-02_r},
                        Real3{-1.742946e-01_r, 1.208272e-01_r, 6.206155e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{4.799338e+00_r, 2.285954e+00_r, -1.563764e+00_r},
                        Real3{7.614653e-01_r, 6.000499e+00_r, -4.436816e-01_r},
                        Real3{-1.452931e+00_r, -8.570227e-01_r, 4.765765e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{-3.442959e-01_r, 6.603560e-02_r, 1.058942e-01_r},
                        Real3{-6.205954e-02_r, -4.436816e-01_r, 4.382840e-01_r},
                        Real3{1.254092e-03_r, -7.666645e-01_r, -3.096783e-01_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{-8.561549e-01_r, -4.791758e-01_r, -2.482210e-01_r},
                        Real3{-1.742946e-01_r, -1.452931e+00_r, 1.254092e-03_r},
                        Real3{6.745571e-01_r, 1.884272e-01_r, -8.412730e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{-6.333809e-01_r, -3.523872e-01_r, -1.003916e-01_r},
                        Real3{1.208272e-01_r, -8.570227e-01_r, -7.666645e-01_r},
                        Real3{1.884272e-01_r, 5.348746e-01_r, -5.986591e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{3.365595e+00_r, 1.900795e+00_r, -9.131640e-01_r},
                        Real3{6.206155e-01_r, 4.765765e+00_r, -3.096783e-01_r},
                        Real3{-8.412730e-01_r, -5.986591e-01_r, 3.174554e+00_r}}},
            }}, // End of TestData
    // Test : isochoric
    // Deformation gradient isochoric
    TestData{
        .deformationGradient =
            NdArray<real, 3, 3>{
                Real3{9.260246e-01_r, 1.286493e-01_r, 4.345008e-01_r},
                Real3{8.777595e-02_r, 9.852412e-01_r, 4.301151e-01_r},
                Real3{1.822116e-01_r, 3.233756e-01_r, 1.314429e+00_r}},
        // Strain Energy
        .strainEnergy = -9.06400e-02_r,
        // Material PK1 isochoric
        .piolaKirchhoffStress =
            NdArray<real, 3, 3>{
                Real3{-5.847547e-02_r, 6.244553e-02_r, 2.202477e-01_r},
                Real3{4.380106e-02_r, -2.941565e-02_r, 2.631674e-01_r},
                Real3{2.028531e-01_r, 2.523737e-01_r, 1.811820e-01_r}},
        // Material Tangent isochoric
        .tangent =
            NdArray<real, 3, 3, 3, 3>{
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{2.720705e+00_r, -6.945923e-02_r, -2.867960e-01_r},
                        Real3{-5.406876e-02_r, 1.834858e+00_r, -4.206255e-01_r},
                        Real3{-7.375858e-01_r, -5.540382e-01_r, 1.490804e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{-6.945923e-02_r, 3.856403e-01_r, 1.170728e-02_r},
                        Real3{4.716803e-01_r, -6.803484e-02_r, -4.541267e-02_r},
                        Real3{-1.289418e-01_r, 2.453327e-02_r, -1.999498e-02_r}},
                    NdArray<real, 3, 3>{
                        Real3{-2.867960e-01_r, 1.170728e-02_r, 4.290346e-01_r},
                        Real3{-1.066304e-01_r, -2.161005e-01_r, 7.887470e-02_r},
                        Real3{4.520289e-01_r, 6.777555e-02_r, -2.141014e-01_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{-5.406876e-02_r, 4.716803e-01_r, -1.066304e-01_r},
                        Real3{3.843631e-01_r, -5.299341e-02_r, 1.501021e-02_r},
                        Real3{1.898582e-02_r, -1.363596e-01_r, 5.709883e-03_r}},
                    NdArray<real, 3, 3>{
                        Real3{1.834858e+00_r, -6.803484e-02_r, -2.161005e-01_r},
                        Real3{-5.299341e-02_r, 2.653909e+00_r, -5.266439e-01_r},
                        Real3{-5.704810e-01_r, -6.958074e-01_r, 1.486941e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{-4.206255e-01_r, -4.541267e-02_r, 7.887470e-02_r},
                        Real3{1.501021e-02_r, -5.266439e-01_r, 5.209706e-01_r},
                        Real3{1.345314e-01_r, 5.074298e-01_r, -4.085443e-01_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{-7.375858e-01_r, -1.289418e-01_r, 4.520289e-01_r},
                        Real3{1.898582e-02_r, -5.704810e-01_r, 1.345314e-01_r},
                        Real3{6.237350e-01_r, 2.339065e-01_r, -5.709874e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{-5.540382e-01_r, 2.453327e-02_r, 6.777555e-02_r},
                        Real3{-1.363596e-01_r, -6.958074e-01_r, 5.074298e-01_r},
                        Real3{2.339065e-01_r, 6.104214e-01_r, -5.445799e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{1.490804e+00_r, -1.999498e-02_r, -2.141014e-01_r},
                        Real3{5.709883e-03_r, 1.486941e+00_r, -4.085443e-01_r},
                        Real3{-5.709874e-01_r, -5.445799e-01_r, 1.847662e+00_r}}},
            }}, // End of TestData
    // Test : symmetric
    // Deformation gradient symmetric
    TestData{
        .deformationGradient =
            NdArray<real, 3, 3>{
                Real3{2.161820e+00_r, 2.319453e+00_r, 1.979647e+00_r},
                Real3{1.738345e+00_r, 2.230582e+00_r, 8.642735e-01_r},
                Real3{9.047993e-01_r, 2.357503e+00_r, 1.739536e+00_r}},
        // Strain Energy
        .strainEnergy = 8.57673e+00_r,
        // Material PK1 symmetric
        .piolaKirchhoffStress =
            NdArray<real, 3, 3>{
                Real3{6.386793e+00_r, -5.484865e+00_r, 6.996521e+00_r},
                Real3{2.651672e+00_r, 6.789146e+00_r, -8.367464e+00_r},
                Real3{-6.632967e+00_r, 5.688352e+00_r, 3.113838e+00_r}},
        // Material Tangent symmetric
        .tangent =
            NdArray<real, 3, 3, 3, 3>{
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{6.326907e+00_r, -7.126666e+00_r, 6.619466e+00_r},
                        Real3{2.014304e+00_r, 1.135523e+01_r, -1.642787e+01_r},
                        Real3{-7.667515e+00_r, 2.480084e+00_r, 9.039269e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{-7.126666e+00_r, 9.142669e+00_r, -8.045280e+00_r},
                        Real3{-7.530200e+00_r, -7.616975e+00_r, 1.424928e+01_r},
                        Real3{1.186020e+01_r, -6.082367e+00_r, -8.137614e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{6.619466e+00_r, -8.045280e+00_r, 7.932862e+00_r},
                        Real3{9.167001e+00_r, 4.428700e+00_r, -1.076186e+01_r},
                        Real3{-1.517807e+01_r, 1.073475e+01_r, 2.839746e+00_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{2.014304e+00_r, -7.530200e+00_r, 9.167001e+00_r},
                        Real3{1.154465e+00_r, 2.152720e+00_r, -3.270513e+00_r},
                        Real3{-2.630098e+00_r, 7.509261e+00_r, -5.917812e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{1.135523e+01_r, -7.616975e+00_r, 4.428700e+00_r},
                        Real3{2.152720e+00_r, 7.161097e+00_r, -1.018962e+01_r},
                        Real3{-1.398398e+01_r, 5.351702e+00_r, 9.011142e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{-1.642787e+01_r, 1.424928e+01_r, -1.076186e+01_r},
                        Real3{-3.270513e+00_r, -1.018962e+01_r, 1.597599e+01_r},
                        Real3{1.926068e+01_r, -1.445960e+01_r, -4.087396e+00_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{-7.667515e+00_r, 1.186020e+01_r, -1.517807e+01_r},
                        Real3{-2.630098e+00_r, -1.398398e+01_r, 1.926068e+01_r},
                        Real3{1.049775e+01_r, -6.544671e+00_r, -3.287102e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{2.480084e+00_r, -6.082367e+00_r, 1.073475e+01_r},
                        Real3{7.509261e+00_r, 5.351702e+00_r, -1.445960e+01_r},
                        Real3{-6.544671e+00_r, 4.737289e+00_r, 2.148807e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{9.039269e+00_r, -8.137614e+00_r, 2.839746e+00_r},
                        Real3{-5.917812e+00_r, 9.011142e+00_r, -4.087396e+00_r},
                        Real3{-3.287102e+00_r, 2.148807e+00_r, 1.542046e+00_r}}},
            }}, // End of TestData
    // Test : dilation
    // Deformation gradient dilation
    TestData{
        .deformationGradient =
            NdArray<real, 3, 3>{
                Real3{4.000000e+00_r, 8.326673e-16_r, 6.661338e-16_r},
                Real3{3.469447e-16_r, 4.000000e+00_r, 1.332268e-15_r},
                Real3{-5.551115e-17_r, 0.000000e+00_r, 4.000000e+00_r}},
        // Strain Energy
        .strainEnergy = 3.41295e+03_r,
        // Material PK1 dilation
        .piolaKirchhoffStress =
            NdArray<real, 3, 3>{
                Real3{1.736152e+03_r, -1.500369e-13_r, 2.437878e-14_r},
                Real3{-3.608589e-13_r, 1.736152e+03_r, 6.214659e-16_r},
                Real3{-2.888425e-13_r, -5.776332e-13_r, 1.736152e+03_r}},
        // Material Tangent dilation
        .tangent =
            NdArray<real, 3, 3, 3, 3>{
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{4.423776e+02_r, -3.832781e-14_r, 6.133717e-15_r},
                        Real3{-9.198936e-14_r, 8.754825e+02_r, 2.113830e-18_r},
                        Real3{-7.359201e-14_r, -2.915921e-13_r, 8.754825e+02_r}},
                    NdArray<real, 3, 3>{
                        Real3{-3.832781e-14_r, 4.664723e-01_r, -5.317035e-31_r},
                        Real3{-4.335714e+02_r, -3.832781e-14_r, -6.017012e-15_r},
                        Real3{1.444083e-13_r, 1.276616e-29_r, -7.593413e-14_r}},
                    NdArray<real, 3, 3>{
                        Real3{6.133717e-15_r, -5.317035e-31_r, 4.664723e-01_r},
                        Real3{-1.276525e-30_r, 1.215073e-14_r, 3.520234e-34_r},
                        Real3{-4.335714e+02_r, 3.760633e-14_r, 6.133717e-15_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{-9.198936e-14_r, -4.335714e+02_r, -1.276525e-30_r},
                        Real3{4.664723e-01_r, -9.198936e-14_r, 1.833455e-34_r},
                        Real3{1.531939e-29_r, 7.220415e-14_r, -1.822445e-13_r}},
                    NdArray<real, 3, 3>{
                        Real3{8.754825e+02_r, -3.832781e-14_r, 1.215073e-14_r},
                        Real3{-9.198936e-14_r, 4.423776e+02_r, 2.113830e-18_r},
                        Real3{-1.457962e-13_r, -1.471839e-13_r, 8.754825e+02_r}},
                    NdArray<real, 3, 3>{
                        Real3{2.113830e-18_r, -6.017012e-15_r, 3.520234e-34_r},
                        Real3{1.833455e-34_r, 2.113830e-18_r, 4.664723e-01_r},
                        Real3{9.025518e-14_r, -4.335714e+02_r, 2.113830e-18_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{-7.359201e-14_r, 1.444083e-13_r, -4.335714e+02_r},
                        Real3{1.531939e-29_r, -1.457962e-13_r, 9.025518e-14_r},
                        Real3{4.664723e-01_r, 2.451104e-29_r, -7.359201e-14_r}},
                    NdArray<real, 3, 3>{
                        Real3{-2.915921e-13_r, 1.276616e-29_r, 3.760633e-14_r},
                        Real3{7.220415e-14_r, -1.471839e-13_r, -4.335714e+02_r},
                        Real3{2.451104e-29_r, 4.664723e-01_r, -1.471839e-13_r}},
                    NdArray<real, 3, 3>{
                        Real3{8.754825e+02_r, -7.593413e-14_r, 6.133717e-15_r},
                        Real3{-1.822445e-13_r, 8.754825e+02_r, 2.113830e-18_r},
                        Real3{-7.359201e-14_r, -1.471839e-13_r, 4.423776e+02_r}}},
            }}, // End of TestData
    // Test : shear
    // Deformation gradient shear
    TestData{
        .deformationGradient =
            NdArray<real, 3, 3>{
                Real3{1.000000e+00_r, 5.000000e-01_r, 2.914335e-16_r},
                Real3{5.000000e-01_r, 1.000000e+00_r, 5.000000e-01_r},
                Real3{2.500000e-01_r, 5.000000e-01_r, 1.000000e+00_r}},
        // Strain Energy
        .strainEnergy = 2.25217e-01_r,
        // Material PK1 shear
        .piolaKirchhoffStress =
            NdArray<real, 3, 3>{
                Real3{-4.521352e-01_r, 6.081958e-01_r, 4.961707e-17_r},
                Real3{7.472397e-01_r, -7.302230e-01_r, 6.081958e-01_r},
                Real3{-1.825558e-01_r, 7.472397e-01_r, -4.521352e-01_r}},
        // Material Tangent shear
        .tangent =
            NdArray<real, 3, 3, 3, 3>{
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{1.390271e+00_r, -4.669109e-01_r, 8.269691e-17_r},
                        Real3{-6.287412e-01_r, 2.194521e-01_r, 8.926473e-02_r},
                        Real3{3.329508e-01_r, -7.256562e-02_r, -1.042086e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{-4.669109e-01_r, 6.341638e-01_r, -3.051866e-17_r},
                        Real3{1.445302e+00_r, -6.287412e-01_r, -2.605216e-02_r},
                        Real3{-7.133609e-01_r, 3.329508e-01_r, 8.926473e-02_r}},
                    NdArray<real, 3, 3>{
                        Real3{8.269691e-17_r, -3.051866e-17_r, 3.821282e-01_r},
                        Real3{-5.561756e-01_r, 2.780878e-01_r, -3.051866e-17_r},
                        Real3{1.112351e+00_r, -5.561756e-01_r, 8.269691e-17_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{-6.287412e-01_r, 1.445302e+00_r, -5.561756e-01_r},
                        Real3{8.229659e-01_r, -8.445150e-01_r, 3.329508e-01_r},
                        Real3{-2.111288e-01_r, 4.408377e-01_r, -7.256562e-02_r}},
                    NdArray<real, 3, 3>{
                        Real3{2.194521e-01_r, -6.287412e-01_r, 2.780878e-01_r},
                        Real3{-8.445150e-01_r, 2.145479e+00_r, -6.287412e-01_r},
                        Real3{4.408377e-01_r, -8.445150e-01_r, 2.194521e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{8.926473e-02_r, -2.605216e-02_r, -3.051866e-17_r},
                        Real3{3.329508e-01_r, -6.287412e-01_r, 6.341638e-01_r},
                        Real3{-7.133609e-01_r, 1.445302e+00_r, -4.669109e-01_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{3.329508e-01_r, -7.133609e-01_r, 1.112351e+00_r},
                        Real3{-2.111288e-01_r, 4.408377e-01_r, -7.133609e-01_r},
                        Real3{4.923376e-01_r, -2.111288e-01_r, 3.329508e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{-7.256562e-02_r, 3.329508e-01_r, -5.561756e-01_r},
                        Real3{4.408377e-01_r, -8.445150e-01_r, 1.445302e+00_r},
                        Real3{-2.111288e-01_r, 8.229659e-01_r, -6.287412e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{-1.042086e-01_r, 8.926473e-02_r, 8.269691e-17_r},
                        Real3{-7.256562e-02_r, 2.194521e-01_r, -4.669109e-01_r},
                        Real3{3.329508e-01_r, -6.287412e-01_r, 1.390271e+00_r}}},
            }}, // End of TestData
}; // end of std::array<TestData>
} // namespace smith_neo_hookean_test_data

namespace st_venant_kirchhoff_test_data {
// The data was acquired for Youngs Modulus = 1.0 and poisson = 0.4
constexpr int kNumTests = 6;
std::array<TestData, kNumTests> const kTestData{
    // Test : // Deformation gradient identity
    TestData{
        .deformationGradient =
            NdArray<real, 3, 3>{
                Real3{1.000000e+00_r, 0.000000e+00_r, 0.000000e+00_r},
                Real3{0.000000e+00_r, 1.000000e+00_r, 0.000000e+00_r},
                Real3{0.000000e+00_r, 0.000000e+00_r, 1.000000e+00_r}},
        // Strain Energy
        .strainEnergy = 0.00000e+00,
        // Material PK1 identity
        .piolaKirchhoffStress =
            NdArray<real, 3, 3>{
                Real3{0.000000e+00_r, 0.000000e+00_r, 0.000000e+00_r},
                Real3{0.000000e+00_r, 0.000000e+00_r, 0.000000e+00_r},
                Real3{0.000000e+00_r, 0.000000e+00_r, 0.000000e+00_r}},
        // Material Tangent identity
        .tangent =
            NdArray<real, 3, 3, 3, 3>{
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{2.142857e+00_r, 0.000000e+00_r, 0.000000e+00_r},
                        Real3{0.000000e+00_r, 1.428571e+00_r, 0.000000e+00_r},
                        Real3{0.000000e+00_r, 0.000000e+00_r, 1.428571e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{0.000000e+00_r, 3.571429e-01_r, 0.000000e+00_r},
                        Real3{3.571429e-01_r, 0.000000e+00_r, 0.000000e+00_r},
                        Real3{0.000000e+00_r, 0.000000e+00_r, 0.000000e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{0.000000e+00_r, 0.000000e+00_r, 3.571429e-01_r},
                        Real3{0.000000e+00_r, 0.000000e+00_r, 0.000000e+00_r},
                        Real3{3.571429e-01_r, 0.000000e+00_r, 0.000000e+00_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{0.000000e+00_r, 3.571429e-01_r, 0.000000e+00_r},
                        Real3{3.571429e-01_r, 0.000000e+00_r, 0.000000e+00_r},
                        Real3{0.000000e+00_r, 0.000000e+00_r, 0.000000e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{1.428571e+00_r, 0.000000e+00_r, 0.000000e+00_r},
                        Real3{0.000000e+00_r, 2.142857e+00_r, 0.000000e+00_r},
                        Real3{0.000000e+00_r, 0.000000e+00_r, 1.428571e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{0.000000e+00_r, 0.000000e+00_r, 0.000000e+00_r},
                        Real3{0.000000e+00_r, 0.000000e+00_r, 3.571429e-01_r},
                        Real3{0.000000e+00_r, 3.571429e-01_r, 0.000000e+00_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{0.000000e+00_r, 0.000000e+00_r, 3.571429e-01_r},
                        Real3{0.000000e+00_r, 0.000000e+00_r, 0.000000e+00_r},
                        Real3{3.571429e-01_r, 0.000000e+00_r, 0.000000e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{0.000000e+00_r, 0.000000e+00_r, 0.000000e+00_r},
                        Real3{0.000000e+00_r, 0.000000e+00_r, 3.571429e-01_r},
                        Real3{0.000000e+00_r, 3.571429e-01_r, 0.000000e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{1.428571e+00_r, 0.000000e+00_r, 0.000000e+00_r},
                        Real3{0.000000e+00_r, 1.428571e+00_r, 0.000000e+00_r},
                        Real3{0.000000e+00_r, 0.000000e+00_r, 2.142857e+00_r}}},
            }}, // End of TestData
    // Test : random
    // Deformation gradient random
    TestData{
        .deformationGradient =
            NdArray<real, 3, 3>{
                Real3{1.561664e+00_r, 3.437668e-01_r, -3.349611e-01_r},
                Real3{-8.652070e-02_r, 1.438568e+00_r, 3.029725e-02_r},
                Real3{3.236146e-01_r, 3.358737e-01_r, 1.517437e+00_r}},
        // Strain Energy
        .strainEnergy = 3.87933e+00,
        // Material PK1 random
        .piolaKirchhoffStress =
            NdArray<real, 3, 3>{
                Real3{5.693126e+00_r, 1.445678e+00_r, -1.155864e+00_r},
                Real3{-4.430136e-02_r, 5.041542e+00_r, 3.338220e-01_r},
                Real3{1.209363e+00_r, 1.477391e+00_r, 5.440944e+00_r}},
        // Material Tangent random
        .tangent =
            NdArray<real, 3, 3, 3, 3>{
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{8.910194e+00_r, 1.144756e+00_r, -9.464769e-01_r},
                        Real3{-1.165407e-01_r, 3.198748e+00_r, 7.794196e-02_r},
                        Real3{9.426579e-01_r, 7.890481e-01_r, 3.346610e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{1.144756e+00_r, 4.676755e+00_r, -4.915678e-02_r},
                        Real3{7.598526e-01_r, 1.007831e+00_r, -1.572155e-01_r},
                        Real3{3.462548e-01_r, 2.463814e-01_r, 7.050262e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{-9.464769e-01_r, -4.915678e-02_r, 4.707250e+00_r},
                        Real3{5.829942e-02_r, -6.846578e-01_r, 1.066162e-01_r},
                        Real3{6.914762e-01_r, 2.558075e-02_r, -8.674483e-01_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{-1.165407e-01_r, 7.598526e-01_r, 5.829942e-02_r},
                        Real3{4.357400e+00_r, -3.616207e-02_r, -1.705704e-02_r},
                        Real3{1.289840e-01_r, 1.247505e-01_r, -1.840551e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{3.198748e+00_r, 1.007831e+00_r, -6.846578e-01_r},
                        Real3{-3.616207e-02_r, 7.950050e+00_r, 2.342952e-01_r},
                        Real3{6.546807e-01_r, 1.041799e+00_r, 3.122115e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{7.794196e-02_r, -1.572155e-01_r, 1.066162e-01_r},
                        Real3{-1.705704e-02_r, 2.342952e-01_r, 4.297360e+00_r},
                        Real3{-3.288258e-02_r, 7.941574e-01_r, 2.610796e-01_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{9.426579e-01_r, 3.462548e-01_r, 6.914762e-01_r},
                        Real3{1.289840e-01_r, 6.546807e-01_r, -3.288258e-02_r},
                        Real3{4.688998e+00_r, 3.801942e-01_r, 8.645253e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{7.890481e-01_r, 2.463814e-01_r, 2.558075e-02_r},
                        Real3{1.247505e-01_r, 1.041799e+00_r, 7.941574e-01_r},
                        Real3{3.801942e-01_r, 4.613957e+00_r, 1.066586e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{3.346610e+00_r, 7.050262e-01_r, -8.674483e-01_r},
                        Real3{-1.840551e-01_r, 3.122115e+00_r, 2.610796e-01_r},
                        Real3{8.645253e-01_r, 1.066586e+00_r, 8.565489e+00_r}}},
            }}, // End of TestData
    // Test : isochoric
    // Deformation gradient isochoric
    TestData{
        .deformationGradient =
            NdArray<real, 3, 3>{
                Real3{1.006409e+00_r, 2.846638e-01_r, 5.985211e-01_r},
                Real3{5.655850e-01_r, 1.211855e+00_r, 2.288555e-01_r},
                Real3{7.161474e-01_r, 1.361985e-01_r, 1.377308e+00_r}},
        // Strain Energy
        .strainEnergy = 2.37125e+00,
        // Material PK1 isochoric
        .piolaKirchhoffStress =
            NdArray<real, 3, 3>{
                Real3{2.736225e+00_r, 1.131293e+00_r, 2.125078e+00_r},
                Real3{1.873501e+00_r, 2.869501e+00_r, 1.173751e+00_r},
                Real3{2.505523e+00_r, 8.784140e-01_r, 3.790802e+00_r}},
        // Material Tangent isochoric
        .tangent =
            NdArray<real, 3, 3, 3, 3>{
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{4.573131e+00_r, 8.935266e-01_r, 1.689263e+00_r},
                        Real3{1.391859e+00_r, 1.799817e+00_r, 4.499298e-01_r},
                        Real3{1.852694e+00_r, 2.686239e-01_r, 2.133275e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{8.935266e-01_r, 2.810072e+00_r, 5.311395e-01_r},
                        Real3{6.655814e-01_r, 9.914328e-01_r, 3.521101e-01_r},
                        Real3{3.401844e-01_r, 6.348964e-01_r, 5.892132e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{1.689263e+00_r, 5.311395e-01_r, 3.569134e+00_r},
                        Real3{5.658502e-01_r, 1.059439e+00_r, 6.200108e-01_r},
                        Real3{1.107376e+00_r, 2.564788e-01_r, 2.037713e+00_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{1.391859e+00_r, 6.655814e-01_r, 5.658502e-01_r},
                        Real3{3.474513e+00_r, 1.605881e+00_r, 8.447629e-01_r},
                        Real3{1.039468e+00_r, 4.199979e-01_r, 1.171369e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{1.799817e+00_r, 9.914328e-01_r, 1.059439e+00_r},
                        Real3{1.605881e+00_r, 5.426688e+00_r, 7.221437e-01_r},
                        Real3{1.267321e+00_r, 6.109156e-01_r, 2.395557e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{4.499298e-01_r, 3.521101e-01_r, 6.200108e-01_r},
                        Real3{8.447629e-01_r, 7.221437e-01_r, 3.161802e+00_r},
                        Real3{5.123436e-01_r, 6.406346e-01_r, 8.790437e-01_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{1.852694e+00_r, 3.401844e-01_r, 1.107376e+00_r},
                        Real3{1.039468e+00_r, 1.267321e+00_r, 5.123436e-01_r},
                        Real3{4.028958e+00_r, 5.561159e-01_r, 2.374975e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{2.686239e-01_r, 6.348964e-01_r, 2.564788e-01_r},
                        Real3{4.199979e-01_r, 6.109156e-01_r, 6.406346e-01_r},
                        Real3{5.561159e-01_r, 3.047164e+00_r, 5.618717e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{2.133275e+00_r, 5.892132e-01_r, 2.037713e+00_r},
                        Real3{1.171369e+00_r, 2.395557e+00_r, 8.790437e-01_r},
                        Real3{2.374975e+00_r, 5.618717e-01_r, 6.665572e+00_r}}},
            }}, // End of TestData
    // Test : symmetric
    // Deformation gradient symmetric
    TestData{
        .deformationGradient =
            NdArray<real, 3, 3>{
                Real3{2.496155e+00_r, 1.421208e+00_r, 5.086858e-01_r},
                Real3{2.463099e+00_r, 2.585904e+00_r, 1.073106e+00_r},
                Real3{1.779090e+00_r, 2.050618e+00_r, 3.213882e+00_r}},
        // Strain Energy
        .strainEnergy = 3.55211e+02,
        // Material PK1 symmetric
        .piolaKirchhoffStress =
            NdArray<real, 3, 3>{
                Real3{8.770124e+01_r, 5.764812e+01_r, 2.914249e+01_r},
                Real3{9.423812e+01_r, 9.535245e+01_r, 5.035276e+01_r},
                Real3{7.734198e+01_r, 8.328420e+01_r, 1.110342e+02_r}},
        // Material Tangent symmetric
        .tangent =
            NdArray<real, 3, 3, 3, 3>{
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{4.584071e+01_r, 1.117961e+01_r, 5.706966e+00_r},
                        Real3{1.468237e+01_r, 1.047137e+01_r, 4.274108e+00_r},
                        Real3{1.114090e+01_r, 8.215391e+00_r, 1.178371e+01_r}},
                    NdArray<real, 3, 3>{
                        Real3{1.117961e+01_r, 3.741005e+01_r, 4.893958e+00_r},
                        Real3{7.306113e+00_r, 1.026600e+01_r, 2.648514e+00_r},
                        Real3{5.440174e+00_r, 8.414951e+00_r, 6.897676e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{5.706966e+00_r, 4.893958e+00_r, 3.384657e+01_r},
                        Real3{2.746576e+00_r, 2.423842e+00_r, 4.678081e+00_r},
                        Real3{4.157978e+00_r, 3.121455e+00_r, 6.130134e+00_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{1.468237e+01_r, 7.306113e+00_r, 2.746576e+00_r},
                        Real3{4.747509e+01_r, 1.621850e+01_r, 8.159481e+00_r},
                        Real3{1.251571e+01_r, 8.858591e+00_r, 1.199057e+01_r}},
                    NdArray<real, 3, 3>{
                        Real3{1.047137e+01_r, 1.026600e+01_r, 2.423842e+00_r},
                        Real3{1.621850e+01_r, 4.767122e+01_r, 8.558247e+00_r},
                        Real3{8.376107e+00_r, 1.415968e+01_r, 1.265846e+01_r}},
                    NdArray<real, 3, 3>{
                        Real3{4.274108e+00_r, 2.648514e+00_r, 4.678081e+00_r},
                        Real3{8.159481e+00_r, 8.558247e+00_r, 3.736797e+01_r},
                        Real3{5.554543e+00_r, 6.111755e+00_r, 1.084921e+01_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{1.114090e+01_r, 5.440174e+00_r, 4.157978e+00_r},
                        Real3{1.251571e+01_r, 8.376107e+00_r, 5.554543e+00_r},
                        Real3{4.364846e+01_r, 1.135939e+01_r, 1.364987e+01_r}},
                    NdArray<real, 3, 3>{
                        Real3{8.215391e+00_r, 8.414951e+00_r, 3.121455e+00_r},
                        Real3{8.858591e+00_r, 1.415968e+01_r, 6.111755e+00_r},
                        Real3{1.135939e+01_r, 4.459429e+01_r, 1.537163e+01_r}},
                    NdArray<real, 3, 3>{
                        Real3{1.178371e+01_r, 6.897676e+00_r, 6.130134e+00_r},
                        Real3{1.199057e+01_r, 1.265846e+01_r, 1.084921e+01_r},
                        Real3{1.364987e+01_r, 1.537163e+01_r, 5.511129e+01_r}}},
            }}, // End of TestData
    // Test : dilation
    // Deformation gradient dilation
    TestData{
        .deformationGradient =
            NdArray<real, 3, 3>{
                Real3{4.000000e+00_r, 8.326673e-16_r, 6.661338e-16_r},
                Real3{3.469447e-16_r, 4.000000e+00_r, 1.332268e-15_r},
                Real3{-5.551115e-17_r, 0.000000e+00_r, 4.000000e+00_r}},
        // Strain Energy
        .strainEnergy = 4.21875e+02,
        // Material PK1 dilation
        .piolaKirchhoffStress =
            NdArray<real, 3, 3>{
                Real3{1.500000e+02_r, 3.796566e-14_r, 2.846929e-14_r},
                Real3{1.975107e-14_r, 1.500000e+02_r, 5.757299e-14_r},
                Real3{1.407604e-15_r, 7.612958e-15_r, 1.500000e+02_r}},
        // Material Tangent dilation
        .tangent =
            NdArray<real, 3, 3, 3, 3>{
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{7.178571e+01_r, 7.632783e-15_r, 5.630417e-15_r},
                        Real3{4.163336e-15_r, 2.285714e+01_r, 7.612958e-15_r},
                        Real3{4.758099e-16_r, -1.650797e-32_r, 2.285714e+01_r}},
                    NdArray<real, 3, 3>{
                        Real3{7.632783e-15_r, 4.321429e+01_r, 1.903239e-15_r},
                        Real3{5.714286e+00_r, 7.632783e-15_r, 9.516197e-16_r},
                        Real3{-6.603188e-32_r, 8.723181e-16_r, 4.758099e-15_r}},
                    NdArray<real, 3, 3>{
                        Real3{5.630417e-15_r, 1.903239e-15_r, 4.321429e+01_r},
                        Real3{1.903239e-15_r, 3.806479e-15_r, 1.685160e-15_r},
                        Real3{5.714286e+00_r, 1.189525e-15_r, 5.630417e-15_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{4.163336e-15_r, 5.714286e+00_r, 1.903239e-15_r},
                        Real3{4.321429e+01_r, 4.163336e-15_r, 8.723181e-16_r},
                        Real3{1.903239e-15_r, -7.930164e-17_r, 1.982541e-15_r}},
                    NdArray<real, 3, 3>{
                        Real3{2.285714e+01_r, 7.632783e-15_r, 3.806479e-15_r},
                        Real3{4.163336e-15_r, 7.178571e+01_r, 1.141944e-14_r},
                        Real3{-3.172066e-16_r, 1.903239e-15_r, 2.285714e+01_r}},
                    NdArray<real, 3, 3>{
                        Real3{7.612958e-15_r, 9.516197e-16_r, 1.685160e-15_r},
                        Real3{8.723181e-16_r, 1.141944e-14_r, 4.321429e+01_r},
                        Real3{4.956353e-16_r, 5.714286e+00_r, 1.141944e-14_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{4.758099e-16_r, -6.603188e-32_r, 5.714286e+00_r},
                        Real3{1.903239e-15_r, -3.172066e-16_r, 4.956353e-16_r},
                        Real3{4.321429e+01_r, 1.685160e-15_r, 4.758099e-16_r}},
                    NdArray<real, 3, 3>{
                        Real3{-1.650797e-32_r, 8.723181e-16_r, 1.189525e-15_r},
                        Real3{-7.930164e-17_r, 1.903239e-15_r, 5.714286e+00_r},
                        Real3{1.685160e-15_r, 4.321429e+01_r, 1.903239e-15_r}},
                    NdArray<real, 3, 3>{
                        Real3{2.285714e+01_r, 4.758099e-15_r, 5.630417e-15_r},
                        Real3{1.982541e-15_r, 2.285714e+01_r, 1.141944e-14_r},
                        Real3{4.758099e-16_r, 1.903239e-15_r, 7.178571e+01_r}}},
            }}, // End of TestData
    // Test : shear
    // Deformation gradient shear
    TestData{
        .deformationGradient =
            NdArray<real, 3, 3>{
                Real3{1.000000e+00_r, 5.000000e-01_r, 2.914335e-16_r},
                Real3{5.000000e-01_r, 1.000000e+00_r, 5.000000e-01_r},
                Real3{2.500000e-01_r, 5.000000e-01_r, 1.000000e+00_r}},
        // Strain Energy
        .strainEnergy = 6.87430e-01,
        // Material PK1 shear
        .piolaKirchhoffStress =
            NdArray<real, 3, 3>{
                Real3{1.071429e+00_r, 8.705357e-01_r, 3.571429e-01_r},
                Real3{9.263393e-01_r, 1.316964e+00_r, 8.705357e-01_r},
                Real3{5.970982e-01_r, 9.263393e-01_r, 1.071429e+00_r}},
        // Material Tangent shear
        .tangent =
            NdArray<real, 3, 3, 3, 3>{
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{3.102679e+00_r, 1.294643e+00_r, 1.785714e-01_r},
                        Real3{1.250000e+00_r, 1.517857e+00_r, 7.142857e-01_r},
                        Real3{6.250000e-01_r, 7.589286e-01_r, 1.428571e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{1.294643e+00_r, 1.830357e+00_r, 3.571429e-01_r},
                        Real3{7.142857e-01_r, 1.250000e+00_r, 3.571429e-01_r},
                        Real3{3.571429e-01_r, 6.250000e-01_r, 7.142857e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{1.785714e-01_r, 3.571429e-01_r, 1.294643e+00_r},
                        Real3{1.785714e-01_r, 8.928571e-02_r, 3.571429e-01_r},
                        Real3{3.571429e-01_r, 1.785714e-01_r, 1.785714e-01_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{1.250000e+00_r, 7.142857e-01_r, 1.785714e-01_r},
                        Real3{1.852679e+00_r, 1.294643e+00_r, 6.250000e-01_r},
                        Real3{6.250000e-01_r, 4.464286e-01_r, 7.589286e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{1.517857e+00_r, 1.250000e+00_r, 8.928571e-02_r},
                        Real3{1.294643e+00_r, 3.258929e+00_r, 1.250000e+00_r},
                        Real3{4.464286e-01_r, 1.294643e+00_r, 1.517857e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{7.142857e-01_r, 3.571429e-01_r, 3.571429e-01_r},
                        Real3{6.250000e-01_r, 1.250000e+00_r, 1.830357e+00_r},
                        Real3{3.571429e-01_r, 7.142857e-01_r, 1.294643e+00_r}}},
                NdArray<real, 3, 3, 3>{
                    NdArray<real, 3, 3>{
                        Real3{6.250000e-01_r, 3.571429e-01_r, 3.571429e-01_r},
                        Real3{6.250000e-01_r, 4.464286e-01_r, 3.571429e-01_r},
                        Real3{1.450893e+00_r, 6.250000e-01_r, 6.250000e-01_r}},
                    NdArray<real, 3, 3>{
                        Real3{7.589286e-01_r, 6.250000e-01_r, 1.785714e-01_r},
                        Real3{4.464286e-01_r, 1.294643e+00_r, 7.142857e-01_r},
                        Real3{6.250000e-01_r, 1.852679e+00_r, 1.250000e+00_r}},
                    NdArray<real, 3, 3>{
                        Real3{1.428571e+00_r, 7.142857e-01_r, 1.785714e-01_r},
                        Real3{7.589286e-01_r, 1.517857e+00_r, 1.294643e+00_r},
                        Real3{6.250000e-01_r, 1.250000e+00_r, 3.102679e+00_r}}},
            }}, // End of TestData
}; // end vector
} // namespace st_venant_kirchhoff_test_data

} // namespace mochi::materials::test
