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

#include <mochi_core/geometry/deep_flow_map.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/time.h>
#include <mochi_core/utils/vmatrix.h>

#include <memory>
#include <string>

using namespace mochi;

namespace {
int constexpr kNumDoFs = 24;

// #define USE_REVISED
#ifdef USE_REVISED
static constexpr real kScale = 1.3309090553162923_r;
static constexpr Real3 kShift = {0.49734753_r, 0.51221045_r, 0.49657665_r};
static std::string const kModelPath = "cube/cube_minimal_flow_revised.pt";
#else
constexpr real kScale = 1.3362546845910275_r;
constexpr Real3 kShift = {0.47520887_r, 0.49131036_r, 0.4907711_r};
std::string const kModelPath = "cube/cube_minimal_flow.pt";
#endif

DynamicArray<Real3> points = {
    Real3(0_r, 0_r, 0_r), // Cube corner
    Real3(1_r, 1_r, 1_r), // Cube corner
    Real3(0.5_r, 0.5_r, 0.5_r), // Cube center
    Real3(0.75_r, 0.85_r, 0.15_r), // Point inside the cube
    Real3(1.25_r, -0.15_r, 0.45_r) // Point outside the cube
};

DynamicArray<int> inds = {0, 1, 2, 3, 4};

DynamicArray<real> dofs = {0.2_r,  -0.1_r, 0.3_r, -0.3_r, 0.1_r,  0.1_r, -0.2_r, -0.2_r,
                           -0.2_r, 0.1_r,  0.3_r, 0.1_r,  -0.1_r, 0.2_r, -0.1_r, 0.3_r,
                           -0.1_r, -0.1_r, 0.1_r, -0.3_r, -0.3_r, 0.2_r, 0.3_r,  -0.2_r};

DynamicArray<Real3> pyMap_Rest = {
    {-0.0507_r, -0.0312_r, -0.0767_r},
    {1.0243_r, 1.0736_r, 1.0506_r},
    {0.4981_r, 0.5071_r, 0.4938_r},
    {0.7704_r, 0.8540_r, 0.1363_r},
    {1.2649_r, -0.1795_r, 0.4560_r}};

DynamicArray<Real3> pyMap_Def = {
    {-0.1016_r, 0.0125_r, -0.2324_r},
    {0.7908_r, 1.0495_r, 1.1209_r},
    {0.4686_r, 0.4833_r, 0.5126_r},
    {0.6635_r, 0.7571_r, 0.2553_r},
    {1.2528_r, 0.0372_r, 0.6637_r}};

DynamicArray<DynamicArray<DynamicArray<real>>> pyDrefDdef_Rest = {
    {{0.9726_r, 0.0960_r, 0.0661_r},
     {-0.0589_r, 0.9448_r, 0.0671_r},
     {0.1268_r, 0.1986_r, 0.9357_r}},

    {{1.0571_r, 0.0395_r, 0.0348_r},
     {0.1111_r, 0.9395_r, 0.0692_r},
     {0.1090_r, -0.0182_r, 0.9901_r}},

    {{1.0208_r, -0.1192_r, -0.0064_r},
     {0.0025_r, 0.9290_r, -0.1091_r},
     {0.0512_r, 0.0316_r, 0.9029_r}},

    {{1.0095_r, 0.0644_r, -0.1349_r},
     {0.0215_r, 0.9771_r, 0.0064_r},
     {-0.1162_r, -0.1084_r, 1.0083_r}},

    {{0.9458_r, -0.1012_r, -0.1498_r},
     {-0.0477_r, 0.9159_r, 0.0165_r},
     {0.0727_r, -0.1182_r, 1.7439_r}}};

DynamicArray<DynamicArray<DynamicArray<real>>> pyDrefDdef_Def = {
    {{0.9228_r, 0.2098_r, -0.0932_r},
     {0.0527_r, 0.9247_r, -0.0986_r},
     {0.2574_r, 0.1873_r, 0.8787_r}},

    {{1.0434_r, 0.0045_r, 0.0075_r},
     {0.1853_r, 0.8944_r, 0.0409_r},
     {0.2501_r, 0.0257_r, 0.9119_r}},

    {{0.5320_r, -0.0158_r, -0.2223_r},
     {0.2213_r, 1.0470_r, -0.1726_r},
     {0.3221_r, 0.0274_r, 0.9161_r}},

    {{0.7226_r, -0.1411_r, -0.1367_r},
     {-0.0909_r, 0.8183_r, 0.2828_r},
     {0.1443_r, -0.0233_r, 0.9802_r}},

    {{0.8999_r, -0.2083_r, -0.2446_r},
     {0.0147_r, 0.9177_r, 0.1073_r},
     {0.1285_r, -0.0293_r, 1.6181_r}}};

DynamicArray<DynamicArray<DynamicArray<real>>> pyDdefDdofs_Rest = {
    {{5.8130e-01_r, 7.2563e-03_r,  2.5231e-02_r, 1.6600e-01_r,  -2.7068e-02_r, -3.5323e-02_r,
      7.0391e-02_r, 1.5291e-02_r,  2.7821e-03_r, 1.5891e-01_r,  -1.4924e-02_r, -2.2274e-02_r,
      5.2160e-04_r, -1.1294e-03_r, 5.3322e-03_r, 1.6111e-02_r,  1.3627e-02_r,  -1.5266e-02_r,
      5.7179e-03_r, -5.8951e-03_r, 9.7012e-03_r, -1.4979e-02_r, 2.1635e-02_r,  1.3501e-02_r},
     {8.1870e-03_r,  6.7241e-01_r,  1.6959e-02_r,  4.3426e-03_r,  8.8210e-02_r,  1.7915e-02_r,
      -4.2025e-03_r, 1.3887e-01_r,  -2.2366e-02_r, -1.2510e-02_r, 1.5533e-01_r,  -1.8060e-02_r,
      4.6769e-03_r,  -1.4673e-02_r, 3.4137e-04_r,  4.6033e-03_r,  1.1011e-02_r,  -1.4027e-02_r,
      -1.1183e-02_r, 1.1237e-02_r,  1.3597e-02_r,  -8.2421e-03_r, -1.7892e-02_r, 2.0170e-02_r},
     {6.6231e-03_r,  -5.9758e-02_r, 6.2287e-01_r,  -1.9394e-02_r, 1.5792e-02_r,  1.5083e-01_r,
      -3.0867e-03_r, 1.3577e-02_r,  1.3825e-01_r,  -1.8931e-03_r, -3.3370e-03_r, 1.1043e-01_r,
      7.2842e-03_r,  2.1870e-02_r,  -4.9014e-03_r, 3.1560e-03_r,  -5.5546e-04_r, 1.5576e-02_r,
      1.3878e-02_r,  1.0050e-02_r,  -2.0092e-02_r, 3.7863e-03_r,  -9.4449e-03_r, -2.2816e-02_r}},

    {{1.8701e-02_r, -1.7172e-03_r, -7.2339e-03_r, 1.6606e-02_r, -1.5125e-02_r, 1.3189e-02_r,
      2.8736e-02_r, 4.5835e-03_r,  -3.1569e-02_r, 2.8629e-02_r, 4.2167e-03_r,  1.7335e-03_r,
      7.7924e-02_r, 5.4722e-03_r,  2.7088e-03_r,  6.3752e-01_r, 1.8713e-02_r,  5.9312e-02_r,
      8.7343e-02_r, -4.5094e-03_r, -1.4936e-02_r, 1.3138e-01_r, -2.4796e-02_r, 8.9730e-03_r},
     {-4.6628e-04_r, -7.0483e-03_r, 1.0670e-02_r,  7.5811e-04_r,  4.3779e-02_r, -9.0393e-03_r,
      1.4105e-02_r,  1.2894e-02_r,  1.2469e-02_r,  -9.4691e-04_r, 3.0731e-02_r, 2.4012e-02_r,
      -3.1412e-02_r, 8.8022e-02_r,  -3.7792e-02_r, 2.3990e-02_r,  6.9322e-01_r, 3.7460e-02_r,
      -2.0496e-02_r, 5.5075e-02_r,  -1.1649e-02_r, -7.6562e-03_r, 9.8732e-02_r, -3.8006e-02_r},
     {-3.2690e-04_r, 7.5811e-03_r,  8.7178e-03_r, -2.2308e-03_r, -2.1678e-03_r, 6.4802e-03_r,
      -3.4722e-03_r, 1.2058e-02_r,  1.3537e-02_r, 1.5764e-02_r,  -2.1674e-02_r, 3.5777e-02_r,
      -7.6468e-03_r, -2.7068e-02_r, 8.6076e-02_r, 2.4861e-02_r,  4.8106e-02_r,  6.5134e-01_r,
      -2.5092e-04_r, -2.5006e-02_r, 1.1283e-01_r, -2.5533e-02_r, -1.1192e-02_r, 6.0708e-02_r}},

    {{3.0553e-02_r, 9.3394e-03_r,  1.0172e-03_r,  2.2416e-01_r, -2.9959e-03_r, -4.4961e-03_r,
      1.6137e-01_r, -1.1867e-02_r, 1.4494e-02_r,  2.1341e-01_r, 7.8985e-03_r,  -2.3260e-02_r,
      4.4812e-02_r, 8.7283e-03_r,  -4.5076e-03_r, 2.1899e-01_r, -2.6645e-02_r, -1.6016e-02_r,
      4.6268e-02_r, 6.6887e-03_r,  1.1731e-02_r,  3.5828e-02_r, 1.0099e-02_r,  2.0219e-03_r},
     {4.0905e-02_r,  5.2824e-02_r, 1.4406e-02_r,  -2.2872e-02_r, 2.3847e-01_r, 2.2992e-02_r,
      -3.1439e-02_r, 1.6555e-01_r, -1.2781e-02_r, 2.4185e-02_r,  1.6900e-01_r, -1.7854e-02_r,
      2.6332e-03_r,  5.7607e-02_r, 2.3281e-02_r,  -3.2513e-02_r, 2.4125e-01_r, 6.6976e-03_r,
      -1.5995e-03_r, 3.6621e-02_r, -1.9969e-02_r, 2.8257e-02_r,  3.2088e-02_r, -2.2164e-02_r},
     {2.1039e-02_r,  3.2420e-02_r,  3.8893e-02_r, -7.5150e-03_r, -3.2313e-02_r, 2.2092e-01_r,
      -1.8108e-02_r, 4.3438e-03_r,  2.3109e-01_r, 1.7517e-02_r,  -1.4635e-02_r, 2.1085e-01_r,
      -9.9996e-03_r, 2.6839e-02_r,  6.6100e-04_r, -2.5244e-02_r, -2.0435e-02_r, 2.3444e-01_r,
      3.8018e-02_r,  -2.2233e-02_r, 1.6934e-02_r, -2.0710e-02_r, 6.3669e-04_r,  2.7515e-02_r}},

    {{-2.1626e-02_r, 8.2451e-04_r,  5.3672e-03_r, 8.5419e-02_r, -4.3323e-02_r, 1.0908e-02_r,
      1.2834e-01_r,  -2.3663e-02_r, 1.2755e-02_r, 1.0997e-02_r, 2.9922e-02_r,  -3.5610e-03_r,
      8.5215e-03_r,  1.3608e-02_r,  1.1662e-02_r, 1.6193e-01_r, -1.7072e-02_r, 8.3837e-03_r,
      2.7505e-02_r,  8.3849e-03_r,  5.3184e-04_r, 6.1199e-01_r, 2.4318e-02_r,  -4.7083e-02_r},
     {6.5502e-03_r,  -5.0770e-03_r, 4.8995e-03_r,  -3.8558e-02_r, 2.4111e-01_r, 1.6558e-02_r,
      1.0863e-02_r,  3.2093e-02_r,  -1.0312e-02_r, 1.5424e-02_r,  2.5077e-03_r, 1.2956e-03_r,
      9.4443e-04_r,  7.8387e-03_r,  9.2921e-03_r,  -1.8540e-02_r, 1.6264e-01_r, -5.2972e-03_r,
      -2.1148e-03_r, -5.7045e-03_r, 3.9704e-03_r,  3.0108e-02_r,  5.4441e-01_r, -2.7049e-02_r},
     {9.4891e-03_r,  -1.2146e-02_r, -5.7165e-03_r, -4.3298e-03_r, -1.3866e-02_r, 2.3196e-01_r,
      6.7265e-03_r,  -2.1376e-02_r, 1.6829e-01_r,  5.2594e-03_r,  2.5965e-03_r,  2.4462e-02_r,
      -5.3753e-03_r, 1.6616e-02_r,  -4.1186e-03_r, 4.5851e-03_r,  4.2435e-03_r,  4.0836e-02_r,
      -5.4935e-03_r, 9.1124e-03_r,  1.2117e-02_r,  -1.0421e-02_r, -1.6507e-02_r, 5.2037e-01_r}},

    {{3.2775e-03_r,  3.5837e-03_r,  2.9428e-03_r,  -1.7026e-02_r, -1.5948e-02_r, -1.2434e-02_r,
      4.8021e-01_r,  6.0087e-03_r,  -2.9715e-02_r, 5.0968e-02_r,  -5.8322e-02_r, -1.6826e-02_r,
      -2.3745e-03_r, -4.1395e-03_r, 1.9922e-02_r,  2.6183e-02_r,  1.7374e-02_r,  -1.7485e-02_r,
      4.0973e-01_r,  4.8886e-02_r,  6.6968e-02_r,  2.1405e-02_r,  8.6514e-03_r,  -1.2898e-02_r},
     {1.1111e-02_r,  3.5194e-02_r, 1.2594e-02_r,  -1.9027e-02_r, 9.2176e-03_r, -1.2978e-02_r,
      -1.2152e-02_r, 4.6539e-01_r, 5.5614e-03_r,  1.2029e-02_r,  1.8262e-02_r, -1.7601e-02_r,
      4.4973e-03_r,  5.0283e-03_r, 1.3208e-02_r,  -6.1143e-02_r, 5.6433e-02_r, 8.1955e-03_r,
      5.8459e-02_r,  3.9778e-01_r, -8.8292e-03_r, -9.3987e-03_r, 1.8108e-02_r, 9.5044e-03_r},
     {6.8332e-03_r,  -7.5458e-03_r, 6.6334e-02_r,  9.5375e-04_r,  2.8250e-02_r,  3.3841e-02_r,
      -8.5221e-02_r, 4.2882e-03_r,  3.6005e-01_r,  1.3799e-02_r,  1.2207e-02_r,  7.2786e-02_r,
      -3.7197e-03_r, 1.3945e-03_r,  -1.1883e-04_r, -2.8103e-02_r, -1.8841e-02_r, 8.5107e-02_r,
      4.6767e-02_r,  -3.1548e-02_r, 2.6744e-01_r,  4.3890e-02_r,  -1.6540e-02_r, 9.3775e-02_r}}};

DynamicArray<DynamicArray<DynamicArray<real>>> pyDdefDdofs_Def = {
    {{5.5976e-01_r,  -3.1054e-02_r, 6.9501e-02_r, 1.5797e-01_r,  -8.3599e-03_r, -4.1249e-02_r,
      9.6724e-02_r,  2.0793e-02_r,  7.4912e-03_r, 1.3587e-01_r,  -2.8672e-02_r, -5.5580e-02_r,
      -4.0787e-03_r, -1.6603e-03_r, 4.5413e-03_r, 1.5212e-02_r,  -1.6684e-03_r, -8.7593e-03_r,
      2.2290e-02_r,  1.4813e-02_r,  9.9703e-03_r, -1.7400e-02_r, 3.4626e-02_r,  -7.0688e-04_r},
     {-2.3424e-02_r, 6.5171e-01_r, 4.8681e-02_r,  -3.6518e-03_r, 9.8686e-02_r,  1.2700e-02_r,
      2.3056e-02_r,  1.6995e-01_r, -3.7399e-02_r, -6.6069e-03_r, 1.3411e-01_r,  -2.1990e-03_r,
      -9.0319e-03_r, 9.7787e-03_r, 1.0109e-02_r,  3.9995e-03_r,  -3.3090e-03_r, -3.3314e-02_r,
      2.0217e-03_r,  1.4853e-03_r, -1.4852e-02_r, 6.1766e-03_r,  -2.9668e-02_r, 3.6414e-02_r},
     {-2.1932e-02_r, -1.2629e-02_r, 5.5542e-01_r, 2.2225e-02_r,  9.4536e-03_r,  1.3398e-01_r,
      1.6743e-03_r,  3.7501e-02_r,  1.7261e-01_r, 6.6710e-03_r,  -3.6441e-02_r, 9.7671e-02_r,
      -5.4088e-03_r, 2.5496e-03_r,  1.8502e-02_r, -1.5811e-02_r, 3.8253e-03_r,  2.7781e-02_r,
      8.8661e-03_r,  1.0189e-02_r,  1.9766e-02_r, 1.3251e-02_r,  -2.3426e-02_r, -9.7575e-03_r}},

    {{1.9122e-02_r, 3.2472e-03_r,  -8.0480e-03_r, -2.0588e-02_r, 7.6148e-03_r,  -1.1340e-02_r,
      4.6312e-02_r, -2.3777e-03_r, -2.8666e-02_r, 2.6304e-02_r,  -2.1549e-03_r, 5.8456e-03_r,
      8.4513e-02_r, 2.3603e-02_r,  3.4950e-03_r,  6.5870e-01_r,  -2.0413e-02_r, 5.9119e-02_r,
      8.5656e-02_r, 5.9279e-03_r,  -1.0905e-02_r, 1.0521e-01_r,  -3.9156e-02_r, 1.3843e-02_r},
     {-5.9617e-03_r, 2.5241e-03_r,  8.5379e-03_r,  1.3215e-02_r,  5.5883e-02_r, 2.5740e-03_r,
      -1.0566e-02_r, -5.1778e-03_r, 2.3458e-02_r,  1.7342e-02_r,  3.3614e-02_r, -1.9880e-03_r,
      -2.4199e-02_r, 1.5640e-01_r,  -3.3186e-02_r, -4.7362e-03_r, 7.1214e-01_r, 1.3134e-02_r,
      -3.5903e-03_r, 2.6974e-02_r,  -1.5908e-02_r, 2.1965e-03_r,  4.1998e-02_r, -3.0333e-02_r},
     {-1.9724e-02_r, 1.5737e-02_r,  2.6993e-02_r, 1.7526e-02_r,  -9.8750e-03_r, 1.4496e-02_r,
      -3.7265e-02_r, 1.8214e-02_r,  4.0432e-02_r, 1.4935e-02_r,  -2.6003e-02_r, 5.7418e-02_r,
      1.8455e-02_r,  -1.9724e-02_r, 1.4449e-01_r, -1.7644e-02_r, 3.0199e-02_r,  6.0990e-01_r,
      1.3746e-02_r,  -3.5106e-02_r, 9.0576e-02_r, 8.2851e-03_r,  1.2431e-02_r,  1.3019e-02_r}},

    {{1.0684e-01_r, 3.6237e-02_r,  8.2507e-03_r,  1.8226e-01_r, 2.4636e-02_r,  -2.9256e-03_r,
      1.6176e-01_r, -1.5459e-02_r, -3.7805e-02_r, 1.8712e-01_r, 6.3625e-02_r,  9.3420e-03_r,
      1.6621e-02_r, 2.8181e-02_r,  -8.8317e-03_r, 2.5963e-01_r, -7.6761e-02_r, -1.6188e-02_r,
      4.0559e-02_r, -2.9605e-02_r, 3.7941e-02_r,  1.5107e-02_r, -2.8393e-02_r, 1.6129e-02_r},
     {3.0275e-02_r,  7.0036e-02_r, -2.1538e-02_r, -1.7151e-02_r, 2.0169e-01_r, 3.4100e-02_r,
      -1.9543e-02_r, 2.2225e-01_r, 2.7767e-02_r,  3.9901e-02_r,  1.9102e-01_r, -5.5984e-02_r,
      1.8839e-02_r,  1.3257e-02_r, 4.3426e-02_r,  -9.4557e-02_r, 2.3777e-01_r, 2.3914e-02_r,
      1.4703e-03_r,  4.7147e-02_r, -4.1990e-02_r, 4.5481e-02_r,  9.2589e-03_r, -6.1349e-03_r},
     {-2.0129e-03_r, 1.9208e-02_r,  4.2150e-02_r, 1.3111e-02_r,  -6.0296e-03_r, 1.5916e-01_r,
      -2.1872e-02_r, 5.1139e-02_r,  1.7368e-01_r, -1.7340e-03_r, -7.3013e-02_r, 2.5494e-01_r,
      6.7788e-05_r,  6.2358e-04_r,  8.1939e-02_r, -1.6342e-02_r, -1.5938e-02_r, 2.0896e-01_r,
      1.3301e-02_r,  -1.7610e-02_r, 5.4521e-02_r, 1.1783e-02_r,  5.1308e-03_r,  3.5902e-02_r}},

    {{-4.2347e-03_r, -2.0130e-03_r, 2.5150e-02_r,  7.9691e-02_r,  4.3180e-02_r,  -6.3538e-03_r,
      2.3849e-01_r,  -6.0215e-02_r, -5.3960e-02_r, -1.7416e-03_r, -1.5903e-02_r, -2.2390e-02_r,
      6.4363e-03_r,  -3.9713e-03_r, 2.1035e-02_r,  2.2364e-01_r,  8.4914e-03_r,  -2.9064e-02_r,
      5.8239e-04_r,  3.0475e-02_r,  1.4417e-02_r,  4.5535e-01_r,  -2.3288e-02_r, 5.7694e-02_r},
     {4.3493e-03_r,  2.2894e-02_r,  1.5830e-02_r,  -4.8761e-02_r, 3.4258e-01_r,  -5.0890e-02_r,
      7.2276e-02_r,  3.2522e-03_r,  -5.1150e-02_r, 2.4979e-02_r,  -5.4184e-02_r, -1.3340e-02_r,
      -6.3725e-04_r, -4.2975e-03_r, 3.1545e-02_r,  -4.3923e-02_r, 2.8662e-01_r,  5.1393e-02_r,
      -1.1556e-02_r, 1.9299e-02_r,  2.6541e-02_r,  4.7289e-03_r,  3.6986e-01_r,  -1.7321e-02_r},
     {1.7092e-02_r,  -1.3298e-02_r, 2.0474e-02_r, 2.4550e-02_r,  -3.4078e-02_r, 2.9708e-01_r,
      -4.1014e-02_r, 1.6072e-02_r,  2.0737e-01_r, 2.2078e-02_r,  -6.6075e-03_r, 3.2764e-02_r,
      1.0933e-02_r,  -4.6643e-03_r, 1.7162e-02_r, 3.0049e-03_r,  -1.5890e-03_r, 4.7652e-02_r,
      2.7281e-04_r,  1.4778e-03_r,  2.0176e-02_r, -3.5788e-02_r, -3.2411e-04_r, 3.6970e-01_r}},

    {{1.4031e-02_r, -3.2613e-03_r, 1.0892e-02_r, -4.3089e-02_r, -1.9676e-03_r, -3.2087e-02_r,
      3.8911e-01_r, 8.9773e-03_r,  9.5406e-03_r, 3.2018e-02_r,  -2.7584e-02_r, 2.2929e-03_r,
      4.4061e-03_r, 1.3796e-02_r,  5.4853e-03_r, 7.9848e-02_r,  -2.2504e-02_r, -4.3892e-02_r,
      4.7597e-01_r, 3.6450e-02_r,  5.9361e-02_r, 2.3352e-02_r,  -4.8228e-03_r, 9.6181e-03_r},
     {1.5826e-02_r,  1.0838e-02_r, 1.1312e-02_r,  -2.3109e-02_r, 1.3091e-02_r, -1.4870e-02_r,
      4.8969e-03_r,  3.7118e-01_r, 5.1972e-02_r,  2.7534e-02_r,  2.8660e-02_r, -4.6671e-03_r,
      -2.8824e-04_r, 1.3806e-02_r, 1.7455e-04_r,  -4.2602e-02_r, 6.5306e-02_r, 1.5581e-02_r,
      2.2743e-02_r,  4.9536e-01_r, -3.8721e-02_r, -1.9594e-02_r, 2.4632e-03_r, -1.3197e-02_r},
     {2.5165e-03_r,  -5.1884e-03_r, 5.7406e-02_r, 2.0350e-02_r,  4.5229e-02_r,  3.5526e-02_r,
      -6.9702e-02_r, 2.8168e-02_r,  3.1629e-01_r, 1.8949e-02_r,  2.7798e-03_r,  9.1529e-02_r,
      -6.0431e-05_r, -2.0525e-02_r, 1.5063e-02_r, -3.6675e-02_r, -4.9419e-03_r, 1.1367e-01_r,
      2.1327e-02_r,  -4.6284e-02_r, 2.6966e-01_r, 3.2371e-02_r,  -2.9821e-02_r, 8.9499e-02_r}}};
}; // namespace

class DeepFlowMapTest : public testing::Test {
 public:
  DeepFlowMapTest() = default;

  void Init(NeuralComputeType computeType) {
    // Create the mapping
    ErrorAssert error;
    std::string path = test::GetAssetPath(kModelPath);
    _flow = LoadDeepFlow(
        path.c_str(),
        kScale,
        kShift,
        kNumDoFs,
        computeType,
        DeepFlow::kMochiSamplesPrealloc,
        error);
    _flowMap = CreateDeepFlowMap(_flow, 1_r, error);
  }

  void Clear() {
    _outPoints.clear();
    _outDDefDDef.clear();
    _outDDefDDofs.clear();
    _outInds.clear();
  }

  void RunQuery() {
    _flowMap->MapPoints(points, inds, nullptr, _outPoints, _outInds, &_outDDefDDef, &_outDDefDDofs);
  }

  void CompareWithPython(
      DynamicArray<Real3> const& pyPoints,
      DynamicArray<DynamicArray<DynamicArray<real>>> const& pyDrefDdef,
      DynamicArray<DynamicArray<DynamicArray<real>>> const& pyDdefDdofs,
      real const tolPoints,
      real const tolDrefDdef,
      real const tolDrefDdof) {
    EXPECT_EQ(_outPoints.size(), points.size());
    EXPECT_EQ(_outInds.size(), points.size());
    for (int i = 0; i < points.size(); i++) {
      EXPECT_EQ(_outInds[i], i);
      EXPECT_NEAR(_outPoints[i][0], pyPoints[i][0], tolPoints);
      EXPECT_NEAR(_outPoints[i][1], pyPoints[i][1], tolPoints);
      EXPECT_NEAR(_outPoints[i][2], pyPoints[i][2], tolPoints);
      for (int j = 0; j < 3; j++) {
        EXPECT_NEAR(Get<0>(_outDDefDDef[i][j]), pyDrefDdef[i][j][0], tolDrefDdef);
        EXPECT_NEAR(Get<1>(_outDDefDDef[i][j]), pyDrefDdef[i][j][1], tolDrefDdef);
        EXPECT_NEAR(Get<2>(_outDDefDDef[i][j]), pyDrefDdef[i][j][2], tolDrefDdef);
      }
      for (int j = 0; j < kNumDoFs; j++) {
        EXPECT_NEAR(Get<0>(_outDDefDDofs[i].jac[j]), pyDdefDdofs[i][0][j], tolDrefDdof);
        EXPECT_NEAR(Get<1>(_outDDefDDofs[i].jac[j]), pyDdefDdofs[i][1][j], tolDrefDdof);
        EXPECT_NEAR(Get<2>(_outDDefDDofs[i].jac[j]), pyDdefDdofs[i][2][j], tolDrefDdof);
      }
    }
  }

  void CompareWithFiniteDifferences(real const tolDrefDdef) {
    constexpr real kEps = 1e-3_r;
    auto fdDrefDdef(pyDrefDdef_Rest);

    RunQuery();
    DynamicArray<Real3> outPointsRef(_outPoints);
    Clear();

    for (int k = 0; k < 3; k++) {
      for (int i = 0; i < points.size(); i++) {
        points[i][k] += kEps;
      }
      RunQuery();
      for (int i = 0; i < points.size(); i++) {
        points[i][k] -= kEps;
      }

      for (int i = 0; i < points.size(); i++) {
        fdDrefDdef[i][0][k] = (_outPoints[i][0] - outPointsRef[i][0]) / kEps;
        fdDrefDdef[i][1][k] = (_outPoints[i][1] - outPointsRef[i][1]) / kEps;
        fdDrefDdef[i][2][k] = (_outPoints[i][2] - outPointsRef[i][2]) / kEps;
      }
      Clear();
    }

    for (int i = 0; i < points.size(); i++) {
      for (int j = 0; j < 3; j++) {
        EXPECT_NEAR(Get<0>(_outDDefDDef[i][j]), fdDrefDdef[i][j][0], tolDrefDdef);
        EXPECT_NEAR(Get<1>(_outDDefDDef[i][j]), fdDrefDdef[i][j][1], tolDrefDdef);
        EXPECT_NEAR(Get<2>(_outDDefDDef[i][j]), fdDrefDdef[i][j][2], tolDrefDdef);
      }
    }
  }

  void SetUp() override {
    // This test might log a warning if CUDA is not available. A warning would normally fail
    // the test, but we need it to continue with the CPU implementation for build agents without
    // GPUs. Therefore we temporarily replace the unit test logging function with the default one
    // (just prints to stdout and the debugger).
    _prevLogFn = GetLogCallback();
    SetLogCallback(nullptr);
  }

  void TearDown() override {
    // Restor previous logging hook
    SetLogCallback(_prevLogFn);
  }

 protected:
  std::shared_ptr<DeepFlow> _flow = nullptr;
  std::unique_ptr<DeepFlowMap> _flowMap = nullptr;
  DynamicArray<Real3> _outPoints;
  DynamicArray<VMatrix3x3r> _outDDefDDef;
  DynamicArray<ColliderJacDofs> _outDDefDDofs;
  DynamicArray<int> _outInds;
  LogFn _prevLogFn = {};
};

// TODO[T152549129] Disabled in debug builds because torch::jit::load crashes or never returns.
// Also disabled when real is type double, because the saved torch files only support floats.
// The model data is not shipped externally.
#if MOCHI_USE_TORCH && !MOCHI_DEBUG && !MOCHI_USE_DOUBLE_PRECISION && MOCHI_INTERNAL
#define MOCHI_CAN_TEST_DEEP_FLOW_MAP 1
#else
#define MOCHI_CAN_TEST_DEEP_FLOW_MAP 0
#endif

// The python data for this test can be generated with the script run_flow_model.py
TEST_IF_F(MOCHI_CAN_TEST_DEEP_FLOW_MAP, DeepFlowMapTest, DeepFlowMap) {
  for (int iCompute = 0; iCompute < static_cast<int>(NeuralComputeType::Count); ++iCompute) {
    Init(static_cast<NeuralComputeType>(iCompute));

    // Compare result with values evaluated in Python for no deformation
    RunQuery();
    CompareWithPython(pyMap_Rest, pyDrefDdef_Rest, pyDdefDdofs_Rest, 3e-4_r, 8e-4_r, 3e-4_r);
    Clear();

    // Compare result with values evaluated in Python for a deformed case
    _flowMap->UpdateMap(dofs);
    RunQuery();
    CompareWithPython(pyMap_Def, pyDrefDdef_Def, pyDdefDdofs_Def, 4e-4_r, 7e-4_r, 3e-4_r);
  }
}

TEST_IF_F(MOCHI_CAN_TEST_DEEP_FLOW_MAP, DeepFlowMapTest, DeepFlowMap_Consistency) {
  for (int iCompute = 0; iCompute < static_cast<int>(NeuralComputeType::Count); ++iCompute) {
    auto computeType = static_cast<NeuralComputeType>(iCompute);
    if (computeType == NeuralComputeType::TorchGpu) {
      continue; // The consistency test fails with TorchGpu
    }
    Init(computeType);

    // Test dref_ddef through finite differences for no deformation
    CompareWithFiniteDifferences(1.e-2_r);

    // Test dref_ddef through finite differences for a deformed case
    _flowMap->UpdateMap(dofs);
    CompareWithFiniteDifferences(1e-2_r);
  }
}
