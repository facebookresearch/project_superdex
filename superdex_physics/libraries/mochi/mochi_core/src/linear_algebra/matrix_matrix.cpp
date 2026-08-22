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

#include <array>

#include <mochi_core/utils/simd.h>

namespace mochi::details {

/** @brief Matrix-matrix multiplication kernel routine.
 * @details The kernel computes C = A * B, where C is N x 6 stored column major, A is N x k stored
 * column major, B is k x 6 with row stride of b_ri and column stride of b_ci, and N is twice the
 * size of the SIMD vector. The kernel is optimized for a SIMD vector size such that 16 vectors fill
 * the floating-point registers.
 * @param a Pointer to the start of matrix A.
 * @param b Pointer to the start of matrix B.
 * @param c Pointer to the start of matrix C.
 * @param k Number of columns of A and rows of B.
 * @param b_ri Row stride of B.
 * @param b_ci Column stride of B.
 */
template <typename VType>
void MatMatKernel(
    typename VType::Scalar const* a,
    typename VType::Scalar const* b,
    typename VType::Scalar* c,
    int k,
    int b_ri,
    int b_ci) {
  static_assert(VType::kIsSupported, "Implementation requires a scalar type with SIMD support");
  constexpr auto kVecSize = VType::kSize;

  std::array<VType, 12> c_r MOCHI_NO_INIT;
  for (int i = 0; i < 12; ++i) {
    c_r[i] = SimdZero<VType>();
  }
  for (int j = 0; j < k; j += 1) {
    auto a_r0 = Load<VType>(a + (2 * j + 0) * kVecSize);
    auto a_r1 = Load<VType>(a + (2 * j + 1) * kVecSize);
    auto b_r0 = Broadcast<VType>(b[j * b_ri + 0 * b_ci]);
    auto b_r1 = Broadcast<VType>(b[j * b_ri + 1 * b_ci]);

    c_r[0] = MulAdd(a_r0, b_r0, c_r[0]);
    c_r[1] = MulAdd(a_r1, b_r0, c_r[1]);
    c_r[2] = MulAdd(a_r0, b_r1, c_r[2]);
    c_r[3] = MulAdd(a_r1, b_r1, c_r[3]);

    b_r0 = Broadcast<VType>(b[j * b_ri + 2 * b_ci]);
    b_r1 = Broadcast<VType>(b[j * b_ri + 3 * b_ci]);
    c_r[4] = MulAdd(a_r0, b_r0, c_r[4]);
    c_r[5] = MulAdd(a_r1, b_r0, c_r[5]);
    c_r[6] = MulAdd(a_r0, b_r1, c_r[6]);
    c_r[7] = MulAdd(a_r1, b_r1, c_r[7]);

    b_r0 = Broadcast<VType>(b[j * b_ri + 4 * b_ci]);
    b_r1 = Broadcast<VType>(b[j * b_ri + 5 * b_ci]);
    c_r[8] = MulAdd(a_r0, b_r0, c_r[8]);
    c_r[9] = MulAdd(a_r1, b_r0, c_r[9]);
    c_r[10] = MulAdd(a_r0, b_r1, c_r[10]);
    c_r[11] = MulAdd(a_r1, b_r1, c_r[11]);
  }

  for (int i = 0; i < 12; ++i) {
    Store(c + i * kVecSize, c_r[i]);
  }
}

template void
MatMatKernel<Vec4f>(float const* a, float const* b, float* c, int k, int b_ri, int b_ci);

template void
MatMatKernel<Vec8f>(float const* a, float const* b, float* c, int k, int b_ri, int b_ci);

template void
MatMatKernel<Vec2d>(double const* a, double const* b, double* c, int k, int b_ri, int b_ci);

template void
MatMatKernel<Vec4d>(double const* a, double const* b, double* c, int k, int b_ri, int b_ci);

} // namespace mochi::details
