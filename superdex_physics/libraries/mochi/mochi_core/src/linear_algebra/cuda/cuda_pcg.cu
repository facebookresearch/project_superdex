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

#if MOCHI_USE_CUDA

#include <mochi_core/linear_algebra/cuda/cuda_lib.h>

#include <cuda_runtime_api.h>

/// @brief Get the alpha scalar and its negative value for PCG
template <typename Scalar>
MOCHI_GPU_KERNEL static void ComputeAlpha(
    Scalar const* dividend,
    Scalar const* divisor,
    Scalar* quotient,
    Scalar* negQuotient,
    bool* or_leq0) {
  quotient[0] = dividend[0] / divisor[0];
  negQuotient[0] = -quotient[0];
  *or_leq0 |= (!isfinite(*quotient) || (*quotient <= 0));
}

/// @brief Compute the update for beta
/// @note
/// This function has a subtraction for the Polak-Ribiere formula.
template <typename Scalar>
MOCHI_GPU_KERNEL static void ComputeBeta(Scalar const* rTzOld, Scalar const* rTz, Scalar* beta) {
  *beta = (*rTz - *beta) / *rTzOld;
}

namespace mochi::details {

template <typename Scalar>
void Alpha(
    Scalar const* dividend,
    Scalar const* divisor,
    Scalar* quotient,
    Scalar* negQuotient,
    bool* or_leq0,
    cudaStream_t mainStream) {
  ComputeAlpha<<<1, 1, 0, mainStream>>>(dividend, divisor, quotient, negQuotient, or_leq0);
}

template void Alpha<double>(
    double const* dividend,
    double const* divisor,
    double* quotient,
    double* negQuotient,
    bool* or_leq0,
    cudaStream_t mainStream);

template void Alpha<float>(
    float const* dividend,
    float const* divisor,
    float* quotient,
    float* negQuotient,
    bool* or_leq0,
    cudaStream_t mainStream);

template <typename Scalar>
void Beta(Scalar const* rTzOld, Scalar const* rTz, Scalar* beta, cudaStream_t mainStream) {
  ComputeBeta<<<1, 1, 0, mainStream>>>(rTzOld, rTz, beta);
}

template void
Beta<double>(double const* rTzOld, double const* rTz, double* beta, cudaStream_t mainStream);

template void
Beta<float>(float const* rTzOld, float const* rTz, float* beta, cudaStream_t mainStream);

} // namespace mochi::details

#endif
