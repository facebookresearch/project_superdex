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

#include <mochi_core/mochi_config.h>

#if MOCHI_USE_CUDA

#include <cuda_runtime_api.h>

namespace mochi::details::gmres {

template <typename Scalar>
size_t GetBufferSize(int n, Scalar* x, Scalar* norm, cudaStream_t stream);

/// @brief Compute the operation y <- y - Q x on the device
///
/// @tparam Scalar
/// @param m Number of rows in y or Q
/// @param n Number of columns in Q
/// @param Q Column-major matrix of dimension m x n
/// @param ldq Leading dimension in Q
/// @param[in] x Vector of length n (on the device)
/// @param[out] y Vector of length m (on the device)
/// @param stream Cuda stream to launch the operation on
template <typename Scalar>
void Gemv(
    int m,
    int const* n,
    Scalar const* Q,
    int ldq,
    Scalar const* x,
    Scalar* y,
    cudaStream_t stream);

/// @brief Compute the operation y <- Q^T x on the device
///
/// @tparam Scalar
/// @param n Number of rows in Q
/// @param k Number of columns in Q
/// @param maxK Upper bound on k (to launch maxK blocks on the device)
/// @param[in] Q Column-major matrix of dimension n x k
/// Each column of Q is "padded" with zeros to be read with float4 or double2.
/// @param[in] ldq Leading dimension in Q
/// @param[in] x Vector of length n (on the device)
/// The vector is "padded" with zeros to be read with float4 or double2.
/// @param[out] y Vector of length k (on the device)
/// @param stream Cuda stream to launch the operation on
template <typename Scalar>
void GemvT(
    int n,
    int const* k,
    int maxK,
    Scalar const* Q,
    int ldq,
    Scalar const* x,
    Scalar* y,
    cudaStream_t stream);

template <typename Scalar>
void GetInverse(Scalar const* a, Scalar* oneOverA, cudaStream_t stream);

template <typename Scalar>
void UpdateHessenberg1(
    int const* d_n,
    Scalar* d_c,
    Scalar* d_s,
    Scalar const* d_H1,
    Scalar const* d_H2,
    Scalar* d_H,
    int ldh,
    cudaStream_t stream);

template <typename Scalar>
void UpdateHessenberg2(
    int const* d_n,
    Scalar* d_c,
    Scalar* d_s,
    Scalar* d_norm,
    Scalar* d_H,
    int ldh,
    Scalar* d_b,
    cudaStream_t stream);

template <typename Scalar>
void DoWhile(
    int* d_iter,
    int maxIter,
    Scalar const* d_b,
    Scalar const* d_aTol,
    Scalar const* d_rTol,
    Scalar const* d_dTol,
    int* d_status,
    cudaGraphConditionalHandle handle,
    cudaStream_t stream);

template <typename Scalar>
void ScaleStore(
    int n,
    int const* dIter,
    Scalar* Q,
    int ldq,
    Scalar* qnew,
    Scalar const* qnorm,
    cudaStream_t stream);

template <typename Scalar>
void NormL2(int n, Scalar* x, Scalar* norm, Scalar* buffer, size_t bufferSize, cudaStream_t stream);

} // namespace mochi::details::gmres

#endif
