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

#include <mochi_core/linear_algebra/cuda/cuda_gmres_kernels.h>
#include <mochi_core/linear_algebra/cuda/cuda_lib.h>
#include <mochi_core/linear_algebra/krylov/iteration_status.h>

#include <cuda_runtime_api.h>
#include <cub/device/device_reduce.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

template <typename Scalar>
MOCHI_GPU MOCHI_FORCE_INLINE void ApplyGivensInPlace(
    Scalar const* MOCHI_RESTRICT c,
    Scalar const* MOCHI_RESTRICT s,
    Scalar* MOCHI_RESTRICT f,
    Scalar* MOCHI_RESTRICT g) {
  Scalar const temp = (*c) * (*f) + (*s) * (*g);
  (*g) = -(*s) * (*f) + (*c) * (*g);
  (*f) = temp;
}

template <typename Scalar>
MOCHI_GPU_KERNEL void ComputeInverse(
    Scalar const* MOCHI_RESTRICT a,
    Scalar* MOCHI_RESTRICT oneOverA) {
  oneOverA[0] = Scalar(1) / a[0];
}

MOCHI_GPU void GetGivens(double const* x, double const* y, double* c, double* s) {
  double const t = rhypot(*x, *y);
  (*c) = (*x) * t;
  (*s) = (*y) * t;
}

MOCHI_GPU void GetGivens(float const* x, float const* y, float* c, float* s) {
  float const t = rhypotf(*x, *y);
  (*c) = (*x) * t;
  (*s) = (*y) * t;
}

template <typename Scalar>
MOCHI_GPU_KERNEL void HessenbergUpdateInPlace1(
    int const* d_n,
    Scalar const* MOCHI_RESTRICT d_c,
    Scalar const* MOCHI_RESTRICT d_s,
    Scalar const* MOCHI_RESTRICT d_H1,
    Scalar const* MOCHI_RESTRICT d_H2,
    Scalar* MOCHI_RESTRICT d_H,
    int ldh) {
  if (blockIdx.x == 0) {
    int const tix = threadIdx.x;
    int const colIdx = (*d_n) - 1;
    Scalar* Hcol = d_H + colIdx * ldh;
    // Update the column of H
    for (int i = tix; i <= colIdx; i += blockDim.x) {
      Hcol[i] = d_H1[i] + d_H2[i];
    }
    __syncthreads();
    if (tix == 0) {
      // Apply the previous rotations
      for (int i = 0; i < colIdx; ++i) {
        ApplyGivensInPlace(d_c + i, d_s + i, Hcol + i, Hcol + i + 1);
      }
    }
  }
}

template <typename Scalar>
MOCHI_GPU_KERNEL void HessenbergUpdateInPlace2(
    int const* d_n,
    Scalar* MOCHI_RESTRICT d_c,
    Scalar* MOCHI_RESTRICT d_s,
    Scalar* d_norm,
    Scalar* MOCHI_RESTRICT d_H,
    int ldh,
    Scalar* MOCHI_RESTRICT d_b) {
  int const gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid == 0) {
    int const colIdx = (*d_n) - 1;
    Scalar* Hcol = d_H + colIdx * ldh;
    Scalar* bvec = d_b + colIdx;
    Hcol[colIdx + 1] = *d_norm;
    GetGivens(Hcol + colIdx, Hcol + colIdx + 1, d_c + colIdx, d_s + colIdx);
    ApplyGivensInPlace(d_c + colIdx, d_s + colIdx, Hcol + colIdx, Hcol + colIdx + 1);
    ApplyGivensInPlace(d_c + colIdx, d_s + colIdx, bvec, bvec + 1);
  }
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// double atomicAdd(..) is provided by Cuda in this situation
#else
MOCHI_GPU double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do {
    assumed = old;
    old = atomicCAS(
        address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case
    // of NaN (since NaN != NaN)
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

/// @brief Kernel for computing y <- Q^T * x where Q is an nxk matrix in column-major format
///
/// @param[in] n Number of rows in matrix Q
/// @param[in] d_k Pointer to number of columns in matrix Q
/// @param[in] Q Matrix in column-major format with leading dimension ldq
/// @param[in] ldq Leading dimension of matrix Q
/// @param[in] x Input vector of length n
/// @param[out] y Output vector of length *d_k, computed as y = Q^T * x
///
/// @note
/// Call KernelGemvT<<< (# of columns in Q) or upper bound, kNThreads >>>
///
template <typename Scalar>
MOCHI_GPU_KERNEL void KernelGemvT(
    int n,
    int const* d_k,
    Scalar const* MOCHI_RESTRICT Q,
    int ldq,
    Scalar const* MOCHI_RESTRICT x,
    Scalar* MOCHI_RESTRICT y) {
  int const numCols = *d_k; // Number of columns in Q (and length of output vector y)
  if (gridDim.x > numCols) {
    // Multiple thread blocks working on each row of the output
    int const teamID = blockIdx.x / numCols;
    int const rowID = blockIdx.x % numCols; // Output element index
    int const globalThreadIdx = teamID * blockDim.x + threadIdx.x;
    // Initialize output element to zero (only one thread per row does this)
    if (globalThreadIdx == 0) {
      y[rowID] = 0;
    }
    int const numTeams = gridDim.x / numCols;
    __syncthreads(); // Ensure y[rowID] is initialized before accumulating

    if (teamID < numTeams) {
      Scalar dotProduct = 0; // Accumulate dot product result
      if constexpr (std::is_same_v<Scalar, float>) {
        // Vectorization
        auto const* Q4 = reinterpret_cast<float4 const*>(Q + rowID * ldq);
        auto const* x4 = reinterpret_cast<float4 const*>(x);
        int const nOver4 = (n + 3) / 4; // Ceiling division for vectorized access
        for (int i = globalThreadIdx; i < nOver4; i += blockDim.x * numTeams) {
          auto const Qval = Q4[i];
          auto const xval = x4[i];
          // Compute dot product using vectorized data
          dotProduct += Qval.x * xval.x + Qval.y * xval.y + Qval.z * xval.z + Qval.w * xval.w;
        }
      } else {
        // Vectorization
        auto const* Q2 = reinterpret_cast<double2 const*>(Q + rowID * ldq);
        auto const* x2 = reinterpret_cast<double2 const*>(x);
        int const nOver2 = (n + 1) / 2; // Ceiling division for vectorized access
        for (int i = globalThreadIdx; i < nOver2; i += blockDim.x * numTeams) {
          auto const Qval = Q2[i];
          auto const xval = x2[i];
          // Compute dot product using vectorized data
          dotProduct += Qval.x * xval.x + Qval.y * xval.y;
        }
      }
      typedef cub::BlockReduce<Scalar, mochi::details::kNThreads> BlockReduceS;
      __shared__ typename BlockReduceS::TempStorage tempStorage;
      // Reduce partial sums within the block
      Scalar result = BlockReduceS(tempStorage).Sum(dotProduct);
      if (threadIdx.x == 0) {
        atomicAdd(&y[rowID], result);
      }
    }
  } else {
    // Case where we have fewer thread blocks than rows
    // Each block processes one or more rows in sequence
    // Only one block per row at a time
    for (int outputIdx = blockIdx.x; outputIdx < numCols; outputIdx += gridDim.x) {
      Scalar dotProduct = 0;
      if constexpr (std::is_same_v<Scalar, float>) {
        // Vectorization
        auto const* Q4 = reinterpret_cast<float4 const*>(Q + outputIdx * ldq);
        auto const* x4 = reinterpret_cast<float4 const*>(x);
        // Each thread processes a portion of the input vectors
        int const nOver4 = (n + 3) / 4; // Ceiling division for vectorized access
        for (int i = threadIdx.x; i < nOver4; i += blockDim.x) {
          auto const Qval = Q4[i];
          auto const xval = x4[i];
          // Compute dot product using vectorized data
          dotProduct += Qval.x * xval.x + Qval.y * xval.y + Qval.z * xval.z + Qval.w * xval.w;
        }
      } else {
        // Vectorization
        auto const* Q2 = reinterpret_cast<double2 const*>(Q + outputIdx * ldq);
        auto const* x2 = reinterpret_cast<double2 const*>(x);
        int const nOver2 = (n + 1) / 2; // Ceiling division for vectorized access
        for (int i = threadIdx.x; i < nOver2; i += blockDim.x) {
          auto const Qval = Q2[i];
          auto const xval = x2[i];
          // Compute dot product using vectorized data
          dotProduct += Qval.x * xval.x + Qval.y * xval.y;
        }
      }
      typedef cub::BlockReduce<Scalar, mochi::details::kNThreads> BlockReduceS;
      __shared__ typename BlockReduceS::TempStorage tempStorage;
      // Reduce partial sums within the block
      Scalar result = BlockReduceS(tempStorage).Sum(dotProduct);
      if (threadIdx.x == 0) {
        y[outputIdx] = result;
      }
    }
  }
}

/// @brief Kernel for computing y <- y - Q*x where Q is an m×n matrix in column-major format
///
/// @param[in] m Number of rows in matrix Q and vectors x, y
/// @param[in] n Pointer to number of columns in matrix Q (passed as pointer for consistency with
/// other kernels)
/// @param[in] Q Matrix in column-major format with leading dimension ldq
/// @param[in] ldq Leading dimension of matrix Q
/// @param[in] x Input vector of length *n
/// @param[in,out] y Output vector of length m,  updated as y = y - Q*x
///
template <typename Scalar>
MOCHI_GPU_KERNEL void KernelGemv(
    int m,
    int const* n,
    Scalar const* MOCHI_RESTRICT Q,
    int ldq,
    Scalar const* MOCHI_RESTRICT x,
    Scalar* MOCHI_RESTRICT y) {
  int const globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int const numCols = *n; // Cache the dereferenced value to avoid multiple dereferences

  // Use a stride loop for coalesced memory access
  for (int rowIdx = globalIdx; rowIdx < m; rowIdx += blockDim.x * gridDim.x) {
    Scalar dotProduct = 0;
    // Different handling based on number of columns for performance
    if (numCols < 4) {
      // For small matrices, simple loop is sufficient
      for (int j = 0; j < numCols; ++j) {
        dotProduct += Q[rowIdx + j * ldq] * x[j];
      }
    } else {
      // For larger matrices, use loop unrolling for better performance
#pragma unroll 4
      for (int j = 0; j < numCols; ++j) {
        dotProduct += Q[rowIdx + j * ldq] * x[j];
      }
    }
    // Update output vector
    y[rowIdx] = y[rowIdx] - dotProduct;
  }
}

template <typename Scalar>
MOCHI_GPU_KERNEL void KernelDoWhileLoop(
    int* d_iter,
    int maxIter,
    Scalar const* d_b,
    Scalar const* d_aTol,
    Scalar const* d_rTol,
    Scalar const* d_dTol,
    int* d_status,
    cudaGraphConditionalHandle handle) {
  Scalar const d_norm = abs(*(d_b + (*d_iter)));
  if (!isfinite(d_norm) || (d_norm > *d_dTol)) {
    *d_status = static_cast<int>(mochi::krylov::IterationStatus::DivergedRes);
    cudaGraphSetConditional(handle, 0);
  } else if ((d_norm <= *d_aTol) || (d_norm <= (*d_rTol))) {
    using mochi::krylov::IterationStatus::ConvergedAtol;
    using mochi::krylov::IterationStatus::ConvergedRtol;
    *d_status =
        (d_norm <= *d_aTol) ? static_cast<int>(ConvergedAtol) : static_cast<int>(ConvergedRtol);
    cudaGraphSetConditional(handle, 0);
  } else if (*d_iter == maxIter) {
    *d_status = static_cast<int>(mochi::krylov::IterationStatus::Active);
    cudaGraphSetConditional(handle, 0);
  } else {
    ++(*d_iter);
  }
}

template <typename Scalar>
MOCHI_GPU_KERNEL void DoScaleStore(
    int n,
    int const* dIter,
    Scalar* MOCHI_RESTRICT Q,
    int ldq,
    Scalar* MOCHI_RESTRICT qnew,
    Scalar const* qnorm) {
  auto const gid = blockIdx.x * blockDim.x + threadIdx.x;
  Scalar* Qcol = Q + ldq * (*dIter);
  Scalar const invN = Scalar(1) / (*qnorm);
  for (int qr = gid; qr < n; qr += blockDim.x * gridDim.x) {
    qnew[qr] *= invN;
    Qcol[qr] = qnew[qr];
  }
}

template <typename Scalar>
MOCHI_GPU_KERNEL void KernelSqrt(Scalar* x) {
  (*x) = sqrt((*x));
}

template <typename Scalar>
struct squareT {
  MOCHI_ANY MOCHI_FORCE_INLINE Scalar operator()(Scalar const& a) const {
    return a * a;
  };
};

namespace mochi::details::gmres {

template <typename Scalar>
size_t GetBufferSize(int n, Scalar* x, Scalar* norm, cudaStream_t stream) {
  size_t wSize = 0;
  void* tempStorage = nullptr;
  size_t tempStorageBytes = 0;
  //
  squareT<Scalar> op;
  cub::TransformInputIterator<Scalar, squareT<Scalar>, Scalar*> input_iter(x, op);
  cub::DeviceReduce::Sum(tempStorage, tempStorageBytes, input_iter, norm, n, stream);
  wSize += tempStorageBytes;
  //
  return wSize;
}

template size_t GetBufferSize<double>(int n, double* x, double* norm, cudaStream_t stream);

template size_t GetBufferSize<float>(int n, float* x, float* norm, cudaStream_t stream);

/// @brief Computes the L2 norm of a vector on the device
///
/// @param[in] n Length of the vector
/// @param[in] x Input vector for which to compute the L2 norm
/// @param[out] norm Pointer to store the resulting L2 norm
/// @param[in] buffer Pre-allocated temporary storage for CUB operations
/// @param[in] bufferSize Size of the buffer in elements (not bytes)
/// @param[in] stream CUDA stream on which to perform the computation
///
/// @note
/// This function computes ||x||_2 = sqrt(sum(x_i^2)) using CUB's device reduction.
///
template <typename Scalar>
void NormL2(
    int n,
    Scalar* x,
    Scalar* norm,
    Scalar* buffer,
    size_t bufferSize,
    cudaStream_t stream) {
  void* tempStorage = buffer; // Cast is unnecessary
  size_t tempStorageBytes = bufferSize * sizeof(Scalar);

  // Create transform iterator to compute squares of elements
  squareT<Scalar> squareOp;
  cub::TransformInputIterator<Scalar, squareT<Scalar>, Scalar*> input_iter(x, squareOp);

  // Sum the squares
  cub::DeviceReduce::Sum(tempStorage, tempStorageBytes, input_iter, norm, n, stream);

  // Take square root of the sum
  KernelSqrt<<<1, 1, 0, stream>>>(norm);
}

template void NormL2<double>(
    int n,
    double* x,
    double* norm,
    double* buffer,
    size_t bufferSize,
    cudaStream_t stream);

template void
NormL2<float>(int n, float* x, float* norm, float* buffer, size_t bufferSize, cudaStream_t stream);

template <typename Scalar>
void ScaleStore(
    int n,
    int const* dIter,
    Scalar* Q,
    int ldq,
    Scalar* qnew,
    Scalar const* qnorm,
    cudaStream_t stream) {
  int const nBlocks = (n + mochi::details::kNThreads - 1) / mochi::details::kNThreads;
  DoScaleStore<<<nBlocks, mochi::details::kNThreads, 0, stream>>>(n, dIter, Q, ldq, qnew, qnorm);
}

template void ScaleStore<double>(
    int n,
    int const* dIter,
    double* Q,
    int ldq,
    double* qnew,
    double const* qnorm,
    cudaStream_t stream);

template void ScaleStore<float>(
    int n,
    int const* dIter,
    float* Q,
    int ldq,
    float* qnew,
    float const* qnorm,
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
    cudaStream_t stream) {
  KernelDoWhileLoop<<<1, 1, 0, stream>>>(
      d_iter, maxIter, d_b, d_aTol, d_rTol, d_dTol, d_status, handle);
}

template void DoWhile<double>(
    int* d_iter,
    int maxIter,
    double const* d_b,
    double const* d_aTol,
    double const* d_rTol,
    double const* d_dTol,
    int* d_status,
    cudaGraphConditionalHandle handle,
    cudaStream_t stream);

template void DoWhile<float>(
    int* d_iter,
    int maxIter,
    float const* d_b,
    float const* d_aTol,
    float const* d_rTol,
    float const* d_dTol,
    int* d_status,
    cudaGraphConditionalHandle handle,
    cudaStream_t stream);

template <typename Scalar>
void Gemv(
    int m,
    int const* n,
    Scalar const* Q,
    int ldq,
    Scalar const* x,
    Scalar* y,
    cudaStream_t stream) {
  using mochi::details::kNThreads;
  int const nBlocks = (m + kNThreads - 1) / kNThreads;
  KernelGemv<<<nBlocks, kNThreads, 0, stream>>>(m, n, Q, ldq, x, y);
}

template void Gemv<double>(
    int m,
    int const* n,
    double const* Q,
    int ldq,
    double const* x,
    double* y,
    cudaStream_t stream);

template void Gemv<float>(
    int m,
    int const* n,
    float const* Q,
    int ldq,
    float const* x,
    float* y,
    cudaStream_t stream);

template <typename Scalar>
void GemvT(
    int n,
    int const* k,
    int maxK,
    Scalar const* Q,
    int ldq,
    Scalar const* x,
    Scalar* y,
    cudaStream_t stream) {
  KernelGemvT<<<maxK, mochi::details::kNThreads, 0, stream>>>(n, k, Q, ldq, x, y);
}

template void GemvT<double>(
    int n,
    int const* k,
    int maxK,
    double const* Q,
    int ldq,
    double const* x,
    double* y,
    cudaStream_t stream);

template void GemvT<float>(
    int n,
    int const* k,
    int maxK,
    float const* Q,
    int ldq,
    float const* x,
    float* y,
    cudaStream_t stream);

template <typename Scalar>
void GetInverse(Scalar const* a, Scalar* oneOverA, cudaStream_t stream) {
  ComputeInverse<<<1, 1, 0, stream>>>(a, oneOverA);
}

template void GetInverse<double>(double const* a, double* oneOverA, cudaStream_t stream);

template void GetInverse<float>(float const* a, float* oneOverA, cudaStream_t stream);

template <typename Scalar>
void UpdateHessenberg1(
    int const* d_n,
    Scalar* d_c,
    Scalar* d_s,
    Scalar const* d_H1,
    Scalar const* d_H2,
    Scalar* d_H,
    int ldh,
    cudaStream_t stream) {
  using mochi::details::kNThreads;
  HessenbergUpdateInPlace1<<<1, kNThreads, 0, stream>>>(d_n, d_c, d_s, d_H1, d_H2, d_H, ldh);
}

template void UpdateHessenberg1<double>(
    int const* d_n,
    double* d_c,
    double* d_s,
    double const* d_H1,
    double const* d_H2,
    double* d_H,
    int ldh,
    cudaStream_t stream);

template void UpdateHessenberg1<float>(
    int const* d_n,
    float* d_c,
    float* d_s,
    float const* d_H1,
    float const* d_H2,
    float* d_H,
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
    cudaStream_t stream) {
  HessenbergUpdateInPlace2<<<1, 1, 0, stream>>>(d_n, d_c, d_s, d_norm, d_H, ldh, d_b);
}

template void UpdateHessenberg2<double>(
    int const* d_n,
    double* d_c,
    double* d_s,
    double* d_norm,
    double* d_H,
    int ldh,
    double* d_b,
    cudaStream_t stream);

template void UpdateHessenberg2<float>(
    int const* d_n,
    float* d_c,
    float* d_s,
    float* d_norm,
    float* d_H,
    int ldh,
    float* d_b,
    cudaStream_t stream);

} // namespace mochi::details::gmres

#endif
