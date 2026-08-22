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

#include "cuda_test_kernels.h"

namespace mochi {

template <typename Scalar>
MOCHI_GPU_KERNEL void TestCuMultiply(Scalar* a, Scalar* b) {
  StridedMatrixView<Scalar, 3, 3> Aview{a};
  StridedMatrixView<Scalar, 3, 3> Bview{b};

  Bview = Aview * Aview;
}

template <typename Scalar, int kStride>
MOCHI_GPU_KERNEL void StridedMultiply(
    StridedView<Scalar, 3, 3, kStride> svA,
    StridedView<Scalar, 3, 3, kStride> svB) {
  int myIndex = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (svA.Has(myIndex) && svB.Has(myIndex)) {
    auto A = svA[myIndex];
    auto B = svB[myIndex];
    B = A * A;
  }
}

template <typename Scalar>
void DoCudaCuSquare3x3(CudaMatrix<Scalar>& A, CudaMatrix<Scalar>& B) {
  TestCuMultiply<<<1, 1>>>(A.data(), B.data());
}

template <typename Scalar, int kStride>
void DoCudaCuSquare3x3(CudaVector<Scalar>& A, CudaVector<Scalar>& B) {
  auto n = std::min(A.Rows(), B.Rows()) / (3 * 3);
  auto nBlocks = (n + 31) / 32;
  StridedView<Scalar, 3, 3, kStride> AV{3, 3, A.GetSpan()};
  StridedView<Scalar, 3, 3, kStride> BV{3, 3, B.GetSpan()};
  StridedMultiply<<<n, nBlocks>>>(AV, BV);
}

template void DoCudaCuSquare3x3<float>(CudaMatrix<float>& A, CudaMatrix<float>& B);
template void DoCudaCuSquare3x3<double>(CudaMatrix<double>& A, CudaMatrix<double>& B);

template void DoCudaCuSquare3x3<float, 1>(CudaVector<float>& A, CudaVector<float>& B);
template void DoCudaCuSquare3x3<float, 32>(CudaVector<float>& A, CudaVector<float>& B);
template void DoCudaCuSquare3x3<double, 1>(CudaVector<double>& A, CudaVector<double>& B);
template void DoCudaCuSquare3x3<double, 32>(CudaVector<double>& A, CudaVector<double>& B);

} // namespace mochi
