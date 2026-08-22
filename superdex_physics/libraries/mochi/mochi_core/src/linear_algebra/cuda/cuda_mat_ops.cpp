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

#include <mochi_core/linear_algebra/cuda/cuda_matrix.h>
#include <mochi_core/utils/transform_rt_inl.h>

namespace mochi::details {
template <typename Scalar>
void DoCudaTranspose(CudaMatrixView<Scalar> out, CudaMatrixView<Scalar const> in);
}

namespace mochi {

template <typename Scalar>
void CudaTranspose(CudaMatrixView<Scalar> out, CudaMatrixView<Scalar const> in) {
  MOCHI_ASSERT_VERBOSE(
      in.Rows() == out.Cols() && in.Cols() == out.Rows(), "Transpose sizes are incompatible.");
  mochi::details::DoCudaTranspose(out, in);
}

template void CudaTranspose<float>(CudaMatrixView<float> out, CudaMatrixView<float const> in);
template void CudaTranspose<double>(CudaMatrixView<double> out, CudaMatrixView<double const> in);
template void CudaTranspose<int>(CudaMatrixView<int> out, CudaMatrixView<int const> in);

} // namespace mochi

#endif // MOCHI_USE_CUDA
