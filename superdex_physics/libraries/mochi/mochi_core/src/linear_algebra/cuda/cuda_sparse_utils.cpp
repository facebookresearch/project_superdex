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

#include <mochi_core/linear_algebra/cuda/cuda_sparse_utils.h>

#include <mochi_core/linear_algebra/cuda/cuda_lib.h>

#include <cusparse.h>

#include <memory>

namespace mochi::details {

void CudaSparseMatDescrRelease(CudaConstSparseMatDescr* p) {
  if (p) {
    MOCHI_CUSPARSE_CHECK(cusparseDestroySpMat(reinterpret_cast<cusparseConstSpMatDescr_t>(p)));
  }
}

void CudaBsrMatDescrRelease(CudaBsrMatDescr* p) {
  if (p) {
    MOCHI_CUSPARSE_CHECK(cusparseDestroyMatDescr(reinterpret_cast<cusparseMatDescr_t>(p)));
  }
}

std::unique_ptr<CudaBsrMatDescr, ReleaseCudaBsrMatDescr> CreateCudaBsrMatDescriptor() {
  cusparseMatDescr_t matDescr = nullptr;
  MOCHI_CUSPARSE_CHECK(cusparseCreateMatDescr(&matDescr));
  return std::unique_ptr<CudaBsrMatDescr, ReleaseCudaBsrMatDescr>{
      reinterpret_cast<CudaBsrMatDescr*>(matDescr)};
}

} // namespace mochi::details

#endif // MOCHI_USE_CUDA
