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

#if MOCHI_USE_CUDA

#include <cuda_runtime_api.h>

namespace mochi::details {

template <typename Scalar>
void Alpha(
    Scalar const* dividend,
    Scalar const* divisor,
    Scalar* quotient,
    Scalar* negQuotient,
    bool* or_leq0,
    cudaStream_t mainStream);

template <typename Scalar>
void Beta(Scalar const* rTzOld, Scalar const* rTz, Scalar* beta, cudaStream_t mainStream);

} // namespace mochi::details

#endif
