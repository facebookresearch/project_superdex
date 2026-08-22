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

#include <mochi_core/linear_algebra/matrix.h>

namespace mochi {

template <
    typename Scalar,
    int kRowsAtCompileTime = krylov::kDynamic,
    int kColsAtCompileTime = krylov::kDynamic,
    krylov::Direction kMajorDirection = krylov::Direction::ColMajor,
    int kLeadingDim = krylov::kAutomaticLeadDim>
using CudaMatrix = Matrix<
    Scalar,
    kRowsAtCompileTime,
    kColsAtCompileTime,
    kMajorDirection,
    krylov::Ownership::Cuda,
    kLeadingDim>;

template <
    typename Scalar,
    int kRowsAtCompileTime = krylov::kDynamic,
    int kColsAtCompileTime = krylov::kDynamic,
    krylov::Direction kMajorDirection = krylov::Direction::ColMajor,
    int kLeadingDim = krylov::kAutomaticLeadDim>
using CudaMatrixView = Matrix<
    Scalar,
    kRowsAtCompileTime,
    kColsAtCompileTime,
    kMajorDirection,
    krylov::Ownership::CudaView,
    kLeadingDim>;

template <typename Scalar, int kRowsAtCompileTime = krylov::kDynamic>
using CudaVector = Matrix<
    Scalar,
    kRowsAtCompileTime,
    1,
    krylov::Direction::ColMajor,
    krylov::Ownership::Cuda,
    krylov::kAutomaticLeadDim>;

template <typename Scalar, int kRowsAtCompileTime = krylov::kDynamic>
using CudaVectorView = Matrix<
    Scalar,
    kRowsAtCompileTime,
    1,
    krylov::Direction::ColMajor,
    krylov::Ownership::CudaView,
    krylov::kAutomaticLeadDim>;

template <typename Scalar>
void CudaTranspose(CudaMatrixView<Scalar> out, CudaMatrixView<Scalar const> in);

} // namespace mochi
