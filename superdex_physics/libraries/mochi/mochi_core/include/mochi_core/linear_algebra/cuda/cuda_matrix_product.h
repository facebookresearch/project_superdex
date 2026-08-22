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

#include <mochi_core/linear_algebra/cuda/cuda_api.h>
#include <mochi_core/linear_algebra/matrix_accessors.h>

namespace mochi::details {

template <typename Scalar>
struct ScaledMatData;

template <typename Scalar>
struct MatDataDest;

void ExecuteCudaProduct(
    MatDataDest<double>& C,
    ScaledMatData<double const>& A,
    ScaledMatData<double const>& B,
    int m,
    int n,
    int k,
    double alpha,
    double beta);

void ExecuteCudaProduct(
    MatDataDest<float>& C,
    ScaledMatData<float const>& A,
    ScaledMatData<float const>& B,
    int m,
    int n,
    int k,
    float alpha,
    float beta);

template <typename Scalar>
struct Multiplier;

template <
    typename Scalar,
    DestOp kOp,
    typename AccessorC,
    typename AccessorA,
    typename AccessorB,
    typename MultScalar>
void CudaProduct(
    DestinationAccessor<kOp, AccessorC>& C,
    AccessorA const& A,
    AccessorB const& B,
    int m,
    int n,
    int k,
    Multiplier<MultScalar> factor) {
  // Check what type of operations should be done on C.
  auto cCoef = (kOp == DestOp::Set || kOp == DestOp::NegSet) ? Scalar{0} : Scalar{1};
  Scalar prodCoef = factor.template Value<Scalar>() *
      ((kOp == DestOp::NegSet || kOp == DestOp::Sub) ? Scalar{-1} : Scalar{1});
  auto dataA = A.ScaledData();
  auto dataB = B.ScaledData();
  auto dataC = C.accessor.DestData();

  ExecuteCudaProduct(dataC, dataA, dataB, m, n, k, prodCoef, cCoef);
}

} // namespace mochi::details
