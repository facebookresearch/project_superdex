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

#include <mochi_core/test/batch_helpers.h>
#include <mochi_core/utils/batch_types.h>

#include <gtest/gtest.h>

#include <type_traits>

using namespace mochi;

/**************************************************************************************************
  BatchTypes static property checks.
*/

template <int kBatchSize>
void TestBatchTypesProperties() {
  using B = BatchTypes<kBatchSize>;
  using V = typename B::Real;
  using Vd = typename B::Double;
  using Vi = typename B::Int;

  static_assert(V::kSize >= kBatchSize, "Scalar register must fit kBatchSize lanes");
  static_assert(
      V::kSize % Simd<real>::kSize == 0, "Scalar width must be a multiple of native width");
  static_assert(Vd::kSize == V::kSize, "Double must have same lane count as Scalar");
  static_assert(Vi::kSize == V::kSize, "Int must have same lane count as Scalar");

  static_assert(B::Real3::size() == 3);
  static_assert(B::Real6::size() == 6);
  static_assert(B::Real9::size() == 9);
  static_assert(B::SymMatrix2x2::size() == 3);
  static_assert(B::SymMatrix3x3::size() == 6);

  static_assert(sizeof(typename B::Real3) == sizeof(V) * 3, "BatchReal3 size mismatch");
  static_assert(sizeof(typename B::Real6) == sizeof(V) * 6, "BatchReal6 size mismatch");
  static_assert(sizeof(typename B::Real9) == sizeof(V) * 9, "BatchReal9 size mismatch");
  static_assert(sizeof(typename B::Real2x2) == sizeof(V) * 2 * 2, "BatchReal2x2 size mismatch");
  static_assert(sizeof(typename B::Real3x3) == sizeof(V) * 3 * 3, "BatchReal3x3 size mismatch");
  static_assert(
      sizeof(typename B::SymMatrix2x2) == sizeof(V) * 3, "BatchSymMatrix2x2 size mismatch");
  static_assert(
      sizeof(typename B::SymMatrix3x3) == sizeof(V) * 6, "BatchSymMatrix3x3 size mismatch");

  static_assert(std::is_same_v<BatchReal<kBatchSize>, V>);
  static_assert(std::is_same_v<BatchDouble<kBatchSize>, Vd>);
  static_assert(std::is_same_v<BatchInt<kBatchSize>, Vi>);
  static_assert(std::is_same_v<BatchReal3<kBatchSize>, typename B::Real3>);
  static_assert(std::is_same_v<BatchReal6<kBatchSize>, typename B::Real6>);
  static_assert(std::is_same_v<BatchReal9<kBatchSize>, typename B::Real9>);
  static_assert(std::is_same_v<BatchReal2x2<kBatchSize>, typename B::Real2x2>);
  static_assert(std::is_same_v<BatchReal3x3<kBatchSize>, typename B::Real3x3>);
  static_assert(std::is_same_v<BatchSymMatrix2x2<kBatchSize>, typename B::SymMatrix2x2>);
  static_assert(std::is_same_v<BatchSymMatrix3x3<kBatchSize>, typename B::SymMatrix3x3>);
}

MOCHI_BATCH_TEST(BatchTypes, Properties, TestBatchTypesProperties)
