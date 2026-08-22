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

#include <mochi_core/geometry/tetrahedral_mesh.h>
#include <mochi_core/geometry/triangular_mesh.h>
#include <mochi_core/utils/mesh_embedding.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/task_scheduler.h>
#include <mochi_core/utils/time.h>

namespace mochi {

/*************************************************************************************************/

LinearMeshEmbedding::LinearMeshEmbedding(
    size_t entriesPerValue,
    Span<int const> indices,
    Span<real const> weights) {
  MOCHI_ASSERT_VERBOSE(entriesPerValue > 0);
  MOCHI_ASSERT_VERBOSE(indices.size() > 0 && indices.size() % entriesPerValue == 0);
  MOCHI_ASSERT_VERBOSE(indices.size() == weights.size());

  _entriesPerValue = entriesPerValue;
  _expectedDstSize = indices.size() / entriesPerValue;
  _indices.reserve(indices.size());
  _weights.reserve(weights.size());
  _indices.insert(_indices.end(), indices.begin(), indices.end());
  _weights.insert(_weights.end(), weights.begin(), weights.end());

  bool hasNegativeIndices = false;
  for (auto& i : _indices) {
    if (i < 0) {
      hasNegativeIndices = true;
      i = 0; // Wrong but safe
    }
  }
  if (hasNegativeIndices) {
    MOCHI_LOG_WARNING("LinearMeshEmbedding detected negative indices. Clamping to zero.");
  }
}

void LinearMeshEmbedding::Update(Span<real const> src, Span<real> dst) const {
  MOCHI_ASSERT_VERBOSE(dst.size() == _expectedDstSize);

  for (size_t i = 0, j = 0; i < dst.size(); ++i) {
    dst[i] = 0.0_r;
    for (size_t m = 0; m < _entriesPerValue; ++m, ++j) {
      MOCHI_ASSERT_VERBOSE(_indices[j] >= 0 && _indices[j] < src.size());
      dst[i] += _weights[j] * src[_indices[j]];
    }
  }
}

void LinearMeshEmbedding::Update(Span<Real3 const> src, Span<Real3> dst) const {
  MOCHI_PROFILE_SCOPE();
  // Multithreaded SIMD Implementation:
  MOCHI_ASSERT_VERBOSE(dst.size() == _expectedDstSize);
  real const* srcBase = Flatten(src).data();
  real const* weightsBase = _weights.data();
  real* dstBase = Flatten(dst).data();
  ParallelForN("LinearMeshEmbedding", isize(dst), 128, [&](int i) {
    int jBegin = i * (int)_entriesPerValue;
    int jEnd = jBegin + (int)_entriesPerValue;
    Vec4r dstValue = SimdZero();
    for (int j = jBegin; j < jEnd; ++j) {
      MOCHI_ASSERT_VERBOSE(_indices[j] < src.size());
      Vec4r srcValue = Load<3, Vec4r>(&srcBase[3 * _indices[j]]);
      dstValue += srcValue * Broadcast<Vec4r>(&weightsBase[j]);
    }
    Store<3>(&dstBase[i * 3], dstValue);
  });

  // Reference Implementation:
  //
  // for (size_t i = 0, j = 0; i < dst.size(); ++i) {
  //   dst[i] = 0.0_r;
  //   for (size_t m = 0; m < _entriesPerValue; ++m, ++j) {
  //     if (_indices[j] >= 0) {
  //       MOCHI_ASSERT_VERBOSE(_indices[j] < src.size());
  //       dst[i] += _weights[j] * src[_indices[j]];
  //     }
  //   }
  // }
}

/*************************************************************************************************/
} // namespace mochi
