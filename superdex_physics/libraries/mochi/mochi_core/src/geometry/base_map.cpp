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

#include <mochi_core/geometry/base_map.h>
#include <mochi_core/utils/matrix_utils.h>

using namespace mochi;

void BaseMap::ReindexResult(
    Span<int const> indsPoints,
    Span<int> outIndices,
    DynamicArray<VMatrix3x3r>* outMapJac,
    DynamicArray<ColliderJacDofs>* outDofsJac) {
#if MOCHI_ASSERT_VERBOSE_ENABLED
  MOCHI_ASSERT_VERBOSE(!outMapJac || (outMapJac->size() == indsPoints.size()), "Size mismatch");
  MOCHI_ASSERT_VERBOSE(!outDofsJac || (outDofsJac->size() == indsPoints.size()), "Size mismatch");
  if (!outIndices.empty()) {
    MOCHI_ASSERT_VERBOSE(outIndices[0] < indsPoints.size(), "Unexpected index");
  }
  for (int i = 1; i < outIndices.size(); ++i) {
    MOCHI_ASSERT_VERBOSE(outIndices[i - 1] < outIndices[i], "Indices are not in growing order");
    MOCHI_ASSERT_VERBOSE(outIndices[i] < indsPoints.size(), "Unexpected index");
  }
#endif

  // If outIndices has the same size as indsPoints, just copy the values. No need to reindex
  // Jacobians.
  if (outIndices.size() == indsPoints.size()) {
    std::copy(indsPoints.begin(), indsPoints.end(), outIndices.begin());
    return;
  }

  // Remap outMapJac if requested
  if (outMapJac) {
    for (int i = 0; i < outIndices.size(); i++) {
      (*outMapJac)[i] = (*outMapJac)[outIndices[i]];
    }
    outMapJac->resize(outIndices.size());
  }

  // Remap outDofsJac if requested
  if (outDofsJac) {
    for (int i = 0; i < outIndices.size(); i++) {
      (*outDofsJac)[i] = (*outDofsJac)[outIndices[i]];
    }
    outDofsJac->resize(outIndices.size());
  }

  // Remap point indices.
  for (auto& index : outIndices) {
    index = indsPoints[index];
  }
}
