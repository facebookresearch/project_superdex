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

#include "mochi_common_components.h"

namespace mochi::rom {

inline void AddAffineShiftVector(
    Span<int const> activeNodes,
    ColumnVectorView<real const> shift,
    ColumnVectorView<real> outDisplacements) {
  if (!activeNodes.empty()) {
    for (int nodeId : activeNodes) {
      outDisplacements(nodeId * 3 + 0) += shift(nodeId * 3 + 0);
      outDisplacements(nodeId * 3 + 1) += shift(nodeId * 3 + 1);
      outDisplacements(nodeId * 3 + 2) += shift(nodeId * 3 + 2);
    }
  } else {
    outDisplacements += shift;
  }
}

} // namespace mochi::rom
