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

#include <functional>
#include <optional>

namespace mochi::rom {

struct ModelProperties {
  // The dimension of the base ROM, not including pivot degrees of freedom
  int baseDim = 0;
  // The dimension of the entire ROM, including pivot degrees of freedom
  int reducedDofsDim = 0; // Number of DoFs
  int reducedPoseDim = 0; // Size of the pose representation (possibly different from DoFs)
  // The dimension of the full-order output space
  int outputDim = 0;
};

using ToRawFunc = std::function<void(ColumnVectorView<real>)>;
using FromRawFunc = std::function<void(ColumnVectorView<real const>)>;
using FromIncrementFunc =
    std::function<void(ColumnVectorView<real const>, ColumnVectorView<real const>)>;
} // namespace mochi::rom
