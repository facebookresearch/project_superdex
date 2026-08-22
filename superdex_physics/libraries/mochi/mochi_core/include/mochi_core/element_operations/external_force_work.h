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

#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/vmatrix.h>

#include <functional>

namespace mochi {

// Arguments to the TractionWorkFunc
struct TractionWorkArgs {
  Vec4r position = {};
  Vec4r predExpPos = {};
  Vec4r predExpVel = {};
  Vec4r refPosition = {};
  int elementIndex = -1;
  int quadPointIndex = -1;
  double* merit = nullptr;
  Vec4r* outForce = nullptr;
  VMatrix3x3r* outDForce = nullptr;
  int baseElementIndex = -1;
};

// User provided function for applied traction. Return false if all outputs are zero.
// Each Vec4r has 4 floats, but only the first 3 (space dims) are important.
using TractionFunc = std::function<bool(TractionWorkArgs const& args)>;

} // namespace mochi
