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

#include <mochi_core/utils/reflection.h>

// When compiling with GCC on Windows, the mingw toolchain's rpcndr.h defines hyper as __int64.
// No thank you.
#ifdef hyper
#undef hyper
#endif

namespace mochi::rom::hyper {

constexpr int kAllElements = 1;
constexpr int kNoElements = -1;

// this struct defines the basic parameters that define
// a sample mesh when subsampling the boundary and internal elements
struct BoundaryAndInternalElementsSubsamplingParameters {
  int stepSizeForBoundaryElementsSelection = kAllElements;
  int stepSizeForInteriorElementsSelection = kAllElements;

  MOCHI_STRUCT_BEGIN(mochi::rom::hyper::BoundaryAndInternalElementsSubsamplingParameters)
  MOCHI_FIELD(stepSizeForBoundaryElementsSelection)
  MOCHI_FIELD(stepSizeForInteriorElementsSelection)
  MOCHI_STRUCT_END()
};

enum SdfLowerBoundAnchorSelection {
  // Use only the sample at the current node to evaluate the lower bound
  Self,
  // Combine all samples at the node and its ancestors to evaluate the lower bound
  Ancestor,
  // Combine all samples at the node, its siblings, and its ancestor to evaluate the lower bound
  AncestorSibling,
  // Count must come last
  Count
};

} // namespace mochi::rom::hyper

MOCHI_ENUM_BEGIN(mochi::rom::hyper::SdfLowerBoundAnchorSelection)
MOCHI_ENUM_ITEM(Self)
MOCHI_ENUM_ITEM(Ancestor)
MOCHI_ENUM_ITEM(AncestorSibling)
MOCHI_ENUM_COUNT(Count)
MOCHI_ENUM_END()
