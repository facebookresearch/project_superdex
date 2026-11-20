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

#include <superdex_physics.h>

#include <cstdint>

namespace superdex::robotics {

/* @brief Episode seed and stable stream identity supplied by the experiment client. */
struct SensorRandomParams {
  /* Reflected JSON preserves all bits as a signed integer; high-bit seeds appear negative. */
  uint64_t episodeSeed = 0;
  DynamicString processKey;

  MOCHI_STRUCT_BEGIN(superdex::robotics::SensorRandomParams)
  MOCHI_FIELD(episodeSeed);
  MOCHI_FIELD(processKey);
  MOCHI_STRUCT_END()
};

} // namespace superdex::robotics
