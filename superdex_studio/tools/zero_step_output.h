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

#include <mochi_core/utils/error.h>

#include <cstdio>
#include <string>

namespace superdex::studio {

inline int PublishZeroStepJson(std::string const& json, mochi::Error const& error) {
  if (!error.IsOK()) {
    std::fprintf(stderr, "%s\n", error.GetDescription());
    return 1;
  }
  std::printf("%s", json.c_str());
  return 0;
}

} // namespace superdex::studio
