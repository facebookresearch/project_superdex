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

namespace mochi {

/**************************************************************************************************
  Visitor Utilities
*/
template <class... Ts>
struct OverloadVisitor : Ts... {
  using Ts::operator()...;

  // work around for msvc ebo (empty base optimization) bug.
  // https://developercommunity.visualstudio.com/t/runtime-stack-corruption-using-stdvisit/346200
  int dummy = 0;
};
template <class... Ts>
OverloadVisitor(Ts...) -> OverloadVisitor<Ts...>;

} // namespace mochi
