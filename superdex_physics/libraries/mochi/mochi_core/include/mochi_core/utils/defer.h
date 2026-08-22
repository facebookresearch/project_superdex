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

#include <mochi_core/mochi_config.h>

#include <utility>

namespace mochi {

/**
  Helper class used to implement the mochi::Defer utility and the MOCHI_DEFER macro.
*/
namespace detail {
template <class F>
class DeferHelper final {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(DeferHelper);

 public:
  explicit DeferHelper(F&& f) : _fun(std::move(f)) {};
  ~DeferHelper() {
    _fun();
  }

 private:
  F _fun;
};
} // namespace detail

/**
  Utility for deferring arbitrary execution to the end of a C++ block, RAII-style.
  Used to ensure proper cleanup, even if the function returns early or throws.
  If you prefer, you can also use the MOCHI_DEFER macro, which makes it even easier.

  Similar to Go's "defer" or Python's "with".

  Example usage:
    {
      FILE* f = fopen("something", "r");
      auto closeOnExit = Defer([&](){ fclose(f); });

      if (complicated_code) {
        return;  // fclose(f) will be called here automatically
      }

      // fclose(f) will be called here, if we didn't return or throw already.
    }
*/
template <class F>
[[nodiscard]] auto Defer(F&& fun) {
  return detail::DeferHelper<F>(std::forward<F>(fun));
}

/**
  Macro for deferring arbitrary execution to the end of a C++ block, RAII-style.
  Used to ensure proper cleanup, even if the function returns early or throws.
  If you prefer, you can also use the non-template AutoCleanup(fn) syntax.

  Similar to Go's "defer" or Python's "with".

  Example usage:
    {
      FILE* f = fopen("something", "r");
      MOCHI_DEFER(fclose(f));

      if (complicated_code) {
        return;  // fclose(f) will be called here automatically
      }

      // fclose(f) will be called here, if we didn't return or throw already.
    }
*/

#define MOCHI_DEFER(statement) \
  auto MOCHI_PP_CAT(auto_cleanup_, __LINE__) = ::mochi::Defer([&]() { statement; })

} // namespace mochi
