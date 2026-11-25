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

#if MOCHI_USE_REFLECTION

#include <mochi_core/utils/error.h>

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

#define IMPL_MOCHI_THROW_ON_ERROR(fn, numArgs) \
  ::mochi::details::MakeThrowOnError<fn>(      \
      decltype(::SReflect::detail::FnTypes(fn)){}, std::make_index_sequence<(numArgs) + 1>{})

// MOCHI_PROPERTY_WITH_ERROR(name, getter) or MOCHI_PROPERTY_WITH_ERROR(name, getter, setter)
#define MOCHI_PROPERTY_WITH_ERROR(name, ...)                                                       \
  IMPL_SR_EXPAND(IMPL_SR_EXPAND(IMPL_SR_PROPERTY_SELECT(                                           \
      __VA_ARGS__, IMPL_MOCHI_PROPERTY_WITH_ERROR_RW, IMPL_MOCHI_PROPERTY_WITH_ERROR_RO, unused))( \
      name, __VA_ARGS__))
// MSVC requires inline lambdas to be named before using them as template arguments.
#define IMPL_MOCHI_PROPERTY_WITH_ERROR_RO(name, getter)                   \
  {                                                                       \
    constexpr auto mochiGetter = getter;                                  \
    IMPL_SR_Property_Get(name, IMPL_MOCHI_THROW_ON_ERROR(mochiGetter, 0)) \
  }
#define IMPL_MOCHI_PROPERTY_WITH_ERROR_RW(name, getter, setter) \
  {                                                             \
    constexpr auto mochiGetter = getter;                        \
    constexpr auto mochiSetter = setter;                        \
    IMPL_SR_Property_GetSet(                                    \
        name,                                                   \
        IMPL_MOCHI_THROW_ON_ERROR(mochiGetter, 0),              \
        IMPL_MOCHI_THROW_ON_ERROR(mochiSetter, 1))              \
  }
#define MOCHI_METHOD_WITH_ERROR(name, fn)                                                        \
  {                                                                                              \
    constexpr auto mochiFn = fn;                                                                 \
    IMPL_SR_Method(                                                                              \
        name, IMPL_MOCHI_THROW_ON_ERROR(mochiFn, ::SReflect::detail::CountMethodArgNames(name))) \
  }

namespace mochi::details {

inline void ThrowIfError(Error const& error) {
  if (!error.IsOK()) {
    throw std::runtime_error(error.GetDescription());
  }
}

// Params starts with the object. The indices select the reflected arguments, excluding Error&.
template <auto kFn, class R, class... Params, std::size_t... kArgIndices>
auto MakeThrowOnError(
    SReflect::detail::MethodTypes<false, R, Params...>,
    std::index_sequence<kArgIndices...>) {
  return +[](std::tuple_element_t<kArgIndices, std::tuple<Params...>>... args) -> R {
    if constexpr (sizeof...(kArgIndices) == sizeof...(Params)) {
      return std::invoke(kFn, std::forward<decltype(args)>(args)...);
    } else {
      static_assert(
          sizeof...(Params) == sizeof...(kArgIndices) + 1,
          "Reflected arguments must match the function without its trailing Error&.");
      static_assert(
          std::is_same_v<
              std::tuple_element_t<sizeof...(kArgIndices), std::tuple<Params...>>,
              Error&>,
          "The omitted parameter must be Error&.");
      Error error;
      if constexpr (std::is_void_v<R>) {
        std::invoke(kFn, std::forward<decltype(args)>(args)..., error);
        ThrowIfError(error);
      } else {
        R result = std::invoke(kFn, std::forward<decltype(args)>(args)..., error);
        ThrowIfError(error);
        return result;
      }
    }
  };
}

} // namespace mochi::details

#else // !MOCHI_USE_REFLECTION

#define MOCHI_PROPERTY_WITH_ERROR(name, ...)
#define MOCHI_METHOD_WITH_ERROR(name, fn)

#endif // MOCHI_USE_REFLECTION
