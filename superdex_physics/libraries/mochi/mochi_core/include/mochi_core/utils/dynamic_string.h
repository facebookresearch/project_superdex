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

#include <mochi_core/memory/stl_allocator.h>
#include <mochi_core/utils/reflection.h>

#include <string>
#include <string_view>

namespace mochi {

/**
 * @brief Alias for std::string using a polymorphic mochi::Allocator pointer.
 *
 * @note It is safe to return this type of string from a DLL exported method (unlike std::string).
 *
 * @code{.cpp}
 * DynamicString myString; // Uses GetDefaultAllocator()
 * DynamicString myStringWithValue("Hello!"); // Uses GetDefaultAllocator()
 * DynamicString myAllocatedString(&allocator);
 * DynamicString myAllocatedStringWithValue("Hello!", &allocator);
 * @endcode
 */
using DynamicString = std::basic_string<char, std::char_traits<char>, StlAllocator<char>>;

} // namespace mochi

/// @cond
template <>
struct std::hash<mochi::DynamicString> {
  std::size_t operator()(mochi::DynamicString const& s) const noexcept {
    return std::hash<std::string_view>{}(std::string_view{s.data(), s.size()});
  }
};
/// @endcond

// Reflection support for mochi::String
#if MOCHI_USE_REFLECTION
template <>
struct SReflectTypeTraits<mochi::DynamicString> {
  static constexpr SReflect::CoreType coreType = SReflect::CoreType::CT_string;
  static SReflect::StringTypeInfo const& GetTypeInfo();
};

#endif // MOCHI_USE_REFLECTION
