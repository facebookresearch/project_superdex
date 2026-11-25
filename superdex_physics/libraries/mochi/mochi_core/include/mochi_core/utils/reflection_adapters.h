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

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/span.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace mochi::reflection {

class ReflectionConversionStorage {
 public:
  template <class T>
  Span<T> KeepArray(DynamicArray<T> values) {
    auto array = std::make_shared<DynamicArray<T>>(std::move(values));
    auto view = MakeSpan(*array);
    _arrays.push_back(std::move(array));
    return view;
  }

 private:
  DynamicArray<std::shared_ptr<void>> _arrays;
};

template <class T>
struct ReflectionValueAdapter;

template <class T>
using ReflectionValue = typename ReflectionValueAdapter<std::remove_cvref_t<T>>::Type;

template <class T>
ReflectionValue<T> ToReflectionValue(T const& value);

template <class T, class Scope>
std::remove_cvref_t<T> FromReflectionValue(
    ReflectionValue<T> const& value,
    Scope const& scope,
    ReflectionConversionStorage& storage);

template <class T>
struct ReflectionValueAdapter {
  using Type = T;
  static Type ToReflection(T const& value) {
    return value;
  }
  template <class Scope>
  static T ToApiValue(Type const& value, Scope const&, ReflectionConversionStorage&) {
    return value;
  }
};

template <class T>
struct ReflectionObjectResolver;

template <class T>
struct ReflectionValueAdapter<T*> {
  using Type = std::remove_cvref_t<decltype(std::declval<T const&>().GetHandle())>;
  static Type ToReflection(T* value) {
    return value != nullptr ? value->GetHandle() : Type{};
  }
  template <class Scope>
  static T* ToApiValue(Type const& value, Scope const& scope, ReflectionConversionStorage&) {
    return ReflectionObjectResolver<T>::Resolve(value, scope);
  }
};

template <>
struct ReflectionValueAdapter<char const*> {
  using Type = std::string;
  static Type ToReflection(char const* value) {
    return value != nullptr ? Type{value} : Type{};
  }
  template <class Scope>
  static char const* ToApiValue(Type const& value, Scope const&, ReflectionConversionStorage&) {
    return value.c_str();
  }
};

template <class T>
  requires std::is_void_v<T>
struct ReflectionValueAdapter<T*> {
  using Type = uint64_t;
  static_assert(sizeof(uintptr_t) <= sizeof(Type));
  static Type ToReflection(T* value) {
    return value == nullptr ? 0 : static_cast<Type>(reinterpret_cast<uintptr_t>(value));
  }
  template <class Scope>
  static T* ToApiValue(Type value, Scope const&, ReflectionConversionStorage&) {
    uintptr_t const pointer = static_cast<uintptr_t>(value);
    if (static_cast<Type>(pointer) != value) {
      throw std::invalid_argument("Value does not fit in a pointer");
    }
    return value == 0 ? nullptr : reinterpret_cast<T*>(pointer);
  }
};

template <>
struct ReflectionValueAdapter<std::string_view> {
  using Type = std::string;
  static Type ToReflection(std::string_view value) {
    return Type{value};
  }
  template <class Scope>
  static std::string_view
  ToApiValue(Type const& value, Scope const&, ReflectionConversionStorage&) {
    return value;
  }
};

template <class T>
struct ReflectionValueAdapter<Span<T>> {
  using Element = std::remove_cv_t<T>;
  using Type = DynamicArray<ReflectionValue<Element>>;
  static Type ToReflection(Span<T> value) {
    Type result;
    result.reserve(value.size());
    for (auto const& element : value) {
      result.push_back(ToReflectionValue(element));
    }
    return result;
  }
  template <class Scope>
  static Span<T>
  ToApiValue(Type const& value, Scope const& scope, ReflectionConversionStorage& storage) {
    if constexpr (std::is_const_v<T> && std::is_same_v<Element, ReflectionValue<Element>>) {
      return MakeSpan(const_cast<Type&>(value));
    } else {
      DynamicArray<Element> result;
      result.reserve(value.size());
      for (auto const& element : value) {
        result.push_back(FromReflectionValue<Element>(element, scope, storage));
      }
      return storage.KeepArray(std::move(result));
    }
  }
};

template <class T>
struct ReflectionValueAdapter<DynamicArray<T>> {
  using Type = DynamicArray<ReflectionValue<T>>;
  static Type ToReflection(DynamicArray<T> const& value) {
    return ReflectionValueAdapter<Span<T const>>::ToReflection(MakeConstSpan(value));
  }
  template <class Scope>
  static DynamicArray<T>
  ToApiValue(Type const& value, Scope const& scope, ReflectionConversionStorage& storage) {
    DynamicArray<T> result;
    result.reserve(value.size());
    for (auto const& element : value) {
      result.push_back(FromReflectionValue<T>(element, scope, storage));
    }
    return result;
  }
};

template <class T>
struct ReflectionValueAdapter<std::optional<T>> {
  using Type = std::optional<ReflectionValue<T>>;
  static Type ToReflection(std::optional<T> const& value) {
    return value ? Type{ToReflectionValue(*value)} : std::nullopt;
  }
  template <class Scope>
  static std::optional<T>
  ToApiValue(Type const& value, Scope const& scope, ReflectionConversionStorage& storage) {
    return value ? std::optional<T>{FromReflectionValue<T>(*value, scope, storage)} : std::nullopt;
  }
};

template <class T>
ReflectionValue<T> ToReflectionValue(T const& value) {
  return ReflectionValueAdapter<std::remove_cvref_t<T>>::ToReflection(value);
}

template <class T, class Scope>
std::remove_cvref_t<T> FromReflectionValue(
    ReflectionValue<T> const& value,
    Scope const& scope,
    ReflectionConversionStorage& storage) {
  return ReflectionValueAdapter<std::remove_cvref_t<T>>::ToApiValue(value, scope, storage);
}

template <class T>
class ReflectionArgument {
 public:
  using Native = std::remove_cvref_t<T>;
  template <class Scope>
  explicit ReflectionArgument(ReflectionValue<T> const& value, Scope const& scope)
      : _value(FromReflectionValue<T>(value, scope, _storage)) {}
  Native& Get() {
    return _value;
  }

 private:
  ReflectionConversionStorage _storage;
  Native _value;
};

} // namespace mochi::reflection

#endif // MOCHI_USE_REFLECTION
