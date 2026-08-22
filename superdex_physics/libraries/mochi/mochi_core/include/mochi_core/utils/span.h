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
#include <mochi_core/utils/concepts.h>
#include <mochi_core/utils/debug.h>

#if MOCHI_USE_EXTERN_TEMPLATE
#include <mochi_core/utils/nd_array.h> // For extern template declarations
#endif // MOCHI_USE_EXTERN_TEMPLATE

#include <cstddef>
#include <iterator>
#include <string>
#include <type_traits>
#include <vector>

namespace mochi {

/**************************************************************************************************
  Span<T>

  Wraps a pointer and length. If type T is a const type, then span provides a read-only view into
  the range of elements. The span does not own the memory nor guarantee its lifespan. To help
  reduce register pressure, the type used to encode the span length is also templatized.
*/
template <typename T, typename SizeT = size_t>
class Span {
 public:
  using value_type = T;
  using size_type = SizeT;
  static_assert(std::is_integral_v<SizeT>, "SizeT should be a signed or unsigned integral type");

  MOCHI_ANY MOCHI_FORCE_INLINE constexpr Span() = default;

  // Construct from pointer + size
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr Span(T* ptr, SizeT len) : _ptr(ptr), _len(len) {}

  // Unfortunately, overload resolution would be ambiguous if someone wrote "Span{ptr, 0}" because 0
  // could convert to SizeT or to T* (sigh). This template wrapper avoids that ambiguity.
  template <typename U>
  struct NotIntegral {
    NotIntegral(SizeT) = delete;
    NotIntegral(U u) : value(u) {}
    U value;
  };

  // Construct from the half open range [begin, end)
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr Span(T* begin, NotIntegral<T*> end)
      : _ptr(begin), _len(static_cast<SizeT>(end.value - begin)) {}

  // Construct from any container whose data pointer is implicitly convertible to T* (e.g.,
  // std::vector<int> -> Span<int const>, Span<int> -> Span<int const>). The SFINAE constraint
  // excludes Span with the same template arguments to avoid hijacking the implicit copy/move
  // constructors.
  template <class ContainerT, MOCHI_CONCEPT((!std::is_same_v<std::decay_t<ContainerT>, Span>))>
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr Span(ContainerT&& container)
      : _ptr(std::data(container)), _len(static_cast<SizeT>(std::size(container))) {}

  // Index operator. May return a const reference if type T is const.
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr T& operator[](SizeT index) const {
    MOCHI_ASSERT_VERBOSE(index >= 0 && index < _len, "Index out-of-range");
    return _ptr[index];
  }

  // Bool operator to mimic non-null pointer semantics. Same empty() == false.
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE explicit constexpr operator bool() const noexcept {
    return !empty();
  }

  // Spans are equal if all elements are equal
  template <typename T2, typename S2>
  [[nodiscard]] MOCHI_ANY bool operator==(Span<T2, S2> const& rhs) const {
    if (static_cast<size_type>(rhs.size()) != _len) {
      return false;
    }
    for (size_type i = 0; i < _len; ++i) {
      if (rhs[i] != _ptr[i]) {
        return false;
      }
    }
    return true;
  }

  template <typename T2, typename S2>
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE bool operator!=(Span<T2, S2> const& rhs) const {
    return !(*this == rhs);
  }

  // These member names are lower case in keeping with the std library conventions.
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr SizeT size() const {
    return _len;
  }
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr bool empty() const {
    return _len == 0;
  }
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr T* begin() const {
    return _ptr;
  }
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr T* end() const {
    return _ptr + _len;
  }
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr T* data() const {
    return _ptr;
  }
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr Span subspan(SizeT start, SizeT len) const {
    MOCHI_ASSERT_VERBOSE(start + len <= _len, "Subspan out-of-range");
    return Span(_ptr + start, len);
  }
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr Span subspan(SizeT start) const {
    MOCHI_ASSERT_VERBOSE(start <= _len, "Start index out-of-range");
    return Span(_ptr + start, _len - start);
  }
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr T& front() const {
    return (*this)[0];
  }
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr T& back() const {
    return (*this)[_len - 1];
  }

 protected:
  T* _ptr = nullptr;
  SizeT _len = 0;
};

// Convert a std::vectors or c-style array to a Spans. The template argument can be deduced
template <typename ContainerT>
[[nodiscard]] MOCHI_FORCE_INLINE auto MakeSpan(ContainerT& container);

/**************************************************************************************************
  Span Inlines
*/

template <typename ContainerT>
[[nodiscard]] MOCHI_FORCE_INLINE auto MakeSpan(ContainerT& container) {
  auto* ptr = std::data(container);
  size_t len = std::size(container);
  return Span<std::remove_reference_t<decltype(*ptr)>>{ptr, len};
}

template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE auto MakeSpan(T (&arr)[N]) {
  return Span<T>{arr, N};
}

template <typename T, typename S>
[[nodiscard]] MOCHI_FORCE_INLINE Span<T, S> const& MakeSpan(Span<T, S> const& alreadyASpan) {
  return alreadyASpan;
}

template <typename ContainerT>
[[nodiscard]] MOCHI_FORCE_INLINE auto MakeConstSpan(ContainerT const& container) {
  auto* ptr = std::data(container);
  size_t len = std::size(container);
  return Span<std::add_const_t<std::remove_reference_t<decltype(*ptr)>>>{ptr, len};
}

template <typename T, size_t N>
[[nodiscard]] MOCHI_FORCE_INLINE auto MakeConstSpan(T (&arr)[N]) {
  return Span<T const>{arr, N};
}

template <typename T, typename S>
[[nodiscard]] MOCHI_FORCE_INLINE Span<T const, S> MakeConstSpan(Span<T, S> const& alreadyASpan) {
  return alreadyASpan;
}

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE Span<T, size_t> MakeSingletonSpan(T& obj) {
  return {&obj, 1};
}

template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE Span<T const, size_t> MakeSingletonConstSpan(T const& obj) {
  return {&obj, 1};
}

#if MOCHI_USE_EXTERN_TEMPLATE
extern template class Span<int>;
extern template class Span<NdArray<int, 2>>;
extern template class Span<NdArray<int, 3>>;
extern template class Span<NdArray<int, 4>>;
extern template class Span<real>;
extern template class Span<NdArray<real, 3>>;
extern template class Span<int, int>;
extern template class Span<NdArray<int, 2>, int>;
extern template class Span<NdArray<int, 3>, int>;
extern template class Span<NdArray<int, 4>, int>;
extern template class Span<real, int>;
extern template class Span<NdArray<real, 3>, int>;
#endif // MOCHI_USE_EXTERN_TEMPLATE

// ScalarType specialization: recursively unwrap value type.
namespace details {
template <class T, class SizeT>
struct ScalarTypeDef<Span<T, SizeT>, void> {
  using type = ScalarType<T>;
};
} // namespace details

} // namespace mochi
