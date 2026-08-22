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
#include <mochi_core/utils/reflection.h>

#include <algorithm>
#include <array>
#include <type_traits>

namespace mochi {

/**************************************************************************************************
  MultiplyConstexpr - Return the product of all arguments as a constexpr. Used in later
  declarations.
*/
template <typename A>
MOCHI_FORCE_INLINE constexpr A MultiplyConstexpr(A a) {
  return a;
}
template <typename A, typename... ARGS>
MOCHI_FORCE_INLINE constexpr A MultiplyConstexpr(A a, ARGS... args) {
  return a * MultiplyConstexpr(args...);
}

/**************************************************************************************************
  NdArray<T, D0, DIMS...>

  Template for fixed-sized N-dimensional arrays like Int2, Real3, Matrix3x3r, etc...
  Most methods can be used in constant (compile-time) expressions. NdArray currently has no
  special alignment requirements and does not use SIMD explicitly.
*/
template <typename T, size_t D0 = 1, size_t... DIMS>
class NdArray final {
 public:
  // Size of each dimension
  constexpr static size_t num_dims = 1 + sizeof...(DIMS);
  constexpr static size_t dims[num_dims] = {D0, DIMS...};
  constexpr static size_t flattened_size = MultiplyConstexpr(D0, DIMS...);

  // If this NdArray is 1D, then "value_type" is just "T". Else, it is an array of dimension (N-1)
  using value_type = typename std::conditional_t<num_dims == 1, T, NdArray<T, DIMS...>>;

  // Indicates the element type "T" in the last dimension of the array.
  using element_type = T;

  // Construct default
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr NdArray() = default;

  // Construct from exactly D0 parameters (array elements that are convertible to T).
  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  template <
      typename... U,
      std::enable_if_t<
          (sizeof...(U) == D0) && std::conjunction_v<std::is_convertible<value_type, U>...>,
          void*> = nullptr>
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr NdArray(U const&... p) : _data{p...} {}

  // clang-format off
  // These member names are lower case in keeping with the std library conventions.
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr static int size() { return D0; } // size of 1st dimension
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr value_type* data() { return _data; }
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr value_type const* data() const { return _data; }
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr auto begin() { return _data; }
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr auto begin() const { return _data; }
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr auto end() { return _data + D0; }
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr auto end() const { return _data + D0; }
  // clang-format on

  // Index operator
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr value_type const& operator[](size_t i) const;
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr value_type& operator[](size_t i);

 private:
  value_type _data[D0];
};

// Type-trait to detect NdArray types
template <typename T>
struct IsNdArray : std::false_type {};
template <typename T, size_t D0, size_t... DIMS>
struct IsNdArray<NdArray<T, D0, DIMS...>> : std::true_type {};
template <typename T>
static constexpr bool kIsNdArray = IsNdArray<std::decay_t<T>>::value;

// Common 1D floating-point arrays types
using Real1 = NdArray<real, 1>;
using Real2 = NdArray<real, 2>;
using Real3 = NdArray<real, 3>;
using Real4 = NdArray<real, 4>;
using Real5 = NdArray<real, 5>;
using Real6 = NdArray<real, 6>;

// For cases that always need 32-bit floats
using Float2 = NdArray<float, 2>;
using Float3 = NdArray<float, 3>;
using Float4 = NdArray<float, 4>;

// For cases that always need 64-bit floats
using Double2 = NdArray<double, 2>;
using Double3 = NdArray<double, 3>;
using Double4 = NdArray<double, 4>;

// Common 2D floating point array types
using Matrix2x2r = NdArray<real, 2, 2>;
using Matrix2x3r = NdArray<real, 2, 3>;
using Matrix3x2r = NdArray<real, 3, 2>;
using Matrix3x3r = NdArray<real, 3, 3>;
using Matrix4x3r = NdArray<real, 4, 3>;
using Matrix4x4r = NdArray<real, 4, 4>;

using Matrix2x2f = NdArray<float, 2, 2>;
using Matrix2x3f = NdArray<float, 2, 3>;
using Matrix3x2f = NdArray<float, 3, 2>;
using Matrix3x3f = NdArray<float, 3, 3>;
using Matrix4x3f = NdArray<float, 4, 3>;
using Matrix4x4f = NdArray<float, 4, 4>;

// Common 1D integral array types
using Int2 = NdArray<int, 2>;
using Int3 = NdArray<int, 3>;
using Int4 = NdArray<int, 4>;

// Common 4d floating point tensors
using Tensor3x3x3x3r = NdArray<real, 3, 3, 3, 3>;
using Tensor3x3x3r = NdArray<real, 3, 3, 3>;

using Tensor3x3x3x3f = NdArray<float, 3, 3, 3, 3>;
using Tensor3x3x3f = NdArray<float, 3, 3, 3>;

/**************************************************************************************************
  Loop unrolling macro
*/

// Unroll a statement BODY over i = 0, ..., D0-1 (manually unrolled when D0 <= 4). `D0` must be in
// scope. BODY may reference `i` as an index and must not introduce its own `i`. BODY is expanded as
// a single statement, so unguarded commas (outside parentheses) must be avoided.
#define MOCHI_DETAILS_UNROLL_D0(BODY) \
  if constexpr (D0 <= 4) {            \
    if constexpr (D0 > 0) {           \
      constexpr size_t i = 0;         \
      BODY;                           \
    }                                 \
    if constexpr (D0 > 1) {           \
      constexpr size_t i = 1;         \
      BODY;                           \
    }                                 \
    if constexpr (D0 > 2) {           \
      constexpr size_t i = 2;         \
      BODY;                           \
    }                                 \
    if constexpr (D0 > 3) {           \
      constexpr size_t i = 3;         \
      BODY;                           \
    }                                 \
  } else {                            \
    for (size_t i = 0; i < D0; ++i) { \
      BODY;                           \
    }                                 \
  }

/**************************************************************************************************
  NdArray Operators
*/

template <typename T, size_t D0, size_t... DIMS>
MOCHI_FORCE_INLINE constexpr NdArray<T, D0, DIMS...> operator-(NdArray<T, D0, DIMS...> const& a) {
  NdArray<T, D0, DIMS...> result{};
  MOCHI_DETAILS_UNROLL_D0(result[i] = -a[i]);
  return result;
}

template <typename T, size_t D0, size_t... DIMS>
MOCHI_FORCE_INLINE constexpr bool operator==(
    NdArray<T, D0, DIMS...> const& lhs,
    NdArray<T, D0, DIMS...> const& rhs) {
  bool isEqual = (lhs[0] == rhs[0]);
  if constexpr (D0 > 1) {
    isEqual &= (lhs[1] == rhs[1]);
  }
  if constexpr (D0 > 2) {
    isEqual &= (lhs[2] == rhs[2]);
  }
  if constexpr (D0 > 3) {
    isEqual &= (lhs[3] == rhs[3]);
  }
  if constexpr (D0 > 4) {
    for (size_t i = 4; i < D0; ++i) {
      isEqual &= (lhs[i] == rhs[i]);
    }
  }
  return isEqual;
}

template <typename T, size_t D0, size_t... DIMS>
MOCHI_FORCE_INLINE constexpr typename NdArray<T, D0, DIMS...>::value_type const&
NdArray<T, D0, DIMS...>::operator[](size_t i) const {
  MOCHI_ASSERT_VERBOSE(i < D0, "Index out-of-range");
  return _data[i];
}

template <typename T, size_t D0, size_t... DIMS>
MOCHI_FORCE_INLINE constexpr typename NdArray<T, D0, DIMS...>::value_type&
NdArray<T, D0, DIMS...>::operator[](size_t i) {
  MOCHI_ASSERT_VERBOSE(i < D0, "Index out-of-range");
  return _data[i];
}

template <typename T, size_t D0, size_t... DIMS>
MOCHI_FORCE_INLINE constexpr bool operator!=(
    NdArray<T, D0, DIMS...> const& lhs,
    NdArray<T, D0, DIMS...> const& rhs) {
  return !(lhs == rhs);
}

// NdArray memberwise math operators (+=, -=, *=, /=, +, -, *, /)
//
// In-place OP_EQ variants (NdArray += NdArray, NdArray += scalar) avoid creating a temporary. For
// large or nested NdArrays, the temporary may otherwise spill out of the register file.
#define MOCHI_DETAILS_NDARRAY_MEMBERWISE_OP(OP_EQ, OP)                          \
                                                                                \
  template <typename T, size_t D0, size_t... DIMS>                              \
  MOCHI_FORCE_INLINE constexpr NdArray<T, D0, DIMS...>& operator OP_EQ(         \
      NdArray<T, D0, DIMS...>& lhs, NdArray<T, D0, DIMS...> const& rhs) {       \
    MOCHI_DETAILS_UNROLL_D0(lhs[i] OP_EQ rhs[i]);                               \
    return lhs;                                                                 \
  }                                                                             \
                                                                                \
  template <typename T, size_t D0, size_t... DIMS>                              \
  MOCHI_FORCE_INLINE constexpr NdArray<T, D0, DIMS...>& operator OP_EQ(         \
      NdArray<T, D0, DIMS...>& lhs, T rhs) {                                    \
    MOCHI_DETAILS_UNROLL_D0(lhs[i] OP_EQ rhs);                                  \
    return lhs;                                                                 \
  }                                                                             \
                                                                                \
  template <typename T, size_t D0, size_t... DIMS, typename AnyRHS>             \
  MOCHI_FORCE_INLINE constexpr NdArray<T, D0, DIMS...>& operator OP_EQ(         \
      NdArray<T, D0, DIMS...>& lhs, AnyRHS const& rhs) {                        \
    MOCHI_DETAILS_UNROLL_D0(lhs[i] OP_EQ rhs);                                  \
    return lhs;                                                                 \
  }                                                                             \
                                                                                \
  template <typename T, size_t D0, size_t... DIMS>                              \
  MOCHI_FORCE_INLINE constexpr NdArray<T, D0, DIMS...> operator OP(             \
      NdArray<T, D0, DIMS...> const& lhs, NdArray<T, D0, DIMS...> const& rhs) { \
    NdArray<T, D0, DIMS...> result{};                                           \
    MOCHI_DETAILS_UNROLL_D0(result[i] = lhs[i] OP rhs[i]);                      \
    return result;                                                              \
  }                                                                             \
                                                                                \
  template <typename T, size_t D0, size_t... DIMS>                              \
  MOCHI_FORCE_INLINE constexpr NdArray<T, D0, DIMS...> operator OP(             \
      NdArray<T, D0, DIMS...> const& lhs, T rhs) {                              \
    NdArray<T, D0, DIMS...> result{};                                           \
    MOCHI_DETAILS_UNROLL_D0(result[i] = lhs[i] OP rhs);                         \
    return result;                                                              \
  }                                                                             \
                                                                                \
  template <typename T, size_t D0, size_t... DIMS>                              \
  MOCHI_FORCE_INLINE constexpr NdArray<T, D0, DIMS...> operator OP(             \
      T lhs, NdArray<T, D0, DIMS...> const& rhs) {                              \
    NdArray<T, D0, DIMS...> result{};                                           \
    MOCHI_DETAILS_UNROLL_D0(result[i] = lhs OP rhs[i]);                         \
    return result;                                                              \
  }

MOCHI_DETAILS_NDARRAY_MEMBERWISE_OP(+=, +);
MOCHI_DETAILS_NDARRAY_MEMBERWISE_OP(-=, -);
MOCHI_DETAILS_NDARRAY_MEMBERWISE_OP(*=, *);
MOCHI_DETAILS_NDARRAY_MEMBERWISE_OP(/=, /);

#undef MOCHI_DETAILS_NDARRAY_MEMBERWISE_OP
#undef MOCHI_DETAILS_UNROLL_D0

// Support for associative containers of small int arrays:
struct Int2Hash {
  MOCHI_FORCE_INLINE size_t operator()(Int2 const& x) const {
    return std::hash<uint64_t>()(static_cast<uint64_t>(x[0]) | (static_cast<uint64_t>(x[1]) << 32));
  }
};
struct Int2Less {
  MOCHI_FORCE_INLINE bool operator()(Int2 const& a, Int2 const& b) const {
    return (a[0] < b[0]) || (a[0] == b[0] && a[1] < b[1]);
  };
};
struct Int3SortAndHash {
  MOCHI_FORCE_INLINE size_t operator()(Int3 const& xUnsorted) const {
    Int3 xSorted = xUnsorted;
    std::sort(xSorted.begin(), xSorted.end());
    return Int2Hash{}({xSorted[0], xSorted[1]}) ^ std::hash<uint64_t>()(xSorted[2]);
  }
};
struct Int3SortAndCompare {
  MOCHI_FORCE_INLINE bool operator()(Int3 const& aUnsorted, Int3 const& bUnsorted) const {
    Int3 aSorted = aUnsorted;
    std::sort(aSorted.begin(), aSorted.end());
    Int3 bSorted = bUnsorted;
    std::sort(bSorted.begin(), bSorted.end());
    return aSorted == bSorted;
  }
};

/************************************************************************************
  By declaring explicit instantiations for common types, we can reduce code bloat
  especially in debug builds where the template functions are not inlined.
*/
#if MOCHI_USE_EXTERN_TEMPLATE
extern template class NdArray<int, 2>;
extern template class NdArray<int, 3>;
extern template class NdArray<int, 4>;
extern template class NdArray<real, 2>;
extern template class NdArray<real, 3>;
extern template class NdArray<real, 4>;
extern template class NdArray<real, 2, 2>;
extern template class NdArray<real, 2, 3>;
extern template class NdArray<real, 3, 2>;
extern template class NdArray<real, 3, 3>;
extern template class NdArray<real, 4, 3>;
#endif // MOCHI_USE_EXTERN_TEMPLATE

// ScalarType specialization: recursively unwrap element type.
namespace details {
template <class T, size_t D0, size_t... DIMS>
struct ScalarTypeDef<NdArray<T, D0, DIMS...>, void> {
  using type = ScalarType<T>;
};
} // namespace details

} // namespace mochi

/************************************************************************************
  Reflection support for NdArray using SReflect::ArrayTypeInfo
    NdArray<T, D0> will be treated as a fixed-size array of type T, similar to std::array<T, D0>.
    NdArray<T, D0, D1> will be treated an array of arrays like std::array<std::array<T, D1>, D0>.
    etc...
*/
#if MOCHI_USE_REFLECTION
template <typename T, size_t D0, size_t... DIMS>
struct SReflectTypeTraits<mochi::NdArray<T, D0, DIMS...>> {
  static constexpr SReflect::CoreType coreType = SReflect::CoreType::CT_array;
  static SReflect::ArrayTypeInfo const& GetTypeInfo() {
    static auto const* s_typeInfo = []() {
      using MyType = mochi::NdArray<T, D0, DIMS...>;
      using InnerType = typename MyType::value_type;
      // Format the name like, "NdArray<int, 2>", "NdArray<real, 3, 3>", etc...
      char dimsStr[128];
      for (size_t i = 0, offset = 0; i < MyType::num_dims; ++i) {
        offset += snprintf(dimsStr + offset, sizeof(dimsStr) - offset, ",%zu", MyType::dims[i]);
      }
      char const* tName = SReflect::GetTypeInfo<T>()._nameWithNamespace;
      char const* myName = SReflect::detail::MakeTypeName("mochi::NdArray<", tName, dimsStr, ">");
      static constexpr bool kFormatAsTemplate = false; // No. Use our formatted name.
      return SReflect::MakeFixedArrayTypeInfo<MyType, InnerType, D0>(myName, kFormatAsTemplate);
    }();
    return *s_typeInfo;
  }
};
#endif // MOCHI_USE_REFLECTION
