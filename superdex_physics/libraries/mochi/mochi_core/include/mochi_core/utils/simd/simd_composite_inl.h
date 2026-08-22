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

#include "../simd.h" // for IntelliSense

#include <type_traits>
#include <utility>

namespace mochi {

/**
  Simd partial specialization for larger values of N.
  Supports any N as long as it can be composed of smaller supported Simd objects.
*/
template <class T, int N>
class Simd<
    T,
    N,
    std::enable_if_t<(N > kSimdDefaultSize<T> && Simd<T>::kIsSupported), SimdConcept>> {
 public:
  using Scalar = T;

  // Data storage is split into two parts
  using NativeType = Simd<T, kSimdDefaultSize<T>>; // Native size
  using First = NativeType; // First part
  using Second = Simd<T, N - First::kSize>; // What's left
  First first;
  Second second;

  static constexpr int kSize = N;
  static constexpr int kSizeFirst = First::kSize;
  static constexpr int kSizeSecond = Second::kSize;
  static constexpr bool kIsSupported = First::kIsSupported && Second::kIsSupported;
  static constexpr bool kIsComposite = true;
  static constexpr bool kIsEmulated = First::kIsEmulated || Second::kIsEmulated;
  static_assert(
      kSizeFirst == 2 || kSizeFirst == 4 || kSizeFirst == 8,
      "Some functions may need to be updated to support other SIMD sizes");
  static_assert(
      !kIsSupported || (First::kIsEmulated == Second::kIsEmulated),
      "Inconsistent SIMD emulation flags");

  // Default construct (may be uninitialized)
  MOCHI_ANY MOCHI_FORCE_INLINE Simd() = default;

  // Copy construct
  MOCHI_ANY MOCHI_FORCE_INLINE Simd(Simd const& a) = default;

  // Construct from Simd parts
  MOCHI_ANY MOCHI_FORCE_INLINE Simd(First const& p0, Second const& p1) : first(p0), second(p1) {}

  // Construct from composite halves (currently just Simd<T, 8> from two Simd<T, 4> composites)
  template <class Half, MOCHI_CONCEPT(Half::kIsComposite && (Half::kSize * 2 == kSize))>
  MOCHI_ANY MOCHI_FORCE_INLINE Simd(Half const& a, Half const& b) {
    static_assert(std::is_same_v<Scalar, typename Half::Scalar>, "Type mismatch");
    static_assert(kSizeFirst == 2 && Half::kSize == 4 && kSize == 8, "Unsupported size");
    first = a.first;
    second.first = a.second;
    second.second.first = b.first;
    second.second.second = b.second;
  }

  // Construct by broadcasting a scalar
  template <class U, MOCHI_REQUIRES_NON_BOOL_SCALAR(U, Scalar)>
  MOCHI_ANY MOCHI_FORCE_INLINE Simd(U val) : first(val), second(val) {}

  // Construct from 2 scalars. Any unspecified values are zero.
  MOCHI_ANY MOCHI_FORCE_INLINE Simd(Scalar a, Scalar b) : first(a, b), second() {}

  // Construct from 3 or 4 scalars. Any unspecified values are zero.
  MOCHI_ANY MOCHI_FORCE_INLINE Simd(Scalar a, Scalar b, Scalar c, Scalar d = Scalar(0)) {
    if constexpr (kSizeFirst == 2) {
      first = First{a, b};
      second = Second{c, d};
    } else {
      first = First{a, b, c, d};
      second = Second{};
    }
  }

  // Construct from 5 or 6 scalars. Any unspecified values are zero.
  MOCHI_ANY MOCHI_FORCE_INLINE
  Simd(Scalar a, Scalar b, Scalar c, Scalar d, Scalar e, Scalar f = Scalar(0)) {
    if constexpr (kSizeFirst == 2) {
      first = First{a, b};
      second = Second{c, d, e, f};
    } else if constexpr (kSizeFirst == 4) {
      first = First{a, b, c, d};
      second = Second{e, f};
    } else {
      first = First{a, b, c, d, e, f};
      second = {};
    }
  }

  // Construct from 7 scalars. Any unspecified values are zero.
  MOCHI_ANY MOCHI_FORCE_INLINE
  Simd(Scalar a, Scalar b, Scalar c, Scalar d, Scalar e, Scalar f, Scalar g) {
    if constexpr (kSizeFirst == 2) {
      first = First{a, b};
      second = Second{c, d, e, f, g};
    } else if constexpr (kSizeFirst == 4) {
      first = First{a, b, c, d};
      second = Second{e, f, g};
    } else {
      first = First{a, b, c, d, e, f, g};
      second = {};
    }
  }

  // Construct from 8 or more scalars. Any unspecified values are zero.
  template <typename... MoreScalars>
  MOCHI_ANY MOCHI_FORCE_INLINE Simd(
      Scalar a,
      Scalar b,
      Scalar c,
      Scalar d,
      Scalar e,
      Scalar f,
      Scalar g,
      Scalar h,
      MoreScalars... args) {
    if constexpr (kSizeFirst == 2) {
      first = First{a, b};
      second = Second{c, d, e, f, g, h, args...};
    } else if constexpr (kSizeFirst == 4) {
      first = First{a, b, c, d};
      second = Second{e, f, g, h, args...};
    } else {
      first = First{a, b, c, d, e, f, g, h};
      static constexpr auto kNumArgs = sizeof...(args);
      if constexpr (kNumArgs == 0) {
        second = Second{};
      } else if constexpr (kNumArgs == 1) {
        second = Second{args..., Scalar(0)};
      } else {
        second = Second{args...};
      }
    }
  }

  MOCHI_ANY MOCHI_FORCE_INLINE static constexpr size_t size() {
    return kSize;
  }

  template <int i>
  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Scalar Get(Simd a) {
    if constexpr (i < kSizeFirst) {
      return First::template Get<i>(a.first);
    } else {
      return Second::template Get<i - kSizeFirst>(a.second);
    }
  }

  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Scalar Get(Simd a, int i) {
    if (i < kSizeFirst) {
      return First::Get(a.first, i);
    } else {
      return Second::Get(a.second, i - kSizeFirst);
    }
  }

  template <int i>
  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE auto GetHalf(Simd a) {
    using Half = Simd<Scalar, N / 2>;
    static_assert(Half::kIsSupported);
    if constexpr (kSizeFirst == kSizeSecond) {
      if constexpr (i == 0) {
        return a.first;
      } else {
        return a.second;
      }
    } else {
      static_assert(
          Half::kIsComposite && kSize == 8 && kSizeFirst == 2, "Not yet supported for other sizes");
      if constexpr (i == 0) {
        return Half{a.first, a.second.first};
      } else {
        return Half{a.second.second.first, a.second.second.second};
      }
    }
  }

  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Simd Set(Simd a, int i, Scalar value) {
    if (i < kSizeFirst) {
      return {First::Set(a.first, i, value), a.second};
    } else {
      return {a.first, Second::Set(a.second, i - kSizeFirst, value)};
    }
  }

  template <int i>
  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Simd Set(Simd a, Scalar value) {
    if constexpr (i < kSizeFirst) {
      return {First::template Set<i>(a.first, value), a.second};
    } else {
      return {a.first, Second::template Set<i - kSizeFirst>(a.second, value)};
    }
  }

  template <int i>
  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Simd SetBasisVector() {
    static_assert(kSize == 4, "Unsupported size");
    return Set<i>(Zero(), Scalar(1));
  }

  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Simd Sequence() {
    alignas(First)
        Scalar constexpr kSequence[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    static_assert(
        std::size(kSequence) >= First::kSize,
        "Vector size is too large. Size of kSequence must be increased.");
    auto first = First::Load(kSequence);
    Second second MOCHI_NO_INIT;
    if constexpr (std::is_same_v<First, Second>) {
      second = first;
    } else if constexpr (Second::kIsComposite) {
      second = Second::Sequence();
    } else {
      static_assert(
          std::size(kSequence) >= Second::kSize,
          "Vector size is too large. Size of kSequence must be increased.");
      second = Second::Load(kSequence);
    }
    return Simd{first, second + Second{static_cast<T>(First::kSize)}};
  }

  template <int x = 0, int y = 1, int z = 2, int w = 3>
  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Simd Shuffle(Simd a, Simd b) {
    static_assert(kSize == 4, "Unsupported size");
    return {Get<x>(a), Get<y>(a), Get<z>(b), Get<w>(b)}; // Not optimized
  }

  template <int x = 0, int y = 1, int z = 2, int w = 3>
  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Simd Shuffle(Simd a) {
    static_assert(kSize == 4, "Unsupported size");
    return Shuffle<x, y, z, w>(a, a);
  }

  template <int SZ = kSize>
  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE bool AllTrue(Simd a) {
    static_assert(SZ <= kSize);
    if constexpr (SZ <= kSizeFirst) {
      return First::template AllTrue<SZ>(a.first);
    } else {
      return First::template AllTrue<kSizeFirst>(a.first) &&
          Second::template AllTrue<SZ - kSizeFirst>(a.second);
    }
  }

  template <int SZ = kSize>
  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE bool AnyTrue(Simd a) {
    if constexpr (SZ <= kSizeFirst) {
      return First::template AnyTrue<SZ>(a.first);
    } else {
      return First::template AnyTrue<kSizeFirst>(a.first) ||
          Second::template AnyTrue<SZ - kSizeFirst>(a.second);
    }
  }

  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Simd AsDirection(Simd a) {
    static_assert(kSize == 4, "Unsupported size");
    return {a.first, Second::template Set<1>(a.second, Scalar(0))};
  }

  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Simd AsPoint(Simd a) {
    static_assert(kSize == 4, "Unsupported size");
    return {a.first, Second::template Set<1>(a.second, Scalar(1))};
  }

  template <int x, int y, int z, int w>
  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Simd Blend(Simd a, Simd b) {
    static_assert(kSize == 4, "Unsupported size");
    return {
        First::template Blend<x, y>(a.first, b.first),
        Second::template Blend<z, w>(a.second, b.second)};
  }

  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Simd Broadcast(Scalar const* p) {
    return {First::Broadcast(p), Second::Broadcast(p)};
  }

  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Simd Broadcast(Scalar a) {
    return {First{a}, Second{a}};
  }

  template <int i>
  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Simd Broadcast(Simd a) {
    if constexpr (kSizeFirst == kSizeSecond) {
      if constexpr (i < kSizeFirst) {
        auto x = First::template Broadcast<i>(a.first);
        return {x, x};
      } else {
        auto x = Second::template Broadcast<i - kSizeFirst>(a.second);
        return {x, x};
      }
    } else {
      Scalar s = Get<i>(a); // Not optimized
      return {First{s}, Second{s}};
    }
  }

  template <int SZ = kSize>
  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Simd Load(T const* ptr) {
    if constexpr (SZ <= kSizeFirst) {
      return {First::template Load<SZ>(ptr), Second{}};
    } else {
      return {
          First::template Load<kSizeFirst>(ptr),
          Second::template Load<SZ - kSizeFirst>(ptr + kSizeFirst)};
    }
  }

  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Simd Load(Scalar const* ptr, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    if (n <= kSizeFirst) {
      return {First::Load(ptr, n), Second{}};
    } else {
      return {First::Load(ptr), Second::Load(ptr + kSizeFirst, n - kSizeFirst)};
    }
  }

  template <typename IntT, MOCHI_CONCEPT(std::is_integral_v<IntT>)>
  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Simd
  LoadIndexed(Scalar const* ptr, Simd<IntT, kSize> const& indices) {
    using IMatch = std::conditional_t<sizeof(Scalar) == 4, int, int64_t>; // int same size as Scalar
    using IVec = Simd<IMatch, kSize>;
    static_assert(IVec::kIsComposite && sizeof(IVec) == sizeof(Simd));
    auto matchingIndices = StaticCast<IVec>(indices); // maybe no change
    return {
        First::LoadIndexed(ptr, matchingIndices.first),
        Second::LoadIndexed(ptr, matchingIndices.second)};
  }

  template <int kTupleCount = kSize, class... OutputVectors>
  static MOCHI_ANY MOCHI_FORCE_INLINE void LoadTransposed(
      Scalar const* ptr,
      OutputVectors&... out) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Invalid kTupleCount");
    if constexpr (kTupleCount <= kSizeFirst) {
      First::template LoadTransposed<kTupleCount>(ptr, out.first...);
      ((out.second = {}), ...); // Zero-fill the second part
    } else {
      First::template LoadTransposed<kSizeFirst>(ptr, out.first...);
      Second::template LoadTransposed<kTupleCount - kSizeFirst>(
          ptr + sizeof...(OutputVectors) * kSizeFirst, out.second...);
    }
  }

  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Simd Select(Simd mask, Simd a, Simd b) {
    return {
        First::Select(mask.first, a.first, b.first),
        Second::Select(mask.second, a.second, b.second)};
  }

  template <int kShift>
  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Simd ShiftRight(Simd a) {
    static_assert(kShift >= 0 && kShift < (8 * sizeof(T)), "Shift amount out-of-range");
    if constexpr (kShift == 0) {
      return a;
    } else {
      return {
          First::template ShiftRight<kShift>(a.first),
          Second::template ShiftRight<kShift>(a.second)};
    }
  }

  template <int SZ = kSize>
  static MOCHI_ANY MOCHI_FORCE_INLINE void Store(Scalar* ptr, Simd a) {
    if constexpr (SZ <= kSizeFirst) {
      First::template Store<SZ>(ptr, a.first);
    } else {
      First::template Store<kSizeFirst>(ptr, a.first);
      Second::template Store<SZ - kSizeFirst>(ptr + kSizeFirst, a.second);
    }
  }

  static MOCHI_ANY MOCHI_FORCE_INLINE void Store(Scalar* ptr, Simd v, int n) {
    // Hopefully the compiler can figure out the best thing to do here.
    // The alternative is to have a runtime branch at each stage.
    memcpy(ptr, &v, sizeof(Scalar) * n);
  }

  static MOCHI_ANY MOCHI_FORCE_INLINE int StoreSelected(Scalar* ptr, Simd condition, Simd values) {
    int count = First::StoreSelected(ptr, condition.first, values.first);
    count += Second::StoreSelected(ptr + count, condition.second, values.second);
    return count;
  }

  template <int kTupleCount = kSize, class... InputVectors>
  static MOCHI_ANY MOCHI_FORCE_INLINE void StoreTransposed(Scalar* ptr, InputVectors... v) {
    First::template StoreTransposed<mochi::Min(kTupleCount, First::kSize)>(ptr, v.first...);
    if constexpr (kTupleCount > First::kSize) {
      Second::template StoreTransposed<kTupleCount - First::kSize>(
          ptr + sizeof...(InputVectors) * kSizeFirst, v.second...);
    }
  }

  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Simd SignBitMask() {
    return {First::SignBitMask(), Second::SignBitMask()};
  }

  template <int SZ = kSize>
  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Scalar HMin(Simd a) {
    if constexpr (SZ <= kSizeFirst) {
      return First::template HMin<SZ>(a.first);
    } else if constexpr (SZ - kSizeFirst == 1) {
      // HMin<1> is not normally supported, so use a scalar
      return mochi::Min(
          First::template HMin<kSizeFirst>(a.first), Second::template Get<0>(a.second));
    } else if constexpr (kSizeFirst == kSizeSecond && SZ == kSize) {
      // Special case for 2x native size. One horizontal operation.
      return First::template HMin<kSizeFirst>(First::Min(a.first, a.second));
    } else if constexpr (kSizeFirst * 2 == kSizeSecond && SZ == kSize) {
      // Special case for 3x native size. One horizontal operation.
      return First::template HMin<kSizeFirst>(
          First::Min(a.first, First::Min(a.second.first, a.second.second)));
    } else if constexpr (kSizeFirst * 3 == kSizeSecond && SZ == kSize) {
      // Special case for 4x native size. One horizontal operation.
      return First::template HMin<kSizeFirst>(First::Min(
          a.first,
          First::Min(a.second.first, First::Min(a.second.second.first, a.second.second.second))));
    } else {
      // May not be as fast due to repeated horizontal operations
      return mochi::Min(
          First::template HMin<kSizeFirst>(a.first),
          Second::template HMin<SZ - kSizeFirst>(a.second));
    }
  }

  template <int SZ = kSize>
  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Scalar HMax(Simd a) {
    if constexpr (SZ <= kSizeFirst) {
      return First::template HMax<SZ>(a.first);
    } else if constexpr (SZ - kSizeFirst == 1) {
      // HMax<1> is not normally supported, so use a scalar
      return mochi::Max(
          First::template HMax<kSizeFirst>(a.first), Second::template Get<0>(a.second));
    } else if constexpr (kSizeFirst == kSizeSecond && SZ == kSize) {
      // Special case for 2x native size. One horizontal operation.
      return First::template HMax<kSizeFirst>(First::Max(a.first, a.second));
    } else if constexpr (kSizeFirst * 2 == kSizeSecond && SZ == kSize) {
      // Special case for 3x native size. One horizontal operation.
      return First::template HMax<kSizeFirst>(
          First::Max(a.first, First::Max(a.second.first, a.second.second)));
    } else if constexpr (kSizeFirst * 3 == kSizeSecond && SZ == kSize) {
      // Special case for 4x native size. One horizontal operation.
      return First::template HMax<kSizeFirst>(First::Max(
          a.first,
          First::Max(a.second.first, First::Max(a.second.second.first, a.second.second.second))));
    } else {
      // May not be as fast due to repeated horizontal operations
      return mochi::Max(
          First::template HMax<kSizeFirst>(a.first),
          Second::template HMax<SZ - kSizeFirst>(a.second));
    }
  }

  template <int SZ = kSize>
  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Scalar HSum(Simd a) {
    if constexpr (SZ <= kSizeFirst) {
      return First::template HSum<SZ>(a.first);
    } else if constexpr (SZ - kSizeFirst == 1) {
      // HSum<1> is not normally supported, so add a scalar
      return First::template HSum<kSizeFirst>(a.first) + Second::template Get<0>(a.second);
    } else if constexpr (kSizeFirst == kSizeSecond && SZ == kSize) {
      // Special case for 2x native size. One horizontal operation.
      return First::template HSum<kSizeFirst>(a.first + a.second);
    } else if constexpr (kSizeFirst * 2 == kSizeSecond && SZ == kSize) {
      // Special case for 3x native size. One horizontal operation.
      return First::template HSum<kSizeFirst>(a.first + a.second.first + a.second.second);
    } else if constexpr (kSizeFirst * 3 == kSizeSecond && SZ == kSize) {
      // Special case for 4x native size. One horizontal operation.
      return First::template HSum<kSizeFirst>(
          a.first + a.second.first + a.second.second.first + a.second.second.second);
    } else {
      // May not be as fast due to repeated horizontal operations
      return First::template HSum<kSizeFirst>(a.first) +
          Second::template HSum<SZ - kSizeFirst>(a.second);
    }
  }

  template <int SZ = kSize>
  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Scalar HProd(Simd a) {
    if constexpr (SZ <= kSizeFirst) {
      return First::template HProd<SZ>(a.first);
    } else if constexpr (SZ - kSizeFirst == 1) {
      // HProd<1> is not normally supported, so multiply a scalar
      return First::template HProd<kSizeFirst>(a.first) * Second::template Get<0>(a.second);
    } else if constexpr (kSizeFirst == kSizeSecond && SZ == kSize) {
      return First::template HProd<kSizeFirst>(a.first * a.second);
    } else {
      // May not be as fast due to repeated horizontal operations
      return First::template HProd<kSizeFirst>(a.first) *
          Second::template HProd<SZ - kSizeFirst>(a.second);
    }
  }

  template <int SZ = kSize>
  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Simd Dot(Simd a, Simd b) {
    return Broadcast(HSum<SZ>(a * b)); // Use HSum to hopefully avoid repeated horizontal operations
  }

  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Simd Zero() {
    return {First::Zero(), Second::Zero()};
  }

  MOCHI_ANY MOCHI_FORCE_INLINE Simd& operator=(Simd const& rhs) = default;

  template <class U, MOCHI_REQUIRES_NON_BOOL_SCALAR(U, Scalar)>
  MOCHI_ANY MOCHI_FORCE_INLINE Simd& operator=(U rhs) {
    first = rhs;
    second = rhs;
    return *this;
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE bool operator==(Simd rhs) const {
    if constexpr (kSizeFirst == kSizeSecond) {
      // Special case for 2x native size. Just one AllTrue.
      return First::template AllTrue<kSizeFirst>(
          First::Equal(this->first, rhs.first) & Second::Equal(this->second, rhs.second));
    } else {
      return (this->first == rhs.first) && (this->second == rhs.second);
    }
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE bool operator!=(Simd rhs) const {
    return !(*this == rhs);
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE Scalar operator[](int i) const {
    return Get(*this, i); /* return by value */
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE Simd operator<<(int i) const {
    return {this->first << i, this->second << i};
  }

  MOCHI_ANY MOCHI_FORCE_INLINE Simd& operator<<=(int i) {
    this->first <<= i;
    this->second <<= i;
    return *this;
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE Simd operator>>(int i) const {
    return {this->first >> i, this->second >> i};
  }

  MOCHI_ANY MOCHI_FORCE_INLINE Simd& operator>>=(int i) {
    this->first >>= i;
    this->second >>= i;
    return *this;
  }

#define MOCHI_SIMD_COMPOSITE_FN_1(FnName)                                 \
  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Simd FnName(Simd a) { \
    return {First::FnName(a.first), Second::FnName(a.second)};            \
  }

#define MOCHI_SIMD_COMPOSITE_FN_2(FnName)                                         \
  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Simd FnName(Simd a, Simd b) { \
    return {First::FnName(a.first, b.first), Second::FnName(a.second, b.second)}; \
  }

#define MOCHI_SIMD_COMPOSITE_FN_3(FnName)                                                        \
  [[nodiscard]] static MOCHI_ANY MOCHI_FORCE_INLINE Simd FnName(Simd a, Simd b, Simd c) {        \
    return {                                                                                     \
        First::FnName(a.first, b.first, c.first), Second::FnName(a.second, b.second, c.second)}; \
  }

#define MOCHI_SIMD_COMPOSITE_OP_1(OP)                                   \
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE Simd operator OP() const { \
    return {OP this->first, OP this->second};                           \
  }

#define MOCHI_SIMD_COMPOSITE_OP_2(OP)                                           \
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE Simd operator OP(Simd rhs) const { \
    return {this->first OP rhs.first, this->second OP rhs.second};              \
  }

#define MOCHI_SIMD_COMPOSITE_OP_EQ(OP_EQ, OP)                 \
  MOCHI_ANY MOCHI_FORCE_INLINE Simd& operator OP_EQ(Simd a) { \
    this->first = this->first OP a.first;                     \
    this->second = this->second OP a.second;                  \
    return *this;                                             \
  }

#define MOCHI_SIMD_COMPOSITE_OP_EQ_WITH_SCALAR(OP_EQ, OP)       \
  MOCHI_SIMD_COMPOSITE_OP_EQ(OP_EQ, OP);                        \
  MOCHI_ANY MOCHI_FORCE_INLINE Simd& operator OP_EQ(Scalar a) { \
    this->first = this->first OP First{a};                      \
    this->second = this->second OP Second{a};                   \
    return *this;                                               \
  }

  // Unary functions:
  MOCHI_SIMD_COMPOSITE_FN_1(Abs);
  MOCHI_SIMD_COMPOSITE_FN_1(ACos);
  MOCHI_SIMD_COMPOSITE_FN_1(ASin);
  MOCHI_SIMD_COMPOSITE_FN_1(ATan);
  MOCHI_SIMD_COMPOSITE_FN_1(Exp);
  MOCHI_SIMD_COMPOSITE_FN_1(Floor);
  MOCHI_SIMD_COMPOSITE_FN_1(FastRound);
  MOCHI_SIMD_COMPOSITE_FN_1(Cos);
  MOCHI_SIMD_COMPOSITE_FN_1(Ln);
  MOCHI_SIMD_COMPOSITE_FN_1(Sin);
  MOCHI_SIMD_COMPOSITE_FN_1(RcpApprox);
  MOCHI_SIMD_COMPOSITE_FN_1(RcpSqrtApprox);
  MOCHI_SIMD_COMPOSITE_FN_1(Sqrt);
  MOCHI_SIMD_COMPOSITE_FN_1(Tan);
  MOCHI_SIMD_COMPOSITE_FN_1(Tanh);

  // Binary functions
  MOCHI_SIMD_COMPOSITE_FN_2(Min);
  MOCHI_SIMD_COMPOSITE_FN_2(Max);
  MOCHI_SIMD_COMPOSITE_FN_2(Equal);
  MOCHI_SIMD_COMPOSITE_FN_2(NotEqual);

  // Ternary functions
  MOCHI_SIMD_COMPOSITE_FN_3(MulAdd);
  MOCHI_SIMD_COMPOSITE_FN_3(MulSub);
  MOCHI_SIMD_COMPOSITE_FN_3(NegMulAdd);
  MOCHI_SIMD_COMPOSITE_FN_3(NegMulSub);

  // Unary operators
  MOCHI_SIMD_COMPOSITE_OP_1(~);
  MOCHI_SIMD_COMPOSITE_OP_1(-);

  // Binary operators
  MOCHI_SIMD_COMPOSITE_OP_2(<);
  MOCHI_SIMD_COMPOSITE_OP_2(>);
  MOCHI_SIMD_COMPOSITE_OP_2(<=);
  MOCHI_SIMD_COMPOSITE_OP_2(>=);
  MOCHI_SIMD_COMPOSITE_OP_2(+);
  MOCHI_SIMD_COMPOSITE_OP_2(-);
  MOCHI_SIMD_COMPOSITE_OP_2(*);
  MOCHI_SIMD_COMPOSITE_OP_2(/);
  MOCHI_SIMD_COMPOSITE_OP_2(&);
  MOCHI_SIMD_COMPOSITE_OP_2(|);
  MOCHI_SIMD_COMPOSITE_OP_2(^);

  // Math assignment operators
  //
  // VS2019 WORKAROUND: These operators should not be necessary because there are generic templates
  //    in simd_inl.h which convert operations like `a += b;` into `a = a + b;`. However, there is
  //    a bug in the VS2019 compiler resulting in incorrect runtime behavior in some of these cases.
  //    The operators here take precedence and shortcut a couple layers of template abstraction,
  //    which appears to be a sufficient work-around. The bug was only observed with VS2019
  //    optimized builds, not with VS2022, Clang, nor GCC.
  MOCHI_SIMD_COMPOSITE_OP_EQ_WITH_SCALAR(+=, +);
  MOCHI_SIMD_COMPOSITE_OP_EQ_WITH_SCALAR(-=, -);
  MOCHI_SIMD_COMPOSITE_OP_EQ_WITH_SCALAR(*=, *);
  MOCHI_SIMD_COMPOSITE_OP_EQ_WITH_SCALAR(/=, /);
  MOCHI_SIMD_COMPOSITE_OP_EQ(&=, &);
  MOCHI_SIMD_COMPOSITE_OP_EQ(|=, |);
  MOCHI_SIMD_COMPOSITE_OP_EQ(^=, ^);

#undef MOCHI_SIMD_COMPOSITE_FN_1
#undef MOCHI_SIMD_COMPOSITE_FN_2
#undef MOCHI_SIMD_COMPOSITE_FN_3
#undef MOCHI_SIMD_COMPOSITE_OP_1
#undef MOCHI_SIMD_COMPOSITE_OP_2
#undef MOCHI_SIMD_COMPOSITE_OP_EQ
#undef MOCHI_SIMD_COMPOSITE_OP_EQ_WITH_SCALAR
};

// ReinterpretCast for composite Simd types
template <
    class To,
    class FromT,
    int FromN,
    MOCHI_CONCEPT((Simd<FromT, FromN>::kIsComposite) && To::kIsComposite)>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE To ReinterpretCast(Simd<FromT, FromN> const& a) {
  return {
      ReinterpretCast<typename To::First>(a.first), ReinterpretCast<typename To::Second>(a.second)};
}

namespace details {
template <typename T>
[[nodiscard]] constexpr bool IsCompositeWithEqualHalves() {
  if constexpr (T::kIsComposite) {
    return T::kSizeFirst == T::kSizeSecond;
  } else {
    return false;
  }
}

template <class To, class From, std::size_t... Is>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE To
StaticCastElementwise(From const& a, std::index_sequence<Is...>) {
  return To{static_cast<typename To::Scalar>(a[Is])...};
}
} // namespace details

// StaticCast when at least one of the types is a Simd composite.
// Also handles the pass-through case where To and From are the same.
template <class To, class From, MOCHI_CONCEPT_DEF(IsSimd<To>&& IsSimd<From>)>
[[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE To StaticCast(From const& a) {
  static_assert(To::kSize == From::kSize, "Size mismatch");
  static_assert(To::kIsSupported, "Unsupported type");
  using ToHalf = Simd<typename To::Scalar, To::kSize / 2>;
  using FromHalf = Simd<typename From::Scalar, From::kSize / 2>;
  if constexpr (std::is_same_v<typename To::Scalar, typename From::Scalar>) {
    return a; // No change
  } else if constexpr (
      To::kIsComposite && From::kIsComposite &&
      (sizeof(typename To::Scalar) == sizeof(typename From::Scalar))) {
    // Casting a composite to another composite, where the native sizes match
    // Examples: Vec16f <--> Vec16i
    return {StaticCast<typename To::First>(a.first), StaticCast<typename To::Second>(a.second)};
  } else if constexpr (
      From::kIsComposite && !To::kIsComposite && ToHalf::kIsSupported &&
      (sizeof(typename To::Scalar) * 2 == sizeof(typename From::Scalar))) {
    // Casting from a composite to a native vector, where the destination scalar type is smaller
    // Examples: Vec8i <-- Vec8d, Vec8f <-- Vec8d (using AVX)
    return {StaticCast<ToHalf>(a.first), StaticCast<ToHalf>(a.second)};
  } else if constexpr (
      To::kIsComposite && !From::kIsComposite && FromHalf::kIsSupported &&
      (sizeof(typename To::Scalar) == 2 * sizeof(typename From::Scalar))) {
    // Casting from a native size to a composite, where the destination scalar type is larger
    // Examples: Vec8d <-- Vec8i, Vec8d <-- Vec8f (using AVX)
    return {
        StaticCast<typename To::First>(From::template GetHalf<0>(a)),
        StaticCast<typename To::Second>(From::template GetHalf<1>(a))}; // Not optimized
  } else if constexpr (
      ::mochi::details::IsCompositeWithEqualHalves<From>() &&
      ::mochi::details::IsCompositeWithEqualHalves<To>() &&
      sizeof(typename To::Scalar) != sizeof(typename From::Scalar)) {
    // Both sides are composite with equal-sized halves (i.e. GetHalf<0/1> can split them
    // symmetrically), but their tree shapes may differ due to different native SIMD widths
    // (e.g. Half↔float or double↔float on ARM), so we can't recurse via .first/.second directly.
    return To{
        StaticCast<ToHalf>(From::template GetHalf<0>(a)),
        StaticCast<ToHalf>(From::template GetHalf<1>(a))};
  } else {
    return mochi::details::StaticCastElementwise<To>(a, std::make_index_sequence<From::kSize>{});
  }
}

} // namespace mochi
