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

#include "../simd.h" // for Intelisense

#if !MOCHI_USE_SIMD
#include <mochi_core/utils/array.h>
#include <mochi_core/utils/basic_utils.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <utility>

namespace mochi {

template <typename F, typename... Args>
concept IsInvokableWithRaw = requires(F&& f, Args&&... args) { f(args.raw[0]...); };

// NOTE: This specialization could support any size N, but it is currently restricted to multiples
// of the default size to better match the behavior of native SIMD implementations.
//
// NOTE: Half (16-bit float) is not currently supported for emulation. We could add this
// functionality in the future for compilers with native support (e.g. using __half for CUDA).
//
// PERFORMANCE: The artificial size restriction sometimes causes users to round up to the next
// multiple of the default size, resulting in wasted loads, stores, and ALU operations. If
// performance of SIMD emulation (e.g. GPU kernel) is important, then consider removing the
// restriction.
//
template <typename T, int N>
  requires(IsSimdSupportedType<T> && !IsHalf<T> && (N % kSimdDefaultSize<T> == 0))
class Simd<T, N> {
  using UIntT = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
  static constexpr T kOnesMask = std::bit_cast<T>(~UIntT{0});

#define MOCHI_SIMD_EMULATOR_OP_1(Op, Expr)                                                     \
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr Simd operator Op(Simd rhs) const {      \
    auto eval = []<size_t... I>(Simd const& lhs, Simd const& rhs, std::index_sequence<I...>) { \
      return Simd{(Expr)...};                                                                  \
    };                                                                                         \
    return eval(*this, rhs, std::make_index_sequence<N>{});                                    \
  }

#define MOCHI_SIMD_EMULATOR_OP_2(Op)                                                           \
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr Simd operator Op(Simd rhs) const {      \
    auto eval = []<size_t... I>(Simd const& lhs, Simd const& rhs, std::index_sequence<I...>) { \
      return Simd(                                                                             \
          std::bit_cast<Scalar>(std::bit_cast<UIntT>(lhs.raw[I])                               \
                                    Op std::bit_cast<UIntT>(rhs.raw[I]))...);                  \
    };                                                                                         \
    return eval(*this, rhs, std::make_index_sequence<N>{});                                    \
  }

#define MOCHI_SIMD_EMULATOR_HREDUCE_BOOL(Name, Op)                                \
  template <int M = kSize>                                                        \
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static constexpr bool Name(Simd a) { \
    static_assert(M >= 1 && M <= kSize, "Unsupported M");                         \
    auto eval = []<size_t... I>(Simd const& a, std::index_sequence<I...>) {       \
      return ((std::bit_cast<UIntT>(a.raw[I]) != UIntT{0}) Op...);                \
    };                                                                            \
    return eval(a, std::make_index_sequence<M>{});                                \
  }

  template <int M = N, auto F>
  MOCHI_ANY MOCHI_FORCE_INLINE static T FoldApply(Simd a) {
    auto eval = []<size_t... I>(Simd const& x, std::index_sequence<I...>) {
      return F(x.raw[I]...);
    };
    return eval(a, std::make_index_sequence<M>{});
  }

  using ST = std::conditional_t<std::floating_point<T>, T, float>;
  template <ST (*F)(ST)>
  MOCHI_ANY MOCHI_FORCE_INLINE static Simd Xapply(
      std::conditional_t<std::floating_point<T>, Simd, struct DoesNotExist>& x) {
    if constexpr (std::floating_point<T>) {
      return Simd(F, x);
    } else {
      return Simd{};
    }
  }

  template <typename S, S (*F)(S, S)>
  MOCHI_ANY MOCHI_FORCE_INLINE static Simd XYapply(Simd const& x, Simd const& y) {
    return Simd(F, x, y);
  }

  template <typename S, S (*F)(S, S, S)>
  MOCHI_ANY MOCHI_FORCE_INLINE static Simd XYZapply(Simd const& x, Simd const& y, Simd const& z) {
    return Simd(F, x, y, z);
  }

 public:
  static constexpr int kSize = N;
  static constexpr bool kIsSupported = true;
  static constexpr bool kIsComposite = false;
  static constexpr bool kIsEmulated = true;
  using Scalar = T;
  // Warning: Do not use std::array as it is not supported in CUDA device code.
  using NativeType = Array<Scalar, N>;
  NativeType raw;

  MOCHI_ANY MOCHI_FORCE_INLINE constexpr Simd() = default;
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr Simd(Simd const& rhs) = default;
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr Simd(NativeType const& rhs) : raw(rhs) {}
  template <class U, MOCHI_REQUIRES_NON_BOOL_SCALAR(U, Scalar)>
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr Simd(U a) {
    for (int i = 0; i < N; ++i) {
      raw[i] = a;
    }
  }
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr Simd(Simd<T, N / 2> a, Simd<T, N / 2> b)
    requires(N > 2 && N % 2 == 0)
  {
    for (int i = 0; i < N / 2; ++i) {
      raw[i] = a.raw[i];
    }
    for (int i = 0; i < N / 2; ++i) {
      raw[(N / 2) + i] = b[i];
    }
  }

  template <typename... Args>
    requires(
        sizeof...(Args) > 1 && sizeof...(Args) <= kSize && (std::convertible_to<Args, T> && ...))
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr Simd(Args... args)
      : raw{static_cast<T>(std::forward<Args>(args))...} {}

  template <typename F, typename... Args>
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr Simd(F const& f, Args const&... args)
    requires IsInvokableWithRaw<F, Args...>
  {
    for (size_t i = 0; i < N; ++i) {
      raw[i] = f(args.raw[i]...);
    }
  }

  MOCHI_ANY MOCHI_FORCE_INLINE static constexpr size_t size() {
    return static_cast<size_t>(kSize);
  }

  MOCHI_ANY MOCHI_FORCE_INLINE constexpr Simd& operator=(Simd const& rhs) = default;

  template <class U, MOCHI_REQUIRES_NON_BOOL_SCALAR(U, Scalar)>
  MOCHI_ANY MOCHI_FORCE_INLINE constexpr Simd& operator=(U a) {
    for (int i = 0; i < N; ++i) {
      raw[i] = a;
    }
    return *this;
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static constexpr Simd Zero() {
    return NativeType{};
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE Scalar constexpr operator[](int i) const {
    return raw[i];
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE Scalar constexpr Get(int i) const {
    return raw[i];
  }

  template <int i>
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static constexpr Scalar Get(Simd v) {
    static_assert(i >= 0 && i < kSize, "Index out of range");
    return v.raw[i];
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static constexpr Scalar Get(Simd v, int i) {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range.");
    return v.raw[i];
  }

  template <int i>
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static constexpr Simd<T, N / 2> GetHalf(Simd v)
    requires(N >= 2 && N % 2 == 0)
  {
    static_assert(i == 0 || i == 1);
    auto eval = [&v]<size_t... I>(std::index_sequence<I...>) {
      return Simd<T, N / 2>{v.raw[i * (N / 2) + I]...};
    };
    return eval(std::make_index_sequence<N / 2>{});
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static constexpr Simd
  Set(Simd v, int i, Scalar value) {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range.");
    auto result = v;
    result.raw[i] = value;
    return result;
  }

  template <int i>
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static Simd Set(Simd v, Scalar value) {
    static_assert(i >= 0 && i < kSize, "Index out of range");
    return Set(v, i, value);
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static constexpr Simd AsPoint(Simd a)
    requires(N == 4)
  {
    return {a.raw[0], a.raw[1], a.raw[2], Scalar{1}};
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static constexpr Simd AsDirection(Simd a)
    requires(N == 4)
  {
    return {a.raw[0], a.raw[1], a.raw[2], Scalar{0}};
  }

  template <int i>
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static constexpr Simd SetBasisVector()
    requires(N == 4)
  {
    static_assert(i >= 0 && i <= 3, "Invalid component index");
    return {
        i == 0 ? Scalar{1} : Scalar{0},
        i == 1 ? Scalar{1} : Scalar{0},
        i == 2 ? Scalar{1} : Scalar{0},
        i == 3 ? Scalar{1} : Scalar{0}};
  }

  template <int... x>
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static constexpr Simd Blend(Simd a, Simd b)
    requires(sizeof...(x) == N)
  {
    static_assert(((x == 0 || x == 1) && ...), "Invalid index");
    auto eval = []<size_t... I>(Simd const& a, Simd const& b, std::index_sequence<I...>) {
      return Simd{(x ? b.raw[I] : a.raw[I])...};
    };
    return eval(a, b, std::make_index_sequence<N>{});
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static constexpr Simd Broadcast(Scalar const* p) {
    return Simd{*p};
  }

  template <int i>
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static constexpr Simd Broadcast(Simd v) {
    static_assert(i >= 0 && i < kSize, "Index out of range");
    return Simd{v.raw[i]};
  }

  template <int x = 0, int y = 1>
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static constexpr Simd Shuffle(Simd a)
    requires(N == 2)
  {
    static_assert(x >= 0 && x < 2 && y >= 0 && y < 2, "Invalid index");
    return Simd{a.raw[x], a.raw[y]};
  }

  template <int x = 0, int y = 1, int z = 2, int w = 3>
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static constexpr Simd Shuffle(Simd a, Simd b)
    requires(N == 4)
  {
    static_assert(
        x >= 0 && x < 4 && y >= 0 && y < 4 && z >= 0 && z < 4 && w >= 0 && w < 4, "Invalid index");
    return Simd{a.raw[x], a.raw[y], b.raw[z], b.raw[w]};
  }

  template <int x = 0, int y = 1, int z = 2, int w = 3>
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static constexpr Simd Shuffle(Simd v)
    requires(N == 4)
  {
    return Shuffle<x, y, z, w>(v, v);
  }

  template <int M = kSize>
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static Simd Load([[maybe_unused]] Scalar const* ptr) {
    static_assert(M >= 0 && M <= kSize);
    if constexpr (M == 0) {
      return Zero();
    } else {
      Simd result;
      std::memcpy(result.raw.data(), ptr, M * sizeof(Scalar));
      if constexpr (M < kSize) {
        std::memset(result.raw.data() + M, 0, (kSize - M) * sizeof(Scalar));
      }
      return result;
    }
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static Simd Load(Scalar const* ptr, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid load size.");
    Simd result;
    for (int i = 0; i < n; ++i) {
      result.raw[i] = ptr[i];
    }
    for (int i = n; i < kSize; ++i) {
      result.raw[i] = Scalar{0};
    }
    return result;
  }

  template <typename IntT>
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static Simd LoadIndexed(
      Scalar const* ptr,
      Simd<IntT, N> const& indices)
    requires(std::integral<IntT>)
  {
    auto eval = []<size_t... I>(
                    Scalar const* ptr, Simd<IntT, N> const& indices, std::index_sequence<I...>) {
      return Simd{ptr[indices.raw[I]]...};
    };
    return eval(ptr, indices, std::make_index_sequence<N>{});
  }

  template <int kTupleCount = kSize>
  MOCHI_ANY MOCHI_FORCE_INLINE static void
  LoadTransposed(Scalar const* ptr, Simd& out0, Simd& out1, Simd& out2) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Invalid kTupleCount");
    constexpr Scalar zero{};
    constexpr int kTupleSize = 3;
    auto eval = [ptr, zero, &out0, &out1, &out2]<size_t... I>(std::index_sequence<I...>) {
      out0 = Simd{(I < kTupleCount ? ptr[I * kTupleSize + 0] : zero)...};
      out1 = Simd{(I < kTupleCount ? ptr[I * kTupleSize + 1] : zero)...};
      out2 = Simd{(I < kTupleCount ? ptr[I * kTupleSize + 2] : zero)...};
    };
    eval(std::make_index_sequence<kSize>{});
  }

  template <int M = kSize>
  MOCHI_ANY MOCHI_FORCE_INLINE static void Store(
      [[maybe_unused]] Scalar* ptr,
      [[maybe_unused]] Simd v) {
    static_assert(M >= 0 && M <= kSize, "Unsupported M");
    if constexpr (M != 0) {
      std::memcpy(ptr, v.raw.data(), M * sizeof(Scalar));
    }
  }

  MOCHI_ANY MOCHI_FORCE_INLINE static void Store(Scalar* ptr, Simd v, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid store size.");
    for (int i = 0; i < n; ++i) {
      ptr[i] = v.raw[i];
    }
  }

  MOCHI_ANY MOCHI_FORCE_INLINE static int StoreSelected(Scalar* ptr, Simd condition, Simd values) {
    int count = 0;
    auto eval = [&count]<size_t... I>(
                    Scalar* ptr,
                    Simd<Scalar, N> const& condition,
                    Simd<Scalar, N> const& values,
                    std::index_sequence<I...>) {
      ((ptr[count] = values.raw[I], count += static_cast<int>(!!condition.raw[I])), ...);
    };
    eval(ptr, condition, values, std::make_index_sequence<N>{});
    return count;
  }

  template <int kTupleCount = kSize>
  MOCHI_ANY MOCHI_FORCE_INLINE static void
  StoreTransposed(Scalar* ptr, Simd out0, Simd out1, Simd out2) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Invalid kTupleCount");
    auto eval = [ptr, &out0, &out1, &out2]<size_t... I>(std::index_sequence<I...>) {
      ((I < kTupleCount ? (void)(ptr[I * 3 + 0] = out0.raw[I],
                                 ptr[I * 3 + 1] = out1.raw[I],
                                 ptr[I * 3 + 2] = out2.raw[I])
                        : (void)0),
       ...);
    };
    eval(std::make_index_sequence<kSize>{});
  }

  template <int M = kSize>
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static constexpr Scalar HMin(Simd a) {
    static_assert(M >= 1 && M <= kSize, "Unsupported M");
    return FoldApply<M, [](auto... v) { return mochi::Min(v...); }>(a);
  }

  template <int M = kSize>
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static constexpr Scalar HMax(Simd a) {
    static_assert(M >= 1 && M <= kSize, "Unsupported M");
    return FoldApply<M, [](auto... v) { return mochi::Max(v...); }>(a);
  }

  template <int M = kSize>
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static constexpr Scalar HSum(Simd a) {
    static_assert(M >= 1 && M <= kSize, "Unsupported M");
    return FoldApply<M, [](auto... v) { return (v + ...); }>(a);
  }

  template <int M = kSize>
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static constexpr Scalar HProd(Simd a) {
    static_assert(M >= 1 && M <= kSize, "Unsupported M");
    return FoldApply<M, [](auto... v) { return (v * ...); }>(a);
  }

  template <int M = kSize>
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static constexpr Simd Dot(Simd a, Simd b) {
    static_assert(M > 0 && M <= kSize, "Unsupported M");
    auto eval = []<size_t... I>(Simd const& a, Simd const& b, std::index_sequence<I...>) {
      return Simd{((a.raw[I] * b.raw[I]) + ...)};
    };
    return eval(a, b, std::make_index_sequence<M>{});
  }

  MOCHI_SIMD_EMULATOR_HREDUCE_BOOL(AllTrue, &&)
  MOCHI_SIMD_EMULATOR_HREDUCE_BOOL(AnyTrue, ||)

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static constexpr Simd
  Select(Simd mask, Simd a, Simd b) {
    auto eval = []<size_t... I>(
                    auto const& mask, Simd const& a, Simd const& b, std::index_sequence<I...>) {
      return Simd{(std::bit_cast<UIntT>(mask.raw[I]) != UIntT{0} ? a.raw[I] : b.raw[I])...};
    };
    return eval(mask, a, b, std::make_index_sequence<N>{});
  }

  // Unary functions.
  // TODO: SinCosImpl in simd_inl.h might be faster than this Cos() and Sin().
  static constexpr auto Sqrt = Xapply<std::sqrt>;
  static constexpr auto Abs = Xapply<std::abs>;
  static constexpr auto Floor = Xapply<std::floor>;
  static constexpr auto FastRound = Xapply<std::round>;
  static constexpr auto RcpApprox = Xapply<[](ST a) { return ST{1} / a; }>;
  static constexpr auto RcpSqrtApprox = Xapply<[](ST a) { return ST{1} / std::sqrt(a); }>;
  static constexpr auto Cos = Xapply<std::cos>;
  static constexpr auto Sin = Xapply<std::sin>;
  static constexpr auto Tan = Xapply<std::tan>;
  static constexpr auto ACos = Xapply<std::acos>;
  static constexpr auto ASin = Xapply<std::asin>;
  static constexpr auto ATan = Xapply<std::atan>;
  static constexpr auto Exp = Xapply<std::exp>;
  static constexpr auto Ln = Xapply<std::log>;
  static constexpr auto Tanh = Xapply<std::tanh>;

  // Binary functions.
  static constexpr auto Min = XYapply<T const&, mochi::Min<T>>;
  static constexpr auto Max = XYapply<T const&, mochi::Max<T>>;

  MOCHI_ANY MOCHI_FORCE_INLINE static Simd Equal(Simd const& x, Simd const& y) {
    return Simd([](T a, T b) { return a == b ? Scalar{kOnesMask} : Scalar{0}; }, x, y);
  }

  MOCHI_ANY MOCHI_FORCE_INLINE static Simd NotEqual(Simd const& x, Simd const& y) {
    return Simd([](T a, T b) { return a != b ? Scalar{kOnesMask} : Scalar{0}; }, x, y);
  }

  // Ternary functions.
  static constexpr auto MulAdd = XYZapply<T, mochi::MulAdd<T, T, T>>;
  static constexpr auto MulSub = XYZapply<T, mochi::MulSub<T, T, T>>;
  static constexpr auto NegMulAdd = XYZapply<T, mochi::NegMulAdd<T, T, T>>;
  static constexpr auto NegMulSub = XYZapply<T, mochi::NegMulSub<T, T, T>>;

  // Unary operators.
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr Simd operator-() const {
    return Simd([](T a) { return -a; }, *this);
  }

  // This one uses a function that may not work on CUDA (std::bit_cast)
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr Simd operator~() const {
    return Simd([](T a) { return std::bit_cast<Scalar>(~std::bit_cast<UIntT>(a)); }, *this);
  }

  // Binary operators.
  MOCHI_SIMD_EMULATOR_OP_1(<, lhs.raw[I] < rhs.raw[I] ? kOnesMask : Scalar{0});
  MOCHI_SIMD_EMULATOR_OP_1(>, lhs.raw[I] > rhs.raw[I] ? kOnesMask : Scalar{0});
  MOCHI_SIMD_EMULATOR_OP_1(<=, lhs.raw[I] <= rhs.raw[I] ? kOnesMask : Scalar{0});
  MOCHI_SIMD_EMULATOR_OP_1(>=, lhs.raw[I] >= rhs.raw[I] ? kOnesMask : Scalar{0});
  MOCHI_SIMD_EMULATOR_OP_1(+, lhs.raw[I] + rhs.raw[I]);
  MOCHI_SIMD_EMULATOR_OP_1(-, lhs.raw[I] - rhs.raw[I]);
  MOCHI_SIMD_EMULATOR_OP_1(*, lhs.raw[I] * rhs.raw[I]);
  MOCHI_SIMD_EMULATOR_OP_1(/, lhs.raw[I] / rhs.raw[I]);
  MOCHI_SIMD_EMULATOR_OP_2(&);
  MOCHI_SIMD_EMULATOR_OP_2(|);
  MOCHI_SIMD_EMULATOR_OP_2(^);
  MOCHI_SIMD_EMULATOR_OP_2(<<);
  MOCHI_SIMD_EMULATOR_OP_2(>>);

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr bool operator==(Simd rhs) const {
    auto eval = []<size_t... I>(Simd const& lhs, Simd const& rhs, std::index_sequence<I...>) {
      return ((lhs.raw[I] == rhs.raw[I]) && ...);
    };
    return eval(*this, rhs, std::make_index_sequence<N>{});
  }

  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE constexpr bool operator!=(Simd rhs) const {
    return !(*this == rhs);
  }

  template <int kShift>
  [[nodiscard]] MOCHI_ANY MOCHI_FORCE_INLINE static constexpr Simd ShiftRight(Simd a) {
    return a >> kShift;
  }

#undef MOCHI_SIMD_EMULATOR_OP_1
#undef MOCHI_SIMD_EMULATOR_OP_2
#undef MOCHI_SIMD_EMULATOR_HREDUCE_BOOL
};

template <class To, class FromT, int FromN>
MOCHI_ANY MOCHI_FORCE_INLINE To ReinterpretCast(Simd<FromT, FromN> const& in)
  requires(IsSimd<To> && Simd<FromT, FromN>::kIsSupported)
{
  if constexpr (std::is_same_v<std::decay_t<To>, Simd<FromT, FromN>>) {
    return in;
  } else {
    static_assert(sizeof(To) == sizeof(FromT) * FromN, "Size mismatch");
    To out;
    std::memcpy(&out.raw, &in.raw, sizeof(in.raw));
    return out;
  }
}

template <class To, class FromT, int FromN>
MOCHI_ANY MOCHI_FORCE_INLINE To StaticCast(Simd<FromT, FromN> const& in)
  requires(IsSimd<To> && Simd<FromT, FromN>::kIsSupported)
{
  if constexpr (std::is_same_v<std::decay_t<To>, Simd<FromT, FromN>>) {
    return in;
  } else {
    static_assert(To::kSize == FromN, "Size mismatch");
    auto eval = [&in]<size_t... I>(std::index_sequence<I...>) {
      return To{static_cast<typename To::Scalar>(in.raw[I])...};
    };
    return eval(std::make_index_sequence<FromN>{});
  }
}

} // namespace mochi
#endif
