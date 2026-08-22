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

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/range_algorithms.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/span_utils.h>

#include <array>
#include <numeric>
#include <utility>
#include <vector>

using namespace mochi;

TEST(BasicTypes, Span) {
  // Default
  {
    Span<int> s;
    EXPECT_EQ(0, s.size());
    EXPECT_EQ(nullptr, s.data());
    static_assert(Span<int>{}.empty());
    static_assert(nullptr == Span<int>{}.data());
  }

  // From ptr + len
  {
    int buffer[] = {11, 22, 33, 44};
    Span<int> s(buffer + 1, 2);
    EXPECT_EQ(2, s.size());
    EXPECT_EQ(22, s[0]);
    EXPECT_EQ(33, s[1]);

    int zero = 0;
    Span<int> s2(buffer + 1, zero);
  }

  // From begin + end
  {
    int buffer[] = {11, 22, 33, 44};
    Span<int> s(buffer + 1, buffer + 3);
    EXPECT_EQ(2, s.size());
    EXPECT_EQ(22, s[0]);
    EXPECT_EQ(33, s[1]);
  }

  // From c-style array
  {
    int buffer[] = {11, 22, 33};
    Span<int> s = buffer;
    EXPECT_EQ(3, s.size());
    EXPECT_EQ(11, s[0]);
    EXPECT_EQ(22, s[1]);
    EXPECT_EQ(33, s[2]);
  }

  // From std::array
  {
    std::array<int, 3> buffer = {11, 22, 33};
    Span<int> s = buffer;
    EXPECT_EQ(3, s.size());
    EXPECT_EQ(11, s[0]);
    EXPECT_EQ(22, s[1]);
    EXPECT_EQ(33, s[2]);
    EXPECT_EQ(11, s.front());
    EXPECT_EQ(33, s.back());
  }

  // From std::vector
  {
    std::vector<int> buffer = {11, 22, 33};
    Span<int> s = buffer;
    EXPECT_EQ(3, s.size());
    EXPECT_EQ(11, s[0]);
    EXPECT_EQ(22, s[1]);
    EXPECT_EQ(33, s[2]);
    EXPECT_EQ(buffer.front(), s.front());
    EXPECT_EQ(buffer.back(), s.back());
  }

  // Copy
  {
    int buffer[] = {11, 22};
    Span<int> s(buffer);
    Span<int> s2(s); // copy construct
    Span<int> s3;
    s3 = s2; // copy assign
    EXPECT_EQ(2, s3.size());
    EXPECT_EQ(11, s3[0]);
    EXPECT_EQ(22, s3[1]);
    EXPECT_EQ(11, s3.front());
    EXPECT_EQ(22, s3.back());
  }

  // Ranged for
  {
    int buffer[] = {11, 22};
    Span<int> s(buffer);
    for (int& x : s) {
      ++x;
    }
    EXPECT_EQ(12, s[0]);
    EXPECT_EQ(23, s[1]);
    EXPECT_EQ(12, s.front());
    EXPECT_EQ(23, s.back());
  }

  // Reverse iterator
  {
    int buffer[] = {11, 22, 66};
    Span<int> s(buffer);
    std::vector<int> v;
    for (int x : ReverseRange(s)) {
      v.push_back(x);
    }
    EXPECT_EQ(3, v.size());
    EXPECT_EQ(66, v[0]);
    EXPECT_EQ(22, v[1]);
    EXPECT_EQ(11, v[2]);
  }

  // subspan (1 argument)
  {
    int buffer[] = {11, 22};
    Span<int> s(buffer);
    EXPECT_EQ(2, s.subspan(0).size());
    EXPECT_EQ(11, s.subspan(0)[0]);
    EXPECT_EQ(22, s.subspan(0)[1]);

    EXPECT_EQ(1, s.subspan(1).size());
    EXPECT_EQ(22, s.subspan(1)[0]);
    EXPECT_EQ(0, s.subspan(2).size());
  }

  // subspan (2 arguments)
  {
    int buffer[] = {11, 22};
    Span<int> s(buffer);
    EXPECT_EQ(0, s.subspan(0, 0).size());
    EXPECT_EQ(1, s.subspan(0, 1).size());
    EXPECT_EQ(11, s.subspan(0, 1)[0]);
    EXPECT_EQ(2, s.subspan(0, 2).size());
    EXPECT_EQ(11, s.subspan(0, 2)[0]);
    EXPECT_EQ(22, s.subspan(0, 2)[1]);

    EXPECT_EQ(0, s.subspan(1, 0).size());
    EXPECT_EQ(1, s.subspan(1, 1).size());
    EXPECT_EQ(22, s.subspan(1, 1)[0]);
    EXPECT_EQ(0, s.subspan(2, 0).size());
  }
}

TEST(BasicTypes, ConstSpan) {
  // Default
  {
    Span<int const> s;
    EXPECT_EQ(0, s.size());
    EXPECT_EQ(nullptr, s.data());
    static_assert(Span<int const>{}.empty());
    static_assert(nullptr == Span<int const>{}.data());
  }

  // From ptr + len
  {
    constexpr int kBuffer[] = {11, 22, 33, 44};
    Span<int const> s(kBuffer + 1, 2);
    EXPECT_EQ(2, s.size());
    EXPECT_EQ(22, s[0]);
    EXPECT_EQ(33, s[1]);
  }

  // From c-style array
  {
    constexpr int kBuffer[] = {11, 22, 33};
    Span<int const> s = kBuffer;
    EXPECT_EQ(3, s.size());
    EXPECT_EQ(11, s[0]);
    EXPECT_EQ(22, s[1]);
    EXPECT_EQ(33, s[2]);
  }

  // From std::array
  {
    constexpr std::array<int, 3> kBuffer = {11, 22, 33};
    Span<int const> s = kBuffer;
    EXPECT_EQ(3, s.size());
    EXPECT_EQ(11, s[0]);
    EXPECT_EQ(22, s[1]);
    EXPECT_EQ(33, s[2]);
  }

  // From std::vector
  {
    std::vector<int> const buffer = {11, 22, 33};
    Span<int const> s = buffer;
    EXPECT_EQ(3, s.size());
    EXPECT_EQ(11, s[0]);
    EXPECT_EQ(22, s[1]);
    EXPECT_EQ(33, s[2]);
  }

  // From std::vector r-value
  {
    std::vector<int> v{11, 22, 33};
    Span<int const> s = std::move(v);
    EXPECT_EQ(3, s.size());
    EXPECT_EQ(11, s[0]);
    EXPECT_EQ(22, s[1]);
    EXPECT_EQ(33, s[2]);
  }

  // Copy
  {
    int const buffer[] = {11, 22};
    Span<int const> s(buffer);
    Span<int const> s2(s); // copy construct
    Span<int const> s3;
    s3 = s2; // copy assign
    EXPECT_EQ(2, s3.size());
    EXPECT_EQ(11, s3[0]);
    EXPECT_EQ(22, s3[1]);
  }

  // Ranged for (read-only)
  {
    int const buffer[] = {11, 22};
    Span<int const> s(buffer);
    int i = 0;
    for (int const& x : s) {
      EXPECT_EQ(i++ ? 22 : 11, x);
    }
  }

  // operator==, operator!=
  {
    // Test various sizes including those large enough to use SIMD internally.
    int constexpr kTestSizes[] = {0, 1, 2, 3, 4, 7, 8, 15, 16, 63, 64};
    for (int sz : kTestSizes) {
      std::vector<int> vecA(sz);
      std::vector<int> vecB(sz);
      std::iota(vecA.begin(), vecA.end(), 0);
      vecB = vecA;

      // Expect equality. Include const & non-const to make sure they all compile.
      EXPECT_TRUE(MakeSpan(vecA) == MakeSpan(vecB));
      EXPECT_TRUE(MakeConstSpan(vecA) == MakeSpan(vecB));
      EXPECT_TRUE(MakeSpan(vecA) == MakeConstSpan(vecB));
      EXPECT_TRUE(MakeConstSpan(vecA) == MakeConstSpan(vecB));
      EXPECT_FALSE(MakeSpan(vecA) != MakeSpan(vecB));
      EXPECT_FALSE(MakeConstSpan(vecA) != MakeSpan(vecB));
      EXPECT_FALSE(MakeSpan(vecA) != MakeConstSpan(vecB));
      EXPECT_FALSE(MakeConstSpan(vecA) != MakeConstSpan(vecB));

      // Size mismatch
      if (sz > 0) {
        EXPECT_FALSE(Span(vecA.data(), sz - 1) == MakeConstSpan(vecB));
        EXPECT_TRUE(Span(vecA.data(), sz - 1) != MakeConstSpan(vecB));
      }

      // Value mismatch
      for (int i = 0; i < isize(vecA); ++i) {
        vecB[i] += 1;
        EXPECT_FALSE(MakeConstSpan(vecA) == MakeConstSpan(vecB));
        vecB[i] -= 1;
        EXPECT_TRUE(MakeConstSpan(vecA) == MakeConstSpan(vecB));
      }
    }
  }
}

TEST(Span, ScalarType) {
  static_assert(std::is_same_v<ScalarType<Span<float>>, float>);
  static_assert(std::is_same_v<ScalarType<Span<double const>>, double>);
  static_assert(std::is_same_v<ScalarType<Span<Span<int const>>>, int>);
  static_assert(std::is_same_v<ScalarType<Span<Span<Simd<int64_t>>>>, int64_t>);
}

namespace mochi::test::details {

template <typename FromScalar, typename FromSz, typename ToScalar, typename ToSz>
void StaticCast(size_t len, unsigned int seed) {
  static_assert(std::is_same_v<std::remove_const_t<ToScalar>, ToScalar>);
  using NonConstFromScalar = std::remove_const_t<FromScalar>;

  if (len == 0) {
    Span<FromScalar, FromSz> srcEmpty;
    Span<ToScalar, ToSz> castEmpty;
    StaticCast(MakeConstSpan(srcEmpty), castEmpty);
    EXPECT_TRUE(castEmpty.empty());
    return;
  }

  FromScalar constexpr frTol = std::numeric_limits<FromScalar>::epsilon();
  ToScalar constexpr toTol = std::numeric_limits<ToScalar>::epsilon();

  ColumnVector<NonConstFromScalar> srcVec(len);
  srcVec.SetRandom(seed);
  Span<NonConstFromScalar, FromSz> srcSpan{srcVec.data(), static_cast<FromSz>(srcVec.Rows())};

  ColumnVector<ToScalar> castVec(len);
  Span<ToScalar, ToSz> castSpan{castVec.data(), static_cast<ToSz>(castVec.Rows())};

  StaticCast(MakeConstSpan(srcSpan), castSpan);

  EXPECT_EQ(srcSpan.size(), castSpan.size());
  if constexpr (std::is_same_v<ToScalar const, FromScalar const>) {
    EXPECT_TRUE(test::NearEqualSpan(castSpan, srcSpan, frTol));
  } else if constexpr (frTol < toTol) {
    EXPECT_TRUE(test::NearEqualSpan(castSpan, srcSpan, toTol));
    // Check that some data is lost in the conversion
    EXPECT_FALSE(test::NearEqualSpan(castSpan, srcSpan, frTol));
  } else {
    // frTol >= toTol
    EXPECT_TRUE(test::NearEqualSpan(castSpan, srcSpan, frTol));
  }
}

} // namespace mochi::test::details

TEST(SpanUtils, StaticCastSpan) {
  for (auto len : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100}) {
    test::details::StaticCast<float, int, float, int>(len, 42 + int(len));
    test::details::StaticCast<float, int, double, int>(len, 42 + int(len));
    test::details::StaticCast<double, int, float, int>(len, 42 + int(len));
    test::details::StaticCast<double, int, double, int>(len, 42 + int(len));
    //
    test::details::StaticCast<float, int, float, size_t>(len, 42 + int(len));
    test::details::StaticCast<float, int, double, size_t>(len, 42 + int(len));
    test::details::StaticCast<double, int, float, size_t>(len, 42 + int(len));
    test::details::StaticCast<double, int, double, size_t>(len, 42 + int(len));
    //
    test::details::StaticCast<float, size_t, float, int>(len, 42 + int(len));
    test::details::StaticCast<float, size_t, double, int>(len, 42 + int(len));
    test::details::StaticCast<double, size_t, float, int>(len, 42 + int(len));
    test::details::StaticCast<double, size_t, double, int>(len, 42 + int(len));
    //
    test::details::StaticCast<float, size_t, float, size_t>(len, 42 + int(len));
    test::details::StaticCast<float, size_t, double, size_t>(len, 42 + int(len));
    test::details::StaticCast<double, size_t, float, size_t>(len, 42 + int(len));
    test::details::StaticCast<double, size_t, double, size_t>(len, 42 + int(len));
  }
}
