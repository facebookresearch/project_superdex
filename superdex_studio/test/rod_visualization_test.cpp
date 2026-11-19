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

#include "rendering/rod_visualization.h"

#include <gtest/gtest.h>

#include <vector>

using namespace superdex::studio;
using mochi::real;
using mochi::Real3;

namespace {

void ExpectNear(Real3 const& actual, Real3 const& expected) {
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(actual[i], expected[i], 1e-6f) << "component " << i;
  }
}

TEST(RodVisualizationTest, AnEmptyConnectivityIsTheOpenPolylineThroughTheNodes) {
  std::vector<mochi::Int2> const expected = {{0, 1}, {1, 2}};
  EXPECT_EQ(PolylineElements(3, {}), expected);
}

TEST(RodVisualizationTest, ExplicitConnectivityIsKept) {
  std::vector<int> const connectivity = {0, 1, 1, 2, 2, 0};
  std::vector<mochi::Int2> const expected = {{0, 1}, {1, 2}, {2, 0}};
  EXPECT_EQ(PolylineElements(3, mochi::MakeConstSpan(connectivity)), expected);
}

TEST(RodVisualizationTest, GlyphsStartAtElementMidpointsAndSpanHalfTheMeanElementLength) {
  // Elements of length 1 and 3 (mean 2), so every glyph is 1 long whatever its element's length.
  std::vector<real> const nodes = {0, 0, 0, 1, 0, 0, 4, 0, 0};
  std::vector<real> const axes = {0, 1, 0, 0, 0, 1};

  std::vector<FrameAxisGlyph> const glyphs =
      ComputeFrameAxisGlyphs(mochi::MakeConstSpan(nodes), {}, mochi::MakeConstSpan(axes));

  ASSERT_EQ(glyphs.size(), 2u);
  ExpectNear(glyphs[0].origin, {0.5f, 0.0f, 0.0f});
  ExpectNear(glyphs[0].vector, {0.0f, 1.0f, 0.0f});
  ExpectNear(glyphs[1].origin, {2.5f, 0.0f, 0.0f});
  ExpectNear(glyphs[1].vector, {0.0f, 0.0f, 1.0f});
}

TEST(RodVisualizationTest, AClosedLoopHasAGlyphOnItsWrapElement) {
  std::vector<real> const nodes = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
  std::vector<int> const connectivity = {0, 1, 1, 2, 2, 3, 3, 0};
  std::vector<real> const axes = {0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1};

  std::vector<FrameAxisGlyph> const glyphs = ComputeFrameAxisGlyphs(
      mochi::MakeConstSpan(nodes), mochi::MakeConstSpan(connectivity), mochi::MakeConstSpan(axes));

  ASSERT_EQ(glyphs.size(), 4u);
  ExpectNear(glyphs[3].origin, {0.0f, 0.5f, 0.0f});
  ExpectNear(glyphs[3].vector, {0.0f, 0.0f, 0.5f});
}

TEST(RodVisualizationTest, MismatchedFrameAxesGiveNoGlyphs) {
  std::vector<real> const nodes = {0, 0, 0, 1, 0, 0, 2, 0, 0};
  std::vector<real> const oneAxis = {0, 1, 0};

  EXPECT_TRUE(ComputeFrameAxisGlyphs(mochi::MakeConstSpan(nodes), {}, mochi::MakeConstSpan(oneAxis))
                  .empty());
}

} // namespace
