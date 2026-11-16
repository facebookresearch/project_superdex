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

#include <mochi_core/test/mochi_test_helpers.h>
#include <superdex_robotics/sensors/sensor_graph.h>
#include <superdex_robotics/sensors/sensor_link_vertex_force.h>
#include <superdex_robotics/sensors/sensor_seed_rect_map.h>

#include <cmath>

using namespace mochi;
using namespace superdex::robotics;

namespace {

class VertexForceSource final : public SensorBase {
 public:
  explicit VertexForceSource(Error& error) : SensorBase(error) {
    real const shoulder = std::sqrt(0.75_r);
    _positions = {
        {0_r, 0_r, 1_r},
        {0.5_r, 0_r, shoulder},
        {-0.5_r, 0_r, shoulder},
        {0_r, 0.5_r, shoulder},
        {0_r, -0.5_r, shoulder},
    };
    _meta.coordinateFrame = "test_sensor";
    _meta.vertexAxis = 0;
    _meta.positionsMeters = MakeConstSpan(_positions);
  }

  static constexpr std::string_view TypeName() {
    return "TEST_VERTEX_FORCE_SOURCE";
  }

  [[nodiscard]] std::string_view GetTypeName() const override {
    return TypeName();
  }

  SensorStepResult Step(StepTime const& now, Error& error) override {
    MOCHI_ERROR_IF_NOT(Inputs().empty(), error, "VertexForceSource accepts no inputs");
    MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
    ++_stepCount;
    for (std::size_t vertex = 0; vertex < _positions.size(); ++vertex) {
      _values[3 * vertex] = static_cast<float>(vertex + 1);
      _values[3 * vertex + 1] = static_cast<float>(now.step);
      _values[3 * vertex + 2] = -static_cast<float>(vertex);
    }
    PublishSample(now, Span<float const>{_values}, error);
    MOCHI_ERROR_RETURN(error, SensorStepResult::NoUpdate);
    return SensorStepResult::Published;
  }

  [[nodiscard]] Span<std::byte const> TryGetMetadata(std::string_view key) const override {
    if (key == SensorMetadataKey<VertexListMeta>::Key) {
      return {reinterpret_cast<std::byte const*>(&_meta), sizeof(_meta)};
    }
    return {};
  }

  [[nodiscard]] int StepCount() const {
    return _stepCount;
  }

 private:
  void ResetSensor() override {
    _stepCount = 0;
  }

  SensorFormat InitSensor(SensorScope const& /*scope*/, Error& error) override {
    MOCHI_ERROR_IF_NOT(Inputs().empty(), error, "VertexForceSource accepts no inputs");
    MOCHI_ERROR_RETURN(error, {});
    _values.resize(3 * _positions.size());
    return {
        .kind = DynamicString{kSensorFormatKindForce3D},
        .shape = {static_cast<int32_t>(_positions.size()), 3},
        .axes =
            {
                {.name = "vertex", .labels = {}, .units = {}},
                MakeForceComponentAxis(),
            },
    };
  }

  DynamicArray<Real3> _positions;
  DynamicArray<float> _values;
  VertexListMeta _meta;
  int _stepCount = 0;
};

SensorSeedRectMapParams RectParams() {
  SensorSeedRectMapParams params;
  params.width = 4;
  params.height = 4;
  params.voronoiOversampling = 2;
  return params;
}

TEST(SensorSeedRectMapTest, MapsVertexForcesAndExposesTypedMetadata) {
  Error constructionError;
  VertexForceSource source(constructionError);
  SensorSeedRectMap sensor(RectParams(), constructionError);
  ASSERT_TRUE(constructionError.IsOK()) << constructionError.ToString();

  SensorGraph graph;
  graph.Add("source", &source, mochi::test::ExpectOK{});
  DynamicArray<DynamicString> const sourceInput{"source"};
  graph.Add("rect", &sensor, sourceInput, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("rect", true, mochi::test::ExpectOK{});

  Error error;
  graph.Build({}, error);
  graph.Reset(error);
  graph.Step({.t = 0.01, .dt = 0.01, .step = 2}, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();

  SensorSample const sample = sensor.Latest();
  EXPECT_EQ(sample.captureStep, 2);
  EXPECT_EQ(sample.elementCount, 4 * 4 * 3);
  auto const* const rect = TryGetSensorMetadata<RectMapMeta>(sensor);
  ASSERT_NE(rect, nullptr);
  EXPECT_EQ(rect->cellToSourceVertex.size(), 16);
  EXPECT_NE(TryGetSensorMetadata<VertexListMeta>(sensor), nullptr);
  for (int32_t const vertex : rect->cellToSourceVertex) {
    EXPECT_GE(vertex, -1);
    EXPECT_LT(vertex, 5);
  }
}

TEST(SensorSeedRectMapTest, RepeatedGraphStepDoesNotRecompute) {
  Error constructionError;
  VertexForceSource source(constructionError);
  SensorSeedRectMap sensor(RectParams(), constructionError);
  ASSERT_TRUE(constructionError.IsOK()) << constructionError.ToString();

  SensorGraph graph;
  graph.Add("source", &source, mochi::test::ExpectOK{});
  DynamicArray<DynamicString> const sourceInput{"source"};
  graph.Add("rect", &sensor, sourceInput, mochi::test::ExpectOK{});
  graph.SetOutputEnabled("rect", true, mochi::test::ExpectOK{});
  Error error;
  graph.Build({}, error);
  graph.Reset(error);
  StepTime const now{.t = 0.02, .dt = 0.01, .step = 4};
  graph.Step(now, error);
  SensorSample const first = sensor.Latest();
  graph.Step(now, error);
  ASSERT_TRUE(error.IsOK()) << error.ToString();
  SensorSample const second = sensor.Latest();
  EXPECT_EQ(source.StepCount(), 1);
  EXPECT_EQ(first.captureStep, second.captureStep);
  EXPECT_EQ(first.data, second.data);
}

} // namespace
