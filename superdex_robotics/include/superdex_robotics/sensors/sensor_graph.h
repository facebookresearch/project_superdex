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

#include <superdex_robotics/sensors/sensor_base.h>

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace superdex::robotics {

/* @brief Owns dependency ordering for a fixed graph of borrowed synchronous sensors. */
class MOCHI_API SensorGraph {
 public:
  SensorGraph() = default;
  ~SensorGraph() = default;

  MOCHI_DECLARE_NO_COPY_NO_MOVE(SensorGraph)

  void Add(std::string_view nodeName, SensorBase* sensor, Span<DynamicString const> inputs = {});
  void AddOutput(std::string_view nodeName);
  void Build(SensorScope const& scope, superdex::Error& error);
  void Reset(superdex::Error& error);
  bool Step(StepTime const& now, superdex::Error& error);

  /* @brief Find a node in the currently built graph; returns null while the graph is unbuilt. */
  [[nodiscard]] SensorBase* Find(std::string_view nodeName) const;

 private:
  struct Node {
    std::string name;
    SensorBase* sensor = nullptr;
    std::vector<std::string> inputNames;
    std::vector<std::size_t> inputIndices;
    std::vector<SensorBase const*> inputs;
    bool active = false;
  };

  std::vector<Node> _nodes;
  std::vector<std::string> _outputNames;
  std::vector<std::size_t> _activeOrder;
  std::unordered_map<std::string, std::size_t> _byName;
  StepTime _lastStep;
  bool _built = false;
  bool _hasLastStep = false;
};

} // namespace superdex::robotics
