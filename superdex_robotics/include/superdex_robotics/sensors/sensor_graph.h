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

#include <superdex_robotics/core/context.h>
#include <superdex_robotics/sensors/sensor_base.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace superdex::robotics {

struct SensorHandle;

struct SensorGraphNodeView {
  std::string_view name;
  SensorHandle handle;
  SensorBase const* sensor = nullptr;
};

/* @brief Graph wiring; inputs contain canonical node names, not sensor pointers. */
struct SensorGraphNode {
  DynamicString name;
  SensorHandle handle;
  SensorBase* sensor = nullptr;
  DynamicArray<DynamicString> inputs;
};

struct SensorGraphNodeIdentity {
  DynamicString name;
  SensorHandle handle;
};

/* @brief Owns dependency ordering for a graph of borrowed synchronous sensors.
 * Graph mutation, stepping and sensor destruction must be serialized by the caller. */
class MOCHI_API SensorGraph {
 public:
  SensorGraph() = default;
  ~SensorGraph();

  MOCHI_DECLARE_NO_COPY_NO_MOVE(SensorGraph)

  /* @brief Add a borrowed sensor under a nonempty, unique node name. Errors leave the graph
   * unchanged. */
  void Add(SensorGraphNode const& node, superdex::Error& error);
  void Add(std::string_view nodeName, SensorBase* sensor, superdex::Error& error) {
    Add(nodeName, sensor, {}, error);
  }
  void Add(
      std::string_view nodeName,
      SensorBase* sensor,
      Span<DynamicString const> inputs,
      superdex::Error& error);
  void Add(
      std::string_view nodeName,
      SensorHandle const& handle,
      SensorBase* sensor,
      Span<DynamicString const> inputs,
      superdex::Error& error);
  /* @brief Select a canonical node name while constructing the graph. */
  void SetOutputEnabled(std::string_view nodeName, bool enabled, superdex::Error& error);
  /* @brief Extend a built graph with explicit nodes and optional enabled output names.
   * Preserves existing sensor state and clock. Empty enabledNames adds no output selections. */
  void Extend(
      Span<SensorGraphNode const> nodes,
      Span<DynamicString const> enabledNames,
      superdex::Error& error);
  /* @brief Remove member sensors and their downstream dependants, preserving other sensor state.
   * Requires a built graph; nonmember handles are ignored, invalid handles are errors.
   * Returns owned identities of removed enabled nodes. Does not destroy sensors. */
  DynamicArray<SensorGraphNodeIdentity> Remove(
      Span<SensorHandle const> handles,
      superdex::Error& error);
  void Clear();
  /* @brief Resolve topology and initialize newly active nodes. Existing state and clock survive;
   * changing the scope requires Clear(). */
  void Build(SensorScope const& scope, superdex::Error& error);
  /* @brief Request an output by handle. The new dependency closure applies on the next reset. */
  void SetOutputEnabled(SensorHandle const& handle, bool enabled, superdex::Error& error);
  /* @brief Whether this node is requested as an output; dependencies may run while this is false.
   */
  [[nodiscard]] bool GetOutputEnabled(SensorHandle const& handle, superdex::Error& error) const;
  /* @brief Enumerate explicitly enabled nodes in dependency order; empty while unbuilt.
   * The borrowed view is invalidated by graph mutation or destruction. Output selection may
   * precede activation, which takes effect on Build() or Reset(). */
  [[nodiscard]] Span<SensorGraphNodeView const> GetEnabledNodes() const;
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
    SensorHandle handle;
    bool requestedOutput = false;
    bool active = false;
    bool initialized = false;
  };

  void RebuildActiveOrder();
  void ResolveTopology(superdex::Error& error);
  void RebuildEnabledNodes();
  void InitializeActiveSensors(superdex::Error& error);

  std::vector<Node> _nodes;
  std::vector<std::string> _requestedOutputNames;
  std::vector<std::size_t> _topologicalOrder;
  std::vector<std::size_t> _activeOrder;
  std::vector<SensorGraphNodeView> _enabledNodes;
  std::unordered_map<std::string, std::size_t> _byName;
  std::unordered_map<SensorHandle, std::size_t> _byHandle;
  SensorScope _scope;
  StepTime _lastStep;
  bool _built = false;
  bool _hasLastStep = false;
  bool _activeOrderNeedsRebuild = false;
};

} // namespace superdex::robotics
