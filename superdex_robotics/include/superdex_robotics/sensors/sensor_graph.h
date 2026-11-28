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
#include <optional>
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

  /* @brief Add a borrowed sensor under a nonempty, unique node name, which becomes the sensor's
   * GraphName() and keys its random streams when its processKey is empty. Errors leave the graph
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
  /* Copy input names into a graph node without constructing a sensor. */
  void AddNode(std::string_view name, Span<DynamicString const> inputs, superdex::Error& error);
  /* Attach a sensor to a built node and refresh readiness; Build initializes active sensors. */
  void BindNode(SensorGraphNode const& binding, superdex::Error& error);
  /* Release bindings in a built graph and their dependants, retaining their input names. */
  DynamicArray<SensorGraphNodeIdentity> UnbindNode(
      Span<SensorHandle const> handles,
      superdex::Error& error);
  /* Whether all upstream bindings of a built node are ready; false while unbuilt. */
  [[nodiscard]] bool InputsReady(std::string_view name) const;
  /* Built node names in dependency order, borrowed until topology changes; empty while unbuilt. */
  [[nodiscard]] DynamicArray<std::string_view> GetDependencyOrder() const;

  /* @brief Select a canonical node name while constructing the graph. */
  void SetOutputEnabled(std::string_view nodeName, bool enabled, superdex::Error& error);
  /* @brief Enable existing outputs and initialize newly active dependencies in a built graph.
   * Empty names do nothing. Failure restores the previous selections; running state and clock
   * are preserved. */
  void AddOutputs(Span<DynamicString const> names, superdex::Error& error);
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
  /* @brief Set the episode seed of every member sensor that draws random streams (noise, outliers,
   * delay jitter, see SensorBase::SetEpisodeSeed). Apply before Reset(). */
  void SetEpisodeSeed(uint64_t episodeSeed);
  /* The graph-wide seed override, if set; Clear removes it. */
  [[nodiscard]] std::optional<uint64_t> GetEpisodeSeed() const {
    return _episodeSeed;
  }
  bool Step(StepTime const& now, superdex::Error& error);
  /* @brief The time of the last successful Step(); absent before the first and after Reset(). */
  [[nodiscard]] std::optional<StepTime> LastStep() const {
    return _hasLastStep ? std::optional<StepTime>{_lastStep} : std::nullopt;
  }

  /* @brief Find a node in the currently built graph; returns null while the graph is unbuilt. */
  [[nodiscard]] SensorBase* Find(std::string_view nodeName) const;
  /* @brief Names of every node in the built graph, active or not, in the order they were added;
   * empty while unbuilt. The names are borrowed until graph mutation or destruction. */
  [[nodiscard]] DynamicArray<std::string_view> GetNodeNames() const;

 private:
  struct NodeNameHash {
    using is_transparent = void;
    std::size_t operator()(std::string_view name) const noexcept {
      return std::hash<std::string_view>{}(name);
    }
  };

  struct Node {
    std::string name;
    SensorBase* sensor = nullptr;
    DynamicArray<DynamicString> inputNames;
    bool ready = false;
    std::vector<std::size_t> inputIndices;
    SensorHandle handle;
    bool requestedOutput = false;
    bool activeOutput = false;
    bool active = false;
    bool initialized = false;
  };

  /* Releases the graph's borrow of the node's sensor and the graph name the node gave it. */
  static void Release(Node const& node);
  void RebuildActiveOrder();
  void RefreshActiveOrder();
  void ResolveTopology(superdex::Error& error);
  /* Recompute readiness from live bindings and upstream readiness; preserves sensor history. */
  void ComputeReadiness();
  void RebuildEnabledNodes();
  void InitializeActiveSensors(superdex::Error& error, bool resetNewSensors);

  std::vector<Node> _nodes;
  std::vector<std::string> _requestedOutputNames;
  std::vector<std::size_t> _topologicalOrder;
  std::vector<std::size_t> _activeOrder;
  std::vector<SensorGraphNodeView> _enabledNodes;
  std::unordered_map<std::string, std::size_t, NodeNameHash, std::equal_to<>> _byName;
  std::unordered_map<SensorHandle, std::size_t> _byHandle;
  SensorScope _scope;
  std::optional<uint64_t> _episodeSeed;
  StepTime _lastStep;
  bool _topologyNeedsRebuild = true;
  bool _built = false;
  bool _hasLastStep = false;
  bool _activeOrderNeedsRebuild = false;
};

} // namespace superdex::robotics
