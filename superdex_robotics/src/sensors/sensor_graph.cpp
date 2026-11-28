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

#include <superdex_robotics/sensors/sensor_graph.h>

#include <superdex_robotics/core/context.h>

#include <algorithm>
#include <cmath>
#include <queue>

using namespace mochi;
using namespace superdex::robotics;

SensorGraph::~SensorGraph() {
  Clear();
}

void SensorGraph::Add(SensorGraphNode const& node, Error& error) {
  Add(std::string_view(node.name.c_str(), node.name.size()),
      node.handle,
      node.sensor,
      node.inputs,
      error);
}

void SensorGraph::Add(
    std::string_view nodeName,
    SensorBase* sensor,
    Span<DynamicString const> inputs,
    Error& error) {
  Add(nodeName, SensorHandle{}, sensor, inputs, error);
}

void SensorGraph::Add(
    std::string_view nodeName,
    SensorHandle const& handle,
    SensorBase* sensor,
    Span<DynamicString const> inputs,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (nodeName.empty() || sensor == nullptr) {
    MOCHI_ERROR_SET(error, "SensorGraph::Add requires a nonempty node name and a non-null sensor");
    return;
  }
  if (std::any_of(
          _nodes.begin(), _nodes.end(), [&](Node const& node) { return node.name == nodeName; })) {
    MOCHI_LOG_WARNING(
        "SensorGraph::Add: duplicate node name '%.*s'",
        static_cast<int>(nodeName.size()),
        nodeName.data());
    MOCHI_ERROR_SET(
        error, "SensorGraph::Add: duplicate node name; see log for the conflicting name");
    return;
  }
  Node node;
  node.name = nodeName;
  node.sensor = sensor;
  node.handle = handle;
  node.inputNames.assign(inputs.begin(), inputs.end());
  _nodes.push_back(std::move(node));
  _topologyNeedsRebuild = true;
  // The first graph to hold a sensor names it, until the last graph releases it.
  if (sensor->_graphBorrowCount++ == 0) {
    sensor->_graphName = DynamicString{nodeName};
  }
  if (_episodeSeed) {
    sensor->SetEpisodeSeed(*_episodeSeed);
  }
  _built = false;
}

void SensorGraph::AddNode(std::string_view name, Span<DynamicString const> inputs, Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (name.empty() ||
      std::ranges::any_of(_nodes, [&](Node const& node) { return node.name == name; })) {
    MOCHI_ERROR_SET(error, "SensorGraph::AddNode requires a nonempty unique name");
    return;
  }
  Node node;
  node.name = name;
  node.inputNames.assign(inputs.begin(), inputs.end());
  _nodes.push_back(std::move(node));
  _topologyNeedsRebuild = true;
  _built = false;
}

void SensorGraph::BindNode(SensorGraphNode const& binding, Error& error) {
  MOCHI_ERROR_RETURN(error);
  auto const found = _byName.find(std::string_view(binding.name));
  if (!_built || found == _byName.end() || binding.sensor == nullptr) {
    MOCHI_ERROR_SET(error, "SensorGraph::BindNode requires a built node and a sensor");
    return;
  }
  Node& node = _nodes[found->second];
  if (!std::ranges::equal(node.inputNames, binding.inputs)) {
    MOCHI_ERROR_SET(error, "SensorGraph binding changes input names; rebuild required");
    return;
  }
  if (node.sensor == binding.sensor && node.handle == binding.handle) {
    return;
  }
  SensorBase* const sensor = binding.sensor;
  SensorHandle const handle = binding.handle;
  if (node.sensor != nullptr || std::ranges::any_of(_nodes, [&](Node const& other) {
        return other.sensor == sensor || (handle.IsValid() && other.handle == handle);
      })) {
    MOCHI_ERROR_SET(error, "SensorGraph::BindNode requires an unbound node and a unique sensor");
    return;
  }
  node.sensor = sensor;
  node.handle = handle;
  if (_episodeSeed) {
    sensor->SetEpisodeSeed(*_episodeSeed);
  }
  if (sensor->_graphBorrowCount++ == 0) {
    sensor->_graphName = binding.name;
  }
  if (handle.IsValid()) {
    _byHandle.emplace(handle, found->second);
  }
  ComputeReadiness();
  RebuildEnabledNodes();
}

DynamicArray<SensorGraphNodeIdentity> SensorGraph::UnbindNode(
    Span<SensorHandle const> handles,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  if (!_built) {
    MOCHI_ERROR_SET(error, "SensorGraph must be built before unbinding nodes");
    return {};
  }
  std::vector<bool> affected(_nodes.size(), false);
  for (SensorHandle const& handle : handles) {
    if (!handle.IsValid()) {
      MOCHI_ERROR_SET(error, "SensorGraph::UnbindNode requires valid handles");
      return {};
    }
    if (auto const node = _byHandle.find(handle); node != _byHandle.end()) {
      affected[node->second] = true;
    }
  }
  DynamicArray<SensorGraphNodeIdentity> removedOutputs;
  for (std::size_t const index : _topologicalOrder) {
    Node& node = _nodes[index];
    for (std::size_t const input : node.inputIndices) {
      affected[index] = affected[index] || affected[input];
    }
    if (!affected[index]) {
      continue;
    }
    if (node.requestedOutput) {
      removedOutputs.push_back({DynamicString(node.name), node.handle});
    }
    Release(node);
    _byHandle.erase(node.handle);
    node.sensor = nullptr;
    node.handle = {};
    node.initialized = false;
    node.requestedOutput = false;
    node.activeOutput = false;
    node.active = false;
  }
  std::erase_if(
      _requestedOutputNames, [&](std::string const& name) { return affected[_byName.at(name)]; });
  ComputeReadiness();
  RefreshActiveOrder();
  return removedOutputs;
}

bool SensorGraph::InputsReady(std::string_view name) const {
  if (!_built) {
    return false;
  }
  auto const node = _byName.find(name);
  return node != _byName.end() &&
      std::ranges::all_of(_nodes[node->second].inputIndices, [&](std::size_t input) {
           return _nodes[input].ready;
         });
}

DynamicArray<std::string_view> SensorGraph::GetDependencyOrder() const {
  DynamicArray<std::string_view> names;
  if (_built) {
    for (std::size_t const index : _topologicalOrder) {
      names.push_back(_nodes[index].name);
    }
  }
  return names;
}

void SensorGraph::SetOutputEnabled(std::string_view nodeName, bool enabled, Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (nodeName.empty()) {
    MOCHI_ERROR_SET(error, "SensorGraph output name must not be empty");
    return;
  }
  auto const output =
      std::find(_requestedOutputNames.begin(), _requestedOutputNames.end(), nodeName);
  if (enabled && output == _requestedOutputNames.end()) {
    _requestedOutputNames.emplace_back(nodeName);
  } else if (!enabled && output != _requestedOutputNames.end()) {
    _requestedOutputNames.erase(output);
  }
  _built = false;
}

void SensorGraph::AddOutputs(Span<DynamicString const> names, Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (!_built) {
    MOCHI_ERROR_SET(error, "SensorGraph must be built before adding outputs");
    return;
  }
  if (names.empty()) {
    return;
  }
  for (DynamicString const& name : names) {
    auto const node = _byName.find(std::string_view(name));
    if (node == _byName.end()) {
      MOCHI_ERROR_SET(error, "SensorGraph selected output name does not match any graph node");
      return;
    }
    if (!_nodes[node->second].ready) {
      MOCHI_ERROR_SET(error, "SensorGraph selected output is waiting for sensor bindings");
      return;
    }
  }
  auto const previousOutputs = _requestedOutputNames;
  for (DynamicString const& name : names) {
    SetOutputEnabled(std::string_view(name), true, error);
  }
  Build(_scope, error);
  if (!error.IsOK()) {
    _requestedOutputNames = previousOutputs;
    Build(_scope, ErrorLog{});
  }
}

void SensorGraph::Release(Node const& node) {
  if (node.sensor != nullptr && --node.sensor->_graphBorrowCount == 0) {
    node.sensor->_graphName = DynamicString{};
  }
}

void SensorGraph::Clear() {
  for (Node const& node : _nodes) {
    Release(node);
  }
  _nodes.clear();
  _requestedOutputNames.clear();
  _topologicalOrder.clear();
  _activeOrder.clear();
  _enabledNodes.clear();
  _byName.clear();
  _byHandle.clear();
  _scope = {};
  _lastStep = {};
  _episodeSeed.reset();
  _topologyNeedsRebuild = true;
  _built = false;
  _hasLastStep = false;
  _activeOrderNeedsRebuild = false;
}

void SensorGraph::Build(SensorScope const& scope, Error& error) {
  MOCHI_ERROR_RETURN(error);
  if ((_scope.scene != scope.scene || _scope.robotics != scope.robotics) &&
      std::any_of(
          _nodes.begin(), _nodes.end(), [](Node const& node) { return node.initialized; })) {
    MOCHI_ERROR_SET(error, "SensorGraph scope change requires Clear before Build");
    return;
  }
  _scope = scope;
  _built = false;
  if (_topologyNeedsRebuild) {
    ResolveTopology(error);
    MOCHI_ERROR_RETURN(error);
  }
  ComputeReadiness();
  for (Node& node : _nodes) {
    node.requestedOutput = false;
  }
  // Resolve the requested output names.
  for (std::string const& outputName : _requestedOutputNames) {
    auto const output = _byName.find(outputName);
    if (output == _byName.end()) {
      MOCHI_ERROR_SET(error, "SensorGraph selected output name does not match any graph node");
      return;
    }
    if (!_nodes[output->second].ready) {
      MOCHI_ERROR_SET(error, "SensorGraph selected output is waiting for sensor bindings");
      return;
    }
    _nodes[output->second].requestedOutput = true;
  }
  // Activate the selected outputs and their upstream dependencies.
  RebuildActiveOrder();
  InitializeActiveSensors(error, _hasLastStep);
  MOCHI_ERROR_RETURN(error);
  _built = true;
}

void SensorGraph::ResolveTopology(Error& error) {
  // Kahn's algorithm provides the dependency order and detects cycles.
  MOCHI_ERROR_RETURN(error);
  _built = false;
  _topologicalOrder.clear();
  _activeOrder.clear();
  _enabledNodes.clear();
  _byName.clear();
  _byHandle.clear();

  // Validate node identities and build the name lookup.
  for (std::size_t i = 0; i < _nodes.size(); ++i) {
    Node& node = _nodes[i];
    node.requestedOutput = false;
    node.active = false;
    node.inputIndices.clear();
    if (node.name.empty()) {
      MOCHI_ERROR_SET(error, "SensorGraph contains an empty name");
      return;
    }
    if (!_byName.emplace(node.name, i).second) {
      MOCHI_ERROR_SET(error, "SensorGraph contains a duplicate sensor name");
      return;
    }
    if (node.handle.IsValid() && !_byHandle.emplace(node.handle, i).second) {
      MOCHI_ERROR_SET(error, "SensorGraph contains a duplicate sensor handle");
      return;
    }
    for (std::size_t previous = 0; previous < i; ++previous) {
      if (node.sensor != nullptr && _nodes[previous].sensor == node.sensor) {
        MOCHI_ERROR_SET(error, "SensorGraph contains the same sensor under multiple names");
        return;
      }
    }
  }

  // Resolve input names and count each node's dependencies.
  std::vector<std::vector<std::size_t>> consumers(_nodes.size());
  std::vector<std::size_t> indegree(_nodes.size(), 0);
  for (std::size_t i = 0; i < _nodes.size(); ++i) {
    Node& node = _nodes[i];
    for (DynamicString const& inputName : node.inputNames) {
      auto const input = _byName.find(std::string_view(inputName));
      if (input == _byName.end()) {
        MOCHI_LOG_ERROR(
            "SensorGraph node '%s' references missing input '%s'",
            node.name.c_str(),
            inputName.c_str());
        MOCHI_ERROR_SET(error, "SensorGraph input name does not match any graph node");
        return;
      }
      node.inputIndices.push_back(input->second);
      consumers.at(input->second).push_back(i);
      ++indegree.at(i);
    }
  }

  // Topologically sort: release consumers when all their inputs have been ordered.
  std::queue<std::size_t> ready;
  for (std::size_t i = 0; i < indegree.size(); ++i) {
    if (indegree[i] == 0) {
      ready.push(i);
    }
  }
  _topologicalOrder.reserve(_nodes.size());
  while (!ready.empty()) {
    std::size_t const index = ready.front();
    ready.pop();
    _topologicalOrder.push_back(index);
    for (std::size_t const consumer : consumers.at(index)) {
      if (--indegree.at(consumer) == 0) {
        ready.push(consumer);
      }
    }
  }
  if (_topologicalOrder.size() != _nodes.size()) {
    MOCHI_ERROR_SET(error, "SensorGraph contains a dependency cycle");
    return;
  }

  _topologyNeedsRebuild = false;
}

void SensorGraph::ComputeReadiness() {
  // Binding readiness follows the same dependency order used for sensor initialization.
  for (std::size_t const index : _topologicalOrder) {
    Node& node = _nodes[index];
    node.ready = node.sensor != nullptr;
    for (std::size_t const input : node.inputIndices) {
      node.ready = node.ready && _nodes[input].ready;
    }
  }
}

void SensorGraph::SetOutputEnabled(SensorHandle const& handle, bool enabled, Error& error) {
  MOCHI_ERROR_RETURN(error);
  auto const node = _byHandle.find(handle);
  MOCHI_ERROR_IF(
      !_built || !handle.IsValid() || node == _byHandle.end(),
      error,
      "SensorGraph::SetOutputEnabled requires a built graph and a member sensor handle");
  MOCHI_ERROR_RETURN(error);
  if (enabled && !_nodes[node->second].ready) {
    MOCHI_ERROR_SET(error, "SensorGraph selected output is waiting for sensor bindings");
    return;
  }
  if (_nodes[node->second].requestedOutput != enabled) {
    Node& selected = _nodes[node->second];
    auto const output =
        std::find(_requestedOutputNames.begin(), _requestedOutputNames.end(), selected.name);
    if (enabled) {
      _requestedOutputNames.push_back(selected.name);
    } else if (output != _requestedOutputNames.end()) {
      _requestedOutputNames.erase(output);
    }
    selected.requestedOutput = enabled;
    RebuildEnabledNodes();
    _activeOrderNeedsRebuild = true;
  }
}

bool SensorGraph::GetOutputEnabled(SensorHandle const& handle, Error& error) const {
  MOCHI_ERROR_RETURN(error, false);
  auto const node = _byHandle.find(handle);
  MOCHI_ERROR_IF(
      !_built || !handle.IsValid() || node == _byHandle.end(),
      error,
      "SensorGraph::GetOutputEnabled requires a built graph and a member sensor handle");
  MOCHI_ERROR_RETURN(error, false);
  return _nodes[node->second].requestedOutput;
}

Span<SensorGraphNodeView const> SensorGraph::GetEnabledNodes() const {
  if (!_built) {
    return {};
  }
  return {_enabledNodes.data(), _enabledNodes.size()};
}

void SensorGraph::RebuildEnabledNodes() {
  _enabledNodes.clear();
  for (std::size_t const index : _topologicalOrder) {
    Node const& node = _nodes[index];
    if (node.requestedOutput && node.ready) {
      _enabledNodes.push_back({.name = node.name, .handle = node.handle, .sensor = node.sensor});
    }
  }
}

void SensorGraph::RebuildActiveOrder() {
  for (Node& node : _nodes) {
    node.activeOutput = node.requestedOutput && node.ready;
  }
  _activeOrderNeedsRebuild = false;
  RefreshActiveOrder();
}

void SensorGraph::RefreshActiveOrder() {
  for (Node& node : _nodes) {
    node.active = false;
  }
  std::vector<std::size_t> pending;
  for (std::size_t i = 0; i < _nodes.size(); ++i) {
    if (_nodes[i].activeOutput && _nodes[i].ready) {
      pending.push_back(i);
    }
  }
  while (!pending.empty()) {
    std::size_t const index = pending.back();
    pending.pop_back();
    Node& node = _nodes[index];
    if (node.active) {
      continue;
    }
    node.active = true;
    pending.insert(pending.end(), node.inputIndices.begin(), node.inputIndices.end());
  }
  _activeOrder.clear();
  for (std::size_t const index : _topologicalOrder) {
    if (_nodes[index].active) {
      _activeOrder.push_back(index);
    }
  }
  RebuildEnabledNodes();
}

void SensorGraph::InitializeActiveSensors(Error& error, bool resetNewSensors) {
  MOCHI_ERROR_RETURN(error);
  // Initialize active sensors in dependency order.
  for (std::size_t const index : _activeOrder) {
    Node& node = _nodes[index];
    if (node.initialized) {
      continue;
    }
    SensorBase* const sensor = node.sensor;
    if (sensor == nullptr) {
      MOCHI_ERROR_SET(error, "Built SensorGraph contains a null sensor");
      return;
    }
    std::vector<SensorBase const*> inputs;
    inputs.reserve(node.inputIndices.size());
    for (std::size_t const input : node.inputIndices) {
      inputs.push_back(_nodes[input].sensor);
    }
    sensor->Init(inputs, _scope, error);
    MOCHI_ERROR_RETURN(error);
    if (resetNewSensors) {
      /* A new node joins the current clock with fresh local history; no catch-up steps. */
      sensor->Reset();
    }
    node.initialized = true;
  }
}

void SensorGraph::Reset(Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (!_built) {
    MOCHI_ERROR_SET(error, "SensorGraph must be built before reset");
    return;
  }
  if (_activeOrderNeedsRebuild) {
    RebuildActiveOrder();
    InitializeActiveSensors(error, /*resetNewSensors=*/false);
    MOCHI_ERROR_RETURN(error);
  }
  for (Node& node : _nodes) {
    if (node.initialized) {
      SensorBase* const sensor = node.sensor;
      if (sensor == nullptr) {
        MOCHI_ERROR_SET(error, "Built SensorGraph contains a null sensor");
        return;
      }
      sensor->Reset();
    }
  }
  _hasLastStep = false;
}

void SensorGraph::SetEpisodeSeed(uint64_t episodeSeed) {
  _episodeSeed = episodeSeed;
  for (Node const& node : _nodes) {
    if (node.sensor != nullptr) {
      node.sensor->SetEpisodeSeed(episodeSeed);
    }
  }
}

bool SensorGraph::Step(StepTime const& now, Error& error) {
  MOCHI_ERROR_RETURN(error, false);
  if (!_built) {
    MOCHI_ERROR_SET(error, "SensorGraph must be built before step");
    return false;
  }
  if (now.step < 0 || now.t < 0.0 || now.dt < 0.0 || !std::isfinite(now.t) ||
      !std::isfinite(now.dt)) {
    MOCHI_ERROR_SET(error, "SensorGraph step time is invalid");
    return false;
  }
  if (_hasLastStep) {
    if (now.step == _lastStep.step) {
      if (now.t != _lastStep.t || now.dt != _lastStep.dt) {
        MOCHI_ERROR_SET(error, "SensorGraph repeated step has inconsistent time");
      }
      return error.IsOK();
    }
    if (now.step < _lastStep.step || now.t < _lastStep.t) {
      MOCHI_ERROR_SET(error, "SensorGraph step time regressed");
      return false;
    }
  }
  for (std::size_t const index : _activeOrder) {
    [[maybe_unused]] SensorStepResult const result = _nodes[index].sensor->Step(now, error);
    MOCHI_ERROR_RETURN(error, false);
  }
  _lastStep = now;
  _hasLastStep = true;
  return true;
}

DynamicArray<std::string_view> SensorGraph::GetNodeNames() const {
  DynamicArray<std::string_view> names;
  if (_built) {
    names.reserve(_nodes.size());
    for (Node const& node : _nodes) {
      names.emplace_back(node.name);
    }
  }
  return names;
}

SensorBase* SensorGraph::Find(std::string_view nodeName) const {
  if (!_built) {
    return nullptr;
  }
  auto const it = _byName.find(nodeName);
  return it == _byName.end() ? nullptr : _nodes[it->second].sensor;
}
