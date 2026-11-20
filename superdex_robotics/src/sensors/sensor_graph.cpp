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
    MOCHI_LOG_WARNING("SensorGraph::Add: duplicate node name '%s'", std::string(nodeName).c_str());
    MOCHI_ERROR_SET(
        error, "SensorGraph::Add: duplicate node name; see log for the conflicting name");
    return;
  }
  Node node;
  node.name = nodeName;
  node.sensor = sensor;
  node.handle = handle;
  node.inputNames.reserve(inputs.size());
  for (DynamicString const& input : inputs) {
    node.inputNames.emplace_back(input.c_str(), input.size());
  }
  _nodes.push_back(std::move(node));
  if (sensor != nullptr) {
    ++sensor->_graphBorrowCount;
  }
  _built = false;
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

void SensorGraph::Extend(
    Span<SensorGraphNode const> nodes,
    Span<DynamicString const> enabledNames,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (!_built) {
    MOCHI_ERROR_SET(error, "SensorGraph must be built before extension");
    return;
  }
  if (nodes.empty() && enabledNames.empty()) {
    return;
  }
  std::size_t const previousSize = _nodes.size();
  auto const previousOutputs = _requestedOutputNames;
  for (SensorGraphNode const& node : nodes) {
    Add(node, error);
    if (!error.IsOK()) {
      break;
    }
  }
  for (DynamicString const& name : enabledNames) {
    SetOutputEnabled(std::string_view(name.c_str(), name.size()), true, error);
  }
  if (error.IsOK()) {
    ResolveTopology(error);
  }
  if (!error.IsOK()) {
    /* Failed additions release their borrows; surviving sensors never need a reset. */
    while (_nodes.size() > previousSize) {
      --_nodes.back().sensor->_graphBorrowCount;
      _nodes.pop_back();
    }
    _requestedOutputNames = previousOutputs;
    ResolveTopology(ErrorLog{});
  }
}

DynamicArray<SensorGraphNodeIdentity> SensorGraph::Remove(
    Span<SensorHandle const> handles,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  if (!_built) {
    MOCHI_ERROR_SET(error, "SensorGraph must be built before removing nodes");
    return {};
  }
  std::vector<bool> removed(_nodes.size(), false);
  std::queue<std::size_t> pending;
  for (SensorHandle const& handle : handles) {
    if (!handle.IsValid()) {
      MOCHI_ERROR_SET(error, "SensorGraph::Remove requires valid sensor handles");
      return {};
    }
    auto const node = _byHandle.find(handle);
    if (node == _byHandle.end()) {
      continue;
    }
    if (!removed[node->second]) {
      removed[node->second] = true;
      pending.push(node->second);
    }
  }
  if (pending.empty()) {
    return {};
  }

  /* Prune the entire downstream closure before releasing any sensor borrow. */
  std::vector<std::vector<std::size_t>> consumers(_nodes.size());
  for (std::size_t i = 0; i < _nodes.size(); ++i) {
    for (std::size_t const input : _nodes[i].inputIndices) {
      consumers[input].push_back(i);
    }
  }
  while (!pending.empty()) {
    std::size_t const index = pending.front();
    pending.pop();
    for (std::size_t const consumer : consumers[index]) {
      if (!removed[consumer]) {
        removed[consumer] = true;
        pending.push(consumer);
      }
    }
  }

  std::erase_if(
      _requestedOutputNames, [&](std::string const& name) { return removed[_byName.at(name)]; });
  std::vector<Node> surviving;
  DynamicArray<SensorGraphNodeIdentity> removedOutputs;
  surviving.reserve(_nodes.size());
  for (std::size_t i = 0; i < _nodes.size(); ++i) {
    if (removed[i]) {
      if (_nodes[i].requestedOutput) {
        removedOutputs.push_back({DynamicString(_nodes[i].name), _nodes[i].handle});
      }
      --_nodes[i].sensor->_graphBorrowCount;
    } else {
      surviving.push_back(std::move(_nodes[i]));
    }
  }
  _nodes = std::move(surviving);
  ResolveTopology(error);
  return removedOutputs;
}

void SensorGraph::Clear() {
  for (Node const& node : _nodes) {
    if (node.sensor != nullptr) {
      --node.sensor->_graphBorrowCount;
    }
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
  ResolveTopology(error);
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
    if (node.name.empty() || node.sensor == nullptr) {
      MOCHI_ERROR_SET(error, "SensorGraph contains an empty name or null sensor");
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
      if (_nodes[previous].sensor == node.sensor) {
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
    for (std::string const& inputName : node.inputNames) {
      auto const input = _byName.find(inputName);
      if (input == _byName.end()) {
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

  // Resolve the requested output names.
  for (std::string const& outputName : _requestedOutputNames) {
    auto const output = _byName.find(outputName);
    if (output == _byName.end()) {
      MOCHI_ERROR_SET(error, "SensorGraph selected output name does not match any graph node");
      return;
    }
    _nodes[output->second].requestedOutput = true;
  }
  // Activate the selected outputs and their upstream dependencies.
  RebuildActiveOrder();
  InitializeActiveSensors(error);
  MOCHI_ERROR_RETURN(error);
  _built = true;
}

void SensorGraph::SetOutputEnabled(SensorHandle const& handle, bool enabled, Error& error) {
  MOCHI_ERROR_RETURN(error);
  auto const node = _byHandle.find(handle);
  MOCHI_ERROR_IF(
      !_built || !handle.IsValid() || node == _byHandle.end(),
      error,
      "SensorGraph::SetOutputEnabled requires a built graph and a member sensor handle");
  MOCHI_ERROR_RETURN(error);
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
    if (node.requestedOutput) {
      _enabledNodes.push_back({.name = node.name, .handle = node.handle, .sensor = node.sensor});
    }
  }
}

void SensorGraph::RebuildActiveOrder() {
  for (Node& node : _nodes) {
    node.active = false;
  }
  std::vector<std::size_t> pending;
  for (std::size_t i = 0; i < _nodes.size(); ++i) {
    if (_nodes[i].requestedOutput) {
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
  _activeOrderNeedsRebuild = false;
  RebuildEnabledNodes();
}

void SensorGraph::InitializeActiveSensors(Error& error) {
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
    if (_hasLastStep) {
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
    InitializeActiveSensors(error);
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

SensorBase* SensorGraph::Find(std::string_view nodeName) const {
  if (!_built) {
    return nullptr;
  }
  auto const it = _byName.find(std::string{nodeName});
  return it == _byName.end() ? nullptr : _nodes[it->second].sensor;
}
