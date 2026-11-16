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

#include <cmath>
#include <queue>

using namespace mochi;
using namespace superdex::robotics;

void SensorGraph::Add(
    std::string_view nodeName,
    SensorBase* sensor,
    Span<DynamicString const> inputs) {
  Node node;
  node.name = nodeName;
  node.sensor = sensor;
  node.inputNames.reserve(inputs.size());
  for (DynamicString const& input : inputs) {
    node.inputNames.emplace_back(input.c_str(), input.size());
  }
  _nodes.push_back(std::move(node));
  _built = false;
}

void SensorGraph::AddOutput(std::string_view nodeName) {
  _outputNames.emplace_back(nodeName);
  _built = false;
}

void SensorGraph::Build(SensorScope const& scope, Error& error) {
  // Kahn's algorithm provides the dependency order and detects cycles.
  MOCHI_ERROR_RETURN(error);
  _built = false;
  _hasLastStep = false;
  _activeOrder.clear();
  _byName.clear();

  // Validate node identities and build the name lookup.
  for (std::size_t i = 0; i < _nodes.size(); ++i) {
    Node& node = _nodes[i];
    node.active = false;
    node.inputIndices.clear();
    node.inputs.clear();
    if (node.name.empty() || node.sensor == nullptr) {
      MOCHI_ERROR_SET(error, "SensorGraph contains an empty name or null sensor");
      return;
    }
    if (!_byName.emplace(node.name, i).second) {
      MOCHI_ERROR_SET(error, "SensorGraph contains a duplicate sensor name");
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
      node.inputs.push_back(_nodes[input->second].sensor);
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
  std::vector<std::size_t> allOrder;
  while (!ready.empty()) {
    std::size_t const index = ready.front();
    ready.pop();
    allOrder.push_back(index);
    for (std::size_t const consumer : consumers.at(index)) {
      if (--indegree.at(consumer) == 0) {
        ready.push(consumer);
      }
    }
  }
  if (allOrder.size() != _nodes.size()) {
    MOCHI_ERROR_SET(error, "SensorGraph contains a dependency cycle");
    return;
  }

  // Walk upstream from selected outputs to mark the required dependencies.
  if (_outputNames.empty()) {
    for (Node& node : _nodes) {
      node.active = true;
    }
  } else {
    std::vector<std::size_t> pending;
    for (std::string const& outputName : _outputNames) {
      auto const output = _byName.find(outputName);
      if (output == _byName.end()) {
        MOCHI_ERROR_SET(error, "SensorGraph selected output name does not match any graph node");
        return;
      }
      pending.push_back(output->second);
    }
    while (!pending.empty()) {
      std::size_t const index = pending.back();
      pending.pop_back();
      Node& node = _nodes[index];
      if (!node.active) {
        node.active = true;
        pending.insert(pending.end(), node.inputIndices.begin(), node.inputIndices.end());
      }
    }
  }

  // Initialize active sensors in dependency order.
  for (std::size_t const index : allOrder) {
    Node& node = _nodes[index];
    if (!node.active) {
      continue;
    }
    if (node.sensor == nullptr) {
      MOCHI_ERROR_SET(error, "SensorGraph contains a null sensor");
      return;
    }
    node.sensor->Init(node.inputs, scope, error);
    MOCHI_ERROR_RETURN(error);
    _activeOrder.push_back(index);
  }
  _built = true;
}

void SensorGraph::Reset(Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (!_built) {
    MOCHI_ERROR_SET(error, "SensorGraph must be built before reset");
    return;
  }
  for (std::size_t const index : _activeOrder) {
    SensorBase* const sensor = _nodes[index].sensor;
    if (sensor == nullptr) {
      MOCHI_ERROR_SET(error, "Built SensorGraph contains a null sensor");
      return;
    }
    sensor->Reset();
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
