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

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_physics/dbg/protocol.h>

#include <cstdint>

namespace mochi::dbg {

struct UiState;

struct ProfilerPanelState {
  float barWidthPx = 1.0f; // Changed with the mouse wheel over the graph
  bool followLatest = true; // Keep the newest step at the right edge of the graph
  uint64_t windowEndSequence = 0; // History sequence one past the rightmost visible bar
  float plotWidthPx = 0.0f; // Plot area width from the previous frame
  float yMaxMs = 0.0f; // Smoothed y-axis limit [ms]

  // Copied from the client's history each frame, so the graph is drawn without holding its lock.
  DynamicArray<protocol::StepProfileData> window; // Visible steps, oldest first
  uint64_t windowBeginSequence = 0;
  uint64_t historyBeginSequence = 0;
  uint64_t historyEndSequence = 0;

  // Stats for the latest step. Refreshed at a readable rate while running.
  protocol::StepProfileData statsProfile;
  float statsStepSec = 0.0f; // Averaged while running, the latest step's while paused
  double statsUpdateTime = -1.0;
};

// Profiler panel: a per-step timing graph and simulation statistics for the selected scene.
void BuildProfilerPanel(UiState& state);

} // namespace mochi::dbg
