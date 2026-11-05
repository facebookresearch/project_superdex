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

#include "profiling_panel.h"

#include "gui/gui.h"

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_debugger/lib/debug_client.h>

#include <imguios/imguios.h>

#include <algorithm>
#include <cmath>

using namespace mochi;
using namespace mochi::dbg;
using StepProfileData = protocol::StepProfileData;

namespace {

struct Segment {
  char const* name = nullptr;
  ImVec4 color = {};
  float StepProfileData::* durationSec = nullptr;
};

struct BarGetterData {
  ProfilerPanelState const* panel = nullptr;
  int segment = 0;
};

struct StatsSnapshot {
  StepProfileData latest;
  float averageStepSec = 0.0f;
  bool hasProfile = false;
};

// Bar segments in display order: the legend, the stats and each bar list them top to bottom.
Segment const kSegments[] = {
    {"PreStep", ImVec4(0.90f, 0.62f, 0.25f, 1.0f), &StepProfileData::preStepSec},
    {"Step", ImVec4(0.30f, 0.60f, 0.90f, 1.0f), &StepProfileData::stepSec},
    {"PostStep", ImVec4(0.55f, 0.80f, 0.40f, 1.0f), &StepProfileData::postStepSec},
};
int constexpr kNumSegments = isize(kSegments);

ImVec4 constexpr kDivergedColor(0.90f, 0.20f, 0.20f, 1.0f);

int constexpr kSmoothingSteps = 30;
float constexpr kMinBarWidthPx = 1.0f;
float constexpr kMaxBarWidthPx = 32.0f;
float constexpr kZoomPerWheelStep = 1.25f;
float constexpr kYHeadroom = 1.2f; // Space above the tallest visible bar
float constexpr kYReleasePerFrame = 0.05f; // Fraction of the gap closed per frame when shrinking
float constexpr kMinYMaxMs = 0.01f;
float constexpr kEmptyYMaxMs = 1.0f;
double constexpr kStatsUpdatePeriodSec = 0.25;
int constexpr kMinHistorySize = 1;
int constexpr kMaxHistorySize = 1'000'000;

} // namespace

static float ToMs(float sec) {
  return 1000.0f * sec;
}

// Height of a bar at the top of `segment`: that segment plus every segment below it [s].
static float StackedSec(StepProfileData const& profile, int segment) {
  float sum = 0.0f;
  for (int i = segment; i < kNumSegments; ++i) {
    sum += profile.*kSegments[i].durationSec;
  }
  return sum;
}

static float TotalSec(StepProfileData const& profile) {
  return StackedSec(profile, 0);
}

static ImPlotPoint GetStackedBar(int index, void* userData) {
  auto const& data = *static_cast<BarGetterData const*>(userData);
  return {
      static_cast<double>(data.panel->window[index].sequence),
      ToMs(StackedSec(data.panel->window[index], data.segment))};
}

// Full-height bar of a diverged step.
static ImPlotPoint GetDivergedBar(int index, void* userData) {
  auto const& panel = *static_cast<ProfilerPanelState const*>(userData);
  int const step = panel.divergedSteps[index];
  return {static_cast<double>(panel.window[step].sequence), ToMs(TotalSec(panel.window[step]))};
}

static void
PlotBars(char const* label, ImVec4 const& color, ImPlotGetter getter, void* data, int count) {
  double constexpr kBarSize = 1.0; // Adjacent bars touch
  ImPlotSpec spec;
  spec.FillColor = color;
  spec.LineColor = color;
  ImPlot::PlotBarsG(label, getter, data, count, kBarSize, spec);
}

static bool HasDiverged(StepProfileData const& profile) {
  return profile.convergenceStatus == ConvergenceStatus::Diverged;
}

// Copy the visible steps and summarize the newest steps from the client's history.
static StatsSnapshot
CopySteps(DebugClient const& client, ProfilerPanelState& panel, int numVisible) {
  StatsSnapshot stats;
  client.ReadProfileHistory([&](StepProfileHistory const& history) {
    uint64_t const historyBegin = history.GetBeginSequence();
    uint64_t const historyEnd = history.GetEndSequence();
    MOCHI_ASSERT_VERBOSE(historyBegin <= historyEnd, "Invalid profile history sequence range");
    auto const visible = static_cast<uint64_t>(numVisible);
    uint64_t const fullWindowEnd = Min(historyEnd, historyBegin + visible);
    uint64_t const end =
        panel.followLatest ? historyEnd : Clamp(panel.windowEndSequence, fullWindowEnd, historyEnd);
    MOCHI_ASSERT_VERBOSE(
        historyBegin <= end && end <= historyEnd, "Invalid profile history window end");
    panel.followLatest = end == historyEnd;
    uint64_t const begin = end - Min(end - historyBegin, visible);
    MOCHI_ASSERT_VERBOSE(
        historyBegin <= begin && begin <= end, "Invalid profile history window begin");
    auto const findIndex = [&](uint64_t sequence) {
      auto const it = std::lower_bound(
          history.begin(), history.end(), sequence, [](auto const& profile, uint64_t value) {
            return profile.sequence < value;
          });
      return static_cast<int>(it - history.begin());
    };
    auto const copy = [&](DynamicArray<StepProfileData>& out, int from, int to) {
      MOCHI_ASSERT_VERBOSE(
          0 <= from && from <= to && to <= isize(history), "Invalid profile history copy range");
      out.resize_noinit(static_cast<size_t>(to - from));
      for (int i = 0; i < isize(out); ++i) {
        out[i] = history[from + i];
      }
    };
    copy(panel.window, findIndex(begin), findIndex(end));

    if (!history.empty()) {
      int const historySize = static_cast<int>(history.size());
      int const recentBegin = Max(0, historySize - kSmoothingSteps);
      float sumSec = 0.0f;
      for (int i = recentBegin; i < historySize; ++i) {
        sumSec += TotalSec(history[i]);
      }
      stats.latest = history.back();
      stats.averageStepSec = sumSec / static_cast<float>(historySize - recentBegin);
      stats.hasProfile = true;
    }

    panel.historyBeginSequence = historyBegin;
    panel.historyEndSequence = historyEnd;
    panel.windowBeginSequence = begin;
    panel.windowEndSequence = end;
  });
  return stats;
}

// Fast attack so spikes show immediately, slow release so the axis does not jitter.
static void UpdateYMax(ProfilerPanelState& panel) {
  float maxTotalMs = 0.0f;
  for (auto const& profile : panel.window) {
    maxTotalMs = Max(maxTotalMs, ToMs(TotalSec(profile)));
  }
  float const target =
      panel.window.empty() ? kEmptyYMaxMs : Max(kMinYMaxMs, kYHeadroom * maxTotalMs);
  panel.yMaxMs = (target > panel.yMaxMs) ? target : Lerp(panel.yMaxMs, target, kYReleasePerFrame);
}

// Change the bar width. While scrolled back, keep the step under the cursor in place.
static void Zoom(ProfilerPanelState& panel, float wheel, float cursorFromLeftPx) {
  float const barWidthPx =
      Clamp(panel.barWidthPx * std::pow(kZoomPerWheelStep, wheel), kMinBarWidthPx, kMaxBarWidthPx);
  if (!panel.followLatest) {
    double const cursorToRightPx = panel.plotWidthPx - cursorFromLeftPx;
    double const cursorSequence =
        static_cast<double>(panel.windowEndSequence) - cursorToRightPx / panel.barWidthPx;
    double const newEnd = Max(0.0, std::round(cursorSequence + cursorToRightPx / barWidthPx));
    panel.windowEndSequence = static_cast<uint64_t>(newEnd);
    panel.followLatest = panel.windowEndSequence >= panel.historyEndSequence;
  }
  panel.barWidthPx = barWidthPx;
}

static void AddRowLabel(
    char const* label,
    ImVec4 const& color = ImGui::GetStyleColorVec4(ImGuiCol_Text)) {
  ImGui::TableNextRow();
  ImGui::TableNextColumn();
  ImGui::TextColored(color, "%s", label);
  ImGui::TableNextColumn();
}

// A duration and the rate it corresponds to, e.g. "2.000 ms (500 Hz)".
static void TextDurationAndRate(float sec) {
  if (sec > 0.0f && IsFinite(sec)) {
    ImGui::Text("%.3f ms (%.0f Hz)", ToMs(sec), 1.0f / sec);
  } else {
    ImGui::Text("%.3f ms", ToMs(sec));
  }
}

// The value is left blank if there is no step to show.
static void AddStepTimeRow(StepProfileData const* profile, float stepSec) {
  AddRowLabel("Step Time");
  if (profile) {
    TextDurationAndRate(stepSec);
  }
}

static void
AddCountRow(char const* label, StepProfileData const* profile, int32_t StepProfileData::* count) {
  AddRowLabel(label);
  if (profile) {
    ImGui::Text("%d", profile->*count);
  }
}

// Rows describing one step, shared by the tooltip and the stats. Values are blank without a step.
static void AddStepRows(StepProfileData const* profile) {
  AddRowLabel("Sim Dt");
  if (profile) {
    TextDurationAndRate(profile->timeStepSec);
  }
  float const totalSec = profile ? TotalSec(*profile) : 0.0f;
  for (auto const& segment : kSegments) {
    AddRowLabel(segment.name, segment.color);
    if (profile) {
      float const sec = profile->*segment.durationSec;
      ImGui::Text("%.3f ms (%.0f%%)", ToMs(sec), totalSec > 0.0f ? 100.0f * sec / totalSec : 0.0f);
    }
  }
  AddCountRow("Active Actors", profile, &StepProfileData::activeActors);
  AddCountRow("Active Constraints", profile, &StepProfileData::activeConstraints);
  AddCountRow("Active Contacts", profile, &StepProfileData::activeContacts);
  AddCountRow("Active DoFs", profile, &StepProfileData::activeDofs);
  AddCountRow("Active Islands", profile, &StepProfileData::activeIslands);
  AddCountRow("Worker Threads", profile, &StepProfileData::numWorkerThreads);
}

static void BuildStepTooltip(StepProfileData const& profile) {
  ImGui::BeginTooltip();
  MOCHI_DEFER(ImGui::EndTooltip());
  if (ImGui::BeginTable("##Step", 2, ImGuiTableFlags_SizingFixedFit)) {
    MOCHI_DEFER(ImGui::EndTable());
    AddRowLabel("Convergence");
    ImGui::TextColored(
        HasDiverged(profile) ? kDivergedColor : ImGui::GetStyleColorVec4(ImGuiCol_Text),
        "%s",
        SReflect::EnumToString(profile.convergenceStatus));
    AddStepTimeRow(&profile, TotalSec(profile));
    AddStepRows(&profile);
  }
}

static void BuildContextMenu(UiState& state) {
  auto& panel = state.profiler;
  if (ImGui::MenuItem("Clear history")) {
    state.client->ClearProfileHistory();
    panel.followLatest = true;
  }
  auto settings = state.client->GetSettings();
  int historySize = settings.maxProfileHistory;
  ImGui::SetNextItemWidth(8.0f * ImGui::GetFontSize());
  if (ImGui::InputInt(
          "History size", &historySize, 1000, 10000, ImGuiInputTextFlags_EnterReturnsTrue)) {
    settings.maxProfileHistory = Clamp(historySize, kMinHistorySize, kMaxHistorySize);
    state.client->SetSettings(settings);
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Number of steps to keep. Press Enter to apply.");
  }
}

// Why the graph is empty.
static char const* GetEmptyGraphMessage(DebugClient const& client) {
  if (client.GetStatus() != net::SocketStatus::Connected) {
    return "Not connected";
  }
  if (!client.GetSelectedScene().IsValid()) {
    return "No scene selected";
  }
  if (client.GetSceneStepMode() == StepMode::Pause) {
    return "Press play to begin...";
  }
  return "Waiting for steps...";
}

// Draw the stacked bar graph, with a tooltip for the hovered step.
static void BuildGraph(UiState& state, ImVec2 size) {
  auto& panel = state.profiler;
  UpdateYMax(panel);

  StepProfileData const* hovered = nullptr;
  bool openContextMenu = false;
  ImPlotFlags constexpr kPlotFlags =
      ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText;
  if (ImPlot::BeginPlot("##Profiler", size, kPlotFlags)) {
    MOCHI_DEFER(ImPlot::EndPlot());

    // We own the view: the wheel sets the bar width and the scrollbar sets the position.
    ImPlot::SetupAxes(
        nullptr,
        "ms",
        ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoTickLabels | ImPlotAxisFlags_NoGridLines,
        ImPlotAxisFlags_Lock);
    double const right = static_cast<double>(panel.windowEndSequence) - 0.5;
    double const left = right - panel.plotWidthPx / panel.barWidthPx;
    ImPlot::SetupAxisLimits(ImAxis_X1, left, right, ImPlotCond_Always);
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, panel.yMaxMs, ImPlotCond_Always);
    ImPlot::SetupLegend(ImPlotLocation_NorthWest, ImPlotLegendFlags_NoButtons);

    // Tallest first: each bar is then covered by the shorter bars of the segments below it.
    int const numSteps = isize(panel.window);
    for (int segment = 0; segment < kNumSegments; ++segment) {
      BarGetterData data{&panel, segment};
      PlotBars(kSegments[segment].name, kSegments[segment].color, &GetStackedBar, &data, numSteps);
    }

    // Diverged steps are covered entirely in red. "##" hides them from the legend.
    panel.divergedSteps.clear();
    for (int i = 0; i < numSteps; ++i) {
      if (HasDiverged(panel.window[i])) {
        panel.divergedSteps.push_back(i);
      }
    }
    PlotBars("##Diverged", kDivergedColor, &GetDivergedBar, &panel, isize(panel.divergedSteps));
    if (panel.window.empty()) {
      ImPlot::PlotText(
          GetEmptyGraphMessage(*state.client), 0.5 * (left + right), 0.5 * panel.yMaxMs);
    }

    panel.plotWidthPx = ImPlot::GetPlotSize().x;
    if (ImPlot::IsPlotHovered()) {
      uint64_t const sequence =
          static_cast<uint64_t>(Max(0.0, std::round(ImPlot::GetPlotMousePos().x)));
      auto const* it = std::lower_bound(
          panel.window.begin(),
          panel.window.end(),
          sequence,
          [](auto const& profile, uint64_t value) { return profile.sequence < value; });
      if (it != panel.window.end() && it->sequence == sequence) {
        hovered = &*it;
      }
      if (float const wheel = ImGui::GetIO().MouseWheel; wheel != 0.0f) {
        Zoom(panel, wheel, ImGui::GetMousePos().x - ImPlot::GetPlotPos().x);
      }
      openContextMenu = ImGui::IsMouseClicked(ImGuiMouseButton_Right);
    }
  }

  if (hovered) {
    BuildStepTooltip(*hovered);
  }
  if (openContextMenu) {
    ImGui::OpenPopup("##ProfilerMenu");
  }
  if (ImGui::BeginPopup("##ProfilerMenu")) {
    MOCHI_DEFER(ImGui::EndPopup());
    BuildContextMenu(state);
  }
}

// A horizontal scrollbar over the whole history. Reaching the end resumes following.
static void BuildScrollbar(ProfilerPanelState& panel, int numVisible) {
  ImVec2 const size(ImGui::GetContentRegionAvail().x, ImGui::GetStyle().ScrollbarSize);
  ImVec2 const pos = ImGui::GetCursorScreenPos();
  ImGui::ItemSize(size);

  MOCHI_ASSERT_VERBOSE(
      panel.historyBeginSequence <= panel.historyEndSequence &&
          panel.historyBeginSequence <= panel.windowEndSequence,
      "Invalid profiler scrollbar sequence range");
  auto const historySize =
      static_cast<ImS64>(panel.historyEndSequence - panel.historyBeginSequence);
  if (historySize <= numVisible) {
    return;
  }
  auto scroll =
      static_cast<ImS64>(panel.windowEndSequence - panel.historyBeginSequence) - numVisible;
  if (ImGui::ScrollbarEx(
          ImRect(pos, pos + size),
          ImGui::GetID("##ProfilerScroll"),
          ImGuiAxis_X,
          &scroll,
          numVisible,
          historySize,
          0)) {
    MOCHI_ASSERT_VERBOSE(scroll + numVisible >= 0, "Invalid profiler scrollbar position");
    panel.windowEndSequence =
        panel.historyBeginSequence + static_cast<uint64_t>(scroll + numVisible);
    panel.followLatest = panel.windowEndSequence >= panel.historyEndSequence;
  }
}

// Show the latest step. Unaffected by hovering, which has its own tooltip. While running, the step
// time is averaged and refreshed at a readable rate. While paused, every new step is shown as it
// arrives.
static void BuildStats(UiState& state, StatsSnapshot const& stats) {
  auto& panel = state.profiler;
  bool const isPaused = state.client->GetSceneStepMode() == StepMode::Pause;
  double const now = ImGui::GetTime();
  if (!stats.hasProfile) {
    panel.statsUpdateTime = -1.0; // Show the first step immediately
  } else if (isPaused) {
    panel.statsProfile = stats.latest;
    panel.statsStepSec = TotalSec(panel.statsProfile);
    panel.statsUpdateTime = -1.0; // Refresh immediately when playback resumes
  } else if (panel.statsUpdateTime < 0.0 || now - panel.statsUpdateTime >= kStatsUpdatePeriodSec) {
    panel.statsProfile = stats.latest;
    panel.statsStepSec = stats.averageStepSec;
    panel.statsUpdateTime = now;
  }

  StepProfileData const* profile = stats.hasProfile ? &panel.statsProfile : nullptr;

  if (!ImGui::BeginTable("##Stats", 2, ImGuiTableFlags_SizingFixedFit)) {
    return;
  }
  MOCHI_DEFER(ImGui::EndTable());
  AddStepTimeRow(profile, panel.statsStepSec);
  if (profile && !isPaused && ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Averaged over recent steps");
  }
  AddStepRows(profile);
}

void dbg::BuildProfilerPanel(UiState& state) {
  auto& panel = state.profiler;
  float const statsWidth = 18.0f * ImGui::GetFontSize();
  if (panel.plotWidthPx <= 0.0f) {
    panel.plotWidthPx = Max(1.0f, ImGui::GetContentRegionAvail().x - statsWidth);
  }
  int const numVisible = Max(1, static_cast<int>(std::ceil(panel.plotWidthPx / panel.barWidthPx)));
  StatsSnapshot const stats = CopySteps(*state.client, panel, numVisible);

  ImGui::BeginChild(
      "##Graph",
      ImVec2(-statsWidth, 0.0f),
      ImGuiChildFlags_None,
      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
  {
    float const graphHeight = ImGui::GetContentRegionAvail().y - ImGui::GetStyle().ScrollbarSize -
        ImGui::GetStyle().ItemSpacing.y;
    BuildGraph(state, ImVec2(-1.0f, Max(1.0f, graphHeight)));
    BuildScrollbar(panel, numVisible);
  }
  ImGui::EndChild();

  ImGui::SameLine();
  ImGui::BeginChild("##Stats");
  BuildStats(state, stats);
  ImGui::EndChild();
}
