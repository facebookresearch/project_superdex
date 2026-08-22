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

// Shared MCP bridge integration code for all rendering backends.
// Each backend implements McpPostSwap() with its own framebuffer readback.

#include <imguios/application.h>

#include "application_mcp.h"

#ifdef IMGUI_ENABLE_TEST_ENGINE
#include "imgui_mcp_bridge.h"
#endif

namespace ImGuios {

#ifdef IMGUI_ENABLE_TEST_ENGINE

void Application::McpInitialize() {
  if (!_mcpConfig.enabled) {
    return;
  }

  _mcpState = std::make_unique<McpState>();

  _mcpState->testEngine = ImGuiTestEngine_CreateContext();
  ImGuiTestEngineIO& test_io = ImGuiTestEngine_GetIO(_mcpState->testEngine);
  test_io.ConfigVerboseLevel = ImGuiTestVerboseLevel_Info;
  test_io.ConfigRunSpeed = ImGuiTestRunSpeed_Fast;
  test_io.ConfigWatchdogWarning = 10.0f;
  test_io.ConfigWatchdogKillTest = 15.0f;
  test_io.ConfigBreakOnError = false;
  ImGuiTestEngine_Start(_mcpState->testEngine, ImGui::GetCurrentContext());
  ImGuiTestEngine_InstallDefaultCrashHandler();

  ImGuiMcpBridgeConfig bridge_config;
  bridge_config.Port = _mcpConfig.port;
  bridge_config.TestEngine = _mcpState->testEngine;
  bridge_config.CaptureFunc =
      [](int x, int y, int w, int h, unsigned int* pixels, void* user_data) -> bool {
    auto* state = static_cast<McpState*>(user_data);
    std::lock_guard<std::mutex> lock(state->cachedFb.mutex);
    if (state->cachedFb.pixels.empty()) {
      return false;
    }

    int fb_w = state->cachedFb.width;
    int fb_h = state->cachedFb.height;
    if (x < 0) {
      x = 0;
    }
    if (y < 0) {
      y = 0;
    }
    if (x + w > fb_w) {
      w = fb_w - x;
    }
    if (y + h > fb_h) {
      h = fb_h - y;
    }
    if (w <= 0 || h <= 0) {
      return false;
    }

    for (int row = 0; row < h; row++) {
      uint8_t const* src =
          state->cachedFb.pixels.data() + (static_cast<size_t>(y + row) * fb_w + x) * 4;
      unsigned int* dst = pixels + row * w;
      memcpy(dst, src, static_cast<size_t>(w) * 4);
    }

    return true;
  };
  bridge_config.CaptureUserData = _mcpState.get();
  ImGuiMcpBridge_Init(bridge_config);
}

void Application::McpShutdown() {
  if (!_mcpState) {
    return;
  }
  ImGuiMcpBridge_Shutdown();
  ImGuiTestEngine_Stop(_mcpState->testEngine);
}

void Application::McpDestroyContext() {
  if (!_mcpState) {
    return;
  }
  ImGuiTestEngine_DestroyContext(_mcpState->testEngine);
  _mcpState.reset();
}

#else // !IMGUI_ENABLE_TEST_ENGINE

void Application::McpInitialize() {}
void Application::McpShutdown() {}
void Application::McpDestroyContext() {}

#endif // IMGUI_ENABLE_TEST_ENGINE

} // namespace ImGuios
