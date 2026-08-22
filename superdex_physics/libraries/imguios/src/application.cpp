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

#include <imguios/application.h>

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_internal.h"

#include "application_mcp.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>

#ifdef IMGUI_ENABLE_TEST_ENGINE
#include "imgui_mcp_bridge.h"
#endif

#ifdef _WIN32
#include <windows.h>
#endif

namespace ImGuios {

Application::Application(std::unique_ptr<Window> mainWindow, McpConfig mcpConfig)
    : _mcpConfig(mcpConfig) {
  // set main window
  if (mainWindow == nullptr) {
    throw std::runtime_error("Main window cannot be null!");
  }
  _mainWindow = std::move(mainWindow);

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
  // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // Enable Docking
  io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable Multi-Viewport / Platform Windows
  // io.ConfigViewportsNoAutoMerge = true;
  // io.ConfigViewportsNoTaskBarIcon = true;

  // Setup Dear ImGui style
  ImGuios::StyleColorsDefault();

  // When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look
  // identical to regular ones.
  ImGuiStyle& style = ImGui::GetStyle();
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
    style.WindowRounding = 0.0f;
    style.Colors[ImGuiCol_WindowBg].w = 1.0f;
  }

#if defined(IMGUI_IMPL_OPENGL_ES2)
  // GL ES 2.0 + GLSL 100
  const char* glsl_version = "#version 100";
#elif defined(__APPLE__)
  // GL 3.2 + GLSL 150
  const char* glsl_version = "#version 150";
#else
  // GL 3.0 + GLSL 130
  const char* glsl_version = "#version 130";
#endif

  // Setup Platform/Renderer backends. On a headless host the main window has no GL
  // context; skip all GL init so we never call through null glad function pointers. The
  // app still constructs (and Run() no-ops) so callers don't have to special-case it.
  if (_mainWindow->HasValidGLContext()) {
    // Query DPI scale for HiDPI support
    float xscale = 1.0f, yscale = 1.0f;
    glfwGetWindowContentScale(_mainWindow->_window, &xscale, &yscale);
    _dpiScale = xscale;

    ImGui_ImplGlfw_InitForOpenGL(_mainWindow->_window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // enable MSAA
    glEnable(GL_MULTISAMPLE);
  }

  // add fonts
  io.Fonts->Clear();

  using namespace ImGuios::Fonts;
  LoadFont("Roboto Regular", 16, Roboto_Regular_ttf, Roboto_Regular_ttf_len);
  LoadFont("Roboto Bold", 16, Roboto_Bold_ttf, Roboto_Bold_ttf_len);
  LoadFont("Roboto Italic", 16, Roboto_Italic_ttf, Roboto_Italic_ttf_len);
  LoadFont("Roboto Mono Regular", 16, RobotoMono_Regular_ttf, RobotoMono_Regular_ttf_len);
  LoadFont("Roboto Mono Bold", 16, RobotoMono_Bold_ttf, RobotoMono_Bold_ttf_len);
  LoadFont("Roboto Mono Italic", 16, RobotoMono_Italic_ttf, RobotoMono_Italic_ttf_len);

  // Font/style DPI compensation differs by platform because of how GLFW reports coordinates:
  // - macOS: ImGui works in logical points (screen coords ≠ framebuffer pixels).
  //   Fonts rasterized at dpiScale * fontSize are oversized in logical coords, so
  //   FontGlobalScale compensates.
  // - Windows/Linux: ImGui works in physical pixels (screen coords == framebuffer pixels
  //   when DPI-aware). Fonts are already the correct physical size. Instead, scale all
  //   style sizes (padding, scrollbar, etc.) so UI elements match the display DPI.
#ifdef __APPLE__
  io.FontGlobalScale = 1.0f / _dpiScale;
#else
  if (_dpiScale > 1.0f) {
    style.ScaleAllSizes(_dpiScale);
  }
#endif
}

static void NormalizeString(std::string& str) {
  str.erase(
      std::remove_if(str.begin(), str.end(), [](unsigned char c) { return std::isspace(c); }),
      str.end());
  std::transform(
      str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::tolower(c); });
}

ImFont* Application::GetFont(const char* name) {
  std::string keyName = name;
  NormalizeString(keyName);
  if (_fonts.count(keyName)) {
    return _fonts[keyName];
  }
  return nullptr;
}

ImFont* Application::LoadFont(const char* name, float fontSize, void* fontData, int fontDataSize) {
  float const scaledSize = fontSize * _dpiScale;
  std::string keyName = name;
  NormalizeString(keyName);
  ImGuiIO& io = ImGui::GetIO();
  ImFontConfig font_cfg;
  font_cfg.FontDataOwnedByAtlas = false;
  ImFontConfig icons_config;
  icons_config.MergeMode = true;
  icons_config.PixelSnapH = true;
  icons_config.GlyphOffset = ImVec2(0, 0);
  icons_config.OversampleH = 1;
  icons_config.OversampleV = 1;
  icons_config.FontDataOwnedByAtlas = false;
  static constexpr auto fa_ranges = std::to_array<ImWchar>({ICON_MIN_FA, ICON_MAX_FA, 0});
  ImStrncpy(font_cfg.Name, name, 40);
  auto font = io.Fonts->AddFontFromMemoryTTF(fontData, fontDataSize, scaledSize, &font_cfg);
  _fonts[keyName] = font;
  io.Fonts->AddFontFromMemoryTTF(
      Fonts::fa_solid_900_ttf,
      Fonts::fa_solid_900_ttf_len,
      scaledSize,
      &icons_config,
      fa_ranges.data());
  return font;
}

Window* Application::GetMainWindow() {
  return _mainWindow.get();
}

Application::~Application() {
  if (_running) {
    Stop();
  }
  // The GL/GLFW backends were only initialized when we had a GL context (see ctor);
  // shutting them down otherwise asserts on null backend data.
  if (_mainWindow && _mainWindow->HasValidGLContext()) {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
  }
  ImPlot::DestroyContext();
  ImGui::DestroyContext();
  McpDestroyContext();
  _mainWindow.reset();
}

void Application::Run() {
  _running = true;
  if (!_mainWindow->HasValidGLContext()) {
    // Headless: no GL context. Skip the render loop — running it would call through null
    // GL function pointers (crash), and with a null window glfwWindowShouldClose() can't
    // terminate the loop (hang). The app is a no-op GUI.
    _running = false;
    return;
  }
  OnInitialize();
  McpInitialize();
  ImGuiIO& io = ImGui::GetIO();
  while (!glfwWindowShouldClose(_mainWindow->_window) && _running) {
    glfwPollEvents();
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    OnUpdate();
    // Rendering
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(_mainWindow->_window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    OnMainWindowRender();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
      GLFWwindow* backup_current_context = glfwGetCurrentContext();
      ImGui::UpdatePlatformWindows();
      ImGui::RenderPlatformWindowsDefault();
      glfwMakeContextCurrent(backup_current_context);
    }
    glfwSwapBuffers(_mainWindow->_window);
    OnPostSwap();
    McpPostSwap();
  }
  OnShutdown();
  McpShutdown();
  _running = false;
}

void Application::OnMainWindowRender() {
  // Default implementation of OnRender. User may override it.
  glClearColor(0, 0, 0, 1);
  glClear(GL_COLOR_BUFFER_BIT);
}

bool Application::IsRunning() const {
  return _running;
}

void Application::Stop() {
  _running = false;
}

//-----------------------------------------------------------------------------
// MCP Bridge Integration
//-----------------------------------------------------------------------------

#ifdef IMGUI_ENABLE_TEST_ENGINE

void Application::McpPostSwap() {
  if (!_mcpState) {
    return;
  }

  ImGuiTestEngine_PostSwap(_mcpState->testEngine);

  // ImGuiMcpBridge_Tick() returns true only when a screenshot request is
  // pending — not every frame while MCP is connected.
  if (ImGuiMcpBridge_Tick()) {
    // Snapshot the framebuffer for the capture callback
    int w, h;
    glfwGetFramebufferSize(_mainWindow->_window, &w, &h);
    if (w > 0 && h > 0) {
      std::vector<uint8_t> pixels(static_cast<size_t>(w) * h * 4);
      glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

      // OpenGL reads bottom-to-top; flip vertically
      std::vector<uint8_t> flipped(static_cast<size_t>(w) * h * 4);
      for (int y = 0; y < h; y++) {
        memcpy(
            flipped.data() + static_cast<size_t>(y) * w * 4,
            pixels.data() + static_cast<size_t>(h - 1 - y) * w * 4,
            static_cast<size_t>(w) * 4);
      }

      std::lock_guard<std::mutex> lock(_mcpState->cachedFb.mutex);
      _mcpState->cachedFb.pixels = std::move(flipped);
      _mcpState->cachedFb.width = w;
      _mcpState->cachedFb.height = h;
    }
  }

  if (ImGuiMcpBridge_WantsShutdown()) {
    Stop();
  }
}

#else // !IMGUI_ENABLE_TEST_ENGINE

void Application::McpPostSwap() {}

#endif // IMGUI_ENABLE_TEST_ENGINE

} // namespace ImGuios
