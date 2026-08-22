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
#include <imguios/imguios_window.h>
#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ImGuios {

struct McpConfig {
  int port = 18086;
  bool enabled = false;
};

class Application {
 public:
  // Constructor.
  Application(std::unique_ptr<Window> mainWindow, McpConfig mcpConfig = {});
  // Destructor.
  virtual ~Application();

  // Called at top of Run.
  virtual void OnInitialize() { /* implement me */ }
  // Called once per frame in Run. Call your ImGui commands here.
  virtual void OnUpdate() { /* implement me */ }
  // Called once per frame after OnUpdate. Call your rendering commands here.
  // The default implementation simply clears the main window to black (OpenGL) or does nothing
  // (Metal).
  virtual void OnMainWindowRender();
  // Called once per frame after swap/present. Used by test engine integration.
  virtual void OnPostSwap() { /* implement me */ }
  // Called at the bottom of Run.
  virtual void OnShutdown() { /* implement me */ }

  // Runs the application.
  void Run();
  // Stops the application.
  void Stop();

  // Load a new font from memory.
  ImFont* LoadFont(const char* name, float fontSize, void* fontData, int fontDataSize);
  // Get a font by name.
  ImFont* GetFont(const char* name);

  // Get the main window.
  Window* GetMainWindow();

  // Is the application running?
  [[nodiscard]] bool IsRunning() const;

  // Returns the DPI scale factor (e.g., 2.0 on Retina displays).
  [[nodiscard]] float GetDpiScale() const {
    return _dpiScale;
  }

 private:
  float _dpiScale = 1.0f;

  friend class Window;
  std::unique_ptr<Window> _mainWindow;
  std::map<std::string, ImFont*> _fonts;
  std::atomic_bool _running = false;

  // MCP bridge support (opt-in via McpConfig)
  McpConfig _mcpConfig;
  struct McpState;
  std::unique_ptr<McpState> _mcpState;
  void McpInitialize();
  void McpPostSwap();
  void McpShutdown();
  void McpDestroyContext();
};

#if defined(__APPLE__)
// Metal-specific accessors (macOS only)
// Returns the MTLDevice used for rendering (id<MTLDevice>)
void* GetMetalDevice();
// Returns the MTLCommandQueue used for rendering (id<MTLCommandQueue>)
void* GetMetalCommandQueue();
// Returns the CAMetalLayer used for rendering
void* GetMetalLayer();
#endif

} // namespace ImGuios
