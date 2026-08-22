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
#include <imguios/common.h>
#include <functional>
#include <string>
#include <vector>

namespace ImGuios {

class Application;

using WindowFlags = int;
enum WindowFlags_ {
  WindowFlags_None = 0,
  WindowFlags_NoResize = 1 << 0,
  WindowFlags_NoFocus = 1 << 1,
  WindowFlags_NoDecorations = 1 << 2,
  WindowFlags_FullScreen = 1 << 3,
  WindowFlags_Maximized = 1 << 4,
  WindowFlags_Floating = 1 << 5,
  WindowFlags_Hidden = 1 << 6,
  WindowFlags_MSAA = 1 << 7,
  WindowFlags_ScaleToMonitor = 1 << 8
};

class Window {
 public:
  Window(int w, int h, const char* title, WindowFlags flags = 0);
  ~Window();

  // Set the title of the window.
  void SetTitle(const char* title);
  // Enable/disable decorations on the window.
  void EnableDecorations(bool enabled);
  // Enable/disable window resizing.
  void EnableResize(bool enabled);
  // Focus the window, i.e. bring to foreground.
  void Focus();
  // Show the window if it was previously hidden.
  void Show();
  // Hide the window, i.e. close without actually closing.
  void Hide();
  // Minimize the window.
  void Minimize();
  // Maximize the window.
  void Maximize();
  // Restore a minimized or maximized window.
  void Restore();
  // Forcefully close the window.
  void Close();

  // Get the window position in pixels.
  [[nodiscard]] ImVec2 GetPos() const;
  // Set the window position in pixels.
  void SetPos(const ImVec2& pos);
  // Get the window size in pixels.
  [[nodiscard]] ImVec2 GetSize() const;
  // Set the window size in pixels.
  void SetSize(const ImVec2& pos);
  // Center the window on the primary display
  void CenterOnPrimaryMonitor();

  // Enable/disable borderless fullscreen: an undecorated window covering the whole monitor it
  // currently sits on. Unlike WindowFlags_FullScreen this does not take exclusive control of the
  // display, so alt-tabbing and overlays keep working. Disabling restores the previous position,
  // size and maximized state. No-op on a headless host.
  void SetBorderlessFullScreen(bool enable);
  // Returns true if the window is in borderless fullscreen.
  [[nodiscard]] bool IsBorderlessFullScreen() const {
    return _borderlessFullScreen;
  }

  // Returns true if the window is hidden.
  [[nodiscard]] bool IsHidden() const;
  // Returns true if the window is minimized.
  [[nodiscard]] bool IsMinimized() const;
  // Returns true if the window is maximized.
  [[nodiscard]] bool IsMaximized() const;

  // Returns the content scale factor for the monitor the window is on.
  [[nodiscard]] float GetContentScale() const;

  // Returns access to the underlying GLFW window
  [[nodiscard]] GLFWwindow* GetGLFW() const;

  // Returns true if a real window and rendering context were created. False on a
  // headless host (no display / no OpenGL driver), where glfwCreateWindow or the GL
  // loader fails. Callers MUST gate all GL/rendering on this — otherwise they call
  // through null GL function pointers and crash.
  [[nodiscard]] bool HasValidGLContext() const {
    return _hasValidContext;
  }

 public:
  // Called when the window is closed. Return true to allow closure, false to keep open.
  std::function<bool()> OnCloseCallback;
  // Called when one or more files are dropped onto the window. A vector of filepaths is passed.
  std::function<void(const std::vector<std::string>&)> OnFileDropCallback;
  // Called for each keyboard event
  std::function<void(Window* window, int key, int scancode, int action, int mods)> OnKeyCallback;

 protected:
  GLFWwindow* _window = nullptr;
  bool _hasValidContext = false;
  bool _borderlessFullScreen = false;
  // Window rect and maximized state captured when entering borderless fullscreen.
  ImVec2 _preBorderlessPos{};
  ImVec2 _preBorderlessSize{};
  bool _preBorderlessMaximized = false;

 private:
  friend class Application;
};

} // namespace ImGuios
