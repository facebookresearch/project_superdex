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
#include <imguios/imguios_window.h>

#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#endif

namespace ImGuios {

static void glfw_window_close_callback(GLFWwindow* glfwWindow) {
  auto* window = static_cast<Window*>(glfwGetWindowUserPointer(glfwWindow));
  if (window && window->OnCloseCallback != nullptr) {
    bool shouldClose = window->OnCloseCallback();
    if (!shouldClose) {
      glfwSetWindowShouldClose(glfwWindow, GLFW_FALSE);
    }
  }
}

static void glfw_drop_callback(GLFWwindow* glfwWindow, int count, const char** paths) {
  auto* window = static_cast<Window*>(glfwGetWindowUserPointer(glfwWindow));
  if (window && window->OnFileDropCallback != nullptr) {
    std::vector<std::string> pathsVec(count);
    for (int i = 0; i < count; ++i) {
      pathsVec[i] = paths[i];
    }
    window->OnFileDropCallback(pathsVec);
  }
}

static int s_glfwWindowCount = 0;

static void glfw_error_callback(int error, const char* description) {
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

#if defined(__linux__)
// Forward-declare X11 types to avoid #include <X11/Xlib.h> which defines Window/Status macros
// that conflict with C++ identifiers in the ImGuios namespace.
struct _XDisplay;
using XErrorHandlerFunc = int (*)(_XDisplay*, void*);
extern "C" XErrorHandlerFunc XSetErrorHandler(XErrorHandlerFunc);
// Non-fatal X error handler. MIT-SHM BadMatch errors occur with shared GLX contexts on Xvfb
// but are non-fatal — rendering still works. The default handler aborts.
static int x11_error_handler(_XDisplay* /*unused*/, void* /*event*/) {
  fprintf(stderr, "[ImGuiOS] X11 error (non-fatal, suppressed)\n");
  return 0;
}
#endif

static void glfw_key_callback(GLFWwindow* glfwWindow, int key, int scancode, int action, int mods) {
  auto* window = static_cast<Window*>(glfwGetWindowUserPointer(glfwWindow));
  if (window && window->OnKeyCallback) {
    window->OnKeyCallback(window, key, scancode, action, mods);
  }
}

Window::Window(int w, int h, const char* title, WindowFlags flags) {
  if (s_glfwWindowCount == 0) {
#ifdef _WIN32
    // Tell Windows we handle DPI ourselves — prevents blurry bitmap scaling
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
#endif
    glfwSetErrorCallback(glfw_error_callback);
#if defined(IMGUIOS_USE_GLFW_NULL)
    // Headless mode: explicitly request GLFW null platform (no display server)
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_NULL);
#endif
#if defined(__linux__)
    // Install non-fatal X error handler before GLFW opens the X display.
    // Prevents abort on MIT-SHM BadMatch with shared GLX contexts on Xvfb.
    XSetErrorHandler(x11_error_handler);
#endif
    if (!glfwInit()) {
      throw std::runtime_error("Failed to initialize GLFW!");
    }
  }
  s_glfwWindowCount++;

#if defined(IMGUIOS_USE_METAL)
  // Metal backend - no OpenGL context needed
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
#elif defined(__APPLE__)
  // macOS OpenGL backend - use 3.2 core profile
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#elif defined(IMGUI_IMPL_OPENGL_ES2)
  // GL ES 2.0 + GLSL 100
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#else
  // GL 3.0 + GLSL 130
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
  // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

  if ((flags & WindowFlags_NoResize) == WindowFlags_NoResize) {
    glfwWindowHint(GLFW_RESIZABLE, false);
  }
  if ((flags & WindowFlags_NoFocus) == WindowFlags_NoFocus) {
    glfwWindowHint(GLFW_FOCUSED, false);
  }
  if ((flags & WindowFlags_NoDecorations) == WindowFlags_NoDecorations) {
    glfwWindowHint(GLFW_DECORATED, false);
  }
  if ((flags & WindowFlags_Maximized) == WindowFlags_Maximized) {
    glfwWindowHint(GLFW_MAXIMIZED, true);
  }
  if ((flags & WindowFlags_Floating) == WindowFlags_Floating) {
    glfwWindowHint(GLFW_FLOATING, true);
  }
  if ((flags & WindowFlags_Hidden) == WindowFlags_Hidden) {
    glfwWindowHint(GLFW_VISIBLE, false);
  }
  if ((flags & WindowFlags_MSAA) == WindowFlags_MSAA) {
    glfwWindowHint(GLFW_SAMPLES, 4);
  }

  if ((flags & WindowFlags_ScaleToMonitor) == WindowFlags_ScaleToMonitor) {
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
  }

  if ((flags & WindowFlags_FullScreen) == WindowFlags_FullScreen) {
    auto monitor = glfwGetPrimaryMonitor();
    auto* mode = glfwGetVideoMode(monitor);
    glfwWindowHint(GLFW_RED_BITS, mode->redBits);
    glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
    glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
    glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
    _window = glfwCreateWindow(mode->width, mode->height, title, monitor, nullptr);
  } else {
    _window = glfwCreateWindow(w, h, title, nullptr, nullptr);
  }

  if (_window == nullptr) {
    // Headless host (no display / no OpenGL driver): glfwCreateWindow failed. Leave
    // _hasValidContext=false and skip context + callback setup so callers that gate on
    // HasValidGLContext() run GUI-less instead of dereferencing null GL pointers later.
    fprintf(stderr, "Failed to initialize GLFW window! Running without a GUI.\n");
    return;
  }

#if !defined(IMGUIOS_USE_METAL)
  // OpenGL context setup (not needed for Metal)
  glfwMakeContextCurrent(_window);
  glfwSwapInterval(1);
  if (gladLoadGLLoader((GLADloadproc)glfwGetProcAddress) == 0) {
    // GL loader failed (no usable driver) — treat as headless: skip the rest so GL
    // function pointers are never called through null.
    fprintf(stderr, "Failed to initialize OpenGL loader! Running without a GUI.\n");
    return;
  }
#endif

  _hasValidContext = true;
  glfwSetWindowUserPointer(_window, this);
  glfwSetWindowCloseCallback(_window, glfw_window_close_callback);
  glfwSetDropCallback(_window, glfw_drop_callback);
  glfwSetKeyCallback(_window, glfw_key_callback);
}

Window::~Window() {
  glfwDestroyWindow(_window);
  _window = nullptr;
  s_glfwWindowCount--;
  if (s_glfwWindowCount == 0) {
    glfwTerminate();
  }
}

void Window::SetTitle(const char* title) {
  glfwSetWindowTitle(_window, title);
}

void Window::EnableDecorations(bool enable) {
  glfwSetWindowAttrib(_window, GLFW_DECORATED, enable ? GLFW_TRUE : GLFW_FALSE);
}

void Window::EnableResize(bool enable) {
  glfwSetWindowAttrib(_window, GLFW_RESIZABLE, enable ? GLFW_TRUE : GLFW_FALSE);
}

void Window::Focus() {
  glfwFocusWindow(_window);
}

void Window::Hide() {
  glfwHideWindow(_window);
}

void Window::Show() {
  glfwShowWindow(_window);
}

void Window::Minimize() {
  glfwIconifyWindow(_window);
}
void Window::Maximize() {
  glfwMaximizeWindow(_window);
}
void Window::Restore() {
  glfwRestoreWindow(_window);
}

void Window::Close() {
  glfwSetWindowShouldClose(_window, true);
}

ImVec2 Window::GetPos() const {
  int x, y;
  glfwGetWindowPos(_window, &x, &y);
  return ImVec2(x, y);
}

void Window::SetPos(const ImVec2& pos) {
  glfwSetWindowPos(_window, (int)pos.x, (int)pos.y);
}

ImVec2 Window::GetSize() const {
  int w, h;
  glfwGetWindowSize(_window, &w, &h);
  return ImVec2(w, h);
}

void Window::SetSize(const ImVec2& pos) {
  glfwSetWindowSize(_window, (int)pos.x, (int)pos.y);
}

void Window::CenterOnPrimaryMonitor() {
  GLFWmonitor* primaryDisplay = glfwGetPrimaryMonitor();

  int primaryMonitorOriginX = 0, primaryMonitorOriginY = 0;
  int primaryMonitorSizeX = 0, primaryMonitorSizeY = 0;

  glfwGetMonitorWorkarea(
      primaryDisplay,
      &primaryMonitorOriginX,
      &primaryMonitorOriginY,
      &primaryMonitorSizeX,
      &primaryMonitorSizeY);

  const ImVec2 size = GetSize();
  SetPos(ImVec2(
      static_cast<float>(primaryMonitorOriginX + (primaryMonitorSizeX - size.x) / 2),
      static_cast<float>(primaryMonitorOriginY + (primaryMonitorSizeY - size.y) / 2)));
}

namespace {

// The monitor whose area contains the center of the given window rect, or the primary monitor if
// the window straddles no monitor (e.g. dragged partly off-screen).
GLFWmonitor* MonitorForWindowRect(const ImVec2& pos, const ImVec2& size) {
  const float centerX = pos.x + size.x * 0.5f;
  const float centerY = pos.y + size.y * 0.5f;

  int monitorCount = 0;
  GLFWmonitor** monitors = glfwGetMonitors(&monitorCount);
  for (int i = 0; i < monitorCount; ++i) {
    int mx = 0, my = 0;
    glfwGetMonitorPos(monitors[i], &mx, &my);
    const GLFWvidmode* mode = glfwGetVideoMode(monitors[i]);
    if (mode == nullptr) {
      continue;
    }
    if (centerX >= mx && centerX < mx + mode->width && centerY >= my &&
        centerY < my + mode->height) {
      return monitors[i];
    }
  }
  return glfwGetPrimaryMonitor();
}

} // namespace

void Window::SetBorderlessFullScreen(bool enable) {
  if (!_hasValidContext || enable == _borderlessFullScreen) {
    return;
  }

  if (enable) {
    _preBorderlessMaximized = IsMaximized();
    if (_preBorderlessMaximized) {
      // Restore first so the saved rect is the windowed one rather than the maximized one.
      Restore();
    }
    _preBorderlessPos = GetPos();
    _preBorderlessSize = GetSize();

    GLFWmonitor* monitor = MonitorForWindowRect(_preBorderlessPos, _preBorderlessSize);
    const GLFWvidmode* mode = monitor != nullptr ? glfwGetVideoMode(monitor) : nullptr;
    if (mode == nullptr) {
      return;
    }
    int mx = 0, my = 0;
    glfwGetMonitorPos(monitor, &mx, &my);

    EnableDecorations(false);
    SetPos(ImVec2(static_cast<float>(mx), static_cast<float>(my)));
    SetSize(ImVec2(static_cast<float>(mode->width), static_cast<float>(mode->height)));
  } else {
    EnableDecorations(true);
    SetPos(_preBorderlessPos);
    SetSize(_preBorderlessSize);
    if (_preBorderlessMaximized) {
      Maximize();
    }
  }
  _borderlessFullScreen = enable;
}

bool Window::IsHidden() const {
  return glfwGetWindowAttrib(_window, GLFW_VISIBLE) == GLFW_FALSE;
}

bool Window::IsMinimized() const {
  return glfwGetWindowAttrib(_window, GLFW_ICONIFIED) == GLFW_TRUE;
}

bool Window::IsMaximized() const {
  return glfwGetWindowAttrib(_window, GLFW_MAXIMIZED) == GLFW_TRUE;
}

float Window::GetContentScale() const {
  float xscale = 1.0f;
  float yscale = 1.0f;
  glfwGetWindowContentScale(_window, &xscale, &yscale);
  return xscale;
}

GLFWwindow* Window::GetGLFW() const {
  return _window;
}

} // namespace ImGuios
