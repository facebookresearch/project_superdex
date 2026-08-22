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

#include <mochi_core/mochi_platform.h>

#include "native_gl_context.h"

// This is the ONLY translation unit that includes <GLFW/glfw3native.h>. On Linux the GLX path pulls
// in <X11/X.h>, which #defines generic tokens (None, Bool, Status, ...). Keeping this file free of
// mochi / Filament / STL headers means those macros have nothing to collide with, so no include
// ordering or #undef gymnastics are needed here.
#if MOCHI_PLATFORM_WINDOWS
#ifndef GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#endif
#ifndef GLFW_EXPOSE_NATIVE_WGL
#define GLFW_EXPOSE_NATIVE_WGL
#endif
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#elif MOCHI_PLATFORM_LINUX
#ifndef GLFW_EXPOSE_NATIVE_GLX
#define GLFW_EXPOSE_NATIVE_GLX
#endif
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#endif

namespace mochi::dbg {

void* GetNativeGlContext([[maybe_unused]] GLFWwindow* window) {
#if MOCHI_PLATFORM_WINDOWS
  return reinterpret_cast<void*>(glfwGetWGLContext(window));
#elif MOCHI_PLATFORM_LINUX
  return reinterpret_cast<void*>(glfwGetGLXContext(window));
#else
  return nullptr;
#endif
}

} // namespace mochi::dbg
