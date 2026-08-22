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

// Forward declaration so consumers don't need any GLFW header.
struct GLFWwindow;

namespace mochi::dbg {

// Returns the platform-native GL context (WGL on Windows, GLX on Linux) for @p window as an opaque
// handle suitable for Filament's Engine sharedContext. Returns nullptr on platforms without a
// native GL context (e.g. macOS, which uses Metal).
//
// This is deliberately isolated in its own translation unit (native_gl_context.cpp) so that the one
// file pulling in <GLFW/glfw3native.h> -- which on Linux drags in <X11/X.h> and its generic `None`,
// `Bool`, `Status`, ... macros -- has no other headers for those macros to clobber. It keeps that
// macro blast radius out of main.cpp and the mochi/Filament headers.
void* GetNativeGlContext(GLFWwindow* window);

} // namespace mochi::dbg
