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

// clang-format off
#ifdef _WIN32
#include <Windows.h>  // Must come before glad.h
#endif // _WIN32
#ifndef IMGUIOS_USE_METAL
#include <glad/glad.h>
#endif // !IMGUIOS_USE_METAL
#include <GLFW/glfw3.h> // Must come after glad.h
// clang-format on

#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#endif

#include <imgui.h>
#include <imguios/fonts/fonts.h>
#include <implot.h>
#include <misc/cpp/imgui_stdlib.h>

#include <imgui_internal.h>
#include <implot_internal.h>

namespace ImGuios {

// Apply the default ImGuios style with an optional accent color.
void StyleColorsDefault(const ImVec4& accentColor = ImVec4(0.000f, 0.455f, 0.898f, 1.000f));
// A button that acts like a toggle, dimming itself when disabled.
bool ToggleButton(const char* label, bool* toggled, const ImVec2& size = ImVec2(0, 0));
// A toggle switch
bool ToggleSwitch(const char* label, bool* toggled);
// A regular button with a color.
bool ButtonColored(const char* label, const ImVec4& color, const ImVec2& size = ImVec2(0, 0));
// A label and a text value
void LabelTextLeft(const char* label, const char* fmt, ...);
// A label and a text value
void LabelTextLeftV(const char* label, const char* fmt, va_list args);
// Checkbox but aligned with other ImGui widgets
bool CheckboxAligned(const char* label, bool* value);

} // namespace ImGuios
