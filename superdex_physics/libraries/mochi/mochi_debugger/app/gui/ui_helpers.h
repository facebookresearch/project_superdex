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

#include <mochi_core/utils/color.h>

namespace mochi::dbg {

// Shared ImGui widget helpers used across panels.

// A button with a hover tooltip. When disabled it is greyed out and never reports a click.
bool UiButton(char const* label, char const* tooltip, bool enabled = true);

// A checkbox with an optional hover tooltip.
bool UiCheckbox(char const* label, bool* value, char const* tooltip = "", bool enabled = true);

// An RGBA color editor (four 0-255 drag fields plus a preview swatch).
bool UiColorPicker(char const* label, Color* color);

} // namespace mochi::dbg
