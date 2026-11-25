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

#include <string>

namespace mochi::dbg {
class AutoCompleter;
struct TerminalPanelState;

// Draw the terminal's command line, with the prompt inside the box, history on Up and Down, and Tab
// completion when completer is not null. Returns true when the user submits the input with Enter.
bool BuildCommandInput(
    TerminalPanelState& terminal,
    AutoCompleter* completer,
    std::string const& prompt);
} // namespace mochi::dbg
