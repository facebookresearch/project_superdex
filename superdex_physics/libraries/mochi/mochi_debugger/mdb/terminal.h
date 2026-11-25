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

#include <mochi_core/mochi_platform.h>
#include <mochi_core/utils/color.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/guarded.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace mochi::dbg {

/**
 * @brief A small cross-platform, readline-style terminal with a pinned input prompt.
 *
 * @note All public methods are thread-safe.
 *
 * @warning Owns exclusive raw-mode state for the process's console (stdin/stdout), so at most one
 * @ref Terminal may exist per process. It is therefore neither copyable nor movable.
 */
class Terminal final {
 public:
  Terminal();
  ~Terminal();

  Terminal(Terminal const&) = delete;
  Terminal& operator=(Terminal const&) = delete;
  Terminal(Terminal&&) = delete;
  Terminal& operator=(Terminal&&) = delete;

  /**
   * @brief Print text above the prompt, then redraw the prompt and current input buffer.
   *
   * @details The caller controls line breaks: include a trailing "\n" in @p text to start the
   * prompt on a fresh line below the message. The prompt and any in-progress input are always
   * redrawn immediately after @p text.
   *
   * @param[in] text The text to print.
   * @param[in] color Optional foreground color. Uses the terminal's default if not specified.
   */
  void Print(std::string_view text, std::optional<Color> color = std::nullopt);

  /**
   * @brief Set the input prompt text and redraw it.
   *
   * @param[in] prompt The new prompt text.
   */
  void SetPrompt(std::string_view prompt);

  /** @brief Return the terminal width in character cells, or zero when unavailable. */
  [[nodiscard]] size_t GetColumnCount() const;

  /** @brief Receives the input line after every edit and cursor movement. */
  using InputFn = std::function<void(std::string_view text, bool isCursorAtEnd)>;

  /** @brief Receives Tab, or Shift+Tab when backward is true, and returns a new input line. */
  using TabFn = std::function<std::optional<std::string>(bool backward)>;

  /**
   * @brief Set the functions that implement Tab completion. Call before @ref ReadLine.
   *
   * @details onInput also receives the line that onTab returns, and an empty line when
   * @ref ReadLine starts. Both run on the thread that calls @ref ReadLine, without the terminal's
   * lock, so they may call @ref SetSuggestion and @ref Print.
   */
  void SetCompletion(InputFn onInput, TabFn onTab);

  /**
   * @brief Show a completion in faint text after the input while the cursor is at its end.
   *
   * @param[in] text The input line the completion is for. It shows only while the input matches.
   * @param[in] preview The text that Tab inserts.
   * @param[in] annotation Information shown after the preview. It is dropped first when the line is
   * too narrow for both.
   */
  void SetSuggestion(std::string_view text, std::string_view preview, std::string_view annotation);

  /**
   * @brief Block until the user submits a line (presses Enter).
   *
   * @details Submitted non-empty lines are appended to the command history; the user can cycle
   * back through previous commands with the Up and Down arrow keys, as in a standard terminal.
   * Left/Right move by character, Ctrl/Alt+Left/Right and Alt+B/F move by word, and Home/End
   * or Ctrl+A/E move to line boundaries. Typing, Backspace and Delete edit at the cursor.
   *
   * @param[out] line The submitted line (without the trailing newline).
   * @return False if input was closed (EOF / error), true otherwise.
   */
  bool ReadLine(std::string& line);

 private:
  /** @brief Prompt, editable line and history state, guarded by @ref _state. */
  struct State {
    std::string buffer; // current (in-progress) input line
    std::size_t cursorPos{0};
    std::string prompt; // text shown before the input buffer
    DynamicArray<std::string> history; // submitted commands, oldest first
    std::string stashedInput; // live input saved while browsing history
    std::size_t stashedCursorPos{0};
    std::size_t historyPos{0}; // browse cursor; == history.size() means the live line
    std::string suggestionText; // the input that suggestion and annotation are for
    std::string suggestion; // completion preview shown after the input
    std::string annotation; // shown after the suggestion
    bool rawModeEnabled{false}; // whether the terminal was switched into raw mode
  };

  // Clears the current line and redraws the prompt followed by the input buffer, and the
  // suggestion while it matches the input and the cursor is at the end.
  // All helpers below assume the caller holds the state lock (i.e. run inside a Mutate/Read).
  void Redraw(State const& state) const;

  // Replaces the buffer with the previous (older) history entry, if any.
  void HistoryUp(State& state) const;

  // Replaces the buffer with the next (newer) history entry, or the stashed live line.
  void HistoryDown(State& state) const;

  struct InputReader;
  std::unique_ptr<InputReader> _input;
  // Escape sequence that starts the faint style of suggestions
  std::string_view const _suggestionStyle;
  InputFn _onInput;
  TabFn _onTab;
  Guarded<State> _state;
};

} // namespace mochi::dbg
