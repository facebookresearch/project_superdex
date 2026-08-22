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

#include "terminal.h"

#include <mochi_core/mochi_platform.h>

#include <cstdio>

#if MOCHI_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <termios.h>
#include <unistd.h>
#endif

namespace mochi::dbg {

// VT escape: carriage-return + erase-to-end-of-line. Used to overwrite the current input line.
static constexpr std::string_view kClearLine = "\r\x1b[K";

// VT escape: reset all text attributes (including foreground color) back to the terminal default.
static constexpr std::string_view kColorReset = "\x1b[0m";

// Writes text to stdout without flushing. Callers flush once per logical operation (see Flush).
static void WriteRaw(std::string_view text) {
  if (!text.empty()) {
    std::fwrite(text.data(), 1, text.size(), stdout);
  }
}

// Flushes buffered stdout so pending output becomes visible.
static void Flush() {
  std::fflush(stdout);
}

// Emits a 24-bit (truecolor) foreground escape for the given RGBA color (alpha is ignored).
static void WriteColorOn(Color const& color) {
  char buf[32];
  int const n = std::snprintf(
      buf,
      sizeof(buf),
      "\x1b[38;2;%u;%u;%um",
      static_cast<unsigned>(color[0]),
      static_cast<unsigned>(color[1]),
      static_cast<unsigned>(color[2]));
  if (n > 0) {
    WriteRaw(std::string_view(buf, static_cast<size_t>(n)));
  }
}

#if MOCHI_PLATFORM_WINDOWS

// Original console modes, saved by EnableRawMode and restored by DisableRawMode. Stored as process
// globals because the console is a per-process resource; only one Terminal may exist at a time.
static DWORD g_originalInputMode = 0;
static DWORD g_originalOutputMode = 0;

static bool EnableRawMode() {
  HANDLE const in = GetStdHandle(STD_INPUT_HANDLE);
  HANDLE const out = GetStdHandle(STD_OUTPUT_HANDLE);
  if (in == INVALID_HANDLE_VALUE || out == INVALID_HANDLE_VALUE) {
    return false;
  }
  if (!GetConsoleMode(in, &g_originalInputMode) || !GetConsoleMode(out, &g_originalOutputMode)) {
    return false;
  }
  // Disable line buffering and echo, and enable VT input so arrow keys arrive as the same escape
  // sequences (ESC [ A/B/...) that POSIX terminals emit.
  DWORD const inMode = (g_originalInputMode & ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT)) |
      ENABLE_VIRTUAL_TERMINAL_INPUT;
  if (!SetConsoleMode(in, inMode)) {
    return false;
  }
  if (!SetConsoleMode(out, g_originalOutputMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
    // Roll back the input mode we just changed so a partial failure doesn't leave the console
    // altered.
    SetConsoleMode(in, g_originalInputMode);
    return false;
  }
  return true;
}

static void DisableRawMode() {
  SetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), g_originalInputMode);
  SetConsoleMode(GetStdHandle(STD_OUTPUT_HANDLE), g_originalOutputMode);
}

// Read a single character, blocking. Returns -1 on EOF/error.
static int ReadCharBlocking() {
  HANDLE const in = GetStdHandle(STD_INPUT_HANDLE);
  char c = 0;
  DWORD read = 0;
  if (!ReadFile(in, &c, 1, &read, nullptr) || read == 0) {
    return -1;
  }
  return static_cast<unsigned char>(c);
}

#else

// Original terminal attributes, saved by EnableRawMode and restored by DisableRawMode. A process
// global because the controlling terminal is a per-process resource; only one Terminal may exist.
static termios g_originalTermios{};

static bool EnableRawMode() {
  if (tcgetattr(STDIN_FILENO, &g_originalTermios) != 0) {
    return false;
  }
  termios raw = g_originalTermios;
  raw.c_lflag &= ~(static_cast<tcflag_t>(ICANON | ECHO));
  raw.c_cc[VMIN] = 1;
  raw.c_cc[VTIME] = 0;
  return tcsetattr(STDIN_FILENO, TCSANOW, &raw) == 0;
}

static void DisableRawMode() {
  tcsetattr(STDIN_FILENO, TCSANOW, &g_originalTermios);
}

// Read a single character, blocking. Returns -1 on EOF/error.
static int ReadCharBlocking() {
  unsigned char c = 0;
  ssize_t const n = read(STDIN_FILENO, &c, 1);
  if (n <= 0) {
    return -1;
  }
  return c;
}

#endif

namespace {

// A single decoded keypress. Multi-byte escape sequences (arrow keys) are collapsed into one Key.
enum class KeyType {
  Eof, // input closed (EOF / error)
  Enter, // submit the current line
  Backspace, // delete the last character
  Up, // previous (older) history entry
  Down, // next (newer) history entry
  Printable, // a printable ASCII character (see Key::ch)
  Ignored, // a recognized-but-unhandled key (e.g. left/right arrow, stray control char)
};

struct Key {
  KeyType type{KeyType::Ignored};
  char ch{0};
};

} // namespace

// Reads and decodes one keypress, blocking. Performs no locking, so async output (Print) may
// run between keystrokes. Arrow keys arrive as a CSI sequence: ESC '[' followed by 'A'/'B'/'C'/'D'.
static Key ReadKey() {
  int const c = ReadCharBlocking();
  if (c < 0) {
    return {KeyType::Eof, 0};
  }
  if (c == '\r' || c == '\n') {
    return {KeyType::Enter, 0};
  }
  if (c == 127 || c == 8) { // DEL / backspace
    return {KeyType::Backspace, 0};
  }
  if (c == 27) { // ESC: possibly a CSI arrow sequence
    int const c1 = ReadCharBlocking();
    if (c1 < 0) {
      return {KeyType::Eof, 0};
    }
    if (c1 == '[') {
      int const c2 = ReadCharBlocking();
      if (c2 < 0) {
        return {KeyType::Eof, 0};
      }
      if (c2 == 'A') {
        return {KeyType::Up, 0};
      }
      if (c2 == 'B') {
        return {KeyType::Down, 0};
      }
    }
    return {KeyType::Ignored, 0}; // unrecognized escape sequence
  }
  if (c >= 32 && c < 127) { // printable ASCII
    return {KeyType::Printable, static_cast<char>(c)};
  }
  return {KeyType::Ignored, 0}; // other control characters
}

Terminal::Terminal() {
  bool const enabled = EnableRawMode();
  _state.Mutate([enabled](State& state) { state.rawModeEnabled = enabled; });
}

Terminal::~Terminal() {
  if (_state.LoadMemberObject(&State::rawModeEnabled)) {
    WriteRaw("\r\n");
    Flush();
    DisableRawMode();
  }
}

void Terminal::Redraw(State const& state) {
  WriteRaw(kClearLine);
  WriteRaw(state.prompt);
  WriteRaw("> ");
  WriteRaw(state.buffer);
}

void Terminal::HistoryUp(State& state) {
  if (state.history.empty() || state.historyPos == 0) {
    return; // no history, or already at the oldest entry
  }
  if (state.historyPos == state.history.size()) {
    state.stashedInput = state.buffer; // leaving the live line: remember it for the trip back down
  }
  --state.historyPos;
  state.buffer = state.history[state.historyPos];
  Redraw(state);
}

void Terminal::HistoryDown(State& state) {
  if (state.historyPos == state.history.size()) {
    return; // already on the live line
  }
  ++state.historyPos;
  if (state.historyPos == state.history.size()) {
    state.buffer = state.stashedInput; // returned to the live line
  } else {
    state.buffer = state.history[state.historyPos];
  }
  Redraw(state);
}

void Terminal::Print(std::string_view text, std::optional<Color> color) {
  _state.Mutate([&](State& state) {
    WriteRaw(kClearLine);
    if (color) {
      WriteColorOn(*color);
    }
    WriteRaw(text);
    if (color) {
      WriteRaw(kColorReset);
    }
    Redraw(state);
    Flush();
  });
}

void Terminal::SetPrompt(std::string_view prompt) {
  _state.Mutate([&](State& state) {
    if (state.prompt != prompt) {
      state.prompt = prompt;
      Redraw(state);
      Flush();
    }
  });
}

bool Terminal::ReadLine(std::string& line) {
  _state.Mutate([](State& state) {
    state.buffer.clear();
    state.historyPos = state.history.size();
    state.stashedInput.clear();
    Redraw(state);
    Flush();
  });

  while (true) {
    Key const key = ReadKey(); // no lock held: async Print can run between keystrokes
    if (key.type == KeyType::Eof) {
      return false;
    }

    std::string submitted;
    bool didSubmit = false;
    _state.Mutate([&](State& state) {
      switch (key.type) {
        case KeyType::Enter: {
          WriteRaw("\r\n");
          submitted = state.buffer;
          // Record non-empty commands, suppressing consecutive duplicates.
          if (!state.buffer.empty() &&
              (state.history.empty() || state.history.back() != state.buffer)) {
            state.history.push_back(state.buffer);
          }
          state.buffer.clear();
          state.historyPos = state.history.size();
          state.stashedInput.clear();
          didSubmit = true;
          break;
        }
        case KeyType::Backspace: {
          if (!state.buffer.empty()) {
            state.buffer.pop_back();
            WriteRaw("\b \b");
          }
          break;
        }
        case KeyType::Up:
          HistoryUp(state);
          break;
        case KeyType::Down:
          HistoryDown(state);
          break;
        case KeyType::Printable: {
          state.buffer.push_back(key.ch);
          WriteRaw(std::string_view(&key.ch, 1));
          break;
        }
        case KeyType::Eof:
        case KeyType::Ignored:
          break;
      }
    });
    Flush(); // make this keystroke's output visible

    if (didSubmit) {
      line = std::move(submitted);
      return true;
    }
  }
}

} // namespace mochi::dbg
