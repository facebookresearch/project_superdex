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
#include <mochi_debugger/lib/terminal_text.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <utility>

#if MOCHI_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#endif

namespace mochi::dbg {

static size_t GetTerminalColumnCount() {
#if MOCHI_PLATFORM_WINDOWS
  CONSOLE_SCREEN_BUFFER_INFO info{};
  if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info)) {
    return static_cast<size_t>(info.srWindow.Right - info.srWindow.Left + 1);
  }
  return 0;
#else
  winsize size{};
  return ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0 ? size.ws_col : 0;
#endif
}

size_t Terminal::GetColumnCount() const {
  return GetTerminalColumnCount();
}

// VT escape: carriage-return + erase-to-end-of-line. Used to overwrite the current input line.
static constexpr std::string_view kClearLine = "\r\x1b[K";

// VT escape: reset all text attributes (including foreground color) back to the terminal default.
static constexpr std::string_view kColorReset = "\x1b[0m";

// Faint text, which follows the terminal's color scheme. The Windows console host does not support
// it, so outside Windows Terminal (WT_SESSION) and VS Code (TERM_PROGRAM) Windows uses dark gray.
static std::string_view GetSuggestionStyle() {
#if MOCHI_PLATFORM_WINDOWS
  if (GetEnvironmentVariableA("WT_SESSION", nullptr, 0) == 0 &&
      GetEnvironmentVariableA("TERM_PROGRAM", nullptr, 0) == 0) {
    return "\x1b[90m";
  }
#endif
  return "\x1b[2m";
}

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

namespace {
struct Key;
} // namespace

struct Terminal::InputReader {
  std::mutex readMutex;
  int pending = -1;
#if MOCHI_PLATFORM_WINDOWS
  DWORD originalInputMode = 0;
  DWORD originalOutputMode = 0;
  UINT originalInputCodePage = 0;
  UINT originalOutputCodePage = 0;
  char readBuffer[4]{};
  DWORD readBegin = 0;
  DWORD readEnd = 0;
#else
  termios originalTermios{};
#endif

  bool EnableRawMode();
  void DisableRawMode();
  int ReadCharBlocking();
  Key ReadKey();
};

#if MOCHI_PLATFORM_WINDOWS

bool Terminal::InputReader::EnableRawMode() {
  HANDLE const in = GetStdHandle(STD_INPUT_HANDLE);
  HANDLE const out = GetStdHandle(STD_OUTPUT_HANDLE);
  if (in == INVALID_HANDLE_VALUE || out == INVALID_HANDLE_VALUE) {
    return false;
  }
  if (!GetConsoleMode(in, &originalInputMode) || !GetConsoleMode(out, &originalOutputMode)) {
    return false;
  }
  originalInputCodePage = GetConsoleCP();
  originalOutputCodePage = GetConsoleOutputCP();
  // Disable line buffering and echo, and enable VT input so arrow keys arrive as the same escape
  // sequences (ESC [ A/B/...) that POSIX terminals emit.
  DWORD const inMode = (originalInputMode & ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT)) |
      ENABLE_VIRTUAL_TERMINAL_INPUT;
  if (!SetConsoleMode(in, inMode)) {
    return false;
  }
  if (!SetConsoleMode(out, originalOutputMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
    // Roll back the input mode we just changed so a partial failure doesn't leave the console
    // altered.
    SetConsoleMode(in, originalInputMode);
    return false;
  }
  if (!SetConsoleCP(CP_UTF8) || !SetConsoleOutputCP(CP_UTF8)) {
    SetConsoleCP(originalInputCodePage);
    SetConsoleOutputCP(originalOutputCodePage);
    SetConsoleMode(in, originalInputMode);
    SetConsoleMode(out, originalOutputMode);
    return false;
  }
  return true;
}

void Terminal::InputReader::DisableRawMode() {
  SetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), originalInputMode);
  SetConsoleMode(GetStdHandle(STD_OUTPUT_HANDLE), originalOutputMode);
  SetConsoleCP(originalInputCodePage);
  SetConsoleOutputCP(originalOutputCodePage);
}

// Read a UTF-8 byte, blocking. The console read buffer can hold a complete code point.
int Terminal::InputReader::ReadCharBlocking() {
  HANDLE const in = GetStdHandle(STD_INPUT_HANDLE);
  if (readBegin == readEnd) {
    DWORD bytesRead = 0;
    if (!ReadFile(in, readBuffer, static_cast<DWORD>(sizeof(readBuffer)), &bytesRead, nullptr) ||
        bytesRead == 0) {
      return -1;
    }
    readBegin = 0;
    readEnd = bytesRead;
  }
  return static_cast<unsigned char>(readBuffer[readBegin++]);
}

#else

bool Terminal::InputReader::EnableRawMode() {
  if (tcgetattr(STDIN_FILENO, &originalTermios) != 0) {
    return false;
  }
  termios raw = originalTermios;
  raw.c_lflag &= ~(static_cast<tcflag_t>(ICANON | ECHO));
  raw.c_cc[VMIN] = 1;
  raw.c_cc[VTIME] = 0;
  return tcsetattr(STDIN_FILENO, TCSANOW, &raw) == 0;
}

void Terminal::InputReader::DisableRawMode() {
  tcsetattr(STDIN_FILENO, TCSANOW, &originalTermios);
}

// Read a single character, blocking. Returns -1 on EOF/error.
int Terminal::InputReader::ReadCharBlocking() {
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
  Backspace, // delete the character before the cursor
  Delete, // delete the character at the cursor
  Up, // previous (older) history entry
  Down, // next (newer) history entry
  Left, // previous character
  Right, // next character
  WordLeft,
  WordRight,
  Home,
  End,
  Tab, // complete the input
  BackTab, // Shift+Tab: complete the input, cycling backward
  Printable, // text to insert (see Key::text)
  Ignored, // an unhandled key or control character
};

struct Key {
  KeyType type{KeyType::Ignored};
  std::string text;

  Key(KeyType keyType, char ch) : type(keyType), text(ch == 0 ? 0 : 1, ch) {}
  Key(KeyType keyType, std::string keyText) : type(keyType), text(std::move(keyText)) {}
};

} // namespace

static bool IsUtf8ContinuationByte(int byte) {
  return (static_cast<unsigned char>(byte) & 0xC0) == 0x80;
}

// Reads and decodes one keypress, blocking. Performs no locking, so async output (Print) may
// run between keystrokes. Terminal escape sequences may include modifier parameters.
Key Terminal::InputReader::ReadKey() {
  int const c = pending >= 0 ? std::exchange(pending, -1) : ReadCharBlocking();
  if (c < 0) {
    return {KeyType::Eof, 0};
  }
  if (c == '\r' || c == '\n') {
    return {KeyType::Enter, 0};
  }
  if (c == 127 || c == 8) { // DEL / backspace
    return {KeyType::Backspace, 0};
  }
  if (c == '\t') {
    return {KeyType::Tab, 0};
  }
  if (c == 1) { // Ctrl+A
    return {KeyType::Home, 0};
  }
  if (c == 2) { // Ctrl+B
    return {KeyType::Left, 0};
  }
  if (c == 5) { // Ctrl+E
    return {KeyType::End, 0};
  }
  if (c == 6) { // Ctrl+F
    return {KeyType::Right, 0};
  }
  if (c == 27) { // ESC: Meta key or terminal escape sequence
    int const c1 = ReadCharBlocking();
    if (c1 < 0) {
      return {KeyType::Eof, 0};
    }
    if (c1 == 'b' || c1 == 'B') {
      return {KeyType::WordLeft, 0};
    }
    if (c1 == 'f' || c1 == 'F') {
      return {KeyType::WordRight, 0};
    }
    if (c1 == 27) {
      pending = c1; // A second ESC starts the next key's escape sequence
      return {KeyType::Ignored, 0};
    }
    if (c1 != '[' && c1 != 'O') {
      return {KeyType::Ignored, 0};
    }
    std::string params;
    while (params.size() < 32) {
      int const c2 = ReadCharBlocking();
      if (c2 < 0) {
        return {KeyType::Eof, 0};
      }
      if (c2 >= 0x40 && c2 <= 0x7E) {
        bool const word = params.size() == 3 && params[0] == '1' && params[1] == ';' &&
            params[2] >= '3' && params[2] <= '8';
        switch (c2) {
          case 'A':
            return {KeyType::Up, 0};
          case 'B':
            return {KeyType::Down, 0};
          case 'C':
            return {word ? KeyType::WordRight : KeyType::Right, 0};
          case 'D':
            return {word ? KeyType::WordLeft : KeyType::Left, 0};
          case 'H':
            return {KeyType::Home, 0};
          case 'F':
            return {KeyType::End, 0};
          case 'Z':
            return {KeyType::BackTab, 0};
          case '~':
            if (params == "1" || params == "7") {
              return {KeyType::Home, 0};
            }
            if (params == "4" || params == "8") {
              return {KeyType::End, 0};
            }
            if (params == "3") {
              return {KeyType::Delete, 0};
            }
            return {KeyType::Ignored, 0};
          default:
            return {KeyType::Ignored, 0};
        }
      }
      params += static_cast<char>(c2);
    }
    return {KeyType::Ignored, 0}; // unrecognized escape sequence
  }
  if (c >= 32 && c < 127) { // printable ASCII
    return {KeyType::Printable, static_cast<char>(c)};
  }
  if (c >= 0xC2 && c <= 0xF4) {
    std::string text(1, static_cast<char>(c));
    size_t const bytes = c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
    while (text.size() < bytes) {
      int const byte = ReadCharBlocking();
      if (byte < 0) {
        return {KeyType::Eof, 0};
      }
      if (!IsUtf8ContinuationByte(byte)) {
        pending = byte;
        return {KeyType::Ignored, 0};
      }
      text += static_cast<char>(byte);
    }
    auto const second = static_cast<unsigned char>(text[1]);
    if ((c == 0xE0 && second < 0xA0) || (c == 0xED && second >= 0xA0) ||
        (c == 0xF0 && second < 0x90) || (c == 0xF4 && second >= 0x90)) {
      return {KeyType::Ignored, 0};
    }
    return {KeyType::Printable, std::move(text)};
  }
  return {KeyType::Ignored, 0}; // other control characters
}

static bool IsWordCharacter(char c) {
  auto const byte = static_cast<unsigned char>(c);
  return byte >= 0x80 || std::isalnum(byte) || c == '_';
}

static size_t PreviousCharacter(std::string_view text, size_t cursor) {
  if (cursor > 0) {
    --cursor;
    while (cursor > 0 && IsUtf8ContinuationByte(text[cursor])) {
      --cursor;
    }
  }
  return cursor;
}

static size_t NextCharacter(std::string_view text, size_t cursor) {
  if (cursor < text.size()) {
    ++cursor;
    while (cursor < text.size() && IsUtf8ContinuationByte(text[cursor])) {
      ++cursor;
    }
  }
  return cursor;
}

Terminal::Terminal()
    : _input(std::make_unique<InputReader>()), _suggestionStyle(GetSuggestionStyle()) {
  bool const enabled = _input->EnableRawMode();
  _state.Mutate([enabled](State& state) { state.rawModeEnabled = enabled; });
}

Terminal::~Terminal() {
  if (_state.LoadMemberObject(&State::rawModeEnabled)) {
    // The cursor is on the pinned prompt, which output printed during shutdown has redrawn.
    WriteRaw(kClearLine);
    Flush();
    _input->DisableRawMode();
  }
}

void Terminal::Redraw(State const& state) const {
  WriteRaw(kClearLine);
  std::string const text = state.prompt + "> " + state.buffer;
  size_t const cursor = state.prompt.size() + 2 + state.cursorPos;
  size_t const columns = GetTerminalColumnCount();
  auto const view = MakeTerminalTextView(text, cursor, columns);
  WriteRaw(view.text);
  size_t end = CountTerminalCells(view.text);
  if (state.cursorPos == state.buffer.size() && state.suggestionText == state.buffer) {
    // Like the input, the suggestion stops before the last column, so the line never wraps.
    size_t const width = columns == 0 ? SIZE_MAX : std::max(size_t{1}, columns - 1);
    std::string const suggestion = FitSuggestion(
        state.buffer, state.suggestion, state.annotations, width > end ? width - end : 0);
    if (!suggestion.empty()) {
      WriteRaw(_suggestionStyle);
      WriteRaw(suggestion);
      WriteRaw(kColorReset);
      end += CountTerminalCells(suggestion);
    }
  }
  if (view.cursorColumn < end) {
    WriteRaw("\x1b[" + std::to_string(end - view.cursorColumn) + "D");
  }
}

void Terminal::HistoryUp(State& state) const {
  if (state.history.empty() || state.historyPos == 0) {
    return; // no history, or already at the oldest entry
  }
  if (state.historyPos == state.history.size()) {
    state.stashedInput = state.buffer; // leaving the live line: remember it for the trip back down
    state.stashedCursorPos = state.cursorPos;
  }
  --state.historyPos;
  state.buffer = state.history[state.historyPos];
  state.cursorPos = state.buffer.size();
  Redraw(state);
}

void Terminal::HistoryDown(State& state) const {
  if (state.historyPos == state.history.size()) {
    return; // already on the live line
  }
  ++state.historyPos;
  if (state.historyPos == state.history.size()) {
    state.buffer = state.stashedInput; // returned to the live line
    state.cursorPos = state.stashedCursorPos;
  } else {
    state.buffer = state.history[state.historyPos];
    state.cursorPos = state.buffer.size();
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

void Terminal::SetCompletion(InputFn onInput, TabFn onTab) {
  _onInput = std::move(onInput);
  _onTab = std::move(onTab);
}

void Terminal::SetSuggestion(
    std::string_view text,
    std::string_view preview,
    Span<std::string const> annotations) {
  _state.Mutate([&](State& state) {
    // A suggestion computed for an earlier input, for example by a reply racing a keystroke,
    // changes nothing on screen.
    if (text != state.buffer ||
        (state.suggestionText == text && state.suggestion == preview &&
         MakeConstSpan(state.annotations) == annotations)) {
      return;
    }
    state.suggestionText = text;
    state.suggestion = preview;
    state.annotations.assign(annotations.begin(), annotations.end());
    Redraw(state);
    Flush();
  });
}

bool Terminal::ReadLine(std::string& line) {
  std::lock_guard<std::mutex> inputLock(_input->readMutex);
  _state.Mutate([&](State& state) {
    state.buffer.clear();
    state.cursorPos = 0;
    state.historyPos = state.history.size();
    state.stashedInput.clear();
    state.stashedCursorPos = 0;
    state.suggestion.clear();
    state.annotations.clear();
    Redraw(state);
    Flush();
  });
  if (_onInput) {
    _onInput("", true);
  }

  while (true) {
    // The state lock is not held while reading, so async Print can run.
    Key const key = _input->ReadKey();
    if (key.type == KeyType::Eof) {
      return false;
    }

    std::string submitted;
    bool didSubmit = false;
    _state.Mutate([&](State& state) {
      switch (key.type) {
        case KeyType::Enter: {
          // The submitted line keeps no suggestion.
          state.suggestion.clear();
          state.annotations.clear();
          WriteRaw(kClearLine);
          WriteRaw(state.prompt + "> " + state.buffer);
          WriteRaw("\r\n");
          submitted = state.buffer;
          // Record non-empty commands, suppressing consecutive duplicates.
          if (!state.buffer.empty() &&
              (state.history.empty() || state.history.back() != state.buffer)) {
            state.history.push_back(state.buffer);
          }
          state.buffer.clear();
          state.cursorPos = 0;
          state.historyPos = state.history.size();
          state.stashedInput.clear();
          state.stashedCursorPos = 0;
          didSubmit = true;
          break;
        }
        case KeyType::Backspace: {
          if (state.cursorPos > 0) {
            size_t const previous = PreviousCharacter(state.buffer, state.cursorPos);
            state.buffer.erase(previous, state.cursorPos - previous);
            state.cursorPos = previous;
            Redraw(state);
          }
          break;
        }
        case KeyType::Delete:
          if (state.cursorPos < state.buffer.size()) {
            state.buffer.erase(
                state.cursorPos, NextCharacter(state.buffer, state.cursorPos) - state.cursorPos);
            Redraw(state);
          }
          break;
        case KeyType::Up:
          HistoryUp(state);
          break;
        case KeyType::Down:
          HistoryDown(state);
          break;
        case KeyType::Left:
          if (state.cursorPos > 0) {
            state.cursorPos = PreviousCharacter(state.buffer, state.cursorPos);
            Redraw(state);
          }
          break;
        case KeyType::Right:
          if (state.cursorPos < state.buffer.size()) {
            state.cursorPos = NextCharacter(state.buffer, state.cursorPos);
            Redraw(state);
          }
          break;
        case KeyType::WordLeft:
          while (state.cursorPos > 0 && !IsWordCharacter(state.buffer[state.cursorPos - 1])) {
            state.cursorPos = PreviousCharacter(state.buffer, state.cursorPos);
          }
          while (state.cursorPos > 0 && IsWordCharacter(state.buffer[state.cursorPos - 1])) {
            state.cursorPos = PreviousCharacter(state.buffer, state.cursorPos);
          }
          Redraw(state);
          break;
        case KeyType::WordRight:
          while (state.cursorPos < state.buffer.size() &&
                 !IsWordCharacter(state.buffer[state.cursorPos])) {
            state.cursorPos = NextCharacter(state.buffer, state.cursorPos);
          }
          while (state.cursorPos < state.buffer.size() &&
                 IsWordCharacter(state.buffer[state.cursorPos])) {
            state.cursorPos = NextCharacter(state.buffer, state.cursorPos);
          }
          Redraw(state);
          break;
        case KeyType::Home:
          state.cursorPos = 0;
          Redraw(state);
          break;
        case KeyType::End:
          state.cursorPos = state.buffer.size();
          Redraw(state);
          break;
        case KeyType::Printable: {
          state.buffer.insert(state.cursorPos, key.text);
          state.cursorPos += key.text.size();
          Redraw(state);
          break;
        }
        case KeyType::Tab:
        case KeyType::BackTab:
        case KeyType::Eof:
        case KeyType::Ignored:
          break;
      }
    });
    if (didSubmit) {
      Flush();
      line = std::move(submitted);
      return true;
    }
    if (key.type == KeyType::Tab || key.type == KeyType::BackTab) {
      if (auto const text = _onTab ? _onTab(key.type == KeyType::BackTab) : std::nullopt) {
        _state.Mutate([&](State& state) {
          state.buffer = *text;
          state.cursorPos = state.buffer.size();
          Redraw(state);
        });
      }
    }
    if (_onInput && key.type != KeyType::Ignored) {
      auto const [text, isCursorAtEnd] = _state.Read([](State const& state) {
        return std::pair{state.buffer, state.cursorPos == state.buffer.size()};
      });
      _onInput(text, isCursorAtEnd);
    }
    // One flush shows the keystroke with its suggestion, so the faint text does not flicker.
    Flush();
  }
}

} // namespace mochi::dbg
