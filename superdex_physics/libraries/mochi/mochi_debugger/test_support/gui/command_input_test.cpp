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

#include "command_input.h"
#include "terminal_panel.h"

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_debugger/lib/auto_completer.h>

#include <gtest/gtest.h>
#include <imgui.h>
#include <imgui_internal.h>

#include <chrono>
#include <string>
#include <utility>

namespace mochi::dbg {
namespace {
using Strings = DynamicArray<std::string>;

class CommandInputTest : public testing::Test {
 protected:
  void SetUp() override {
    ImGui::CreateContext();
    auto& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(800, 400);
    io.DeltaTime = 1.0f / 60.0f;
    // The app's keyboard navigation and docking flags
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_DockingEnable;
    io.Fonts->AddFontDefault();
    unsigned char* pixels = nullptr;
    int width = 0;
    int height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    terminal.refocus = true;
    Frame();
    Frame();
    ASSERT_TRUE(IsInputActive());
  }
  void TearDown() override {
    ImGui::DestroyContext();
  }
  // Run a frame and return whether the input was submitted.
  bool Frame() {
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(800, 400));
    ImGui::Begin("Test", nullptr, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoTitleBar);
    ImGui::SetCursorPosY(300);
    inputId = ImGui::GetID("##cmd");
    bool const submit = BuildCommandInput(terminal, &completer, "/> ");
    ImGui::End();
    ImGui::Render();
    return submit;
  }
  bool Key(ImGuiKey key, bool shift = false) {
    auto& io = ImGui::GetIO();
    io.AddKeyEvent(ImGuiMod_Shift, shift);
    io.AddKeyEvent(key, true);
    bool const submit = Frame();
    io.AddKeyEvent(key, false);
    io.AddKeyEvent(ImGuiMod_Shift, false);
    return Frame() || submit;
  }
  void Type(char const* text) {
    ImGui::GetIO().AddInputCharactersUTF8(text);
    Frame();
  }
  void Reply(Strings candidates) {
    ASSERT_FALSE(sent.empty());
    protocol::AutoCompleteReply reply(sent.back());
    reply.result._candidates.assign(candidates.begin(), candidates.end());
    completer.OnReply(std::move(reply));
    Frame();
  }
  void Click(ImVec2 position) {
    auto& io = ImGui::GetIO();
    io.AddMousePosEvent(position.x, position.y);
    io.AddMouseButtonEvent(0, true);
    Frame();
    io.AddMouseButtonEvent(0, false);
    Frame();
  }
  bool IsInputActive() const {
    return ImGui::GetActiveID() == inputId;
  }

  DynamicArray<protocol::AutoCompleteRequest> sent;
  AutoCompleter completer{AutoCompleterHost{
      .send = [this](protocol::AutoCompleteRequest const& request) { sent.push_back(request); },
      .workingDirectory = [] { return std::string{"/"}; },
      .commandKind = [](std::string_view) { return ConsoleCommandKind::Remote; },
      .connectionGeneration = [] { return uint64_t{0}; },
      .now = [] { return std::chrono::nanoseconds{std::chrono::seconds{1}}; }}};
  TerminalPanelState terminal;
  ImGuiID inputId = 0;
};

TEST_F(CommandInputTest, UpAndDownBrowseHistory) {
  terminal.history = {"first", "second"};
  terminal.historyPos = 2;
  Type("live");
  Key(ImGuiKey_UpArrow);
  EXPECT_EQ(terminal.input, "second");
  Key(ImGuiKey_UpArrow);
  Key(ImGuiKey_UpArrow);
  EXPECT_EQ(terminal.input, "first");
  Key(ImGuiKey_DownArrow);
  Key(ImGuiKey_DownArrow);
  EXPECT_EQ(terminal.input, "live");
  EXPECT_TRUE(Key(ImGuiKey_Enter));
  EXPECT_EQ(terminal.input, "live");
}

TEST_F(CommandInputTest, TabFillsInAndOpensTheMenu) {
  Type("/s");
  ASSERT_EQ(sent.size(), 1);
  Reply({"scenes", "settings"});
  EXPECT_FALSE(terminal.isMenuOpen);
  Key(ImGuiKey_Tab);
  EXPECT_EQ(terminal.input, "/scenes");
  EXPECT_TRUE(terminal.isMenuOpen);
  EXPECT_EQ(terminal.menuCandidates, (Strings{"scenes", "settings"}));
  EXPECT_EQ(terminal.menuSelected, 0);
  Key(ImGuiKey_Tab);
  EXPECT_EQ(terminal.input, "/settings");
  Key(ImGuiKey_Tab);
  EXPECT_EQ(terminal.input, "/scenes");
  EXPECT_TRUE(IsInputActive());
}

TEST_F(CommandInputTest, ShiftTabCyclesBackwardWithoutMovingFocus) {
  Type("/s");
  Reply({"scenes", "settings", "speed"});
  Key(ImGuiKey_Tab, /*shift*/ true);
  EXPECT_EQ(terminal.input, "/speed");
  EXPECT_TRUE(terminal.isMenuOpen);
  Key(ImGuiKey_Tab, /*shift*/ true);
  EXPECT_EQ(terminal.input, "/settings");
  EXPECT_TRUE(IsInputActive());
}

TEST_F(CommandInputTest, CtrlTabStillSwitchesWindows) {
  auto& io = ImGui::GetIO();
  io.AddKeyEvent(ImGuiMod_Ctrl, true);
  Frame();
  io.AddKeyEvent(ImGuiKey_Tab, true);
  Frame();
  EXPECT_NE(ImGui::GetCurrentContext()->NavWindowingTarget, nullptr);
  io.AddKeyEvent(ImGuiKey_Tab, false);
  io.AddKeyEvent(ImGuiMod_Ctrl, false);
  Frame();
}

TEST_F(CommandInputTest, ArrowsMoveThroughTheMenuWithoutWrapping) {
  Type("/s");
  Reply({"scenes", "settings"});
  Key(ImGuiKey_Tab);
  Key(ImGuiKey_DownArrow);
  EXPECT_EQ(terminal.input, "/settings");
  Key(ImGuiKey_DownArrow);
  EXPECT_EQ(terminal.input, "/settings");
  Key(ImGuiKey_UpArrow);
  Key(ImGuiKey_UpArrow);
  EXPECT_EQ(terminal.input, "/scenes");
  EXPECT_EQ(terminal.menuSelected, 0);
}

TEST_F(CommandInputTest, EscapeClosesTheMenuAndKeepsTheInput) {
  Type("/s");
  Reply({"scenes", "settings"});
  Key(ImGuiKey_Tab);
  Key(ImGuiKey_Escape);
  EXPECT_FALSE(terminal.isMenuOpen);
  EXPECT_EQ(terminal.input, "/scenes");
  EXPECT_TRUE(IsInputActive());

  // With the menu closed, the arrows browse history again.
  terminal.history = {"previous"};
  terminal.historyPos = 1;
  Key(ImGuiKey_UpArrow);
  EXPECT_EQ(terminal.input, "previous");
}

TEST_F(CommandInputTest, TypingFiltersTheMenuAndClosesItWhenNothingMatches) {
  Type("/s");
  Reply({"scenes", "settings"});
  Key(ImGuiKey_Tab);
  Key(ImGuiKey_Backspace);
  Key(ImGuiKey_Backspace);
  Key(ImGuiKey_Backspace);
  Key(ImGuiKey_Backspace);
  Key(ImGuiKey_Backspace);
  EXPECT_EQ(terminal.input, "/s");
  EXPECT_EQ(terminal.menuCandidates, (Strings{"scenes", "settings"}));
  Type("et");
  EXPECT_EQ(terminal.menuCandidates, (Strings{"settings"}));
  Type("x");
  EXPECT_FALSE(terminal.isMenuOpen);
}

TEST_F(CommandInputTest, EnterClosesTheMenuAndSubmits) {
  Type("/s");
  Reply({"scenes", "settings"});
  Key(ImGuiKey_Tab);
  EXPECT_TRUE(Key(ImGuiKey_Enter));
  EXPECT_FALSE(terminal.isMenuOpen);
  EXPECT_EQ(terminal.input, "/scenes");
}

TEST_F(CommandInputTest, ClickingACandidateFillsItIn) {
  Type("/s");
  Reply({"scenes", "settings"});
  Key(ImGuiKey_Tab);
  ImGuiWindow const* menu = ImGui::FindWindowByName("##completionMenu");
  ASSERT_NE(menu, nullptr);
  float const rowHeight = ImGui::GetTextLineHeightWithSpacing();
  Click(ImVec2(
      menu->Pos.x + menu->WindowPadding.x + 2,
      menu->Pos.y + menu->WindowPadding.y + 1.5f * rowHeight));
  Frame(); // Focus returns to the input
  Frame();
  EXPECT_EQ(terminal.input, "/settings");
  EXPECT_TRUE(terminal.isMenuOpen);
  EXPECT_TRUE(IsInputActive());
}

TEST_F(CommandInputTest, ClickingElsewhereClosesTheMenu) {
  Type("/s");
  Reply({"scenes", "settings"});
  Key(ImGuiKey_Tab);
  Click(ImVec2(400, 100));
  EXPECT_FALSE(terminal.isMenuOpen);
  EXPECT_EQ(terminal.input, "/scenes");
}
} // namespace
} // namespace mochi::dbg
