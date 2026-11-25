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

#include <gtest/gtest.h>
#include <imgui.h>
#include <imgui_internal.h>

namespace mochi::dbg {
namespace {
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
    bool const submit = BuildCommandInput(terminal, "/> ");
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
  bool IsInputActive() const {
    return ImGui::GetActiveID() == inputId;
  }

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
} // namespace
} // namespace mochi::dbg
