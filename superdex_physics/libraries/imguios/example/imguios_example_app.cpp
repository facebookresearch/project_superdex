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

#include <imguios/application.h>
#include <array>

// Macro to disable console
#define APP_NO_CONSOLE
#if defined(_WIN32) && defined(APP_NO_CONSOLE)
#pragma comment(linker, "/subsystem:windows /ENTRY:mainCRTStartup")
#endif

// ImGuios Application

class ImGuiosTestApp : public ImGuios::Application {
 public:
  ImGuiosTestApp()
      : ImGuios::Application(
            std::make_unique<ImGuios::Window>(
                kDefaultWindowWidth,
                kDefaultWindowHeight,
                "ImGuios Test App",
                ImGuios::WindowFlags_MSAA)) {}

  // Called at top of Run.
  virtual void OnInitialize() override {}

  // Called at the bottom of Run.
  virtual void OnShutdown() override {}

  // Update or Tick, called once per frame in Run.
  virtual void OnUpdate() override {
    // [MENU] Main Menu
    ShowMainMenuBar();

    // [DOCKSPACE]
    ImGui::DockSpaceOverViewport(ImGui::GetMainViewport()->ID);

    // [WINDOW] Test Window
    ShowTestWindow();

    // [WINDOW] ImGui Stack Tool
    if (_showImGuiStackToolWindow) {
      ImGui::ShowStackToolWindow(&_showImGuiStackToolWindow);
    }
    // [WINDOW] ImGui Demo
    if (_showImGuiDemoWindow) {
      ImGui::ShowDemoWindow(&_showImGuiDemoWindow);
    }
    // [WINDOW] ImGui Metrics
    if (_showImGuiMetricsWindow) {
      ImGui::ShowMetricsWindow(&_showImGuiMetricsWindow);
    }
    // [WINDOW] ImPlot Demo
    if (_showImPlotDemoWindow) {
      ImPlot::ShowDemoWindow(&_showImPlotDemoWindow);
    }
    // [WINDOW] ImPlot Metrics
    if (_showImPlotMetricsWindow) {
      ImPlot::ShowMetricsWindow(&_showImPlotMetricsWindow);
    }
  }

  void ShowMainMenuBar() {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
    auto sepColor = ImGui::GetStyleColorVec4(ImGuiCol_Text);
    sepColor.w = 0.1f;
    ImGui::PushStyleColor(ImGuiCol_Separator, sepColor);
    if (ImGui::BeginMainMenuBar()) {
      if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Save Project", "Ctrl + S")) {
          // TODO
        }
        ImGui::Separator();
        if (ImGui::MenuItem("New Project", "Ctrl + N")) {
          // TODO
        }
        if (ImGui::MenuItem("Open Project...", "Ctrl + O")) {
          // TODO
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Exit", "Alt + F4")) {
          this->Stop();
        }
        ImGui::EndMenu();
      }
      if (ImGui::BeginMenu("Edit")) {
        if (ImGui::MenuItem("Undo", "Ctrl + Z", nullptr, true)) {
          // TODO
        }
        if (ImGui::MenuItem("Redo", "Ctrl + Y", nullptr, true)) {
          // TODO
        }
        ImGui::EndMenu();
      }
      if (ImGui::BeginMenu("ImGui")) {
        ImGui::MenuItem("Demo", nullptr, &_showImGuiDemoWindow);
        ImGui::MenuItem("Metrics", nullptr, &_showImGuiMetricsWindow);
        ImGui::MenuItem("Stack Tool", nullptr, &_showImGuiStackToolWindow);
        ImGui::EndMenu();
      }
      if (ImGui::BeginMenu("ImPlot")) {
        ImGui::MenuItem("Demo", nullptr, &_showImPlotDemoWindow);
        ImGui::MenuItem("Metrics", nullptr, &_showImPlotMetricsWindow);
        ImGui::EndMenu();
      }

      ImGui::EndMainMenuBar();
    }
    ImGui::PopStyleColor(); // ImGuiCol_Separator
    ImGui::PopStyleVar(); // ImGuiStyleVar_WindowBorderSize
  }

  static void ShowTestWindow() {
    ImGui::Begin("Test Window");
    {
      static float sliderValue = 0.5f;
      static std::array<char, 64> textBuffer = {"Type In Me"};
      static std::array<float, 10> plotData = {
          0.1f, 0.4f, 0.2f, 0.3f, 0.6f, 0.8f, 0.9f, 0.3f, 0.5f, 0.8f};

      ImGui::SliderFloat("Slider", &sliderValue, 0.0f, 1.0f);
      ImGui::InputText("Input Text", textBuffer.data(), textBuffer.size());

      if (ImGui::Button("Press Me")) {
        // TODO
      }

      if (ImPlot::BeginPlot("My Plot")) {
        ImPlot::SetupAxes("X-Axis", "Y-Axis");
        ImPlot::PlotLine("My Line", plotData.data(), 10);
        ImPlot::EndPlot();
      }

      ImGui::Separator();

      ImGui::TextUnformatted("How to use ImGui effectively:");
      ImGui::Indent();
      ImGui::TextUnformatted("1) Explore the ImGui and ImPlot demos from the menus above.");
      ImGui::TextUnformatted("2) Find samples for the widgets you need to build your app.");
      ImGui::TextUnformatted(
          "3) Find code for these sampeles in imgui_demo.cpp or implot_demo.cpp (in ImGuios/third-party/).");
      ImGui::TextUnformatted("4) Copy and modify code to you needs.");
      ImGui::Unindent();
    }
    ImGui::End();
  }

 private:
  static constexpr int kDefaultWindowWidth = 1600;
  static constexpr int kDefaultWindowHeight = 900;

  bool _showImGuiDemoWindow = false;
  bool _showImGuiMetricsWindow = false;
  bool _showImGuiStackToolWindow = false;
  bool _showImPlotDemoWindow = false;
  bool _showImPlotMetricsWindow = false;
};

// Main / Application Entry

int main(int argc, char** argv) {
  ImGuiosTestApp app;
  app.Run();
}
