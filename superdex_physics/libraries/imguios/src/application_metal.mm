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

// Metal backend implementation for macOS

#include <imguios/application.h>

#include "imgui_impl_glfw.h"
#include "imgui_impl_metal.h"
#include "imgui_internal.h"

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>

#ifdef IMGUI_ENABLE_TEST_ENGINE
#include "application_mcp.h"
#include "imgui_mcp_bridge.h"
#endif

namespace ImGuios {

#ifndef IMGUI_ENABLE_TEST_ENGINE
// McpState must be defined so unique_ptr<McpState> destructor has a complete type.
struct Application::McpState {};
#endif

// Metal-specific state
static id<MTLDevice> g_mtlDevice = nil;
static id<MTLCommandQueue> g_mtlCommandQueue = nil;
static CAMetalLayer* g_metalLayer = nil;
static id<CAMetalDrawable> g_lastDrawable = nil;
static id<MTLCommandBuffer> g_lastCommandBuffer = nil;

Application::Application(std::unique_ptr<Window> mainWindow, McpConfig mcpConfig)
    : _mcpConfig(mcpConfig) {
  // set main window
  if (mainWindow == nullptr) {
    throw std::runtime_error("Main window cannot be null!");
  }
  _mainWindow = std::move(mainWindow);

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
  // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // Enable Docking
  // Note: Multi-viewport with Metal requires additional setup, disable for now
  // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

  // Setup Dear ImGui style
  ImGuios::StyleColorsDefault();

  // Setup Metal
  g_mtlDevice = MTLCreateSystemDefaultDevice();
  g_mtlCommandQueue = [g_mtlDevice newCommandQueue];

  // Get the native Cocoa window and set up Metal layer
  NSWindow* nswindow = glfwGetCocoaWindow(_mainWindow->_window);
  NSView* nsview = [nswindow contentView];

  g_metalLayer = [CAMetalLayer layer];
  g_metalLayer.device = g_mtlDevice;
  g_metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;
  g_metalLayer.contentsScale = [nswindow backingScaleFactor];
  [nsview setLayer:g_metalLayer];
  [nsview setWantsLayer:YES];

  // Query DPI scale for Retina/HiDPI support
  float xscale = 1.0f, yscale = 1.0f;
  glfwGetWindowContentScale(_mainWindow->_window, &xscale, &yscale);
  _dpiScale = xscale;

  // Setup Platform/Renderer backends
  ImGui_ImplGlfw_InitForOther(_mainWindow->_window, true);
  ImGui_ImplMetal_Init(g_mtlDevice);

  // add fonts
  io.Fonts->Clear();

  using namespace ImGuios::Fonts;
  LoadFont("Roboto Regular", 16, Roboto_Regular_ttf, Roboto_Regular_ttf_len);
  LoadFont("Roboto Bold", 16, Roboto_Bold_ttf, Roboto_Bold_ttf_len);
  LoadFont("Roboto Italic", 16, Roboto_Italic_ttf, Roboto_Italic_ttf_len);
  LoadFont("Roboto Mono Regular", 16, RobotoMono_Regular_ttf, RobotoMono_Regular_ttf_len);
  LoadFont("Roboto Mono Bold", 16, RobotoMono_Bold_ttf, RobotoMono_Bold_ttf_len);
  LoadFont("Roboto Mono Italic", 16, RobotoMono_Italic_ttf, RobotoMono_Italic_ttf_len);

  // Fonts are rasterized at dpiScale * logical size for crispness. Compensate so ImGui
  // renders them at the correct logical size (not 2x as large on Retina).
  io.FontGlobalScale = 1.0f / _dpiScale;
}

static void NormalizeString(std::string& str) {
  str.erase(
      std::remove_if(str.begin(), str.end(), [](unsigned char c) { return std::isspace(c); }),
      str.end());
  std::transform(
      str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::tolower(c); });
}

ImFont* Application::GetFont(const char* name) {
  std::string keyName = name;
  NormalizeString(keyName);
  if (_fonts.count(keyName)) {
    return _fonts[keyName];
  }
  return nullptr;
}

ImFont* Application::LoadFont(const char* name, float fontSize, void* fontData, int fontDataSize) {
  float const scaledSize = fontSize * _dpiScale;
  std::string keyName = name;
  NormalizeString(keyName);
  ImGuiIO& io = ImGui::GetIO();
  ImFontConfig font_cfg;
  font_cfg.FontDataOwnedByAtlas = false;
  ImFontConfig icons_config;
  icons_config.MergeMode = true;
  icons_config.PixelSnapH = true;
  icons_config.GlyphOffset = ImVec2(0, 0);
  icons_config.OversampleH = 1;
  icons_config.OversampleV = 1;
  icons_config.FontDataOwnedByAtlas = false;
  static constexpr auto fa_ranges = std::to_array<ImWchar>({ICON_MIN_FA, ICON_MAX_FA, 0});
  ImStrncpy(font_cfg.Name, name, 40);
  auto font = io.Fonts->AddFontFromMemoryTTF(fontData, fontDataSize, scaledSize, &font_cfg);
  _fonts[keyName] = font;
  io.Fonts->AddFontFromMemoryTTF(
      Fonts::fa_solid_900_ttf,
      Fonts::fa_solid_900_ttf_len,
      scaledSize,
      &icons_config,
      fa_ranges.data());
  return font;
}

Window* Application::GetMainWindow() {
  return _mainWindow.get();
}

Application::~Application() {
  if (_running) {
    Stop();
  }
  ImGui_ImplMetal_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImPlot::DestroyContext();
  ImGui::DestroyContext();
  McpDestroyContext();
  _mainWindow.reset();

  // Metal cleanup is handled by ARC
  g_lastDrawable = nil;
  g_lastCommandBuffer = nil;
  g_mtlCommandQueue = nil;
  g_mtlDevice = nil;
  g_metalLayer = nil;
}

void Application::Run() {
  _running = true;
  OnInitialize();
  McpInitialize();
  while (!glfwWindowShouldClose(_mainWindow->_window) && _running) {
    @autoreleasepool {
      glfwPollEvents();

      // Update Metal layer size
      int width = 0, height = 0;
      glfwGetFramebufferSize(_mainWindow->_window, &width, &height);
      g_metalLayer.drawableSize = CGSizeMake(width, height);

      id<CAMetalDrawable> drawable = [g_metalLayer nextDrawable];
      if (drawable == nil) {
        continue;
      }

      MTLRenderPassDescriptor* renderPassDescriptor =
          [MTLRenderPassDescriptor renderPassDescriptor];
      renderPassDescriptor.colorAttachments[0].texture = drawable.texture;
      renderPassDescriptor.colorAttachments[0].loadAction = MTLLoadActionClear;
      renderPassDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
      renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);

      id<MTLCommandBuffer> commandBuffer = [g_mtlCommandQueue commandBuffer];

      // Start the Dear ImGui frame
      ImGui_ImplMetal_NewFrame(renderPassDescriptor);
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();
      OnUpdate();
      // Rendering
      ImGui::Render();

      // Call OnMainWindowRender for any custom Metal rendering
      OnMainWindowRender();

      id<MTLRenderCommandEncoder> renderEncoder =
          [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
      [renderEncoder pushDebugGroup:@"ImGui"];
      ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), commandBuffer, renderEncoder);
      [renderEncoder popDebugGroup];
      [renderEncoder endEncoding];

      g_lastDrawable = drawable;
      g_lastCommandBuffer = commandBuffer;
      [commandBuffer presentDrawable:drawable];
      [commandBuffer commit];
      OnPostSwap();
      McpPostSwap();
    }
  }
  OnShutdown();
  McpShutdown();
  _running = false;
}

void Application::OnMainWindowRender() {
  // Default implementation - nothing to do for Metal
  // User may override for custom rendering
}

bool Application::IsRunning() const {
  return _running;
}

void Application::Stop() {
  _running = false;
}

// McpPostSwap — Metal framebuffer readback for MCP screenshot capture.
// McpInitialize, McpShutdown, McpDestroyContext are in application_mcp.cpp (shared).

#ifdef IMGUI_ENABLE_TEST_ENGINE

void Application::McpPostSwap() {
  if (!_mcpState) {
    return;
  }

  ImGuiTestEngine_PostSwap(_mcpState->testEngine);

  // ImGuiMcpBridge_Tick() returns true only when a screenshot request is
  // pending — not every frame while MCP is connected.
  if (ImGuiMcpBridge_Tick()) {
    // Capture framebuffer from the last rendered Metal drawable
    if (g_lastCommandBuffer && g_lastDrawable) {
      [g_lastCommandBuffer waitUntilCompleted];
      id<MTLTexture> texture = g_lastDrawable.texture;
      int w = static_cast<int>(texture.width);
      int h = static_cast<int>(texture.height);
      if (w > 0 && h > 0) {
        std::vector<uint8_t> pixels(static_cast<size_t>(w) * h * 4);
        [texture getBytes:pixels.data()
              bytesPerRow:static_cast<NSUInteger>(w * 4)
               fromRegion:MTLRegionMake2D(
                              0, 0, static_cast<NSUInteger>(w), static_cast<NSUInteger>(h))
              mipmapLevel:0];

        // Metal uses BGRA pixel format — convert to RGBA
        for (size_t i = 0; i < pixels.size(); i += 4) {
          std::swap(pixels[i], pixels[i + 2]);
        }

        // Metal reads top-to-bottom (no vertical flip needed, unlike OpenGL)
        std::lock_guard<std::mutex> lock(_mcpState->cachedFb.mutex);
        _mcpState->cachedFb.pixels = std::move(pixels);
        _mcpState->cachedFb.width = w;
        _mcpState->cachedFb.height = h;
      }
    }
  }

  if (ImGuiMcpBridge_WantsShutdown()) {
    Stop();
  }
}

#else // !IMGUI_ENABLE_TEST_ENGINE

void Application::McpPostSwap() {}

#endif // IMGUI_ENABLE_TEST_ENGINE

// Metal-specific accessors - return as void* for C++ compatibility
void* GetMetalDevice() {
  return (__bridge void*)g_mtlDevice;
}

void* GetMetalCommandQueue() {
  return (__bridge void*)g_mtlCommandQueue;
}

void* GetMetalLayer() {
  return (__bridge void*)g_metalLayer;
}

} // namespace ImGuios
