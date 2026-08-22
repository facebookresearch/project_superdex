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

#include <filament/Engine.h>
#include <filament/Renderer.h>

#include <mochi_core/utils/error.h>

#include <cstdint>
#include <vector>

namespace mochi_renderer {
class Scene;
} // namespace mochi_renderer

namespace filament {
class View;
} // namespace filament

namespace superdex::studio {

class RenderTarget;

// Optional second render for the selection highlight, composited over the main image. The overlay
// view re-renders the highlighted meshes into overlayTarget with its own cleared (transparent)
// color and depth -- an isolated pass that yields a clean nearest-surface silhouette unaffected by
// scene occlusion. The composite view (a fullscreen TRANSLUCENT pass) then blends that overlay back
// over the main render target. Null members mean "no highlight this frame" and the passes are
// skipped.
struct HighlightPass {
  filament::View* overlayView = nullptr;
  RenderTarget* overlayTarget = nullptr;
  filament::View* compositeView = nullptr;
};

class Renderer {
 public:
  static std::unique_ptr<Renderer>
  Create(filament::Engine* engine, filament::Renderer* filamentRenderer, int w, int h);
  ~Renderer();
  void SetClearColor(filament::math::float4 color);
  filament::math::float4 GetClearColor() const;
  // When beginEndFrame is true (default) this method owns the full filament frame
  // lifetime (beginFrame / render / endFrame) and respects frame pacing. When
  // false, the caller MUST guarantee a filament frame is already active on this
  // renderer (beginFrame returned true externally) - render() will be called
  // without beginFrame. Filament requires render() between beginFrame/endFrame;
  // violating that triggers a debug invariant failure. For true offscreen work
  // with no active frame, use filament::Renderer::renderStandaloneView() instead.
  void Render(
      mochi_renderer::Scene* scene,
      RenderTarget* renderTarget,
      bool flushAndWait,
      HighlightPass const* highlight = nullptr,
      std::optional<filament::math::float4> clearColor = std::nullopt,
      bool beginEndFrame = true) const;

  // Explicit multi-render frame control. Bracket several offscreen Render() calls with a
  // single BeginFrame()/EndFrame() so they share one Filament beginFrame/endFrame -- i.e.
  // one swapchain present (and one frame-pacing wait) per app frame. Rendering each target
  // in its own frame instead presents/paces multiple times per app frame and proportionally
  // lowers the framerate. While a frame is active, Render() will not open a nested frame
  // regardless of its beginEndFrame argument. Returns false if the frame was skipped for
  // pacing, in which case bracketed Render() calls become no-ops until EndFrame(). Callers
  // that don't use these keep the previous behavior: each Render() manages its own frame.
  bool BeginFrame();
  void EndFrame();

  // Reads the color attachment of @p target back to CPU memory as tightly-packed RGBA8
  // (4 bytes/pixel, width*height*4 total), with the top row first (top-left origin) — ready to
  // hand to a top-left-origin encoder such as PNG without flipping. Synchronous: issues the
  // readback and blocks on flushAndWait() plus the completion callback before returning. @p target
  // must be non-empty and its color texture must have been created with TextureUsage::BLIT_SRC.
  void ReadPixels(RenderTarget const& target, std::vector<uint8_t>& outRgba, mochi::Error& error)
      const;

  // Whether a Render() issued right now would actually execute this app frame. Inside an explicit
  // BeginFrame()/EndFrame() bracket this reflects whether Filament accepted the frame (false when
  // skipped for pacing, so bracketed Render() calls are no-ops). Callers that accumulate state
  // across Render() (e.g. a persistent ping-pong canvas) should skip their work when this is false,
  // otherwise they mutate state for renders that never happen. Outside a bracket, returns true.
  bool IsFrameStarted() const {
    return _frameActive ? _frameStarted : true;
  }

 private:
  Renderer(filament::Engine* engine, filament::Renderer* filamentRenderer, int w, int h);

 private:
  filament::Engine* _engine = nullptr;
  filament::SwapChain* _swapChain = nullptr;
  filament::Renderer* _renderer = nullptr;
  bool _ownsRenderer = false;
  // Set between BeginFrame()/EndFrame(); makes Render() share the active frame instead of
  // opening its own. _frameStarted caches whether beginFrame() accepted the frame.
  // Marked mutable because Render() is const but needs to observe the shared-frame state.
  mutable bool _frameActive = false;
  mutable bool _frameStarted = false;
};

} // namespace superdex::studio
