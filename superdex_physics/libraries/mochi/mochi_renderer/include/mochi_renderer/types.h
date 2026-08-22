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

#include <mochi_core/utils/reflection.h>

#include <cstdint>
#include <vector>

namespace mochi_renderer {

enum class PipelineMode : uint8_t {
  Synchronized, // Block until readback completes. Simplest, highest latency.
  OneFrameDelay, // Return previous frame's images. 1-frame latency.
  MaxPerformance, // Async pipeline with ring of render targets. 2-3 frame latency.
};

enum class ImageFormat : uint8_t {
  RGBA8, // 4 channels, 8 bits per channel
  // Future: DEPTH32F, RGB8, etc.
};

struct RenderResult {
  std::vector<uint8_t> pixels;
  int width = 0;
  int height = 0;
  int channels = 4;
  ImageFormat format = ImageFormat::RGBA8;

  bool IsValid() const {
    return !pixels.empty() && width > 0 && height > 0;
  }
};

// Note: These are the default Filament Tone Mappers without customizations
// there are others, but are only useful if their properties are exposed too
enum class ToneMapperSetting : uint8_t {
  Linear,
  ACES,
  Filmic,
  PBRNeutral,
  GT7, // Fidelity in Gran Turismo 7, SIGGRAPH 2025, by Yasutomi, Suzuki, and Uchimura.
  COUNT, // keep last
};

} // namespace mochi_renderer

// Registered before SceneViewSettings below, which reflects a field of this type.
MOCHI_ENUM_BEGIN(mochi_renderer::ToneMapperSetting)
MOCHI_ENUM_ITEM(Linear)
MOCHI_ENUM_ITEM(ACES)
MOCHI_ENUM_ITEM(Filmic)
MOCHI_ENUM_ITEM(PBRNeutral)
MOCHI_ENUM_ITEM(GT7)
MOCHI_ENUM_END()

namespace mochi_renderer {

// OpenCV/ROS camera model: pinhole intrinsics + radial/tangential distortion.
// Follows the OpenCV calib3d convention and ROS sensor_msgs/CameraInfo format.
// Intrinsics matrix K = [fx 0 cx; 0 fy cy; 0 0 1]
// Distortion coefficients D = [k1, k2, p1, p2, k3, k4, k5, k6]
/// @warning Only 5-parameter distortion is implemented (k1, k2, k3, p1, p2).
/// The rational model coefficients k4, k5, k6 are stored but NOT applied
/// during rendering. Setting them to non-zero values will NOT produce
/// correct distortion. Full 8-parameter support is planned for a future update.
struct OpenCVCameraModel {
  // Focal lengths in pixels
  float fx = 0;
  float fy = 0;
  // Principal point in pixels
  float cx = 0;
  float cy = 0;

  // Radial distortion coefficients
  float k1 = 0;
  float k2 = 0;
  float k3 = 0;
  float k4 = 0;
  float k5 = 0;
  float k6 = 0;
  // Tangential distortion coefficients
  float p1 = 0;
  float p2 = 0;

  bool HasIntrinsics() const {
    return fx > 0 && fy > 0;
  }

  bool HasDistortion() const {
    return k1 != 0 || k2 != 0 || k3 != 0 || k4 != 0 || k5 != 0 || k6 != 0 || p1 != 0 || p2 != 0;
  }
};

struct SceneViewSettings {
  bool msaaEnabled = true;
  int msaaSampleCount = 4;
  bool shadowsEnabled = true;
  bool ssaoEnabled = true;
  bool postProcessingEnabled = true;
  bool bloomEnabled = true;
  float bloomStrength = 0.05f;
  bool vignetteEnabled = true;
  float exposure = 0.0f;
  ToneMapperSetting toneMapper = ToneMapperSetting::Filmic;
  bool showSkybox = false;

  MOCHI_STRUCT_BEGIN(mochi_renderer::SceneViewSettings)
  MOCHI_FIELD(msaaEnabled)
  MOCHI_FIELD(msaaSampleCount)
  MOCHI_FIELD(shadowsEnabled)
  MOCHI_FIELD(ssaoEnabled)
  MOCHI_FIELD(postProcessingEnabled)
  MOCHI_FIELD(bloomEnabled)
  MOCHI_FIELD(bloomStrength)
  MOCHI_FIELD(vignetteEnabled)
  MOCHI_FIELD(exposure)
  MOCHI_FIELD(toneMapper)
  MOCHI_FIELD(showSkybox)
  MOCHI_STRUCT_END()
};

} // namespace mochi_renderer
