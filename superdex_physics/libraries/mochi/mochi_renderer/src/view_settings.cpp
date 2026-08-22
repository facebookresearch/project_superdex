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

#include <mochi_renderer/windows_compat.h> // Must be first — cleans up Windows macros before Filament headers

#include "view_settings.h"

#include <filament/ColorGrading.h>
#include <filament/ColorSpace.h>
#include <filament/Engine.h>
#include <filament/ToneMapper.h>
#include <filament/View.h>

#include <memory>

namespace mochi_renderer {

namespace {

filament::ToneMapper const* CreateToneMapper(ToneMapperSetting setting) {
  switch (setting) {
    case ToneMapperSetting::Linear:
      return new filament::LinearToneMapper;
    case ToneMapperSetting::ACES:
      return new filament::ACESToneMapper;
    case ToneMapperSetting::Filmic:
      return new filament::FilmicToneMapper;
    case ToneMapperSetting::PBRNeutral:
      return new filament::PBRNeutralToneMapper;
    case ToneMapperSetting::GT7:
      return new filament::GT7ToneMapper;
    case ToneMapperSetting::COUNT:
      return new filament::LinearToneMapper;
  }
  // Unreachable: the switch above is exhaustive over the enum.
  return new filament::LinearToneMapper;
}

} // namespace

void ApplyViewSettingsToView(
    filament::Engine* engine,
    filament::View* view,
    filament::ColorGrading*& colorGrading,
    SceneViewSettings const& settings) {
  // Post-processing must be enabled for bloom, SSAO, tone mapping, etc.
  view->setPostProcessingEnabled(settings.postProcessingEnabled);

  filament::View::MultiSampleAntiAliasingOptions msaaOptions;
  msaaOptions.enabled = settings.msaaEnabled;
  msaaOptions.sampleCount = settings.msaaSampleCount;
  view->setMultiSampleAntiAliasingOptions(msaaOptions);

  filament::View::BloomOptions bloomOptions;
  bloomOptions.enabled = settings.bloomEnabled;
  bloomOptions.strength = settings.bloomStrength;
  bloomOptions.resolution = 360;
  bloomOptions.levels = 6;
  bloomOptions.blendMode = filament::View::BloomOptions::BlendMode::ADD;
  view->setBloomOptions(bloomOptions);

  // Ambient occlusion: GTAO (not the default SAO) so occlusion is stable under view changes instead
  // of popping between discrete levels as the camera moves. SAO selects a depth mip from the
  // world-space radius projected to screen space, so crease occlusion jumped between levels on
  // rotate/dolly. GTAO integrates over screen-space slices; more slices (rotational stability) and
  // steps per slice (distance stability) than the default keep it steady. A full-resolution AO
  // buffer plus a HIGH low-pass filter remove upsampling shimmer. Screen-space contact shadows stay
  // off: being screen-space, their darkening is view-dependent.
  filament::View::AmbientOcclusionOptions ssaoOptions;
  ssaoOptions.enabled = settings.ssaoEnabled;
  ssaoOptions.aoType = filament::View::AmbientOcclusionOptions::AmbientOcclusionType::GTAO;
  ssaoOptions.radius = 0.1f;
  ssaoOptions.power = 1.0f;
  ssaoOptions.intensity = 1.0f;
  ssaoOptions.resolution = 1.0f;
  ssaoOptions.quality = filament::View::QualityLevel::HIGH;
  ssaoOptions.lowPassFilter = filament::View::QualityLevel::HIGH;
  ssaoOptions.upsampling = filament::View::QualityLevel::HIGH;
  ssaoOptions.gtao.sampleSliceCount = 8;
  ssaoOptions.gtao.sampleStepsPerSlice = 4;
  ssaoOptions.ssct.enabled = false;
  view->setAmbientOcclusionOptions(ssaoOptions);

  filament::View::VignetteOptions vignetteOptions;
  vignetteOptions.enabled = settings.vignetteEnabled;
  vignetteOptions.midPoint = 0.75f;
  view->setVignetteOptions(vignetteOptions);

  view->setShadowingEnabled(settings.shadowsEnabled);
  view->setShadowType(filament::View::ShadowType::PCF);
  filament::View::VsmShadowOptions vsmOptions;
  vsmOptions.anisotropy = 1;
  view->setVsmShadowOptions(vsmOptions);

  {
    using namespace filament::color;
    std::unique_ptr<filament::ToneMapper const> toneMapper(CreateToneMapper(settings.toneMapper));
    if (colorGrading) {
      engine->destroy(colorGrading);
    }
    colorGrading = filament::ColorGrading::Builder()
                       .quality(filament::ColorGrading::QualityLevel::HIGH)
                       .exposure(settings.exposure)
                       .outputColorSpace(Rec709 - sRGB - D65) // yes, operator `-` is overloaded
                       .toneMapper(toneMapper.get())
                       .build(*engine);
  }
  view->setColorGrading(colorGrading);
}

} // namespace mochi_renderer
