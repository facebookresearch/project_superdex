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

#include <mochi_renderer/ibl.h>

#include <mochi_core/utils/debug.h>

#include <filament/Engine.h>
#include <filament/IndirectLight.h>
#include <filament/Skybox.h>
#include <filament/Texture.h>

#include <filament-iblprefilter/IBLPrefilterContext.h>
#include <imageio/ImageDecoder.h>
#include <ktxreader/Ktx1Reader.h>

#include <stb_image.h>

#include <utils/Path.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <string>

using namespace filament;
using namespace filament::math;
using namespace ktxreader;
using namespace utils;

namespace {

constexpr float kIblIntensity = 30000.0f;

// The image decoding, cubemap prefiltering, and KTX/cubemap loading below are derived from
// Filament's sample IBL utility (https://github.com/google/filament), Apache-2.0 licensed.

// Loads a single cubemap mip level (6 faces) from disk into a freshly built (level 0) or
// existing texture. The detailed overload fills the caller-provided pixel buffer and texture
// dimension; the convenience overload uploads the result to the texture immediately.
bool LoadCubemapLevel(
    Engine& engine,
    Texture** texture,
    Texture::PixelBufferDescriptor* outBuffer,
    uint32_t* dim,
    Path const& path,
    size_t level,
    std::string const& levelPrefix) {
  static char const* faceSuffix[6] = {"px", "nx", "py", "ny", "pz", "nz"};

  size_t size = 0;
  size_t numLevels = 1;

  { // this is just a scope to avoid variable name hiding below
    int w = 0, h = 0;
    std::string faceName = levelPrefix + faceSuffix[0] + ".rgb32f";
    Path facePath(Path::concat(path, faceName));
    if (!facePath.exists()) {
      MOCHI_LOG_ERROR("[IBL] The face %s does not exist", faceName.c_str());
      return false;
    }
    stbi_info(facePath.getAbsolutePath().c_str(), &w, &h, nullptr);
    if (w != h) {
      MOCHI_LOG_ERROR("[IBL] Cubemap face width != height");
      return false;
    }

    size = (size_t)w;

    if (!levelPrefix.empty()) {
      numLevels = (size_t)std::log2(size) + 1;
    }

    if (level == 0) {
      *texture = Texture::Builder()
                     .width((uint32_t)size)
                     .height((uint32_t)size)
                     .levels((uint8_t)numLevels)
                     .format(Texture::InternalFormat::R11F_G11F_B10F)
                     .sampler(Texture::Sampler::SAMPLER_CUBEMAP)
                     .build(engine);
    }
  }

  // RGB_10_11_11_REV encoding: 4 bytes per pixel
  size_t const faceSize = size * size * sizeof(uint32_t);
  *dim = size;

  Texture::PixelBufferDescriptor buffer(
      malloc(faceSize * 6),
      faceSize * 6,
      Texture::Format::RGB,
      Texture::Type::UINT_10F_11F_11F_REV,
      (Texture::PixelBufferDescriptor::Callback)&free);

  bool success = true;
  auto* p = static_cast<uint8_t*>(buffer.buffer);

  for (size_t j = 0; j < 6; j++) {
    std::string faceName = levelPrefix + faceSuffix[j] + ".rgb32f";
    Path facePath(Path::concat(path, faceName));
    if (!facePath.exists()) {
      MOCHI_LOG_ERROR("[IBL] The face %s does not exist", faceName.c_str());
      success = false;
      break;
    }

    int w = 0, h = 0, n = 0;
    unsigned char* data = stbi_load(facePath.getAbsolutePath().c_str(), &w, &h, &n, 4);
    if (w != h || w != (int)size) {
      MOCHI_LOG_ERROR(
          "[IBL] Face %s has a wrong size %d x %d, instead of %zu x %zu",
          faceName.c_str(),
          w,
          h,
          size,
          size);
      success = false;
      break;
    }

    if (data == nullptr || n != 4) {
      MOCHI_LOG_ERROR("[IBL] Could not decode face %s", faceName.c_str());
      success = false;
      break;
    }

    memcpy(p + faceSize * j, data, (size_t)w * h * sizeof(uint32_t));

    stbi_image_free(data);
  }

  if (!success) {
    return false;
  }

  *outBuffer = std::move(buffer);

  return true;
}

bool LoadCubemapLevel(
    Engine& engine,
    Texture** texture,
    Path const& path,
    size_t level = 0,
    std::string const& levelPrefix = "") {
  uint32_t dim = 0;
  Texture::PixelBufferDescriptor buffer;
  if (LoadCubemapLevel(engine, texture, &buffer, &dim, path, level, levelPrefix)) {
    (*texture)->setImage(engine, level, 0, 0, 0, dim, dim, 6, std::move(buffer));
    return true;
  }
  return false;
}

} // namespace

namespace mochi_renderer {

IBL::IBL(filament::Engine* engine, std::string const& name, mochi::Path const& path)
    : Resource(engine, name, path, ResourceType::Ibl) {}

IBL::~IBL() {
  _engine->destroy(_indirectLight);
  _engine->destroy(_reflectionsTexture);
  _engine->destroy(_skybox);
  _engine->destroy(_skyboxTexture);
  _engine->destroy(_fogTexture);
}

filament::IndirectLight* IBL::GetIndirectLight() const {
  return _indirectLight;
}

filament::Skybox* IBL::GetSkybox() const {
  return _skybox;
}

filament::Texture* IBL::GetFogTexture() const {
  return _fogTexture;
}

bool IBL::HasSphericalHarmonics() const {
  return _hasSphericalHarmonics;
}

filament::math::float3 const* IBL::GetSphericalHarmonics() const {
  return _sphericalHarmonics;
}

bool IBL::LoadFromEquirect(mochi::Path const& path) {
  utils::Path filaPath(path.ToString());
  if (!filaPath.exists()) {
    return false;
  }

  int w = 0, h = 0;
  int n = 0;
  size_t size = 0;
  void* data = nullptr;
  void* user = nullptr;
  Texture::PixelBufferDescriptor::Callback destroyer{};

  if (filaPath.getExtension() == "exr") {
    std::ifstream in_stream(filaPath.getAbsolutePath().c_str(), std::ios::binary);
    auto* image = new image::LinearImage(
        image::ImageDecoder::decode(in_stream, filaPath.getAbsolutePath().c_str()));
    w = image->getWidth();
    h = image->getHeight();
    n = image->getChannels();
    size = w * h * n * sizeof(float);
    data = image->getPixelRef();
    user = image;
    destroyer = [](void*, size_t, void* user) {
      delete reinterpret_cast<image::LinearImage*>(user);
    };
  } else {
    stbi_info(filaPath.getAbsolutePath().c_str(), &w, &h, nullptr);
    // load image as float
    size = w * h * sizeof(float3);
    data = (float3*)stbi_loadf(filaPath.getAbsolutePath().c_str(), &w, &h, &n, 3);
    destroyer = [](void* data, size_t, void*) { stbi_image_free(data); };
  }

  if (data == nullptr || n != 3) {
    MOCHI_LOG_ERROR("[IBL] Could not decode image: %s", filaPath.getAbsolutePath().c_str());
    destroyer(data, size, user);
    return false;
  }

  if (w != h * 2) {
    MOCHI_LOG_ERROR("[IBL] Not an equirectangular image: %s", filaPath.getAbsolutePath().c_str());
    destroyer(data, size, user);
    return false;
  }

  // now load texture
  Texture::PixelBufferDescriptor buffer(
      data, size, Texture::Format::RGB, Texture::Type::FLOAT, destroyer, user);

  Texture* const equirect = Texture::Builder()
                                .width((uint32_t)w)
                                .height((uint32_t)h)
                                .levels(0xff)
                                .format(Texture::InternalFormat::R11F_G11F_B10F)
                                .sampler(Texture::Sampler::SAMPLER_2D)
                                .usage(Texture::Usage::DEFAULT | Texture::Usage::GEN_MIPMAPPABLE)
                                .build(*_engine);

  equirect->setImage(*_engine, 0, std::move(buffer));

  IBLPrefilterContext context(*_engine);
  IBLPrefilterContext::EquirectangularToCubemap equirectangularToCubemap(context);
  IBLPrefilterContext::SpecularFilter specularFilter(context);
  IBLPrefilterContext::IrradianceFilter irradianceFilter(context);

  _skyboxTexture = equirectangularToCubemap(equirect);

  _engine->destroy(equirect);

  _reflectionsTexture = specularFilter(_skyboxTexture);

  _fogTexture = irradianceFilter({.generateMipmap = true}, _skyboxTexture);
  _fogTexture->generateMipmaps(*_engine);

  _indirectLight = IndirectLight::Builder()
                       .reflections(_reflectionsTexture)
                       .intensity(kIblIntensity)
                       .build(*_engine);

  _skybox = Skybox::Builder().environment(_skyboxTexture).showSun(true).build(*_engine);

  return true;
}

bool IBL::LoadFromKtx(std::string const& prefix) {
  utils::Path iblPath(prefix + "_ibl.ktx");
  if (!iblPath.exists()) {
    return false;
  }
  utils::Path skyPath(prefix + "_skybox.ktx");
  if (!skyPath.exists()) {
    return false;
  }

  auto createKtx = [](utils::Path const& path) {
    std::ifstream file(path.getPath(), std::ios::binary);
    std::vector<uint8_t> contents((std::istreambuf_iterator<char>(file)), {});
    return std::make_unique<image::Ktx1Bundle>(contents.data(), contents.size());
  };

  std::unique_ptr<image::Ktx1Bundle> iblKtx = createKtx(iblPath);
  std::unique_ptr<image::Ktx1Bundle> skyKtx = createKtx(skyPath);

  // Read spherical harmonics before handing the bundles to createTexture, which takes
  // ownership and destroys them after the (async) GPU upload completes.
  if (!iblKtx->getSphericalHarmonics(_sphericalHarmonics)) {
    return false;
  }
  _hasSphericalHarmonics = true;

  // createTexture takes ownership of the bundle, so release it from the unique_ptr.
  _skyboxTexture = Ktx1Reader::createTexture(_engine, skyKtx.release(), false);
  _reflectionsTexture = Ktx1Reader::createTexture(_engine, iblKtx.release(), false);

  _indirectLight = IndirectLight::Builder()
                       .reflections(_reflectionsTexture)
                       .intensity(kIblIntensity)
                       .build(*_engine);

  _skybox = Skybox::Builder().environment(_skyboxTexture).showSun(true).build(*_engine);

  return true;
}

bool IBL::LoadFromDirectory(mochi::Path const& path) {
  utils::Path dir(path.ToString());
  // First check if KTX files are available.
  if (LoadFromKtx(utils::Path::concat(dir, dir.getName()))) {
    return true;
  }
  // Read spherical harmonics
  utils::Path sh(utils::Path::concat(dir, "sh.txt"));
  if (sh.exists()) {
    std::ifstream shReader(sh);
    shReader >> std::skipws;

    std::string line;
    for (float3& band : _sphericalHarmonics) {
      std::getline(shReader, line);
      int n = sscanf(line.c_str(), "(%f,%f,%f)", &band.r, &band.g, &band.b); // NOLINT(cert-err34-c)
      if (n != 3) {
        return false;
      }
    }
  } else {
    return false;
  }
  _hasSphericalHarmonics = true;

  // Read mip-mapped cubemap
  std::string const prefix = "m";
  if (!LoadCubemapLevel(*_engine, &_reflectionsTexture, dir, 0, prefix + "0_")) {
    return false;
  }

  size_t numLevels = _reflectionsTexture->getLevels();
  for (size_t i = 1; i < numLevels; i++) {
    std::string const levelPrefix = prefix + std::to_string(i) + "_";
    LoadCubemapLevel(*_engine, &_reflectionsTexture, dir, i, levelPrefix);
  }

  if (!LoadCubemapLevel(*_engine, &_skyboxTexture, dir)) {
    return false;
  }

  _indirectLight = IndirectLight::Builder()
                       .reflections(_reflectionsTexture)
                       .irradiance(3, _sphericalHarmonics)
                       .intensity(kIblIntensity)
                       .build(*_engine);

  _skybox = Skybox::Builder().environment(_skyboxTexture).showSun(true).build(*_engine);

  return true;
}

} // namespace mochi_renderer
