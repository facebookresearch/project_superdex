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

#include <mochi_renderer/resource.h>

#include <math/vec3.h>

#include <string>

namespace filament {
class Engine;
class IndirectLight;
class Skybox;
class Texture;
} // namespace filament

namespace utils {
class Path;
}

namespace mochi_renderer {

//--------------------------------------------------------------------------------------------------
// IBL (IMAGE-BASED LIGHTING)
//--------------------------------------------------------------------------------------------------

// An image-based lighting environment: a skybox paired with a precomputed indirect light
// (and optional spherical harmonics / fog texture). Owns its filament resources and is
// referenced (non-owning) by multiple scenes simultaneously via @ref Scene::SetIbl.
// Created and lifetime-managed by @ref ResourceManager (see @ref ResourceManager::LoadIbl).
// Not instanceable.
class IBL : public Resource {
 public:
  ~IBL() override;

  filament::IndirectLight* GetIndirectLight() const;
  filament::Skybox* GetSkybox() const;
  filament::Texture* GetFogTexture() const;
  bool HasSphericalHarmonics() const;
  filament::math::float3 const* GetSphericalHarmonics() const;

 private:
  friend class ResourceManager;
  IBL(filament::Engine* engine, std::string const& name, mochi::Path const& path);

  // Populate this IBL's filament resources from disk. Driven by @ref ResourceManager::LoadIbl.
  bool LoadFromEquirect(mochi::Path const& path);
  bool LoadFromDirectory(mochi::Path const& path);
  bool LoadFromKtx(std::string const& prefix);

  filament::math::float3 _sphericalHarmonics[9] = {};
  bool _hasSphericalHarmonics = false;
  filament::Texture* _reflectionsTexture = nullptr;
  filament::IndirectLight* _indirectLight = nullptr;
  filament::Texture* _skyboxTexture = nullptr;
  filament::Texture* _fogTexture = nullptr;
  filament::Skybox* _skybox = nullptr;
};
} // namespace mochi_renderer
