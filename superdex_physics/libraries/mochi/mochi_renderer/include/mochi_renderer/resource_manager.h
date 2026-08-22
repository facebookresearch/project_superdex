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

#include <mochi_renderer/material.h>
#include <mochi_renderer/path.h>
#include <mochi_renderer/resource.h>

#include <filament/Engine.h>

#include <gltfio/MaterialProvider.h>
#include <gltfio/ResourceLoader.h>

#include <utils/NameComponentManager.h>
#include <utils/Path.h>

#include <map>
#include <vector>

#include <utils/unwindows.h> // Clean up Windows macros (near, far, OPAQUE, etc.)

namespace mochi {
class CoordinateSpaceConverter;
} // namespace mochi

namespace mochi_renderer {

struct MeshSection;

class IBL;

class ResourceManager {
 public:
  static std::unique_ptr<ResourceManager> Create(
      filament::Engine* engine,
      mochi::CoordinateSpaceConverter* spaceConverter = nullptr);

  ResourceManager(ResourceManager const&) = delete;
  ResourceManager& operator=(ResourceManager const&) = delete;
  ResourceManager(ResourceManager&&) = delete;
  ResourceManager& operator=(ResourceManager&&) = delete;
  ~ResourceManager();

  filament::Engine* GetEngine() const {
    return _engine;
  }

  mochi::CoordinateSpaceConverter const* GetSpaceConverter() const {
    return _spaceConverter;
  }

  using ResourceMap = std::map<mochi::Path, std::unique_ptr<Resource>>;

  std::shared_ptr<MaterialInstance> CreateLitOpaqueMaterial(
      filament::math::float3 baseColor,
      float roughness = 0.5f,
      float metallic = 0.0f) const;
  std::shared_ptr<MaterialInstance> CreateFlatLitOpaqueMaterial(
      filament::math::float3 baseColor,
      float roughness = 0.5f,
      float metallic = 0.0f) const;
  std::shared_ptr<MaterialInstance> CreateLitSeeThroughMaterial(
      filament::math::float3 baseColor,
      float alpha = 0.5f,
      float roughness = 0.5f,
      float metallic = 0.0f) const;
  std::shared_ptr<MaterialInstance> CreateFlatLitSeeThroughMaterial(
      filament::math::float3 baseColor,
      float alpha = 0.5f,
      float roughness = 0.5f,
      float metallic = 0.0f) const;
  std::shared_ptr<MaterialInstance> CreateOutlineMaterial(
      filament::math::float4 color,
      float thickness = 0.003f) const;
  std::shared_ptr<MaterialInstance> CreateUnlitSeeThroughMaterial(
      filament::math::float4 color) const;
  std::shared_ptr<MaterialInstance> CreateWireframeMaterial(
      filament::math::float4 color,
      float lineWidth = 1.0f,
      float depthBias = 0.0005f) const;
  std::shared_ptr<MaterialInstance> CreateWireframeDepthMaterial() const;
  // Fullscreen composite material for the highlight overlay pass: samples @p sourceTexture (the
  // isolated highlight render) and blends it over the scene at global opacity @p alpha. The caller
  // rebinds sourceTexture when the overlay target is resized.
  std::shared_ptr<MaterialInstance> CreateHighlightCompositeMaterial(
      filament::Texture const* sourceTexture,
      float alpha) const;

  void ForEachResource(std::function<void(Resource*, mochi::Path const&)> const& callback) const;
  Resource* LoadResource(mochi::Path const& path);
  RenderModel* LoadRenderModel(mochi::Path const& path);
  RenderModel* LoadGltf(mochi::Path const& path, bool async = false);
  RenderModel* LoadStl(mochi::Path const& path);
  RenderModel* LoadObj(mochi::Path const& path);
  RenderModel* LoadCollada(mochi::Path const& path);
  RenderModel* LoadMochiModel(mochi::Path const& path);
  RenderModel* LoadMochiModel(mochi::Path const& path, mochi::ModelData const& modelData);
  // Builds a render model from in-memory mesh sections (no file on disk). @p path is used only as
  // the resource-map key and the model's name. Used by assets whose geometry is generated in
  // memory (e.g. tessellated CAD models).
  RenderModel* LoadModelFromMeshSections(
      mochi::Path const& path,
      std::vector<MeshSection> sections,
      RenderModelFormat format);
  IBL* LoadIbl(mochi::Path const& path);
  Resource* FindResourceByPath(mochi::Path const& path, bool silent = true) const;
  template <typename TResource>
  TResource* FindResourceByPath(mochi::Path const& path, bool silent = true) const {
    return dynamic_cast<TResource*>(FindResourceByPath(path, silent));
  }
  bool UnloadResource(mochi::Path const& path);
  void UnloadAllResources();
  Resource* RegisterResource(mochi::Path const& path, std::unique_ptr<Resource> resource);
  bool RewriteResourcePath(mochi::Path const& oldPath, mochi::Path const& newPath);

  void PumpAsyncLoad() const;
  void WaitAsyncLoad() const;

 private:
  ResourceManager(filament::Engine* engine, mochi::CoordinateSpaceConverter* spaceConverter);
  filament::gltfio::FilamentAsset* LoadFilamentGltfAsset(
      utils::Path const& filename,
      FilamentInstanceVector& instances) const;
  filament::gltfio::FilamentAsset* LoadFilamentGltfAssetFromBuffer(
      std::vector<uint8_t> const& glbData,
      FilamentInstanceVector& instances) const;
  bool LoadFilamentGltfResources(
      utils::Path const& filename,
      filament::gltfio::FilamentAsset* asset,
      bool async) const;
  // Parses @p path with @p reader and builds a render model from the resulting sections. When
  // @p sourceIsMochiSpace the geometry is brought into @ref RenderSpace via
  // @ref SourceToRenderConverter first; pass false for a reader that already emits renderer-space
  // geometry.
  RenderModel* LoadMeshAsset(
      mochi::Path const& path,
      std::vector<MeshSection> (*reader)(char const*),
      RenderModelFormat format,
      char const* label,
      bool sourceIsMochiSpace);

  // Maps geometry authored in Mochi's space (@ref mochi::CoordinateSpace::Default) into
  // @ref RenderSpace: the converter the owner injected, or the default mapping when there is none.
  mochi::CoordinateSpaceConverter const& SourceToRenderConverter() const;

 private:
  filament::Engine* _engine = nullptr;
  utils::NameComponentManager* _nameManager = nullptr;

  filament::gltfio::MaterialProvider* _gltfMaterialProvider = nullptr;
  filament::gltfio::AssetLoader* _gltfAssetLoader = nullptr;
  filament::gltfio::ResourceLoader* _gltfResourceLoader = nullptr;
  filament::gltfio::TextureProvider* _gltfStbProvider = nullptr;
  filament::gltfio::TextureProvider* _gltfKtxProvider = nullptr;

  filament::Material* _materialLitOpaque = nullptr;
  filament::Material* _materialFlatLitOpaque = nullptr;
  filament::Material* _materialLitSeeThrough = nullptr;
  filament::Material* _materialFlatLitSeeThrough = nullptr;
  filament::Material* _materialOutline = nullptr;
  filament::Material* _materialUnlitSeeThrough = nullptr;
  filament::Material* _materialWireframe = nullptr;
  filament::Material* _materialWireframeDepth = nullptr;
  filament::Material* _materialHighlightComposite = nullptr;

  mochi::CoordinateSpaceConverter* _spaceConverter = nullptr;
  ResourceMap _resources;
};

} // namespace mochi_renderer
