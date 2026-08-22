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
#include <mochi_renderer/mesh.h>
#include <mochi_renderer/render_space.h>
#include <mochi_renderer/resource_manager.h>
#include <mochi_renderer/utils.h>

#include "materials.h"

#include <filament/RenderableManager.h>
#include <filament/Texture.h>
#include <filament/TextureSampler.h>
#include <gltfio/TextureProvider.h>
#include <materials/uberarchive.h>
#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/debug.h>

#include <cgltf.h>
#include <fstream>
#include <set>

#include "filament/TransformManager.h"
#include "mochi_core/geometry/tetrahedral_mesh.h"
#include "mochi_core/utils/hdf5_utils.h"

namespace mochi_renderer {

namespace {

utils::Path GetPathForGltfAsset(std::string_view string) {
  auto isGLTF = [](utils::Path file) -> bool {
    return file.getExtension() == "gltf" || file.getExtension() == "glb";
  };

  MOCHI_LOG("[gltf] GetPathForGltfAsset: input = '%.*s'", (int)string.size(), string.data());
  utils::Path filename{string};
  if (!filename.exists()) {
    MOCHI_LOG_ERROR("[gltf] File not found: %s", filename.c_str());
    return {};
  }
  MOCHI_LOG("[gltf] File exists: %s", filename.c_str());

  if (filename.isDirectory()) {
    MOCHI_LOG("[gltf] Path is a directory, scanning for glTF files...");
    std::vector<utils::Path> files = filename.listContents();
    auto it = std::find_if(files.cbegin(), files.cend(), isGLTF);
    if (it == files.end()) {
      MOCHI_LOG_ERROR("[gltf] No glTF/glb file found in directory: %s", filename.c_str());
      return {};
    }
    filename = *it;
    MOCHI_LOG("[gltf] Found glTF in directory: %s", filename.c_str());
  } else if (!isGLTF(filename)) {
    MOCHI_LOG_ERROR(
        "[gltf] File is not a glTF/glb: %s (extension: %s)",
        filename.c_str(),
        filename.getExtension().c_str());
    return {};
  }

  MOCHI_LOG("[gltf] Resolved glTF path: %s", filename.c_str());
  return filename;
}

std::ifstream::pos_type GetFileSize(char const* filename) {
  std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
  return in.tellg();
}

bool CheckGltfAsset(utils::Path const& filename) {
  MOCHI_LOG("[gltf] CheckGltfAsset: %s", filename.c_str());
  // Peek at the file size to allow pre-allocation.
  long const contentSize = static_cast<long>(GetFileSize(filename.c_str()));
  if (contentSize <= 0) {
    MOCHI_LOG_ERROR("[gltf] Unable to open (size=%ld): %s", contentSize, filename.c_str());
    return false;
  }
  MOCHI_LOG("[gltf] File size: %ld bytes", contentSize);
  // Consume the glTF file.
  std::ifstream in(filename.c_str(), std::ifstream::binary | std::ifstream::in);
  std::vector<uint8_t> buffer(static_cast<unsigned long>(contentSize));
  if (!in.read((char*)buffer.data(), contentSize)) {
    MOCHI_LOG_ERROR("[gltf] Unable to read: %s", filename.c_str());
    return false;
  }
  // Try parsing the glTF file to check the validity of the file format.
  cgltf_options options{};
  cgltf_data* sourceAsset = nullptr;
  cgltf_result result = cgltf_parse(&options, buffer.data(), contentSize, &sourceAsset);
  cgltf_free(sourceAsset);
  if (result != cgltf_result_success) {
    MOCHI_LOG(
        "[gltf] Unable to parse glTF file (cgltf_result=%d): %s", (int)result, filename.c_str());
    return false;
  }
  MOCHI_LOG("[gltf] glTF parse OK: %s", filename.c_str());
  return true;
};

} // namespace

std::unique_ptr<ResourceManager> ResourceManager::Create(
    filament::Engine* engine,
    mochi::CoordinateSpaceConverter* spaceConverter) {
  MOCHI_ASSERT(engine != nullptr);
  return std::unique_ptr<ResourceManager>(new ResourceManager(engine, spaceConverter));
}

ResourceManager::ResourceManager(
    filament::Engine* engine,
    mochi::CoordinateSpaceConverter* spaceConverter)
    : _spaceConverter(spaceConverter) {
  _engine = engine;
  _gltfMaterialProvider = filament::gltfio::createUbershaderProvider(
      _engine, UBERARCHIVE_DEFAULT_DATA, UBERARCHIVE_DEFAULT_SIZE);
  _nameManager = new utils::NameComponentManager(utils::EntityManager::get());
  _gltfAssetLoader =
      filament::gltfio::AssetLoader::create({_engine, _gltfMaterialProvider, _nameManager});
  filament::gltfio::ResourceConfiguration config{};
  config.engine = _engine;
  config.normalizeSkinningWeights = true;
  _gltfResourceLoader = new filament::gltfio::ResourceLoader(config);
  _gltfStbProvider = filament::gltfio::createStbProvider(_engine);
  _gltfKtxProvider = filament::gltfio::createKtx2Provider(_engine);
  _gltfResourceLoader->addTextureProvider("image/png", _gltfStbProvider);
  _gltfResourceLoader->addTextureProvider("image/jpeg", _gltfStbProvider);
  _gltfResourceLoader->addTextureProvider("image/ktx2", _gltfKtxProvider);

  _materialLitOpaque =
      filament::Material::Builder()
          .package(MOCHI_RENDERER_MATERIALS_LITOPAQUE_DATA, MOCHI_RENDERER_MATERIALS_LITOPAQUE_SIZE)
          .build(*_engine);
  _materialFlatLitOpaque = filament::Material::Builder()
                               .package(
                                   MOCHI_RENDERER_MATERIALS_FLATLITOPAQUE_DATA,
                                   MOCHI_RENDERER_MATERIALS_FLATLITOPAQUE_SIZE)
                               .build(*_engine);
  _materialLitSeeThrough = filament::Material::Builder()
                               .package(
                                   MOCHI_RENDERER_MATERIALS_LITSEETHROUGH_DATA,
                                   MOCHI_RENDERER_MATERIALS_LITSEETHROUGH_SIZE)
                               .build(*_engine);
  _materialFlatLitSeeThrough = filament::Material::Builder()
                                   .package(
                                       MOCHI_RENDERER_MATERIALS_FLATLITSEETHROUGH_DATA,
                                       MOCHI_RENDERER_MATERIALS_FLATLITSEETHROUGH_SIZE)
                                   .build(*_engine);
  _materialOutline =
      filament::Material::Builder()
          .package(MOCHI_RENDERER_MATERIALS_OUTLINE_DATA, MOCHI_RENDERER_MATERIALS_OUTLINE_SIZE)
          .build(*_engine);
  _materialUnlitSeeThrough = filament::Material::Builder()
                                 .package(
                                     MOCHI_RENDERER_MATERIALS_UNLITSEETHROUGH_DATA,
                                     MOCHI_RENDERER_MATERIALS_UNLITSEETHROUGH_SIZE)
                                 .build(*_engine);
  _materialWireframe =
      filament::Material::Builder()
          .package(MOCHI_RENDERER_MATERIALS_WIREFRAME_DATA, MOCHI_RENDERER_MATERIALS_WIREFRAME_SIZE)
          .build(*_engine);
  _materialWireframeDepth = filament::Material::Builder()
                                .package(
                                    MOCHI_RENDERER_MATERIALS_WIREFRAMEDEPTH_DATA,
                                    MOCHI_RENDERER_MATERIALS_WIREFRAMEDEPTH_SIZE)
                                .build(*_engine);
  _materialHighlightComposite = filament::Material::Builder()
                                    .package(
                                        MOCHI_RENDERER_MATERIALS_HIGHLIGHTCOMPOSITE_DATA,
                                        MOCHI_RENDERER_MATERIALS_HIGHLIGHTCOMPOSITE_SIZE)
                                    .build(*_engine);
}

ResourceManager::~ResourceManager() {
  _resources.clear();
  _engine->destroy(_materialLitOpaque);
  _engine->destroy(_materialFlatLitOpaque);
  _engine->destroy(_materialLitSeeThrough);
  _engine->destroy(_materialFlatLitSeeThrough);
  _engine->destroy(_materialOutline);
  _engine->destroy(_materialUnlitSeeThrough);
  _engine->destroy(_materialWireframe);
  _engine->destroy(_materialWireframeDepth);
  _engine->destroy(_materialHighlightComposite);
  filament::gltfio::AssetLoader::destroy(&_gltfAssetLoader);
  _gltfMaterialProvider->destroyMaterials();
  delete _gltfMaterialProvider;
  delete _nameManager;
  delete _gltfResourceLoader;
  delete _gltfStbProvider;
  delete _gltfKtxProvider;
}

std::shared_ptr<MaterialInstance> ResourceManager::CreateLitOpaqueMaterial(
    filament::math::float3 baseColor,
    float roughness,
    float metallic) const {
  auto* mi = _materialLitOpaque->createInstance();
  mi->setParameter("baseColor", baseColor);
  mi->setParameter("roughness", roughness);
  mi->setParameter("metallic", metallic);
  return std::make_shared<MaterialInstance>(_engine, mi);
}

std::shared_ptr<MaterialInstance> ResourceManager::CreateFlatLitOpaqueMaterial(
    filament::math::float3 baseColor,
    float roughness,
    float metallic) const {
  auto* mi = _materialFlatLitOpaque->createInstance();
  mi->setParameter("baseColor", baseColor);
  mi->setParameter("roughness", roughness);
  mi->setParameter("metallic", metallic);
  return std::make_shared<MaterialInstance>(_engine, mi);
}

std::shared_ptr<MaterialInstance> ResourceManager::CreateLitSeeThroughMaterial(
    filament::math::float3 baseColor,
    float alpha,
    float roughness,
    float metallic) const {
  auto* mi = _materialLitSeeThrough->createInstance();
  mi->setParameter("baseColor", baseColor);
  mi->setParameter("alpha", alpha);
  mi->setParameter("roughness", roughness);
  mi->setParameter("metallic", metallic);
  // The .mat bakes depthCulling off, so transparents otherwise ignore the depth buffer. Re-enable
  // both depth test and depth write so see-through surfaces are depth-correct: opaque geometry in
  // front occludes them and nearer parts of a mesh obscure farther parts. This is single-layer
  // translucency (only the nearest surface shows at each pixel) -- the cleanest option that keeps
  // correct ordering and color. Correctly showing stacked interior layers would require
  // order-independent transparency, which Filament does not provide here.
  mi->setDepthCulling(true);
  mi->setDepthWrite(true);
  return std::make_shared<MaterialInstance>(_engine, mi);
}

std::shared_ptr<MaterialInstance> ResourceManager::CreateFlatLitSeeThroughMaterial(
    filament::math::float3 baseColor,
    float alpha,
    float roughness,
    float metallic) const {
  auto* mi = _materialFlatLitSeeThrough->createInstance();
  mi->setParameter("baseColor", baseColor);
  mi->setParameter("alpha", alpha);
  mi->setParameter("roughness", roughness);
  mi->setParameter("metallic", metallic);
  // Depth-correct (test + write) single-layer translucency; see CreateLitSeeThroughMaterial.
  mi->setDepthCulling(true);
  mi->setDepthWrite(true);
  return std::make_shared<MaterialInstance>(_engine, mi);
}

std::shared_ptr<MaterialInstance> ResourceManager::CreateOutlineMaterial(
    filament::math::float4 color,
    float thickness) const {
  auto* mi = _materialOutline->createInstance();
  mi->setParameter("color", color);
  mi->setParameter("thickness", thickness);
  return std::make_shared<MaterialInstance>(_engine, mi);
}

std::shared_ptr<MaterialInstance> ResourceManager::CreateUnlitSeeThroughMaterial(
    filament::math::float4 color) const {
  auto* mi = _materialUnlitSeeThrough->createInstance();
  mi->setParameter("color", color);
  // Depth-correct (test + write) single-layer translucency; see CreateLitSeeThroughMaterial.
  mi->setDepthCulling(true);
  mi->setDepthWrite(true);
  return std::make_shared<MaterialInstance>(_engine, mi);
}

std::shared_ptr<MaterialInstance> ResourceManager::CreateWireframeMaterial(
    filament::math::float4 color,
    float lineWidth,
    float depthBias) const {
  auto* mi = _materialWireframe->createInstance();
  mi->setParameter("color", color);
  mi->setParameter("lineWidth", lineWidth);
  mi->setParameter("depthBias", depthBias);
  return std::make_shared<MaterialInstance>(_engine, mi);
}

std::shared_ptr<MaterialInstance> ResourceManager::CreateWireframeDepthMaterial() const {
  auto* mi = _materialWireframeDepth->createInstance();
  mi->setColorWrite(false);
  mi->setDepthWrite(true);
  return std::make_shared<MaterialInstance>(_engine, mi);
}

std::shared_ptr<MaterialInstance> ResourceManager::CreateHighlightCompositeMaterial(
    filament::Texture const* sourceTexture,
    float alpha) const {
  auto* mi = _materialHighlightComposite->createInstance();
  filament::TextureSampler const sampler(
      filament::TextureSampler::MinFilter::LINEAR, filament::TextureSampler::MagFilter::LINEAR);
  mi->setParameter("sourceTexture", sourceTexture, sampler);
  mi->setParameter("alpha", alpha);
  return std::make_shared<MaterialInstance>(_engine, mi);
}

void ResourceManager::ForEachResource(
    std::function<void(Resource*, mochi::Path const&)> const& callback) const {
  for (auto const& [path, resource] : _resources) {
    callback(resource.get(), path);
  }
}

Resource* ResourceManager::LoadResource(mochi::Path const& path) {
  if (_resources.contains(path)) {
    MOCHI_LOG_VERBOSE("Resource is already loaded from path: %s", path.ToString().c_str());
    return _resources[path].get();
  }
  std::string extension = path.GetExtension();
  std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
    return std::tolower(c);
  });
  if (extension == ".gltf" || extension == ".glb") {
    return LoadGltf(path);
  }
  if (extension == ".stl") {
    return LoadStl(path);
  }
  if (extension == ".obj") {
    return LoadObj(path);
  }
  if (extension == ".dae") {
    return LoadCollada(path);
  }
  if (extension == ".h5") {
    return LoadMochiModel(path);
  }
  if (extension == ".hdr" || extension == ".exr") {
    return LoadIbl(path);
  }
  if (extension == ".json") {
    std::string stem = path.GetStem();
    std::transform(stem.begin(), stem.end(), stem.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (stem.ends_with(".mochi")) {
      return LoadMochiModel(path);
    }
  }
  MOCHI_LOG_WARNING(
      "Failed to load unsupported resource type with extension %s: %s",
      extension.c_str(),
      path.ToString().c_str());
  return nullptr;
}

RenderModel* ResourceManager::LoadRenderModel(mochi::Path const& path) {
  if (_resources.contains(path)) {
    MOCHI_LOG_VERBOSE("Resource is already loaded from path: %s", path.ToString().c_str());
    if (_resources[path].get()->GetType() == ResourceType::RenderModel) {
      return static_cast<RenderModel*>(_resources[path].get());
    }
    MOCHI_LOG_ERROR("Failed to cast resource to RenderModel: %s", path.ToString().c_str());
    return nullptr;
  }
  std::string extension = path.GetExtension();
  std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
    return std::tolower(c);
  });
  if (extension == ".gltf" || extension == ".glb") {
    return LoadGltf(path);
  }
  if (extension == ".stl") {
    return LoadStl(path);
  }
  if (extension == ".obj") {
    return LoadObj(path);
  }
  if (extension == ".dae") {
    return LoadCollada(path);
  }
  if (extension == ".h5") {
    return LoadMochiModel(path);
  }
  if (extension == ".json") {
    std::string stem = path.GetStem();
    std::transform(stem.begin(), stem.end(), stem.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (stem.ends_with(".mochi")) {
      return LoadMochiModel(path);
    }
  }
  MOCHI_LOG_WARNING(
      "Failed to load unsupported RenderModel type with extension %s: %s",
      extension.c_str(),
      path.ToString().c_str());
  return nullptr;
}

RenderModel* ResourceManager::LoadGltf(mochi::Path const& path, bool async) {
  MOCHI_LOG("[gltf] LoadGltf called with path: %s (async=%d)", path.ToString().c_str(), async);
  if (_resources.contains(path)) {
    MOCHI_LOG_VERBOSE("[gltf] Resource is already loaded from path: %s", path.ToString().c_str());
    if (_resources[path].get()->GetType() == ResourceType::RenderModel) {
      return static_cast<RenderModel*>(_resources[path].get());
    }
    MOCHI_LOG_ERROR("Failed to cast resource to RenderModel: %s", path.ToString().c_str());
    return nullptr;
  }
  utils::Path filename = GetPathForGltfAsset(path.ToString());
  if (!filename.isEmpty()) {
    MOCHI_LOG("[gltf] CheckGltfAsset for: %s", filename.c_str());
    if (CheckGltfAsset(filename)) {
      MOCHI_LOG("[gltf] CheckGltfAsset passed, loading asset...");
      _gltfResourceLoader->asyncCancelLoad();
      _gltfResourceLoader->evictResourceData();
      auto renderModel = std::unique_ptr<RenderModel>(new RenderModel(
          _engine, filename.getNameWithoutExtension(), path, RenderModelFormat::Gltf));
      if (auto* fila_asset = LoadFilamentGltfAsset(filename, renderModel->_instances)) {
        MOCHI_LOG("[gltf] LoadFilamentGltfAsset succeeded, loading resources...");
        if (LoadFilamentGltfResources(filename, fila_asset, async)) {
          MOCHI_LOG("[gltf] LoadFilamentGltfResources succeeded for: %s", path.ToString().c_str());
          renderModel->_primaryAsset = fila_asset;
          renderModel->_assetLoader = _gltfAssetLoader;
          renderModel->InitializeInitialInstance();
          auto* raw = renderModel.get();
          _resources[path] = std::move(renderModel);
          return raw;
        } else {
          MOCHI_LOG_ERROR(
              "[gltf] LoadFilamentGltfResources FAILED for: %s", path.ToString().c_str());
        }
      } else {
        MOCHI_LOG_ERROR("[gltf] LoadFilamentGltfAsset FAILED for: %s", filename.c_str());
      }
    } else {
      MOCHI_LOG_ERROR("[gltf] CheckGltfAsset FAILED for: %s", filename.c_str());
    }
  } else {
    MOCHI_LOG_ERROR("[gltf] GetPathForGltfAsset returned empty for: %s", path.ToString().c_str());
  }
  return nullptr;
}

void ResourceManager::PumpAsyncLoad() const {
  _gltfResourceLoader->asyncUpdateLoad();
}

void ResourceManager::WaitAsyncLoad() const {
  while (_gltfResourceLoader->asyncGetLoadProgress() < 1.0f) {
    _gltfResourceLoader->asyncUpdateLoad();
  }
}

RenderModel* ResourceManager::LoadModelFromMeshSections(
    mochi::Path const& path,
    std::vector<MeshSection> sections,
    RenderModelFormat format) {
  if (_resources.contains(path)) {
    MOCHI_LOG_VERBOSE("Resource is already loaded from path: %s", path.ToString().c_str());
    if (_resources[path].get()->GetType() == ResourceType::RenderModel) {
      return static_cast<RenderModel*>(_resources[path].get());
    }
    MOCHI_LOG_ERROR("Failed to cast resource to RenderModel: %s", path.ToString().c_str());
    return nullptr;
  }

  if (sections.empty()) {
    MOCHI_LOG_ERROR("Model has no geometry: %s", path.ToString().c_str());
    return nullptr;
  }

  // Compute angle-weighted vertex normals for any section whose source left them
  // missing (STL supplies only per-face normals; OBJ/COLLADA/CAD may omit them).
  for (MeshSection& section : sections) {
    if (!section.hasNormals || section.normals.size() != section.positions.size()) {
      std::vector<float> faceNormals;
      ComputeFaceNormals(section.positions, section.indices, faceNormals);
      ComputeVertexNormalsAngleWeighted(
          section.positions, faceNormals, section.indices, section.normals);
      section.hasNormals = true;
    }
  }

  // Convert to in-memory GLB and load through the glTF pipeline.
  auto glbData = BuildGlbFromMeshSections(sections);

  _gltfResourceLoader->asyncCancelLoad();
  _gltfResourceLoader->evictResourceData();

  auto renderModel =
      std::unique_ptr<RenderModel>(new RenderModel(_engine, path.GetStem(), path, format));
  auto* fila_asset = LoadFilamentGltfAssetFromBuffer(glbData, renderModel->_instances);
  if (!fila_asset) {
    MOCHI_LOG_ERROR("Failed to load model as glTF asset: %s", path.ToString().c_str());
    return nullptr;
  }
  renderModel->_primaryAsset = fila_asset;
  renderModel->_assetLoader = _gltfAssetLoader;
  renderModel->InitializeInitialInstance();
  auto* raw = renderModel.get();
  _resources[path] = std::move(renderModel);
  return raw;
}

mochi::CoordinateSpaceConverter const& ResourceManager::SourceToRenderConverter() const {
  static mochi::CoordinateSpaceConverter const kDefaultConverter(
      mochi::CoordinateSpace::Default(), RenderSpace());
  return _spaceConverter != nullptr ? *_spaceConverter : kDefaultConverter;
}

RenderModel* ResourceManager::LoadMeshAsset(
    mochi::Path const& path,
    std::vector<MeshSection> (*reader)(char const*),
    RenderModelFormat format,
    char const* label,
    bool sourceIsMochiSpace) {
  if (_resources.contains(path)) {
    MOCHI_LOG_VERBOSE("Resource is already loaded from path: %s", path.ToString().c_str());
    if (_resources[path].get()->GetType() == ResourceType::RenderModel) {
      return static_cast<RenderModel*>(_resources[path].get());
    }
    MOCHI_LOG_ERROR("Failed to cast resource to RenderModel: %s", path.ToString().c_str());
    return nullptr;
  }

  // Parse the mesh file into sections (one per material).
  std::vector<MeshSection> sections = reader(path.ToString().c_str());
  if (sections.empty()) {
    MOCHI_LOG_ERROR("%s file contains no geometry: %s", label, path.ToString().c_str());
    return nullptr;
  }
  if (sourceIsMochiSpace) {
    ConvertMeshSectionsSpace(sections, SourceToRenderConverter());
  }
  return LoadModelFromMeshSections(path, std::move(sections), format);
}

// OBJ and STL declare no axis convention of their own, so their raw coordinates are whatever the
// authoring tool wrote. Mochi authors both in @ref mochi::CoordinateSpace::Default (Z-up), which is
// a quarter turn about X away from @ref RenderSpace -- hence the conversion. COLLADA is the
// exception: @ref ReadColladaFromFile already resolves the document's own up-axis into glTF's Y-up
// convention, so its output is renderer-space already.
RenderModel* ResourceManager::LoadStl(mochi::Path const& path) {
  return LoadMeshAsset(
      path, &ReadStlFromFile, RenderModelFormat::Stl, "STL", /*sourceIsMochiSpace=*/true);
}

RenderModel* ResourceManager::LoadObj(mochi::Path const& path) {
  return LoadMeshAsset(
      path, &ReadObjFromFile, RenderModelFormat::Obj, "OBJ", /*sourceIsMochiSpace=*/true);
}

RenderModel* ResourceManager::LoadCollada(mochi::Path const& path) {
  return LoadMeshAsset(
      path,
      &ReadColladaFromFile,
      RenderModelFormat::Collada,
      "COLLADA",
      /*sourceIsMochiSpace=*/false);
}

RenderModel* ResourceManager::LoadMochiModel(mochi::Path const& path) {
  if (_resources.contains(path)) {
    MOCHI_LOG_VERBOSE("Resource is already loaded from path: %s", path.ToString().c_str());
    if (_resources[path].get()->GetType() == ResourceType::RenderModel) {
      return static_cast<RenderModel*>(_resources[path].get());
    }
    MOCHI_LOG_ERROR("Failed to cast resource to RenderModel: %s", path.ToString().c_str());
    return nullptr;
  }
  mochi::ErrorLog error;
  MOCHI_ERROR_IF(path.IsEmpty(), error, "Invalid file path");
  MOCHI_ERROR_RETURN(error, nullptr);
  mochi::ModelData modelData = mochi::model::LoadFromFile(path.ToString(), error);
  MOCHI_ERROR_RETURN(error, nullptr);
  return LoadMochiModel(path, modelData);
}

RenderModel* ResourceManager::LoadMochiModel(
    mochi::Path const& path,
    mochi::ModelData const& modelData) {
  if (_resources.contains(path)) {
    MOCHI_LOG_VERBOSE("Resource is already loaded from path: %s", path.ToString().c_str());
    if (_resources[path].get()->GetType() == ResourceType::RenderModel) {
      return static_cast<RenderModel*>(_resources[path].get());
    }
    MOCHI_LOG_ERROR("Failed to cast resource to RenderModel: %s", path.ToString().c_str());
    return nullptr;
  }
  mochi::CoordinateSpaceConverter const& spaceConverter = SourceToRenderConverter();
  std::vector<float> positions;
  std::vector<float> vertexNormals;
  std::vector<int> indices;
  if (!BuildMochiModelGeometry(modelData, &spaceConverter, positions, vertexNormals, indices)) {
    MOCHI_LOG_ERROR("Mochi model has empty or invalid geometry: %s", path.ToString().c_str());
    return nullptr;
  }

  // Package the single-material geometry into one section and convert to an
  // in-memory GLB to load through the glTF pipeline. BuildMochiModelGeometry
  // always fills one normal per vertex, so mark the section accordingly.
  MeshSection section;
  section.positions = positions;
  section.normals = vertexNormals;
  section.indices = indices;
  section.hasNormals = true;
  auto glbData = BuildGlbFromMeshSections({std::move(section)});

  _gltfResourceLoader->asyncCancelLoad();
  _gltfResourceLoader->evictResourceData();

  auto renderModel = std::unique_ptr<RenderModel>(new RenderModel(
      _engine, Resource::GetNameFromPath(path), path, RenderModelFormat::MochiModel));
  auto* fila_asset = LoadFilamentGltfAssetFromBuffer(glbData, renderModel->_instances);
  if (!fila_asset) {
    MOCHI_LOG_ERROR("Failed to load Mochi model as glTF asset: %s", path.ToString().c_str());
    return nullptr;
  }
  renderModel->_primaryAsset = fila_asset;
  renderModel->_assetLoader = _gltfAssetLoader;
  renderModel->InitializeInitialInstance();
  auto* raw = renderModel.get();
  _resources[path] = std::move(renderModel);
  return raw;
}

IBL* ResourceManager::LoadIbl(mochi::Path const& path) {
  if (auto it = _resources.find(path); it != _resources.end()) {
    MOCHI_LOG_VERBOSE("Resource is already loaded from path: %s", path.ToString().c_str());
    if (it->second->GetType() == ResourceType::Ibl) {
      return static_cast<IBL*>(it->second.get());
    }
    MOCHI_LOG_ERROR("Failed to cast resource to IBL: %s", path.ToString().c_str());
    return nullptr;
  }

  utils::Path iblPath(path.ToString());
  if (!iblPath.exists()) {
    MOCHI_LOG_ERROR("The specified IBL path does not exist: %s", path.ToString().c_str());
    return nullptr;
  }

  auto ibl = std::unique_ptr<IBL>(new IBL(_engine, Resource::GetNameFromPath(path), path));
  bool const ok =
      iblPath.isDirectory() ? ibl->LoadFromDirectory(path) : ibl->LoadFromEquirect(path);
  if (!ok) {
    MOCHI_LOG_ERROR("Failed to load IBL: %s", path.ToString().c_str());
    return nullptr;
  }

  MOCHI_LOG("Loaded IBL: %s", path.ToString().c_str());
  auto* raw = ibl.get();
  _resources[path] = std::move(ibl);
  return raw;
}

Resource* ResourceManager::FindResourceByPath(mochi::Path const& path, bool silent) const {
  auto it = _resources.find(path);
  if (it != _resources.end()) {
    return it->second.get();
  }
  if (!silent) {
    MOCHI_LOG_ERROR("Resource not found: %s", path.ToString().c_str());
  }
  return nullptr;
}

bool ResourceManager::UnloadResource(mochi::Path const& path) {
  auto it = _resources.find(path);
  if (it == _resources.end()) {
    return false;
  }
  _resources.erase(it);
  return true;
}

void ResourceManager::UnloadAllResources() {
  _resources.clear();
}

Resource* ResourceManager::RegisterResource(
    mochi::Path const& path,
    std::unique_ptr<Resource> resource) {
  if (_resources.contains(path)) {
    return nullptr;
  }
  auto* raw = resource.get();
  _resources[path] = std::move(resource);
  return raw;
}

bool ResourceManager::RewriteResourcePath(mochi::Path const& oldPath, mochi::Path const& newPath) {
  auto it = _resources.find(oldPath);
  if (it == _resources.end() || _resources.contains(newPath)) {
    return false;
  }
  auto node = _resources.extract(it);
  Resource* resource = node.mapped().get();
  resource->SetPath(newPath);
  // Handle double extensions like .mochi.h5
  resource->SetName(Resource::GetNameFromPath(newPath));
  node.key() = newPath;
  _resources.insert(std::move(node));
  return true;
}

filament::gltfio::FilamentAsset* ResourceManager::LoadFilamentGltfAsset(
    utils::Path const& filename,
    FilamentInstanceVector& instances) const {
  // Peek at the file size to allow pre-allocation.
  long const contentSize = static_cast<long>(GetFileSize(filename.c_str()));
  if (contentSize <= 0) {
    MOCHI_LOG_ERROR("Unable to open: %s", filename.c_str());
    return nullptr;
  }

  // Consume the glTF file.
  std::ifstream in(filename.c_str(), std::ifstream::binary | std::ifstream::in);
  std::vector<uint8_t> buffer(static_cast<unsigned long>(contentSize));
  if (!in.read((char*)buffer.data(), contentSize)) {
    MOCHI_LOG_ERROR("Unable to read: %s", filename.c_str());
    return nullptr;
  }

  // Create exactly one initial instance; further instances grow lazily on demand.
  instances.resize(1);
  filament::gltfio::FilamentAsset* asset = _gltfAssetLoader->createInstancedAsset(
      buffer.data(), buffer.size(), instances.data(), instances.size());
  if (!asset) {
    MOCHI_LOG_ERROR("Unable to parse: %s", filename.c_str());
    return nullptr;
  }

  // pre-compile all material variants
  std::set<filament::Material*> materials;
  filament::RenderableManager const& rcm = _engine->getRenderableManager();
  utils::Slice<utils::Entity const> const renderables{
      asset->getRenderableEntities(), asset->getRenderableEntityCount()};
  for (utils::Entity const e : renderables) {
    auto ri = rcm.getInstance(e);
    size_t const c = rcm.getPrimitiveCount(ri);
    for (size_t i = 0; i < c; i++) {
      filament::MaterialInstance* const mi = rcm.getMaterialInstanceAt(ri, i);
      auto* ma = const_cast<filament::Material*>(mi->getMaterial());
      materials.insert(ma);
    }
  }
  for (filament::Material* ma : materials) {
    // First compile high priority variants
    ma->compile(
        filament::Material::CompilerPriorityQueue::HIGH,
        filament::UserVariantFilterBit::DIRECTIONAL_LIGHTING |
            filament::UserVariantFilterBit::DYNAMIC_LIGHTING |
            filament::UserVariantFilterBit::SHADOW_RECEIVER);
    // and then, everything else at low priority, except STE, which is very uncommon.
    ma->compile(
        filament::Material::CompilerPriorityQueue::LOW,
        filament::UserVariantFilterBit::FOG | filament::UserVariantFilterBit::SKINNING |
            filament::UserVariantFilterBit::SSR | filament::UserVariantFilterBit::VSM);
  }

  buffer.clear();
  buffer.shrink_to_fit();
  return asset;
}

filament::gltfio::FilamentAsset* ResourceManager::LoadFilamentGltfAssetFromBuffer(
    std::vector<uint8_t> const& glbData,
    FilamentInstanceVector& instances) const {
  // Create exactly one initial instance; further instances grow lazily on demand.
  instances.resize(1);
  filament::gltfio::FilamentAsset* asset = _gltfAssetLoader->createInstancedAsset(
      glbData.data(), static_cast<uint32_t>(glbData.size()), instances.data(), instances.size());
  if (!asset) {
    MOCHI_LOG_ERROR("Unable to parse in-memory GLB");
    return nullptr;
  }

  // Pre-compile all material variants.
  std::set<filament::Material*> materials;
  filament::RenderableManager const& rcm = _engine->getRenderableManager();
  utils::Slice<utils::Entity const> const renderables{
      asset->getRenderableEntities(), asset->getRenderableEntityCount()};
  for (utils::Entity const e : renderables) {
    auto ri = rcm.getInstance(e);
    size_t const c = rcm.getPrimitiveCount(ri);
    for (size_t i = 0; i < c; i++) {
      auto* const mi = rcm.getMaterialInstanceAt(ri, i);
      if (!mi) {
        continue;
      }
      auto* ma = const_cast<filament::Material*>(mi->getMaterial());
      materials.insert(ma);
    }
  }
  for (filament::Material* ma : materials) {
    ma->compile(
        filament::Material::CompilerPriorityQueue::HIGH,
        filament::UserVariantFilterBit::DIRECTIONAL_LIGHTING |
            filament::UserVariantFilterBit::DYNAMIC_LIGHTING |
            filament::UserVariantFilterBit::SHADOW_RECEIVER);
    ma->compile(
        filament::Material::CompilerPriorityQueue::LOW,
        filament::UserVariantFilterBit::FOG | filament::UserVariantFilterBit::SKINNING |
            filament::UserVariantFilterBit::SSR | filament::UserVariantFilterBit::VSM);
  }

  // Load resources (vertex/index buffers) from the embedded binary chunk.
  // The glbData buffer must remain alive for this call since cgltf's bin pointer
  // references into it. No external file I/O is needed for embedded GLB.
  filament::gltfio::ResourceConfiguration configuration{};
  configuration.engine = _engine;
  configuration.gltfPath = ""; // Must not be nullptr — Filament assigns it to std::string.
  configuration.normalizeSkinningWeights = true;
  _gltfResourceLoader->setConfiguration(configuration);
  if (!_gltfResourceLoader->loadResources(asset)) {
    MOCHI_LOG_ERROR("Unable to load resources from in-memory GLB");
    return nullptr;
  }

  // Source data is intentionally retained (no releaseSourceData) so instances can grow lazily
  // via AssetLoader::createInstance for the model's lifetime. Per-instance setup (stencil config +
  // bounding-box recompute) is applied by RenderModel::ConfigureInstance.
  return asset;
}

bool ResourceManager::LoadFilamentGltfResources(
    utils::Path const& filename,
    filament::gltfio::FilamentAsset* asset,
    bool async) const {
  // Load external textures and buffers.
  std::string const gltfPath = filename.getAbsolutePath();
  filament::gltfio::ResourceConfiguration configuration{};
  configuration.engine = _engine;
  configuration.gltfPath = gltfPath.c_str();
  configuration.normalizeSkinningWeights = true;
  _gltfResourceLoader->setConfiguration(configuration);
  if (async) {
    if (!_gltfResourceLoader->asyncBeginLoad(asset)) {
      MOCHI_LOG_ERROR("Unable to start loading resources for: %s", filename.c_str());
      return false;
    }
  } else {
    if (!_gltfResourceLoader->loadResources(asset)) {
      MOCHI_LOG_ERROR("Unable to load resources for: %s", filename.c_str());
      return false;
    }
  }
  // Source data is intentionally retained (no releaseSourceData) so instances can grow lazily
  // via AssetLoader::createInstance for the model's lifetime. Per-instance setup (stencil config +
  // bounding-box recompute) is applied by RenderModel::ConfigureInstance.
  return true;
}

} // namespace mochi_renderer
