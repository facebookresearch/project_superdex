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

#include "io/glb_export.h"

#include <superdex_robotics/core/context.h>
#include <superdex_robotics/core/loader.h>
#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/bot_utils.h>

#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/coordinate_space.h>
#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/transform_rt.h>
#include <mochi_core/utils/transform_rt_utils.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_renderer/render_space.h>
#include <mochi_renderer/utils.h>

// Only the writer half: cgltf's parser implementation is already linked in through
// mochi_renderer (src/third_party/cgltf_impl.cpp), so defining CGLTF_IMPLEMENTATION here would
// duplicate those symbols.
#define CGLTF_WRITE_IMPLEMENTATION
#include <cgltf_write.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace superdex::studio {
namespace {

struct CombinedMesh {
  mochi::DynamicArray<float> positions;
  mochi::DynamicArray<float> normals;
  mochi::DynamicArray<float> texcoords; // 2 per vertex (0,0 for untextured geometry)
  mochi::DynamicArray<uint8_t> joints;
  mochi::DynamicArray<float> weights;
  mochi::DynamicArray<uint32_t> indices;
  int totalVertices = 0;
  int totalIndices = 0;
};

// The glTF material textures the export carries over from the source render meshes.
enum TextureSlot {
  kBaseColor,
  kMetallicRoughness,
  kNormal,
  kOcclusion,
  kEmissive,
  kNumTextureSlots,
};

// Marks a texture slot with no image.
constexpr int kNoImage = -1;

constexpr std::array<int, kNumTextureSlots> NoImages() {
  std::array<int, kNumTextureSlots> images{};
  images.fill(kNoImage);
  return images;
}

// An image from a source render mesh, re-embedded verbatim into the export.
struct ImageData {
  std::vector<uint8_t> bytes;
  std::string mimeType;
};

// PBR material of a source render mesh, doubling as the key that merges primitives drawn alike.
struct MaterialFactors {
  std::array<float, 4> baseColor;
  float metallic;
  float roughness;
  std::array<float, 3> emissive;
  float emissiveStrength;
  float normalScale = 1.0f;
  float occlusionStrength = 1.0f;
  // Per TextureSlot, an index into the images list, or kNoImage.
  std::array<int, kNumTextureSlots> images = NoImages();

  // Exact comparison is what merging wants: the values are copied verbatim out of cgltf's parse of
  // the source GLBs.
  bool operator==(MaterialFactors const&) const = default;
};

// One glTF primitive: geometry plus the material it draws with, or -1 for untextured collision.
struct MeshPart {
  CombinedMesh geometry;
  int materialIndex = -1;
};

// One glTF mesh, covering a contiguous run of parts so it satisfies cgltf_mesh's pointer + count.
struct MeshGroup {
  char const* meshName;
  char const* nodeName;
  int firstPart;
  int numParts;
};

// Mochi authors in Z-up (@ref mochi::CoordinateSpace::Default) while glTF is Y-up, a quarter turn
// about X apart. Targeting @ref mochi_renderer::RenderSpace keeps this export on the same
// convention.
mochi::CoordinateSpaceConverter GltfFromMochi() {
  return {mochi::CoordinateSpace::Default(), mochi_renderer::RenderSpace()};
}

// Sizes @p array once and zeroes it, since DynamicArray::resize leaves POD elements uninitialized
// and cgltf expects absent fields to be null.
template <typename T>
void ResizeZeroed(mochi::DynamicArray<T>& array, int count) {
  array.resize(count);
  if (count > 0) {
    std::memset(array.data(), 0, sizeof(T) * static_cast<size_t>(count));
  }
}

// Appends a triangle mesh, carrying it from its own frame into the bind-pose root frame and on into
// the GLB's basis. @p preScale is applied in the mesh's own frame, before @p rootFromMesh.
//
// @p normals may be empty, in which case per-vertex normals are accumulated from face normals.
// Source normals matter: the render meshes are unwelded triangle soups, so every vertex has exactly
// one adjacent face and accumulation could only ever produce faceted shading. @p texcoords may be
// empty for untextured geometry.
template <typename T>
void AppendMeshGeometry(
    CombinedMesh& combined,
    mochi::Span<T const> positions,
    mochi::Span<T const> normals,
    mochi::Span<float const> texcoords,
    mochi::Span<int const> indices,
    mochi::TransformRT const& rootFromMesh,
    mochi::Real3 const& preScale,
    mochi::CoordinateSpaceConverter const& gltfFromMochi,
    int linkIndex) {
  int const numNodes = mochi::isize(positions) / 3;
  int const numElements = mochi::isize(indices) / 3;
  if (numNodes == 0 || numElements == 0) {
    return;
  }
  bool const hasSourceNormals = mochi::isize(normals) == numNodes * 3;
  bool const hasTexcoords = mochi::isize(texcoords) == numNodes * 2;

  // Normals transform by the inverse transpose of the linear part. That part is a permutation times
  // a rotation times diag(preScale), and the first two are orthogonal, so only the scale inverts.
  // Guard each axis: a zero (or near-zero) renderModelScale component would make 1/s inf, and
  // 0 * inf is NaN, which NormalizeNormals cannot recover (inf normalizes to NaN). Skipping the
  // inverse-scale on degenerate axes keeps the export finite.
  constexpr mochi::real kMinScale = 1e-8_r;
  auto const safeInverse = [](mochi::real s) -> mochi::real {
    return std::abs(s) > kMinScale ? 1_r / s : 1_r;
  };
  mochi::Real3 const inversePreScale{
      safeInverse(preScale[0]), safeInverse(preScale[1]), safeInverse(preScale[2])};

  auto const vertexOffset = static_cast<uint32_t>(combined.totalVertices);

  for (int i = 0; i < numNodes; ++i) {
    mochi::Real3 const pos{
        static_cast<mochi::real>(positions[i * 3 + 0]) * preScale[0],
        static_cast<mochi::real>(positions[i * 3 + 1]) * preScale[1],
        static_cast<mochi::real>(positions[i * 3 + 2]) * preScale[2]};
    mochi::Real3 const bindPos =
        gltfFromMochi.TranslationToOutput(rootFromMesh.TransformPoint(pos));
    combined.positions.push_back(static_cast<float>(bindPos[0]));
    combined.positions.push_back(static_cast<float>(bindPos[1]));
    combined.positions.push_back(static_cast<float>(bindPos[2]));

    if (hasSourceNormals) {
      mochi::Real3 const normal{
          static_cast<mochi::real>(normals[i * 3 + 0]) * inversePreScale[0],
          static_cast<mochi::real>(normals[i * 3 + 1]) * inversePreScale[1],
          static_cast<mochi::real>(normals[i * 3 + 2]) * inversePreScale[2]};
      mochi::Real3 const bindNormal =
          gltfFromMochi.DirectionToOutput(rootFromMesh.TransformDirection(normal));
      combined.normals.push_back(static_cast<float>(bindNormal[0]));
      combined.normals.push_back(static_cast<float>(bindNormal[1]));
      combined.normals.push_back(static_cast<float>(bindNormal[2]));
    } else {
      // Placeholder, accumulated from face normals below.
      combined.normals.push_back(0.0f);
      combined.normals.push_back(0.0f);
      combined.normals.push_back(0.0f);
    }

    // Every primitive carries a TEXCOORD_0 so the writer's vertex attribute layout stays uniform
    // across parts; untextured geometry gets placeholder UVs that go unused.
    if (hasTexcoords) {
      combined.texcoords.push_back(texcoords[i * 2 + 0]);
      combined.texcoords.push_back(texcoords[i * 2 + 1]);
    } else {
      combined.texcoords.push_back(0.0f);
      combined.texcoords.push_back(0.0f);
    }

    combined.joints.push_back(static_cast<uint8_t>(linkIndex));
    combined.joints.push_back(0);
    combined.joints.push_back(0);
    combined.joints.push_back(0);

    combined.weights.push_back(1.0f);
    combined.weights.push_back(0.0f);
    combined.weights.push_back(0.0f);
    combined.weights.push_back(0.0f);
  }

  int numAppendedElements = 0;
  int numSkippedElements = 0;
  for (int e = 0; e < numElements; ++e) {
    int const s0 = indices[e * 3 + 0];
    int const s1 = indices[e * 3 + 1];
    int const s2 = indices[e * 3 + 2];
    // An index outside the source vertex range would read and write past the arrays below, and even
    // on the source-normals path would reach the exported index accessor and invalidate the glTF.
    if (s0 < 0 || s0 >= numNodes || s1 < 0 || s1 >= numNodes || s2 < 0 || s2 >= numNodes) {
      ++numSkippedElements;
      continue;
    }
    uint32_t const i0 = vertexOffset + static_cast<uint32_t>(s0);
    uint32_t const i1 = vertexOffset + static_cast<uint32_t>(s1);
    uint32_t const i2 = vertexOffset + static_cast<uint32_t>(s2);
    combined.indices.push_back(i0);
    combined.indices.push_back(i1);
    combined.indices.push_back(i2);
    ++numAppendedElements;

    if (hasSourceNormals) {
      continue;
    }

    // Compute face normal and accumulate to vertices
    float const* p0 = &combined.positions[static_cast<size_t>(i0) * 3];
    float const* p1 = &combined.positions[static_cast<size_t>(i1) * 3];
    float const* p2 = &combined.positions[static_cast<size_t>(i2) * 3];
    std::array<float, 3> const e1 = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
    std::array<float, 3> const e2 = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
    std::array<float, 3> const n = {
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0]};
    for (uint32_t const vi : {i0, i1, i2}) {
      combined.normals[vi * 3 + 0] += n[0];
      combined.normals[vi * 3 + 1] += n[1];
      combined.normals[vi * 3 + 2] += n[2];
    }
  }

  if (numSkippedElements > 0) {
    MOCHI_LOG_WARNING(
        "AppendMeshGeometry: skipped %d of %d triangles with out-of-range vertex indices.",
        numSkippedElements,
        numElements);
  }

  combined.totalVertices += numNodes;
  combined.totalIndices += numAppendedElements * 3;
}

void NormalizeNormals(CombinedMesh& combined) {
  for (int i = 0; i < combined.totalVertices; ++i) {
    float* n = &combined.normals[static_cast<size_t>(i) * 3];
    float const len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    // std::isfinite also rejects inf: an inf normal would otherwise normalize to NaN
    // (inf * (1/inf) == inf * 0 == NaN) and corrupt the NORMAL accessor.
    if (len > 1e-8f && std::isfinite(len) && std::isfinite(n[0]) && std::isfinite(n[1]) &&
        std::isfinite(n[2])) {
      float const inv = 1.0f / len;
      n[0] *= inv;
      n[1] *= inv;
      n[2] *= inv;
    } else {
      n[0] = 0.0f;
      n[1] = 1.0f;
      n[2] = 0.0f;
    }
  }
}

void ComputeBounds(
    CombinedMesh const& combined,
    std::array<float, 3>& outMin,
    std::array<float, 3>& outMax) {
  outMin.fill(std::numeric_limits<float>::max());
  outMax.fill(std::numeric_limits<float>::lowest());
  for (int i = 0; i < combined.totalVertices; ++i) {
    for (int c = 0; c < 3; ++c) {
      float const v = combined.positions[i * 3 + c];
      outMin[c] = std::min(outMin[c], v);
      outMax[c] = std::max(outMax[c], v);
    }
  }
}

void WriteInverseBindMatrix(float* out, mochi::TransformRT const& rootFromLink) {
  mochi::TransformRT const ibm = Invert(rootFromLink);
  mochi::Matrix3x3r const rotation = mochi::GetRotationMatrix(ibm);
  mochi::Real3 const t = ibm.GetTranslation();

  // glTF is column-major, so column c of the rotation occupies out[c * 4 .. c * 4 + 2].
  for (int c = 0; c < 3; ++c) {
    out[c * 4 + 0] = static_cast<float>(rotation[0][c]);
    out[c * 4 + 1] = static_cast<float>(rotation[1][c]);
    out[c * 4 + 2] = static_cast<float>(rotation[2][c]);
    out[c * 4 + 3] = 0.0f;
  }
  out[12] = static_cast<float>(t[0]);
  out[13] = static_cast<float>(t[1]);
  out[14] = static_cast<float>(t[2]);
  out[15] = 1.0f;
}

int FindOrAddMaterial(
    mochi::DynamicArray<MaterialFactors>& materials,
    MaterialFactors const& factors) {
  auto const it = std::ranges::find(materials, factors);
  if (it != materials.end()) {
    return static_cast<int>(it - materials.begin());
  }
  materials.push_back(factors);
  return mochi::isize(materials) - 1;
}

// Index of @p image in @p images, adding it if no image there has the same bytes, so a texture used
// by several sections or links is embedded once. kNoImage for an empty image.
int FindOrAddImage(mochi::DynamicArray<ImageData>& images, ImageData image) {
  if (image.bytes.empty()) {
    return kNoImage;
  }
  for (int i = 0; i < mochi::isize(images); ++i) {
    if (images[i].bytes == image.bytes) {
      return i;
    }
  }
  images.push_back(std::move(image));
  return mochi::isize(images) - 1;
}

// The articulated actor's link order need not match the prefab's declaration order, so links are
// paired by name.
robotics::BotLinkPrefab const* FindLinkPrefab(
    robotics::BotPrefab const& botPrefab,
    mochi::DynamicString const& linkName) {
  for (auto const& link : botPrefab.links) {
    if (link.name == linkName) {
      return &link;
    }
  }
  return nullptr;
}

// Appends one part per distinct material, merging every link's render mesh into those parts.
void GatherRenderParts(
    robotics::BotPrefab const& botPrefab,
    mochi::Span<mochi::DynamicString const> linkNames,
    mochi::Span<mochi::TransformRT const> bindFromLinks,
    mochi::CoordinateSpaceConverter const& gltfFromMochi,
    mochi::Span<bool const> hiddenLinks,
    mochi::DynamicArray<ImageData>& images,
    mochi::DynamicArray<MaterialFactors>& materials,
    mochi::DynamicArray<MeshPart>& parts) {
  // Render GLBs are authored in the renderer's basis. Bringing them back into Mochi space lets them
  // share one conversion path with the collision geometry and the link transforms.
  mochi::CoordinateSpaceConverter const mochiFromGltf(
      mochi_renderer::RenderSpace(), mochi::CoordinateSpace::Default());

  int const numLinks = mochi::isize(bindFromLinks);
  for (int i = 0; i < numLinks; ++i) {
    // A link the skin's render model stands in for is represented by it, so skip its rigid render
    // mesh.
    if (hiddenLinks[i]) {
      continue;
    }
    auto const* linkPrefab = FindLinkPrefab(botPrefab, linkNames[i]);
    if (linkPrefab == nullptr || linkPrefab->renderModelFile.empty()) {
      continue;
    }

    auto sections = mochi_renderer::ReadGlbFromFile(linkPrefab->renderModelFile.c_str());
    if (sections.empty()) {
      MOCHI_LOG_WARNING(
          "ExportSkeletalGlb: no render geometry read from '%s' for link '%s'; skipping it.",
          linkPrefab->renderModelFile.c_str(),
          linkNames[i].c_str());
      continue;
    }
    mochi_renderer::ConvertMeshSectionsSpace(sections, mochiFromGltf);

    // Unlike the collision shape, whose offset IBotLoader::LoadShape bakes in, a link's render
    // model offset is baked into no asset and has to be applied here.
    mochi::TransformRT const rootFromRenderModel = bindFromLinks[i] *
        mochi::TransformRT(linkPrefab->renderModelRotation, linkPrefab->renderModelTranslation);

    for (auto const& section : sections) {
      // Check for at least one triangle before allocating a material and a part: an empty primitive
      // would carry a zero-count accessor, which is invalid glTF.
      if (section.positions.size() < 3 || section.indices.size() < 3) {
        continue;
      }
      MaterialFactors factors{
          section.baseColor,
          section.metallic,
          section.roughness,
          section.emissive,
          section.emissiveStrength};
      // ReadGlbFromFile reads only the base color texture, which needs the section's UVs.
      if (section.hasTexcoords && section.baseColorTexture != nullptr) {
        factors.images[kBaseColor] = FindOrAddImage(
            images, ImageData{section.baseColorTexture->bytes, section.baseColorTexture->mimeType});
      }
      int const materialIndex = FindOrAddMaterial(materials, factors);
      // Materials grow one at a time, so a new material is always the next part, which keeps render
      // parts indexed by material.
      if (materialIndex == mochi::isize(parts)) {
        MeshPart part;
        part.materialIndex = materialIndex;
        parts.push_back(std::move(part));
      }
      AppendMeshGeometry(
          parts[materialIndex].geometry,
          mochi::MakeConstSpan(section.positions),
          mochi::MakeConstSpan(section.normals),
          section.hasTexcoords ? mochi::MakeConstSpan(section.texcoords)
                               : mochi::Span<float const>{},
          mochi::MakeConstSpan(section.indices),
          rootFromRenderModel,
          linkPrefab->renderModelScale,
          gltfFromMochi,
          i);
    }
  }
}

// Appends a single untextured part merging every link's collision mesh, if any link has one.
void GatherCollisionPart(
    mochi::Scene* scene,
    mochi::Context* context,
    mochi::Span<mochi::ActorHandle const> linkActorHandles,
    mochi::Span<mochi::TransformRT const> bindFromLinks,
    mochi::CoordinateSpaceConverter const& gltfFromMochi,
    mochi::Span<bool const> coveredLinks,
    mochi::DynamicArray<MeshPart>& parts) {
  MeshPart part;
  int const numLinks = mochi::isize(bindFromLinks);
  for (int i = 0; i < numLinks; ++i) {
    // A link whose collision the skin covers is represented by the skin's collision mesh, so skip
    // its own.
    if (coveredLinks[i]) {
      continue;
    }
    auto* linkActor = scene->GetActor(linkActorHandles[i]);
    if (linkActor == nullptr) {
      continue;
    }

    mochi::ShapeHandle linkShape;
    {
      mochi::ErrorLog e;
      linkShape = linkActor->GetReferenceShape(e);
      if (!e.IsOK()) {
        continue;
      }
    }

    // The visual mesh is deliberately not consulted: only a link's collision shape is ever loaded
    // into Mochi, so a visual mesh here would be someone else's geometry, not what was asked for.
    mochi::MeshDataView meshView{};
    {
      mochi::ErrorLog e;
      meshView = context->GetShapeSurfaceMesh(linkShape, e);
      if (!e.IsOK() || meshView.GetNumNodes() == 0) {
        meshView = {};
      }
    }
    if (meshView.GetNumNodes() == 0) {
      mochi::ErrorLog e;
      meshView = context->GetShapeMesh(linkShape, e);
      if (!e.IsOK()) {
        continue;
      }
    }

    // Only process triangle meshes
    if (meshView.nodesPerElement != 3 || meshView.GetNumNodes() == 0) {
      continue;
    }

    AppendMeshGeometry(
        part.geometry,
        meshView.coordinates,
        mochi::Span<mochi::real const>{},
        mochi::Span<float const>{},
        meshView.connectivity,
        bindFromLinks[i],
        mochi::Real3{1_r, 1_r, 1_r},
        gltfFromMochi,
        i);
  }

  if (part.geometry.totalVertices > 0) {
    parts.push_back(std::move(part));
  }
}

// ------------------------------------------------------------------------------------------------
// Skin support: a bot may carry a deformable skin (@ref robotics::BotPrefab::skin) skinned over the
// articulation's links. Unlike a rigid link (every vertex weighted 1.0 to its own bone), a skin
// vertex blends several links, so it is exported with its real per-vertex weights. The links the
// skin stands in for are represented by the skin, so their rigid meshes are skipped (see
// hiddenLinks and coveredLinks).
// ------------------------------------------------------------------------------------------------

// One skinned triangle section read from a skin's render GLB: geometry plus up-to-4 bone influences
// per vertex. Bone indices are in the SKIN's own bone order; the caller remaps them to composed
// links. Positions/normals are in the renderer's (Y-up) basis, as authored.
struct SkinnedSection {
  std::vector<float> positions; // 3 per vertex
  std::vector<float> normals; // 3 per vertex (may be empty)
  std::vector<float> texcoords; // 2 per vertex (may be empty)
  std::vector<int> indices; // 3 per triangle
  std::vector<int> boneIndices; // 4 per vertex
  std::vector<mochi::real> boneWeights; // 4 per vertex
  // The section's material. Its images are set once the section's images join an export's list.
  MaterialFactors material{{1.0f, 1.0f, 1.0f, 1.0f}, 0.0f, 1.0f, {0.0f, 0.0f, 0.0f}, 1.0f};
  // This section's own embedded images, per TextureSlot (empty for none). Per section rather than
  // per file: a skin GLB may texture its sections differently.
  std::array<ImageData, kNumTextureSlots> images;
};

// The image of a texture embedded in the GLB (via a buffer view), or empty for none. External-URI
// images are not handled, and neither are textures on a UV set other than TEXCOORD_0, the only one
// the export writes; both are reported.
ImageData ReadEmbeddedTexture(cgltf_texture_view const& view, char const* path) {
  cgltf_texture const* tex = view.texture;
  if (tex == nullptr || tex->image == nullptr) {
    return {};
  }
  cgltf_buffer_view const* bv = tex->image->buffer_view;
  // cgltf leaves checking that a view lies inside its buffer to cgltf_validate.
  bool const embedded = bv != nullptr && bv->buffer != nullptr && bv->buffer->data != nullptr &&
      bv->offset <= bv->buffer->size && bv->size <= bv->buffer->size - bv->offset;
  if (view.texcoord != 0 || !embedded) {
    MOCHI_LOG_WARNING_ONCE(
        "ExportSkeletalGlb: a texture of skin '%s' is external or uses a UV set other than "
        "TEXCOORD_0; it is left out of the export.",
        path);
    return {};
  }
  auto const* src = static_cast<uint8_t const*>(bv->buffer->data) + bv->offset;
  return ImageData{
      std::vector<uint8_t>(src, src + bv->size),
      tex->image->mime_type != nullptr ? tex->image->mime_type : "image/png"};
}

// Reads a skinned GLB (POSITION / NORMAL / JOINTS_0 / WEIGHTS_0 / indices + PBR material) via
// cgltf -- unlike mochi_renderer::ReadGlbFromFile, which drops skinning. Returns empty on failure.
std::vector<SkinnedSection> ReadSkinnedGlbSections(char const* path) {
  std::vector<SkinnedSection> out;
  cgltf_options parseOptions = {};
  cgltf_data* data = nullptr;
  if (cgltf_parse_file(&parseOptions, path, &data) != cgltf_result_success) {
    return out;
  }
  MOCHI_DEFER(cgltf_free(data));
  if (cgltf_load_buffers(&parseOptions, data, path) != cgltf_result_success) {
    return out;
  }
  for (cgltf_size mi = 0; mi < data->meshes_count; ++mi) {
    cgltf_mesh const& mesh = data->meshes[mi];
    for (cgltf_size pi = 0; pi < mesh.primitives_count; ++pi) {
      cgltf_primitive const& prim = mesh.primitives[pi];
      if (prim.type != cgltf_primitive_type_triangles || prim.indices == nullptr) {
        continue;
      }
      cgltf_accessor const* posA = nullptr;
      cgltf_accessor const* nrmA = nullptr;
      cgltf_accessor const* jntA = nullptr;
      cgltf_accessor const* wgtA = nullptr;
      cgltf_accessor const* texA = nullptr;
      for (cgltf_size a = 0; a < prim.attributes_count; ++a) {
        switch (prim.attributes[a].type) {
          case cgltf_attribute_type_position:
            posA = prim.attributes[a].data;
            break;
          case cgltf_attribute_type_normal:
            nrmA = prim.attributes[a].data;
            break;
          case cgltf_attribute_type_joints:
            jntA = prim.attributes[a].data;
            break;
          case cgltf_attribute_type_weights:
            wgtA = prim.attributes[a].data;
            break;
          case cgltf_attribute_type_texcoord:
            if (texA == nullptr) {
              texA = prim.attributes[a].data;
            }
            break;
          // No default: every other attribute is deliberately ignored, and leaving the switch
          // exhaustive keeps -Wswitch flagging attribute types added to cgltf later.
          case cgltf_attribute_type_invalid:
          case cgltf_attribute_type_tangent:
          case cgltf_attribute_type_color:
          case cgltf_attribute_type_custom:
          case cgltf_attribute_type_max_enum:
            break;
        }
      }
      if (posA == nullptr) {
        continue;
      }
      SkinnedSection section;
      cgltf_size const n = posA->count;
      section.positions.resize(n * 3);
      cgltf_accessor_unpack_floats(posA, section.positions.data(), n * 3);
      if (nrmA != nullptr && nrmA->count == n) {
        section.normals.resize(n * 3);
        cgltf_accessor_unpack_floats(nrmA, section.normals.data(), n * 3);
      }
      if (texA != nullptr && texA->count == n) {
        section.texcoords.resize(n * 2);
        cgltf_accessor_unpack_floats(texA, section.texcoords.data(), n * 2);
      }
      section.boneIndices.assign(n * 4, 0);
      section.boneWeights.assign(n * 4, mochi::real(0));
      if (jntA != nullptr && wgtA != nullptr && jntA->count == n && wgtA->count == n) {
        std::vector<float> jointFloats(n * 4);
        std::vector<float> weightFloats(n * 4);
        // JOINTS_0 is an (unnormalized) integer accessor; unpack_floats returns the indices as
        // floats, which round-trip exactly for the small counts a skin uses.
        cgltf_accessor_unpack_floats(jntA, jointFloats.data(), n * 4);
        cgltf_accessor_unpack_floats(wgtA, weightFloats.data(), n * 4);
        for (cgltf_size i = 0; i < n * 4; ++i) {
          section.boneIndices[i] = static_cast<int>(std::lround(jointFloats[i]));
          section.boneWeights[i] = static_cast<mochi::real>(weightFloats[i]);
        }
      }
      cgltf_size const numIndices = prim.indices->count;
      section.indices.resize(numIndices);
      for (cgltf_size i = 0; i < numIndices; ++i) {
        section.indices[i] = static_cast<int>(cgltf_accessor_read_index(prim.indices, i));
      }
      if (prim.material != nullptr) {
        cgltf_material const& material = *prim.material;
        MaterialFactors& factors = section.material;
        if (material.has_pbr_metallic_roughness) {
          auto const& pbr = material.pbr_metallic_roughness;
          std::copy_n(pbr.base_color_factor, 4, factors.baseColor.begin());
          factors.metallic = pbr.metallic_factor;
          factors.roughness = pbr.roughness_factor;
          section.images[kBaseColor] = ReadEmbeddedTexture(pbr.base_color_texture, path);
          section.images[kMetallicRoughness] =
              ReadEmbeddedTexture(pbr.metallic_roughness_texture, path);
        }
        section.images[kNormal] = ReadEmbeddedTexture(material.normal_texture, path);
        if (!section.images[kNormal].bytes.empty()) {
          factors.normalScale = material.normal_texture.scale;
        }
        section.images[kOcclusion] = ReadEmbeddedTexture(material.occlusion_texture, path);
        if (!section.images[kOcclusion].bytes.empty()) {
          factors.occlusionStrength = material.occlusion_texture.scale;
        }
        section.images[kEmissive] = ReadEmbeddedTexture(material.emissive_texture, path);
        // The emissive factor scales the emissive texture. Without it, the factor alone would make
        // the whole surface glow (this skin's factor is white over a black texture), so drop both.
        if (material.emissive_texture.texture == nullptr ||
            !section.images[kEmissive].bytes.empty()) {
          std::copy_n(material.emissive_factor, 3, factors.emissive.begin());
          if (material.has_emissive_strength) {
            factors.emissiveStrength = material.emissive_strength.emissive_strength;
          }
        }
      }
      out.push_back(std::move(section));
    }
  }
  return out;
}

// Transforms a SkinnedSection's positions/normals from the renderer's basis into Mochi space, so it
// can share AppendSkinnedGeometry's one conversion path with the (already-Mochi) collision skin.
void ConvertSkinnedSectionToMochi(
    SkinnedSection& section,
    mochi::CoordinateSpaceConverter const& mochiFromGltf) {
  int const numNodes = mochi::isize(section.positions) / 3;
  for (int i = 0; i < numNodes; ++i) {
    mochi::Real3 const p{
        section.positions[i * 3 + 0], section.positions[i * 3 + 1], section.positions[i * 3 + 2]};
    mochi::Real3 const pm = mochiFromGltf.TranslationToOutput(p);
    section.positions[i * 3 + 0] = static_cast<float>(pm[0]);
    section.positions[i * 3 + 1] = static_cast<float>(pm[1]);
    section.positions[i * 3 + 2] = static_cast<float>(pm[2]);
  }
  if (mochi::isize(section.normals) == numNodes * 3) {
    for (int i = 0; i < numNodes; ++i) {
      mochi::Real3 const nr{
          section.normals[i * 3 + 0], section.normals[i * 3 + 1], section.normals[i * 3 + 2]};
      mochi::Real3 const nm = mochiFromGltf.DirectionToOutput(nr);
      section.normals[i * 3 + 0] = static_cast<float>(nm[0]);
      section.normals[i * 3 + 1] = static_cast<float>(nm[1]);
      section.normals[i * 3 + 2] = static_cast<float>(nm[2]);
    }
  }
}

// Like AppendMeshGeometry, but writes real per-vertex skin weights instead of a single rigid bone.
// @p boneIndices / @p boneWeights hold @p weightsPerNode influences per vertex in the skin's own
// bone order; @p boneToLink remaps each to a composed link index (an influence whose bone maps to
// no link, or has non-positive weight, is dropped, then the kept influences are renormalized). Only
// the 4 glTF influence slots are kept. Positions arrive in Mochi space.
template <typename T>
void AppendSkinnedGeometry(
    CombinedMesh& combined,
    mochi::Span<T const> positions,
    mochi::Span<T const> normals,
    mochi::Span<float const> texcoords,
    mochi::Span<int const> indices,
    mochi::Span<int const> boneIndices,
    mochi::Span<mochi::real const> boneWeights,
    int weightsPerNode,
    mochi::TransformRT const& rootFromMesh,
    mochi::Real3 const& preScale,
    mochi::CoordinateSpaceConverter const& gltfFromMochi,
    mochi::Span<int const> boneToLink) {
  int const numNodes = mochi::isize(positions) / 3;
  int const numElements = mochi::isize(indices) / 3;
  if (numNodes == 0 || numElements == 0 || weightsPerNode <= 0) {
    return;
  }
  bool const hasSourceNormals = mochi::isize(normals) == numNodes * 3;
  mochi::Real3 const inversePreScale{1_r / preScale[0], 1_r / preScale[1], 1_r / preScale[2]};
  auto const vertexOffset = static_cast<uint32_t>(combined.totalVertices);

  for (int i = 0; i < numNodes; ++i) {
    mochi::Real3 const pos{
        static_cast<mochi::real>(positions[i * 3 + 0]) * preScale[0],
        static_cast<mochi::real>(positions[i * 3 + 1]) * preScale[1],
        static_cast<mochi::real>(positions[i * 3 + 2]) * preScale[2]};
    mochi::Real3 const bindPos =
        gltfFromMochi.TranslationToOutput(rootFromMesh.TransformPoint(pos));
    combined.positions.push_back(static_cast<float>(bindPos[0]));
    combined.positions.push_back(static_cast<float>(bindPos[1]));
    combined.positions.push_back(static_cast<float>(bindPos[2]));

    if (hasSourceNormals) {
      mochi::Real3 const normal{
          static_cast<mochi::real>(normals[i * 3 + 0]) * inversePreScale[0],
          static_cast<mochi::real>(normals[i * 3 + 1]) * inversePreScale[1],
          static_cast<mochi::real>(normals[i * 3 + 2]) * inversePreScale[2]};
      mochi::Real3 const bindNormal =
          gltfFromMochi.DirectionToOutput(rootFromMesh.TransformDirection(normal));
      combined.normals.push_back(static_cast<float>(bindNormal[0]));
      combined.normals.push_back(static_cast<float>(bindNormal[1]));
      combined.normals.push_back(static_cast<float>(bindNormal[2]));
    } else {
      combined.normals.push_back(0.0f);
      combined.normals.push_back(0.0f);
      combined.normals.push_back(0.0f);
    }

    if (i * 2 + 1 < mochi::isize(texcoords)) {
      combined.texcoords.push_back(texcoords[i * 2 + 0]);
      combined.texcoords.push_back(texcoords[i * 2 + 1]);
    } else {
      combined.texcoords.push_back(0.0f);
      combined.texcoords.push_back(0.0f);
    }

    std::array<uint8_t, 4> outJoints{0, 0, 0, 0};
    std::array<float, 4> outWeights{0.0f, 0.0f, 0.0f, 0.0f};
    int kept = 0;
    float weightSum = 0.0f;
    for (int k = 0; k < weightsPerNode && kept < 4; ++k) {
      int const bone = boneIndices[i * weightsPerNode + k];
      auto const weight = static_cast<float>(boneWeights[i * weightsPerNode + k]);
      if (weight <= 0.0f || bone < 0 || bone >= mochi::isize(boneToLink)) {
        continue;
      }
      int const link = boneToLink[bone];
      if (link < 0) {
        continue;
      }
      outJoints[kept] = static_cast<uint8_t>(link);
      outWeights[kept] = weight;
      weightSum += weight;
      ++kept;
    }
    if (weightSum > 0.0f) {
      for (int k = 0; k < kept; ++k) {
        outWeights[k] /= weightSum;
      }
    } else {
      // No usable influence (shouldn't happen for a valid skin); bind rigidly to the root link.
      outJoints[0] = 0;
      outWeights[0] = 1.0f;
    }
    for (int k = 0; k < 4; ++k) {
      combined.joints.push_back(outJoints[k]);
    }
    for (int k = 0; k < 4; ++k) {
      combined.weights.push_back(outWeights[k]);
    }
  }

  int numAppendedElements = 0;
  for (int e = 0; e < numElements; ++e) {
    // Indices come from an external GLB (or a .mochi.h5), so validate them against this section's
    // own vertex count before offsetting: an out-of-range value would index past the vertices just
    // appended, and the face-normal path below writes through it.
    int const raw0 = indices[e * 3 + 0];
    int const raw1 = indices[e * 3 + 1];
    int const raw2 = indices[e * 3 + 2];
    if (raw0 < 0 || raw0 >= numNodes || raw1 < 0 || raw1 >= numNodes || raw2 < 0 ||
        raw2 >= numNodes) {
      MOCHI_LOG_WARNING_ONCE(
          "ExportSkeletalGlb: skinned mesh has triangle indices outside its vertex range; those "
          "triangles are dropped from the export.");
      continue;
    }
    uint32_t const i0 = vertexOffset + static_cast<uint32_t>(raw0);
    uint32_t const i1 = vertexOffset + static_cast<uint32_t>(raw1);
    uint32_t const i2 = vertexOffset + static_cast<uint32_t>(raw2);
    combined.indices.push_back(i0);
    combined.indices.push_back(i1);
    combined.indices.push_back(i2);
    ++numAppendedElements;

    if (hasSourceNormals) {
      continue;
    }
    float const* p0 = &combined.positions[i0 * 3];
    float const* p1 = &combined.positions[i1 * 3];
    float const* p2 = &combined.positions[i2 * 3];
    std::array<float, 3> const edge1 = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
    std::array<float, 3> const edge2 = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
    std::array<float, 3> const faceNormal = {
        edge1[1] * edge2[2] - edge1[2] * edge2[1],
        edge1[2] * edge2[0] - edge1[0] * edge2[2],
        edge1[0] * edge2[1] - edge1[1] * edge2[0]};
    for (uint32_t const vi : {i0, i1, i2}) {
      combined.normals[vi * 3 + 0] += faceNormal[0];
      combined.normals[vi * 3 + 1] += faceNormal[1];
      combined.normals[vi * 3 + 2] += faceNormal[2];
    }
  }

  combined.totalVertices += numNodes;
  combined.totalIndices += numAppendedElements * 3;
}

// Appends the skin's render mesh (@ref BotPrefab::skin->renderModelFile) to the render parts, per
// distinct material, with real per-vertex weights remapped through @p skinBoneToLink. No-op if the
// bot has no skin or the skin has no render model.
void GatherSkinRenderParts(
    robotics::BotPrefab const& botPrefab,
    mochi::CoordinateSpaceConverter const& gltfFromMochi,
    mochi::Span<int const> skinBoneToLink,
    mochi::TransformRT const& rootFromSkinRoot,
    mochi::DynamicArray<ImageData>& images,
    mochi::DynamicArray<MaterialFactors>& materials,
    mochi::DynamicArray<MeshPart>& parts) {
  if (!botPrefab.skin.has_value() || botPrefab.skin->renderModelFile.empty()) {
    return;
  }
  auto sections = ReadSkinnedGlbSections(botPrefab.skin->renderModelFile.c_str());
  if (sections.empty()) {
    MOCHI_LOG_WARNING(
        "ExportSkeletalGlb: no skinned render geometry read from skin '%s'; skipping it.",
        botPrefab.skin->renderModelFile.c_str());
    return;
  }
  mochi::CoordinateSpaceConverter const mochiFromGltf(
      mochi_renderer::RenderSpace(), mochi::CoordinateSpace::Default());
  // The skin mesh is authored in the skin's own root-link frame. rootFromSkinRoot carries that
  // frame to the composed root (identity for a standalone bot; the hand's offset onto the arm for a
  // mod bot), and the skin's render-model transform is its static offset within that frame. The
  // skinning then carries it to the pose.
  mochi::TransformRT const rootFromMesh = rootFromSkinRoot *
      mochi::TransformRT(botPrefab.skin->renderModelRotation,
                         botPrefab.skin->renderModelTranslation);
  for (auto& section : sections) {
    if (section.positions.size() < 9 || section.indices.size() < 3) {
      continue;
    }
    ConvertSkinnedSectionToMochi(section, mochiFromGltf);
    // Sections sharing identical image bytes share one embedded image, so a skin split across
    // several primitives by material does not embed the same texture repeatedly.
    MaterialFactors factors = section.material;
    for (int slot = 0; slot < kNumTextureSlots; ++slot) {
      factors.images[slot] = FindOrAddImage(images, std::move(section.images[slot]));
    }
    int const materialIndex = FindOrAddMaterial(materials, factors);
    // Materials grow one at a time and every new material immediately gets its own part, so a
    // material index is also its part index. Both render gatherers run before any collision part is
    // pushed (which would add a part with no material), so the two stay in step here.
    if (materialIndex == mochi::isize(parts)) {
      MeshPart part;
      part.materialIndex = materialIndex;
      parts.push_back(std::move(part));
    }
    AppendSkinnedGeometry<float>(
        parts[materialIndex].geometry,
        mochi::MakeConstSpan(section.positions),
        mochi::MakeConstSpan(section.normals),
        mochi::MakeConstSpan(section.texcoords),
        mochi::MakeConstSpan(section.indices),
        mochi::MakeConstSpan(section.boneIndices),
        mochi::MakeConstSpan(section.boneWeights),
        /*weightsPerNode=*/4,
        rootFromMesh,
        botPrefab.skin->renderModelScale,
        gltfFromMochi,
        skinBoneToLink);
  }
}

// Appends the skin's collision mesh (a preloaded @ref BotPrefab::skin->shapeFile .mochi.h5, already
// in Mochi space and root-relative) as one untextured collision part, with real per-vertex weights
// remapped through @p skinBoneToLink. No-op if the mesh carries no triangle skinning.
void GatherSkinCollisionPart(
    mochi::ModelData const& skinModel,
    mochi::CoordinateSpaceConverter const& gltfFromMochi,
    mochi::Span<int const> skinBoneToLink,
    mochi::TransformRT const& rootFromSkinRoot,
    mochi::DynamicArray<MeshPart>& parts) {
  if (!skinModel.mesh.has_value() || !skinModel.mesh->skinning.has_value()) {
    return;
  }
  auto const& mesh = *skinModel.mesh;
  if (mesh.nodesPerElement != 3 || mesh.GetNumNodes() == 0) {
    return;
  }
  MeshPart part; // untextured (materialIndex stays -1)
  AppendSkinnedGeometry<mochi::real>(
      part.geometry,
      mochi::MakeConstSpan(mesh.coordinates),
      mochi::Span<mochi::real const>{},
      mochi::Span<float const>{},
      mochi::MakeConstSpan(mesh.connectivity),
      mochi::MakeConstSpan(mesh.skinning->indices),
      mochi::MakeConstSpan(mesh.skinning->weights),
      mesh.skinning->weightsPerNode,
      rootFromSkinRoot,
      mochi::Real3{1_r, 1_r, 1_r},
      gltfFromMochi,
      skinBoneToLink);
  if (part.geometry.totalVertices > 0) {
    parts.push_back(std::move(part));
  }
}

} // namespace

void ExportSkeletalGlb(
    char const* outputPath,
    robotics::BotPrefab const& botPrefab,
    mochi::Context* context,
    robotics::RoboticsContext* roboticsContext,
    robotics::IBotLoader const& loader,
    GlbExportOptions const& options,
    mochi::Error& error) {
  MOCHI_ERROR_RETURN(error);

  // Create temporary scene and bot for data extraction
  auto* scene = context->CreateScene("ExportGlbScene");
  MOCHI_ERROR_IF(scene == nullptr, error, "Failed to create export scene.");
  MOCHI_ERROR_RETURN(error);
  MOCHI_DEFER(context->DestroyScene(scene));

  auto* bot = roboticsContext->CreateBot(scene, botPrefab, loader, error);
  MOCHI_ERROR_IF(bot == nullptr, error, "Failed to create export bot.");
  MOCHI_ERROR_RETURN(error);
  MOCHI_DEFER(robotics::DestroyBot(scene, bot));

  auto* actor = bot->GetArticulatedActor();
  MOCHI_ERROR_IF(actor == nullptr, error, "Bot has no articulated actor.");
  MOCHI_ERROR_RETURN(error);

  auto const artInfo = actor->GetArticulatedShapeInfo(error);
  MOCHI_ERROR_RETURN(error);

  int const numLinks = mochi::isize(artInfo.rootFromLinksAtRest);
  MOCHI_ERROR_IF(numLinks == 0, error, "Articulated actor has no links.");
  MOCHI_ERROR_IF(numLinks > 255, error, "Too many links for UBYTE joint indices (max 255).");
  MOCHI_ERROR_RETURN(error);

  // GatherCollisionPart pairs these with the link transforms by index, so a shorter span would read
  // out of bounds.
  auto const linkActorHandles = actor->GetNestedLinkActors(error);
  MOCHI_ERROR_IF(
      mochi::isize(linkActorHandles) != numLinks,
      error,
      "Articulated actor reported a different number of link actors than links.");
  MOCHI_ERROR_RETURN(error);

  // CreateBot poses the articulation at BotPrefab::defaultPose, which is what the editor shows, so
  // the live link transforms are the pose to export. They come back world-relative (folding in
  // BotPrefab::worldFromRoot) while rootFromLinksAtRest is root-relative, so rebase onto the root.
  mochi::DynamicArray<mochi::TransformRT> rootFromLinks;
  rootFromLinks.resize(numLinks);
  actor->GetArticulatedLinkTransforms(mochi::MakeSpan(rootFromLinks), error);
  MOCHI_ERROR_RETURN(error);
  mochi::TransformRT const rootFromWorld = Invert(actor->GetRootTransform());
  for (auto& rootFromLink : rootFromLinks) {
    rootFromLink = rootFromWorld * rootFromLink;
  }

  // Vertex positions and inverse bind matrices together define the bind pose; the joint node
  // transforms further down always carry the current pose.
  mochi::Span<mochi::TransformRT const> const bindFromLinks =
      options.bindPose == GlbBindPose::Current ? mochi::MakeConstSpan(rootFromLinks)
                                               : artInfo.rootFromLinksAtRest;

  // Link transforms stay in Mochi space through the code below and are converted at each point
  // where they reach the GLB, keeping one conversion per written value.
  mochi::CoordinateSpaceConverter const gltfFromMochi = GltfFromMochi();

  // Skin (optional). The skin's bone indices are remapped to composed link indices -- identity for
  // a standalone skinned bot, offset onto the composed skeleton for a mod bot (via _skinBoneLinks).
  // The links the skin stands in for are represented by the skin, so their per-link rigid meshes
  // are skipped (hiddenLinks, coveredLinks).
  mochi::DynamicArray<int> const skinBoneToLink =
      robotics::ComputeSkinBoneLinkIndices(botPrefab, artInfo.linkNames);
  for (int b = 0; b < mochi::isize(skinBoneToLink); ++b) {
    if (skinBoneToLink[b] == robotics::kIndexNone) {
      // Left unmapped so the vertex loop drops the influence and renormalizes, but report it: the
      // exported skin silently loses that bone's deformation otherwise.
      MOCHI_LOG_WARNING_ONCE(
          "ExportSkeletalGlb: skin bone references link '%s', which is not in the exported "
          "skeleton; that bone's influence is dropped.",
          botPrefab._skinBoneLinks[b].c_str());
    }
  }
  // The skin mesh is authored in its own root-link frame (skin bone 0 = the skin's root link).
  // rootFromSkinRoot maps that frame to the composed root: identity for a standalone skinned bot
  // (bone 0 -> link 0, whose root-relative rest is identity), and the hand's offset onto the arm
  // for a mod bot. Skin vertices are placed through it so they land where the composed skeleton is.
  int const skinRootLink =
      (mochi::isize(skinBoneToLink) > 0 && skinBoneToLink[0] >= 0) ? skinBoneToLink[0] : 0;
  mochi::TransformRT const rootFromSkinRoot =
      (skinRootLink >= 0 && skinRootLink < mochi::isize(bindFromLinks))
      ? bindFromLinks[skinRootLink]
      : mochi::TransformRT{};
  // Load the collision skin once; it is reused below (collision covered-set derivation when
  // nonCollidingLinks is unset) and by GatherSkinCollisionPart.
  mochi::ModelData skinCollisionModel;
  bool loadFailed = false;
  if (botPrefab.skin.has_value() && !botPrefab.skin->shapeFile.empty()) {
    mochi::ErrorLog skinError;
    skinCollisionModel = loader.LoadModelData(botPrefab.skin->shapeFile, skinError);
    if (!skinError.IsOK() || !skinCollisionModel.mesh.has_value()) {
      // Continuing would drop the collision skin AND leave coveredLinks empty, so every link the
      // skin covers would export its rigid mesh as well as the skin -- overlapping geometry in a
      // file that otherwise looks fine. A declared skin shape that will not load is an error.
      MOCHI_LOG_ERROR(
          "ExportSkeletalGlb: could not load the skin collision mesh '%s'.",
          botPrefab.skin->shapeFile.c_str());
      loadFailed = true;
    }
  }
  // JOINTS_0 is emitted as unsigned bytes (see CombinedGeometry::joints), so link indices past
  // 255 wrap and bind geometry to the wrong bone. Fail rather than write a file that is quietly
  // wrong; widening the accessor to 16-bit is the real fix if this is ever hit.
  if (mochi::isize(artInfo.linkNames) > 256) {
    MOCHI_LOG_ERROR("ExportSkeletalGlb: bot has %d links.", mochi::isize(artInfo.linkNames));
  }
  MOCHI_ERROR_IF(
      mochi::isize(artInfo.linkNames) > 256,
      error,
      "Bot has more links than the 256 addressable by unsigned-byte GLB joint indices.");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      loadFailed, error, "The bot's skin declares a collision mesh that could not be loaded.");
  MOCHI_ERROR_RETURN(error);
  // The skin represents some links, so their rigid meshes are skipped: hiddenLinks are the links
  // whose render meshes its render model stands in for, coveredLinks those whose collision meshes
  // its collision mesh covers.
  mochi::DynamicArray<bool> const hiddenLinks =
      robotics::ComputeSkinHiddenLinks(botPrefab, artInfo.linkNames);
  mochi::SkinningData const* const skinCollisionSkinning =
      skinCollisionModel.mesh.has_value() && skinCollisionModel.mesh->skinning.has_value()
      ? &*skinCollisionModel.mesh->skinning
      : nullptr;
  mochi::DynamicArray<bool> const coveredLinks =
      robotics::ComputeSkinCoveredLinks(botPrefab, artInfo.linkNames, skinCollisionSkinning);

  // Gather geometry, render parts first, so each mesh owns a contiguous run of parts.
  mochi::DynamicArray<ImageData> images;
  mochi::DynamicArray<MaterialFactors> materialFactors;
  mochi::DynamicArray<MeshPart> parts;
  if (options.includeRenderMeshes) {
    GatherRenderParts(
        botPrefab,
        artInfo.linkNames,
        bindFromLinks,
        gltfFromMochi,
        mochi::MakeConstSpan(hiddenLinks),
        images,
        materialFactors,
        parts);
    GatherSkinRenderParts(
        botPrefab,
        gltfFromMochi,
        mochi::MakeConstSpan(skinBoneToLink),
        rootFromSkinRoot,
        images,
        materialFactors,
        parts);
  }
  int const numRenderParts = mochi::isize(parts);
  if (options.includeCollisionMeshes) {
    GatherCollisionPart(
        scene,
        context,
        linkActorHandles,
        bindFromLinks,
        gltfFromMochi,
        mochi::MakeConstSpan(coveredLinks),
        parts);
    GatherSkinCollisionPart(
        skinCollisionModel,
        gltfFromMochi,
        mochi::MakeConstSpan(skinBoneToLink),
        rootFromSkinRoot,
        parts);
  }
  int const numCollisionParts = mochi::isize(parts) - numRenderParts;

  // Exporting only part of what was asked for would be the more confusing outcome, so a requested
  // source that yields nothing is an error. Individual links without geometry are merely skipped.
  MOCHI_ERROR_IF(
      options.includeRenderMeshes && numRenderParts == 0,
      error,
      "Requested render meshes, but no link in this bot has a render model with geometry.");
  MOCHI_ERROR_IF(
      options.includeCollisionMeshes && numCollisionParts == 0,
      error,
      "Requested collision meshes, but no link in this bot has collision geometry.");
  MOCHI_ERROR_RETURN(error);

  int const numParts = mochi::isize(parts);
  int const numMaterials = mochi::isize(materialFactors);
  // Without geometry there is nothing for a skin to deform, and an unreferenced skin is an invalid
  // object, so a skeleton-only export is joint nodes alone.
  bool const hasSkin = numParts > 0;

  mochi::DynamicArray<MeshGroup> meshGroups;
  if (numRenderParts > 0) {
    meshGroups.push_back(MeshGroup{"BotRenderMesh", "BotRenderMeshNode", 0, numRenderParts});
  }
  if (numCollisionParts > 0) {
    meshGroups.push_back(
        MeshGroup{"BotCollisionMesh", "BotCollisionMeshNode", numRenderParts, numCollisionParts});
  }
  int const numMeshes = mochi::isize(meshGroups);

  for (auto& part : parts) {
    NormalizeNormals(part.geometry);
  }

  // Inverse bind matrices, shared by every mesh since they all bind to the one skeleton.
  mochi::DynamicArray<float> ibmData;
  if (hasSkin) {
    ibmData.resize(static_cast<size_t>(numLinks) * 16);
    for (int i = 0; i < numLinks; ++i) {
      WriteInverseBindMatrix(
          &ibmData[static_cast<size_t>(i) * 16], gltfFromMochi.TransformToOutput(bindFromLinks[i]));
    }
  }

  // From here on cgltf_data holds raw pointers into the arrays below, and cgltf_write derives every
  // JSON index from those pointers, so each array is sized once up front and never grows again.
  int const numImages = mochi::isize(images);
  int const numAccessors = numParts * 6 + (hasSkin ? 1 : 0);
  int const imageViewBase = numAccessors;
  int const numBufferViews = numAccessors + numImages;
  mochi::DynamicArray<cgltf_buffer_view> bufferViews;
  ResizeZeroed(bufferViews, numBufferViews);
  mochi::DynamicArray<cgltf_accessor> accessors;
  ResizeZeroed(accessors, numAccessors);

  // Every geometry element size here is a multiple of 4 bytes (position 12, normal 12, texcoord 8,
  // joints 4, weights 16, index 4, inverse-bind 64), so appending sequentially keeps each view's
  // offset 4-byte aligned as glTF requires. Image blobs, appended last, are padded to 4 explicitly.
  mochi::DynamicArray<uint8_t> bufferData;
  auto const appendBytes = [&bufferData](void const* src, cgltf_size numBytes) {
    auto const offset = bufferData.size();
    bufferData.resize(offset + numBytes);
    std::memcpy(&bufferData[offset], src, numBytes);
    return static_cast<cgltf_size>(offset);
  };

  for (int p = 0; p < numParts; ++p) {
    auto const& geometry = parts[p].geometry;
    auto const numVertices = static_cast<cgltf_size>(geometry.totalVertices);
    int const view = p * 6;

    std::array<float, 3> posMin{};
    std::array<float, 3> posMax{};
    ComputeBounds(geometry, posMin, posMax);

    bufferViews[view + 0].offset = appendBytes(geometry.positions.data(), numVertices * 12);
    bufferViews[view + 0].size = numVertices * 12;
    bufferViews[view + 0].stride = 12;
    bufferViews[view + 0].type = cgltf_buffer_view_type_vertices;
    accessors[view + 0].component_type = cgltf_component_type_r_32f;
    accessors[view + 0].type = cgltf_type_vec3;
    accessors[view + 0].count = numVertices;
    accessors[view + 0].buffer_view = &bufferViews[view + 0];
    accessors[view + 0].has_min = true;
    accessors[view + 0].has_max = true;
    std::memcpy(accessors[view + 0].min, posMin.data(), 12);
    std::memcpy(accessors[view + 0].max, posMax.data(), 12);

    bufferViews[view + 1].offset = appendBytes(geometry.normals.data(), numVertices * 12);
    bufferViews[view + 1].size = numVertices * 12;
    bufferViews[view + 1].stride = 12;
    bufferViews[view + 1].type = cgltf_buffer_view_type_vertices;
    accessors[view + 1].component_type = cgltf_component_type_r_32f;
    accessors[view + 1].type = cgltf_type_vec3;
    accessors[view + 1].count = numVertices;
    accessors[view + 1].buffer_view = &bufferViews[view + 1];

    bufferViews[view + 2].offset = appendBytes(geometry.texcoords.data(), numVertices * 8);
    bufferViews[view + 2].size = numVertices * 8;
    bufferViews[view + 2].stride = 8;
    bufferViews[view + 2].type = cgltf_buffer_view_type_vertices;
    accessors[view + 2].component_type = cgltf_component_type_r_32f;
    accessors[view + 2].type = cgltf_type_vec2;
    accessors[view + 2].count = numVertices;
    accessors[view + 2].buffer_view = &bufferViews[view + 2];

    bufferViews[view + 3].offset = appendBytes(geometry.joints.data(), numVertices * 4);
    bufferViews[view + 3].size = numVertices * 4;
    bufferViews[view + 3].stride = 4;
    bufferViews[view + 3].type = cgltf_buffer_view_type_vertices;
    accessors[view + 3].component_type = cgltf_component_type_r_8u;
    accessors[view + 3].type = cgltf_type_vec4;
    accessors[view + 3].count = numVertices;
    accessors[view + 3].buffer_view = &bufferViews[view + 3];

    bufferViews[view + 4].offset = appendBytes(geometry.weights.data(), numVertices * 16);
    bufferViews[view + 4].size = numVertices * 16;
    bufferViews[view + 4].stride = 16;
    bufferViews[view + 4].type = cgltf_buffer_view_type_vertices;
    accessors[view + 4].component_type = cgltf_component_type_r_32f;
    accessors[view + 4].type = cgltf_type_vec4;
    accessors[view + 4].count = numVertices;
    accessors[view + 4].buffer_view = &bufferViews[view + 4];

    auto const numIndices = static_cast<cgltf_size>(geometry.totalIndices);
    bufferViews[view + 5].offset = appendBytes(geometry.indices.data(), numIndices * 4);
    bufferViews[view + 5].size = numIndices * 4;
    bufferViews[view + 5].type = cgltf_buffer_view_type_indices;
    accessors[view + 5].component_type = cgltf_component_type_r_32u;
    accessors[view + 5].type = cgltf_type_scalar;
    accessors[view + 5].count = numIndices;
    accessors[view + 5].buffer_view = &bufferViews[view + 5];
  }

  int const ibmView = numParts * 6;
  if (hasSkin) {
    auto const ibmSize = static_cast<cgltf_size>(numLinks) * 64;
    bufferViews[ibmView].offset = appendBytes(ibmData.data(), ibmSize);
    bufferViews[ibmView].size = ibmSize;
    accessors[ibmView].component_type = cgltf_component_type_r_32f;
    accessors[ibmView].type = cgltf_type_mat4;
    accessors[ibmView].count = static_cast<cgltf_size>(numLinks);
    accessors[ibmView].buffer_view = &bufferViews[ibmView];
  }

  // Image bytes (textures) follow the geometry, each as its own buffer view -- a glTF image
  // references a buffer view directly, with no accessor. Pad to 4 bytes to keep later data aligned.
  for (int i = 0; i < numImages; ++i) {
    auto const imageSize = static_cast<cgltf_size>(images[i].bytes.size());
    bufferViews[imageViewBase + i].offset = appendBytes(images[i].bytes.data(), imageSize);
    bufferViews[imageViewBase + i].size = imageSize;
    while (bufferData.size() % 4 != 0) {
      bufferData.push_back(0);
    }
  }

  cgltf_buffer buffer = {};
  buffer.size = bufferData.size();
  buffer.data = bufferData.data();
  for (auto& bufferView : bufferViews) {
    bufferView.buffer = &buffer;
  }

  // Texture objects for the embedded images (one shared trilinear/repeat sampler, as the source
  // assets use). These outlive cgltf_write_file; the material loop below points textured materials
  // at them.
  cgltf_sampler sampler = {};
  sampler.mag_filter = cgltf_filter_type_linear;
  sampler.min_filter = cgltf_filter_type_linear_mipmap_linear;
  sampler.wrap_s = cgltf_wrap_mode_repeat;
  sampler.wrap_t = cgltf_wrap_mode_repeat;
  mochi::DynamicArray<cgltf_image> cgltfImages;
  mochi::DynamicArray<cgltf_texture> cgltfTextures;
  mochi::DynamicArray<std::array<char, 24>> imageNames;
  ResizeZeroed(cgltfImages, numImages);
  ResizeZeroed(cgltfTextures, numImages);
  imageNames.resize(numImages);
  for (int i = 0; i < numImages; ++i) {
    std::snprintf(imageNames[i].data(), imageNames[i].size(), "Image_%d", i);
    cgltfImages[i].name = imageNames[i].data();
    cgltfImages[i].buffer_view = &bufferViews[imageViewBase + i];
    cgltfImages[i].mime_type = const_cast<char*>(images[i].mimeType.c_str());
    cgltfTextures[i].image = &cgltfImages[i];
    cgltfTextures[i].sampler = &sampler;
  }

  // cgltf_material::name is non-owning, so its storage has to outlive cgltf_write_file. Fixed-size
  // buffers sidestep both reallocation and DynamicString's small-string optimization, either of
  // which would move the characters out from under the pointer.
  mochi::DynamicArray<cgltf_material> materials;
  mochi::DynamicArray<std::array<char, 24>> materialNames;
  ResizeZeroed(materials, numMaterials);
  materialNames.resize(numMaterials);
  for (int i = 0; i < numMaterials; ++i) {
    std::snprintf(materialNames[i].data(), materialNames[i].size(), "Material_%d", i);
    materials[i].name = materialNames[i].data();
    materials[i].has_pbr_metallic_roughness = 1;
    auto& pbr = materials[i].pbr_metallic_roughness;
    std::memcpy(
        pbr.base_color_factor, materialFactors[i].baseColor.data(), sizeof(pbr.base_color_factor));
    pbr.metallic_factor = materialFactors[i].metallic;
    pbr.roughness_factor = materialFactors[i].roughness;
    std::memcpy(
        materials[i].emissive_factor,
        materialFactors[i].emissive.data(),
        sizeof(materials[i].emissive_factor));
    // emissive_factor is clamped to [0, 1] by the spec, so anything brighter has to live in
    // KHR_materials_emissive_strength. Only flag the extension when it carries something, since
    // cgltf_write emits an extensionsUsed entry for every material that sets it.
    if (materialFactors[i].emissiveStrength != 1.0f) {
      materials[i].has_emissive_strength = 1;
      materials[i].emissive_strength.emissive_strength = materialFactors[i].emissiveStrength;
    }
    materials[i].alpha_mode =
        materialFactors[i].baseColor[3] < 1.0f ? cgltf_alpha_mode_blend : cgltf_alpha_mode_opaque;
    // Winding in the source render meshes is not dependable, so draw both sides rather than risk
    // geometry disappearing to back-face culling.
    materials[i].double_sided = 1;
    auto const textureOf = [&](TextureSlot slot) -> cgltf_texture* {
      int const image = materialFactors[i].images[slot];
      return image >= 0 && image < numImages ? &cgltfTextures[image] : nullptr;
    };
    pbr.base_color_texture.texture = textureOf(kBaseColor);
    pbr.metallic_roughness_texture.texture = textureOf(kMetallicRoughness);
    materials[i].normal_texture.texture = textureOf(kNormal);
    materials[i].normal_texture.scale = materialFactors[i].normalScale;
    materials[i].occlusion_texture.texture = textureOf(kOcclusion);
    materials[i].occlusion_texture.scale = materialFactors[i].occlusionStrength;
    materials[i].emissive_texture.texture = textureOf(kEmissive);
  }

  mochi::DynamicArray<cgltf_attribute> attributes;
  ResizeZeroed(attributes, numParts * 5);
  mochi::DynamicArray<cgltf_primitive> primitives;
  ResizeZeroed(primitives, numParts);
  for (int p = 0; p < numParts; ++p) {
    int const view = p * 6;
    int const attribute = p * 5;
    attributes[attribute + 0].name = const_cast<char*>("POSITION");
    attributes[attribute + 0].type = cgltf_attribute_type_position;
    attributes[attribute + 0].index = 0;
    attributes[attribute + 0].data = &accessors[view + 0];
    attributes[attribute + 1].name = const_cast<char*>("NORMAL");
    attributes[attribute + 1].type = cgltf_attribute_type_normal;
    attributes[attribute + 1].index = 0;
    attributes[attribute + 1].data = &accessors[view + 1];
    attributes[attribute + 2].name = const_cast<char*>("TEXCOORD_0");
    attributes[attribute + 2].type = cgltf_attribute_type_texcoord;
    attributes[attribute + 2].index = 0;
    attributes[attribute + 2].data = &accessors[view + 2];
    attributes[attribute + 3].name = const_cast<char*>("JOINTS_0");
    attributes[attribute + 3].type = cgltf_attribute_type_joints;
    attributes[attribute + 3].index = 0;
    attributes[attribute + 3].data = &accessors[view + 3];
    attributes[attribute + 4].name = const_cast<char*>("WEIGHTS_0");
    attributes[attribute + 4].type = cgltf_attribute_type_weights;
    attributes[attribute + 4].index = 0;
    attributes[attribute + 4].data = &accessors[view + 4];

    primitives[p].type = cgltf_primitive_type_triangles;
    primitives[p].indices = &accessors[view + 5];
    primitives[p].attributes = &attributes[attribute];
    primitives[p].attributes_count = 5;
    if (parts[p].materialIndex >= 0) {
      primitives[p].material = &materials[parts[p].materialIndex];
    }
  }

  mochi::DynamicArray<cgltf_mesh> meshes;
  ResizeZeroed(meshes, numMeshes);
  for (int m = 0; m < numMeshes; ++m) {
    meshes[m].name = const_cast<char*>(meshGroups[m].meshName);
    meshes[m].primitives = &primitives[meshGroups[m].firstPart];
    meshes[m].primitives_count = static_cast<cgltf_size>(meshGroups[m].numParts);
  }

  // Nodes: one per link, plus one per mesh
  int const totalNodes = numLinks + numMeshes;
  mochi::DynamicArray<cgltf_node> nodes;
  ResizeZeroed(nodes, totalNodes);

  // Build per-node children lists
  mochi::DynamicArray<mochi::DynamicArray<cgltf_node*>> childrenLists;
  childrenLists.resize(numLinks);

  for (int i = 0; i < numLinks; ++i) {
    nodes[i].name = const_cast<char*>(artInfo.linkNames[i].c_str());
    nodes[i].has_translation = true;
    nodes[i].has_rotation = true;
    nodes[i].has_scale = false;

    // Compute local transform relative to parent. Converting the composed local transform matches
    // converting each link transform first: the conversion is a similarity transform, so the
    // change of basis cancels between a parent's inverse and its child.
    mochi::TransformRT localTransform;
    if (artInfo.parents[i] < 0) {
      localTransform = rootFromLinks[i];
    } else {
      localTransform = Invert(rootFromLinks[artInfo.parents[i]]) * rootFromLinks[i];
    }
    localTransform = gltfFromMochi.TransformToOutput(localTransform);

    mochi::Real3 const translation = localTransform.GetTranslation();
    nodes[i].translation[0] = static_cast<float>(translation[0]);
    nodes[i].translation[1] = static_cast<float>(translation[1]);
    nodes[i].translation[2] = static_cast<float>(translation[2]);

    // glTF quaternion is XYZW, same as Mochi
    mochi::Real4 const rotation = localTransform.GetRotation().ToReal4();
    nodes[i].rotation[0] = static_cast<float>(rotation[0]);
    nodes[i].rotation[1] = static_cast<float>(rotation[1]);
    nodes[i].rotation[2] = static_cast<float>(rotation[2]);
    nodes[i].rotation[3] = static_cast<float>(rotation[3]);

    if (artInfo.parents[i] >= 0) {
      nodes[i].parent = &nodes[artInfo.parents[i]];
      childrenLists[artInfo.parents[i]].push_back(&nodes[i]);
    }
  }

  // Assign children pointers only once every list is complete, so no list can still reallocate.
  for (int i = 0; i < numLinks; ++i) {
    if (!childrenLists[i].empty()) {
      nodes[i].children = childrenLists[i].data();
      nodes[i].children_count = childrenLists[i].size();
    }
  }

  // One skin shared by every mesh: a glTF skin is not tied to a mesh, so both meshes deform with
  // the same joint hierarchy and an importer builds a single armature.
  cgltf_skin skin = {};
  mochi::DynamicArray<cgltf_node*> jointPtrs;
  if (hasSkin) {
    jointPtrs.resize(numLinks);
    for (int i = 0; i < numLinks; ++i) {
      jointPtrs[i] = &nodes[i];
    }
    skin.name = const_cast<char*>("BotSkeleton");
    skin.joints = jointPtrs.data();
    skin.joints_count = static_cast<cgltf_size>(numLinks);
    skin.inverse_bind_matrices = &accessors[ibmView];
    // Find skeleton root (first node with parent == -1)
    for (int i = 0; i < numLinks; ++i) {
      if (artInfo.parents[i] < 0) {
        skin.skeleton = &nodes[i];
        break;
      }
    }
  }

  for (int m = 0; m < numMeshes; ++m) {
    int const node = numLinks + m;
    nodes[node].name = const_cast<char*>(meshGroups[m].nodeName);
    nodes[node].mesh = &meshes[m];
    nodes[node].skin = &skin;
    nodes[node].has_translation = false;
    nodes[node].has_rotation = false;
    nodes[node].has_scale = false;
  }

  // Scene: root skeleton nodes + mesh nodes
  mochi::DynamicArray<cgltf_node*> sceneRoots;
  for (int i = 0; i < numLinks; ++i) {
    if (artInfo.parents[i] < 0) {
      sceneRoots.push_back(&nodes[i]);
    }
  }
  for (int m = 0; m < numMeshes; ++m) {
    sceneRoots.push_back(&nodes[numLinks + m]);
  }

  cgltf_scene gltfScene = {};
  gltfScene.name = const_cast<char*>("Scene");
  gltfScene.nodes = sceneRoots.data();
  gltfScene.nodes_count = sceneRoots.size();

  // Assemble cgltf_data
  cgltf_data data = {};
  data.file_type = cgltf_file_type_glb;
  data.asset.version = const_cast<char*>("2.0");
  data.asset.generator = const_cast<char*>("SuperDex Studio");
  data.nodes = nodes.data();
  data.nodes_count = static_cast<cgltf_size>(totalNodes);
  data.scenes = &gltfScene;
  data.scenes_count = 1;
  data.scene = &gltfScene;
  if (numParts > 0) {
    data.meshes = meshes.data();
    data.meshes_count = static_cast<cgltf_size>(numMeshes);
    data.accessors = accessors.data();
    data.accessors_count = accessors.size();
    data.buffer_views = bufferViews.data();
    data.buffer_views_count = bufferViews.size();
    data.buffers = &buffer;
    data.buffers_count = 1;
    data.skins = &skin;
    data.skins_count = 1;
    data.bin = bufferData.data();
    data.bin_size = bufferData.size();
  }
  if (numMaterials > 0) {
    data.materials = materials.data();
    data.materials_count = static_cast<cgltf_size>(numMaterials);
  }
  if (numImages > 0) {
    data.images = cgltfImages.data();
    data.images_count = static_cast<cgltf_size>(numImages);
    data.textures = cgltfTextures.data();
    data.textures_count = static_cast<cgltf_size>(numImages);
    data.samplers = &sampler;
    data.samplers_count = 1;
  }

  // Write GLB file
  cgltf_options opts = {};
  opts.type = cgltf_file_type_glb;
  cgltf_result const result = cgltf_write_file(&opts, outputPath, &data);
  MOCHI_ERROR_IF(result != cgltf_result_success, error, "Failed to write GLB file.");
}

} // namespace superdex::studio
