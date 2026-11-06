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

#include <mochi_renderer/utils.h>

#include "collada_reader.h"
#include "deformable_source.h"
#include "third_party/stl_reader.h"

#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/log.h>

#include <cgltf.h>
#include <draco/compression/decode.h>
#include <meshoptimizer.h>
#include <tiny_obj_loader.h>
#include <tinyxml2.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mochi_renderer {

// Appends a "%.9g"-formatted float to `out` (the GLB writer's numeric precision).
static void AppendFloat(std::string& out, float value) {
  char buf[32];
  int const n = snprintf(buf, sizeof(buf), "%.9g", value);
  out.append(buf, static_cast<size_t>(n));
}

// Appends a decimal integer to `out`.
static void AppendUint(std::string& out, size_t value) {
  char buf[32];
  int const n = snprintf(buf, sizeof(buf), "%zu", value);
  out.append(buf, static_cast<size_t>(n));
}

// Appends `bytes` to `bin`, then pads to a 4-byte boundary so the next view stays aligned.
static size_t AppendAligned(std::vector<uint8_t>& bin, void const* bytes, size_t size) {
  size_t const offset = bin.size();
  bin.resize(offset + size);
  if (size > 0) {
    auto const* const source = static_cast<uint8_t const*>(bytes);
    std::copy(source, source + size, bin.begin() + static_cast<std::ptrdiff_t>(offset));
  }
  bin.resize((bin.size() + 3) & ~size_t{3}, 0);
  return offset;
}

std::vector<uint8_t> BuildGlbFromMeshSections(std::vector<MeshSection> const& sections) {
  static_assert(sizeof(int) == sizeof(uint32_t), "int must be 32 bits for GLB index data");

  // A byte range in the binary buffer, emitted as one bufferView.
  struct View {
    size_t offset;
    size_t bytes;
  };

  // Per-section bookkeeping needed to emit bufferViews/accessors after the
  // binary buffer layout is fixed. Material fields are copied here so the JSON
  // emitters never re-index the original `sections` (which may contain skipped
  // empty sections, making positional lookups unsafe).
  struct SectionLayout {
    size_t vertexCount;
    size_t indexCount;
    // Accessor (and bufferView) indices; accessors and their views are 1:1. Optional streams
    // are kNone when absent.
    size_t posView;
    size_t normView;
    size_t uvView;
    size_t idxView;
    size_t texture; // index into the glTF textures, or kNone
    float minPos[3];
    float maxPos[3];
    std::array<float, 4> baseColor;
    float metallic;
    float roughness;
    std::array<float, 3> emissive;
    float emissiveStrength;
  };
  constexpr size_t kNone = std::numeric_limits<size_t>::max();

  std::vector<SectionLayout> layouts;
  layouts.reserve(sections.size());
  std::vector<View> views; // accessor views first, then image views
  std::vector<TextureImage const*> images; // distinct textures, in first-use order

  // Binary buffer layout: for each non-empty section, [positions | normals | texcoords |
  // indices], all 4-byte aligned (float/uint32); then each distinct image.
  std::vector<uint8_t> bin;
  auto addView = [&](void const* data, size_t bytes) {
    views.push_back(View{AppendAligned(bin, data, bytes), bytes});
    return views.size() - 1;
  };
  for (MeshSection const& s : sections) {
    if (s.positions.empty() || s.indices.empty()) {
      continue;
    }
    SectionLayout layout{};
    layout.vertexCount = s.positions.size() / 3;
    layout.indexCount = s.indices.size();
    // `hasNormals` / `hasTexcoords` are the single source of truth for presence (see
    // @ref MeshSection). A section that advertises a stream must supply exactly one entry per
    // vertex; otherwise we would emit an accessor that outruns its bufferView.
    MOCHI_ASSERT(
        !s.hasNormals || s.normals.size() == s.positions.size(),
        "MeshSection.hasNormals is set but normals do not match positions");
    MOCHI_ASSERT(
        !s.hasTexcoords || s.texcoords.size() == 2 * layout.vertexCount,
        "MeshSection.hasTexcoords is set but texcoords do not match positions");
    layout.baseColor = s.baseColor;
    layout.metallic = s.metallic;
    layout.roughness = s.roughness;
    layout.emissive = s.emissive;
    layout.emissiveStrength = s.emissiveStrength;

    for (int j = 0; j < 3; ++j) {
      layout.minPos[j] = std::numeric_limits<float>::max();
      layout.maxPos[j] = std::numeric_limits<float>::lowest();
    }
    for (size_t i = 0; i < layout.vertexCount; ++i) {
      for (int j = 0; j < 3; ++j) {
        float const v = s.positions[i * 3 + j];
        layout.minPos[j] = std::min(layout.minPos[j], v);
        layout.maxPos[j] = std::max(layout.maxPos[j], v);
      }
    }

    layout.posView = addView(s.positions.data(), s.positions.size() * sizeof(float));
    layout.normView =
        s.hasNormals ? addView(s.normals.data(), s.normals.size() * sizeof(float)) : kNone;
    layout.uvView =
        s.hasTexcoords ? addView(s.texcoords.data(), s.texcoords.size() * sizeof(float)) : kNone;
    layout.idxView = addView(s.indices.data(), layout.indexCount * sizeof(uint32_t));

    layout.texture = kNone;
    if (s.hasTexcoords && s.baseColorTexture && !s.baseColorTexture->bytes.empty()) {
      auto const it = std::find(images.begin(), images.end(), s.baseColorTexture.get());
      layout.texture = static_cast<size_t>(it - images.begin());
      if (it == images.end()) {
        images.push_back(s.baseColorTexture.get());
      }
    }
    layouts.push_back(layout);
  }

  if (layouts.empty()) {
    return {};
  }

  size_t const numAccessors = views.size();
  for (TextureImage const* image : images) {
    addView(image->bytes.data(), image->bytes.size());
  }
  size_t const binSize = bin.size();

  // A glTF extension must be declared at the top level before any material may reference it.
  bool const usesEmissiveStrength =
      std::any_of(layouts.begin(), layouts.end(), [](SectionLayout const& layout) {
        return layout.emissiveStrength != 1.0f;
      });

  // Build the glTF JSON descriptor. One primitive + one material per section.
  std::string json;
  json.reserve(1024 + layouts.size() * 1024);
  json += R"({"asset":{"version":"2.0","generator":"mochi_renderer"},)";
  if (usesEmissiveStrength) {
    json += R"("extensionsUsed":["KHR_materials_emissive_strength"],)";
  }
  json += R"("scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],)"
          R"("meshes":[{"primitives":[)";
  for (size_t i = 0; i < layouts.size(); ++i) {
    SectionLayout const& layout = layouts[i];
    if (i > 0) {
      json += ',';
    }
    json += R"({"attributes":{"POSITION":)";
    AppendUint(json, layout.posView);
    if (layout.normView != kNone) {
      json += R"(,"NORMAL":)";
      AppendUint(json, layout.normView);
    }
    if (layout.uvView != kNone) {
      json += R"(,"TEXCOORD_0":)";
      AppendUint(json, layout.uvView);
    }
    json += R"(},"indices":)";
    AppendUint(json, layout.idxView);
    json += R"(,"material":)";
    AppendUint(json, i);
    json += '}';
  }
  json += R"(]}],"materials":[)";
  for (size_t i = 0; i < layouts.size(); ++i) {
    SectionLayout const& layout = layouts[i];
    if (i > 0) {
      json += ',';
    }
    json += R"({"pbrMetallicRoughness":{"baseColorFactor":[)";
    AppendFloat(json, layout.baseColor[0]);
    json += ',';
    AppendFloat(json, layout.baseColor[1]);
    json += ',';
    AppendFloat(json, layout.baseColor[2]);
    json += ',';
    AppendFloat(json, layout.baseColor[3]);
    json += ']';
    if (layout.texture != kNone) {
      json += R"(,"baseColorTexture":{"index":)";
      AppendUint(json, layout.texture);
      json += '}';
    }
    json += R"(,"metallicFactor":)";
    AppendFloat(json, layout.metallic);
    json += R"(,"roughnessFactor":)";
    AppendFloat(json, layout.roughness);
    json += '}';
    // glTF defaults emissiveFactor to [0,0,0] and emissiveStrength to 1, so both are omitted when
    // unused rather than written out redundantly.
    if (layout.emissive != std::array<float, 3>{0.0f, 0.0f, 0.0f}) {
      json += R"(,"emissiveFactor":[)";
      AppendFloat(json, layout.emissive[0]);
      json += ',';
      AppendFloat(json, layout.emissive[1]);
      json += ',';
      AppendFloat(json, layout.emissive[2]);
      json += ']';
    }
    if (layout.emissiveStrength != 1.0f) {
      json += R"(,"extensions":{"KHR_materials_emissive_strength":{"emissiveStrength":)";
      AppendFloat(json, layout.emissiveStrength);
      json += R"(}})";
    }
    // Imported DAE/OBJ/STL geometry often has inconsistent triangle winding;
    // render both faces so sections never silently disappear to back-face
    // culling. This is the minimum-viable choice (see @ref MeshSection).
    json += R"(,"doubleSided":true})";
  }
  json += ']';
  if (!images.empty()) {
    // One linear, mipmapped, repeating sampler serves every texture.
    json += R"(,"samplers":[{"magFilter":9729,"minFilter":9987,"wrapS":10497,"wrapT":10497}])";
    json += R"(,"textures":[)";
    for (size_t t = 0; t < images.size(); ++t) {
      json += t > 0 ? R"(,{"sampler":0,"source":)" : R"({"sampler":0,"source":)";
      AppendUint(json, t);
      json += '}';
    }
    json += R"(],"images":[)";
    for (size_t t = 0; t < images.size(); ++t) {
      json += t > 0 ? R"(,{"bufferView":)" : R"({"bufferView":)";
      AppendUint(json, numAccessors + t);
      json += R"(,"mimeType":")";
      json += images[t]->mimeType;
      json += R"("})";
    }
    json += ']';
  }
  json += R"(,"accessors":[)";
  for (size_t i = 0; i < layouts.size(); ++i) {
    SectionLayout const& layout = layouts[i];
    if (i > 0) {
      json += ',';
    }
    json += R"({"bufferView":)";
    AppendUint(json, layout.posView);
    json += R"(,"componentType":5126,"count":)";
    AppendUint(json, layout.vertexCount);
    json += R"(,"type":"VEC3","min":[)";
    AppendFloat(json, layout.minPos[0]);
    json += ',';
    AppendFloat(json, layout.minPos[1]);
    json += ',';
    AppendFloat(json, layout.minPos[2]);
    json += R"(],"max":[)";
    AppendFloat(json, layout.maxPos[0]);
    json += ',';
    AppendFloat(json, layout.maxPos[1]);
    json += ',';
    AppendFloat(json, layout.maxPos[2]);
    json += R"(]})";
    if (layout.normView != kNone) {
      json += R"(,{"bufferView":)";
      AppendUint(json, layout.normView);
      json += R"(,"componentType":5126,"count":)";
      AppendUint(json, layout.vertexCount);
      json += R"(,"type":"VEC3"})";
    }
    if (layout.uvView != kNone) {
      json += R"(,{"bufferView":)";
      AppendUint(json, layout.uvView);
      json += R"(,"componentType":5126,"count":)";
      AppendUint(json, layout.vertexCount);
      json += R"(,"type":"VEC2"})";
    }
    json += R"(,{"bufferView":)";
    AppendUint(json, layout.idxView);
    json += R"(,"componentType":5125,"count":)";
    AppendUint(json, layout.indexCount);
    json += R"(,"type":"SCALAR"})";
  }
  json += R"(],"bufferViews":[)";
  for (size_t v = 0; v < views.size(); ++v) {
    if (v > 0) {
      json += ',';
    }
    json += R"({"buffer":0,"byteOffset":)";
    AppendUint(json, views[v].offset);
    json += R"(,"byteLength":)";
    AppendUint(json, views[v].bytes);
    json += '}';
  }
  json += R"(],"buffers":[{"byteLength":)";
  AppendUint(json, binSize);
  json += R"(}]})";

  size_t const jsonLen = json.size();
  size_t const jsonPadding = (4 - (jsonLen % 4)) % 4;
  size_t const jsonChunkSize = jsonLen + jsonPadding;
  size_t const binPadding = (4 - (binSize % 4)) % 4;
  size_t const binChunkSize = binSize + binPadding;

  // GLB = header(12) + json_chunk(8 + data) + bin_chunk(8 + data)
  size_t const totalSize = 12 + 8 + jsonChunkSize + 8 + binChunkSize;
  std::vector<uint8_t> glb(totalSize, 0);
  uint8_t* ptr = glb.data();

  // GLB header
  auto magic = uint32_t{0x46546C67}; // "glTF"
  auto version = uint32_t{2};
  auto length = static_cast<uint32_t>(totalSize);
  std::memcpy(ptr, &magic, 4);
  ptr += 4;
  std::memcpy(ptr, &version, 4);
  ptr += 4;
  std::memcpy(ptr, &length, 4);
  ptr += 4;

  // JSON chunk
  auto jsonChunkLen = static_cast<uint32_t>(jsonChunkSize);
  auto jsonChunkType = uint32_t{0x4E4F534A}; // "JSON"
  std::memcpy(ptr, &jsonChunkLen, 4);
  ptr += 4;
  std::memcpy(ptr, &jsonChunkType, 4);
  ptr += 4;
  std::memcpy(ptr, json.data(), jsonLen);
  ptr += jsonLen;
  std::memset(ptr, ' ', jsonPadding); // JSON padding uses spaces per GLB spec
  ptr += jsonPadding;

  // BIN chunk
  auto binChunkLen = static_cast<uint32_t>(binChunkSize);
  auto binChunkType = uint32_t{0x004E4942}; // "BIN\0"
  std::memcpy(ptr, &binChunkLen, 4);
  ptr += 4;
  std::memcpy(ptr, &binChunkType, 4);
  ptr += 4;
  std::memcpy(ptr, bin.data(), binSize);
  ptr += binSize;
  std::memset(ptr, 0, binPadding);

  return glb;
}

// --- Compressed geometry -----------------------------------------------------
//
// cgltf parses KHR_draco_mesh_compression and EXT_meshopt_compression but
// decodes neither: it leaves the affected accessors reading from nothing, and
// cgltf_accessor_unpack_floats then silently zero-fills. Both extensions are
// resolved below, before extraction, by populating whatever
// cgltf_buffer_view_data returns, so the extraction code needs no knowledge of
// compression.

namespace {

// One decoded Draco stream (indices or a single attribute), exposed to cgltf as
// a synthetic buffer view. These objects are not part of cgltf_data, so
// cgltf_free does not release them; their owner must outlive extraction.
struct DecodedStream {
  std::vector<uint8_t> bytes;
  cgltf_buffer buffer{};
  cgltf_buffer_view view{};
};

using DecodedStreams = std::vector<std::unique_ptr<DecodedStream>>;

} // namespace

// Allocates `byteSize` bytes owned by `streams`, points `accessor` at them and
// returns the writable storage. `accessor.stride` is deliberately left alone: a
// Draco accessor has no buffer view, so cgltf already set it to the tightly
// packed element size that the decoded data uses.
static uint8_t*
AttachDecodedStream(cgltf_accessor& accessor, size_t byteSize, DecodedStreams& streams) {
  std::unique_ptr<DecodedStream> const& stream =
      streams.emplace_back(std::make_unique<DecodedStream>());
  stream->bytes.resize(byteSize);
  stream->buffer.size = byteSize;
  stream->buffer.data = stream->bytes.data();
  stream->view.buffer = &stream->buffer;
  stream->view.size = byteSize;
  stream->view.data = stream->bytes.data();
  accessor.offset = 0;
  accessor.buffer_view = &stream->view;
  return stream->bytes.data();
}

template <typename T>
static void WriteDracoFaces(draco::Mesh const& mesh, uint8_t* dest) {
  auto* out = reinterpret_cast<T*>(dest);
  for (uint32_t f = 0, n = mesh.num_faces(); f < n; ++f) {
    draco::Mesh::Face const& face = mesh.face(draco::FaceIndex(f));
    out[f * 3 + 0] = static_cast<T>(face[0].value());
    out[f * 3 + 1] = static_cast<T>(face[1].value());
    out[f * 3 + 2] = static_cast<T>(face[2].value());
  }
}

template <typename T>
static void WriteDracoAttribute(
    draco::PointAttribute const& attr,
    uint32_t pointCount,
    int8_t numComponents,
    uint8_t* dest) {
  auto* out = reinterpret_cast<T*>(dest);
  for (uint32_t p = 0; p < pointCount; ++p, out += numComponents) {
    attr.ConvertValue(attr.mapped_index(draco::PointIndex(p)), numComponents, out);
  }
}

// Fills the primitive's index accessor from the decoded mesh's face list.
static bool
DecodeDracoIndices(cgltf_accessor& indices, draco::Mesh const& mesh, DecodedStreams& streams) {
  cgltf_size const indexCount = cgltf_size{mesh.num_faces()} * 3;
  if (indices.count != indexCount) {
    MOCHI_LOG_WARNING(
        "ReadGlb: Draco index count mismatch (accessor %zu, decoded %zu).",
        static_cast<size_t>(indices.count),
        static_cast<size_t>(indexCount));
    return false;
  }
  uint8_t* dest = AttachDecodedStream(indices, indexCount * indices.stride, streams);
  switch (indices.component_type) {
    case cgltf_component_type_r_8u:
      WriteDracoFaces<uint8_t>(mesh, dest);
      return true;
    case cgltf_component_type_r_16u:
      WriteDracoFaces<uint16_t>(mesh, dest);
      return true;
    case cgltf_component_type_r_32u:
      WriteDracoFaces<uint32_t>(mesh, dest);
      return true;
    case cgltf_component_type_invalid:
    case cgltf_component_type_r_8:
    case cgltf_component_type_r_16:
    case cgltf_component_type_r_32f:
    case cgltf_component_type_max_enum:
      break;
  }
  MOCHI_LOG_WARNING("ReadGlb: unsupported component type for Draco indices.");
  return false;
}

// Fills one attribute accessor from the decoded mesh, converting to the
// component type the accessor declares.
static bool DecodeDracoAttribute(
    cgltf_accessor& target,
    draco::Mesh const& mesh,
    draco::PointAttribute const& attr,
    DecodedStreams& streams) {
  auto const pointCount = static_cast<cgltf_size>(mesh.num_points());
  if (target.count != pointCount) {
    MOCHI_LOG_WARNING(
        "ReadGlb: Draco vertex count mismatch (accessor %zu, decoded %zu).",
        static_cast<size_t>(target.count),
        static_cast<size_t>(pointCount));
    return false;
  }
  auto const numComponents = static_cast<int8_t>(attr.num_components());
  if (static_cast<cgltf_size>(numComponents) != cgltf_num_components(target.type)) {
    MOCHI_LOG_WARNING("ReadGlb: Draco attribute component count does not match its accessor.");
    return false;
  }

  uint8_t* dest = AttachDecodedStream(target, pointCount * target.stride, streams);
  auto const count = static_cast<uint32_t>(pointCount);
  switch (target.component_type) {
    case cgltf_component_type_r_8:
      WriteDracoAttribute<int8_t>(attr, count, numComponents, dest);
      return true;
    case cgltf_component_type_r_8u:
      WriteDracoAttribute<uint8_t>(attr, count, numComponents, dest);
      return true;
    case cgltf_component_type_r_16:
      WriteDracoAttribute<int16_t>(attr, count, numComponents, dest);
      return true;
    case cgltf_component_type_r_16u:
      WriteDracoAttribute<uint16_t>(attr, count, numComponents, dest);
      return true;
    case cgltf_component_type_r_32u:
      WriteDracoAttribute<uint32_t>(attr, count, numComponents, dest);
      return true;
    case cgltf_component_type_r_32f:
      WriteDracoAttribute<float>(attr, count, numComponents, dest);
      return true;
    case cgltf_component_type_invalid:
    case cgltf_component_type_max_enum:
      break;
  }
  MOCHI_LOG_WARNING("ReadGlb: unsupported component type for Draco vertices.");
  return false;
}

// Rewires one KHR_draco_mesh_compression primitive's accessors onto `mesh`.
static bool DecodeDracoPrimitive(
    cgltf_data const& data,
    cgltf_primitive& prim,
    draco::Mesh const& mesh,
    DecodedStreams& streams) {
  if (prim.indices != nullptr && prim.indices->buffer_view == nullptr &&
      !DecodeDracoIndices(*prim.indices, mesh, streams)) {
    return false;
  }

  cgltf_draco_mesh_compression const& compression = prim.draco_mesh_compression;
  for (cgltf_size a = 0; a < compression.attributes_count; ++a) {
    cgltf_attribute const& source = compression.attributes[a];
    // In the extension, an attribute's value is its id within the Draco stream,
    // not an accessor index; cgltf still resolves it against `data.accessors`.
    auto const attributeId = static_cast<uint32_t>(source.data - data.accessors);

    cgltf_accessor* target = nullptr;
    for (cgltf_size i = 0; i < prim.attributes_count; ++i) {
      if (prim.attributes[i].type == source.type && prim.attributes[i].index == source.index) {
        target = prim.attributes[i].data;
        break;
      }
    }
    if (target == nullptr || target->buffer_view != nullptr) {
      continue; // No matching accessor, or another primitive already filled it.
    }

    draco::PointAttribute const* attr = mesh.GetAttributeByUniqueId(attributeId);
    if (attr == nullptr) {
      MOCHI_LOG_WARNING("ReadGlb: Draco stream has no attribute with id %u.", attributeId);
      return false;
    }
    if (!DecodeDracoAttribute(*target, mesh, *attr, streams)) {
      return false;
    }
  }
  return true;
}

// Decodes every KHR_draco_mesh_compression primitive in `data`. Primitives
// sharing a compressed buffer view decode it only once.
static bool DecodeDracoMeshes(cgltf_data& data, DecodedStreams& streams) {
  std::unordered_map<cgltf_buffer_view const*, std::unique_ptr<draco::Mesh>> decoded;
  for (cgltf_size m = 0; m < data.meshes_count; ++m) {
    cgltf_mesh& mesh = data.meshes[m];
    for (cgltf_size p = 0; p < mesh.primitives_count; ++p) {
      cgltf_primitive& prim = mesh.primitives[p];
      if (!prim.has_draco_mesh_compression) {
        continue;
      }
      cgltf_buffer_view const* source = prim.draco_mesh_compression.buffer_view;
      auto [it, inserted] = decoded.try_emplace(source);
      if (inserted) {
        uint8_t const* compressed = cgltf_buffer_view_data(source);
        if (compressed == nullptr) {
          MOCHI_LOG_WARNING("ReadGlb: Draco buffer view has no data.");
          return false;
        }
        draco::DecoderBuffer buffer;
        buffer.Init(reinterpret_cast<char const*>(compressed), source->size);
        draco::Decoder decoder;
        draco::StatusOr<std::unique_ptr<draco::Mesh>> result =
            decoder.DecodeMeshFromBuffer(&buffer);
        if (!result.ok()) {
          MOCHI_LOG_WARNING("ReadGlb: Draco decoding failed: %s", result.status().error_msg());
          return false;
        }
        it->second = std::move(result).value();
      }
      if (!DecodeDracoPrimitive(data, prim, *it->second, streams)) {
        return false;
      }
    }
  }
  return true;
}

// Decodes every EXT_meshopt_compression buffer view in place. The decoded bytes
// are malloc'd because cgltf_free takes ownership of `buffer_view.data`.
static bool DecodeMeshoptBufferViews(cgltf_data& data) {
  for (cgltf_size i = 0; i < data.buffer_views_count; ++i) {
    cgltf_buffer_view& view = data.buffer_views[i];
    if (!view.has_meshopt_compression) {
      continue;
    }
    cgltf_meshopt_compression const& compression = view.meshopt_compression;
    if (compression.stride == 0 ||
        compression.count > std::numeric_limits<size_t>::max() / compression.stride) {
      MOCHI_LOG_WARNING("ReadGlb: meshopt buffer view has an invalid stride or count.");
      return false;
    }
    auto const* source = static_cast<uint8_t const*>(compression.buffer->data);
    if (source == nullptr) {
      MOCHI_LOG_WARNING("ReadGlb: meshopt buffer has no data.");
      return false;
    }
    source += compression.offset;

    size_t const byteSize = compression.count * compression.stride;
    void* destination = std::malloc(byteSize);
    if (destination == nullptr) {
      MOCHI_LOG_WARNING("ReadGlb: meshopt decompression could not allocate %zu bytes.", byteSize);
      return false;
    }

    int error = -1;
    switch (compression.mode) {
      case cgltf_meshopt_compression_mode_attributes:
        error = meshopt_decodeVertexBuffer(
            destination, compression.count, compression.stride, source, compression.size);
        break;
      case cgltf_meshopt_compression_mode_triangles:
        error = meshopt_decodeIndexBuffer(
            destination, compression.count, compression.stride, source, compression.size);
        break;
      case cgltf_meshopt_compression_mode_indices:
        error = meshopt_decodeIndexSequence(
            destination, compression.count, compression.stride, source, compression.size);
        break;
      case cgltf_meshopt_compression_mode_invalid:
      case cgltf_meshopt_compression_mode_max_enum:
        break;
    }
    if (error != 0) {
      MOCHI_LOG_WARNING("ReadGlb: meshopt decompression failed with error %d.", error);
      std::free(destination);
      return false;
    }

    switch (compression.filter) {
      case cgltf_meshopt_compression_filter_octahedral:
        meshopt_decodeFilterOct(destination, compression.count, compression.stride);
        break;
      case cgltf_meshopt_compression_filter_quaternion:
        meshopt_decodeFilterQuat(destination, compression.count, compression.stride);
        break;
      case cgltf_meshopt_compression_filter_exponential:
        meshopt_decodeFilterExp(destination, compression.count, compression.stride);
        break;
      case cgltf_meshopt_compression_filter_none:
      case cgltf_meshopt_compression_filter_max_enum:
        break;
    }

    // cgltf_free releases buffer_view.data, so ownership transfers here.
    view.data = destination;
  }
  return true;
}

// The MIME type of PNG or JPEG bytes, from their signature, or null for anything else.
static char const* DetectImageMimeType(uint8_t const* data, size_t size) {
  static constexpr uint8_t kPng[] = {0x89, 'P', 'N', 'G'};
  static constexpr uint8_t kJpeg[] = {0xFF, 0xD8, 0xFF};
  if (size >= sizeof(kPng) && std::memcmp(data, kPng, sizeof(kPng)) == 0) {
    return "image/png";
  }
  if (size >= sizeof(kJpeg) && std::memcmp(data, kJpeg, sizeof(kJpeg)) == 0) {
    return "image/jpeg";
  }
  return nullptr;
}

// Images already extracted from one glTF file, so primitives sharing an image share one
// TextureImage.
using ImageCache = std::unordered_map<cgltf_image const*, std::shared_ptr<TextureImage const>>;

// The PNG or JPEG behind a glTF image: embedded in a bufferView, or an external file resolved
// relative to `gltfPath` (data URIs are not supported). Null if unavailable.
static std::shared_ptr<TextureImage const>
ExtractImage(cgltf_image const* image, char const* gltfPath, ImageCache& cache) {
  if (image == nullptr) {
    return nullptr;
  }
  auto [it, inserted] = cache.try_emplace(image);
  if (!inserted) {
    return it->second;
  }
  if (image->buffer_view != nullptr) {
    cgltf_buffer_view const* view = image->buffer_view;
    // cgltf leaves checking that a view lies inside its buffer to cgltf_validate.
    // Compared without adding the two sizes, which a corrupt file could make overflow.
    bool const inBounds = view->buffer != nullptr && view->offset <= view->buffer->size &&
        view->size <= view->buffer->size - view->offset;
    auto const* data =
        inBounds ? static_cast<uint8_t const*>(cgltf_buffer_view_data(view)) : nullptr;
    if (char const* mimeType = data ? DetectImageMimeType(data, view->size) : nullptr) {
      auto texture = std::make_shared<TextureImage>();
      texture->bytes.assign(data, data + view->size);
      texture->mimeType = mimeType;
      it->second = std::move(texture);
    }
  } else if (
      image->uri != nullptr && std::strncmp(image->uri, "data:", 5) != 0 && gltfPath != nullptr) {
    // glTF URIs are percent-encoded (e.g. a space is %20).
    std::string uri = image->uri;
    uri.resize(cgltf_decode_uri(uri.data()));
    std::filesystem::path const file = std::filesystem::path(gltfPath).parent_path() / uri;
    it->second = ReadTextureImageFromFile(file.string().c_str());
  }
  if (it->second == nullptr) {
    MOCHI_LOG_WARNING(
        "ReadGlb: could not load base color image '%s'; using the base color factor instead.",
        image->uri != nullptr && std::strncmp(image->uri, "data:", 5) != 0 ? image->uri
                                                                           : "(embedded)");
  }
  return it->second;
}

// Builds a MeshSection from a single triangle primitive, or nullopt if the
// primitive is not a usable triangle mesh (non-triangle, missing POSITION,
// empty geometry, or indices that are not whole triangles of existing vertices).
// Material properties are read only when `images` is non-null. Assumes
// cgltf_load_buffers has already populated buffer data.
static std::optional<MeshSection>
ExtractPrimitive(cgltf_primitive const& prim, char const* gltfPath, ImageCache* images) {
  static_assert(sizeof(int) == 4, "cgltf index unpack requires 32-bit int");

  if (prim.type != cgltf_primitive_type_triangles) {
    return std::nullopt;
  }

  cgltf_accessor const* posAccessor = nullptr;
  cgltf_accessor const* normAccessor = nullptr;
  cgltf_accessor const* uvAccessor = nullptr;
  for (cgltf_size a = 0; a < prim.attributes_count; ++a) {
    cgltf_attribute const& attr = prim.attributes[a];
    if (attr.type == cgltf_attribute_type_position) {
      posAccessor = attr.data;
    } else if (attr.type == cgltf_attribute_type_normal) {
      normAccessor = attr.data;
    } else if (attr.type == cgltf_attribute_type_texcoord && attr.index == 0) {
      uvAccessor = attr.data;
    }
  }
  if (posAccessor == nullptr) {
    return std::nullopt; // POSITION is required.
  }

  MeshSection section;

  cgltf_size const posFloats = cgltf_accessor_unpack_floats(posAccessor, nullptr, 0);
  section.positions.resize(posFloats);
  cgltf_accessor_unpack_floats(posAccessor, section.positions.data(), posFloats);

  // Normals are optional; accept only if they match the position count.
  if (normAccessor != nullptr) {
    cgltf_size const normFloats = cgltf_accessor_unpack_floats(normAccessor, nullptr, 0);
    if (normFloats == posFloats) {
      section.normals.resize(normFloats);
      cgltf_accessor_unpack_floats(normAccessor, section.normals.data(), normFloats);
      section.hasNormals = true;
    }
  }

  cgltf_size const vertexCount = posFloats / 3;
  // Texture coordinates are optional; accept only if there is one pair per vertex.
  if (uvAccessor != nullptr) {
    cgltf_size const uvFloats = cgltf_accessor_unpack_floats(uvAccessor, nullptr, 0);
    if (uvFloats == 2 * vertexCount) {
      section.texcoords.resize(uvFloats);
      cgltf_accessor_unpack_floats(uvAccessor, section.texcoords.data(), uvFloats);
      section.hasTexcoords = true;
    }
  }
  if (prim.indices != nullptr) {
    cgltf_size const indexCount =
        cgltf_accessor_unpack_indices(prim.indices, nullptr, sizeof(int), 0);
    section.indices.resize(indexCount);
    cgltf_accessor_unpack_indices(prim.indices, section.indices.data(), sizeof(int), indexCount);
  } else {
    // Non-indexed primitive: vertices are consumed sequentially.
    section.indices.resize(vertexCount);
    for (cgltf_size i = 0; i < vertexCount; ++i) {
      section.indices[i] = static_cast<int>(i);
    }
  }

  if (prim.material != nullptr && images != nullptr) {
    if (prim.material->has_pbr_metallic_roughness) {
      cgltf_pbr_metallic_roughness const& pbr = prim.material->pbr_metallic_roughness;
      section.baseColor = {
          pbr.base_color_factor[0],
          pbr.base_color_factor[1],
          pbr.base_color_factor[2],
          pbr.base_color_factor[3]};
      section.metallic = pbr.metallic_factor;
      section.roughness = pbr.roughness_factor;
      if (pbr.base_color_texture.texture != nullptr) {
        section.baseColorTexture =
            ExtractImage(pbr.base_color_texture.texture->image, gltfPath, *images);
      }
    }
    // emissiveFactor sits on the material itself, not under pbrMetallicRoughness, so it is read
    // even for materials with no metallic-roughness model.
    section.emissive = {
        prim.material->emissive_factor[0],
        prim.material->emissive_factor[1],
        prim.material->emissive_factor[2]};
    if (prim.material->has_emissive_strength) {
      section.emissiveStrength = prim.material->emissive_strength.emissive_strength;
    }
  }

  if (section.positions.empty() || section.indices.empty() || section.indices.size() % 3 != 0 ||
      std::any_of(section.indices.begin(), section.indices.end(), [&](int index) {
        return index < 0 || static_cast<cgltf_size>(index) >= vertexCount;
      })) {
    MOCHI_LOG_WARNING("ReadGlb: triangle primitive has invalid geometry.");
    return std::nullopt;
  }
  return section;
}

// Transforms a section's geometry by the column-major 4x4 `world` matrix:
// positions by the full affine transform, normals by the upper-left 3x3 with
// renormalization. This matches glTF node-to-world semantics (exact for rigid
// and uniform-scale nodes; normals are approximate under non-uniform scale).
static void ApplyWorldTransform(MeshSection& section, float const world[16]) {
  for (size_t i = 0; i + 2 < section.positions.size(); i += 3) {
    float const x = section.positions[i];
    float const y = section.positions[i + 1];
    float const z = section.positions[i + 2];
    section.positions[i] = world[0] * x + world[4] * y + world[8] * z + world[12];
    section.positions[i + 1] = world[1] * x + world[5] * y + world[9] * z + world[13];
    section.positions[i + 2] = world[2] * x + world[6] * y + world[10] * z + world[14];
  }
  for (size_t i = 0; i + 2 < section.normals.size(); i += 3) {
    float const x = section.normals[i];
    float const y = section.normals[i + 1];
    float const z = section.normals[i + 2];
    float nx = world[0] * x + world[4] * y + world[8] * z;
    float ny = world[1] * x + world[5] * y + world[9] * z;
    float nz = world[2] * x + world[6] * y + world[10] * z;
    float const len = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (len > 0.0f) {
      float const invLen = 1.0f / len;
      nx *= invLen;
      ny *= invLen;
      nz *= invLen;
    }
    section.normals[i] = nx;
    section.normals[i + 1] = ny;
    section.normals[i + 2] = nz;
  }
}

// Builds one MeshSection per triangle primitive, returning geometry in world
// space. Each mesh is emitted once per referencing scene node with that node's
// world transform applied, so node placement/rotation/scale is preserved. GLBs
// with no node graph fall back to emitting each mesh once at identity.
static std::vector<MeshSection> ExtractSections(cgltf_data const& data, char const* gltfPath) {
  std::vector<MeshSection> sections;
  ImageCache images;

  bool anyNodeMesh = false;
  for (cgltf_size n = 0; n < data.nodes_count; ++n) {
    cgltf_node const& node = data.nodes[n];
    if (node.mesh == nullptr) {
      continue;
    }
    anyNodeMesh = true;
    float world[16];
    cgltf_node_transform_world(&node, world);
    for (cgltf_size p = 0; p < node.mesh->primitives_count; ++p) {
      std::optional<MeshSection> section =
          ExtractPrimitive(node.mesh->primitives[p], gltfPath, &images);
      if (section.has_value()) {
        ApplyWorldTransform(*section, world);
        sections.push_back(std::move(*section));
      }
    }
  }

  // Fallback for GLBs without a scene-node graph: emit each mesh at identity.
  if (!anyNodeMesh) {
    for (cgltf_size m = 0; m < data.meshes_count; ++m) {
      cgltf_mesh const& mesh = data.meshes[m];
      for (cgltf_size p = 0; p < mesh.primitives_count; ++p) {
        std::optional<MeshSection> section =
            ExtractPrimitive(mesh.primitives[p], gltfPath, &images);
        if (section.has_value()) {
          sections.push_back(std::move(*section));
        }
      }
    }
  }
  return sections;
}

// COLOR_0 of @p prim as RGBA, or empty if the primitive lacks it or it holds a different vertex
// count. An RGB COLOR_0 gets an opaque alpha.
static std::vector<float> UnpackColors(cgltf_primitive const& prim, size_t vertexCount) {
  for (cgltf_size a = 0; a < prim.attributes_count; ++a) {
    cgltf_attribute const& attr = prim.attributes[a];
    if (attr.type != cgltf_attribute_type_color || attr.index != 0 || attr.data == nullptr ||
        attr.data->count != vertexCount) {
      continue;
    }
    size_t const stored = cgltf_num_components(attr.data->type);
    std::vector<float> values(vertexCount * stored);
    cgltf_accessor_unpack_floats(attr.data, values.data(), values.size());
    if (stored == 4) {
      return values;
    }
    if (stored == 3) {
      std::vector<float> padded(vertexCount * 4, 1.0f);
      for (size_t v = 0; v < vertexCount; ++v) {
        std::copy_n(&values[3 * v], 3, &padded[4 * v]);
      }
      return padded;
    }
  }
  return {};
}

static bool IsAncestorOrSelf(cgltf_node const* ancestor, cgltf_node const* node) {
  for (; node != nullptr; node = node->parent) {
    if (node == ancestor) {
      return true;
    }
  }
  return false;
}

static DeformableSourceData ExtractDeformableSource(cgltf_data const& data, char const* /*path*/) {
  DeformableSourceData result;
  cgltf_node const* meshNode = nullptr;
  for (cgltf_size i = 0; i < data.nodes_count; ++i) {
    if (data.nodes[i].mesh == nullptr) {
      continue;
    }
    if (meshNode != nullptr) {
      result.error = "The GLB has more than one node with a mesh.";
      return result;
    }
    meshNode = &data.nodes[i];
  }
  if (meshNode == nullptr) {
    result.error = "The GLB has no node with a mesh.";
    return result;
  }
  if (meshNode->mesh->primitives_count != 1) {
    result.error = "The GLB's mesh does not have exactly one primitive.";
    return result;
  }
  cgltf_primitive const& primitive = meshNode->mesh->primitives[0];
  if (primitive.type != cgltf_primitive_type_triangles) {
    result.error = "The GLB's primitive is not a triangle list.";
    return result;
  }
  if (meshNode->skin != nullptr) {
    result.error = "The GLB's mesh is skinned.";
    return result;
  }
  if (primitive.targets_count != 0) {
    result.error = "The GLB's primitive has morph targets.";
    return result;
  }
  if (meshNode->has_mesh_gpu_instancing) {
    result.error = "The GLB's mesh uses GPU instancing.";
    return result;
  }
  if (data.variants_count != 0 || primitive.mappings_count != 0) {
    result.error = "The GLB has material variants.";
    return result;
  }
  for (cgltf_size a = 0; a < primitive.attributes_count; ++a) {
    cgltf_attribute const& attribute = primitive.attributes[a];
    if (attribute.type == cgltf_attribute_type_texcoord && attribute.index != 0) {
      result.error = "The GLB's primitive has a texture-coordinate set other than TEXCOORD_0.";
      return result;
    }
  }
  for (cgltf_size a = 0; a < data.animations_count; ++a) {
    cgltf_animation const& animation = data.animations[a];
    for (cgltf_size c = 0; c < animation.channels_count; ++c) {
      if (IsAncestorOrSelf(animation.channels[c].target_node, meshNode)) {
        result.error = "The GLB animates its mesh node or one of its ancestors.";
        return result;
      }
    }
  }
  for (cgltf_size b = 0; b < data.buffers_count; ++b) {
    char const* const uri = data.buffers[b].uri;
    if (uri != nullptr && std::strncmp(uri, "data:", 5) != 0) {
      result.error = "The GLB references an external buffer file.";
      return result;
    }
  }

  std::optional<MeshSection> section = ExtractPrimitive(primitive, nullptr, nullptr);
  if (!section.has_value()) {
    result.error = "The GLB's primitive geometry is invalid.";
    return result;
  }
  size_t const vertexCount = section->positions.size() / 3;
  result.positions = std::move(section->positions);
  result.normals = std::move(section->normals);
  result.texcoords = std::move(section->texcoords);
  result.indices = std::move(section->indices);
  result.colors = UnpackColors(primitive, vertexCount);
  cgltf_node_transform_world(meshNode, result.rootFromNode.data());
  return result;
}

// Parses `data`/`size` as a GLB blob and passes the parsed asset and `path` to
// `extract`. `path` (may be null) resolves external .bin buffers (through
// cgltf_load_buffers) and images relative to the source file; pass null for
// self-contained embedded-GLB bytes. Returns an empty result on failure.
template <typename Extract>
static auto ReadGlbImpl(uint8_t const* data, size_t size, char const* path, Extract&& extract)
    -> decltype(extract(std::declval<cgltf_data const&>(), path)) {
  if (data == nullptr || size == 0) {
    return {};
  }
  cgltf_options options{};
  cgltf_data* parsed = nullptr;
  if (cgltf_parse(&options, data, size, &parsed) != cgltf_result_success || parsed == nullptr) {
    MOCHI_LOG_WARNING("ReadGlb: failed to parse GLB data.");
    return {};
  }
  MOCHI_DEFER(cgltf_free(parsed));
  if (cgltf_load_buffers(&options, parsed, path) != cgltf_result_success) {
    MOCHI_LOG_WARNING("ReadGlb: failed to load GLB buffers.");
    return {};
  }
  if (cgltf_validate(parsed) != cgltf_result_success) {
    MOCHI_LOG_WARNING("ReadGlb: GLB validation failed.");
    return {};
  }
  // Must outlive `extract`, which reads through the accessors these streams
  // back; it copies everything out, so scoping them here is enough.
  DecodedStreams dracoStreams;
  if (!DecodeMeshoptBufferViews(*parsed) || !DecodeDracoMeshes(*parsed, dracoStreams)) {
    return {};
  }
  return extract(*parsed, path);
}

// Reads the whole file at @p path, or returns an empty buffer (with a warning naming @p reader).
static std::vector<uint8_t> ReadFileBytes(char const* path, char const* reader) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    MOCHI_LOG_WARNING("%s: failed to open '%s'.", reader, path);
    return {};
  }
  std::streamsize const size = in.tellg();
  if (size <= 0) {
    return {};
  }
  in.seekg(0);
  std::vector<uint8_t> buffer(static_cast<size_t>(size));
  if (!in.read(reinterpret_cast<char*>(buffer.data()), size)) {
    MOCHI_LOG_WARNING("%s: failed to read '%s'.", reader, path);
    return {};
  }
  return buffer;
}

std::vector<MeshSection> ReadGlbFromFile(char const* path) {
  if (path == nullptr) {
    return {};
  }
  std::vector<uint8_t> const buffer = ReadFileBytes(path, "ReadGlbFromFile");
  return ReadGlbImpl(buffer.data(), buffer.size(), path, ExtractSections);
}

size_t HashGlbBytes(mochi::Span<uint8_t const> bytes) {
  return std::hash<std::string_view>{}(
      std::string_view(reinterpret_cast<char const*>(bytes.data()), bytes.size()));
}

DeformableSourceData ReadDeformableGlb(char const* path, size_t expectedHash) {
  std::vector<uint8_t> const glb = ReadFileBytes(path, "ReadDeformableGlb");
  if (glb.empty()) {
    return {.error = "The GLB file could not be read."};
  }
  if (HashGlbBytes(mochi::MakeConstSpan(glb)) != expectedHash) {
    return {
        .error =
            "The GLB file has changed since the render model was loaded; load it again to "
            "deform it."};
  }
  constexpr uint32_t kGlbMagic = 0x46546C67; // "glTF"
  uint32_t magic = 0;
  if (glb.size() >= sizeof(magic)) {
    std::memcpy(&magic, glb.data(), sizeof(magic));
  }
  if (magic != kGlbMagic) {
    return {.error = "Only binary GLB files can be deformed."};
  }
  DeformableSourceData result = ReadGlbImpl(glb.data(), glb.size(), path, ExtractDeformableSource);
  if (result.error == nullptr && !result.IsSupported()) {
    result.error = "The GLB's data could not be read; the log says why.";
  }
  return result;
}

std::vector<MeshSection> ReadColladaFromFile(char const* path) {
  if (path == nullptr) {
    return {};
  }
  tinyxml2::XMLDocument doc;
  if (doc.LoadFile(path) != tinyxml2::XML_SUCCESS) {
    MOCHI_LOG_WARNING("ReadColladaFromFile: failed to read '%s'.", path);
    return {};
  }
  return collada_reader::ReadSections(doc);
}

std::vector<MeshSection> ReadObjFromFile(char const* path) {
  if (path == nullptr) {
    return {};
  }

  tinyobj::ObjReaderConfig config;
  config.triangulate = true; // fan-triangulate polygon faces into triangles
  config.vertex_color = false; // MeshSection has no vertex-color channel
  // mtl_search_path defaults to "" → resolve `mtllib` next to the .obj file.

  tinyobj::ObjReader reader;
  if (!reader.ParseFromFile(path, config)) {
    MOCHI_LOG_WARNING("ReadObjFromFile: failed to read '%s': %s", path, reader.Error().c_str());
    return {};
  }

  // An unresolvable `mtllib` is only a tinyobj warning: parsing succeeds, no materials come back,
  // and every face silently falls back to the MeshSection default color -- so a renamed or missing
  // .mtl looks like "materials are unsupported" rather than a broken asset. Say so. Keyed on
  // tinyobj's "Material file [ x ] not found" text so an .obj that names no material library at
  // all (which warns about nothing) stays quiet, as does a `usemtl` naming an absent material.
  if (reader.Warning().find("Material file [") != std::string::npos) {
    MOCHI_LOG_WARNING(
        "ReadObjFromFile: '%s' names a material library that could not be loaded; its geometry "
        "will use default colors. %s",
        path,
        reader.Warning().c_str());
  }

  tinyobj::attrib_t const& attrib = reader.GetAttrib();
  std::vector<tinyobj::material_t> const& materials = reader.GetMaterials();
  bool const hasNormals = !attrib.normals.empty();
  bool const hasTexcoords = !attrib.texcoords.empty();

  // Diffuse maps, by path, so materials sharing an image share one TextureImage.
  std::unordered_map<std::string, std::shared_ptr<TextureImage const>> textures;
  auto loadTexture = [&](std::string const& name) -> std::shared_ptr<TextureImage const> {
    std::string const file = (std::filesystem::path(path).parent_path() / name).string();
    auto [it, inserted] = textures.try_emplace(file);
    if (inserted) {
      it->second = ReadTextureImageFromFile(file.c_str());
      if (!it->second) {
        MOCHI_LOG_WARNING(
            "ReadObjFromFile: '%s' uses diffuse map '%s', which is missing or not a PNG/JPEG.",
            path,
            file.c_str());
      }
    }
    return it->second;
  };

  // One section per material in first-seen order across all shapes; faces with
  // no material (id -1) accumulate into a single default section.
  std::vector<MeshSection> sections;
  std::unordered_map<int, size_t> sectionByMaterial;
  // Parallel to `sections`: false once any corner in that section is missing a
  // valid normal index. Such sections cannot be trusted to have per-vertex
  // normals, so their (partial) normals are dropped below to force the caller to
  // recompute them. Texture coordinates are tracked the same way.
  std::vector<bool> sectionNormalsComplete;
  std::vector<bool> sectionTexcoordsComplete;

  auto sectionFor = [&](int materialId) -> size_t {
    auto const [it, inserted] = sectionByMaterial.try_emplace(materialId, sections.size());
    if (inserted) {
      MeshSection section;
      section.hasNormals = hasNormals;
      section.hasTexcoords = hasTexcoords;
      if (materialId >= 0 && materialId < static_cast<int>(materials.size())) {
        tinyobj::material_t const& mat = materials[materialId];
        section.baseColor = {
            static_cast<float>(mat.diffuse[0]),
            static_cast<float>(mat.diffuse[1]),
            static_cast<float>(mat.diffuse[2]),
            static_cast<float>(mat.dissolve)};
        auto const ns = static_cast<float>(mat.shininess);
        section.roughness = std::clamp(std::sqrt(2.0f / (ns + 2.0f)), 0.0f, 1.0f);
        if (!mat.diffuse_texname.empty()) {
          section.baseColorTexture = loadTexture(mat.diffuse_texname);
        }
      }
      sections.push_back(std::move(section));
      sectionNormalsComplete.push_back(true);
      sectionTexcoordsComplete.push_back(true);
    }
    return it->second;
  };

  // Emit vertices flat per triangle corner (no deduplication); per-section
  // indices are local and sequential, as required by BuildGlbFromMeshSections.
  for (tinyobj::shape_t const& shape : reader.GetShapes()) {
    tinyobj::mesh_t const& mesh = shape.mesh;
    size_t indexOffset = 0;
    for (size_t f = 0; f < mesh.num_face_vertices.size(); ++f) {
      int const faceVertices = mesh.num_face_vertices[f];
      int const materialId = f < mesh.material_ids.size() ? mesh.material_ids[f] : -1;
      size_t const sectionIdx = sectionFor(materialId);
      MeshSection& section = sections[sectionIdx];

      for (int v = 0; v < faceVertices; ++v) {
        tinyobj::index_t const& idx = mesh.indices[indexOffset + v];

        int const posIdx = idx.vertex_index * 3;
        if (idx.vertex_index >= 0 && posIdx + 2 < static_cast<int>(attrib.vertices.size())) {
          section.positions.push_back(static_cast<float>(attrib.vertices[posIdx]));
          section.positions.push_back(static_cast<float>(attrib.vertices[posIdx + 1]));
          section.positions.push_back(static_cast<float>(attrib.vertices[posIdx + 2]));
        } else {
          section.positions.insert(section.positions.end(), {0.0f, 0.0f, 0.0f});
        }

        if (hasNormals) {
          int const normIdx = idx.normal_index * 3;
          if (idx.normal_index >= 0 && normIdx + 2 < static_cast<int>(attrib.normals.size())) {
            section.normals.push_back(static_cast<float>(attrib.normals[normIdx]));
            section.normals.push_back(static_cast<float>(attrib.normals[normIdx + 1]));
            section.normals.push_back(static_cast<float>(attrib.normals[normIdx + 2]));
          } else {
            // This corner has no normal: keep arrays aligned for now, but mark
            // the section so its normals are dropped and recomputed downstream.
            section.normals.insert(section.normals.end(), {0.0f, 0.0f, 0.0f});
            sectionNormalsComplete[sectionIdx] = false;
          }
        }

        if (hasTexcoords) {
          // Range-checked before scaling, which a corrupt index could overflow.
          if (idx.texcoord_index >= 0 &&
              static_cast<size_t>(idx.texcoord_index) < attrib.texcoords.size() / 2) {
            size_t const uvIdx = 2 * static_cast<size_t>(idx.texcoord_index);
            // OBJ puts v = 0 at the bottom of the image; glTF puts it at the top.
            section.texcoords.push_back(static_cast<float>(attrib.texcoords[uvIdx]));
            section.texcoords.push_back(1.0f - static_cast<float>(attrib.texcoords[uvIdx + 1]));
          } else {
            section.texcoords.insert(section.texcoords.end(), {0.0f, 0.0f});
            sectionTexcoordsComplete[sectionIdx] = false;
          }
        }

        section.indices.push_back(static_cast<int>(section.indices.size()));
      }
      indexOffset += static_cast<size_t>(faceVertices);
    }
  }

  // Drop partially-specified normals so the caller recomputes them instead of
  // shading with the placeholder zero normals inserted above.
  for (size_t i = 0; i < sections.size(); ++i) {
    if (!sectionNormalsComplete[i]) {
      sections[i].normals.clear();
      sections[i].hasNormals = false;
    }
    if (!sectionTexcoordsComplete[i]) {
      sections[i].texcoords.clear();
      sections[i].hasTexcoords = false;
    }
  }

  std::erase_if(sections, [](MeshSection const& section) {
    return section.positions.empty() || section.indices.empty();
  });

  return sections;
}

std::shared_ptr<TextureImage const> ReadTextureImageFromFile(char const* path) {
  if (path == nullptr) {
    return nullptr;
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return nullptr;
  }
  std::vector<uint8_t> bytes(
      (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  char const* mimeType = DetectImageMimeType(bytes.data(), bytes.size());
  if (mimeType == nullptr) {
    return nullptr;
  }
  auto texture = std::make_shared<TextureImage>();
  texture->bytes = std::move(bytes);
  texture->mimeType = mimeType;
  return texture;
}

std::vector<MeshSection> ReadStlFromFile(char const* path) {
  if (path == nullptr) {
    return {};
  }
  std::vector<float> positions;
  std::vector<float> faceNormals;
  std::vector<int> indices;
  std::vector<int> solids;
  try {
    stl_reader::ReadStlFile(path, positions, faceNormals, indices, solids);
  } catch (std::exception const& e) {
    MOCHI_LOG_WARNING("ReadStlFromFile: failed to read '%s': %s", path, e.what());
    return {};
  }
  // `solids` holds [begin, end) ranges, so a single solid yields size 2.
  if (solids.size() > 2) {
    MOCHI_LOG_WARNING("ReadStlFromFile: '%s' has multiple solids; unsupported.", path);
    return {};
  }
  if (positions.empty() || indices.empty()) {
    return {};
  }
  MeshSection section;
  section.positions = std::move(positions);
  section.indices = std::move(indices);
  // STL provides only per-face normals; leave hasNormals false so the caller
  // computes per-vertex normals (matching the other readers' contract).
  return {std::move(section)};
}

void ConvertMeshSectionsSpace(
    std::vector<MeshSection>& sections,
    mochi::CoordinateSpaceConverter const& converter) {
  for (MeshSection& section : sections) {
    converter.TranslationsToOutput(mochi::MakeSpan(section.positions), mochi::ErrorAssert{});
    converter.DirectionsToOutput(mochi::MakeSpan(section.normals), mochi::ErrorAssert{});
  }
}

void ComputeFaceNormals(
    std::vector<float> const& positions,
    std::vector<int> const& indices,
    std::vector<float>& faceNormalsOut) {
  size_t const numTris = indices.size() / 3;
  faceNormalsOut.clear();
  faceNormalsOut.reserve(numTris * 3);
  for (size_t t = 0; t < numTris; ++t) {
    int const i0 = indices[t * 3 + 0];
    int const i1 = indices[t * 3 + 1];
    int const i2 = indices[t * 3 + 2];
    float const* p0 = &positions[i0 * 3];
    float const* p1 = &positions[i1 * 3];
    float const* p2 = &positions[i2 * 3];
    float const e1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
    float const e2[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
    float n[3] = {
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0]};
    float const len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (len > 0.0f) {
      n[0] /= len;
      n[1] /= len;
      n[2] /= len;
    }
    faceNormalsOut.push_back(n[0]);
    faceNormalsOut.push_back(n[1]);
    faceNormalsOut.push_back(n[2]);
  }
}

void ComputeVertexNormalsAngleWeighted(
    std::vector<float> const& positions,
    std::vector<float> const& faceNormals,
    std::vector<int> const& indices,
    std::vector<float>& vertexNormalsOut) {
  size_t const numVertices = positions.size() / 3;
  size_t const numTris = indices.size() / 3;
  vertexNormalsOut.assign(numVertices * 3, 0.0f);
  for (size_t t = 0; t < numTris; ++t) {
    int const i0 = indices[t * 3 + 0];
    int const i1 = indices[t * 3 + 1];
    int const i2 = indices[t * 3 + 2];
    float const* p0 = &positions[i0 * 3];
    float const* p1 = &positions[i1 * 3];
    float const* p2 = &positions[i2 * 3];
    float const* fn = &faceNormals[t * 3];
    float const e01[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
    float const e02[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
    float const e12[3] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
    auto length = [](float const* v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); };
    auto dot = [](float const* a, float const* b) {
      return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };
    float const len01 = length(e01), len02 = length(e02), len12 = length(e12);
    if (len01 < 1e-12f || len02 < 1e-12f || len12 < 1e-12f) {
      continue;
    }
    float const angle0 = std::acos(std::clamp(dot(e01, e02) / (len01 * len02), -1.0f, 1.0f));
    float const e10[3] = {-e01[0], -e01[1], -e01[2]};
    float const angle1 = std::acos(std::clamp(dot(e10, e12) / (len01 * len12), -1.0f, 1.0f));
    float const angle2 = std::numbers::pi_v<float> - angle0 - angle1;
    int const idx[3] = {i0, i1, i2};
    float const angles[3] = {angle0, angle1, angle2};
    for (int c = 0; c < 3; ++c) {
      int const vi = idx[c];
      float const w = angles[c];
      vertexNormalsOut[vi * 3 + 0] += fn[0] * w;
      vertexNormalsOut[vi * 3 + 1] += fn[1] * w;
      vertexNormalsOut[vi * 3 + 2] += fn[2] * w;
    }
  }
  for (size_t v = 0; v < numVertices; ++v) {
    float* vn = &vertexNormalsOut[v * 3];
    float const len = std::sqrt(vn[0] * vn[0] + vn[1] * vn[1] + vn[2] * vn[2]);
    if (len > 1e-12f) {
      vn[0] /= len;
      vn[1] /= len;
      vn[2] /= len;
    }
  }
}

void ComputeVertexNormalsAreaWeighted(
    std::vector<float> const& positions,
    std::vector<int> const& indices,
    std::vector<float>& vertexNormalsOut) {
  size_t const numVertices = positions.size() / 3;
  size_t const numTris = indices.size() / 3;
  vertexNormalsOut.assign(numVertices * 3, 0.0f);
  for (size_t t = 0; t < numTris; ++t) {
    int const i0 = indices[t * 3 + 0];
    int const i1 = indices[t * 3 + 1];
    int const i2 = indices[t * 3 + 2];
    float const* p0 = &positions[i0 * 3];
    float const* p1 = &positions[i1 * 3];
    float const* p2 = &positions[i2 * 3];
    float const e1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
    float const e2[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
    // Unnormalized cross product; its magnitude is twice the triangle area, so
    // accumulating it weights each face normal by area.
    float const n[3] = {
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0]};
    int const idx[3] = {i0, i1, i2};
    for (int const vi : idx) {
      vertexNormalsOut[vi * 3 + 0] += n[0];
      vertexNormalsOut[vi * 3 + 1] += n[1];
      vertexNormalsOut[vi * 3 + 2] += n[2];
    }
  }
  for (size_t v = 0; v < numVertices; ++v) {
    float* vn = &vertexNormalsOut[v * 3];
    float const len = std::sqrt(vn[0] * vn[0] + vn[1] * vn[1] + vn[2] * vn[2]);
    if (len > 1e-12f) {
      vn[0] /= len;
      vn[1] /= len;
      vn[2] /= len;
    }
  }
}

bool WriteGlbToFile(char const* path, std::vector<MeshSection> const& sections) {
  std::vector<uint8_t> const glb = BuildGlbFromMeshSections(sections);
  std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!out) {
    return false;
  }
  out.write(reinterpret_cast<char const*>(glb.data()), static_cast<std::streamsize>(glb.size()));
  return out.good();
}

bool WriteObjToFile(char const* path, std::vector<MeshSection> const& sections) {
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) {
    return false;
  }
  out << "# Exported by SuperDex\n";
  int base = 1; // OBJ vertex indices are 1-based, offset across concatenated sections
  for (auto const& section : sections) {
    for (std::size_t i = 0; i + 3 <= section.positions.size(); i += 3) {
      out << "v " << section.positions[i] << ' ' << section.positions[i + 1] << ' '
          << section.positions[i + 2] << '\n';
    }
    for (std::size_t t = 0; t + 3 <= section.indices.size(); t += 3) {
      out << "f " << (base + section.indices[t]) << ' ' << (base + section.indices[t + 1]) << ' '
          << (base + section.indices[t + 2]) << '\n';
    }
    base += static_cast<int>(section.positions.size() / 3);
  }
  return out.good();
}

} // namespace mochi_renderer
