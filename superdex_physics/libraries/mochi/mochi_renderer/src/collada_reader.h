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

// Minimal COLLADA (.dae) reader for mochi_renderer.
// Parses <library_geometries> into flat positions/normals/indices grouped by
// material (one @ref MeshSection per primitive), resolving COLLADA's non-PBR
// materials to simple solid-color glTF PBR factors. Designed to feed
// BuildGlbFromMeshSections; no image textures, animations, or skinning.

#pragma once

#include <mochi_renderer/utils.h>

#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array.h>

#include <tinyxml2.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace collada_reader {

using mochi_renderer::MeshSection;

namespace details {

using mochi::Matrix4x4r;
using mochi::real;
using mochi::Real3;
using mochi::Real4;

// COLLADA documents declare their vertical axis; glTF is Y-up, so non-Y files
// require a fixed rotation applied to every position and normal.
enum class UpAxis { X, Y, Z };

// A COLLADA <source>: flat float payload plus the accessor stride (components
// per logical element, typically 3 for positions/normals).
struct Source {
  std::vector<float> data;
  int stride = 3;
};

// Resolved solid material for a primitive.
struct EffectInfo {
  std::array<float, 4> color = {0.6f, 0.6f, 0.6f, 1.0f};
  float shininess = 0.0f;
  bool hasShininess = false;
};

inline std::vector<float> ParseFloats(char const* text) {
  std::vector<float> out;
  if (text == nullptr) {
    return out;
  }
  char const* p = text;
  while (*p != '\0') {
    char* end = nullptr;
    float const v = std::strtof(p, &end);
    if (end == p) {
      ++p; // Skip a stray non-numeric character and continue.
      continue;
    }
    out.push_back(v);
    p = end;
  }
  return out;
}

inline std::vector<int> ParseInts(char const* text) {
  std::vector<int> out;
  if (text == nullptr) {
    return out;
  }
  char const* p = text;
  while (*p != '\0') {
    char* end = nullptr;
    long const v = std::strtol(p, &end, 10);
    if (end == p) {
      ++p;
      continue;
    }
    out.push_back(static_cast<int>(v));
    p = end;
  }
  return out;
}

// Strips a leading '#' from a COLLADA URL/id reference (e.g. "#Effect" → "Effect").
inline std::string StripHash(char const* url) {
  if (url == nullptr) {
    return {};
  }
  std::string s(url);
  if (!s.empty() && s.front() == '#') {
    s.erase(0, 1);
  }
  return s;
}

// Rotates a vector (position or normal) from the file's up-axis into glTF Y-up.
// The transforms are pure rotations, so they apply identically to normals.
inline void ApplyUpAxis(UpAxis up, float& x, float& y, float& z) {
  switch (up) {
    case UpAxis::Y:
      return;
    case UpAxis::Z: {
      // -90° about X: (x, y, z) → (x, z, -y).
      float const ny = z;
      float const nz = -y;
      y = ny;
      z = nz;
      return;
    }
    case UpAxis::X: {
      // +90° about Z: (x, y, z) → (-y, x, z).
      float const nx = -y;
      float const ny = x;
      x = nx;
      y = ny;
      return;
    }
  }
}

// Applies the full affine transform to a position (homogeneous w = 1).
inline Real3 TransformPoint(Matrix4x4r const& m, Real3 const& p) {
  Real4 const r = mochi::DotMatVec(m, Real4{p[0], p[1], p[2], real(1)});
  return {r[0], r[1], r[2]};
}

// Applies the linear (3x3) part to a normal (homogeneous w = 0) and
// renormalizes. Correct for rotation and uniform scale (the common COLLADA node
// transform); for non-uniform scale this is an approximation.
inline Real3 TransformNormal(Matrix4x4r const& m, Real3 const& n) {
  Real4 const r = mochi::DotMatVec(m, Real4{n[0], n[1], n[2], real(0)});
  real const len = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
  if (len > real(0)) {
    return {r[0] / len, r[1] / len, r[2] / len};
  }
  return {r[0], r[1], r[2]};
}

// Parses a single node transform element (<matrix>/<translate>/<rotate>/<scale>)
// into a 4x4 affine matrix. Returns identity for unrecognized or malformed
// elements.
inline Matrix4x4r ParseTransformElement(tinyxml2::XMLElement* t) {
  char const* name = t->Name();
  std::vector<float> const v = ParseFloats(t->GetText());
  if (std::strcmp(name, "matrix") == 0) {
    // COLLADA stores the 4x4 in row-major order, matching Matrix4x4r.
    if (v.size() >= 16) {
      return Matrix4x4r{
          Real4{v[0], v[1], v[2], v[3]},
          Real4{v[4], v[5], v[6], v[7]},
          Real4{v[8], v[9], v[10], v[11]},
          Real4{v[12], v[13], v[14], v[15]}};
    }
  } else if (std::strcmp(name, "translate") == 0) {
    if (v.size() >= 3) {
      Matrix4x4r m = mochi::Eye<4>();
      m[0][3] = v[0];
      m[1][3] = v[1];
      m[2][3] = v[2];
      return m;
    }
  } else if (std::strcmp(name, "scale") == 0) {
    if (v.size() >= 3) {
      Matrix4x4r m = mochi::Eye<4>();
      m[0][0] = v[0];
      m[1][1] = v[1];
      m[2][2] = v[2];
      return m;
    }
  } else if (std::strcmp(name, "rotate") == 0) {
    // axis (x, y, z) plus angle in degrees.
    if (v.size() >= 4) {
      Real3 axis{v[0], v[1], v[2]};
      real const axisLen = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
      if (axisLen > real(0)) {
        axis = {axis[0] / axisLen, axis[1] / axisLen, axis[2] / axisLen};
      }
      real const rad = real(v[3]) * mochi::kRadiansPerDegree;
      real const c = std::cos(rad);
      real const s = std::sin(rad);
      real const t1 = real(1) - c;
      real const ax = axis[0];
      real const ay = axis[1];
      real const az = axis[2];
      Matrix4x4r m = mochi::Eye<4>();
      m[0][0] = c + ax * ax * t1;
      m[0][1] = ax * ay * t1 - az * s;
      m[0][2] = ax * az * t1 + ay * s;
      m[1][0] = ay * ax * t1 + az * s;
      m[1][1] = c + ay * ay * t1;
      m[1][2] = ay * az * t1 - ax * s;
      m[2][0] = az * ax * t1 - ay * s;
      m[2][1] = az * ay * t1 + ax * s;
      m[2][2] = c + az * az * t1;
      return m;
    }
  }
  return mochi::Eye<4>();
}

// Recursively walks the node hierarchy, composing each node's transform
// elements (in document order) with the accumulated parent transform, and
// records geometry id → world transform(s) for every <instance_geometry>. A
// geometry instanced by multiple nodes accumulates one transform per instance.
inline void CollectNodeTransforms(
    tinyxml2::XMLElement* elem,
    Matrix4x4r const& parent,
    std::unordered_map<std::string, std::vector<Matrix4x4r>>& geomToTransforms) {
  for (auto* node = elem->FirstChildElement("node"); node != nullptr;
       node = node->NextSiblingElement("node")) {
    Matrix4x4r world = parent;
    for (auto* child = node->FirstChildElement(); child != nullptr;
         child = child->NextSiblingElement()) {
      char const* name = child->Name();
      if (std::strcmp(name, "matrix") == 0 || std::strcmp(name, "translate") == 0 ||
          std::strcmp(name, "rotate") == 0 || std::strcmp(name, "scale") == 0) {
        world = mochi::Dot(world, ParseTransformElement(child));
      }
    }
    for (auto* ig = node->FirstChildElement("instance_geometry"); ig != nullptr;
         ig = ig->NextSiblingElement("instance_geometry")) {
      geomToTransforms[StripHash(ig->Attribute("url"))].push_back(world);
    }
    CollectNodeTransforms(node, world, geomToTransforms);
  }
}

// Recursively collects every <instance_material symbol=.. target=..> binding
// found under `elem` into `map` (symbol → material id).
inline void CollectInstanceMaterials(
    tinyxml2::XMLElement* elem,
    std::unordered_map<std::string, std::string>& map) {
  for (auto* child = elem->FirstChildElement(); child != nullptr;
       child = child->NextSiblingElement()) {
    if (std::strcmp(child->Name(), "instance_material") == 0) {
      char const* symbol = child->Attribute("symbol");
      char const* target = child->Attribute("target");
      if (symbol != nullptr && target != nullptr) {
        map[symbol] = StripHash(target);
      }
    }
    CollectInstanceMaterials(child, map);
  }
}

} // namespace details

// Parses an already-loaded COLLADA document into one @ref MeshSection per
// primitive. Returns an empty vector when no geometry is found; tolerates
// missing optional elements and skips unknown semantics.
inline std::vector<MeshSection> ReadSections(tinyxml2::XMLDocument& doc) {
  std::vector<MeshSection> sections;

  tinyxml2::XMLElement* root = doc.RootElement();
  if (root == nullptr) {
    return sections;
  }

  // --- Asset: up axis (default Y-up) ---
  details::UpAxis up = details::UpAxis::Y;
  if (auto* asset = root->FirstChildElement("asset")) {
    if (auto* ua = asset->FirstChildElement("up_axis")) {
      char const* t = ua->GetText();
      if (t != nullptr) {
        if (std::strcmp(t, "Z_UP") == 0) {
          up = details::UpAxis::Z;
        } else if (std::strcmp(t, "X_UP") == 0) {
          up = details::UpAxis::X;
        }
      }
    }
  }

  // --- Effects: effect id → diffuse color + shininess ---
  std::unordered_map<std::string, details::EffectInfo> effects;
  if (auto* libFx = root->FirstChildElement("library_effects")) {
    for (auto* fx = libFx->FirstChildElement("effect"); fx != nullptr;
         fx = fx->NextSiblingElement("effect")) {
      char const* id = fx->Attribute("id");
      if (id == nullptr) {
        continue;
      }
      details::EffectInfo info;
      if (auto* profile = fx->FirstChildElement("profile_COMMON")) {
        if (auto* tech = profile->FirstChildElement("technique")) {
          tinyxml2::XMLElement* shader = nullptr;
          for (char const* name : {"phong", "lambert", "blinn"}) {
            shader = tech->FirstChildElement(name);
            if (shader != nullptr) {
              break;
            }
          }
          if (shader != nullptr) {
            if (auto* diffuse = shader->FirstChildElement("diffuse")) {
              // Solid colors only; textured diffuse falls back to default gray.
              if (auto* color = diffuse->FirstChildElement("color")) {
                std::vector<float> const vals = details::ParseFloats(color->GetText());
                for (size_t i = 0; i < 4 && i < vals.size(); ++i) {
                  info.color[i] = vals[i];
                }
              }
            }
            if (auto* sh = shader->FirstChildElement("shininess")) {
              if (auto* f = sh->FirstChildElement("float")) {
                std::vector<float> const vals = details::ParseFloats(f->GetText());
                if (!vals.empty()) {
                  info.shininess = vals[0];
                  info.hasShininess = true;
                }
              }
            }
          }
        }
      }
      effects[id] = info;
    }
  }

  // --- Materials: material id → effect id ---
  std::unordered_map<std::string, std::string> materialToEffect;
  if (auto* libMat = root->FirstChildElement("library_materials")) {
    for (auto* m = libMat->FirstChildElement("material"); m != nullptr;
         m = m->NextSiblingElement("material")) {
      char const* id = m->Attribute("id");
      if (id == nullptr) {
        continue;
      }
      if (auto* ie = m->FirstChildElement("instance_effect")) {
        materialToEffect[id] = details::StripHash(ie->Attribute("url"));
      }
    }
  }

  // --- Visual scenes: primitive material symbol → material id ---
  std::unordered_map<std::string, std::string> symbolToMaterial;
  // --- Visual scenes: geometry id → node world transform(s) ---
  // Blender (and other) COLLADA exports commonly store geometry in local
  // coordinates (e.g. millimeters) with the unit scale and reorientation baked
  // into the <node> transform. Ignoring it yields meshes ~1000x too large. A
  // single geometry may be instanced by several nodes, so each maps to a list of
  // world transforms (one section set is emitted per instance).
  std::unordered_map<std::string, std::vector<mochi::Matrix4x4r>> geomToTransforms;
  if (auto* libScenes = root->FirstChildElement("library_visual_scenes")) {
    details::CollectInstanceMaterials(libScenes, symbolToMaterial);
    for (auto* scene = libScenes->FirstChildElement("visual_scene"); scene != nullptr;
         scene = scene->NextSiblingElement("visual_scene")) {
      details::CollectNodeTransforms(scene, mochi::Eye<4>(), geomToTransforms);
    }
  }

  // Resolves a primitive's material symbol through binding → material → effect.
  // Falls back to treating the symbol as a material/effect id directly, then to
  // a default gray effect.
  auto resolveEffect = [&](char const* symbol) -> details::EffectInfo {
    if (symbol == nullptr) {
      return {};
    }
    std::string materialId = symbol;
    if (auto it = symbolToMaterial.find(symbol); it != symbolToMaterial.end()) {
      materialId = it->second;
    }
    std::string effectId = materialId;
    if (auto it = materialToEffect.find(materialId); it != materialToEffect.end()) {
      effectId = it->second;
    }
    if (auto it = effects.find(effectId); it != effects.end()) {
      return it->second;
    }
    return {};
  };

  // --- Geometry ---
  auto* libGeo = root->FirstChildElement("library_geometries");
  if (libGeo == nullptr) {
    return sections;
  }

  for (auto* geo = libGeo->FirstChildElement("geometry"); geo != nullptr;
       geo = geo->NextSiblingElement("geometry")) {
    auto* mesh = geo->FirstChildElement("mesh");
    if (mesh == nullptr) {
      continue;
    }

    // World transforms placing this geometry into the scene. A geometry may be
    // instanced by multiple nodes; we emit one section set per instance. Empty
    // when the geometry is not referenced by any node (emit once in local space).
    std::vector<mochi::Matrix4x4r> instanceXforms;
    if (char const* geoId = geo->Attribute("id")) {
      if (auto it = geomToTransforms.find(geoId); it != geomToTransforms.end()) {
        instanceXforms = it->second;
      }
    }
    // Working transform for the current instance, updated by the instance loop
    // below and read by processPrimitive via capture.
    mochi::Matrix4x4r nodeXform = mochi::Eye<4>();
    bool hasNodeXform = false;

    // Index all <source> by id.
    std::unordered_map<std::string, details::Source> sources;
    for (auto* src = mesh->FirstChildElement("source"); src != nullptr;
         src = src->NextSiblingElement("source")) {
      char const* id = src->Attribute("id");
      if (id == nullptr) {
        continue;
      }
      details::Source s;
      if (auto* fa = src->FirstChildElement("float_array")) {
        s.data = details::ParseFloats(fa->GetText());
      }
      if (auto* tc = src->FirstChildElement("technique_common")) {
        if (auto* acc = tc->FirstChildElement("accessor")) {
          s.stride = acc->IntAttribute("stride", 3);
        }
      }
      sources[id] = std::move(s);
    }

    // Map <vertices id> → its POSITION source id.
    std::unordered_map<std::string, std::string> verticesToSource;
    for (auto* v = mesh->FirstChildElement("vertices"); v != nullptr;
         v = v->NextSiblingElement("vertices")) {
      char const* id = v->Attribute("id");
      if (id == nullptr) {
        continue;
      }
      for (auto* in = v->FirstChildElement("input"); in != nullptr;
           in = in->NextSiblingElement("input")) {
        char const* sem = in->Attribute("semantic");
        if (sem != nullptr && std::strcmp(sem, "POSITION") == 0) {
          verticesToSource[id] = details::StripHash(in->Attribute("source"));
        }
      }
    }

    // Builds one MeshSection from a primitive element (triangles/polylist/polygons)
    // by expanding each triangle corner into flat per-corner arrays.
    auto processPrimitive = [&](tinyxml2::XMLElement* prim, char const* tag) {
      std::string posSourceId;
      std::string normSourceId;
      int posOffset = 0;
      int normOffset = -1;
      int maxOffset = 0;
      for (auto* in = prim->FirstChildElement("input"); in != nullptr;
           in = in->NextSiblingElement("input")) {
        char const* sem = in->Attribute("semantic");
        int const offset = in->IntAttribute("offset", 0);
        maxOffset = std::max(maxOffset, offset);
        if (sem == nullptr) {
          continue;
        }
        if (std::strcmp(sem, "VERTEX") == 0) {
          std::string const vid = details::StripHash(in->Attribute("source"));
          if (auto it = verticesToSource.find(vid); it != verticesToSource.end()) {
            posSourceId = it->second;
          } else {
            posSourceId = vid;
          }
          posOffset = offset;
        } else if (std::strcmp(sem, "NORMAL") == 0) {
          normSourceId = details::StripHash(in->Attribute("source"));
          normOffset = offset;
        }
        // TEXCOORD and other semantics are ignored.
      }
      int const stride = maxOffset + 1;

      auto posIt = sources.find(posSourceId);
      if (posIt == sources.end()) {
        return; // No positions; nothing to emit.
      }
      details::Source const& posSrc = posIt->second;
      details::Source const* normSrc = nullptr;
      if (normOffset >= 0) {
        if (auto it = sources.find(normSourceId); it != sources.end()) {
          normSrc = &it->second;
        }
      }

      MeshSection section;
      section.hasNormals = (normSrc != nullptr);

      details::EffectInfo const effect = resolveEffect(prim->Attribute("material"));
      section.baseColor = effect.color;
      section.metallic = 0.0f;
      float const shininess = effect.hasShininess ? effect.shininess : 20.0f;
      section.roughness = std::clamp(std::sqrt(2.0f / (shininess + 2.0f)), 0.0f, 1.0f);

      int running = 0;
      auto emitCorner = [&](int corner, std::vector<int> const& p) {
        int const base = corner * stride;
        if (base + posOffset >= static_cast<int>(p.size())) {
          return;
        }
        int const posIndex = p[base + posOffset];
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        int const off = posIndex * posSrc.stride;
        if (off >= 0 && off + 2 < static_cast<int>(posSrc.data.size())) {
          x = posSrc.data[off];
          y = posSrc.data[off + 1];
          z = posSrc.data[off + 2];
        }
        if (hasNodeXform) {
          details::Real3 const tp = details::TransformPoint(nodeXform, details::Real3{x, y, z});
          x = static_cast<float>(tp[0]);
          y = static_cast<float>(tp[1]);
          z = static_cast<float>(tp[2]);
        }
        details::ApplyUpAxis(up, x, y, z);
        section.positions.push_back(x);
        section.positions.push_back(y);
        section.positions.push_back(z);

        if (normSrc != nullptr && base + normOffset < static_cast<int>(p.size())) {
          int const normIndex = p[base + normOffset];
          float nx = 0.0f;
          float ny = 0.0f;
          float nz = 0.0f;
          int const noff = normIndex * normSrc->stride;
          if (noff >= 0 && noff + 2 < static_cast<int>(normSrc->data.size())) {
            nx = normSrc->data[noff];
            ny = normSrc->data[noff + 1];
            nz = normSrc->data[noff + 2];
          }
          if (hasNodeXform) {
            details::Real3 const tn =
                details::TransformNormal(nodeXform, details::Real3{nx, ny, nz});
            nx = static_cast<float>(tn[0]);
            ny = static_cast<float>(tn[1]);
            nz = static_cast<float>(tn[2]);
          }
          details::ApplyUpAxis(up, nx, ny, nz);
          section.normals.push_back(nx);
          section.normals.push_back(ny);
          section.normals.push_back(nz);
        }

        section.indices.push_back(running++);
      };

      if (std::strcmp(tag, "triangles") == 0) {
        auto* pe = prim->FirstChildElement("p");
        if (pe == nullptr) {
          return;
        }
        std::vector<int> const p = details::ParseInts(pe->GetText());
        int const corners = static_cast<int>(p.size()) / stride;
        for (int c = 0; c < corners; ++c) {
          emitCorner(c, p);
        }
      } else if (std::strcmp(tag, "polylist") == 0) {
        auto* vce = prim->FirstChildElement("vcount");
        auto* pe = prim->FirstChildElement("p");
        if (vce == nullptr || pe == nullptr) {
          return;
        }
        std::vector<int> const vcount = details::ParseInts(vce->GetText());
        std::vector<int> const p = details::ParseInts(pe->GetText());
        int cursor = 0;
        for (int const n : vcount) {
          // Fan-triangulate the polygon (assumes convex).
          for (int i = 1; i + 1 < n; ++i) {
            emitCorner(cursor, p);
            emitCorner(cursor + i, p);
            emitCorner(cursor + i + 1, p);
          }
          cursor += n;
        }
      } else if (std::strcmp(tag, "polygons") == 0) {
        for (auto* pe = prim->FirstChildElement("p"); pe != nullptr;
             pe = pe->NextSiblingElement("p")) {
          std::vector<int> const p = details::ParseInts(pe->GetText());
          int const n = static_cast<int>(p.size()) / stride;
          for (int i = 1; i + 1 < n; ++i) {
            emitCorner(0, p);
            emitCorner(i, p);
            emitCorner(i + 1, p);
          }
        }
      }

      if (!section.positions.empty() && !section.indices.empty()) {
        sections.push_back(std::move(section));
      }
    };

    // Emit one section set per instance of this geometry. When the geometry is
    // not referenced by any node, emit a single set in local space.
    int const instanceCount = instanceXforms.empty() ? 1 : static_cast<int>(instanceXforms.size());
    for (int inst = 0; inst < instanceCount; ++inst) {
      if (instanceXforms.empty()) {
        nodeXform = mochi::Eye<4>();
        hasNodeXform = false;
      } else {
        nodeXform = instanceXforms[inst];
        hasNodeXform = true;
      }
      for (auto* prim = mesh->FirstChildElement(); prim != nullptr;
           prim = prim->NextSiblingElement()) {
        char const* name = prim->Name();
        if (std::strcmp(name, "triangles") == 0 || std::strcmp(name, "polylist") == 0 ||
            std::strcmp(name, "polygons") == 0) {
          processPrimitive(prim, name);
        }
      }
    }
  }

  return sections;
}

} // namespace collada_reader
