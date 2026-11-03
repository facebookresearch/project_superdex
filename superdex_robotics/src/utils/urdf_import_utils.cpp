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

#include "urdf_utils.h"

#include "geom_utils.h"

#include <superdex_robotics/utils/bot_utils.h>

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/transform_rt.h>

#include <tinyxml2.h>

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

using namespace mochi;
using namespace superdex::robotics;

// ---------------------------------------------------------------------------
// tinyxml2 URDF parsing
// ---------------------------------------------------------------------------

namespace {

namespace tx = tinyxml2;

// Intermediate parse structures. These mirror only the URDF subset the importer consumes
// (geometry, materials, inertial, joints).

struct ParsedInertial {
  real mass = 0_r;
  Real3 com = {};
  Real6 inertia = {};
};

struct ParsedMaterial {
  std::optional<Real4> rgba;
  std::string texture; // raw <texture filename>, or empty
};

// A <visual> or <collision> element. Mesh and texture paths are resolved later, against the file.
struct ParsedGeom {
  GeomReference geom;
  std::string textureRef;
};

struct ParsedLink {
  std::string name;
  std::optional<ParsedInertial> inertial;
  DynamicArray<ParsedGeom> geoms;
};

struct ParsedJoint {
  std::string name;
  std::string type;
  std::string parentLink;
  std::string childLink;
  Quaternion originRotation;
  Real3 originTranslation = {};
  Real3 axis = {}; // urdfdom default is (0,0,0); overridden per type below.
  bool hasDynamics = false;
  real damping = 0_r;
  real friction = 0_r;
  bool hasLimits = false;
  real lower = 0_r;
  real upper = 0_r;
  real effort = 0_r;
};

struct ParsedModel {
  std::string name;
  DynamicArray<ParsedLink> links;
  DynamicArray<ParsedJoint> joints;
  DynamicArray<DynamicString> warnings;
};

// Parse a single real from an attribute string; returns fallback if null or unparseable.
real ParseReal(char const* str, real fallback) {
  if (str == nullptr) {
    return fallback;
  }
  char* end = nullptr;
  double const v = std::strtod(str, &end);
  if (end == str) {
    return fallback;
  }
  return static_cast<real>(v);
}

// Parse exactly N whitespace-separated reals from an attribute string. A null string (absent
// attribute) yields the fallback without error. A present string that does not contain exactly
// N parseable components is reported via error (matching urdfdom's stricter validation), so
// malformed vectors are rejected rather than silently filled from the fallback.
template <int N>
NdArray<real, N> ParseReals(char const* str, NdArray<real, N> const& fallback, Error& error) {
  if (str == nullptr) {
    return fallback;
  }
  NdArray<real, N> out = fallback;
  char const* p = str;
  int i = 0;
  while (i < N) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') {
      ++p;
    }
    if (*p == '\0') {
      break;
    }
    char* end = nullptr;
    double const v = std::strtod(p, &end);
    if (end == p) {
      break;
    }
    out[i] = static_cast<real>(v);
    ++i;
    p = end;
  }
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') {
    ++p;
  }
  MOCHI_ERROR_IF(
      i != N || *p != '\0', error, "URDF vector attribute has the wrong number of components");
  return out;
}

Real3 ParseReal3(char const* str, Real3 const& fallback, Error& error) {
  return ParseReals<3>(str, fallback, error);
}

// URDF fixed-axis XYZ Euler angles → quaternion, composed from Mochi rotation primitives.
Quaternion RpyToQuaternion(Real3 const& rpy) {
  return Quaternion::RotationZ(rpy[2]) * Quaternion::RotationY(rpy[1]) *
      Quaternion::RotationX(rpy[0]);
}

// Parse an <origin xyz rpy> element into rotation + translation. Missing element/attributes
// default to identity / zeros.
void ParseOrigin(
    tx::XMLElement const* originXml,
    Quaternion& outRotation,
    Real3& outTranslation,
    Error& error) {
  outRotation = Quaternion::Identity();
  outTranslation = Real3{};
  if (originXml == nullptr) {
    return;
  }
  outTranslation = ParseReal3(originXml->Attribute("xyz"), Real3{}, error);
  outRotation = RpyToQuaternion(ParseReal3(originXml->Attribute("rpy"), Real3{}, error));
}

using MaterialMap = std::unordered_map<std::string, ParsedMaterial>;

// The color and texture a <material> element defines itself, if any.
std::optional<ParsedMaterial> ParseMaterialDefinition(
    tx::XMLElement const* materialXml,
    Error& error) {
  ParsedMaterial material;
  bool defined = false;
  if (tx::XMLElement const* colorXml = materialXml->FirstChildElement("color")) {
    material.rgba = ParseReals<4>(colorXml->Attribute("rgba"), Real4{0_r, 0_r, 0_r, 1_r}, error);
    defined = true;
  }
  if (tx::XMLElement const* textureXml = materialXml->FirstChildElement("texture")) {
    if (char const* filename = textureXml->Attribute("filename")) {
      material.texture = filename;
      defined = true;
    }
  }
  return defined ? std::optional<ParsedMaterial>(material) : std::nullopt;
}

// A <visual>'s material: its own color or texture, else the robot-level material of that name. A
// named definition is also recorded for later references, as urdfdom does.
std::optional<ParsedMaterial>
ParseVisualMaterial(tx::XMLElement const* materialXml, MaterialMap& materials, Error& error) {
  char const* name = materialXml->Attribute("name");
  if (std::optional<ParsedMaterial> material = ParseMaterialDefinition(materialXml, error)) {
    if (name != nullptr) {
      materials.try_emplace(name, *material);
    }
    return material;
  }
  if (name != nullptr) {
    if (auto const it = materials.find(name); it != materials.end()) {
      return it->second;
    }
  }
  return std::nullopt;
}

// The <geometry> of a <visual> or <collision>, or nullopt (with a warning) if it is absent or not
// supported.
std::optional<GeomReference> ParseGeometry(
    tx::XMLElement const* geometryXml,
    std::string const& where,
    DynamicArray<DynamicString>& warnings,
    Error& error) {
  tx::XMLElement const* shapeXml =
      geometryXml != nullptr ? geometryXml->FirstChildElement() : nullptr;
  if (shapeXml == nullptr) {
    warnings.push_back(DynamicString{"Skipped " + where + ", which has no geometry."});
    return std::nullopt;
  }
  GeomReference geom;
  std::string_view const shape = shapeXml->Name();
  if (shape == "mesh") {
    char const* filename = shapeXml->Attribute("filename");
    if (filename == nullptr) {
      warnings.push_back(DynamicString{"Skipped " + where + ", whose mesh has no filename."});
      return std::nullopt;
    }
    geom.type = GeomType::Mesh;
    geom.meshRef = DynamicString{filename};
    geom.meshScale = ParseReal3(shapeXml->Attribute("scale"), Real3{1_r, 1_r, 1_r}, error);
  } else if (shape == "box") {
    geom.type = GeomType::Box;
    geom.size = ParseReal3(shapeXml->Attribute("size"), Real3{}, error) * 0.5_r;
  } else if (shape == "cylinder") {
    geom.type = GeomType::Cylinder;
    geom.size = Real3{
        ParseReal(shapeXml->Attribute("radius"), 0_r),
        0.5_r * ParseReal(shapeXml->Attribute("length"), 0_r),
        0_r};
  } else if (shape == "sphere") {
    geom.type = GeomType::Sphere;
    geom.size = Real3{ParseReal(shapeXml->Attribute("radius"), 0_r), 0_r, 0_r};
  } else {
    warnings.push_back(
        DynamicString{"Skipped " + where + ": <" + std::string(shape) + "> is not supported."});
    return std::nullopt;
  }
  // A sphere uses only its radius; a cylinder its radius and half-length.
  int const numSizes = geom.type == GeomType::Sphere ? 1 : geom.type == GeomType::Cylinder ? 2 : 3;
  for (int i = 0; geom.type != GeomType::Mesh && i < numSizes; ++i) {
    if (!IsFinite(geom.size[i]) || geom.size[i] <= 0_r) {
      warnings.push_back(
          DynamicString{
              "Skipped " + where + ": its <" + std::string(shape) + "> has no positive size."});
      return std::nullopt;
    }
  }
  return geom;
}

ParsedLink ParseLink(
    tx::XMLElement const* linkXml,
    MaterialMap& materials,
    DynamicArray<DynamicString>& warnings,
    Error& error) {
  ParsedLink link;
  char const* name = linkXml->Attribute("name");
  MOCHI_ERROR_IF(name == nullptr, error, "URDF <link> missing name attribute");
  MOCHI_ERROR_RETURN(error, link);
  link.name = name;

  // Inertial (first).
  if (tx::XMLElement const* inertialXml = linkXml->FirstChildElement("inertial")) {
    ParsedInertial inertial;
    Quaternion rotation;
    ParseOrigin(inertialXml->FirstChildElement("origin"), rotation, inertial.com, error);
    if (tx::XMLElement const* massXml = inertialXml->FirstChildElement("mass")) {
      inertial.mass = ParseReal(massXml->Attribute("value"), 0_r);
    }
    if (tx::XMLElement const* inertiaXml = inertialXml->FirstChildElement("inertia")) {
      inertial.inertia = Real6{
          ParseReal(inertiaXml->Attribute("ixx"), 0_r),
          ParseReal(inertiaXml->Attribute("ixy"), 0_r),
          ParseReal(inertiaXml->Attribute("ixz"), 0_r),
          ParseReal(inertiaXml->Attribute("iyy"), 0_r),
          ParseReal(inertiaXml->Attribute("iyz"), 0_r),
          ParseReal(inertiaXml->Attribute("izz"), 0_r)};
    }
    link.inertial = inertial;
  }

  // Visual and collision elements, in document order.
  for (tx::XMLElement const* elem = linkXml->FirstChildElement(); elem != nullptr;
       elem = elem->NextSiblingElement()) {
    std::string_view const tag = elem->Name();
    bool const isVisual = tag == "visual";
    if (!isVisual && tag != "collision") {
      continue;
    }
    std::string const where = "a <" + std::string(tag) + "> of link \"" + link.name + "\"";
    std::optional<GeomReference> geom =
        ParseGeometry(elem->FirstChildElement("geometry"), where, warnings, error);
    MOCHI_ERROR_RETURN(error, link);
    if (!geom) {
      continue;
    }
    ParsedGeom parsed;
    parsed.geom = std::move(*geom);
    if (char const* geomName = elem->Attribute("name")) {
      parsed.geom.name = DynamicString{geomName};
    }
    Quaternion rotation;
    Real3 translation;
    ParseOrigin(elem->FirstChildElement("origin"), rotation, translation, error);
    parsed.geom.linkFromGeom = TransformRT{rotation, translation};
    parsed.geom.isVisual = isVisual;
    parsed.geom.isCollision = !isVisual;
    parsed.geom.keepMeshMaterials = true;
    if (tx::XMLElement const* materialXml = elem->FirstChildElement("material");
        isVisual && materialXml != nullptr) {
      if (std::optional<ParsedMaterial> material =
              ParseVisualMaterial(materialXml, materials, error)) {
        parsed.geom.hasMaterial = true;
        parsed.geom.rgba = material->rgba.value_or(Real4{1_r, 1_r, 1_r, 1_r});
        parsed.textureRef = material->texture;
      }
    }
    MOCHI_ERROR_RETURN(error, link);
    link.geoms.push_back(std::move(parsed));
  }
  return link;
}

ParsedJoint ParseJoint(tx::XMLElement const* jointXml, Error& error) {
  ParsedJoint joint;
  char const* name = jointXml->Attribute("name");
  MOCHI_ERROR_IF(name == nullptr, error, "URDF <joint> missing name attribute");
  MOCHI_ERROR_RETURN(error, joint);
  joint.name = name;

  char const* type = jointXml->Attribute("type");
  MOCHI_ERROR_IF(type == nullptr, error, "URDF <joint> missing type attribute");
  MOCHI_ERROR_RETURN(error, joint);
  joint.type = type;

  ParseOrigin(
      jointXml->FirstChildElement("origin"), joint.originRotation, joint.originTranslation, error);

  if (tx::XMLElement const* parentXml = jointXml->FirstChildElement("parent")) {
    if (char const* linkName = parentXml->Attribute("link")) {
      joint.parentLink = linkName;
    }
  }
  if (tx::XMLElement const* childXml = jointXml->FirstChildElement("child")) {
    if (char const* linkName = childXml->Attribute("link")) {
      joint.childLink = linkName;
    }
  }

  // Axis: urdfdom leaves fixed/floating joints at (0,0,0); other types default to (1,0,0) when
  // no <axis> element is present, or (0,0,0) when <axis> lacks an xyz attribute.
  if (joint.type != "fixed" && joint.type != "floating") {
    tx::XMLElement const* axisXml = jointXml->FirstChildElement("axis");
    if (axisXml == nullptr) {
      joint.axis = Real3{1_r, 0_r, 0_r};
    } else if (char const* xyz = axisXml->Attribute("xyz")) {
      joint.axis = ParseReal3(xyz, Real3{}, error);
    }
    MOCHI_ERROR_RETURN(error, joint);
    MOCHI_ERROR_IF(
        NormSqr(joint.axis) <= 0_r,
        error,
        "URDF non-fixed/non-floating joint has a degenerate zero axis");
    MOCHI_ERROR_RETURN(error, joint);
    // Normalize so the stored axis is unit length. The importer scales limits by this axis
    // (minLimit = axis * lower) while the exporter recovers them via Dot(limitVec,
    // Normalize(axis)); normalizing here keeps a non-unit URDF <axis> lossless across a round-trip.
    joint.axis = Normalize(joint.axis);
  }

  if (tx::XMLElement const* dynamicsXml = jointXml->FirstChildElement("dynamics")) {
    joint.hasDynamics = true;
    joint.damping = ParseReal(dynamicsXml->Attribute("damping"), 0_r);
    joint.friction = ParseReal(dynamicsXml->Attribute("friction"), 0_r);
  }

  if (tx::XMLElement const* limitXml = jointXml->FirstChildElement("limit")) {
    joint.hasLimits = true;
    joint.lower = ParseReal(limitXml->Attribute("lower"), 0_r);
    joint.upper = ParseReal(limitXml->Attribute("upper"), 0_r);
    joint.effort = ParseReal(limitXml->Attribute("effort"), 0_r);
  }
  return joint;
}

ParsedModel ParseUrdfDocument(tx::XMLDocument const& doc, Error& error) {
  ParsedModel model;
  tx::XMLElement const* robot = doc.RootElement();
  MOCHI_ERROR_IF(robot == nullptr, error, "URDF has no root element");
  MOCHI_ERROR_RETURN(error, model);
  MOCHI_ERROR_IF(
      std::string_view(robot->Name()) != "robot", error, "URDF root element is not <robot>");
  MOCHI_ERROR_RETURN(error, model);

  char const* robotName = robot->Attribute("name");
  MOCHI_ERROR_IF(robotName == nullptr, error, "URDF <robot> missing name attribute");
  MOCHI_ERROR_RETURN(error, model);
  model.name = robotName;

  MaterialMap materials;
  for (tx::XMLElement const* materialXml = robot->FirstChildElement("material");
       materialXml != nullptr;
       materialXml = materialXml->NextSiblingElement("material")) {
    char const* materialName = materialXml->Attribute("name");
    std::optional<ParsedMaterial> material = ParseMaterialDefinition(materialXml, error);
    MOCHI_ERROR_RETURN(error, model);
    if (materialName != nullptr && material) {
      materials[materialName] = *material;
    }
  }
  for (tx::XMLElement const* linkXml = robot->FirstChildElement("link"); linkXml != nullptr;
       linkXml = linkXml->NextSiblingElement("link")) {
    ParsedLink link = ParseLink(linkXml, materials, model.warnings, error);
    MOCHI_ERROR_RETURN(error, model);
    model.links.push_back(std::move(link));
  }
  for (tx::XMLElement const* jointXml = robot->FirstChildElement("joint"); jointXml != nullptr;
       jointXml = jointXml->NextSiblingElement("joint")) {
    ParsedJoint joint = ParseJoint(jointXml, error);
    MOCHI_ERROR_RETURN(error, model);
    model.joints.push_back(std::move(joint));
  }
  return model;
}

ArticulatedJointType ToJointType(std::string const& type) {
  if (type == "revolute" || type == "continuous") {
    return ArticulatedJointType::Revolute;
  }
  if (type == "prismatic") {
    return ArticulatedJointType::Prismatic;
  }
  if (type == "floating") {
    return ArticulatedJointType::Free;
  }
  // fixed and any other (planar, unknown) → Hard, matching the old importer.
  return ArticulatedJointType::Hard;
}

BotJointPrefab FromParsedJoint(ParsedJoint const& parsedJoint) {
  BotJointPrefab outJoint;
  outJoint.name = DynamicString{parsedJoint.name};
  outJoint.type = ToJointType(parsedJoint.type);
  outJoint.parentLinkFromJoint =
      TransformRT{parsedJoint.originRotation, parsedJoint.originTranslation};
  outJoint.axis = parsedJoint.axis;

  if (parsedJoint.hasDynamics) {
    outJoint.friction.viscous = parsedJoint.damping;
    outJoint.friction.coulomb = parsedJoint.friction;
  }

  if (parsedJoint.hasLimits) {
    outJoint.minLimit = parsedJoint.axis * parsedJoint.lower;
    outJoint.maxLimit = parsedJoint.axis * parsedJoint.upper;
    // URDF <limit effort="..."> is the max actuation effort [N·m / N]. Import it verbatim
    // (< 0 unbounded, 0 non-actuated, > 0 finite). When no <limit> is present the joint keeps the
    // BotJointPrefab default (kEffortUnbounded).
    outJoint.effortLimit = parsedJoint.effort;
  }

  if (parsedJoint.type == "continuous") {
    outJoint.minLimit = std::nullopt;
    outJoint.maxLimit = std::nullopt;
  }
  return outJoint;
}

void PopulateFromParsedModel(
    BotPrefab& outData,
    ParsedModel const& model,
    std::string_view meshBasePath,
    UrdfImportInfo& outInfo,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  outData.name = DynamicString{model.name};
  outInfo = UrdfImportInfo{};
  outInfo.warnings = model.warnings;

  int const numLinks = isize(model.links);
  int const numJoints = isize(model.joints);
  MOCHI_ERROR_IF(
      numJoints != numLinks - 1, error, "URDF joint count not equal to one less than link count");
  MOCHI_ERROR_RETURN(error);

  // Map link name → index.
  std::unordered_map<std::string_view, int> linkIndexByName;
  linkIndexByName.reserve(static_cast<size_t>(numLinks));
  for (int i = 0; i < numLinks; ++i) {
    // Reject duplicate link names: a collision would silently collapse both links to the first
    // index, mis-wiring the tree (urdfdom rejected these outright).
    bool const inserted = linkIndexByName.emplace(std::string_view(model.links[i].name), i).second;
    MOCHI_ERROR_IF(!inserted, error, "URDF contains duplicate link names");
    MOCHI_ERROR_RETURN(error);
  }

  // Build parent → children adjacency and each link's parent joint from the joint references.
  DynamicArray<DynamicArray<int>> childrenByLink(numLinks);
  DynamicArray<int> parentJointByLink(numLinks, -1);
  DynamicArray<bool> isChild(numLinks, false);
  for (int j = 0; j < numJoints; ++j) {
    ParsedJoint const& joint = model.joints[j];
    auto const parentIt = linkIndexByName.find(std::string_view(joint.parentLink));
    auto const childIt = linkIndexByName.find(std::string_view(joint.childLink));
    MOCHI_ERROR_IF(
        parentIt == linkIndexByName.end() || childIt == linkIndexByName.end(),
        error,
        "URDF joint references an unknown link");
    MOCHI_ERROR_RETURN(error);
    int const parentIdx = parentIt->second;
    int const childIdx = childIt->second;
    childrenByLink[parentIdx].push_back(childIdx);
    parentJointByLink[childIdx] = j;
    isChild[childIdx] = true;
  }

  // Root = the link never referenced as any joint's child. Exactly one is required.
  int rootLink = -1;
  int rootCount = 0;
  for (int i = 0; i < numLinks; ++i) {
    if (!isChild[i]) {
      rootLink = i;
      ++rootCount;
    }
  }
  MOCHI_ERROR_IF(rootCount != 1, error, "URDF must have exactly one root link");
  MOCHI_ERROR_RETURN(error);

  outData.links.resize(numLinks);
  outData.joints.reserve(numLinks);
  outInfo.links.resize(numLinks);

  auto& outLinks = outData.links;
  auto& outJoints = outData.joints;

  DynamicArray<bool> visited(numLinks, false);
  int iLink = -1;
  std::function<void(int, int, BotLinkPrefab*)> recurse =
      [&](int linkIdx, int depth, BotLinkPrefab* parentLink) {
        if (!error.IsOK()) {
          return;
        }
        if (visited[linkIdx]) {
          MOCHI_ERROR_SET(error, "URDF link graph contains a cycle");
          return;
        }
        visited[linkIdx] = true;
        bool const bIsRoot = depth == 0;
        ++iLink;
        ParsedLink const& parsedLink = model.links[linkIdx];
        BotLinkPrefab* link = &outLinks[iLink];
        link->name = DynamicString{parsedLink.name};
        link->_index = iLink;
        link->_parentFromLink = TransformRT::Identity();

        // Inertial properties.
        if (parsedLink.inertial) {
          link->mass = parsedLink.inertial->mass;
          link->centerOfMass = parsedLink.inertial->com;
          link->momentOfInertia = parsedLink.inertial->inertia;
        }

        // Geometry. Without a base path (an in-memory URDF), no file resolves.
        UrdfLinkReference& linkRef = outInfo.links[iLink];
        linkRef.link = link->name;
        for (ParsedGeom const& parsed : parsedLink.geoms) {
          GeomReference geom = parsed.geom;
          if (!meshBasePath.empty() && geom.type == GeomType::Mesh) {
            if (auto const resolved = ResolveMeshPath(geom.meshRef, meshBasePath)) {
              geom.meshFile = DynamicString{*resolved};
              geom.shapeFile = geom.meshFile;
              geom.renderModelFile = geom.meshFile;
            }
          }
          if (!meshBasePath.empty() && !parsed.textureRef.empty()) {
            if (auto const resolved = ResolveMeshPath(parsed.textureRef, meshBasePath)) {
              geom.textureFile = DynamicString{*resolved};
            }
          }
          linkRef.geoms.push_back(std::move(geom));
        }

        // Parent/child relations.
        if (parentLink != nullptr) {
          link->parentLink = parentLink->_index;
          parentLink->_childrenIndices.push_back(iLink);
        } else {
          link->parentLink = kIndexNone;
        }

        // Buffer this link's parent joint.
        int const parentJoint = parentJointByLink[linkIdx];
        if (parentJoint >= 0) {
          if (bIsRoot) {
            MOCHI_ERROR_SET(error, "Root link has parent joint");
            return;
          }
          ParsedJoint const& parsedJoint = model.joints[parentJoint];
          outJoints.push_back(FromParsedJoint(parsedJoint));
          link->_parentFromLink =
              TransformRT{parsedJoint.originRotation, parsedJoint.originTranslation};
          link->_rootFromLink = parentLink->_rootFromLink * link->_parentFromLink;
        }

        // Sort child links alphanumerically, then recurse.
        DynamicArray<int> childLinks = childrenByLink[linkIdx];
        std::ranges::sort(childLinks, [&](int a, int b) {
          return AlphaNumCompare(model.links[a].name.c_str(), model.links[b].name.c_str()) < 0;
        });
        for (int child : childLinks) {
          recurse(child, depth + 1, link);
        }
      };
  recurse(rootLink, 0, nullptr);
  MOCHI_ERROR_RETURN(error);

  // Every link must be reachable from the root via the joint tree. A count mismatch means the
  // graph is disconnected (e.g. a separate cyclic component that still satisfies the joint/link
  // count and single-root checks above).
  MOCHI_ERROR_IF(
      iLink + 1 != numLinks,
      error,
      "URDF link graph is not a fully connected tree (unreachable or cyclic links)");
  MOCHI_ERROR_RETURN(error);

  // Inject world joint at index 0 (URDF has N links, N-1 joints → we make N/N).
  BotJointPrefab worldJoint;
  worldJoint.name = "world_joint";
  worldJoint.type = ArticulatedJointType::Free;
  outJoints.push_back(worldJoint);
  std::ranges::rotate(outJoints, outJoints.end() - 1);
  superdex::robotics::RebuildBotData(outData, error);
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      outJoints.size() != outLinks.size(),
      error,
      "Joint count must equal link count after world joint injection");
  MOCHI_ERROR_RETURN(error);
}

} // namespace

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

void superdex::robotics::LoadBotPrefabFromUrdfFile(
    BotPrefab& outData,
    std::string_view path,
    UrdfImportInfo& outInfo,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  tx::XMLDocument doc;
  tx::XMLError const result = doc.LoadFile(std::string(path).c_str());
  MOCHI_ERROR_IF(result != tx::XML_SUCCESS, error, "Failed to load .urdf file");
  MOCHI_ERROR_RETURN(error);
  ParsedModel const model = ParseUrdfDocument(doc, error);
  MOCHI_ERROR_RETURN(error);
  PopulateFromParsedModel(outData, model, path, outInfo, error);
}

void superdex::robotics::LoadBotPrefabFromUrdfXml(
    BotPrefab& outData,
    std::string_view xmlString,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  tx::XMLDocument doc;
  tx::XMLError const result = doc.Parse(xmlString.data(), xmlString.size());
  MOCHI_ERROR_IF(result != tx::XML_SUCCESS, error, "Failed to parse URDF XML string");
  MOCHI_ERROR_RETURN(error);
  ParsedModel const model = ParseUrdfDocument(doc, error);
  MOCHI_ERROR_RETURN(error);
  UrdfImportInfo info;
  PopulateFromParsedModel(outData, model, {}, info, error);
}

void superdex::robotics::ExpandUrdfGeomLinks(
    BotPrefab& botPrefab,
    UrdfImportInfo const& info,
    Error& error) {
  DynamicArray<LinkGeoms> links;
  links.reserve(info.links.size());
  for (UrdfLinkReference const& link : info.links) {
    links.push_back(LinkGeoms{std::string_view(link.link), MakeConstSpan(link.geoms)});
  }
  ExpandGeomLinks(botPrefab, MakeConstSpan(links), error);
}
