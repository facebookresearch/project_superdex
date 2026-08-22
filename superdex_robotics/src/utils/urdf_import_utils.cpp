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

#include <superdex_robotics/utils/bot_utils.h>

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/quaternion.h>

#include <tinyxml2.h>

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>

using namespace mochi;
using namespace superdex::robotics;

// ---------------------------------------------------------------------------
// tinyxml2 URDF parsing
// ---------------------------------------------------------------------------

namespace {

namespace tx = tinyxml2;

// Intermediate parse structures. These mirror only the URDF subset the old urdfdom-based
// importer consumed (mesh geometry, first visual/collision, inertial, joints).

struct ParsedInertial {
  real mass = 0_r;
  Real3 com = {};
  Real6 inertia = {};
};

struct ParsedMesh {
  std::string filename;
  Real3 scale = {1_r, 1_r, 1_r};
  Quaternion originRotation;
  Real3 originTranslation = {};
};

struct ParsedLink {
  std::string name;
  std::optional<ParsedInertial> inertial;
  std::optional<ParsedMesh> visual;
  std::optional<ParsedMesh> collision;
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

// Parse exactly three whitespace-separated reals from an attribute string. A null string (absent
// attribute) yields the fallback without error. A present string that does not contain exactly
// three parseable components is reported via error (matching urdfdom's stricter validation), so
// malformed vectors are rejected rather than silently filled from the fallback.
Real3 ParseReal3(char const* str, Real3 const& fallback, Error& error) {
  if (str == nullptr) {
    return fallback;
  }
  Real3 out = fallback;
  char const* p = str;
  int i = 0;
  while (i < 3) {
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
      i != 3 || *p != '\0', error, "URDF vector attribute must have exactly three components");
  return out;
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

// Parse the <mesh> under a <geometry> element. Returns nullopt for non-mesh geometry (box,
// cylinder, sphere) or a mesh without a filename — matching the old importer's mesh-only behavior.
std::optional<ParsedMesh> ParseGeometryMesh(tx::XMLElement const* geometryXml, Error& error) {
  if (geometryXml == nullptr) {
    return std::nullopt;
  }
  tx::XMLElement const* meshXml = geometryXml->FirstChildElement("mesh");
  if (meshXml == nullptr) {
    return std::nullopt;
  }
  char const* filename = meshXml->Attribute("filename");
  if (filename == nullptr) {
    return std::nullopt;
  }
  ParsedMesh mesh;
  mesh.filename = filename;
  if (char const* scaleStr = meshXml->Attribute("scale")) {
    mesh.scale = ParseReal3(scaleStr, Real3{1_r, 1_r, 1_r}, error);
  }
  return mesh;
}

// Parse the first mesh-bearing <visual>/<collision> child (tag), baking its <origin> into the
// returned mesh. Elements whose geometry is a non-mesh primitive (box/cylinder/sphere) are
// skipped, so a mesh in a later same-tag sibling is still found.
std::optional<ParsedMesh>
ParseVisualOrCollision(tx::XMLElement const* parentXml, char const* tag, Error& error) {
  for (tx::XMLElement const* elem = parentXml->FirstChildElement(tag); elem != nullptr;
       elem = elem->NextSiblingElement(tag)) {
    std::optional<ParsedMesh> mesh = ParseGeometryMesh(elem->FirstChildElement("geometry"), error);
    MOCHI_ERROR_RETURN(error, std::nullopt);
    if (!mesh) {
      continue;
    }
    ParseOrigin(
        elem->FirstChildElement("origin"), mesh->originRotation, mesh->originTranslation, error);
    return mesh;
  }
  return std::nullopt;
}

ParsedLink ParseLink(tx::XMLElement const* linkXml, Error& error) {
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

  // Visual / collision (first mesh of each).
  link.visual = ParseVisualOrCollision(linkXml, "visual", error);
  link.collision = ParseVisualOrCollision(linkXml, "collision", error);
  MOCHI_ERROR_RETURN(error, link);
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

  for (tx::XMLElement const* linkXml = robot->FirstChildElement("link"); linkXml != nullptr;
       linkXml = linkXml->NextSiblingElement("link")) {
    ParsedLink link = ParseLink(linkXml, error);
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
    UrdfMeshReferences* meshRefs,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  outData.name = DynamicString{model.name};

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
  if (meshRefs != nullptr) {
    meshRefs->links.resize(numLinks);
  }

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

        // Visual geometry (mesh path, transform, scale).
        if (!meshBasePath.empty() && parsedLink.visual) {
          ParsedMesh const& mesh = *parsedLink.visual;
          if (auto const resolved = ResolveMeshPath(mesh.filename, meshBasePath)) {
            link->renderModelFile = DynamicString{*resolved};
          }
          if (meshRefs != nullptr) {
            meshRefs->links[iLink].visual = DynamicString{mesh.filename};
          }
          link->renderModelRotation = mesh.originRotation;
          link->renderModelTranslation = mesh.originTranslation;
          link->renderModelScale = mesh.scale;
        }

        // Collision geometry (mesh path, transform, scale).
        if (!meshBasePath.empty() && parsedLink.collision) {
          ParsedMesh const& mesh = *parsedLink.collision;
          if (auto const resolved = ResolveMeshPath(mesh.filename, meshBasePath)) {
            link->shapeFile = DynamicString{*resolved};
          }
          if (meshRefs != nullptr) {
            meshRefs->links[iLink].collision = DynamicString{mesh.filename};
          }
          link->shapeRotation = mesh.originRotation;
          link->shapeTranslation = mesh.originTranslation;
          link->shapeScale = mesh.scale;
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
        std::sort(childLinks.begin(), childLinks.end(), [&](int a, int b) {
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
  std::rotate(outJoints.begin(), outJoints.end() - 1, outJoints.end());
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
    UrdfMeshReferences* meshRefs,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  tx::XMLDocument doc;
  tx::XMLError const result = doc.LoadFile(std::string(path).c_str());
  MOCHI_ERROR_IF(result != tx::XML_SUCCESS, error, "Failed to load .urdf file");
  MOCHI_ERROR_RETURN(error);
  ParsedModel const model = ParseUrdfDocument(doc, error);
  MOCHI_ERROR_RETURN(error);
  PopulateFromParsedModel(outData, model, path, meshRefs, error);
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
  PopulateFromParsedModel(outData, model, {}, nullptr, error);
}
