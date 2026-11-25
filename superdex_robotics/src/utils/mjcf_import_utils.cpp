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

#include "mjcf_utils.h"

#include "geom_utils.h"

#include <superdex_robotics/utils/bot_utils.h>

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/quaternion_utils.h>
#include <mochi_core/utils/transform_rt.h>

#include <tinyxml2.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

using namespace mochi;
using namespace superdex::robotics;

namespace {

namespace tx = tinyxml2;
namespace fs = std::filesystem;

constexpr std::string_view kMainClass = "main";
constexpr std::string_view kActuatorDefaultsKey = "actuator";
constexpr int kMaxIncludeDepth = 32;
// Set on elements spliced in from an <include>: the directory of the file they came from.
constexpr char const* kIncludeDirAttribute = "superdex_include_dir";

// MuJoCo actuator element names. Their defaults share one attribute set, as in MuJoCo.
constexpr std::array<std::string_view, 10> kActuatorTags = {
    "general",
    "motor",
    "position",
    "velocity",
    "intvelocity",
    "damper",
    "cylinder",
    "muscle",
    "adhesion",
    "plugin"};

bool IsActuatorTag(std::string_view tag) {
  return std::find(kActuatorTags.begin(), kActuatorTags.end(), tag) != kActuatorTags.end();
}

// ---------------------------------------------------------------------------
// Small math helpers
// ---------------------------------------------------------------------------

// Quaternion of the rotation whose matrix columns are the orthonormal axes x, y, z.
Quaternion QuaternionFromAxes(Real3 const& x, Real3 const& y, Real3 const& z) {
  real const trace = x[0] + y[1] + z[2];
  real qx = 0_r;
  real qy = 0_r;
  real qz = 0_r;
  real qw = 1_r;
  if (trace > 0_r) {
    real const s = 0.5_r / std::sqrt(trace + 1_r);
    qw = 0.25_r / s;
    qx = (y[2] - z[1]) * s;
    qy = (z[0] - x[2]) * s;
    qz = (x[1] - y[0]) * s;
  } else if (x[0] > y[1] && x[0] > z[2]) {
    real const s = 2_r * std::sqrt(1_r + x[0] - y[1] - z[2]);
    qw = (y[2] - z[1]) / s;
    qx = 0.25_r * s;
    qy = (y[0] + x[1]) / s;
    qz = (z[0] + x[2]) / s;
  } else if (y[1] > z[2]) {
    real const s = 2_r * std::sqrt(1_r + y[1] - x[0] - z[2]);
    qw = (z[0] - x[2]) / s;
    qx = (y[0] + x[1]) / s;
    qy = 0.25_r * s;
    qz = (z[1] + y[2]) / s;
  } else {
    real const s = 2_r * std::sqrt(1_r + z[2] - x[0] - y[1]);
    qw = (x[1] - y[0]) / s;
    qx = (z[0] + x[2]) / s;
    qy = (z[1] + y[2]) / s;
    qz = 0.25_r * s;
  }
  return Normalize(Quaternion{qx, qy, qz, qw});
}

// Minimal rotation taking +Z onto the unit vector z.
Quaternion QuaternionFromZAxis(Real3 const& z) {
  Real3 const up{0_r, 0_r, 1_r};
  real const cosAngle = Clamp(Dot(up, z), -1_r, 1_r);
  Real3 const axis = Cross(up, z);
  real const sinAngle = Norm(axis);
  if (sinAngle < 1e-9_r) {
    return cosAngle > 0_r ? Quaternion::Identity() : Quaternion::RotationX(kPI);
  }
  return Quaternion::FromAxisAngle(axis / sinAngle, std::atan2(sinAngle, cosAngle));
}

// I = R diag(d) R^T, as [ixx, ixy, ixz, iyy, iyz, izz].
Real6 RotateDiagonalInertia(Real3 const& diagonal, Quaternion const& rotation) {
  Real6 out = {};
  for (int k = 0; k < 3; ++k) {
    Real3 axis = {};
    axis[k] = 1_r;
    Real3 const r = rotation * axis;
    out[0] += diagonal[k] * r[0] * r[0];
    out[1] += diagonal[k] * r[0] * r[1];
    out[2] += diagonal[k] * r[0] * r[2];
    out[3] += diagonal[k] * r[1] * r[1];
    out[4] += diagonal[k] * r[1] * r[2];
    out[5] += diagonal[k] * r[2] * r[2];
  }
  return out;
}

std::string SanitizeName(std::string name) {
  std::replace(name.begin(), name.end(), '/', '_');
  std::replace(name.begin(), name.end(), '\\', '_');
  return name;
}

// ---------------------------------------------------------------------------
// Parsed model
// ---------------------------------------------------------------------------

using AttrMap = std::map<std::string, std::string, std::less<>>;

struct DefaultClass {
  std::map<std::string, AttrMap, std::less<>> tags; // element tag -> attributes
};

struct MeshAsset {
  std::string ref;
  std::string file;
  Real3 scale = {1_r, 1_r, 1_r};
  TransformRT geomFromMesh;
};

struct MaterialAsset {
  Real4 rgba = {1_r, 1_r, 1_r, 1_r};
  std::string texture;
  std::array<real, 2> texRepeat = {1_r, 1_r};
  real shininess = 0.5_r;
};

struct TextureAsset {
  std::string file;
  bool is2d = false;
};

struct ParsedInertial {
  real mass = 0_r;
  Real3 com = {};
  Real6 moi = {};
};

struct ParsedJoint {
  std::string name;
  ArticulatedJointType type = ArticulatedJointType::Revolute;
  bool isFree = false;
  Real3 pos = {};
  Real3 axis = {0_r, 0_r, 1_r};
  std::optional<std::array<real, 2>> range;
  real damping = 0_r;
  real frictionLoss = 0_r;
  real armature = 0_r;
  std::optional<real> effortLimit;
  int qposAddress = 0;
};

struct ParsedSite {
  int body = -1;
  Real3 pos = {};
};

struct ParsedConnect {
  int body1 = -1;
  int body2 = -1;
  Real3 anchor = {}; // in body1's frame
};

struct ParsedBody {
  std::string name;
  int parent = -1;
  TransformRT parentFromBody;
  DynamicArray<ParsedJoint> joints;
  std::optional<ParsedInertial> inertial;
  DynamicArray<GeomReference> geoms; // linkFromGeom holds bodyFromGeom until emission
  real gravcomp = 0_r;
  DynamicArray<int> children;
};

struct Compiler {
  bool degrees = true;
  std::string eulerSeq = "xyz";
  std::optional<fs::path> meshDir;
  std::optional<fs::path> textureDir;
  std::optional<fs::path> assetDir;
  bool autoLimits = true;
  bool inertiaFromGeomAlways = false;
  real boundMass = 0_r;
  real boundInertia = 0_r;
};

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

class MjcfReader {
 public:
  MjcfReader(fs::path baseDir, MjcfImportInfo& info, Error& error)
      : _baseDir(std::move(baseDir)), _info(info), _error(error) {}

  void Read(tx::XMLDocument& doc, std::string const& fallbackName, BotPrefab& out);

 private:
  // Parsing
  void ExpandIncludes(tx::XMLDocument& doc, tx::XMLElement* parent, fs::path const& dir, int depth);
  void ParseCompiler(tx::XMLElement const* e);
  void ParseDefault(tx::XMLElement const* e, DefaultClass const& parent);
  void ParseAssets(tx::XMLElement const* assetXml);
  void ParseBodyChildren(
      tx::XMLElement const* parentXml,
      int bodyIndex,
      TransformRT const& bodyFromFrame,
      std::string const& childClass);
  void ParseBody(
      tx::XMLElement const* e,
      int parent,
      TransformRT const& parentFromFrame,
      std::string const& childClass);
  void ParseJoint(tx::XMLElement const* e, std::string const& cls, ParsedBody& body);
  void ParseInertial(tx::XMLElement const* e, ParsedBody& body);
  void ParseGeom(
      tx::XMLElement const* e,
      std::string const& cls,
      TransformRT const& bodyFromFrame,
      int bodyIndex);
  void ParseContact(tx::XMLElement const* contactXml);
  void ParseEquality(tx::XMLElement const* equalityXml);
  void ParseActuators(tx::XMLElement const* actuatorXml);
  void ParseKeyframe(tx::XMLElement const* keyframeXml);

  // Prefab construction
  void BuildPrefab(BotPrefab& out);
  void
  EmitBody(int bodyIndex, int parentLink, int parentInfo, TransformRT const& parentLinkFromBody);
  int AddLink(BotLinkPrefab link, BotJointPrefab joint);
  std::string ClaimJointName(std::string const& name);
  void AppendDefaultPose(ParsedJoint const& joint);

  // Attribute access
  char const* DefaultAttr(std::string_view tag, std::string const& cls, char const* name) const;
  char const* Attr(tx::XMLElement const* e, std::string const& cls, char const* name) const;
  std::optional<DynamicArray<real>> Reals(char const* text, char const* what);
  std::optional<real> GetReal(tx::XMLElement const* e, std::string const& cls, char const* name);
  std::optional<Real3> GetReal3(tx::XMLElement const* e, std::string const& cls, char const* name);
  std::optional<int> GetInt(tx::XMLElement const* e, std::string const& cls, char const* name);
  std::string GetString(tx::XMLElement const* e, std::string const& cls, char const* name) const;
  bool IsLimited(
      tx::XMLElement const* e,
      std::string const& cls,
      char const* limitedName,
      char const* rangeName) const;
  Quaternion ParseOrientation(tx::XMLElement const* e, std::string const& cls, bool useDefaults);
  TransformRT ParsePose(tx::XMLElement const* e, std::string const& cls, bool useDefaults);
  real Angle(real value) const;
  std::string ClassOf(tx::XMLElement const* e, std::string const& inherited);
  fs::path ResolveAsset(
      std::string const& file,
      std::optional<fs::path> const& dir,
      tx::XMLElement const* e) const;

  void Skip(std::string const& what);
  void Warn(std::string const& message);

  fs::path _baseDir;
  MjcfImportInfo& _info;
  Error& _error;

  Compiler _compiler;
  std::unordered_map<std::string, DefaultClass> _defaults;
  std::unordered_map<std::string, MeshAsset> _meshes;
  std::unordered_map<std::string, MaterialAsset> _materials;
  std::unordered_map<std::string, TextureAsset> _textures;

  std::string _modelName;
  DynamicArray<ParsedBody> _bodies;
  DynamicArray<int> _rootBodies;
  std::unordered_map<std::string, int> _bodyByName;
  std::unordered_map<std::string, int> _geomBodyByName;
  std::unordered_map<std::string, ParsedSite> _sites;
  std::unordered_map<std::string, std::optional<real>> _actuatorEffort; // joint name -> limit
  DynamicArray<std::pair<std::string, std::string>> _excludes;
  DynamicArray<std::pair<int, int>> _pairs;
  DynamicArray<ParsedConnect> _connects;
  DynamicArray<real> _keyQpos;
  int _qposSize = 0;

  std::map<std::string, int> _skipped; // element description -> count
  std::unordered_set<std::string> _warnedClasses;

  // Emission state
  BotPrefab* _out = nullptr;
  // Links and joints share one namespace in a BotPrefab. MJCF joint names are reserved up front so
  // joints keep them; a link whose body shares a joint's name gets a "_link" suffix.
  std::unordered_set<std::string> _names;
  std::unordered_set<std::string> _reservedJointNames;
  std::unordered_map<int, int> _infoByBody;
};

void MjcfReader::Skip(std::string const& what) {
  ++_skipped[what];
}

void MjcfReader::Warn(std::string const& message) {
  _info.warnings.push_back(DynamicString{message});
}

real MjcfReader::Angle(real value) const {
  return _compiler.degrees ? value * (kPI / 180_r) : value;
}

char const* MjcfReader::DefaultAttr(std::string_view tag, std::string const& cls, char const* name)
    const {
  auto classIt = _defaults.find(cls);
  if (classIt == _defaults.end()) {
    classIt = _defaults.find(std::string(kMainClass));
    if (classIt == _defaults.end()) {
      return nullptr;
    }
  }
  std::string_view const key = IsActuatorTag(tag) ? kActuatorDefaultsKey : tag;
  auto const tagIt = classIt->second.tags.find(key);
  if (tagIt == classIt->second.tags.end()) {
    return nullptr;
  }
  auto const attrIt = tagIt->second.find(std::string_view(name));
  return attrIt == tagIt->second.end() ? nullptr : attrIt->second.c_str();
}

char const* MjcfReader::Attr(tx::XMLElement const* e, std::string const& cls, char const* name)
    const {
  if (char const* value = e->Attribute(name)) {
    return value;
  }
  return DefaultAttr(e->Name(), cls, name);
}

std::optional<DynamicArray<real>> MjcfReader::Reals(char const* text, char const* what) {
  if (text == nullptr) {
    return std::nullopt;
  }
  DynamicArray<real> values;
  size_t at = 0;
  while (true) {
    while (text[at] == ' ' || text[at] == '\t' || text[at] == '\n' || text[at] == '\r') {
      ++at;
    }
    if (text[at] == '\0') {
      break;
    }
    char* end = nullptr;
    double const v = std::strtod(text + at, &end);
    auto const parsed = static_cast<size_t>(end - (text + at));
    if (parsed == 0) {
      MOCHI_ERROR_SET(_error, "MJCF attribute is not a list of numbers");
      MOCHI_LOG_ERROR("MJCF attribute '%s' is not a list of numbers: \"%s\"", what, text);
      return std::nullopt;
    }
    values.push_back(static_cast<real>(v));
    at += parsed;
  }
  return values;
}

std::optional<real>
MjcfReader::GetReal(tx::XMLElement const* e, std::string const& cls, char const* name) {
  auto const values = Reals(Attr(e, cls, name), name);
  if (!values || values->empty()) {
    return std::nullopt;
  }
  return (*values)[0];
}

std::optional<Real3>
MjcfReader::GetReal3(tx::XMLElement const* e, std::string const& cls, char const* name) {
  auto const values = Reals(Attr(e, cls, name), name);
  if (!values) {
    return std::nullopt;
  }
  if (values->size() != 3) {
    MOCHI_ERROR_SET(_error, "MJCF vector attribute must have three components");
    MOCHI_LOG_ERROR("MJCF attribute '%s' on <%s> must have three components", name, e->Name());
    return std::nullopt;
  }
  return Real3{(*values)[0], (*values)[1], (*values)[2]};
}

std::optional<int>
MjcfReader::GetInt(tx::XMLElement const* e, std::string const& cls, char const* name) {
  auto const value = GetReal(e, cls, name);
  return value ? std::optional<int>(static_cast<int>(*value)) : std::nullopt;
}

std::string MjcfReader::GetString(tx::XMLElement const* e, std::string const& cls, char const* name)
    const {
  char const* value = Attr(e, cls, name);
  return value ? std::string(value) : std::string();
}

bool MjcfReader::IsLimited(
    tx::XMLElement const* e,
    std::string const& cls,
    char const* limitedName,
    char const* rangeName) const {
  std::string const limited = GetString(e, cls, limitedName);
  if (limited == "true") {
    return true;
  }
  if (limited == "false") {
    return false;
  }
  // "auto" (the default): limited when a range is given and autolimits is on.
  return _compiler.autoLimits && Attr(e, cls, rangeName) != nullptr;
}

std::string MjcfReader::ClassOf(tx::XMLElement const* e, std::string const& inherited) {
  char const* cls = e->Attribute("class");
  if (cls == nullptr) {
    return inherited;
  }
  if (_defaults.find(cls) == _defaults.end() && _warnedClasses.insert(cls).second) {
    Warn(std::string("Unknown default class \"") + cls + "\"; using the main class.");
  }
  return cls;
}

Quaternion
MjcfReader::ParseOrientation(tx::XMLElement const* e, std::string const& cls, bool useDefaults) {
  auto parseWith = [&](auto const& get) -> std::optional<Quaternion> {
    if (auto const quat = Reals(get("quat"), "quat")) {
      MOCHI_ERROR_IF(quat->size() != 4, _error, "MJCF quat must have four components");
      MOCHI_ERROR_RETURN(_error, std::nullopt);
      // MJCF quaternions are (w, x, y, z).
      return Normalize(Quaternion{(*quat)[1], (*quat)[2], (*quat)[3], (*quat)[0]});
    }
    if (auto const axisAngle = Reals(get("axisangle"), "axisangle")) {
      MOCHI_ERROR_IF(axisAngle->size() != 4, _error, "MJCF axisangle must have four components");
      MOCHI_ERROR_RETURN(_error, std::nullopt);
      Real3 const axis{(*axisAngle)[0], (*axisAngle)[1], (*axisAngle)[2]};
      MOCHI_ERROR_IF(NormSqr(axis) <= 0_r, _error, "MJCF axisangle has a zero axis");
      MOCHI_ERROR_RETURN(_error, std::nullopt);
      return Quaternion::FromAxisAngle(Normalize(axis), Angle((*axisAngle)[3]));
    }
    if (auto const euler = Reals(get("euler"), "euler")) {
      MOCHI_ERROR_IF(euler->size() != 3, _error, "MJCF euler must have three components");
      MOCHI_ERROR_RETURN(_error, std::nullopt);
      Quaternion q = Quaternion::Identity();
      for (int i = 0; i < 3; ++i) {
        char const axisChar = _compiler.eulerSeq[i];
        real const angle = Angle((*euler)[i]);
        char const lower = static_cast<char>(std::tolower(static_cast<unsigned char>(axisChar)));
        Quaternion const r = lower == 'x' ? Quaternion::RotationX(angle)
            : lower == 'y'                ? Quaternion::RotationY(angle)
                                          : Quaternion::RotationZ(angle);
        // Lowercase axes rotate with the frame (intrinsic); uppercase stay fixed (extrinsic).
        q = (axisChar == lower) ? q * r : r * q;
      }
      return Normalize(q);
    }
    if (auto const xyAxes = Reals(get("xyaxes"), "xyaxes")) {
      MOCHI_ERROR_IF(xyAxes->size() != 6, _error, "MJCF xyaxes must have six components");
      MOCHI_ERROR_RETURN(_error, std::nullopt);
      Real3 const x = Normalize(Real3{(*xyAxes)[0], (*xyAxes)[1], (*xyAxes)[2]});
      Real3 y{(*xyAxes)[3], (*xyAxes)[4], (*xyAxes)[5]};
      y = Normalize(y - Dot(y, x) * x);
      return QuaternionFromAxes(x, y, Cross(x, y));
    }
    if (auto const zAxis = Reals(get("zaxis"), "zaxis")) {
      MOCHI_ERROR_IF(zAxis->size() != 3, _error, "MJCF zaxis must have three components");
      MOCHI_ERROR_RETURN(_error, std::nullopt);
      return QuaternionFromZAxis(Normalize(Real3{(*zAxis)[0], (*zAxis)[1], (*zAxis)[2]}));
    }
    return std::nullopt;
  };
  // An orientation on the element itself wins over any in its default class.
  if (auto const q = parseWith([&](char const* name) { return e->Attribute(name); })) {
    return *q;
  }
  if (useDefaults) {
    if (auto const q =
            parseWith([&](char const* name) { return DefaultAttr(e->Name(), cls, name); })) {
      return *q;
    }
  }
  return Quaternion::Identity();
}

TransformRT
MjcfReader::ParsePose(tx::XMLElement const* e, std::string const& cls, bool useDefaults) {
  char const* posText = useDefaults ? Attr(e, cls, "pos") : e->Attribute("pos");
  Real3 pos = {};
  if (auto const values = Reals(posText, "pos")) {
    MOCHI_ERROR_IF(values->size() != 3, _error, "MJCF pos must have three components");
    MOCHI_ERROR_RETURN(_error, {});
    pos = Real3{(*values)[0], (*values)[1], (*values)[2]};
  }
  return TransformRT{ParseOrientation(e, cls, useDefaults), pos};
}

fs::path MjcfReader::ResolveAsset(
    std::string const& file,
    std::optional<fs::path> const& dir,
    tx::XMLElement const* e) const {
  fs::path path(file);
  if (path.is_absolute()) {
    return path;
  }
  auto resolveFrom = [&](fs::path const& modelDir) {
    fs::path base = modelDir;
    if (std::optional<fs::path> const& assetDir = dir ? dir : _compiler.assetDir) {
      base = assetDir->is_absolute() ? *assetDir : modelDir / *assetDir;
    }
    return (base / path).lexically_normal();
  };
  fs::path resolved = resolveFrom(_baseDir);
  // Files named in an included file may instead be relative to that file, as MuJoCo allows.
  std::error_code ec;
  char const* includeDir = e->Attribute(kIncludeDirAttribute);
  if (includeDir != nullptr && !fs::exists(resolved, ec)) {
    fs::path fromInclude = resolveFrom(fs::path(includeDir));
    if (fs::exists(fromInclude, ec)) {
      return fromInclude;
    }
  }
  return resolved;
}

// Records on every element of an included file that does not yet know its source directory (from a
// deeper include) the directory of that file.
void TagIncludeDir(tx::XMLElement* e, std::string const& dir) {
  if (e->Attribute(kIncludeDirAttribute) == nullptr) {
    e->SetAttribute(kIncludeDirAttribute, dir.c_str());
  }
  for (tx::XMLElement* child = e->FirstChildElement(); child != nullptr;
       child = child->NextSiblingElement()) {
    TagIncludeDir(child, dir);
  }
}

void MjcfReader::ExpandIncludes(
    tx::XMLDocument& doc,
    tx::XMLElement* parent,
    fs::path const& dir,
    int depth) {
  MOCHI_ERROR_RETURN(_error);
  MOCHI_ERROR_IF(depth > kMaxIncludeDepth, _error, "MJCF <include> nesting is too deep");
  MOCHI_ERROR_RETURN(_error);
  tx::XMLElement* child = parent->FirstChildElement();
  while (child != nullptr) {
    if (std::string_view(child->Name()) != "include") {
      ExpandIncludes(doc, child, dir, depth);
      MOCHI_ERROR_RETURN(_error);
      child = child->NextSiblingElement();
      continue;
    }
    char const* file = child->Attribute("file");
    MOCHI_ERROR_IF(file == nullptr, _error, "MJCF <include> is missing its file attribute");
    MOCHI_ERROR_IF(dir.empty(), _error, "MJCF <include> cannot be resolved without a file path");
    MOCHI_ERROR_RETURN(_error);
    // Resolve relative to the including file, falling back to the main model file.
    fs::path includePath = (dir / file).lexically_normal();
    std::error_code ec;
    if (!fs::exists(includePath, ec)) {
      includePath = (_baseDir / file).lexically_normal();
    }
    tx::XMLDocument includeDoc;
    if (includeDoc.LoadFile(includePath.string().c_str()) != tx::XML_SUCCESS ||
        includeDoc.RootElement() == nullptr) {
      MOCHI_LOG_ERROR("Failed to load MJCF include '%s'", includePath.string().c_str());
      MOCHI_ERROR_SET(_error, "Failed to load MJCF <include> file");
      return;
    }
    ExpandIncludes(includeDoc, includeDoc.RootElement(), includePath.parent_path(), depth + 1);
    MOCHI_ERROR_RETURN(_error);
    TagIncludeDir(includeDoc.RootElement(), includePath.parent_path().string());
    tx::XMLNode* insertAfter = child;
    for (tx::XMLElement const* inc = includeDoc.RootElement()->FirstChildElement(); inc != nullptr;
         inc = inc->NextSiblingElement()) {
      tx::XMLNode* clone = inc->DeepClone(&doc);
      parent->InsertAfterChild(insertAfter, clone);
      insertAfter = clone;
    }
    tx::XMLElement* const next = insertAfter->NextSiblingElement();
    parent->DeleteChild(child);
    child = next;
  }
}

void MjcfReader::ParseCompiler(tx::XMLElement const* e) {
  if (char const* angle = e->Attribute("angle")) {
    _compiler.degrees = std::string_view(angle) != "radian";
  }
  if (char const* seq = e->Attribute("eulerseq")) {
    MOCHI_ERROR_IF(std::string_view(seq).size() != 3, _error, "MJCF eulerseq must have 3 axes");
    _compiler.eulerSeq = seq;
  }
  if (char const* dir = e->Attribute("meshdir")) {
    _compiler.meshDir = fs::path(dir);
  }
  if (char const* dir = e->Attribute("texturedir")) {
    _compiler.textureDir = fs::path(dir);
  }
  if (char const* dir = e->Attribute("assetdir")) {
    _compiler.assetDir = fs::path(dir);
  }
  if (char const* autoLimits = e->Attribute("autolimits")) {
    _compiler.autoLimits = std::string_view(autoLimits) == "true";
  }
  if (char const* fromGeom = e->Attribute("inertiafromgeom")) {
    _compiler.inertiaFromGeomAlways = std::string_view(fromGeom) == "true";
  }
  if (e->BoolAttribute("fitaabb", false)) {
    Skip("compiler fitaabb (primitives are fitted to the mesh's inertia box)");
  }
  _compiler.boundMass = static_cast<real>(e->DoubleAttribute("boundmass", _compiler.boundMass));
  _compiler.boundInertia =
      static_cast<real>(e->DoubleAttribute("boundinertia", _compiler.boundInertia));
}

void MjcfReader::ParseDefault(tx::XMLElement const* e, DefaultClass const& parent) {
  char const* name = e->Attribute("class");
  DefaultClass cls = parent;
  for (tx::XMLElement const* child = e->FirstChildElement(); child != nullptr;
       child = child->NextSiblingElement()) {
    std::string_view const tag = child->Name();
    if (tag == "default") {
      continue;
    }
    std::string const key(IsActuatorTag(tag) ? kActuatorDefaultsKey : tag);
    for (tx::XMLAttribute const* attr = child->FirstAttribute(); attr != nullptr;
         attr = attr->Next()) {
      cls.tags[key][attr->Name()] = attr->Value();
    }
  }
  _defaults[name ? std::string(name) : std::string(kMainClass)] = cls;
  for (tx::XMLElement const* child = e->FirstChildElement("default"); child != nullptr;
       child = child->NextSiblingElement("default")) {
    ParseDefault(child, cls);
  }
}

void MjcfReader::ParseAssets(tx::XMLElement const* assetXml) {
  for (tx::XMLElement const* e = assetXml->FirstChildElement(); e != nullptr;
       e = e->NextSiblingElement()) {
    std::string_view const tag = e->Name();
    std::string const cls = ClassOf(e, std::string(kMainClass));
    if (tag == "mesh") {
      MeshAsset mesh;
      std::string const file = GetString(e, cls, "file");
      std::string name = GetString(e, cls, "name");
      if (name.empty()) {
        name = fs::path(file).stem().string();
      }
      if (file.empty()) {
        Skip("<mesh> without a file (inline vertices)");
        continue;
      }
      mesh.ref = file;
      fs::path const resolved = ResolveAsset(file, _compiler.meshDir, e);
      std::error_code ec;
      if (!_baseDir.empty() && fs::exists(resolved, ec)) {
        mesh.file = resolved.string();
      }
      if (auto const scale = GetReal3(e, cls, "scale")) {
        mesh.scale = *scale;
      }
      // MuJoCo subtracts refpos from the (scaled) vertices, then rotates them by conj(refquat).
      Real3 refPos = {};
      if (auto const pos = GetReal3(e, cls, "refpos")) {
        refPos = *pos;
      }
      Quaternion refQuat = Quaternion::Identity();
      if (auto const quat = Reals(Attr(e, cls, "refquat"), "refquat"); quat && quat->size() == 4) {
        refQuat = Normalize(Quaternion{(*quat)[1], (*quat)[2], (*quat)[3], (*quat)[0]});
      }
      Quaternion const refInv = Conjugate(refQuat);
      mesh.geomFromMesh = TransformRT{refInv, -(refInv * refPos)};
      _meshes[name] = mesh;
    } else if (tag == "material") {
      MaterialAsset material;
      if (auto const rgba = Reals(Attr(e, cls, "rgba"), "rgba"); rgba && rgba->size() == 4) {
        material.rgba = Real4{(*rgba)[0], (*rgba)[1], (*rgba)[2], (*rgba)[3]};
      }
      material.texture = GetString(e, cls, "texture");
      if (auto const repeat = Reals(Attr(e, cls, "texrepeat"), "texrepeat");
          repeat && repeat->size() == 2) {
        material.texRepeat = {(*repeat)[0], (*repeat)[1]};
      }
      material.shininess = GetReal(e, cls, "shininess").value_or(material.shininess);
      _materials[GetString(e, cls, "name")] = material;
    } else if (tag == "texture") {
      TextureAsset texture;
      std::string const file = GetString(e, cls, "file");
      texture.is2d = GetString(e, cls, "type") == "2d";
      if (!file.empty()) {
        texture.file = ResolveAsset(file, _compiler.textureDir, e).string();
      }
      std::string name = GetString(e, cls, "name");
      if (name.empty()) {
        name = fs::path(file).stem().string();
      }
      _textures[name] = texture;
    } else {
      Skip("<" + std::string(tag) + "> asset");
    }
    MOCHI_ERROR_RETURN(_error);
  }
}

void MjcfReader::ParseBodyChildren(
    tx::XMLElement const* parentXml,
    int bodyIndex,
    TransformRT const& bodyFromFrame,
    std::string const& childClass) {
  for (tx::XMLElement const* e = parentXml->FirstChildElement(); e != nullptr;
       e = e->NextSiblingElement()) {
    MOCHI_ERROR_RETURN(_error);
    std::string_view const tag = e->Name();
    if (tag == "body") {
      ParseBody(e, bodyIndex, bodyFromFrame, childClass);
    } else if (tag == "frame") {
      std::string const frameClass =
          e->Attribute("childclass") ? std::string(e->Attribute("childclass")) : childClass;
      ParseBodyChildren(e, bodyIndex, bodyFromFrame * ParsePose(e, childClass, false), frameClass);
    } else if (bodyIndex < 0) {
      // Children of <worldbody> that are not bodies are scene content, not part of a robot.
      if (tag == "geom") {
        Skip("<geom> attached to the world");
      } else if (tag == "site") {
        Skip("<site>");
      } else if (tag != "light" && tag != "camera") {
        Skip("<" + std::string(tag) + "> in <worldbody>");
      }
    } else if (tag == "joint" || tag == "freejoint") {
      ParseJoint(e, ClassOf(e, childClass), _bodies[bodyIndex]);
    } else if (tag == "inertial") {
      ParseInertial(e, _bodies[bodyIndex]);
    } else if (tag == "geom") {
      ParseGeom(e, ClassOf(e, childClass), bodyFromFrame, bodyIndex);
    } else if (tag == "site") {
      std::string const cls = ClassOf(e, childClass);
      std::string const name = GetString(e, cls, "name");
      if (!name.empty()) {
        _sites[name] =
            ParsedSite{bodyIndex, (bodyFromFrame * ParsePose(e, cls, true)).GetTranslation()};
      }
      Skip("<site>");
    } else if (tag == "camera") {
      Skip("<camera>");
    } else if (tag == "light") {
      Skip("<light>");
    } else {
      Skip("<" + std::string(tag) + "> in <body>");
    }
  }
}

void MjcfReader::ParseBody(
    tx::XMLElement const* e,
    int parent,
    TransformRT const& parentFromFrame,
    std::string const& childClass) {
  MOCHI_ERROR_RETURN(_error);
  int const index = isize(_bodies);
  _bodies.emplace_back();
  ParsedBody& body = _bodies.back();
  char const* name = e->Attribute("name");
  body.name = name ? std::string(name) : "body_" + std::to_string(index);
  body.parent = parent;
  body.parentFromBody = parentFromFrame * ParsePose(e, childClass, false);
  body.gravcomp = static_cast<real>(e->DoubleAttribute("gravcomp", 0.0));
  if (e->Attribute("mocap") != nullptr && std::string_view(e->Attribute("mocap")) == "true") {
    Warn("Body \"" + body.name + "\" is a mocap body; imported as a regular body.");
  }
  _bodyByName[body.name] = index;
  if (parent >= 0) {
    _bodies[parent].children.push_back(index);
  } else {
    _rootBodies.push_back(index);
  }
  std::string const bodyChildClass =
      e->Attribute("childclass") ? std::string(e->Attribute("childclass")) : childClass;
  ParseBodyChildren(e, index, TransformRT::Identity(), bodyChildClass);
}

void MjcfReader::ParseJoint(tx::XMLElement const* e, std::string const& cls, ParsedBody& body) {
  ParsedJoint joint;
  char const* name = e->Attribute("name");
  joint.name = name ? std::string(name) : std::string();
  std::string const type = std::string_view(e->Name()) == "freejoint"
      ? std::string("free")
      : (GetString(e, cls, "type").empty() ? std::string("hinge") : GetString(e, cls, "type"));
  joint.qposAddress = _qposSize;
  if (type == "free") {
    joint.type = ArticulatedJointType::Free;
    joint.isFree = true;
    _qposSize += 7;
    body.joints.push_back(joint);
    return;
  }
  if (type == "ball") {
    joint.type = ArticulatedJointType::Spherical;
    _qposSize += 4;
    if (IsLimited(e, cls, "limited", "range")) {
      Warn("Ball joint \"" + joint.name + "\" has a cone limit, which is not imported.");
    }
  } else {
    joint.type = type == "slide" ? ArticulatedJointType::Prismatic : ArticulatedJointType::Revolute;
    _qposSize += 1;
    MOCHI_ERROR_IF(type != "slide" && type != "hinge", _error, "MJCF <joint> has an unknown type");
    MOCHI_ERROR_RETURN(_error);
    if (auto const axis = GetReal3(e, cls, "axis")) {
      MOCHI_ERROR_IF(NormSqr(*axis) <= 0_r, _error, "MJCF <joint> has a zero axis");
      MOCHI_ERROR_RETURN(_error);
      joint.axis = Normalize(*axis);
    }
    if (IsLimited(e, cls, "limited", "range")) {
      auto const range = Reals(Attr(e, cls, "range"), "range");
      if (!range || range->size() != 2) {
        MOCHI_ERROR_SET(_error, "MJCF joint range needs two values");
        return;
      }
      bool const isAngle = joint.type == ArticulatedJointType::Revolute;
      real const lo = (*range)[0];
      real const hi = (*range)[1];
      joint.range = std::array<real, 2>{isAngle ? Angle(lo) : lo, isAngle ? Angle(hi) : hi};
    }
  }
  if (auto const pos = GetReal3(e, cls, "pos")) {
    joint.pos = *pos;
  }
  joint.damping = GetReal(e, cls, "damping").value_or(0_r);
  joint.frictionLoss = GetReal(e, cls, "frictionloss").value_or(0_r);
  joint.armature = GetReal(e, cls, "armature").value_or(0_r);
  if (IsLimited(e, cls, "actuatorfrclimited", "actuatorfrcrange")) {
    auto const range = Reals(Attr(e, cls, "actuatorfrcrange"), "actuatorfrcrange");
    if (range && range->size() == 2) {
      joint.effortLimit = Max(Abs((*range)[0]), Abs((*range)[1]));
    }
  }
  if (GetReal(e, cls, "stiffness").value_or(0_r) != 0_r) {
    Skip("joint stiffness (springs)");
  }
  if (Attr(e, cls, "springdamper") != nullptr) {
    Skip("joint springdamper (inertia-derived spring and damping)");
  }
  if (GetReal(e, cls, "ref").value_or(0_r) != 0_r) {
    Skip("joint ref offset");
  }
  body.joints.push_back(joint);
}

void MjcfReader::ParseInertial(tx::XMLElement const* e, ParsedBody& body) {
  ParsedInertial inertial;
  TransformRT const pose = ParsePose(e, std::string(kMainClass), false);
  inertial.com = pose.GetTranslation();
  inertial.mass = static_cast<real>(e->DoubleAttribute("mass", 0.0));
  if (auto const full = Reals(e->Attribute("fullinertia"), "fullinertia")) {
    MOCHI_ERROR_IF(full->size() != 6, _error, "MJCF fullinertia needs six values");
    MOCHI_ERROR_RETURN(_error);
    // MJCF order is (ixx, iyy, izz, ixy, ixz, iyz); Mochi's is (ixx, ixy, ixz, iyy, iyz, izz).
    auto const& f = *full;
    inertial.moi = Real6{f[0], f[3], f[4], f[1], f[5], f[2]};
  } else if (auto const diag = Reals(e->Attribute("diaginertia"), "diaginertia")) {
    MOCHI_ERROR_IF(diag->size() != 3, _error, "MJCF diaginertia needs three values");
    MOCHI_ERROR_RETURN(_error);
    Real3 const principal{
        Max((*diag)[0], _compiler.boundInertia),
        Max((*diag)[1], _compiler.boundInertia),
        Max((*diag)[2], _compiler.boundInertia)};
    inertial.moi = RotateDiagonalInertia(principal, pose.GetRotation());
  }
  inertial.mass = Max(inertial.mass, _compiler.boundMass);
  MOCHI_ERROR_RETURN(_error);
  if (_compiler.inertiaFromGeomAlways) {
    return;
  }
  if (inertial.mass <= 0_r) {
    Warn("Body \"" + body.name + "\" has a massless <inertial>, which is ignored.");
    return;
  }
  body.inertial = inertial;
}

void MjcfReader::ParseGeom(
    tx::XMLElement const* e,
    std::string const& cls,
    TransformRT const& bodyFromFrame,
    int bodyIndex) {
  GeomReference geom;
  if (char const* name = e->Attribute("name")) {
    geom.name = name;
    _geomBodyByName[name] = bodyIndex;
  }
  std::string type = GetString(e, cls, "type");
  std::string const meshName = GetString(e, cls, "mesh");
  if (type.empty()) {
    type = meshName.empty() ? "sphere" : "mesh";
  }
  static std::unordered_map<std::string_view, GeomType> const kTypes = {
      {"plane", GeomType::Plane},
      {"hfield", GeomType::Hfield},
      {"sphere", GeomType::Sphere},
      {"capsule", GeomType::Capsule},
      {"ellipsoid", GeomType::Ellipsoid},
      {"cylinder", GeomType::Cylinder},
      {"box", GeomType::Box},
      {"mesh", GeomType::Mesh},
      {"sdf", GeomType::Sdf}};
  auto const typeIt = kTypes.find(type);
  MOCHI_ERROR_IF(typeIt == kTypes.end(), _error, "MJCF <geom> has an unknown type");
  MOCHI_ERROR_RETURN(_error);
  geom.type = typeIt->second;

  if (auto const size = Reals(Attr(e, cls, "size"), "size")) {
    for (int i = 0; i < Min(3, isize(*size)); ++i) {
      geom.size[i] = (*size)[i];
    }
  }

  TransformRT bodyFromGeom;
  if (auto const fromTo = Reals(e->Attribute("fromto"), "fromto")) {
    MOCHI_ERROR_IF(fromTo->size() != 6, _error, "MJCF fromto needs six values");
    MOCHI_ERROR_RETURN(_error);
    Real3 const from{(*fromTo)[0], (*fromTo)[1], (*fromTo)[2]};
    Real3 const to{(*fromTo)[3], (*fromTo)[4], (*fromTo)[5]};
    Real3 const dir = to - from;
    real const length = Norm(dir);
    Quaternion const rotation =
        length > 0_r ? QuaternionFromZAxis(dir / length) : Quaternion::Identity();
    bodyFromGeom = TransformRT{rotation, 0.5_r * (from + to)};
    // Capsules and cylinders take (radius, half-length); boxes and ellipsoids keep their X and Y
    // half-sizes and take the half-length along Z.
    if (geom.type == GeomType::Capsule || geom.type == GeomType::Cylinder) {
      geom.size[1] = 0.5_r * length;
    } else {
      geom.size[2] = 0.5_r * length;
    }
  } else {
    bodyFromGeom = ParsePose(e, cls, true);
  }
  geom.linkFromGeom = bodyFromFrame * bodyFromGeom;

  bool const isFitted = !meshName.empty() && IsPrimitiveGeom(geom.type);
  if (geom.type == GeomType::Mesh || isFitted) {
    auto const meshIt = _meshes.find(meshName);
    if (meshIt == _meshes.end()) {
      Warn("Geom references unknown mesh \"" + meshName + "\".");
    } else {
      geom.meshRef = meshIt->second.ref;
      geom.meshFile = meshIt->second.file;
      geom.meshScale = meshIt->second.scale;
      geom.geomFromMesh = meshIt->second.geomFromMesh;
      if (geom.type == GeomType::Mesh) {
        geom.shapeFile = geom.meshFile;
        geom.renderModelFile = geom.meshFile;
      }
    }
    if (isFitted) {
      geom.fitScale = GetReal(e, cls, "fitscale").value_or(1_r);
    }
  }

  // An explicit rgba (on the geom or its class) wins over the material's color.
  std::string const materialName = GetString(e, cls, "material");
  auto const materialIt = _materials.find(materialName);
  if (materialIt != _materials.end()) {
    MaterialAsset const& material = materialIt->second;
    geom.hasMaterial = true;
    geom.rgba = material.rgba;
    geom.texRepeat = material.texRepeat;
    geom.shininess = material.shininess;
    if (!material.texture.empty()) {
      auto const textureIt = _textures.find(material.texture);
      if (textureIt != _textures.end() && textureIt->second.is2d &&
          !textureIt->second.file.empty()) {
        geom.textureFile = textureIt->second.file;
      } else {
        Skip("non-2D or builtin texture");
      }
    }
  } else if (!materialName.empty()) {
    Warn("Geom references unknown material \"" + materialName + "\".");
  }
  if (auto const rgba = Reals(Attr(e, cls, "rgba"), "rgba"); rgba && rgba->size() == 4) {
    geom.rgba = Real4{(*rgba)[0], (*rgba)[1], (*rgba)[2], (*rgba)[3]};
    geom.hasMaterial = true;
  }

  geom.mass = GetReal(e, cls, "mass");
  geom.density = GetReal(e, cls, "density");
  geom.friction = GetReal(e, cls, "friction");
  geom.contype = GetInt(e, cls, "contype").value_or(1);
  geom.conaffinity = GetInt(e, cls, "conaffinity").value_or(1);
  geom.group = GetInt(e, cls, "group").value_or(0);
  geom.isCollision = (geom.contype | geom.conaffinity) != 0;
  geom.isVisual = geom.group >= 0 && geom.group <= 2 && geom.rgba[3] > 0_r;
  MOCHI_ERROR_RETURN(_error);
  _bodies[bodyIndex].geoms.push_back(std::move(geom));
}

void MjcfReader::ParseContact(tx::XMLElement const* contactXml) {
  for (tx::XMLElement const* e = contactXml->FirstChildElement(); e != nullptr;
       e = e->NextSiblingElement()) {
    std::string_view const tag = e->Name();
    if (tag == "exclude") {
      char const* body1 = e->Attribute("body1");
      char const* body2 = e->Attribute("body2");
      if (body1 != nullptr && body2 != nullptr) {
        _excludes.emplace_back(body1, body2);
      }
    } else if (tag == "pair") {
      char const* geom1 = e->Attribute("geom1");
      char const* geom2 = e->Attribute("geom2");
      auto const it1 = geom1 ? _geomBodyByName.find(geom1) : _geomBodyByName.end();
      auto const it2 = geom2 ? _geomBodyByName.find(geom2) : _geomBodyByName.end();
      if (it1 != _geomBodyByName.end() && it2 != _geomBodyByName.end() &&
          it1->second != it2->second) {
        _pairs.emplace_back(it1->second, it2->second);
      } else {
        Skip("<pair> with a world or unknown geom");
      }
    }
  }
}

void MjcfReader::ParseEquality(tx::XMLElement const* equalityXml) {
  for (tx::XMLElement const* e = equalityXml->FirstChildElement(); e != nullptr;
       e = e->NextSiblingElement()) {
    std::string_view const tag = e->Name();
    if (tag != "connect") {
      Skip("<equality " + std::string(tag) + ">");
      continue;
    }
    ParsedConnect connect;
    if (char const* site1 = e->Attribute("site1")) {
      char const* site2 = e->Attribute("site2");
      auto const it1 = _sites.find(site1);
      auto const it2 = site2 ? _sites.find(site2) : _sites.end();
      if (it1 == _sites.end() || it2 == _sites.end()) {
        Skip("<equality connect> with an unknown site");
        continue;
      }
      connect.body1 = it1->second.body;
      connect.body2 = it2->second.body;
      connect.anchor = it1->second.pos;
    } else {
      char const* body1 = e->Attribute("body1");
      char const* body2 = e->Attribute("body2");
      auto const it1 = body1 ? _bodyByName.find(body1) : _bodyByName.end();
      auto const it2 = body2 ? _bodyByName.find(body2) : _bodyByName.end();
      if (it1 == _bodyByName.end() || it2 == _bodyByName.end()) {
        Skip("<equality connect> to the world");
        continue;
      }
      connect.body1 = it1->second;
      connect.body2 = it2->second;
      if (auto const anchor = Reals(e->Attribute("anchor"), "anchor");
          anchor && anchor->size() == 3) {
        connect.anchor = Real3{(*anchor)[0], (*anchor)[1], (*anchor)[2]};
      }
    }
    _connects.push_back(connect);
  }
}

void MjcfReader::ParseActuators(tx::XMLElement const* actuatorXml) {
  for (tx::XMLElement const* e = actuatorXml->FirstChildElement(); e != nullptr;
       e = e->NextSiblingElement()) {
    std::string const cls = ClassOf(e, std::string(kMainClass));
    char const* joint = e->Attribute("joint");
    if (joint == nullptr) {
      joint = e->Attribute("jointinparent");
    }
    if (joint == nullptr) {
      Skip("actuator without a joint transmission");
      continue;
    }
    Skip("actuator gains and control ranges");
    std::optional<real>& limit = _actuatorEffort[joint];
    if (!IsLimited(e, cls, "forcelimited", "forcerange")) {
      continue;
    }
    auto const range = Reals(Attr(e, cls, "forcerange"), "forcerange");
    if (!range || range->size() != 2) {
      continue;
    }
    real gear = 1_r;
    if (auto const gears = Reals(Attr(e, cls, "gear"), "gear"); gears && !gears->empty()) {
      gear = Abs((*gears)[0]);
    }
    real const effort = gear * Max(Abs((*range)[0]), Abs((*range)[1]));
    limit = limit ? Min(*limit, effort) : effort;
  }
}

void MjcfReader::ParseKeyframe(tx::XMLElement const* keyframeXml) {
  tx::XMLElement const* chosen = nullptr;
  for (tx::XMLElement const* key = keyframeXml->FirstChildElement("key"); key != nullptr;
       key = key->NextSiblingElement("key")) {
    char const* name = key->Attribute("name");
    if (chosen == nullptr || (name != nullptr && std::string_view(name) == "home")) {
      chosen = key;
    }
  }
  if (chosen == nullptr) {
    return;
  }
  if (auto const qpos = Reals(chosen->Attribute("qpos"), "qpos")) {
    _keyQpos = *qpos;
  }
}

std::string MjcfReader::ClaimJointName(std::string const& name) {
  std::string sanitized = SanitizeName(name);
  if (_reservedJointNames.erase(sanitized) > 0) {
    return sanitized;
  }
  return ClaimUniqueName(sanitized, _names, "_joint");
}

int MjcfReader::AddLink(BotLinkPrefab link, BotJointPrefab joint) {
  link.name = DynamicString{ClaimUniqueName(SanitizeName(std::string(link.name)), _names, "_link")};
  joint.name = DynamicString{ClaimJointName(std::string(joint.name))};
  _out->links.push_back(std::move(link));
  _out->joints.push_back(std::move(joint));
  return isize(_out->links) - 1;
}

void MjcfReader::AppendDefaultPose(ParsedJoint const& joint) {
  auto qpos = [&](int offset) {
    int const index = joint.qposAddress + offset;
    return index < isize(_keyQpos) ? _keyQpos[index] : 0_r;
  };
  if (joint.type == ArticulatedJointType::Spherical) {
    // Keyframes hold ball joints as (w, x, y, z) quaternions; bot poses use rotation vectors.
    Quaternion const q = isize(_keyQpos) >= joint.qposAddress + 4
        ? Normalize(Quaternion{qpos(1), qpos(2), qpos(3), qpos(0)})
        : Quaternion::Identity();
    Real3 const rotationVector = q.ToRotationVector();
    for (int i = 0; i < 3; ++i) {
      _out->defaultPose.push_back(rotationVector[i]);
    }
  } else if (
      joint.type == ArticulatedJointType::Revolute ||
      joint.type == ArticulatedJointType::Prismatic) {
    _out->defaultPose.push_back(qpos(0));
  }
}

void MjcfReader::EmitBody(
    int bodyIndex,
    int parentLink,
    int parentInfo,
    TransformRT const& parentLinkFromBody) {
  ParsedBody const& body = _bodies[bodyIndex];
  for (ParsedJoint const& joint : body.joints) {
    MOCHI_ERROR_IF(joint.isFree, _error, "MJCF free joints are only supported on top-level bodies");
  }
  MOCHI_ERROR_RETURN(_error);

  // Each joint gets its own link, framed at the joint; the last one is the body's link. Links
  // keep the body's orientation, so linkFromBody is a pure translation.
  int link = parentLink;
  TransformRT linkFromBody = TransformRT::Identity();
  int const numJoints = isize(body.joints);
  for (int k = 0; k < Max(numJoints, 1); ++k) {
    BotLinkPrefab outLink;
    BotJointPrefab outJoint;
    bool const isBodyLink = k == Max(numJoints, 1) - 1;
    outLink.parentLink = link;
    if (numJoints == 0) {
      outLink.name = DynamicString{body.name};
      outJoint.name = DynamicString{body.name + "_fixed"};
      outJoint.type = ArticulatedJointType::Hard;
      outJoint.parentLinkFromJoint = parentLinkFromBody;
    } else {
      ParsedJoint const& joint = body.joints[k];
      std::string const jointName =
          joint.name.empty() ? body.name + "_joint" + std::to_string(k) : joint.name;
      outLink.name = DynamicString{isBodyLink ? body.name : body.name + "__" + jointName};
      outJoint.name = DynamicString{jointName};
      outJoint.type = joint.type;
      outJoint.parentLinkFromJoint = k == 0 ? parentLinkFromBody * TransformRT{joint.pos}
                                            : TransformRT{joint.pos - body.joints[k - 1].pos};
      if (joint.type != ArticulatedJointType::Spherical) {
        outJoint.axis = joint.axis;
        if (joint.range) {
          outJoint.minLimit = joint.axis * (*joint.range)[0];
          outJoint.maxLimit = joint.axis * (*joint.range)[1];
        }
      }
      outJoint.friction.viscous = joint.damping;
      outJoint.friction.coulomb = joint.frictionLoss;
      if (joint.armature > 0_r) {
        outJoint.inertia = joint.armature;
      }
      std::optional<real> effort = joint.effortLimit;
      if (auto const it = _actuatorEffort.find(joint.name);
          it != _actuatorEffort.end() && it->second) {
        effort = effort ? Min(*effort, *it->second) : *it->second;
      }
      if (effort) {
        outJoint.effortLimit = *effort;
      }
      linkFromBody = TransformRT{-joint.pos};
      AppendDefaultPose(joint);
    }
    if (isBodyLink) {
      if (body.inertial) {
        outLink.mass = body.inertial->mass;
        outLink.centerOfMass = linkFromBody.TransformPoint(body.inertial->com);
        outLink.momentOfInertia = body.inertial->moi;
      }
      if (body.gravcomp >= 1_r) {
        outLink.hasGravity = false;
      } else if (body.gravcomp > 0_r) {
        Skip("partial gravity compensation");
      }
    }
    link = AddLink(std::move(outLink), std::move(outJoint));
  }

  MjcfBodyReference ref;
  ref.name = DynamicString{body.name};
  ref.parentBody = parentInfo;
  ref.link = _out->links[link].name;
  ref.linkFromBody = linkFromBody;
  ref.hasInertial = body.inertial.has_value();
  ref.hasJoints = numJoints > 0;
  for (GeomReference geom : body.geoms) {
    geom.linkFromGeom = linkFromBody * geom.linkFromGeom;
    ref.geoms.push_back(std::move(geom));
  }
  int const infoIndex = isize(_info.bodies);
  _infoByBody[bodyIndex] = infoIndex;
  _info.bodies.push_back(std::move(ref));

  for (int child : body.children) {
    EmitBody(child, link, infoIndex, linkFromBody * _bodies[child].parentFromBody);
    MOCHI_ERROR_RETURN(_error);
  }
}

void MjcfReader::BuildPrefab(BotPrefab& out) {
  _out = &out;
  for (ParsedBody const& body : _bodies) {
    for (ParsedJoint const& joint : body.joints) {
      if (!joint.name.empty() && _names.insert(SanitizeName(joint.name)).second) {
        _reservedJointNames.insert(SanitizeName(joint.name));
      }
    }
  }
  out.name = DynamicString{_modelName};
  MOCHI_ERROR_IF(_rootBodies.empty(), _error, "MJCF <worldbody> contains no bodies");
  MOCHI_ERROR_RETURN(_error);

  // A top-level body with a free joint is the floating robot, and other top-level bodies are scene
  // content that cannot share its root. Otherwise, a single jointless top-level body is the welded
  // root, and anything else hangs off a shapeless world link.
  auto* const freeRoot = std::find_if(_rootBodies.begin(), _rootBodies.end(), [&](int body) {
    auto const& joints = _bodies[body].joints;
    return joints.size() == 1 && joints[0].isFree;
  });
  int rootBody = -1;
  if (freeRoot != _rootBodies.end()) {
    rootBody = *freeRoot;
  } else if (_rootBodies.size() == 1 && _bodies[_rootBodies[0]].joints.empty()) {
    rootBody = _rootBodies[0];
  }

  if (rootBody >= 0) {
    for (int other : _rootBodies) {
      if (other != rootBody) {
        Warn(
            "Skipped top-level body \"" + _bodies[other].name + "\", which is not attached to \"" +
            _bodies[rootBody].name + "\".");
      }
    }
    ParsedBody& body = _bodies[rootBody];
    bool const isFree = !body.joints.empty();
    out.worldFromRoot = body.parentFromBody;
    std::string const jointName =
        isFree && !body.joints[0].name.empty() ? body.joints[0].name : "world_joint";
    // Emit as a jointless body, then turn its joint into the root joint. A free joint's keyframe
    // entries are not bot DOFs, so dropping it from the joint list skips them.
    body.joints.clear();
    EmitBody(rootBody, kIndexNone, -1, TransformRT::Identity());
    MOCHI_ERROR_RETURN(_error);
    out.joints[0].type = isFree ? ArticulatedJointType::Free : ArticulatedJointType::Hard;
    _names.erase(std::string(out.joints[0].name));
    out.joints[0].name = DynamicString{ClaimJointName(jointName)};
    // A floating root moves on its free joint, so contact filtering must not weld it to the world.
    _info.bodies[_infoByBody.at(rootBody)].hasJoints = isFree;
  } else {
    BotLinkPrefab world;
    world.name = "world";
    world.parentLink = kIndexNone;
    BotJointPrefab worldJoint;
    worldJoint.name = "world_joint";
    worldJoint.type = ArticulatedJointType::Hard;
    int const worldLink = AddLink(std::move(world), std::move(worldJoint));
    for (int body : _rootBodies) {
      EmitBody(body, worldLink, -1, _bodies[body].parentFromBody);
      MOCHI_ERROR_RETURN(_error);
    }
  }

  // Cycle-closing joints from <equality connect>. The pivot is the anchor on body1, expressed in
  // body1's link frame; body2 follows from the initial pose.
  std::unordered_map<std::string_view, int> linkIndexByName;
  for (int i = 0; i < isize(out.links); ++i) {
    linkIndexByName[std::string_view(out.links[i].name)] = i;
  }
  for (ParsedConnect const& connect : _connects) {
    auto const it1 = _infoByBody.find(connect.body1);
    auto const it2 = _infoByBody.find(connect.body2);
    if (it1 == _infoByBody.end() || it2 == _infoByBody.end()) {
      continue;
    }
    MjcfBodyReference const& body1 = _info.bodies[it1->second];
    MjcfBodyReference const& body2 = _info.bodies[it2->second];
    Real3 const pivot = body1.linkFromBody.TransformPoint(connect.anchor);
    ArticulatedCycleJointParams cycle;
    cycle.childLink = linkIndexByName[std::string_view(body1.link)];
    cycle.parentLink = linkIndexByName[std::string_view(body2.link)];
    cycle.jointFromChildLink = Invert(TransformRT{pivot});
    out.cycles.push_back(cycle);
  }

  // Contact rules, by imported body index.
  for (auto const& [name1, name2] : _excludes) {
    auto const b1 = _bodyByName.find(name1);
    auto const b2 = _bodyByName.find(name2);
    if (b1 == _bodyByName.end() || b2 == _bodyByName.end()) {
      continue;
    }
    auto const i1 = _infoByBody.find(b1->second);
    auto const i2 = _infoByBody.find(b2->second);
    if (i1 != _infoByBody.end() && i2 != _infoByBody.end()) {
      _info.excludes.push_back(MjcfBodyPair{i1->second, i2->second});
    }
  }
  for (auto const& [body1, body2] : _pairs) {
    auto const i1 = _infoByBody.find(body1);
    auto const i2 = _infoByBody.find(body2);
    if (i1 != _infoByBody.end() && i2 != _infoByBody.end()) {
      _info.pairs.push_back(MjcfBodyPair{i1->second, i2->second});
    }
  }

  RebuildBotData(out, _error);
}

void MjcfReader::Read(tx::XMLDocument& doc, std::string const& fallbackName, BotPrefab& out) {
  MOCHI_ERROR_RETURN(_error);
  tx::XMLElement* root = doc.RootElement();
  MOCHI_ERROR_IF(root == nullptr, _error, "MJCF has no root element");
  MOCHI_ERROR_RETURN(_error);
  MOCHI_ERROR_IF(
      std::string_view(root->Name()) != "mujoco", _error, "MJCF root element is not <mujoco>");
  MOCHI_ERROR_RETURN(_error);
  char const* model = root->Attribute("model");
  _modelName = model ? std::string(model) : fallbackName;
  if (_modelName.empty()) {
    _modelName = "mjcf_robot";
  }

  ExpandIncludes(doc, root, _baseDir, 0);
  MOCHI_ERROR_RETURN(_error);

  for (tx::XMLElement const* e = root->FirstChildElement("compiler"); e != nullptr;
       e = e->NextSiblingElement("compiler")) {
    ParseCompiler(e);
  }
  MOCHI_ERROR_RETURN(_error);
  _defaults[std::string(kMainClass)] = DefaultClass{};
  // Every top-level <default> (several can come from includes) adds to the one main class.
  for (tx::XMLElement const* e = root->FirstChildElement("default"); e != nullptr;
       e = e->NextSiblingElement("default")) {
    DefaultClass const main = _defaults[std::string(kMainClass)];
    ParseDefault(e, main);
  }
  for (tx::XMLElement const* e = root->FirstChildElement("asset"); e != nullptr;
       e = e->NextSiblingElement("asset")) {
    ParseAssets(e);
    MOCHI_ERROR_RETURN(_error);
  }
  for (tx::XMLElement const* e = root->FirstChildElement("worldbody"); e != nullptr;
       e = e->NextSiblingElement("worldbody")) {
    ParseBodyChildren(e, -1, TransformRT::Identity(), std::string(kMainClass));
    MOCHI_ERROR_RETURN(_error);
  }
  for (tx::XMLElement const* e = root->FirstChildElement(); e != nullptr;
       e = e->NextSiblingElement()) {
    std::string_view const tag = e->Name();
    if (tag == "contact") {
      ParseContact(e);
    } else if (tag == "equality") {
      ParseEquality(e);
    } else if (tag == "actuator") {
      ParseActuators(e);
    } else if (tag == "keyframe") {
      ParseKeyframe(e);
    } else if (tag == "tendon" || tag == "sensor" || tag == "deformable" || tag == "flexcomp") {
      Skip("<" + std::string(tag) + ">");
    }
    MOCHI_ERROR_RETURN(_error);
  }

  BuildPrefab(out);
  MOCHI_ERROR_RETURN(_error);

  for (auto const& [what, count] : _skipped) {
    Warn("Skipped " + std::to_string(count) + " x " + what + ".");
  }
  if (!_info.warnings.empty()) {
    std::string notes;
    for (DynamicString const& warning : _info.warnings) {
      notes += "\n  ";
      notes += warning;
    }
    MOCHI_LOG_WARNING("MJCF import of \"%s\":%s", _modelName.c_str(), notes.c_str());
  }
}

void LoadMjcfDocument(
    tx::XMLDocument& doc,
    fs::path const& baseDir,
    std::string const& fallbackName,
    BotPrefab& outData,
    MjcfImportInfo& outInfo,
    Error& error) {
  outInfo = MjcfImportInfo{};
  MjcfReader reader(baseDir, outInfo, error);
  reader.Read(doc, fallbackName, outData);
}

} // namespace

// ---------------------------------------------------------------------------
// Internal entry points
// ---------------------------------------------------------------------------

void superdex::robotics::LoadBotPrefabFromMjcfFile(
    BotPrefab& outData,
    std::string_view path,
    MjcfImportInfo& outInfo,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  tx::XMLDocument doc;
  tx::XMLError const result = doc.LoadFile(std::string(path).c_str());
  MOCHI_ERROR_IF(result != tx::XML_SUCCESS, error, "Failed to load MJCF file");
  MOCHI_ERROR_RETURN(error);
  fs::path const filePath(path);
  fs::path const baseDir = filePath.parent_path().empty() ? fs::path(".") : filePath.parent_path();
  LoadMjcfDocument(doc, baseDir, filePath.stem().string(), outData, outInfo, error);
}

void superdex::robotics::LoadBotPrefabFromMjcfXml(
    BotPrefab& outData,
    std::string_view xmlString,
    MjcfImportInfo& outInfo,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  tx::XMLDocument doc;
  tx::XMLError const result = doc.Parse(xmlString.data(), xmlString.size());
  MOCHI_ERROR_IF(result != tx::XML_SUCCESS, error, "Failed to parse MJCF XML string");
  MOCHI_ERROR_RETURN(error);
  LoadMjcfDocument(doc, fs::path(), std::string(), outData, outInfo, error);
}

// ---------------------------------------------------------------------------
// Public geometry and contact helpers
// ---------------------------------------------------------------------------

void superdex::robotics::ExpandMjcfGeomLinks(
    BotPrefab& botPrefab,
    MjcfImportInfo const& info,
    Error& error) {
  DynamicArray<LinkGeoms> links;
  links.reserve(info.bodies.size());
  for (MjcfBodyReference const& body : info.bodies) {
    links.push_back(LinkGeoms{std::string_view(body.link), MakeConstSpan(body.geoms)});
  }
  ExpandGeomLinks(botPrefab, MakeConstSpan(links), error);
}

void superdex::robotics::ApplyMjcfContactFilter(BotPrefab& botPrefab, MjcfImportInfo const& info) {
  int const numBodies = isize(info.bodies);
  int const numLinks = isize(botPrefab.links);

  // MuJoCo welds a jointless body to its parent, and a jointless top-level body to the world
  // (mj_filterBodyPair): bodies on the same weld never collide, and a weld never collides with its
  // parent weld unless either one is the world.
  constexpr int kWorldWeld = -1;
  DynamicArray<int> weld(numBodies);
  for (int b = 0; b < numBodies; ++b) {
    int const parent = info.bodies[b].parentBody;
    weld[b] = info.bodies[b].hasJoints ? b : (parent < 0 ? kWorldWeld : weld[parent]);
  }
  auto parentWeld = [&](int w) {
    int const parent = info.bodies[w].parentBody;
    return parent < 0 ? kWorldWeld : weld[parent];
  };

  // Union of each body's collision bitmasks.
  DynamicArray<int> contype(numBodies, 0);
  DynamicArray<int> conaffinity(numBodies, 0);
  for (int b = 0; b < numBodies; ++b) {
    for (GeomReference const& geom : info.bodies[b].geoms) {
      if (geom.isCollision) {
        contype[b] |= geom.contype;
        conaffinity[b] |= geom.conaffinity;
      }
    }
  }

  auto isPair = [](DynamicArray<MjcfBodyPair> const& pairs, int a, int b) {
    return std::any_of(pairs.begin(), pairs.end(), [&](MjcfBodyPair const& p) {
      return (p.body1 == a && p.body2 == b) || (p.body1 == b && p.body2 == a);
    });
  };
  auto mujocoCollides = [&](int a, int b) {
    if (isPair(info.pairs, a, b)) {
      return true;
    }
    if (isPair(info.excludes, a, b)) {
      return false;
    }
    int const wa = weld[a];
    int const wb = weld[b];
    if (wa == wb) {
      return false;
    }
    if (wa != kWorldWeld && wb != kWorldWeld && (parentWeld(wa) == wb || parentWeld(wb) == wa)) {
      return false;
    }
    return (contype[a] & conaffinity[b]) != 0 || (contype[b] & conaffinity[a]) != 0;
  };

  // Map each shape-bearing link to its body: the body's own link or one of its collision children.
  DynamicArray<int> bodyOfLink(numLinks, -1);
  for (int l = 0; l < numLinks; ++l) {
    if (botPrefab.links[l].shapeFile.empty()) {
      continue;
    }
    std::string_view const name(botPrefab.links[l].name);
    for (int b = 0; b < numBodies; ++b) {
      std::string const bodyLink(info.bodies[b].link);
      if (name == bodyLink ||
          (name.starts_with(bodyLink) && name.substr(bodyLink.size()).starts_with("_collision_"))) {
        bodyOfLink[l] = b;
        break;
      }
    }
  }

  auto hasOverride = [&](DynamicString const& a, DynamicString const& b) {
    return std::any_of(
        botPrefab.contactOverrides.begin(),
        botPrefab.contactOverrides.end(),
        [&](BotContactOverride const& o) {
          return (o.linkA == a && o.linkB == b) || (o.linkA == b && o.linkB == a);
        });
  };

  for (int a = 0; a < numLinks; ++a) {
    for (int b = a + 1; b < numLinks; ++b) {
      if (bodyOfLink[a] < 0 || bodyOfLink[b] < 0) {
        continue;
      }
      bool const collides =
          bodyOfLink[a] != bodyOfLink[b] && mujocoCollides(bodyOfLink[a], bodyOfLink[b]);
      bool const mochiCollides = !IsContactImplicitlyDisabled(botPrefab, a, b);
      if (collides == mochiCollides ||
          hasOverride(botPrefab.links[a].name, botPrefab.links[b].name)) {
        continue;
      }
      BotContactOverride contactOverride;
      contactOverride.linkA = botPrefab.links[a].name;
      contactOverride.linkB = botPrefab.links[b].name;
      contactOverride.enable = collides;
      botPrefab.contactOverrides.push_back(std::move(contactOverride));
    }
  }
}
