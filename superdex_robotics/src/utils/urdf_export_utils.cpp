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

#include <superdex_robotics/sensors/camera_sensor.h>

#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/quaternion_utils.h>
#include <mochi_physics/utils/mochi_prefab.h>

#include <tinyxml2.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace mochi;
using namespace superdex::robotics;

namespace {

namespace tx = tinyxml2;

// URDF joint-type strings (replaces the ::urdf::Joint::* enum constants).
constexpr char const* kJointTypeRevolute = "revolute";
constexpr char const* kJointTypeContinuous = "continuous";
constexpr char const* kJointTypePrismatic = "prismatic";
constexpr char const* kJointTypeFixed = "fixed";
constexpr char const* kJointTypeFloating = "floating";

std::string FormatDouble(double v) {
  // Avoid printing "-0" — normalize negative zero to positive zero.
  if (v == 0.0) {
    v = 0.0;
  }
  // Emit enough significant digits to round-trip a `real` value exactly (9 for float, 17 for
  // double). The stream's default (6) silently truncated exported origins, inertia, and limits.
  std::ostringstream ss;
  ss << std::setprecision(std::numeric_limits<real>::max_digits10) << v;
  return ss.str();
}

std::string FormatXyz(double x, double y, double z) {
  return FormatDouble(x) + " " + FormatDouble(y) + " " + FormatDouble(z);
}

// Quaternion → URDF fixed-axis XYZ Euler angles (roll, pitch, yaw). Inverts the importer's
// RotationZ(yaw) * RotationY(pitch) * RotationX(roll). Uses atan2 and handles gimbal lock at
// pitch ≈ ±π/2.
std::string FormatRpy(Quaternion const& q) {
  Matrix3x3r const rot = ToMatrix3x3(q);
  auto const r00 = static_cast<double>(rot[0][0]);
  auto const r01 = static_cast<double>(rot[0][1]);
  auto const r10 = static_cast<double>(rot[1][0]);
  auto const r11 = static_cast<double>(rot[1][1]);
  auto const r20 = static_cast<double>(rot[2][0]);
  auto const r21 = static_cast<double>(rot[2][1]);
  auto const r22 = static_cast<double>(rot[2][2]);

  double const cosPitch = std::sqrt(r00 * r00 + r10 * r10);
  constexpr double kHalfPi = static_cast<double>(kPI) / 2.0;
  // Near pitch = ±π/2 the yaw/roll terms (r00, r10 and r21, r22) all collapse toward zero, so the
  // atan2-based split of yaw and roll becomes ill-conditioned. Two effects compound near the pole:
  // (1) exactly at the singularity r21/r10 vanish identically, losing the coupled angle; and (2)
  // with `real` == float, those near-zero entries come from `1 - 2(...)` cancellation in the
  // quaternion→matrix step, so they carry ~1e-7 absolute error — an O(1) relative error once
  // cosPitch drops into the ~1e-6 range. Either way the non-gimbal split returns garbage roll/yaw.
  // Switch to the gimbal branch (which reads the O(1) entries r01/r11) well before that. kGimbalEps
  // bounds the pitch snap error (≈ cosPitch): 1e-4 stays within the round-trip rotation tolerance
  // yet is large enough to capture truncated right angles like rpy="1.5708" (cosPitch ≈ 3.7e-6),
  // which sit far inside the float-cancellation dead zone.
  constexpr double kGimbalEps = 1e-4;

  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
  if (cosPitch > kGimbalEps) {
    pitch = std::atan2(-r20, cosPitch);
    yaw = std::atan2(r10, r00);
    roll = std::atan2(r21, r22);
  } else {
    // Gimbal lock: pitch = ±π/2, roll and yaw are coupled. Pin yaw = 0.
    yaw = 0.0;
    if (r20 < 0.0) {
      pitch = kHalfPi;
      roll = std::atan2(r01, r11);
    } else {
      pitch = -kHalfPi;
      roll = std::atan2(-r01, r11);
    }
  }
  return FormatDouble(roll) + " " + FormatDouble(pitch) + " " + FormatDouble(yaw);
}

void WriteOrigin(tx::XMLElement* parentXml, Real3 const& translation, Quaternion const& rotation) {
  tx::XMLElement* originXml = parentXml->GetDocument()->NewElement("origin");
  originXml->SetAttribute(
      "xyz",
      FormatXyz(
          static_cast<double>(translation[0]),
          static_cast<double>(translation[1]),
          static_cast<double>(translation[2]))
          .c_str());
  originXml->SetAttribute("rpy", FormatRpy(rotation).c_str());
  parentXml->InsertEndChild(originXml);
}

void WriteJointLinks(tx::XMLElement* jointXml, std::string_view parent, std::string_view child) {
  for (auto const& [tag, link] : {std::pair{"parent", parent}, std::pair{"child", child}}) {
    tx::XMLElement* linkXml = jointXml->GetDocument()->NewElement(tag);
    linkXml->SetAttribute("link", std::string(link).c_str());
    jointXml->InsertEndChild(linkXml);
  }
}

// Determine the URDF joint-type string. Revolute with any infinite limit → "continuous".
char const* ToUrdfJointTypeName(
    ArticulatedJointType type,
    std::optional<Real3> const& minLimit,
    std::optional<Real3> const& maxLimit) {
  switch (type) {
    case ArticulatedJointType::Free:
      return kJointTypeFloating;
    case ArticulatedJointType::Revolute: {
      bool const hasInfLimit =
          !minLimit || !IsFinite(*minLimit) || !maxLimit || !IsFinite(*maxLimit);
      return hasInfLimit ? kJointTypeContinuous : kJointTypeRevolute;
    }
    case ArticulatedJointType::Prismatic:
      return kJointTypePrismatic;
    case ArticulatedJointType::Hard:
    case ArticulatedJointType::Spherical:
    case ArticulatedJointType::Cycle:
    case ArticulatedJointType::Count:
      return kJointTypeFixed;
  }
  return kJointTypeFixed;
}

// Extract scalar limit from axis-scaled limit vector: limit = Dot(limitVec, Normalize(axis)).
double ExtractScalarLimit(std::optional<Real3> const& limitVec, Real3 const& axis) {
  if (limitVec.has_value()) {
    auto dLimitVec = StaticCast<NdArray<double, 3>>(*limitVec);
    auto dAxis = Normalize(StaticCast<NdArray<double, 3>>(axis));
    return Dot(dLimitVec, dAxis);
  }
  return std::numeric_limits<double>::infinity();
}

std::string MakeRelativeMeshPath(
    std::string_view absolutePath,
    std::filesystem::path const& urdfDir) {
  namespace fs = std::filesystem;
  if (absolutePath.empty()) {
    return {};
  }
  fs::path const meshPath(absolutePath);
  if (!meshPath.is_absolute()) {
    return std::string(absolutePath);
  }
  auto const rel = fs::relative(meshPath, urdfDir);
  return "./" + rel.generic_string();
}

void WriteInertial(tx::XMLElement* linkXml, BotLinkPrefab const& link) {
  tx::XMLDocument* doc = linkXml->GetDocument();
  double const mass = link.mass.value_or(0.0);
  Real3 const com = link.centerOfMass.value_or(Real3{});
  Real6 const moi = link.momentOfInertia.value_or(Real6{});

  tx::XMLElement* inertialXml = doc->NewElement("inertial");
  linkXml->InsertEndChild(inertialXml);

  tx::XMLElement* massXml = doc->NewElement("mass");
  massXml->SetAttribute("value", FormatDouble(mass).c_str());
  inertialXml->InsertEndChild(massXml);

  tx::XMLElement* originXml = doc->NewElement("origin");
  originXml->SetAttribute(
      "xyz",
      FormatXyz(
          static_cast<double>(com[0]), static_cast<double>(com[1]), static_cast<double>(com[2]))
          .c_str());
  originXml->SetAttribute("rpy", "0 0 0");
  inertialXml->InsertEndChild(originXml);

  tx::XMLElement* inertiaXml = doc->NewElement("inertia");
  inertiaXml->SetAttribute("ixx", FormatDouble(static_cast<double>(moi[0])).c_str());
  inertiaXml->SetAttribute("ixy", FormatDouble(static_cast<double>(moi[1])).c_str());
  inertiaXml->SetAttribute("ixz", FormatDouble(static_cast<double>(moi[2])).c_str());
  inertiaXml->SetAttribute("iyy", FormatDouble(static_cast<double>(moi[3])).c_str());
  inertiaXml->SetAttribute("iyz", FormatDouble(static_cast<double>(moi[4])).c_str());
  inertiaXml->SetAttribute("izz", FormatDouble(static_cast<double>(moi[5])).c_str());
  inertialXml->InsertEndChild(inertiaXml);
}

void WriteMeshElement(
    tx::XMLElement* linkXml,
    char const* tag,
    std::string_view name,
    UrdfMeshReference const& mesh,
    Real3 const& scale,
    Quaternion const& rotation,
    Real3 const& translation) {
  if (mesh.filename.empty()) {
    return;
  }
  tx::XMLDocument* doc = linkXml->GetDocument();

  tx::XMLElement* elem = doc->NewElement(tag);
  elem->SetAttribute("name", std::string(name).c_str());
  linkXml->InsertEndChild(elem);

  // Origin (baked transform).
  WriteOrigin(elem, translation, rotation);

  tx::XMLElement* geometryXml = doc->NewElement("geometry");
  elem->InsertEndChild(geometryXml);

  tx::XMLElement* meshXml = doc->NewElement("mesh");
  meshXml->SetAttribute("filename", mesh.filename.c_str());
  // Only emit scale if non-default.
  bool const defaultScale = (scale[0] == 1_r && scale[1] == 1_r && scale[2] == 1_r);
  if (!defaultScale) {
    meshXml->SetAttribute(
        "scale",
        FormatXyz(
            static_cast<double>(scale[0]),
            static_cast<double>(scale[1]),
            static_cast<double>(scale[2]))
            .c_str());
  }
  geometryXml->InsertEndChild(meshXml);

  if (mesh.rgba) {
    tx::XMLElement* materialXml = doc->NewElement("material");
    materialXml->SetAttribute("name", (std::string(name) + "_material").c_str());
    tx::XMLElement* colorXml = doc->NewElement("color");
    Real4 const& c = *mesh.rgba;
    colorXml->SetAttribute(
        "rgba",
        (FormatXyz(
             static_cast<double>(c[0]), static_cast<double>(c[1]), static_cast<double>(c[2])) +
         " " + FormatDouble(static_cast<double>(c[3])))
            .c_str());
    materialXml->InsertEndChild(colorXml);
    elem->InsertEndChild(materialXml);
  }
}

void ValidateBotForUrdf(BotPrefab const& bot, Error& error) {
  MOCHI_ERROR_RETURN(error);

  int const numLinks = static_cast<int>(bot.links.size());
  int const numJoints = static_cast<int>(bot.joints.size());
  MOCHI_ERROR_IF(numLinks == 0, error, "Cannot export empty bot to URDF");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      numJoints != numLinks, error, "Joint count must equal link count (including world_joint)");
  MOCHI_ERROR_RETURN(error);

  // Index 0 is the synthetic root joint (URDF injects a Free "world_joint" here on import) and is
  // skipped on export, since URDF's root link has no parent joint. Its name is irrelevant — what
  // matters is that link 0 really is the root, so the index-1..N joint/link pairing below is valid.
  MOCHI_ERROR_IF(
      bot.links[0].parentLink >= 0,
      error,
      "First link must be the root (no parent joint) for URDF export");
  MOCHI_ERROR_RETURN(error);

  // URDF has no spherical joint type — reject these bots up-front rather than silently lowering
  // them to something else. (Skip index 0, the injected world_joint.)
  bool const hasSphericalJoint =
      std::any_of(bot.joints.begin() + 1, bot.joints.end(), [](BotJointPrefab const& joint) {
        return joint.type == ArticulatedJointType::Spherical;
      });
  MOCHI_ERROR_IF(hasSphericalJoint, error, "URDF does not support spherical joints.");
}

// Generate URDF XML string from BotPrefab. When urdfDir is non-empty, mesh elements are included
// with paths resolved relative to urdfDir. When urdfDir is empty, mesh elements are omitted.
std::string
GenerateUrdfXml(BotPrefab const& bot, std::filesystem::path const& urdfDir, Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  tx::XMLDocument doc;
  tx::XMLElement* robotXml = doc.NewElement("robot");
  robotXml->SetAttribute("name", bot.name.c_str());
  doc.InsertEndChild(robotXml);

  UrdfBotWriter writer;
  if (!urdfDir.empty()) {
    writer.mesh = [&urdfDir](BotLinkPrefab const& link, bool visual, Error&) {
      return UrdfMeshReference{
          .filename = MakeRelativeMeshPath(visual ? link.renderModelFile : link.shapeFile, urdfDir),
          .rgba = {}};
    };
  }
  AppendBotToUrdf(robotXml, bot, writer, error);
  MOCHI_ERROR_RETURN(error, {});
  return PrintUrdf(doc);
}

} // namespace

void superdex::robotics::AppendBotToUrdf(
    tx::XMLElement* robotXml,
    BotPrefab const& bot,
    UrdfBotWriter const& writer,
    Error& error) {
  ValidateBotForUrdf(bot, error);
  MOCHI_ERROR_RETURN(error);

  auto const rename = [&writer](std::string_view name) {
    return writer.rename ? writer.rename(name) : std::string(name);
  };
  int const numLinks = static_cast<int>(bot.links.size());
  tx::XMLDocument* doc = robotXml->GetDocument();

  // Write links (all links are emitted).
  for (int i = 0; i < numLinks; ++i) {
    auto const& link = bot.links[i];
    std::string const linkName = rename(link.name);
    tx::XMLElement* linkXml = doc->NewElement("link");
    linkXml->SetAttribute("name", linkName.c_str());
    robotXml->InsertEndChild(linkXml);

    if (writer.mesh) {
      // Visual.
      if (!link.renderModelFile.empty()) {
        WriteMeshElement(
            linkXml,
            "visual",
            linkName,
            writer.mesh(link, true, error),
            link.renderModelScale,
            link.renderModelRotation,
            link.renderModelTranslation);
      }

      // Collision.
      if (!link.shapeFile.empty()) {
        WriteMeshElement(
            linkXml,
            "collision",
            linkName + "_collision",
            writer.mesh(link, false, error),
            link.shapeScale,
            link.shapeRotation,
            link.shapeTranslation);
      }
      MOCHI_ERROR_RETURN(error);
    }

    // Inertial: only emit when the source link has a mass. Links without <inertial> (frames, sensor
    // mounts, tool-center-point links) must round-trip as absent — emitting a zero-mass block here
    // would make them re-import with mass/com/inertia set, breaking fidelity. URDF has no density,
    // so a density-specified link gets no <inertial> either and re-imports at the default density.
    if (link.mass) {
      WriteInertial(linkXml, link);
    }
  }

  // Write joints (skip index 0 = world_joint).
  for (int i = 1; i < numLinks; ++i) {
    auto const& joint = bot.joints[i];
    auto const& link = bot.links[i];

    char const* const typeName = ToUrdfJointTypeName(joint.type, joint.minLimit, joint.maxLimit);

    // Find parent link name.
    std::string parentName;
    if (link.parentLink >= 0 && link.parentLink < numLinks) {
      parentName = rename(bot.links[link.parentLink].name);
    }

    tx::XMLElement* jointXml = doc->NewElement("joint");
    jointXml->SetAttribute("name", rename(joint.name).c_str());
    jointXml->SetAttribute("type", typeName);
    robotXml->InsertEndChild(jointXml);

    // URDF has no separate joint frame: a <joint> <origin> is the full parent-link -> child-link
    // transform and <axis> is expressed in the child link frame (urdfdom: "child link frame is the
    // same as the Joint frame"). Fold mochi's two-part chain -- parentLinkFromJoint (parent link ->
    // joint) composed with parentJointFromLink (joint -> child link) -- into that single origin,
    // and re-express the axis through the joint -> child-link rotation below. For URDF-imported
    // bots parentJointFromLink is identity, so this reduces to the raw parentLinkFromJoint / axis.
    TransformRT const xf = joint.parentLinkFromJoint * link.parentJointFromLink;
    WriteOrigin(jointXml, xf.GetTranslation(), xf.GetRotation());

    // Axis, re-expressed in the child link frame (see the origin note above). The joint's
    // axis-aligned limits are exported as scalars via ExtractScalarLimit, which is
    // frame-independent, so they need no adjustment here.
    Real3 const axis = link.parentJointFromLink.GetRotation().GetConjugate() * joint.axis;
    tx::XMLElement* axisXml = doc->NewElement("axis");
    axisXml->SetAttribute(
        "xyz",
        FormatXyz(
            static_cast<double>(axis[0]),
            static_cast<double>(axis[1]),
            static_cast<double>(axis[2]))
            .c_str());
    jointXml->InsertEndChild(axisXml);

    WriteJointLinks(jointXml, parentName, rename(link.name));

    // Dynamics.
    double const damping = joint.friction.viscous;
    double const friction = joint.friction.coulomb;
    tx::XMLElement* dynamicsXml = doc->NewElement("dynamics");
    dynamicsXml->SetAttribute("damping", FormatDouble(damping).c_str());
    dynamicsXml->SetAttribute("friction", FormatDouble(friction).c_str());
    jointXml->InsertEndChild(dynamicsXml);

    bool const isContinuous = typeName == std::string_view(kJointTypeContinuous);
    if (isContinuous || (joint.minLimit.has_value() && joint.maxLimit.has_value())) {
      // URDF requires the effort attribute; emit the effortLimit verbatim so it round-trips
      // (< 0 unbounded, 0 non-actuated, > 0 finite).
      auto const effort = static_cast<double>(joint.effortLimit);
      // Velocity is unmodeled: fixed joints, and every joint when there is no velocity callback,
      // get a placeholder.
      double const velocity = writer.velocity && typeName != std::string_view(kJointTypeFixed)
          ? writer.velocity(joint, error)
          : 1.0;
      MOCHI_ERROR_RETURN(error);
      tx::XMLElement* limitXml = doc->NewElement("limit");
      limitXml->SetAttribute("effort", FormatDouble(effort).c_str());
      limitXml->SetAttribute("velocity", FormatDouble(velocity).c_str());
      if (!isContinuous) {
        double const lower = ExtractScalarLimit(joint.minLimit, joint.axis);
        double const upper = ExtractScalarLimit(joint.maxLimit, joint.axis);
        limitXml->SetAttribute("lower", FormatDouble(lower).c_str());
        limitXml->SetAttribute("upper", FormatDouble(upper).c_str());
      }
      jointXml->InsertEndChild(limitXml);
    }
  }
}

void superdex::robotics::AppendUrdfLink(tx::XMLElement* robotXml, std::string_view name) {
  tx::XMLElement* linkXml = robotXml->GetDocument()->NewElement("link");
  linkXml->SetAttribute("name", std::string(name).c_str());
  robotXml->InsertEndChild(linkXml);
}

void superdex::robotics::AppendUrdfFixedJoint(
    tx::XMLElement* robotXml,
    std::string_view name,
    std::string_view parent,
    std::string_view child,
    TransformRT const& parentFromChild) {
  tx::XMLDocument* doc = robotXml->GetDocument();
  tx::XMLElement* jointXml = doc->NewElement("joint");
  jointXml->SetAttribute("name", std::string(name).c_str());
  jointXml->SetAttribute("type", kJointTypeFixed);
  robotXml->InsertEndChild(jointXml);
  WriteOrigin(jointXml, parentFromChild.GetTranslation(), parentFromChild.GetRotation());
  WriteJointLinks(jointXml, parent, child);
}

std::string superdex::robotics::PrintUrdf(tx::XMLDocument const& doc) {
  tx::XMLPrinter printer;
  doc.Print(&printer);
  return {printer.CStr(), printer.CStrSize() > 0 ? printer.CStrSize() - 1 : 0};
}

std::string superdex::robotics::ExportBotPrefabToUrdfXml(BotPrefab const& botPrefab, Error& error) {
  return GenerateUrdfXml(botPrefab, {}, error);
}

void superdex::robotics::ExportBotPrefabToUrdfFile(
    BotPrefab const& botPrefab,
    std::string_view path,
    Error& error) {
  MOCHI_ERROR_RETURN(error);

  namespace fs = std::filesystem;
  fs::path const urdfPath(path);
  fs::path const urdfDir = urdfPath.parent_path();

  std::string const xmlStr = GenerateUrdfXml(botPrefab, urdfDir, error);
  MOCHI_ERROR_RETURN(error);

  // Write to file. Creates parent directories as needed.
  mochi::WriteFile(urdfPath, xmlStr, error);
}

namespace {

constexpr std::string_view kWorldLink = "world";

/* "link_NN_name" / "joint_NN_name" -> "name". */
std::string StripIndexPrefix(std::string_view name) {
  for (std::string_view const prefix : {std::string_view("link_"), std::string_view("joint_")}) {
    if (!name.starts_with(prefix)) {
      continue;
    }
    size_t i = prefix.size();
    while (i < name.size() && std::isdigit(static_cast<unsigned char>(name[i]))) {
      ++i;
    }
    if (i > prefix.size() && i + 1 < name.size() && name[i] == '_') {
      return std::string(name.substr(i + 1));
    }
  }
  return std::string(name);
}

bool IsCamera(std::string_view type) {
  // "fixed_camera" and "wrist_camera" are legacy names for SENSOR_CAMERA.
  return type == CameraSensor::TypeName() || type == "fixed_camera" || type == "wrist_camera";
}

class SceneUrdfExporter {
 public:
  SceneUrdfExporter(
      BotScenePrefab const& scene,
      std::string_view rootPath,
      SceneUrdfOptions const& options,
      SceneUrdfMeshWriter const& mesh)
      : _scene(scene), _rootPath(rootPath), _options(options), _mesh(mesh) {}

  std::string Export(Error& error) {
    MOCHI_ERROR_RETURN(error, {});
    std::string robotName(_options.robotName);
    if (robotName.empty()) {
      robotName = _scene.metadata.name.empty() ? "scene" : std::string(_scene.metadata.name);
    }
    _robotXml = _doc.NewElement("robot");
    _robotXml->SetAttribute("name", robotName.c_str());
    _doc.InsertEndChild(_robotXml);
    AppendUrdfLink(_robotXml, kWorldLink);

    if (!_scene.scene.baseScene.empty()) {
      AddBaseScene(error);
    }
    for (auto const& entry : _scene.bots) {
      AddBotEntry(entry, error);
    }
    MOCHI_ERROR_RETURN(error, {});
    int index = 0;
    for (auto const& sensor : _scene.sensors) {
      _sensors.push_back(
          {std::string(sensor.name),
           std::string(sensor.type),
           "",
           std::string(kWorldLink),
           index++,
           sensor.parentFromSensor});
    }
    AddSensors();
    MakeNamesUnique(error);
    MOCHI_ERROR_RETURN(error, {});
    return PrintUrdf(_doc);
  }

 private:
  /* A sensor and where it attaches, named once every sensor in the scene is known. */
  struct Sensor {
    std::string name;
    std::string type;
    /* The prefix of its bot's links, or empty for a scene sensor. */
    std::string prefix;
    /* URDF link it attaches to. */
    std::string parent;
    /* Its index among the sensors on @ref parent. */
    int index = 0;
    TransformRT parentFromSensor;
  };

  std::string Name(std::string_view name) const {
    return _options.stripIndexPrefixes ? StripIndexPrefix(name) : std::string(name);
  }

  void AddBotEntry(BotEntry const& entry, Error& error) {
    MOCHI_ERROR_RETURN(error);
    std::string prefix = std::string(entry.name) + "_";
    bool inSlot = false;
    for (auto const& slot : _scene.slots) {
      if (Contains(slot.controlledBots, entry.name)) {
        // A URDF holds every bot it lists, so it can hold only one of a slot's candidates.
        MOCHI_ERROR_IF(
            slot.controlledBots.size() > 1,
            error,
            "A slot with several candidate bots cannot be exported to URDF");
        // The slot names the bot's links, so a bot in two slots would need two names.
        MOCHI_ERROR_IF(inSlot, error, "A bot in several slots cannot be exported to URDF");
        inSlot = true;
        prefix = std::string(slot.name) + "_" + std::string(_options.instanceName) + "_";
      }
    }
    MOCHI_ERROR_RETURN(error);
    BotPrefab const bot = LoadBotPrefabFromFile(std::string_view(entry.path), error);
    MOCHI_ERROR_RETURN(error);
    AddBot(bot, prefix, entry.parentFromBot, error);
  }

  /* Write @p bot with names prefixed by @p prefix, fixed to `world` at @p worldFromBot, with its
   * link aliases and sensors. */
  void AddBot(
      BotPrefab const& bot,
      std::string const& prefix,
      TransformRT const& worldFromBot,
      Error& error) {
    auto const urdfName = [this, &prefix](std::string_view name) { return prefix + Name(name); };
    UrdfBotWriter writer;
    writer.rename = urdfName;
    if (_mesh) {
      writer.mesh = [this, &urdfName](BotLinkPrefab const& link, bool visual, Error& meshError) {
        SceneUrdfMesh const mesh = _mesh(link, urdfName(link.name), visual, meshError);
        return UrdfMeshReference{.filename = std::string(mesh.filename), .rgba = mesh.rgba};
      };
    }
    writer.velocity = [this](BotJointPrefab const& joint, Error& velocityError) {
      return Velocity(joint, velocityError);
    };
    AppendBotToUrdf(_robotXml, bot, writer, error);
    MOCHI_ERROR_RETURN(error);

    std::string const root = urdfName(bot.links[0].name);
    AppendUrdfFixedJoint(_robotXml, root + "_placement", kWorldLink, root, worldFromBot);
    for (auto const& link : bot.links) {
      for (auto const& alias : _options.linkAliases) {
        if (Name(link.name) == std::string_view(alias.link)) {
          std::string const aliasName = prefix + std::string(alias.alias);
          AppendUrdfLink(_robotXml, aliasName);
          AppendUrdfFixedJoint(_robotXml, aliasName + "_alias", urdfName(link.name), aliasName, {});
        }
      }
      int index = 0;
      for (auto const& sensor : link.sensors) {
        _sensors.push_back(
            {std::string(sensor.name),
             std::string(sensor.type),
             prefix,
             urdfName(link.name),
             index++,
             sensor.parentFromSensor});
      }
    }
  }

  double Velocity(BotJointPrefab const& joint, Error& error) const {
    std::string const name = Name(joint.name);
    for (auto const& limit : _options.jointVelocityLimits) {
      if (std::string_view(limit.joint) == name) {
        return static_cast<double>(limit.velocity);
      }
    }
    if (_options.requireVelocityLimits) {
      MOCHI_LOG_ERROR("No velocity limit for joint %s", name.c_str());
      MOCHI_ERROR_SET(error, "Joint has no entry in SceneUrdfOptions::jointVelocityLimits");
    }
    return 1.0;
  }

  void AddBaseScene(Error& error) {
    prefab::ScenePrefab base =
        prefab::ShallowLoadFromFile(std::string_view(_scene.scene.baseScene), error);
    MOCHI_ERROR_RETURN(error);
    prefab::LoadNestedPrefabs(base, _rootPath, error);
    AddPrefab(base, "", TransformRT::Identity(), error);
  }

  void AddPrefab(
      prefab::ScenePrefab const& prefab,
      std::string const& prefix,
      TransformRT const& worldFromPrefab,
      Error& error) {
    MOCHI_ERROR_RETURN(error);
    for (auto const& nested : prefab.prefabs) {
      MOCHI_ERROR_IF(nested.scale != 1_r, error, "Scaled prefabs cannot be exported to URDF");
      if (nested.prefab) {
        AddPrefab(
            *nested.prefab,
            prefix + std::string(nested.name) + "_",
            worldFromPrefab * TransformRT(nested.rotation, nested.translation),
            error);
      }
    }
    std::string_view const prefabFile =
        prefab.sourceFilePath ? std::string_view(*prefab.sourceFilePath) : std::string_view();
    auto const fullPath = [&](DynamicString const& file) {
      return file.empty() ? file : prefab::GetPrefabFullPath(file, _rootPath, prefabFile);
    };

    for (auto const& actor : prefab.actors.articulated) {
      MOCHI_ERROR_IF(actor.scale != 1_r, error, "Scaled actors cannot be exported to URDF");
      MOCHI_ERROR_RETURN(error);
      BotPrefab bot;
      for (auto const& link : actor.links) {
        bot.links.emplace_back();
        BotLinkPrefab& botLink = bot.links.back();
        static_cast<prefab::ArticulatedLinkPrefab&>(botLink) = link;
        botLink.shapeFile = fullPath(link.shapeFile);
        botLink.renderModelFile = fullPath(link.renderModelFile);
      }
      for (auto const& joint : actor.joints) {
        bot.joints.emplace_back();
        static_cast<prefab::ArticulatedJointPrefab&>(bot.joints.back()) = joint;
      }
      AddBot(bot, prefix, worldFromPrefab * TransformRT(actor.rotation, actor.translation), error);
    }

    for (auto const& actor : prefab.actors.rigid) {
      if (!actor.isStatic) {
        continue;
      }
      MOCHI_ERROR_IF(
          (actor.scale != Real3{1_r, 1_r, 1_r}), error, "Scaled actors cannot be exported to URDF");
      MOCHI_ERROR_RETURN(error);
      BotPrefab bot;
      bot.links.emplace_back();
      BotLinkPrefab& link = bot.links.back();
      link.name = actor.name;
      link.shapeFile = fullPath(actor.shapeFile);
      link.shapeRotation = actor.shapeRotation;
      link.shapeTranslation = actor.shapeTranslation;
      link.renderModelFile = fullPath(actor.renderModelFile);
      link.renderModelScale = actor.renderModelScale;
      link.renderModelRotation = actor.renderModelRotation;
      link.renderModelTranslation = actor.renderModelTranslation;
      // The root joint, which a URDF does not write.
      bot.joints.emplace_back();
      AddBot(bot, prefix, worldFromPrefab * TransformRT(actor.rotation, actor.translation), error);
    }
  }

  /* Name every sensor and write its frames. A name only one sensor has is kept. */
  void AddSensors() {
    std::unordered_map<std::string, int> uses;
    for (auto const& sensor : _sensors) {
      ++uses[sensor.name];
    }
    for (auto const& sensor : _sensors) {
      std::string name;
      if (sensor.name.empty()) {
        name.append(sensor.parent).append("_sensor_").append(std::to_string(sensor.index));
      } else {
        if (uses[sensor.name] > 1) {
          name = sensor.prefix;
        }
        name += sensor.name;
      }
      AddSensor(name, sensor.type, sensor.parent, sensor.parentFromSensor);
    }
  }

  void AddSensor(
      std::string const& name,
      std::string_view type,
      std::string_view parent,
      TransformRT const& parentFromSensor) {
    std::string const frame = name + "_frame";
    AppendUrdfLink(_robotXml, frame);
    if (!IsCamera(type)) {
      AppendUrdfFixedJoint(_robotXml, name + "_joint", parent, frame, parentFromSensor);
      return;
    }
    // A camera's parentFromSensor is its optical frame (Z forward, X right, Y down). Its frame
    // follows ROS REP-103 (X forward, Z up).
    TransformRT const frameFromOptical(Quaternion(-0.5_r, 0.5_r, -0.5_r, 0.5_r));
    std::string const optical = name + "_optical_frame";
    AppendUrdfLink(_robotXml, optical);
    AppendUrdfFixedJoint(
        _robotXml, name + "_joint", parent, frame, parentFromSensor * Invert(frameFromOptical));
    AppendUrdfFixedJoint(_robotXml, name + "_optical_joint", frame, optical, frameFromOptical);
  }

  /* URDF link names must be unique, and so must joint names. SuperDex also needs joint names to
   * differ from link names, so a joint named like a link gets a "_joint" suffix. */
  void MakeNamesUnique(Error& error) {
    auto const duplicate = [&error](char const* name) {
      MOCHI_LOG_ERROR("Scene URDF has two links or two joints named %s", name);
      MOCHI_ERROR_SET(error, "Scene URDF has duplicate link or joint names");
    };
    std::unordered_set<std::string> links;
    for (auto const* link = _robotXml->FirstChildElement("link"); link != nullptr;
         link = link->NextSiblingElement("link")) {
      if (!links.insert(link->Attribute("name")).second) {
        duplicate(link->Attribute("name"));
        return;
      }
    }
    std::unordered_set<std::string> joints;
    for (auto* joint = _robotXml->FirstChildElement("joint"); joint != nullptr;
         joint = joint->NextSiblingElement("joint")) {
      std::string name = joint->Attribute("name");
      if (links.contains(name)) {
        name += "_joint";
        joint->SetAttribute("name", name.c_str());
      }
      if (links.contains(name) || !joints.insert(name).second) {
        duplicate(name.c_str());
        return;
      }
    }
  }

  BotScenePrefab const& _scene;
  std::string _rootPath;
  SceneUrdfOptions const& _options;
  SceneUrdfMeshWriter const& _mesh;
  std::vector<Sensor> _sensors;
  tx::XMLDocument _doc;
  tx::XMLElement* _robotXml = nullptr;
};

} // namespace

DynamicString superdex::robotics::SaveBotSceneToUrdfString(
    BotScenePrefab const& scene,
    std::string_view rootPath,
    SceneUrdfOptions const& options,
    SceneUrdfMeshWriter const& mesh,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  std::string const xml = SceneUrdfExporter(scene, rootPath, options, mesh).Export(error);
  MOCHI_ERROR_RETURN(error, {});
  return DynamicString(xml.c_str(), xml.size());
}

DynamicString superdex::robotics::SaveBotSceneToUrdfString(
    BotScenePrefab const& scene,
    std::string_view rootPath,
    SceneUrdfOptions const& options,
    Error& error) {
  return SaveBotSceneToUrdfString(scene, rootPath, options, {}, error);
}
