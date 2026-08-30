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

#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/quaternion_utils.h>

#include <tinyxml2.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

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
    DynamicString const& path,
    Real3 const& scale,
    Quaternion const& rotation,
    Real3 const& translation,
    std::filesystem::path const& urdfDir) {
  std::string const relPath = MakeRelativeMeshPath(path.c_str(), urdfDir);
  if (relPath.empty()) {
    return;
  }
  tx::XMLDocument* doc = linkXml->GetDocument();

  tx::XMLElement* elem = doc->NewElement(tag);
  elem->SetAttribute("name", std::string(name).c_str());
  linkXml->InsertEndChild(elem);

  // Origin (baked transform).
  tx::XMLElement* originXml = doc->NewElement("origin");
  originXml->SetAttribute(
      "xyz",
      FormatXyz(
          static_cast<double>(translation[0]),
          static_cast<double>(translation[1]),
          static_cast<double>(translation[2]))
          .c_str());
  originXml->SetAttribute("rpy", FormatRpy(rotation).c_str());
  elem->InsertEndChild(originXml);

  tx::XMLElement* geometryXml = doc->NewElement("geometry");
  elem->InsertEndChild(geometryXml);

  tx::XMLElement* meshXml = doc->NewElement("mesh");
  meshXml->SetAttribute("filename", relPath.c_str());
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
}

// Generate URDF XML string from BotPrefab. When urdfDir is non-empty, mesh elements are included
// with paths resolved relative to urdfDir. When urdfDir is empty, mesh elements are omitted.
std::string
GenerateUrdfXml(BotPrefab const& bot, std::filesystem::path const& urdfDir, Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  int const numLinks = static_cast<int>(bot.links.size());
  int const numJoints = static_cast<int>(bot.joints.size());
  MOCHI_ERROR_IF(numLinks == 0, error, "Cannot export empty bot to URDF");
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(
      numJoints != numLinks, error, "Joint count must equal link count (including world_joint)");
  MOCHI_ERROR_RETURN(error, {});

  // Index 0 is the synthetic root joint (URDF injects a Free "world_joint" here on import) and is
  // skipped on export, since URDF's root link has no parent joint. Its name is irrelevant — what
  // matters is that link 0 really is the root, so the index-1..N joint/link pairing below is valid.
  MOCHI_ERROR_IF(
      bot.links[0].parentLink >= 0,
      error,
      "First link must be the root (no parent joint) for URDF export");
  MOCHI_ERROR_RETURN(error, {});

  // URDF has no spherical joint type — reject these bots up-front rather than silently lowering
  // them to something else. (Skip index 0, the injected world_joint.)
  bool const hasSphericalJoint =
      std::any_of(bot.joints.begin() + 1, bot.joints.end(), [](BotJointPrefab const& joint) {
        return joint.type == ArticulatedJointType::Spherical;
      });
  MOCHI_ERROR_IF(hasSphericalJoint, error, "URDF does not support spherical joints.");
  MOCHI_ERROR_RETURN(error, {});

  bool const emitMeshes = !urdfDir.empty();

  tx::XMLDocument doc;
  tx::XMLElement* robotXml = doc.NewElement("robot");
  robotXml->SetAttribute("name", bot.name.c_str());
  doc.InsertEndChild(robotXml);

  // Write links (all links are emitted).
  for (int i = 0; i < numLinks; ++i) {
    auto const& link = bot.links[i];
    tx::XMLElement* linkXml = doc.NewElement("link");
    linkXml->SetAttribute("name", link.name.c_str());
    robotXml->InsertEndChild(linkXml);

    if (emitMeshes) {
      // Visual.
      if (!link.renderModelFile.empty()) {
        WriteMeshElement(
            linkXml,
            "visual",
            link.name,
            link.renderModelFile,
            link.renderModelScale,
            link.renderModelRotation,
            link.renderModelTranslation,
            urdfDir);
      }

      // Collision.
      if (!link.shapeFile.empty()) {
        std::string const collisionName = std::string(link.name) + "_collision";
        WriteMeshElement(
            linkXml,
            "collision",
            collisionName,
            link.shapeFile,
            link.shapeScale,
            link.shapeRotation,
            link.shapeTranslation,
            urdfDir);
      }
    }

    // Inertial: only emit when the source link actually carried one. Links without <inertial>
    // (frames, sensor mounts, tool-center-point links) must round-trip as absent — emitting a
    // zero-mass block here would make them re-import with mass/com/inertia set, breaking fidelity.
    if (link.mass || link.centerOfMass || link.momentOfInertia) {
      WriteInertial(linkXml, link);
    }
  }

  // Write joints (skip index 0 = world_joint).
  for (int i = 1; i < numJoints; ++i) {
    auto const& joint = bot.joints[i];
    auto const& link = bot.links[i];

    char const* const typeName = ToUrdfJointTypeName(joint.type, joint.minLimit, joint.maxLimit);

    // Find parent link name.
    std::string_view parentName;
    if (link.parentLink >= 0 && link.parentLink < numLinks) {
      parentName = bot.links[link.parentLink].name.c_str();
    }

    tx::XMLElement* jointXml = doc.NewElement("joint");
    jointXml->SetAttribute("name", joint.name.c_str());
    jointXml->SetAttribute("type", typeName);
    robotXml->InsertEndChild(jointXml);

    // URDF has no separate joint frame: a <joint> <origin> is the full parent-link -> child-link
    // transform and <axis> is expressed in the child link frame (urdfdom: "child link frame is the
    // same as the Joint frame"). Fold mochi's two-part chain -- parentLinkFromJoint (parent link ->
    // joint) composed with parentJointFromLink (joint -> child link) -- into that single origin,
    // and re-express the axis through the joint -> child-link rotation below. For URDF-imported
    // bots parentJointFromLink is identity, so this reduces to the raw parentLinkFromJoint / axis.
    TransformRT const xf = joint.parentLinkFromJoint * link.parentJointFromLink;
    Real3 const& t = xf.GetTranslation();
    tx::XMLElement* originXml = doc.NewElement("origin");
    originXml->SetAttribute(
        "xyz",
        FormatXyz(static_cast<double>(t[0]), static_cast<double>(t[1]), static_cast<double>(t[2]))
            .c_str());
    originXml->SetAttribute("rpy", FormatRpy(xf.GetRotation()).c_str());
    jointXml->InsertEndChild(originXml);

    // Axis, re-expressed in the child link frame (see the origin note above). The joint's
    // axis-aligned limits are exported as scalars via ExtractScalarLimit, which is
    // frame-independent, so they need no adjustment here.
    Real3 const axis = link.parentJointFromLink.GetRotation().GetConjugate() * joint.axis;
    tx::XMLElement* axisXml = doc.NewElement("axis");
    axisXml->SetAttribute(
        "xyz",
        FormatXyz(
            static_cast<double>(axis[0]),
            static_cast<double>(axis[1]),
            static_cast<double>(axis[2]))
            .c_str());
    jointXml->InsertEndChild(axisXml);

    // Parent / child.
    tx::XMLElement* parentXml = doc.NewElement("parent");
    parentXml->SetAttribute("link", std::string(parentName).c_str());
    jointXml->InsertEndChild(parentXml);

    tx::XMLElement* childXml = doc.NewElement("child");
    childXml->SetAttribute("link", link.name.c_str());
    jointXml->InsertEndChild(childXml);

    // Dynamics.
    double const damping = joint.friction.viscous;
    double const friction = joint.friction.coulomb;
    tx::XMLElement* dynamicsXml = doc.NewElement("dynamics");
    dynamicsXml->SetAttribute("damping", FormatDouble(damping).c_str());
    dynamicsXml->SetAttribute("friction", FormatDouble(friction).c_str());
    jointXml->InsertEndChild(dynamicsXml);

    // Limits (only emit when both limits are set and the joint is not continuous).
    if (typeName != std::string_view(kJointTypeContinuous) && joint.minLimit.has_value() &&
        joint.maxLimit.has_value()) {
      double const lower = ExtractScalarLimit(joint.minLimit, joint.axis);
      double const upper = ExtractScalarLimit(joint.maxLimit, joint.axis);
      // URDF requires the effort attribute; emit the effortLimit verbatim so it round-trips
      // (< 0 unbounded, 0 non-actuated, > 0 finite). Velocity is unmodeled, so emit a placeholder.
      auto const effort = static_cast<double>(joint.effortLimit);
      tx::XMLElement* limitXml = doc.NewElement("limit");
      limitXml->SetAttribute("effort", FormatDouble(effort).c_str());
      limitXml->SetAttribute("velocity", "1");
      limitXml->SetAttribute("lower", FormatDouble(lower).c_str());
      limitXml->SetAttribute("upper", FormatDouble(upper).c_str());
      jointXml->InsertEndChild(limitXml);
    }
  }

  tx::XMLPrinter printer;
  doc.Print(&printer);
  return {printer.CStr(), printer.CStrSize() > 0 ? printer.CStrSize() - 1 : 0};
}

} // namespace

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
