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

#include "rendering/bot_visualization.h"
#include "app/app.h"
#include "rendering/scene_stage.h"

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/decomposition_utils.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/vmatrix.h>

#include <mochi_renderer/type_conversions.h>

#include <array>
#include <cmath>
#include <optional>

using namespace mochi_renderer;

namespace superdex::studio {

// The intended DebugDraw here is mochi_renderer's. Disambiguate it from mochi::DebugDraw,
// which is now visible in `superdex` via superdex_physics.h's `using namespace mochi;`.
using mochi_renderer::DebugDraw;

namespace {

constexpr float defaultFrameScale = 0.05f;

/// Computes link scale from AABB if the link has a visual mesh, otherwise returns default.
float ComputeLinkScale(SceneStage const& stage, int linkIndex) {
  auto const& actors = stage.GetActors();
  if (linkIndex >= 0 && linkIndex < stage.GetNumActors()) {
    SceneObject* linkObject = actors[linkIndex].sceneObject;
    if (linkObject) {
      filament::Box aabb = linkObject->GetAABB();
      return std::max(0.02f, length(aabb.halfExtent) * 0.3f);
    }
  }
  return defaultFrameScale;
}

/// The equivalent solid box of a moment of inertia tensor, in the link's local frame.
struct InertiaBox {
  /// Half extents along the box's own axes.
  mochi::Real3 halfExtents;
  /// Principal axes of the tensor, as unit-length columns.
  mochi::Matrix3x3r principalAxes;
};

/// Converts a moment of inertia tensor to the equivalent box of uniform density.
/// Diagonalizing the tensor yields its principal moments l0, l1, l2 and the axes they act about.
/// A solid box with extents (w, h, d) along those axes satisfies:
/// l0 = (1/12) * m * (h² + d²)
/// l1 = (1/12) * m * (w² + d²)
/// l2 = (1/12) * m * (w² + h²)
/// Summing gives l0 + l1 + l2 = (1/6) * m * (w² + h² + d²), so each extent follows from
/// w² = (6/m) * (l1 + l2 - l0), i.e. (6/m) * (sum of moments - 2 * l0).
InertiaBox
InertiaToBox(mochi::Real6 const& momentOfInertia, mochi::real mass, mochi::real minSize = 0.001_r) {
  InertiaBox box{{minSize, minSize, minSize}, mochi::Eye<3>()};
  if (mass <= 0_r) {
    return box;
  }

  // momentOfInertia holds the upper triangle [Ixx, Ixy, Ixz, Iyy, Iyz, Izz] of the symmetric
  // tensor.
  mochi::Matrix3x3r const inertia{
      mochi::Real3{momentOfInertia[0], momentOfInertia[1], momentOfInertia[2]},
      mochi::Real3{momentOfInertia[1], momentOfInertia[3], momentOfInertia[4]},
      mochi::Real3{momentOfInertia[2], momentOfInertia[4], momentOfInertia[5]}};

  mochi::Real3 principalMoments;
  mochi::Matrix3x3r principalAxes;
  mochi::AnalyticalEigendecompSym(inertia, principalMoments, &principalAxes);
  if (!mochi::IsFinite(principalMoments) || !mochi::IsFinite(principalAxes)) {
    return box;
  }

  mochi::real const factor = 6_r / mass;
  mochi::real const momentSum = mochi::Sum(principalMoments);
  for (int i = 0; i < 3; ++i) {
    // Clamp to avoid negative values from numerical issues or non-physical inertias.
    mochi::real const sizeSq = factor * (momentSum - 2_r * principalMoments[i]);
    box.halfExtents[i] = sizeSq > 0_r ? std::sqrt(sizeSq) * 0.5_r : minSize;
  }
  box.principalAxes = principalAxes;
  return box;
}

} // namespace

namespace {

/* @brief Computes world position of a waypoint from its link's local frame.
 * Transforms the localPosition (in link local frame — same frame in which link mesh
 * and geometry are authored) to world/render space.
 * @return World position, or std::nullopt if linkIndex is out of range. */
std::optional<filament::math::float3> ComputeWaypointWorldPosition(
    SceneStage const& stage,
    int linkIndex,
    mochi::Real3 const& localPosition,
    mochi::CoordinateSpaceConverter const& spaceConverter) {
  if (linkIndex < 0 || linkIndex >= stage.GetNumActors()) {
    return std::nullopt;
  }

  auto const& actors = stage.GetActors();
  mochi::TransformRT const& worldTransform = actors[linkIndex].worldTransform;

  // Convert link transform to render space
  auto const rtRender = spaceConverter.TransformToOutput(worldTransform);

  // Build world transform matrix for point transformations
  auto const worldTransformMat = ToFilament<float>(ToNdArray(ToVMatrix4x4(rtRender)));

  // Transform local position from link local frame to world space
  filament::math::float3 localRender =
      ToFilament<float>(spaceConverter.TranslationToOutput(localPosition));

  return (worldTransformMat * filament::math::float4{localRender, 1.0f}).xyz;
}

/* @brief Draws a transmission segment as a solid cylinder between two points. */
void DrawTransmissionSegment(
    DebugDraw* debugDraw,
    filament::math::float3 const& startPos,
    filament::math::float3 const& endPos,
    filament::math::float4 const& color,
    float radius = 0.0005f) {
  filament::math::float3 direction = endPos - startPos;
  debugDraw->DrawSolidCylinder(startPos, direction, radius, color, 8, true, true);
}

} // namespace

void DrawLinkVisualization(
    DebugDraw* debugDraw,
    SceneStage const& stage,
    superdex::robotics::BotPrefab const& botPrefab,
    int linkIndex,
    mochi::CoordinateSpaceConverter const& spaceConverter,
    BotVisualizationFlags const& flags,
    LinkVisualizationSettings const& settings) {
  if (!debugDraw) {
    return;
  }

  bool hasAnythingToDraw =
      flags.showInertiaBox || flags.showCenterOfMass || flags.showLocalTransform;
  if (!hasAnythingToDraw) {
    return;
  }

  if (linkIndex < 0 || linkIndex >= static_cast<int>(botPrefab.links.size())) {
    return;
  }
  auto const& actors = stage.GetActors();
  if (linkIndex >= stage.GetNumActors()) {
    return;
  }

  auto const& link = botPrefab.links[linkIndex];

  mochi::TransformRT const& worldTransform = actors[linkIndex].worldTransform;

  // Compute visualization scale based on AABB if available
  float linkScale = ComputeLinkScale(stage, linkIndex);
  float visualScale = settings.linkVisualizationScale;
  float scale = (settings.scaleToLinkSize ? linkScale : defaultFrameScale) * visualScale;

  // Convert link transform to render space
  auto const rtRender = spaceConverter.TransformToOutput(worldTransform);
  auto const linkPosWorld = ToFilament<float>(rtRender.GetTranslation());
  auto const linkRotWorld = filament::math::mat3f(ToFilament<float>(rtRender.GetRotation()));

  // Build world transform matrix for point transformations
  auto const worldTransformMat = ToFilament<float>(ToNdArray(ToVMatrix4x4(rtRender)));

  // Draw Center of Mass
  if (flags.showCenterOfMass && link.centerOfMass.has_value()) {
    filament::math::float3 comLocalRender =
        ToFilament<float>(spaceConverter.TranslationToOutput(*link.centerOfMass));

    // Transform to world space
    filament::math::float3 comWorld =
        (worldTransformMat * filament::math::float4{comLocalRender, 1.0f}).xyz;

    float comRadius = scale * 0.2f;
    debugDraw->DrawSolidSphere(comWorld, comRadius, ToFilament(settings.comColor), 12, 16, true);
  }

  // Draw Inertia Box
  if (flags.showInertiaBox && link.momentOfInertia.has_value() && link.mass.has_value()) {
    InertiaBox const inertiaBox = InertiaToBox(*link.momentOfInertia, *link.mass);

    // Apply space conversion scaling
    auto const extentScale = static_cast<float>(std::abs(spaceConverter.GetScale()));

    // Center at CoM if available, otherwise at link origin
    filament::math::float3 boxCenter = linkPosWorld;
    if (link.centerOfMass.has_value()) {
      filament::math::float3 comLocalRender =
          ToFilament<float>(spaceConverter.TranslationToOutput(*link.centerOfMass));
      boxCenter = (worldTransformMat * filament::math::float4{comLocalRender, 1.0f}).xyz;
    }

    // The box is oriented along the tensor's principal axes, which are expressed in the link's
    // local (robot space) frame. Convert each to render space, then to world by the link rotation.
    std::array<filament::math::float3, 3> boxAxes{};
    for (int i = 0; i < 3; ++i) {
      mochi::Real3 const axisRobot{
          inertiaBox.principalAxes[0][i],
          inertiaBox.principalAxes[1][i],
          inertiaBox.principalAxes[2][i]};
      auto const axisRender = ToFilament<float>(spaceConverter.DirectionToOutput(axisRobot));
      boxAxes[i] =
          linkRotWorld * axisRender * (static_cast<float>(inertiaBox.halfExtents[i]) * extentScale);
    }

    debugDraw->DrawSolidOrientedBox(
        boxCenter, boxAxes[0], boxAxes[1], boxAxes[2], ToFilament(settings.inertiaColor), true);
  }

  // Draw Local Transform Axes (in robot space) using cylinders
  if (flags.showLocalTransform) {
    float axisLen = scale;
    float axisRadius = scale * 0.08f;

    // Convert robot space axes to render space, then transform by link's world rotation
    // This shows the robot's local frame, not the render space frame
    auto const robotX =
        ToFilament<float>(spaceConverter.DirectionToOutput(mochi::Float3{1.0f, 0.0f, 0.0f}));
    auto const robotY =
        ToFilament<float>(spaceConverter.DirectionToOutput(mochi::Float3{0.0f, 1.0f, 0.0f}));
    auto const robotZ =
        ToFilament<float>(spaceConverter.DirectionToOutput(mochi::Float3{0.0f, 0.0f, 1.0f}));

    // X axis (red)
    filament::math::float3 xAxis = linkRotWorld * robotX;
    debugDraw->DrawSolidCylinder(
        linkPosWorld, xAxis * axisLen, axisRadius, ToFilament(settings.xAxisColor), 8, true, true);

    // Y axis (green)
    filament::math::float3 yAxis = linkRotWorld * robotY;
    debugDraw->DrawSolidCylinder(
        linkPosWorld, yAxis * axisLen, axisRadius, ToFilament(settings.yAxisColor), 8, true, true);

    // Z axis (blue)
    filament::math::float3 zAxis = linkRotWorld * robotZ;
    debugDraw->DrawSolidCylinder(
        linkPosWorld, zAxis * axisLen, axisRadius, ToFilament(settings.zAxisColor), 8, true, true);
  }
}

void DrawAllLinkVisualizations(
    DebugDraw* debugDraw,
    SceneStage const& stage,
    superdex::robotics::BotPrefab const& botPrefab,
    mochi::CoordinateSpaceConverter const& spaceConverter,
    BotVisualizationFlags const& flags,
    LinkVisualizationSettings const& settings) {
  bool hasAnythingToDraw =
      flags.showInertiaBox || flags.showCenterOfMass || flags.showLocalTransform;
  if (!hasAnythingToDraw || !debugDraw) {
    return;
  }
  for (int i = 0; i < static_cast<int>(botPrefab.links.size()); ++i) {
    DrawLinkVisualization(debugDraw, stage, botPrefab, i, spaceConverter, flags, settings);
  }
}

namespace {

struct JointRenderFrameInfo {
  filament::math::float3 position;
  filament::math::float3 axis;
  filament::math::float3 arcAxis;
  filament::math::mat3f rotation;
  filament::math::float3 referenceDir;
  float scale;
};

void LinkTransformToOutputSpace(
    mochi::TransformRT const& linkToRootTransform,
    mochi::CoordinateSpaceConverter const& spaceConverter,
    filament::math::float3& outPos,
    filament::math::mat3f& outRot) {
  auto const& t = linkToRootTransform.GetTranslation();
  auto const& r = linkToRootTransform.GetRotation();

  outPos = ToFilament<float>(spaceConverter.TranslationToOutput(t));
  outRot = filament::math::mat3f(ToFilament<float>(spaceConverter.RotationToOutput(r)));
}

JointRenderFrameInfo ComputeJointRenderFrame(
    SceneStage const& stage,
    superdex::robotics::BotPrefab const& botPrefab,
    int linkIndex,
    mochi::CoordinateSpaceConverter const& spaceConverter) {
  JointRenderFrameInfo info{};

  if (linkIndex < 0 || linkIndex >= static_cast<int>(botPrefab.links.size())) {
    return info;
  }

  if (linkIndex >= static_cast<int>(botPrefab.joints.size())) {
    return info;
  }

  auto const& actors = stage.GetActors();
  if (linkIndex >= stage.GetNumActors()) {
    return info;
  }

  auto const& joint = botPrefab.joints[linkIndex];
  auto const& link = botPrefab.links[linkIndex];
  SceneObject* linkObject = actors[linkIndex].sceneObject;

  // if the link has a visual mesh, scale the joint visualization to the AABB
  if (linkObject) {
    filament::Box aabb = linkObject->GetAABB();
    info.scale = std::max(0.02f, length(aabb.halfExtent) * 0.3f);
  } else {
    info.scale = defaultFrameScale;
  }

  auto parentRotation = filament::math::mat3f(1.0f);
  filament::math::float3 parentPosWorld = {0.0f, 0.0f, 0.0f};
  if (link.parentLink >= 0 && link.parentLink < stage.GetNumActors()) {
    LinkTransformToOutputSpace(
        actors[link.parentLink].worldTransform, spaceConverter, parentPosWorld, parentRotation);
  }

  auto const& jointToParent = joint.parentLinkFromJoint;

  filament::math::float3 const jointPosRender =
      ToFilament<float>(spaceConverter.DirectionToOutput(jointToParent.GetTranslation()));

  filament::math::float3 jointPosInParent = parentRotation * jointPosRender;
  info.position = parentPosWorld + jointPosInParent;

  filament::math::float3 localAxisRender =
      ToFilament<float>(spaceConverter.DirectionToOutput(joint.axis));
  localAxisRender = normalize(localAxisRender);

  filament::math::quatf jointQuatRender =
      ToFilament<float>(spaceConverter.RotationToOutput(jointToParent.GetRotation()));

  auto jointRotMatrix = filament::math::mat3f(jointQuatRender);
  info.rotation = parentRotation * jointRotMatrix;

  filament::math::float3 childPosWorld;
  filament::math::mat3f childRotation;
  LinkTransformToOutputSpace(
      actors[linkIndex].worldTransform, spaceConverter, childPosWorld, childRotation);

  info.axis = normalize(childRotation * localAxisRender);

  info.arcAxis = normalize(info.rotation * localAxisRender);

  filament::math::float3 refDirWorld;
  bool hasValidRefDir = false;

  // The joint's zero reference points toward the link's "bulk". Prefer the center of mass when the
  // bot specifies it explicitly; otherwise fall back to the center of the link's bounding box. The
  // bounding-box center is available for any link with a visual mesh, whereas a
  // mass/density-derived center of mass is only computed at runtime and is absent here. This keeps
  // CoM-authored bots unchanged while giving the others a robust, roughly-equivalent reference.
  if (link.centerOfMass.has_value()) {
    filament::math::float3 comLocalRender =
        ToFilament<float>(spaceConverter.TranslationToOutput(*link.centerOfMass));

    refDirWorld = info.rotation * comLocalRender;
    float const refDirLength = length(refDirWorld);
    if (refDirLength > 1e-6f) {
      refDirWorld = refDirWorld / refDirLength;
      hasValidRefDir = true;
    }
  }

  if (!hasValidRefDir && linkObject != nullptr) {
    // Direction toward the center of the link's object-aligned bounding box, taken in the link's
    // local frame and placed via the joint's zero-DOF frame (info.rotation) exactly like the center
    // of mass above. Using the local-frame center (rather than the posed world center) keeps the
    // reference fixed to the joint's zero pose, so the fan does not drift as the joint articulates.
    if (std::optional<filament::math::float3> const localBoundsCenter =
            linkObject->GetLocalBoundsCenter()) {
      refDirWorld = info.rotation * *localBoundsCenter;
      float const refDirLength = length(refDirWorld);
      if (refDirLength > 1e-6f) {
        refDirWorld = refDirWorld / refDirLength;
        hasValidRefDir = true;
      }
    }
  }

  if (!hasValidRefDir) {
    filament::math::float3 localRefDir = {1.0f, 0.0f, 0.0f};
    float dotWithAxis = dot(localRefDir, localAxisRender);
    if (std::abs(dotWithAxis) > 0.9f) {
      localRefDir = {0.0f, 1.0f, 0.0f};
    }
    localRefDir = normalize(localRefDir - localAxisRender * dot(localRefDir, localAxisRender));
    info.referenceDir = normalize(info.rotation * localRefDir);
  } else {
    refDirWorld = refDirWorld - info.axis * dot(refDirWorld, info.axis);
    float projLength = length(refDirWorld);
    if (projLength > 1e-6f) {
      info.referenceDir = refDirWorld / projLength;
    } else {
      filament::math::float3 localRefDir = {1.0f, 0.0f, 0.0f};
      float dotWithAxis = dot(localRefDir, localAxisRender);
      if (std::abs(dotWithAxis) > 0.9f) {
        localRefDir = {0.0f, 1.0f, 0.0f};
      }
      localRefDir = normalize(localRefDir - localAxisRender * dot(localRefDir, localAxisRender));
      info.referenceDir = normalize(info.rotation * localRefDir);
    }
  }

  return info;
}

void DrawRevoluteLimits(
    DebugDraw* debugDraw,
    JointRenderFrameInfo const& frame,
    superdex::robotics::BotJointPrefab const& joint,
    superdex::robotics::BotLinkPrefab const& link,
    mochi::CoordinateSpaceConverter const& spaceConverter,
    JointVisualizationSettings const& settings) {
  float const visualScale = settings.jointVisualizationScale;
  float const scale = (settings.scaleToLinkSize ? frame.scale : defaultFrameScale) * visualScale;

  float cylinderLength = scale * 2.0f;
  float cylinderRadius = scale * 0.08f;

  debugDraw->DrawSolidCylinder(
      frame.position - frame.axis * (cylinderLength * 0.5f),
      frame.axis * cylinderLength,
      cylinderRadius,
      ToFilament(settings.jointColor),
      16,
      true,
      true);

  if (joint.minLimit.has_value() && IsFinite(*joint.minLimit) && joint.maxLimit.has_value() &&
      IsFinite(*joint.maxLimit)) {
    auto const axisNormalized = Normalize(joint.axis);
    auto minAngle = static_cast<float>(Dot(*joint.minLimit, axisNormalized));
    auto maxAngle = static_cast<float>(Dot(*joint.maxLimit, axisNormalized));
    if (minAngle != 0.0f || maxAngle != 0.0f) {
      float arcRadius = scale * 1.2f;
      int numSegments = std::max(
          8, static_cast<int>(32 * std::abs(maxAngle - minAngle) / filament::math::f::TAU));

      filament::math::float3 refPointLocal;
      if (link.centerOfMass.has_value()) {
        refPointLocal = ToFilament<float>(spaceConverter.DirectionToOutput(*link.centerOfMass));
      } else {
        refPointLocal = {0.0f, 0.0f, 0.0f};
      }

      auto rotateAroundAxis = [&](filament::math::float3 const& v,
                                  filament::math::float3 const& axis,
                                  float angle) -> filament::math::float3 {
        float c = std::cos(angle);
        float s = std::sin(angle);
        return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0f - c);
      };

      filament::math::float3 localAxisRender =
          normalize(ToFilament<float>(spaceConverter.DirectionToOutput(joint.axis)));

      filament::math::float3 comAtMin =
          frame.rotation * rotateAroundAxis(refPointLocal, localAxisRender, minAngle);
      filament::math::float3 comAtMax =
          frame.rotation * rotateAroundAxis(refPointLocal, localAxisRender, maxAngle);

      filament::math::float3 minDir = comAtMin - frame.axis * dot(comAtMin, frame.axis);
      filament::math::float3 maxDir = comAtMax - frame.axis * dot(comAtMax, frame.axis);

      float minDirLen = length(minDir);
      float maxDirLen = length(maxDir);

      if (minDirLen > 1e-6f && maxDirLen > 1e-6f) {
        minDir = minDir / minDirLen;
        maxDir = maxDir / maxDirLen;

        float expectedArcAngle = maxAngle - minAngle;

        debugDraw->DrawSolidArc(
            frame.position,
            frame.axis,
            minDir,
            expectedArcAngle,
            cylinderRadius,
            arcRadius,
            ToFilament(settings.jointFanColor),
            numSegments,
            true);
      } else {
        float arcAngle = maxAngle - minAngle;
        float cosMin = std::cos(minAngle);
        float sinMin = std::sin(minAngle);
        filament::math::float3 startDir = frame.referenceDir * cosMin +
            cross(frame.arcAxis, frame.referenceDir) * sinMin +
            frame.arcAxis * dot(frame.arcAxis, frame.referenceDir) * (1.0f - cosMin);

        debugDraw->DrawSolidArc(
            frame.position,
            frame.axis,
            startDir,
            arcAngle,
            cylinderRadius,
            arcRadius,
            ToFilament(settings.jointFanColor),
            numSegments,
            true);
      }
    }
  }
}

void DrawPrismaticLimits(
    DebugDraw* debugDraw,
    JointRenderFrameInfo const& frame,
    superdex::robotics::BotJointPrefab const& joint,
    JointVisualizationSettings const& settings) {
  if (joint.minLimit.has_value() && IsFinite(*joint.minLimit) && joint.maxLimit.has_value() &&
      IsFinite(*joint.maxLimit)) {
    float const visualScale = settings.jointVisualizationScale;
    float const scale = (settings.scaleToLinkSize ? frame.scale : defaultFrameScale) * visualScale;

    auto const axisNormalized = Normalize(joint.axis);
    auto minLimit = static_cast<float>(Dot(*joint.minLimit, axisNormalized));
    auto maxLimit = static_cast<float>(Dot(*joint.maxLimit, axisNormalized));

    float cylinderRadius = scale * 0.1f;
    float cylinderLength = std::max(0.01f, maxLimit - minLimit);
    if (cylinderLength > 0.0f) {
      filament::math::float3 startPos = frame.position + frame.axis * minLimit;

      debugDraw->DrawSolidCylinder(
          startPos,
          frame.axis * cylinderLength,
          cylinderRadius,
          ToFilament(settings.jointColor),
          16,
          true,
          true);

      float capRadius = cylinderRadius * 1.5f;
      debugDraw->DrawSolidSphere(
          frame.position + frame.axis * minLimit,
          capRadius,
          ToFilament(settings.jointColor),
          8,
          12,
          true);
      debugDraw->DrawSolidSphere(
          frame.position + frame.axis * maxLimit,
          capRadius,
          ToFilament(settings.jointColor),
          8,
          12,
          true);
    }
  }
}

void DrawSphericalLimits(
    DebugDraw* debugDraw,
    JointRenderFrameInfo const& frame,
    superdex::robotics::BotJointPrefab const& joint,
    mochi::CoordinateSpaceConverter const& spaceConverter,
    JointVisualizationSettings const& settings) {
  if (joint.minLimit.has_value() && IsFinite(*joint.minLimit) && joint.maxLimit.has_value() &&
      IsFinite(*joint.maxLimit)) {
    float const visualScale = settings.jointVisualizationScale;
    float const scale = (settings.scaleToLinkSize ? frame.scale : defaultFrameScale) * visualScale;

    // Spherical joints have 3 rotational DOFs with independent limits per axis.
    // Draw three arc fans, one for each rotation axis (X, Y, Z).
    float const arcRadius = scale * 1.2f;
    float const cylinderRadius = scale * 0.06f;

    auto const min = StaticCast<mochi::Float3>(*joint.minLimit);
    auto const max = StaticCast<mochi::Float3>(*joint.maxLimit);

    filament::math::float3 renderX =
        ToFilament<float>(spaceConverter.DirectionToOutput(mochi::Float3{1.0f, 0.0f, 0.0f}));
    filament::math::float3 renderY =
        ToFilament<float>(spaceConverter.DirectionToOutput(mochi::Float3{0.0f, 1.0f, 0.0f}));
    filament::math::float3 renderZ =
        ToFilament<float>(spaceConverter.DirectionToOutput(mochi::Float3{0.0f, 0.0f, 1.0f}));

    filament::math::float3 const xAxis = normalize(frame.rotation * renderX);
    filament::math::float3 const yAxis = normalize(frame.rotation * renderY);
    filament::math::float3 const zAxis = normalize(frame.rotation * renderZ);

    filament::math::float4 const xColor = {1.0f, 0.3f, 0.3f, 0.5f};
    filament::math::float4 const yColor = {0.3f, 1.0f, 0.3f, 0.5f};
    filament::math::float4 const zColor = {0.3f, 0.3f, 1.0f, 0.5f};

    auto rotateAroundAxis = [](filament::math::float3 const& v,
                               filament::math::float3 const& axis,
                               float angle) -> filament::math::float3 {
      float c = std::cos(angle);
      float s = std::sin(angle);
      return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0f - c);
    };

    // Draw X-axis rotation arc (rotation around X, perpendicular to Y-Z plane)
    if (min[0] != 0.0f || max[0] != 0.0f) {
      float arcAngle = max[0] - min[0];
      int numSegments =
          std::max(8, static_cast<int>(32 * std::abs(arcAngle) / filament::math::f::TAU));
      filament::math::float3 startDir = rotateAroundAxis(yAxis, xAxis, min[0]);
      debugDraw->DrawSolidArc(
          frame.position,
          xAxis,
          startDir,
          arcAngle,
          cylinderRadius,
          arcRadius,
          xColor,
          numSegments,
          true);
    }

    // Draw Y-axis rotation arc (rotation around Y, perpendicular to X-Z plane)
    if (min[1] != 0.0f || max[1] != 0.0f) {
      float arcAngle = max[1] - min[1];
      int numSegments =
          std::max(8, static_cast<int>(32 * std::abs(arcAngle) / filament::math::f::TAU));
      filament::math::float3 startDir = rotateAroundAxis(zAxis, yAxis, min[1]);
      debugDraw->DrawSolidArc(
          frame.position,
          yAxis,
          startDir,
          arcAngle,
          cylinderRadius,
          arcRadius,
          yColor,
          numSegments,
          true);
    }

    // Draw Z-axis rotation arc (rotation around Z, perpendicular to X-Y plane)
    if (min[2] != 0.0f || max[2] != 0.0f) {
      float arcAngle = max[2] - min[2];
      int numSegments =
          std::max(8, static_cast<int>(32 * std::abs(arcAngle) / filament::math::f::TAU));
      filament::math::float3 startDir = rotateAroundAxis(xAxis, zAxis, min[2]);
      debugDraw->DrawSolidArc(
          frame.position,
          zAxis,
          startDir,
          arcAngle,
          cylinderRadius,
          arcRadius,
          zColor,
          numSegments,
          true);
    }

    // Draw small sphere at joint center to indicate spherical joint type
    float sphereRadius = scale * 0.15f;
    debugDraw->DrawSolidSphere(
        frame.position, sphereRadius, ToFilament(settings.jointColor), 8, 12, true);
  }
}

} // namespace

void DrawJointLimitVisualization(
    DebugDraw* debugDraw,
    SceneStage const& stage,
    superdex::robotics::BotPrefab const& botPrefab,
    int linkIndex,
    mochi::CoordinateSpaceConverter const& spaceConverter,
    BotVisualizationFlags const& flags,
    JointVisualizationSettings const& settings) {
  if (!flags.showJointLimits || !debugDraw) {
    return;
  }

  if (linkIndex < 0 || linkIndex >= static_cast<int>(botPrefab.links.size())) {
    return;
  }

  if (linkIndex >= static_cast<int>(botPrefab.joints.size())) {
    return;
  }

  auto const& joint = botPrefab.joints[linkIndex];
  auto const& link = botPrefab.links[linkIndex];
  JointRenderFrameInfo frame = ComputeJointRenderFrame(stage, botPrefab, linkIndex, spaceConverter);

  if (frame.scale <= 0.0f) {
    return;
  }

  switch (joint.type) {
    case mochi::ArticulatedJointType::Revolute:
      if (flags.showRevoluteLimits) {
        DrawRevoluteLimits(debugDraw, frame, joint, link, spaceConverter, settings);
      }
      break;

    case mochi::ArticulatedJointType::Prismatic:
      if (flags.showPrismaticLimits) {
        DrawPrismaticLimits(debugDraw, frame, joint, settings);
      }
      break;

    case mochi::ArticulatedJointType::Spherical:
      if (flags.showSphericalLimits) {
        DrawSphericalLimits(debugDraw, frame, joint, spaceConverter, settings);
      }
      break;

    default:
      break;
  }
}

void DrawAllJointLimits(
    DebugDraw* debugDraw,
    SceneStage const& stage,
    superdex::robotics::BotPrefab const& botPrefab,
    mochi::CoordinateSpaceConverter const& spaceConverter,
    BotVisualizationFlags const& flags,
    JointVisualizationSettings const& settings) {
  if (!flags.showJointLimits || !debugDraw) {
    return;
  }

  for (int i = 1; i < static_cast<int>(botPrefab.links.size()); ++i) {
    DrawJointLimitVisualization(debugDraw, stage, botPrefab, i, spaceConverter, flags, settings);
  }
}

void DrawCycleJointVisualization(
    DebugDraw* debugDraw,
    SceneStage const& stage,
    superdex::robotics::BotPrefab const& botPrefab,
    int cycleIndex,
    mochi::CoordinateSpaceConverter const& spaceConverter,
    BotVisualizationFlags const& flags,
    JointVisualizationSettings const& settings) {
  if (!flags.showJointLimits || !flags.showCycles || !debugDraw) {
    return;
  }
  if (cycleIndex < 0 || cycleIndex >= static_cast<int>(botPrefab.cycles.size())) {
    return;
  }

  auto const& cycle = botPrefab.cycles[cycleIndex];

  // The pivot is the cycle joint's origin. jointFromChildLink maps the child link frame to the
  // joint frame, so its inverse expresses the joint origin (the pivot) in the child link's frame.
  mochi::Real3 const pivotLocal = mochi::Invert(cycle.jointFromChildLink).GetTranslation();
  std::optional<filament::math::float3> const pivotWorld =
      ComputeWaypointWorldPosition(stage, cycle.childLink, pivotLocal, spaceConverter);
  if (!pivotWorld) {
    return;
  }

  // Match the joint-marker sizing convention (see the spherical-joint center marker).
  float const linkScale =
      settings.scaleToLinkSize ? ComputeLinkScale(stage, cycle.childLink) : defaultFrameScale;
  float const sphereRadius = linkScale * settings.jointVisualizationScale * 0.15f;

  debugDraw->DrawSolidSphere(
      *pivotWorld, sphereRadius, ToFilament(settings.jointColor), 16, 24, true);
}

void DrawAllCycleJointVisualizations(
    DebugDraw* debugDraw,
    SceneStage const& stage,
    superdex::robotics::BotPrefab const& botPrefab,
    mochi::CoordinateSpaceConverter const& spaceConverter,
    BotVisualizationFlags const& flags,
    JointVisualizationSettings const& settings) {
  if (!debugDraw || !flags.showJointLimits || !flags.showCycles) {
    return;
  }
  for (int i = 0; i < static_cast<int>(botPrefab.cycles.size()); ++i) {
    DrawCycleJointVisualization(debugDraw, stage, botPrefab, i, spaceConverter, flags, settings);
  }
}

namespace {

/* @brief Joint frame data including position and axis in world space. */
struct JointFrameData {
  filament::math::float3 position; // Joint origin in world/render space
  filament::math::float3 axis; // Joint axis in world/render space (unit vector)
};

/* @brief Helper to compute joint frame data (position and axis) in world space. */
JointFrameData ComputeJointFrameData(
    SceneStage const& stage,
    superdex::robotics::BotPrefab const& botPrefab,
    int jointIndex,
    mochi::CoordinateSpaceConverter const& spaceConverter) {
  JointFrameData result{{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};

  if (jointIndex < 0 || jointIndex >= static_cast<int>(botPrefab.joints.size())) {
    return result;
  }

  auto const& joint = botPrefab.joints[jointIndex];
  auto const& link = botPrefab.links[jointIndex];
  auto const& actors = stage.GetActors();

  // Get parent link index
  int parentLinkIndex = link.parentLink;
  if (parentLinkIndex < 0 || parentLinkIndex >= stage.GetNumActors()) {
    // Root joint or invalid parent - use origin and default axis
    return result;
  }

  // Get parent link's world transform
  mochi::TransformRT const& parentWorldTransform = actors[parentLinkIndex].worldTransform;

  // Get joint transform relative to parent
  mochi::TransformRT const& jointToParent = joint.parentLinkFromJoint;

  // Combine transforms: world = parentWorld * jointToParent
  mochi::TransformRT worldJointTransform = parentWorldTransform * jointToParent;

  // Extract translation
  result.position =
      ToFilament<float>(spaceConverter.TranslationToOutput(worldJointTransform.GetTranslation()));

  // Extract rotation and transform joint axis
  auto const rotRender =
      ToFilament<float>(spaceConverter.RotationToOutput(worldJointTransform.GetRotation()));
  auto rotMatrix = filament::math::mat3f(rotRender);

  // Joint axis in local space
  filament::math::float3 localAxisRender =
      ToFilament<float>(spaceConverter.DirectionToOutput(joint.axis));

  // Transform to world space
  result.axis = normalize(rotMatrix * localAxisRender);

  return result;
}

/* @brief Compute attachment point for a joint given the adjacent joint position and alignment flag.
 * For revolute joints: computes tangent point on a circle (pulley model).
 * For prismatic joints: uses joint origin + axis displacement (linear slider model). */
filament::math::float3 ComputeAttachmentPoint(
    JointFrameData const& joint,
    filament::math::float3 const& adjacentPos,
    float radius,
    bool alignment,
    float axisDisp,
    bool isPrismatic) {
  // For prismatic joints, the transmission attaches at the joint origin (with axis displacement)
  // The "radius" parameter is a dimensionless transmission ratio, not a physical radius
  if (isPrismatic) {
    return joint.position + joint.axis * axisDisp;
  }

  // If radius is zero or very small, use joint origin
  if (radius < 1e-6f) {
    // Apply axis displacement even when radius is zero
    return joint.position + joint.axis * axisDisp;
  }

  // Vector from joint to adjacent joint
  filament::math::float3 d = adjacentPos - joint.position;
  float dLen = length(d);
  if (dLen < 1e-6f) {
    // Coincident joints - use arbitrary perpendicular direction
    d = {1.0f, 0.0f, 0.0f};
    dLen = 1.0f;
  }
  d = d / dLen;

  // Project d onto plane perpendicular to joint axis
  float dotAD = dot(joint.axis, d);
  filament::math::float3 dPerp = d - joint.axis * dotAD;
  float dPerpLen = length(dPerp);

  // If d is parallel to axis, use arbitrary perpendicular direction
  if (dPerpLen < 1e-6f) {
    // Choose an arbitrary vector not parallel to axis
    filament::math::float3 arbitrary = (std::abs(joint.axis.x) < 0.9f)
        ? filament::math::float3{1.0f, 0.0f, 0.0f}
        : filament::math::float3{0.0f, 1.0f, 0.0f};
    dPerp = arbitrary - joint.axis * dot(joint.axis, arbitrary);
    dPerpLen = length(dPerp);
  }
  dPerp = dPerp / dPerpLen;

  // Right vector (tangent direction) - perpendicular to both axis and dPerp
  // Use cross(axis, dPerp) to align with plane normal cross(axis, d)
  // This makes alignment=true correspond to positive side of the plane
  // Note: right is guaranteed to be a unit vector since joint.axis and dPerp
  // are both unit vectors and perpendicular by construction.
  filament::math::float3 right = cross(joint.axis, dPerp);

  // Choose side based on alignment flag
  // true = positive side, false = negative side
  float sign = alignment ? 1.0f : -1.0f;

  // Attachment point = joint position + radius * offset direction + axis displacement
  return joint.position + right * (radius * sign) + joint.axis * axisDisp;
}

} // namespace

void DrawLinearTransmissionVisualization(
    DebugDraw* debugDraw,
    SceneStage const& stage,
    superdex::robotics::BotPrefab const& botPrefab,
    int transmissionIndex,
    mochi::CoordinateSpaceConverter const& spaceConverter,
    BotVisualizationFlags const& flags,
    TransmissionVisualizationSettings const& settings,
    float displacement) {
  if (!debugDraw || !flags.showLinearTransmissions) {
    return;
  }

  if (transmissionIndex < 0 ||
      transmissionIndex >= static_cast<int>(botPrefab.linearTransmissions.size())) {
    return;
  }

  auto const& transmission = botPrefab.linearTransmissions[transmissionIndex];

  // Need at least 2 joints to draw a line
  if (transmission.jointIndices.size() < 2) {
    return;
  }

  // Transmission color based on displacement: binary selection between stretched and compressed
  // states. Use base color for stretched (displacement >= 0), compressed color for slack/compressed
  // (displacement < 0).
  filament::math::float4 const transmissionColor = (displacement >= 0.0f)
      ? ToFilament(settings.linearTransmissionColor)
      : ToFilament(settings.linearTransmissionCompressedColor);

  // Draw line segments connecting consecutive joints
  for (size_t i = 0; i + 1 < transmission.jointIndices.size(); ++i) {
    int jointIdx0 = transmission.jointIndices[i];
    int jointIdx1 = transmission.jointIndices[i + 1];

    filament::math::float3 pos0;
    filament::math::float3 pos1;

    // Get joint frame data for both joints
    JointFrameData joint0 = ComputeJointFrameData(stage, botPrefab, jointIdx0, spaceConverter);
    JointFrameData joint1 = ComputeJointFrameData(stage, botPrefab, jointIdx1, spaceConverter);

    // Determine if each endpoint is an actual end of the transmission
    bool isFirstJointOfTransmission = (i == 0);
    bool isLastJointOfTransmission = (i + 1 == transmission.jointIndices.size() - 1);

    // Get parameters for joint 0
    float radius0 = (i < transmission.jointCoefficients.size())
        ? std::abs(static_cast<float>(transmission.jointCoefficients[i]))
        : 0.0f;
    bool align0 = (i < transmission.jointCoefficients.size())
        ? transmission.jointCoefficients[i] >= 0.0f
        : false;
    float axisDisp0 = (i < transmission.jointAxisDisps.size())
        ? static_cast<float>(transmission.jointAxisDisps[i])
        : 0.0f;

    // Get parameters for joint 1
    float radius1 = (i + 1 < transmission.jointCoefficients.size())
        ? std::abs(static_cast<float>(transmission.jointCoefficients[i + 1]))
        : 0.0f;
    bool align1 = (i + 1 < transmission.jointCoefficients.size())
        ? transmission.jointCoefficients[i + 1] >= 0.0f
        : false;
    float axisDisp1 = (i + 1 < transmission.jointAxisDisps.size())
        ? static_cast<float>(transmission.jointAxisDisps[i + 1])
        : 0.0f;

    // Check if joints are prismatic (transmission attaches at origin, not on a circle)
    bool isPrismatic0 = false;
    bool isPrismatic1 = false;
    if (jointIdx0 >= 0 && jointIdx0 < static_cast<int>(botPrefab.joints.size())) {
      isPrismatic0 = botPrefab.joints[jointIdx0].type == mochi::ArticulatedJointType::Prismatic;
    }
    if (jointIdx1 >= 0 && jointIdx1 < static_cast<int>(botPrefab.joints.size())) {
      isPrismatic1 = botPrefab.joints[jointIdx1].type == mochi::ArticulatedJointType::Prismatic;
    }

    // Compute position for joint 0 endpoint
    if (isFirstJointOfTransmission) {
      // First joint of transmission: use tangent attachment point
      pos0 =
          ComputeAttachmentPoint(joint0, joint1.position, radius0, align0, axisDisp0, isPrismatic0);
    } else {
      // Middle joint: use joint origin
      pos0 = joint0.position;
    }

    // Compute position for joint 1 endpoint
    if (isLastJointOfTransmission) {
      // Last joint of transmission: use tangent attachment point
      pos1 =
          ComputeAttachmentPoint(joint1, joint0.position, radius1, align1, axisDisp1, isPrismatic1);
    } else {
      // Middle joint: use joint origin
      pos1 = joint1.position;
    }

    // Draw solid cylinder using helper function
    DrawTransmissionSegment(debugDraw, pos0, pos1, transmissionColor);
  }
}

void DrawAllLinearTransmissionVisualizations(
    DebugDraw* debugDraw,
    SceneStage const& stage,
    superdex::robotics::BotPrefab const& botPrefab,
    mochi::CoordinateSpaceConverter const& spaceConverter,
    BotVisualizationFlags const& flags,
    TransmissionVisualizationSettings const& settings,
    mochi::DynamicArray<float> const* transmissionDisplacements) {
  if (!debugDraw || !flags.showLinearTransmissions) {
    return;
  }

  for (int i = 0; i < static_cast<int>(botPrefab.linearTransmissions.size()); ++i) {
    float displacement = 0.0f;
    if (transmissionDisplacements && i < static_cast<int>(transmissionDisplacements->size())) {
      displacement = (*transmissionDisplacements)[i];
    }
    DrawLinearTransmissionVisualization(
        debugDraw, stage, botPrefab, i, spaceConverter, flags, settings, displacement);
  }
}

void DrawSpatialTendonVisualization(
    DebugDraw* debugDraw,
    SceneStage const& stage,
    superdex::robotics::BotPrefab const& botPrefab,
    int tendonIndex,
    mochi::CoordinateSpaceConverter const& spaceConverter,
    BotVisualizationFlags const& flags,
    TransmissionVisualizationSettings const& settings,
    float displacement) {
  if (!debugDraw || !flags.showSpatialTendons) {
    return;
  }

  if (tendonIndex < 0 || tendonIndex >= static_cast<int>(botPrefab.spatialTendons.size())) {
    return;
  }

  auto const& tendon = botPrefab.spatialTendons[tendonIndex];

  // Need at least 2 routing elements to draw segments
  if (tendon.routingElements.size() < 2) {
    return;
  }

  // Tendon color based on displacement: binary selection between stretched and compressed states.
  // Use base color for stretched (displacement >= 0), compressed color for slack/compressed
  // (displacement < 0).
  filament::math::float4 const tendonColor = (displacement >= 0.0f)
      ? ToFilament(settings.spatialTendonColor)
      : ToFilament(settings.spatialTendonCompressedColor);

  // Iterate through routing elements, tracking the previous waypoint position
  std::optional<filament::math::float3> prevWaypointPos;

  for (auto const& element : tendon.routingElements) {
    if (element.type == mochi::RoutingElementType::Waypoint) {
      // Compute world position of the waypoint
      mochi::Real3 localPositionInLinkFrame = element.localPosition;
      std::optional<filament::math::float3> waypointPosOpt = ComputeWaypointWorldPosition(
          stage, element.index, localPositionInLinkFrame, spaceConverter);

      if (!waypointPosOpt.has_value()) {
        // Invalid link index — skip this waypoint to avoid misleading visualization
        // at world origin and to surface potential prefab data corruption.
        prevWaypointPos = std::nullopt;
        continue;
      }
      filament::math::float3 waypointPos = waypointPosOpt.value();

      // If we have a previous waypoint, draw a segment between them
      if (prevWaypointPos.has_value()) {
        DrawTransmissionSegment(debugDraw, prevWaypointPos.value(), waypointPos, tendonColor);
      }

      // Update previous waypoint position
      prevWaypointPos = waypointPos;
    } else if (element.type == mochi::RoutingElementType::LinearJoint) {
      // Visualize the linear joint using ComputeAttachmentPoint to find the attachment point
      // on the joint pulley, similar to how LinearTransmission does it.
      // The LinearJoint is treated as a waypoint-like element that connects to adjacent waypoints.
      JointFrameData jointData =
          ComputeJointFrameData(stage, botPrefab, element.index, spaceConverter);

      // Determine joint type for prismatic check
      bool isPrismatic = false;
      if (element.index >= 0 && element.index < static_cast<int>(botPrefab.joints.size())) {
        isPrismatic =
            botPrefab.joints[element.index].type == mochi::ArticulatedJointType::Prismatic;
      }

      // Compute attachment point using the coefficient as the moment arm (radius)
      // Use previous waypoint position as adjacentPos if available, to help choose the correct
      // tangent. Otherwise, use joint position (will use arbitrary direction).
      // TODO: If there's no previous waypoint, we should use the next waypoint or joint position.
      auto radius = static_cast<float>(mochi::Abs(element.coefficient));
      bool alignment = element.coefficient >= 0.0f;
      filament::math::float3 adjacentPos =
          prevWaypointPos.has_value() ? prevWaypointPos.value() : jointData.position;
      filament::math::float3 attachmentPos = ComputeAttachmentPoint(
          jointData, adjacentPos, radius, alignment, 0.0f /*axisDisp*/, isPrismatic);

      // If we have a previous waypoint, draw a segment connecting to the LinearJoint
      if (prevWaypointPos.has_value()) {
        DrawTransmissionSegment(debugDraw, prevWaypointPos.value(), attachmentPos, tendonColor);
      }

      // Update previous waypoint position to the LinearJoint attachment point
      // (so the next waypoint will connect to it)
      prevWaypointPos = attachmentPos;
    }
  }
}

void DrawAllSpatialTendonVisualizations(
    DebugDraw* debugDraw,
    SceneStage const& stage,
    superdex::robotics::BotPrefab const& botPrefab,
    mochi::CoordinateSpaceConverter const& spaceConverter,
    BotVisualizationFlags const& flags,
    TransmissionVisualizationSettings const& settings,
    mochi::DynamicArray<float> const* transmissionDisplacements,
    size_t transmissionOffset) {
  if (!debugDraw || !flags.showSpatialTendons) {
    return;
  }

  for (int i = 0; i < static_cast<int>(botPrefab.spatialTendons.size()); ++i) {
    float displacement = 0.0f;
    size_t const transmissionIndex = transmissionOffset + i;
    if (transmissionDisplacements && transmissionIndex < transmissionDisplacements->size()) {
      displacement = (*transmissionDisplacements)[transmissionIndex];
    }
    DrawSpatialTendonVisualization(
        debugDraw, stage, botPrefab, i, spaceConverter, flags, settings, displacement);
  }
}

} // namespace superdex::studio
