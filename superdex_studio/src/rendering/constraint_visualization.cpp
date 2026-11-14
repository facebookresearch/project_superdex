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

#include "rendering/constraint_visualization.h"

#include "rendering/debug_text.h"
#include "rendering/measure_tool.h"
#include "rendering/scene_stage.h"
#include "ui/imgui_widgets.h"

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_renderer/debug.h>
#include <mochi_renderer/type_conversions.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <type_traits>

using mochi::operator""_r;

namespace superdex::studio {

using filament::math::float3;
using filament::math::float4;
using mochi::Quaternion;
using mochi::real;
using mochi::Real3;
using mochi::TransformRT;

namespace {

// Marker sizes are fractions of the camera distance, so they read the same at any zoom (the asset
// tuner sizes its load arrows the same way).
constexpr float kThicknessPerDistance = 0.0035f;
constexpr float kTubeRadius = 2.0f; // x thickness x a segment's thickness scale
constexpr float kGizmoRadius = 28.0f; // x thickness: default arc / triad size
constexpr float kMinLinkRadius = 16.0f; // x thickness: smallest arc that follows the link length
constexpr float kMaxLinkRadius = 60.0f; // x thickness: largest arc that follows the link length

constexpr float4 kRangeColor{0.25f, 0.80f, 0.95f, 0.30f};
constexpr float4 kRangeEdgeColor{0.25f, 0.80f, 0.95f, 0.90f};
constexpr float4 kTargetColor{1.00f, 0.62f, 0.10f, 1.00f};
constexpr float4 kCurrentColor{0.95f, 0.95f, 0.95f, 1.00f};
constexpr float4 kAxisColor{0.80f, 0.80f, 0.80f, 0.80f};
constexpr float4 kAnchorAColor{0.30f, 0.60f, 1.00f, 1.00f};
constexpr float4 kAnchorBColor{1.00f, 0.40f, 0.70f, 1.00f};
constexpr float4 kErrorColor{1.00f, 0.20f, 0.20f, 1.00f};
constexpr float4 kTieColor{0.70f, 0.70f, 0.70f, 0.60f};
constexpr std::array<float4, 3> kAxisColors{
    float4{1.00f, 0.30f, 0.30f, 1.00f},
    float4{0.30f, 1.00f, 0.30f, 1.00f},
    float4{0.35f, 0.45f, 1.00f, 1.00f}};
constexpr std::array<float4, 3> kRangeAxisColors{
    float4{1.00f, 0.30f, 0.30f, 0.35f},
    float4{0.30f, 1.00f, 0.30f, 0.35f},
    float4{0.35f, 0.45f, 1.00f, 0.35f}};

// A gap this large between two points a constraint holds together is drawn as an error line [m].
constexpr real kGapTolerance = 1e-3_r;
// Most dashes one dashed line is drawn with, however close the camera is.
constexpr int kMaxDashes = 24;

TransformRT Scaled(TransformRT const& t, real scale) {
  return TransformRT(t.GetRotation(), t.GetTranslation() * scale);
}

// Signed angle of `q` about unit `axis` (its swing-twist twist angle) [rad], in (-pi, pi].
real TwistAngle(Quaternion const& q, Real3 const& axis) {
  mochi::Real4 const v = q.ToReal4();
  real angle = 2_r * std::atan2(mochi::Dot(Real3{v[0], v[1], v[2]}, axis), v[3]);
  if (angle > mochi::kPI) {
    angle -= 2_r * mochi::kPI;
  } else if (angle <= -mochi::kPI) {
    angle += 2_r * mochi::kPI;
  }
  return angle;
}

// Draws in editor space: converts every point and direction to renderer space and sizes markers by
// their distance to the camera.
class Painter {
 public:
  Painter(ConstraintDrawContext const& ctx, ConstraintEmphasis emphasis)
      : _ctx(ctx), _emphasis(emphasis) {}

  bool Labels() const {
    return _emphasis == ConstraintEmphasis::Selected && _ctx.debugText != nullptr;
  }

  float3 P(Real3 const& p) const {
    return mochi_renderer::ToFilament<float>(_ctx.editorToRenderer.TranslationToOutput(p));
  }
  float3 D(Real3 const& d) const {
    return mochi_renderer::ToFilament<float>(_ctx.editorToRenderer.DirectionToOutput(d));
  }

  // Line thickness at `p` [m]; thinner when the constraint is only shown with all the others.
  float Thickness(Real3 const& p) const {
    return _emphasis == ConstraintEmphasis::Normal ? 0.6f * BaseThickness(p) : BaseThickness(p);
  }
  // Marker size at `p`: `factor` base thicknesses [m] (editor and renderer lengths match); smaller
  // when the constraint is only shown with all the others.
  real Size(Real3 const& p, float factor) const {
    float const scale = _emphasis == ConstraintEmphasis::Normal ? 0.7f : 1.0f;
    return static_cast<real>(scale * factor * BaseThickness(p));
  }

  void Segment(Real3 const& a, Real3 const& b, float4 const& color, float thicknessScale = 0.35f)
      const {
    float3 const pa = P(a);
    float3 const pb = P(b);
    if (length(pb - pa) > 1e-7f) {
      _ctx.debugDraw.DrawSolidCylinder(
          pa, pb - pa, kTubeRadius * thicknessScale * Thickness(a), color, 8, true, true);
    }
  }

  // A diamond (two cones), drawn over the scene since DrawSolidSphere has no overlay mode. Its
  // faces tilt up and down so the scene light reads it from any side.
  void Dot(Real3 const& at, float4 const& color, float sizeScale = 1.2f) const {
    float3 const center = P(at);
    float const radius = 1.5f * sizeScale * Thickness(at);
    float3 const up{0.0f, 1.4f * radius, 0.0f}; // renderer space is Y-up
    _ctx.debugDraw.DrawSolidCone(center, up, radius, color, 12, false, true);
    _ctx.debugDraw.DrawSolidCone(center, -up, radius, color, 12, false, true);
  }

  // A dashed line, for ties that only say which things belong together.
  void Dashed(Real3 const& a, Real3 const& b, float4 const& color) const {
    real const total = mochi::Norm(b - a);
    real const dash = std::max(Size(a, 3.0f), total / (2_r * kMaxDashes));
    if (mochi::NearZero(total)) {
      return;
    }
    Real3 const dir = (b - a) * (1_r / total);
    for (real t = 0; t < total; t += 2 * dash) {
      Segment(a + dir * t, a + dir * std::min(t + dash, total), color, 0.2f);
    }
  }

  // Filled wedge in the plane normal to `axis`, from direction `start` turning `angle` [rad] about
  // `axis` by the right-hand rule (editor space).
  void Wedge(
      Real3 const& center,
      Real3 const& axis,
      Real3 const& start,
      real angle,
      real radius,
      float4 const& color) const {
    if (std::abs(angle) < 1e-6f || radius <= 0) {
      return;
    }
    // DrawSolidArc turns counter-clockwise about its normal in renderer space; mirror the normal
    // when the conversion flips handedness so the wedge keeps the editor-space sense.
    float3 normal = D(axis);
    if (_ctx.editorToRenderer.FlipsHandedness()) {
      normal = -normal;
    }
    int const segments = std::clamp(static_cast<int>(std::abs(angle) * 12.0f), 6, 64);
    _ctx.debugDraw.DrawSolidArc(
        P(center),
        normal,
        D(start),
        static_cast<float>(angle),
        0.0f,
        static_cast<float>(radius),
        color,
        segments,
        true);
  }

  // Tube along the arc from direction `start` turning `angle` about `axis` at `radius`.
  void ArcLine(
      Real3 const& center,
      Real3 const& axis,
      Real3 const& start,
      real angle,
      real radius,
      float4 const& color,
      float thicknessScale = 0.3f) const {
    int const segments = std::clamp(static_cast<int>(std::abs(angle) * 16.0f), 2, 64);
    Real3 previous = center + start * radius;
    for (int s = 1; s <= segments; ++s) {
      real const a = angle * static_cast<real>(s) / static_cast<real>(segments);
      Real3 const next = center + (Quaternion::FromAxisAngle(axis, a) * start) * radius;
      Segment(previous, next, color, thicknessScale);
      previous = next;
    }
  }

  // Axes of orientation `rotation` at `at` (X red, Y green, Z blue); `ghost` draws them faded.
  void Triad(Real3 const& at, Quaternion const& rotation, real length, bool ghost) const {
    for (int axis = 0; axis < 3; ++axis) {
      float4 color = kAxisColors[static_cast<size_t>(axis)];
      if (ghost) {
        color = float4{0.5f * (color.xyz + float3{1.0f}), 0.55f};
      }
      Segment(
          at,
          at + (rotation * mochi::BasisVector<real, 3>(axis)) * length,
          color,
          ghost ? 0.3f : 0.4f);
    }
  }

  // Ties the axis tips of two orientations at `at` (where a target pulls the current one).
  void TriadSprings(
      Real3 const& at,
      Quaternion const& current,
      Quaternion const& target,
      real length) const {
    for (int axis = 0; axis < 3; ++axis) {
      Segment(
          at + (current * mochi::BasisVector<real, 3>(axis)) * length,
          at + (target * mochi::BasisVector<real, 3>(axis)) * length,
          kTargetColor,
          0.2f);
    }
  }

  void Label(Real3 const& at, std::string const& text, float4 const& color) const {
    if (Labels()) {
      _ctx.debugText->Draw(P(at), text, color, ImVec2{0.0f, -18.0f});
    }
  }

  ConstraintDrawContext const& Context() const {
    return _ctx;
  }

 private:
  float BaseThickness(Real3 const& p) const {
    return ScreenScaledSize(
        P(p), _ctx.cameraPosition, kThicknessPerDistance, 1e-7f, std::numeric_limits<float>::max());
  }

  ConstraintDrawContext const& _ctx;
  ConstraintEmphasis _emphasis;
};

std::string FormatDof(real value, bool rotation) {
  char text[64];
  std::snprintf(
      text,
      sizeof(text),
      GetUnitFormat(
          UnitFormat::Position,
          rotation ? mochi::ArticulatedJointType::Revolute
                   : mochi::ArticulatedJointType::Prismatic),
      static_cast<float>(rotation ? value * mochi::kDegreesPerRadian : value));
  return text;
}

// The arc radius for a revolute joint: the link's reach from the joint when it has one (so the
// needle runs along the link), clamped to a readable size.
real RevoluteRadius(Painter const& painter, Real3 const& center, real reach) {
  real const minimum = painter.Size(center, kMinLinkRadius);
  real const maximum = painter.Size(center, kMaxLinkRadius);
  return reach > minimum ? std::min(reach, maximum) : painter.Size(center, kGizmoRadius);
}

//--------------------------------------------------------------------------------------------------
// Articulated joint constraints
//--------------------------------------------------------------------------------------------------

struct SingleDofDrawing {
  std::optional<std::pair<real, real>> range; // [min, max], as simulated (scaled)
  std::optional<real> target; // as simulated (scaled)
  std::string label;
};

void DrawSingleDof(
    Painter const& painter,
    mochi::prefab::ArticulatedActorPrefab const& actor,
    int jointIndex,
    int dofIndex,
    SingleDofDrawing const& drawing) {
  auto const frames = ComputeArticulatedJointFrames(actor, jointIndex, painter.Context().stage);
  if (!frames) {
    return;
  }
  real const current = MeasureJointDof(*frames, dofIndex);
  Real3 const center = frames->parent.GetTranslation();
  Quaternion const& frame = frames->parent.GetRotation();

  if (IsRotationDof(frames->type, dofIndex)) {
    // Angle: wedge for the range, needles for the current and target angles. The zero direction
    // points from the joint toward the child link's origin, so the needle runs along the link.
    Real3 const axis = mochi::Normalize(frame * frames->axis);
    auto const& link = actor.links[static_cast<size_t>(jointIndex)];
    Real3 reach = link.parentJointFromLink.GetTranslation() * frames->scale;
    reach = reach - frames->axis * mochi::Dot(reach, frames->axis);
    real const reachLength = mochi::Norm(reach);
    Real3 const zero = mochi::Normalize(
        frame * (reachLength > 1e-6f ? reach : mochi::OrthogonalVector(frames->axis)));
    real const radius = RevoluteRadius(painter, center, reachLength);
    auto const direction = [&](real angle) {
      return Quaternion::FromAxisAngle(axis, angle) * zero;
    };

    painter.Segment(center - axis * (0.4_r * radius), center + axis * (0.4_r * radius), kAxisColor);
    if (drawing.range) {
      auto const [lo, hi] = *drawing.range;
      painter.Wedge(center, axis, direction(lo), hi - lo, radius, kRangeColor);
      painter.Segment(center, center + direction(lo) * radius, kRangeEdgeColor, 0.25f);
      painter.Segment(center, center + direction(hi) * radius, kRangeEdgeColor, 0.25f);
      painter.Label(center + direction(hi) * radius, drawing.label, kRangeEdgeColor);
    }
    if (drawing.target) {
      Real3 const tip = center + direction(*drawing.target) * (1.1_r * radius);
      painter.Segment(center, tip, kTargetColor, 0.35f);
      painter.Dot(tip, kTargetColor);
      painter.ArcLine(
          center,
          axis,
          direction(current),
          *drawing.target - current,
          0.8_r * radius,
          kTargetColor,
          0.2f);
      painter.Label(tip, drawing.label, kTargetColor);
    }
    Real3 const needle = center + direction(current) * (1.1_r * radius);
    painter.Segment(center, needle, kCurrentColor, 0.3f);
    painter.Dot(needle, kCurrentColor, 0.9f);
    return;
  }

  // Distance: a bar for the range, dots for the current and target positions along the axis.
  Real3 const localAxis = frames->type == mochi::ArticulatedJointType::Prismatic
      ? frames->axis
      : mochi::BasisVector<real, 3>(std::clamp(dofIndex, 0, 2));
  Real3 const axis = mochi::Normalize(frame * localAxis);
  auto const at = [&](real value) { return center + axis * value; };
  Real3 const now = at(current);
  real const extent = painter.Size(center, kGizmoRadius);
  painter.Segment(center - axis * extent, center + axis * extent, kAxisColor, 0.2f);
  if (drawing.range) {
    auto const [lo, hi] = *drawing.range;
    painter.Segment(at(lo), at(hi), kRangeColor, 1.2f);
    painter.Dot(at(lo), kRangeEdgeColor, 1.4f);
    painter.Dot(at(hi), kRangeEdgeColor, 1.4f);
    painter.Label(at(hi), drawing.label, kRangeEdgeColor);
  }
  if (drawing.target) {
    painter.Segment(now, at(*drawing.target), kTargetColor, 0.25f);
    painter.Dot(at(*drawing.target), kTargetColor, 1.5f);
    painter.Label(at(*drawing.target), drawing.label, kTargetColor);
  }
  painter.Dot(now, kCurrentColor);
}

void Draw3dRotation(
    Painter const& painter,
    mochi::prefab::ArticulatedActorPrefab const& actor,
    int jointIndex,
    std::optional<std::pair<Real3, Real3>> const& range,
    std::optional<Quaternion> const& target,
    std::string const& label) {
  auto const frames = ComputeArticulatedJointFrames(actor, jointIndex, painter.Context().stage);
  if (!frames) {
    return;
  }
  Real3 const center = frames->parent.GetTranslation();
  Quaternion const& frame = frames->parent.GetRotation();
  Quaternion const current = frames->child.GetRotation();
  real const radius = painter.Size(center, kGizmoRadius);
  if (range) {
    // One wedge per rotation axis, as the bot editor draws spherical joint limits.
    for (int axis = 0; axis < 3; ++axis) {
      real const lo = range->first[axis];
      real const hi = range->second[axis];
      if (lo == 0 && hi == 0) {
        continue;
      }
      Real3 const n = frame * mochi::BasisVector<real, 3>(axis);
      Real3 const start =
          Quaternion::FromAxisAngle(n, lo) * (frame * mochi::BasisVector<real, 3>((axis + 1) % 3));
      painter.Wedge(
          center,
          n,
          start,
          std::min(hi - lo, 2_r * mochi::kPI),
          radius,
          kRangeAxisColors[static_cast<size_t>(axis)]);
    }
    painter.Label(
        center + (frame * mochi::BasisVector<real, 3>(2)) * radius, label, kRangeEdgeColor);
  }
  if (target) {
    Quaternion const goal = frame * *target;
    painter.Triad(center, goal, radius, true);
    painter.TriadSprings(center, current, goal, radius);
    painter.Label(center + (goal * mochi::BasisVector<real, 3>(2)) * radius, label, kTargetColor);
  }
  painter.Triad(center, current, 0.8_r * radius, false);
  painter.Dot(center, kCurrentColor);
}

//--------------------------------------------------------------------------------------------------
// Dispatch
//--------------------------------------------------------------------------------------------------

template <class C>
void DrawTyped(Painter const& painter, C const& c) {
  using namespace mochi::prefab;
  ConstraintDrawContext const& ctx = painter.Context();
  auto const staged = [&](mochi::DynamicString const& name) {
    return ctx.stage.FindActorByName(std::string_view(name));
  };
  // Mochi scales local anchors by the actor's metric scale.
  auto const scaleOf = [&](mochi::DynamicString const& name) {
    return GetConstraintActorMetricScale(ctx.prefab, std::string_view(name));
  };

  if constexpr (
      std::is_same_v<C, ArticulatedSingleDofTargetConstraintPrefab> ||
      std::is_same_v<C, ArticulatedSingleDofRangeConstraintPrefab>) {
    auto const* actor = FindArticulatedActor(ctx.prefab, std::string_view(c.actorName));
    if (actor == nullptr || c.jointIndex < 0 ||
        c.jointIndex >= static_cast<int>(actor->joints.size())) {
      return;
    }
    auto const type = actor->joints[static_cast<size_t>(c.jointIndex)].type;
    bool const rotation = IsRotationDof(type, c.dofIndex);
    real const valueScale = rotation ? 1_r : actor->scale;
    SingleDofDrawing drawing;
    if constexpr (std::is_same_v<C, ArticulatedSingleDofTargetConstraintPrefab>) {
      drawing.target = c.targetValue * valueScale;
      if (painter.Labels()) {
        drawing.label = "Target " + FormatDof(c.targetValue * valueScale, rotation);
      }
    } else {
      drawing.range = std::make_pair(c.minValue * valueScale, c.maxValue * valueScale);
      if (painter.Labels()) {
        drawing.label = "Range " + FormatDof(c.minValue * valueScale, rotation) + " .. " +
            FormatDof(c.maxValue * valueScale, rotation);
      }
    }
    DrawSingleDof(painter, *actor, c.jointIndex, c.dofIndex, drawing);
  } else if constexpr (std::is_same_v<C, Articulated3dRotationTargetConstraintPrefab>) {
    if (auto const* actor = FindArticulatedActor(ctx.prefab, std::string_view(c.actorName))) {
      Draw3dRotation(
          painter,
          *actor,
          c.jointIndex,
          std::nullopt,
          mochi::Normalize(c.target),
          "Rotation target");
    }
  } else if constexpr (std::is_same_v<C, Articulated3dRotationRangeConstraintPrefab>) {
    if (auto const* actor = FindArticulatedActor(ctx.prefab, std::string_view(c.actorName))) {
      Draw3dRotation(
          painter,
          *actor,
          c.jointIndex,
          std::make_pair(c.minValues, c.maxValues),
          std::nullopt,
          "Rotation range");
    }
  } else if constexpr (std::is_same_v<C, RigidSphericalJointConstraintPrefab>) {
    StagedActor const* a = staged(c.actorNameA);
    StagedActor const* b = staged(c.actorNameB);
    if (!a || !b) {
      return;
    }
    Real3 const pa = a->worldTransform.TransformPoint(c.localPosA * scaleOf(c.actorNameA));
    Real3 const pb = b->worldTransform.TransformPoint(c.localPosB * scaleOf(c.actorNameB));
    painter.Dashed(a->worldTransform.GetTranslation(), pa, kAnchorAColor);
    painter.Dashed(b->worldTransform.GetTranslation(), pb, kAnchorBColor);
    if (mochi::Norm(pb - pa) > kGapTolerance) {
      painter.Segment(pa, pb, kErrorColor, 0.3f);
    }
    painter.Dot(pa, kAnchorAColor, 1.6f);
    painter.Dot(pb, kAnchorBColor, 1.2f);
    painter.Label(pa, "Spherical joint", kAnchorAColor);
  } else if constexpr (std::is_same_v<C, RigidPrismaticJointConstraintPrefab>) {
    StagedActor const* a = staged(c.actorNameA);
    StagedActor const* b = staged(c.actorNameB);
    if (!a || !b || mochi::NearZero(mochi::Norm(c.freeAxis))) {
      return;
    }
    // The axis is authored in world space at rest and then rides on actor A, through B's rest
    // position; the limits are measured from that rest position.
    TransformRT const aFromRest = a->worldTransform * mochi::Invert(a->defaultWorldTransform);
    Real3 const axis = mochi::Normalize(aFromRest.TransformDirection(mochi::Normalize(c.freeAxis)));
    Real3 const origin = aFromRest.TransformPoint(b->defaultWorldTransform.GetTranslation());
    real const extent = painter.Size(origin, kGizmoRadius);
    real const lo = c.min.value_or(-extent);
    real const hi = c.max.value_or(extent);
    painter.Segment(
        origin + axis * lo, origin + axis * hi, kRangeColor, c.min && c.max ? 1.2f : 0.5f);
    if (c.min) {
      painter.Dot(origin + axis * lo, kRangeEdgeColor, 1.4f);
    }
    if (c.max) {
      painter.Dot(origin + axis * hi, kRangeEdgeColor, 1.4f);
    }
    Real3 const pb = b->worldTransform.GetTranslation();
    Real3 const onAxis = origin + axis * mochi::Dot(pb - origin, axis);
    if (mochi::Norm(pb - onAxis) > kGapTolerance) {
      painter.Segment(pb, onAxis, kErrorColor, 0.3f);
    }
    painter.Dashed(a->worldTransform.GetTranslation(), origin, kAnchorAColor);
    painter.Dot(pb, kAnchorBColor, 1.4f);
    painter.Label(origin + axis * hi, "Prismatic joint", kRangeEdgeColor);
  } else if constexpr (
      std::is_same_v<C, JointRotationTrackingConstraintPrefab> ||
      std::is_same_v<C, JointRotationRangeConstraintPrefab>) {
    StagedActor const* a = staged(c.actorNameA);
    StagedActor const* b = staged(c.actorNameB);
    if (!a || !b) {
      return;
    }
    // Both are measured at B's origin, in a frame that rides on actor A.
    Quaternion const aFromRest =
        a->worldTransform.GetRotation() * a->defaultWorldTransform.GetRotation().GetConjugate();
    Real3 const at = b->worldTransform.GetTranslation();
    real const length = painter.Size(at, kGizmoRadius);
    Quaternion const current = b->worldTransform.GetRotation();
    painter.Dashed(a->worldTransform.GetTranslation(), at, kTieColor);
    if constexpr (std::is_same_v<C, JointRotationTrackingConstraintPrefab>) {
      // The target is B's rest rotation relative to A.
      Quaternion const goal = aFromRest * b->defaultWorldTransform.GetRotation();
      painter.Triad(at, goal, length, true);
      painter.TriadSprings(at, current, goal, length);
      painter.Label(
          at + (goal * mochi::BasisVector<real, 3>(2)) * length, "Rotation target", kTargetColor);
    } else {
      // Mochi measures B's rotation in the reference frame (authored in world space at rest, then
      // riding on A) as the rotation vector of q = frame' * Rb * Ra_rest' * qRef, offset by its
      // rest value when the range is around rest.
      Quaternion const reference = Quaternion::FromRotationVector(c.refFrameRotVec);
      Quaternion const frame = aFromRest * reference;
      Quaternion const childFromRef =
          a->defaultWorldTransform.GetRotation().GetConjugate() * reference;
      Real3 const restOffset = c.rangeAroundRest
          ? (reference.GetConjugate() * b->defaultWorldTransform.GetRotation() * childFromRef)
                .ToRotationVector()
          : Real3{0_r, 0_r, 0_r};
      std::array<mochi::Real2 const*, 3> const ranges{
          &c.angleRangeX, &c.angleRangeY, &c.angleRangeZ};
      for (int axis = 0; axis < 3; ++axis) {
        mochi::Real2 const& r = *ranges[static_cast<size_t>(axis)];
        Real3 const n = frame * mochi::BasisVector<real, 3>(axis);
        Real3 const start = Quaternion::FromAxisAngle(n, r[0] + restOffset[axis]) *
            (frame * mochi::BasisVector<real, 3>((axis + 1) % 3));
        painter.Wedge(
            at,
            n,
            start,
            std::min(r[1] - r[0], 2_r * mochi::kPI),
            length,
            kRangeAxisColors[static_cast<size_t>(axis)]);
      }
      painter.Label(
          at + (frame * mochi::BasisVector<real, 3>(2)) * length,
          "Rotation range",
          kRangeEdgeColor);
      // The measured frame, so the triad's axes move through the wedges.
      painter.Triad(at, current * childFromRef, 0.8_r * length, false);
      return;
    }
    painter.Triad(at, current, 0.8_r * length, false);
  } else if constexpr (
      std::is_same_v<C, RigidPivotPositionConstraintPrefab> ||
      std::is_same_v<C, RigidPivotToRigidTargetConstraintPrefab>) {
    StagedActor const* a = staged(c.actorName);
    if (!a) {
      return;
    }
    Real3 const local = c.localPosition * scaleOf(c.actorName);
    Real3 const pivot = a->worldTransform.TransformPoint(local);
    Real3 goal;
    if constexpr (std::is_same_v<C, RigidPivotPositionConstraintPrefab>) {
      goal = c.targetPosition;
    } else {
      goal = c.targetTransform.TransformPoint(local);
      painter.Triad(
          c.targetTransform.GetTranslation(),
          mochi::Normalize(c.targetTransform.GetRotation()),
          painter.Size(goal, kGizmoRadius),
          true);
    }
    painter.Dashed(a->worldTransform.GetTranslation(), pivot, kTieColor);
    painter.Segment(pivot, goal, kTargetColor, 0.3f);
    painter.Dot(goal, kTargetColor, 1.6f);
    painter.Dot(pivot, kCurrentColor);
    painter.Label(goal, "Pivot target", kTargetColor);
  } else if constexpr (std::is_same_v<C, RigidPivotRotationConstraintPrefab>) {
    StagedActor const* a = staged(c.actorName);
    if (!a) {
      return;
    }
    Real3 const at = a->worldTransform.GetTranslation();
    real const length = painter.Size(at, kGizmoRadius);
    Quaternion const current =
        a->worldTransform.GetRotation() * Quaternion::FromRotationVector(c.localRotation);
    Quaternion const goal = Quaternion::FromRotationVector(c.targetRotation);
    painter.Triad(at, goal, length, true);
    painter.TriadSprings(at, current, goal, length);
    painter.Triad(at, current, 0.8_r * length, false);
    painter.Label(
        at + (goal * mochi::BasisVector<real, 3>(2)) * length, "Pivot rotation", kTargetColor);
  } else if constexpr (std::is_same_v<C, DeformableNodePositionConstraintPrefab>) {
    // Node positions are not staged, so the target is tied to the actor's origin and labeled with
    // the node it moves.
    StagedActor const* a = staged(c.actorName);
    if (a) {
      painter.Dashed(a->worldTransform.GetTranslation(), c.position, kTieColor);
    }
    painter.Dot(c.position, kTargetColor, 1.6f);
    if (painter.Labels()) {
      painter.Label(c.position, "Node " + std::to_string(c.nodeIndex) + " target", kTargetColor);
    }
  } else if constexpr (std::is_same_v<C, DeformableNodeToRigidConstraintPrefab>) {
    StagedActor const* rigid = staged(c.rigidActorName);
    StagedActor const* soft = staged(c.deformableActorName);
    if (!rigid || !soft) {
      return;
    }
    Real3 const anchor = c.fixToDeformablePos
        ? rigid->worldTransform.GetTranslation()
        : rigid->worldTransform.TransformPoint(c.rigidLocalPos * scaleOf(c.rigidActorName));
    painter.Dashed(anchor, soft->worldTransform.GetTranslation(), kTieColor);
    painter.Dot(anchor, kAnchorAColor, 1.4f);
    if (painter.Labels()) {
      painter.Label(
          anchor,
          c.findClosest ? std::string("Closest node")
                        : "Node " + std::to_string(c.deformableNodeIndex),
          kAnchorAColor);
    }
  } else if constexpr (std::is_same_v<C, DeformableNodeToDeformableNodeConstraintPrefab>) {
    StagedActor const* a = staged(c.actorNameA);
    StagedActor const* b = staged(c.actorNameB);
    if (!a || !b) {
      return;
    }
    Real3 const pa = a->worldTransform.GetTranslation();
    Real3 const pb = b->worldTransform.GetTranslation();
    painter.Dashed(pa, pb, kTieColor);
    painter.Dot(pa, kAnchorAColor);
    painter.Dot(pb, kAnchorBColor);
    if (painter.Labels()) {
      painter.Label(
          (pa + pb) * 0.5_r,
          c.findClosest
              ? std::string("Closest nodes")
              : "Nodes " + std::to_string(c.nodeIndexA) + " & " + std::to_string(c.nodeIndexB),
          kTieColor);
    }
  }
}

} // namespace

std::optional<ArticulatedJointFrames> ComputeArticulatedJointFrames(
    mochi::prefab::ArticulatedActorPrefab const& actor,
    int jointIndex,
    SceneStage const& stage) {
  int const numLinks = static_cast<int>(actor.links.size());
  if (jointIndex < 0 || jointIndex >= numLinks ||
      jointIndex >= static_cast<int>(actor.joints.size())) {
    return std::nullopt;
  }
  auto const linkWorld = [&](int link) -> std::optional<TransformRT> {
    StagedActor const* staged = stage.FindActorByName(
        std::string(actor.name) + "/" + std::string(actor.links[static_cast<size_t>(link)].name));
    return staged ? std::optional<TransformRT>(staged->worldTransform) : std::nullopt;
  };
  auto const child = linkWorld(jointIndex);
  if (!child) {
    return std::nullopt;
  }
  auto const& joint = actor.joints[static_cast<size_t>(jointIndex)];
  auto const& link = actor.links[static_cast<size_t>(jointIndex)];
  int const parentLink = link.parentLink;
  std::optional<TransformRT> parent;
  if (parentLink >= 0 && parentLink < numLinks) {
    parent = linkWorld(parentLink);
  } else {
    parent = TransformRT(mochi::Normalize(actor.rotation), actor.translation);
  }
  if (!parent) {
    return std::nullopt;
  }
  ArticulatedJointFrames frames;
  frames.scale = actor.scale;
  frames.type = joint.type;
  frames.parent = *parent * Scaled(joint.parentLinkFromJoint, actor.scale);
  frames.child = *child * mochi::Invert(Scaled(link.parentJointFromLink, actor.scale));
  real const axisLength = mochi::Norm(joint.axis);
  frames.axis = axisLength > 0 ? joint.axis * (1_r / axisLength) : Real3{0_r, 0_r, 0_r};
  return frames;
}

real MeasureJointDof(ArticulatedJointFrames const& frames, int dofIndex) {
  TransformRT const relative = mochi::Invert(frames.parent) * frames.child;
  switch (frames.type) {
    case mochi::ArticulatedJointType::Revolute:
      return TwistAngle(relative.GetRotation(), frames.axis);
    case mochi::ArticulatedJointType::Prismatic:
      return mochi::Dot(relative.GetTranslation(), frames.axis);
    case mochi::ArticulatedJointType::Free:
      if (dofIndex >= 0 && dofIndex < 3) {
        return relative.GetTranslation()[dofIndex];
      }
      break;
    case mochi::ArticulatedJointType::Spherical:
    case mochi::ArticulatedJointType::Hard:
    case mochi::ArticulatedJointType::Cycle:
    case mochi::ArticulatedJointType::Count:
      break;
  }
  return 0_r;
}

void DrawConstraint(
    ConstraintDrawContext const& ctx,
    ConstraintRef ref,
    ConstraintEmphasis emphasis) {
  Painter const painter(ctx, emphasis);
  VisitConstraint(
      ctx.prefab.constraints, ref, [&](auto const& constraint) { DrawTyped(painter, constraint); });
}

} // namespace superdex::studio
