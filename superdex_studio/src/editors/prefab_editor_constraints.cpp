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

#include "editors/prefab_editor_constraints.h"

#include "app/app.h"
#include "assets/mochi_prefab_asset.h"
#include "editors/prefab_editor.h"
#include "rendering/constraint_visualization.h"
#include "rendering/scene_stage.h"
#include "ui/imgui_widgets.h"

#include <mochi_core/articulated_body/articulated_body.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/math_utils.h>

#include <imguios/fonts/icons_font_awesome5.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

using mochi::operator""_r;

namespace superdex::studio {

using mochi::ConstraintType;
using mochi::Quaternion;
using mochi::real;
using mochi::Real3;
using mochi::TransformRT;
using mochi::prefab::ArticulatedActorPrefab;

namespace {

constexpr float kMaxValue = std::numeric_limits<float>::max();

// Whether constraint prefab type `C` acts on one DoF of its joint.
template <class C>
constexpr bool kIsSingleDofConstraint =
    std::is_same_v<C, mochi::prefab::ArticulatedSingleDofTargetConstraintPrefab> ||
    std::is_same_v<C, mochi::prefab::ArticulatedSingleDofRangeConstraintPrefab>;

// The joint type whose units (N/m vs N m/rad, m vs deg) a value of this kind is shown in.
mochi::ArticulatedJointType UnitJointType(bool rotation) {
  return rotation ? mochi::ArticulatedJointType::Revolute : mochi::ArticulatedJointType::Prismatic;
}

// Picks an actor of the right kind, or one inside a nested prefab (whose kind is not tracked).
bool ActorCombo(
    char const* label,
    mochi::DynamicString& value,
    std::vector<std::string> const& names,
    std::vector<std::string> const& nested) {
  std::vector<std::string> candidates = names;
  candidates.insert(candidates.end(), nested.begin(), nested.end());
  return ImGui::ActorNameCombo(label, value, candidates);
}

std::string JointLabel(ArticulatedActorPrefab const& actor, int jointIndex) {
  return JointDisplayName(&actor, jointIndex) + " (" +
      SReflect::EnumToString(actor.joints[static_cast<size_t>(jointIndex)].type) + ")";
}

// Joint picker for `actor`, listing only the joints the constraint fits (the current joint is
// always listed so a mismatched one can be seen and replaced).
bool JointCombo(
    char const* label,
    int& jointIndex,
    ArticulatedActorPrefab const* actor,
    bool singleDof) {
  int const numJoints = actor ? static_cast<int>(actor->joints.size()) : 0;
  bool const valid = jointIndex >= 0 && jointIndex < numJoints;
  std::string const preview =
      valid ? JointLabel(*actor, jointIndex) : "joint " + std::to_string(jointIndex) + " (missing)";
  bool const mismatch = !valid ||
      !JointFitsConstraint(actor->joints[static_cast<size_t>(jointIndex)].type, singleDof);
  if (mismatch) {
    ImGui::PushStyleColor(ImGuiCol_FrameBg, kNameConflictColor);
    ImGui::PushStyleColor(ImGuiCol_Button, kNameConflictColor);
  }
  bool const open = ImGui::BeginCombo(label, preview.c_str());
  if (mismatch) {
    ImGui::PopStyleColor(2);
  }
  bool changed = false;
  if (open) {
    for (int j = 0; j < numJoints; ++j) {
      if (j != jointIndex &&
          !JointFitsConstraint(actor->joints[static_cast<size_t>(j)].type, singleDof)) {
        continue;
      }
      bool const selected = j == jointIndex;
      if (ImGui::Selectable(JointLabel(*actor, j).c_str(), selected) && !selected) {
        jointIndex = j;
        changed = true;
      }
    }
    ImGui::EndCombo();
  }
  return changed;
}

char const* DofName(mochi::ArticulatedJointType type, int dofIndex) {
  switch (type) {
    case mochi::ArticulatedJointType::Prismatic:
      return dofIndex == 0 ? "Translation" : "?";
    case mochi::ArticulatedJointType::Revolute:
      return dofIndex == 0 ? "Rotation" : "?";
    case mochi::ArticulatedJointType::Free: {
      constexpr std::array<char const*, 3> kNames{
          "Translation X", "Translation Y", "Translation Z"};
      return dofIndex >= 0 && dofIndex < 3 ? kNames[static_cast<size_t>(dofIndex)] : "?";
    }
    case mochi::ArticulatedJointType::Spherical:
    case mochi::ArticulatedJointType::Hard:
    case mochi::ArticulatedJointType::Cycle:
    case mochi::ArticulatedJointType::Count:
      break;
  }
  return "?";
}

bool DofCombo(char const* label, int& dofIndex, mochi::ArticulatedJointType type) {
  bool changed = false;
  if (ImGui::BeginCombo(label, DofName(type, dofIndex))) {
    auto const [translationDofs, rotationDofs] = mochi::articulated::GetJointTypeNumDofs(type);
    for (int d = 0; d < translationDofs + rotationDofs; ++d) {
      if (!IsSingleDofConstraintDof(type, d)) {
        continue;
      }
      bool const selected = d == dofIndex;
      if (ImGui::Selectable(DofName(type, d), selected) && !selected) {
        dofIndex = d;
        changed = true;
      }
    }
    ImGui::EndCombo();
  }
  return changed;
}

// A rotation vector [rad] edited as roll / pitch / yaw [deg].
bool DragRotationVector(char const* label, Real3& rotationVector) {
  Quaternion q = Quaternion::FromRotationVector(rotationVector);
  if (ImGui::DragQuaternion(label, q, ImGui::QuaternionMode::RPY)) {
    rotationVector = mochi::Normalize(q).ToRotationVector();
    return true;
  }
  return false;
}

// An angle range [rad] edited as min / max [deg].
bool DragAngleRange(char const* label, mochi::Real2& range) {
  float v[2] = {
      static_cast<float>(range[0] * mochi::kDegreesPerRadian),
      static_cast<float>(range[1] * mochi::kDegreesPerRadian)};
  if (ImGui::DragFloat2(label, v, 0.5f, -360.0f, 360.0f, GetUnitFormat(UnitFormat::Degrees))) {
    range[0] = static_cast<real>(v[0]) * mochi::kRadiansPerDegree;
    range[1] = static_cast<real>(v[1]) * mochi::kRadiansPerDegree;
    return true;
  }
  return false;
}

bool DragPosition(char const* label, Real3& position) {
  return ImGui::DragRealXYZ(label, position, 0.001f, 0.0f, 0.0f, GetUnitFormat(UnitFormat::Length));
}

// Stiffness, damping and saturation, in the units of a translation or a rotation.
bool CommonParams(mochi::ConstraintParams& params, bool rotation) {
  bool changed = false;
  mochi::ArticulatedJointType const units = UnitJointType(rotation);
  ImGui::SeparatorText("Spring");
  changed |= ImGui::DragReal(
      "Stiffness",
      &params.stiffness,
      0.01f,
      0.0f,
      kMaxValue,
      GetUnitFormat(UnitFormat::Stiffness, units, params.stiffness));
  changed |= ImGui::DragReal(
      "Damping",
      &params.damping,
      0.001f,
      0.0f,
      kMaxValue,
      GetUnitFormat(UnitFormat::Damping, units, params.damping));
  // Saturation caps the spring force at stiffness x saturation; negative turns it off.
  bool saturates = params.saturation > 0;
  if (ImGui::Checkbox("##saturates", &saturates)) {
    params.saturation = saturates ? (rotation ? 10_r * mochi::kRadiansPerDegree : 0.01_r) : -1_r;
    changed = true;
  }
  ImGui::ItemTooltipWrapped(
      "Caps the spring force at Stiffness x Saturation: past this deviation the constraint pulls no "
      "harder. Off: no cap.");
  ImGui::SameLine();
  ImGui::BeginDisabled(!saturates);
  real shown = saturates ? params.saturation : 0_r;
  float const scale = rotation ? mochi::kDegreesPerRadian : 1.0f;
  if (ImGui::DragReal(
          "Saturation",
          &shown,
          rotation ? 0.1f : 0.0005f,
          0.0f,
          kMaxValue,
          GetUnitFormat(UnitFormat::Position, units),
          scale) &&
      saturates && shown > 0_r) {
    params.saturation = shown;
    changed = true;
  }
  ImGui::EndDisabled();
  return changed;
}

template <class C>
bool EditFields(ConstraintEditContext const& ctx, C& c) {
  using namespace mochi::prefab;
  ConstraintActorIndex const& actors = ctx.actors;
  bool changed = false;
  bool rotation = false;

  if constexpr (kIsJointConstraint<C>) {
    ImGui::SeparatorText("Joint");
    changed |= ActorCombo("Actor", c.actorName, actors.articulated, actors.nested);
    ArticulatedActorPrefab const* actor =
        FindArticulatedActor(ctx.prefab, std::string_view(c.actorName));
    if (JointCombo("Joint", c.jointIndex, actor, kIsSingleDofConstraint<C>)) {
      changed = true;
      if constexpr (kIsSingleDofConstraint<C>) {
        c.dofIndex = 0;
      }
    }
    bool const validJoint =
        actor && c.jointIndex >= 0 && c.jointIndex < static_cast<int>(actor->joints.size());
    auto const jointType = validJoint ? actor->joints[static_cast<size_t>(c.jointIndex)].type
                                      : mochi::ArticulatedJointType::Revolute;
    if constexpr (kIsSingleDofConstraint<C>) {
      changed |= DofCombo("DoF", c.dofIndex, jointType);
      rotation = IsRotationDof(jointType, c.dofIndex);
      auto const units = UnitJointType(rotation);
      float const scale = rotation ? mochi::kDegreesPerRadian : 1.0f;
      float const speed = rotation ? 0.5f : 0.0005f;
      char const* format = GetUnitFormat(UnitFormat::Position, units);
      ImGui::SeparatorText(
          std::is_same_v<C, ArticulatedSingleDofTargetConstraintPrefab> ? "Target" : "Range");
      if constexpr (std::is_same_v<C, ArticulatedSingleDofTargetConstraintPrefab>) {
        changed |= ImGui::DragReal("Target", &c.targetValue, speed, 0.0f, 0.0f, format, scale);
      } else {
        changed |= ImGui::DragReal("Min", &c.minValue, speed, 0.0f, 0.0f, format, scale);
        changed |= ImGui::DragReal("Max", &c.maxValue, speed, 0.0f, 0.0f, format, scale);
      }
      if (validJoint) {
        if (auto const frames = ComputeArticulatedJointFrames(*actor, c.jointIndex, ctx.stage)) {
          real const current = MeasureJointDof(*frames, c.dofIndex) /
              (rotation || actor->scale == 0 ? 1_r : actor->scale);
          char text[64];
          std::snprintf(text, sizeof(text), format, static_cast<float>(current) * scale);
          ImGui::LabelText("Current", "%s", text);
        }
      }
    } else {
      rotation = true;
      if constexpr (std::is_same_v<C, Articulated3dRotationTargetConstraintPrefab>) {
        ImGui::SeparatorText("Target");
        changed |= ImGui::DragQuaternion("Target", c.target, ImGui::QuaternionMode::RPY);
      } else {
        ImGui::SeparatorText("Range");
        char const* format = GetUnitFormat(UnitFormat::Degrees);
        changed |= ImGui::DragRealXYZ(
            "Min", c.minValues, 0.5f, 0.0f, 0.0f, format, 0, mochi::kDegreesPerRadian);
        changed |= ImGui::DragRealXYZ(
            "Max", c.maxValues, 0.5f, 0.0f, 0.0f, format, 0, mochi::kDegreesPerRadian);
      }
    }
  } else if constexpr (kIsPairConstraint<C>) {
    constexpr bool kDeformable = std::is_same_v<C, DeformableNodeToDeformableNodeConstraintPrefab>;
    auto const& names = kDeformable ? actors.deformable : actors.rigid;
    ImGui::SeparatorText("Actors");
    changed |= ActorCombo("Actor A", c.actorNameA, names, actors.nested);
    changed |= ActorCombo("Actor B", c.actorNameB, names, actors.nested);
    if constexpr (std::is_same_v<C, RigidSphericalJointConstraintPrefab>) {
      ImGui::SeparatorText("Anchors");
      changed |= DragPosition("Anchor on A", c.localPosA);
      changed |= DragPosition("Anchor on B", c.localPosB);
      StagedActor const* a = ctx.stage.FindActorByName(std::string_view(c.actorNameA));
      StagedActor const* b = ctx.stage.FindActorByName(std::string_view(c.actorNameB));
      ImGui::BeginDisabled(!a || !b);
      if (ImGui::Button("Move Anchor A onto Anchor B") && a && b) {
        // Where B's anchor rests, in A's frame: the joint then starts closed. Mochi scales local
        // anchors by each actor's metric scale.
        real const scaleA =
            GetConstraintActorMetricScale(ctx.prefab, std::string_view(c.actorNameA));
        real const scaleB =
            GetConstraintActorMetricScale(ctx.prefab, std::string_view(c.actorNameB));
        c.localPosA = a->defaultWorldTransform.TransformPointInverse(
                          b->defaultWorldTransform.TransformPoint(c.localPosB * scaleB)) /
            scaleA;
        changed = true;
      }
      ImGui::EndDisabled();
    } else if constexpr (std::is_same_v<C, RigidPrismaticJointConstraintPrefab>) {
      ImGui::SeparatorText("Axis");
      changed |= ImGui::DragRealXYZ("Free Axis", c.freeAxis, 0.01f);
      ImGui::ItemTooltipWrapped(
          "The direction B slides in, in world coordinates at rest. It then turns with actor A.");
      changed |= ImGui::DragOptionalReal(
          "Min", c.min, 0.0005f, 0.0f, 0.0f, GetUnitFormat(UnitFormat::Length));
      changed |= ImGui::DragOptionalReal(
          "Max", c.max, 0.0005f, 0.0f, 0.0f, GetUnitFormat(UnitFormat::Length));
    } else if constexpr (std::is_same_v<C, JointRotationTrackingConstraintPrefab>) {
      rotation = true;
      ImGui::SeparatorText("Frame");
      changed |= DragRotationVector("Reference Frame", c.refFrameRotVec);
      ImGui::ItemTooltipWrapped(
          "The frame the rotation deviation is measured in, in world coordinates at rest. The "
          "target is B's rest rotation relative to A.");
    } else if constexpr (std::is_same_v<C, JointRotationRangeConstraintPrefab>) {
      rotation = true;
      ImGui::SeparatorText("Range");
      changed |= DragRotationVector("Reference Frame", c.refFrameRotVec);
      changed |= ImGui::Checkbox("Range Around Rest", &c.rangeAroundRest);
      ImGui::ItemTooltipWrapped(
          "On: the ranges are measured from the actors' rest rotation. Off: they are absolute, in "
          "the reference frame.");
      changed |= DragAngleRange("X Range", c.angleRangeX);
      changed |= DragAngleRange("Y Range", c.angleRangeY);
      changed |= DragAngleRange("Z Range", c.angleRangeZ);
    } else if constexpr (kDeformable) {
      ImGui::SeparatorText("Nodes");
      changed |= ImGui::Checkbox("Find Closest", &c.findClosest);
      ImGui::ItemTooltipWrapped(
          "Picks the boundary node of each actor closest to the midpoint between their origins.");
      ImGui::BeginDisabled(c.findClosest);
      changed |= ImGui::InputInt("Node on A", &c.nodeIndexA);
      changed |= ImGui::InputInt("Node on B", &c.nodeIndexB);
      ImGui::EndDisabled();
    }
  } else if constexpr (std::is_same_v<C, DeformableNodeToRigidConstraintPrefab>) {
    ImGui::SeparatorText("Actors");
    changed |= ActorCombo("Rigid Actor", c.rigidActorName, actors.rigid, actors.nested);
    changed |=
        ActorCombo("Deformable Actor", c.deformableActorName, actors.deformable, actors.nested);
    ImGui::SeparatorText("Attachment");
    changed |= ImGui::Checkbox("Find Closest", &c.findClosest);
    ImGui::ItemTooltipWrapped("Picks the boundary node closest to the attachment point.");
    ImGui::BeginDisabled(c.findClosest);
    changed |= ImGui::InputInt("Node", &c.deformableNodeIndex);
    ImGui::EndDisabled();
    changed |= ImGui::Checkbox("Attach Where the Node Rests", &c.fixToDeformablePos);
    ImGui::BeginDisabled(c.fixToDeformablePos && !c.findClosest);
    changed |= DragPosition("Point on Rigid", c.rigidLocalPos);
    ImGui::EndDisabled();
  } else {
    ImGui::SeparatorText("Actor");
    bool const deformable = std::is_same_v<C, DeformableNodePositionConstraintPrefab>;
    changed |= ActorCombo(
        "Actor", c.actorName, deformable ? actors.deformable : actors.rigid, actors.nested);
    StagedActor const* staged = ctx.stage.FindActorByName(std::string_view(c.actorName));
    real const metricScale =
        GetConstraintActorMetricScale(ctx.prefab, std::string_view(c.actorName));
    if constexpr (std::is_same_v<C, RigidPivotPositionConstraintPrefab>) {
      ImGui::SeparatorText("Pivot");
      changed |= DragPosition("Pivot", c.localPosition);
      changed |= DragPosition("Target", c.targetPosition);
      ImGui::BeginDisabled(staged == nullptr);
      if (ImGui::Button("Set Target to Pivot") && staged) {
        c.targetPosition =
            staged->defaultWorldTransform.TransformPoint(c.localPosition * metricScale);
        changed = true;
      }
      ImGui::EndDisabled();
    } else if constexpr (std::is_same_v<C, RigidPivotToRigidTargetConstraintPrefab>) {
      ImGui::SeparatorText("Pivot");
      changed |= DragPosition("Pivot", c.localPosition);
      changed |= ImGui::DragTransformRT("Target Pose", c.targetTransform);
      ImGui::BeginDisabled(staged == nullptr);
      if (ImGui::Button("Set Target to Rest Pose") && staged) {
        c.targetTransform = staged->defaultWorldTransform;
        changed = true;
      }
      ImGui::EndDisabled();
    } else if constexpr (std::is_same_v<C, RigidPivotRotationConstraintPrefab>) {
      rotation = true;
      ImGui::SeparatorText("Pivot");
      changed |= DragRotationVector("Pivot Frame", c.localRotation);
      changed |= DragRotationVector("Target", c.targetRotation);
      ImGui::BeginDisabled(staged == nullptr);
      if (ImGui::Button("Set Target to Rest Rotation") && staged) {
        c.targetRotation = (staged->defaultWorldTransform.GetRotation() *
                            Quaternion::FromRotationVector(c.localRotation))
                               .ToRotationVector();
        changed = true;
      }
      ImGui::EndDisabled();
    } else if constexpr (std::is_same_v<C, DeformableNodePositionConstraintPrefab>) {
      ImGui::SeparatorText("Node");
      changed |= ImGui::InputInt("Node", &c.nodeIndex);
      changed |= DragPosition("Target", c.position);
    }
  }
  changed |= CommonParams(c, rotation);
  return changed;
}

} // namespace

bool ConstraintEditor(
    ConstraintEditContext const& ctx,
    mochi::prefab::ConstraintLists& lists,
    ConstraintRef ref) {
  bool changed = false;
  VisitConstraint(lists, ref, [&](auto& constraint) { changed = EditFields(ctx, constraint); });
  return changed;
}

void AttachConstraint(
    ConstraintEditContext const& ctx,
    mochi::prefab::ConstraintLists& lists,
    ConstraintRef ref,
    ConstraintTargets const& targets) {
  auto const name = [](std::vector<std::string> const& list, size_t i) {
    return i < list.size() ? mochi::DynamicString{list[i]} : mochi::DynamicString{};
  };
  auto const rest = [&](std::vector<std::string> const& list, size_t i) -> StagedActor const* {
    return i < list.size() ? ctx.stage.FindActorByName(list[i]) : nullptr;
  };
  VisitConstraint(lists, ref, [&](auto& c) {
    using C = std::decay_t<decltype(c)>;
    using namespace mochi::prefab;
    if constexpr (kIsJointConstraint<C>) {
      c.actorName = mochi::DynamicString{targets.articulatedActor};
      ArticulatedActorPrefab const* actor =
          FindArticulatedActor(ctx.prefab, targets.articulatedActor);
      if (actor == nullptr) {
        return;
      }
      auto const fits = [&](int j) {
        return j >= 0 && j < static_cast<int>(actor->joints.size()) &&
            JointFitsConstraint(
                   actor->joints[static_cast<size_t>(j)].type, kIsSingleDofConstraint<C>);
      };
      // The preferred joint, else the first that fits after the root (usually the free joint that
      // places the actor), else the root.
      int joint = fits(targets.joint) ? targets.joint : -1;
      for (int j = 1; joint < 0 && j < static_cast<int>(actor->joints.size()); ++j) {
        joint = fits(j) ? j : -1;
      }
      if (joint < 0) {
        joint = fits(0) ? 0 : std::max(targets.joint, 0);
      }
      c.jointIndex = joint;
      if (joint >= static_cast<int>(actor->joints.size())) {
        return;
      }
      auto const& j = actor->joints[static_cast<size_t>(joint)];
      if constexpr (std::is_same_v<C, ArticulatedSingleDofRangeConstraintPrefab>) {
        // The joint's own limits, along its axis; a sensible span when it has none.
        c.dofIndex = 0;
        bool const rotation = IsRotationDof(j.type, 0);
        Real3 const axis =
            mochi::Norm(j.axis) > 0_r ? mochi::Normalize(j.axis) : Real3{1_r, 0_r, 0_r};
        real const span = rotation ? 45_r * mochi::kRadiansPerDegree : 0.05_r;
        c.minValue =
            j.minLimit && mochi::IsFinite(*j.minLimit) ? mochi::Dot(*j.minLimit, axis) : -span;
        c.maxValue =
            j.maxLimit && mochi::IsFinite(*j.maxLimit) ? mochi::Dot(*j.maxLimit, axis) : span;
        // A one-sided limit outside the default span, or limits authored reversed.
        if (c.minValue > c.maxValue) {
          std::swap(c.minValue, c.maxValue);
        }
      } else if constexpr (std::is_same_v<C, Articulated3dRotationRangeConstraintPrefab>) {
        real const span = 45_r * mochi::kRadiansPerDegree;
        c.minValues =
            j.minLimit && mochi::IsFinite(*j.minLimit) ? *j.minLimit : Real3{-span, -span, -span};
        c.maxValues =
            j.maxLimit && mochi::IsFinite(*j.maxLimit) ? *j.maxLimit : Real3{span, span, span};
      } else if constexpr (kIsSingleDofConstraint<C>) {
        c.dofIndex = 0;
      }
    } else if constexpr (kIsPairConstraint<C>) {
      constexpr bool kDeformable =
          std::is_same_v<C, DeformableNodeToDeformableNodeConstraintPrefab>;
      auto const& list = kDeformable ? targets.deformable : targets.rigid;
      c.actorNameA = name(list, 0);
      c.actorNameB = name(list, 1);
      StagedActor const* a = rest(list, 0);
      StagedActor const* b = rest(list, 1);
      if constexpr (std::is_same_v<C, RigidSphericalJointConstraintPrefab>) {
        // The ball sits at B's origin, so the joint starts closed.
        if (a && b) {
          c.localPosA = a->defaultWorldTransform.TransformPointInverse(
                            b->defaultWorldTransform.GetTranslation()) /
              GetConstraintActorMetricScale(ctx.prefab, std::string_view(c.actorNameA));
        }
      } else if constexpr (std::is_same_v<C, RigidPrismaticJointConstraintPrefab>) {
        Real3 axis{0_r, 0_r, 1_r};
        if (a && b) {
          Real3 const d =
              b->defaultWorldTransform.GetTranslation() - a->defaultWorldTransform.GetTranslation();
          if (!mochi::NearZero(mochi::Norm(d))) {
            axis = mochi::Normalize(d);
          }
        }
        c.freeAxis = axis;
      } else if constexpr (std::is_same_v<C, JointRotationRangeConstraintPrefab>) {
        // A loose cone around the rest rotation rather than the unbounded default.
        real const span = 30_r * mochi::kRadiansPerDegree;
        c.rangeAroundRest = true;
        c.angleRangeX = mochi::Real2{-span, span};
        c.angleRangeY = mochi::Real2{-span, span};
        c.angleRangeZ = mochi::Real2{-span, span};
      } else if constexpr (kDeformable) {
        c.findClosest = true;
      }
    } else if constexpr (std::is_same_v<C, DeformableNodeToRigidConstraintPrefab>) {
      c.rigidActorName = name(targets.rigid, 0);
      c.deformableActorName = name(targets.deformable, 0);
      c.findClosest = true;
    } else {
      bool const deformable = std::is_same_v<C, DeformableNodePositionConstraintPrefab>;
      auto const& list = deformable ? targets.deformable : targets.rigid;
      c.actorName = name(list, 0);
      if constexpr (std::is_same_v<C, DeformableNodePositionConstraintPrefab>) {
        c.nodeIndex = 0;
      }
      StagedActor const* a = rest(list, 0);
      if (a == nullptr) {
        return;
      }
      if constexpr (std::is_same_v<C, RigidPivotPositionConstraintPrefab>) {
        c.targetPosition = a->defaultWorldTransform.GetTranslation();
      } else if constexpr (std::is_same_v<C, RigidPivotToRigidTargetConstraintPrefab>) {
        c.targetTransform = a->defaultWorldTransform;
      } else if constexpr (std::is_same_v<C, RigidPivotRotationConstraintPrefab>) {
        c.targetRotation = a->defaultWorldTransform.GetRotation().ToRotationVector();
      } else if constexpr (std::is_same_v<C, DeformableNodePositionConstraintPrefab>) {
        c.position = a->defaultWorldTransform.GetTranslation();
      }
    }
  });
}

//--------------------------------------------------------------------------------------------------
// PrefabEditor: constraints
//--------------------------------------------------------------------------------------------------

ConstraintActorIndex PrefabEditor::BuildConstraintActorIndex() const {
  std::vector<std::string> nested;
  for (auto const& staged : _stage.GetActors()) {
    if (staged.source == StagedActorSource::NestedPrefab) {
      nested.push_back(staged.name);
    }
  }
  return superdex::studio::BuildConstraintActorIndex(_prefabAsset->GetPrefab(), nested);
}

void PrefabEditor::ShowCreateActorMenuItems(HierarchyEdits& edits) {
  if (ImGui::Selectable("Create Rigid")) {
    edits.createRigid = true;
  }
  if (ImGui::Selectable("Create Soft")) {
    edits.createSoft = true;
  }
  if (ImGui::Selectable("Create Articulated")) {
    edits.createArticulated = true;
  }
  if (ImGui::Selectable("Create Nested Prefab")) {
    edits.createNested = true;
  }
}

void PrefabEditor::ShowCreateConstraintMenuItems(Selection const& context, HierarchyEdits& edits) {
  // A row's own menu offers what attaches to that kind of actor; the toolbar offers everything.
  auto const offered = [&](ConstraintCategory category) {
    switch (context.kind) {
      case Selection::Kind::ArticulatedActor:
      case Selection::Kind::ArticulatedLink:
        return category == ConstraintCategory::ArticulatedJoint;
      case Selection::Kind::Rigid:
        return category != ConstraintCategory::ArticulatedJoint;
      case Selection::Kind::Soft:
        return category == ConstraintCategory::Deformable;
      case Selection::Kind::NestedPrefab:
      case Selection::Kind::Constraint:
      case Selection::Kind::None:
        break;
    }
    return true;
  };
  struct Group {
    ConstraintCategory category;
    char const* label;
  };
  constexpr std::array kGroups{
      Group{ConstraintCategory::ArticulatedJoint, "Articulated Joint"},
      Group{ConstraintCategory::RigidPair, "Between Two Rigid Actors"},
      Group{ConstraintCategory::RigidPivot, "Rigid Actor to World"},
      Group{ConstraintCategory::Deformable, "Deformable"},
  };
  bool first = true;
  for (Group const& group : kGroups) {
    if (!offered(group.category)) {
      continue;
    }
    if (!first) {
      ImGui::Spacing();
    }
    first = false;
    ImGui::SeparatorText(group.label);
    for (ConstraintType type : PrefabConstraintTypes()) {
      if (GetConstraintCategory(type) != group.category) {
        continue;
      }
      if (ImGui::Selectable(ConstraintTypeLabel(type))) {
        edits.createConstraint = type;
        edits.constraintContext = context;
      }
      ImGui::ItemTooltipWrapped(ConstraintTypeDescription(type));
    }
  }
}

void PrefabEditor::ShowHierarchySettingsCog() {
  // Mirrors the Bot Hierarchy's cog: drawn over the right end of the first row.
  ImVec2 const backup = ImGui::GetCursorPos();
  float const gearWidth = ImGui::CalcTextSize(ICON_FA_COG).x;
  ImGui::SetCursorPos(ImGui::GetCursorStartPos());
  ImGui::SetCursorPosX(ImGui::GetWindowContentRegionMax().x - gearWidth);
  if (ImGui::TextButton(ICON_FA_COG)) {
    ImGui::OpenPopup("##PrefabHierarchySettings");
  }
  ImGui::SetCursorPos(backup);
  if (ImGui::BeginPopup("##PrefabHierarchySettings")) {
    bool& show = _studio->GetAppSettings().prefabEditor.showConstraintsUnderActors;
    ImGui::MenuItem("Show Constraints Under Actors", nullptr, &show);
    ImGui::SetItemTooltip(
        "List each constraint on the Actors tab too, under the link or actors it constrains.");
    ImGui::EndPopup();
  }
}

void PrefabEditor::ShowHierarchyToolbar(
    bool simulating,
    HierarchyEdits& edits,
    bool withActorButton) {
  ImGui::BeginDisabled(simulating);
  if (withActorButton) {
    if (ImGui::Button(ICON_FA_PLUS " Actor")) {
      ImGui::OpenPopup("CreateActorPopup");
    }
    ImGui::SetItemTooltip("Create a rigid, soft, articulated or nested prefab actor.");
    ImGui::SameLine();
  }
  if (ImGui::Button(ICON_FA_PLUS " Constraint")) {
    ImGui::OpenPopup("CreateConstraintPopup");
  }
  ImGui::SetItemTooltip(
      "Create a constraint. It attaches to the selected actors or link when they fit, otherwise to "
      "the first actors that do.");
  ImGui::EndDisabled();
  if (ImGui::BeginPopup("CreateActorPopup")) {
    ShowCreateActorMenuItems(edits);
    ImGui::EndPopup();
  }
  if (ImGui::BeginPopup("CreateConstraintPopup")) {
    // From the toolbar every type is offered; the selection only decides what it attaches to.
    ShowCreateConstraintMenuItems({}, edits);
    if (edits.createConstraint != ConstraintType::None) {
      Selection const* const primary = PrimarySelection();
      edits.constraintContext = primary ? *primary : Selection{};
    }
    ImGui::EndPopup();
  }
}

std::string_view PrefabEditor::PrefabActorName(Selection::Kind kind, int index) const {
  auto const& prefab = _prefabAsset->GetPrefab();
  auto const nameAt = [index](auto const& array) -> std::string_view {
    return index >= 0 && index < static_cast<int>(array.size())
        ? std::string_view(array[static_cast<size_t>(index)].name)
        : std::string_view();
  };
  switch (kind) {
    case Selection::Kind::Rigid:
      return nameAt(prefab.actors.rigid);
    case Selection::Kind::Soft:
      return nameAt(prefab.actors.soft);
    case Selection::Kind::ArticulatedActor:
    case Selection::Kind::ArticulatedLink:
      return nameAt(prefab.actors.articulated);
    case Selection::Kind::NestedPrefab:
      return nameAt(prefab.prefabs);
    case Selection::Kind::Constraint:
    case Selection::Kind::None:
      break;
  }
  return {};
}

void PrefabEditor::ShowConstraintRow(
    ConstraintRef ref,
    char const* name,
    char const* type,
    bool simulating,
    HierarchyEdits& edits) {
  ImGui::PushID(static_cast<int>(ref.type));
  ImGui::PushID(ref.index);
  ImGui::TableNextRow();
  ImGui::TableNextColumn();
  Selection const* const primary = PrimarySelection();
  bool const selected =
      primary && primary->kind == Selection::Kind::Constraint && primary->constraint == ref;
  std::string const error =
      ValidateConstraint(_prefabAsset->GetPrefab(), _constraintActorIndex, ref);
  std::string const label = std::string(ICON_FA_LINK " ") + name;
  if (!error.empty()) {
    ImGui::PushStyleColor(ImGuiCol_Text, kErrorTextColor);
  }
  bool const clicked =
      ImGui::Selectable(label.c_str(), selected, ImGuiSelectableFlags_SpanAllColumns);
  if (!error.empty()) {
    ImGui::PopStyleColor();
    ImGui::SetItemTooltip("%s", error.c_str());
  }
  if (clicked) {
    SetSingleSelection({Selection::Kind::Constraint, -1, -1, ref});
    FocusWindowIfOpen("Details");
  }
  if (ImGui::IsItemHovered()) {
    _hoveredConstraint = ref;
    HighlightConstraintActors(ref);
  }
  if (!simulating &&
      ImGui::BeginPopupContextItem("ConstraintPopup", ImGuiPopupFlags_MouseButtonRight)) {
    if (ImGui::Selectable("Duplicate")) {
      edits.duplicateConstraint = ref;
    }
    if (ImGui::Selectable("Delete")) {
      edits.deleteConstraint = ref;
    }
    ImGui::EndPopup();
  }
  ImGui::TableNextColumn();
  ImGui::TextDisabled("%s", type);
  ImGui::PopID();
  ImGui::PopID();
}

void PrefabEditor::ShowConstraintsTable(bool simulating, HierarchyEdits& edits) {
  auto const& prefab = _prefabAsset->GetPrefab();
  std::vector<ConstraintRef> refs = CollectConstraints(prefab.constraints);
  if (refs.empty()) {
    ImGui::TextDisabled("No constraints. Add one with " ICON_FA_PLUS " Constraint.");
    return;
  }
  constexpr ImGuiTableFlags kFlags = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
      ImGuiTableFlags_Resizable | ImGuiTableFlags_Sortable;
  // Match the actor table's styling.
  ImGui::PushStyleColor(ImGuiCol_Header, ImGui::GetStyleColorVec4(ImGuiCol_HeaderActive));
  if (!ImGui::BeginTable("##PrefabConstraints", 2, kFlags)) {
    ImGui::PopStyleColor();
    return;
  }
  ImGui::TableSetupScrollFreeze(0, 1);
  ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_DefaultSort);
  ImGui::TableSetupColumn("Type");
  ImGui::TableHeadersRow();
  // Reuse the row buffer across frames (like the actor table's) to avoid reallocating it.
  auto& rows = _constraintRowsScratch;
  rows.clear();
  for (ConstraintRef const& ref : refs) {
    rows.emplace_back(DescribeConstraint(prefab, ref), ref);
  }
  if (ImGuiTableSortSpecs* sortSpecs = ImGui::TableGetSortSpecs();
      sortSpecs && sortSpecs->SpecsCount > 0) {
    ImGuiTableColumnSortSpecs const& spec = sortSpecs->Specs[0];
    bool const ascending = spec.SortDirection == ImGuiSortDirection_Ascending;
    std::stable_sort(rows.begin(), rows.end(), [&](auto const& lhs, auto const& rhs) {
      int cmp = 0;
      if (spec.ColumnIndex == 1) {
        cmp =
            std::strcmp(ConstraintTypeLabel(lhs.second.type), ConstraintTypeLabel(rhs.second.type));
      }
      if (cmp == 0) {
        cmp = lhs.first.compare(rhs.first);
      }
      return ascending ? cmp < 0 : cmp > 0;
    });
  }
  for (auto const& [name, ref] : rows) {
    ShowConstraintRow(ref, name.c_str(), ConstraintTypeLabel(ref.type), simulating, edits);
  }
  ImGui::EndTable();
  ImGui::PopStyleColor(); // ImGuiCol_Header
}

bool PrefabEditor::ApplyConstraintEdits(HierarchyEdits const& edits) {
  if (edits.createConstraint != ConstraintType::None) {
    CreateConstraint(edits.createConstraint, edits.constraintContext);
    return true;
  }
  if (edits.duplicateConstraint) {
    DuplicateConstraintAndSelect(*edits.duplicateConstraint);
    return true;
  }
  if (edits.deleteConstraint) {
    DeleteConstraint(*edits.deleteConstraint);
    return true;
  }
  return false;
}

void PrefabEditor::DuplicateConstraintAndSelect(ConstraintRef ref) {
  if (auto const copy = DuplicateConstraint(_prefabAsset->GetPrefab().constraints, ref)) {
    SetSingleSelection({Selection::Kind::Constraint, -1, -1, *copy});
    _prefabAsset->SetDirty(true);
    GetUndoStack().PushNow();
  }
}

bool PrefabEditor::DeleteConstraint(ConstraintRef ref) {
  if (!RemoveConstraint(_prefabAsset->GetPrefab().constraints, ref)) {
    return false;
  }
  // Removing shifts the later constraints of its array, so constraint selections go stale.
  std::erase_if(
      _selections, [](Selection const& s) { return s.kind == Selection::Kind::Constraint; });
  SyncViewportSelection();
  _prefabAsset->SetDirty(true);
  GetUndoStack().PushNow();
  return true;
}

bool PrefabEditor::RenameActorInConstraints(std::string_view name, bool collides) {
  // Constraints follow the rename once it is committed, not on every keystroke: an intermediate
  // empty or colliding name would detach their references or merge them into another actor's.
  if (ImGui::IsItemActivated()) {
    _renameStartName = std::string(name);
  }
  return ImGui::IsItemDeactivatedAfterEdit() && !name.empty() && !collides &&
      RenameConstraintActor(_prefabAsset->GetPrefab().constraints, _renameStartName, name) > 0;
}

void PrefabEditor::CreateConstraint(ConstraintType type, Selection const& context) {
  auto& prefab = _prefabAsset->GetPrefab();
  // The context row (or the selection it belongs to) decides what the constraint attaches to.
  bool const contextIsSelected =
      std::any_of(_selections.begin(), _selections.end(), [&](auto const& s) {
        return s.kind == context.kind && s.actorIndex == context.actorIndex;
      });
  std::vector<Selection> picked;
  if (context.kind != Selection::Kind::None) {
    picked.push_back(context);
  }
  if (contextIsSelected || context.kind == Selection::Kind::None) {
    for (Selection const& s : _selections) {
      if (s.kind != context.kind || s.actorIndex != context.actorIndex) {
        picked.push_back(s);
      }
    }
  }
  ConstraintTargets targets;
  auto const addUnique = [](std::vector<std::string>& list, mochi::DynamicString const& name) {
    std::string value(name);
    if (!value.empty() && !mochi::Contains(list, value)) {
      list.push_back(std::move(value));
    }
  };
  for (Selection const& s : picked) {
    switch (s.kind) {
      case Selection::Kind::Rigid:
        if (s.actorIndex >= 0 && s.actorIndex < static_cast<int>(prefab.actors.rigid.size())) {
          addUnique(targets.rigid, prefab.actors.rigid[s.actorIndex].name);
        }
        break;
      case Selection::Kind::Soft:
        if (s.actorIndex >= 0 && s.actorIndex < static_cast<int>(prefab.actors.soft.size())) {
          addUnique(targets.deformable, prefab.actors.soft[s.actorIndex].name);
        }
        break;
      case Selection::Kind::ArticulatedActor:
      case Selection::Kind::ArticulatedLink:
        if (targets.articulatedActor.empty() && s.actorIndex >= 0 &&
            s.actorIndex < static_cast<int>(prefab.actors.articulated.size())) {
          targets.articulatedActor = std::string(prefab.actors.articulated[s.actorIndex].name);
          targets.joint = s.kind == Selection::Kind::ArticulatedLink ? s.linkIndex : -1;
        }
        break;
      case Selection::Kind::NestedPrefab:
      case Selection::Kind::Constraint:
      case Selection::Kind::None:
        break;
    }
  }
  // Fill what the selection left open with the prefab's first actors of each kind.
  for (auto const& actor : prefab.actors.rigid) {
    addUnique(targets.rigid, actor.name);
  }
  for (auto const& actor : prefab.actors.soft) {
    addUnique(targets.deformable, actor.name);
  }
  if (targets.articulatedActor.empty() && !prefab.actors.articulated.empty()) {
    targets.articulatedActor = std::string(prefab.actors.articulated.front().name);
  }

  ConstraintRef const ref = AppendConstraint(prefab.constraints, type);
  ConstraintActorIndex const actors = BuildConstraintActorIndex();
  AttachConstraint({prefab, actors, _stage}, prefab.constraints, ref, targets);
  SetSingleSelection({Selection::Kind::Constraint, -1, -1, ref});
  FocusWindowIfOpen("Details");
  _prefabAsset->SetDirty(true);
  GetUndoStack().PushNow();
}

void PrefabEditor::ShowConstraintDetails(ConstraintRef ref) {
  auto& prefab = _prefabAsset->GetPrefab();
  ConstraintActorIndex const& actors = _constraintActorIndex;
  ImGui::SeparatorText("General");
  ImGui::LabelText("Type", "%s", ConstraintTypeLabel(ref.type));
  ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
  ImGui::TextWrapped("%s", ConstraintTypeDescription(ref.type));
  ImGui::PopStyleColor();
  std::string const error = ValidateConstraint(prefab, actors, ref);
  if (!error.empty()) {
    ImGui::PushStyleColor(ImGuiCol_Text, kErrorTextColor);
    ImGui::TextWrapped(
        ICON_FA_EXCLAMATION_TRIANGLE " %s The prefab simulates without it.", error.c_str());
    ImGui::PopStyleColor();
  }
  ImGui::PushID("Constraint");
  bool const changed = ConstraintEditor({prefab, actors, _stage}, prefab.constraints, ref);
  ImGui::PopID();
  if (changed) {
    // Constraints do not change the staged actors, so no restage.
    _prefabAsset->SetDirty(true);
    GetUndoStack().MarkEdited();
  }
}

void PrefabEditor::HighlightConstraintActors(ConstraintRef ref) {
  auto const& prefab = _prefabAsset->GetPrefab();
  auto const highlight = [&](std::string_view name) {
    if (StagedActor const* staged = _stage.FindActorByName(name)) {
      HighlightStagedObject(staged->sceneObject);
    }
  };
  VisitConstraint(prefab.constraints, ref, [&](auto const& c) {
    using C = std::decay_t<decltype(c)>;
    if constexpr (kIsJointConstraint<C>) {
      // A joint constraint moves the joint's child link.
      auto const* actor = FindArticulatedActor(prefab, std::string_view(c.actorName));
      if (actor != nullptr && c.jointIndex >= 0 &&
          c.jointIndex < static_cast<int>(actor->links.size())) {
        highlight(std::string(actor->name) + "/" + std::string(actor->links[c.jointIndex].name));
      }
    } else {
      ForEachConstraintActorName(
          c, [&](mochi::DynamicString const& name) { highlight(std::string_view(name)); });
    }
  });
}

void PrefabEditor::DrawConstraints() {
  auto* debugDraw = _viewport->GetRenderScene()->GetDebugDraw();
  if (debugDraw == nullptr) {
    return;
  }
  auto const& prefab = _prefabAsset->GetPrefab();
  filament::math::double3 const camera = _viewport->GetRenderScene()->GetCameraPosition();
  ConstraintDrawContext const ctx{
      prefab,
      _stage,
      *debugDraw,
      _viewport->GetDebugText(),
      _studio->GetEditorToRendererSpaceConverter(),
      filament::math::float3{
          static_cast<float>(camera.x),
          static_cast<float>(camera.y),
          static_cast<float>(camera.z)}};

  std::optional<ConstraintRef> selected;
  std::vector<ConstraintRef> emphasized; // drawn bold: hovered, or on the selected link's joint
  if (Selection const* const primary = PrimarySelection()) {
    if (primary->kind == Selection::Kind::Constraint) {
      selected = primary->constraint;
    } else if (
        primary->kind == Selection::Kind::ArticulatedLink && primary->actorIndex >= 0 &&
        primary->actorIndex < static_cast<int>(prefab.actors.articulated.size())) {
      emphasized = FindJointConstraints(
          prefab.constraints,
          std::string_view(prefab.actors.articulated[primary->actorIndex].name),
          primary->linkIndex);
    }
  }
  if (_hoveredConstraint && _hoveredConstraint != selected) {
    emphasized.push_back(*_hoveredConstraint);
  }
  auto const isEmphasized = [&](ConstraintRef const& ref) {
    return ref == selected || mochi::Contains(emphasized, ref);
  };
  if (_showAllConstraints) {
    for (ConstraintRef const& ref : CollectConstraints(prefab.constraints)) {
      if (!isEmphasized(ref)) {
        DrawConstraint(ctx, ref, ConstraintEmphasis::Normal);
      }
    }
  }
  for (ConstraintRef const& ref : emphasized) {
    if (ref != selected) {
      DrawConstraint(ctx, ref, ConstraintEmphasis::Hovered);
    }
  }
  if (selected) {
    DrawConstraint(ctx, *selected, ConstraintEmphasis::Selected);
  }
}

} // namespace superdex::studio
