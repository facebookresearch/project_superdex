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

#include "assets/prefab_constraints.h"

#include <mochi_core/articulated_body/articulated_body.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/container_utils.h>
#include <mochi_core/utils/math_utils.h>

#include <algorithm>
#include <array>
#include <cmath>

using mochi::operator""_r;

namespace superdex::studio {

using mochi::ConstraintType;
using mochi::prefab::ConstraintLists;
using mochi::prefab::ScenePrefab;

namespace {

constexpr std::array kPrefabConstraintTypes{
    ConstraintType::ArticulatedSingleDofTarget,
    ConstraintType::ArticulatedSingleDofRange,
    ConstraintType::Articulated3dRotationTarget,
    ConstraintType::Articulated3dRotationRange,
    ConstraintType::RigidSphericalJoint,
    ConstraintType::RigidPrismaticJoint,
    ConstraintType::JointRotationTracking,
    ConstraintType::JointRotationRange,
    ConstraintType::RigidPivotPosition,
    ConstraintType::RigidPivotToRigidTarget,
    ConstraintType::RigidPivotRotation,
    ConstraintType::DeformableNodePosition,
    ConstraintType::DeformableNodeToRigid,
    ConstraintType::DeformableNodeToDeformableNode,
};

// Joint springs and limits are soft by design; ConstraintParams defaults (1e6, no damping) are a
// rigid weld that would snap a joint to its target.
constexpr mochi::real kJointTargetStiffness = 1_r;
constexpr mochi::real kJointTargetDamping = 0.01_r;
constexpr mochi::real kJointRangeStiffness = 100_r;

// Erases the elements of `array` for which `pred(element)` holds. Returns the number erased.
template <class Array, class Pred>
int EraseIf(Array& array, Pred pred) {
  int removed = 0;
  for (int i = static_cast<int>(array.size()) - 1; i >= 0; --i) {
    if (pred(array[static_cast<size_t>(i)])) {
      array.erase(array.begin() + i);
      ++removed;
    }
  }
  return removed;
}

// Checks the stiffness, damping and saturation every constraint shares.
std::string ValidateCommon(mochi::ConstraintParams const& params) {
  if (!mochi::IsFinite(params.stiffness) || params.stiffness < 0) {
    return "Stiffness must be finite and not negative.";
  }
  if (!mochi::IsFinite(params.damping) || params.damping < 0) {
    return "Damping must be finite and not negative.";
  }
  if (!mochi::IsFinite(params.saturation) || params.saturation == 0_r) {
    return "Saturation must be finite and not zero (negative disables it).";
  }
  return {};
}

} // namespace

std::span<ConstraintType const> PrefabConstraintTypes() {
  return kPrefabConstraintTypes;
}

char const* ConstraintTypeLabel(ConstraintType type) {
  switch (type) {
    case ConstraintType::ArticulatedSingleDofTarget:
      return "Joint Target";
    case ConstraintType::ArticulatedSingleDofRange:
      return "Joint Range";
    case ConstraintType::Articulated3dRotationTarget:
      return "Joint Rotation Target";
    case ConstraintType::Articulated3dRotationRange:
      return "Joint Rotation Range";
    case ConstraintType::RigidSphericalJoint:
      return "Spherical Joint";
    case ConstraintType::RigidPrismaticJoint:
      return "Prismatic Joint";
    case ConstraintType::JointRotationTracking:
      return "Relative Rotation Target";
    case ConstraintType::JointRotationRange:
      return "Relative Rotation Range";
    case ConstraintType::RigidPivotPosition:
      return "Pivot Position";
    case ConstraintType::RigidPivotToRigidTarget:
      return "Pivot Target Pose";
    case ConstraintType::RigidPivotRotation:
      return "Pivot Rotation";
    case ConstraintType::DeformableNodePosition:
      return "Node Position";
    case ConstraintType::DeformableNodeToRigid:
      return "Node to Rigid";
    case ConstraintType::DeformableNodeToDeformableNode:
      return "Node to Node";
    case ConstraintType::None:
    case ConstraintType::RodElementRotationToRigid:
    case ConstraintType::RodElementRotation:
    case ConstraintType::Count:
      break;
  }
  return "Constraint";
}

char const* ConstraintTypeDescription(ConstraintType type) {
  switch (type) {
    case ConstraintType::ArticulatedSingleDofTarget:
      return "A spring that pulls one degree of freedom of a joint toward a target value.";
    case ConstraintType::ArticulatedSingleDofRange:
      return "Keeps one degree of freedom of a joint between a minimum and a maximum.";
    case ConstraintType::Articulated3dRotationTarget:
      return "A spring that turns a spherical or free joint toward a target orientation.";
    case ConstraintType::Articulated3dRotationRange:
      return "Keeps each rotation angle of a spherical or free joint within a range.";
    case ConstraintType::RigidSphericalJoint:
      return "A ball joint: holds a point on actor A on a point on actor B, rotation free.";
    case ConstraintType::RigidPrismaticJoint:
      return "Lets actor B slide relative to actor A along one axis only (rotation free; pair it "
             "with a Relative Rotation Target).";
    case ConstraintType::JointRotationTracking:
      return "Holds the rotation of actor B relative to actor A at its rest value.";
    case ConstraintType::JointRotationRange:
      return "Keeps each angle of actor B's rotation relative to actor A within a range.";
    case ConstraintType::RigidPivotPosition:
      return "Pulls a point on a rigid actor toward a position in the world.";
    case ConstraintType::RigidPivotToRigidTarget:
      return "Pulls a point on a rigid actor to where it would be at a target pose.";
    case ConstraintType::RigidPivotRotation:
      return "Turns a frame on a rigid actor toward an orientation in the world.";
    case ConstraintType::DeformableNodePosition:
      return "Pulls one node of a deformable actor toward a position in the world.";
    case ConstraintType::DeformableNodeToRigid:
      return "Attaches one node of a deformable actor to a point on a rigid actor.";
    case ConstraintType::DeformableNodeToDeformableNode:
      return "Ties a node of one deformable actor to a node of another.";
    case ConstraintType::None:
    case ConstraintType::RodElementRotationToRigid:
    case ConstraintType::RodElementRotation:
    case ConstraintType::Count:
      break;
  }
  return "";
}

ConstraintCategory GetConstraintCategory(ConstraintType type) {
  switch (type) {
    case ConstraintType::ArticulatedSingleDofTarget:
    case ConstraintType::ArticulatedSingleDofRange:
    case ConstraintType::Articulated3dRotationTarget:
    case ConstraintType::Articulated3dRotationRange:
      return ConstraintCategory::ArticulatedJoint;
    case ConstraintType::RigidSphericalJoint:
    case ConstraintType::RigidPrismaticJoint:
    case ConstraintType::JointRotationTracking:
    case ConstraintType::JointRotationRange:
      return ConstraintCategory::RigidPair;
    case ConstraintType::RigidPivotPosition:
    case ConstraintType::RigidPivotToRigidTarget:
    case ConstraintType::RigidPivotRotation:
      return ConstraintCategory::RigidPivot;
    case ConstraintType::DeformableNodePosition:
    case ConstraintType::DeformableNodeToRigid:
    case ConstraintType::DeformableNodeToDeformableNode:
    case ConstraintType::None:
    case ConstraintType::RodElementRotationToRigid:
    case ConstraintType::RodElementRotation:
    case ConstraintType::Count:
      break;
  }
  return ConstraintCategory::Deformable;
}

std::vector<ConstraintRef> CollectConstraints(ConstraintLists const& lists) {
  std::vector<ConstraintRef> refs;
  ForEachConstraintArray(lists, [&](ConstraintType type, auto const& array) {
    for (int i = 0; i < static_cast<int>(array.size()); ++i) {
      refs.push_back({type, i});
    }
  });
  return refs;
}

ConstraintRef AppendConstraint(ConstraintLists& lists, ConstraintType type) {
  ConstraintRef ref;
  ForEachConstraintArray(lists, [&](ConstraintType arrayType, auto& array) {
    if (arrayType != type) {
      return;
    }
    array.emplace_back();
    auto& constraint = array.back();
    if (type == ConstraintType::ArticulatedSingleDofTarget ||
        type == ConstraintType::Articulated3dRotationTarget) {
      constraint.stiffness = kJointTargetStiffness;
      constraint.damping = kJointTargetDamping;
    } else if (
        type == ConstraintType::ArticulatedSingleDofRange ||
        type == ConstraintType::Articulated3dRotationRange) {
      constraint.stiffness = kJointRangeStiffness;
    }
    ref = {type, static_cast<int>(array.size()) - 1};
  });
  return ref;
}

std::optional<ConstraintRef> DuplicateConstraint(ConstraintLists& lists, ConstraintRef ref) {
  std::optional<ConstraintRef> copy;
  ForEachConstraintArray(lists, [&](ConstraintType type, auto& array) {
    if (type == ref.type && ref.index >= 0 && ref.index < static_cast<int>(array.size())) {
      auto element = array[static_cast<size_t>(ref.index)];
      array.push_back(std::move(element));
      copy = ConstraintRef{type, static_cast<int>(array.size()) - 1};
    }
  });
  return copy;
}

bool RemoveConstraint(ConstraintLists& lists, ConstraintRef ref) {
  bool removed = false;
  ForEachConstraintArray(lists, [&](ConstraintType type, auto& array) {
    if (type == ref.type && ref.index >= 0 && ref.index < static_cast<int>(array.size())) {
      array.erase(array.begin() + ref.index);
      removed = true;
    }
  });
  return removed;
}

std::vector<ConstraintRef>
FindJointConstraints(ConstraintLists const& lists, std::string_view actorName, int jointIndex) {
  std::vector<ConstraintRef> refs;
  ForEachConstraintArray(lists, [&](ConstraintType type, auto const& array) {
    using C = typename std::decay_t<decltype(array)>::value_type;
    if constexpr (kIsJointConstraint<C>) {
      for (int i = 0; i < static_cast<int>(array.size()); ++i) {
        auto const& c = array[static_cast<size_t>(i)];
        if (c.jointIndex == jointIndex && std::string_view(c.actorName) == actorName) {
          refs.push_back({type, i});
        }
      }
    }
  });
  return refs;
}

std::vector<ConstraintRef> FindActorConstraints(
    ConstraintLists const& lists,
    std::string_view actorName) {
  std::vector<ConstraintRef> refs;
  ForEachConstraintArray(lists, [&](ConstraintType type, auto const& array) {
    using C = typename std::decay_t<decltype(array)>::value_type;
    for (int i = 0; i < static_cast<int>(array.size()); ++i) {
      auto const& c = array[static_cast<size_t>(i)];
      if constexpr (kIsJointConstraint<C>) {
        if (std::string_view(c.actorName) == actorName) {
          continue;
        }
      }
      bool refers = false;
      ForEachConstraintActorName(c, [&](mochi::DynamicString const& name) {
        refers |= ActorNameRefersTo(std::string_view(name), actorName);
      });
      if (refers) {
        refs.push_back({type, i});
      }
    }
  });
  return refs;
}

bool ActorNameRefersTo(std::string_view name, std::string_view actor) {
  // Whole path segments only: "arm" covers "arm" and "arm/hand", not "armrest".
  return !actor.empty() && name.starts_with(actor) &&
      (name.size() == actor.size() || name[actor.size()] == '/');
}

int RenameConstraintActor(
    ConstraintLists& lists,
    std::string_view oldName,
    std::string_view newName) {
  if (oldName.empty() || oldName == newName) {
    return 0;
  }
  int renamed = 0;
  ForEachConstraintArray(lists, [&](ConstraintType, auto& array) {
    for (auto& constraint : array) {
      ForEachConstraintActorName(constraint, [&](mochi::DynamicString& name) {
        std::string_view const current(name);
        if (ActorNameRefersTo(current, oldName)) {
          name = mochi::DynamicString{
              std::string(newName) + std::string(current.substr(oldName.size()))};
          ++renamed;
        }
      });
    }
  });
  return renamed;
}

int RemoveConstraintsReferencingActor(ConstraintLists& lists, std::string_view name) {
  int removed = 0;
  ForEachConstraintArray(lists, [&](ConstraintType, auto& array) {
    removed += EraseIf(array, [&](auto const& constraint) {
      bool refers = false;
      ForEachConstraintActorName(constraint, [&](mochi::DynamicString const& actor) {
        refers |= ActorNameRefersTo(std::string_view(actor), name);
      });
      return refers;
    });
  });
  return removed;
}

int RemapJointConstraints(
    ConstraintLists& lists,
    std::string_view actorName,
    std::span<int const> oldToNew) {
  int removed = 0;
  ForEachConstraintArray(lists, [&](ConstraintType, auto& array) {
    using C = typename std::decay_t<decltype(array)>::value_type;
    if constexpr (kIsJointConstraint<C>) {
      removed += EraseIf(array, [&](C& c) {
        if (std::string_view(c.actorName) != actorName || c.jointIndex < 0 ||
            c.jointIndex >= static_cast<int>(oldToNew.size())) {
          return false;
        }
        int const newIndex = oldToNew[static_cast<size_t>(c.jointIndex)];
        if (newIndex < 0) {
          return true;
        }
        c.jointIndex = newIndex;
        return false;
      });
    }
  });
  return removed;
}

int CopyJointConstraints(ConstraintLists& lists, std::string_view from, std::string_view to) {
  int copied = 0;
  ForEachConstraintArray(lists, [&](ConstraintType, auto& array) {
    using C = typename std::decay_t<decltype(array)>::value_type;
    if constexpr (kIsJointConstraint<C>) {
      int const count = static_cast<int>(array.size());
      for (int i = 0; i < count; ++i) {
        if (std::string_view(array[static_cast<size_t>(i)].actorName) == from) {
          C copy = array[static_cast<size_t>(i)];
          copy.actorName = mochi::DynamicString{std::string(to)};
          array.push_back(std::move(copy));
          ++copied;
        }
      }
    }
  });
  return copied;
}

mochi::prefab::ArticulatedActorPrefab const* FindArticulatedActor(
    ScenePrefab const& prefab,
    std::string_view name) {
  for (auto const& actor : prefab.actors.articulated) {
    if (std::string_view(actor.name) == name) {
      return &actor;
    }
  }
  return nullptr;
}

std::string JointDisplayName(mochi::prefab::ArticulatedActorPrefab const* actor, int jointIndex) {
  if (actor != nullptr && jointIndex >= 0 && jointIndex < static_cast<int>(actor->joints.size()) &&
      !actor->joints[static_cast<size_t>(jointIndex)].name.empty()) {
    return std::string(actor->joints[static_cast<size_t>(jointIndex)].name);
  }
  return "joint_" + std::to_string(jointIndex);
}

mochi::real GetConstraintActorMetricScale(ScenePrefab const& prefab, std::string_view name) {
  mochi::real scale = 1_r;
  ScenePrefab const* level = &prefab;
  while (level != nullptr) {
    size_t const slash = name.find('/');
    if (slash == std::string_view::npos) {
      break;
    }
    std::string_view const first = name.substr(0, slash);
    ScenePrefab const* next = nullptr;
    for (auto const& nested : level->prefabs) {
      if (std::string_view(nested.name) == first && nested.prefab) {
        scale *= nested.scale;
        next = nested.prefab.get();
        break;
      }
    }
    if (next == nullptr) {
      // "arm/link": a link of an articulated actor, whose local distances also scale with it.
      if (auto const* actor = FindArticulatedActor(*level, first)) {
        scale *= actor->scale;
      }
      break;
    }
    level = next;
    name = name.substr(slash + 1);
  }
  return scale;
}

std::string DescribeConstraint(ScenePrefab const& prefab, ConstraintRef ref) {
  auto const nameOr = [](mochi::DynamicString const& name) {
    return name.empty() ? std::string("(no actor)") : std::string(name);
  };
  std::string text;
  VisitConstraint(prefab.constraints, ref, [&](auto const& c) {
    using C = std::decay_t<decltype(c)>;
    using namespace mochi::prefab;
    if constexpr (kIsJointConstraint<C>) {
      text = nameOr(c.actorName) + "/" +
          JointDisplayName(
                 FindArticulatedActor(prefab, std::string_view(c.actorName)), c.jointIndex);
    } else if constexpr (std::is_same_v<C, DeformableNodeToDeformableNodeConstraintPrefab>) {
      text = c.findClosest ? nameOr(c.actorNameA) + " & " + nameOr(c.actorNameB)
                           : nameOr(c.actorNameA) + " #" + std::to_string(c.nodeIndexA) + " & " +
              nameOr(c.actorNameB) + " #" + std::to_string(c.nodeIndexB);
    } else if constexpr (kIsPairConstraint<C>) {
      text = nameOr(c.actorNameA) + " & " + nameOr(c.actorNameB);
    } else if constexpr (std::is_same_v<C, DeformableNodeToRigidConstraintPrefab>) {
      text = nameOr(c.deformableActorName) +
          (c.findClosest ? std::string() : " #" + std::to_string(c.deformableNodeIndex)) + " & " +
          nameOr(c.rigidActorName);
    } else if constexpr (std::is_same_v<C, DeformableNodePositionConstraintPrefab>) {
      text = nameOr(c.actorName) + " #" + std::to_string(c.nodeIndex);
    } else {
      text = nameOr(c.actorName);
    }
  });
  return text;
}

ConstraintActorIndex BuildConstraintActorIndex(
    ScenePrefab const& prefab,
    std::span<std::string const> nestedActorNames) {
  ConstraintActorIndex index;
  for (auto const& actor : prefab.actors.rigid) {
    if (!actor.name.empty()) {
      index.rigid.emplace_back(actor.name);
    }
  }
  for (auto const& actor : prefab.actors.soft) {
    if (!actor.name.empty()) {
      index.deformable.emplace_back(actor.name);
    }
  }
  for (auto const& actor : prefab.actors.articulated) {
    if (actor.name.empty()) {
      continue;
    }
    index.articulated.emplace_back(actor.name);
    // Mochi registers each link as an actor, so rigid constraints can name it.
    for (auto const& link : actor.links) {
      index.rigid.push_back(std::string(actor.name) + "/" + std::string(link.name));
    }
  }
  for (std::string const& name : nestedActorNames) {
    index.nested.push_back(name);
    // The articulated actor of a nested link ("nested/arm" of "nested/arm/link"); never the
    // nested prefab reference itself.
    size_t const first = name.find('/');
    size_t const last = name.rfind('/');
    if (last != first && !mochi::Contains(index.nested, name.substr(0, last))) {
      index.nested.push_back(name.substr(0, last));
    }
  }
  return index;
}

bool IsSingleDofConstraintDof(mochi::ArticulatedJointType type, int dofIndex) {
  switch (type) {
    case mochi::ArticulatedJointType::Prismatic:
    case mochi::ArticulatedJointType::Revolute:
      return dofIndex == 0;
    case mochi::ArticulatedJointType::Free:
      return dofIndex >= 0 && dofIndex < 3;
    case mochi::ArticulatedJointType::Spherical:
    case mochi::ArticulatedJointType::Hard:
    case mochi::ArticulatedJointType::Cycle:
    case mochi::ArticulatedJointType::Count:
      break;
  }
  return false;
}

bool Is3dRotationConstraintJoint(mochi::ArticulatedJointType type) {
  return type == mochi::ArticulatedJointType::Spherical ||
      type == mochi::ArticulatedJointType::Free;
}

bool JointFitsConstraint(mochi::ArticulatedJointType type, bool singleDof) {
  return singleDof ? IsSingleDofConstraintDof(type, 0) : Is3dRotationConstraintJoint(type);
}

bool IsRotationDof(mochi::ArticulatedJointType type, int dofIndex) {
  auto const [translationDofs, rotationDofs] = mochi::articulated::GetJointTypeNumDofs(type);
  return dofIndex >= translationDofs && dofIndex < translationDofs + rotationDofs;
}

std::string ValidateConstraint(
    ScenePrefab const& prefab,
    ConstraintActorIndex const& actors,
    ConstraintRef ref) {
  using namespace mochi::prefab;
  std::string error = "The constraint does not exist.";
  VisitConstraint(prefab.constraints, ref, [&](auto const& c) {
    using C = std::decay_t<decltype(c)>;
    error.clear();
    // `kind` lists the actors of the right kind for `name`; nested actors are accepted anywhere.
    auto checkActor = [&](char const* field,
                          mochi::DynamicString const& name,
                          std::vector<std::string> const& kind,
                          char const* kindName) {
      std::string const value(name);
      if (value.empty()) {
        error = std::string(field) + " is not set.";
      } else if (!mochi::Contains(kind, value) && !mochi::Contains(actors.nested, value)) {
        error = std::string(field) + " \"" + value + "\" is not a " + kindName +
            " actor of this prefab.";
      }
      return error.empty();
    };
    if constexpr (kIsJointConstraint<C>) {
      if (!checkActor("Actor", c.actorName, actors.articulated, "articulated")) {
        return;
      }
      auto const* actor = FindArticulatedActor(prefab, std::string_view(c.actorName));
      if (actor == nullptr) {
        return; // Inside a nested prefab: its joints are checked when it loads.
      }
      if (c.jointIndex < 0 || c.jointIndex >= static_cast<int>(actor->joints.size())) {
        error = "Joint " + std::to_string(c.jointIndex) + " does not exist.";
        return;
      }
      auto const jointType = actor->joints[static_cast<size_t>(c.jointIndex)].type;
      constexpr bool kSingleDof = std::is_same_v<C, ArticulatedSingleDofTargetConstraintPrefab> ||
          std::is_same_v<C, ArticulatedSingleDofRangeConstraintPrefab>;
      if constexpr (kSingleDof) {
        if (!IsSingleDofConstraintDof(jointType, c.dofIndex)) {
          error = Is3dRotationConstraintJoint(jointType)
              ? "This DoF is a rotation of a 3D joint: use a Joint Rotation constraint."
              : "The joint has no DoF " + std::to_string(c.dofIndex) + ".";
          return;
        }
      } else if (!Is3dRotationConstraintJoint(jointType)) {
        error = "3D rotation constraints need a spherical or free joint.";
        return;
      }
      if constexpr (std::is_same_v<C, ArticulatedSingleDofRangeConstraintPrefab>) {
        if (!(c.minValue <= c.maxValue)) {
          error = "The minimum is above the maximum.";
          return;
        }
      } else if constexpr (std::is_same_v<C, Articulated3dRotationRangeConstraintPrefab>) {
        for (int i = 0; i < 3; ++i) {
          if (!(c.minValues[i] <= c.maxValues[i])) {
            error = "A minimum is above its maximum.";
            return;
          }
        }
      }
    } else if constexpr (kIsPairConstraint<C>) {
      constexpr bool kDeformable =
          std::is_same_v<C, DeformableNodeToDeformableNodeConstraintPrefab>;
      auto const& kind = kDeformable ? actors.deformable : actors.rigid;
      char const* kindName = kDeformable ? "deformable" : "rigid";
      if (!checkActor("Actor A", c.actorNameA, kind, kindName) ||
          !checkActor("Actor B", c.actorNameB, kind, kindName)) {
        return;
      }
      if (c.actorNameA == c.actorNameB) {
        error = "Actor A and actor B are the same actor.";
        return;
      }
      if constexpr (std::is_same_v<C, RigidPrismaticJointConstraintPrefab>) {
        if (!mochi::IsFinite(c.freeAxis) || mochi::NearZero(mochi::Norm(c.freeAxis))) {
          error = "The free axis must be finite and not zero.";
          return;
        }
        if (c.min.has_value() && c.max.has_value() && !(*c.min <= *c.max)) {
          error = "The minimum is above the maximum.";
          return;
        }
      } else if constexpr (std::is_same_v<C, JointRotationRangeConstraintPrefab>) {
        for (auto const* range : {&c.angleRangeX, &c.angleRangeY, &c.angleRangeZ}) {
          if (!((*range)[0] <= (*range)[1])) {
            error = "A minimum angle is above its maximum.";
            return;
          }
        }
      } else if constexpr (kDeformable) {
        if (!c.findClosest && (c.nodeIndexA < 0 || c.nodeIndexB < 0)) {
          error = "Set both node indices, or turn on Find Closest.";
          return;
        }
      }
    } else if constexpr (std::is_same_v<C, DeformableNodeToRigidConstraintPrefab>) {
      if (!checkActor("Rigid actor", c.rigidActorName, actors.rigid, "rigid") ||
          !checkActor("Deformable actor", c.deformableActorName, actors.deformable, "deformable")) {
        return;
      }
      if (!c.findClosest && c.deformableNodeIndex < 0) {
        error = "Set the node index, or turn on Find Closest.";
        return;
      }
    } else if constexpr (std::is_same_v<C, DeformableNodePositionConstraintPrefab>) {
      if (!checkActor("Actor", c.actorName, actors.deformable, "deformable")) {
        return;
      }
      if (c.nodeIndex < 0) {
        error = "Set the node index.";
        return;
      }
    } else if (!checkActor("Actor", c.actorName, actors.rigid, "rigid")) {
      return;
    }
    error = ValidateCommon(c);
  });
  return error;
}

std::vector<std::string> RemoveInvalidConstraints(
    ScenePrefab& prefab,
    ConstraintActorIndex const& actors) {
  std::vector<std::string> reasons;
  std::vector<ConstraintRef> refs = CollectConstraints(prefab.constraints);
  // Remove from the back so earlier indices of the same array stay valid.
  for (auto it = refs.rbegin(); it != refs.rend(); ++it) {
    std::string reason = ValidateConstraint(prefab, actors, *it);
    if (!reason.empty()) {
      reasons.push_back(
          std::string(ConstraintTypeLabel(it->type)) + " (" + DescribeConstraint(prefab, *it) +
          "): " + reason);
      RemoveConstraint(prefab.constraints, *it);
    }
  }
  return reasons;
}

} // namespace superdex::studio
