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

#include "mochi_articulated_actor_params.h"

#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/rigid_body_utils.h>

#include <limits>
#include <numeric>
#include <string>
#include <string_view>

using namespace mochi;

static constexpr real kUnitVectorTolerance = 5e-6_r;

std::shared_ptr<ArticulatedBodyShape> mochi::GetArticulatedShape(
    ArticulatedActorParams const& params,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});

#if MOCHI_ASSERT_VERBOSE_ENABLED
  Validate(params, error);
  MOCHI_ASSERT_VERBOSE(error.IsOK(), "Please call Validate first.");
  MOCHI_ERROR_RETURN(error, {});
#endif

  auto articulatedShape = std::make_shared<ArticulatedBodyShape>(params, error);
  MOCHI_ERROR_RETURN(error, {});
  return articulatedShape;
}

template <typename Items, typename GetName>
static void FillEmptyNames(Items& items, char const* prefix, GetName const& getName) {
  int nextIndex = 0;
  for (auto& item : items) {
    auto& name = getName(item);
    if (name.empty()) {
      std::string candidate;
      bool collision = false;
      do {
        candidate = std::string(prefix) + "_" + std::to_string(nextIndex);
        ++nextIndex;
        collision = false;
        for (auto const& other : items) {
          auto const& otherName = getName(other);
          if (std::string_view(otherName) == candidate) {
            collision = true;
            break;
          }
        }
      } while (collision);
      name = candidate;
    }
  }
}

[[nodiscard]] static bool HasNan(Real3 const& v) {
  return std::isnan(v[0]) || std::isnan(v[1]) || std::isnan(v[2]);
}

[[nodiscard]] static bool HasNan(std::optional<Real3> const& opt) {
  return opt.has_value() && HasNan(*opt);
}

static void ReplaceInfWithHuge(Real3& v) {
  // JSON specification does not allow non-finite values. Therefore, we replace
  // them with huge values instead (practically infinite);
  real constexpr kHuge = std::numeric_limits<float>::max(); // float max not real max
  for (int i = 0; i < 3; ++i) {
    if (!IsFinite(v[i])) {
      v[i] = (v[i] < 0_r) ? -kHuge : kHuge;
    }
  }
}

void mochi::AutoCorrect(ArticulatedJointParams& joint, Error& error) {
  MOCHI_ERROR_RETURN(error);

  joint.parentLinkFromJoint = NormalizeRotation(joint.parentLinkFromJoint);

  switch (joint.type) {
    case ArticulatedJointType::Free:
    case ArticulatedJointType::Hard: {
      // Clear params not allowed for Free (6 DOF) or Hard (0 DOF) joints
      joint.axis = {};
      joint.minLimit = std::nullopt;
      joint.maxLimit = std::nullopt;
      joint.inertia = std::nullopt;
      joint.friction = {};
    } break;

    case ArticulatedJointType::Prismatic:
    case ArticulatedJointType::Revolute: {
      MOCHI_ERROR_IF(
          !IsFinite(joint.axis) || (Real3{} == joint.axis),
          error,
          "Joint axis must be non-zero and finite.");
      MOCHI_ERROR_IF(
          HasNan(joint.minLimit) || HasNan(joint.maxLimit),
          error,
          "NaN value detected in joint limits.");
      MOCHI_ERROR_RETURN(error); // We don't try to fix NaNs.

      joint.axis = Normalize(joint.axis);

      // If joint limits are used, then they will be projected onto the joint axis.
      real minVal = joint.minLimit ? Dot(joint.axis, *joint.minLimit) : -kInf;
      real maxVal = joint.maxLimit ? Dot(joint.axis, *joint.maxLimit) : kInf;

      // Clear limits that are infinite when projected. This can happen even when all values
      // are technically finite on their own.
      if (!IsFinite(minVal)) {
        joint.minLimit = std::nullopt;
        minVal = -kInf;
      }
      if (!IsFinite(maxVal)) {
        joint.maxLimit = std::nullopt;
        maxVal = kInf;
      }

      // Swap limits if they are out-of-order
      if (minVal > maxVal) {
        std::swap(joint.minLimit, joint.maxLimit);
        std::swap(minVal, maxVal);
      }
    } break;

    case ArticulatedJointType::Spherical: {
      MOCHI_ERROR_IF(
          HasNan(joint.minLimit) || HasNan(joint.maxLimit),
          error,
          "NaN value detected in joint limits.");
      MOCHI_ERROR_RETURN(error); // We don't try to fix NaNs.

      // No axis for a spherical joint
      joint.axis = {};

      // Spherical joint limits are applied per axis.
      Real3 minVal = joint.minLimit.value_or(-kInf3);
      Real3 maxVal = joint.maxLimit.value_or(kInf3);
      for (int i = 0; i < 3; ++i) {
        if (minVal[i] > maxVal[i]) {
          // Fix the order
          std::swap(minVal[i], maxVal[i]);
        }
      }
      joint.minLimit = minVal;
      joint.maxLimit = maxVal;

      // Clear infinite limits.
      if (minVal == -kInf3) {
        joint.minLimit = std::nullopt; // No limit
      }
      if (maxVal == kInf3) {
        joint.maxLimit = std::nullopt; // No limit
      }

      // A spherical joint could have a mix of finite and non-finite values. However, JSON doesn't
      // allow non-finite values. The work-around is to replace non-finite values with huge finite
      // values.
      if (joint.minLimit.has_value() && !IsFinite(*joint.minLimit)) {
        ReplaceInfWithHuge(*joint.minLimit);
      }
      if (joint.maxLimit.has_value() && !IsFinite(*joint.maxLimit)) {
        ReplaceInfWithHuge(*joint.maxLimit);
      }

    } break;

    case ArticulatedJointType::Cycle:
    case ArticulatedJointType::Count:
    default:
      static_assert(
          static_cast<int>(ArticulatedJointType::Count) == 6,
          "Please update this code if you add a new joint type.");
      MOCHI_ERROR_SET(error, "Invalid joint type.");
      break;
  }

  // For joints without limits, set damping and stiffness to their defaults.
  if (!joint.minLimit.has_value() && !joint.maxLimit.has_value()) {
    joint.limitDamping = ArticulatedJointParams{}.limitDamping;
    joint.limitStiffness = ArticulatedJointParams{}.limitStiffness;
  }
}

void mochi::AutoCorrect(ArticulatedCycleJointParams& cycle, Error& error) {
  MOCHI_ERROR_RETURN(error);
  cycle.jointFromChildLink = NormalizeRotation(cycle.jointFromChildLink);
}

void mochi::AutoCorrect(ArticulatedLinkParams& link, Error& error) {
  MOCHI_ERROR_RETURN(error);
  link.parentJointFromLink = NormalizeRotation(link.parentJointFromLink);
}

void mochi::AutoCorrect(ArticulatedActorParams& params, Error& error) {
  MOCHI_ERROR_RETURN(error);

  // Normalize worldFromRoot quaternion
  params.worldFromRoot = NormalizeRotation(params.worldFromRoot);

  // Fill empty link names with unique defaults ("link_0", "link_1", ...)
  FillEmptyNames(params.links, "link", [](auto& link) -> auto& { return link.name; });

  // Fill empty joint names with unique defaults ("joint_0", "joint_1", ...)
  FillEmptyNames(params.joints, "joint", [](auto& joint) -> auto& { return joint.name; });

  // Joints
  for (auto& joint : params.joints) {
    AutoCorrect(joint, error);
  }

  // Links
  for (auto& link : params.links) {
    AutoCorrect(link, error);
  }

  // Cycles
  for (auto& cycle : params.cycles) {
    AutoCorrect(cycle, error);
  }
}

#define MOCHI_IMPL_VALIDATE_TRANSFORM_RT(transform, error, description)                           \
  {                                                                                               \
    MOCHI_ERROR_IF(!IsFinite(transform), error, description " transform values must be finite."); \
    MOCHI_ERROR_IF(                                                                               \
        !NearEqual(1_r, Norm(transform.GetRotation()), kUnitVectorTolerance),                     \
        error,                                                                                    \
        description " quaternion should be unit length.")                                         \
  }

void mochi::ValidateFriction(ArticulatedJointFrictionParams const& friction, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(!IsFinite(friction.viscous), error, "Joint friction viscous is not finite.");
  MOCHI_ERROR_IF(friction.viscous < 0_r, error, "Joint friction viscous is negative.");
  MOCHI_ERROR_IF(!IsFinite(friction.coulomb), error, "Joint friction coulomb is not finite.");
  MOCHI_ERROR_IF(friction.coulomb < 0_r, error, "Joint friction coulomb is negative.");
  MOCHI_ERROR_IF(!IsFinite(friction.falloffVel), error, "Joint friction falloffVel is not finite.");
  MOCHI_ERROR_IF(friction.falloffVel < 0_r, error, "Joint friction falloffVel is negative.");
  MOCHI_ERROR_IF(
      !IsFinite(friction.stictionExtra), error, "Joint friction stictionExtra is not finite.");
  MOCHI_ERROR_IF(friction.stictionExtra < 0_r, error, "Joint friction stictionExtra is negative.");
  MOCHI_ERROR_IF(
      !IsFinite(friction.stribeckVel), error, "Joint friction stribeckVel is not finite.");
  MOCHI_ERROR_IF(friction.stribeckVel < 0_r, error, "Joint friction stribeckVel is negative.");
}

void mochi::ValidateInertia(real inertia, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(!IsFinite(inertia), error, "Joint inertia is not finite.");
  MOCHI_ERROR_IF(inertia < 0_r, error, "Joint inertia is negative.");
}

static void ValidateJoint(ArticulatedJointParams const& joint, Error& error) {
  MOCHI_ERROR_RETURN(error);

  MOCHI_ERROR_IF(joint.name.empty(), error, "Joint has an empty name.");
  MOCHI_ERROR_IF(
      static_cast<int>(joint.type) < 0 || joint.type >= ArticulatedJointType::Count ||
          joint.type == ArticulatedJointType::Cycle,
      error,
      "Invalid joint type.");
  MOCHI_IMPL_VALIDATE_TRANSFORM_RT(joint.parentLinkFromJoint, error, "Parent-link-from-joint");

  // Axis
  MOCHI_ERROR_IF(!IsFinite(joint.axis), error, "Joint axis is not finite.");
  if (joint.type == ArticulatedJointType::Prismatic ||
      joint.type == ArticulatedJointType::Revolute) {
    MOCHI_ERROR_IF(
        !NearEqual(1_r, Norm(joint.axis), kUnitVectorTolerance),
        error,
        "Joint axis should be a unit vector.");
  }

  // Limits
  MOCHI_ERROR_IF(
      HasNan(joint.minLimit) || HasNan(joint.maxLimit),
      error,
      "NaN value detected in joint limits.");

  // Friction
  ValidateFriction(joint.friction, error);

  // Inertia
  if (joint.inertia.has_value()) {
    ValidateInertia(*joint.inertia, error);
  }

  // Limit stiffness
  MOCHI_ERROR_IF(!IsFinite(joint.limitStiffness), error, "Joint limitStiffness is not finite.");
  MOCHI_ERROR_IF(joint.limitStiffness < 0_r, error, "Joint limitStiffness is negative.");

  // Limit damping
  MOCHI_ERROR_IF(!IsFinite(joint.limitDamping), error, "Joint limitDamping is not finite.");
  MOCHI_ERROR_IF(joint.limitDamping < 0_r, error, "Joint limitDamping is negative.");
}

static void ValidateLink(int index, ArticulatedLinkParams const& link, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(link.name.empty(), error, "Link has an empty name.");
  MOCHI_ERROR_IF(
      HasInvalidNestedActorNameCharacter(link.name),
      error,
      "Link name must not contain '/', '\\', or embedded NUL characters.");

  // Parent link index must be in [-1, index) for parent-first order
  if (index == 0) {
    MOCHI_ERROR_IF(link.parentLink != -1, error, "Root link must have parent index of -1.");
  } else if (link.parentLink != -1) {
    MOCHI_ERROR_IF(
        link.parentLink < 0 || link.parentLink >= index, error, "Parent link index out-of-range.");
  }
  MOCHI_IMPL_VALIDATE_TRANSFORM_RT(link.parentJointFromLink, error, "Parent-joint-from-link");

  // The following fields are ignored for dummy links (links with no shape)
  if (link.shape.IsValid()) {
    // Density
    if (link.density.has_value()) {
      MOCHI_ERROR_IF(
          !IsFinite(*link.density) || (*link.density <= 0_r),
          error,
          "Link density must be positive and finite.");
    }

    // Mass
    if (link.mass.has_value()) {
      MOCHI_ERROR_IF(
          !IsFinite(*link.mass) || (*link.mass <= 0_r),
          error,
          "Link mass must be positive and finite.");
    }
    MOCHI_ERROR_IF(
        link.density.has_value() && link.mass.has_value(),
        error,
        "Link may specify density or mass, but not both.");

    // Center of mass
    if (link.centerOfMass.has_value()) {
      MOCHI_ERROR_IF(!IsFinite(*link.centerOfMass), error, "Link center-of-mass is not finite.");
    }

    // Moment of inertia
    if (link.momentOfInertia.has_value()) {
      MOCHI_ERROR_IF_NOT(
          IsFinite(*link.momentOfInertia), error, "Link moment-of-inertia tensor must be finite.");
      MOCHI_ERROR_RETURN(error);
      if (!IsMomentOfInertiaValid(*link.momentOfInertia)) {
        MOCHI_LOG_WARNING(
            "Moment-of-inertia tensor of link \"%s\" is not physically valid: principal moments must be non-negative and satisfy the triangle inequality.",
            link.name.c_str());
      }
    }
  }
}

static void
ValidateCycleJoint(ArticulatedCycleJointParams const& cycle, int numLinks, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      cycle.childLink < 0 || cycle.childLink >= numLinks,
      error,
      "Cycle joint child index is out of range.");
  MOCHI_ERROR_IF(
      cycle.parentLink < 0 || cycle.parentLink >= numLinks,
      error,
      "Cycle joint parent index is out of range.");
  MOCHI_ERROR_IF(
      cycle.childLink == cycle.parentLink, error, "Cycle joint child and parent must differ.");
  MOCHI_IMPL_VALIDATE_TRANSFORM_RT(cycle.jointFromChildLink, error, "Cycle joint-from-child-link");
  MOCHI_ERROR_IF(
      !IsFinite(cycle.stiffness) || (cycle.stiffness < 0_r),
      error,
      "Cycle joint stiffness must be non-negative and finite.");
}

template <typename GetNameFn>
static bool HasDuplicateNames(int n, GetNameFn const& getName) {
  if (n == 0) {
    return false;
  }
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 128 * sizeof(int));
  DynamicArray<int> indices(&allocator);
  indices.resize(n);
  std::iota(indices.begin(), indices.end(), 0);
  std::ranges::sort(indices, [&getName](int a, int b) { return getName(a) < getName(b); });
  for (int i = 1; i < n; ++i) {
    if (getName(indices[i - 1]) == getName(indices[i])) {
      return true;
    }
  }
  return false;
}

void mochi::Validate(ArticulatedActorParams const& params, Error& error) {
  MOCHI_ERROR_RETURN(error);

  int const numJoints = isize(params.joints);
  int const numLinks = isize(params.links);

  // Structural checks
  MOCHI_ERROR_IF(numLinks == 0, error, "Expected at least one link.");
  MOCHI_ERROR_IF(numJoints != numLinks, error, "Expected an equal number of joints and links.");
  MOCHI_ERROR_RETURN(error);

  // Validate worldFromRoot
  MOCHI_IMPL_VALIDATE_TRANSFORM_RT(params.worldFromRoot, error, "World-from-root");

  // Validate name uniqueness
  MOCHI_ERROR_IF(
      HasDuplicateNames(numLinks, [&params](int i) { return params.links[i].name; }),
      error,
      "Duplicate link name.");
  MOCHI_ERROR_IF(
      HasDuplicateNames(numJoints, [&params](int i) { return params.joints[i].name; }),
      error,
      "Duplicate joint name.");
  MOCHI_ERROR_RETURN(error);

  // Validate joints
  for (auto const& joint : params.joints) {
    ValidateJoint(joint, error);
  }
  MOCHI_ERROR_RETURN(error);

  // Validate links
  for (int i = 0; i < numLinks; ++i) {
    ValidateLink(i, params.links[i], error);
  }
  MOCHI_ERROR_RETURN(error);

  // Validate cycle joints
  for (auto const& cycle : params.cycles) {
    ValidateCycleJoint(cycle, numLinks, error);
  }

  // Validate joint velocities
  if (params.jointVelocities.has_value() && !params.jointVelocities->empty()) {
    MOCHI_ERROR_IF(
        !IsFinite(MakeConstSpan(*params.jointVelocities)),
        error,
        "Joint velocity values must be finite.");
  }
}
