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

#include "mochi_transmission.h"
#include "mochi_articulated_body.h"
#include "mochi_group.h"

#include <mochi_core/articulated_body/transmission.h>
#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/utils/dynamic_array.h>

#include <memory>

namespace mochi::transmission {

// Internal helper function for transmission access with error handling
static Transmission*
GetTransmission(entt::registry& reg, entt::entity e, int transmissionIndex, Error& error) {
  MOCHI_ERROR_IF_NOT(
      reg.all_of<TagArticulatedActor>(e),
      error,
      "Transmissions are currently only supported for articulated actors.");
  MOCHI_ERROR_RETURN(error, nullptr);
  auto& transmissions = reg.get<CTransmissions>(e).transmissions;
  MOCHI_ERROR_IF_NOT(
      transmissionIndex >= 0 && isize(transmissions) > transmissionIndex,
      error,
      "Invalid transmission index.");
  MOCHI_ERROR_RETURN(error, nullptr);
  Transmission* transmission = transmissions[transmissionIndex].get();
  MOCHI_ERROR_IF(transmission == nullptr, error, "Invalid transmission at given index.");
  MOCHI_ERROR_RETURN(error, nullptr);
  return transmission;
}

// Internal helper function to attach any transmission actuator type
static void AttachTransmissionActuatorImpl(
    entt::registry& reg,
    entt::entity e,
    int transmissionIndex,
    std::unique_ptr<TransmissionActuator>&& actuator,
    Error& error) {
  Transmission* transmission = GetTransmission(reg, e, transmissionIndex, error);
  MOCHI_ERROR_RETURN(error);
  transmission->SetActuator(std::move(actuator));
}

int AddLinearTransmission(
    entt::registry& reg,
    entt::entity e,
    experimental::LinearTransmissionParams const& params,
    Error& error) {
  MOCHI_ERROR_IF_NOT(
      reg.all_of<TagArticulatedActor>(e),
      error,
      "AddLinearTransmission is currently only supported for articulated actors.");
  MOCHI_ERROR_RETURN(error, kSentinelIndex);
  // Validate that the input joint indices are valid and all refer to single-DoF joints, which is
  // required for the linear transmission model.
  auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
  auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(e);
  int const numJoints = isize(joints->dofInfo);
  MOCHI_ERROR_IF_NOT(
      isize(params.jointIndices) == isize(params.jointCoefficients),
      error,
      "Inconsistent numbers of joints and coefficients in linear transmission parameters");
  MOCHI_ERROR_RETURN(error, kSentinelIndex);
  for (int jointIndex : params.jointIndices) {
    MOCHI_ERROR_IF(
        (jointIndex < 0) || (jointIndex >= numJoints),
        error,
        "Invalid joint index given in linear transmission parameters");
    MOCHI_ERROR_RETURN(error, kSentinelIndex);
    MOCHI_ERROR_IF_NOT(
        joints->dofInfo[jointIndex].GetSize() == 1,
        error,
        "Linear transmission can only be routed through single-DoF joints.");
  }
  MOCHI_ERROR_RETURN(error, kSentinelIndex);

  auto& transmissions = reg.get<CTransmissions>(e).transmissions;
  // This new transmission will have a null actuator, which will need to be set separately, using
  // one of the API methods for different actuator types.
  transmissions.emplace_back(
      std::make_unique<LinearTransmission>(
          params.jointIndices, params.jointCoefficients, joints->dofInfo, poseInfo));
  return isize(transmissions) - 1;
}

int AddSpatialTendon(
    entt::registry& reg,
    entt::entity e,
    experimental::SpatialTendonParams const& params,
    Error& error) {
  MOCHI_ERROR_IF_NOT(
      reg.all_of<TagArticulatedActor>(e),
      error,
      "AddSpatialTendon is currently only supported for articulated actors.");
  MOCHI_ERROR_RETURN(error, kSentinelIndex);
  auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
  auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(e);
  auto const& parents = reg.get<CArticulatedParents const>(e);
  auto const& restTransforms = reg.get<CArticulatedRestTransforms const>(e);
  int const numLinks = isize(parents);
  int const numJoints = isize(joints->dofInfo);
  MOCHI_ERROR_IF(
      params.routingElements.empty(), error, "A spatial tendon must have at least one element");
  MOCHI_ERROR_RETURN(error, kSentinelIndex);
  for (auto const& element : params.routingElements) {
    switch (element.type) {
      case RoutingElementType::Waypoint:
        MOCHI_ERROR_IF(
            (element.index < 0) || (element.index >= numLinks),
            error,
            "Invalid waypoint link index given in spatial tendon parameters");
        MOCHI_ERROR_IF_NOT(
            IsFinite(element.localPosition),
            error,
            "Spatial tendon waypoint local position must be finite");
        break;
      case RoutingElementType::LinearJoint:
        MOCHI_ERROR_IF_NOT(
            (element.index >= 0) && (element.index < numJoints),
            error,
            "Invalid joint index given in spatial tendon linear-joint element");
        MOCHI_ERROR_RETURN(error, kSentinelIndex);
        MOCHI_ERROR_IF_NOT(
            joints->dofInfo[element.index].GetSize() == 1,
            error,
            "Spatial tendon linear-joint elements can only be routed through single-DoF joints.");
        MOCHI_ERROR_IF_NOT(
            IsFinite(element.coefficient),
            error,
            "Spatial tendon linear-joint coefficient must be finite");
        break;
    }
    MOCHI_ERROR_RETURN(error, kSentinelIndex);
  }
  // A waypoint forms a tendon segment only with an adjacent waypoint; one sitting between
  // non-waypoint elements (or at a chain end next to one) contributes nothing and is almost
  // certainly a routing mistake, so reject it.
  auto const& elements = params.routingElements;
  int const numElements = isize(elements);
  for (int i = 0; i < numElements; ++i) {
    if (elements[i].type != RoutingElementType::Waypoint) {
      continue;
    }
    bool const prevIsWaypoint = (i > 0) && (elements[i - 1].type == RoutingElementType::Waypoint);
    bool const nextIsWaypoint =
        (i + 1 < numElements) && (elements[i + 1].type == RoutingElementType::Waypoint);
    MOCHI_ERROR_IF_NOT(
        prevIsWaypoint || nextIsWaypoint,
        error,
        "Each spatial tendon waypoint must be adjacent to another waypoint; an isolated waypoint forms no segment");
  }
  MOCHI_ERROR_RETURN(error, kSentinelIndex);
  // Warn if a linear-joint element is adjacent to a waypoint whose link is neither of the joint's
  // two links: the waypoint then contributes no joint-relevant routing geometry and the pairing is
  // likely a routing mistake.
  for (int i = 0; i < numElements; ++i) {
    if (elements[i].type != RoutingElementType::LinearJoint) {
      continue;
    }
    int const jointIdx = elements[i].index;
    int const jointParent = joints->jointsParentLinks[jointIdx];
    int const jointChild = joints->jointsChildLinks[jointIdx];
    for (int const neighbor : {i - 1, i + 1}) {
      if (neighbor < 0 || neighbor >= numElements ||
          elements[neighbor].type != RoutingElementType::Waypoint) {
        continue;
      }
      int const linkIndex = elements[neighbor].index;
      if (linkIndex != jointParent && linkIndex != jointChild) {
        MOCHI_LOG_WARNING(
            "Spatial tendon linear-joint element at routing index %d (joint index %d) is adjacent "
            "to waypoint at routing index %d (link index %d), which is on neither the joint's "
            "parent link (%d) nor child link (%d).",
            i,
            elements[i].index,
            neighbor,
            linkIndex,
            jointParent,
            jointChild);
      }
    }
  }
  // Gather each link's CoM offset in its local frame (the same user-authored frame in which the
  // link's mesh and `RoutingElement::localPosition` live), used by the SpatialTendon constructor
  // to convert each waypoint's `localPosition` from the public-API local frame to the internal
  // CoM frame.
  auto const& links = reg.get<CGroupMembers const>(e).actors;
  DynamicArray<Real3> comLocals;
  comLocals.reserve(numLinks);
  for (auto const link : links) {
    comLocals.emplace_back(ToReal3(reg.get<CRigidBodyInertia const>(link).GetCenterOfMassLocal()));
  }
  auto& transmissions = reg.get<CTransmissions>(e).transmissions;
  // This new transmission will have a null actuator, which will need to be set separately, using
  // one of the API methods for different actuator types.
  transmissions.emplace_back(
      std::make_unique<SpatialTendon>(
          params.routingElements,
          joints->jointTypes,
          joints->jointAxes,
          joints->dofInfo,
          poseInfo,
          parents,
          restTransforms,
          MakeConstSpan(comLocals)));
  return isize(transmissions) - 1;
}

void AttachDisplacementControlActuator(
    entt::registry& reg,
    entt::entity e,
    int transmissionIndex,
    experimental::DisplacementControlActuatorParams const& params,
    Error& error) {
  AttachTransmissionActuatorImpl(
      reg,
      e,
      transmissionIndex,
      std::make_unique<DisplacementControlActuator>(
          params.targetDisplacement,
          params.stiffness,
          params.damping,
          params.allowCompressiveForce),
      error);
}

void AttachForceControlActuator(
    entt::registry& reg,
    entt::entity e,
    int transmissionIndex,
    experimental::ForceControlActuatorParams const& params,
    Error& error) {
  auto actuator =
      std::make_unique<ForceControlActuator>(params.force, error, params.allowCompressiveForce);
  MOCHI_ERROR_RETURN(error);
  AttachTransmissionActuatorImpl(reg, e, transmissionIndex, std::move(actuator), error);
}

void AttachMcKibbenActuator(
    entt::registry& reg,
    entt::entity e,
    int transmissionIndex,
    experimental::McKibbenActuatorParams const& params,
    Error& error) {
  MOCHI_ERROR_IF(
      params.deflatedEquilibriumLength <= 0_r,
      error,
      "Deflated equilibrium length must be positive");
  MOCHI_ERROR_IF(params.deflatedStiffness <= 0_r, error, "Deflated stiffness must be positive");
  MOCHI_ERROR_IF(params.numberOfWraps <= 0_r, error, "Number of wraps must be positive");
  MOCHI_ERROR_IF(params.threadLength <= 0_r, error, "Thread length must be positive");
  MOCHI_ERROR_RETURN(error);
  // Pressures can be negative, depending on an arbitrary gague pressure, as long as it is the same
  // for both pressure and minimumPressure.
  AttachTransmissionActuatorImpl(
      reg,
      e,
      transmissionIndex,
      std::make_unique<McKibbenActuator>(
          params.pressure,
          params.minimumPressure,
          params.threadLength,
          params.numberOfWraps,
          params.deflatedStiffness,
          params.deflatedEquilibriumLength),
      error);
}

void SetTransmissionActuatorStateVariables(
    entt::registry& reg,
    entt::entity e,
    int transmissionIndex,
    Span<real const> stateVariables,
    Error& error) {
  Transmission* transmission = GetTransmission(reg, e, transmissionIndex, error);
  MOCHI_ERROR_RETURN(error);
  auto* actuator = transmission->GetActuator();
  MOCHI_ERROR_IF_NOT(actuator != nullptr, error, "Transmission has no actuator attached");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF_NOT(
      actuator->GetNumStateVariables() == isize(stateVariables),
      error,
      "Incorrect number of state variables for the transmission's actuator");
  MOCHI_ERROR_RETURN(error);
  actuator->SetStateVariables(stateVariables, error);
  MOCHI_ERROR_RETURN(error);
}

real GetTransmissionDisplacement(
    entt::registry& reg,
    entt::entity e,
    int transmissionIndex,
    Error& error) {
  Transmission const* transmission = GetTransmission(reg, e, transmissionIndex, error);
  MOCHI_ERROR_RETURN(error, 0_r);
  auto const& props = reg.get<CArticulatedProps const>(e);
  auto const& pose = reg.get<CArticulatedReducedPose<TimeStep::Current> const>(e).value;

  // Reserve stack memory for the transforms (up to 256 links).
  MOCHI_FILO_STACK_ALLOCATOR(allocator, sizeof(TransformRT) * 256);
  DynamicArray<TransformRT> linkTransforms(&allocator);
  linkTransforms.resize_noinit(props.numLinks);
  articulated::compound::GetLinkTransformsComFromPose(reg, e, pose, linkTransforms);
  return transmission->Displacement(linkTransforms, pose);
}

void GetTransmissionDisplacementJacobian(
    entt::registry& reg,
    entt::entity e,
    int transmissionIndex,
    Span<real> outJacobian,
    Error& error) {
  Transmission const* transmission = GetTransmission(reg, e, transmissionIndex, error);
  MOCHI_ERROR_RETURN(error);
  int const numDofs = reg.get<CActorDofInfo const>(e).dofsSize;
  MOCHI_ERROR_IF_NOT(
      isize(outJacobian) == numDofs,
      error,
      "Transmission displacement Jacobian size must equal Actor::GetNumDofs().");
  MOCHI_ERROR_RETURN(error);

  // The link transforms and the Jacobian are up to date.
  auto const& linkTransforms = reg.get<CArticulatedLinkTransforms<TimeStep::Current> const>(e);
  auto const& bodyJacobian = reg.get<CArticulatedJacobian const>(e).value;
  transmission->DisplacementJacobian(
      MakeConstSpan(linkTransforms), AsConstView(bodyJacobian), outJacobian);
}

void GetTransmissionActuatorStateVariables(
    entt::registry& reg,
    entt::entity e,
    int transmissionIndex,
    Span<real> outStateVariables,
    Error& error) {
  Transmission const* transmission = GetTransmission(reg, e, transmissionIndex, error);
  MOCHI_ERROR_RETURN(error);
  auto const* actuator = transmission->GetActuator();
  MOCHI_ERROR_IF_NOT(actuator != nullptr, error, "Transmission has no actuator attached");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF_NOT(
      actuator->GetNumStateVariables() == isize(outStateVariables),
      error,
      "Incorrect number of state variables for the transmission's actuator");
  MOCHI_ERROR_RETURN(error);
  actuator->GetStateVariables(outStateVariables);
}

int GetNumTransmissionActuatorStateVariables(
    entt::registry& reg,
    entt::entity e,
    int transmissionIndex,
    Error& error) {
  Transmission const* transmission = GetTransmission(reg, e, transmissionIndex, error);
  MOCHI_ERROR_RETURN(error, 0);
  auto const* actuator = transmission->GetActuator();
  MOCHI_ERROR_IF_NOT(actuator != nullptr, error, "Transmission has no actuator attached");
  MOCHI_ERROR_RETURN(error, 0);
  return actuator->GetNumStateVariables();
}
} // namespace mochi::transmission
