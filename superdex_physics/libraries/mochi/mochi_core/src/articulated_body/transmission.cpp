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

#include <mochi_core/articulated_body/transmission.h>

#include <mochi_core/articulated_body/articulated_body.h>
#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/rigid_body_size.h>

#include <algorithm>

namespace mochi {

LinearTransmission::LinearTransmission(
    Span<int const> jointIndices,
    Span<real const> jointCoefficients,
    Span<ArticulatedDofInfo const> dofInfo,
    Span<ArticulatedPoseInfo const> poseInfo,
    std::unique_ptr<TransmissionActuator>&& actuator)
    : Transmission(std::move(actuator)) {
  int const numJoints = isize(jointIndices);
  MOCHI_ASSERT_VERBOSE(isize(jointCoefficients) == numJoints, "Inconsistent sizes of input arrays");
  _terms.reserve(numJoints);
  for (int i = 0; i < numJoints; i++) {
    // Assumes a single-DoF joint (revolute/prismatic): a scalar pose entry and a scalar DoF entry,
    // enforced by the caller (AddLinearTransmission).
    int const jointIndex = jointIndices[i];
    MOCHI_ASSERT_VERBOSE(
        dofInfo[jointIndex].GetSize() == 1, "LinearTransmission only allowed on single-DoF joints");
    _terms.push_back(
        {.jointIndex = jointIndex,
         .poseOffset = poseInfo[jointIndex].offset,
         .dofsOffset = dofInfo[jointIndex].offset,
         .coefficient = jointCoefficients[i]});
  }
}

real LinearTransmission::Displacement(
    Span<TransformRT const> /*linkTransforms*/,
    Span<real const> pose) const {
  real displacement = 0_r;
  for (auto const& term : _terms) {
    displacement += pose[term.poseOffset] * term.coefficient;
  }
  return displacement;
}

void LinearTransmission::DisplacementJacobian(
    Span<TransformRT const> /*linkTransforms*/,
    [[maybe_unused]] ArticulatedJacobian const& bodyJacobian,
    Span<real> outJacobian) const {
  MOCHI_ASSERT_VERBOSE(
      isize(outJacobian) == bodyJacobian.NumDofs(),
      "Displacement Jacobian size must equal the reduced DoF dimension");
  std::fill(outJacobian.begin(), outJacobian.end(), 0_r);
  for (auto const& term : _terms) {
    outJacobian[term.dofsOffset] += term.coefficient;
  }
}

void LinearTransmission::AddObjResDRes(
    Span<TransformRT const> currentLinkTransforms,
    Span<TransformRT const> stageStartLinkTransforms,
    ArticulatedJacobian const& /*bodyJacobian*/,
    Span<real const> currentPose,
    Span<real const> stageStartPose,
    real timeStep,
    AssemblyParams const& params,
    double& outObjective,
    Span<real> outResidual,
    Span<real> /*outLinkResidual*/,
    Matrix<real>& outDResidual) const {
  if (!(params.assemObj || params.assemRes || params.assemDRes) || !HasActuator()) {
    return;
  }
  real const currentDisplacement = Displacement(currentLinkTransforms, currentPose);
  real const stageStartDisplacement = Displacement(stageStartLinkTransforms, stageStartPose);

  real energy{}, force{}, stiffness{};
  _actuator->EnergyGradientHessian(
      currentDisplacement,
      stageStartDisplacement,
      timeStep,
      params.assemObj ? &energy : nullptr,
      params.assemRes ? &force : nullptr,
      params.assemDRes ? &stiffness : nullptr);

  if (params.assemObj) {
    outObjective += energy;
  }
  if (params.assemRes && force != 0_r) {
    for (auto const& term : _terms) {
      outResidual[term.dofsOffset] += force * term.coefficient;
    }
  }
  if (params.assemDRes) {
    if (params.psdDRes) {
      stiffness = Max(stiffness, 0_r);
    }
    if (stiffness == 0_r) {
      return;
    }
    for (auto const& termI : _terms) {
      real const stiffnessCoefficientI = stiffness * termI.coefficient;
      for (auto const& termJ : _terms) {
        outDResidual(termI.dofsOffset, termJ.dofsOffset) +=
            stiffnessCoefficientI * termJ.coefficient;
      }
    }
  }
}

// Accumulates the Euclidean length of each adjacent-waypoint segment plus coefficient * jointAngle
// for each linear-joint element. `linearJointOffsets` is advanced past each LinearJoint element so
// that the precomputed offsets stay in sync with the routing list.
static real ComputeElementSum(
    Span<RoutingElement const> routingElements,
    Span<TransformRT const> linkTransforms,
    SpatialTendon::LinearJointOffsets const* linearJointOffsets,
    Span<real const> pose) {
  real sum = 0_r;
  bool prevWasWaypoint = false;
  Real3 prevWaypointWorld{};
  for (auto const& element : routingElements) {
    switch (element.type) {
      case RoutingElementType::Waypoint: {
        Real3 const world = linkTransforms[element.index].TransformPoint(element.localPosition);
        if (prevWasWaypoint) {
          sum += Norm(world - prevWaypointWorld);
        }
        prevWaypointWorld = world;
        prevWasWaypoint = true;
        break;
      }
      case RoutingElementType::LinearJoint: {
        sum += element.coefficient * pose[linearJointOffsets->poseOffset];
        ++linearJointOffsets;
        prevWasWaypoint = false;
        break;
      }
    }
  }
  return sum;
}

// Overwrites the gradient of the element sum with respect to the pose, split in two parts: the
// waypoint segments in link space, one 6-vector per link with waypoints (the derivative with
// respect to the translation of the link's CoM and the rotation about it), and the linear-joint
// elements in joint space (one value per reduced DoF). `waypointSlots` gives, for each Waypoint
// element, the index in `outLinkGradients` of its link.
static void ComputeElementSumGradient(
    Span<RoutingElement const> routingElements,
    Span<SpatialTendon::LinearJointOffsets const> linearJointOffsets,
    Span<int const> waypointSlots,
    Span<TransformRT const> linkTransforms,
    Span<ColumnVector<real, RigidSize::kDAll>> outLinkGradients,
    ColumnVectorView<real> outJointGradient) {
  for (auto& linkGradient : outLinkGradients) {
    linkGradient.SetZero();
  }
  outJointGradient.SetZero();

  // A segment with unit direction u lengthens by u . dP_end - u . dP_start, and a point P on a link
  // moves by dP = v + omega x lever, with lever = P - CoM. Hence the gradient with respect to
  // [v; omega] of the link of the end waypoint is [u; lever x u], and the opposite for the start.
  auto addWaypointGradient = [&](int slot, Real3 const& lever, Real3 const& direction) {
    Real3 const angular = Cross(lever, direction);
    auto& linkGradient = outLinkGradients[slot];
    for (int i = 0; i < RigidSize::kDTrans; ++i) {
      linkGradient[i] += direction[i];
      linkGradient[RigidSize::kDTrans + i] += angular[i];
    }
  };

  int waypointIndex = 0;
  int linearJointIndex = 0;
  bool prevWasWaypoint = false;
  int prevSlot = -1;
  Real3 prevWorld{};
  Real3 prevLever{};
  for (auto const& element : routingElements) {
    switch (element.type) {
      case RoutingElementType::Waypoint: {
        int const slot = waypointSlots[waypointIndex++];
        Real3 const world = linkTransforms[element.index].TransformPoint(element.localPosition);
        Real3 const lever = world - linkTransforms[element.index].GetTranslation();
        if (prevWasWaypoint) {
          // Normalize regularizes the ill-posed direction of a zero-length segment to zero; the
          // length itself is smooth in pose, but the unit direction is undefined there.
          Real3 const unit = Normalize(world - prevWorld);
          addWaypointGradient(slot, lever, unit);
          addWaypointGradient(prevSlot, prevLever, -unit);
        }
        prevWorld = world;
        prevLever = lever;
        prevSlot = slot;
        prevWasWaypoint = true;
        break;
      }
      case RoutingElementType::LinearJoint: {
        // Single-DoF joints only; precomputed in the constructor.
        auto const& linearJointOffset = linearJointOffsets[linearJointIndex++];
        outJointGradient[linearJointOffset.dofsOffset] += element.coefficient;
        prevWasWaypoint = false;
        break;
      }
    }
  }
}

// outJointGradient += bodyJacobian^T * linkGradients, with linkGradients[s] the link-space gradient
// of link waypointLinks[s].
static void AddJacobianTransposeProduct(
    Span<int const> waypointLinks,
    ArticulatedJacobian const& bodyJacobian,
    Span<ColumnVector<real, RigidSize::kDAll> const> linkGradients,
    ColumnVectorView<real> outJointGradient) {
  MOCHI_ASSERT_VERBOSE(isize(linkGradients) == isize(waypointLinks), "Inconsistent sizes");
  for (int s = 0; s < isize(waypointLinks); ++s) {
    bodyJacobian.AddLinkTransposeProduct(waypointLinks[s], linkGradients[s], outJointGradient);
  }
}

SpatialTendon::SpatialTendon(
    Span<RoutingElement const> routingElements,
    Span<ArticulatedJointType const> jointTypes,
    Span<Real3 const> jointAxes,
    Span<ArticulatedDofInfo const> dofInfo,
    Span<ArticulatedPoseInfo const> poseInfo,
    Span<int const> parents,
    Span<ArticulatedRestTransform const> restTransforms,
    Span<Real3 const> comLocals,
    std::unique_ptr<TransmissionActuator>&& actuator)
    : Transmission(std::move(actuator)),
      _routingElements(routingElements.begin(), routingElements.end()) {
  // Inputs are fully validated by the calling public API; asserts here only guard internal
  // invariants in debug builds.
  MOCHI_ASSERT_VERBOSE(!_routingElements.empty(), "SpatialTendon requires at least one element");
  MOCHI_ASSERT_VERBOSE(
      isize(comLocals) == isize(parents), "comLocals must have one entry per link");

  // Convert each waypoint's `localPosition` from the link's local frame (the user-facing frame
  // in which the link's mesh and geometry are authored) to its CoM frame (internal
  // representation): localPosition_com = localPosition_local - comLocal.
  for (auto& element : _routingElements) {
    if (element.type == RoutingElementType::Waypoint) {
      element.localPosition = element.localPosition - comLocals[element.index];
    }
  }

  // LinearJoint elements read the joint's pose offset as a scalar coordinate and write the gradient
  // at a single DoF, so they are only valid for single-DoF joints (revolute/prismatic). This
  // excludes hard (0), spherical (3), and free (6) joints. The check is the macro's argument, so it
  // compiles out entirely (no residual loop) when verbose asserts are disabled.
  MOCHI_ASSERT_VERBOSE(
      std::all_of(
          _routingElements.begin(),
          _routingElements.end(),
          [&](RoutingElement const& element) {
            return element.type != RoutingElementType::LinearJoint ||
                dofInfo[element.index].GetSize() == 1;
          }),
      "SpatialTendon LinearJoint elements require a single-DoF joint");

  // Every Waypoint must be adjacent to at least one other Waypoint in the routing list; an isolated
  // waypoint contributes no segment length and is therefore meaningless.
  MOCHI_ASSERT_VERBOSE(
      [&]() {
        int const n = isize(_routingElements);
        for (int i = 0; i < n; ++i) {
          if (_routingElements[i].type != RoutingElementType::Waypoint) {
            continue;
          }
          bool const prevIsWaypoint =
              i > 0 && _routingElements[i - 1].type == RoutingElementType::Waypoint;
          bool const nextIsWaypoint =
              i + 1 < n && _routingElements[i + 1].type == RoutingElementType::Waypoint;
          if (!prevIsWaypoint && !nextIsWaypoint) {
            return false;
          }
        }
        return true;
      }(),
      "SpatialTendon Waypoint elements must be adjacent to at least one other Waypoint");

  // Precompute the links with waypoints, and the linear-joint offsets so that Displacement and
  // AddObjResDRes never need the joint layout.
  for (auto const& element : _routingElements) {
    if (element.type == RoutingElementType::LinearJoint) {
      _linearJointOffsets.push_back(
          {.poseOffset = poseInfo[element.index].offset,
           .dofsOffset = dofInfo[element.index].offset});
    } else if (element.type == RoutingElementType::Waypoint) {
      _waypointLinks.push_back(element.index);
    }
  }
  std::sort(_waypointLinks.begin(), _waypointLinks.end());
  _waypointLinks.erase(
      std::unique(_waypointLinks.begin(), _waypointLinks.end()), _waypointLinks.end());
  for (auto const& element : _routingElements) {
    if (element.type == RoutingElementType::Waypoint) {
      auto const* it =
          std::lower_bound(_waypointLinks.begin(), _waypointLinks.end(), element.index);
      _waypointSlots.push_back(static_cast<int>(it - _waypointLinks.begin()));
    }
  }

  int const numLinks = isize(parents);
  int const reducedDofsDim = articulated::GetReducedDofsSize(dofInfo);
  int const reducedPoseDim = articulated::GetReducedPoseSize(poseInfo);

  // Sized to hold zeroDofs + restPose + 2 * link transforms; 16 KiB comfortably covers
  // ~100-link articulated bodies even at double precision, and overflow falls back to heap.
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 16 * 1024);
  // The rest pose is the all-zero reduced-DoF pose; ConvertDofsToPose fills in identity quaternions
  // for ball/free joints.
  DynamicArray<real> zeroDofs(reducedDofsDim, 0_r, &allocator);
  DynamicArray<real> restPose(reducedPoseDim, 0_r, &allocator);
  articulated::ConvertDofsToPose(
      jointTypes, dofInfo, poseInfo, AsConstView(zeroDofs), AsView(restPose));

  // Run forward kinematics once at the rest pose (identity world-from-root, since routed length
  // is invariant to it) to get the link transforms used for the rest-length computation.
  DynamicArray<TransformRT> jointTransforms(numLinks, &allocator);
  DynamicArray<TransformRT> linkTransforms(numLinks, &allocator);
  articulated::ComputeTransformsFromReducedPose(
      jointTypes,
      jointAxes,
      poseInfo,
      parents,
      restTransforms,
      TransformRT{},
      AsConstView(restPose),
      MakeSpan(jointTransforms),
      MakeSpan(linkTransforms));
  // Fixed-joint elements contribute zero at the all-zero rest pose, so this is the rest
  // segment-length sum.
  _restLength = ComputeElementSum(
      MakeConstSpan(_routingElements),
      MakeConstSpan(linkTransforms),
      _linearJointOffsets.data(),
      MakeConstSpan(restPose));
}

real SpatialTendon::Displacement(Span<TransformRT const> linkTransforms, Span<real const> pose)
    const {
  return ComputeElementSum(
             MakeConstSpan(_routingElements), linkTransforms, _linearJointOffsets.data(), pose) -
      _restLength;
}

void SpatialTendon::DisplacementJacobian(
    Span<TransformRT const> linkTransforms,
    ArticulatedJacobian const& bodyJacobian,
    Span<real> outJacobian) const {
  MOCHI_ASSERT_VERBOSE(
      isize(outJacobian) == bodyJacobian.NumDofs(),
      "Displacement Jacobian size must equal the reduced DoF dimension");
  // The Jacobian is the joint-space gradient plus the link-space gradient mapped to joint DoFs.
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 64 * RigidSize::kDAll * sizeof(real));
  DynamicArray<ColumnVector<real, RigidSize::kDAll>> linkGradients(
      isize(_waypointLinks), &allocator);
  ComputeElementSumGradient(
      _routingElements,
      _linearJointOffsets,
      _waypointSlots,
      linkTransforms,
      linkGradients,
      AsView(outJacobian));
  AddJacobianTransposeProduct(_waypointLinks, bodyJacobian, linkGradients, AsView(outJacobian));
}

void SpatialTendon::AddObjResDRes(
    Span<TransformRT const> currentLinkTransforms,
    Span<TransformRT const> stageStartLinkTransforms,
    ArticulatedJacobian const& bodyJacobian,
    Span<real const> currentPose,
    Span<real const> stageStartPose,
    real timeStep,
    AssemblyParams const& params,
    double& outObjective,
    Span<real> outResidual,
    Span<real> outLinkResidual,
    Matrix<real>& outDResidual) const {
  if (!(params.assemObj || params.assemRes || params.assemDRes) || !HasActuator()) {
    return;
  }

  real const currentDisplacement = Displacement(currentLinkTransforms, currentPose);
  real const stageStartDisplacement = Displacement(stageStartLinkTransforms, stageStartPose);

  // TODO: Use dE/dDisp_prev for GradTarget::Previous.
  real energy{}, force{}, stiffness{};
  _actuator->EnergyGradientHessian(
      currentDisplacement,
      stageStartDisplacement,
      timeStep,
      params.assemObj ? &energy : nullptr,
      params.assemRes ? &force : nullptr,
      params.assemDRes ? &stiffness : nullptr);

  if (params.assemObj) {
    outObjective += energy;
  }
  if (params.assemDRes && params.psdDRes) {
    stiffness = Max(stiffness, 0_r);
  }
  MOCHI_ASSERT_VERBOSE(
      !params.assemRes || isize(outLinkResidual) == isize(currentLinkTransforms) * RigidSize::kDAll,
      "Expected a link-space residual with 6 values per link.");
  bool const needResidual = params.assemRes && force != 0_r;
  bool const needDResidual = params.assemDRes && stiffness != 0_r;
  if (!needResidual && !needDResidual) {
    return;
  }

  // The gradient is evaluated on the link transforms of the grad target, which are the ones of the
  // projection of the link-space residual to joint DoFs. The dresidual is only assembled for
  // GradTarget::Current, so it shares the gradient with the residual.
  MOCHI_ASSERT_VERBOSE(
      !needDResidual || params.gradTarget == GradTarget::Current,
      "The dresidual requires GradTarget::Current.");
  Span<TransformRT const> const linkTransforms =
      params.gradTarget == GradTarget::Previous ? stageStartLinkTransforms : currentLinkTransforms;
  MOCHI_FILO_STACK_ALLOCATOR(allocator, (64 * RigidSize::kDAll + 256) * sizeof(real));
  DynamicArray<ColumnVector<real, RigidSize::kDAll>> linkGradients(
      isize(_waypointLinks), &allocator);
  ColumnVector<real> jointGradient(bodyJacobian.NumDofs(), &allocator);
  ComputeElementSumGradient(
      _routingElements,
      _linearJointOffsets,
      _waypointSlots,
      linkTransforms,
      linkGradients,
      jointGradient);

  // The waypoint segments go to the link-space residual, and the linear-joint elements to the
  // joint-space residual.
  if (needResidual) {
    auto outLinkResidualView = AsView(outLinkResidual);
    for (int s = 0; s < isize(_waypointLinks); ++s) {
      outLinkResidualView.Slice<RigidSize::kDAll>(
          _waypointLinks[s] * RigidSize::kDAll, RigidSize::kDAll) += force * linkGradients[s];
    }
    AsView(outResidual) += force * jointGradient;
  }

  // Complete the displacement Jacobian g = J^T linkGradients + jointGradient. Then,
  // dresidual += stiffness * g (outer) g (Gauss-Newton approximation that drops the
  // force * d2(Displacement)/dq2 term, matching LinearTransmission and constraints).
  if (needDResidual) {
    AddJacobianTransposeProduct(_waypointLinks, bodyJacobian, linkGradients, jointGradient);
    outDResidual += stiffness * (jointGradient * jointGradient.Transpose());
  }
}

Span<RoutingElement const> SpatialTendon::GetRoutingElements() const {
  return MakeConstSpan(_routingElements);
}
} // namespace mochi
