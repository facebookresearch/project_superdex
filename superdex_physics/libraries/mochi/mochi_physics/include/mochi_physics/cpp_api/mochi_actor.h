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

#pragma once

#include "mochi_enums.h"
#include "mochi_handle.h"
#include "mochi_structs.h"

/********************************************************************************
 IMPORTANT: PLEASE KEEP HEADER INCLUDES TO A MINIMUM.
    If you must include a mochi_core header, then please make sure that it only
    declares the data types (not containing other implementation details).
*********************************************************************************/
#include <mochi_core/articulated_body/articulated_body_params.h>
#include <mochi_core/contact/contact_params.h>
#include <mochi_core/geometry/aabb.h>
#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/geometry/obb.h>
#include <mochi_core/geometry/sphere.h>
#include <mochi_core/materials/material_params.h>
#include <mochi_core/solvers/nonlinear_solver_params.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/transform_rt.h>

#include <functional>
#include <string_view>

namespace mochi {

class Scene;
class Constraint;
class Context;

class Actor {
 public:
  [[nodiscard]] virtual ActorHandle GetHandle() const = 0;

  [[nodiscard]] virtual Context* GetContext() = 0;
  [[nodiscard]] virtual Context const* GetContext() const = 0;

  [[nodiscard]] virtual Scene* GetScene() = 0;
  [[nodiscard]] virtual Scene const* GetScene() const = 0;

  [[nodiscard]] virtual char const* GetName() const = 0;

  [[nodiscard]] virtual ActorType GetType() const = 0;

  [[nodiscard]] virtual ConvergenceStatus GetConvergenceStatus() const = 0;

  [[nodiscard]] virtual bool IsStatic() const = 0;

  [[nodiscard]] virtual ContactParams GetContactParams(Error& error) const = 0;

  virtual void SetContactParams(ContactParams const& params, Error& error) = 0;

  [[nodiscard]] virtual ColliderType GetColliderType() const = 0;

  [[nodiscard]] virtual bool HasRootTransform() const = 0;

  [[nodiscard]] virtual TransformRT GetRootTransform() const = 0;

  virtual void SetRootTransform(TransformRT const& worldFromLocal, Error& error) = 0;

  [[nodiscard]] virtual Real3 GetRigidCenterOfMassLocal(Error& error) const = 0;

  [[nodiscard]] virtual TransformRT GetCenterOfMassTransform(Error& error) const = 0;

  virtual void SetCenterOfMassTransform(TransformRT const& worldFromCom, Error& error) = 0;

  [[nodiscard]] virtual Real6 GetRigidMomentOfInertiaLocal(Error& error) const = 0;

  virtual void SetVelocity(Real3 const& linearVel, Real3 const& angularVel, Error& error) = 0;

  [[nodiscard]] virtual Real3 GetLinearVelocity(Error& error) const = 0;

  [[nodiscard]] virtual Real3 GetAngularVelocity(Error& error) const = 0;

  [[nodiscard]] virtual real GetMass(Error& error) const = 0;

  [[nodiscard]] virtual real GetDensity(Error& error) const = 0;

  virtual void SetDensity(real density, Error& error) = 0;

  virtual void SetInertiaProperties(
      real mass,
      Real3 const& centerOfMass,
      Real6 const& momentOfInertia,
      Error& error) = 0;

  [[nodiscard]] virtual real GetElasticEnergy(Error& error) const = 0;

  [[nodiscard]] virtual RecenteringParams GetRecenteringParams() const = 0;

  virtual void SetRecenteringParams(RecenteringParams const& params, Error& error) = 0;

  [[nodiscard]] virtual Span<real const> GetDisplacements(Error& error) const = 0;

  virtual void SetDisplacements(Span<real const> displacements, Error& error) = 0;

  [[nodiscard]] virtual SoftMaterialParams GetSoftMaterialParams(Error& error) const = 0;

  virtual void SetSoftMaterialParams(SoftMaterialParams const& params, Error& error) = 0;

  [[nodiscard]] virtual MeshDataView GetMesh() const = 0;

  [[nodiscard]] virtual MeshDataView GetSurfaceMesh() const = 0;

  [[nodiscard]] virtual MeshDataView GetVisualMesh() const = 0;

  [[nodiscard]] virtual Span<real const> GetSurfaceMeshNodePositionsLocal(Error& error) const = 0;

  [[nodiscard]] virtual Span<real const> GetSurfaceMeshNodeNormalsLocal(Error& error) const = 0;

  [[nodiscard]] virtual ShapeHandle GetReferenceShape(Error& error) const = 0;

  [[nodiscard]] virtual std::string_view GetContactLayer() const = 0;

  virtual void SetContactLayer(std::string_view const& layer) = 0;

  [[nodiscard]] virtual int GetNumDofs() const = 0;

  virtual void GetDofValues(Span<int const> dofIndices, Span<real> outDofValues, Error& error)
      const = 0;

  virtual void QueryNodesInVolumeLocal(
      Aabb const& volume,
      bool boundaryOnly,
      std::function<void(int, Real3)> const& callback,
      Error& error) const = 0;

  virtual void QueryNodesInVolumeLocal(
      Obb const& volume,
      bool boundaryOnly,
      std::function<void(int, Real3)> const& callback,
      Error& error) const = 0;

  virtual void QueryNodesInVolumeLocal(
      Sphere const& volume,
      bool boundaryOnly,
      std::function<void(int, Real3)> const& callback,
      Error& error) const = 0;

  virtual void GetPointsDistanceToSurface(
      Span<Real3 const> pointsWorld,
      Span<real> outDistances,
      Error& error) const = 0;

  [[nodiscard]] virtual Aabb GetAabbLocal(Error& error) const = 0;

  [[nodiscard]] virtual Aabb GetAabbWorld(Error& error) const = 0;

  [[nodiscard]] virtual Span<real const> GetNodePositionsLocal(Error& error) const = 0;

  virtual void SetZeroDisplacementsAndVelocities(Error& error) = 0;

  [[nodiscard]] virtual Span<real const> GetElementsDeformationGradient(Error& error) const = 0;

  virtual void SetNodePositionsLocal(Span<real const> positionsLocal, Error& error) = 0;

  virtual void SetNodeVelocitiesLocal(Span<real const> velocitiesLocal, Error& error) = 0;

  [[nodiscard]] virtual Span<real const> GetVisualMeshNodePositionsLocal(Error& error) const = 0;

  [[nodiscard]] virtual Span<real const> GetVisualMeshNodeNormalsLocal(Error& error) const = 0;

  [[nodiscard]] virtual Span<int const> GetBoundaryConditionDofIndices() const = 0;

  [[nodiscard]] virtual Span<real const> GetBoundaryConditionDofValuesWorld() const = 0;

  virtual void AddBoundaryConditionDofsWorld(
      Span<int const> dofIndices,
      Span<real const> dofValuesWorld,
      Error& error) = 0;

  virtual void AddBoundaryConditionDofsWorldPermanent(
      Span<int const> dofIndices,
      Span<real const> dofValuesWorld,
      Error& error) = 0;

  virtual void AddBoundaryConditionNodesWorld(
      Span<int const> nodeIndices,
      Span<real const> nodePositionsWorld,
      Error& error) = 0;

  virtual void AddBoundaryConditionNodesWorldPermanent(
      Span<int const> nodeIndices,
      Span<real const> nodePositionsWorld,
      Error& error) = 0;

  virtual void AddBoundaryConditionConstrainedNodesAtRest(Error& error) = 0;

  virtual void AddBoundaryConditionConstrainedNodesAtRestPermanent(Error& error) = 0;

  virtual void ClearBoundaryConditions() = 0;

  virtual void SetExternalForcesOnDofs(
      Span<int const> dofIndices,
      Span<real const> forceValues,
      Error& error) = 0;

  virtual void ClearExternalForces() = 0;

  virtual void GetExternalForces(Span<real> outForces, Error& error) = 0;

  [[nodiscard]] virtual Span<ContactPoint const> GetContactPointsWorld(Error& error) const = 0;

  [[nodiscard]] virtual Span<NodeContactForce const> GetNodeContactForcesWorld(
      Error& error) const = 0;

  [[nodiscard]] virtual Real3 GetContactForceWorld(Error& error) const = 0;

  [[nodiscard]] virtual Real3 GetContactTorqueWorld(Error& error) const = 0;

  [[nodiscard]] virtual Real3 GetContactForceFromActorWorld(Actor const* other, Error& error)
      const = 0;

  [[nodiscard]] virtual SdfDistances GetSdfDistances(Error& error) const = 0;

  virtual QueryHandle RegisterQuery(QueryType type, Error& error) = 0;

  virtual QueryHandle RegisterQueryAndCompute(QueryType type, Error& error) = 0;

  virtual void CancelQuery(QueryHandle handle) = 0;

  [[nodiscard]] virtual bool IsQuerySupported(QueryType type) const = 0;

  virtual void SetUserData(void* userData) = 0;

  virtual void* GetUserData() const = 0;

  [[nodiscard]] virtual Span<ActorHandle const> GetNestedLinkActors(Error& error) const = 0;

  [[nodiscard]] virtual Span<ActorHandle const> GetNestedSoftActors(Error& error) const = 0;

  [[nodiscard]] virtual ArticulatedShapeInfo GetArticulatedShapeInfo(Error& error) const = 0;

  [[nodiscard]] virtual Span<Constraint* const> GetArticulatedJointLimitConstraints(
      Error& error) const = 0;

  [[nodiscard]] virtual Span<ArticulatedJointFrictionParams const>
  GetArticulatedJointFrictionParams(Error& error) const = 0;

  virtual void SetArticulatedJointFrictionParams(
      Span<ArticulatedJointFrictionParams const> friction,
      Error& error) = 0;

  [[nodiscard]] virtual Span<real const> GetArticulatedJointInertiaParams(Error& error) const = 0;

  virtual void SetArticulatedJointInertiaParams(Span<real const> inertia, Error& error) = 0;

  virtual void GetArticulatedDofLimits(Span<Real2> outDofLimits, Error& error) const = 0;

  virtual void GetArticulatedPose(Span<real> outPose, Error& error) const = 0;

  virtual void GetArticulatedLinkTransforms(Span<TransformRT> outWorldFromLinks, Error& error)
      const = 0;

  virtual void GetArticulatedJointVelocities(Span<real> outVelocities, Error& error) const = 0;

  virtual void SetArticulatedPoseFromLinks(
      Span<TransformRT const> worldFromLinks,
      Error& error) = 0;

  virtual void SetArticulatedPoseFromJoints(Span<real const> pose, Error& error) = 0;

  virtual void SetArticulatedJointVelocities(Span<real const> velocities, Error& error) = 0;

  virtual void AddArticulatedDeltaToPose(
      Span<real const> pose,
      Span<real const> deltaDofs,
      Span<real> outPose,
      Error& error) const = 0;

  virtual void ComputeArticulatedPoseDelta(
      Span<real const> poseBase,
      Span<real const> poseTarget,
      Span<real> outDeltaDofs,
      Error& error) const = 0;

  virtual void GetArticulatedPoseDistance(
      Span<real const> poseA,
      Span<real const> poseB,
      Span<real> outTransDistances,
      Span<real> outRotDistances,
      Error& error) const = 0;

  [[nodiscard]] virtual bool HasArticulatedPoseController(Error& error) const = 0;

  virtual void AddArticulatedPoseController(PoseControllerParams const& params, Error& error) = 0;

  virtual void RemoveArticulatedPoseController(Error& error) = 0;

  virtual void SetArticulatedTargetLinkTransforms(
      Span<TransformRT const> worldFromTargets,
      Error& error) = 0;

  virtual void SetArticulatedTargetPose(Span<real const> pose, Error& error) = 0;

  virtual void ResetArticulatedTargetLinkTransforms(
      Span<TransformRT const> worldFromTargets,
      Error& error) = 0;

  virtual void ResetArticulatedTargetPose(Span<real const> pose, Error& error) = 0;

  virtual void SetArticulatedTargetVelocity(Span<real const> velocity, Error& error) = 0;

  [[nodiscard]] virtual Span<PoseConstraintInfo const> GetArticulatedPoseConstraints(
      Error& error) const = 0;

  virtual void GetArticulatedPoseControllerParams(PoseControllerParams& outParams, Error& error)
      const = 0;

  virtual void SetArticulatedPoseControllerParams(
      PoseControllerParams const& params,
      Error& error) = 0;

  virtual void GetArticulatedTargetPose(Span<real> outPose, Error& error) const = 0;

  virtual void GetArticulatedTargetLinkTransforms(
      Span<TransformRT> outWorldFromTargets,
      Error& error) const = 0;

  [[nodiscard]] virtual Span<real const> GetArticulatedControllerForce(Error& error) const = 0;

  [[nodiscard]] virtual bool IsNestedLinkActor() const = 0;

  [[nodiscard]] virtual bool IsNestedSoftActor() const = 0;

  [[nodiscard]] virtual ActorHandle GetArticulatedActor(Error& error) const = 0;

  [[nodiscard]] virtual Span<real const> GetArticulatedJacobian(Error& error) const = 0;

 protected:
  // Don't delete the Actor pointer. Call Scene::DestroyActor.
  virtual ~Actor() = default;
};

} // namespace mochi
