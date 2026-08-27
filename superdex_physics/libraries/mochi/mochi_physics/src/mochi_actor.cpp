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

#include "mochi_actor.h"

#include "mochi_actor_convergence.h"
#include "mochi_articulated_actor_params.h"
#include "mochi_articulated_body.h"
#include "mochi_blended.h"
#include "mochi_contact.h"
#include "mochi_contact_filter.h"
#include "mochi_context.h"
#include "mochi_differentiable.h"
#include "mochi_ecs_utils.h"
#include "mochi_group.h"
#include "mochi_integration.h"
#include "mochi_query.h"
#include "mochi_rigid.h"
#include "mochi_rod.h"
#include "mochi_shell.h"
#include "mochi_simulation.h"
#include "mochi_soft.h"
#include "mochi_soft_rom_systems.h"
#include "mochi_soft_skinned.h"
#include "mochi_solve.h"
#include "mochi_transmission.h"

#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/materials/batched_smith_neo_hookean.h>
#include <mochi_core/materials/material_params_utils.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/mesh_embedding.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/rigid_body_utils.h>
#include <mochi_physics/diffsim/mochi_diffsim.h>
#include <mochi_physics/mochi_physics_experimental.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mochi {

bool actor::CanOwnNestedActors(ActorType type) {
  static_assert(
      static_cast<int>(ActorType::Count) == 6,
      "Update actor::CanOwnNestedActors if ActorType changes.");

  switch (type) {
    case ActorType::Articulated:
      return true;
    case ActorType::None:
    case ActorType::Rigid:
    case ActorType::Soft:
    case ActorType::Shell:
    case ActorType::Rod:
      return false;
    case ActorType::Count:
      MOCHI_ASSERT_VERBOSE(false, "ActorType::Count is not a valid actor type.");
      return false;
  }

  return false;
}

template <typename MeshType>
static MeshDataView MakeActorMeshDataView(MeshType const& mesh) {
  MeshDataView view{};
  view.nodesPerElement = mesh.GetNumNodesPerElement();
  view.coordinates = Flatten(mesh.GetNodeCoordinates());
  view.connectivity = mesh.GetFlatConnectivity();
  return view;
}

static MeshDataView MakeActorSurfaceMeshDataView(TriangularMesh const& mesh) {
  MeshDataView view{};
  view.nodesPerElement = mesh.GetNumNodesPerElement();
  view.coordinates = Flatten(mesh.GetActiveNodeCoordinates());
  view.connectivity = mesh.GetActiveNodesFlatConnectivity();
  return view;
}

namespace {
// Mesh representation exposed by Actor::GetMesh().
enum class ActorMeshSource {
  Simplicial,
  Polyline,
  Surface,
  None,
};
} // namespace

[[nodiscard]] static ActorMeshSource GetActorMeshSource(entt::registry const& reg, entt::entity e) {
  // Use the primary simplicial simulation mesh before falling back to surface meshes.
  // Note: Skinned and blended actors also have a surface mesh, but their primary mesh is the
  // skin/blended simulation mesh.
  if (reg.all_of<CSimplicialMesh>(e)) {
    return ActorMeshSource::Simplicial;
  }

  // Rod actors store their simulation mesh as a polyline instead of a simplicial mesh.
  if (reg.all_of<CPolylineMesh>(e)) {
    return ActorMeshSource::Polyline;
  }

  // Mesh-backed rigid actors store their simulation mesh as a surface mesh.
  if (reg.any_of<TagRigidActor, TagStaticActor>(e)) {
    return ActorMeshSource::Surface;
  }

  return ActorMeshSource::None;
}

template <typename ShapeT>
static void QueryNodesInVolumeLocalImpl(
    entt::registry const& reg,
    entt::entity e,
    ShapeT const& volumeLocal,
    bool boundaryOnly,
    std::function<void(int, Real3)> const& callback,
    Error& error) {
  if (boundaryOnly) {
    auto const* queryPos = MOCHI_TRY_GET(CQuerySurfaceNodePositions, reg, e, error);
    MOCHI_ERROR_RETURN(error);
    Span<Real3 const> positions = Unflatten<Real3 const>(queryPos->nodePositions);
    auto const* surfaceMeshComponent = MOCHI_TRY_GET(CSurfaceMesh, reg, e, error);
    MOCHI_ERROR_RETURN(error);

    ActorMeshSource const actorMeshSource = GetActorMeshSource(reg, e);
    MOCHI_ERROR_IF(
        actorMeshSource != ActorMeshSource::Simplicial &&
            actorMeshSource != ActorMeshSource::Surface,
        error,
        "Boundary node indices cannot be represented in Actor::GetMesh ordering.");
    MOCHI_ERROR_RETURN(error);

    auto const& mesh = surfaceMeshComponent->mesh;
    bool const usesCompactSurfaceNodeOrdering = actorMeshSource == ActorMeshSource::Surface;
    for (int i = 0; i < mesh->GetNumActiveNodes(); ++i) {
      if (ContainsPoint(volumeLocal, positions[i])) {
        int const nodeIndex =
            usesCompactSurfaceNodeOrdering ? i : mesh->GetActiveToAllNodesIndexMap(i);
        callback(nodeIndex, positions[i]);
      }
    }
  } else {
    auto const* queryPos = MOCHI_TRY_GET(CQueryNodePositions, reg, e, error);
    MOCHI_ERROR_RETURN(error);
    Span<Real3 const> positions = Unflatten<Real3 const>(queryPos->nodePositions);

    int const numNodes = isize(positions);
    for (int i = 0; i < numNodes; ++i) {
      if (ContainsPoint(volumeLocal, positions[i])) {
        callback(i, positions[i]);
      }
    }
  }
}

namespace {
// Application specific user data associated with an actor.
struct CUserData : public NoCopy {
  explicit CUserData(void* userData) : userData(userData) {}
  void* userData = nullptr;
};

/**
  EcsActor implements the Actor API by calling ECS-style functions.
*/
class ActorInterfaceImpl : public ActorInterface {
 public:
  ActorInterfaceImpl(entt::registry& reg, entt::entity e, Scene* scene)
      : reg(reg), e(e), scene(scene) {
    MOCHI_ASSERT(e != entt::null);
    MOCHI_ASSERT(scene != nullptr);
  }

  // The ONLY data members
  entt::registry& reg;
  entt::entity e;
  Scene* scene;

  ActorHandle GetHandle() const override {
    return GetActorHandle(e, scene->GetHandle());
  }

  Context* GetContext() override {
    return scene->GetContext();
  }

  Context const* GetContext() const override {
    return scene->GetContext();
  }

  Scene* GetScene() override {
    return scene;
  }

  Scene const* GetScene() const override {
    return scene;
  }

  char const* GetName() const override {
    return reg.get<CActorInfo>(e).name.c_str();
  }

  ActorType GetType() const override {
    return reg.get<CActorInfo>(e).type;
  }

  ConvergenceStatus GetConvergenceStatus() const override {
    auto* convergence = reg.try_get<CConvergenceStatus>(e);
    if (convergence) {
      return convergence->stepStatus;
    } else {
      return ConvergenceStatus::None;
    }
  }

  bool IsStatic() const override {
    return reg.all_of<TagStaticActor>(e);
  }

  bool HasRootTransform() const override {
    return reg.try_get<CRootTransform const>(e) != nullptr;
  }

  TransformRT GetRootTransform() const override {
    auto const* transform = reg.try_get<CRootTransform const>(e);
    return transform ? transform->worldFromLocal : TransformRT{};
  }

  static void SetRootTransformImpl(
      entt::registry& reg,
      entt::entity e,
      TransformRT const& worldFromLocal,
      Error& error) {
    MOCHI_ERROR_IF(
        reg.any_of<TagArticulatedLinkActor>(e),
        error,
        "The transform of a nested link actor cannot be directly set. Please use the articulated actor's transform and pose instead.");
    MOCHI_ERROR_IF(
        reg.any_of<TagSoftSkinnedActor>(e),
        error,
        "The transform of a nested soft actor cannot be directly set.");
    MOCHI_ERROR_RETURN(error);

    // Articulated actors: set the root and recompute derived state.
    if (reg.all_of<TagArticulatedActor>(e)) {
      articulated::compound::SetArticulatedRootTransform(reg, e, worldFromLocal);
      return;
    }

    // If this is a rigid dynamic actor, ensure the rigid state is also set
    if (ecs::TryInvokeOnEntity(&rigid::SetRootTransform, reg, e, std::cref(worldFromLocal))) {
    } else {
      // Otherwise, just set the root transform
      MOCHI_ASSERT_VERBOSE(reg.all_of<CRootTransform>(e), "Expected a root transform");
      reg.get<CRootTransform>(e).worldFromLocal = worldFromLocal;
    }

    // External state changes invalidate step history.
    InvalidateActorStepHistory(reg, e);
  }

  void SetRootTransform(TransformRT const& worldFromLocal, Error& error) override {
    SetRootTransformImpl(reg, e, worldFromLocal, error);
  }

  Real3 GetRigidCenterOfMassLocal(Error& error) const override {
    MOCHI_ERROR_RETURN(error, {});

    if (auto const* rbInertia = reg.try_get<CRigidBodyInertia const>(e)) {
      // Rigid actor.
      return ToReal3(rbInertia->GetCenterOfMassLocal());
    } else if (auto const* volDisc = reg.try_get<CFemVolumeDiscretizationP1Q4 const>(e)) {
      // Soft actor. Compute the CoM on the fly.
      Vec4r comLocal;
      real mass = 0_r;
      ComputeCenterOfMassFem(MakeSpan(volDisc->femElements), 1_r, comLocal, mass);
      return ToReal3(comLocal);
    } else {
      MOCHI_ERROR_SET(error, "Center of mass not supported for this type of actor.");
      return {};
    }
  }

  TransformRT GetCenterOfMassTransform(Error& error) const override {
    MOCHI_ERROR_RETURN(error, {});
    auto const* rbInertia = MOCHI_TRY_GET(CRigidBodyInertia const, reg, e, error);
    MOCHI_ERROR_RETURN(error, {});

    // Get the root transform. Reuse the rotation, modify the translation.
    TransformRT transform = reg.get<CRootTransform const>(e).worldFromLocal;
    transform.SetTranslation(transform.TransformPoint(rbInertia->GetCenterOfMassLocal()));
    return transform;
  }

  void SetCenterOfMassTransform(TransformRT const& worldFromCom, Error& error) override {
    MOCHI_ERROR_RETURN(error);
    auto const* rbInertia = reg.try_get<CRigidBodyInertia const>(e);
    MOCHI_ERROR_IF_NOT(rbInertia, error, "This is not a dynamic rigid actor");
    MOCHI_ERROR_RETURN(error);
    TransformRT rootTransform;
    rigid::RigidStateToRootTransform(
        rbInertia->GetCenterOfMassLocal(), worldFromCom, rootTransform);
    SetRootTransformImpl(reg, e, rootTransform, error);
  }

  Real6 GetRigidMomentOfInertiaLocal(Error& error) const override {
    // If the entity is a rigid body, simply return the data in the component.
    auto const* rbInertia = reg.try_get<CRigidBodyInertia const>(e);
    if (rbInertia) {
      auto const& moi = rbInertia->GetMomentOfInertiaLocal();
      // Return just the upper-right triangle of the 3x3 matrix
      return Real6{moi[0][0], moi[0][1], moi[0][2], moi[1][1], moi[1][2], moi[2][2]};
    } else {
      MOCHI_ERROR_SET(error, "Not a dynamic rigid actor");
    }
    return {};
  }

  void SetVelocity(Real3 const& linearVel, Real3 const& angularVel, Error& error) override {
    bool const isStatic = reg.all_of<TagStaticActor>(e);
    MOCHI_ERROR_IF(
        isStatic && linearVel != Real3{},
        error,
        "Cannot set the velocity of a static actor to any value other than zero.");
    MOCHI_ERROR_IF(
        isStatic && angularVel != Real3{},
        error,
        "Cannot set the velocity of a static actor to any value other than zero.");
    MOCHI_ERROR_IF(
        (!reg.any_of<TagStaticActor, TagRigidActor, TagSoftActor>(e) ||
         reg.any_of<TagArticulatedLinkActor, TagRomActor, TagSoftSkinnedActor>(e)),
        error,
        "SetVelocity is not supported for this actor type.");
    MOCHI_ERROR_IF_NOT(IsFinite(linearVel), error, "Linear velocity must be finite.");
    MOCHI_ERROR_IF_NOT(IsFinite(angularVel), error, "Angular velocity must be finite.");
    MOCHI_ERROR_RETURN(error);

    if (isStatic) {
      // Make previous and current transforms match.
      auto& rootTransform = reg.get<CRootTransform>(e);
      rootTransform.worldFromLocalPrev = rootTransform.worldFromLocal;
    } else if (reg.all_of<TagRigidActor>(e)) {
      auto& rigidVel = reg.get<CRigidVel<TimeStep::Current>>(e).value;

      // Set the velocity of the center of mass
      rigidVel.SetVCom(ToSimd(linearVel));

      // Set the angular velocity
      rigidVel.SetOmega(ToSimd(angularVel));

    } else {
      // TODO[Nate] - It would be great if there was a way to give a soft actor an "angular"
      //              velocity about some pivot point.
      MOCHI_ASSERT_VERBOSE(reg.all_of<TagSoftActor>(e), "Unexpected actor type.");
      MOCHI_ERROR_IF(
          Norm(angularVel) > 0_r, error, "Soft actors do not support setting angular velocity.");

      auto const& actorDofInfo = reg.get<CActorDofInfo const>(e);
      int numNodes = GetNumNodes();
      MOCHI_ERROR_IF(
          actorDofInfo.dofsSize != numNodes * 3,
          error,
          "Number of DOFs does not match number of nodes.");
      MOCHI_ERROR_RETURN(error);

      // Transform linear velocity from world-space to local-space.
      auto const localFromWorld = Invert(GetRootTransform());
      auto const localLinearVel = localFromWorld.TransformDirection(linearVel);

      auto& currVel = reg.get<CVelocitySlice<real, TimeStep::Current>>(e);
      for (int i = 0; i < numNodes; ++i) {
        for (int k = 0; k < 3; ++k) {
          currVel.value[3 * i + k] = localLinearVel[k];
        }
      }
    }

    // All soft, rigid and static actors have CPrevRigidVelocity
    auto& prevRigidVelocity = reg.get<CPrevRigidVelocity>(e);
    prevRigidVelocity.linearVelocityWorld = ToSimd(linearVel);
    prevRigidVelocity.angularVelocityWorld = ToSimd(angularVel);

    // External state changes invalidate step history.
    InvalidateActorStepHistory(reg, e);
  }

  Real3 GetLinearVelocity(Error& error) const override {
    auto const* prev = MOCHI_TRY_GET(CPrevRigidVelocity const, reg, e, error);
    MOCHI_ERROR_RETURN(error, {});
    return ToReal3(prev->linearVelocityWorld);
  }

  Real3 GetAngularVelocity(Error& error) const override {
    auto const* prev = MOCHI_TRY_GET(CPrevRigidVelocity const, reg, e, error);
    MOCHI_ERROR_RETURN(error, {});
    return ToReal3(prev->angularVelocityWorld);
  }

  real GetMass(Error& error) const override {
    MOCHI_ERROR_IF(reg.any_of<TagStaticActor>(e), error, "Cannot get mass of static actor");
    MOCHI_ERROR_RETURN(error, 0_r);

    static_assert(
        static_cast<int>(ActorType::Count) == 6,
        "Please update the logic below if new actor types are introduced");
    if (reg.any_of<TagArticulatedActor>(e)) {
      return articulated::GetActorMass(reg, e);
    } else if (reg.any_of<TagRigidActor>(e)) {
      return rigid::GetActorMass(reg, e);
    } else if (reg.any_of<TagSoftActor>(e)) {
      return soft::GetActorMass(reg, e);
    } else if (reg.any_of<TagShellActor>(e)) {
      return shell::GetActorMass(reg, e);
    } else if (reg.any_of<TagRodActor>(e)) {
      return rod::GetActorMass(reg, e);
    } else {
      MOCHI_ERROR_SET(error, "Unsupported actor type");
      return 0_r;
    }
  }

  real GetDensity(Error& error) const override {
    MOCHI_ERROR_IF(
        reg.any_of<TagStaticActor>(e), error, "GetDensity not supported for static actors.");
    MOCHI_ERROR_RETURN(error, 0_r);

    if (auto const* rigid = reg.try_get<CRigidBodyInertia const>(e)) {
      // Rigid
      return rigid->GetDensity();
    } else if (auto const* material = reg.try_get<CSoftMaterialParams const>(e)) {
      // Soft
      return material->density;
    } else {
      MOCHI_ERROR_SET(error, "GetDensity not supported for this actor type.");
      return 0_r;
    }
  }

  void SetDensity(real density, Error& error) override {
    MOCHI_ERROR_IF(
        !IsFinite(density) || (density <= 0_r), error, "Density must be positive and finite.");
    MOCHI_ERROR_IF(
        reg.any_of<TagStaticActor>(e), error, "SetDensity not supported for static actors.");
    MOCHI_ERROR_RETURN(error);

    if (auto* rigidInertia = reg.try_get<CRigidBodyInertia>(e)) {
      // Rigid
      rigidInertia->SetDensity(density);
    } else if (auto* material = reg.try_get<CSoftMaterialParams>(e)) {
      // Soft
      auto const prevDensity = material->density;
      material->density = density;
      if (density != prevDensity) {
        ecs::TryInvokeOnEntity(&soft::UpdateSoftMass, reg, e);
      }
    } else {
      MOCHI_ERROR_SET(error, "SetDensity not supported for this actor type.");
    }

    // Invalidate actor convergence weights.
    InvalidateActorConvergenceWeights(reg, e);
  }

  void SetInertiaProperties(
      real mass,
      Real3 const& centerOfMass,
      Real6 const& momentOfInertia,
      Error& error) override {
    MOCHI_ERROR_IF(
        reg.any_of<TagStaticActor>(e),
        error,
        "SetInertiaProperties is not supported for static actors.");
    MOCHI_ERROR_IF(
        reg.any_of<TagArticulatedLinkActor>(e),
        error,
        "SetInertiaProperties is not supported for articulated links. It is only supported for standalone dynamic rigid actors.");
    MOCHI_ERROR_IF(!IsFinite(mass) || mass <= 0_r, error, "Mass must be positive and finite.");
    MOCHI_ERROR_IF(!IsFinite(centerOfMass), error, "Center of mass must be finite.");
    MOCHI_ERROR_IF(!IsFinite(momentOfInertia), error, "Moment of inertia tensor must be finite.");
    MOCHI_ERROR_RETURN(error);

    auto* rigidInertia = reg.try_get<CRigidBodyInertia>(e);
    MOCHI_ERROR_IF_NOT(
        rigidInertia, error, "SetInertiaProperties is only supported for rigid actors.");
    MOCHI_ERROR_RETURN(error);

    if (!IsMomentOfInertiaValid(momentOfInertia)) {
      MOCHI_LOG_WARNING(
          "New moment-of-inertia tensor for actor \"%s\" is not physically valid: principal moments must be non-negative and satisfy the triangle inequality.",
          GetName());
    }

    // Convert Real3 → Vec4r (xyz, w=0)
    Vec4r const comLocal = ToSimd(centerOfMass);

    // Convert Real6 (upper triangle: xx, xy, xz, yy, yz, zz) → VMatrix3x3r (symmetric 3×3)
    VMatrix3x3r const moi = {
        Vec4r{momentOfInertia[0], momentOfInertia[1], momentOfInertia[2]},
        Vec4r{momentOfInertia[1], momentOfInertia[3], momentOfInertia[4]},
        Vec4r{momentOfInertia[2], momentOfInertia[4], momentOfInertia[5]}};

    rigidInertia->SetInertiaProperties(mass, comLocal, moi);

    // Invalidate actor convergence weights.
    InvalidateActorConvergenceWeights(reg, e);
  }

  real GetElasticEnergy(Error& error) const override {
    auto const* energy = reg.try_get<CQueryElasticEnergy const>(e);
    MOCHI_ERROR_IF_NOT(energy, error, "Elastic energy query not registered.");
    MOCHI_ERROR_RETURN(error, {});

    MOCHI_ERROR_IF_NOT(
        energy->isEnergyAtRestInitialized, error, "Wait one step before querying the energy.");
    MOCHI_ERROR_RETURN(error, {});

    return energy->energy - energy->energyAtRest;
  }

  ContactParams GetContactParams(Error& error) const override {
    auto const* params = MOCHI_TRY_GET(ContactParams, reg, e, error);
    MOCHI_ERROR_RETURN(error, {})
    return *params;
  }

  void SetContactParams(ContactParams const& newParams, Error& error) override {
    auto* params = MOCHI_TRY_GET(ContactParams, reg, e, error);
    MOCHI_ERROR_RETURN(error);
    ValidateContactParams(newParams, error);
    MOCHI_ERROR_RETURN(error);
    *params = newParams;
  }

  ColliderType GetColliderType() const override {
    auto const* colliderInfo = reg.try_get<CColliderInfo const>(e);
    return colliderInfo ? colliderInfo->type : ColliderType::None;
  }

  RecenteringParams GetRecenteringParams() const override {
    auto const* comp = reg.try_get<CRecenteringParams const>(e);
    return comp ? *comp : RecenteringParams{.useRecentering = false};
  }

  void SetRecenteringParams(RecenteringParams const& params, Error& error) override {
    auto* comp = MOCHI_TRY_GET(CRecenteringParams, reg, e, error);
    MOCHI_ERROR_RETURN(error);
    *comp = params;
  }

  Span<real const> GetDisplacements(Error& error) const override {
    using CurrentDisplacement = CFinalDisplacementRef<TimeStep::Current>;
    auto const* displacement = MOCHI_TRY_GET(CurrentDisplacement, reg, e, error);
    MOCHI_ERROR_RETURN(error, {});
    return displacement->value;
  }

  void SetDisplacements(Span<real const> displacements, Error& error) override {
    using CurrentDisplacement = CDisplacementSlice<real, TimeStep::Current>;
    auto* currentDisplacement = MOCHI_TRY_GET(CurrentDisplacement, reg, e, error);
    MOCHI_ERROR_RETURN(error);
    MOCHI_ERROR_IF_NOT(
        currentDisplacement->value.size() == displacements.size(),
        error,
        "Incorrect displacements size.");
    MOCHI_ERROR_IF_NOT(IsFinite(displacements), error, "Displacements must be finite.");
    MOCHI_ERROR_RETURN(error);

    currentDisplacement->value = AsConstView(displacements);

    // External state changes invalidate step history.
    InvalidateActorStepHistory(reg, e);
  }

  SoftMaterialParams GetSoftMaterialParams(Error& error) const override {
    MOCHI_ERROR_IF_NOT(
        reg.all_of<TagSoftActor>(e),
        error,
        "GetSoftMaterialParams is only supported for soft actors.");
    MOCHI_ERROR_RETURN(error, {});

    // TODO[T135019972] - Store the SoftMaterialParams as a component, instead of translating
    // between SoftMaterialParams and SmithNeoHookean here.
    SoftMaterialParams outParams;
    soft::GetMaterialParams(reg.get<CSoftMaterialParams>(e), outParams);
    return outParams;
  }

  void SetSoftMaterialParams(SoftMaterialParams const& params, Error& error) override {
    MOCHI_ERROR_IF_NOT(
        reg.all_of<TagSoftActor>(e),
        error,
        "SetSoftMaterialParams is only supported for soft actors.");
    MOCHI_ERROR_RETURN(error);
    ValidateSoftMaterialParams(params, error);
    MOCHI_ERROR_RETURN(error);

    auto& softMaterialParams = reg.get<CSoftMaterialParams>(e);
    auto prevDensity = softMaterialParams.density;
    soft::SetMaterialParams(params, softMaterialParams);
    if (softMaterialParams.density != prevDensity) {
      ecs::TryInvokeOnEntity(&soft::UpdateSoftMass, reg, e);
    }

    // Invalidate actor convergence weights.
    InvalidateActorConvergenceWeights(reg, e);

    // Invalidate cached reference elastic energy.
    if (auto* query = reg.try_get<CQueryElasticEnergy>(e)) {
      query->isEnergyAtRestInitialized = false;
    }
  }

  void
  SetSoftMaterialParamsField(SoftMaterialParams const& params, int elementIndex, Error& error) {
    MOCHI_ERROR_IF_NOT(
        reg.all_of<TagSoftActor>(e),
        error,
        "SetSoftMaterialParamsField is only supported for soft actors.");
    MOCHI_ERROR_RETURN(error);

    soft::SetMaterialParamsField(
        params, elementIndex, GetNumElements(), reg.get<CSoftMaterialParams>(e), error);
    MOCHI_ERROR_RETURN(error);

    // Invalidate actor convergence weights.
    InvalidateActorConvergenceWeights(reg, e);

    // Invalidate cached reference elastic energy.
    if (auto* query = reg.try_get<CQueryElasticEnergy>(e)) {
      query->isEnergyAtRestInitialized = false;
    }
  }

  SoftMaterialParams GetSoftMaterialParamsField(int elementIndex, Error& error) const {
    MOCHI_ERROR_IF_NOT(
        reg.all_of<TagSoftActor>(e),
        error,
        "GetSoftMaterialParamsField is only supported for soft actors.");
    MOCHI_ERROR_RETURN(error, {});

    int const numElements = GetNumElements();
    MOCHI_ERROR_IF(
        elementIndex < 0 || elementIndex >= numElements,
        error,
        "Element index should be greater than or equal to zero and smaller than the number of elements.");
    MOCHI_ERROR_RETURN(error, {});

    // TODO[T135019972] - Store the SoftMaterialParams as a component, instead of translating
    // between SoftMaterialParams and SmithNeoHookean here.
    SoftMaterialParams outParams;
    soft::GetMaterialParamsField(reg.get<CSoftMaterialParams>(e), elementIndex, outParams);
    return outParams;
  }

  std::string_view GetContactLayer() const override {
    auto const* layer = reg.try_get<CContactLayer>(e);
    return layer ? std::string_view(layer->name) : std::string_view();
  }

  void SetContactLayer(std::string_view const& layer) override {
    auto& table = reg.ctx<CContactFilterTable>();
    auto& comp = reg.get<CContactLayer>(e);
    comp.id = GetOrAddContactLayerId(table, layer);
    comp.name = std::string(layer);
  }

  int GetNumDofs() const override {
    auto* dofInfo = reg.try_get<CActorDofInfo>(e);
    return dofInfo ? dofInfo->dofsSize : 0;
  }

  void GetDofValues(Span<int const> dofIndices, Span<real> outDofValues, Error& error)
      const override {
    MOCHI_ERROR_RETURN(error);
    MOCHI_ERROR_IF(
        !dofIndices.empty() && dofIndices.size() != outDofValues.size(), error, "Invalid sizes");
    MOCHI_ERROR_RETURN(error);
    auto const* dofInfo = MOCHI_TRY_GET(CActorDofInfo, reg, e, error);
    MOCHI_ERROR_RETURN(error);
    MOCHI_ERROR_IF(
        dofIndices.empty() && outDofValues.size() != dofInfo->dofsSize, error, "Invalid sizes");
    if (!dofIndices.empty()) {
      auto const [minIndex, maxIndex] = MinMax(dofIndices);
      MOCHI_ERROR_IF(
          minIndex < 0 || maxIndex >= dofInfo->dofsSize, error, "DoF index out-of-range.");
    }
    MOCHI_ERROR_RETURN(error);

    // Get the pose from the actor. Create a temporary container if #dofs and #pose do not match or
    // if the caller requests specific dofs.
    Span<real> pose = outDofValues;
    std::vector<real> poseContainer;
    if (!dofIndices.empty() || dofInfo->dofsSize != dofInfo->poseSize) {
      poseContainer.resize(dofInfo->poseSize);
      pose = poseContainer;
    }
    GetSolutions(
        AsView(pose), reg, MakeSingletonConstSpan(e), reg.get<CDofOffset const>(e).poseOffset);

    // Check if we're done
    if (poseContainer.empty()) {
      return;
    }

    // If necessary, convert the pose to dofs. Create a temporary container if the caller requests
    // specific dofs.
    Span<real> dofs = outDofValues;
    std::vector<real> dofsContainer;
    if (dofInfo->dofsSize != dofInfo->poseSize) {
      if (!dofIndices.empty()) {
        dofsContainer.resize(dofInfo->dofsSize);
        dofs = dofsContainer;
      }
      if (reg.all_of<TagRigidActor>(e)) {
        ConvertRigidPoseToDofs(pose, dofs);
      } else if (reg.all_of<TagArticulatedActor>(e)) {
        auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
        auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(e);
        articulated::ConvertPoseToDofs(
            joints->jointTypes, joints->dofInfo, poseInfo, AsConstView(pose), AsView(dofs));
      } else if (reg.all_of<TagRodActor>(e)) {
        // For rods, DoFs are the first dofsSize entries of the pose; the remainder are axes.
        std::copy_n(pose.data(), dofInfo->dofsSize, dofs.data());
      } else {
        MOCHI_ASSERT_VERBOSE(reg.all_of<TagRomActor>(e), "Unexpected actor type");
        rom::pivoted::ConvertPoseToDofs(pose, dofs);
      }

      // Check if we're done
      if (dofsContainer.empty()) {
        return;
      }
    } else {
      dofs = pose;
    }

    // Select the dofs requested by the caller
    for (size_t i = 0; i < dofIndices.size(); ++i) {
      outDofValues[i] = dofs[dofIndices[i]];
    }
  }

  void QueryNodesInVolumeLocal(
      Aabb const& volume,
      bool boundaryOnly,
      std::function<void(int, Real3)> const& callback,
      Error& error) const override {
    return QueryNodesInVolumeLocalImpl(reg, e, volume, boundaryOnly, callback, error);
  }

  void QueryNodesInVolumeLocal(
      Obb const& volume,
      bool boundaryOnly,
      std::function<void(int, Real3)> const& callback,
      Error& error) const override {
    return QueryNodesInVolumeLocalImpl(reg, e, volume, boundaryOnly, callback, error);
  }

  void QueryNodesInVolumeLocal(
      Sphere const& volume,
      bool boundaryOnly,
      std::function<void(int, Real3)> const& callback,
      Error& error) const override {
    return QueryNodesInVolumeLocalImpl(reg, e, volume, boundaryOnly, callback, error);
  }

  void GetPointsDistanceToSurface(
      Span<Real3 const> pointsWorld,
      Span<real> outDistances,
      Error& error) const override {
    auto const numPoints = isize(pointsWorld);
    MOCHI_ERROR_IF(
        numPoints != isize(outDistances), error, "Size mismatch between input and output");
    MOCHI_ERROR_RETURN(error);

    // Retrieve Actor's collider.
    auto const* colliderInfo = MOCHI_TRY_GET(CColliderInfo, reg, e, error);
    MOCHI_ERROR_IF(
        !colliderInfo || colliderInfo->type == ColliderType::None, error, "Actor has no collider");
    MOCHI_ERROR_IF(
        reg.all_of<CSdfMapping<TimeStep::Current>>(e),
        error,
        "Distance queries against soft actors is not yet supported");
    MOCHI_ERROR_RETURN(error);

    // Mapped SDF colliders are not supported.
    if (colliderInfo->type == ColliderType::Sdf) {
      auto const* map = reg.try_get<CSdfMapping<TimeStep::Current> const>(e);
      MOCHI_ERROR_IF(map, error, "Distance queries on soft actors are not supported");
      MOCHI_ERROR_RETURN(error);
    }

    // Transform given points from world-space to actor-space.
    auto const worldFromActor = GetRootTransform();
    auto const actorFromWorld = Invert(worldFromActor);
    DynamicArray<Real3> pointsLocal;
    pointsLocal.resize_noinit(numPoints);
    ArrayTransformPoints<false>(MakeSpan(pointsLocal), pointsWorld, actorFromWorld);

    // Handle query according to collider type.
    ContactDetectionParams params{.tolerance = kInf};
    DynamicArray<int> indices;
    indices.reserve(numPoints);
    DynamicArray<Real3> contacts;
    contacts.reserve(numPoints);
    SdfInfo sdf;
    sdf.reserve(numPoints);
    bool unused = {};

    static_assert(
        static_cast<int>(ColliderType::Count) == 8,
        "Please update the following switch statement if the ColliderType enum changes.");

    switch (colliderInfo->type) {
      case ColliderType::Sphere:
        FindPointContactsParallel(
            pointsLocal,
            &reg.get<CSphereCollider const>(e).shape,
            params,
            TransformRT{},
            indices,
            contacts,
            sdf,
            unused);
        break;
      case ColliderType::Box:
        FindPointContactsParallel(
            pointsLocal,
            &reg.get<CBoxCollider const>(e).shape,
            params,
            TransformRT{},
            indices,
            contacts,
            sdf,
            unused);
        break;
      case ColliderType::Mesh:
        FindPointContactsParallel(
            pointsLocal,
            static_cast<MeshCollider const*>(&reg.get<CMeshCollider const>(e)),
            params,
            TransformRT{},
            indices,
            contacts,
            sdf,
            unused);
        break;
      case ColliderType::Plane:
        FindPointContactsParallel(
            pointsLocal,
            &reg.get<CPlaneCollider const>(e).shape,
            params,
            TransformRT{},
            indices,
            contacts,
            sdf,
            unused);
        break;
      case ColliderType::Sdf: {
        // NOTE: SDF generation is asynchronous. If not yet ready, wait until it is available.
        std::shared_ptr<Sdf const> sdfShape = reg.get<CSdfCollider const>(e).shape;
        if (!sdfShape) {
          auto const& gridSdfShape = reg.get<CSdfColliderPending const>(e).gridSdfShape;
          gridSdfShape->GetGridSdfSemaphore().Wait();
          sdfShape = gridSdfShape->GetGridSdf();
        }

        FindPointContactsParallel(
            pointsLocal,
            static_cast<GridSdf const*>(sdfShape.get()),
            params,
            TransformRT{},
            indices,
            contacts,
            sdf,
            unused);
        break;
      }
      case ColliderType::Auto:
        MOCHI_ASSERT(false, "ColliderType::Auto should have been resolved before reaching here.");
        break;
      case ColliderType::PointCloud:
      case ColliderType::None:
      default:
        MOCHI_ERROR_SET(error, "Unsupported collider type");
        break;
    };
    MOCHI_ERROR_RETURN(error);

    // Ensure all points have a corresponding contact in the results.
    MOCHI_ASSERT(isize(indices) == numPoints, "Not all input points generated a contact.");

    // Copy distances to output.
    std::ranges::copy(sdf.val, outDistances.begin());
  }

  Aabb GetAabbLocal(Error& error) const override {
    auto const* bv = MOCHI_TRY_GET(CBoundingVolume<TimeStep::Current>, reg, e, error);
    MOCHI_ERROR_RETURN(error, {});
    return GetAabb(bv->localShape);
  }

  Aabb GetAabbWorld(Error& error) const override {
    auto const* bv = MOCHI_TRY_GET(CBoundingVolume<TimeStep::Current>, reg, e, error);
    MOCHI_ERROR_RETURN(error, {});
    return GetAabb(TransformShape(GetRootTransform(), bv->localShape));
  }

  int GetNumNodes() const {
    return GetMesh().GetNumNodes();
  }

  int GetNumElements() const {
    return GetMesh().GetNumElements();
  }

  MeshDataView GetMesh() const override {
    switch (GetActorMeshSource(reg, e)) {
      case ActorMeshSource::Simplicial: {
        auto const& meshComp = reg.get<CSimplicialMesh>(e);
        MeshDataView view = MakeActorMeshDataView(*meshComp.mesh);
        if (auto const* shapeComp = reg.try_get<CShape const>(e)) {
          if (auto const& skinning = shapeComp->shape->GetMeshSkinning()) {
            view.skinning = SkinningDataView{*skinning};
          }
        }
        return view;
      }
      case ActorMeshSource::Polyline: {
        auto const& plMeshComp = reg.get<CPolylineMesh>(e);
        MeshDataView view{};
        view.nodesPerElement = 2;
        view.coordinates = Flatten(plMeshComp.nodes);
        view.connectivity = plMeshComp.flatConnectivity;
        return view;
      }
      case ActorMeshSource::Surface:
        return GetSurfaceMesh();
      case ActorMeshSource::None:
        return {};
    }

    MOCHI_ASSERT(false, "Unexpected actor mesh source.");
    return {};
  }

  MeshDataView GetSurfaceMesh() const override {
    if (auto const* component = reg.try_get<CSurfaceMesh>(e)) {
      return MakeActorSurfaceMeshDataView(*component->mesh);
    }

    return {};
  }

  MeshDataView GetVisualMesh() const override {
    auto const* component = reg.try_get<CVisualMesh>(e);
    if (!component) {
      return {};
    }

    MeshDataView view = MakeActorMeshDataView(*component->mesh);

    // Skinning data.
    // Note: Rod visual mesh embedding is nonlinear and incompatible with SkinningDataView.
    if (auto const* linearEmbedding =
            dynamic_cast<LinearMeshEmbedding const*>(component->embedding.get())) {
      view.skinning.emplace();
      view.skinning->weightsPerNode =
          static_cast<int>(linearEmbedding->GetNumSkinningWeightsPerEntry());
      view.skinning->indices = linearEmbedding->GetIndices();
      view.skinning->weights = linearEmbedding->GetWeights();
    }

    return view;
  }

  Span<real const> GetSurfaceMeshNodePositionsLocal(Error& error) const override {
    return GetQueryResultSpan(&CQuerySurfaceNodePositions::nodePositions, error);
  }

  ShapeHandle GetReferenceShape(Error& error) const override {
    MOCHI_ERROR_RETURN(error, {});
    auto const* shapeComp = reg.try_get<CShape const>(e);
    MOCHI_ERROR_IF_NOT(shapeComp && shapeComp->shape, error, "Actor has no shape.");
    MOCHI_ERROR_RETURN(error, {});
    auto* contextImpl = assert_cast<ContextImpl*>(scene->GetContext());
    return contextImpl->RegisterShape(shapeComp->shape, error);
  }

  ArticulatedShapeInfo GetArticulatedShapeInfo(Error& error) const override {
    MOCHI_ERROR_RETURN(error, {});
    auto const* articulated = MOCHI_TRY_GET(CArticulatedBodyShape, reg, e, error);
    MOCHI_ERROR_RETURN(error, {});
    return articulated->shape->GetArticulatedShapeInfo();
  }

  Span<Constraint* const> GetArticulatedJointLimitConstraints(Error& error) const override {
    MOCHI_ERROR_RETURN(error, {});
    auto const* constraints = MOCHI_TRY_GET(CArticulatedJointLimits, reg, e, error);
    MOCHI_ERROR_RETURN(error, {});
    return *constraints;
  }

  Span<ArticulatedJointFrictionParams const> GetArticulatedJointFrictionParams(
      Error& error) const override {
    MOCHI_ERROR_IF_NOT(
        reg.all_of<TagArticulatedActor>(e),
        error,
        "GetArticulatedJointFrictionParams is only supported for articulated actors.");
    MOCHI_ERROR_RETURN(error, {});

    // The joint-friction component is always present on articulated actors.
    return reg.get<CArticulatedJointFrictionParams const>(e);
  }

  void SetArticulatedJointFrictionParams(
      Span<ArticulatedJointFrictionParams const> friction,
      Error& error) override {
    MOCHI_ERROR_IF_NOT(
        reg.all_of<TagArticulatedActor>(e),
        error,
        "SetArticulatedJointFrictionParams is only supported for articulated actors.");
    MOCHI_ERROR_RETURN(error);

    auto& current = reg.get<CArticulatedJointFrictionParams>(e);
    MOCHI_ERROR_IF_NOT(
        isize(friction) == isize(current),
        error,
        "SetArticulatedJointFrictionParams size must equal the number of joints.");
    MOCHI_ERROR_RETURN(error);

    // Validate through the shared per-joint validator so the runtime API and the build-time
    // ArticulatedActorParams validation stay in sync.
    for (auto const& params : friction) {
      ValidateFriction(params, error);
    }
    MOCHI_ERROR_RETURN(error);

    std::copy(friction.begin(), friction.end(), current.begin());
  }

  Span<real const> GetArticulatedJointInertiaParams(Error& error) const override {
    MOCHI_ERROR_IF_NOT(
        reg.all_of<TagArticulatedActor>(e),
        error,
        "GetArticulatedJointInertiaParams is only supported for articulated actors.");
    MOCHI_ERROR_RETURN(error, {});

    // The joint-inertia component is always present on articulated actors.
    return reg.get<CArticulatedInertiaParams const>(e);
  }

  void SetArticulatedJointInertiaParams(Span<real const> inertia, Error& error) override {
    MOCHI_ERROR_IF_NOT(
        reg.all_of<TagArticulatedActor>(e),
        error,
        "SetArticulatedJointInertiaParams is only supported for articulated actors.");
    MOCHI_ERROR_RETURN(error);

    auto& current = reg.get<CArticulatedInertiaParams>(e);
    MOCHI_ERROR_IF_NOT(
        isize(inertia) == isize(current),
        error,
        "SetArticulatedJointInertiaParams size must equal the number of joints.");
    MOCHI_ERROR_RETURN(error);

    // Validate through the shared per-joint validator (short-circuits once error is set),
    // mirroring the friction setter above.
    for (real const value : inertia) {
      ValidateInertia(value, error);
    }
    MOCHI_ERROR_RETURN(error);

    std::copy(inertia.begin(), inertia.end(), current.begin());
  }

#define MOCHI_ERROR_RETURN_IF_NOT_ARTICULATED(...)                                            \
  MOCHI_ERROR_RETURN(error, __VA_ARGS__);                                                     \
  MOCHI_ERROR_IF_NOT(reg.all_of<TagArticulatedActor>(e), error, "Not an articulated actor."); \
  MOCHI_ERROR_RETURN(error, __VA_ARGS__);

  void GetArticulatedDofLimits(Span<Real2> outDofLimits, Error& error) const override {
    MOCHI_ERROR_RETURN_IF_NOT_ARTICULATED();

    // Check if outLimits has the correct size
    auto const N = GetNumDofs();
    MOCHI_ERROR_IF_NOT(outDofLimits.size() == N, error, "Invalid size.");
    MOCHI_ERROR_RETURN(error);

    // Initialize all limits as unconstrained (i.e. can range in [-inf, inf])
    Fill(outDofLimits, Real2{-kInf, kInf});

    // Populate from the joint-limit constraints.
    auto const& constraints = reg.get<CArticulatedJointLimits const>(e);
    for (auto const* constr : constraints) {
      auto dofs = constr->GetDofIndicesForActor(0);
      auto minValues = constr->GetLimitMinValues(error);
      auto maxValues = constr->GetLimitMaxValues(error);
      MOCHI_ERROR_RETURN(error);

      auto const M = isize(dofs);
      MOCHI_ASSERT_VERBOSE(M == isize(minValues), "Size mismatch");
      MOCHI_ASSERT_VERBOSE(M == isize(maxValues), "Size mismatch");

      for (int i = 0; i < M; ++i) {
        int const d = dofs[i];
        MOCHI_ASSERT_VERBOSE(d < N, "Dof index out of range");
        outDofLimits[d] = Real2{minValues[i], maxValues[i]};
      }
    }
  }

  Span<ActorHandle const> GetNestedLinkActors(Error& error) const override {
    MOCHI_ERROR_RETURN_IF_NOT_ARTICULATED({});
    return reg.get<CBoneHandles const>(e).bones;
  }

  Span<ActorHandle const> GetNestedSoftActors(Error& error) const override {
    MOCHI_ERROR_RETURN_IF_NOT_ARTICULATED({});
    if (auto const* blended = reg.try_get<CBlendedComposition const>(e)) {
      return blended->softHandles;
    } else {
      return {};
    }
  }

  void GetArticulatedPose(Span<real> outPose, Error& error) const override {
    MOCHI_ERROR_RETURN_IF_NOT_ARTICULATED();
    GetDofValues({}, outPose, error);
  }

  void GetArticulatedLinkTransforms(Span<TransformRT> outWorldFromLinks, Error& error)
      const override {
    MOCHI_ERROR_RETURN_IF_NOT_ARTICULATED();
    articulated::compound::GetLinkTransforms(reg, e, outWorldFromLinks, error);
  }

  void GetArticulatedJointVelocities(Span<real> outVelocities, Error& error) const override {
    MOCHI_ERROR_RETURN_IF_NOT_ARTICULATED();
    articulated::compound::GetArticulatedJointVelocities(reg, e, outVelocities, error);
  }

  static void SetArticulatedTargetPoseOwner(
      TargetOwner owner,
      uint64_t step,
      CTargetOwners& outComp,
      bool oldAndNew) {
    // Set the new-pose target owner
    outComp.newPoseOwner = owner;
    outComp.newPoseStep = step;
    // If externally requested or if velocity is set in the same step, set also the old-pose target
    // owner.
    if (oldAndNew || (outComp.velStep == outComp.newPoseStep)) {
      outComp.oldPoseOwner = outComp.newPoseOwner;
      outComp.oldPoseStep = outComp.newPoseStep;
    }
  }

  static void
  SetArticulatedTargetVelOwner(TargetOwner owner, uint64_t step, CTargetOwners& outComp) {
    // Set the velocity target owner
    outComp.velOwner = owner;
    outComp.velStep = step;
    // Make the old-pose target owner match the new-pose target owner
    outComp.oldPoseOwner = outComp.newPoseOwner;
    outComp.oldPoseStep = outComp.newPoseStep;
  }

  // Set differentiability target owners after a target reset. No-op for non-differentiable actors.
  void SetArticulatedTargetOwnersAfterReset(TargetOwner owner) {
    auto* targetOwners = reg.try_get<CTargetOwners>(e);
    if (targetOwners == nullptr) {
      return;
    }
    auto const step = reg.ctx<CSceneStepCounter const>().value;
    SetArticulatedTargetPoseOwner(owner, step, *targetOwners, /*oldAndNew*/ true);
    // Reset APIs overwrite any pending target velocity with a constant zero target velocity. No
    // public velocity-setting API owns that overwritten velocity target.
    SetArticulatedTargetVelOwner(TargetOwner::ControllerInit, step, *targetOwners);
  }

  void SetArticulatedPoseFromLinks(Span<TransformRT const> worldFromLinks, Error& error) override {
    MOCHI_ERROR_RETURN_IF_NOT_ARTICULATED();

    auto const& links = reg.get<CGroupMembers const>(e).actors;
    MOCHI_ERROR_IF(
        isize(links) != isize(worldFromLinks), error, "Incorrect number of link transforms");
    MOCHI_ERROR_IF_NOT(IsFinite(worldFromLinks), error, "Link transforms must be finite.");
    MOCHI_ERROR_RETURN(error);

    // Set the root transform of each rigid body within the articulation.
    for (int i = 0; i < isize(worldFromLinks); ++i) {
      ecs::InvokeOnEntity(&rigid::SetRootTransform, reg, links[i], std::cref(worldFromLinks[i]));
    }

    // Project the link transforms onto the DOFs of the articulation.
    articulated::compound::SetArticulatedPoseFromLinks(reg, e);

    // If there is a controller, reset the target pose.
    if (reg.all_of<CControllerConstraints>(e)) {
      articulated::compound::ResetTargetLinkTransforms(reg, e, worldFromLinks, error);
      MOCHI_ERROR_RETURN(error);
      SetArticulatedTargetOwnersAfterReset(TargetOwner::PoseFromLinks);
    }
  }

  Span<real const> ConvertArticulatedDofsToPoseIfNeeded(
      Span<real const> dofs,
      DynamicArray<real>& poseContainer,
      Error& error) const {
    MOCHI_ERROR_RETURN(error, {});

    auto const& dofInfo = reg.get<CActorDofInfo const>(e);
    MOCHI_ERROR_IF(isize(dofs) != dofInfo.dofsSize, error, "Invalid pose size");
    MOCHI_ERROR_RETURN(error, {});

    if (dofInfo.poseSize != dofInfo.dofsSize) {
      poseContainer.resize_noinit(dofInfo.poseSize);
      auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
      auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(e);
      articulated::ConvertDofsToPose(
          joints->jointTypes, joints->dofInfo, poseInfo, AsConstView(dofs), AsView(poseContainer));
      return poseContainer;
    } else {
      return dofs;
    }
  }

  void SetArticulatedPoseFromJoints(Span<real const> pose, Error& error) override {
    MOCHI_ERROR_RETURN_IF_NOT_ARTICULATED();

    // If necessary, convert dofs to pose
    MOCHI_FILO_STACK_ALLOCATOR(allocator, sizeof(real) * 4 * 256); // 4 values x 256 joints
    DynamicArray<real> poseContainer(&allocator);
    Span<real const> poseSpan = ConvertArticulatedDofsToPoseIfNeeded(pose, poseContainer, error);
    MOCHI_ERROR_RETURN(error);

    articulated::compound::SetArticulatedBodyPose(reg, e, poseSpan, error);

    // If there is a controller, reset the target pose.
    if (reg.all_of<CControllerConstraints>(e)) {
      articulated::compound::ResetTargetPose(reg, e, poseSpan, error);
      MOCHI_ERROR_RETURN(error);
      SetArticulatedTargetOwnersAfterReset(TargetOwner::PoseFromJoints);
    }
  }

  void SetArticulatedJointVelocities(Span<real const> velocities, Error& error) override {
    MOCHI_ERROR_RETURN_IF_NOT_ARTICULATED();
    articulated::compound::SetArticulatedJointVelocities(reg, e, velocities, error);
    MOCHI_ERROR_RETURN(error);

    // If there is a controller, set the target velocity
    if (reg.all_of<CControllerConstraints>(e)) {
      articulated::compound::SetTargetJointVelocities(reg, e, velocities, error);
      MOCHI_ERROR_RETURN(error);
      if (auto* targetOwners = reg.try_get<CTargetOwners>(e)) {
        SetArticulatedTargetVelOwner(
            TargetOwner::JointVelocities, reg.ctx<CSceneStepCounter const>().value, *targetOwners);
      }
    }
  }

  void AddArticulatedDeltaToPose(
      Span<real const> pose,
      Span<real const> deltaDofs,
      Span<real> outPose,
      Error& error) const override {
    MOCHI_ERROR_RETURN_IF_NOT_ARTICULATED();
    auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
    auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(e);
    auto const& actorDofInfo = reg.get<CActorDofInfo const>(e);
    int const numDofs = actorDofInfo.dofsSize;
    MOCHI_ERROR_IF(isize(pose) != numDofs, error, "Invalid size of pose");
    MOCHI_ERROR_IF(isize(deltaDofs) != numDofs, error, "Invalid size of delta DoFs");
    MOCHI_ERROR_IF(isize(outPose) != numDofs, error, "Invalid size of outPose");
    MOCHI_ERROR_RETURN(error);

    MOCHI_ERROR_IF_NOT(IsFinite(pose), error, "Pose values must be finite.");
    MOCHI_ERROR_IF_NOT(IsFinite(deltaDofs), error, "Delta DoF values must be finite.");
    MOCHI_ERROR_RETURN(error);

    int const poseSize = actorDofInfo.poseSize;
    if (poseSize == numDofs) {
      // If pose size and dofs match, just add the delta
      AsView(outPose) = AsConstView(deltaDofs) + AsConstView(pose);
    } else {
      // Convert dofs to pose
      MOCHI_FILO_STACK_ALLOCATOR(allocator, sizeof(real) * 4 * 256); // 4 values x 256 joints
      ColumnVector<real> poseQuaternions(poseSize, &allocator);
      articulated::ConvertDofsToPose(
          joints->jointTypes, joints->dofInfo, poseInfo, AsConstView(pose), poseQuaternions);

      // Add Lie delta
      articulated::AddLieDeltaToReducedPose(
          joints->jointTypes,
          joints->dofInfo,
          poseInfo,
          poseQuaternions,
          AsConstView(deltaDofs),
          poseQuaternions);

      // Convert back to dofs
      articulated::ConvertPoseToDofs(
          joints->jointTypes, joints->dofInfo, poseInfo, poseQuaternions, AsView(outPose));
    }
  }

  void ComputeArticulatedPoseDelta(
      Span<real const> poseBase,
      Span<real const> poseTarget,
      Span<real> outDeltaDofs,
      Error& error) const override {
    MOCHI_ERROR_RETURN_IF_NOT_ARTICULATED();
    auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
    auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(e);
    auto const& actorDofInfo = reg.get<CActorDofInfo const>(e);
    int const numDofs = actorDofInfo.dofsSize;
    MOCHI_ERROR_IF(isize(outDeltaDofs) != numDofs, error, "Invalid size of delta");
    MOCHI_ERROR_RETURN(error);

    // If necessary, convert dofs to pose
    MOCHI_FILO_STACK_ALLOCATOR(allocator, sizeof(real) * 8 * 256); // 2 x 4 values x 256 joints
    DynamicArray<real> poseContainerBase(&allocator);
    Span<real const> poseSpanBase =
        ConvertArticulatedDofsToPoseIfNeeded(poseBase, poseContainerBase, error);
    MOCHI_ERROR_RETURN(error);
    DynamicArray<real> poseContainerTarget(&allocator);
    Span<real const> poseSpanTarget =
        ConvertArticulatedDofsToPoseIfNeeded(poseTarget, poseContainerTarget, error);
    MOCHI_ERROR_RETURN(error);

    // Compute delta.
    articulated::ComputeLieDeltaReducedPose(
        joints->jointTypes,
        joints->dofInfo,
        poseInfo,
        AsConstView(poseSpanBase),
        AsConstView(poseSpanTarget),
        AsView(outDeltaDofs));
  }

  void GetArticulatedPoseDistance(
      Span<real const> poseA,
      Span<real const> poseB,
      Span<real> outTransDistances,
      Span<real> outRotDistances,
      Error& error) const override {
    MOCHI_ERROR_RETURN_IF_NOT_ARTICULATED();
    auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
    auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(e);
    MOCHI_ERROR_IF_NOT(
        outTransDistances.size() == isize(joints->jointTypes), error, "Invalid output size.");
    MOCHI_ERROR_IF_NOT(
        outRotDistances.size() == isize(joints->jointTypes), error, "Invalid output size.");
    MOCHI_ERROR_RETURN(error);

    // If necessary, convert dofs to pose
    MOCHI_FILO_STACK_ALLOCATOR(allocator, sizeof(real) * 8 * 256); // 2 x 4 values x 256 joints
    DynamicArray<real> poseContainerA(&allocator);
    Span<real const> poseSpanA = ConvertArticulatedDofsToPoseIfNeeded(poseA, poseContainerA, error);
    MOCHI_ERROR_RETURN(error);
    DynamicArray<real> poseContainerB(&allocator);
    Span<real const> poseSpanB = ConvertArticulatedDofsToPoseIfNeeded(poseB, poseContainerB, error);
    MOCHI_ERROR_RETURN(error);

    // Compute distances.
    articulated::ReducedPoseDistance(
        joints->jointTypes,
        poseInfo,
        AsConstView(poseSpanA),
        AsConstView(poseSpanB),
        outTransDistances,
        outRotDistances);
  }

  bool HasArticulatedPoseController(Error& error) const override {
    MOCHI_ERROR_RETURN_IF_NOT_ARTICULATED(false);
    return reg.all_of<CControllerConstraints>(e);
  }

  void AddArticulatedPoseController(PoseControllerParams const& params, Error& error) override {
    MOCHI_ERROR_RETURN_IF_NOT_ARTICULATED();
    articulated::compound::AddPoseController(reg, e, scene, params, error);
  }

  void SetArticulatedForceAndTargetPose(
      Span<real const> target,
      Span<experimental::ControlType const> controlTypes,
      Span<int const> dofOrLinkIndices,
      Error& error) {
    using experimental::ControlType;
    // Local helper function - returns value count for each control type
    auto const getValueCount = [](ControlType type) -> int {
      switch (type) {
        case ControlType::SingleDof:
        case ControlType::Force:
          return 1;
        case ControlType::LinkPos:
        case ControlType::LinkRot:
          return 3;
        case ControlType::Count:
          return 0;
      }
      return 0;
    };

    MOCHI_ERROR_RETURN_IF_NOT_ARTICULATED();

    // Validate sizes
    MOCHI_ERROR_IF(
        isize(controlTypes) != isize(dofOrLinkIndices),
        error,
        "controlTypes and dofOrLinkIndices size mismatch");
    MOCHI_ERROR_RETURN(error);

    // Compute expected target size and validate
    int expectedTargetSize = 0;
    for (auto const type : controlTypes) {
      expectedTargetSize += getValueCount(type);
    }
    MOCHI_ERROR_IF(isize(target) != expectedTargetSize, error, "Target size mismatch");
    MOCHI_ERROR_IF_NOT(IsFinite(target), error, "Target values must be finite.");
    MOCHI_ERROR_RETURN(error);

    // Validate dofOrLinkIndices values and check for duplicates
    int const numDofs = GetNumDofs();
    auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
    int const numLinks = isize(GetNestedLinkActors(ErrorAssert{})); // = num active joints

    {
      // Bool bitset arrays for O(1) duplicate detection — scoped for release after use
      {
        MOCHI_FILO_STACK_ALLOCATOR(seenAlloc, sizeof(bool) * 256 * 4);
        DynamicArray<bool> seenDofs(numDofs, false, &seenAlloc);
        DynamicArray<bool> seenLinkPos(numLinks, false, &seenAlloc);
        DynamicArray<bool> seenLinkRot(numLinks, false, &seenAlloc);
        DynamicArray<bool> jointControlledLinkJointPair(numLinks, false, &seenAlloc);

        // Check which control types exist and validate indices
        bool hasLinkTarget = false;
        bool hasPoseTarget = false;
        for (int i = 0; i < isize(controlTypes); ++i) {
          auto const type = controlTypes[i];
          int const idx = dofOrLinkIndices[i];
          switch (type) {
            case ControlType::SingleDof:
              hasPoseTarget = true;
              MOCHI_ERROR_IF(idx < 0 || idx >= numDofs, error, "DOF index out of range");
              MOCHI_ERROR_RETURN(error);
              MOCHI_ERROR_IF(seenDofs[idx], error, "DOF indices are not unique");
              seenDofs[idx] = true;
              // Mark which joint this DOF belongs to
              for (int jointIdx = 0; jointIdx < numLinks; ++jointIdx) {
                auto const& dofs = joints->dofInfo[jointIdx];
                if (idx >= dofs.offset && idx < (dofs.offset + dofs.GetSize())) {
                  jointControlledLinkJointPair[jointIdx] = true;
                  break;
                }
              }
              break;
            case ControlType::Force:
              MOCHI_ERROR_IF(idx < 0 || idx >= numDofs, error, "DOF index out of range");
              MOCHI_ERROR_RETURN(error);
              MOCHI_ERROR_IF(seenDofs[idx], error, "DOF indices are not unique");
              seenDofs[idx] = true;
              break;
            case ControlType::LinkPos:
              hasLinkTarget = true;
              MOCHI_ERROR_IF(idx < 0 || idx >= numLinks, error, "Link index out of range");
              MOCHI_ERROR_RETURN(error);
              MOCHI_ERROR_IF(
                  seenLinkPos[idx], error, "Link position control indices are not unique");
              seenLinkPos[idx] = true;
              break;
            case ControlType::LinkRot:
              hasLinkTarget = true;
              MOCHI_ERROR_IF(idx < 0 || idx >= numLinks, error, "Link index out of range");
              MOCHI_ERROR_RETURN(error);
              MOCHI_ERROR_IF(
                  seenLinkRot[idx], error, "Link rotation control indices are not unique");
              seenLinkRot[idx] = true;
              break;
            case ControlType::Count:
              MOCHI_ERROR_IF(true, error, "Invalid ControlType::Count");
              break;
          }
          MOCHI_ERROR_RETURN(error);
        }

        MOCHI_ERROR_IF(
            (hasLinkTarget || hasPoseTarget) && !reg.all_of<CControllerConstraints>(e),
            error,
            "No pose controller.");
        MOCHI_ERROR_RETURN(error);

        // Validate no link-joint pair has both link and joint controls
        for (int i = 0; i < numLinks; ++i) {
          MOCHI_ERROR_IF(
              (seenLinkPos[i] || seenLinkRot[i]) && jointControlledLinkJointPair[i],
              error,
              "Cannot have both link and joint controls on the same link-joint pair");
        }
        MOCHI_ERROR_RETURN(error);

        // Get current pose
        MOCHI_FILO_STACK_ALLOCATOR(poseAlloc, sizeof(real) * 256);
        DynamicArray<real> pose(&poseAlloc);
        pose.resize_noinit(numDofs);
        GetArticulatedPose(pose, ErrorAssert{});

        // Process single-DOF controls and forces
        int targetIdx = 0;
        for (int i = 0; i < isize(controlTypes); ++i) {
          auto const type = controlTypes[i];
          if (type == ControlType::SingleDof) {
            int const idx = dofOrLinkIndices[i];
            pose[idx] = target[targetIdx];
          }
          targetIdx += getValueCount(type);
        }

        // Process 3D link controls (only if present)
        if (hasLinkTarget) {
          MOCHI_FILO_STACK_ALLOCATOR(linkAlloc, sizeof(TransformRT) * 128);
          DynamicArray<TransformRT> linkTransforms(&linkAlloc);
          linkTransforms.resize_noinit(numLinks);
          GetArticulatedLinkTransforms(linkTransforms, ErrorAssert{});

          targetIdx = 0;
          for (int i = 0; i < isize(controlTypes); ++i) {
            auto const type = controlTypes[i];
            int const idx = dofOrLinkIndices[i];
            int const valueCount = getValueCount(type);

            switch (type) {
              case ControlType::LinkPos: {
                Real3 const pos{target[targetIdx], target[targetIdx + 1], target[targetIdx + 2]};
                linkTransforms[idx].SetTranslation(pos);
                break;
              }
              case ControlType::LinkRot: {
                Real3 const rotVec{target[targetIdx], target[targetIdx + 1], target[targetIdx + 2]};
                Quaternion const rot = Quaternion::FromRotationVector(rotVec);
                linkTransforms[idx].SetRotation(rot);
                break;
              }
              case ControlType::SingleDof:
              case ControlType::Force:
              case ControlType::Count:
                break;
            }
            targetIdx += valueCount;
          }
          articulated::compound::CombinePoseAndLinkTargets(
              reg, e, pose, seenLinkPos, seenLinkRot, linkTransforms);
          SetArticulatedTargetLinkTransforms(linkTransforms, error);
          MOCHI_ERROR_RETURN(error);
        } else if (hasPoseTarget) {
          SetArticulatedTargetPose(pose, error);
          MOCHI_ERROR_RETURN(error);
        }
      } // bitset arrays released

      // Apply forces — scoped for release after use
      {
        MOCHI_FILO_STACK_ALLOCATOR(forceAlloc, (sizeof(real) + sizeof(int)) * 256);
        DynamicArray<real> forceVals(&forceAlloc);
        DynamicArray<int> forceDofs(&forceAlloc);
        forceVals.reserve(isize(controlTypes));
        forceDofs.reserve(isize(controlTypes));
        int forceTargetIdx = 0;
        for (int i = 0; i < isize(controlTypes); ++i) {
          auto const type = controlTypes[i];
          if (type == ControlType::Force) {
            forceDofs.push_back(dofOrLinkIndices[i]);
            forceVals.push_back(target[forceTargetIdx]);
          }
          forceTargetIdx += getValueCount(type);
        }
        if (!forceDofs.empty()) {
          SetExternalForcesOnDofs(forceDofs, forceVals, ErrorAssert{});
        }
      } // force arrays released
    }
  }

#undef MOCHI_ERROR_RETURN_IF_NOT_ARTICULATED

#define MOCHI_ERROR_RETURN_IF_NO_CONTROLLER(...)                                           \
  MOCHI_ERROR_RETURN(error, __VA_ARGS__);                                                  \
  MOCHI_ERROR_IF_NOT(reg.all_of<CControllerConstraints>(e), error, "No pose controller."); \
  MOCHI_ERROR_RETURN(error, __VA_ARGS__);

  void RemoveArticulatedPoseController(Error& error) override {
    MOCHI_ERROR_RETURN_IF_NO_CONTROLLER();

    // Destroy controller constraints
    auto const& constraints = reg.get<CControllerConstraints>(e).impl;
    for (auto const& constraint : constraints) {
      scene->DestroyConstraint(constraint.constraint);
    }

    // Remove controller components
    reg.remove<
        CLinkPosController,
        CLinkRotController,
        CJointController,
        CControllerConstraints,
        CControllerTarget<TimeStep::Current>,
        CControllerTarget<TimeStep::Previous>,
        CControllerTargetVelocity>(e);

    // Remove controller components for differentiability
    articulated::compound::RemoveDifferentiablePoseController(reg, e);
  }

  void SetArticulatedTargetLinkTransforms(Span<TransformRT const> worldFromTargets, Error& error)
      override {
    MOCHI_ERROR_RETURN_IF_NO_CONTROLLER();
    MOCHI_ERROR_IF_NOT(IsFinite(worldFromTargets), error, "Target link transforms must be finite.");
    MOCHI_ERROR_RETURN(error);
    articulated::compound::SetTargetLinkTransforms(reg, e, worldFromTargets, error);
    MOCHI_ERROR_RETURN(error);
    if (auto* targetOwners = reg.try_get<CTargetOwners>(e)) {
      SetArticulatedTargetPoseOwner(
          TargetOwner::TargetLinkTransforms,
          reg.ctx<CSceneStepCounter const>().value,
          *targetOwners,
          /*oldAndNew*/ false);
    }
  }

  void SetArticulatedTargetPose(Span<real const> pose, Error& error) override {
    MOCHI_ERROR_RETURN_IF_NO_CONTROLLER();
    MOCHI_ERROR_IF_NOT(IsFinite(pose), error, "Target pose values must be finite.");
    MOCHI_ERROR_RETURN(error);

    // If necessary, convert dofs to pose
    MOCHI_FILO_STACK_ALLOCATOR(allocator, sizeof(real) * 4 * 256); // 4 values x 256 joints
    DynamicArray<real> poseContainer(&allocator);
    Span<real const> poseSpan = ConvertArticulatedDofsToPoseIfNeeded(pose, poseContainer, error);
    MOCHI_ERROR_RETURN(error);

    articulated::compound::SetTargetPose(reg, e, poseSpan, error);
    MOCHI_ERROR_RETURN(error);
    if (auto* targetOwners = reg.try_get<CTargetOwners>(e)) {
      SetArticulatedTargetPoseOwner(
          TargetOwner::TargetPose,
          reg.ctx<CSceneStepCounter const>().value,
          *targetOwners,
          /*oldAndNew*/ false);
    }
  }

  void ResetArticulatedTargetLinkTransforms(Span<TransformRT const> worldFromTargets, Error& error)
      override {
    MOCHI_ERROR_RETURN_IF_NO_CONTROLLER();
    MOCHI_ERROR_IF_NOT(IsFinite(worldFromTargets), error, "Target link transforms must be finite.");
    MOCHI_ERROR_RETURN(error);
    articulated::compound::ResetTargetLinkTransforms(reg, e, worldFromTargets, error);
    MOCHI_ERROR_RETURN(error);
    SetArticulatedTargetOwnersAfterReset(TargetOwner::ResetTargetLinkTransforms);
  }

  void ResetArticulatedTargetPose(Span<real const> pose, Error& error) override {
    MOCHI_ERROR_RETURN_IF_NO_CONTROLLER();
    MOCHI_ERROR_IF_NOT(IsFinite(pose), error, "Target pose values must be finite.");
    MOCHI_ERROR_RETURN(error);

    // If necessary, convert dofs to pose
    MOCHI_FILO_STACK_ALLOCATOR(allocator, sizeof(real) * 4 * 256); // 4 values x 256 joints
    DynamicArray<real> poseContainer(&allocator);
    Span<real const> poseSpan = ConvertArticulatedDofsToPoseIfNeeded(pose, poseContainer, error);
    MOCHI_ERROR_RETURN(error);

    articulated::compound::ResetTargetPose(reg, e, poseSpan, error);
    MOCHI_ERROR_RETURN(error);
    SetArticulatedTargetOwnersAfterReset(TargetOwner::ResetTargetPose);
  }

  void SetArticulatedTargetVelocity(Span<real const> velocity, Error& error) override {
    MOCHI_ERROR_RETURN_IF_NO_CONTROLLER();
    MOCHI_ERROR_IF_NOT(IsFinite(velocity), error, "Target velocity values must be finite.");
    MOCHI_ERROR_RETURN(error);
    articulated::compound::SetTargetJointVelocities(reg, e, velocity, error);
    MOCHI_ERROR_RETURN(error);
    if (auto* targetOwners = reg.try_get<CTargetOwners>(e)) {
      SetArticulatedTargetVelOwner(
          TargetOwner::TargetVelocity, reg.ctx<CSceneStepCounter const>().value, *targetOwners);
    }
  }

  Span<PoseConstraintInfo const> GetArticulatedPoseConstraints(Error& error) const override {
    MOCHI_ERROR_RETURN(error, {});
    auto const* constraints = MOCHI_TRY_GET(CControllerConstraints const, reg, e, error);
    MOCHI_ERROR_RETURN(error, {});
    return constraints->info;
  }

  void GetArticulatedPoseControllerParams(PoseControllerParams& outParams, Error& error)
      const override {
    MOCHI_ERROR_RETURN_IF_NO_CONTROLLER();
    articulated::compound::GetPoseControllerParams(reg, e, outParams, error);
  }

  void SetArticulatedPoseControllerParams(PoseControllerParams const& params, Error& error)
      override {
    MOCHI_ERROR_RETURN_IF_NO_CONTROLLER();
    articulated::compound::SetPoseControllerParams(reg, e, params, error);
  }

  void GetArticulatedTargetPose(Span<real> outPose, Error& error) const override {
    MOCHI_ERROR_RETURN_IF_NO_CONTROLLER();
    MOCHI_ERROR_IF(outPose.size() != GetNumDofs(), error, "Invalid pose size");
    MOCHI_ERROR_RETURN(error);

    auto pose = MakeConstSpan(reg.get<CControllerTarget<TimeStep::Current> const>(e).JointPose());

    // Copy the pose or convert pose to dofs if necessary
    if (pose.size() == outPose.size()) {
      std::copy(pose.begin(), pose.end(), outPose.begin());
    } else {
      auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
      auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(e);
      articulated::ConvertPoseToDofs(
          joints->jointTypes, joints->dofInfo, poseInfo, AsConstView(pose), AsView(outPose));
    }
  }

  void GetArticulatedTargetLinkTransforms(Span<TransformRT> outWorldFromTargets, Error& error)
      const override {
    MOCHI_ERROR_RETURN_IF_NO_CONTROLLER();
    articulated::compound::GetTargetLinkTransforms(reg, e, outWorldFromTargets, error);
  }

  Span<real const> GetArticulatedControllerForce(Error& error) const override {
    MOCHI_ERROR_RETURN_IF_NO_CONTROLLER({});
    return articulated::compound::GetPoseControllerForce(reg, e, scene, error);
  }

#undef MOCHI_ERROR_RETURN_IF_NO_CONTROLLER

  // Look up a component and return one of its member variables as a Span<real const>.
  // Used for accessing query results.
  template <typename ComponentT, typename MemberT>
  Span<real const> GetQueryResultSpan(MemberT ComponentT::* member, Error& error) const {
    auto const* component = MOCHI_TRY_GET(ComponentT, reg, e, error);
    MOCHI_ERROR_RETURN(error, {});
    auto result = MakeSpan(component->*member);
    MOCHI_ERROR_IF(
        result.empty(),
        error,
        "The data is not available. If you registered the query, then the results should be available after the next simulation step.");
    return result;
  }

  Span<real const> GetNodePositionsLocal(Error& error) const override {
    return GetQueryResultSpan(&CQueryNodePositions::nodePositions, error);
  }

  void SetZeroDisplacementsAndVelocities(Error& error) override {
    mochi::SetZeroDisplacements(reg, e, error);
    mochi::SetZeroVelocities(reg, e, error);
  }

  Span<real const> GetElementsDeformationGradient(Error& error) const override {
    return GetQueryResultSpan(
        &CQueryElementsDeformationGradient::elementsDeformationGradient, error);
  }

  void SetNodePositionsLocal(Span<real const> positionsLocal, Error& error) override {
    mochi::SetNodePositionsLocal(reg, e, positionsLocal, error);
  }

  void SetNodeVelocitiesLocal(Span<real const> velocitiesLocal, Error& error) override {
    mochi::SetNodeVelocitiesLocal(reg, e, velocitiesLocal, error);
  }

  Span<real const> GetSurfaceMeshNodeNormalsLocal(Error& error) const override {
    return GetQueryResultSpan(&CQuerySurfaceNodeNormals::nodeNormals, error);
  }

  Span<real const> GetVisualMeshNodePositionsLocal(Error& error) const override {
    return GetQueryResultSpan(&CQueryVisualNodePositions::nodePositions, error);
  }

  Span<real const> GetVisualMeshNodeNormalsLocal(Error& error) const override {
    return GetQueryResultSpan(&CQueryVisualNodeNormals::nodeNormals, error);
  }

  Span<int const> GetBoundaryConditionDofIndices() const override {
    auto const* bc = reg.try_get<CDofPositionsBC>(e); // optional component
    return bc ? bc->dofIndices : Span<int const>{};
  }

  Span<real const> GetBoundaryConditionDofValuesWorld() const override {
    auto const* bc = reg.try_get<CDofPositionsBC>(e); // optional component
    return bc ? bc->dofValues : Span<real const>{};
  }

  static void ErrorIfBoundaryConditionSetterUnsupported(
      entt::registry const& reg,
      entt::entity e,
      Error& error) {
    MOCHI_ERROR_IF(
        reg.any_of<TagSoftSkinnedActor>(e),
        error,
        "Boundary condition setters are not supported for nested soft actors.");
  }

  void AddBoundaryConditionDofsWorldImpl(
      Span<int const> dofIndices,
      Span<real const> dofValuesWorld,
      bool isPermanent,
      Error& error) {
    MOCHI_PROFILE_SCOPE();
    ErrorIfBoundaryConditionSetterUnsupported(reg, e, error);
    MOCHI_ERROR_RETURN(error);

    // General error checking
    auto const* dofInfo = MOCHI_TRY_GET(CActorDofInfo, reg, e, error);
    MOCHI_ERROR_IF(
        (reg.any_of<TagRomActor>(e)),
        error,
        "ROM actors do not yet support boundary conditions yet.");
    MOCHI_ERROR_IF(
        !reg.all_of<CDirichletBC<real>>(e),
        error,
        "This type of actor does not support boundary conditions.");
    MOCHI_ERROR_IF(
        reg.try_ctx<TagDifferentiableScene>() != nullptr,
        error,
        "Dirichlet boundary conditions are not supported in differentiable scenes.");
    MOCHI_ERROR_IF(dofIndices.size() != dofValuesWorld.size(), error, "Array size mismatch");
    MOCHI_ERROR_RETURN(error);
    int const numDofs = GetNumDofs();
    for (int i : dofIndices) {
      MOCHI_ERROR_IF(i < 0 || i >= numDofs, error, "Dof index out-of-range");
    }
    MOCHI_ERROR_RETURN(error);

    Span<int const> poseIndices = dofIndices;
    Span<real const> poseValues = dofValuesWorld;
    auto addBCs = [&]() {
      // Add CDofPositionsBC the first time BCs are added to an actor
      auto& bc = reg.get_or_emplace<CDofPositionsBC>(e);
      int const dofBegin = isize(bc.dofIndices);
      int const poseBegin = isize(bc.poseIndices);
      int const colValueBegin = isize(bc.colValueIndices);

      // Append BCs
      bc.poseIndices.insert(bc.poseIndices.end(), poseIndices.begin(), poseIndices.end());
      bc.dofIndices.insert(bc.dofIndices.end(), dofIndices.begin(), dofIndices.end());
      bc.poseValues.insert(bc.poseValues.end(), poseValues.begin(), poseValues.end());
      bc.dofValues.insert(bc.dofValues.end(), dofValuesWorld.begin(), dofValuesWorld.end());

      // Append cache for efficient column zeroing in sparse and block sparse actor matrices.
      if (auto const* fullSparsity = reg.try_get<CFullSparsityPattern const>(e)) {
        AppendColValueIndexCache(fullSparsity->graph, dofIndices, bc.colValueIndices);
      }

      if (isPermanent) {
        bc.permanentRanges.push_back(
            CDofPositionsBC::PermanentRange{
                .dofBegin = dofBegin,
                .dofCount = isize(bc.dofIndices) - dofBegin,
                .poseBegin = poseBegin,
                .poseCount = isize(bc.poseIndices) - poseBegin,
                .colValueBegin = colValueBegin,
                .colValueCount = isize(bc.colValueIndices) - colValueBegin,
            });
      }

      // A duplicate-free set has at most one boundary condition per DoF, so exceeding the DoF count
      // means some DoF has duplicates. This usually happens when boundary conditions are added to
      // the same DoFs without an intervening ClearBoundaryConditions, which increases the per-step
      // cost. It is legal (last value wins) but almost always a mistake.
      if (isize(bc.dofIndices) > numDofs) {
        MOCHI_LOG_WARNING(
            "Boundary condition count exceeds the actor's degree-of-freedom count, indicating at "
            "least one DoF has duplicate boundary conditions. Duplicates increase the per-step "
            "cost. Consider calling ClearBoundaryConditions before re-adding.");
      }
    };

    if (reg.any_of<TagSoftActor, TagShellActor>(e)) {
      // Soft actor
      MOCHI_ERROR_IF(
          (dofIndices.size() % 3) != 0,
          error,
          "For soft and shell actors, you must specify all 3 DOFs for each node.");
      MOCHI_ERROR_RETURN(error);
      for (int i = 0; i < isize(dofIndices); i += 3) {
        MOCHI_ERROR_IF(
            ((dofIndices[i] % 3) != 0) || (dofIndices[i + 1] != dofIndices[i] + 1) ||
                (dofIndices[i + 2] != dofIndices[i] + 2),
            error,
            "For soft and shell actors, you must specify all 3 DOFs for each node, in order.");
      }
      MOCHI_ERROR_RETURN(error);
      addBCs();
      return;
    }

    if (reg.any_of<TagRodActor>(e)) {
      // Validity checking specific to rod actors:
      int const numBcDofs = isize(dofIndices);
      for (int i = 0; i < numBcDofs; i++) {
        int const dofIndex = dofIndices[i];
        int const component = dofIndex % 4;
        bool const isTwist = (component == 3);
        if (!isTwist) {
          // Displacement DoFs must be given in ordered blocks of 3, to be able to transform the
          // world-space positions here to local displacements.
          bool invalidDisplacementDofs = (component != 0) || (i + 3 > numBcDofs) ||
              (dofIndices[i + 1] != dofIndex + 1) || (dofIndices[i + 2] != dofIndex + 2);
          MOCHI_ERROR_IF(
              invalidDisplacementDofs,
              error,
              "Displacement boundary conditions on rod actor nodes must specify all "
              "displacement components of constrained node in order");
          MOCHI_ERROR_RETURN(error);
          i += 2;
        } else {
          // Scalar torsional DoFs are incremental, and do not have a notion of transformation
          // independent of position DoFs. They are only allowed to be fixed to zero by this type
          // of hard DoF-level boundary condition. Imposing nonzero rotations in world space
          // requires a more elaborate constraint involving both twist and position DoFs.
          MOCHI_ERROR_IF(
              dofValuesWorld[i] != 0_r,
              error,
              "Twist degrees of freedom on rod actors may only be constrained to zero by boundary "
              "conditions");
          MOCHI_ERROR_RETURN(error);
        }
      }
      addBCs();
      return;
    }

    // Rigid and articulated actors require sorted indices
    for (int i = 1; i < isize(dofIndices); ++i) {
      MOCHI_ERROR_IF(dofIndices[i] <= dofIndices[i - 1], error, "Indices must be sorted");
    }
    MOCHI_ERROR_RETURN(error);

    // If necessary, convert dof indices and values to pose
    std::vector<int> poseIndContainer;
    std::vector<real> poseValContainer;
    if (dofInfo->poseSize != dofInfo->dofsSize) {
      // Create temp vectors for all dofs and indices
      std::vector<real> dofs(dofInfo->dofsSize, 0_r);
      DynamicArray<bool> indsDofs(dofInfo->dofsSize, false);
      for (int i = 0; i < dofIndices.size(); ++i) {
        dofs[dofIndices[i]] = dofValuesWorld[i];
        indsDofs[dofIndices[i]] = true;
      }

      // Convert to pose and indices
      std::vector<real> pose(dofInfo->poseSize);
      DynamicArray<bool> indsPose(dofInfo->poseSize, false);
      if (reg.all_of<TagRigidActor>(e)) {
        ConvertRigidDofsToPose(dofs, pose);
        ConvertRigidDofFlagsToPoseFlags(indsDofs, indsPose);
      } else {
        MOCHI_ASSERT_VERBOSE(reg.all_of<TagArticulatedActor>(e), "Unexpected actor type");
        auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
        auto const& poseInfo = reg.get<CArticulatedJointPoseInfo const>(e);
        articulated::ConvertDofsToPose(
            joints->jointTypes, joints->dofInfo, poseInfo, AsConstView(dofs), AsView(pose));
        articulated::ConvertDofFlagsToPoseFlags(
            joints->jointTypes, joints->dofInfo, poseInfo, indsDofs, indsPose);
      }

      // Extract the required pose values and indices
      poseIndContainer.reserve(dofInfo->poseSize);
      poseValContainer.reserve(dofInfo->poseSize);
      for (int i = 0; i < indsPose.size(); ++i) {
        if (indsPose[i]) {
          poseIndContainer.push_back(i);
          poseValContainer.push_back(pose[i]);
        }
      }
      poseIndices = poseIndContainer;
      poseValues = poseValContainer;
    }

    auto hasSomeRotation = [&](Span<int const> rotInds) {
      return std::find_first_of(
                 dofIndices.begin(), dofIndices.end(), rotInds.begin(), rotInds.end()) !=
          dofIndices.end();
    };

    auto hasAllRotations = [&](Span<int const> rotInds) {
      return std::includes(dofIndices.begin(), dofIndices.end(), rotInds.begin(), rotInds.end());
    };

    auto hasPartialRotationBCs = [&](Span<int const> rotInds) {
      return hasSomeRotation(rotInds) != hasAllRotations(rotInds);
    };

    if (reg.all_of<TagRigidActor>(e)) {
      // Rigid actor
      std::array<int, RigidSize::kDRot> constexpr kRotInds = {
          RigidSize::kDTrans, RigidSize::kDTrans + 1, RigidSize::kDTrans + 2};
      MOCHI_ERROR_IF(hasPartialRotationBCs(kRotInds), error, "Rotation BCs must be all or none.");
      MOCHI_ERROR_RETURN(error);
      addBCs();
    } else {
      // Articulated actor
      MOCHI_ASSERT_VERBOSE(reg.all_of<TagArticulatedActor>(e), "Unexpected actor type");
      auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
      for (auto const& dof : joints->dofInfo) {
        if (dof.rotSize <= 1) {
          continue;
        }
        std::vector<int> rotInds(dof.rotSize);
        std::iota(rotInds.begin(), rotInds.end(), dof.GetRotOffset());
        MOCHI_ERROR_IF(hasPartialRotationBCs(rotInds), error, "Rotation BCs must be all or none.");
      }
      MOCHI_ERROR_RETURN(error);
      addBCs();
    }
  }

  void AddBoundaryConditionDofsWorld(
      Span<int const> dofIndices,
      Span<real const> dofValuesWorld,
      Error& error) override {
    constexpr bool kIsPermanent = false;
    AddBoundaryConditionDofsWorldImpl(dofIndices, dofValuesWorld, kIsPermanent, error);
  }

  void AddBoundaryConditionDofsWorldPermanent(
      Span<int const> dofIndices,
      Span<real const> dofValuesWorld,
      Error& error) override {
    constexpr bool kIsPermanent = true;
    AddBoundaryConditionDofsWorldImpl(dofIndices, dofValuesWorld, kIsPermanent, error);
  }

  // Length of nodePositionsWorld = kSpaceDim * nodeIndices.size(). This is equivalent to calling
  // AddBoundaryConditionDofsWorld with 3 times as many indices of the form
  // ((3 * i + 0), (3 * i + 1), (3 * i + 2)) for each node. Optionally, make the BCs permanent
  // (i.e., they cannot be cleared once created).
  void AddBoundaryConditionNodesWorldImpl(
      Span<int const> nodeIndices,
      Span<real const> nodePositionsWorld,
      bool isPermanent,
      Error& error) {
    ErrorIfBoundaryConditionSetterUnsupported(reg, e, error);
    MOCHI_ERROR_RETURN(error);

    MOCHI_ERROR_IF(
        !(reg.any_of<TagSoftActor, TagShellActor>(e)),
        error,
        "This function is only supported by soft and shell actors. For other types of actors, use AddBoundaryConditionDofsWorld.");
    MOCHI_ERROR_RETURN(error);

    // NOTE: To avoid the allocation of a temporary std::vector, we could write to CDofPositionsBC
    // directly, but for now it is nice to keep everything going through one code path.
    std::vector<int> dofIndices;
    dofIndices.reserve(nodeIndices.size());
    for (int i = 0; i < isize(nodeIndices); ++i) {
      int iDof = 3 * nodeIndices[i];
      dofIndices.push_back(iDof + 0);
      dofIndices.push_back(iDof + 1);
      dofIndices.push_back(iDof + 2);
    }
    return AddBoundaryConditionDofsWorldImpl(dofIndices, nodePositionsWorld, isPermanent, error);
  }

  void AddBoundaryConditionNodesWorld(
      Span<int const> nodeIndices,
      Span<real const> nodePositionsWorld,
      Error& error) override {
    constexpr bool kIsPermanent = false;
    AddBoundaryConditionNodesWorldImpl(nodeIndices, nodePositionsWorld, kIsPermanent, error);
  }

  void AddBoundaryConditionNodesWorldPermanent(
      Span<int const> nodeIndices,
      Span<real const> nodePositionsWorld,
      Error& error) override {
    constexpr bool kIsPermanent = true;
    AddBoundaryConditionNodesWorldImpl(nodeIndices, nodePositionsWorld, kIsPermanent, error);
  }

  void AddBoundaryConditionConstrainedNodesAtRestImpl(bool isPermanent, Error& error) {
    ErrorIfBoundaryConditionSetterUnsupported(reg, e, error);
    MOCHI_ERROR_RETURN(error);

    MOCHI_ERROR_IF(
        !(reg.any_of<TagSoftActor, TagShellActor>(e)),
        error,
        "This function is only supported by soft and shell actors.");
    auto const* shapeComp = MOCHI_TRY_GET(CShape, reg, e, error);
    MOCHI_ERROR_RETURN(error);

    Span<int const> constrainedNodes;
    Span<real const> referencePositions;
    if (auto const* tetShape = dynamic_cast<TetrahedralMeshShape const*>(shapeComp->shape.get())) {
      referencePositions = Flatten(tetShape->GetMesh()->GetNodeCoordinates());
      if (auto const& data = tetShape->GetMeshConstrainedNodes()) {
        constrainedNodes = data->constrainedNodes;
      }
    } else if (
        auto const* triShape = dynamic_cast<TriangularMeshShape const*>(shapeComp->shape.get())) {
      referencePositions = Flatten(triShape->GetMesh()->GetNodeCoordinates());
      if (auto const& data = triShape->GetMeshConstrainedNodes()) {
        constrainedNodes = data->constrainedNodes;
      }
    } else {
      MOCHI_ERROR_SET(error, "This function requires a tetrahedral or triangular mesh.");
    }

    if (!constrainedNodes.empty()) {
      VMatrix4x4r const worldFromLocalT =
          ToVMatrix4x4Transpose(reg.get<CRootTransform const>(e).worldFromLocal);
      auto const referencePositions3 = Unflatten<Real3 const>(referencePositions);
      DynamicArray<Real3> constrainedPositionsWorld;
      constrainedPositionsWorld.reserve(constrainedNodes.size());
      for (int idx : constrainedNodes) {
        Vec4r const referencePositionLocal = ToSimd(referencePositions3[idx], 1_r);
        constrainedPositionsWorld.emplace_back(
            ToReal3(DotVecMat4x4(referencePositionLocal, worldFromLocalT)));
      }

      AddBoundaryConditionNodesWorldImpl(
          constrainedNodes, Flatten(MakeSpan(constrainedPositionsWorld)), isPermanent, error);
    }
  }

  void AddBoundaryConditionConstrainedNodesAtRest(Error& error) override {
    constexpr bool kIsPermanent = false;
    AddBoundaryConditionConstrainedNodesAtRestImpl(kIsPermanent, error);
  }

  void AddBoundaryConditionConstrainedNodesAtRestPermanent(Error& error) override {
    constexpr bool kIsPermanent = true;
    AddBoundaryConditionConstrainedNodesAtRestImpl(kIsPermanent, error);
  }

  void ClearBoundaryConditions() override {
    if (auto* bc = reg.try_get<CDofPositionsBC>(e)) { // Optional component
      bc->Clear();
    }
  }

  void SetExternalForcesOnDofs(
      Span<int const> dofIndices,
      Span<real const> forceValues,
      Error& error) override {
    MOCHI_ERROR_RETURN(error);

    MOCHI_ERROR_IF(dofIndices.size() != forceValues.size(), error, "Sizes do not match.");
    MOCHI_ERROR_IF_NOT(IsFinite(forceValues), error, "External forces must be finite.");
    auto* externalForces = reg.try_get<CExternalForces>(e);
    MOCHI_ERROR_IF_NOT(externalForces, error, "This actor does not support external forces.");
    MOCHI_ERROR_RETURN(error);

    if (dofIndices.empty()) {
      externalForces->Clear();
      return;
    }

    int const numDofs = reg.get<CActorDofInfo const>(e).dofsSize;
    MOCHI_ERROR_IF(
        Min(dofIndices) < 0 || Max(dofIndices) >= numDofs,
        error,
        "External force DoFs out of range.");
    MOCHI_ERROR_RETURN(error);
    MOCHI_ERROR_IF_NOT(IsUnique(dofIndices), error, "External force DoF indices must be unique.");
    MOCHI_ERROR_RETURN(error);
    if (reg.any_of<TagRodActor>(e) && !reg.get<CPolylineMesh>(e).isClosedLoop) {
      int const paddingDofIndex = numDofs - 1;
      for (int i = 0; i < isize(dofIndices); ++i) {
        MOCHI_ERROR_IF(
            dofIndices[i] == paddingDofIndex && forceValues[i] != 0_r,
            error,
            "Attempting to apply external torque to non-physical padding DoF.");
      }
      MOCHI_ERROR_RETURN(error);
    }

    // Assign the dofs and forces for this actor, regardless if some other dofs were receiving
    // forces.
    externalForces->dofs.assign(dofIndices.begin(), dofIndices.end());
    externalForces->forces.assign(forceValues.begin(), forceValues.end());
  }

  void ClearExternalForces() override {
    auto* externalForces = reg.try_get<CExternalForces>(e); // optional component
    if (externalForces) {
      externalForces->Clear();
    }
  }

  void GetExternalForces(Span<real> outForces, Error& error) override {
    int const numDofs = GetNumDofs();
    MOCHI_ERROR_IF(isize(outForces) != numDofs, error, "outForces size must equal number of DOFs.");
    MOCHI_ERROR_RETURN(error);
    std::fill(outForces.begin(), outForces.end(), real{});
    auto const* ef = reg.try_get<CExternalForces const>(e);
    if (ef && !ef->Empty()) {
      for (int i = 0; i < isize(ef->dofs); ++i) {
        outForces[ef->dofs[i]] = ef->forces[i];
      }
    }
  }

  Span<ContactPoint const> GetContactPointsWorld(Error& error) const override {
    auto* query = reg.try_get<CQueryContactPoints>(e);
    MOCHI_ERROR_IF(
        !query || !query->isInitialized,
        error,
        "If you registered the query, then the results should be available after the next simulation step.");
    MOCHI_ERROR_RETURN(error, {});
    return query->contactPoints;
  }

  SdfDistances GetSdfDistances(Error& error) const override {
    auto* query = reg.try_get<CQuerySdfDistances>(e);
    MOCHI_ERROR_IF(
        !query,
        error,
        "If you registered the query, then the results should be available after the next simulation step.");
    MOCHI_ERROR_RETURN(error, {});
    return SdfDistances{
        .sampleIndices = query->sampleIndices,
        .worldPositions = query->worldPositions,
        .distances = query->distances,
        .distanceGrads = query->distanceGrads,
        .maxSdfFarDistanceEvaluation = query->maxSdfFarDistanceEvaluation,
    };
  }

  Span<NodeContactForce const> GetNodeContactForcesWorld(Error& error) const override {
    auto* query = reg.try_get<CQueryNodeContactForces>(e);
    MOCHI_ERROR_IF(
        !query || !query->isInitialized,
        error,
        "If you registered the query, then the results should be available after the next simulation step.");
    MOCHI_ERROR_RETURN(error, {});
    return query->nodeContactForces;
  }

#define MOCHI_ERROR_RETURN_IF_NO_CONTACT_QUERY(...)                                                         \
  MOCHI_ERROR_RETURN(error, __VA_ARGS__);                                                                   \
  auto* query = reg.try_get<CQueryActorContactForces>(e);                                                   \
  MOCHI_ERROR_IF(                                                                                           \
      !query || !query->isInitialized,                                                                      \
      error,                                                                                                \
      "If you registered the query, then the results should be available after the next simulation step."); \
  MOCHI_ERROR_RETURN(error, __VA_ARGS__);

  Real3 GetContactForceWorld(Error& error) const override {
    MOCHI_ERROR_RETURN_IF_NO_CONTACT_QUERY({});

    return ToReal3(query->GetTotalForce());
  }

  Real3 GetContactTorqueWorld(Error& error) const override {
    MOCHI_ERROR_IF_NOT(
        reg.all_of<TagRigidActor>(e),
        error,
        "GetContactTorqueWorld only supported for rigid actors.");
    MOCHI_ERROR_RETURN_IF_NO_CONTACT_QUERY({});

    return ToReal3(query->GetTotalTorque());
  }

  Real3 GetContactForceFromActorWorld(Actor const* other, Error& error) const override {
    MOCHI_ERROR_RETURN_IF_NO_CONTACT_QUERY({});

    // Check that the other actor is valid
    MOCHI_ERROR_IF(!other, error, "Invalid actor");
    MOCHI_ERROR_RETURN(error, {});
    auto otherEntity = GetEntity(reg, other->GetHandle(), error);
    MOCHI_ERROR_RETURN(error, {});

    // Add the force
    return ToReal3(query->GetForce(otherEntity));
  }

#undef MOCHI_ERROR_RETURN_IF_NO_CONTACT_QUERY

  QueryHandle RegisterQuery(QueryType type, Error& error) override {
    constexpr bool kComputeImmediately = false;
    return mochi::RegisterQuery(reg, e, type, kComputeImmediately, error);
  }

  QueryHandle RegisterQueryAndCompute(QueryType type, Error& error) override {
    constexpr bool kComputeImmediately = true;
    return mochi::RegisterQuery(reg, e, type, kComputeImmediately, error);
  }

  void CancelQuery(QueryHandle handle) override {
    mochi::CancelQuery(reg, e, handle);
  }

  bool IsQuerySupported(QueryType type) const override {
    Error error;
    SetErrorIfQueryNotSupported(reg, e, type, error);
    return error.IsOK();
  }

  void SetUserData(void* userData) override {
    reg.emplace_or_replace<CUserData>(e, userData);
  }

  void* GetUserData() const override {
    auto const* cdata = reg.try_get<CUserData>(e);
    return cdata ? cdata->userData : nullptr;
  }

  bool IsNestedLinkActor() const override {
    return reg.all_of<TagArticulatedLinkActor>(e);
  }

  ActorHandle GetArticulatedActor(Error& error) const override {
    MOCHI_ERROR_RETURN(error, {});
    if (auto const* articulated = reg.try_get<CArticulatedEntity const>(e)) {
      return GetActorHandle(articulated->entity, scene->GetHandle());
    }
    if (auto const* composition = reg.try_get<CSkinnedComposition const>(e)) {
      return composition->articulatedHandle;
    }
    MOCHI_ERROR_SET(error, "Actor is not an articulated link or soft skinned actor");
    return {};
  }

#define MOCHI_ERROR_RETURN_IF_NOT_LINK(...)                                   \
  MOCHI_ERROR_RETURN(error, __VA_ARGS__);                                     \
  auto const* articulated = MOCHI_TRY_GET(CArticulatedEntity, reg, e, error); \
  MOCHI_ERROR_IF_NOT(articulated, error, "Actor is not an articulated link"); \
  MOCHI_ERROR_RETURN(error, __VA_ARGS__);

  bool IsNestedSoftActor() const override {
    // Check if this soft actor is nested within a soft skinned articulation
    return reg.all_of<TagSoftSkinnedActor>(e);
  }

  Span<real const> GetArticulatedJacobian(Error& error) const override {
    MOCHI_ERROR_RETURN_IF_NOT_LINK({});
    auto const& artJacobian = reg.get<CArticulatedJacobian const>(articulated->entity).value;
    auto const& dofOffset = reg.get<CDofOffset const>(e).dofsOffset;
    auto const& numDofs = reg.get<CActorDofInfo const>(e).dofsSize;
    return artJacobian.MiddleRows(dofOffset, numDofs);
  }
#undef MOCHI_ERROR_RETURN_IF_NOT_LINK
};

} // namespace

// Experimental API

void experimental::SetArticulatedForceAndTargetPose(
    Actor* actor,
    Span<real const> target,
    Span<experimental::ControlType const> controlTypes,
    Span<int const> dofOrLinkIndices,
    Error& error) {
  MOCHI_ERROR_IF(!actor, error, "Invalid actor.");
  MOCHI_ERROR_RETURN(error);
  auto* actorImpl = dynamic_cast<ActorInterfaceImpl*>(actor);
  MOCHI_ERROR_IF(!actorImpl, error, "Invalid actor implementation.");
  MOCHI_ERROR_RETURN(error);
  actorImpl->SetArticulatedForceAndTargetPose(target, controlTypes, dofOrLinkIndices, error);
}

int experimental::AddLinearTransmission(
    Actor* actor,
    LinearTransmissionParams const& params,
    Error& error) {
  MOCHI_ERROR_IF(!actor, error, "Invalid actor");
  MOCHI_ERROR_RETURN(error, -1);
  auto* actorImpl = dynamic_cast<ActorInterfaceImpl*>(actor);
  MOCHI_ERROR_IF(!actorImpl, error, "Invalid actor implementation");
  MOCHI_ERROR_RETURN(error, -1);
  return transmission::AddLinearTransmission(actorImpl->reg, actorImpl->e, params, error);
}

int experimental::AddSpatialTendon(Actor* actor, SpatialTendonParams const& params, Error& error) {
  MOCHI_ERROR_IF(!actor, error, "Invalid actor");
  MOCHI_ERROR_RETURN(error, -1);
  auto* actorImpl = dynamic_cast<ActorInterfaceImpl*>(actor);
  MOCHI_ERROR_IF(!actorImpl, error, "Invalid actor implementation");
  MOCHI_ERROR_RETURN(error, -1);
  return transmission::AddSpatialTendon(actorImpl->reg, actorImpl->e, params, error);
}

void experimental::AttachDisplacementControlActuator(
    Actor* actor,
    int transmissionIndex,
    experimental::DisplacementControlActuatorParams const& params,
    Error& error) {
  MOCHI_ERROR_IF(!actor, error, "Invalid actor");
  MOCHI_ERROR_RETURN(error);
  auto* actorImpl = dynamic_cast<ActorInterfaceImpl*>(actor);
  MOCHI_ERROR_IF(!actorImpl, error, "Invalid actor implementation");
  MOCHI_ERROR_RETURN(error);
  transmission::AttachDisplacementControlActuator(
      actorImpl->reg, actorImpl->e, transmissionIndex, params, error);
}

void experimental::AttachForceControlActuator(
    Actor* actor,
    int transmissionIndex,
    experimental::ForceControlActuatorParams const& params,
    Error& error) {
  MOCHI_ERROR_IF(!actor, error, "Invalid actor");
  MOCHI_ERROR_RETURN(error);
  auto* actorImpl = dynamic_cast<ActorInterfaceImpl*>(actor);
  MOCHI_ERROR_IF(!actorImpl, error, "Invalid actor implementation");
  MOCHI_ERROR_RETURN(error);
  transmission::AttachForceControlActuator(
      actorImpl->reg, actorImpl->e, transmissionIndex, params, error);
}

void experimental::AttachMcKibbenActuator(
    Actor* actor,
    int transmissionIndex,
    experimental::McKibbenActuatorParams const& params,
    Error& error) {
  MOCHI_ERROR_IF(!actor, error, "Invalid actor");
  MOCHI_ERROR_RETURN(error);
  auto* actorImpl = dynamic_cast<ActorInterfaceImpl*>(actor);
  MOCHI_ERROR_IF(!actorImpl, error, "Invalid actor implementation");
  MOCHI_ERROR_RETURN(error);
  transmission::AttachMcKibbenActuator(
      actorImpl->reg, actorImpl->e, transmissionIndex, params, error);
}

void experimental::SetTransmissionActuatorStateVariables(
    Actor* actor,
    int transmissionIndex,
    Span<real const> stateVariables,
    Error& error) {
  MOCHI_ERROR_IF(!actor, error, "Invalid actor");
  MOCHI_ERROR_RETURN(error);
  auto* actorImpl = dynamic_cast<ActorInterfaceImpl*>(actor);
  MOCHI_ERROR_IF(!actorImpl, error, "Invalid actor implementation");
  MOCHI_ERROR_RETURN(error);
  transmission::SetTransmissionActuatorStateVariables(
      actorImpl->reg, actorImpl->e, transmissionIndex, stateVariables, error);
}

real experimental::GetTransmissionDisplacement(
    Actor const* actor,
    int transmissionIndex,
    Error& error) {
  MOCHI_ERROR_IF(!actor, error, "Invalid actor");
  MOCHI_ERROR_RETURN(error, 0_r);
  auto const* actorImpl = dynamic_cast<ActorInterfaceImpl const*>(actor);
  MOCHI_ERROR_IF(!actorImpl, error, "Invalid actor implementation");
  MOCHI_ERROR_RETURN(error, 0_r);
  return transmission::GetTransmissionDisplacement(
      actorImpl->reg, actorImpl->e, transmissionIndex, error);
}

void experimental::GetTransmissionDisplacementJacobian(
    Actor const* actor,
    int transmissionIndex,
    Span<real> outJacobian,
    Error& error) {
  MOCHI_ERROR_IF(!actor, error, "Invalid actor");
  MOCHI_ERROR_RETURN(error);
  auto const* actorImpl = dynamic_cast<ActorInterfaceImpl const*>(actor);
  MOCHI_ERROR_IF(!actorImpl, error, "Invalid actor implementation");
  MOCHI_ERROR_RETURN(error);
  transmission::GetTransmissionDisplacementJacobian(
      actorImpl->reg, actorImpl->e, transmissionIndex, outJacobian, error);
}

void experimental::GetTransmissionActuatorStateVariables(
    Actor const* actor,
    int transmissionIndex,
    Span<real> outStateVariables,
    Error& error) {
  MOCHI_ERROR_IF(!actor, error, "Invalid actor");
  MOCHI_ERROR_RETURN(error);
  auto const* actorImpl = dynamic_cast<ActorInterfaceImpl const*>(actor);
  MOCHI_ERROR_IF(!actorImpl, error, "Invalid actor implementation");
  MOCHI_ERROR_RETURN(error);
  transmission::GetTransmissionActuatorStateVariables(
      actorImpl->reg, actorImpl->e, transmissionIndex, outStateVariables, error);
}

int experimental::GetNumTransmissionActuatorStateVariables(
    Actor const* actor,
    int transmissionIndex,
    Error& error) {
  MOCHI_ERROR_IF(!actor, error, "Invalid actor");
  MOCHI_ERROR_RETURN(error, 0);
  auto const* actorImpl = dynamic_cast<ActorInterfaceImpl const*>(actor);
  MOCHI_ERROR_IF(!actorImpl, error, "Invalid actor implementation");
  MOCHI_ERROR_RETURN(error, 0);
  return transmission::GetNumTransmissionActuatorStateVariables(
      actorImpl->reg, actorImpl->e, transmissionIndex, error);
}
void experimental::ConstrainNodesByPosition(
    Actor* actor,
    std::function<bool(int, Real3 const&)> callback,
    Error& error) {
  MOCHI_ERROR_IF(!actor, error, "Invalid actor");
  MOCHI_ERROR_RETURN(error);
  std::vector<int> nodesToFix;
  std::vector<real> fixedDofValues;
  ShapeHandle shape = actor->GetReferenceShape(error);
  MOCHI_ERROR_RETURN(error);
  auto const& mesh = actor->GetContext()->GetShapeMesh(shape, error);
  MOCHI_ERROR_IF(
      mesh.IsEmpty(), error, "Cannot constrain nodes: actor reference shape has no mesh.");
  MOCHI_ERROR_RETURN(error);
  Span<real const> referencePositions = mesh.coordinates;
  int const numNodes = mesh.GetNumNodes();
  nodesToFix.reserve(numNodes);
  fixedDofValues.reserve(3 * numNodes);
  // Transform reference positions from mesh description to world space.
  VMatrix4x4r const worldFromLocalT = ToVMatrix4x4Transpose(actor->GetRootTransform());
  for (int nodeIndex = 0; nodeIndex < numNodes; ++nodeIndex) {
    int const dofOffset = 3 * nodeIndex;
    Vec4r const localPos = ToSimdPoint(ToSimd(
        Real3{
            referencePositions[dofOffset],
            referencePositions[dofOffset + 1],
            referencePositions[dofOffset + 2]}));
    // Assumes affine transform where we can discard 4th component.
    Real3 const worldPos = ToReal3(DotVecMat4x4(localPos, worldFromLocalT));
    // Constrain the current node to its world-space position if the callback returns true.
    if (callback(nodeIndex, worldPos)) {
      nodesToFix.push_back(nodeIndex);
      for (int i = 0; i < 3; i++) {
        fixedDofValues.push_back(worldPos[i]);
      }
    }
  }
  actor->AddBoundaryConditionNodesWorld(nodesToFix, fixedDofValues, error);
}

void experimental::EnableNewtonEulerInertia(Actor* actor, bool enable, Error& error) {
  MOCHI_ERROR_IF(!actor, error, "Invalid actor");
  MOCHI_ERROR_RETURN(error);
  auto const* actorImpl = dynamic_cast<ActorInterfaceImpl const*>(actor);
  MOCHI_ERROR_IF(!actorImpl, error, "Invalid actor implementation");
  MOCHI_ERROR_RETURN(error);
  auto& reg = actorImpl->reg;
  auto e = actorImpl->e;

  MOCHI_ERROR_IF_NOT(
      (reg.any_of<TagRigidActor, TagArticulatedActor>(e)),
      error,
      "EnableNewtonEulerInertia is only supported for rigid and articulated actors.");
  MOCHI_ERROR_RETURN(error);

  // Newton-Euler inertia should not be used with differentiability. But perhaps the user is not
  // setting it up to compute derivatives, so just log a warning.
  if (enable && reg.any_of<CActorDerivedStateInfo>(e)) {
    MOCHI_LOG_WARNING_ONCE("\nNewton-Euler inertia is not supported with differentiability.");
  }

  auto enableActor = [&](entt::entity actor) {
    if (enable) {
      reg.emplace_or_replace<TagUseNewtonEulerInertia>(actor);
    } else {
      reg.remove<TagUseNewtonEulerInertia>(actor);
    }
  };

  // Enable/disable on this actor
  enableActor(e);

  // If the actor is articulated, enable/disable on all of its links
  auto const* links = reg.try_get<CBoneHandles const>(e);
  if (!links) {
    return;
  }
  for (auto const& link : links->bones) {
    enableActor(GetEntity(reg, link, ErrorAssert{}));
  }
}

bool experimental::IsNewtonEulerInertiaEnabled(Actor const* actor, Error& error) {
  MOCHI_ERROR_IF(!actor, error, "Invalid actor");
  MOCHI_ERROR_RETURN(error, {});

  auto const* actorImpl = assert_cast<ActorInterfaceImpl const*>(actor);
  auto const& reg = actorImpl->reg;
  auto e = actorImpl->e;

  MOCHI_ERROR_IF_NOT(
      (reg.any_of<TagRigidActor, TagArticulatedActor>(e)),
      error,
      "IsNewtonEulerInertiaEnabled is only supported for rigid and articulated actors.");
  MOCHI_ERROR_RETURN(error, {});

  return reg.all_of<TagUseNewtonEulerInertia>(e);
}

#define MOCHI_ERROR_RETURN_IF_BACKWARD_NOT_SUPPORTED(CONST_QUALIFIER)                         \
  MOCHI_ERROR_IF(!actor, error, "Invalid actor");                                             \
  MOCHI_ERROR_RETURN(error);                                                                  \
  auto CONST_QUALIFIER* actorImpl = dynamic_cast<ActorInterfaceImpl CONST_QUALIFIER*>(actor); \
  MOCHI_ERROR_IF(!actorImpl, error, "Invalid actor");                                         \
  MOCHI_ERROR_RETURN(error);                                                                  \
  auto CONST_QUALIFIER& reg = actorImpl->reg;                                                 \
  MOCHI_ERROR_IF(                                                                             \
      !reg.try_ctx<TagDifferentiableScene>(), error, "The scene is not differentiable");      \
  MOCHI_ERROR_RETURN(error);                                                                  \
  [[maybe_unused]] auto CONST_QUALIFIER e = actorImpl->e;

static void GetRigidTransformBackward(
    Actor* actor,
    Span<real const> gradOutput,
    bool isRootTransform,
    Error& error) {
  MOCHI_ERROR_RETURN_IF_BACKWARD_NOT_SUPPORTED();
  MOCHI_ERROR_IF(
      actor->GetType() != ActorType::Rigid || !reg.all_of<CRigidBodyInertia>(actorImpl->e),
      error,
      "Only rigid actors (including links) are supported.");
  MOCHI_ERROR_IF_NOT(isize(gradOutput) == RigidSize::kAll, error, "gradOutput size must be 7.");
  MOCHI_ERROR_IF_NOT(IsFinite(gradOutput), error, "gradOutput must be finite.");
  MOCHI_ERROR_RETURN(error);

  // Convert gradient from (translation, quaternion) to (translation, Lie rotation).
  ColumnVector<real, RigidSize::kDAll> gradLie;
  auto const transform =
      isRootTransform ? actor->GetRootTransform() : actor->GetCenterOfMassTransform(ErrorAssert{});
  diffsim::ConvertRigidGradientQuaternionToLie(transform, gradOutput, gradLie, ErrorAssert{});

  if (isRootTransform) {
    // The rigid state stores the center-of-mass transform, while GetRootTransform returns
    // x_root = x_com - R * com_local. Add the indirect rotation gradient from the root translation.
    auto const& rbInertia = reg.get<CRigidBodyInertia const>(actorImpl->e);
    Vec4r const comOffsetWorld = transform.GetRotation() * rbInertia.GetCenterOfMassLocal();
    auto const gradTranslation = Load<Vec4r>(gradOutput.data()); // The 4th entry is not used.
    Vec4r const gradRotationFromTranslation = Cross3(gradTranslation, comOffsetWorld);
    gradLie.BottomRows<RigidSize::kDRot>(RigidSize::kDRot) +=
        AsColumnVectorView<RigidSize::kDRot>(gradRotationFromTranslation);
  }

  if (actor->IsNestedLinkActor()) {
    // For link actors, project through the FK Jacobian to the parent articulated actor's DOFs.
    auto jacSpan = actor->GetArticulatedJacobian(ErrorAssert{});
    int const parentDofs = isize(jacSpan) / RigidSize::kDAll;
    RowMatrixView<real const, RigidSize::kDAll> jac(jacSpan.data(), RigidSize::kDAll, parentDofs);

    auto const& artEntity = reg.get<CArticulatedEntity>(actorImpl->e);
    auto& stateGrad = reg.get<CDiffStateGrad>(artEntity.entity);
    stateGrad.value += jac.Transpose() * gradLie;
  } else {
    // For standalone rigid actors, accumulate directly.
    auto& stateGrad = reg.get<CDiffStateGrad>(actorImpl->e);
    stateGrad.value += gradLie;
  }
}

void diffsim::GetCenterOfMassTransformBackward(
    Actor* actor,
    Span<real const> gradOutput,
    Error& error) {
  GetRigidTransformBackward(actor, gradOutput, /*isRootTransform*/ false, error);
}

void diffsim::GetRootTransformBackward(Actor* actor, Span<real const> gradOutput, Error& error) {
  GetRigidTransformBackward(actor, gradOutput, /*isRootTransform*/ true, error);
}

static void AccumulateContactForceWorldAdjoints(
    entt::registry& reg,
    entt::entity colliding,
    ContactDetectionResult& collisionResult,
    Vec4r gradForce) {
  int const numContacts = isize(collisionResult.posColliding);
  if (numContacts == 0) {
    // Broad-phase pairs are retained without contacts (e.g. for warm-starting). Such entries have
    // an empty `jacColliderFromWorld`, so there is nothing to accumulate and `[0]` would be
    // invalid.
    return;
  }
  MOCHI_ASSERT_VERBOSE(
      numContacts == isize(collisionResult.forcePerUnitArea),
      "Expected prepared force adjoints for every contact point.");
  MOCHI_ASSERT_VERBOSE(
      collisionResult.jacColliderFromWorld.size() == 1,
      "Deformable colliders not supported in differentiability");

  // Transform to collider space
  gradForce = DotMatVec3x3(collisionResult.jacColliderFromWorld[0], gradForce);

  // Accumulate into each contact's adjoint
  auto const& collidingSamples = reg.get<CContactSamples<TimeStep::Current> const>(colliding);
  for (int i = 0; i < numContacts; ++i) {
    // Multiply by the contact weight of the colliding actor's sample
    int const sample = collisionResult.sampleIndices[i];
    collisionResult.forcePerUnitArea[i] += ToReal3(collidingSamples.weights[sample] * gradForce);
  }
}

static void GetContactForceWorldBackwardImpl(
    Actor* actor,
    Actor const* exclusiveCollider,
    Span<real const> gradOutput,
    Error& error) {
  MOCHI_ERROR_RETURN_IF_BACKWARD_NOT_SUPPORTED();
  MOCHI_ERROR_IF(
      actor->GetType() != ActorType::Rigid || !reg.all_of<CRigidBodyInertia>(actorImpl->e),
      error,
      "Only rigid actors (including links) are supported.");
  MOCHI_ERROR_IF_NOT(isize(gradOutput) == RigidSize::kDTrans, error, "gradOutput size must be 3.");
  MOCHI_ERROR_IF_NOT(IsFinite(gradOutput), error, "gradOutput must be finite.");
  MOCHI_ERROR_IF_NOT(reg.all_of<CQueryActorContactForces>(e), error, "No contact query.");
  MOCHI_ERROR_RETURN(error);

  auto const exclusiveEntity =
      exclusiveCollider ? GetEntity(reg, exclusiveCollider->GetHandle(), error) : entt::null;
  MOCHI_ERROR_RETURN(error);

  Vec4r const gradForceWorld = Load<RigidSize::kDTrans, Vec4r>(gradOutput.data());
  if (auto* collisionsAsync =
          reg.try_get<CActiveCollisions<ContactType::Async, TimeStep::Current>>(e)) {
    for (auto& collision : *collisionsAsync) {
      if (exclusiveEntity == entt::null || collision.colliderEntity == exclusiveEntity) {
        AccumulateContactForceWorldAdjoints(reg, e, collision.collisionResult, gradForceWorld);
      }
    }
  }
  if (auto* collisionsSync =
          reg.try_get<CActiveCollisions<ContactType::Sync, TimeStep::Current>>(e)) {
    for (auto& collision : *collisionsSync) {
      if (exclusiveEntity == entt::null || collision.colliderEntity == exclusiveEntity) {
        AccumulateContactForceWorldAdjoints(reg, e, collision.collisionResult, gradForceWorld);
      }
    }
  }
  if (auto* colliderJacs = reg.try_get<CCollJacs<CollRole::Collider>>(e)) {
    for (auto& jac : *colliderJacs) {
      if (exclusiveEntity == entt::null || jac.otherEntity == exclusiveEntity) {
        // Entities that use "far sdf" queries are not compatible with contact point reporting.
        // If the other actor uses that feature, then do not attempt to store adjoints
        // because we won't have the matching forcePerUnitArea data.
        if (reg.any_of<CRequiresFarSdfEvaluation>(jac.otherEntity)) {
          continue;
        }
        AccumulateContactForceWorldAdjoints(reg, jac.otherEntity, *jac.query, -gradForceWorld);
      }
    }
  }
}

void diffsim::GetContactForceWorldBackward(
    Actor* actor,
    Span<real const> gradOutput,
    Error& error) {
  GetContactForceWorldBackwardImpl(actor, /*exclusiveCollider*/ nullptr, gradOutput, error);
}

void diffsim::GetContactForceFromActorWorldBackward(
    Actor* actor,
    Actor const* other,
    Span<real const> gradOutput,
    Error& error) {
  MOCHI_ERROR_IF(!other, error, "Invalid actor");
  MOCHI_ERROR_RETURN(error);
  GetContactForceWorldBackwardImpl(actor, /*exclusiveCollider*/ other, gradOutput, error);
}

void diffsim::GetArticulatedPoseBackward(Actor* actor, Span<real const> gradOutput, Error& error) {
  MOCHI_ERROR_RETURN_IF_BACKWARD_NOT_SUPPORTED();
  MOCHI_ERROR_IF_NOT(
      actor->GetType() == ActorType::Articulated, error, "Only articulated actors are supported.");
  int const numDofs = actor->GetNumDofs();
  MOCHI_ERROR_IF_NOT(
      isize(gradOutput) == numDofs, error, "gradOutput size must equal GetNumDofs().");
  MOCHI_ERROR_IF_NOT(IsFinite(gradOutput), error, "gradOutput must be finite.");
  MOCHI_ERROR_RETURN(error);

  // GetArticulatedPose returns joint DOFs directly (identity Jacobian).
  // Convert gradient from rotation-vector to Lie representation, then accumulate into state grad.
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 512 * sizeof(real));
  ColumnVector<real> gradLie(numDofs, &allocator);
  gradLie = AsConstView(gradOutput);

  DynamicArray<real> pose(&allocator);
  pose.resize_noinit(numDofs);
  actor->GetArticulatedPose(pose, ErrorAssert{});
  ConvertArticulatedGradientRotationVectorToLie(actor, pose, gradLie, ErrorAssert{});

  auto& stateGrad = reg.get<CDiffStateGrad>(actorImpl->e);
  stateGrad.value += gradLie;
}

static void ConvertArticulatedGradientLieToRotationVectorImpl(
    Actor const* actor,
    bool useTarget,
    Span<real> outGrad) {
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 256 * sizeof(real));
  DynamicArray<real> pose(&allocator);
  pose.resize_noinit(actor->GetNumDofs());
  if (useTarget) {
    actor->GetArticulatedTargetPose(pose, ErrorAssert{});
  } else {
    actor->GetArticulatedPose(pose, ErrorAssert{});
  }
  diffsim::ConvertArticulatedGradientLieToRotationVector(actor, pose, outGrad, ErrorAssert{});
}

void diffsim::SetArticulatedTargetPoseBackward(
    Actor const* actor,
    Span<real> outGradTargetPose,
    Error& error) {
  MOCHI_ERROR_RETURN_IF_BACKWARD_NOT_SUPPORTED(const);
  MOCHI_ERROR_IF_NOT(reg.all_of<CControllerConstraints>(e), error, "No pose controller.");
  MOCHI_ERROR_IF_NOT(
      isize(outGradTargetPose) == actor->GetNumDofs(),
      error,
      "outGradTargetPose size must equal GetNumDofs()");
  MOCHI_ERROR_RETURN(error);

  // Only claim CurrentInput if this backward function's forward counterpart was the last to set it.
  auto const& owner = reg.get<CTargetOwners const>(e);
  if (owner.newPoseOwner != TargetOwner::TargetPose) {
    AsView(outGradTargetPose).SetZero();
    return;
  }

  // Read Lie gradient from target pose gradient.
  auto const& targetPoseGrad = reg.get<CDiffTargetPoseGrad const>(e);
  AsView(outGradTargetPose) = AsConstView(targetPoseGrad.current);

  // If targetOld was set at the same step and depends on targetNew, add its gradient contribution.
  // This happens when: targetOld = targetNew - dt * velocity (so d(targetOld)/d(targetNew) = 1).
  // The total gradient is dL/d(targetNew) + dL/d(targetOld).
  if (owner.oldPoseStep == owner.newPoseStep && owner.oldPoseOwner == TargetOwner::TargetPose) {
    AsView(outGradTargetPose) += AsConstView(targetPoseGrad.previous);
  }

  // Convert from Lie gradient to rotation vector gradient.
  ConvertArticulatedGradientLieToRotationVectorImpl(actor, /*useTarget*/ true, outGradTargetPose);
}

void diffsim::SetArticulatedPoseFromJointsBackward(
    Actor const* actor,
    Span<real> outGradPose,
    Error& error) {
  MOCHI_ERROR_RETURN_IF_BACKWARD_NOT_SUPPORTED(const);
  MOCHI_ERROR_IF_NOT(
      actor->GetType() == ActorType::Articulated, error, "Only articulated actors are supported.");
  MOCHI_ERROR_IF_NOT(
      isize(outGradPose) == actor->GetNumDofs(), error, "outGradPose size must equal GetNumDofs()");
  MOCHI_ERROR_RETURN(error);

  // When a controller is present, SetArticulatedPoseFromJoints sets the body pose and resets the
  // controller target pose to the provided pose with zero target velocity. The input pose can
  // therefore own state, current-target, and previous-target gradients. Only add target gradients
  // that are still owned by SetArticulatedPoseFromJoints; later target setters may overwrite them.
  AsView(outGradPose) = reg.get<CDiffStateGrad const>(e).value;
  if (reg.all_of<CControllerConstraints>(e)) {
    auto const& targetPoseGrad = reg.get<CDiffTargetPoseGrad const>(e);
    auto const& owner = reg.get<CTargetOwners const>(e);
    if (owner.newPoseOwner == TargetOwner::PoseFromJoints) {
      AsView(outGradPose) += AsConstView(targetPoseGrad.current);
    }
    if (owner.oldPoseStep == owner.newPoseStep &&
        owner.oldPoseOwner == TargetOwner::PoseFromJoints) {
      AsView(outGradPose) += AsConstView(targetPoseGrad.previous);
    }
  }

  // Convert from Lie gradient to rotation vector gradient.
  ConvertArticulatedGradientLieToRotationVectorImpl(actor, /*useTarget*/ false, outGradPose);
}

void diffsim::SetArticulatedTargetVelocityBackward(
    Actor const* actor,
    Span<real> outGradTargetVelocity,
    Error& error) {
  MOCHI_ERROR_RETURN_IF_BACKWARD_NOT_SUPPORTED(const);
  MOCHI_ERROR_IF_NOT(reg.all_of<CControllerConstraints>(e), error, "No pose controller.");
  MOCHI_ERROR_IF_NOT(
      isize(outGradTargetVelocity) == actor->GetNumDofs(),
      error,
      "outGradTargetVelocity size must equal GetNumDofs()");
  MOCHI_ERROR_RETURN(error);

  // Only claim PreviousInput if this backward function's forward counterpart was the last to set
  // it.
  if (reg.get<CTargetOwners const>(e).velOwner != TargetOwner::TargetVelocity) {
    AsView(outGradTargetVelocity).SetZero();
    return;
  }

  // The velocity target sets the previous controller target as:
  //   target_old = target_current - dt * velocity
  // Therefore: dL/dvelocity = - dt * dL/dtarget_old
  auto const& targetPoseGrad = reg.get<CDiffTargetPoseGrad const>(e);
  auto const dt = static_cast<real>(reg.ctx<CSceneTime const>().DeltaTime());
  AsView(outGradTargetVelocity) = AsConstView(targetPoseGrad.previous);
  AsView(outGradTargetVelocity) *= -dt;

  // Convert from Lie gradient to rotation vector gradient.
  ConvertArticulatedGradientLieToRotationVectorImpl(
      actor, /*useTarget*/ true, outGradTargetVelocity);
}

void diffsim::SetExternalForcesOnDofsBackward(
    Actor const* actor,
    Span<int const> dofIndices,
    Span<real> outGradForceValues,
    Error& error) {
  MOCHI_ERROR_RETURN_IF_BACKWARD_NOT_SUPPORTED(const);
  auto const actorType = actor->GetType();
  MOCHI_ERROR_IF_NOT(
      actorType == ActorType::Articulated ||
          (actorType == ActorType::Rigid && !actor->IsNestedLinkActor()),
      error,
      "Only articulated and standalone rigid actors are supported.");
  MOCHI_ERROR_IF(dofIndices.size() != outGradForceValues.size(), error, "Sizes do not match.");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF_NOT(IsUnique(dofIndices), error, "External force DoF indices must be unique.");
  MOCHI_ERROR_RETURN(error);

  if (dofIndices.empty()) {
    return;
  }

  int const numDofs = actor->GetNumDofs();
  auto const& forceGrad = reg.get<CDiffForceGrad const>(e);

  for (int i = 0; i < isize(dofIndices); ++i) {
    MOCHI_ERROR_IF(dofIndices[i] >= numDofs || dofIndices[i] < 0, error, "DOF index out of range.");
    MOCHI_ERROR_RETURN(error);
    outGradForceValues[i] = forceGrad[dofIndices[i]];
  }
}

void diffsim::SetArticulatedJointVelocitiesBackward(
    Actor const* actor,
    Span<real> outGradVelocities,
    Error& error) {
  MOCHI_ERROR_RETURN_IF_BACKWARD_NOT_SUPPORTED(const);
  MOCHI_ERROR_IF_NOT(
      actor->GetType() == ActorType::Articulated, error, "Only articulated actors are supported.");
  int const numDofs = actor->GetNumDofs();
  MOCHI_ERROR_IF_NOT(
      isize(outGradVelocities) == numDofs, error, "outGradVelocities size must equal GetNumDofs()");
  MOCHI_ERROR_RETURN(error);

  // SetArticulatedJointVelocities sets joint velocities v. Two gradient contributions:
  //
  // 1. Velocity path: v determines the derived step Δx = v · dt, so
  //    dL/dv = dt · ProjectDerivedStateGradient(dL/dΔx).
  //    For articulated actors, the projection is:
  //    outGrad = derivedStepGrad[:numDofs] + J^T * derivedStepGrad[numDofs:]
  //
  // 2. Controller path (if pose controller exists): v sets target_old = target_current - dt * v,
  //    so dL/dv += -dt · dL/d(target_old).
  auto const dt = static_cast<real>(reg.ctx<CSceneTime const>().DeltaTime());
  auto const& derivedStepGrad = reg.get<CDiffDerivedStepGrad const>(e);
  auto const& jacobian = reg.get<CArticulatedJacobian const>(e);

  // Contribution 1: velocity path via ProjectDerivedStateGradient.
  int const linkDofs = jacobian.value.Rows();
  auto outGrad = AsView(outGradVelocities);
  outGrad = dt * derivedStepGrad.value.TopRows(numDofs);
  outGrad += dt * (jacobian.value.Transpose() * derivedStepGrad.value.BottomRows(linkDofs));

  // Contribution 2: controller path (if pose controller exists and this function owns
  // PreviousInput).
  if (reg.all_of<CControllerConstraints>(e) &&
      reg.get<CTargetOwners const>(e).velOwner == TargetOwner::JointVelocities) {
    outGrad += (-dt) * AsConstView(reg.get<CDiffTargetPoseGrad const>(e).previous);
  }

  // Convert from Lie gradient to rotation vector gradient.
  ConvertArticulatedGradientLieToRotationVectorImpl(actor, /*useTarget*/ false, outGradVelocities);
}

void diffsim::SetVelocityBackward(
    Actor const* actor,
    Span<real> outGradLinearVel,
    Span<real> outGradAngularVel,
    Error& error) {
  MOCHI_ERROR_RETURN_IF_BACKWARD_NOT_SUPPORTED(const);
  MOCHI_ERROR_IF_NOT(
      actor->GetType() == ActorType::Rigid && !actor->IsNestedLinkActor(),
      error,
      "Only standalone rigid actors are supported.");
  MOCHI_ERROR_IF_NOT(
      isize(outGradLinearVel) == RigidSize::kDTrans, error, "outGradLinearVel size must be 3.");
  MOCHI_ERROR_IF_NOT(
      isize(outGradAngularVel) == RigidSize::kDRot, error, "outGradAngularVel size must be 3.");
  MOCHI_ERROR_RETURN(error);

  // SetVelocity sets the rigid body velocity v = (v_com, ω). The velocity determines the
  // derived step Δx = v · dt. Therefore: dL/dv = dt · dL/dΔx.
  auto const dt = static_cast<real>(reg.ctx<CSceneTime const>().DeltaTime());
  auto const& derivedStepGrad = reg.get<CDiffDerivedStepGrad const>(e);

  // Linear velocity gradient: first RigidSize::kDTrans components.
  for (int i = 0; i < RigidSize::kDTrans; ++i) {
    outGradLinearVel[i] = dt * derivedStepGrad.value[i];
  }
  // Angular velocity gradient: next RigidSize::kDRot components.
  for (int i = 0; i < RigidSize::kDRot; ++i) {
    outGradAngularVel[i] = dt * derivedStepGrad.value[RigidSize::kDTrans + i];
  }
}

void diffsim::SetCenterOfMassTransformBackward(
    Actor const* actor,
    Span<real> outGradTransform,
    Error& error) {
  MOCHI_ERROR_RETURN_IF_BACKWARD_NOT_SUPPORTED(const);
  MOCHI_ERROR_IF_NOT(
      actor->GetType() == ActorType::Rigid && !actor->IsNestedLinkActor(),
      error,
      "Only standalone rigid actors are supported.");
  MOCHI_ERROR_IF_NOT(
      isize(outGradTransform) == RigidSize::kAll, error, "outGradTransform size must be 7.");
  MOCHI_ERROR_RETURN(error);

  // SetCenterOfMassTransform directly sets the rigid body's state (pose). The gradient
  // dL/d(transform) = dL/d(state) is stored in CDiffStateGrad in Lie representation. The output
  // is in quaternion representation.
  auto const gradLie = AsConstView(reg.get<CDiffStateGrad const>(e).value);
  auto const transform = actor->GetCenterOfMassTransform(ErrorAssert{});
  ConvertRigidGradientLieToQuaternion(transform, gradLie, outGradTransform, ErrorAssert{});
}

#undef MOCHI_ERROR_RETURN_IF_BACKWARD_NOT_SUPPORTED

void experimental::SetSoftMaterialParamsField(
    Actor* actor,
    SoftMaterialParams const& params,
    int elementIndex,
    Error& error) {
  MOCHI_ERROR_IF(!actor, error, "Invalid actor");
  MOCHI_ERROR_RETURN(error);
  auto* actorImpl = dynamic_cast<ActorInterfaceImpl*>(actor);
  MOCHI_ERROR_IF(!actorImpl, error, "Invalid actor implementation");
  MOCHI_ERROR_RETURN(error);
  actorImpl->SetSoftMaterialParamsField(params, elementIndex, error);
}

SoftMaterialParams
experimental::GetSoftMaterialParamsField(Actor const* actor, int elementIndex, Error& error) {
  MOCHI_ERROR_IF(!actor, error, "Invalid actor");
  MOCHI_ERROR_RETURN(error, {});
  auto const* actorImpl = dynamic_cast<ActorInterfaceImpl const*>(actor);
  MOCHI_ERROR_IF(!actorImpl, error, "Invalid actor implementation");
  MOCHI_ERROR_RETURN(error, {});
  return actorImpl->GetSoftMaterialParamsField(elementIndex, error);
}

void diffsim::ConvertRigidGradientLieToRotationVector(
    TransformRT const& state,
    Span<real> outGrad,
    Error& error) {
  MOCHI_ERROR_IF(isize(outGrad) != RigidSize::kDAll, error, "Invalid gradient size");
  MOCHI_ERROR_RETURN(error);

  // Convert the rotation gradient
  TransportInputOfLieJacobian(
      state.GetRotation().VToRotationVector(),
      AsView(outGrad).BottomRows<RigidSize::kDRot>(RigidSize::kDRot).Transpose());
}

void diffsim::ConvertArticulatedGradientLieToRotationVector(
    Actor const* actor,
    Span<real const> pose,
    Span<real> outGrad,
    Error& error) {
  MOCHI_ERROR_IF(actor == nullptr, error, "Actor must not be null");
  auto const* actorImpl = dynamic_cast<ActorInterfaceImpl const*>(actor);
  MOCHI_ERROR_IF(!actorImpl, error, "Invalid actor implementation");
  MOCHI_ERROR_RETURN(error);
  auto const& reg = actorImpl->reg;
  auto e = actorImpl->e;

  auto const* shapePtr = reg.try_get<CArticulatedBodyShape const>(e);
  MOCHI_ERROR_IF(!shapePtr, error, "Actor must be articulated");
  MOCHI_ERROR_RETURN(error);

  auto const* joints = shapePtr->shape->GetJointsData();
  auto const numDofs = reg.get<CActorDofInfo const>(e).dofsSize;
  MOCHI_ERROR_IF(isize(pose) != numDofs, error, "Invalid pose size");
  MOCHI_ERROR_IF(isize(outGrad) != numDofs, error, "Invalid gradient size");
  MOCHI_ERROR_RETURN(error);

  // Unlike other functions of articulated vectors, in this one there's no need to convert the pose,
  // because the following function expects 3D rotations in rotation-vector representation.
  articulated::TransportInputOfLieJacobian(
      joints->jointTypes, joints->dofInfo, AsConstView(pose), AsView(outGrad).Transpose());
}

void diffsim::ConvertRigidGradientQuaternionToLie(
    TransformRT const& state,
    Span<real const> inGrad,
    Span<real> outGrad,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(isize(inGrad) != RigidSize::kAll, error, "Invalid input gradient size");
  MOCHI_ERROR_IF(isize(outGrad) != RigidSize::kDAll, error, "Invalid output gradient size");
  MOCHI_ERROR_RETURN(error);

  // Copy the translation gradient
  std::copy(inGrad.begin(), inGrad.begin() + RigidSize::kDTrans, outGrad.begin());

  // Convert the rotation gradient
  auto const dgdquat = Load<Vec4r>(inGrad.data() + RigidSize::kDTrans);
  Vec4r const dgdrot = DotVecMat4x4(dgdquat, lie::DQuatDRot(state.GetRotation()));
  Store<RigidSize::kDRot>(outGrad.data() + RigidSize::kDTrans, dgdrot);
}

void diffsim::ConvertRigidGradientLieToQuaternion(
    TransformRT const& state,
    Span<real const> inGrad,
    Span<real> outGrad,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(isize(inGrad) != RigidSize::kDAll, error, "Invalid input gradient size");
  MOCHI_ERROR_IF(isize(outGrad) != RigidSize::kAll, error, "Invalid output gradient size");
  MOCHI_ERROR_RETURN(error);

  // Copy the translation gradient
  std::copy(inGrad.begin(), inGrad.begin() + RigidSize::kDTrans, outGrad.begin());

  // Convert the rotation gradient. Pad the data with zeros before the multiplication.
  Vec4r const dgdrot = Load<RigidSize::kDRot, Vec4r>(inGrad.data() + RigidSize::kDTrans);
  Vec4r const dgdrotPadded = Blend<0, 0, 0, 1>(dgdrot, Vec4r{});
  VMatrix3x4r const drotdquat = lie::DRotDQuat(state.GetRotation());
  VMatrix4x4r const drotdquatPadded = {drotdquat[0], drotdquat[1], drotdquat[2], Vec4r{}};
  Vec4r const dgdquat = DotVecMat4x4(dgdrotPadded, drotdquatPadded);
  Store<RigidSize::kRot>(outGrad.data() + RigidSize::kTrans, dgdquat);
}

void diffsim::ConvertArticulatedGradientRotationVectorToLie(
    Actor const* actor,
    Span<real const> pose,
    Span<real> outGrad,
    Error& error) {
  MOCHI_ERROR_IF(actor == nullptr, error, "Actor must not be null");
  auto const* actorImpl = dynamic_cast<ActorInterfaceImpl const*>(actor);
  MOCHI_ERROR_IF(!actorImpl, error, "Invalid actor implementation");
  MOCHI_ERROR_RETURN(error);
  auto const& reg = actorImpl->reg;
  auto e = actorImpl->e;

  auto const* shapePtr = reg.try_get<CArticulatedBodyShape const>(e);
  MOCHI_ERROR_IF(!shapePtr, error, "Actor must be articulated");
  MOCHI_ERROR_RETURN(error);

  auto const* joints = shapePtr->shape->GetJointsData();
  auto const numDofs = reg.get<CActorDofInfo const>(e).dofsSize;
  MOCHI_ERROR_IF(isize(pose) != numDofs, error, "Invalid pose size");
  MOCHI_ERROR_IF(isize(outGrad) != numDofs, error, "Invalid gradient size");
  MOCHI_ERROR_RETURN(error);

  // Unlike other functions of articulated vectors, in this one there's no need to convert the pose,
  // because the following function expects 3D rotations in rotation-vector representation.
  articulated::TransportOutputOfLieJacobian(
      joints->jointTypes,
      joints->dofInfo,
      AsConstView(pose),
      RowMatrixView<real>(outGrad.data(), isize(outGrad), 1));
}

ActorInterfacePtr CreateActorInterface(entt::registry& reg, entt::entity e, Scene* scene) {
  return std::make_unique<ActorInterfaceImpl>(reg, e, scene);
}

MOCHI_API real experimental::GetContactForceWorldBatch(
    Span<Actor const* const> actors,
    Span<Actor const* const> colliders,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  real result = 0_r;
  for (auto const* actor : actors) {
    MOCHI_ERROR_IF(!actor, error, "Invalid actor");
    MOCHI_ERROR_RETURN(error, {});
    for (auto const* collider : colliders) {
      result += Norm(actor->GetContactForceFromActorWorld(collider, error));
      MOCHI_ERROR_RETURN(error, {});
    }
  }
  return result;
}

namespace actor {
void InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<CUserData>(reg);
}
} // namespace actor

} // namespace mochi
