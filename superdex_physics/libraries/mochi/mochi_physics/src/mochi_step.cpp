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

#include "mochi_step.h"

#include "mochi_articulated_body.h"
#include "mochi_common_components.h"
#include "mochi_compound.h"
#include "mochi_constraint.h"
#include "mochi_contact.h"
#include "mochi_deformable.h"
#include "mochi_discretization_functions.h"
#include "mochi_group.h"
#include "mochi_island.h"
#include "mochi_pose_controller.h"
#include "mochi_query.h"
#include "mochi_rigid.h"
#include "mochi_rod.h"
#include "mochi_shell.h"
#include "mochi_snle.h"
#include "mochi_soft.h"
#include "mochi_soft_rom_systems.h"
#include "mochi_soft_skinned.h"
#include "mochi_solve.h"

#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/rigid_body_utils.h>

using namespace mochi;

// Update CBoundingVolume<TimeStep::Current> for actor types whose local bounds can change.
template <typename Invoke>
static void ForEachCurrentBoundsUpdateSystem(Invoke&& invoke) {
  invoke(&soft::UpdateBounds<TimeStep::Current>);
  invoke(&articulated::compound::UpdateBounds<TimeStep::Current>);
  invoke(&shell::UpdateBounds<TimeStep::Current>);
  invoke(&rod::UpdateSurfaceContactBounds<TimeStep::Current>);
  invoke(&rod::UpdateBounds<TimeStep::Current>);
}

static void UpdateConservativeStepBounds(entt::registry& reg) {
  MOCHI_PROFILE_SCOPE();

  // Global context
  auto const& time = reg.ctx<CSceneTime const>();
  real const currTimeStep = static_cast<real>(time.DeltaTime());

  // If timestep is infinity, then we are doing quasistatic optimization, simply set all the bounds
  // to be infinity, because actors can move over arbitrarily large distances, as determined by the
  // optimizer, so we cannot use current configuration to determine a reasonable bound.
  if (!IsFinite(currTimeStep)) {
    for (auto&& [e, outStepBounds] : reg.view<CConservativeStepBounds>().each()) {
      outStepBounds.worldAabb.SetMin(
          Real3{
              -std::numeric_limits<real>::infinity(),
              -std::numeric_limits<real>::infinity(),
              -std::numeric_limits<real>::infinity()});
      outStepBounds.worldAabb.SetMax(
          Real3{
              std::numeric_limits<real>::infinity(),
              std::numeric_limits<real>::infinity(),
              std::numeric_limits<real>::infinity()});
    }
    return;
  }

  real const prevTimeStep = static_cast<real>(time.DeltaTimePrev());
  Vec4r const gravityAccel = reg.ctx<CSceneGravity const>().accel;
  real const gravitySpeedDelta = Norm<3>(gravityAccel) * currTimeStep;
  MOCHI_ASSERT(
      IsFinite(prevTimeStep) && prevTimeStep > 0_r,
      "Previous time step must be positive and finite.")

  // For each entity with CConservativeStepBounds
  for (auto&& [e, outStepBounds] : reg.view<CConservativeStepBounds>().each()) {
    // NOTE: The conservative step bounds are heuristics based on previous/current AABB deltas,
    // deformable velocity fields, simple acceleration padding, one-step actor-size relaxation after
    // history invalidation, etc.
    //
    // TODO: Known gaps:
    // - Contact-generated motion, especially from off-center impacts or collisions between actors
    //   whose masses differ by orders of magnitude.
    // - Force-driven motion from transmission actuators and non-target constraints such as joint
    //   limits.
    // - Rotation that moves geometry farther than changes in its AABB extrema indicate, especially
    //   for highly anisotropic actors.
    // - Rigid-link velocities do not expand the conservative bounds owned by a colliding
    //   articulated skin.
    // - Motion of rod visual-mesh vertices caused by twist or material-frame rotation, which is not
    //   captured by centerline translational velocities.

    // Every actor with CConservativeStepBounds should have these
    auto const& root = reg.get<CRootTransform const>(e);
    auto const& prevBounds = reg.get<CBoundingVolume<TimeStep::Previous> const>(e);
    auto const& currBounds = reg.get<CBoundingVolume<TimeStep::Current> const>(e);

    // Start with tight fitting world-space bounds
    Aabb prevWorldAabb = GetAabb(TransformShape(root.worldFromLocalPrev, prevBounds.localShape));
    Aabb currWorldAabb = GetAabb(TransformShape(root.worldFromLocal, currBounds.localShape));
    Aabb stepBounds = currWorldAabb;

    // Use finite difference to compute a speed value from the change in world-space bounds (max of
    // any cartesian direction). This indirectly accounts for linear motion, angular motion, and
    // local deformation (if any).
    Vec4r deltaMin = currWorldAabb.VGetMin() - prevWorldAabb.VGetMin();
    Vec4r deltaMax = currWorldAabb.VGetMax() - prevWorldAabb.VGetMax();
    real predictedMaxSpeed = HMax<3>(Max(Abs(deltaMin), Abs(deltaMax))) / prevTimeStep;

    // Improve the predicted speed for specific actor types.
    if (reg.all_of<TagRigidActor>(e)) {
      auto const& rigidState = reg.get<CRigidState<TimeStep::Current> const>(e).value;
      auto const& rigidVel = reg.get<CRigidVel<TimeStep::Current> const>(e).value;
      auto const [omega, vSym] = rigidVel.GetOmegaAndVSym();

      // Build the transposed rotation-velocity gradient directly for faster SIMD products.
      VMatrix3x3r const rotationVelocityGradientT = SimdSymToFull(vSym) - Skew3(omega);
      Vec4r const aabbCenterOffset = currWorldAabb.VGetCenter() - rigidState.VGetTranslation();
      Vec4r const aabbCenterVelocity =
          rigidVel.GetVCom() + DotVecMat3x3(aabbCenterOffset, rotationVelocityGradientT);

      // halfExtents * Abs(rotationVelocityGradientT) is the exact component-wise velocity radius.
      Vec4r const velocityRadius =
          DotVecMat3x3(currWorldAabb.VGetHalfExtents(), Abs(rotationVelocityGradientT));

      // Per axis, velocities span centerVelocity +/- velocityRadius, so the maximum absolute
      // component over the AABB is abs(centerVelocity) + velocityRadius.
      real const rigidMaxSpeed = HMax<3>(Abs(aabbCenterVelocity) + velocityRadius);
      predictedMaxSpeed = Max(predictedMaxSpeed, rigidMaxSpeed);
    }

    if (reg.any_of<TagSoftActor, TagShellActor>(e)) {
      // Get the velocity of each DoF.
      auto const& velocityPerDofLocal =
          reg.get<CVelocitySlice<real, TimeStep::Current> const>(e).value;

      // Use the maximum local-space speed of any DoF from the previous step. World-space speed
      // should be the same because CRootTransfrom does not have scale.
      // PERFORMANCE NOTE: For volumetric soft actors, we could check just the boundary DoFs, but
      // this is already pretty fast.
      MOCHI_ASSERT(!velocityPerDofLocal.empty());
      real maxDofSpeedLocal = MaxAbs(MakeSpan(velocityPerDofLocal));
      predictedMaxSpeed = Max(predictedMaxSpeed, maxDofSpeedLocal);
    }

    if (reg.all_of<TagRodActor>(e)) {
      // Get the velocity of each DoF (rods have 4 DoFs per node: 3 displacement + 1 twist).
      auto const& velocityPerDofLocal =
          reg.get<CVelocitySlice<real, TimeStep::Current> const>(e).value;

      // Use the maximum local-space speed of any displacement DoF from the previous step.
      // We only consider displacement DoFs (not twist) for spatial velocity bounds.
      // Rod DoFs are laid out as [dx0, dy0, dz0, twist0, dx1, dy1, dz1, twist1, ...]
      MOCHI_ASSERT(!velocityPerDofLocal.empty());
      real maxDofSpeedLocal = 0_r;
      int const numDofs = isize(velocityPerDofLocal);
      for (int i = 0; i < numDofs; i += fem::kNumRodFields) {
        // Only check the 3 displacement components, skip the twist component
        for (int j = 0; j < 3; ++j) {
          maxDofSpeedLocal = Max(maxDofSpeedLocal, Abs(velocityPerDofLocal[i + j]));
        }
      }
      predictedMaxSpeed = Max(predictedMaxSpeed, maxDofSpeedLocal);
    }

    // Exaggerate the speed.
    static real kSpeedScalar = 2_r;
    predictedMaxSpeed *= kSpeedScalar;

    // Add some additional acceleration. This will have the effect of expanding the step bounds even
    // if the actor was not originally moving. The amount of expansion will depend on the time step.
    static real kMaxPredicatedAccel = 400_r;
    predictedMaxSpeed += kMaxPredicatedAccel * currTimeStep;

    // Add the acceleration of gravity
    if (reg.all_of<TagUseGravity>(e)) {
      predictedMaxSpeed += gravitySpeedDelta;
    }

    // Extend stepBounds to account for speed. To be conservative, we extend it in all directions
    // because the kinetic energy could be redirected (e.g. elastic collision).
    stepBounds = ExpandShape(stepBounds, predictedMaxSpeed * currTimeStep);

    // Extend stepBounds to account for contact penalty falloff (if any)
    stepBounds = ExpandConservativeBoundsWithContactPadding(stepBounds, reg, e);

    // On top of everything else, add some small padding in absolute coordinates.
    static real kAbsPadding = 0.01_r;
    stepBounds = ExpandShape(stepBounds, kAbsPadding);

    // When previous-step history is unavailable or no longer trustworthy, the prediction above can
    // be unreliable, e.g. a newly-created actor may be far from equilibrium, and an externally
    // reset actor may have discontinuous state. Apply a finite, actor-size-based relaxation for
    // this step only, then clear the flag.
    if (outStepBounds.needsNextStepRelaxation) {
      constexpr real kNextStepRelaxationScale = 3_r; // Empirical value.
      real const actorDiagonal = Norm(currWorldAabb.GetMax() - currWorldAabb.GetMin());
      stepBounds = ExpandShape(stepBounds, kNextStepRelaxationScale * actorDiagonal);
      outStepBounds.needsNextStepRelaxation = false;
    }

    // Store
    outStepBounds.worldAabb = stepBounds;
  }
}

void mochi::PreStepEcs(entt::registry& reg) {
  // Called before each simulation step to prepare actors for simulation
  MOCHI_PROFILE_SCOPE();

  // Set velocities imposed externally that require knowledge of the time-step size
  ecs::InvokeForEachGlobal(&rigid::UpdateVSym, reg);
  ecs::InvokeForEachGlobal(&articulated::compound::UpdateVSym, reg);

  // Update CBoundingVolume<TimeStep::Current> to reflect any changes since the previous step, e.g.,
  // due to actor creation, API calls, etc. Must come BEFORE UpdateConservativeStepBounds which
  // reads CBoundingVolume<Current>.
  ForEachCurrentBoundsUpdateSystem(
      [&](auto const& system) { ecs::InvokeForEachGlobal(system, reg); });

  // Update CConservativeStepBounds for all dynamic actors.
  // Must come BEFORE CRootTransform.worldFromLocalPrev or CBoundingVolume<TimeStep::Previous> are
  // updated for this step.
  UpdateConservativeStepBounds(reg);

  // Compute linear and angular velocity about the center-of-mass (static rigid actors only)
  // Must come BEFORE CRootTransform.worldFromLocalPrev is updated for this step.
  ecs::InvokeForEachGlobal(&rigid::UpdateRigidVelocity_Static, reg);

  // RomFomSwitchingPipeline emplaces and removes several components/tags. It must be invoked BEFORE
  // any other systems that rely on such components/tags.
  reg.view<CRomFomSwitchingParams>().each(
      [&](entt::entity e, CRomFomSwitchingParams const& params) {
        rom::RomFomSwitchingPipeline(reg, e, params);
      });

  // Initialize previous state (position and velocity)
  {
    MOCHI_PROFILE_SCOPE_N("Shift State");
    ecs::InvokeForEachGlobal(&soft::EntityIncrementStep, reg);
    ecs::InvokeForEachGlobal(&rigid::EntityIncrementStep, reg);
    ecs::InvokeForEachGlobal(&shell::EntityIncrementStep, reg);
    ecs::InvokeForEachGlobal(&rod::EntityIncrementStep, reg);
    articulated::compound::PreStepPipeline(reg);
    rom::PreStepPipeline(reg);
    ecs::InvokeForEachGlobal(&skinned::EntityIncrementStep, reg);

    // Set CRootTransform.worldFromLocalPrev equal to the current state, except for static actors.
    ecs::InvokeForEachGlobal(
        +[](ecs::Excluded<TagStaticActor>, CRootTransform& root) {
          root.worldFromLocalPrev = root.worldFromLocal;
        },
        reg);

    // Set CBoundingVolume<TimeStep::Previous> equal to the current state, except for static and
    // rigid actors.
    ecs::InvokeForEachGlobal(
        +[](ecs::Excluded<TagStaticActor, TagRigidActor>,
            CBoundingVolume<TimeStep::Current> const& current,
            CBoundingVolume<TimeStep::Previous>& outPrevious) {
          outPrevious.localShape = current.localShape;
        },
        reg);
  }

  // Wait for any pending colliders to finish initialization
  ecs::InvokeForEachGlobal(&WaitForPendingSdfCollider, reg);
  reg.clear<CSdfColliderPending>();

  // Set old targets of pose controllers if velocity was imposed externally
  ecs::InvokeForEachGlobal<ecs::policy::AllowFullRegistryAccess>(
      &articulated::compound::SetOldControllerTargets, reg);

  // Create compounds automatically for any newly added constraints.
  // Split automatically created compounds if constraints were removed.
  compound::UpdateAutoCompounds(reg);

  // Update CPotentialColliders for all dynamic actors with contact.
  // Must come AFTER CConservativeStepBounds has been updated (see above).
  contact::UpdateConservativePotentialColliders(reg);

  // Update all islands, merging or splitting them as necessary.
  // Must come AFTER CPotentialColliders has been updated (see above).
  // May add TagDofsOffsetChanged to actors.
  island::PreStep(reg);

  // Handle TagGlobalDofsChanged
  ecs::InvokeForEachGlobal<ecs::policy::AllowFullRegistryAccess>(
      &compound::OnGlobalDofsChanged, reg);

  // If DoFs offsets change, then all such changes have been handled by this point.
  reg.clear<TagGlobalDofsChanged>();
}

void mochi::PreStepIslandAsync(entt::registry& reg, CIslandDescendants const& descendants) {
  TaskSemaphore sem;

  for (auto e : descendants.softActors) {
    Schedule(
        sem, "PreStepDeformableActorAsync", [&reg, e]() { PreStepDeformableActorAsync(reg, e); });
  }

  for (auto e : descendants.shellActors) {
    Schedule(
        sem, "PreStepDeformableActorAsync", [&reg, e]() { PreStepDeformableActorAsync(reg, e); });
  }

  for (auto e : descendants.rodActors) {
    PreStepRodActorAsync(reg, e);
  }

  // PreStepRigidActorAsync is used for both (normal) rigid bodies and for articulated rigid bodies.
  for (auto e : descendants.rigidActors) {
    PreStepRigidActorAsync(reg, e);
  }

  for (auto e : descendants.compoundActors) {
    articulated::compound::PreStepArticulatedBodyActorAsync(reg, e);
  }

  // Wait for any scheduled tasks
  sem.Wait();
}

/**
 * @brief Update derived state, including queries, for a single actor.
 *
 * @remarks
 * - Called by PostStepIslandAsync during simulation step. Also called by UpdateAllActorQueries.
 * - Can be called concurrently for different actors.
 * - May schedule additional work. Wait for the semaphore to ensure completion.
 */
static void UpdateActorQueriesAsync(TaskSemaphore sem, entt::registry& reg, entt::entity e) {
  bool isDeformable = reg.any_of<TagSoftActor, TagBlendedActor, TagShellActor, TagRodActor>(e);
  if (isDeformable) {
    // Update CBoundingVolume<TimeStep::Current> for actors that deform
    // NOTE: Invoke in this thread since CBoundingVolume is NOT a CQuery component and it's read by
    // some of the UpdateQuery systems below.
    ForEachCurrentBoundsUpdateSystem(
        [&](auto const& system) { ecs::TryInvokeOnEntity(system, reg, e); });

    // Writes CQueryElasticEnergy for soft actors, including nested soft actors. It assembles the
    // energy when stress is enabled and otherwise sets it to zero.
    ecs::TryScheduleInvokeOnEntity(
        sem, "UpdateQueryElasticEnergy", &soft::UpdateQueryElasticEnergy, reg, e);

    // Reads CDisplacementSlice, and others
    // Writes CQueryElementsDeformationGradients
    ecs::TryScheduleInvokeOnEntity(
        sem,
        "soft::UpdateQueryElementsDeformationGradient",
        &soft::UpdateQueryElementsDeformationGradient,
        reg,
        e);

    // Reads CDisplacementSlice, and others
    // Writes CQueryQuadraturePointsPosition
    ecs::TryScheduleInvokeOnEntity(
        sem,
        "soft::UpdateQueryQuadraturePointsPosition",
        &soft::UpdateQueryQuadraturePointsPosition,
        reg,
        e);
  }

  // Soft & Actors with skinned mesh.
  // Reads CDisplacementSlice, and others
  // Writes CQueryNodePositions (must happen before queries that read it)
  ecs::TryInvokeOnEntity(&UpdateQueryNodePositions, reg, e);

  // Rigid & Soft
  // Reads CDisplacementSlice, and others
  // Writes CQuerySurfaceNodePositions (must happen before queries that read it)
  ecs::TryInvokeOnEntity(&UpdateQuerySurfaceNodePositions, reg, e);

  // Soft & Rigid
  // Reads CQuerySurfaceNodePositions
  // Writes CQuerySurfaceNodeNormals
  ecs::TryScheduleInvokeOnEntity(
      sem, "UpdateQuerySurfaceNodeNormals", &UpdateQuerySurfaceNodeNormals, reg, e);

  // Actors with a visual mesh (and embeddings)
  // Writes CQueryVisualNodePositions, and (optionally) CQueryVisualNodeNormals
  ecs::TryScheduleInvokeOnEntity(
      sem,
      "rod::UpdateQueryVisualNodePositionsAndNormals",
      &rod::UpdateQueryVisualNodePositionsAndNormals,
      reg,
      e);
  // Reads CQueryNodePositions (for non-rod deformable actors)
  ecs::TryScheduleInvokeOnEntity(
      sem, "UpdateQueryVisual", &UpdateQueryVisualNodePositionsAndNormals, reg, e);

  // Soft or Rigid
  // Writes CQueryContactSamples
  ecs::TryScheduleInvokeOnEntity(
      sem, "UpdateQueryContactSamples", &UpdateQueryContactSamples, reg, e);

  ecs::TryScheduleInvokeOnEntity(sem, "UpdateQuerySdfDistances", &UpdateQuerySdfDistances, reg, e);

  // For any actor with an SDF collider.
  // Write CQuerySdfSurface
  ecs::TryScheduleInvokeOnEntity(sem, "UpdateQuerySdfSurface", &UpdateQuerySdfSurface, reg, e);

  // All dynamic actors
  // Writes CQueryContactPoints and/or CQueryNodeContactForces. Needs
  // entt::registry to look up info on the collider entity.
  ecs::TryScheduleInvokeOnEntity<ecs::policy::AllowFullRegistryAccess>(
      sem, "UpdateQueryActiveContactsWorldSpace", &UpdateQueryActiveContactsWorldSpace, reg, e);
  // Writes CQueryActorContactForces. Needs entt::registry to look up info on the collider entity.
  ecs::TryScheduleInvokeOnEntity<ecs::policy::AllowFullRegistryAccess>(
      sem, "UpdateQueryActorContactForces", &UpdateQueryActorContactForces, reg, e);
}

void mochi::UpdateAllActorQueries(entt::registry& reg) {
  TaskSemaphore sem;
  reg.view<CActorInfo const, CIslandMemberInfo const>().each(
      [&](entt::entity e, auto const& /*actorInfo*/, auto const& islandMemberInfo) {
        // Queries are normally updated for actors in islands (not static actors).
        // Therefore, we will process those same actors here.
        if (islandMemberInfo.island != entt::null) {
          Schedule(sem, "UpdateDynamicActorQueriesAsync", [&reg, sem, e]() {
            UpdateActorQueriesAsync(sem, reg, e);
          });
        }
      });
  sem.Wait();
}

static void PostStepIslandAsync(entt::registry& reg, CIslandDescendants const& descendants) {
  MOCHI_PROFILE_SCOPE();
  TaskSemaphore sem;

  // Optionally verify that each actor's post-solve bounding volume fits within the
  // CConservativeStepBounds that were predicted at the beginning of the step.
  //
  // Ordering constraints:
  //   1. UpdateBounds must run first (so the check sees the current bounding volumes).
  //   2. The check must run before recentering (which modifies worldFromLocal for non-ROM soft
  //      actors, invalidating the check logic).
  static constexpr bool kVerifyConservativeStepBounds = false;
  if constexpr (kVerifyConservativeStepBounds) {
    // Update CBoundingVolume<TimeStep::Current> before the check.
    for (auto e : descendants.actors) {
      ForEachCurrentBoundsUpdateSystem(
          [&](auto const& system) { ecs::TryInvokeOnEntity(system, reg, e); });
    }
    for (auto e : descendants.actors) {
      CheckConservativeStepBounds(reg, e);
    }
  }

  for (auto e : descendants.actors) {
    if (reg.any_of<TagSoftActor, TagBlendedActor, TagShellActor>(e)) {
      // These actor types need to call PostStepDeformableActorAsync. It may be quite expensive, so
      // we schedule an async task for each one.
      Schedule(sem, "PostStepDeformableActorAsync", [sem, &reg, e]() {
        ecs::TryInvokeOnEntity(
            &soft::RecenterSolutionUsingRigidTransformEval, reg, e); // Non-ROM only

        UpdateActorQueriesAsync(sem, reg, e);
      });
    } else {
      UpdateActorQueriesAsync(sem, reg, e);
    }
  }

  // Wait for any scheduled tasks
  sem.Wait();
}

void mochi::StepEcs(entt::registry& reg) {
  MOCHI_PROFILE_SCOPE();
  TaskSemaphore eachTask;

  reg.view<CIslandDescendants const>().each([&](entt::entity island,
                                                CIslandDescendants const& descendants) {
    Schedule(eachTask, "StepIsland", [island, &reg, &descendants]() {
      // Keep nested island parallelism under TSAN so CI exercises synchronization paths.
      bool const useLocalSingleThreadedMode =
          !MOCHI_COMPILER_TSAN && island::ShouldRunSingleThreaded(reg, island, descendants);
      if (useLocalSingleThreadedMode) {
        TaskScheduler::PushLocalSingleThreadedMode();
      }
      MOCHI_DEFER(if (useLocalSingleThreadedMode) { TaskScheduler::PopLocalSingleThreadedMode(); });

      PreStepIslandAsync(reg, descendants);
      solver::StepIslandNewtonAsync(reg, island, descendants);
      PostStepIslandAsync(reg, descendants);
    });
  });

  eachTask.Wait();
}

void mochi::PostStepEcs(entt::registry& reg) {
  MOCHI_PROFILE_SCOPE();

  // Set CRootTransform.worldFromLocalPrev equal to the current state for static actors.
  ecs::InvokeForEachGlobal(
      +[](ecs::RequiredTag<TagStaticActor>, CRootTransform& root) {
        root.worldFromLocalPrev = root.worldFromLocal;
      },
      reg);

  // Update CPrevRigidVelocity
  ecs::InvokeForEachGlobal(&rigid::UpdateRigidVelocity_Dynamic, reg);
  ecs::InvokeForEachGlobal(&soft::UpdateRigidVelocity, reg);

  // Swap active elements, if necessary.
  reg.view<TagRomActor>().each([&](entt::entity e) { rom::SwapActiveElements(reg, e); });

  // Update constraints that receive external targets
  ecs::InvokeForEachGlobal(&UpdateConstraintOldTarget<real>, reg);
  ecs::InvokeForEachGlobal(&UpdateConstraintOldTarget<Real3>, reg);
  ecs::InvokeForEachGlobal(&UpdateConstraintOldTarget<Quaternion>, reg);
  ecs::InvokeForEachGlobal(&UpdateConstraintOldTarget<TransformRT>, reg);

  // Update the external targets of pose controllers
  ecs::InvokeForEachGlobal(&controller::UpdateOldTargets, reg);
}
