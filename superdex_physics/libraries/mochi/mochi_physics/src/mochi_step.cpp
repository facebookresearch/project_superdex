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
#include "mochi_blended.h"
#include "mochi_capture.h"
#include "mochi_common_components.h"
#include "mochi_compound.h"
#include "mochi_constraint.h"
#include "mochi_contact.h"
#include "mochi_contact_anchors.h"
#include "mochi_deformable.h"
#include "mochi_discretization_functions.h"
#include "mochi_group.h"
#include "mochi_island.h"
#include "mochi_linear_contact_skin.h"
#include "mochi_pose_controller.h"
#include "mochi_query.h"
#include "mochi_rigid.h"
#include "mochi_rod.h"
#include "mochi_scene_profiler.h"
#include "mochi_shell.h"
#include "mochi_sleep.h"
#include "mochi_snle.h"
#include "mochi_soft.h"
#include "mochi_soft_rom_systems.h"
#include "mochi_soft_skinned.h"
#include "mochi_solve.h"

#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/profile.h>

using namespace mochi;

// Update CBoundingVolume for actor types whose local bounds can change.
template <typename Invoke>
static void ForEachCurrentBoundsUpdateSystem(Invoke&& invoke) {
  invoke(&linear_contact_skin::UpdateBounds<TimeStep::Current>);
  invoke(&soft::UpdateBounds<TimeStep::Current>);
  invoke(&articulated::compound::UpdateBounds<TimeStep::Current>);
  invoke(&shell::UpdateBounds<TimeStep::Current>);
  invoke(&rod::UpdateSurfaceContactBounds<TimeStep::Current>);
  invoke(&rod::UpdateBounds<TimeStep::Current>);
}

static void UpdateSkinnedMaxGeometrySpeed(
    CVelocitySlice<real, TimeStep::Current, DisplacementLayer::Skinned> const& velocity,
    CConservativeStepBounds& outStepBounds) {
  outStepBounds.maxGeometrySpeed = MaxPackedVector3Norm<kSpaceDim3>(velocity.value.GetConstSpan());
}

void mochi::UpdateMaxGeometrySpeeds(entt::registry& reg) {
  MOCHI_PROFILE_SCOPE();
#if MOCHI_ASSERT_VERBOSE_ENABLED
  auto const stepBoundsView = reg.view<CConservativeStepBounds>(entt::exclude<TagStaticActor>);
  for (entt::entity const entity : stepBoundsView) {
    stepBoundsView.get<CConservativeStepBounds>(entity).maxGeometrySpeed =
        std::numeric_limits<real>::quiet_NaN();
  }
#endif

  // On-demand update of skinned velocities, not updated during the solve. Current rigid velocities
  // must have clean vsym; PreStepEcs satisfies this before reaching this function.
  ecs::InvokeForEachGlobal(&articulated::compound::UpdateSkinningVelocity, reg);
  ecs::InvokeForEachGlobal<ecs::policy::AllowReadWriteSameComponent>(
      &skinned::UpdateSkinningVelocity</*kIsState*/ false>, reg);
  ecs::InvokeForEachGlobal<ecs::policy::AllowReadWriteSameComponent>(
      &blended::UpdateBlendingVelocity, reg);

  // Compute velocity-based bounds for all actors.
  ecs::InvokeForEachGlobal(&rigid::UpdateMaxGeometrySpeed, reg);
  ecs::InvokeForEachGlobal(&deformable::UpdateMaxGeometrySpeed, reg);
  ecs::InvokeForEachGlobal(&rod::UpdateMaxGeometrySpeed, reg);
  ecs::InvokeForEachGlobal(&UpdateSkinnedMaxGeometrySpeed, reg);

#if MOCHI_ASSERT_VERBOSE_ENABLED
  for (entt::entity const entity : stepBoundsView) {
    auto const& stepBounds = stepBoundsView.get<CConservativeStepBounds>(entity);
    MOCHI_ASSERT_VERBOSE(
        IsFinite(stepBounds.maxGeometrySpeed) && stepBounds.maxGeometrySpeed >= 0_r,
        "Every dynamic geometry owner must have a finite nonnegative maximum geometry speed.");
  }
#endif
}

static void UpdateConservativeStepBounds(entt::registry& reg) {
  MOCHI_PROFILE_SCOPE();

  // Global context
  auto const& time = reg.ctx<CSceneTime const>();
  real const currTimeStep = static_cast<real>(time.DeltaTime());
  auto const stepBoundsView = reg.view<CConservativeStepBounds>(entt::exclude<TagStaticActor>);

  // If timestep is infinity, then we are doing quasistatic optimization, simply set all the bounds
  // to be infinity, because actors can move over arbitrarily large distances, as determined by the
  // optimizer, so we cannot use current configuration to determine a reasonable bound.
  if (!IsFinite(currTimeStep)) {
    for (auto&& [e, outStepBounds] : stepBoundsView.each()) {
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

  UpdateMaxGeometrySpeeds(reg);

  Vec4r const gravityAccel = reg.ctx<CSceneGravity const>().accel;
  real const gravitySpeedDelta = Norm<3>(gravityAccel) * currTimeStep;

  // For each dynamic entity with CConservativeStepBounds.
  for (auto&& [e, outStepBounds] : stepBoundsView.each()) {
    // NOTE: The conservative step bounds are heuristics based on geometry velocities, simple
    // acceleration padding, one-step actor-size relaxation after state discontinuities, etc.
    //
    // TODO: Known gaps:
    // - Contact-generated motion, especially from off-center impacts or collisions between actors
    //   whose masses differ by orders of magnitude.
    // - Force-driven motion from transmission actuators and non-target constraints such as joint
    //   limits.

    // Every actor with CConservativeStepBounds should have these
    auto const& root = reg.get<CRootTransform const>(e);
    auto const& bounds = reg.get<CBoundingVolume const>(e);

    // Start with tight fitting world-space bounds.
    Aabb const currWorldAabb = GetAabb(TransformShape(root.worldFromLocal, bounds.localShape));
    Aabb stepBounds = currWorldAabb;

    // Exaggerate the speed.
    static real kSpeedScalar = 2_r;
    real predictedMaxSpeed = kSpeedScalar * outStepBounds.maxGeometrySpeed;

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

    // A newly-created actor may be far from equilibrium, and an externally reset actor may have a
    // discontinuous state. Apply a finite, actor-size-based relaxation for this step only, then
    // clear the flag.
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

  // Update CBoundingVolume to reflect any changes since the previous step, e.g., due to actor
  // creation or API calls. Must come BEFORE UpdateConservativeStepBounds, which reads it.
  ForEachCurrentBoundsUpdateSystem(
      [&](auto const& system) { ecs::InvokeForEachGlobal(system, reg); });

  // Update conservative step bounds for all dynamic actors. This also updates velocity-based bounds
  // for finite time steps. Must come before current velocities are cleared.
  UpdateConservativeStepBounds(reg);

  // A solve-free assembly (@ref TagAssemblyPreserveHistory) reproduces the current conditions
  // instead of starting a new step.
  bool const advanceStep = !reg.try_ctx<TagAssemblyPreserveHistory>();

  // Set CRootTransform.worldFromLocalPrev equal to the current state. Must come before previous
  // state reconstruction, so worldFromLocalPrev is overwritten for rigid actors. Standalone static
  // actors instead set the start and end poses of the step from their prescribed motion.
  ecs::InvokeForEachGlobal(
      +[](ecs::Excluded<CStaticPrescribedMotion>, CRootTransform& root) {
        root.worldFromLocalPrev = root.worldFromLocal;
      },
      reg);
  ecs::InvokeForEachGlobal(&rigid::ApplyStaticStepStart, reg);

  // Adopt the friction anchors advanced at the end of the last step, apply queued friction-preload
  // resets, and index the final anchors, before collision detection copies them into the contacts.
  // A solve-free assembly keeps the anchors the last step used, but applies the resets, as it
  // applies other inputs prescribed since the last step.
  PrepareFrictionAnchors(reg, /*promotePending*/ advanceStep);

  // Update of previous state and trajectory-dependent derived state.
  if (advanceStep) {
    // RomFomSwitchingPipeline emplaces and removes several components/tags. Must run before the
    // state-shift systems below, whose dispatch depends on TagRomActor.
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
    }
  } else {
    // Reconstruct previous state (position and velocity)
    ecs::InvokeForEachGlobal(&soft::EntityReconstructPreviousStep, reg);
    ecs::InvokeForEachGlobal(&rigid::EntityReconstructPreviousStep, reg);
    ecs::InvokeForEachGlobal(&shell::EntityReconstructPreviousStep, reg);
    ecs::InvokeForEachGlobal(&rod::EntityReconstructPreviousStep, reg);
    articulated::compound::ReconstructPreviousStepPipeline(reg);
    ecs::InvokeForEachGlobal(&skinned::EntityReconstructPreviousStep, reg);
  }

  // Wake islands swept by moving static actors. Must come BEFORE island::PreStep, which applies
  // wake requests. A solve-free assembly does not change the sleep state.
  if (advanceStep) {
    reg.view<CStaticPrescribedMotion const, CRootTransform const>().each(
        [&](entt::entity e, auto const&, CRootTransform const& root) {
          if (root.worldFromLocalPrev != root.worldFromLocal) {
            sleep::WakeUp(reg, e);
          }
        });
  }

  // Wait for any pending colliders to finish initialization
  ecs::InvokeForEachGlobal(&WaitForPendingSdfCollider, reg);
  reg.clear<CSdfColliderPending>();

  // Set the old targets of pose controllers from their prescribed motion. A target velocity also
  // determines the old targets of the controller constraints, so it must come before the constraint
  // shifts below.
  ecs::InvokeForEachGlobal<ecs::policy::AllowFullRegistryAccess>(
      &articulated::compound::ApplyControllerStepStart, reg);

  // Start the step at the end targets of the last step, unless a new target was applied since then.
  // Between steps, old and current targets hold the targets at the start and at the end of the last
  // step.
  ecs::InvokeForEachGlobal(&ApplyConstraintTargetStepStart<real>, reg);
  ecs::InvokeForEachGlobal(&ApplyConstraintTargetStepStart<Real3>, reg);
  ecs::InvokeForEachGlobal(&ApplyConstraintTargetStepStart<Quaternion>, reg);
  ecs::InvokeForEachGlobal(&ApplyConstraintTargetStepStart<TransformRT>, reg);

  // Create compounds automatically for any newly added constraints.
  // Split automatically created compounds if constraints were removed.
  compound::UpdateAutoCompounds(reg);

  // Update CPotentialColliders for all dynamic actors with contact.
  // Must come AFTER CConservativeStepBounds has been updated (see above).
  contact::UpdateConservativePotentialColliders(reg);

  // Update all islands, merging or splitting them as necessary.
  // Must come AFTER CPotentialColliders has been updated (see above).
  // May add TagDofsOffsetChanged to actors.
  island::PreStep(reg, /*updateSleepState*/ advanceStep);

  // Handle TagGlobalDofsChanged
  ecs::InvokeForEachGlobal<ecs::policy::AllowFullRegistryAccess>(
      &compound::OnGlobalDofsChanged, reg);

  // If DoFs offsets change, then all such changes have been handled by this point.
  reg.clear<TagGlobalDofsChanged>();
}

void mochi::PreStepIslandAsync(entt::registry& reg, CIslandDescendants const& descendants) {
  TaskSemaphore sem;

  // A solve-free assembly (@ref TagAssemblyPreserveHistory) runs on state that is restored
  // afterwards, so it must not recenter: queries computed in a recentered frame would not match the
  // restored root transform.
  bool const advanceStep = !reg.try_ctx<TagAssemblyPreserveHistory>();

  // Recenter only in islands that are about to be solved, so that a deformable actor's root
  // transform stays fixed from here until the next step and all of its state, contact data, and
  // queries share one local frame. Must come BEFORE anything reads the root transform.
  for (auto e : descendants.softActors) {
    Schedule(sem, "PreStepDeformableActorAsync", [&reg, e, advanceStep]() {
      if (advanceStep) {
        ecs::TryInvokeOnEntity(
            &soft::RecenterSolutionUsingRigidTransformEval, reg, e); // Non-ROM only
      }
      PreStepDeformableActorAsync(reg, e);
    });
  }

  for (auto e : descendants.shellActors) {
    Schedule(sem, "PreStepDeformableActorAsync", [&reg, e, advanceStep]() {
      if (advanceStep) {
        ecs::TryInvokeOnEntity(&shell::RecenterSolutionUsingPivotNode, reg, e);
      }
      PreStepDeformableActorAsync(reg, e);
    });
  }

  for (auto e : descendants.rodActors) {
    if (advanceStep) {
      ecs::TryInvokeOnEntity(&rod::RecenterSolutionUsingPivotNode, reg, e);
    }
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
 * - If @p includeAssemblyQueries is false, queries that require contact detection or force
 *   assembly (see QueryRequiresAssembly) are left unchanged.
 */
static void UpdateActorQueriesAsync(
    TaskSemaphore const& sem,
    entt::registry& reg,
    entt::entity e,
    bool includeAssemblyQueries) {
  bool isDeformable = reg.any_of<TagSoftActor, TagBlendedActor, TagShellActor, TagRodActor>(e);
  if (isDeformable) {
    // Update CBoundingVolume for actors that deform
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

  // Rigid, Soft, Shell & Articulated
  // Reads CDisplacementSlice, and others
  // Writes CQuerySurfaceNodePositions (must happen before queries that read it)
  ecs::TryInvokeOnEntity(&UpdateQuerySurfaceNodePositions, reg, e);

  // Rigid, Soft, Shell & Articulated
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

  // Actors with a contact skin (and embeddings)
  // Writes CQueryContactSkinNodePositions, and (optionally) CQueryContactSkinNodeNormals
  ecs::TryScheduleInvokeOnEntity(
      sem,
      "rod::UpdateQueryContactSkinNodePositionsAndNormals",
      &rod::UpdateQueryContactSkinNodePositionsAndNormals,
      reg,
      e);
  // Reads CQueryNodePositions (for non-rod deformable actors)
  ecs::TryScheduleInvokeOnEntity(
      sem, "UpdateQueryContactSkin", &UpdateQueryContactSkinNodePositionsAndNormals, reg, e);

  // Soft or Rigid
  // Writes CQueryContactSamples
  ecs::TryScheduleInvokeOnEntity(
      sem, "UpdateQueryContactSamples", &UpdateQueryContactSamples, reg, e);

  // For any actor with an SDF collider.
  // Write CQuerySdfSurface
  ecs::TryScheduleInvokeOnEntity(sem, "UpdateQuerySdfSurface", &UpdateQuerySdfSurface, reg, e);

  if (!includeAssemblyQueries) {
    return;
  }

  ecs::TryScheduleInvokeOnEntity(sem, "UpdateQuerySdfDistances", &UpdateQuerySdfDistances, reg, e);

  // All dynamic actors
  // Writes CQueryContactPoints and/or CQueryNodeContactForces. Needs
  // entt::registry to look up info on the collider entity.
  ecs::TryScheduleInvokeOnEntity<ecs::policy::AllowFullRegistryAccess>(
      sem, "UpdateQueryActiveContactsWorldSpace", &UpdateQueryActiveContactsWorldSpace, reg, e);
  // Writes CQueryActorContactForces. Needs entt::registry to look up info on the collider entity.
  ecs::TryScheduleInvokeOnEntity<ecs::policy::AllowFullRegistryAccess>(
      sem, "UpdateQueryActorContactForces", &UpdateQueryActorContactForces, reg, e);
}

void mochi::UpdateAllActorQueries(entt::registry& reg, bool includeAssemblyQueries) {
  TaskSemaphore sem;
  reg.view<CActorInfo const, CIslandMemberInfo const>().each(
      [&](entt::entity e, auto const& /*actorInfo*/, auto const& islandMemberInfo) {
        // Queries are normally updated for actors in islands (not static actors).
        // Therefore, we will process those same actors here.
        if (islandMemberInfo.island != entt::null) {
          Schedule(sem, "UpdateDynamicActorQueriesAsync", [&reg, sem, e, includeAssemblyQueries]() {
            UpdateActorQueriesAsync(sem, reg, e, includeAssemblyQueries);
          });
        }
      });
  sem.Wait();
}

static void PostStepIslandAsync(entt::registry& reg, CIslandDescendants const& descendants) {
  MOCHI_PROFILE_SCOPE();
  TaskSemaphore sem;

  // Optionally verify that each actor's post-solve bounding volume fits within the
  // CConservativeStepBounds that were predicted at the beginning of the step. UpdateBounds must run
  // first, so that the check sees the current bounding volumes.
  static constexpr bool kVerifyConservativeStepBounds = false;
  if constexpr (kVerifyConservativeStepBounds) {
    // Update CBoundingVolume before the check.
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
      // Query updates for these actor types may be quite expensive, so we schedule an async task
      // for each one.
      Schedule(sem, "PostStepDeformableActorAsync", [sem, &reg, e]() {
        UpdateActorQueriesAsync(sem, reg, e, /*includeAssemblyQueries*/ true);
      });
    } else {
      UpdateActorQueriesAsync(sem, reg, e, /*includeAssemblyQueries*/ true);
    }
  }

  // Wait for any scheduled tasks
  sem.Wait();
}

void mochi::StepEcs(entt::registry& reg) {
  MOCHI_PROFILE_SCOPE();
  TaskSemaphore eachTask;

  reg.view<CIslandDescendants const, TagIslandIsAwake const>().each(
      [&](entt::entity island, CIslandDescendants const& descendants) {
        Schedule(eachTask, "StepIsland", [island, &reg, &descendants]() {
          // Keep nested island parallelism under TSAN so CI exercises synchronization paths.
          bool const useLocalSingleThreadedMode =
              !MOCHI_COMPILER_TSAN && island::ShouldRunSingleThreaded(reg, island, descendants);
          if (useLocalSingleThreadedMode) {
            TaskScheduler::PushLocalSingleThreadedMode();
          }
          MOCHI_DEFER(
              if (useLocalSingleThreadedMode) { TaskScheduler::PopLocalSingleThreadedMode(); });

          if (auto* stats = reg.try_get<CIslandStats>(island)) {
            *stats = {};
          }
          PreStepIslandAsync(reg, descendants);
          solver::StepIslandNewtonAsync(reg, island, descendants);
          profiler::ScopedIslandTimer const queryTimer(reg, island, &CIslandStats::queriesSec);
          PostStepIslandAsync(reg, descendants);
        });
      });

  eachTask.Wait();
}

void mochi::PostStepEcs(entt::registry& reg) {
  MOCHI_PROFILE_SCOPE();

  // Update CPrevRigidVelocity of soft actors
  ecs::InvokeForEachGlobal(&soft::UpdateRigidVelocity, reg);

  // Swap active elements, if necessary.
  reg.view<TagRomActor>().each([&](entt::entity e) { rom::SwapActiveElements(reg, e); });
}

namespace mochi::step {
void InitializeOnce(entt::registry& reg) {
  // CBoundingVolume is not captured, so recompute deforming bounds after restore. Must run after
  // the articulated post-restore systems, which rebuild the skinned displacements read here.
  ForEachCurrentBoundsUpdateSystem(
      [&](auto const& system) { capture::RegisterPostRestoreSystem(system, reg); });
}
} // namespace mochi::step
