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

#include "mochi_scene.h"

#include "mochi_actor.h"
#include "mochi_actor_convergence.h"
#include "mochi_articulated_actor_params.h"
#include "mochi_articulated_body.h"
#include "mochi_blended.h"
#include "mochi_capture.h"
#include "mochi_compound.h"
#include "mochi_constraint.h"
#include "mochi_constraint_interface.h"
#include "mochi_contact.h"
#include "mochi_contact_filter.h"
#include "mochi_contact_pair_params.h"
#include "mochi_context.h"
#include "mochi_debug_draw.h"
#include "mochi_differentiable.h"
#include "mochi_ecs_init.h"
#include "mochi_ecs_utils.h"
#include "mochi_hdf5.h"
#include "mochi_island.h"
#include "mochi_query.h"
#include "mochi_rigid.h"
#include "mochi_rod.h"
#include "mochi_scene_debugger.h"
#include "mochi_shape.h"
#include "mochi_shell_init.h"
#include "mochi_simulation.h"
#include "mochi_soft.h"
#include "mochi_soft_init.h"
#include "mochi_soft_rom_init.h"
#include "mochi_soft_skinned.h"
#include "mochi_step.h"

#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/integration/integration_utils.h>
#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/mochi_platform.h>
#include <mochi_core/solvers/linear_solver.h>
#include <mochi_core/utils/cuda_utils.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/group_rw.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/time.h>

#include <mochi_physics/diffsim/mochi_diffsim.h>
#include <mochi_physics/mochi_physics_config.h>
#include <mochi_physics/utils/mochi_prefab.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace mochi::experimental;

namespace mochi {

namespace {
// RAII guard that binds the calling thread to the task scheduler when it is not already bound, and
// unbinds it again on scope exit. Public API entry points that run parallel work use this so the
// work still parallelizes when they are called from a thread that has not been bound to the
// scheduler; otherwise the parallel primitives silently fall back to single-threaded execution.
class ScopedSchedulerBinding {
 public:
  explicit ScopedSchedulerBinding(ContextImpl& context)
      : _context(TaskScheduler::TryGet() == nullptr ? &context : nullptr) {
    if (_context != nullptr) {
      _context->BindThisThread();
    }
  }
  ~ScopedSchedulerBinding() {
    if (_context != nullptr) {
      _context->UnbindThisThread();
    }
  }

  MOCHI_DECLARE_NO_COPY_NO_MOVE(ScopedSchedulerBinding);

 private:
  ContextImpl* _context;
};

// Rolls back independent cleanup roots during multi-actor construction. When ownership transfers
// to an aggregate, replace its children with that owner. Roots are destroyed in reverse
// registration order so later-registered dependents are removed before actors they reference.
class ScopedActorCreationRollback {
 public:
  explicit ScopedActorCreationRollback(SceneImpl& scene) : _scene(scene) {}
  ~ScopedActorCreationRollback() noexcept {
    if (!_active) {
      return;
    }
    for (int i = isize(_actors); i-- > 0;) {
#if MOCHI_ASSERT_VERBOSE_ENABLED
      auto const entity = GetEntityUnchecked(_actors[i]);
      auto const& registry = _scene.GetRegistry();
      MOCHI_ASSERT_VERBOSE(
          !registry.valid(entity) || !registry.all_of<CGroupMemberInfo>(entity),
          "Actor creation rollback cannot directly own a nested actor.");
#endif // MOCHI_ASSERT_VERBOSE_ENABLED
      _scene.DestroyActor(_actors[i]);
    }
  }

  MOCHI_DECLARE_NO_COPY_NO_MOVE(ScopedActorCreationRollback);

  void Add(ActorHandle actor) {
    if (actor.IsValid()) {
      _actors.push_back(actor);
    }
  }

  void ReplaceWithOwner(ActorHandle actor) {
    _actors.clear();
    Add(actor);
  }

  void Release() {
    _active = false;
  }

 private:
  SceneImpl& _scene;
  DynamicArray<ActorHandle> _actors;
  bool _active = true;
};
} // namespace

static void
DestroyAllItemsInArticulatedActor(SceneImpl& scene, entt::registry& registry, entt::entity e) {
  MOCHI_ASSERT(
      registry.valid(e) && registry.all_of<TagArticulatedActor>(e),
      "Not a valid articulated actor.");

  auto& groupMembers = registry.get<CGroupMembers>(e);
  std::vector<entt::entity> memberActorsCopy = groupMembers.actors;
  std::vector<entt::entity> memberConstraintsCopy = groupMembers.constraints;

  // Destroy constraints in the articulation.
  for (auto const& constraint : memberConstraintsCopy) {
    scene.DestroyConstraint(GetConstraintHandle(constraint, scene.GetHandle()));
  }

  // Detach actors from the articulation.
  groupMembers.actors.clear();
  for (auto const& actor : memberActorsCopy) {
    registry.erase<CGroupMemberInfo>(actor);
  }

  // Destroy actors in the articulation (legal now that they are detached).
  for (auto const& actor : memberActorsCopy) {
    scene.DestroyActor(GetActorHandle(actor, scene.GetHandle()));
  }
}

static void DestroyActorEntity(SceneImpl& scene, entt::registry& registry, entt::entity e) {
  // If this actor was affected by constraint, then destroy those constraints.
  if (auto* constraintMemberInfo = registry.try_get<CConstraintMemberInfo>(e)) {
    auto constraintsCopy = constraintMemberInfo->constraints;
    for (entt::entity c : constraintsCopy) {
      scene.DestroyConstraint(GetConstraintHandle(c, scene.GetHandle()));
    }
  }

  // If this actor belongs to a compound, then remove it from that compound.
  auto* groupInfo = registry.try_get<CGroupMemberInfo>(e);
  if (groupInfo) {
    RemoveActorFromCompound(registry, groupInfo->group, e, ErrorAssert{});
  }

  // If this actor is an articulation, destroy its members.
  if (registry.all_of<TagArticulatedActor>(e)) {
    DestroyAllItemsInArticulatedActor(scene, registry, e);
  }

  // Clean actor-vs-actor contact tables.
  registry.ctx<CContactFilterTable>().RemoveEntity(e);
  registry.ctx<CContactPairParamsOverrideTable>().RemoveEntity(e);

  // Remove actor from its island (if any)
  island::RemoveActor(registry, e);

  // Destroy the ECS entity and all components
  registry.destroy(e);
}

template <typename EnumT>
[[nodiscard]] static bool IsValidEnumValue(EnumT value, EnumT count) {
  auto const valueInt = static_cast<int>(value);
  return valueInt >= 0 && valueInt < static_cast<int>(count);
}

static void CheckStateCaptureSupported(entt::registry const& reg, Error& error) {
  MOCHI_ERROR_IF(
      !reg.storage<TagRomActor>().empty(),
      error,
      "State capture is not supported for scenes with ROM actors.");
}

[[nodiscard]] static bool ActorCanOwnNestedActors(Actor const& actor) {
  bool const canOwnNestedActors = actor::CanOwnNestedActors(actor.GetType());

#if MOCHI_ASSERT_VERBOSE_ENABLED
  Error nestedLinkError;
  [[maybe_unused]] auto const nestedLinks = actor.GetNestedLinkActors(nestedLinkError);
  Error nestedSoftError;
  [[maybe_unused]] auto const nestedSoft = actor.GetNestedSoftActors(nestedSoftError);
  MOCHI_ASSERT_VERBOSE(
      (nestedLinkError.IsOK() == nestedSoftError.IsOK()) &&
          (nestedLinkError.IsOK() == canOwnNestedActors),
      "Nested actor accessor contract changed. Update actor::CanOwnNestedActors.");
#endif // MOCHI_ASSERT_VERBOSE_ENABLED

  return canOwnNestedActors;
}

[[nodiscard]] static bool IsValidIncludeNestedActors(IncludeNestedActors includeNestedActors) {
  static_assert(
      static_cast<int>(IncludeNestedActors::Count) == 2,
      "Update IsValidIncludeNestedActors if IncludeNestedActors enum changes.");
  switch (includeNestedActors) {
    case IncludeNestedActors::No:
    case IncludeNestedActors::Yes:
      return true;
    case IncludeNestedActors::Count:
      return false;
  }
  return false;
}

static void CollectActorContactHandles(
    entt::registry const& registry,
    Scene const& scene,
    ActorHandle handle,
    IncludeNestedActors includeNestedActors,
    DynamicArray<ActorHandle>& outHandles,
    Error& error) {
  [[maybe_unused]] auto const entity = GetEntity(registry, handle, error);
  MOCHI_ERROR_RETURN(error);

  Actor const* actor = scene.GetActor(handle);
  MOCHI_ASSERT_VERBOSE(
      actor != nullptr, "ActorHandle maps to a valid entity, but that entity is not an actor.");

  if ((includeNestedActors == IncludeNestedActors::Yes) && ActorCanOwnNestedActors(*actor)) {
    auto const nestedLinks = actor->GetNestedLinkActors(ErrorAssert{});
    auto const nestedSoft = actor->GetNestedSoftActors(ErrorAssert{});
    outHandles.reserve(outHandles.size() + nestedLinks.size() + nestedSoft.size() + 1);
    outHandles.append(nestedLinks);
    outHandles.append(nestedSoft);
  }

  outHandles.push_back(handle);
}

// Helper function to compute the aggregate solver stats for a step.
static void ComputeAggregateSolverSceneStats(
    entt::registry& reg,
    SolverStats& outStats,
    DebugStats& outDebugStats) {
  // Retrieve the integration method from solver parameters to determine how many integration
  // stages are used in the time-stepping scheme (e.g., backward Euler has 1 stage, while
  // higher-order methods may have multiple stages).
  auto const intMethod = reg.ctx<CSimulationParams const>().integrationMethod;
  auto const numStages = GetNumStages(intMethod);
  MOCHI_ASSERT(numStages <= kMaxIntegrationStages, "Unexpected number of integration stages");

  // Aggregate solver statistics across all islands in the scene. We compute:
  // 1. The total squared residual norm across all islands and integration stages.
  // 2. The maximum number of non-linear solver iterations across all islands and stages.
  // NOTE: We accumulate squared residual norms (without taking the square root) because we'll
  // compute the root-mean-square (RMS) across all stages at the end.
  outStats.residualNorm = 0.0;
  outStats.maxNonLinearIters = 0;
  outStats.maxLineSearchIters = 0;
  outDebugStats.maxResidualNormRelativeError = 0_r;
  reg.view<CIslandSolverStats const>().each(
      [numStages, &outStats, &outDebugStats](auto const& islandSolverStats) {
        int const islandNumStages = isize(islandSolverStats.stages);
        MOCHI_ASSERT(islandNumStages == numStages, "Unexpected number of integration stages");
        for (int stageIdx = 0; stageIdx < islandNumStages; ++stageIdx) {
          auto const& stage = islandSolverStats.stages[stageIdx];
          outStats.residualNorm += Sqr((double)stage.resNorm);
          outStats.maxNonLinearIters = Max(outStats.maxNonLinearIters, stage.numIterDone);
          outStats.maxLineSearchIters = Max(outStats.maxLineSearchIters, stage.numLSIterDone);
          outDebugStats.maxResidualNormRelativeError =
              Max(outDebugStats.maxResidualNormRelativeError, stage.resNormError);
        }
      });

  // Compute the final RMS residual norm.
  outStats.residualNorm = Sqrt(outStats.residualNorm / numStages);

  // Aggregate convergence status across all actors.
  outStats.convergenceStatus = ConvergenceStatus::None;
  reg.view<CConvergenceStatus const>().each([&outStats](auto const& convergence) {
    outStats.convergenceStatus = Max(outStats.convergenceStatus, convergence.stepStatus);
  });
}

// Helper function to compute the aggregate backprop solver stats for a step.
static void ComputeAggregateBackPropSolverSceneStats(
    entt::registry& reg,
    diffsim::BackPropagationSceneStats& outStats) {
  double sqrResNorm = 0.0;
  outStats.maxOuterIters = 0;
  outStats.finiteDiffValid = true;
  reg.view<CIslandBackPropSolverStats>().each([&](auto& islandSolverStats) {
    auto const& stats = islandSolverStats.stats;
    sqrResNorm += Sqr((double)stats.resNorm);
    outStats.maxOuterIters = Max(outStats.maxOuterIters, stats.numIterDone);
    outStats.finiteDiffValid = outStats.finiteDiffValid && islandSolverStats.finiteDiffValid;
  });
  outStats.residualNorm = Sqrt(sqrResNorm);
}

// Declared in MochiDebugDrawSystems.cpp
void RegisterDebugDrawSystems(DebugDrawInternal& debugDraw);

// This ECS component simply ensures that its entity cannot be accidentally
// destroyed before final shutdown.
namespace {
struct ZeroEntityWatchdog {
  bool isShuttingDown = false;
  ~ZeroEntityWatchdog() {
    MOCHI_ASSERT(
        isShuttingDown,
        "It is illegal to destroy the entity with identifier zero until the whole "
        "registry is destroyed. This ensures that the value zero cannot be used for "
        "actors.");
  }
};

// This ECS component stores a pointer to an implementation of the virtual mochi::Actor
// interface. Mochi Physics internals should use the other ECS components directly and should
// not need to call the virtual API, which is why it is only available in this file.
struct CActorInterface {
  ActorInterfacePtr ptr;
};

// This ECS component stores a pointer to an implementation of the virtual mochi::Constraint
// interface. Similar to CActorInterface (see above).
struct CConstraintInterface {
  ConstraintInterfacePtr ptr;
};

// Detect premature destruction of the entity with entt::entity value zero.
using ZeroEntityWatchdogPtr = std::unique_ptr<ZeroEntityWatchdog>;
} // namespace

SceneImpl::SceneImpl(ContextImpl* context, std::string_view name, uint64_t uniqueId)
    : _context(context), _name(name), _sceneId(uniqueId) {
  MOCHI_ASSERT(context != nullptr);

  // This must be called once before ecs::RegisterComponent
  ecs::InitializeComponentRegistryOnce(_registry);

  // Register the component types that are unique to this cpp file
  ecs::RegisterComponent<CActorInterface>(_registry);
  ecs::RegisterComponent<CConstraintInterface>(_registry);
  ecs::RegisterComponent<ZeroEntityWatchdogPtr>(_registry);

  // Let other sub-systems perform one-time initialization, including the registration of all of
  // their ECS component types.
  ecs::InitializeOnce(_registry);

  // It is illegal to register additional component types after this point
  ecs::FinalizeComponentRegistration(_registry);

  // The first entity created by the EnTT registry will have an identifier value of zero, but the
  // Mochi public API handle's assume that zero will always be an invalid value. We can make that
  // assumption true by claiming ID zero now, and never releasing it.
  entt::entity zeroEntity = _registry.create();
  MOCHI_ASSERT((int)zeroEntity == 0, "Expected the first call to registry::create to return zero");

  // Detect premature destruction of zeroEntity
  _registry.emplace<ZeroEntityWatchdogPtr>(zeroEntity, std::make_unique<ZeroEntityWatchdog>());

  // Create a component to store the scene's time step (global for this registry)
  _registry.set<CSceneTime>();

  // Create a component to store the scene's time step number  (global for this registry).
  _registry.set<CSceneStepCounter>();

  // Create a component to store the scene's handle (global for this registry).
  _registry.set<CSceneHandle>(SceneHandle{_sceneId});

  // Set the default gravity (global for this registry)
  auto& gravity = _registry.set<CSceneGravity>();
  gravity.accel = ToSimd(kDefaultGravity);

  // Set the default solver parameters (global for this registry)
  _registry.set<CSimulationParams>();

  // Create the one DebugDrawImpl
  _debugDraw = DebugDrawInternal::Create(_registry);
  RegisterDebugDrawSystems(*_debugDraw);
}

SceneImpl::~SceneImpl() {
  // Stop recording before we touch the registry.
  _recorder.reset();

  _debugDraw.reset();

  // Wait for any pending colliders to finish initialization
  ecs::InvokeForEachGlobal(&WaitForPendingSdfCollider, _registry);
  _registry.clear<CSdfColliderPending>();

  // Indicate that it is now OK to destroy the entity with ID zero
  _registry.get<ZeroEntityWatchdogPtr>((entt::entity)0)->isShuttingDown = true;

  // Destroy all remaining entities and components
  _registry.clear();
}

bool SceneImpl::TryClaimOwnership() {
  return !_ownershipClaimed.exchange(true, std::memory_order_relaxed);
}

char const* SceneImpl::GetName() const {
  return _name.c_str();
}

Context* SceneImpl::GetContext() {
  return _context;
}

Context const* SceneImpl::GetContext() const {
  return _context;
}

SceneHandle SceneImpl::GetHandle() const {
  MOCHI_ASSERT_VERBOSE(_registry.ctx<CSceneHandle>().value == SceneHandle{_sceneId});
  return SceneHandle{_sceneId};
}

void SceneImpl::SetGravity(Real3 const& gravityAccelWorld) {
  // Store in the CSceneGravity component (global to our registry)
  _registry.ctx<CSceneGravity>().accel = ToSimd(gravityAccelWorld, 0_r);

  // Invalidate actor convergence weights.
  for (auto&& [e, weights] : _registry.view<CActorConvergenceWeights>().each()) {
    InvalidateActorConvergenceWeights(_registry, e);
  }
}

SolverParams SceneImpl::GetSolverParams() const {
  return _registry.ctx<CSimulationParams const>();
}

void SceneImpl::SetSolverParams(SolverParams const& params, Error& error) {
  MOCHI_ERROR_IF_NOT(
      IsValidEnumValue(params.nonLinearSolver.solverType, NonLinearSolverType::Count),
      error,
      "Invalid non-linear solver type (NonLinearSolverParams::solverType).");
  MOCHI_ERROR_IF_NOT(
      IsValidEnumValue(params.nonLinearSolver.psdProjMode, PsdProjectionMode::Count),
      error,
      "Invalid PSD projection mode (NonLinearSolverParams::psdProjMode).");
  MOCHI_ERROR_IF_NOT(
      IsValidEnumValue(params.nonLinearSolver.lineSearchType, LineSearchType::Count),
      error,
      "Invalid line search type (NonLinearSolverParams::lineSearchType).");
  MOCHI_ERROR_IF_NOT(
      IsValidEnumValue(
          params.nonLinearSolver.linearToleranceStrategy, LinearToleranceStrategy::Count),
      error,
      "Invalid strategy for adaptive linear solver tolerance (NonLinearSolverParams::linearToleranceStrategy).");
  MOCHI_ERROR_IF_NOT(
      IsValidEnumValue(
          params.nonLinearSolver.convergenceMode, NonLinearSolverConvergenceMode::Count),
      error,
      "Invalid non-linear solver convergence mode (NonLinearSolverParams::convergenceMode).");
  MOCHI_ERROR_IF_NOT(
      IsValidEnumValue(params.linearSolver.solverType, LinearSolverType::Count),
      error,
      "Invalid linear solver type (LinearSolverParams::solverType).");
  MOCHI_ERROR_IF_NOT(
      IsValidEnumValue(params.linearSolver.preconditionerType, PreconditionerType::Count),
      error,
      "Invalid preconditioner type (LinearSolverParams::preconditionerType).");
  MOCHI_ERROR_IF_NOT(
      IsValidEnumValue(params.linearSolver.normType, LinearSolverConvergenceNorm::Count),
      error,
      "Invalid linear solver norm type (LinearSolverParams::normType).");
  MOCHI_ERROR_IF_NOT(
      IsValidEnumValue(params.integrationMethod, IntegrationMethod::Count),
      error,
      "Invalid time integration method.");
  MOCHI_ERROR_RETURN(error);

  MOCHI_ERROR_IF(
      !MOCHI_USE_EIGEN && (params.linearSolver.solverType == LinearSolverType::AugmentedCG),
      error,
      "AugmentedCG solver is not supported in this build. To enable, include Eigen in your build setup and define MOCHI_USE_EIGEN=1");

  if (params.linearSolver.solverType != LinearSolverType::Auto &&
      details::IsCudaSolver(params.linearSolver.solverType)) {
    MOCHI_ERROR_IF_NOT(
        (MOCHI_USE_CUDA),
        error,
        "CUDA solvers require building with CUDA. To build with CUDA, include the CUDA dependencies in your build setup and define MOCHI_USE_CUDA=1");
    MOCHI_ERROR_IF_NOT(
        IsCudaAvailable(),
        error,
        "CUDA solvers require a CUDA device but no CUDA device was found.");
    if (details::IsIterativeSolver(params.linearSolver.solverType)) {
      MOCHI_ERROR_IF_NOT(
          params.linearSolver.preconditionerType == PreconditionerType::None ||
              params.linearSolver.preconditionerType == PreconditionerType::Jacobi ||
              params.linearSolver.preconditionerType == PreconditionerType::BlockJacobi,
          error,
          "CUDA iterative solvers only support None, Jacobi and BlockJacobi preconditioners.");
    }
  }

  MOCHI_ERROR_IF(
      params.linearSolver.maxIter != kAutoLinearSolverMaxIter && params.linearSolver.maxIter < 0,
      error,
      "Maximum number of linear solver iterations (LinearSolverParams::maxIter) must not be negative.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.linearSolver.absTol) && params.linearSolver.absTol >= 0_r,
      error,
      "Linear solver absolute tolerance (LinearSolverParams::absTol) must be non-negative and finite.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.linearSolver.relTol) && params.linearSolver.relTol >= 0_r,
      error,
      "Linear solver relative tolerance (LinearSolverParams::relTol) must be non-negative and finite.");

  MOCHI_ERROR_IF_NOT(
      params.nonLinearSolver.maxIter >= 1,
      error,
      "At least 1 non-linear solver iteration (NonLinearSolverParams::maxIter) is required.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.nonLinearSolver.absTol) && params.nonLinearSolver.absTol >= 0_r,
      error,
      "Non-linear solver absolute tolerance (NonLinearSolverParams::absTol) must be non-negative and finite.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.nonLinearSolver.relTol) && params.nonLinearSolver.relTol >= 0_r,
      error,
      "Non-linear solver relative tolerance (NonLinearSolverParams::relTol) must be non-negative and finite.");
  if (params.nonLinearSolver.lineSearchType != LineSearchType::None) {
    MOCHI_ERROR_IF_NOT(
        params.nonLinearSolver.lineSearchMaxIter >= 1,
        error,
        "At least 1 line search iteration (NonLinearSolverParams::lineSearchMaxIter) is required. To run without line search, use 'LineSearchType::None'.");
  }

  MOCHI_ERROR_IF(
      params.experimentalEval.implicitNormalForceForDissipation &&
          params.experimentalEval.explicitNormals,
      error,
      "implicitNormalForceForDissipation = true with explicitNormals = true is not supported.");
  MOCHI_ERROR_RETURN(error);

  // Store in CSimulationParams (global to our registry)
  SolverParams& storedParams = _registry.ctx<CSimulationParams>();
  storedParams = params;

  if (_registry.try_ctx<TagDifferentiableScene>() && !params.experimentalEval.explicitNormals) {
    MOCHI_LOG_WARNING(
        "Differentiable scenes require explicit normals for contact. Overriding input params.");
    storedParams.experimentalEval.explicitNormals = true;
  }
}

void SceneImpl::Step(double timeStepSec) {
  MOCHI_PROFILE_SCOPE();

  if (!(timeStepSec >= 0.0))
    MOCHI_UNLIKELY {
      MOCHI_LOG_ERROR(
          "The time-step size (%.3f s) is invalid. No step will be taken.", timeStepSec);
      return;
    }

  Timer totalStepTimer;

  auto& stepCounter = _registry.ctx<CSceneStepCounter>().value;
  stepCounter++;

  // Enforce scheduler binding to this thread, for parallel work.
  ScopedSchedulerBinding schedulerBinding(*_context);

  Timer timer;

  if (_recorder) {
    _recorder->OnStepBegin(timeStepSec, GetGravity());
  }

  TimeSpan recordingStepDuration = timer.GetElapsed();

  // Additional info for the profiler
  MOCHI_PROFILE_DESCRIPTION_F(
      "Step: %llu\nDT: %g (sec)", static_cast<unsigned long long>(stepCounter), timeStepSec);

  StepInfo stepInfo;
  stepInfo.scene = this;
  stepInfo.timeStepSec = timeStepSec;

  timer.Reset();

  // Fire pre-step callbacks one at a time
  {
    MOCHI_PROFILE_SCOPE_N("PreStepCallbacks");
    _preStepCallbacks.Call(stepInfo);
  }

  TimeSpan preStepDuration = timer.GetElapsed();
  _lastPerformanceStats.preStepCallbacksDurationSec = ToSeconds(preStepDuration);

  double stepDuration = 0.0;
  if (timeStepSec > 0.0) {
    timer.Reset();

    // Store time step
    _registry.ctx<CSceneTime>().Advance(timeStepSec);

    // ECS compute velocities and prepare actors
    PreStepEcs(_registry);

    // ECS (step actors):
    StepEcs(_registry);

    // ECS
    PostStepEcs(_registry);

    stepDuration = ToSeconds(timer.GetElapsed());

    ComputeAggregateSolverSceneStats(_registry, _lastSolverStats, _lastDebugStats);
  } else {
    // If the user steps the scene with zero time-step, then the simulation does not advance,
    // but we still update the queries. This gives the user a way to force queries to refresh.
    // For example: They might do this before the first real simulation step to gather data for
    // rendering, or they might do it after calling Scene::RestoreState to update rendering while
    // the scene is paused.
    UpdateAllActorQueries(_registry);

    // Clear the stats
    _lastSolverStats.maxNonLinearIters = 0;
    _lastSolverStats.residualNorm = 0.0;
    _lastSolverStats.maxLineSearchIters = 0;
    _lastSolverStats.convergenceStatus = ConvergenceStatus::None;
    _lastDebugStats.maxResidualNormRelativeError = 0.0;
  }

  _lastPerformanceStats.solveStepDurationSec = stepDuration;

  timer.Reset();

  // Fire post-step callbacks one at a time
  {
    MOCHI_PROFILE_SCOPE_N("PostStepCallbacks");
    _postStepCallbacks.Call(stepInfo);
  }

  // Check if a SceneDebugger needs to be cleaned up on this thread
  DynamicArray<std::shared_ptr<dbg::SceneDebugger>> debuggersToShutdown;
  _debugger.Mutate([&](auto& info) { debuggersToShutdown = std::move(info.pendingShutdown); });
  for (auto& ptr : debuggersToShutdown) {
    ptr->ShutdownOnSceneThread(this);
  }

  TimeSpan postStepDuration = timer.GetElapsed();
  _lastPerformanceStats.postStepCallbacksDurationSec = ToSeconds(postStepDuration);

  timer.Reset();

  if (_recorder) {
    _recorder->OnStepEnd();
  }

  recordingStepDuration += timer.GetElapsed();
  _lastPerformanceStats.recordingStepDurationSec = ToSeconds(recordingStepDuration);

  TimeSpan totalStepDuration = totalStepTimer.GetElapsed();
  _lastPerformanceStats.totalStepDurationSec = ToSeconds(totalStepDuration);

#if MOCHI_DEBUG
  // Log a warning if any ECS component types have been used with the entt::registry, which were not
  // registered up front via ecs::RegistryComponent.
  ecs::DetectUnregisteredComponents(_registry);
#endif // MOCHI_DEBUG

  MOCHI_PROFILE_END_FRAME();
}

double SceneImpl::GetLastTimeStep() const {
  return _registry.ctx<CSceneTime const>().DeltaTime();
}

double SceneImpl::GetTotalSimulationTime() const {
  return _registry.ctx<CSceneTime const>().StepEndTime();
}

SolverStats SceneImpl::GetSolverStats() const {
  return _lastSolverStats;
}

PerformanceStats SceneImpl::GetPerformanceStats() const {
  return _lastPerformanceStats;
}

bool SceneImpl::GetForceSingleIsland() const {
  return island::GetForceSingleIsland(_registry);
}

void SceneImpl::SetForceSingleIsland(bool forceSingleIsland) {
  island::SetForceSingleIsland(_registry, forceSingleIsland);
}

StateHandle SceneImpl::CaptureState(Error& error) {
  MOCHI_PROFILE_SCOPE();
  CheckStateCaptureSupported(_registry, error);
  MOCHI_ERROR_RETURN(error, {});

  // Recycle the previously released state buffer (if any)
  DynamicArray<uint8_t> stateBuffer(std::move(_stateBufferForRecycling));
  MOCHI_ASSERT_VERBOSE(stateBuffer.empty(), "Recycled buffer should have been cleared");

  // Repeated captures are often the same size. Start by reserving the size we used last time.
  stateBuffer.reserve(_prevStateBufferSize);

  // Capture state to stateBuffer
  capture::CaptureState(_registry, stateBuffer, error);
  MOCHI_ERROR_RETURN(error, {});

  // Remember the size for next time
  _prevStateBufferSize = stateBuffer.size();

  // Generate a new StateHandle. Some of the bits come from this scene's unique ID in order to
  // prevent the StateHandle from being applied to the wrong scene. This isn't bullet proof. It is
  // meant to catch mistakes.
  MOCHI_ASSERT_VERBOSE(
      _nextStateHandleValue + 1 < 0x0000FFFFFFFFFFFFull,
      "State handle values overflowed or got corrupted");
  StateHandle handle((_sceneId << 48) | (_nextStateHandleValue++));
  MOCHI_ASSERT_VERBOSE(handle.IsValid());
  MOCHI_ASSERT_VERBOSE(_capturedState.find(handle) == _capturedState.end(), "Handle not unique");

  // Store the StateHandle + buffer pair
  _capturedState[handle] = std::move(stateBuffer);
  return handle;
}

Span<uint8_t const> SceneImpl::FindState(StateHandle handle, Error& error) const {
  MOCHI_ERROR_RETURN(error, {});
  auto it = _capturedState.find(handle);
  if (it == _capturedState.end()) {
    MOCHI_ERROR_SET(error, "Invalid state handle");
    return {};
  }
  return it->second;
}

void SceneImpl::RestoreState(StateHandle handle, bool releaseImmediately, Error& error) {
  RestorePartialState(handle, releaseImmediately, {}, error);
}

void SceneImpl::CaptureStateToBytes(DynamicArray<uint8_t>& outData, Error& error) {
  MOCHI_PROFILE_SCOPE();
  CheckStateCaptureSupported(_registry, error);
  MOCHI_ERROR_RETURN(error);

  capture::CaptureState(_registry, outData, error);
}

void SceneImpl::RestoreStateFromBytes(Span<uint8_t const> data, Error& error) {
  MOCHI_PROFILE_SCOPE();
  capture::RestoreState(_registry, data, error);
}

void SceneImpl::RestorePartialState(
    StateHandle handle,
    bool releaseImmediately,
    Span<SReflect::TypeId const> excludedAttributes,
    Error& error) {
  MOCHI_PROFILE_SCOPE();

  // Look it up in the map
  auto const& stateBuffer = FindState(handle, error);

  // Restore state from stateBuffer
  capture::RestorePartialState(_registry, stateBuffer, excludedAttributes, error);

  // Optionally release the handle now (even if !error.IsOK())
  if (releaseImmediately) {
    ReleaseState(handle);
  }
}

void SceneImpl::ReleaseState(StateHandle handle) {
  // Find and remove. Ignore redundant attempts.
  auto it = _capturedState.find(handle);
  if (it != _capturedState.end()) {
    // Preserve the most recently released buffer for recycling.
    _stateBufferForRecycling = std::move(it->second);
    _stateBufferForRecycling.clear();
    _capturedState.erase(it);
  }
}

void SceneImpl::ReleaseAllStates() {
  _capturedState.clear();
  _stateBufferForRecycling.reset();
}

bool SceneImpl::IsEqualState(StateHandle a, StateHandle b) const {
  Error error;
  auto stateA = FindState(a, error);
  auto stateB = FindState(b, error);
  return error.IsOK() && capture::IsEqualState(_registry, stateA, stateB);
}

void SceneImpl::CaptureStateToFile(std::string_view filePath, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(filePath.empty(), error, "Empty file path");
  MOCHI_ERROR_IF(
      !_registry.storage<TagRomActor>().empty(),
      error,
      "CaptureStateToFile is not supported for scenes with ROM actors.");
  MOCHI_ERROR_RETURN(error);
  std::string json = capture::CaptureStateToJson(_registry, /*prettyMultiLine*/ true, error);
  WriteFile(filePath, json, error);
}

int SceneImpl::GetNumActors() const {
  return _numActors;
}

void SceneImpl::GetActors(Span<Actor*> outActors, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(outActors.size() != _numActors, error, "Output array must be the correct size");
  MOCHI_ERROR_RETURN(error);
  int index = 0;
  ForEachActor([&](Actor* actor) { outActors[index++] = actor; });
}

void SceneImpl::GetActors(Span<Actor const*> outActors, Error& error) const {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(outActors.size() != _numActors, error, "Output array must be the correct size");
  MOCHI_ERROR_RETURN(error);
  int index = 0;
  ForEachActor([&](Actor const* actor) { outActors[index++] = actor; });
}

void SceneImpl::ForEachActor(std::function<void(Actor*)> const& callback) {
  for (auto&& [e, api] : _registry.view<CActorInterface const>().each()) {
    callback(api.ptr.get());
  }
}

// Const overload
void SceneImpl::ForEachActor(std::function<void(Actor const*)> const& callback) const {
  for (auto&& [e, api] : _registry.view<CActorInterface const>().each()) {
    callback(api.ptr.get());
  }
}

DebugDraw& SceneImpl::GetDebugDraw() {
  return *_debugDraw;
}

DebugDraw const& SceneImpl::GetDebugDraw() const {
  return *_debugDraw;
}

void SceneImpl::UpdateDebugger() {
  std::shared_ptr<dbg::SceneDebugger> debugger;
  DynamicArray<std::shared_ptr<dbg::SceneDebugger>> debuggersToShutdown;
  _debugger.Mutate([&](auto& info) {
    debugger = info.debugger; // Copy the shared_ptr
    debuggersToShutdown = std::move(info.pendingShutdown);
  });

  // Cleanup any old debuggers that were waiting to finally shutdown on the scene's thread.
  for (auto& ptr : debuggersToShutdown) {
    ptr->ShutdownOnSceneThread(this);
  }

  // Then update the current debugger (if any)
  if (debugger) {
    debugger->UpdateOnSceneThread(this);
  }
}

bool SceneImpl::IsRecording() const {
  return _recorder != nullptr;
}

void SceneImpl::StartRecording(
    std::string_view filePath,
    RecordingParams const& params,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (IsRecording()) {
    StopRecording();
  }

  // Attempt to open an HDF5 file for writing
  auto writer = CreateGroupWriterHDF5(filePath, error);
  MOCHI_ERROR_RETURN(error);

  // Compression may reduce file size, but the impact on performance has not been evaluated.
  // For now, we always disable compression.
  writer->SetCompression(GroupWriter::kNoCompression);

  // Create a SceneRecorder
  auto recorder = std::make_unique<SceneRecorder>(std::move(writer), _registry, params);

  if (!recorder->IsOK()) {
    error = recorder->GetError().Copy();
    return;
  }

  // If we make it this far, then it should be OK to continue recording
  _recorder = std::move(recorder);
}

void SceneImpl::StopRecording() {
  _recorder.reset();
}

static void EnableLayerContactImpl(
    entt::registry& reg,
    std::string_view layerA,
    std::string_view layerB,
    bool enable,
    bool symmetric,
    Error& error) {
  MOCHI_ERROR_IF(layerA.empty() || layerB.empty(), error, "Layer names cannot be empty");
  MOCHI_ERROR_RETURN(error);

  auto& table = reg.ctx<CContactFilterTable>();
  auto a = GetOrAddContactLayerId(table, layerA);
  auto b = GetOrAddContactLayerId(table, layerB);
  table.EnableLayerContact(a, b, enable);
  if (symmetric) {
    table.EnableLayerContact(b, a, enable);
  }
}

void SceneImpl::EnableLayerContactAsymmetric(
    std::string_view layerA,
    std::string_view layerB,
    bool enable,
    Error& error) {
  EnableLayerContactImpl(_registry, layerA, layerB, enable, false /*symmetric*/, error);
}

void SceneImpl::EnableLayerContactSymmetric(
    std::string_view layerA,
    std::string_view layerB,
    bool enable,
    Error& error) {
  EnableLayerContactImpl(_registry, layerA, layerB, enable, true /*symmetric*/, error);
}

bool SceneImpl::IsLayerContactEnabled(std::string_view layerA, std::string_view layerB) const {
  auto const& table = _registry.ctx<CContactFilterTable const>();
  auto a = GetContactLayerId(table, layerA);
  auto b = GetContactLayerId(table, layerB);
  return table.IsLayerContactEnabled(a, b);
}

int SceneImpl::GetNumContactLayers() const {
  auto const& table = _registry.ctx<CContactFilterTable const>();
  return isize(table.layerNameToId);
}

void SceneImpl::EnumerateContactLayerNames(
    std::function<void(std::string_view name)> const& callback) const {
  auto const& config = _registry.ctx<CContactFilterTable const>();
  for (auto const& [name, id] : config.layerNameToId) {
    callback(name);
  }
}

void SceneImpl::EnableActorContactAsymmetric(
    ActorHandle colliding,
    ActorHandle collider,
    bool enable,
    IncludeNestedActors includeNestedActors,
    Error& error) {
  MOCHI_ERROR_IF_NOT(
      IsValidIncludeNestedActors(includeNestedActors), error, "Invalid IncludeNestedActors.");
  MOCHI_ERROR_RETURN(error);

  MOCHI_FILO_STACK_ALLOCATOR(collidingAllocator, 64 * sizeof(ActorHandle));
  MOCHI_FILO_STACK_ALLOCATOR(colliderAllocator, 64 * sizeof(ActorHandle));
  DynamicArray<ActorHandle> collidingHandles(&collidingAllocator);
  DynamicArray<ActorHandle> colliderHandles(&colliderAllocator);
  CollectActorContactHandles(
      _registry, *this, colliding, includeNestedActors, collidingHandles, error);
  MOCHI_ERROR_RETURN(error);
  CollectActorContactHandles(
      _registry, *this, collider, includeNestedActors, colliderHandles, error);
  MOCHI_ERROR_RETURN(error);

  auto& table = _registry.ctx<CContactFilterTable>();
  for (auto collidingHandle : collidingHandles) {
    auto eColliding = GetEntity(_registry, collidingHandle, error);
    MOCHI_ERROR_RETURN(error);
    for (auto colliderHandle : colliderHandles) {
      auto eCollider = GetEntity(_registry, colliderHandle, error);
      MOCHI_ERROR_RETURN(error);
      table.EnableEntityContact(eColliding, eCollider, enable);
    }
  }
}

void SceneImpl::EnableActorContactSymmetric(
    ActorHandle actorA,
    ActorHandle actorB,
    bool enable,
    IncludeNestedActors includeNestedActors,
    Error& error) {
  EnableActorContactAsymmetric(actorA, actorB, enable, includeNestedActors, error);
  MOCHI_ERROR_RETURN(error);
  if (actorA == actorB) {
    return;
  }
  EnableActorContactAsymmetric(actorB, actorA, enable, includeNestedActors, error);
}

void SceneImpl::SetContactPairParamsOverride(
    ActorHandle actorA,
    ActorHandle actorB,
    ContactPairParamsOverride const& paramsOverride,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  ValidateContactPairParamsOverride(paramsOverride, error);
  MOCHI_ERROR_RETURN(error);

  entt::entity const entityA = GetEntity(_registry, actorA, error);
  entt::entity const entityB = GetEntity(_registry, actorB, error);
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF_NOT(
      _registry.all_of<CContactParams>(entityA) && _registry.all_of<CContactParams>(entityB),
      error,
      "Both actors must have contact parameters.");
  MOCHI_ERROR_RETURN(error);

  _registry.ctx<CContactPairParamsOverrideTable>().Set(entityA, entityB, paramsOverride);
}

void SceneImpl::ClearContactPairParamsOverride(
    ActorHandle actorA,
    ActorHandle actorB,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  entt::entity const entityA = GetEntity(_registry, actorA, error);
  entt::entity const entityB = GetEntity(_registry, actorB, error);
  MOCHI_ERROR_RETURN(error);

  _registry.ctx<CContactPairParamsOverrideTable>().Clear(entityA, entityB);
}

bool SceneImpl::HasContactPairParamsOverride(ActorHandle actorA, ActorHandle actorB, Error& error)
    const {
  MOCHI_ERROR_RETURN(error, false);
  entt::entity const entityA = GetEntity(_registry, actorA, error);
  entt::entity const entityB = GetEntity(_registry, actorB, error);
  MOCHI_ERROR_RETURN(error, false);
  return _registry.ctx<CContactPairParamsOverrideTable const>().Find(entityA, entityB) != nullptr;
}

ContactPairParamsOverride SceneImpl::GetContactPairParamsOverride(
    ActorHandle actorA,
    ActorHandle actorB,
    Error& error) const {
  MOCHI_ERROR_RETURN(error, {});
  entt::entity const entityA = GetEntity(_registry, actorA, error);
  entt::entity const entityB = GetEntity(_registry, actorB, error);
  MOCHI_ERROR_RETURN(error, {});

  auto const* paramsOverride =
      _registry.ctx<CContactPairParamsOverrideTable const>().Find(entityA, entityB);
  MOCHI_ERROR_IF(paramsOverride == nullptr, error, "Contact pair has no parameter override.");
  MOCHI_ERROR_RETURN(error, {});
  return *paramsOverride;
}

void SceneImpl::RestoreStatePair(StateHandle curr, StateHandle prev, Error& err) {
  MOCHI_ERROR_RETURN(err);

  // Set the previous state, excluding adjoints.
  SReflect::TypeId excludeAdjoints[] = {attribute::HasAdjoint::GetTypeId()};
  RestorePartialState(prev, /*releaseImmediately*/ false, excludeAdjoints, err);
  MOCHI_ERROR_RETURN(err);

  // Advance time and pre-step ECS, to set TimeStep::Previous components
  auto timeStepSec = _registry.ctx<CSceneTime>().DeltaTime();
  _registry.ctx<CSceneTime>().Advance(timeStepSec);
  PreStepEcs(_registry);

  // Set the current state, excluding adjoints and old targets.
  SReflect::TypeId excludeAdjointsAndOldTargets[] = {
      attribute::HasAdjoint::GetTypeId(), attribute::HasOldTarget::GetTypeId()};
  RestorePartialState(curr, /*releaseImmediately*/ false, excludeAdjointsAndOldTargets, err);
}

bool SceneImpl::IsActorContactEnabled(ActorHandle colliding, ActorHandle collider, Error& error)
    const {
  auto eColliding = GetEntity(_registry, colliding, error);
  auto eCollider = GetEntity(_registry, collider, error);
  MOCHI_ERROR_RETURN(error, false);

  auto const& table = _registry.ctx<CContactFilterTable>();
  return table.IsEntityContactEnabled(eColliding, eCollider);
}

experimental::DebugStats SceneImpl::GetDebugStats() const {
  return _lastDebugStats;
}

void SceneImpl::ApplyImprovedConvergenceSettings(bool logWarnings) {
  // Set ContactParams.frictionWithColliderNormal = true.
  // Possibly log warnings if values were non-default.
  ForEachActor([&](Actor* actor) {
    Error error;
    ContactParams contactParams = actor->GetContactParams(error);
    if (error.IsOK() && !contactParams.frictionWithColliderNormal) {
      if (logWarnings) {
        MOCHI_LOG_WARNING_ONCE(
            "\nOverriding ContactParams.frictionWithColliderNormal = false. Setting to true.");
      }
      contactParams.frictionWithColliderNormal = true;
      actor->SetContactParams(contactParams, ErrorAssert{});
    }
  });

  // Set EnableNewtonEulerInertia(false).
  // Possibly log warning if value was non-default.
  ForEachActor([&](Actor* actor) {
    Error error;
    bool const hasNewtonEulerInertia = IsNewtonEulerInertiaEnabled(actor, error);
    if (error.IsOK() && hasNewtonEulerInertia) {
      if (logWarnings) {
        MOCHI_LOG_WARNING_ONCE("\nOverriding EnableNewtonEulerInertia(). Setting to false.");
      }
      EnableNewtonEulerInertia(actor, false, ErrorAssert{});
    }
  });

  // Set SolverParams.experimentalEval.implicitNormalForceForDissipation = false.
  // Possibly log warnings if value was non-default.
  auto solverParams = GetSolverParams();
  if (logWarnings && solverParams.experimentalEval.implicitNormalForceForDissipation) {
    MOCHI_LOG_WARNING_ONCE(
        "\nOverriding SolverParams.experimentalEval.implicitNormalForceForDissipation = true. Setting to false");
  }
  solverParams.experimentalEval.implicitNormalForceForDissipation = false;

  // Set SolverParams.experimentalEval.explicitNormals = true.
  // Do not log warning; this is a non-default value.
  solverParams.experimentalEval.explicitNormals = true;

  // Set SolverParams.nonlinearSolver.lineSearchType = LineSearchType::Armijo.
  // Do not log warning; this is a non-default value.
  solverParams.nonLinearSolver.lineSearchType = LineSearchType::Armijo;

  // Set at least one line search iteration (required by LineSearchType::Armijo).
  solverParams.nonLinearSolver.lineSearchMaxIter =
      Max(1, solverParams.nonLinearSolver.lineSearchMaxIter);

  SetSolverParams(solverParams, ErrorAssert{});
}

void SceneImpl::WarnIfNotImprovedConvergenceSettings() const {
  // Validate ContactParams.frictionWithColliderNormal = true.
  ForEachActor([&](Actor const* actor) {
    Error error;
    ContactParams contactParams = actor->GetContactParams(error);
    if (error.IsOK() && !contactParams.frictionWithColliderNormal) {
      MOCHI_LOG_WARNING_ONCE("\nExpected ContactParams.frictionWithColliderNormal = true.");
    }
  });

  // Validate EnableNewtonEulerInertia(false).
  ForEachActor([&](Actor const* actor) {
    Error error;
    bool const hasNewtonEulerInertia = IsNewtonEulerInertiaEnabled(actor, error);
    if (error.IsOK() && hasNewtonEulerInertia) {
      MOCHI_LOG_WARNING_ONCE("\nExpected EnableNewtonEulerInertia(false).");
    }
  });

  // Validate SolverParams.experimentalEval.implicitNormalForceForDissipation = false.
  auto solverParams = GetSolverParams();
  if (solverParams.experimentalEval.implicitNormalForceForDissipation) {
    MOCHI_LOG_WARNING_ONCE(
        "\nExpected SolverParams.experimentalEval.implicitNormalForceForDissipation = false.");
  }

  // Validate SolverParams.experimentalEval.explicitNormals = true.
  if (!solverParams.experimentalEval.explicitNormals) {
    MOCHI_LOG_WARNING_ONCE("\nExpected SolverParams.experimentalEval.explicitNormals = true.");
  }

  // Validate SolverParams.nonlinearSolver.lineSearchType = LineSearchType::Armijo.
  if (solverParams.nonLinearSolver.lineSearchType != LineSearchType::Armijo) {
    MOCHI_LOG_WARNING_ONCE(
        "\nExpected SolverParams.nonLinearSolver.lineSearchType = LineSearchType::Armijo.");
  }
}

void SceneImpl::ResetBackPropagation() {
  ecs::InvokeForEachGlobal(&ResetBackPropagationContainers, _registry);
  _registry.unset<TagBackPropagationPrepared>();
}

void SceneImpl::PrepareBackPropagate(StateHandle stateNew, StateHandle stateOld, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_PROFILE_SCOPE();

  MOCHI_ERROR_IF(
      _registry.try_ctx<TagBackPropagationPrepared>(),
      error,
      "PrepareBackPropagate was called twice.");
  FindState(stateNew, error);
  FindState(stateOld, error);
  MOCHI_ERROR_RETURN(error);

  // Enforce scheduler binding to this thread, for parallel work.
  ScopedSchedulerBinding schedulerBinding(*_context);

  RestoreStatePair(stateNew, stateOld, error);
  MOCHI_ERROR_RETURN(error);

  _registry.ctx<CStatePair>() = CStatePair{.stateNew = stateNew, .stateOld = stateOld};

  PrepareBackPropagation(_registry);
  _registry.set<TagBackPropagationPrepared>();
}

void SceneImpl::BackPropagate(Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_PROFILE_SCOPE();

  Timer totalBackPropTimer;

  MOCHI_ERROR_IF_NOT(
      _registry.try_ctx<TagBackPropagationPrepared>(),
      error,
      "PrepareBackPropagate must be called before BackPropagate.");
  MOCHI_ERROR_RETURN(error);
  MOCHI_DEFER({ _registry.unset<TagBackPropagationPrepared>(); });

  // Validate that the old state is still available
  StateHandle stateOld = _registry.ctx<CStatePair>().stateOld;
  FindState(stateOld, error);
  MOCHI_ERROR_RETURN(error);

  // Differentiable actors. Currently only rigid and articulated actors are supported.
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 100 * sizeof(entt::entity)); // Stack mem for 100 actors
  DynamicArray<entt::entity> actors(&allocator);
  actors.reserve(GetNumActors()); // Conservative
  bool allActorsValid = true;
  ForEachActor([&](Actor* actor) {
    allActorsValid &=
        (actor->GetType() == ActorType::Rigid || actor->GetType() == ActorType::Articulated);
    if (!allActorsValid) {
      return;
    }
    if (!actor->IsStatic() && !actor->IsNestedLinkActor()) {
      actors.push_back(GetEntity(_registry, actor->GetHandle(), ErrorAssert{}));
    }
  });
  MOCHI_ERROR_IF(!allActorsValid, error, "All actors must be rigid or articulated");
  MOCHI_ERROR_RETURN(error);

  // Enforce scheduler binding to this thread, for parallel work.
  ScopedSchedulerBinding schedulerBinding(*_context);

  // Validate necessary solver settings for accurate differentiability
  WarnIfNotImprovedConvergenceSettings();

  // Process the rigid current and previous contact force adjoints.
  AccumulateContactForceAdjoints(_registry);

  // Add to the pose adjoint the projected derived-step adjoint. Store in CDiffContainerState.
  // λq_k += λΔx_k * ∂Δx_k/∂x_k * ∂x_k/∂q_k; with ∂Δx_k/∂x_k = I
  // Also add the contact-force adjoints wrt the current state.
  ParallelForEach("PrepareBackPropagationSolve", actors, 1, [&](entt::entity e) {
    AsView(_registry.get<CDiffContainerDerivedState>(e)) =
        _registry.get<CDiffDerivedStepGrad const>(e).value;

    ecs::TryInvokeOnEntity(rigid::ProjectDerivedStateGradient, _registry, e);
    ecs::TryInvokeOnEntity(articulated::compound::ProjectDerivedStateGradient, _registry, e);
    ecs::TryInvokeOnEntity<ecs::policy::AllowReadWriteSameComponent>(
        articulated::compound::ProjectContactForceAdjoints<GradTarget::Current>, _registry, e);

    auto container = AsView(_registry.get<CDiffContainerState>(e));
    container += _registry.get<CDiffStateGrad const>(e).value;
    container += _registry.get<CDiffContactGrad<GradTarget::Current> const>(e);
  });

  // Perform back-propagation solve (linear solve for z and gradient assembly):
  // - CDiffContainerState stores z * ∂r_k/∂q_{k-1}
  // - CDiffContainerDerivedState stores z * ∂r_k/∂Δx_{k-1}
  Timer timer;
  BackPropagationSolve(_registry);
  auto& backPropStats = _registry.ctx<CBackPropagationSceneStats>();
  backPropStats.solveDurationSec = ToSeconds(timer.GetElapsed());

  // Update the adjoints from the solve results.
  // λΔx_{k-1} = z * ∂r_k/∂Δx_{k-1}
  // λq_{k-1} = z * ∂r_k/∂q_{k-1} + λΔx_k * ∂Δx_k/∂x_{k-1} * ∂x_{k-1}/∂q_{k-1}
  // The operation λΔx_k * ∂Δx_k/∂x_{k-1} * ∂x_{k-1}/∂q_{k-1} requires 3 steps:
  // 1. Shift λΔx_k ⇐ λΔx_k * ∂Δx_k/∂x_{k-1}
  ParallelForEach("UpdateAdjointsStep1", actors, 1, [&](entt::entity e) {
    auto derivedStateContainer = AsView(_registry.get<CDiffContainerDerivedState>(e));
    auto gradDerivedStep = AsView(_registry.get<CDiffDerivedStepGrad>(e).value);
    // Swap: container gets the accumulated adjoint (for shift), accumulated gets the solve result.
    MOCHI_FILO_STACK_ALLOCATOR(allocator, 256 * sizeof(real));
    ColumnVector<real> temp(derivedStateContainer, &allocator);
    derivedStateContainer = gradDerivedStep;
    gradDerivedStep = temp;

    ecs::TryInvokeOnEntity(rigid::ShiftDerivedStateGradient, _registry, e);
    ecs::TryInvokeOnEntity(articulated::compound::ShiftDerivedStateGradient, _registry, e);

    _registry.get<CDiffStateGrad>(e).value = _registry.get<CDiffContainerState const>(e);
  });

  // 2. Restore state q_{k-1} (to get the correct Jacobian for projection), but not adjoints.
  SReflect::TypeId excludeAdjoints[] = {attribute::HasAdjoint::GetTypeId()};
  RestorePartialState(stateOld, /*releaseImmediately*/ false, excludeAdjoints, error);
  MOCHI_ERROR_RETURN(error);

  // 3. Project and add λq_{k-1} += λΔx_k * ∂x_{k-1}/∂q_{k-1}
  // Also add the contact-force adjoints wrt the previous state.
  ParallelForEach("UpdateAdjointsStep3", actors, 1, [&](entt::entity e) {
    ecs::TryInvokeOnEntity(rigid::ProjectDerivedStateGradient, _registry, e);
    ecs::TryInvokeOnEntity(articulated::compound::ProjectDerivedStateGradient, _registry, e);
    ecs::TryInvokeOnEntity<ecs::policy::AllowReadWriteSameComponent>(
        articulated::compound::ProjectContactForceAdjoints<GradTarget::Previous>, _registry, e);

    auto stateGrad = AsView(_registry.get<CDiffStateGrad>(e).value);
    stateGrad += _registry.get<CDiffContainerState const>(e);
    stateGrad += _registry.get<CDiffContactGrad<GradTarget::Previous> const>(e);
  });

  backPropStats.totalDurationSec = ToSeconds(totalBackPropTimer.GetElapsed());
  ComputeAggregateBackPropSolverSceneStats(_registry, backPropStats);
}

void SceneImpl::GetStepJacobian(
    StateHandle stateNew,
    StateHandle stateCurr,
    StateHandle stateOld,
    Span<real> outJacCurr,
    Span<real> outJacOld,
    Error& error) {
  MOCHI_ERROR_RETURN(error);

  MOCHI_PROFILE_SCOPE();

  // Check the state handles
  FindState(stateNew, error);
  FindState(stateCurr, error);
  FindState(stateOld, error);
  MOCHI_ERROR_RETURN(error);

  // Check Jacobian sizes
  int sceneDofs = 0;
  ForEachActor([&sceneDofs](Actor* actor) { sceneDofs += actor->GetNumDofs(); });
  MOCHI_ERROR_IF(isize(outJacCurr) != sceneDofs * sceneDofs, error, "Invalid Jacobian size");
  MOCHI_ERROR_IF(isize(outJacOld) != sceneDofs * sceneDofs, error, "Invalid Jacobian size");
  MOCHI_ERROR_RETURN(error);

  // Check if all actors are valid. Currently only rigid actors are supported.
  bool allActorsValid = true;
  ForEachActor([&](Actor* actor) { allActorsValid &= actor->GetType() == ActorType::Rigid; });
  MOCHI_ERROR_IF(!allActorsValid, error, "All actors must be rigid");
  MOCHI_ERROR_RETURN(error);

  // Check time integrator is backward Euler.
  MOCHI_ERROR_IF(
      _registry.ctx<CSimulationParams const>().integrationMethod !=
          IntegrationMethod::BackwardEuler,
      error,
      "GetStepJacobian is only supported with backward Euler.");

  // Validate necessary solver settings for accurate differentiability
  WarnIfNotImprovedConvergenceSettings();

  // Enforce scheduler binding to this thread, for parallel work.
  ScopedSchedulerBinding schedulerBinding(*_context);

  // Compute scene state offset
  int dofOffset = 0;
  ForEachActor([&](Actor* actor) {
    MOCHI_ASSERT(actor->GetType() == ActorType::Rigid, "GetStepJacobian only supports rigid actors")
    MOCHI_ASSERT(!actor->IsNestedLinkActor(), "GetStepJacobian does not support articulated body")
    if (!actor->IsStatic()) {
      int const numDofs = actor->GetNumDofs();
      auto e = GetEntity(_registry, actor->GetHandle(), ErrorAssert{});
      _registry.get<CSceneStateOffset>(e).dofsOffset = dofOffset;
      dofOffset += numDofs;
    }
  });

  // Set the new state (with the current state as previous).
  RestoreStatePair(stateNew, stateCurr, error);
  MOCHI_ERROR_RETURN(error);

  // Compute dq_t/dq_t-1 and dq_t/dDx
  // Store dq_t/dq_t-1 in jacCurrView
  // Store dq_t/dDx_t-1 in per-actor CForwardPropContainerDerivedState
  MatrixView<real> jacCurrView(outJacCurr.data(), sceneDofs, sceneDofs);
  StepJacobianSolve(_registry, jacCurrView);

  // Set the current state (with the old state as previous).
  RestoreStatePair(stateCurr, stateOld, error);
  MOCHI_ERROR_RETURN(error);

  // Accumulate dq_t/dq_t-1 += dq_t/dδ * dδ/dx_t-1 in jacCurrView
  // Compute dq_t/dq_t-2 = dq_t/dδ dδ/dx_t-2 in jacOldView
  MatrixView<real> jacOldView(outJacOld.data(), sceneDofs, sceneDofs);
  StepJacobianShiftAndProject(_registry, jacCurrView, jacOldView);
}

#define MOCHI_DESTROY_AND_RETURN_IF_ERROR()         \
  if (!error.IsOK()) {                              \
    if (_registry.all_of<TagArticulatedActor>(e)) { \
      DestroyActorEntity(*this, _registry, e);      \
    } else {                                        \
      _registry.destroy(e);                         \
    }                                               \
  }                                                 \
  MOCHI_ERROR_RETURN(error, {});

Actor* SceneImpl::CreateRigidActorImpl(
    RigidActorParams const& params,
    bool isArticulatedLink,
    bool useContact,
    std::shared_ptr<Shape const> shapePtr,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF(
      (isArticulatedLink || params.isStatic) && params.linearVelocity,
      error,
      "Static actors and articulated links do not allow initial velocity");
  MOCHI_ERROR_IF(
      (isArticulatedLink || params.isStatic) && params.angularVelocity,
      error,
      "Static actors and articulated links do not allow initial velocity");
  MOCHI_ERROR_RETURN(error, {});

  // Create an ECS entity
  entt::entity e = _registry.create();
  ActorHandle newHandle = GetActorHandle(e, GetHandle());

  // Create an implementation of mochi::Actor to satisfy the public interface
  _registry.emplace<CActorInterface>(e, CreateActorInterface(_registry, e, this));

  // Add all the other components without the help of the virtual API pointer
  InitRigidActor(_registry, e, params, useContact, shapePtr, error);
  MOCHI_DESTROY_AND_RETURN_IF_ERROR();

  // Initialization specific to a differentiable scene
  if (!params.isStatic && _registry.try_ctx<TagDifferentiableScene>()) {
    InitDifferentiableRigidActor(_registry, e, isArticulatedLink);
  }

  ++_numActors;
  if (!params.isStatic) {
    // Must be in an island to simulate
    island::CreateForActor(_registry, e);
  }
  _registry.emplace<TagFullyInitialized>(e);
  ValidateNewActorComposition(e);
  return GetActor(newHandle);
}

Actor* SceneImpl::CreateRigidActor(RigidActorParams const& params, Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  // Get the shape
  auto shapePtr = _context->GetShapeSharedPtr(params.shape);
  MOCHI_ERROR_IF_NOT(shapePtr, error, "Invalid ShapeHandle");
  MOCHI_ERROR_RETURN(error, {});

  return CreateRigidActorImpl(
      params, false /*isArticulatedLink*/, true /*useContact*/, shapePtr, error);
}

Actor* SceneImpl::CreateSoftActorImpl(
    SoftActorParams const& params,
    ExperimentalSoftActorParams const& experimentalParams,
    bool isNestedSoft,
    std::shared_ptr<TetrahedralMeshShape const> shapePtr,
    Error& error) {
  MOCHI_ERROR_IF(
      !MOCHI_ENABLE_ROM_ACTORS && experimentalParams.rom.has_value(),
      error,
      "ROM actor creation is not supported in this build. To enable, define MOCHI_ENABLE_ROM_ACTORS=1");
  MOCHI_ERROR_IF(
      !MOCHI_ENABLE_DEEP_FLOW_ACTORS && experimentalParams.flow.IsValid(),
      error,
      "Deep Flow actor creation is not supported in this build. To enable, define MOCHI_ENABLE_DEEP_FLOW_ACTORS=1");
  MOCHI_ERROR_RETURN(error, {});

  // Create an ECS entity
  entt::entity e = _registry.create();
  ActorHandle newHandle = GetActorHandle(e, GetHandle());

  // Initialize it as a soft actor
  bool const useContact = !isNestedSoft;
  std::shared_ptr<DeepFlowShape const> flow = std::dynamic_pointer_cast<DeepFlowShape const>(
      _context->GetShapeSharedPtr(experimentalParams.flow));
  InitSoftActor(
      _registry, e, params, experimentalParams, useContact, isNestedSoft, shapePtr, flow, error);
  MOCHI_DESTROY_AND_RETURN_IF_ERROR();

  // If necessary, initialize ROM actor
  if (experimentalParams.rom) {
    bool const hasExternalRigidDofs = isNestedSoft;
    rom::InitSoftActorRom(
        _registry, e, *experimentalParams.rom, shapePtr, flow, hasExternalRigidDofs, error);
    MOCHI_DESTROY_AND_RETURN_IF_ERROR();
  }

  // Create an implementation of mochi::Actor to satisfy the public interface
  _registry.emplace<CActorInterface>(e, CreateActorInterface(_registry, e, this));

  island::CreateForActor(_registry, e); // Must be in an island to simulate
  _registry.emplace<TagFullyInitialized>(e);
  ValidateNewActorComposition(e);
  ++_numActors;
  return GetActor(newHandle);
}

Actor* SceneImpl::CreateShellActorImpl(
    ShellActorParams const& params,
    std::shared_ptr<TriangularMeshShape const> shapePtr,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {})

  // Create an ECS entity
  entt::entity e = _registry.create();
  ActorHandle newHandle = GetActorHandle(e, GetHandle());

  InitShellActor(_registry, e, params, shapePtr, error);
  MOCHI_DESTROY_AND_RETURN_IF_ERROR();

  // Create an implementation of mochi::Actor to satisfy the public interface
  _registry.emplace<CActorInterface>(e, CreateActorInterface(_registry, e, this));

  island::CreateForActor(_registry, e); // Must be in an island to simulate
  _registry.emplace<TagFullyInitialized>(e);
  ValidateNewActorComposition(e);
  ++_numActors;
  return GetActor(newHandle);
}

Actor* SceneImpl::CreateRodActorImpl(
    RodActorParams const& params,
    std::shared_ptr<PolylineShape const> shapePtr,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {})

  entt::entity e = _registry.create();
  SceneHandle sceneHandle = GetHandle();
  ActorHandle newHandle = GetActorHandle(e, sceneHandle);

  InitRodActor(_registry, e, params, shapePtr, error);
  MOCHI_DESTROY_AND_RETURN_IF_ERROR();

  _registry.emplace<CActorInterface>(e, CreateActorInterface(_registry, e, this));

  island::CreateForActor(_registry, e);
  _registry.emplace<TagFullyInitialized>(e);
  ValidateNewActorComposition(e);
  ++_numActors;
  return GetActor(newHandle);
}

Actor* SceneImpl::CreateSoftActor(SoftActorParams const& params, Error& error) {
  // Use the experimental API as the common code path.
  return experimental::CreateSoftActor(this, params, {}, error);
}

// Experimental API
MOCHI_API Actor* mochi::experimental::CreateSoftActor(
    Scene* scene,
    SoftActorParams const& params,
    ExperimentalSoftActorParams const& experimentalParams,
    Error& error) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ERROR_IF(!scene, error, "Invalid scene");
  MOCHI_ERROR_RETURN(error, {});
  auto* sceneImpl = assert_cast<SceneImpl*>(scene);
  auto* contextImpl = assert_cast<ContextImpl*>(sceneImpl->GetContext());

  // A tetrahedral mesh is required
  auto shapePtr = std::dynamic_pointer_cast<TetrahedralMeshShape const>(
      contextImpl->GetShapeSharedPtr(params.shape));
  MOCHI_ERROR_IF_NOT(shapePtr, error, "Cannot create actor. Requires a TetrahedralMeshShape.");
  MOCHI_ERROR_RETURN(error, {});

  return sceneImpl->CreateSoftActorImpl(
      params, experimentalParams, /* isNestedSoft */ false, shapePtr, error);
}

// Store skin params that are consumed during InitSkinMesh and cannot be recovered afterward.
static void StoreSkinDataForExport(
    entt::registry& reg,
    Actor const* actor,
    std::optional<ArticulatedSkinParams> const& params) {
  if (!params) {
    return;
  }
  auto e = GetEntity(reg, actor->GetHandle(), ErrorAssert{});
  auto& skinExport = reg.emplace<CArticulatedSkinExportParams>(e);
  skinExport.boundaryElementType = params->boundaryElementType;
  skinExport.boundarySubsampling = params->boundarySubsampling;
}

// If the user did not provide a parent actor name, then pretend it is "unnamed_articulation", so
// that all link names follow the same "parent_name/link_name" convention.
[[nodiscard]] static std::string GetNestedActorParentName(std::string_view parentActorName) {
  return std::string(parentActorName.empty() ? "unnamed_articulation" : parentActorName);
}

static void ReserveNestedActorLocalName(
    std::unordered_set<std::string>& usedLocalNames,
    std::string_view name,
    Error& error) {
  MOCHI_ERROR_IF(name.empty(), error, "Nested actor local name must be non-empty.");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      HasInvalidNestedActorNameCharacter(name),
      error,
      "Nested actor local name must not contain '/', '\\', or embedded NUL characters.");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      !usedLocalNames.insert(std::string(name)).second,
      error,
      "Duplicate nested actor local name.");
}

static void AutoCorrectNestedSoftActorNames(SoftSkinnedActorParams& params) {
  std::unordered_set<std::string> usedLocalNames;
  for (auto const& link : params.skeletonParams.links) {
    MOCHI_ASSERT(!link.name.empty(), "Expected skeleton link names to be autocorrected first.");
    usedLocalNames.insert(std::string(link.name));
  }

  for (auto const& softParams : params.softParams) {
    std::string_view const softName = softParams.name;
    if (softName.empty()) {
      continue;
    }
    usedLocalNames.insert(std::string(softName));
  }

  for (int i = 0; i < isize(params.softParams); ++i) {
    auto& name = params.softParams[i].name;
    if (!name.empty()) {
      continue;
    }

    std::string candidate = Format("soft_%d", i);
    if (usedLocalNames.count(candidate) != 0) {
      for (int n = 0;; ++n) {
        candidate = Format("soft_%d", n);
        if (usedLocalNames.count(candidate) == 0) {
          break;
        }
      }
    }
    name = candidate;
    usedLocalNames.insert(std::move(candidate));
  }
}

static void ValidateSoftSkinnedNestedActorNamesAndAttachLinks(
    SoftSkinnedActorParams const& params,
    Error& error) {
  MOCHI_ERROR_RETURN(error);

  std::unordered_set<std::string> usedLocalNames;
  std::unordered_set<std::string> linkLocalNames;
  for (auto const& link : params.skeletonParams.links) {
    std::string_view const linkName = link.name;
    ReserveNestedActorLocalName(usedLocalNames, linkName, error);
    MOCHI_ERROR_RETURN(error);
    linkLocalNames.insert(std::string(linkName));
  }

  for (auto const& softParams : params.softParams) {
    ReserveNestedActorLocalName(usedLocalNames, softParams.name, error);
    MOCHI_ERROR_RETURN(error);
  }

  if (!params.softAttachLinks.empty()) {
    MOCHI_ERROR_IF(
        isize(params.softAttachLinks) != isize(params.softParams),
        error,
        "SoftSkinnedActorParams::softAttachLinks must be empty or 1-to-1 with softParams.");
    MOCHI_ERROR_RETURN(error);
    for (auto const& softAttachLink : params.softAttachLinks) {
      std::string_view const linkName = softAttachLink;
      MOCHI_ERROR_IF(
          linkName.empty(),
          error,
          "SoftSkinnedActorParams::softAttachLinks entries must be non-empty.");
      MOCHI_ERROR_IF(
          HasInvalidNestedActorNameCharacter(linkName),
          error,
          "SoftSkinnedActorParams::softAttachLinks entries must not contain '/', '\\', or embedded NUL characters.");
      MOCHI_ERROR_IF(
          linkLocalNames.count(std::string(linkName)) == 0,
          error,
          "SoftSkinnedActorParams::softAttachLinks entry does not match any skeleton link local name.");
      MOCHI_ERROR_RETURN(error);
    }
  }
}

static void ValidateSoftSkinnedActorEnergyParams(
    SoftSkinnedActorParams const& params,
    Error& error) {
  MOCHI_ERROR_RETURN(error);

  for (auto const& softParams : params.softParams) {
    MOCHI_ERROR_IF(softParams.hasGravity, error, "Must disable gravity for nested soft actors.");
    MOCHI_ERROR_IF_NOT(
        params.hasGravity || params.hasInertia || params.hasStress || softParams.hasInertia ||
            softParams.hasStress,
        error,
        "Each nested soft actor must have at least one energy term enabled: gravity, "
        "inertia or stress on the SoftSkinnedActorParams (posed), or inertia or stress on its "
        "SoftActorParams (unposed).");
  }
}

static void ValidateAndAutoCorrect(SoftSkinnedActorParams& params, Error& error) {
  MOCHI_ERROR_RETURN(error);

  // Auto-correct before validating: name validation depends on the auto-assigned/corrected skeleton
  // link and nested soft actor names.
  AutoCorrect(params.skeletonParams, error);
  MOCHI_ERROR_RETURN(error);
  AutoCorrectNestedSoftActorNames(params);

  Validate(params.skeletonParams, error);
  MOCHI_ERROR_RETURN(error);
  ValidateSoftSkinnedNestedActorNamesAndAttachLinks(params, error);
  MOCHI_ERROR_RETURN(error);
  ValidateSoftSkinnedActorEnergyParams(params, error);
  MOCHI_ERROR_RETURN(error);

  // Nested soft actor shapes must be defined directly in the articulated actor's reference frame,
  // so their worldFromLocal must be identity: a non-identity transform would introduce a spurious
  // offset that breaks attachment. Reject meaningfully non-identity transforms, then snap the rest
  // to exact identity so downstream skinning can rely on exact equality.
  for (auto& softParams : params.softParams) {
    MOCHI_ERROR_IF(
        !NearEqual(softParams.worldFromLocal, TransformRT::Identity()),
        error,
        "SoftActorParams::worldFromLocal must be identity for nested soft actors of a "
        "soft-skinned actor; the soft actor's shape must be defined directly in the articulated "
        "actor's reference frame.");
    MOCHI_ERROR_RETURN(error);
    softParams.worldFromLocal = TransformRT::Identity();
  }
}

// Experimental API
MOCHI_API Actor* mochi::experimental::CreateSoftSkinnedActor(
    Scene* scene,
    SoftSkinnedActorParams const& params,
    experimental::ExperimentalSoftSkinnedActorParams const& experimentalParams,
    Error& error) {
  MOCHI_ERROR_IF(!scene, error, "Null scene pointer.");
  MOCHI_ERROR_RETURN(error, nullptr);

  auto paramsCopy = params;
  ValidateAndAutoCorrect(paramsCopy, error);
  MOCHI_ERROR_RETURN(error, nullptr);

  // Build the articulated skeleton shape from the params.
  auto articulatedShapePtr = GetArticulatedShape(paramsCopy.skeletonParams, error);
  MOCHI_ERROR_RETURN(error, nullptr);

  auto* sceneImpl = assert_cast<SceneImpl*>(scene);
  Actor* newActor = sceneImpl->CreateSoftSkinnedActorImpl(
      paramsCopy, experimentalParams, std::move(articulatedShapePtr), error);
  MOCHI_ERROR_RETURN(error, nullptr);

  StoreSkinDataForExport(sceneImpl->GetRegistry(), newActor, paramsCopy.skeletonParams.skin);

  return newActor;
}

// Experimental API
MOCHI_API Actor*
experimental::CreateShellActor(Scene* scene, ShellActorParams const& params, Error& error) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ERROR_IF(!scene, error, "Invalid scene");
  MOCHI_ERROR_RETURN(error, {});
  auto* context = assert_cast<ContextImpl*>(scene->GetContext());

  // A tri mesh is required
  auto shapePtr = std::dynamic_pointer_cast<TriangularMeshShape const>(
      context->GetShapeSharedPtr(params.shape));
  MOCHI_ERROR_IF_NOT(shapePtr, error, "Cannot create actor. Requires a TriangularMeshShape.");
  MOCHI_ERROR_RETURN(error, {});

  return assert_cast<SceneImpl*>(scene)->CreateShellActorImpl(params, shapePtr, error);
}

Actor* SceneImpl::CreateSoftSkinnedActor(SoftSkinnedActorParams const& params, Error& error) {
  // Share code with the experimental API
  return experimental::CreateSoftSkinnedActor(this, params, {}, error);
}

MOCHI_API Actor*
experimental::CreateRodActor(Scene* scene, RodActorParams const& params, Error& error) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ERROR_IF(!scene, error, "Invalid scene");
  MOCHI_ERROR_RETURN(error, {});
  auto* context = assert_cast<ContextImpl*>(scene->GetContext());

  // A polyline mesh is required
  auto shapePtr =
      std::dynamic_pointer_cast<PolylineShape const>(context->GetShapeSharedPtr(params.shape));
  MOCHI_ERROR_IF_NOT(shapePtr, error, "Cannot create actor. Requires a PolylineShape.");
  MOCHI_ERROR_RETURN(error, {});

  return assert_cast<SceneImpl*>(scene)->CreateRodActorImpl(params, shapePtr, error);
}

static std::vector<bool> IdentifyStaticLinks(ArticulatedBodyShape const* shape) {
  MOCHI_ASSERT(shape != nullptr, "Invalid shape");
  auto const& jointTypes = shape->GetJointsData()->jointTypes;
  auto const& parents = *shape->GetBoneParents();
  std::vector<bool> isStatic(parents.size());
  for (int i = 0; i < parents.size(); ++i) {
    // A link is static if its joint is hard and it has no parent or its parent is static
    bool isStaticParent = parents[i] == -1 ? true : isStatic[parents[i]];
    isStatic[i] = isStaticParent && jointTypes[i] == ArticulatedJointType::Hard;
  }
  return isStatic;
}

namespace {
struct GraphNode {
  ActorHandle actor{};
  bool hasShape{};
  DynamicArray<int> edges{};
};

struct GraphEdge {
  int nodeA{};
  int nodeB{};
  bool isHard{};
};
} // namespace

static void CreateGraphOfLinks(
    Span<ActorHandle const> links,
    ArticulatedBodyShape const* shapePtr,
    Span<ShapeHandle const> linkShapes,
    DynamicArray<GraphNode>& outNodes,
    DynamicArray<GraphEdge>& outEdges) {
  MOCHI_ASSERT_VERBOSE(
      isize(links) == isize(linkShapes), "Expected one link shape per link actor.");
  outNodes.reserve(links.size());
  for (int i = 0; i < isize(links); ++i) {
    outNodes.emplace_back(GraphNode{.actor = links[i], .hasShape = linkShapes[i].IsValid()});
  }

  auto const& children = shapePtr->GetJointsData()->jointsChildLinks;
  auto const& parents = shapePtr->GetJointsData()->jointsParentLinks;
  auto const& types = shapePtr->GetJointsData()->jointTypes;
  outEdges.reserve(children.size());
  for (auto i = 0; i < isize(children); ++i) {
    if (parents[i] >= 0) {
      int const nodeA = children[i];
      int const nodeB = parents[i];
      outNodes[nodeA].edges.push_back(isize(outEdges));
      outNodes[nodeB].edges.push_back(isize(outEdges));
      bool isHard = types[i] == ArticulatedJointType::Hard;
      outEdges.emplace_back(GraphEdge{.nodeA = nodeA, .nodeB = nodeB, .isHard = isHard});
    }
  }
}

static void AddSoftActorsToGraph(
    Span<ActorHandle const> actors,
    Span<int const> parents,
    DynamicArray<GraphNode>& outNodes,
    DynamicArray<GraphEdge>& outEdges) {
  for (int i = 0; i < isize(parents); ++i) {
    int const nodeA = isize(outNodes);
    int const nodeB = parents[i];
    outNodes.push_back(GraphNode{.actor = actors[i], .hasShape = true});
    outNodes[nodeA].edges.push_back(isize(outEdges));
    outNodes[nodeB].edges.push_back(isize(outEdges));
    // Since the soft actor deforms, we do not consider the edge with its parent hard.
    outEdges.emplace_back(GraphEdge{.nodeA = nodeA, .nodeB = nodeB, .isHard = false});
  }
}

// Disable contact for actors based on graph connectivity. Contact is disabled if two actors have
// shape and:
// - The actors are connected.
// - OR the actors are connected by a sequence of hard joints.
// - OR the actors are connected by a sequence of actors with no shape.
static void DisableContactForAdjacentActors(
    SceneImpl* scene,
    Span<GraphNode const> nodes,
    Span<GraphEdge const> edges,
    Error& error) {
  MOCHI_ERROR_RETURN(error);

  // Reusable containers
  DynamicArray<bool> visited(nodes.size(), false);
  DynamicArray<int> cache;
  cache.reserve(nodes.size());
  for (int i = 0; i < isize(nodes); ++i) {
    if (!nodes[i].hasShape) {
      continue;
    }

    // Recursive lambda
    std::function<void(int)> dfsTraverse = [&](int current) {
      for (int edge : nodes[current].edges) {
        int neighbor = edges[edge].nodeA == current ? edges[edge].nodeB : edges[edge].nodeA;
        if (visited[neighbor]) {
          continue;
        }

        visited[neighbor] = true;
        cache.push_back(neighbor);

        // Disable contact if neighbor has shape (consider symmetry)
        if (nodes[neighbor].hasShape && neighbor > i) {
          scene->EnableActorContactSymmetric(
              nodes[i].actor,
              nodes[neighbor].actor,
              /*enable*/ false,
              IncludeNestedActors::No,
              error);
          MOCHI_ERROR_RETURN(error);
        }

        // Continue traversal through hard edges or shapeless nodes
        if (edges[edge].isHard || !nodes[neighbor].hasShape) {
          dfsTraverse(neighbor);
        }
      }
    };

    // DFS to find all nodes reachable from node i
    visited[i] = true;
    cache.push_back(i);
    dfsTraverse(i);

    // Reset visited nodes
    for (int j : cache) {
      visited[j] = false;
    }
    cache.clear();
  }
}

void SceneImpl::CreateArticulatedLinkActorsImpl(
    std::string_view parentActorName,
    Span<ArticulatedLinkParams const> params,
    bool useContact,
    std::shared_ptr<ArticulatedBodyShape const> shapePtr,
    TransformRT const& rootTransform,
    Span<ActorHandle> outLinks,
    Span<ShapeHandle> outLinkShapes,
    Error& error) {
  MOCHI_ERROR_IF(!shapePtr, error, "Requires an articulated shape");
  MOCHI_ERROR_RETURN(error);

  auto const& transforms = shapePtr->GetBoneData()->restRootFromBone;
  auto const& names = shapePtr->GetBoneData()->boneNames;
  MOCHI_ASSERT(
      transforms.size() == names.size(),
      "Bone data arrays should be equal length for any ArticulatedBodyShape that was created successfully.");
  int const numLinks = isize(transforms);
  MOCHI_ASSERT(numLinks > 0, "Every ArticulatedBodyShape should have at least one link");

  MOCHI_ERROR_IF(
      isize(params) != numLinks,
      error,
      "Incorrect number of link params. Expected one per link in the articulated shape.");
  MOCHI_ERROR_IF(
      isize(outLinks) != numLinks,
      error,
      "Output array is the wrong length. Expected one entry per link.");
  MOCHI_ERROR_IF(
      isize(outLinkShapes) != numLinks,
      error,
      "Output shape array is the wrong length. Expected one entry per link.");
  MOCHI_ERROR_RETURN(error);

  std::string const parentName = GetNestedActorParentName(parentActorName);

  // Create link actors. Disable contact for static actors.
  auto const isStatic = IdentifyStaticLinks(shapePtr.get());
  for (int i = 0; i < numLinks; ++i) {
    auto const& link = params[i];
    outLinkShapes[i] = link.shape;

    // Configure rigid params for this link. The link name comes from the shape's bone names, not
    // from the link params, matching how the articulated shape is built.
    RigidActorParams linkParams;
    linkParams.layer = link.layer;
    linkParams.colliderType = link.colliderType;
    linkParams.isStatic = false;
    linkParams.contact = link.contact;
    linkParams.hasGravity = link.hasGravity;
    linkParams.density = link.density;
    linkParams.mass = link.mass;
    linkParams.centerOfMass = link.centerOfMass;
    linkParams.momentOfInertia = link.momentOfInertia;
    linkParams.boundaryElementType = link.boundaryElementType;
    linkParams.boundarySubsampling = link.boundarySubsampling;
    linkParams.worldFromLocal = rootTransform * transforms[i];

    // Name formatting
    std::string_view childNameView = names[i];
    std::string childName =
        childNameView.empty() ? Format("link_%d", i) : std::string(childNameView);
    linkParams.name = Format("%s/%s", parentName.c_str(), childName.c_str());

    bool useContactLink = useContact && !isStatic[i];

    auto linkShapePtr = _context->GetShapeSharedPtr(link.shape);
    MOCHI_ERROR_IF(
        link.shape.IsValid() && !linkShapePtr,
        error,
        "Articulated link shape must be empty or refer to an existing ShapeHandle in this Context.");
    MOCHI_ERROR_RETURN(error);

    // Create actor and assign to output vector
    Actor* actor = CreateRigidActorImpl(
        linkParams, true /*isArticulatedLink*/, useContactLink, linkShapePtr, error);
    if (actor) {
      outLinks[i] = actor->GetHandle();
    }
    MOCHI_ERROR_RETURN(error);
  }
}

void SceneImpl::CreateArticulatedActorJointLimitsImpl(
    entt::entity e,
    ArticulatedActorParams const& params,
    std::shared_ptr<ArticulatedBodyShape const> shape,
    Error& error) {
  MOCHI_ERROR_RETURN(error);

  // Emplace the joint-limits component
  auto& constraints = _registry.emplace<CArticulatedJointLimits>(e);

  // Return if there are no limits
  auto const* jointsData = shape->GetJointsData();
  if (jointsData->jointMinLimits.empty()) {
    return;
  }

  // Limit stiffness/damping come from the active joints, which are index-aligned with params.joints
  // (cycle joints carry type Cycle and do not support joint limits).
  int const numJoints = isize(params.joints);

  // Traverse joint data and create constraints
  constraints.reserve(numJoints);
  for (int i = 0; i < numJoints; i++) {
    auto const& type = jointsData->jointTypes[i];
    Real3 const minLimit = jointsData->jointMinLimits[i];
    Real3 const maxLimit = jointsData->jointMaxLimits[i];
    if ((minLimit == -kInf3) && (maxLimit == kInf3)) {
      continue; // No limits
    }

    switch (type) {
      case ArticulatedJointType::Free:
      case ArticulatedJointType::Hard:
        // These joint types do not use joint limits
        continue;

      case ArticulatedJointType::Spherical: {
        // This is a 3D rotation joint. Define a 3D rotation range constraint.
        Articulated3dRotationRangeConstraintParams dofRangeParams;
        dofRangeParams.actor = GetActorHandle(e, GetHandle());
        dofRangeParams.minValues = minLimit;
        dofRangeParams.maxValues = maxLimit;
        dofRangeParams.jointIndex = i;
        dofRangeParams.stiffness = params.joints[i].limitStiffness;
        dofRangeParams.damping = params.joints[i].limitDamping;

        // Create constraint
        constraints.emplace_back(
            CreateArticulated3dRotationRangeConstraint(dofRangeParams, ErrorAssert{}));
      } break;

      case ArticulatedJointType::Prismatic: // Fallthrough
      case ArticulatedJointType::Revolute: {
        // Compute axis-aligned limits, necessary for 1D joints
        Real3 const& axis = jointsData->jointAxes[i];
        real minValue = Dot(axis, minLimit);
        real maxValue = Dot(axis, maxLimit);
        if (minValue > maxValue) {
          std::swap(minValue, maxValue); // Possibly swap, e.g. if the axis is negative
        }

        // TODO[T261627106] - Historically both positive infinity and negative infinity were
        // used to signal "no limit". Similarly, a mix of finite and non-finite values within
        // a Real3 would result in a NaN dot product, which was silently ignored. The new
        // parameter validation code is stricter, so we should be able to turn this into an
        // error once the old articulated shape files are removed.
        if (!IsFinite(minValue) && !IsFinite(maxValue)) {
          continue;
        } else {
          if (!IsFinite(minValue)) {
            minValue = -kInf;
          }
          if (!IsFinite(maxValue)) {
            maxValue = kInf;
          }
        }

        ArticulatedSingleDofRangeConstraintParams dofRangeParams;
        dofRangeParams.actor = GetActorHandle(e, GetHandle());
        dofRangeParams.minValue = minValue;
        dofRangeParams.maxValue = maxValue;
        dofRangeParams.jointIndex = i;
        dofRangeParams.dofIndex = 0;
        dofRangeParams.stiffness = params.joints[i].limitStiffness;
        dofRangeParams.damping = params.joints[i].limitDamping;

        // Create constraint
        constraints.emplace_back(
            CreateArticulatedSingleDofRangeConstraint(dofRangeParams, ErrorAssert{}));
      } break;

      case ArticulatedJointType::Cycle: // Fallthrough
      case ArticulatedJointType::Count: // Fallthrough
      default:
        static_assert(
            static_cast<int>(ArticulatedJointType::Count) == 6,
            "Please update this code if you add a new joint type");
        MOCHI_ERROR_SET(error, "Invalid articulated joint type");
        return;
    }
  }
}

void SceneImpl::CreateArticulatedActorCycleJointsImpl(
    ArticulatedActorParams const& params,
    std::shared_ptr<ArticulatedBodyShape const> shape,
    entt::entity articulatedActorEntity,
    Span<ActorHandle const> links,
    Error& error) {
  MOCHI_ERROR_RETURN(error);

  auto const& children = shape->GetJointsData()->jointsChildLinks;
  auto const& parents = shape->GetJointsData()->jointsParentLinks;
  auto const& jointFromChildLink = shape->GetJointsData()->jointFromChildLink;
  auto const& restTxs = shape->GetBoneData()->restRootFromBone;

  // The active joints are index-aligned with the links; the passive (cycle) joints follow them and
  // are index-aligned with params.cycles.
  int const numLinks = isize(links);
  int const numCycles = isize(children) - numLinks;
  MOCHI_ERROR_IF(numCycles < 0, error, "Array size mismatch");
  MOCHI_ERROR_RETURN(error);
  if (numCycles > 0) {
    auto& constraints = _registry.emplace<CArticulatedCycleJoints>(articulatedActorEntity);
    constraints.reserve(numCycles);
    for (int i = numLinks; i < isize(children); ++i) {
      auto const parent = parents[i];
      auto const child = children[i];
      auto const& parentTx = restTxs[parent];
      auto const& childTx = restTxs[child];
      RigidSphericalJointConstraintParams jointParams;
      jointParams.actorA = links[parent];
      jointParams.actorB = links[child];
      Real3 const jointPos = Invert(jointFromChildLink[i]).GetTranslation();
      jointParams.localPosA = parentTx.TransformPointInverse(childTx.TransformPoint(jointPos));
      jointParams.localPosB = jointPos;
      jointParams.stiffness = params.cycles[i - numLinks].stiffness;
      auto* newConstraint = CreateRigidSphericalJointConstraint(jointParams, error);
      MOCHI_ERROR_RETURN(error);
      constraints.push_back(newConstraint->GetHandle());
    }
  }
}

// Returns nullptr without setting `error` when `handle` is invalid (i.e. no skin shape was
// requested). A non-null handle that fails validation sets `error` and returns nullptr.
static std::shared_ptr<Shape const>
ResolveSkinShape(ContextImpl const* context, ShapeHandle handle, Error& error) {
  MOCHI_ERROR_RETURN(error, nullptr);
  if (!handle.IsValid()) {
    return nullptr;
  }
  auto shape = context->GetShapeSharedPtr(handle);
  MOCHI_ERROR_IF_NOT(shape, error, "Invalid skin shape handle.");
  MOCHI_ERROR_RETURN(error, nullptr);
  MOCHI_ERROR_IF_NOT(
      dynamic_cast<TetrahedralMeshShape const*>(shape.get()) ||
          dynamic_cast<TriangularMeshShape const*>(shape.get()),
      error,
      "Skin shape must be a TetrahedralMeshShape or TriangularMeshShape.");
  MOCHI_ERROR_RETURN(error, nullptr);
  MOCHI_ERROR_IF_NOT(
      shape->GetMeshSkinning() != nullptr,
      error,
      "Skin shape must contain skinning data (per-node weights and link indices).");
  MOCHI_ERROR_RETURN(error, nullptr);
  return shape;
}

Actor* SceneImpl::CreateArticulatedActorImpl(
    ArticulatedActorParams const& params,
    bool useContact,
    std::shared_ptr<ArticulatedBodyShape const> shapePtr,
    Span<ActorHandle const> links,
    std::shared_ptr<Shape const> skinShape,
    Error& error) {
  MOCHI_ERROR_RETURN(error, nullptr);

  // Create an ECS entity
  entt::entity e = _registry.create();

  // Create an implementation of mochi::Actor to satisfy the public interface
  _registry.emplace<CActorInterface>(e, CreateActorInterface(_registry, e, this));

  // Initialize articulated body actor.
  articulated::compound::InitArticulatedBodyActor(
      _registry, e, params, useContact, shapePtr, skinShape, links, error);
  MOCHI_DESTROY_AND_RETURN_IF_ERROR();

  // Add cycle joints (for closed kinematic chains)
  CreateArticulatedActorCycleJointsImpl(params, shapePtr, e, links, error);
  MOCHI_DESTROY_AND_RETURN_IF_ERROR();

  // Add joint limits
  CreateArticulatedActorJointLimitsImpl(e, params, shapePtr, error);
  MOCHI_DESTROY_AND_RETURN_IF_ERROR();

  // Tag static links, once initialized and added to compounds and islands.
  auto const isStatic = IdentifyStaticLinks(shapePtr.get());
  for (auto i = 0; i < links.size(); ++i) {
    if (isStatic[i]) {
      auto link = GetEntity(_registry, links[i], error);
      _registry.emplace_or_replace<TagStaticActor>(link);
    }
  }

  // Initialization specific to a differentiable scene
  if (_registry.try_ctx<TagDifferentiableScene>()) {
    articulated::compound::InitDifferentiableActor(_registry, e, error);
    MOCHI_DESTROY_AND_RETURN_IF_ERROR();
  }

  ++_numActors;
  _registry.emplace<TagFullyInitialized>(e); // Do this last
  ValidateNewActorComposition(e);
  return GetActor(GetActorHandle(e, GetHandle()));
}

Actor* SceneImpl::CreateArticulatedActorImpl(
    ArticulatedActorParams const& params,
    std::shared_ptr<ArticulatedBodyShape const> shapePtr,
    Error& error) {
  MOCHI_ERROR_RETURN(error, nullptr);

  ShapeHandle const skinHandle = params.skin.has_value() ? params.skin->shape : ShapeHandle{};
  auto skinShape = ResolveSkinShape(_context, skinHandle, error);
  MOCHI_ERROR_RETURN(error, nullptr);

  // A zero-DOF articulated actor (every joint Hard/weld, so reducedDofsDim == 0) is a static welded
  // structure with nothing to solve. A skin adds considerable complexity and is not handled.
  if (skinShape != nullptr) {
    int const reducedDofsDim = articulated::GetReducedDofsSize(shapePtr->GetJointsData()->dofInfo);
    MOCHI_ERROR_IF(
        reducedDofsDim == 0,
        error,
        "A zero-DOF articulated actor (all joints are Hard) cannot have a skin.");
    MOCHI_ERROR_RETURN(error, nullptr);
  }

  // Create link actors from the articulated link params.
  DynamicArray<ActorHandle> links(shapePtr->GetNumBones());
  DynamicArray<ShapeHandle> linkShapes(shapePtr->GetNumBones());
  bool const useContactLinks = skinShape == nullptr;
  ScopedActorCreationRollback actorRollback(*this);
  CreateArticulatedLinkActorsImpl(
      params.name,
      params.links,
      useContactLinks,
      shapePtr,
      params.worldFromRoot,
      links,
      linkShapes,
      error);
  for (auto link : links) {
    actorRollback.Add(link);
  }
  MOCHI_ERROR_RETURN(error, {});

  // Disable contact for actors that are adjacent in the hierarchy. Adjacency also considers hard
  // joints and shapeless links.
  DynamicArray<GraphNode> nodes;
  DynamicArray<GraphEdge> edges;
  CreateGraphOfLinks(links, shapePtr.get(), MakeConstSpan(linkShapes), nodes, edges);
  DisableContactForAdjacentActors(this, nodes, edges, error);
  MOCHI_ERROR_RETURN(error, {});

  // Finally create the articulated body actor
  Actor* actor =
      CreateArticulatedActorImpl(params, /* useContact */ true, shapePtr, links, skinShape, error);
  if (actor) {
    actorRollback.ReplaceWithOwner(actor->GetHandle());
  }
  MOCHI_ERROR_RETURN(error, {});
  actorRollback.Release();
  return actor;
}

Actor* SceneImpl::CreateArticulatedActor(ArticulatedActorParams const& params, Error& error) {
  MOCHI_ERROR_RETURN(error, nullptr);

  auto paramsCopy = params;
  AutoCorrect(paramsCopy, error);
  Validate(paramsCopy, error);
  MOCHI_ERROR_RETURN(error, nullptr);

  // Build the articulated shape from the params.
  auto shapePtr = GetArticulatedShape(paramsCopy, error);
  Actor* newActor = CreateArticulatedActorImpl(paramsCopy, shapePtr, error);
  MOCHI_ERROR_RETURN(error, nullptr);

  StoreSkinDataForExport(_registry, newActor, paramsCopy.skin);

  return newActor;
}

static DynamicArray<int> FindSoftLinkParents(
    SceneImpl* scene,
    Span<ActorHandle const> links,
    Span<DynamicString const> softAttachLinks,
    Error& error) {
  DynamicArray<int> outParents;
  MOCHI_ERROR_RETURN(error, outParents);

  // Build a hash map of link names
  std::unordered_map<std::string, int> linkNameToIndex;
  linkNameToIndex.reserve(links.size());
  for (int i = 0; i < isize(links); ++i) {
    auto const* actor = scene->GetActor(links[i]);
    MOCHI_ASSERT(actor, "Invalid actor");
    char const* linkName = actor->GetName();
    // The link name is formatted like "parentName/linkName".
    // We are interested in the link name, starting after the last forward slash.
    char const* slash = std::strrchr(linkName, '/');
    linkName = slash ? slash + 1 : linkName;
    linkNameToIndex[linkName] = i;
  }

  outParents.reserve(softAttachLinks.size());
  for (auto const& softAttachLink : softAttachLinks) {
    auto it = linkNameToIndex.find(std::string(softAttachLink));
    MOCHI_ERROR_IF(it == linkNameToIndex.end(), error, "Invalid link id");
    MOCHI_ERROR_RETURN(error, {});
    outParents.emplace_back(it->second);
  }
  return outParents;
}

// Return a TetrahedralMeshShape which is identical to the source shape, except that it has new
// SkinningData.
static std::shared_ptr<TetrahedralMeshShape const> CreateDuplicateShapeWithSkinning(
    std::shared_ptr<TetrahedralMeshShape const> srcShape,
    SkinningData skinning,
    Error& error) {
  MOCHI_ERROR_IF(!srcShape, error, "Requires a tetrahedral mesh shape");
  MOCHI_ERROR_RETURN(error, {});

  // If a GridSdf was requested on demand (not loaded from file), then it might still be pending.
  // Wait for it to complete so that we can share the same shared_ptr<GridSdf const>.
  srcShape->GetGridSdfSemaphore().Wait();

  // Combine the new skinning data with a copy of the rest of the data from the source
  // shape. Most fields are of the form std::shared_ptr<T const> so those are shallow copies.
  return std::make_shared<TetrahedralMeshShape>(
      srcShape->GetMesh(),
      std::make_shared<SkinningData const>(std::move(skinning)),
      srcShape->GetMeshConstrainedNodes(),
      srcShape->GetMeshBlending(),
      srcShape->GetVisualMesh(),
      srcShape->GetVisualEmbedding(),
      srcShape->GetGridSdf(),
      srcShape->GetRomData(), // Deep copy
      srcShape->GetSampleMeshes(), // Deep copy
      srcShape->GetBoundingSphereHierarchies(), // Deep copy
      srcShape->GetSoftMaterialParamsField());
}

Actor* SceneImpl::CreateSoftSkinnedActorImpl(
    SoftSkinnedActorParams const& params,
    experimental::ExperimentalSoftSkinnedActorParams const& experimentalParams,
    std::shared_ptr<ArticulatedBodyShape const> articulatedShapePtr,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  // Validate experimental params size
  int const numSoftActors = isize(params.softParams);
  bool const hasExperimentalParams = !experimentalParams.softParams.empty();
  MOCHI_ERROR_IF(
      hasExperimentalParams && isize(experimentalParams.softParams) != numSoftActors,
      error,
      "Unexpected number of experimental soft actor params. Expected one for each soft actor or zero.");
  MOCHI_ERROR_RETURN(error, {});

  // Get the articulated shape pointer
  MOCHI_ERROR_IF_NOT(articulatedShapePtr, error, "Invalid articulated shape.");
  MOCHI_ERROR_RETURN(error, {});

  // A soft-skinned (or blended) actor deforms its soft bodies with the skeleton through the joint
  // Jacobian. A zero-DOF skeleton (every joint Hard/weld, so reducedDofsDim == 0) provides no such
  // coupling, so this combination is not supported. Reject it explicitly at creation.
  MOCHI_ERROR_IF(
      articulated::GetReducedDofsSize(articulatedShapePtr->GetJointsData()->dofInfo) == 0,
      error,
      "A soft-skinned actor cannot be bound to a zero-DOF skeleton (all joints are Hard).");
  MOCHI_ERROR_RETURN(error, {});

  // If provided, get the blended mesh shape; a tetrahedral or triangular mesh shape.
  auto const& skeletonParams = params.skeletonParams;
  ShapeHandle const blendedHandle =
      skeletonParams.skin.has_value() ? skeletonParams.skin->shape : ShapeHandle{};
  auto blendedShapePtr = ResolveSkinShape(_context, blendedHandle, error);
  MOCHI_ERROR_RETURN(error, {});
  // Do not pass subsampling settings if there's no blended surface
  MOCHI_ERROR_IF(
      skeletonParams.skin.has_value() && skeletonParams.skin->boundarySubsampling.has_value() &&
          !blendedShapePtr,
      error,
      "Do not pass skin subsampling settings to non-blended soft skinned actor");
  MOCHI_ERROR_RETURN(error, {});

  // Get the soft-actor shapes
  std::vector<std::shared_ptr<TetrahedralMeshShape const>> softShapes;
  softShapes.reserve(numSoftActors);
  for (auto const& softParams : params.softParams) {
    auto softShapePtr = std::dynamic_pointer_cast<TetrahedralMeshShape const>(
        _context->GetShapeSharedPtr(softParams.shape));
    MOCHI_ERROR_IF_NOT(softShapePtr, error, "Invalid soft shape.");
    MOCHI_ERROR_RETURN(error, {});
    softShapes.emplace_back(softShapePtr);
  }

  // Create bone actors. Their names will be formatted like "skeletonName/linkName".
  DynamicArray<ActorHandle> links(articulatedShapePtr->GetNumBones());
  DynamicArray<ShapeHandle> linkShapes(articulatedShapePtr->GetNumBones());
  ScopedActorCreationRollback actorRollback(*this);
  CreateArticulatedLinkActorsImpl(
      skeletonParams.name,
      skeletonParams.links,
      /* useContact */ params.enableCollidingLinks,
      articulatedShapePtr,
      skeletonParams.worldFromRoot,
      links,
      linkShapes,
      error);
  for (auto link : links) {
    actorRollback.Add(link);
  }
  MOCHI_ERROR_RETURN(error, {});

  // Create the articulated body. If there's blending, skin the blended surface.
  Actor* skeletonActor = CreateArticulatedActorImpl(
      skeletonParams,
      /* useContact */ false,
      articulatedShapePtr,
      links,
      blendedShapePtr,
      error);
  if (skeletonActor) {
    actorRollback.ReplaceWithOwner(skeletonActor->GetHandle());
  }
  MOCHI_ERROR_RETURN(error, {});

  // If soft actors are externally attached, identify parent links
  auto softLinkParents = FindSoftLinkParents(this, links, params.softAttachLinks, error);
  MOCHI_ERROR_RETURN(error, {});

  // If externally given, add attachment info to soft shapes.
  if (!softLinkParents.empty()) {
    MOCHI_ERROR_IF(isize(softLinkParents) != numSoftActors, error, "Invalid attachments params");
    MOCHI_ERROR_RETURN(error, {});
    for (int i = 0; i < numSoftActors; ++i) {
      // Create SkinningData which will attach the soft shape to the parent link.
      auto const numNodes = softShapes[i]->GetMesh()->GetNumNodes();
      SkinningData skinning;
      skinning.weightsPerNode = 1;
      skinning.weights.resize(numNodes, 1_r);
      skinning.indices.resize(numNodes, softLinkParents[i]);

      // Create a new TetrahedralMeshShape which is identical to the original, except for the new
      // SkinningData.
      softShapes[i] = CreateDuplicateShapeWithSkinning(softShapes[i], std::move(skinning), error);
      MOCHI_ERROR_RETURN(error, {});
    }
  }

  // Create the soft actors
  std::string const softParentName = GetNestedActorParentName(skeletonActor->GetName());
  std::vector<ActorHandle> softActors(numSoftActors);
  for (int i = 0; i < numSoftActors; ++i) {
    auto softParams = params.softParams[i]; // Copy

    // Nested soft actors get slash-separated names, just like the nested link actors.
    softParams.name = Format("%s/%s", softParentName.c_str(), softParams.name.c_str());

    // Use experimental params if provided, otherwise use empty experimental params.
    auto experimentalSoftParams =
        hasExperimentalParams ? experimentalParams.softParams[i] : ExperimentalSoftActorParams{};

    // Make sure recentering is disabled.
    experimentalSoftParams.useRecentering = false;

    Actor* softActor = CreateSoftActorImpl(
        softParams, experimentalSoftParams, /* isNestedSoft */ true, softShapes[i], error);
    if (softActor) {
      softActors[i] = softActor->GetHandle();
      actorRollback.Add(softActors[i]);
    }
    MOCHI_ERROR_RETURN(error, {});

    // If not ROM, set Dirichlet boundary conditions on constrained nodes (as specified by the
    // soft shape) to couple soft actor to articulated actor.
    if (!experimentalSoftParams.rom) {
      softActor->AddBoundaryConditionConstrainedNodesAtRestPermanent(error);
      MOCHI_ERROR_RETURN(error, {});
    }
  }

  // Disable contact for actors that are adjacent in the hierarchy. Adjacency also considers hard
  // joints and shapeless links.
  DynamicArray<GraphNode> nodes;
  DynamicArray<GraphEdge> edges;
  CreateGraphOfLinks(links, articulatedShapePtr.get(), MakeConstSpan(linkShapes), nodes, edges);
  AddSoftActorsToGraph(softActors, softLinkParents, nodes, edges);
  DisableContactForAdjacentActors(this, nodes, edges, error);
  MOCHI_ERROR_RETURN(error, {});

  // Initialize the nested soft actor components. The entities are the same as the soft actors.
  bool const useNestedSoftContact = !blendedShapePtr;
  for (int i = 0; i < numSoftActors; ++i) {
    auto entity = GetEntity(_registry, softActors[i], error);
    skinned::InitSkinnedActor(
        _registry, entity, params, useNestedSoftContact, skeletonActor->GetHandle(), error);
    MOCHI_ERROR_RETURN(error, {});
  }

  // Initialize the blended actor components. The entity is the same as the skeleton.
  auto entity = GetEntity(_registry, skeletonActor->GetHandle(), error);
  blended::InitBlendedActor(_registry, entity, skeletonParams, softActors, blendedShapePtr, error);
  MOCHI_ERROR_RETURN(error, {});

  // The skeleton can now destroy its nested soft actors, so it is the only rollback root needed.
  actorRollback.ReplaceWithOwner(skeletonActor->GetHandle());

  // Store the soft attachment links in the ECS for export purposes
  if (!softLinkParents.empty()) {
    _registry.emplace<CSoftAttachmentLinks>(entity, std::move(softLinkParents));
  }

  // Return the skeleton actor
  actorRollback.Release();
  return skeletonActor;
}

void SceneImpl::DestroyActor(ActorHandle actorHandle) {
  if (!actorHandle.IsValid()) {
    return;
  }
  if (!ActorHandleBelongsToScene(actorHandle, GetHandle())) {
    MOCHI_LOG_WARNING(
        "Ignoring attempt to destroy an actor using an ActorHandle from a different scene.");
    return;
  }

  entt::entity e = GetEntityUnchecked(actorHandle);
  if (!_registry.valid(e)) {
    // Nothing to destroy, not a valid entity anymore.
    return;
  }

  if (GetActor(actorHandle) == nullptr) {
    // Handle value does not encode an actor type.
    return;
  }

  MOCHI_ASSERT(
      (!_registry.any_of<TagCompoundActor>(e) && !_registry.any_of<CGroupMembers>(e)) ||
          _registry.all_of<TagArticulatedActor>(e),
      "Non-articulated compounds must not be accessible through the public API.");

  // It is illegal to destroy a member of an articulated actor via the public interface.
  // The user must destroy the entire articulated actor instead. Because the user may simply be
  // trying to destroy all of their actors, in no particular order.
  if (_registry.all_of<TagArticulatedLinkActor, CGroupMemberInfo>(e)) {
    MOCHI_LOG_WARNING(
        "Articulated links cannot be destroyed individually. Please destroy the entire articulated actor instead.");
    return;
  }

  // If this is an articulated actor with nested soft actors, destroy them first.
  auto const* blended = _registry.try_get<CBlendedComposition const>(e);
  if (blended) {
    for (auto soft : blended->softHandles) {
      DestroyActor(soft);
    }
  }

  MOCHI_ASSERT(_numActors >= 1);
  --_numActors;
  DestroyActorEntity(*this, _registry, e);
}

Actor* SceneImpl::GetActor(ActorHandle actor) {
  if (!actor.IsValid()) {
    return nullptr;
  }
  if (!ActorHandleBelongsToScene(actor, GetHandle())) {
    MOCHI_LOG_WARNING(
        "Ignoring attempt to get an actor using an ActorHandle from a different scene.");
    return nullptr;
  }
  entt::entity e = GetEntityUnchecked(actor); // We'll check it ourselves
  if (_registry.valid(e)) {
    auto* api = _registry.try_get<CActorInterface>(e);
    if (api) {
      return api->ptr.get();
    }
  }
  return nullptr;
}

Actor const* SceneImpl::GetActor(ActorHandle actor) const {
  return const_cast<SceneImpl*>(this)->GetActor(actor); // Share code with the non-const overload
}

Constraint* SceneImpl::CreateConstraintImpl(
    std::function<void(entt::registry&, entt::entity, Error&)> const& init,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  // Create an ECS entity
  entt::entity e = _registry.create();
  ConstraintHandle newHandle = GetConstraintHandle(e, GetHandle());

  // Create an implementation of mochi::Constraint to satisfy the public interface
  _registry.emplace<CConstraintInterface>(e, CreateConstraintInterface(_registry, e, this));

  // Initialize constraint
  init(_registry, e, error);
  MOCHI_DESTROY_AND_RETURN_IF_ERROR();

  // Upate CConstraintMemberInfo on each of the affected actors, so that they point back to the
  // constraint entity.
  auto const& info = _registry.get<CConstraintInfo>(e);
  for (entt::entity actor : info.actors) {
    auto& actorInfo = _registry.get_or_emplace<CConstraintMemberInfo>(actor);
    // It's possible that a constraint acts on two nodes of the same actor. Skip if already present
    if (std::find(actorInfo.constraints.begin(), actorInfo.constraints.end(), e) ==
        actorInfo.constraints.end()) {
      actorInfo.constraints.push_back(e);
    }
  }

  // Needed by every type of constraint
  _registry.emplace<CConstraintGlobalSparsityCache>(e);

  ++_numConstraints;

  // If the user doesn't manually add this constraint to a compound, then we will do it
  // automatically on the next step.
  _registry.emplace<TagEnsureEntityInCompound>(e);

  _registry.emplace<TagFullyInitialized>(e);

  auto* constraintPtr = GetConstraint(newHandle);
  MOCHI_ASSERT(constraintPtr != nullptr);
  return constraintPtr;
}

#undef MOCHI_DESTROY_AND_RETURN_IF_ERROR

Constraint* SceneImpl::CreateRigidSphericalJointConstraint(
    RigidSphericalJointConstraintParams const& params,
    Error& error) {
  return CreateConstraintImpl(
      [&](entt::registry& reg, entt::entity e, Error& error) {
        InitConstraint_RigidSphericalJoint(reg, e, params, error);
      },
      error);
}

Constraint* SceneImpl::CreateRigidPrismaticJointConstraint(
    RigidPrismaticJointConstraintParams const& params,
    Error& error) {
  return CreateConstraintImpl(
      [&](entt::registry& reg, entt::entity e, Error& error) {
        InitConstraint_RigidPrismaticJoint(reg, e, params, error);
      },
      error);
}

Constraint* SceneImpl::CreateDeformableNodeToDeformableNodeConstraint(
    DeformableNodeToDeformableNodeConstraintParams const& params,
    Error& error) {
  return CreateConstraintImpl(
      [&](entt::registry& reg, entt::entity e, Error& error) {
        InitConstraint_DeformableNodeToDeformableNode(reg, e, params, error);
      },
      error);
}

Constraint* SceneImpl::CreateDeformableNodeToRigidConstraint(
    DeformableNodeToRigidConstraintParams const& params,
    Error& error) {
  return CreateConstraintImpl(
      [&](entt::registry& reg, entt::entity e, Error& error) {
        InitConstraint_DeformableNodeToRigid(reg, e, params, error);
      },
      error);
}

Constraint* SceneImpl::CreateJointRotationRangeConstraint(
    JointRotationRangeConstraintParams const& params,
    Error& error) {
  return CreateConstraintImpl(
      [&](entt::registry& reg, entt::entity e, Error& error) {
        InitConstraint_JointRotationRange(reg, e, params, error);
      },
      error);
}

Constraint* SceneImpl::CreateJointRotationTrackingConstraint(
    JointRotationTrackingConstraintParams const& params,
    Error& error) {
  return CreateConstraintImpl(
      [&](entt::registry& reg, entt::entity e, Error& error) {
        InitConstraint_JointRotationTracking(reg, e, params, error);
      },
      error);
}

Constraint* SceneImpl::CreateRodElementRotationToRigidConstraint(
    RodElementRotationToRigidConstraintParams const& params,
    Error& error) {
  return CreateConstraintImpl(
      [&](entt::registry& reg, entt::entity e, Error& error) {
        InitConstraint_RodElementRotationToRigid(reg, e, params, error);
      },
      error);
}

Constraint* SceneImpl::CreateRigidPivotPositionConstraint(
    RigidPivotPositionConstraintParams const& params,
    Error& error) {
  return CreateConstraintImpl(
      [&](entt::registry& reg, entt::entity e, Error& error) {
        InitConstraint_RigidPivotPosition(reg, e, params, error);
      },
      error);
}

Constraint* SceneImpl::CreateRigidPivotToRigidTargetConstraint(
    RigidPivotToRigidTargetConstraintParams const& params,
    Error& error) {
  return CreateConstraintImpl(
      [&](entt::registry& reg, entt::entity e, Error& error) {
        InitConstraint_RigidPivotToRigidTarget(reg, e, params, error);
      },
      error);
}

Constraint* SceneImpl::CreateRigidPivotRotationConstraint(
    RigidPivotRotationConstraintParams const& params,
    Error& error) {
  return CreateConstraintImpl(
      [&](entt::registry& reg, entt::entity e, Error& error) {
        InitConstraint_RigidPivotRotation(reg, e, params, error);
      },
      error);
}

Constraint* SceneImpl::CreateDeformableNodePositionConstraint(
    DeformableNodePositionConstraintParams const& params,
    Error& error) {
  return CreateConstraintImpl(
      [&](entt::registry& reg, entt::entity e, Error& error) {
        InitConstraint_DeformableNodePosition(reg, e, params, error);
      },
      error);
}

Constraint* SceneImpl::CreateArticulated3dRotationTargetConstraint(
    Articulated3dRotationTargetConstraintParams const& params,
    Error& error) {
  return CreateConstraintImpl(
      [&](entt::registry& reg, entt::entity e, Error& error) {
        InitConstraint_Articulated3dRotationTarget(reg, e, params, error);
      },
      error);
}

Constraint* SceneImpl::CreateArticulatedSingleDofTargetConstraint(
    ArticulatedSingleDofTargetConstraintParams const& params,
    Error& error) {
  return CreateConstraintImpl(
      [&](entt::registry& reg, entt::entity e, Error& error) {
        InitConstraint_ArticulatedSingleDofTarget(reg, e, params, error);
      },
      error);
}

Constraint* SceneImpl::CreateArticulated3dRotationRangeConstraint(
    Articulated3dRotationRangeConstraintParams const& params,
    Error& error) {
  return CreateConstraintImpl(
      [&](entt::registry& reg, entt::entity e, Error& error) {
        InitConstraint_Articulated3dRotationRange(reg, e, params, error);
      },
      error);
}

Constraint* SceneImpl::CreateArticulatedSingleDofRangeConstraint(
    ArticulatedSingleDofRangeConstraintParams const& params,
    Error& error) {
  return CreateConstraintImpl(
      [&](entt::registry& reg, entt::entity e, Error& error) {
        InitConstraint_ArticulatedSingleDofRange(reg, e, params, error);
      },
      error);
}

Constraint* SceneImpl::GetConstraint(ConstraintHandle constraint) {
  if (!constraint.IsValid()) {
    return nullptr;
  }
  if (!ConstraintHandleBelongsToScene(constraint, GetHandle())) {
    MOCHI_LOG_WARNING(
        "Ignoring attempt to get a constraint using a ConstraintHandle from a different scene.");
    return nullptr;
  }
  entt::entity e = GetEntityUnchecked(constraint); // We'll check it ourselves
  if (_registry.valid(e)) {
    auto* api = _registry.try_get<CConstraintInterface>(e);
    if (api) {
      return api->ptr.get();
    }
  }
  return nullptr;
}

Constraint const* SceneImpl::GetConstraint(ConstraintHandle constraint) const {
  return const_cast<SceneImpl*>(this)->GetConstraint(
      constraint); // Share code with the non-const overload
}

void SceneImpl::DestroyConstraint(Constraint* constraint) {
  if (constraint) {
    DestroyConstraint(constraint->GetHandle());
  }
}

void SceneImpl::DestroyConstraint(ConstraintHandle constraint) {
  if (!constraint.IsValid()) {
    return;
  }
  if (!ConstraintHandleBelongsToScene(constraint, GetHandle())) {
    MOCHI_LOG_WARNING(
        "Ignoring attempt to destroy a constraint using a ConstraintHandle from a different scene.");
    return;
  }

  entt::entity constraintEntity = GetEntityUnchecked(constraint);
  if (!_registry.valid(constraintEntity)) {
    // Nothing to destroy, not a valid entity anymore.
    return;
  }

  if (GetConstraint(constraint) == nullptr) {
    // Handle value does not encode a constraint type.
    return;
  }

  // Update CConstraintMemberInfo on each affected actor, so that they no longer point
  // back to this constraint entity.
  auto& constraintInfo = _registry.get<CConstraintInfo>(constraintEntity);
  std::unordered_set<entt::entity> processedActors;
  for (entt::entity actor : constraintInfo.actors) {
    if (!processedActors.insert(actor).second) {
      continue;
    }

    auto& actorConstraints = _registry.get<CConstraintMemberInfo>(actor).constraints;
    auto itr = std::find(actorConstraints.begin(), actorConstraints.end(), constraintEntity);
    MOCHI_ASSERT(
        itr != actorConstraints.end(),
        "CConstraintInfo on the constraint entity is out-of-sync with CConstraintMemberInfo on the actor");
    actorConstraints.erase(itr);
    if (actorConstraints.empty()) {
      // This actor is no longer affected by any constraints, so remove the component.
      _registry.erase<CConstraintMemberInfo>(actor);
    }
  }

  // The constraint probably belongs to a compound. If so, remove it from the compound.
  if (auto* groupMember = _registry.try_get<CGroupMemberInfo>(constraintEntity)) {
    RemoveConstraintFromCompound(_registry, groupMember->group, constraintEntity, ErrorAssert{});
  }

  // Destroy the ECS entity and all components
  _numConstraints--;
  _registry.destroy(constraintEntity);
}

int SceneImpl::GetNumConstraints() const {
  return _numConstraints;
}

void SceneImpl::ForEachConstraint(std::function<void(Constraint* constraint)> const& callback) {
  for (auto&& [e, api] : _registry.view<CConstraintInterface const>().each()) {
    callback(api.ptr.get());
  }
}

void SceneImpl::ForEachConstraint(
    std::function<void(Constraint const* constraint)> const& callback) const {
  for (auto&& [e, api] : _registry.view<CConstraintInterface const>().each()) {
    callback(api.ptr.get());
  }
}

CallbackHandle SceneImpl::RegisterPreStepCallback(
    std::string_view debugName,
    std::function<void(StepInfo const&)> callback,
    int priority) {
  std::string nameCopy{debugName};
  auto newHandle = GenerateNewCallbackHandle(CallbackType::PreStep);
  _preStepCallbacks.Register(
      newHandle.value,
      [callback = std::move(callback), nameCopy = std::move(nameCopy)](StepInfo const& stepInfo) {
        MOCHI_PROFILE_SCOPE_N("PreStepCallback");
        MOCHI_PROFILE_LABEL(nameCopy);
        callback(stepInfo);
      },
      priority);
  return newHandle;
}

CallbackHandle SceneImpl::RegisterPostStepCallback(
    std::string_view debugName,
    std::function<void(StepInfo const&)> callback,
    int priority) {
  std::string nameCopy{debugName};
  auto newHandle = GenerateNewCallbackHandle(CallbackType::PostStep);
  _postStepCallbacks.Register(
      newHandle.value,
      [callback = std::move(callback), nameCopy = std::move(nameCopy)](StepInfo const& stepInfo) {
        MOCHI_PROFILE_SCOPE_N("PostStepCallback");
        MOCHI_PROFILE_LABEL(nameCopy);
        callback(stepInfo);
      },
      priority);
  return newHandle;
}

void SceneImpl::CancelCallback(CallbackHandle handle) {
  // Deregister the callback according to its type.
  // Redundant deregistration will be ignored.
  if (handle.IsValid()) {
    if (!IsProbablyMyCallbackHandle(handle)) {
      MOCHI_LOG_WARNING("Ignoring attempt to cancel a CallbackHandle from a different scene.");
      return;
    }
    switch (GetCallbackType(handle)) {
      case CallbackType::PreStep:
        _preStepCallbacks.Deregister(handle.value);
        break;
      case CallbackType::PostStep:
        _postStepCallbacks.Deregister(handle.value);
        break;
    }
  }
}

void SceneImpl::SetThreadAffinity() {
  _debugDraw->SetThreadAffinity();
}

QueryHandle SceneImpl::NewQueryHandle(QueryType type) {
  return _registry.ctx<CQueryHandleAllocator>().NewHandleThreadSafe(type);
}

void SceneImpl::RegisterActorQuery(
    Actor* actor,
    QueryHandle preallocatedHandle,
    bool computeImmediately,
    Error& error) {
  MOCHI_ERROR_IF(!actor, error, "Invalid Actor");
  MOCHI_ERROR_IF(!preallocatedHandle.IsValid(), error, "Invalid QueryHandle");
  MOCHI_ERROR_RETURN(error);
  entt::entity e = GetEntity(_registry, actor->GetHandle(), error);
  RegisterQuery(_registry, e, preallocatedHandle, computeImmediately, error);
}

// Called when a new actor is created. Attempts to catch bugs caused by inconsistently
// configured ECS components.
void SceneImpl::ValidateNewActorComposition(entt::entity e) const {
#if MOCHI_ASSERT_ENABLED
  MOCHI_ASSERT(_registry.valid(e));
  MOCHI_ASSERT(_registry.all_of<CActorInfo>(e), "Missing required component");
  MOCHI_ASSERT(_registry.all_of<TagFullyInitialized>(e), "Missing required component");
  MOCHI_ASSERT(
      !_registry.all_of<CActorSnle>(e) || _registry.all_of<CActorConvergenceWeights>(e),
      "Actors with CActorSnle must also have CActorConvergenceWeights, even if useInSolver = false at initialization.");

  // Stuff needed for contact
  auto const* colliderInfo = _registry.try_get<CColliderInfo const>(e);
  bool const hasCollider = colliderInfo && (colliderInfo->type != ColliderType::None);
  bool const canDetectContact = _registry.all_of<TagUseContact>(e);
  if (hasCollider || canDetectContact) {
    MOCHI_ASSERT(
        _registry.all_of<CBoundingVolume<TimeStep::Previous>>(e), "Missing required component");
    MOCHI_ASSERT(
        _registry.all_of<CBoundingVolume<TimeStep::Current>>(e), "Missing required component");
    MOCHI_ASSERT(_registry.all_of<CContactLayer>(e), "Missing required component");
    MOCHI_ASSERT(_registry.all_of<CContactParams>(e), "Missing required component");
  }
  if (canDetectContact) {
    MOCHI_ASSERT(
        !_registry.any_of<TagStaticActor>(e), "Static actor should not have TagUseContact");
    MOCHI_ASSERT(_registry.all_of<CConservativeStepBounds>(e), "Missing required component");
    MOCHI_ASSERT(
        _registry.all_of<CPotentialColliders<ContactType::Async>>(e), "Missing required component");
    MOCHI_ASSERT(
        _registry.all_of<CPotentialColliders<ContactType::Sync>>(e), "Missing required component");
  }

  if (_registry.all_of<TagRodActor>(e)) {
    bool const usesContactSkin = _registry.all_of<TagRodSurfaceContact>(e);
    bool const hasContactSkinComponents = _registry.all_of<
        CRodContactSkin,
        CRodContactSkinningData,
        CRodDeformedContactSkinNodes,
        CFemSurfaceDiscretization,
        CSkinnedContactSnle,
        TagSkinnedContact>(e);
    MOCHI_ASSERT(
        usesContactSkin == hasContactSkinComponents,
        "Rod contact-skin tag and components must be installed together.");
    MOCHI_ASSERT(
        usesContactSkin != _registry.all_of<CFemSegmentDiscretization>(e),
        "Rod contact must use exactly one of contact-skin or centerline discretization.");
  }

  // All actors must have a CConvergenceStatus, except static actors and internal-only compounds
  // (not accessible through the public API).
  bool const isStatic = _registry.any_of<TagStaticActor>(e);
  bool const isInternalCompound =
      _registry.all_of<TagCompoundActor>(e) && !_registry.any_of<TagArticulatedActor>(e);
  if (!isStatic && !isInternalCompound) {
    MOCHI_ASSERT(
        _registry.all_of<CConvergenceStatus>(e), "Missing required CConvergenceStatus component.");
  }
#endif // MOCHI_ASSERT_ENABLED
}

CallbackHandle SceneImpl::GenerateNewCallbackHandle(CallbackType type) {
  auto newCallbackId = _nextCallbackId++;
  if (_nextCallbackId == 0) {
    MOCHI_LOG_WARNING(
        "Too many callbacks have been registered. The CallbackHandle may not be unique.");
    _nextCallbackId++;
  }

  // Pack the callback type into the handle, so we know what to do in CancelCallback.
  // Include at least part of the scene ID as well, to help catch mistakes.
  return CallbackHandle{
      (_sceneId << 40) | (static_cast<uint64_t>(type) << 32) |
      static_cast<uint64_t>(newCallbackId)};
}

SceneImpl::CallbackType SceneImpl::GetCallbackType(CallbackHandle handle) {
  return static_cast<CallbackType>((handle.value >> 32) & 0xFF);
}

bool SceneImpl::IsProbablyMyCallbackHandle(CallbackHandle handle) const {
  return (handle.value >> 40) == (_sceneId & 0x0000000000FFFFFFull);
}

void SceneImpl::SetDebugger(std::shared_ptr<dbg::SceneDebugger> debugger) {
  _debugger.Mutate([&](auto& info) {
    if (info.debugger) {
      // Retire the old one, but don't delete it yet. It might have some final
      // cleanup to do on the scene's controlling thread.
      info.pendingShutdown.emplace_back(std::move(info.debugger));
    }
    info.debugger = std::move(debugger);
  });
}

std::shared_ptr<dbg::SceneDebugger> SceneImpl::GetDebugger() const {
  return _debugger.Read(&DebuggerInfo::debugger);
}

MOCHI_API void experimental::ApplyImprovedConvergenceSettings(Scene* scene, Error& error) {
  MOCHI_ERROR_IF(!scene, error, "Invalid scene");
  MOCHI_ERROR_RETURN(error);
  auto* sceneImpl = assert_cast<SceneImpl*>(scene);
  sceneImpl->ApplyImprovedConvergenceSettings();
}

void MakeSceneDifferentiableInternal(Scene* scene, Error& error) {
  MOCHI_ERROR_IF(!scene, error, "Invalid scene");
  MOCHI_ERROR_RETURN(error);
  auto* sceneImpl = assert_cast<SceneImpl*>(scene);

  // Skip if the scene is already differentiable.
  auto& reg = sceneImpl->GetRegistry();
  if (reg.try_ctx<TagDifferentiableScene>()) {
    return;
  }

  // The adjoint keeps a single step of state history and treats the stage solution as the
  // step-end state, so backward Euler is the only supported integrator.
  MOCHI_ERROR_IF(
      sceneImpl->GetSolverParams().integrationMethod != IntegrationMethod::BackwardEuler,
      error,
      "Differentiable scenes require SolverParams.integrationMethod to be backward Euler.");
  MOCHI_ERROR_RETURN(error);

  // Check all actors are differentiable.
  sceneImpl->ForEachActor([&](Actor* actor) {
    auto e = GetEntity(reg, actor->GetHandle(), error);
    MOCHI_ERROR_RETURN(error);
    if (actor->IsStatic() || actor->GetType() == ActorType::Rigid) {
      return;
    } else if (actor->GetType() == ActorType::Articulated) {
      articulated::compound::ValidateDifferentiabilitySupport(reg, e, error);
    } else {
      MOCHI_ERROR_SET(error, "Actor type is not differentiable.");
    }
  });
  MOCHI_ERROR_RETURN(error);

  // Apply robust convergence settings. Log warnings if scene or actor settings are overridden.
  sceneImpl->ApplyImprovedConvergenceSettings(/*logWarnings*/ true);

  // Loop through the scene actors and emplace differentiability components.
  sceneImpl->ForEachActor([&](Actor* actor) {
    auto e = GetEntity(reg, actor->GetHandle(), ErrorAssert{});
    if (actor->IsStatic()) {
      // Skip static actors.
      return;
    } else if (actor->GetType() == ActorType::Rigid) {
      // Handle rigid actor
      InitDifferentiableRigidActor(reg, e, reg.all_of<TagArticulatedLinkActor>(e));
    } else {
      // Handle articulated actor
      MOCHI_ASSERT(actor->GetType() == ActorType::Articulated);
      articulated::compound::InitDifferentiableActor(reg, e, ErrorAssert{});
    }

    // Emplace differentiability components for the actor's island.
    auto island = reg.get<CIslandMemberInfo const>(e).island;
    island::InitDifferentiableIsland(reg, island);
  });

  // Create a tag of differentiability
  reg.set<TagDifferentiableScene>();

  // Create default solver params
  reg.set<CBackPropagationSolverParams>();

  // Create the component holding the metrics of the last back-propagation step
  reg.set<CBackPropagationSceneStats>();

  // Create the component to store state pairs
  reg.set<CStatePair>();
}

MOCHI_API void diffsim::MakeSceneDifferentiable(Scene* scene, Error& error) {
  MakeSceneDifferentiableInternal(scene, error);
}

void experimental::RestoreStateFromScene(
    Scene* sceneTo,
    Scene const* sceneFrom,
    StateHandle handleFrom,
    Error& error) {
  MOCHI_ERROR_IF(!sceneFrom, error, "Invalid scene");
  MOCHI_ERROR_IF(!sceneTo, error, "Invalid scene");
  MOCHI_ERROR_RETURN(error);

  Span<uint8_t const> stateBuffer =
      assert_cast<SceneImpl const*>(sceneFrom)->FindState(handleFrom, error);
  MOCHI_ERROR_RETURN(error);

  capture::RestoreState(assert_cast<SceneImpl*>(sceneTo)->GetRegistry(), stateBuffer, error);
}

#define MOCHI_RETURN_IF_NOT_DIFFERENTIABLE(CONST_QUALIFIER, ...)                                  \
  MOCHI_ERROR_IF(!scene, error, "Invalid scene");                                                 \
  MOCHI_ERROR_RETURN(error, __VA_ARGS__);                                                         \
  auto CONST_QUALIFIER* sceneImpl = dynamic_cast<SceneImpl CONST_QUALIFIER*>(scene);              \
  MOCHI_ERROR_IF(!sceneImpl, error, "Invalid scene");                                             \
  MOCHI_ERROR_RETURN(error, __VA_ARGS__);                                                         \
  auto CONST_QUALIFIER& reg = sceneImpl->GetRegistry();                                           \
  MOCHI_ERROR_IF_NOT(reg.try_ctx<TagDifferentiableScene>(), error, "Not a differentiable scene"); \
  MOCHI_ERROR_RETURN(error, __VA_ARGS__);

// [Differentiability] Get solver parameters.
MOCHI_API diffsim::BackPropagationSolverParams diffsim::GetBackPropagationSolverParams(
    Scene const* scene,
    Error& error) {
  MOCHI_RETURN_IF_NOT_DIFFERENTIABLE(const, {});
  return reg.ctx<CBackPropagationSolverParams>();
}

// [Differentiability] Set solver parameters.
MOCHI_API void diffsim::SetBackPropagationSolverParams(
    Scene* scene,
    BackPropagationSolverParams const& params,
    Error& error) {
  MOCHI_RETURN_IF_NOT_DIFFERENTIABLE(, );
  reg.ctx<CBackPropagationSolverParams>() = params;
}

// [Differentiability] Get the performance metrics of the last back-propagation step.
[[nodiscard]] MOCHI_API diffsim::BackPropagationSceneStats diffsim::GetBackPropagationSceneStats(
    Scene const* scene,
    Error& error) {
  MOCHI_RETURN_IF_NOT_DIFFERENTIABLE(const, {});
  return reg.ctx<CBackPropagationSceneStats>();
}

// [Differentiability] Reset accumulated gradient containers used during backpropagation.
MOCHI_API void diffsim::ResetBackPropagation(Scene* scene, Error& error) {
  MOCHI_RETURN_IF_NOT_DIFFERENTIABLE(, );
  sceneImpl->ResetBackPropagation();
}

MOCHI_API void diffsim::PrepareBackPropagate(
    Scene* scene,
    StateHandle stateNew,
    StateHandle stateOld,
    Error& error) {
  MOCHI_RETURN_IF_NOT_DIFFERENTIABLE(, );
  sceneImpl->PrepareBackPropagate(stateNew, stateOld, error);
}

MOCHI_API void diffsim::BackPropagate(Scene* scene, Error& error) {
  MOCHI_RETURN_IF_NOT_DIFFERENTIABLE(, );
  sceneImpl->BackPropagate(error);
}

MOCHI_API void diffsim::GetStepJacobian(
    Scene* scene,
    StateHandle stateNew,
    StateHandle stateCurr,
    StateHandle stateOld,
    Span<real> outJacCurr,
    Span<real> outJacOld,
    Error& error) {
  MOCHI_RETURN_IF_NOT_DIFFERENTIABLE(, );
  sceneImpl->GetStepJacobian(stateNew, stateCurr, stateOld, outJacCurr, outJacOld, error);
}

#undef MOCHI_RETURN_IF_NOT_DIFFERENTIABLE

MOCHI_API experimental::DebugStats experimental::GetDebugStats(Scene const* scene, Error& error) {
  MOCHI_ERROR_IF(scene == nullptr, error, "Invalid scene pointer");
  MOCHI_ERROR_RETURN(error, {});
  return assert_cast<SceneImpl const*>(scene)->GetDebugStats();
}

Actor* CopyActorToSingletonScene(Actor const* actor, Context* context, Error& error) {
  Scene const* originalScene = actor->GetScene();

  // Export the actor to a temporary directory (writes .mochi_scene + .h5 meshes).
  auto tempDir = CreateTempDirectory("actor_clone", error);
  MOCHI_ERROR_RETURN(error, nullptr);
  prefab::ExportActor(actor, "actor", tempDir.Path().string(), error);
  MOCHI_ERROR_RETURN(error, nullptr);

  // Reload the freshly-written prefab, then create a new scene to host the cloned actor.
  std::filesystem::path const prefabFilePath = tempDir.Path() / "actor" / "actor.mochi_scene";
  auto scenePrefab =
      prefab::LoadFromFile(prefabFilePath.string(), tempDir.Path().string(), context, error);
  MOCHI_ERROR_RETURN(error, nullptr);

  Scene* newScene = context->CreateScene("singleton_scene");
  MOCHI_DEFER(if (!error.IsOK()) { context->DestroyScene(newScene); });
  newScene->SetGravity(originalScene->GetGravity());
  newScene->SetSolverParams(originalScene->GetSolverParams(), ErrorAssert{});

  // Instantiate the cloned actor in the new scene.
  auto const result = prefab::AddToScene(scenePrefab, newScene, {}, error);
  MOCHI_ERROR_RETURN(error, nullptr);
  auto const articulatedActors = result.Filter(ActorType::Articulated);
  MOCHI_ASSERT(
      !articulatedActors.empty(), "Exported prefab must contain at least one articulated actor");

  return articulatedActors[0];
}

} // namespace mochi
