# Changelog – Project SuperDex

All notable changes to this repository will be documented here.

## [1.1.0]

This release adds Unreal integration and repeatable asset tuning, expands robot
and soft-body authoring, and improves contact and solver behavior.

### What's New

- **SuperDex in Unreal (early preview).** A source project brings physics,
  robotics, editor tools, and manipulation examples to Unreal Engine 5.5.
  `hydrate_plugins.py` stages native dependencies; `package_plugins.py` packages
  reusable plugins. Generated parameter/model types preserve native defaults
  and optional values. Win64 is supported; Linux is for development. Standalone
  Meta Quest packaging with UE 5.8 is experimental.
- **Tracked-hand robot control.** Author inverse kinematics configurations in SuperDex
  Studio to retarget OpenXR hand poses to robot hands and grippers. Supplied example
  configurations connect tracking to Wuji, Tesollo DG-5F, and Robotiq 2F-85 controllers.
- **TeleopNodeZenoh.** An optional, off-by-default plugin provides a basic
  zenoh-pico and Protocol Buffers backend for teleoperation across processes or
  machines on Win64 and Android.
- **Blueprint episode recording on Win64.** Record simulation state, application
  transforms and skeletons, text events, and scalar values into Rerun episodes. Users
  can define additional channels with their own data and structures. Native recorder
  builds also read and restore rigid, articulated, standalone soft, and rod state and
  pack episodes into `.superdex_framebank` files. Restoration cold-starts integration;
  replay is not bit-exact but is close to float precision. The separate `rrd_video`
  source tool renders existing complete episodes to video without re-simulating them.
- **Combined bot scenes and tasks.** Load bots, physics prefabs, controllers,
  and supported sensors together from `.superdex_scene` files. In Studio, author
  controllers and task spawns and preview nominal `.superdex_task` reset states.
  Task APIs apply caller-supplied pose offsets and prefab choices for each run.
- **Reusable scene archives and snapshots.** Edit/save scene archives in Studio.
  Native scene loading can apply asset snapshots to supplied pinned file versions;
  this is not an automatic asset-version creation or download service.
- **Asset Tuner.** SuperDex Asset Tuner is a physics workbench of sorts that is
  used primarily for tuning and calibrating various physics parameters related
  to rigid bodies and articulations in a prefab. The tuner is designed to run
  repeated experiments with boundary conditions (loads) on the same scene/prefab
  multiple times with different parameters (overrides) to achieve the desired behavior.
  Asset Tuner has feature-rich integration within Studio and supports `.rrd` (Rerun)
  and `.csv` recording. Scene tuners have experimental support for driving
  `.superdex_bot` bots with waypoints and joint keys in `.superdex_task` environments.
- **More complete robot imports.** Studio imports MuJoCo MJCF and URDF robots with
  primitives and multiple visual/collision geometries per link as a unioned or
  shrink-wrapped mesh, retaining supported materials and textures. Collision
  processing recipes can be reopened and rebuilt. Unsupported MJCF features are
  reported; this is not full MuJoCo simulator compatibility.
- **Soft-body authoring.** Studio can tetrahedralize surface and CAD models,
  extract their boundaries, embed separate visual/contact meshes, and export soft
  models. Constrained Delaunay tetrahedralization preserves boundary shape,
  allowing subdivision unless "Strict Boundary" is selected. Compatible textured
  GLB render models can deform with soft actors in Studio and Unreal.
- **Prefab editing and inspection.** Author and visualize supported joint, pivot,
  and attachment constraints in Studio and import supported prefab constraints and rod
  actors into Unreal. SDF and embedded-mesh views, geometry statistics, and hide/isolate
  controls make complex assets easier to inspect. Robot skin visibility can be
  controlled independently of collision coverage. `hiddenLinks` controls skin
  visibility, independently of `nonCollidingLinks`; unset and empty are distinct.
- **Elasto-plastic friction by default.** Contacts with rigid colliders, static or
  dynamic, including articulated links, now use elasto-plastic Coulomb friction. A
  contact holds a tangential load below the Coulomb limit without slip, so held objects
  no longer creep. Contacts with deformable colliders keep velocity-regularized friction.
  So do all contacts with `DIRK23`, `SymplecticDIRK12` and `SymplecticDIRK22`, whose
  steps do not end at their last stage. Elasto-plastic friction uses
  `coulombFrictionCoefficient` and the new `ContactParams::preslidingDisplacement`
  (defaults to `1e-4` m): friction ramps up to full strength over this tangential
  displacement, and an unloaded contact springs back by up to that displacement.
  Within this displacement range, friction adds no damping. `frictionFalloffVel`
  now applies only to velocity-regularized contacts. A pair uses elasto-plastic friction
  only if both actors prefer it. To turn it off, set the experimental
  `ContactParams::preferElastoPlasticFriction` (defaults to `true`) to `false` on either
  actor, or override the preference per pair through `ContactPairParamsOverride`.
- **Elasto-plastic joint friction.** Dry friction in articulated joints now holds a load
  below its static peak without creep. It ramps up over the new
  `ArticulatedJointFrictionParams::preslidingDisplacement` (defaults to `1e-4` m or
  rad).
- **Per-actor linear stopping.** Opt-in
  `LinearSolverConvergenceNorm::ResidualPerActorWeighted` stops Newton's linear solves
  relative to each actor's own residual, so lightly loaded or stiff actors are no longer
  under-solved. When using this norm, a negative `LinearSolverParams::absTol` stops
  the linear solves at 1% of Newton's tolerance.
- **More capable rods and shells.** Rod creation is now in the main native API.
  Rod pose getters/setters preserve element orientation, and prefab export can
  capture current rod poses. A rod-element rotation constraint avoids a dummy
  rigid body. Rods gain a default axial-collapse barrier; shells gain optional
  tension-only strain stiffening, disabled by default.
- **Rods in Studio.** View rod (polyline) models and create, edit, and simulate rod
  actors. Drag rods by their centerline, and wrap rod models in contact skins and visual
  meshes.
- **Profiling in Physics Debugger.** The Physics Debugger now includes a real-time
  profiler with a scrolling graph of simulation step times, helping you spot performance
  trends and spikes. Hover over a step to see details, or use the mouse wheel to zoom.
  Enable single-threaded simulation to see each step broken down by cost category and
  identify where time is spent. The profiler appears at the bottom of the debugger; if it
  isn't visible, choose "Reset Layout".
- **SuperDex CAD Exporter.** Support for exporting `.superdex_prefab` directly and
  workflow improvements for NX.
- **SuperDex Recorder.** The Rerun-based recorder in `superdex_recorder/`, with C++ and
  Python APIs for recording physics and robotics episodes, now ships in the Python
  wheels and is built by default. It captures rigid, articulated, soft-body, and rod
  state, robot controllers, and optional contact forces in standard `.rrd` files
  viewable with the regular Rerun viewer. Chunked background writing and transactional
  commits support efficient episode recording. `EpisodeReader` provides selective frame
  reads, while `FrameBank` and `EpisodeRestorer` enable "Kinematic Resume" so learning
  curricula can initialize rollouts from recorded mid-episode states. The included
  `rrd_video` tool generates real-time video clips directly from `.rrd` recordings using
  headless Rerun rendering, with automatic camera framing and support for selecting a
  time range.
- **New robot assets.** Tianji Marvin M6 arms are published under CC BY 4.0. Arm-hand
  combinations for Tianji M6 + Wuji Hand 2 and FR3 V2 + DG-5F SEED are available in
  SuperDex and Unreal, as are OpenXR hand bots.
- **Robotics scenes.** Load scenes that specify prefab assets, robots, controllers
  and sensor graphs. Configure robot placement and initial joint poses, select robots
  to instantiate, and spawn or remove prefab instances.
- **Domain randomization in Robotics.** Derive deterministic seeds for named
  quantities and random streams, and draw from common distributions. Save
  per-episode seeds, draws, sampled values, applied values and other bookkeeping metadata.
- **Robotics SensorGraph.** Connect numeric observation sources and transforms
  into a graph that synchronously runs in dependency order. Current source sensor
  types include contact forces, joint positions and velocities, joint wrenches,
  and tactile responses. Transforms include aggregation, concatenation, delay, history,
  spatial mapping and noise. Async sensors (e.g., cameras) are deferred.

#### Performance

Optimizations target rigid, articulated, soft, shell, and rod workloads, both with and
without contact. Shell and cloth scenes are 14–25% faster, soft scenes 4–9% faster,
rod scenes 16–20% faster, and soft actors with contact skins up to 8x faster. Gains
depend on the scene and settings.

- Reduced contact and solver setup work for articulated and soft/shell contact skins.
- Reduced cloth/rod point-cloud contact work, including self-contact.
- Reduced repeated articulated-body calculations.
- Reduced scheduling overhead in parallel stepping.
- Faster contact Jacobians for soft, shell, cloth, and rod actors.
- Avoided excessive temporary storage and computation in large BFGS/SR1 solves.
- Rotation-aware AMG preconditioner for soft and shell actors.
- Opt-in `AdaptivePerActor` to automatically select the most appropriate
  preconditioner type for each actor.
- Reduced fixed per-step overhead in small rigid scenes.
- Lock-free scene lookup improves scaling when many threads step many scenes.
- Faster contact detection against a soft actor using an SDF collider.

### Fixes

- Improved accuracy in small-rotation calculations, mesh mass properties, shell bending,
  and world-space constraints and soft-body recentering far from the origin.
- Fixed missed contacts on tilted planes, shared-axis boxes, and stage-start point
  clouds.
- Fixed inconsistent fixed-DoF handling and invalid nested-actor setups that
  could corrupt state or produce incorrect constraints. Apply articulated-link
  boundary conditions through the articulation root. Nested soft actors require
  `ColliderType::None` and reject deformable-node constraints; free joints reject
  nonzero friction.
- Fast-spinning or slender rigid bodies and free or spherical joints, including joints
  dominated by viscous damping, no longer produce singular or uphill Newton steps: their
  rotational inertia and damping Hessians stay positive definite.
- With `frictionWithColliderNormal = false`, friction against soft and skinned SDF
  colliders uses a unit normal, so its forces match its energy and its Hessian stays
  positive definite.
- Line searches account for the work of friction, contact damping, external torques, and
  Newton-Euler inertia, reducing solver freezes and failed searches by up to 10x.
- Rope-mode displacement-control and McKibben actuators have continuous energies where
  they go slack, so merit line searches no longer reject steps across that point.
  McKibben actuators stay slack at negative muscle length instead of pulling ever harder.
- Quasi-static and inertialess-soft steps avoid infinite-time-step inertia NaNs
  and incorrect extrapolation. Sleeping islands wake for quasi-static solves;
  IK disables sleeping.
- Prescribed static-body motion and controller/constraint targets retain coherent
  motion history through state capture.
- Adaptive preconditioner policy state is captured rather than restarted on restore.
- GLB reimport refreshes materials/textures without conflating differently
  colored same-named assets. Articulation reimport unregisters skeletal meshes
  before replacing their render data. Paused preview changes preserve poses.
- Shrink Wrap converts millimeters to meters correctly; unfused STEP sources are
  no longer unnecessarily round-tripped. Sphere-pole meshing and visual normals
  are corrected.
- Python arrays bound in submodules pickle correctly.
- Growing a Python `DynamicArray` from its own elements no longer reads freed memory.
- Nested soft reduced-order actors no longer miss contacts, actors with a negative
  contact threshold no longer sink into colliders, and implicit sphere and plane rigid
  shapes now respect `ColliderType::None`.
- Destroying a nested soft actor individually is rejected with a warning instead of
  crashing, and articulations with several nested soft actors step deterministically.
- Fixed one-sided joint limits on all-negative axes, zeros from `GetDofValues` on
  articulated links, and spurious link velocities after a diverged step.
- Experimental reduced-order soft actors no longer gain energy or stall under `BDF2`,
  `BDF3`, and `DIRK22`.
- `QueryNodesInVolumeLocal` no longer crashes before the first step.
- `superdex.physics.mesh` now imports from the installed `superdex-physics` wheels.

### API Changes

- **Newton defaults:** `LineSearchType::Default` is now `WolfeStrong`, rather than
  `ResidualNorm`; `lineSearchMaxIter` increases from `4` to `8`. Nonlinear `relTol`
  changes from `1e-6` to `0`, disabling relative stopping. Set these explicitly
  when relying on prior behavior. `ArmijoOrResidualNorm` and
  `LinearToleranceStrategy::EisenstatWalker3` are removed.
- **Contact behavior:** `ContactParams::alignmentSmoothingFraction` (defaults to `0.1`)
  eliminates the abrupt cutoff of contacts due to normal alignment, which helps solver
  convergence. Used only with implicit normals; not needed with explicit normals. Set it
  to `0` for the previous abrupt cutoff. The all-parameters constructor adds
  `collidingPenaltyThresholdPadding`, `alignmentSmoothingFraction`,
  `preferElastoPlasticFriction`, and `preslidingDisplacement`; update positional calls.
  Existing normal/friction heuristics are now marked experimental.
  `experimental::ApplyImprovedConvergenceSettings` no longer forces
  `frictionWithColliderNormal` on; its default remains true.
- **Friction model:** the experimental `CoulombFrictionModel` enum and
  `ExperimentalEvalParams::frictionModel` (Python `friction_model`) are removed; default
  simulations don't change. Delete any references, and resave prefabs that store
  `frictionModel`, which now load with a warning.
- **Joint friction:** `ArticulatedJointFrictionParams::falloffVel` is deprecated and
  ignored; tune the new `preslidingDisplacement` instead. The new field precedes
  `falloffVel`, so update positional C++ initializers. `NewtonEulerTerms` now excludes
  joint friction.
- **Articulated skins:** skin shapes and nested soft actor shapes are now defined in the
  articulated actor's frame, not the root link's. Re-author skins whose root joint or
  root link has an offset.
- **Query refresh and timing:** use `Scene::UpdateQueries(error)` or Python
  `scene.update_queries()` to refresh registered queries without simulating a
  step. Forces are evaluated automatically when registered queries need them.
  `Step(0)` does the same refresh, but it also runs pre- and post-step
  callbacks, and changes made by pre-step callbacks are included. It no longer
  increments the step counter. Both also work after a state restore. Before the
  first step, queries that require contact detection or force evaluation are
  evaluated with a default 10 ms time step.
- **Articulated Jacobian views:** `Actor::GetArticulatedJacobian` returns values
  computed at the call, not a view refreshed by later steps. Call it again after
  stepping when current values are needed.
- **Rod API:** move `RodActorParams`, `RodMaterialParams`, and
  `PointCloudColliderParams` from `mochi::experimental` to `mochi`; replace
  `experimental::CreateRodActor` with `Scene::CreateRodActor`. Use `GetRodPose` /
  `SetRodPose` for node displacements and element frame axes, not generic
  displacement setters. Rod boundary-condition setters now reject rods; use
  node-position and rod-element rotation constraints instead.
- **Recentering:** shell/rod translation recentering is on by default; their
  local displacements and root transforms can change frame. Use world-space
  values when appropriate, or set recentering explicitly. New rod
  `axialCollapseBarrierScale` defaults to `1e-3`; set `0` to disable it. Shell
  `membraneStrainStiffeningScale` defaults to `0`.
- **Static-body motion:** On a static body, `SetRootTransform` prescribes the
  pose at the end of the next step, and the body moves there during the step,
  with a velocity implied by the change of pose. `SetVelocity` overrides that
  velocity: the body moves at constant velocity during the step and still ends
  at the prescribed pose. To teleport a static body without implied motion,
  call `SetVelocity` with zero velocity before the next step.
- **Prefab files:** Studio's New/Export Prefab actions now write `.superdex_prefab`;
  existing `.mochi_prefab` files still open and save in place.
- **Python dependency:** `superdex-physics` no longer depends on scikit-image.
  Declare it yourself if you use it directly.
- **Native library names:** CMake FP64 shared libraries now have a `_double` suffix, and
  AVX-512 builds an `_avx512` suffix; update custom link lines.
- **Robot scene APIs.** `BotScenePrefab` describes a scene; `LoadBotScene` instantiates
  it. `BotScene` provides named component lookup, robot initialization/removal, prefab
  spawning/destruction and task termination criteria.
- **Domain randomization:** Deterministic, keyed 64-bit seeds can be consistently derived
  in C++ and Python. `Uniform`, `Normal` and `UniformQuat` functions draw values from a
  caller-owned RNG state. Per-episode randomized state and its application-specific
  metadata can be serialized and deserialized in C++ and Python.
- **Sensors and SensorGraph:** Declare sensors and their inputs in bot/scene files, or
  construct and connect them programmatically. Create, build and select outputs from the
  `SensorGraph`; it can then be stepped using caller-supplied simulation time, and each
  output's sample is accessed with `Latest`. A `SensorFormat` struct describes each
  output sample's shape, units and labels. Sensors must be removed from graphs before
  they are destroyed. Adding or removing sensors preserves the state of unaffected
  sensors.


## [1.0.1]

This release expands Python support, improves simulation performance, and adds
new tools for recording episodes, authoring robot scenes, and processing CAD.

### What's New

- **Python 3.13 and 3.14 support.** Native wheels now use the CPython stable ABI
  and support standard CPython 3.12+. The nanobind port preserves the Python API
  and enables parallel calls in free-threaded CPython source builds.
- **Experimental sleeping islands.** Settled islands can skip simulation work
  and wake when affected by scene changes. This is a work-in-progress. Enable
  it through `experimental::SleepParams`. Off by default.
- **Experimental independent contact skins.** Rods, shells, and soft bodies can
  use contact meshes independently of their simulation and visual meshes. New
  contact-skin queries expose deformed positions and normals.
- **More robust shells.** A new membrane-collapse barrier guards against
  instability under strong in-plane compression, eliminating solution explosions
  in many scenarios.
- **More control over contact.** Override contact parameters for specific actor
  pairs in code or prefabs. Skinned articulations can also list links whose rigid
  contact is replaced by skin contact through
  `ArticulatedSkinParams::nonCollidingLinks`.
- **Actor preconditioner overrides.** Select a supported preconditioner for each
  actor with `experimental::SetActorPreconditionerType` (Python:
  `experimental.set_actor_preconditioner_type`) when using
  `PreconditionerType::PerActor`.
- **Prefab pose snapshots.** Save and restore soft deformation and articulated
  joint poses with `ExportParams::exportCurrentPose` (`export_current_pose`).
  This is opt-in; velocities are not included.
- **Episode recording for source builds.** The new `superdex.recorder` API writes
  transactional Rerun episodes with rigid and articulated state, robot
  controllers, step durations, body parameters, and optional contact forces.
  `EpisodeReader` supports selective frame reads. Robot and keypoint shadows add
  visualization without changing the simulation.
- **Skinned bots in Studio.** Render and edit bot skins, attach skins to composed
  bots, drag a skin through its nearest link, and export skinned GLB meshes and
  simulation prefabs. A measurement tool measures vertices, triangles, and
  facets, including while simulation is paused.
- **Studio automation.** Run processing pipelines without a window using
  `--process`, redirect exports with `--out`, and explicitly allow conflicting
  exports with `--override`. Capture repeatable screenshots with `--screenshot`,
  camera and framing controls, and a chosen output size. Screenshots require a
  display or Xvfb.
- **Better CAD workflows.** Control STEP mesh density by CAD face color and
  preview models in their authored colors. Mesh exports have improved UV
  packing. The CAD exporter has a DPI-aware UI, restored SolidWorks part export,
  per-link meshing overrides, and an NX installer.
- **Filament rendering for SuperDex Lab.** Lab now uses the Filament-based viewer
  with automatic server launch, per-actor colors, ground planes, grids, and
  working camera controls. Camera roll and background-color issues are fixed.
- **New robot assets.** Added skinned Wuji hand variants, FR3/Wuji and FR3/DG5F
  combinations, and DG5F/Seed mounting-adapter CAD. Wuji collisions and joints and
  the Robotiq 2F-85 gripper have updated tuning. Left-hand Wuji kinematics,
  visual meshes, and colliders have corrected mirroring.
- **Debugger profiling.** A new profiler panel displays per-step timings
  and statistics. If you don't see it docked at the bottom, then use the
  "Reset Layout" option in the menu.
- **Debugger debug-draw.** Enable various debug draw options to visualize
  contacts, forces, bounds, and more. Controlled through the properties panel.
- **Native neural inference.** Native MLPs now support SiLU activation.

#### Performance

- Various optimizations resulting in up to 2X overall performance improvement,
  depending on the scene and settings.
- Soft and shell actors use closed-form element mass and store fewer redundant
  matrices and sparsity patterns, reducing assembly work, memory use, and scene
  creation time.
- Reduced contact work with sphere-tree culling, sort-and-sweep broadphase,
  cached collider bounds, and specialized point-cloud and GridSDF queries.
- Reworked skinned contact assembly around sparse bone maps, Jacobian-free
  sums, and batched SIMD reductions. Articulated Hessians use compact link
  Jacobians and symmetric assembly.
- Reduced PCG and AMG synchronization, improved per-actor preconditioner
  scheduling, and removed redundant solver setup, temporary storage, and copies.
- Nonlinear solver skips redundant retries.
- Expanded opt-in AVX-512 support. AVX-512 remains disabled by default because it
  requires a compatible CPU and it isn't always faster. This is expected to
  improve in future versions. You're welcome to try it out now by building with
  CMake flag `-DMOCHI_USE_AVX512=ON`.
- Improved SIMD processing, small-matrix operations, eigensolvers, and FEM
  assembly.
- Reduced FEM and ECS memory overhead and eliminated unnecessary copies in NumPy
  mesh conversion, and viewer frame reception. Shared-library calls also have less
  dynamic-linking overhead. Renderer capture can reuse caller-owned frame buffers,
  and rigid and articulated boundary-condition updates use fewer temporary
  allocations.

### Fixes

- Articulation pose and velocity resets now update link, soft-skin, and blended
  state consistently. Rods now support `SetZeroDisplacementsAndVelocities`.
- Revolute joint velocities now report exact angle rates at large rotations per
  step. Changing a body's center of mass preserves its root transform, and
  inertia validation accepts valid rotated slender bodies.
- Mass and density queries now work on static articulated links. Setting density
  on shapeless links no longer produces NaN inertia.
- Fixed missed contacts after point-cloud threshold changes, incorrect rod
  contact bounds, and oversized sphere colliders on non-cubic meshes. SDF distance
  queries report correct world positions for recentered soft actors.
- Fixed slow drift of resting rigid actors far from the world origin,
  incorrect forces and normals against rotated soft colliders, and contact
  torques on reduced-order soft colliders.
- Fixed crashes and incorrect force reporting for partitioned skinned contacts.
- Failed IK target replacement and solver creation preserve existing targets
  and scenes. Pose-controller parameter updates are all-or-nothing. IK no longer
  produces NaN actuator energy from an infinite time step or constraint damping.
- Newton-Euler mass and Coriolis terms no longer include contact forces.
- `PsdProjectionMode::Never` is now honored on Newton retries. Non-finite
  line-search values are checked consistently in optimized builds.
- Corrected triangle quadrature weights and Smith neo-Hookean PSD projection
  near rest states.
- Partial-rotation boundary conditions return an error instead of triggering
  an assertion in debug builds.
- Python handles now work as dictionary keys and set members.
- Restoring state now refreshes debugger visual meshes. Static actors support
  visual-node queries, and the debugger handles playback-time resets safely.
- Prefab reload preserves in-memory nested prefabs. Studio refreshes stale
  models, handles inaccessible paths without crashing, and avoids invalid link
  selections.
- Source and wheel builds respect compiler selection. CMake builds handle
  jemalloc compatibility correctly, and Android builds without FP16 arithmetic
  use a supported fallback. CUDA sparse solver builds support cuDSS 0.8.

### API Changes

- **Legacy viewer dependency:** Polyscope is now optional. Install
  `polyscope>=2.5.0,<2.6` separately if you use the legacy viewer. This range
  has no Python 3.14 wheels.
- **Equality comparisons:** constraint and prefab comparisons now include
  derived fields such as actor names and translations. Compare selected
  fields explicitly if your code relied on the previous partial comparison.
  Defaulted C++ prefab equality operators require C++20.
- **Prefab export:** replace `prefab::ExportSceneExcluding` with `ExportScene`
  and `ExportParams::excludeActors`. In Python, use `export_scene` with
  `ExportParams(exclude_actors=...)`. Existing exports without options still work.
- **Newton fallback:** removed `gradientDescentFallback` (Python:
  `gradient_descent_fallback`) from the solver configuration. Default behavior is
  unchanged.
- **Computed inertia:** rigid bodies and articulated links using `P1Q1` boundary
  elements now get exact mesh-derived inertia.
- **Python precision names:** use `fp32` and `fp64` in configuration and precision
  comparisons. `PRECISION_NAME` now returns those names. Existing `double`
  environment aliases and the `double` source-build extra remain accepted.
- **Rod contact:** replace `useVisualMeshContact` with `useContactSkin` and
  `visualMeshContactElementType` with `contactSkinElementType`. In Python, use
  `use_contact_skin` and `contact_skin_element_type`. Move contact geometry and
  its skinning data into `ModelData::contactSkinMesh` (`contact_skin_mesh`).
- **Contact-skin access:** surface-mesh getters no longer return deformable
  contact skins. Use `GetShapeContactSkinMesh` or `Actor::GetContactSkinMesh`.
  Register `ContactSkinNodePositions` and `ContactSkinNodeNormals` for deformed
  data, then read the corresponding contact-skin position and normal getters.
- **Lab environments:** import `superdex.lab.gym` and construct registered
  environments with `gymnasium.make`, for example
  `gymnasium.make("superdex_gym/CartPole-v0", cfg={})`. Update discovery-based
  environment names in scripts, training recipes, and checkpoint configuration
  to the registered `superdex_gym/<Name>-v0` IDs.
- **Lab rendering:** remove `renderer_backend`, `render_coordinate_system`, and
  the sample runner's `--renderer` option. Configure `mochi_renderer_cfg` instead.
  Public builds do not include the new interactive `render_mode="human"` viewer.
  Use `render_mode="rgb_array"` or `None`. Pass `up_dir` explicitly to camera-view
  calls if your observations depend on the previous orientation.
- **Shell material defaults:** `membraneCollapseBarrierScale` now defaults to
  `1e-3`. Set it to `0` to retain the previous plain StVK membrane response.
- **Mesh authoring:** remove nodes that no element references before creating
  soft or shell actors, articulated skins, or blended skins. Contact-skin models
  also require skinning data at shape creation. Blended vertices must have
  matching rest coordinates and skinning transforms in the articulated and
  nested-soft meshes. Previously accepted invalid assets can now return errors.
  Runtime SDF collider creation also rejects empty meshes, oversized grids,
  and unusable cell spacing. Check mesh padding and grid resolution when
  actor creation returns an SDF error.
- **Solver limits:** iterative solvers reject `maxIter = 0`. Use the automatic
  limit or a positive value. The default maximum is now 500 rather than 1000.
  Set `LinearSolverParams::maxIter` explicitly if you need the previous limit.
- **Native linking:** shared Physics libraries no longer export `mochi_core`
  implementation symbols. Link `mochi_core` directly if your application uses
  those symbols. Replace `MOCHI_PMR_USES_JEMALLOC` with
  `MOCHI_PMR_ENABLE_JEMALLOC_COMPATIBILITY` in custom build configuration.
- **Experimental scene recording:** HDF5 scene recording has moved to the
  experimental API. It will probably be removed in the future. All new recording
  efforts are directed toward SuperDex Recorder (`superdex.recorder`), described
  above.


## [2026-08-24]

- Initial release.
