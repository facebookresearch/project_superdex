# Changelog – Project SuperDex

All notable changes to this repository will be documented here.

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
