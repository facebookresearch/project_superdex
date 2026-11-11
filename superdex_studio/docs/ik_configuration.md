# IK Configuration

SuperDex Studio can author and preview inverse-kinematics goals for a Bot asset.
An IK configuration is a `.superdex_ik` JSON document. It references its output bot;
preview does not save the bot or change its authored default pose.

## Create and Edit

Select one Bot asset in the Asset Browser and choose **Create IK Configuration...**.
Studio creates a unique sibling document and opens it. The initial goals are the
origins and frames of every leaf link, computed from the bot's authored default
pose. This includes leaves attached through hard joints.

Use **IK Goals** to select, add, remove, or enable goals. Each goal has a
unique display name and output-link name, an input-side constraint frame,
position stiffness in N/m,
and rotation stiffness in N m/rad. Initial stiffnesses are 100 for position and 1
for rotation. Zero stiffness disables that component;
disabling a goal disables both. Each link can have one position constraint and
one rotation constraint. **Output Link Frame From Constraint Frame** edits the output-side attachment's
translation and rotation in the selected output link's coordinates. Its default identity frame matches the link origin and orientation;
an offset frame can match a tool tip or tool frame.

Each transform has a **Zero** button that resets its local translation and rotation
to identity. **Match Input Constraint Frame** sets the output-side constraint frame to the
resolved input-side world constraint pose;
**Match Output Constraint Frame** sets the input-side constraint frame to the
current output-side world constraint pose. Matching uses
the output bot's displayed pose (configured when stopped, solved when paused) and
includes the input bot's kinematics, root placement, and retarget scale when bound.
Only the clicked frame changes; the other frame and joint configurations stay intact.
Matches require resolvable output-bot and input-bot bindings. All four actions support
undo/redo and save.

Select a goal in the list or pick its viewport marker. Translate and rotate it
with the existing viewport gizmos and snapping controls. Picking a robot link
selects its associated goal, if present. Goal scale gizmos and physics-force
dragging are not available.
**Reset Goal to Initial Output Constraint Pose** restores that goal to the
attachment frame's pose in the authored initial configuration, preserving the
output-side constraint frame. Refreshing or reopening never regenerates authored goals.

Save and undo/redo use the normal Studio commands. Numeric and gizmo drags form a
single undo operation. Selection is editor-only; solved robot poses are transient.
Read-only documents support inspection and preview, but not pose or goal editing.

## Output Bot Pose

In **IK Solver**, expand **Output Bot Pose** below the Output Bot reference to edit the output
bot's starting joint configuration. Both **Output Bot Pose** and **Input Bot Pose**
are collapsed by default. The joint controls and Zero/Min/Max/Mid/Random shortcuts
match Bot Details. Output-bot-pose controls are disabled while running or while
recording playback owns the pose (including Pause and Step).

The output configuration is saved as `outputBotConfiguration` in the IK document, not
in the bot asset. An omitted or empty array uses the bot's authored default pose.
The stopped viewport displays this starting configuration, including skinned
geometry and output attachment frames; goals remain unchanged.
**Reset Output Bot Pose** clears the override and restores the authored default.
Selecting a different Output Bot also clears the old override.

Pause and single-step retain the solve result for inspection. Editing Output Bot Pose
while paused ends that preview and displays the new starting configuration.
Pose edits and Reset support normal undo/redo and save/reload.

## Coordinate Frames

The two constraint frames are authored in different reference frames:

- `linkFrameFromConstraintFrame` maps the output-side constraint frame into the
  selected `outputBotLink`. **Output Link Frame From Constraint Frame** edits it.
- `baseOrInputLinkFrameFromConstraintFrame` maps the input-side constraint frame
  into the selected `inputBotLink`, or the output bot's authored initial base when
  unbound. **Output Base / Input Link Frame From Constraint Frame** edits it.
- `outputBaseFrameFromInputBaseFrame` maps the input bot's base frame into the
  output bot's authored initial base. Its UI label is **Output Base Frame From
  Input Base Frame**.

The output-side world constraint pose is
`worldFrameFromOutputLinkFrame * linkFrameFromConstraintFrame`. The unbound
input-side pose is `worldFrameFromOutputBaseFrame *
baseOrInputLinkFrameFromConstraintFrame`; a bound pose uses the input link's
world frame instead. The solver matches these two world-space constraint poses.
The output base anchor is captured when the preview prepares the output bot, so a
floating output root can move without dragging unbound authored constraints.

These are the same transforms as earlier schemas: this naming change does not
invert any transform, change axes, alter scaling, or change multiplication order.
Gizmo and Match actions still use the inverse of the resolved reference anchor to
store the edited local constraint frame.

Current saves use version 9 and emit `outputBot`, `outputBotConfiguration`,
`inputBot`, `inputBotConfiguration`, `outputBaseFrameFromInputBaseFrame`, and
`inputBotScale`. Each goal uses `outputBotLink`, `inputBotLink`,
`linkFrameFromConstraintFrame`, and `baseOrInputLinkFrameFromConstraintFrame`.
`goals`, names, enabled flags, stiffnesses, and joint couplings keep their semantics.

Versions 1-8 remain strict legacy inputs. Versions 1-7 retain their historical
`bot`, `botConfiguration`, `targets`, `targetBot*`, `baseFromTargetBot`, and
`targetFrame` vocabulary; version 1 position/rotation vectors become a constraint
transform. Version 8 retains `bot`, `botConfiguration`, `referenceBot*`,
`baseFromReferenceBot`, `goals`, `link`, `linkRelativeFrame`, and `goalFrame` as
migration input only. Older omitted `linkRelativeFrame` uses identity. Versions 1
and 2 load with unit input-bot scale; versions 1-3 load without an output-pose
override. Migration preserves numerical transforms and tuning. Native C++ APIs
have no old-name aliases. Unknown fields, duplicate/mixed vocabulary, and malformed
values fail transactionally without replacing the current document.

Versions 5 and 6 discard their historical `targetBotRecording` references with a
warning. Versions 7 and 8 reject recording keys, and version 9 does not introduce a
persisted `inputBotRecording` field. Recording selection remains local Studio state.

## Retargeting From Another Bot

In **IK Solver**, select an optional **Input Bot**. It appears in the main
viewport with a translucent blue material. **Output Base Frame From Input Base Frame** places its
root relative to the output bot's initial base. Click any part of the input bot to
select it and translate or rotate its root with the viewport gizmo. **Input Bot
Uniform Scale** scales its geometry, internal kinematic translations, and bound
input-side constraint-frame translation offsets about that root. Root placement and all
rotations are unchanged. Scale must be finite and positive; nonuniform scale and
scale gizmos are not supported.
Expand **Input Bot Pose** for the same joint sliders and pose presets as Bot
Details. These settings are saved in the IK document; they do not modify the
input bot asset. Input-bot-pose controls remain editable while running without a
recording. While recording playback owns the pose, **Input Bot Pose**, its reset,
**Output Base Frame From Input Base Frame**, and the input-bot root gizmo are disabled, including
while paused or single-stepping. **Stop** restores pose authoring; goals
and uniform retarget scale remain editable.

**Snapshot Input Bot Pose**, beside **Reset Input Bot Pose**, captures the input pose
currently applied in the main viewport into the IK document. It remains available
while recording playback is running or paused; read-only documents, missing input
models, and incomplete or invalid displayed poses disable it. A manual default pose
is materialized as explicit joint coordinates.

The snapshot writes only `inputBotConfiguration` and
`outputBaseFrameFromInputBaseFrame`, using the output bot's authored initial base
anchor, not its solved floating root. It does not scale joint values, change the
input scale or model default, copy the output solve, or modify any goal/coupling.
Both input fields are one undo/redo edit and survive save/reopen without a recording
reference in the document. Repeating an already-authored pose adds no undo entry.

Snapshot stops playback and returns the input pose controls to authoring, but keeps
the selected RRD, loaded clip, local recording preference, and current output
preview. The playback cursor returns to the first sample; the next Run uses the
captured root as its new authored anchor without applying the previous wrist delta
twice. Reset remains disabled while recording playback owns the pose. Snapshot does
not infer tracking confidence from simulated recording poses.

For each selected IK goal, choose an **Input Bot Link** or **Output Bot Base**.
A bound goal resolves through `worldFrameFromOutputBaseFrame * outputBaseFrameFromInputBaseFrame`, the input
bot's forward kinematics, and `baseOrInputLinkFrameFromConstraintFrame`. The retarget scale multiplies the
root-relative link translations and the link-local input constraint-frame translation.
An unbound goal remains `worldFrameFromOutputBaseFrame * baseOrInputLinkFrameFromConstraintFrame`. Changing the selected
reference preserves the goal's current world pose.
Edit **Output Base / Input Link Frame From Constraint Frame** to adjust the offset in that
reference. Clearing the input bot converts bound goals back to the output base frame.
Unknown links or an incompatible input-bot pose prevent solving and can be repaired
in the editor. Both bot references are tracked across renames and Save As.

At runtime, `BotIK::SetInputBotConfiguration` accepts a prepared input bot,
joint positions excluding root DOFs, its explicit world root placement, and the
`BotIKGoalConfig` bindings. It computes FK and replaces the IK constraints;
call `SolveIK` afterwards. Its scale-aware overload accepts the uniform retarget
scale after root placement; the original overload uses scale 1. Empty joint
positions use the input bot's default pose.
Disabled or zero-stiffness components are cleared. Invalid input preserves the
previous constraints. Unbound goals stay relative to the output bot's authored initial
base, including when the output bot has a floating base.

The bot reference is relative to the IK document, not the process working
directory. Studio tracks output-bot and document renames and moves. A missing output bot
or unknown goal link remains visible for repair, but prevents solving. Duplicate
names and link goals in externally edited documents are diagnosed rather than
silently replacing constraints. Unsupported versions and malformed values fail
loading without replacing the active document.

## Joint Couplings

In **IK Goals**, **Joint Couplings** has Enabled, Joint A, Joint B, Ratio,
Stiffness, and Remove controls. Choose two distinct, nonroot, single-DOF source
joints of the same kind (revolute or prismatic), then **Add Coupling**. Joint names
are case-sensitive; duplicate unordered pairs are rejected. New rows default to
ratio 1 and stiffness 1, in N m/rad for revolute joints or N/m for prismatic joints.

A coupling is a bilateral soft penalty `q_B ~= ratio * q_A`, with zero offset:
both joints may move, and the relation is not a hard mechanical linkage. Ratios
must be finite and nonzero; negative ratios are allowed. Zero stiffness or Disabled
makes a row inactive; inactive rows may retain stale joint names. Edits stop/reset
preview while retaining authored poses and the loaded session recording. Couplings
support undo/redo and save; read-only documents allow inspection only.

## Session Recording

Assign a **Input Bot**, then drop one existing `.rrd` file from the Asset Browser
onto the IK viewport, or use **Browse RRD** in **IK Solver**. **Clear RRD** returns
to manual input-bot pose; **Reload RRD** reloads the selected file. This design-only
selection is remembered per IK document in local Studio preferences, not in
`.superdex_ik`, its undo history, or runtime asset references. Selection and playback
are available for read-only documents. An unavailable remembered file reports a
warning and falls back to manual input-bot pose.

Playback reads native `PhysicsRecorder` scene RRDs, including the two-hand
`SuperDexRecorderPhysicsSceneRRD` capture. Select the left or right Input Bot;
Studio selects exactly one compatible recorded articulation using the native
`robot_prefab_name` joined through the original `robot_id`/`runtime_name`. Without
robotics metadata it requires an exact native actor, runtime, or spawn identity.
It never guesses a hand from joint names (left and right can have identical layouts)
or arbitrarily chooses a duplicate actor. Missing, ambiguous, or incompatible
articulations report their identity/layout diagnostics rather than enabling playback.

The native `data/physics/actors/<encodedId>/joint_layout` supplies joint names,
types, DOF offsets, and translation/rotation counts. Studio maps each nonroot joint's
components into input-bot DOF order and validates that spans partition the
recorded `num_dofs`. Hard joints add no coordinates; a free root is decoded from its
own declared span, not by blindly stripping six values. The complete
`superdex.physics.joint_positions`, `root_translation_xyz`, and
`root_quaternion_xyzw` fields must align on exact native integer `frame` and
`sim_time` keys, even when stored in separate chunks. Duplicate frames, time reversal,
missing fields, incorrect widths, nonfinite values, static state, and nonunit
quaternions fail the entire clip. Episode coordinates must be static FLU/meters.

These are **physics-scene articulation poses**, not raw tracked targets or controller
targets. Every accepted snapshot has a pose; there is no invented tracking-loss or
confidence signal. Native model identity and joint layout are checked, but this
recording does not store joint axes/rest frames: the selected input asset supplies
those FK conventions. Geometry/native size is not a compatibility requirement:
runtime hand scaling can differ from the input asset. Playback does not change
the authored Input Bot Uniform Scale or rescale native joint/root coordinates.
Input roots require free/hard joints with identity rest
frames. Fixed-base preview keeps the authored input-bot root. Floating-base
preview composes the recorded actor anchor with its native free-root xyz/rotation
vector, then applies simulated wrist motion relative to the first effective snapshot.
This matters when the actor anchor is constant but its free joint moves.

Recordings are supplied separately and are not bundled with the open-source
assets. Both left and right OpenXR Input Bots can play their own articulation
from the same compatible bilateral physics-scene recording.
The retired custom `projected_target` capture rows are not a fallback format.

**Run** plays at 1x looping while solving output IK; **Pause** retains both preview
poses, **Step** advances to the next recorded timestamp and solves once, and
**Stop** resets playback and restores authored poses. Only enabled goals bound to
input-bot links transfer recording motion to output IK. Native decoding eagerly
loads the clip on a background task; sampling performs no file I/O, but IK solving
remains synchronous on the UI thread. **IK Solver** shows load/compatibility,
articulation availability, and playback-time diagnostics. Files have a 1 GiB limit;
decoded clip construction has a 256 MiB result/pose budget. Native eager decoding's
internal allocations are not capped or cancellable through the public reader API.
The viewport shows solve status/errors,
the selected RRD filename, and measured solve time.

## Frame Visualization

**IK Constraint Frames** displays both sides of each constraint in the viewport's
working frame. Larger XYZ/RGB axes show the current output-link attachment frame
(`worldFrameFromOutputLinkFrame * linkFrameFromConstraintFrame`). Smaller axes show the fully resolved goal,
including the input bot's pose and scale where bound. Goal axis colors have a
+30-degree hue shift, preserving saturation and brightness, so coincident frames
remain distinguishable.

Attachment axes are approximately 10% of the output robot's largest authored
visible extent; goal axes are 80% of that length. Thickness and picking markers
scale proportionally. Sizes stay stable while solving or moving the input bot.
Missing geometry falls back to the source's kinematic extent. Disabled goals
retain dimmed frames. Output attachments follow the solved pose during preview
and return to the configured starting pose on Stop; resolved goals remain anchored to their
authored input-link or output-base reference.

## Preview

**IK Solver** exposes both creation options and every numerical solver setting.
New documents use a fixed base, exclude collision shapes, and use the native
iteration defaults with a 10 ms elapsed-time budget. Changing fixed-base or
collision-shape options recreates the private solver from the configured starting pose.
Numerical settings and goal edits apply at the next solve tick.

- **Run** repeatedly solves at approximately 60 Hz, warm-starting from the current
  configuration. Goals remain editable while running.
- **Pause** retains the solved pose without solving.
- **Step** performs one solve and stays paused, creating a solver when needed.
- **Stop** releases the solver and restores the configured Bot Pose without
  resetting edited goals.

Preview is synchronous on the UI thread. There is no physics integration time
step and no catch-up solving after slow frames. The elapsed-time setting is a
solver termination budget, not hard preemption. An unlimited budget may block
interaction; keep a bounded budget for live editing and inspect measured solve
time in the viewport.

An unreached result is a best-effort finite pose, not a fatal error. It is shown
while live editing continues. A real error pauses solving and preserves the last
valid display. With no active goals, the current pose is retained and the solver
is not called. Switching editor tabs pauses preview. Closing or refreshing releases
its private solver before the shared Studio physics context is destroyed.

IK preview is quasistatic and does not simulate contact or deformable skins.
Rigid links and GPU-skinned surfaces follow the solved configuration in both
render and collision display modes. Solutions are not exported back to the bot.
