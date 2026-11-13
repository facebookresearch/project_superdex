# SuperDex Asset Tuner

SuperDex Asset Tuner is a physics workbench of sorts that is used primarily for tuning and calibrating various physics parameters related to rigid bodies and articulations in a prefab. The tuner is designed to run repeated experiments with boundary conditions (loads) on the same scene/prefab multiple times with different parameters (overrides) to achieve the desired behavior.

For a more illustrative example: if we have an ethernet port and jack pair, and we have measured that it requires 7 Newtons of force to plug in an ethernet jack, we can set up a prefab scene containing both assets and a 7 N ramp of force pressing in the jack. Then we dial in the contact parameters and latch’s spring stiffness by observing how far the jack enters the port. This can be done by hand in SuperDex Studio, or via an agent using the accompanying CLI. Asset Tuner also produces signal plots that can be used for further analysis and live plots in Studio.

The following is supported:

- **Loads**: forces and torques on rigid bodies or articulated links, joint efforts and pose paths (implicit PD controller dragging body/link), each with a time profile
- **Probes**: body pose, velocity, contact force, joint state, solver stats and wall-clock solve time, sampled every step
- **Prefab overrides**: temporary parameter changes applied to the prefab for each run
- **Stop conditions**: end the run early when a body leaves or enters a bounding box, or a probe’s channel crosses a threshold

This guide is primarily for the CLI which is more useful for agents or after scenes have been set up in Studio, please refer to the documentation under SuperDex Studio for using Asset Tuner in Studio.

## Commands

```
superdex_asset_tuner run <file.superdex_tuner> [--set path=value]... [--out DIR] [--tag TAG] [--no-files]
superdex_asset_tuner describe <file.superdex_tuner | file.mochi_scene>
```

### describe

Lists everything loads, probes and stop conditions can target, after prefab overrides: rigid actors (with mass and whether they are static), articulated actors with their links and joints (with DoF offsets and counts), and the ground height the run will use.

Run this first on any new prefab. **Names in nested prefabs are paths**: an actor `brick` inside a nested prefab named `mobile` is `mobile/brick`; a link `link2` of articulated actor `arm` is `arm/link2`.

### run

Simulates for `run.duration` (or until a stop condition triggers), prints `summary.json` to stdout, and, unless `--no-files` is given, writes a run folder:

```
<--out, default ./tuner_runs>/<UTC timestamp>_<tuner name>[_<tag>]/
  tuner.resolved.json   the spec actually run: overrides applied, absolute prefab path, ground height
  signals.csv           one row per step: t, then one column per channel
  summary.json          status, message, steps, sim/wall time, per-channel min/max/final
  sim.rrd               the simulation over time, as a SuperDex episode (see below)
```

`tuner.resolved.json` is a complete tuner file: running it again reproduces the run

`sim.rrd` is written unless `run.record.enabled` is false, and only when the tuner is built with the SuperDex recorder (`SUPERDEX_BUILD_RECORDER` in CMake). It is a [Rerun](https://rerun.io) recording: open it in the Rerun viewer to see the bodies move and every channel plotted. It holds:

- the scene's actors at every step, under `data/physics`, drawn at `world`,
- with `run.record.contactForces`, each step's contact points and forces under `data/physics/contacts`, drawn as arrows at `world/contact_forces`,
- every channel, under `data/tuner/channels/<channel>`,
- each load as applied (point, vector and pose-path target), under `data/tuner/loads/<load>`,
- the resolved spec and the summary, as text on `data/tuner`.

`signals.csv` holds the same channels for tools that don't read Rerun recordings.

`summary.json`'s `status` is one of:

| status | meaning | exit code |
| :---- | :---- | :---- |
| `ok` | ran the full duration | 0 |
| `stopped` | a stop condition ended the run; `stopCondition` and `message` say which and why | 0 |
| `diverged` | the state became non-finite | 2 |

Setup problems (missing file, bad JSON, unknown target, missing channel) will print a message to stderr and exit 1\. Messages about unknown names list the available ones.

## Overriding values with `--set`

`--set path=value` changes any field of the tuner file before the run, without editing it. Repeat it as needed.

- **Paths** are dot-separated keys with array selectors: `[index]`, or `[field=value]` to pick the element whose string `field` equals `value`:

```
--set run.dt=0.0005
--set loads[0].enabled=false
--set "loads[name=pull_up].profile.magnitude=3"
--set "stopConditions[name=fell].threshold=-0.1"
```

- **Values** are JSON (`3`, `true`, `[0, 0, 1]`). Text that isn't valid JSON is used as a string, and a field that is already a string stays a string, so `--set prefabOverrides[0].value=0.8` sets the text `"0.8"`.

- **Quoting:** quote arguments with brackets in bash (`--set 'loads[name=push].profile.magnitude=3'`) so the shell doesn't glob them. PowerShell doesn't need it, but quoting is harmless.

- **Existing entries only:** `--set` can change an array element but can't append one. To sweep a parameter the tuner doesn't override yet, add the override to the file once and `--set` its value.

## .superdex\_tuner file reference

All quantities are assumed SI: m, s, kg, N, N m, rad. Fields left out will use defaults.

### Top level

| field | description |
| :---- | :---- |
| `prefab` | The prefab to simulate. `./`\-prefixed paths are relative to the tuner file, others to the assets root. |
| `assetsRoot` | Root for asset-relative paths. Empty: the nearest `.superdex_root` marker above the prefab (as in Studio), else `$SUPERDEX_ASSETS_PATH`, else the tuner file's folder. |
| `prefabOverrides` | `[{ "path", "value" }]`: JSON paths into the prefab, applied in order. |
| `loads`, `probes`, `stopConditions` | See below. |
| `run` | Time step, duration, gravity, ground and recording. |
| `plots` | Studio's plot layout. Unused in the CLI |

### Prefab overrides

Paths use the same syntax as `--set`, into the prefab's JSON. `prefabs[name=X].` reaches into one nested prefab reference; the change applies to that reference only, and the prefab file on disk is untouched:

```json
{ "path": "actors.rigid[name=block].density", "value": "2000" }
{ "path": "prefabs[name=stationary].actors.rigid[name=brick].isStatic", "value": "true" }
```

The simulated prefab does change, so `describe` reflects the overrides.

### Loads

| Field | Description |
| :---- | :---- |
| `name` | Unique. Its magnitude is the channel `load.<name>.mag`. |
| `enabled` | Disabled loads are skipped |
| `type` | `Force`, `Torque`, `JointEffort` or `PosePath` (below). |
| `target` | A dynamic rigid body or `actor/link` (`Force`, `Torque`, `PosePath`), or `actor/joint` (`JointEffort`). |
| `frame` | `World`: the direction is fixed in the world and the point rides on the body (a string tied to the body, pulled in a fixed direction). `Local`: the direction turns with the body. `WorldPoint` (forces only): the direction and `pointWorld` are both fixed in the world, so the line of action stays put while the body moves under it, even once the body has left it. |
| `direction` | Force or torque direction, normalized when applied. For `JointEffort`, per-DoF weights (not normalized). |
| `pointLocal` | Application point in the body or link frame (forces in `World` and `Local`). Off-center-of-mass points also apply the resulting torque. |
| `pointWorld` | Application point in world space (forces in `WorldPoint`). The torque is taken about the body's current center of mass. |
| `profile` | Magnitude over time, see below. |

Loads on the same body are summed internally. A `Force` or `Torque` on a link is mapped to joint forces through the link Jacobian, so it acts on the link the way it would on a free rigid body.

| `profile.kind` | Behavior | Fields used |
| :---- | :---- | :---- |
| `Constant` | `magnitude` from `start` to `end` | `magnitude`, `start`, `end` |
| `Envelope` | Rises linearly to `magnitude` over `riseTime`, holds, then falls over `fallTime` to reach 0 at `end` | \+ `riseTime`, `fallTime` |
| `Sine` | `offset + magnitude * sin(2π freqHz (t - start) + phase)` | \+ `offset`, `freqHz`, `phase` |
| `Keyframes` | Curve through `keys` (`time`, `value`, `interpolation` to the next key). Holds the first value before the keys and the last after | `keys` |

An `end` at or before `start` means the load never turns off. `Envelope` without an `end` holds forever (`fallTime` is ignored). Files that still say `Ramp` / `rampTime` load as `Envelope` / `riseTime`.

`interpolation` (keys and waypoints) is `Step` (hold until the next time), `Linear`, or `Smooth` (a Catmull-Rom curve through the neighbors that starts and ends at rest). Keys and waypoints are sorted by time when the tuner loads.

With `Keyframes`, the keys can also animate the load's direction and application point:

| Field | Description |
| :---- | :---- |
| `profile.animateDirection` | `Force`, `Torque`, `JointEffort`: the direction comes from each key's `direction` instead of the load's `direction`. Default false. |
| `profile.animatePoint` | `Force` only: the application point comes from each key's `point` instead of `pointLocal` / `pointWorld`. Default false. |
| `keys[].direction` | In the load's `frame`, like `direction`. A `Force` or `Torque` direction turns between keys along the shortest arc (Linear at constant angular speed, Smooth as a Catmull-Rom turn), so it never passes through zero, even between opposite directions; its length is ignored. `JointEffort` weights are blended as values. |
| `keys[].point` | Like the load's own point: body-local for `World` and `Local`, world for `WorldPoint`. A `WorldPoint` force with animated points moves its line of action through the world. |

```json
"profile": { "kind": "Keyframes", "animateDirection": true,
  "keys": [ { "time": 0, "value": 2, "direction": [1, 0, 0] },
            { "time": 1, "value": 2, "direction": [0, 1, 0] } ] }
```

#### Pose paths

A `PosePath` load pulls `pointLocal` on its target (a dynamic rigid body or `actor/link`) along timed `waypoints`, and turns the body to each waypoint's orientation, with a spring-damper similar to SuperDex Studio's Physics Drag (left click in the viewport). Use PosePaths to apply loads to bodies implicitly (using one body to push another), or for pulling bodies along complex trajectories.

```json
{ "name": "lift", "type": "PosePath", "target": "block", "pointLocal": [0.05, 0.05, 0.05],
  "waypoints": [
    { "time": 0, "position": [0.05, 0.05, 0.05], "rotation": [0, 0, 0] },
    { "time": 1, "position": [0.05, 0.05, 0.25], "rotation": [0, 0, 1.5708],
      "interpolation": "Linear" }
  ],
  "drive": { "responseHz": 6, "dampingRatio": 1, "maxAcceleration": 30, "rotation": true } }
```

| Field | Description |
| :---- | :---- |
| `waypoints[].time` | When the body should reach the pose \[s\]. |
| `waypoints[].position` | World position of `pointLocal` \[m\]. |
| `waypoints[].rotation` | World orientation of the body, as a rotation vector \[rad\]. |
| `waypoints[].interpolation` | How it moves on to the next waypoint (default `Smooth`). |
| `waypoints[].drive` | What the drive does from this waypoint until the next (default `On`): `On` drives the point and, with `drive.rotation`, the orientation; `PositionOnly` drives only the point, so the body turns freely about it; `Off` releases the body. Add a waypoint at time 0 with Off to start with the PosePath disabled. |
| `drive.responseHz`, `drive.dampingRatio` | Position spring: stiffness m(2πf)², damping 2ζm(2πf), with m the body's mass (a link uses its whole articulation's). Defaults 6 Hz, 1\. |
| `drive.maxAcceleration` | Caps the spring's pull at m × this \[m/s²\], so a blocked body is not pulled without bound. The damper is not capped: on a blocked body it adds 2ζm(2πf) × the path's speed while the path moves, and nothing once it holds. 0: no cap. Default 30\. |
| `drive.rotation` | Also drive the orientation (default true). Off: the body turns freely about the point. |
| `drive.rotResponseHz`, `drive.rotDampingRatio`, `drive.maxAngularAcceleration` | The same for orientation, scaled by the body's inertia. Defaults 4 Hz, 1, 30 rad/s². |

Its channels are `load.<name>.mag` (the drive's force \[N\]), `load.<name>.error` (distance from the path \[m\]), `load.<name>.angleError` (angle from the path's orientation \[rad\]) and `load.<name>.drive` (0 released, 1 position driven, 2 position and orientation driven). The errors read 0 for what the drive leaves free.

### Probes and channels

| Kind | Target | Channels |
| :---- | :---- | :---- |
| `BodyPose` | body or `actor/link` | `<target>.pos.{x,y,z}`, `<target>.rotvec.{x,y,z}`, `<target>.rotvecUnwrapped.{x,y,z}` |
| `BodyVelocity` | body or `actor/link` | `<target>.vel.{x,y,z}`, `<target>.angvel.{x,y,z}` |
| `ContactForce` | body or `actor/link` | `<target>.contactForce.{x,y,z}` (total, world frame) |
| `JointState` | `actor/joint` | `<target>.q<i>`, `<target>.qd<i>` per DoF |
| `SolverStats` | (none) | `solver.iters`, `solver.residual`, `solver.lineSearchIters`, `solver.convergence` (0 none, 1 converged, 2 stopped, 3 diverged) |
| `Performance` | (none) | `perf.solveMs` (wall-clock solve time; varies between runs, so runs are not bit-identical) |
| `Penetration` | body or `actor/link` | `<target>.penetration.volume` \[m³\]. Set `ignoreGround: true` to leave out ground planes. |

Every load adds `load.<name>.mag`. Channels are the `signals.csv` columns and the `summary.json` `channels` entries.

`rotvec` is the body's orientation relative to the world axes, as axis \* angle with the angle capped at π, so it flips sign when the body turns through 180°. `rotvecUnwrapped` is the same orientation unwrapped from step to step: it stays continuous and keeps growing past π. It assumes the body turns less than π per step.

`penetration.volume` approximates how far the target overlaps other actors: each overlapping contact point's depth times the contact area it covers, summed. Contact always overlaps a little, and how much grows with load and body size, so plot it before choosing a threshold.

### Stop conditions

Checked after every step, from `start` seconds on. With `holdTime` set, a condition must hold continuously for that long before it triggers, e.g. to ignore spikes on impact. The first one that triggers, in list order, ends the run.

| Kind | Triggers when | Fields |
| :---- | :---- | :---- |
| `LeavesBox` | the target's origin is outside `[boxMin, boxMax]` (world, axis-aligned) | `target`, `boxMin`, `boxMax` |
| `EntersBox` | the target's origin is inside the box | `target`, `boxMin`, `boxMax` |
| `ChannelThreshold` | `channel` goes `Above` / `Below` `threshold` | `channel`, `comparison`, `threshold` |

```json
"stopConditions": [
  { "name": "fell", "kind": "ChannelThreshold", "channel": "block.pos.z",
    "comparison": "Below", "threshold": -0.05 },
  { "name": "escaped", "kind": "LeavesBox", "target": "block",
    "boxMin": [-0.5, -0.5, -0.1], "boxMax": [0.5, 0.5, 0.5], "start": 0.2 },
  { "name": "interpenetrated", "kind": "ChannelThreshold", "channel": "block.penetration.volume",
    "comparison": "Above", "threshold": 1e-5, "holdTime": 0.05 }
]
```

A threshold's channel must be produced by a probe or load. If it isn't, the run fails and the error lists the channels that exist. Box conditions test the body's origin, not its whole shape.

### Run settings

| Field | Default | Description |
| :---- | :---- | :---- |
| `dt` | 0.01 | Fixed time step to use. |
| `duration` | 2 | Simulation time. |
| `singleThreaded` | true | Run SuperDex Physics on one thread |
| `gravity` | unset | Overrides the prefab's scene gravity. Unset: the prefab's, otherwise (0, 0, \-9.81). |
| `groundPlane` | true | Add a static ground plane, normal \+Z. |
| `groundHeight` | unset | Ground plane height. Unset: the lowest point of the prefab's bodies. |
| `groundContact` | unset | The ground plane's contact parameters (SuperDex Physics' `ContactParams` fields, e.g. `{ "coulombFrictionCoefficient": 0.8, "penaltySmoothingHalfDistance": 1e-5 }`). Unset: SuperDex defaults. |
| `record.enabled` | true | Write `sim.rrd` into the run folder. |
| `record.contactForces` | false | Include each step's contact points and forces in `sim.rrd` (larger files). Only scenes of rigid and articulated actors can record them. Formerly `contactPoints`. |
