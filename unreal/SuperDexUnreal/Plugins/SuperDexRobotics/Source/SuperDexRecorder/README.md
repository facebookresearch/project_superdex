# SuperDex Recorder for Unreal

The `SuperDexRecorder` module of the SuperDexRobotics plugin records a Mochi scene and its SuperDex
robot controllers to one transactional RRD per episode. It buffers sampled columns in memory and
writes them when the episode commits. A two-arm teleoperation scene records about 1 MB/s, and
committing blocks the calling thread while it writes, so keep episodes to minutes.

Create an `FSuperDexSceneRecorder`, call `BeginEpisode`, then call either `CommitEpisode` or
`AbortEpisode`. It records the SuperDexPhysics game-instance scene unless the config's `Scene` names
another Mochi scene, and only the robots whose articulation lives in that scene. To record several
scenes, use one recorder (one episode) per scene. Two recorders capturing the same scene both record
everything in it, so the second logs a warning.

Each sample is recorded on three timelines: `frame`, `sim_time` (the simulation time) and
`wall_time`, the wall-clock time since the episode began, on a monotonic clock, at which the
frame's physics state was recorded. Commands, skeletons, sources and events sampled for a frame
share its wall time. Comparing `wall_time` with `sim_time` shows when steps were late or missing.

The required scene archive filename and XXH3-64 content hash are stored in the RRD; restoring or
rebuilding the scene offline must use an archive with the recorded hash. An optional
`.superdex_task` filename and hash can be recorded as well. Set `bRecordContactForces` to record
Mochi's contact forces directly.

Set `SpawnNames` to record the bodies of task-spawned actors under their spawn names. A scene
rebuilt from the task then matches them to its spawns.

Set `SkeletonSources` to record keypoint skeletons, such as an operator's tracked hands, beside
their robots' `command` shadows. Each sampler runs on the Mochi thread after every step. It returns
false while its skeleton is untracked, which hides the skeleton; a non-finite keypoint fails the
episode.

Skeletons without a `Robot`, and `TransformSources` such as a headset pose, are recorded on their
own under `world/sources/<Id>`. Their samplers also run on the Mochi thread after every step.

Call `RecordEvent(Stream, Text)`, `RecordScalar(Name, Value)` or `RecordScalars(Name, Values)` from
any thread while capturing. They're recorded at the next Mochi step under `data/events/<Stream>` (a
Rerun text log) and `data/scalars/<Name>`. Texts raised on one stream within one step share that
step's sample. Values still queued when the episode ends (for example an outcome decided in the
same frame) are written at the last recorded step. A scalar list's length isn't fixed by the
recording: keep it consistent per name and interpret the values when reading.

Set `bRequireSceneArchive` to false to record a level that has no scene archive; the episode then
stores no `scene.archive` reference and can't be rebuilt offline from an archive.

## Blueprint

Place a `SuperDex Recorder` actor (`ASuperDexRecorder`) in the level; Blueprint subclasses of it can
hold study logic beside the recording. By default it starts recording once the Mochi scene is ready
(gate it with `bPreventAutomaticStart` and `WaitForActorsBeforeStart`) and commits when play ends;
or call `StartRecording`, `CommitRecording` and `AbortRecording` yourself. Set `Properties` for
per-episode metadata.

A world records with one recorder: a second one would record every body and robot again. The first
recorder to initialize records; any other logs an error naming them all and never records. `Get
Active Recorder` returns the recording one from any Blueprint, including in `BeginPlay`, so study
code can call `RecordEvent(Stream, Text)`, `RecordScalar(Name, Value)` and `RecordScalars(Name,
Values)` without holding a reference.

To record a study block by block, call `StartRecording` and `CommitRecording` once per block,
setting `Properties` (e.g. the block or condition) before each `StartRecording`. Each block is its
own episode file. Commit at a pause in the study: it blocks while the file is written.

To record questionnaire answers, bind the `On Response` event of a SuperDexStudy question prompt
(`ASuperDexStudyQuestionPrompt`) to `Record Scalar` on `Get Active Recorder`, named after the
question (e.g. its data asset), and use `Record Event` for study milestones such as a block starting.

The recorder's `bRequireSceneArchive` defaults to false, unlike `FSuperDexSceneRecorderConfig`'s,
so it records hand-built levels that have no scene archive. Set `SceneArchive` when the level was
loaded from one. Or set `bExportLiveScene`: when the first episode starts, the recorder exports the
live Mochi scene, task and spawnables included, to
`<OutputDirectory>/scenes/<Map>_<UTC time>.superdex_scene_archive` and references it. The
simulation pauses while it exports. Later episodes reuse that archive; call
`ExportLiveSceneAtNextEpisode` after adding or removing physics actors or bots. The editor's "Export
SuperDex Bot Scene Archive (Live Scene)" menu entry exports the same way. Only editor builds (PIE
included) can export: bots are exported from their source files, whose paths are editor-only data.
Packaged builds record without a scene archive (or refuse to record with `bRequireSceneArchive`).

The physics recorder already records every Mochi body, and the robotics recorder every SuperDex bot.
For anything else, add source components to the recorder (or to an actor in `SourceActors`):
- `SuperDex Recording Player Camera Source` records the first local player's camera (the headset in
  VR) under `world/sources/head`.
- `SuperDex Recording Transform Source` records the world transforms of chosen scene components,
  such as objects without Mochi physics, under `world/sources/objects.<Id>`.

New sources derive from `USuperDexRecordingSourceComponent`, add their streams in
`ConfigureRecording` and release them in `OnRecordingEnded`; they tick only while an episode is
recording. Values they read on the game thread must be cached for the Mochi-thread sampler, as
`FSuperDexRecordingCachedTransform` does.
