# Sensor graphs

SensorGraph provides synchronous numeric CPU observations with separate capture and
publication timestamps. Each sensor publishes one fixed-shape Float32 tensor.
Public callers construct and step the graph programmatically.

[`SensorBase`](./sensor_base.h) is the common interface for numeric observation sources and nodes
that transform other sensor outputs. [`SensorGraph`](./sensor_graph.h) connects them into a directed
acyclic graph and steps them in dependency order. Sources and transforms are both called sensors.

A graph can model the path from SuperDex physics to the observations a policy receives. For example:

```text
surface forces → noise model → selected output
```

Some measurement models acquire physics data themselves; they do not need a separate source node.
Other sensors aggregate surface forces, map them to a spatial tensor, or transform an upstream
sample. Inputs are bound during initialization; explicit graph operations can extend or prune the
topology while preserving surviving sensor state.

- Each sensor publishes a `SensorSample`: a borrowed view of a flat Float32 payload, with capture
  timestamps. `SensorFormat` describes its tensor shape, units and labels.
- Select the outputs you need. The graph activates those nodes and their upstream dependencies;
  other nodes remain inactive.
- Call `graph.Step(now, error)` after advancing physics. Each active sensor can publish a sample
  or return `NoUpdate`, for example when its sampling period is not due.
- Read `sensor.Latest()` to obtain the most recently published sample without triggering another
  acquisition. Policy reads and sensor acquisitions can run at different rates.

## Quick start

This example reads the total contact force on a supported rigid actor or articulated link.
It assumes an existing `Scene& scene`, `Actor* linkActor` and `Error& error`. Keep the actor and
scene alive until the sensor is destroyed; declare the graph after its sensors.

```cpp
#include <superdex_robotics/sensors/sensor_link_contact_force.h>
#include <superdex_robotics/sensors/sensor_graph.h>

using namespace superdex::robotics;

SensorLinkContactForce::Params params;
params.sampleRateHz = 60;
SensorLinkContactForce forces(linkActor, params, error);
if (error) {
  return;
}
SensorGraph graph;
graph.Add("link_force", &forces, error);
graph.SetOutputEnabled("link_force", true, error);
graph.Build({.scene = &scene}, error);
graph.Reset(error);
if (error) {
  return;
}
for (int64_t step = 1; step <= 20; ++step) {
  scene.Step(0.005);
  graph.Step({.t = step * 0.005, .dt = 0.005, .step = step}, error);
  if (error) {
    return;
  }
  SensorSample const sample = forces.Latest();
  if (sample.captureStep < 0) {
    continue;
  }
  Span<float const> const values = sample.Floats();
  // Consume values before the next publication, or copy them into owning storage.
}
```

The output is `[Fx, Fy, Fz]` in newtons, expressed in the sensor frame. The force acts from the
environment onto the queried actor. It includes all contacts on that actor, including contacts
whose sampling points lie on the other body. It does not isolate a sensitive patch, provide a
pressure map or include joint loads. A contact-free acquisition publishes a zero force.

The source registers Mochi's `TotalContactForce` query. The actor must support that query, and the
integration method must end its step at the last integration stage so the force and pose used for
frame conversion describe the same state. `sampleRateHz = 0` captures on every positive-duration
graph step; a positive rate is sampled on the available graph steps, without a separate clock or
interpolation. Read `Latest()` between captures to obtain the held sample.

Use `SensorLinkVertexForce` when contact location or spatial distribution is required, for example
as the input to a `SensorSeedRectMap`.

## Available sensors

The headers linked below specify constructor parameters, frames and metadata requirements. `N`
denotes the source's vertex or joint count; `H` and `W` are map dimensions.
Matching shape alone does not establish compatible units, frames or row identities.

### Public sensors

| Sensor | Purpose | Output shape |
|--------|---------|--------------|
| [`SensorLinkContactForce`](./sensor_link_contact_force.h) | Whole-actor contact force in the sensor frame | `[3]` |
| [`SensorLinkVertexForce`](./sensor_link_vertex_force.h) | Surface contact forces, with vertex positions and normals in metadata | `[N, 3]` |
| [`SensorArticulatedJointWrench`](./sensor_articulated_joint_wrench.h) | Solver-backed joint forces and torques, with row-to-joint metadata | `[N, 6]` |
| [`SensorJointEncoder`](./sensor_joint_encoder.h) | The owning bot's one-DOF joint positions (plus calibration offsets) and velocities | `[2, N]` |
| [`SensorSeedV1MLP`](./sensor_seed_v1_mlp.h) | Contact acquisition and an optional MLP response model | `[3]` |
| [`SensorAggregateForce`](./sensor_aggregate_force.h) | Sum a borrowed vertex-force input | `[3]` |
| [`SensorSeedRectMap`](./sensor_seed_rect_map.h) | CPU force map from a vertex-force input and its spatial metadata | `[H, W, 3]` |
| [`SensorDelayLine`](./sensor_delay_line.h) | Fixed delay measured from upstream publication | Same as input |
| [`SensorTemporalWindow`](./sensor_temporal_window.h) | Oldest-first history of upstream publications | `[historyLength, ...input shape]` |

Joint wrench rows omit joints whose child link is welded to the world through an entirely static
ancestry. Check the source headers for these restrictions before choosing an observation layout.

SEED V1 uses raw contact force when the model is bypassed or no weights are loaded. Supply the
weights to obtain the learned response; an output sample alone does not confirm that inference ran.

## Samples, formats and capture identity

[`SensorFormat`](./sensor_format.h) is the immutable description of one sensor output:

```cpp
struct SensorAxis {
  DynamicString name;
  DynamicArray<DynamicString> labels;
  DynamicArray<DynamicString> units;
};

struct SensorFormat {
  DynamicString kind;
  DynamicArray<int32_t> shape;
  SensorDType dtype = SensorDType::Float32;
  DynamicArray<SensorAxis> axes;
};
```

`shape` gives the row-major tensor dimensions. `axes` has the same length and order as `shape`.
Labels are either empty or contain one entry per element of that axis. Units are empty, contain one
broadcast unit, or contain one unit per element; at most one axis declares units.

`kind` describes the output independently of the sensor class. For example, `Wrench6D` is one
force/torque vector `[6]`; `Wrench6DList` is an ordered list `[N, 6]`, with metadata identifying
each row. Different implementations can produce the same format.

`SensorTensor` is an owning flat `DynamicArray<float>` for copying and serialization boundaries.
Its flat storage does not restrict the tensor rank.

### Capture identity

`StepTime` supplies simulation or hardware time: `t` and `dt` in seconds, and an integer `step`.
Capture and publication have separate identities:

```cpp
// Excerpt from SensorSample.
int64_t captureStep;
double captureTime;
int64_t publishedStep;
double publishedTime;
void const* data;
int64_t elementCount;

// SensorSample::Floats() returns the bounded Float32 view.
```

`captureStep` and `captureTime` identify the acquisition. A transform passes its source sample to
`PublishSample()` to preserve that identity. In the quick start, the surface-force sample and
its derived force map retain the same capture timestamp. Reading the map again does not acquire
new data or recompute it.

`captureStep == -1` means no sample has been published. Check it before consuming values;
allocated output storage alone does not establish a valid observation. `publishedStep` and
`publishedTime` identify when this sensor made that capture available. For example, a capture at
1.000 s passed through a 50 ms delay retains `captureTime = 1.000` and receives
`publishedTime = 1.050` on the first graph step at or after its due time.

`SensorDelayLine` measures its configured `delayMs` from the upstream publication, preserving the
original capture identity. Configure `bufferMs` to cover the delay; storage is sized once from the
input's publication-frequency estimate, or `bufferAssumedFrequencyHzEstimate` when unavailable.
An underestimated capacity reports an error rather than discarding a sample silently.

`SensorTemporalWindow` prepends an oldest-first history axis. During startup or after reset, some
rows are empty; their row capture/publication steps are `-1`. Use the row timestamp getters when
interpreting a partially filled window or samples with different acquisition times.

An environment can advance physics to settle its initial state during reset. Its policy step
counter can therefore still be zero when the sensor graph has already published samples. Policy
steps, graph steps and source acquisition rates are distinct.

### Ownership and metadata

`SensorSample` borrows producer-owned storage. Its view remains valid until that sensor publishes
again, resets, is initialized again or is destroyed. Graph mutation, stepping and borrowed reads
must be serialized by the caller. Copy `Floats()` into owning storage, or use `CopyLatestValues()`,
when the values must outlive that view.

Some formats need immutable information beyond shape and axis labels, such as vertex positions
or joint-row identities. Use the typed helper to retrieve it:

```cpp
auto const* metadata = TryGetSensorMetadata<VertexListMeta>(sensor);
```

`SensorMetadataKey<T>` supplies a stable library-owned key. The helper checks byte size and
alignment; unsupported metadata returns `nullptr`. The returned object remains producer-owned
and is valid only while that initialization remains live. The raw bytes returned by
`TryGetMetadata()` are an object view, not a serialization format.

## Enablement and lifecycle

### Construct and build

`SensorGraph` borrows sensor instances. Loaded bot/scene sensors are owned by `RoboticsContext`;
standalone callers may own sensors themselves. Sensors must outlive the graph's use of them.
Programmatic construction supplies an explicit graph-unique name and the names of upstream inputs
to `Add()`. A standalone node can be named `source` or `map`; bot-attached callers can use
`bot/link/sensor-instance`. Programmatic names have no attachment-based enforcement.

`Add()` rejects empty or duplicate names and null sensors. `Build()` rejects missing inputs,
duplicate nonzero handles, reuse of a sensor under multiple names and dependency cycles. It resolves
names to a topological execution order and initializes newly active sensors with the supplied
`SensorScope`. Select outputs by name before building; forward references are validated at build.

**Enabled nodes** are explicitly selected outputs. **Active nodes** also include their transitive
upstream dependencies. With no requested outputs, no nodes run.

### Reset, step and read

1. `SensorBase::Init()` binds inputs, calls `InitSensor()` and allocates an output buffer from the
   returned format.
2. `Reset()` clears the published sample and calls `ResetSensor()` to clear episode state. Reset
   itself publishes no sample.
3. `Step()` performs acquisition or transformation. It returns `Published` after successful
   publication or `NoUpdate` when no new observation is available. A non-OK `Error` always means
   failure, regardless of the returned result.
4. `Latest()` reads the already-published sample without acquisition, allocation or recomputation.

`SensorGraph::Step()` reports execution success. Repeating the same step and time is idempotent;
inconsistent repeats, regressing time, negative values and non-finite time are errors. Check
`error` after graph operations before consuming outputs.

### Change outputs or topology

The handle overload of `Add()` associates an existing `SensorHandle` with a node. After building,
`SetOutputEnabled(handle, enabled, error)` changes output selection; the new active dependency
closure takes effect on the next `Reset()`. `GetOutputEnabled()` reports output selection, even
when an unselected node is needed as a dependency.

`GetEnabledNodes()` returns a borrowed list of selected-output names, handles and sensor pointers
in dependency order. A standalone node may have an invalid handle. Fetch this list again after
changing enablement or topology; graph mutation or destruction invalidates the view. Selection
can precede activation, so the list alone does not imply that a new output has acquired a sample.
`Find(name)` returns null while the graph is unbuilt or when the name is absent.

`Extend(nodes, enabledNames, error)` adds explicit declarations to a built graph and optionally
selects additional outputs. An empty enabled-name list preserves existing selections.
`Remove(handles, error)` prunes those sensors and their downstream dependants, returning owned
identities for removed enabled outputs. It does not destroy sensors. These operations preserve
surviving sensor state and the graph clock. `Clear()` removes all nodes and output requests;
changing the graph's initialization scope requires clearing it.

See [`sensor_graph.h`](./sensor_graph.h) for the complete graph interface.

### Random streams

Noise, outlier and jittered-delay nodes draw from `Derive(episodeSeed, key, step)`, where the key is
the node's `random.processKey` or, when that is empty, its graph name. A sensor takes its graph name
from `SensorGraph::Add`, and `SensorGraph::SetEpisodeSeed` seeds every such node of a graph. The
graph's seed overrides `random.episodeSeed` and survives `SetParams`. Leave `processKey` out of
scene and bot files: BotScene names nodes `<bot>/<link>/<sensor>`, so two instances of one bot draw
independently, while a shared explicit key would give them identical noise. Set an explicit key only
to name a stream across graphs or code changes.

## Recording hooks

[`SensorRecordingSink`](./sensor_recording.h) provides a backend-neutral destination for constants
and timestamped samples. `ConfigureRecording()` declares sensor metadata and additional fields;
`Record()` writes the latest sample and any sensor-specific sidecars. Storage, encoding and
persistence belong to the sink. A graph step does not itself invoke these recording hooks.

These hooks do not supply a concrete recorder or replay backend. Those integrations are separate
from the R2 sensor additions.

## Creating sensors

A synchronous concrete sensor implements `InitSensor()`, `Step()` and `ResetSensor()`, and provides
a constructor that calls the appropriate attached or unattached `SensorBase` constructor:

```cpp
class SensorExample final : public SensorBase {
 public:
  explicit SensorExample(Error& error) : SensorBase(error) {}
  std::string_view GetTypeName() const override {
    return "SENSOR_EXAMPLE";
  }
  SensorStepResult Step(StepTime const& now, Error& error) override;

 private:
  SensorFormat InitSensor(SensorScope const& scope, Error& error) override;
  void ResetSensor() override;
};
```

Sources use no inputs. Transforms validate `Inputs()` during `InitSensor()` and read already-published
upstream samples during `Step()`. Return `NoUpdate` when the upstream sample is unpublished or
when no new input needs processing. Use `PublishSample(now, values, error, &source)` to retain a
source's capture identity. `Reset()` is final; implement an explicit empty `ResetSensor()` for a
stateless sensor. Implement `GetTypeName()` from `ComponentBase` and keep construction parameters
typed on the concrete sensor.

A source computed by a C++ function needs no class of its own: [`SensorFunction`](./sensor_function.h)
publishes the values its function writes at every graph step, the reset step included. The function
returns the sample its values derive from, such as another node's `Latest()`, to keep that sample's
capture step and time, or `std::nullopt` for values captured at the step. This mirror republishes a
measurement node at every step of another graph, keeping its format and capture:

```cpp
SensorFunction mirror(
    measured.GetFormat(),
    [&measured](Span<float> out) -> std::optional<SensorSample> {
      SensorSample const sample = measured.Latest();
      std::copy_n(sample.Floats().data(), out.size(), out.data());
      return sample;
    },
    error);
```

`SensorFunction::ValuesFormat(width)` is the format of `width` plain values.

## Python and sensor graphs

### Sensors implemented in C++

The common Python workflow keeps every graph sensor in C++: built-in types, or custom types
registered from C++. Python needs no binding for a particular sensor to use it. It selects the
sensor by graph name and reads it through `SensorBase`: `latest()`, `copy_latest_values()` and
`get_format()`. The built-in sensors and their `Params` are also declared in
`superdex_robotics.mochi_gen`, which generates typed Python classes with `get_params()` and
`set_params()`. A custom C++ sensor gets typed Python access only when it is declared there or in
bindings of its own; see [Creating sensors](#creating-sensors).

The order of operations is:

1. Register custom types before loading. The loader skips an unregistered type with a warning.
2. Load the scene. Loading creates each bot's declared sensors and the scene-level sensors, builds
   the graph with no outputs selected, and resets acquisition.
3. Select outputs with `add_sensor_outputs()`.
4. Step the physics scene. BotScene steps the graph after each physics step.
5. Check for graph errors, then read the selected outputs.

```python
import superdex.physics as sdp
import superdex.robotics as sdr
from superdex.physics.utils.scene_helpers import load_bot_scene

sdp.initialize(num_worker_threads=0)
bots_context = sdr.create_context()  # Must outlive the BotScene.
bot_scene = load_bot_scene("hand.superdex_scene", bots_context)
bot_scene.add_sensor_outputs(["hand/tip/noisy_force"])

scene = bot_scene.get_scene()
for _ in range(100):
    scene.step(0.005)  # Physics, then one graph step.
bot_scene.check_sensor_graph()  # Raises the first graph error since the last reset.
for output in bot_scene.get_sensor_outputs():
    sample = output.sensor.latest()
    if sample.capture_step >= 0:  # -1 means nothing has been published yet.
        values = output.sensor.copy_latest_values()  # An owning copy.
```

Points to note:

- **Python does not build or step a `SensorGraph`.** BotScene builds it from the declared sensors and
  steps it once after every `scene.step()`, with that physics step's time, so graph steps and
  physics steps are in lockstep. A source with a positive `sampleRateHz` samples more slowly than
  physics, never faster. Graph timing independent of physics requires a standalone C++ graph.
- **Graph errors do not interrupt `scene.step()`.** BotScene records the first error, stops stepping
  the graph and keeps the last published samples. `check_sensor_graph()` raises the recorded error
  until `reset_sensors()` or `rebuild_sensor_graph()` clears it, so one check before consuming
  observations covers every step since the last check. The explicit check trades streamlining for
  control: the application decides when to look, and whether to end the episode or reset the
  sensors and continue.
- **Only declared sensors join the BotScene graph.** A sensor created at runtime with
  `create_sensor()` is owned by the context but is not a graph node, and `rebuild_sensor_graph()`
  rebuilds from declarations only.
- **Some interfaces are C++-only:** constructing a `SensorGraph` or a `SensorFunction`, the
  `RegisterSensor<T>()` templates (including registering `SENSOR_TEMPORAL_WINDOW`), the native
  publication timestamps (Python exposes `capture_step` and `capture_time`), and typed metadata
  such as `VertexListMeta`.

### Python-only sensors

`register_python_sensor(context, typeName, factory)` adds a Python class to the same factory table
as C++ types. Link entries in `.superdex_bot` files and `sensors` entries in `.superdex_scene` files
can then declare the type, and `RoboticsContext.create_sensor` or `Bot.create_sensor` creates it at
runtime. The factory receives the link actor, or `None` for an unattached sensor, and the params
string. Creation fails if the instance has no `reset()` method. The context owns a `PythonSensor`
adapter that holds a strong reference to the instance and is destroyed with its bot or the
context; `get_python_sensor(context, handle)` returns the instance.

The adapter is not graph-enabled. Its `InitSensor()` fails with "Python sensor is not
graph-enabled" and its `Step()` always returns `NoUpdate`. A declared Python sensor is therefore an
inactive graph node: selecting it as an output, or naming it as another node's input, fails graph
initialization. It has no `SensorFormat`, publishes no `SensorSample`, carries no capture or
publication time, and is not reached by recording hooks. The application calls the instance's own
methods to compute its signal.

`SensorGraph::Reset()` is the only framework path that resets sensors, and it resets initialized
nodes only. No framework path therefore calls a Python sensor's `reset()`; call it from the
application between episodes.

A bot or scene file that declares a Python type loads that sensor only where the type is
registered. Runtimes without it, such as SuperDex Unreal or a C++-only application, skip the entry
with a warning.

## Bot-relative inputs

The planned combined scene structure resolves bot-relative inputs such as `./force` and
`../front_pad/force` into complete `bot/link/sensor` graph names. This declarative integration
is future work; public callers currently supply graph names programmatically.
Scene-owned sensors use complete names. Other relative path shapes are rejected; missing
inputs and cycles fail graph construction. Graph node declarations own their input-name strings.

## Release scope and integration

SensorGraph supports synchronous numeric CPU observations, including image-like tactile tensors.
Camera image samples, asynchronous graph execution and GPU processing are future work.

Public callers advance physics and then step their graph explicitly, as shown above. A combined
scene structure with declarative sensors and automatic acquisition is future work. All graph
operations and borrowed sample reads must remain serialized by the caller.

Some sensors retain specialized `ComputeSignal()` methods for compatibility. Deriving from
`SensorBase` alone does not make an existing implementation usable in a graph; unsupported graph
initialization is rejected explicitly. Python sensors are not graph-enabled; see
[Python and sensor graphs](#python-and-sensor-graphs).

### Force aggregation

`SensorAggregateForce` sums one borrowed `[N,3]` input directly, without copying
its tensor, and preserves the input's capture identity and interaction sense.
It transforms the supplied signal; it does not query physics or recover contact
contributions absent from its input. Use a total-contact-force source when the
required signal is the link's net physical contact force.
