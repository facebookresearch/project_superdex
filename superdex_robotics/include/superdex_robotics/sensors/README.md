# Sensor graphs

This README covers **SensorGraph Stack 1 (diffs 1–6)**: synchronous numeric sources, transforms,
formats, graph execution and recording hooks. **Stack 2 (diffs 1–14)** extends that foundation with
acquisition and publication timing, noise and delay, concrete recording and replay, additional
sensor models and domain randomization. **Beyond Stack 2** are asynchronous execution,
GPU-rendered observations and camera samples. The examples below use Stack 1 sensors.

[`SensorBase`](./sensor_base.h) is the common interface for numeric observation sources and nodes
that transform other sensor outputs. [`SensorGraph`](./sensor_graph.h) connects them into a directed
acyclic graph and steps them in dependency order. Sources and transforms are both called sensors.

A graph can model the path from SuperDex physics to the observations a policy receives:

```text
surface forces → spatial force map → selected output
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

This example maps surface forces on a fingertip to a rectangular tactile tensor. It assumes an
existing `Scene& scene` and `Actor* linkActor` with a mesh or contact skin that supports vertex-force
queries. Set the map center and orientation for the fingertip geometry; the defaults describe a
hemisphere centered at the sensor origin, with its tip along +Z.

The graph borrows these sensors, so declare it after them and keep the actor and scene alive
throughout its use.

```cpp
#include <superdex_robotics/sensors/sensor_link_vertex_force.h>
#include <superdex_robotics/sensors/sensor_seed_rect_map.h>
#include <superdex_robotics/sensors/sensor_graph.h>

using namespace superdex::robotics;

SensorLinkVertexForce::Params forceParams;
SensorLinkVertexForce forces(linkActor, forceParams, error);
SensorSeedRectMap::Params mapParams;
mapParams.width = 64;
mapParams.height = 64;
SensorSeedRectMap map(mapParams, error);
if (error) {
  return;
}

SensorGraph graph;
graph.Add("surface_forces", &forces, error);
DynamicArray<DynamicString> mapInputs{"surface_forces"};
graph.Add("force_map", &map, mapInputs, error);
graph.SetOutputEnabled("force_map", true, error);
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
  SensorSample const sample = map.Latest();
  if (sample.captureStep < 0) {
    continue; // The output has not published yet.
  }
  Span<float const> const values = sample.Floats();
  // Consume values before the next graph step, or copy them into owning storage.
}
```

The source output is `[N, 3]`, with vertex positions and normals in metadata. The map uses that
spatial metadata and publishes `[64, 64, 3]`. Only `force_map` is selected as an output;
`surface_forces` runs as its dependency. Contact-based sources require a supported integration
method; see the [integration notes](./internal/README.md#botscene-physics-integration).

## Available sensors

The headers linked below specify constructor parameters, frames and metadata requirements. `N`
denotes the source's vertex or joint count; `H` and `W` are map dimensions.
Matching shape alone does not establish compatible units, frames or row identities.

### Public sensors

| Sensor | Purpose | Output shape |
|--------|---------|--------------|
| [`SensorLinkVertexForce`](./sensor_link_vertex_force.h) | Surface contact forces, with vertex positions and normals in metadata | `[N, 3]` |
| [`SensorArticulatedJointWrench`](./sensor_articulated_joint_wrench.h) | Solver-backed joint forces and torques, with row-to-joint metadata | `[N, 6]` |
| [`SensorSeedV1MLP`](./sensor_seed_v1_mlp.h) | Contact acquisition and an optional MLP response model | `[3]` |
| [`SensorAggregateForce`](./sensor_aggregate_force.h) | Sum a borrowed vertex-force input | `[3]` |
| [`SensorSeedRectMap`](./sensor_seed_rect_map.h) | CPU force map from a vertex-force input and its spatial metadata | `[H, W, 3]` |

Joint wrench rows omit joints whose child link is welded to the world through an entirely static
ancestry. Check the source headers for these restrictions before choosing an observation layout.

SEED V1 uses raw contact force when the model is bypassed or no weights are loaded. Supply the
weights to obtain the learned response; an output sample alone does not confirm that inference ran.

### Internal sensors

These implementations are in `internal/` and are not part of the OSS export:

| Sensor | Purpose | Output shape |
|--------|---------|--------------|
| [`SensorAggregateForce`](./internal/sensor_aggregate_force.h) | Sum a vertex-force input | `[3]` |
| [`SensorAggregateWrench`](./internal/sensor_aggregate_wrench.h) | Sum forces and moments using vertex and interaction metadata | `[6]` |
| [`SensorMitsumiMms101V1`](./internal/sensor_mitsumi_mms101_v1.h) | Mitsumi contact-based wrench observation | `[6]` |

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
The sample fields used by Stack 1 consumers are:

```cpp
// Excerpt from SensorSample.
int64_t captureStep;
double captureTime;
void const* data;
int64_t elementCount;

// SensorSample::Floats() returns the bounded Float32 view.
```

`captureStep` and `captureTime` identify the acquisition. A transform passes its source sample to
`PublishSample()` to preserve that identity. In the quick start, the surface-force sample and
its derived force map retain the same capture timestamp. Reading the map again does not acquire
new data or recompute it.

`captureStep == -1` means no sample has been published. Check it before consuming values;
allocated output storage alone does not establish a valid observation. Stack 2 adds a separate
publication identity to describe when a transformed or delayed sample becomes available.

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

## Recording hooks

[`SensorRecordingSink`](./sensor_recording.h) provides a backend-neutral destination for constants
and timestamped samples. `ConfigureRecording()` declares sensor metadata and additional fields;
`Record()` writes the latest sample and any sensor-specific sidecars. Storage, encoding and
persistence belong to the sink. A graph step does not itself invoke these recording hooks.

Stack 1 provides these hooks. The concrete graph recorder, recording/replay integration and
episode randomization belong to Stack 2.

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

## Bot-relative inputs

A bot file can refer to another sensor on the same link with `./force`, or one on another
link of the same bot with `../front_pad/force`. `BotScene` resolves these inputs once, using
the scene's bot instance name, into complete `bot/link/sensor` graph names. The same bot file
can therefore be instantiated under different scene names without editing its input paths.
Scene-owned sensors use complete names. Other relative path shapes are rejected; missing
inputs and cycles fail graph construction. Graph node declarations own their input-name strings.

## Release scope and integration

Stack 1 supports synchronous numeric CPU observations, including image-like tactile tensors.
Asynchronous execution, GPU-rendered observations and camera samples are beyond Stack 2.
[`SensorCamera`](./sensor_camera.h) does not yet produce graph image observations.

The initial OSS release supports separate bots and physics scenes. Declarative graph construction
through the combined scene workflow and its automatic post-physics acquisition callback are
internal integrations; see the [internal integration notes](./internal/README.md). Public callers
construct and step the graph programmatically as shown above.

Some sensors retain specialized `ComputeSignal()` methods for compatibility. Deriving from
`SensorBase` alone does not make an existing implementation usable in a graph; unsupported graph
initialization is rejected explicitly. Python sensors keep the required `reset()` method, invoked
through the shared adapter after the common publication has been cleared.

### Force aggregation

`SensorAggregateForce` sums one borrowed `[N,3]` input directly, without copying
its tensor, and preserves the input's capture identity and interaction sense.
It transforms the supplied signal; it does not query physics or recover contact
contributions absent from its input. Use a total-contact-force source when the
required signal is the link's net physical contact force.
