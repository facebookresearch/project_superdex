# Loading sensors: choose the cast before starting the show

Imagine a scene with two bot slots. One is a hand, and the other can hold either
of two grippers. The scene file describes all these possibilities. That doesn't
mean we should open every bot file, construct every sensor, or run every edit
callback just to discover which gripper the caller actually wants.

The caller chooses the bots. We prepare their sensor declarations, check that
they form a valid graph, and then load the bots. The graph can know its wiring
before it has all its live sensors.

## Four words, four jobs

| Word | What it means |
| --- | --- |
| Selected | The bot belongs to this configuration, including bots needed by sensor dependencies. |
| Prepared | Its prefab has been read, edited and cached, and its sensor declarations participate in the validated graph. |
| Loaded | The bot and its sensors have been constructed. |
| Bound | A live sensor has been attached to its already-declared graph node. |

These give us three useful bot states: unselected; selected and prepared but
not loaded; selected and loaded. Binding is a sensor-level operation: a loaded
bot's downstream sensor can still be waiting for an upstream sensor on another
selected bot.

An unbound node owns its name and input names but has no live sensor pointer.
We sometimes called this "dormant" in conversation; the API calls it unbound.
`BindNode` attaches a live sensor; `UnbindNode` releases bindings, including
those of downstream dependants. `ComputeReadiness` updates the cached readiness
flags from the bindings and their dependencies. It doesn't clear bindings or
reset sensor history.

## Preparation checks the wiring, not just the files

`BotScene::PrepareBots(selectedBotNames, error)` takes the complete selection.
It reads only those bots' prefabs and applies `BotPrefabEdit` once per bot entry.
The cache is keyed by the scene's bot instance name: two instances of the same
file can receive different edits.

The selected bots' sensor declarations and the scene sensor declarations must
form a valid DAG:

- Node names are non-empty and unique.
- Every input names another declared node.
- Dependencies have no cycles.

`SensorGraph` checks this with Kahn's topological sorting algorithm. It does not
need live sensors to do so. Input tensor shapes and other sensor-specific
contracts are checked later, when the live sensors are initialized.

Suppose `reader/base/double` takes input from `source/base/clock`. Selecting
`reader` alone is invalid. Select `source` too, even if you aren't ready to load
it. This rule applies equally to bot sensors and scene sensors. We don't
silently select dependencies, or read an unselected slot alternative to make a
reference work.

If preparation fails, handle the error before loading or stepping that
configuration. The graph is not a way to sneak an invalid selection into a
running simulation.

## Selected does not mean loaded immediately

Here is the two-phase workflow, after loading the scene with an empty
`botsToInitialize` list:

```cpp
DynamicArray<DynamicString> const selected{"reader", "source"};
botScene.PrepareBots(MakeConstSpan(selected), error);
// Check error before continuing.
botScene.InitializeBot("reader", error);
// Load source later, when it is wanted.
botScene.InitializeBot("source", error);
DynamicArray<DynamicString> const outputs{"reader/base/double"};
botScene.AddSensorOutputs(outputs, error);
```

Loading a bot doesn't select its outputs. `AddSensorOutputs` takes the required
canonical output names as a `Span<DynamicString const>`.

`InitializeBot` requires the bot to be selected already. It uses the prepared
prefab, constructs the bot, and binds its sensors. Ready scene sensors are
constructed and bound too. Construction order need not be dependency order:
the declaration graph already describes which sensors must wait.

A valid graph can step while a selected bot remains unloaded. Outputs with
fully bound dependency chains can run. Trying to enable an output that needs
an unloaded bot returns an error; we don't invent a sample or silently enable
only half the chain. Once the dependency bot loads, explicitly enable the
output again. Unrelated running sensors keep their state and acquisition clock.
Here, "unrelated" means sensors that neither belong to that bot nor depend on
its sensors, directly or indirectly.

For ordinary one-phase loading, `LoadBotScene` still prepares and loads all
bots by default. A non-empty `botsToInitialize` list selects and loads just
those bots; dependencies must be included. An explicit empty list defers bot
selection to `PrepareBots`, which fits the Unreal slot-selection workflow.
If the scene declares no bots, its scene-only graph can be prepared at load.

Existing callers that load with an empty initialization list must now call
`PrepareBots` before their first `InitializeBot`. A caller can prepare every
declared alternative once, then load only its current slot choices. That lets
it switch loaded alternatives while other bots keep running; it also means
all those alternatives' files must be readable and their combined graph valid.
A caller that needs to ignore unused files should prepare only its chosen set.

## Changing the cast starts a new configuration

We are not promising hot slot swaps halfway through an episode. To change the
selection, the caller stops stepping and removes the loaded bots, then calls
`PrepareBots` with the new complete selection. Preparation rebuilds and resets
the graph. Output selections must be supplied again. Loading and enabling the
new bots' outputs then starts the next configuration.

Changing selection while bots are loaded is an error. Preparing the same set
again does nothing, even if the names are passed in a different order.

Loading a bot that was already selected is a different operation. That binds
existing nodes and does not rebuild or reset the unrelated running graph.
Removing a loaded bot unbinds its sensors and affected downstream sensors;
the selected declarations remain, so the bot can be loaded again. Its fresh
buffers and running averages naturally need to warm up again.

The prepared prefab is a snapshot. Rebuilds and reinitialization don't reread
the file or run its edit callback again. To pick up changed prefab files or
edits, load a new BotScene. Changing a prepared entry's `path` is rejected;
it cannot silently load the old cached file. Set path overrides before
preparation. A `bot_paths` loader loads with no bots, updates the entries,
then prepares and initializes the full set.

## What the toy tests show

The tests use little clock and doubling sensors so the loading behavior is
visible without a complicated robot:

- A missing file for an unselected slot alternative is never opened or edited.
- A bot referencing an unselected dependency is rejected before construction.
- Edited names and input references are used by both declaration and construction.
- Selected bots can load in either order, including an A-to-B-to-A chain with no cycle.
- An independent output keeps stepping while another selected bot is unloaded.
- An unavailable output can be enabled after its dependency loads.
- A selection change requires unloading and resets the graph's clock and outputs.
- Rebuilding and reloading use the same prepared snapshot.
- Prepared alternatives can switch while an unrelated bot's clock keeps advancing.
- Changing a prepared bot's path is rejected rather than loading the stale file.

Finding all valid combinations of slot choices could be useful later. For now,
we validate the one complete selection the caller proposes.
