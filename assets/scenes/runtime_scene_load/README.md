# Runtime Scene Load

The scene of SuperDex Unreal's sampler level (`SuperDexExamples/Content/Sampler/L_Sampler`): the
workbench, the arms and proxy hands the level's roboperator spawner offers each hand, and the
prefabs its prefab spawner places on the table that make a task.

| File | What |
|---|---|
| `runtime_scene_load.superdex_scene` | The bots and their controllers (parameters in each bot's `control/` folder), the `left` and `right` slots, the spawnable prefabs and the task root |
| `base_scene/` | The base scene, `runtime_scene_load.mochi_scene`: a floor plane at z = 0 and the workbench, under Unreal's default gravity (9.8 m/s²) |
| `tasks/` | One task per spawnable prefab, each placing it at the task root, in `BP_Sampler_PrefabSpawner`'s order |

## Tasks

The tasks are `L_Sampler`'s prefabs with a clear objective; five start partway, from a variant
beside the prefab:

| Task | Start | Objective |
|---|---|---|
| `01_paper_cup_pyramid` | `paper_cup_pyramid_half_built`: the bottom two rows, three cups nested beside them | Finish the pyramid |
| `02_shape_box` | `shape_box_four_out`: the easiest piece for each face on the table | Put the four pieces in the box |
| `03_puzzle_cube` | `puzzle_cube_scrambled`: two quarter turns from solved | Solve the cube |
| `04_functional_dexterity_test` | `functional_dexterity_test_partly_flipped`: four pegs not yet flipped | Flip and reinsert them |
| `05_nine_hole_peg_test` | `nine_hole_peg_test_transfer`: the test and an empty board on the operator's right | Move the pegs to the right board |
| `06_brick_toys` | The eight bricks | Stack them in one tower |
| `07_jar` | The closed jar | Take the lid off |

## Matching the Unreal Level

- Each bot sits where `BP_RoboperatorSpawner` places its roboperator, starts in that roboperator's
  initial pose, and names it as its `roboperatorClass`.
- The controllers split the roboperators' `UController_ArmOsc_HandJsc` into a `BASIC_OSC_PD` arm and
  a `BASIC_JSC_PD` hand, with the gains of their `BP_Controller_*` Blueprints.
- The task root is where `BP_Sampler_PrefabSpawner` places prefabs: the middle of the table top,
  facing the operator.
- Each slot offers the FR3 + DG-5F SEED and Tianji M6 + Wuji Hand 2 arms, then the proxy hands
  (OpenXR, Wuji Hand 2, DG-5F SEED and 2F-85), which a `MOCHI_ARTICULATED_POSE` controller moves to
  the tracked hand. `BP_RoboperatorSpawner` starts on the OpenXR hand; the slots start on the FR3.
- As in Unreal, the bots are built without gravity (`hasGravity: false`). Unreal's arm controller
  also pulls the arm toward its initial pose in the null space of the end effector; the scene's
  does not.

See the READMEs, LICENSEs and NOTICEs of the bots and prefabs the scene references.

**You are responsible for ensuring your use is compatible with all third-party licenses of the
referenced sub-assets.**
