# Box and blocks: FR3 V2 + 2F-85 pick and place

An FR3 V2 arm with a Robotiq 2F-85 gripper (`@bots/arm_hand_combos/fr3_v2_2f_85`) on the SuperDex workbench, picking a block out of a Box and Block Test box and placing it over the partition. It uses the prefabs `@prefabs/workbench` and `@prefabs/box_and_blocks/box_and_two_blocks.superdex_prefab` (the box with one red and one blue block in one compartment).

| File | What it is |
| :---- | :---- |
| `box_and_blocks.superdex_scene` | The workbench (top at z = 0.9 m, the scene's `taskRoot`; the base scene has no floor), and the arm on the workbench top 0.14 m behind its center, facing +x. Controllers: `osc` (`BASIC_OSC_PD`, tool at the finger pads, 0.13 m past `fr3_link8`) and `gripper` (`BASIC_JSC_PD`, driving only the two knuckle joints). The arm has no gravity: its controller stands in for the real arm's gravity compensation. |
| `box_and_blocks.superdex_task` | Spawns the box (static) centered 0.15 m in front of the workbench center, long side along y, its blocks in the -y compartment. Success: for 0.5 s, the red block rests in the +y compartment and the gripper is at least 20 cm from it (released). Failure: the red block falls off the workbench. Time limit 20 s. |
| `pick_place_red.superdex_tuner` | A SuperDex Asset Tuner script that solves the task: tool waypoints for `fr3/osc` and knuckle keys for `fr3/gripper`. |

The task's criteria name the red block as `box_and_blocks.Block_red/Block_red` and the box as `box_and_blocks.Box`. Both have their origin at a corner, as in `box_and_blocks.mochi_prefab`: in the box's frame, the +y compartment is x 0.2735 to 0.527 m, y 0.01 to 0.244 m, above its floor at z = 0.01 m.

## Scripted solve

| Time [s] | Arm (`arm` load, tool pose in the arm's root frame, pointing down) | Gripper (`grip` load) |
| :---- | :---- | :---- |
| 0 – 1.5 | home to 10 cm above the red block, fingers closing along x | open |
| 1.5 – 2.5 | down until the pads reach the block's upper 2 cm | open |
| 2.6 – 3.3 | holds | closes (knuckles to 0.8 rad; the block stops them at 0.6) |
| 3.4 – 4.4 | lifts to 26 cm above the workbench, clear of the 16 cm partition | holds |
| 4.4 – 5.9 | carries the block over to the +y compartment | holds |
| 5.9 – 6.9 | lowers it to 6 mm above the compartment floor | holds |
| 6.9 – 7.5 | holds | opens |
| 7.6 – 8.6 | retreats 8 cm up | open |

The task succeeds at 8.85 s. Every segment eases in and out.

From `assets/`:

```
superdex_asset_tuner run tasks/box_and_blocks/pick_place_red.superdex_tuner
```

It runs in about 1.2 s at its 10 ms step, and also succeeds at 5 ms (1.8 s) and 2 ms (4 s). Moving the grasp by ±5 mm in x and y still succeeds; with the `grip` load disabled the block is never picked and the run fails at 12 s. In SuperDex Studio, open the tuner and press Play to watch it, with the solve time, the block's position and the loads' errors plotted; open the scene to edit the task's criteria (Task Preview).
