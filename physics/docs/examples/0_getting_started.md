---
title: Overview
sidebar_position: 1
---

# Get Started with SuperDex Physics

These examples walk through SuperDex Physics' example programs, explaining the key concepts and API patterns demonstrated by each one. Each example corresponds to a runnable program in the `examples/` directory.

## Examples

- Basic
  - [Rigid Bodies](basic/rigid_bodies.md) — `examples/example_rigid_bodies.py`
  - [Soft Duck](basic/soft_duck.md) — `examples/example_soft_duck.py`
  - [State Capture](basic/state_capture.md) — `examples/example_state_capture_restore.py`
- Contact & Constraints
  - [Contact Filtering](contact_constraints/contact_filtering.md) — `examples/example_contact_filtering.py`
  - [Constraints](contact_constraints/constraints.md) — `examples/example_constraints_double_pendulum.py`
- Articulations
  - [Double Pendulum on Rail](articulations/double_pendulum_on_rail.md) — `examples/example_articulations_double_pendulum_on_rail.py`
  - [Skinned Double Pendulum](articulations/skinned_double_pendulum.md) — `examples/example_articulations_skinned_double_pendulum.py`
  - [Soft-Skinned Double Pendulum](articulations/soft_skinned_double_pendulum.md) — `examples/example_articulations_soft_skinned_double_pendulum.py`
  - [Articulation Pose Controller](articulations/pose_controller.md) — `examples/example_articulations_pose_controller.py`
  - [Inverse Kinematics](articulations/ik.md) — `examples/example_ik.py`
- Shell
  - [T-shirt on Plane](shell/tshirt_on_plane.md) — `examples/example_tshirt_on_plane.py`
  - [Slit Annular Ring](shell/slit_annular_ring.md) — `examples/example_slit_annular_ring.py`
- Tendons & Rods
  - [Mass on Rod Spring](tendons_rods/mass_on_rod_spring.md) — `examples/example_mass_on_rod_spring.py`
  - [Tendon Comparison](tendons_rods/tendon_model_fidelity.md) — `examples/example_tendon_comparison.py`
- Advanced
  - [Soft Duck with Visual Mesh](advanced/soft_duck_visual_mesh.md) — `examples/example_soft_duck_visual_mesh.py`
  - [Damping Parameter Sweep](advanced/damping_sweep.md) — `examples/example_damping_sweep.py`
  - [Cross-Thread Capture/Restore](advanced/cross_thread_capture_restore.md) — `examples/example_cross_thread_capture_restore.py`

## Running Examples

Examples can be run with `uv`:

```bash
cd <path_to_superdex_physics>
uv run --no-project examples/example_rigid_bodies.py
```

Most examples automatically launch the SuperDex Physics Debugger for inspecting and controlling the simulated scene. See [Inspecting Scenes](../debugging_scenes.md) for navigation and playback controls.
