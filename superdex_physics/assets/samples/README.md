# Sample Scene Assemblies

## Overview

This directory contains SuperDex Physics scene assemblies with composition, simulation parameters, and static-environment configuration authored by Meta. The scenes reference assets stored in the source packages below.

## Referenced Asset Packages

| Package | Attribution and license |
| --- | --- |
| [`cube`](../cube/README.md) | Meta-authored, CC-BY-4.0 |
| [`implicit`](../implicit/README.md) | Meta-authored, CC-BY-4.0 |
| [`letters/alphanumeric`](../letters/alphanumeric/README.md) | Meta-authored mesh construction under CC-BY-4.0; Century Gothic Bold rights are excluded |
| [`sphere`](../sphere/README.md) | Meta-authored, CC-BY-4.0 |

## Scene Inventory

| Scene file | Referenced asset packages |
| --- | --- |
| `articulations_double_pendulum_on_rail.mochi_scene` | [`cube`](../cube/README.md), [`sphere`](../sphere/README.md), [static environments](#static-environments) |
| `articulations_pose_controller.mochi_scene` | [`cube`](../cube/README.md), [`sphere`](../sphere/README.md), [static environments](#static-environments) |
| `articulations_skinned_double_pendulum.mochi_scene` | [`cube`](../cube/README.md), [`sphere`](../sphere/README.md), [samples-local geometry](#samples-local-geometry), [static environments](#static-environments) |
| `articulations_soft_skinned_double_pendulum.mochi_scene` | [`cube`](../cube/README.md), [`sphere`](../sphere/README.md), [samples-local geometry](#samples-local-geometry), [static environments](#static-environments) |
| `constraints_double_pendulum.mochi_scene` | [`cube`](../cube/README.md) |
| `tendon_comparison_articulation.mochi_scene` | [`cube`](../cube/README.md), [`letters/alphanumeric`](../letters/alphanumeric/README.md) |

## Static Environments

| File | Referenced asset packages |
| --- | --- |
| `static_environments/ground_plane.mochi_prefab` | [`implicit`](../implicit/README.md) |

## Samples-Local Geometry

The following assets are the original work of Meta Platforms, Inc. and affiliates:

- `slit_annular_ring.mochi.h5`
- `articulations_parts/skin.mochi.json`
- `articulations_parts/soft.mochi.json`

## License

The Meta-authored scene composition, simulation parameters, static-environment configuration, and samples-local geometry in the following files are the original work of Meta Platforms, Inc. and affiliates and are distributed under CC BY 4.0; see [`LICENSE`](LICENSE):

- `articulations_double_pendulum_on_rail.mochi_scene`
- `articulations_pose_controller.mochi_scene`
- `articulations_skinned_double_pendulum.mochi_scene`
- `articulations_soft_skinned_double_pendulum.mochi_scene`
- `constraints_double_pendulum.mochi_scene`
- `tendon_comparison_articulation.mochi_scene`
- `static_environments/ground_plane.mochi_prefab`
- `slit_annular_ring.mochi.h5`
- `articulations_parts/skin.mochi.json`
- `articulations_parts/soft.mochi.json`

The local license applies only to the Meta-authored files named above and does not relicense referenced assets. Referenced assets remain governed by the licenses and notices in their source directories.

**You are responsible for ensuring your use is compatible with all third-party licenses of the referenced sub-assets.**
