# superdex_mesh_cli — the GPL-isolated geometry helper

`superdex_mesh_cli` is the **only** Mochi target that links **CGAL**. CGAL is **GPL**, which cannot
be linked into the open-source shipping libraries (`mochi_core`, `mochi_mesh`). To keep those
libraries free of GPL code, the genuine CGAL geometry algorithms live here, in a separate executable
that the shipping libraries invoke **over a pipe** (spawn-per-request). No shipping library links any
CGAL symbol.

## What runs here

- `RemeshSurface` — alpha-wrap / ACVD / Surface-Delaunay remeshing and repair.
- `ReconstructSurfaceFromSdf` — marching-cubes isosurface from a Cartesian SDF grid.
- `ApproximateHausdorffDistance` — one-sided approximate Hausdorff distance between two meshes.

`main()` reads one framed request from stdin to EOF, dispatches, writes a framed response to stdout,
and exits. The wire format is defined in `mochi_mesh/protocol` (CGAL-free). Coordinates are
`double` on the wire, so it is lossless for both single- and double-precision callers.

## How callers find it

`mochi_mesh/src/mesh_cli_client.{h,cpp}` resolves the helper via, in order:

1. the `SUPERDEX_MESH_CLI_PATH` environment variable (tests and packaging), then
2. the helper sitting next to the current executable, then
3. `../../superdex_mesh_cli/_native/` relative to the current executable — the wheel layout, where
   the GPL helper is its own distribution and so installs beside its caller's rather than inside it.

A missing or unspawnable helper yields a clean error (never a crash).

## Packaging / licensing

- **Buck:** built on demand; `mochi_mesh`'s pybind module and tests package this executable as a
  runtime resource and wire `SUPERDEX_MESH_CLI_PATH`.
- **CMake:** a standalone build, **not** part of the default aggregate. Enable it explicitly with
  `-DMOCHI_BUILD_MESH_CLI=ON`; the core/mesh CMake builds never link CGAL.
- **Open-source release:** this helper, the vendored CGAL, and the internal `mochi_cgal_utils` asset
  tool are the GPL components. They are shipped/licensed **separately** from the permissive core. Any
  bundling of this helper with the OSS libraries is a licensing decision — route it through OSS /
  Third-Party review.
