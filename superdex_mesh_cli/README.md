# SuperDex Mesh CLI

An out-of-process mesh geometry helper for
[Project SuperDex](https://github.com/facebookresearch/project_superdex).

This package ships a single helper executable and has no importable API. It reads one framed
request from stdin, runs a single geometry operation, writes a framed response to stdout, and
then exits. SuperDex Studio launches it for you, so you never need to run it yourself.

## Why it ships on its own

This is the only SuperDex binary that links [CGAL](https://www.cgal.org/), whose combinatorial
and geometry code is licensed GPL-3.0-or-later. Keeping it in a package of its own lets the rest
of SuperDex (including SuperDex Studio, which depends on it) stay outside that GPL boundary:
other components talk to the helper over a pipe and never link against it.

The helper also statically links [OpenCASCADE](https://dev.opencascade.org/) (OCCT) so it
can read STEP files. OpenCASCADE is licensed LGPL-2.1 with an exception.

The OCCT source is patched at build time by
[`patch_occt.cmake`](../superdex_physics/libraries/mochi/superdex_mesh_cli/cmake/patch_occt.cmake),
so the matching release tag includes everything needed to rebuild the executable yourself: the
helper source, the build scripts, and the pinned location of the OCCT source. See
[`thirdparty_licenses/occt`](thirdparty_licenses/occt) for the LGPL text, its exception, and the
notice describing our changes to OCCT.

## How callers find it

`mochi_mesh` looks for the helper in this order:

1. the `SUPERDEX_MESH_CLI_PATH` environment variable;
2. the helper sitting next to the program that calls it;
3. `../../superdex_mesh_cli/_native/`, relative to that program. This is the layout `pip`
   installs, where the helper lands beside its caller rather than inside it.

## License

This package is **GPL-3.0-or-later**. Other SuperDex packages license their own code under
Apache-2.0, and their third-party code keeps whatever license it already carries. The
corresponding source for this helper and its CGAL dependency is at
<https://github.com/facebookresearch/project_superdex>.
