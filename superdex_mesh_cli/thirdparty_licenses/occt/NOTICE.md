# Open CASCADE Technology 7.9.3

`superdex_mesh_cli` statically links Open CASCADE Technology (OCCT) 7.9.3 for STEP import and
tessellation. OCCT is licensed under the GNU Lesser General Public License version 2.1 with the
Open CASCADE exception included in this directory.

The build applies Meta-authored changes to OCCT's CMake files using
[`patch_occt.cmake`](../../../superdex_physics/libraries/mochi/superdex_mesh_cli/cmake/patch_occt.cmake).
The matching Project SuperDex release tag contains the helper source, build scripts, patch, and
pinned OCCT source location needed to rebuild the executable with a modified OCCT copy.
