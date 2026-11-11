# Third-party software

This plugin incorporates the components listed below. Their license texts are
provided in the adjacent directories.

The plugin carries two native libraries: the SuperDex Physics library (`mochi_physics`) and the
third-party Marl library (`marl`) that it links. Together they incorporate:

| Component | License | Source |
| --- | --- | --- |
| CityHash | MIT | https://github.com/google/cityhash |
| Eigen | MPL-2.0 and bundled third-party terms | https://gitlab.com/libeigen/eigen |
| EnTT | MIT | https://github.com/skypjack/entt |
| happly | MIT | https://github.com/nmwsharp/happly |
| HDF5 | HDF5 license | https://github.com/HDFGroup/hdf5 |
| Marl | Apache-2.0 | https://github.com/google/marl |
| picojson derivative (Steven Lovegrove modifications) | BSD-2-Clause | https://github.com/kazuho/picojson |
| tinyobjloader | MIT | https://github.com/tinyobjloader/tinyobjloader |
| zlib | zlib | https://github.com/madler/zlib |

The `SuperDexPhysics` module's force visualizer includes:

| Component | License | Source |
| --- | --- | --- |
| Turbo colormap lookup table (Copyright 2019 Google LLC) | Apache-2.0 | https://gist.github.com/mikhailov-work/6a308c20e494d9e0ccc29036b28faa7a |

The vendored picojson header identifies Kazuho Oku's repository as its original upstream and
retains a 2014 modification notice for Steven Lovegrove.
