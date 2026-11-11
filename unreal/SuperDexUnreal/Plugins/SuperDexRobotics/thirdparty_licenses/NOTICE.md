# Third-party software

This plugin incorporates the components listed below. License texts for all but the Rust crates
are provided in the adjacent directories.

The SuperDex Robotics native library (`superdex_robotics`) incorporates:

| Component | License | Source |
| --- | --- | --- |
| CityHash | MIT | https://github.com/google/cityhash |
| Eigen | MPL-2.0 and bundled third-party terms | https://gitlab.com/libeigen/eigen |
| happly | MIT | https://github.com/nmwsharp/happly |
| HDF5 | HDF5 license | https://github.com/HDFGroup/hdf5 |
| Marl | Apache-2.0 | https://github.com/google/marl |
| miniz | MIT | https://github.com/richgel999/miniz |
| picojson derivative (Steven Lovegrove modifications) | BSD-2-Clause | https://github.com/kazuho/picojson |
| tinyobjloader | MIT | https://github.com/tinyobjloader/tinyobjloader |
| TinyXML-2 | zlib | https://github.com/leethomason/tinyxml2 |
| xxHash | BSD-2-Clause | https://github.com/Cyan4973/xxHash |
| zlib | zlib | https://github.com/madler/zlib |

The SuperDex Recorder native library (`superdex_recorder_shared`, Win64) incorporates the
components above and:

| Component | License | Source |
| --- | --- | --- |
| Apache Arrow C++ and its bundled third-party code | Apache-2.0 and bundled third-party terms | https://github.com/apache/arrow |
| FlatBuffers | Apache-2.0 | https://github.com/google/flatbuffers |
| mimalloc | MIT | https://github.com/microsoft/mimalloc |
| Rerun C++ SDK | MIT or Apache-2.0 | https://github.com/rerun-io/rerun |
| Rust crates linked into Rerun's C library (`rerun_c`) | Each crate's own license | https://github.com/rerun-io/rerun/tree/0.33.0/crates/top/rerun_c |
| xsimd | BSD-3-Clause | https://github.com/xtensor-stack/xsimd |

The vendored picojson header identifies Kazuho Oku's repository as its original upstream and
retains a 2014 modification notice for Steven Lovegrove.
