# Third-party software

SuperDex Recorder incorporates the components listed below. License texts for all but the Rust
crates are provided in the adjacent directories.

| Component | License | Source |
| --- | --- | --- |
| Apache Arrow C++ | Apache-2.0 and bundled third-party terms | https://github.com/apache/arrow |
| LZ4 library | BSD-2-Clause | https://github.com/lz4/lz4 |
| mimalloc | MIT | https://github.com/microsoft/mimalloc |
| Rerun | MIT or Apache-2.0 | https://github.com/rerun-io/rerun |
| Rust crates linked into Rerun's C library (`rerun_c`) | Each crate's own license | https://github.com/rerun-io/rerun/tree/0.33.0/crates/top/rerun_c |
| xsimd | BSD-3-Clause | https://github.com/xtensor-stack/xsimd |

The Apache Arrow build bundles mimalloc and xsimd.
