# Third-party software

This wheel incorporates the components listed below. License texts for all but the Rust
crates are provided in the adjacent directories.

| Component | License | Source |
| --- | --- | --- |
| Apache Arrow C++ | Apache-2.0 and bundled third-party terms | https://github.com/apache/arrow |
| Basis Universal and its bundled Zstandard code | Apache-2.0, BSD, zlib, and Zstandard terms | https://github.com/BinomialLLC/basis_universal |
| cgltf | MIT | https://github.com/jkuhlmann/cgltf |
| CityHash | MIT | https://github.com/google/cityhash |
| Dear ImGui | MIT | https://github.com/ocornut/imgui |
| Draco | Apache-2.0 | https://github.com/google/draco |
| Eigen | MPL-2.0 and bundled third-party terms | https://gitlab.com/libeigen/eigen |
| EnTT | MIT | https://github.com/skypjack/entt |
| FidelityFX Super Resolution 1 | MIT | https://github.com/GPUOpen-Effects/FidelityFX-FSR |
| Filament | Apache-2.0 | https://github.com/google/filament |
| Font Awesome Free 5 font | SIL OFL-1.1 | https://github.com/FortAwesome/Font-Awesome |
| GLAD generated code and Khronos platform header | Public domain and MIT | https://github.com/Dav1dde/glad |
| GLFW | zlib | https://github.com/glfw/glfw |
| happly | MIT | https://github.com/nmwsharp/happly |
| HDF5 | HDF5 license | https://github.com/HDFGroup/hdf5 |
| ImGuizmo | MIT | https://github.com/CedricGuillemet/ImGuizmo |
| ImPlot | MIT | https://github.com/epezent/implot |
| libpng | libpng license | https://github.com/pnggroup/libpng |
| LZ4 library | BSD-2-Clause | https://github.com/lz4/lz4 |
| Marl | Apache-2.0 | https://github.com/google/marl |
| meshoptimizer | MIT | https://github.com/zeux/meshoptimizer |
| MikkTSpace | zlib | https://github.com/mmikk/MikkTSpace |
| mimalloc | MIT | https://github.com/microsoft/mimalloc |
| miniz | MIT | https://github.com/richgel999/miniz |
| picojson derivative (Steven Lovegrove modifications) | BSD-2-Clause | https://github.com/kazuho/picojson |
| Rerun | MIT or Apache-2.0 | https://github.com/rerun-io/rerun |
| Rust crates linked into Rerun's C library (`rerun_c`) | Each crate's own license | https://github.com/rerun-io/rerun/tree/0.33.0/crates/top/rerun_c |
| robin-map | MIT | https://github.com/Tessil/robin-map |
| Roboto and Roboto Mono fonts | Apache-2.0 | https://github.com/googlefonts/roboto |
| smol-v | Apache-2.0 | https://github.com/aras-p/smol-v |
| Snapdragon Game Super Resolution | BSD-3-Clause | https://github.com/SnapdragonStudios/snapdragon-gsr |
| SPIRV-Headers | MIT | https://github.com/KhronosGroup/SPIRV-Headers |
| stb | MIT or public domain | https://github.com/nothings/stb |
| stl_reader | BSD-2-Clause | https://github.com/sreiter/stl_reader |
| tinyexr | BSD-3-Clause | https://github.com/syoyo/tinyexr |
| tinyfiledialogs | zlib | https://sourceforge.net/projects/tinyfiledialogs/ |
| tinyobjloader | MIT | https://github.com/tinyobjloader/tinyobjloader |
| TinyXML-2 | zlib | https://github.com/leethomason/tinyxml2 |
| Vulkan Memory Allocator | MIT | https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator |
| xsimd | BSD-3-Clause | https://github.com/xtensor-stack/xsimd |
| xxHash | BSD-2-Clause | https://github.com/Cyan4973/xxHash |
| zlib | zlib | https://github.com/madler/zlib |

The Apache Arrow build bundles mimalloc and xsimd. The vendored picojson header identifies
Kazuho Oku's repository as its original upstream and retains a 2014 modification notice for
Steven Lovegrove.
