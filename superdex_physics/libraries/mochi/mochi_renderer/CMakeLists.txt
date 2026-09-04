# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

mochi_add_library(mochi_renderer)

# mochi_add_library globs src/ recursively, which pulls in
# src/third_party/cgltf_impl.cpp. That TU compiles the cgltf implementation, but
# the Filament-full renderer already obtains cgltf's symbols from gltfio_core, so
# compiling it here produces duplicate-symbol link errors. Only the Filament-free
# mochi_renderer_utils target (Buck-only) is meant to compile it. Exclude it here
# to match Buck's non-recursive glob(["src/*.cpp"]).
get_target_property(_mochi_renderer_srcs mochi_renderer SOURCES)
list(REMOVE_ITEM _mochi_renderer_srcs
    "${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/cgltf_impl.cpp"
)
set_target_properties(mochi_renderer PROPERTIES SOURCES "${_mochi_renderer_srcs}")
unset(_mochi_renderer_srcs)

# HACK: mochi_renderer does not have DLL export macros (e.g. MOCHI_RENDERER_API)
# on specific symbols. To function as a shared library, it currently exports ALL
# symbols
if (MOCHI_BUILD_SHARED)
    set_target_properties(mochi_renderer PROPERTIES
        C_VISIBILITY_PRESET default
        CXX_VISIBILITY_PRESET default
        WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif ()

if (MOCHI_BUILD_TESTS)
    # WORK-AROUND: the export-all hack above causes linker errors when mochi_core types
    # are used, because mochi_core is statically linked into mochi_renderer_test and all
    # mochi_core symbols are also exported from mochi_renderer. The real fix should be to
    # add export macros (e.g. MOCHI_RENDERER_API) to specific functions. If mochi_core
    # were a shared library, that would also fix it.
    if (MOCHI_BUILD_SHARED)
        target_compile_definitions(mochi_renderer_test PRIVATE MOCHI_USE_EXTERN_TEMPLATE=0)
    endif ()

    # stl_import_test.cpp includes "third_party/stl_reader.h", which is a private header
    # in the src directory.
    target_include_directories(mochi_renderer_test
        PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    # collada_reader_test.cpp includes the private collada_reader.h, which pulls in
    # <tinyxml2.h>. tinyxml2 is a PRIVATE dep of mochi_renderer, so it does not
    # propagate to the test; link it directly. glb_sections_test.cpp encodes its
    # meshopt fixtures with meshoptimizer, which is private for the same reason.
    target_link_libraries(mochi_renderer_test PRIVATE tinyxml2 meshoptimizer)

    # Meshes loaded from disk by glb_sections_test.cpp and stl_import_test.cpp.
    # Buck stages these; ctest reads them from the source tree. APPEND so the
    # MOCHI_ASSETS_PATH that mochi_add_test_executable sets survives.
    set_property(TEST mochi_renderer_test APPEND PROPERTY
        ENVIRONMENT "MOCHI_RENDERER_TEST_ASSETS=${CMAKE_CURRENT_SOURCE_DIR}/test/assets/")
endif ()

# -----------------------------------------------------------------------------
# Generated material resources
# -----------------------------------------------------------------------------
# Convert each resources/*.bin to a C source via Filament's bin_to_c.py.
# We run bin_to_c.py at CONFIGURE TIME (not via add_custom_command) so the
# generated .c files exist on disk when the target's Makefile is generated.
# The .bin files only change when someone updates the pre-built material
# blobs, so re-running on every configure is acceptable.
set(_mochi_renderer_materials
    Distortion
    FlatLitOpaque
    FlatLitSeeThrough
    GroundShadow
    HighlightComposite
    InstancedSphere
    LitOpaque
    LitSeeThrough
    LitVertexColor
    Outline
    UnlitSeeThrough
    UnlitVertexColor
    Wireframe
    WireframeDepth
)

set(_mochi_renderer_generated_resources_dir "${CMAKE_CURRENT_BINARY_DIR}/resources")
file(MAKE_DIRECTORY "${_mochi_renderer_generated_resources_dir}")

# bin_to_c.py ships with Filament.
set(_mochi_renderer_bin_to_c
    "${MOCHI_FILAMENT_SOURCE_DIR}/tools/bin_to_c.py"
)

set(_mochi_renderer_generated_sources)
foreach (_name IN LISTS _mochi_renderer_materials)
    string(TOUPPER "${_name}" _upper)
    set(_symbol "MOCHI_RENDERER_MATERIALS_${_upper}")
    set(_bin "${CMAKE_CURRENT_SOURCE_DIR}/resources/${_name}.bin")
    set(_c "${_mochi_renderer_generated_resources_dir}/${_name}.c")

    execute_process(
        COMMAND "${Python3_EXECUTABLE}"
            "${_mochi_renderer_bin_to_c}"
            "${_bin}"
            "${_c}"
            "${_symbol}"
        RESULT_VARIABLE _rc
    )
    if (NOT _rc EQUAL 0)
        message(FATAL_ERROR "[Mochi] bin_to_c.py failed for ${_name} (rc=${_rc})")
    endif ()

    list(APPEND _mochi_renderer_generated_sources ${_c})
endforeach ()
unset(_mochi_renderer_materials)
unset(_mochi_renderer_bin_to_c)

target_sources(mochi_renderer PRIVATE ${_mochi_renderer_generated_sources})

# -----------------------------------------------------------------------------
# Include directories
# -----------------------------------------------------------------------------
target_include_directories(mochi_renderer
    PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/resources"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# -----------------------------------------------------------------------------
# Preprocessor defines
# -----------------------------------------------------------------------------
if (WIN32)
    target_compile_definitions(mochi_renderer
        PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
endif ()

# -----------------------------------------------------------------------------
# Link libraries
# -----------------------------------------------------------------------------
target_link_libraries(mochi_renderer
    PUBLIC
        mochi_core
        filament
        gltfio_core
        math
        utils
    PRIVATE
        camutils
        filameshio
        filament-iblprefilter
        image
        imageio
        ktxreader
        cgltf
        dracodec
        meshoptimizer
        uberarchive
        tinyxml2
)

# Platform link libraries
if (APPLE)
    target_link_libraries(mochi_renderer
        PRIVATE
        "-framework Cocoa"
        "-framework CoreFoundation"
        "-framework CoreGraphics"
        "-framework CoreVideo"
        "-framework IOKit"
        "-framework IOSurface"
        "-framework Metal"
        "-framework OpenGL"
        "-framework QuartzCore"
    )
elseif (WIN32)
    target_link_libraries(mochi_renderer
        PRIVATE
        user32
        gdi32
        opengl32
    )
endif ()
