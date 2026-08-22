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

# superdex_studio is a Filament-based editor app. Its sources live under src/. Following the Buck
# build (superdex_studio/BUCK), it is split into a static library (superdex_studio_static) holding
# all non-main sources and a thin executable (superdex_studio) that links it. See below.

#-----------------------------------------------------------------------------
# imguizmo (transform gizmo)
#-----------------------------------------------------------------------------

# imguizmo is a vendored third-party library, and only the studio uses it. Only ImGuizmo.cpp is
# needed (the other translation units -- sequencer, curve/gradient editors, graph editor -- are
# unused here). It builds against the imgui target provided by imguios.
if (NOT TARGET imguizmo)
    # Prefer the studio-local copy; fall back to the repository-wide third-party copy.
    set(_imguizmo_oss "${CMAKE_CURRENT_SOURCE_DIR}/third_party/imguizmo")
    set(_imguizmo_central "${CMAKE_CURRENT_SOURCE_DIR}/../../../../third-party/imguizmo")
    if (EXISTS "${_imguizmo_oss}/ImGuizmo.cpp")
        set(_imguizmo_dir "${_imguizmo_oss}")
    else ()
        set(_imguizmo_dir "${_imguizmo_central}")
    endif ()
    add_library(imguizmo STATIC "${_imguizmo_dir}/ImGuizmo.cpp")
    target_include_directories(imguizmo PUBLIC "${_imguizmo_dir}")
    target_link_libraries(imguizmo PUBLIC imgui)
    target_compile_definitions(imguizmo PUBLIC IMGUI_DEFINE_MATH_OPERATORS)
    set_target_properties(imguizmo PROPERTIES
        FOLDER "third_party"
        POSITION_INDEPENDENT_CODE ${MOCHI_BUILD_SHARED}
    )
    mochi_target_ignore_all_warnings(imguizmo)
    unset(_imguizmo_dir)
    unset(_imguizmo_oss)
    unset(_imguizmo_central)
endif ()

#-----------------------------------------------------------------------------
# superdex_studio_static – static library containing all non-main sources
#-----------------------------------------------------------------------------

file(GLOB_RECURSE _studio_static_srcs
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
)
list(FILTER _studio_static_srcs EXCLUDE REGEX "/internal/")
# Exclude the entry point – matches Buck's exclude = ["app/main.cpp"]
list(FILTER _studio_static_srcs EXCLUDE REGEX "/app/main\\.cpp$")

add_library(superdex_studio_static STATIC ${_studio_static_srcs})

target_include_directories(superdex_studio_static
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

# third_party/ holds the vendored stl_reader.h; SYSTEM keeps its diagnostics out of Studio's build.
target_include_directories(superdex_studio_static SYSTEM PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

# PUBLIC so that consumers (the superdex_studio executable) inherit its link dependencies.
target_link_libraries(superdex_studio_static
    PUBLIC
        mochi_core
        mochi_physics
        mochi_renderer
        superdex_robotics
        mochi_mesh
        imguizmo
        tinyfiledialogs
        picojson
        simple_reflection
)

target_compile_definitions(superdex_studio_static
    PRIVATE
        FILAMENT_DISABLE_MATOPT=0
        IMGUI_DEFINE_MATH_OPERATORS
        MOCHI_USE_REFLECTION=1
)

if (MSVC)
    # Studio TUs freely mix Windows-macro-polluting headers with Filament's math headers (which use
    # `near`/`far`/`min`/`max` as identifiers). Force-include mochi_renderer's windows_compat.h before
    # every TU so Windows.h is pulled in first and its conflicting macros are undef'd -- mirroring the
    # Buck build, which force-includes the same header on MSVC.
    target_compile_options(superdex_studio_static PRIVATE
        "/FI${SUPERDEX_MOCHI_SOURCE_DIR}/mochi_renderer/include/mochi_renderer/windows_compat.h")
endif ()

if (APPLE)
    # Mirrors Buck's select macOS glob(["**/*.mm"]) in superdex_studio_static.
    target_sources(superdex_studio_static
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src/rendering/render_target_metal.mm")
    target_link_libraries(superdex_studio_static PUBLIC imguios_metal)
else ()
    target_link_libraries(superdex_studio_static PUBLIC imguios)
endif ()

mochi_set_compiler_options(superdex_studio_static)
mochi_set_output_directories(superdex_studio_static)

if (MOCHI_BUILD_SHARED)
    set_target_properties(superdex_studio_static PROPERTIES
        POSITION_INDEPENDENT_CODE ${MOCHI_BUILD_SHARED}
    )
endif ()

#-----------------------------------------------------------------------------
# superdex_studio – thin executable linking the static library
#-----------------------------------------------------------------------------

add_executable(superdex_studio "${CMAKE_CURRENT_SOURCE_DIR}/src/app/main.cpp")

target_include_directories(superdex_studio
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_include_directories(superdex_studio SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

target_link_libraries(superdex_studio
    PRIVATE
        superdex_studio_static
)

target_compile_definitions(superdex_studio
    PRIVATE
        FILAMENT_DISABLE_MATOPT=0
        IMGUI_DEFINE_MATH_OPERATORS
        MOCHI_USE_REFLECTION=1
)

if (MSVC)
    target_compile_options(superdex_studio PRIVATE
        "/FI${SUPERDEX_MOCHI_SOURCE_DIR}/mochi_renderer/include/mochi_renderer/windows_compat.h")
endif ()

mochi_set_compiler_options(superdex_studio)
mochi_set_output_directories(superdex_studio)
mochi_set_sibling_library_rpath(superdex_studio)
mochi_set_gui_subsystem(superdex_studio)

#-----------------------------------------------------------------------------
# Runtime resources
#-----------------------------------------------------------------------------

# Stage runtime resources next to the executable so the app finds them without the source tree:
#   - processing_presets/ : discovered relative to the exe by preset_discovery (mirrors Buck's
#                           `resources`); can also be overridden via MOCHI_STUDIO_PRESETS_DIR.
#   - assets/             : basic shapes and the IBL/environment map used by the editor.
# $<TARGET_FILE_DIR:...> resolves to the real per-config output dir (e.g. bin/Release on MSVC).
foreach (_studio_resource_dir processing_presets assets)
    add_custom_command(TARGET superdex_studio POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/${_studio_resource_dir}"
            "$<TARGET_FILE_DIR:superdex_studio>/${_studio_resource_dir}"
        COMMENT "[superdex_studio] Staging ${_studio_resource_dir}/ next to the executable"
        VERBATIM)
endforeach ()
unset(_studio_resource_dir)

# The `superdex_studio` component tag lets a consumer select exactly this app's files and leave the
# debugger and mesh helper behind.
mochi_install_runtime_dependencies(superdex_studio COMPONENT superdex_studio)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/processing_presets"
    DESTINATION . COMPONENT superdex_studio)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/assets"
    DESTINATION . COMPONENT superdex_studio)

unset(_studio_static_srcs)
