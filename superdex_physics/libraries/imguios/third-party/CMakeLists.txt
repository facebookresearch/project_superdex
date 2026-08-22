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

include(FetchContent)

function (imguios_target_ignore_all_warnings target_name)
    if (MSVC)
        target_compile_options(${target_name} PRIVATE /W0)
    else ()
        target_compile_options(${target_name} PRIVATE -w)
    endif ()
endfunction ()

#############################################################
# glad
#############################################################

# Overwrite GLAD_SOURCE_DIR if you want to build glad source from a different location.
set(GLAD_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/glad" CACHE STRING "Location of glad source")

FetchContent_Declare(
        glad
        SOURCE_DIR ${GLAD_SOURCE_DIR}
)
FetchContent_MakeAvailable(glad)
add_library(glad)
target_sources(glad
        PRIVATE
        ${glad_SOURCE_DIR}/src/glad.c
        PUBLIC FILE_SET HEADERS
        BASE_DIRS ${glad_SOURCE_DIR}/include/
        FILES
        ${glad_SOURCE_DIR}/include/glad/glad.h
)

imguios_target_ignore_all_warnings(glad)
target_compile_features(glad PRIVATE cxx_std_17)
set_target_properties(glad PROPERTIES FOLDER "third_party")

#############################################################
# glfw
#############################################################

# Overwrite GLFW_SOURCE_DIR if you want to build glfw source from a different location.
set(GLFW_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/glfw" CACHE STRING "Location of glfw source")

# Disable stuff we don't need
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Build the GLFW example programs" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "Build the GLFW test programs" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "Build the GLFW documentation" FORCE)
set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "Build the GLFW documentation" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "Generate installation target" FORCE)

# Check Linux packages
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        find_package(X11 QUIET)
        if (NOT X11_FOUND OR NOT X11_Xrandr_INCLUDE_PATH OR NOT X11_Xinerama_INCLUDE_PATH OR NOT X11_Xkb_INCLUDE_PATH OR NOT X11_Xcursor_INCLUDE_PATH OR NOT X11_Xi_INCLUDE_PATH OR NOT X11_Xshape_INCLUDE_PATH)
                message(WARNING "Failed to find X11 package contents. Install the X11 development libraries and try again. On Fedora run: sudo dnf install libX11-devel libXext-devel libXft-devel libXi-devel libXinerama-devel libXrandr-devel libXrender-devel libXt-devel libXtst-devel libXv-devel libXxf86vm-devel libXcursor-devel libxkbcommon-devel")
                set(GLFW_BUILD_X11 OFF CACHE BOOL "Build GLFW with X11 support" FORCE)
        else ()
                set(GLFW_BUILD_X11 ON CACHE BOOL "Build GLFW with X11 support" FORCE)
        endif ()

        find_package(OpenGL QUIET)
        if (NOT OpenGL_FOUND)
                message(WARNING "Failed to find OpenGl package. Install the OpenGL development libraries and try again. On Fedora run: sudo dnf install mesa-libGL-devel")
        endif ()
endif()

FetchContent_Declare(
        glfw
        SOURCE_DIR ${GLFW_SOURCE_DIR}
)
FetchContent_MakeAvailable(glfw)

imguios_target_ignore_all_warnings(glfw)
target_compile_features(glfw PRIVATE cxx_std_17)
set_target_properties(glfw PROPERTIES FOLDER "third_party")
set_target_properties(update_mappings PROPERTIES FOLDER "third_party")

#############################################################
# imgui
#############################################################

# Overwrite IMGUI_SOURCE_DIR if you want to build imgui source from a different location.
set(IMGUI_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/imgui" CACHE STRING "Location of imgui source")

FetchContent_Declare(
        imgui
        SOURCE_DIR ${IMGUI_SOURCE_DIR}
)
FetchContent_MakeAvailable(imgui)
add_library(imgui)
target_sources(imgui
        PRIVATE
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_internal.h
        ${imgui_SOURCE_DIR}/imstb_rectpack.h
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imstb_textedit.h
        ${imgui_SOURCE_DIR}/imstb_truetype.h
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.h
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.h
        ${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp
        ${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.h
        PUBLIC FILE_SET HEADERS
        BASE_DIRS
        ${imgui_SOURCE_DIR}
        FILES
        ${imgui_SOURCE_DIR}/imgui.h
        ${imgui_SOURCE_DIR}/imconfig.h
)

imguios_target_ignore_all_warnings(imgui)
target_compile_features(imgui PRIVATE cxx_std_17)
target_include_directories(imgui PUBLIC "${imgui_SOURCE_DIR}/backends")
# SYSTEM marks vendored headers as external so their diagnostics are suppressed in consumer code.
set_property(TARGET imgui PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${imgui_SOURCE_DIR};${imgui_SOURCE_DIR}/backends")
target_link_libraries(imgui PUBLIC glfw)
set_target_properties(imgui PROPERTIES FOLDER "third_party")

#############################################################
# imgui_metal (macOS only)
#############################################################

if (APPLE AND IMGUIOS_BUILD_METAL)
add_library(imgui_metal)
target_sources(imgui_metal
        PRIVATE
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_internal.h
        ${imgui_SOURCE_DIR}/imstb_rectpack.h
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imstb_textedit.h
        ${imgui_SOURCE_DIR}/imstb_truetype.h
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.h
        ${imgui_SOURCE_DIR}/backends/imgui_impl_metal.mm
        ${imgui_SOURCE_DIR}/backends/imgui_impl_metal.h
        ${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp
        ${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.h
        PUBLIC FILE_SET HEADERS
        BASE_DIRS
        ${imgui_SOURCE_DIR}
        FILES
        ${imgui_SOURCE_DIR}/imgui.h
        ${imgui_SOURCE_DIR}/imconfig.h
)

imguios_target_ignore_all_warnings(imgui_metal)
target_compile_features(imgui_metal PRIVATE cxx_std_17)
target_include_directories(imgui_metal PUBLIC "${imgui_SOURCE_DIR}/backends")
# SYSTEM marks vendored headers as external so their diagnostics are suppressed in consumer code.
set_property(TARGET imgui_metal PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${imgui_SOURCE_DIR};${imgui_SOURCE_DIR}/backends")
target_link_libraries(imgui_metal PUBLIC glfw)
target_link_libraries(imgui_metal PUBLIC "-framework Metal" "-framework QuartzCore")
set_target_properties(imgui_metal PROPERTIES FOLDER "third_party")
endif()

#############################################################
# implot
#############################################################

# Overwrite IMPLOT_SOURCE_DIR if you want to build implot source from a different location.
set(IMPLOT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/implot" CACHE STRING "Location of implot source")

FetchContent_Declare(
        implot
        SOURCE_DIR ${IMPLOT_SOURCE_DIR}
)
FetchContent_MakeAvailable(implot)
add_library(implot)
target_sources(implot
        PRIVATE
        ${implot_SOURCE_DIR}/implot.cpp
        ${implot_SOURCE_DIR}/implot_demo.cpp
        ${implot_SOURCE_DIR}/implot_internal.h
        ${implot_SOURCE_DIR}/implot_items.cpp
        PUBLIC FILE_SET HEADERS
        BASE_DIRS ${implot_SOURCE_DIR}/
        FILES
        ${implot_SOURCE_DIR}/implot.h
)

imguios_target_ignore_all_warnings(implot)
target_compile_features(implot PRIVATE cxx_std_17)
# SYSTEM marks vendored headers as external so their diagnostics are suppressed in consumer code.
set_property(TARGET implot PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${implot_SOURCE_DIR}")
target_link_libraries(implot PUBLIC imgui)
set_target_properties(implot PROPERTIES FOLDER "third_party")

#############################################################
# imguios_fonts
#############################################################

# Overwrite IMGUIOS_FONTS_SOURCE_DIR if you want to build imguios_fonts source from a different location.
set(IMGUIOS_FONTS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/imguios_fonts" CACHE STRING "Location of imguios_fonts source")

FetchContent_Declare(
        imguios_fonts
        SOURCE_DIR ${IMGUIOS_FONTS_SOURCE_DIR}
)
FetchContent_MakeAvailable(imguios_fonts)
add_library(imguios_fonts)
target_sources(imguios_fonts
        PRIVATE
        ${imguios_fonts_SOURCE_DIR}/src/font_awesome5.cpp
        ${imguios_fonts_SOURCE_DIR}/src/roboto_bold.cpp
        ${imguios_fonts_SOURCE_DIR}/src/roboto_mono_italic.cpp
        ${imguios_fonts_SOURCE_DIR}/src/roboto_mono_regular.cpp
        ${imguios_fonts_SOURCE_DIR}/src/roboto_mono_bold.cpp
        ${imguios_fonts_SOURCE_DIR}/src/roboto_regular.cpp
        ${imguios_fonts_SOURCE_DIR}/src/roboto_italic.cpp
        PUBLIC FILE_SET HEADERS
        BASE_DIRS ${imguios_fonts_SOURCE_DIR}/include/
        FILES
        ${imguios_fonts_SOURCE_DIR}/include/imguios/fonts/fonts.h
        ${imguios_fonts_SOURCE_DIR}/include/imguios/fonts/icons_font_awesome5.h
)

imguios_target_ignore_all_warnings(imguios_fonts)
target_compile_features(imguios_fonts PRIVATE cxx_std_17)
set_target_properties(imguios_fonts PROPERTIES FOLDER "third_party")
