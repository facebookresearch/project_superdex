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

include_guard(GLOBAL)

# Places zenoh-pico and nanopb's C runtime into the plugin's modules, where UnrealBuildTool compiles
# them as part of the TeleopNodeZenoh module. Unchanged files are left alone, so they don't
# recompile, and anything else in a destination is deleted, so files from an earlier layout don't
# linger (UnrealBuildTool compiles every source it finds). Links OSIRIS placed there (the internal
# build links the destinations or their top-level entries to its cache) are removed first, so files
# are never written through them.

#-----------------------------------------------------------------------------
# zenoh-pico -> Source/TeleopNodeZenoh/Private/ZenohPico
#
#   include/          public headers, with the configured zenoh-pico/config.h
#   src/common/       platform-independent sources
#   src/Windows/      Windows sources
#   src/Android/      POSIX sources, used on Android
#   src/atomic.c.inl  collections/atomic.c, which the module compiles as C++ (MSVC only has C11
#                     <stdatomic.h> behind a flag a module can't set)
#
# Sources are renamed after their path under src/ ("link/config/tcp.c" -> "link_config_tcp.c"):
# UnrealBuildTool's non-unity builds name object files after the source file alone and reject a
# module with two sources of the same name, and zenoh-pico has many (three encoding.c, four
# transport.c, ...). The sources only include "zenoh-pico/..." and system headers, so their
# location doesn't matter. Windows/ and Android/ are folder names UnrealBuildTool skips when building
# for another platform.
#-----------------------------------------------------------------------------

function(teleop_node_zenoh_stage_zenoh_pico)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "SOURCE_DIR;CONFIG_H;DEST" "")
    set(src "${ARG_SOURCE_DIR}/src")

    file(GLOB_RECURSE common_srcs
        "${src}/api/*.c"
        "${src}/collections/*.c"
        "${src}/net/*.c"
        "${src}/protocol/*.c"
        "${src}/runtime/*.c"
        "${src}/session/*.c"
        "${src}/transport/*.c"
        "${src}/utils/*.c"
        "${src}/system/common/*.c"
    )
    file(GLOB common_link_srcs
        "${src}/link/config/*.c"
        "${src}/link/multicast/*.c"
        "${src}/link/unicast/*.c"
        "${src}/link/transport/common/*.c"
    )
    list(APPEND common_srcs ${common_link_srcs}
        "${src}/link/endpoint.c"
        "${src}/link/link.c"
        "${src}/link/transport/tcp/address.c"
        "${src}/link/transport/udp/address.c"
        "${src}/link/transport/upper/serial_protocol.c"
        "${src}/link/transport/upper/tls_stream.c"
    )
    set(atomic_src "${src}/collections/atomic.c")
    list(REMOVE_ITEM common_srcs "${atomic_src}")
    set(posix_srcs
        "${src}/link/transport/serial/tty_posix.c"
        "${src}/link/transport/tcp/tcp_posix.c"
        "${src}/link/transport/udp/raweth_unix.c"
        "${src}/link/transport/udp/udp_multicast_posix.c"
        "${src}/link/transport/udp/udp_posix.c"
        "${src}/system/unix/network.c"
        "${src}/system/unix/system.c"
    )
    set(windows_srcs
        "${src}/link/transport/tcp/tcp_windows.c"
        "${src}/link/transport/udp/udp_multicast_windows.c"
        "${src}/link/transport/udp/udp_windows.c"
        "${src}/system/windows/network.c"
        "${src}/system/windows/system.c"
    )

    _teleop_node_zenoh_unlink("${ARG_DEST}")
    set(placed "")

    file(GLOB_RECURSE headers RELATIVE "${ARG_SOURCE_DIR}/include" "${ARG_SOURCE_DIR}/include/*.h")
    foreach (header IN LISTS headers)
        _teleop_node_zenoh_copy("${ARG_SOURCE_DIR}/include/${header}" "${ARG_DEST}/include/${header}")
    endforeach ()
    _teleop_node_zenoh_copy("${ARG_CONFIG_H}" "${ARG_DEST}/include/zenoh-pico/config.h")

    foreach (group IN ITEMS common Android Windows)
        if (group STREQUAL "common")
            set(group_srcs ${common_srcs})
        elseif (group STREQUAL "Android")
            set(group_srcs ${posix_srcs})
        else ()
            set(group_srcs ${windows_srcs})
        endif ()
        foreach (source IN LISTS group_srcs)
            file(RELATIVE_PATH relative "${src}" "${source}")
            string(REPLACE "/" "_" flat "${relative}")
            _teleop_node_zenoh_copy("${source}" "${ARG_DEST}/src/${group}/${flat}")
        endforeach ()
    endforeach ()
    _teleop_node_zenoh_copy("${atomic_src}" "${ARG_DEST}/src/atomic.c.inl")

    _teleop_node_zenoh_copy("${ARG_SOURCE_DIR}/LICENSE" "${ARG_DEST}/LICENSE")
    _teleop_node_zenoh_copy("${ARG_SOURCE_DIR}/NOTICE.md" "${ARG_DEST}/NOTICE.md")
    _teleop_node_zenoh_prune("${ARG_DEST}" "${placed}")
    message(STATUS "[TeleopNodeZenoh] Placed zenoh-pico in ${ARG_DEST}")
endfunction()

#-----------------------------------------------------------------------------
# nanopb's C runtime -> Source/TeleopNodeZenoh/Private/Nanopb
#-----------------------------------------------------------------------------

function(teleop_node_zenoh_stage_nanopb)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "SOURCE_DIR;DEST" "")
    _teleop_node_zenoh_unlink("${ARG_DEST}")
    set(placed "")
    foreach (name IN ITEMS
            LICENSE.txt pb.h pb_common.c pb_common.h pb_decode.c pb_decode.h pb_encode.c pb_encode.h)
        _teleop_node_zenoh_copy("${ARG_SOURCE_DIR}/${name}" "${ARG_DEST}/${name}")
    endforeach ()
    _teleop_node_zenoh_prune("${ARG_DEST}" "${placed}")
    message(STATUS "[TeleopNodeZenoh] Placed nanopb in ${ARG_DEST}")
endfunction()

# Copies source to dest unless dest already has the same content, and records dest in the caller's
# `placed` list. A link at dest (e.g. left by another tool) is replaced, never written through.
macro(_teleop_node_zenoh_copy source dest)
    if (NOT EXISTS "${source}")
        message(FATAL_ERROR "[TeleopNodeZenoh] Missing ${source}")
    endif ()
    if (IS_SYMLINK "${dest}")
        file(REMOVE "${dest}")
    endif ()
    get_filename_component(_teleop_node_zenoh_dest_dir "${dest}" DIRECTORY)
    file(MAKE_DIRECTORY "${_teleop_node_zenoh_dest_dir}")
    file(COPY_FILE "${source}" "${dest}" ONLY_IF_DIFFERENT)
    list(APPEND placed "${dest}")
endmacro()

# Removes dest if it is a link, and any link directly inside it.
function(_teleop_node_zenoh_unlink dest)
    if (IS_SYMLINK "${dest}")
        file(REMOVE "${dest}")
        return ()
    endif ()
    file(GLOB entries LIST_DIRECTORIES true "${dest}/*")
    foreach (entry IN LISTS entries)
        if (IS_SYMLINK "${entry}")
            file(REMOVE "${entry}")
        endif ()
    endforeach ()
endfunction()

# Deletes everything under dest that is not in the placed list.
function(_teleop_node_zenoh_prune dest placed)
    file(GLOB_RECURSE existing LIST_DIRECTORIES false "${dest}/*")
    foreach (path IN LISTS existing)
        if (NOT path IN_LIST placed)
            file(REMOVE "${path}")
        endif ()
    endforeach ()
endfunction()
