/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <mochi_core/utils/console.h>

#if MOCHI_PLATFORM_WINDOWS
#include <windows.h>
#include <cstdio>
#endif

namespace mochi {

#if MOCHI_PLATFORM_WINDOWS

void AttachParentConsole() {
  if (!AttachConsole(ATTACH_PARENT_PROCESS)) {
    // Either this process already has a console or the parent has none; nothing to reattach.
    return;
  }
  // AttachConsole grants a console but leaves the CRT's stdio handles unbound. Rebind them so
  // printf/std::cout reach the terminal.
  FILE* stream = nullptr;
  freopen_s(&stream, "CONOUT$", "w", stdout);
  freopen_s(&stream, "CONOUT$", "w", stderr);
  freopen_s(&stream, "CONIN$", "r", stdin);
}

#else

void AttachParentConsole() {}

#endif

} // namespace mochi
