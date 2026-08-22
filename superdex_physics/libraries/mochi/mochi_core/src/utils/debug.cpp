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

#include <mochi_core/utils/debug.h>

#if MOCHI_PLATFORM_LINUX
#include <fstream>
#include <iostream>
#include <string>
#elif MOCHI_PLATFORM_MACOS
#include <sys/sysctl.h>
#include <sys/types.h>
#include <unistd.h>
#elif MOCHI_PLATFORM_WINDOWS
#include <Windows.h>
#endif

bool mochi::IsDebuggerAttached() {
#if MOCHI_PLATFORM_LINUX
  try {
    std::ifstream statusFile("/proc/self/status");
    std::string line;
    while (std::getline(statusFile, line)) {
      if (line.find("TracerPid:") == 0) {
        int tracerPid = std::stoi(line.substr(10));
        return tracerPid != 0;
      }
    }
  } catch (...) {
  }
  // Debugger presence can't be determined.
  return false;
#elif MOCHI_PLATFORM_MACOS
  struct kinfo_proc info{};
  size_t size = sizeof(info);
  int mib[] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()};
  if (sysctl(mib, std::size(mib), &info, &size, nullptr, 0) != -1) {
    // Debugger is attached if the P_TRACED flag is set.
    return (info.kp_proc.p_flag & P_TRACED) != 0;
  } else {
    // Debugger presence can't be determined.
    return false;
  }
#elif MOCHI_PLATFORM_WINDOWS
  // Win32 API
  return (::IsDebuggerPresent() == TRUE);
#else
  // Got another platform? Add support for it here.
  return false;
#endif
}
