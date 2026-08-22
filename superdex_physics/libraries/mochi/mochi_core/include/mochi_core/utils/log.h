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

#pragma once

#include <mochi_core/mochi_config.h>

#if MOCHI_PLATFORM_WINDOWS && !defined(__UNREAL__)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif // MOCHI_PLATFORM_WINDOWS && !defined(__UNREAL__)

#include <array>
#include <cstdarg>
#include <functional>
#include <mutex>
#include <string>

namespace mochi {

// Log messages are grouped into channels that can be enabled/disabled
enum class LogChannel { Verbose, Info, Warning, Error, Count };

// Macros: Output to the log if the specified channel is enabled.
// clang-format off
#define MOCHI_LOG(...) MOCHI_LOG_IMPL(::mochi::LogChannel::Info, __FILE__, __LINE__, __VA_ARGS__)
#define MOCHI_LOG_VERBOSE(...) MOCHI_LOG_IMPL(::mochi::LogChannel::Verbose, __FILE__, __LINE__, __VA_ARGS__)
#define MOCHI_LOG_WARNING(...) MOCHI_LOG_IMPL(::mochi::LogChannel::Warning, __FILE__, __LINE__, __VA_ARGS__)
#define MOCHI_LOG_ERROR(...) MOCHI_LOG_IMPL(::mochi::LogChannel::Error, __FILE__, __LINE__, __VA_ARGS__)
#define MOCHI_LOG_TO(channel, ...) MOCHI_LOG_IMPL(channel, __FILE__, __LINE__, __VA_ARGS__)
// clang-format on

// Same but log only the first time.
#define MOCHI_LOG_ONCE_HELPER(channel, ...)                     \
  do {                                                          \
    [[maybe_unused]] static bool logged_##__LINE__ = [&]() {    \
      MOCHI_LOG_IMPL(channel, __FILE__, __LINE__, __VA_ARGS__); \
      return true;                                              \
    }();                                                        \
  } while (0)
#define MOCHI_LOG_ONCE(...) MOCHI_LOG_ONCE_HELPER(::mochi::LogChannel::Info, __VA_ARGS__)
#define MOCHI_LOG_VERBOSE_ONCE(...) MOCHI_LOG_ONCE_HELPER(::mochi::LogChannel::Verbose, __VA_ARGS__)
#define MOCHI_LOG_WARNING_ONCE(...) MOCHI_LOG_ONCE_HELPER(::mochi::LogChannel::Warning, __VA_ARGS__)
#define MOCHI_LOG_ERROR_ONCE(...) MOCHI_LOG_ONCE_HELPER(::mochi::LogChannel::Error, __VA_ARGS__)

// Return a std::string initialized with printf-style formatting
[[nodiscard]] std::string Format(char const* format, ...);

// Signature of a logging callback function
using LogFn = std::function<void(LogChannel, char const* message, char const* file, int line)>;

// Get the current logging function.
[[nodiscard]] LogFn GetLogCallback();

// Set the logging function to redirect output. Pass LogFn{} to restore default logging.
void SetLogCallback(LogFn fn);

// Enable or disable log messages on the specified channel
void EnableLogChannel(LogChannel channel, bool enable);

// Return true if messages on the specified channel should be shown
[[nodiscard]] bool IsLogChannelEnabled(LogChannel channel);

// Output to the current logging function.
void Log(LogChannel channel, std::string msg, char const* file, int line, bool newline);

// Ensure that recent logging has been flushed to the output pipe.
void FlushLog();

//------------------------------------------------------------------------------------------------
// Implementation Details
//------------------------------------------------------------------------------------------------

#define MOCHI_LOG_IMPL(channel, file, line, ...)                           \
  if (::mochi::IsLogChannelEnabled(channel)) {                             \
    ::mochi::Log(channel, ::mochi::Format(__VA_ARGS__), file, line, true); \
  } else {                                                                 \
  }

// Supports Format(__VA_ARGS__) when __VA_ARGS__ is empty
[[nodiscard]] inline std::string Format() {
  return {};
}

[[nodiscard]] inline std::string Format(char const* format, ...) {
  va_list args1, args2;
  va_start(args1, format);
  va_copy(args2, args1);
  int len = vsnprintf(nullptr, 0, format, args1);
  std::string str;
  str.resize((size_t)len); // +1 makes it cheap to append "\n" for logging
  if (len)
    MOCHI_LIKELY {
      vsnprintf(&str[0], len + 1, format, args2); // This use of &str[0] guaranteed by C++11
    }
  va_end(args2);
  va_end(args1);
  return str;
}

// These implementation functions are inlined so that logging macros can be used via header-only
// dependency on MochiCore. They do not require linking with MochiCore.
namespace log_impl {

#if MOCHI_ARCH_CPU
inline auto& GetLogMutexRef() {
  static std::recursive_mutex s_logMutex;
  return s_logMutex;
}
inline auto& GetLogFnRef() {
  static LogFn s_logFn;
  return s_logFn;
}
inline auto& GetLogChannelsDisabledArrayRef() {
  static std::array<bool, (size_t)LogChannel::Count> s_logChannelsDisabled = []() {
    // All channels enabled by default
    std::array<bool, (size_t)LogChannel::Count> disabledChannels = {};
    // Except for these, which are disabled:
    disabledChannels[(size_t)LogChannel::Verbose] = true;
    return disabledChannels;
  }();
  return s_logChannelsDisabled;
}
#endif // MOCHI_ARCH_CPU

inline void DefaultLogFn(LogChannel channel, std::string msg, char const* file, int line) {
  char const* channelTag = (channel == LogChannel::Warning) ? " [WARNING]"
      : (channel == LogChannel::Error)                      ? " [ERROR]"
                                                            : "";
  auto fmtMsg = Format("%s(%d): [Mochi]%s %s", file, line, channelTag, msg.c_str());
  printf("%s", fmtMsg.c_str());

#if MOCHI_PLATFORM_WINDOWS && !defined(__UNREAL__)
  // Logging to stdout is not automatically displayed in the Visual Studio debugger,
  // so we have to do that explicitly.
  ::OutputDebugStringA(fmtMsg.c_str());
#endif // MOCHI_PLATFORM_WINDOWS
}

} // namespace log_impl

inline void FlushLog() {
  // TODO: Allow users who register a custom logging callback to also implement FlushLog.
#if MOCHI_ARCH_CPU
  fflush(stdout);
  fflush(stderr);
#endif
}

inline LogFn GetLogCallback() {
#if MOCHI_ARCH_CPU
  std::lock_guard<std::recursive_mutex> lock(log_impl::GetLogMutexRef());
  auto const& logFn = log_impl::GetLogFnRef();
  return logFn ? logFn : LogFn{&log_impl::DefaultLogFn};
#else
  return LogFn{&log_impl::DefaultLogFn};
#endif
}

inline void SetLogCallback([[maybe_unused]] LogFn fn) {
#if MOCHI_ARCH_CPU
  std::lock_guard<std::recursive_mutex> lock(log_impl::GetLogMutexRef());
  auto& logFn = log_impl::GetLogFnRef();
  logFn = fn ? fn : LogFn{&log_impl::DefaultLogFn};
#endif
}

inline void EnableLogChannel(LogChannel channel, [[maybe_unused]] bool enable) {
  if ((size_t)channel >= (size_t)LogChannel::Count)
    MOCHI_UNLIKELY {
      MOCHI_LOG_WARNING("Requested LogChannel (%d) is invalid and will be ignored.", (int)channel);
      return;
    }
#if MOCHI_ARCH_CPU
  std::lock_guard<std::recursive_mutex> lock(log_impl::GetLogMutexRef());
  log_impl::GetLogChannelsDisabledArrayRef()[(size_t)channel] = !enable;
#endif
}

inline bool IsLogChannelEnabled(LogChannel channel) {
  if ((size_t)channel >= (size_t)LogChannel::Count)
    MOCHI_UNLIKELY {
      return false;
    }
#if MOCHI_ARCH_CPU
  std::lock_guard<std::recursive_mutex> lock(log_impl::GetLogMutexRef());
  auto const& isChannelDisabled = log_impl::GetLogChannelsDisabledArrayRef();
  return !isChannelDisabled[(size_t)channel];
#else
  return true;
#endif
}

inline void Log(LogChannel channel, std::string msg, char const* file, int line, bool newline) {
  if (newline) {
    msg += "\n";
  }
  GetLogCallback()(channel, msg.c_str(), file, line);
}

} // namespace mochi
