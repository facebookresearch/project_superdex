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

#include <chrono>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include <mochi_core/utils/debug.h>

namespace mochi {

/**
  Shorthand to make std::chrono easier to use
*/
using TimePoint = std::chrono::high_resolution_clock::time_point;
using TimeSpan = std::chrono::high_resolution_clock::duration;

/**
  TimeSpan Conversions:
*/
template <class Clock = std::chrono::high_resolution_clock>
[[nodiscard]] inline double ToNanoseconds(typename Clock::duration ts) {
  return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(ts).count());
}

template <class Clock = std::chrono::high_resolution_clock>
[[nodiscard]] inline double ToMicroseconds(typename Clock::duration ts) {
  return ToNanoseconds<Clock>(ts) * 1.0e-3;
}

template <class Clock = std::chrono::high_resolution_clock>
[[nodiscard]] inline double ToMilliseconds(typename Clock::duration ts) {
  return ToNanoseconds<Clock>(ts) * 1.0e-6;
}

template <class Clock = std::chrono::high_resolution_clock>
[[nodiscard]] inline double ToSeconds(typename Clock::duration ts) {
  return ToNanoseconds<Clock>(ts) * 1.0e-9;
}

template <class Clock = std::chrono::high_resolution_clock>
[[nodiscard]] inline typename Clock::duration TimeSpanFromSeconds(double seconds) {
  return std::chrono::duration_cast<typename Clock::duration>(
      std::chrono::duration<double>(seconds));
}

/**
  High resolution timer used to measure performance.
  Measurement starts when the Timer is constructed or Reset.
*/
class Timer {
 public:
  [[nodiscard]] TimeSpan GetElapsed() const {
    return {Now() - _start};
  }
  void Reset() {
    _start = Now();
  }
  [[nodiscard]] static TimePoint Now() {
    return std::chrono::high_resolution_clock::now();
  }
  TimePoint _start = Now();
};

enum TimeUnit { Seconds, Milliseconds, Microseconds, Nanoseconds };

[[nodiscard]] inline std::string_view GetUnitString(TimeUnit unit) {
  switch (unit) {
    case TimeUnit::Seconds:
      return "s";
    case TimeUnit::Milliseconds:
      return "ms";
    case TimeUnit::Microseconds:
      return "us";
    case TimeUnit::Nanoseconds:
      return "ns";
    default:
      return "?";
  }
}

template <class Clock = std::chrono::high_resolution_clock>
[[nodiscard]] inline double ToTimeUnit(typename Clock::duration span, TimeUnit unit) {
  switch (unit) {
    case TimeUnit::Seconds:
      return ToSeconds(span);
    case TimeUnit::Milliseconds:
      return ToMilliseconds(span);
    case TimeUnit::Microseconds:
      return ToMicroseconds(span);
    case TimeUnit::Nanoseconds:
      return ToNanoseconds(span);
    default:
      throw std::runtime_error("Unknown time unit!");
  }
}

/*
    A simple profiling utility. To performance measure a chunk of code,
    wrap the code in the following:

    ```
    static Profiler profiler("NAME OF BLOCK");
    profiler([&]() {
        // Code here
    });
    ```

    Alternatively:

    ```
    static Profiler profiler("NAME OF BLOCK");
    profiler.Begin();
        // Code here
    profiler.End();
    ```

    If the code is within a loop, you can use:

    ```
    static Profiler profiler("NAME OF BLOCK");

    for (int i = 0; i < n; ++i) {
        // Code here

        profiler.Begin();
            // Block of code to measure
        profiler.Pause();

        // Code here
    }

    profiler.End();
    ```
*/
struct Profiler {
  size_t numInstances = 0;
  double totalTime = 0.0;

  std::string name;
  TimeUnit timeUnit;
  size_t logEvery;

  Timer timer;
  bool started = false;

  inline Profiler(
      std::string_view name,
      TimeUnit timeUnit = TimeUnit::Milliseconds,
      size_t logEvery = 1000)
      : name(name), timeUnit(timeUnit), logEvery(logEvery) {}

  void Begin() {
    MOCHI_ASSERT(!started, "Profiler has already begun!");
    timer.Reset();
    started = true;
  }

  void Pause() {
    if (started) {
      totalTime += ToTimeUnit(timer.GetElapsed(), timeUnit);
    }
    started = false;
  }

  void End() {
    Pause();
    numInstances++;

    if (numInstances % logEvery == 0) {
      auto unitStr = GetUnitString(timeUnit);
      double meanTime = totalTime / numInstances;
      MOCHI_LOG("%s: %f %s", name.c_str(), meanTime, unitStr.data());
    }
  }

  template <typename T>
  inline auto operator()(T t) {
    using return_t = decltype(t());
    Begin();
    if constexpr (std::is_void_v<return_t>) {
      t();
      End();
    } else {
      auto result = t();
      End();
      return result;
    }
  }
};

} // namespace mochi
