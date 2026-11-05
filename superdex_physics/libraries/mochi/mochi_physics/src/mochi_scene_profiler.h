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

#include "mochi_ecs.h"

#include <mochi_core/utils/no_copy.h>
#include <mochi_core/utils/time.h>

namespace mochi {

// Present while the debugger's scene profiler is enabled.
struct TagProfilerEnabled {};

// Coarse wall times of an island's latest step [s]. Sleeping islands retain stale values.
// Only the island's own pipeline writes these fields, ordered by its task semaphores.
struct CIslandStats {
  double contactSec = 0.0; // Collision detection and contact Jacobians
  double assemblySec = 0.0; // Assembly excluding the contact interval within it
  double solveSec = 0.0; // Linear operator setup and linear solves
  double queriesSec = 0.0; // Post-solve query update, including async completion

  bool operator==(CIslandStats const&) const = default;
};

namespace profiler {

void InitializeOnce(entt::registry& reg);
void Enable(entt::registry& reg);
void Disable(entt::registry& reg);
[[nodiscard]] bool IsEnabled(entt::registry const& reg);

// Adds a coarse operation's time to the island component. Reads no clock when it is absent.
class ScopedIslandTimer : NoCopy {
 public:
  ScopedIslandTimer(entt::registry& reg, entt::entity island, double CIslandStats::* field)
      : _stats(reg.try_get<CIslandStats>(island)),
        _field(field),
        _start(_stats ? Timer::Now() : TimePoint{}) {}

  ~ScopedIslandTimer() {
    if (_stats) {
      _stats->*_field += ToSeconds(Timer::Now() - _start);
    }
  }

 private:
  CIslandStats* _stats;
  double CIslandStats::* _field;
  TimePoint _start;
};

} // namespace profiler
} // namespace mochi
