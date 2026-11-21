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

#include "mochi_scene_table.h"

#include <mochi_core/utils/basic_utils.h>

namespace mochi {

static_assert(
    std::atomic<uint32_t>::is_always_lock_free && std::atomic<Scene*>::is_always_lock_free,
    "GetScene is documented never to block.");

SceneTable::SceneTable(uint32_t capacity, uint32_t firstHandle)
    : _slots(std::make_unique<Slot[]>(capacity)), _mask(capacity - 1), _next(firstHandle) {
  MOCHI_ASSERT_VERBOSE(
      IsPowerOfTwo(capacity) && capacity <= kMaxCapacity && firstHandle != 0,
      "SceneTable needs a power-of-two capacity, at most %u, and a nonzero first handle.",
      kMaxCapacity);
}

SceneHandle SceneTable::Reserve() {
  std::scoped_lock lock(_writeMutex);
  // _next wraps to 0 after the last 32-bit value, which ends all reservations.
  for (uint32_t i = 0; i <= _mask && _next != 0; ++i) {
    uint32_t const value = _next++;
    Slot& entry = _slots[value & _mask];
    if (entry.handle.load() == 0) {
      entry.handle.store(value);
      return SceneHandle{value};
    }
  }
  return SceneHandle{};
}

void SceneTable::Publish(SceneHandle handle, Scene* scene) {
  std::scoped_lock lock(_writeMutex);
  Slot& entry = _slots[handle.value & _mask];
  bool const reserved =
      handle.IsValid() && entry.handle.load() == handle.value && !entry.scene.load();
  MOCHI_ASSERT(reserved);
  if (reserved) {
    entry.scene.store(scene);
  }
}

void SceneTable::Release(SceneHandle handle) {
  std::scoped_lock lock(_writeMutex);
  Slot& entry = _slots[handle.value & _mask];
  bool const owned = handle.IsValid() && entry.handle.load() == handle.value;
  MOCHI_ASSERT(owned);
  if (owned) {
    entry.handle.store(0);
    entry.scene.store(nullptr);
  }
}

} // namespace mochi
