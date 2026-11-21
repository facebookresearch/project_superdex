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

#include <mochi_core/mochi_platform.h>
#include <mochi_physics/cpp_api/mochi_handle.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>

namespace mochi {

class Scene;

// Maps scene handles to scenes. Find takes no lock and writes nothing, so any number of threads can
// resolve handles at once without contending. Reserve, Publish and Release lock the table's own
// mutex, so any thread may call them. A reserved handle is published at most once and released
// once.
//
// A handle's slot is its value modulo the capacity. Reserve tries values in increasing order and
// skips those whose slot is taken, so handles increase with creation order and are never reused.
// Every value tried is used up, and Reserve fails once all 32-bit values are.
class SceneTable {
  MOCHI_DECLARE_NO_COPY_NO_MOVE(SceneTable);

 public:
  // At most 65,536, so live scenes differ in the low 16 bits of their handles. State handles keep
  // those bits to catch a state restored to the wrong scene.
  static constexpr uint32_t kMaxCapacity = 65'536;

  // capacity: a power of two, at most kMaxCapacity. firstHandle: the first value Reserve tries.
  explicit SceneTable(uint32_t capacity = kMaxCapacity, uint32_t firstHandle = 1);
  ~SceneTable() = default;

  // The scene published under the handle, or nullptr. Never another handle's scene.
  [[nodiscard]] Scene* Find(SceneHandle handle) const;

  // Claims a free slot and returns its handle, or an invalid handle if every slot is taken or every
  // value has been used.
  [[nodiscard]] SceneHandle Reserve();

  // Makes the scene of a reserved handle visible to Find.
  void Publish(SceneHandle handle, Scene* scene);

  // Frees the slot of a reserved or published handle. Find returns nullptr for it from now on.
  void Release(SceneHandle handle);

 private:
  struct Slot {
    std::atomic<uint32_t> handle = 0; // reserved handle, or 0
    std::atomic<Scene*> scene = nullptr; // nullptr until published
  };

  std::unique_ptr<Slot[]> _slots;
  uint32_t _mask = 0;
  // Serializes Reserve, Publish and Release: _next and every slot write. Find never takes it.
  // NOLINTNEXTLINE(facebook-thread-safety-analysis)
  std::mutex _writeMutex;
  uint32_t _next = 0; // the next value Reserve tries; 0 once all are used
};

inline Scene* SceneTable::Find(SceneHandle handle) const {
  Slot const& entry = _slots[handle.value & _mask];
  // Handle 0 marks a free slot, whose scene may not be cleared yet.
  if (!handle.IsValid() || entry.handle.load() != handle.value) {
    return nullptr;
  }
  // Read the slot's handle, its scene, then its handle again. If both handle reads match, the scene
  // is this handle's own and not one published later in the same slot.
  Scene* const scene = entry.scene.load();
  return entry.handle.load() == handle.value ? scene : nullptr;
}

} // namespace mochi
