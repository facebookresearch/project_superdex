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

/********************************************************************************
 IMPORTANT: PLEASE KEEP HEADER INCLUDES TO A MINIMUM.
    If you must include a mochi_core header, then please make sure that it only
    declares the data types (not containing other implementation details).
*********************************************************************************/

#include <cstdint>

namespace mochi {

inline constexpr uint64_t kInvalidHandle = 0;

struct Handle {
  using ValueType = uint64_t;
  Handle() = default;
  explicit Handle(ValueType raw);
  ValueType value = kInvalidHandle;

  bool IsValid() const;
  bool operator==(Handle const& rhs) const;
  bool operator!=(Handle const& rhs) const;
  bool operator<(Handle const& rhs) const; // for ordered containers
  uint64_t GetHash() const; // for unordered containers
};

// Strongly typed handles

struct SceneHandle : public Handle {
  using Handle::Handle;
};

struct ActorHandle : public Handle {
  using Handle::Handle;
};

// ShapeHandle is reference-counted in C++/Python via a private cleanup token that has
// no DSL counterpart, so its definition is hand-written in mochi_handle_inl.h (the
// DSL marks it [no_cpp_definition]). Forward-declared here to keep this header aligned
// 1:1 with mochi_physics_handle.mochi_gen.
struct ShapeHandle;

struct CallbackHandle : public Handle {
  using Handle::Handle;
};

struct ConstraintHandle : public Handle {
  using Handle::Handle;
};

struct StateHandle : public Handle {
  using Handle::Handle;
};

struct QueryHandle : public Handle {
  using Handle::Handle;
};

// PrefabHandle is a std::shared_ptr subclass (to a loaded prefab) that has no DSL counterpart,
// so its definition is hand-written in mochi_handle_inl.h (the DSL marks it [no_cpp_definition]).
// Forward-declared here to keep this header aligned with mochi_physics_handle.mochi_gen.
struct PrefabHandle;

} // namespace mochi

#include <mochi_physics/cpp_api/mochi_handle_inl.h>
