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

#include <cstddef>
#include <functional>
#include <memory>

#include <mochi_physics/cpp_api/mochi_handle.h>

namespace mochi {

inline Handle::Handle(ValueType raw) : value(raw) {}

inline bool Handle::IsValid() const {
  return value != kInvalidHandle;
}
inline bool Handle::operator==(Handle const& rhs) const {
  return value == rhs.value;
}
inline bool Handle::operator!=(Handle const& rhs) const {
  return value != rhs.value;
}
inline bool Handle::operator<(Handle const& rhs) const {
  return value < rhs.value;
}
inline uint64_t Handle::GetHash() const {
  return std::hash<ValueType>{}(value);
}

// Definition kept out of the DSL-mirrored mochi_handle.h (ShapeHandle is marked
// [no_cpp_definition] in mochi_physics_handle.mochi_gen): it carries a private,
// C++-only auto-cleanup token that the DSL cannot express.
struct ShapeHandle : public Handle {
  using Handle::Handle;

 private:
  friend class ContextImpl;
  struct AutoCleanup {
    virtual ~AutoCleanup() = default;
  };
  std::shared_ptr<AutoCleanup> _cleanup = {};
};

namespace prefab {
struct ScenePrefab;
} // namespace prefab

// Definition kept out of the DSL-mirrored mochi_prefab.h (PrefabHandle is declared in
// mochi_physics_handle.mochi_gen and marked [no_cpp_definition]): it wraps a std::shared_ptr, which
// the DSL cannot express.
//
// Runtime handle to a loaded nested prefab (the resolved target of a @ref prefab::PrefabReference).
// It is a strongly-typed std::shared_ptr<@ref prefab::ScenePrefab> and behaves exactly like one
// (copy shares ownership, move transfers, deref/operator-> access the prefab). The pointed-to
// prefab is runtime state: it is populated during loading (e.g. @ref prefab::LoadNestedPrefabs) and
// is never serialized.
struct PrefabHandle : public std::shared_ptr<prefab::ScenePrefab> {
  // Bring in std::shared_ptr's assignment operators (otherwise hidden by the implicit copy/move
  // assignment) so a PrefabHandle can be assigned from a shared_ptr or nullptr. Constructors are
  // intentionally NOT inherited: doing so would make `nullptr` convertible to PrefabHandle and make
  // `handle = nullptr` ambiguous. Default/copy/move construction (all we need) stay implicit.
  using std::shared_ptr<prefab::ScenePrefab>::operator=;
};

} // namespace mochi

// Support for mochi::Handle in std::unordered_set and std::unordered_map
/// @cond
namespace std {
template <>
struct hash<mochi::Handle> {
  std::size_t operator()(mochi::Handle h) const {
    return h.GetHash();
  }
};
template <>
struct hash<mochi::SceneHandle> : public hash<mochi::Handle> {};
template <>
struct hash<mochi::ActorHandle> : public hash<mochi::Handle> {};
template <>
struct hash<mochi::ShapeHandle> : public hash<mochi::Handle> {};
template <>
struct hash<mochi::CallbackHandle> : public hash<mochi::Handle> {};
template <>
struct hash<mochi::ConstraintHandle> : public hash<mochi::Handle> {};
template <>
struct hash<mochi::StateHandle> : public hash<mochi::Handle> {};
template <>
struct hash<mochi::QueryHandle> : public hash<mochi::Handle> {};
} // namespace std
/// @endcond
