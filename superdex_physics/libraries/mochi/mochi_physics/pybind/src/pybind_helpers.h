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

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <typeindex>
#include <unordered_map>

#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/mochi_physics_experimental.h>
#include <mochi_physics/utils/mochi_prefab.h>

namespace mochi {

// Name of the module for single- or double-precision
#if MOCHI_USE_DOUBLE_PRECISION
#define MOCHI_PHYSICS_MODULE_NAME mochi_physics_double
#define MOCHI_PHYSICS_MODULE_NAME_STR "mochi_physics_double"
#else
#define MOCHI_PHYSICS_MODULE_NAME mochi_physics
#define MOCHI_PHYSICS_MODULE_NAME_STR "mochi_physics"
#endif

inline auto const kQuaternionIdentity = Quaternion::Identity();
inline auto const kTransformRTIdentity = TransformRT::Identity();
inline auto const kGridSdfParamsDefault = GridSdfParams{};

// Use reflection to format the Python __str__ for any supported type.
template <typename T>
inline std::string ToPyString(T const& obj) {
  // Use "pretty" multi-line formatting
  auto json = SReflect::ToJsonString(obj, true /*pretty*/);
  MOCHI_ASSERT(!json.empty());

  // Trim trailing newline
  size_t end = json.find_last_not_of("\n\r");
  if (end != std::string::npos) {
    json.resize(end + 1);
  }

  return json;
}

// Use reflection to format the Python __repl__ string for any supported type.
template <typename T>
inline std::string ToPyReplString(T const& obj) {
  return Format("%s(%s)", SReflect::GetTypeInfo<T>()._name, ToPyString(obj).c_str());
}

// Types like Optional<T> don't need to be registered. Pybind handles them automatically. This
// function exists because the code generator emits DefX for every template class X, including
// "Optional".
template <class T>
void DefOptional(pybind11::module& /*m*/, char const* /*name*/) {}

// Type-erased registry of already-declared pybind11 classes, keyed by C++ type.
//
// Enables two-phase registration of the generated bindings: a declaration phase
// registers every py::class_ up front (via StoreClass) so that all types exist before
// any constructor default argument is converted to a Python object; a later definition
// phase retrieves the same handle (via GetClass) to attach members. Handles are stored
// type-erased as pybind11::object and recovered as the concrete
// pybind11::class_<T, Opts...> on retrieval.
//
// This is a local object created during module initialization and passed by reference
// to the generated Declare*/Define* functions; it is not a global and holds no state
// past import (its borrowed handles are released when it goes out of scope, while the
// classes themselves stay owned by the module).
class PybindRegistry {
 public:
  // Registers a freshly-declared class handle, keyed by its C++ type T.
  template <typename T, typename... Opts>
  void StoreClass(pybind11::class_<T, Opts...> cls) {
    auto [it, inserted] = _handles.emplace(std::type_index(typeid(T)), std::move(cls));
    (void)it;
    MOCHI_ASSERT(inserted);
  }

  // Retrieves the handle previously stored for T, reinterpreted as its concrete
  // pybind11::class_<T, Opts...>. Opts must match the declaration exactly.
  template <typename T, typename... Opts>
  pybind11::class_<T, Opts...> GetClass() const {
    auto it = _handles.find(std::type_index(typeid(T)));
    MOCHI_ASSERT(it != _handles.end());
    return pybind11::reinterpret_borrow<pybind11::class_<T, Opts...>>(it->second);
  }

 private:
  std::unordered_map<std::type_index, pybind11::object> _handles;
};

// Entry point for the generated bindings. Declares all classes and functions in the extension.
void DefineAll(pybind11::module& m);

} // namespace mochi
