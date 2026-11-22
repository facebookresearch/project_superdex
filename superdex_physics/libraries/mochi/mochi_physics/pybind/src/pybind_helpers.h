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

#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <memory>
#include <type_traits>
#include <typeindex>
#include <unordered_map>

#include <mochi_core/pybind/pybind_module_name.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/mochi_physics_experimental.h>
#include <mochi_physics/utils/mochi_prefab.h>

namespace mochi {

namespace nb = nanobind;

#define MOCHI_PHYSICS_MODULE_NAME MOCHI_PYBIND_MODULE_NAME(mochi_physics)
#define MOCHI_PHYSICS_MODULE_NAME_STR MOCHI_PP_STRINGIFY(MOCHI_PHYSICS_MODULE_NAME)

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

// Types like Optional<T> don't need to be registered. Nanobind handles them automatically. This
// function exists because the code generator emits DefX for every template class X, including
// "Optional".
template <class T>
void DefOptional(nb::module_& /*m*/, char const* /*name*/) {}

// Build an owning 1-D numpy array holding a copy of the given contiguous data.
template <typename T>
inline nb::object MakeOwningNumpy1D(T const* data, size_t size) {
  using Array = nb::ndarray<nb::numpy, T const, nb::ndim<1>>;
  if (size == 0) {
    T const emptyValue{};
    return Array(&emptyValue, {size}).cast(nb::rv_policy::copy);
  }
  return Array(data, {size}).cast(nb::rv_policy::copy);
}

template <typename>
inline constexpr bool kUnsupportedPythonBufferType = false;

template <typename T>
constexpr char const* PythonBufferFormat() {
  using U = std::remove_cv_t<T>;
  if constexpr (std::is_same_v<U, bool>) {
    return "?";
  } else if constexpr (std::is_same_v<U, char> || std::is_same_v<U, signed char>) {
    return "b";
  } else if constexpr (std::is_same_v<U, unsigned char>) {
    return "B";
  } else if constexpr (std::is_same_v<U, short>) {
    return "h";
  } else if constexpr (std::is_same_v<U, unsigned short>) {
    return "H";
  } else if constexpr (std::is_same_v<U, int>) {
    return "i";
  } else if constexpr (std::is_same_v<U, unsigned int>) {
    return "I";
  } else if constexpr (std::is_same_v<U, long>) {
    return "l";
  } else if constexpr (std::is_same_v<U, unsigned long>) {
    return "L";
  } else if constexpr (std::is_same_v<U, long long>) {
    return "q";
  } else if constexpr (std::is_same_v<U, unsigned long long>) {
    return "Q";
  } else if constexpr (std::is_same_v<U, float>) {
    return "f";
  } else if constexpr (std::is_same_v<U, double>) {
    return "d";
  } else {
    static_assert(kUnsupportedPythonBufferType<U>, "Unsupported Python buffer scalar type");
  }
}

struct PythonBufferLayout {
  DynamicArray<Py_ssize_t> shape;
  DynamicArray<Py_ssize_t> strides;
};

template <typename Fn>
bool TranslatePythonCasterExceptions(Fn&& fn, char const* fallbackMessage) noexcept {
  try {
    return fn();
  } catch (std::bad_alloc const&) {
    PyErr_NoMemory();
  } catch (nb::python_error& error) {
    error.restore();
  } catch (std::exception const& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError, fallbackMessage);
  }
  return false;
}

inline int FillPythonBuffer(
    PyObject* exporter,
    Py_buffer* view,
    int flags,
    void* data,
    Py_ssize_t itemSize,
    char const* format,
    bool readOnly,
    Py_ssize_t const* shape,
    size_t numDims) noexcept {
  try {
    Py_ssize_t length = itemSize;
    for (size_t i = numDims; i-- > 0;) {
      length *= shape[i];
    }

    std::unique_ptr<PythonBufferLayout> layout;
    if ((flags & PyBUF_ND) != 0) {
      layout = std::make_unique<PythonBufferLayout>();
      layout->shape.assign(shape, shape + numDims);
      layout->strides.resize(numDims);
      Py_ssize_t stride = itemSize;
      for (size_t i = numDims; i-- > 0;) {
        layout->strides[i] = stride;
        stride *= layout->shape[i];
      }
    }

    if (PyBuffer_FillInfo(view, exporter, data, length, readOnly, flags) != 0) {
      return -1;
    }

    view->itemsize = itemSize;
    if ((flags & PyBUF_FORMAT) != 0) {
      view->format = const_cast<char*>(format);
    }
    if ((flags & PyBUF_ND) == 0) {
      return 0;
    }

    view->ndim = static_cast<int>(numDims);
    view->shape = layout->shape.data();
    view->strides = (flags & PyBUF_STRIDES) ? layout->strides.data() : nullptr;
    if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS && !PyBuffer_IsContiguous(view, 'F')) {
      Py_CLEAR(view->obj);
      PyErr_SetString(PyExc_BufferError, "Mochi buffer is not Fortran-contiguous");
      return -1;
    }
    view->internal = layout.release();
    return 0;
  } catch (...) {
    PyErr_NoMemory();
    return -1;
  }
}

inline void ReleasePythonBuffer(PyObject*, Py_buffer* view) noexcept {
  delete static_cast<PythonBufferLayout*>(view->internal);
  view->internal = nullptr;
}

// Type-erased registry of already-declared nanobind classes, keyed by C++ type.
//
// Enables two-phase registration of the generated bindings: a declaration phase
// registers every nb::class_ up front (via StoreClass) so that all types exist before
// any constructor default argument is converted to a Python object; a later definition
// phase retrieves the same handle (via GetClass) to attach members. Handles are stored
// type-erased as nb::object and recovered as the concrete nb::class_<T, Opts...> on retrieval.
//
// This is a local object created during module initialization and passed by reference
// to the generated Declare*/Define* functions; it is not a global and holds no state
// past import (its borrowed handles are released when it goes out of scope, while the
// classes themselves stay owned by the module).
class PybindRegistry {
 public:
  // Registers a freshly-declared class handle, keyed by its C++ type T.
  template <typename T, typename... Opts>
  void StoreClass(nb::class_<T, Opts...> cls) {
    auto [it, inserted] = _handles.emplace(std::type_index(typeid(T)), std::move(cls));
    (void)it;
    MOCHI_ASSERT(inserted);
  }

  // Retrieves the handle previously stored for T, reinterpreted as its concrete
  // nb::class_<T, Opts...>. Opts must match the declaration exactly.
  template <typename T, typename... Opts>
  nb::class_<T, Opts...> GetClass() const {
    auto it = _handles.find(std::type_index(typeid(T)));
    MOCHI_ASSERT(it != _handles.end());
    return nb::borrow<nb::class_<T, Opts...>>(it->second);
  }

 private:
  std::unordered_map<std::type_index, nb::object> _handles;
};

// Entry point for the generated bindings. Declares all classes and functions in the extension.
void DefineAll(nb::module_& m);

} // namespace mochi
