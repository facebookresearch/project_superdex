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

#include "pybind_helpers.h"

namespace mochi {

template <class T>
inline auto DefDynamicArray(pybind11::module& m, char const* pyName) {
  using DynamicArrayT = DynamicArray<T>;
  auto c = pybind11::class_<DynamicArrayT>(m, pyName, pybind11::buffer_protocol());
  c.def(pybind11::init<>());
  c.def(pybind11::init<size_t, T const&>(), pybind11::arg("size"), pybind11::arg("value") = T{});
  c.def(
      pybind11::init([](pybind11::sequence sequence) {
        DynamicArrayT array;
        array.reserve(pybind11::len(sequence));
        for (auto item : sequence) {
          array.push_back(pybind11::cast<T>(item));
        }
        return array;
      }),
      pybind11::arg("sequence"));
  if constexpr (std::is_arithmetic_v<T>) {
    c.def(
        "__array__",
        [](DynamicArrayT const& self,
           pybind11::object dtype,
           pybind11::object /*copy*/) -> pybind11::object {
          auto result = pybind11::array_t<T>(self.size());
          pybind11::buffer_info buf = result.request();
          T* ptr = static_cast<T*>(buf.ptr);
          for (size_t i = 0; i < self.size(); ++i) {
            ptr[i] = self[i];
          }
          if (dtype.is_none()) {
            return result;
          } else {
            return pybind11::cast<pybind11::array>(result).attr("astype")(dtype);
          }
        },
        pybind11::arg("dtype") = pybind11::none(),
        pybind11::arg("copy") = pybind11::none());
  }
  c.def("__len__", &DynamicArrayT::size);
  c.def("__bool__", [](DynamicArrayT const& self) { return !self.empty(); });
  // For arithmetic types, return by value (Python ints/floats are immutable anyway).
  // For all other types, return by reference so that arr[i].field = value modifies the
  // actual element — matching standard Python list semantics.
  if constexpr (std::is_arithmetic_v<T>) {
    c.def("__getitem__", [](DynamicArrayT const& self, size_t index) -> T {
      if (index >= self.size()) {
        throw pybind11::index_error();
      }
      return self[index];
    });
  } else {
    c.def(
        "__getitem__",
        [](DynamicArrayT& self, size_t index) -> T& {
          if (index >= self.size()) {
            throw pybind11::index_error();
          }
          return self[index];
        },
        pybind11::return_value_policy::reference_internal);
  }
  c.def("__setitem__", [](DynamicArrayT& self, size_t index, T const& value) {
    if (index >= self.size()) {
      throw pybind11::index_error();
    }
    self[index] = value;
  });
  c.def(
      "__iter__",
      [](DynamicArrayT& self) { return pybind11::make_iterator(self.begin(), self.end()); },
      pybind11::keep_alive<0, 1>());
  c.def("__reduce__", [pyName](DynamicArrayT const& self) {
    if constexpr (std::is_arithmetic_v<T>) {
      // Create a numpy array that shares the same memory (no copy)
      auto array = pybind11::array_t<T>(self.size(), self.data(), pybind11::cast(self));
      return pybind11::make_tuple(
          pybind11::module::import(MOCHI_PHYSICS_MODULE_NAME_STR).attr(pyName),
          pybind11::make_tuple(array));
    } else {
      return pybind11::make_tuple(
          pybind11::module::import(MOCHI_PHYSICS_MODULE_NAME_STR).attr(pyName),
          pybind11::make_tuple(std::vector<T>(self.begin(), self.end())));
    }
  });
  c.def(
      "append",
      pybind11::overload_cast<T const&>(&DynamicArrayT::push_back),
      pybind11::arg("item"));
  c.def(
      "extend",
      [](DynamicArrayT& self, pybind11::sequence sequence) {
        self.reserve(self.size() + pybind11::len(sequence));
        for (auto item : sequence) {
          self.push_back(pybind11::cast<T>(item));
        }
      },
      pybind11::arg("sequence"));
  c.def(
      "extend",
      [](DynamicArrayT& self, DynamicArrayT const& sequence) { self.append(sequence); },
      pybind11::arg("sequence"));
  c.def("clear", &DynamicArrayT::clear);
  c.def("empty", &DynamicArrayT::empty);
  c.def("size", &DynamicArrayT::size);
  c.def("capacity", &DynamicArrayT::capacity);
  c.def("reserve", &DynamicArrayT::reserve, pybind11::arg("capacity"));
  c.def(
      "resize",
      static_cast<void (DynamicArrayT::*)(size_t)>(&DynamicArrayT::resize),
      pybind11::arg("size"));
  c.def(
      "resize",
      static_cast<void (DynamicArrayT::*)(size_t, T const&)>(&DynamicArrayT::resize),
      pybind11::arg("size"),
      pybind11::arg("value"));
  c.def("tolist", [](DynamicArrayT const& self) {
    pybind11::list result;
    for (auto const& item : self) {
      result.append(item);
    }
    return result;
  });

  // ToPyReplString and ToPyString depend on SReflect support.
  if constexpr (SReflect::IsSupportedType<T>()) {
    c.def("__repr__", [](DynamicArrayT const& self) { return ToPyReplString(self); });
    c.def("__str__", [](DynamicArrayT const& self) { return ToPyString(self); });
  }

  // Add buffer protocol support for arithmetic types
  if constexpr (std::is_arithmetic_v<T>) {
    c.def_buffer([](DynamicArrayT& self) -> pybind11::buffer_info {
      return pybind11::buffer_info(
          self.data(), // Pointer to buffer
          sizeof(T), // Size of one item
          pybind11::format_descriptor<T>::format(), // Python struct-style format descriptor
          1, // Number of dimensions
          {self.size()}, // Buffer dimensions
          {sizeof(T)} // Strides (in bytes) for each index
      );
    });
  }

  // Equality operators (only if T supports them)
  if constexpr (requires(T const& a, T const& b) { a == b; }) {
    c.def(pybind11::self == pybind11::self);
  }
  if constexpr (requires(T const& a, T const& b) { a != b; }) {
    c.def(pybind11::self != pybind11::self);
  }

  // copy.copy / copy.deepcopy. DynamicArray owns its memory so the C++ copy
  // constructor produces an independent copy — correct for both shallow and
  // deep semantics.
  c.def("__copy__", [](DynamicArrayT const& self) { return DynamicArrayT(self); });
  c.def(
      "__deepcopy__",
      [](DynamicArrayT const& self, pybind11::dict) { return DynamicArrayT(self); },
      pybind11::arg("memo"));

  // Allow implicit conversion using the pybind11::sequence initializer (above)
  pybind11::implicitly_convertible<pybind11::sequence, DynamicArray<T>>();

  return c;
}

} // namespace mochi
