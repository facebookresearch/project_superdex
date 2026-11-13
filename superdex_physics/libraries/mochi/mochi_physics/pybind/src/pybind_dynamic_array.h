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

inline constexpr char kDynamicArrayDoc[] = R"doc(A resizable array with contiguous storage.

Warning:
    Instances are not thread-safe. Concurrent access to the same instance requires
    external synchronization if any access mutates the array or its exported storage.)doc";

template <class T>
struct DynamicArrayBufferProtocol {
  using DynamicArrayT = DynamicArray<T>;

  static int GetBuffer(PyObject* exporter, Py_buffer* view, int flags) noexcept {
    auto* self = nanobind::inst_ptr<DynamicArrayT>(nanobind::handle(exporter));
    std::array<Py_ssize_t, 1> shape{static_cast<Py_ssize_t>(self->size())};
    return FillPythonBuffer(
        exporter,
        view,
        flags,
        self->data(),
        sizeof(T),
        PythonBufferFormat<T>(),
        false,
        shape.data(),
        shape.size());
  }

  inline static PyType_Slot kSlots[] = {
      {Py_bf_getbuffer, reinterpret_cast<void*>(GetBuffer)},
      {Py_bf_releasebuffer, reinterpret_cast<void*>(ReleasePythonBuffer)},
      {0, nullptr},
  };
};

template <class T>
T CastDynamicArrayItem(nanobind::handle item) {
  try {
    return nanobind::cast<T>(item);
  } catch (nanobind::cast_error const&) {
    throw nanobind::type_error("DynamicArray sequence item has an incompatible type");
  }
}

template <class T>
inline auto DefDynamicArray(nanobind::module_& m, char const* pyName) {
  namespace nb = nanobind;
  using DynamicArrayT = DynamicArray<T>;
  auto c = [&]() {
    if constexpr (std::is_arithmetic_v<T>) {
      return nb::class_<DynamicArrayT>(
          m, pyName, kDynamicArrayDoc, nb::type_slots(DynamicArrayBufferProtocol<T>::kSlots));
    } else {
      return nb::class_<DynamicArrayT>(m, pyName, kDynamicArrayDoc);
    }
  }();
  nb::handle scope = c; // Non-owning handle to the class (owned by the module) for make_iterator.
  c.def(nb::init<>());
  c.def(
      "__init__",
      [](DynamicArrayT* self, size_t size, T const& value) {
        new (self) DynamicArrayT(size, value);
      },
      nb::arg("size"),
      nb::arg("value") = T{});
  if constexpr (std::is_same_v<T, char>) {
    c.def("__init__", [](DynamicArrayT* self, nb::bytes values) {
      new (self) DynamicArrayT(values.c_str(), values.c_str() + values.size());
    });
  }
  c.def(
      "__init__",
      [](DynamicArrayT* self, nb::sequence sequence) {
        DynamicArrayT result;
        result.reserve(nb::len(sequence));
        for (auto item : sequence) {
          result.push_back(CastDynamicArrayItem<T>(item));
        }
        new (self) DynamicArrayT(std::move(result));
      },
      nb::arg("sequence"));
  if constexpr (std::is_arithmetic_v<T>) {
    c.def(
        "__array__",
        [](DynamicArrayT const& self, nb::object dtype, nb::object /*copy*/) -> nb::object {
          nb::object result = MakeOwningNumpy1D<T>(self.data(), self.size());
          if (!dtype.is_none()) {
            result = result.attr("astype")(dtype);
          }
          return result;
        },
        nb::arg("dtype") = nb::none(),
        nb::arg("copy") = nb::none());
  }
  c.def("__len__", &DynamicArrayT::size);
  c.def("__bool__", [](DynamicArrayT const& self) { return !self.empty(); });
  // For arithmetic types, return by value (Python ints/floats are immutable anyway).
  // For all other types, return by reference so that arr[i].field = value modifies the
  // actual element — matching standard Python list semantics.
  if constexpr (std::is_arithmetic_v<T>) {
    c.def("__getitem__", [](DynamicArrayT const& self, size_t index) -> T {
      if (index >= self.size()) {
        throw nb::index_error();
      }
      return self[index];
    });
  } else {
    c.def(
        "__getitem__",
        [](DynamicArrayT& self, size_t index) -> T& {
          if (index >= self.size()) {
            throw nb::index_error();
          }
          return self[index];
        },
        nb::rv_policy::reference_internal);
  }
  c.def("__setitem__", [](DynamicArrayT& self, size_t index, T const& value) {
    if (index >= self.size()) {
      throw nb::index_error();
    }
    self[index] = value;
  });
  if constexpr (std::is_arithmetic_v<T>) {
    c.def(
        "__iter__",
        [scope](DynamicArrayT& self) {
          return nb::make_iterator<nb::rv_policy::copy>(
              scope, "Iterator", self.begin(), self.end());
        },
        nb::keep_alive<0, 1>());
  } else {
    c.def(
        "__iter__",
        [scope](DynamicArrayT& self) {
          return nb::make_iterator<nb::rv_policy::reference_internal>(
              scope, "Iterator", self.begin(), self.end());
        },
        nb::keep_alive<0, 1>());
  }
  c.def("__reduce__", [](DynamicArrayT const& self) {
    nb::object values;
    // The item casters reject the NumPy scalars of bool and char arrays, and a char only converts
    // to a str if it is ASCII.
    if constexpr (std::is_same_v<T, char>) {
      values = nb::bytes(self.data(), self.size());
    } else if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) {
      if (self.empty()) {
        values = nb::list();
      } else {
        using ArrayView = nb::ndarray<nb::numpy, T const, nb::ndim<1>, nb::c_contig>;
        values = ArrayView(self.data(), {self.size()})
                     .cast(nb::rv_policy::reference_internal, nb::find(self));
      }
    } else {
      nb::list items;
      for (auto const& item : self) {
        items.append(item);
      }
      values = std::move(items);
    }
    return nb::make_tuple(nb::type<DynamicArrayT>(), nb::make_tuple(values));
  });
  c.def("append", nb::overload_cast<T const&>(&DynamicArrayT::push_back), nb::arg("item"));
  c.def(
      "extend",
      [](DynamicArrayT& self, nb::sequence sequence) {
        self.reserve(self.size() + nb::len(sequence));
        for (auto item : sequence) {
          self.push_back(CastDynamicArrayItem<T>(item));
        }
      },
      nb::arg("sequence"));
  c.def(
      "extend",
      [](DynamicArrayT& self, DynamicArrayT const& sequence) { self.append(sequence); },
      nb::arg("sequence"));
  c.def("clear", &DynamicArrayT::clear);
  c.def("empty", &DynamicArrayT::empty);
  c.def("size", &DynamicArrayT::size);
  c.def("capacity", &DynamicArrayT::capacity);
  c.def("reserve", &DynamicArrayT::reserve, nb::arg("capacity"));
  c.def(
      "resize",
      static_cast<void (DynamicArrayT::*)(size_t)>(&DynamicArrayT::resize),
      nb::arg("size"));
  c.def(
      "resize",
      static_cast<void (DynamicArrayT::*)(size_t, T const&)>(&DynamicArrayT::resize),
      nb::arg("size"),
      nb::arg("value"));
  c.def("tolist", [](DynamicArrayT const& self) {
    nb::list result;
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

  // Equality operators (only if T supports them)
  if constexpr (requires(T const& a, T const& b) { a == b; }) {
    c.def(nb::self == nb::self);
  }
  if constexpr (requires(T const& a, T const& b) { a != b; }) {
    c.def(nb::self != nb::self);
  }

  // copy.copy / copy.deepcopy. DynamicArray owns its memory so the C++ copy
  // constructor produces an independent copy — correct for both shallow and
  // deep semantics.
  c.def("__copy__", [](DynamicArrayT const& self) { return DynamicArrayT(self); });
  c.def(
      "__deepcopy__",
      [](DynamicArrayT const& self, nb::dict) { return DynamicArrayT(self); },
      nb::arg("memo"));

  // Allow implicit conversion using the nb::sequence initializer (above)
  nb::implicitly_convertible<nb::sequence, DynamicArray<T>>();

  return c;
}

} // namespace mochi
