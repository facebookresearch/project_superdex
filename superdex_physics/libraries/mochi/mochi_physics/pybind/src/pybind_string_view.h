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

namespace pybind11::detail {

// Full specialization of type_caster for std::string_view that keeps the source Python object
// alive. pybind11's default partial specialization points directly into CPython's internal UTF-8
// buffer without holding a reference to the Python str object.
//
// This is normally safe while the GIL is held, but becomes unsafe when the GIL is released via
// py::call_guard<py::gil_scoped_release>(): another thread could garbage-collect the string,
// invalidating the string_view.
//
// This specialization stores a pybind11::object to prevent GC, following the same pattern as
// type_caster<mochi::Span<T>> in pybind_span.h.
template <>
struct type_caster<std::string_view> {
  PYBIND11_TYPE_CASTER(std::string_view, _("str"));

  // Conversion from Python to C++
  bool load(handle src, bool) {
    if (!src) {
      return false;
    }
    if (PyUnicode_Check(src.ptr())) {
      Py_ssize_t size = 0;
      char const* buffer = PyUnicode_AsUTF8AndSize(src.ptr(), &size);
      if (!buffer) {
        PyErr_Clear();
        return false;
      }
      value = std::string_view(buffer, static_cast<size_t>(size));
      _source = reinterpret_borrow<object>(src);
      return true;
    }
    if (PyBytes_Check(src.ptr())) {
      char const* buffer = PyBytes_AsString(src.ptr());
      if (!buffer) {
        PyErr_Clear();
        return false;
      }
      value = std::string_view(buffer, static_cast<size_t>(PyBytes_Size(src.ptr())));
      _source = reinterpret_borrow<object>(src);
      return true;
    }
    return false;
  }

  // Conversion from C++ to Python
  static handle cast(std::string_view src, return_value_policy, handle) {
    return str(src.data(), src.size()).release();
  }

 private:
  object _source; // Prevent GC of the Python str/bytes while the string_view is alive
};

} // namespace pybind11::detail
