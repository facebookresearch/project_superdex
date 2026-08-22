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

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/string_utils.h>

namespace mochi {

template <typename T>
constexpr bool IsArithmeticNdArray() {
  if constexpr (kIsNdArray<T>) {
    using E = typename T::element_type;
    return std::is_arithmetic_v<E>;
  }
  return false;
}

// A wrapper for the Span class, which Python can iterate without necessarily copying the data
template <typename T>
class PySpan {
 public:
  using NonConstT = std::remove_const_t<T>;

  PySpan() = default;
  PySpan(mochi::Span<T> span) : _span(span) {}

  // Iterator support for Python
  T* begin() const {
    return _span.data();
  }
  T* end() const {
    return _span.end();
  }

  // Size and indexing
  size_t size() const {
    return _span.size();
  }
  T& operator[](size_t index) const {
    if (index >= _span.size()) {
      throw pybind11::index_error();
    }
    return _span[index];
  }

  // Buffer protocol support
  pybind11::buffer_info get_buffer_info() {
    // Specialization for NdArrays with arithmetic element type.
    if constexpr (IsArithmeticNdArray<T>()) {
      using E = typename T::element_type;
      int constexpr kNumDims = T::num_dims;
      std::array<pybind11::ssize_t, 1 + kNumDims> dims{};
      std::array<pybind11::ssize_t, 1 + kNumDims> strides{};

      // Extract the dimensions and strides from the NdArray.
      dims[0] = static_cast<pybind11::ssize_t>(_span.size());
      for (int i = 0; i < kNumDims; ++i) {
        dims[1 + i] = static_cast<pybind11::ssize_t>(T::dims[i]);
      }

      // Calculate strides. Work backwards from innermost to outermost.
      strides[kNumDims] = static_cast<pybind11::ssize_t>(sizeof(E));
      for (int i = kNumDims - 1; i >= 0; --i) {
        strides[i] = strides[i + 1] * dims[i + 1];
      }

      return pybind11::buffer_info(
          const_cast<NonConstT*>(_span.data()), // Pointer to buffer
          sizeof(E), // Size of one item
          pybind11::format_descriptor<E>::format(), // Python struct-style format descriptor
          1 + kNumDims, // Number of dimensions (span + ndarray dims)
          dims, // Buffer dimensions
          strides, // Strides (in bytes) for each index
          std::is_const_v<T> // Set readonly flag for const types
      );
    }

    // Default behavior for non-NdArray types
    return pybind11::buffer_info(
        const_cast<NonConstT*>(_span.data()), // Pointer to buffer
        sizeof(T), // Size of one item
        pybind11::format_descriptor<NonConstT>::format(), // Python struct-style format descriptor
        1, // Number of dimensions
        {_span.size()}, // Buffer dimensions
        {sizeof(T)}, // Strides (in bytes) for each index
        std::is_const_v<T> // Set readonly flag for const types
    );
  }

 private:
  mochi::Span<T> _span;
};

} // namespace mochi

namespace pybind11::detail {

// Cast Python list to Span<T>
template <class T>
struct type_caster<mochi::Span<T>> {
 public:
  using NonConstT = std::remove_const_t<T>;
  PYBIND11_TYPE_CASTER(mochi::Span<T>, _("Span[") + make_caster<T>::name + _("]"));

  // Conversion from Python to C++
  bool load(handle src, bool) {
    if constexpr (std::is_arithmetic_v<T>) {
      // Special handling for pybind11::bytes --> mochi::Span<char const>.
      if constexpr (std::is_same_v<NonConstT, char>) {
        if (pybind11::isinstance<pybind11::bytes>(src)) {
          char* buffer = nullptr;
          pybind11::ssize_t size = 0;
          if (PYBIND11_BYTES_AS_STRING_AND_SIZE(src.ptr(), &buffer, &size) == 0) {
            value = mochi::Span<T>(buffer, static_cast<size_t>(size));
            _source = pybind11::reinterpret_borrow<pybind11::object>(src);
            return true;
          }
        }
      }
      // Since the inner type is an arithmetic type, we might be able to get direct access to the
      // source memory using Python's buffer interface. This woks for source types like
      // mochi::DynamicArray<T>. It also works for numpy arrays, as long as the numpy dtype is
      // correct.
      if (pybind11::isinstance<pybind11::buffer>(src)) {
        try {
          auto buf = pybind11::cast<pybind11::buffer>(src);
          pybind11::buffer_info info = buf.request();
          if (info.ndim == 1) {
            // Check if the element type is equivalent. This check fails for pybind11::bytearray (a
            // type of pybind11::buffer), but we still allow it for arrays of characters. Example
            // usage: see LoadShapeFromBytes.
            bool sameElementType = info.item_type_is_equivalent_to<NonConstT>() ||
                (std::is_same_v<NonConstT, char> && pybind11::isinstance<pybind11::bytearray>(src));
            if (sameElementType) {
              value = mochi::Span<T>(static_cast<T*>(info.ptr), info.size);
              _source = std::move(buf); // Keep the Python object alive
              return true;
            }
          }
        } catch (...) {
          // Fall through to iterable approach
        }
      }
    } else {
      // For non-arithmetic types, we might still be able to get direct access to the source memory
      // if we can recognize the type. For example, it might be something like
      // mochi::DynamicArray<mochi::TransformRT>.
      try {
        if (auto* dynamicArray =
                pybind11::cast<mochi::DynamicArray<std::remove_const_t<T>>*>(src)) {
          value = mochi::Span<T>(dynamicArray->data(), dynamicArray->size());
          _source =
              pybind11::reinterpret_borrow<pybind11::object>(src); // Keep the Python object alive
          return true;
        }
      } catch (...) {
        // Fall through to iterable approach
      }
    }

    // A non-const Span can be used as an output parameter, but we must guard against outputting
    // data to a temporary variable. Therefore conversion to a non-const Span is only allowed if we
    // can get direct access to the destination memory (see above).
    if constexpr (!std::is_const_v<T>) {
      std::string innerTypeName = SReflect::GetTypeInfo<T>()._name;
      std::string mochiRecommendation;
      std::string numpyRecommendation;
      if constexpr (std::is_same_v<NonConstT, int>) {
        mochiRecommendation = "DynamicArrayInt";
        numpyRecommendation = "int32";
      } else if constexpr (std::is_same_v<NonConstT, mochi::real>) {
        mochiRecommendation = "DynamicArrayReal";
        numpyRecommendation = MOCHI_USE_DOUBLE_PRECISION ? "float64" : "float32";
      } else if (!innerTypeName.empty() && (innerTypeName[0] >= 'A' && innerTypeName[0] <= 'Z')) {
        mochiRecommendation = mochi::Format("DynamicArray%s", innerTypeName.c_str());
      }
      std::string message = mochi::Format(
          "Incompatible argument type. Mochi requires a contiguous span of values of type %s that it can write to.",
          innerTypeName.c_str());
      if (!mochiRecommendation.empty()) {
        message += mochi::Format(" Consider using Mochi type %s", mochiRecommendation.c_str());
      }
      if (!numpyRecommendation.empty()) {
        message += mochi::Format(" or a numpy array with dtype=%s", numpyRecommendation.c_str());
      }
      message += ".";
      throw pybind11::type_error(message);
    }

    // Copy the Python iterable object into a temporary mochi::DynamicArray
    size_t len = pybind11::len(src);
    _storage.resize(len);
    try {
      size_t i = 0;
      for (auto item : src) {
        _storage[i++] = pybind11::cast<T>(item);
      }
    } catch (pybind11::cast_error const&) {
      return false;
    }

    // The cast will result in a Span pointing to our _storage
    value = mochi::Span<T>(_storage.data(), _storage.size());
    return true;
  }

  // Zero-copy conversion from C++ to Python
  static handle cast(mochi::Span<T> src, return_value_policy policy, handle parent) {
    // Return a wrapper with buffer protocol support
    auto wrapper = mochi::PySpan<T>(src);
    return pybind11::cast(std::move(wrapper), policy, parent).release();
  }

 private:
  mochi::DynamicArray<NonConstT> _storage; // To keep the data alive
  pybind11::object _source; // To keep the source Python object alive (buffer, bytes, etc.)
};

} // namespace pybind11::detail

namespace mochi {

template <typename T>
auto DefSpan(pybind11::module& m, char const* pyName) {
  using NonConstT = std::remove_const_t<T>;
  auto c = pybind11::class_<PySpan<T>>(m, pyName, pybind11::buffer_protocol());
  c.def(pybind11::init<>());
  c.def(
      pybind11::init(
          [](DynamicArray<NonConstT>& arr) { return PySpan<T>(Span<T>(arr.data(), arr.size())); }),
      pybind11::keep_alive<1, 2>(),
      pybind11::arg("array"));
  c.def("__len__", &PySpan<T>::size);
  // For arithmetic types, return by value (Python ints/floats are immutable anyway).
  // For const spans, also return by copy to prevent mutation through a const view.
  // For all other types, return by reference so that span[i].field = value modifies the
  // actual element — matching standard Python list semantics.
  if constexpr (std::is_arithmetic_v<std::remove_const_t<T>> || std::is_const_v<T>) {
    c.def("__getitem__", &PySpan<T>::operator[]);
  } else {
    c.def("__getitem__", &PySpan<T>::operator[], pybind11::return_value_policy::reference_internal);
  }
  c.def("__setitem__", [](PySpan<T>& self, size_t index, T const& value) {
    if constexpr (std::is_const_v<T>) {
      throw pybind11::type_error("Cannot modify a const Span");
    } else {
      self[index] = value;
    }
  });
  if constexpr (std::is_arithmetic_v<std::remove_const_t<T>> || std::is_const_v<T>) {
    c.def(
        "__iter__",
        [](PySpan<T>& s) {
          return pybind11::make_iterator<pybind11::return_value_policy::copy>(s.begin(), s.end());
        },
        pybind11::keep_alive<0, 1>());
  } else {
    c.def(
        "__iter__",
        [](PySpan<T>& s) { return pybind11::make_iterator(s.begin(), s.end()); },
        pybind11::keep_alive<0, 1>());
  }
  c.def("tolist", [](PySpan<T> const& s) {
    return std::vector<std::remove_const_t<T>>(s.begin(), s.end());
  });

  // copy.copy / copy.deepcopy. A Span is a non-owning view. Copy it to a DynamicArray instead.
  c.def("__copy__", [](PySpan<T> const& self) {
    return DynamicArray<NonConstT>(self.begin(), self.end());
  });
  c.def(
      "__deepcopy__",
      [](PySpan<T> const& self, pybind11::dict) {
        return DynamicArray<NonConstT>(self.begin(), self.end());
      },
      pybind11::arg("memo"));

  if constexpr (std::is_arithmetic_v<T>) {
    c.def_buffer(&PySpan<T>::get_buffer_info);
    c.def(
        "__array__",
        [](PySpan<T>& s,
           pybind11::object /*dtype*/,
           pybind11::object /*copy*/) -> pybind11::array_t<NonConstT> {
          auto info = s.get_buffer_info();
          return pybind11::array_t<NonConstT>(
              info.size,
              static_cast<T*>(info.ptr),
              pybind11::cast(s) // Keep wrapper alive
          );
        },
        pybind11::arg("dtype") = pybind11::none(),
        pybind11::arg("copy") = pybind11::none());
  }

  if constexpr (IsArithmeticNdArray<T>()) {
    using E = typename T::element_type;
    using NonConstE = std::remove_const_t<E>;
    c.def_buffer(&PySpan<T>::get_buffer_info);
    c.def(
        "__array__",
        [](PySpan<T>& s,
           pybind11::object /*dtype*/,
           pybind11::object /*copy*/) -> pybind11::array_t<NonConstE> {
          auto info = s.get_buffer_info();
          return pybind11::array_t<NonConstE>(
              info.size,
              static_cast<E*>(info.ptr),
              pybind11::cast(s) // Keep wrapper alive
          );
        },
        pybind11::arg("dtype") = pybind11::none(),
        pybind11::arg("copy") = pybind11::none());
  }

  return c;
}

} // namespace mochi
