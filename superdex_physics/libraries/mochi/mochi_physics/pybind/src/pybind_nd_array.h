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

// Clang/GCC gets worried about operators like +=, -=, *=, /=. They work correctly even if the left
// and right sides are the same object
MOCHI_WARNING_PUSH()
MOCHI_WARNING_IGNORE_CLANG(clang diagnostic ignored "-Wself-assign-overloaded")

template <class T, int N>
inline auto DefNdArray(pybind11::module& m, char const* pyName, char const* doc) {
  using NdArrayT = NdArray<T, N>;
  auto c = pybind11::class_<NdArrayT>(m, pyName, doc);
  c.def(pybind11::init<>());
  c.def(pybind11::init([pyName](pybind11::sequence seq) {
    if (pybind11::len(seq) != N) {
      throw std::runtime_error(Format("%s requires exactly %d elements", pyName, N));
    }
    auto* arr = new NdArrayT;
    for (int i = 0; i < N; ++i) {
      (*arr)[i] = pybind11::cast<T>(seq[i]);
    }
    return arr;
  }));
  c.def(
      "__array__",
      [](NdArrayT const& self,
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
  c.def("__getitem__", [](NdArrayT const& self, size_t index) -> T {
    if (index >= N) {
      throw pybind11::index_error();
    }
    return self[index];
  });
  c.def("__setitem__", [](NdArrayT& self, size_t index, T value) {
    if (index >= N) {
      throw pybind11::index_error();
    }
    self[index] = value;
  });
  c.def("__len__", [](NdArrayT const& /*self*/) { return size_t(N); });
  c.def("__reduce__", [pyName](NdArrayT const& self) {
    // Create a numpy array that shares the same memory (no copy)
    auto array = pybind11::array_t<T>(self.size(), self.data(), pybind11::cast(self));
    return pybind11::make_tuple(
        pybind11::module::import(MOCHI_PHYSICS_MODULE_NAME_STR).attr(pyName),
        pybind11::make_tuple(array));
  });
  c.def("__repr__", [](NdArrayT const& self) { return ToPyReplString(self); });
  c.def("__str__", [](NdArrayT const& self) { return ToPyString(self); });
  c.def("tolist", [](NdArrayT const& self) { return std::vector<real>(self.begin(), self.end()); });
  c.def(pybind11::self + pybind11::self);
  c.def(pybind11::self - pybind11::self);
  c.def(pybind11::self * pybind11::self);
  c.def(pybind11::self / pybind11::self);
  c.def(pybind11::self + T());
  c.def(pybind11::self - T());
  c.def(pybind11::self * T());
  c.def(pybind11::self / T());
  c.def(T() + pybind11::self);
  c.def(T() - pybind11::self);
  c.def(T() * pybind11::self);
  c.def(T() / pybind11::self);
  c.def(pybind11::self += pybind11::self);
  c.def(pybind11::self -= pybind11::self);
  c.def(pybind11::self *= pybind11::self);
  c.def(pybind11::self /= pybind11::self);
  c.def(pybind11::self += T());
  c.def(pybind11::self -= T());
  c.def(pybind11::self *= T());
  c.def(pybind11::self /= T());
  c.def(-pybind11::self);
  c.def(pybind11::self == pybind11::self);
  c.def(pybind11::self != pybind11::self);

  // copy.copy / copy.deepcopy. NdArray stores its elements inline so the
  // C++ copy constructor produces an independent copy — correct for both
  // shallow and deep semantics.
  c.def("__copy__", [](NdArrayT const& self) { return NdArrayT(self); });
  c.def(
      "__deepcopy__",
      [](NdArrayT const& self, pybind11::dict) { return NdArrayT(self); },
      pybind11::arg("memo"));

  // Allow implicit conversion using the pybind11::sequence initializer (above)
  pybind11::implicitly_convertible<pybind11::sequence, NdArrayT>();

  return c;
}

MOCHI_WARNING_POP()

} // namespace mochi
