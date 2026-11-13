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
inline auto DefNdArray(nanobind::module_& m, char const* pyName, char const* doc) {
  namespace nb = nanobind;
  using NdArrayT = NdArray<T, N>;
  auto c = nb::class_<NdArrayT>(m, pyName, doc);
  c.def(nb::init<>());
  c.def("__init__", [pyName](NdArrayT* self, nb::sequence seq) {
    if (nb::len(seq) != N) {
      throw std::runtime_error(Format("%s requires exactly %d elements", pyName, N));
    }
    new (self) NdArrayT;
    for (int i = 0; i < N; ++i) {
      (*self)[i] = nb::cast<T>(seq[i]);
    }
  });
  c.def(
      "__array__",
      [](NdArrayT const& self, nb::object dtype, nb::object /*copy*/) -> nb::object {
        nb::object result = MakeOwningNumpy1D<T>(self.data(), self.size());
        if (!dtype.is_none()) {
          result = result.attr("astype")(dtype);
        }
        return result;
      },
      nb::arg("dtype") = nb::none(),
      nb::arg("copy") = nb::none());
  c.def("__getitem__", [](NdArrayT const& self, size_t index) -> T {
    if (index >= N) {
      throw nb::index_error();
    }
    return self[index];
  });
  c.def("__setitem__", [](NdArrayT& self, size_t index, T value) {
    if (index >= N) {
      throw nb::index_error();
    }
    self[index] = value;
  });
  c.def("__len__", [](NdArrayT const& /*self*/) { return size_t(N); });
  c.def("__reduce__", [](NdArrayT const& self) {
    nb::list values;
    for (auto const& v : self) {
      values.append(v);
    }
    return nb::make_tuple(nb::type<NdArrayT>(), nb::make_tuple(values));
  });
  c.def("__repr__", [](NdArrayT const& self) { return ToPyReplString(self); });
  c.def("__str__", [](NdArrayT const& self) { return ToPyString(self); });
  c.def("tolist", [](NdArrayT const& self) { return std::vector<real>(self.begin(), self.end()); });
  // clang-tidy misreads the nb::self operator-registration idiom (e.g. `nb::self / nb::self`)
  // as a redundant self-operation; nb::self is a binding marker, not a value. These register
  // __sub__/__truediv__/__eq__/__ne__ etc., so suppress the false positive over the block.
  // NOLINTBEGIN(misc-redundant-expression)
  c.def(nb::self + nb::self);
  c.def(nb::self - nb::self);
  c.def(nb::self * nb::self);
  c.def(nb::self / nb::self);
  c.def(nb::self + T());
  c.def(nb::self - T());
  c.def(nb::self * T());
  c.def(nb::self / T());
  c.def(T() + nb::self);
  c.def(T() - nb::self);
  c.def(T() * nb::self);
  c.def(T() / nb::self);
  c.def(nb::self += nb::self);
  c.def(nb::self -= nb::self);
  c.def(nb::self *= nb::self);
  c.def(nb::self /= nb::self);
  c.def(nb::self += T());
  c.def(nb::self -= T());
  c.def(nb::self *= T());
  c.def(nb::self /= T());
  c.def(-nb::self);
  c.def(nb::self == nb::self);
  c.def(nb::self != nb::self);
  // NOLINTEND(misc-redundant-expression)

  // copy.copy / copy.deepcopy. NdArray stores its elements inline so the
  // C++ copy constructor produces an independent copy — correct for both
  // shallow and deep semantics.
  c.def("__copy__", [](NdArrayT const& self) { return NdArrayT(self); });
  c.def(
      "__deepcopy__",
      [](NdArrayT const& self, nb::dict) { return NdArrayT(self); },
      nb::arg("memo"));

  // Allow implicit conversion using the nb::sequence initializer (above)
  nb::implicitly_convertible<nb::sequence, NdArrayT>();

  return c;
}

MOCHI_WARNING_POP()

} // namespace mochi
