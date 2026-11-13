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

// @not-generated: This file is maintained by hand. The generated code goes into the "generated"
// folder and is invoked via mochi::DefineAll.

#include "pybind_dynamic_array.h"
#include "pybind_helpers.h"
#include "pybind_nd_array.h"
#include "pybind_span.h"
#include "pybind_string_view.h"

#include <mochi_physics/pybind/core/pybind_core.h>

#include <mochi_core/geometry/model_utils.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/quaternion_utils.h>
#include <mochi_physics/utils/mochi_prefab.h>

using namespace mochi;
namespace nb = nanobind;

NB_MODULE(MOCHI_PHYSICS_MODULE_NAME, m) {
  // The extension module lives until interpreter shutdown, so its bound types, functions, and
  // eagerly-created default-argument instances are still alive when nanobind's leak checker runs
  // at teardown. That is benign (the OS reclaims the memory on process exit) but the checker's
  // stderr output otherwise pollutes the test-listing protocol. pybind11 never performed this
  // check, so disable it to match.
  nb::set_leak_warnings(false);

  // Build configuration
  m.def(
      "is_debug",
      []() { return bool(MOCHI_DEBUG); },
      "Return whether the loaded native library is a debug build.");

  // MochiErrorException
  [[maybe_unused]] auto errorException =
      nb::exception<MochiErrorException>(m, "Error", PyExc_RuntimeError);

  // WARNING: If defining more NdArray types, please add them to `is_ndarray_type_alias`
  // in `emit_pybind.rs`.

  // Int3
  DefNdArray<int, 3>(m, "Int3", "Fixed-size three-element integer array.")
      .def(nb::init<int, int, int>(), nb::arg("x") = 0, nb::arg("y") = 0, nb::arg("z") = 0);

  // Real2
  DefNdArray<real, 2>(m, "Real2", "Fixed-size two-element floating-point array.")
      .def(nb::init<real, real>(), nb::arg("x") = real(0), nb::arg("y") = real(0));

  // Real3
  DefNdArray<real, 3>(m, "Real3", "Fixed-size three-element floating-point array.")
      .def(
          nb::init<real, real, real>(),
          nb::arg("x") = real(0),
          nb::arg("y") = real(0),
          nb::arg("z") = real(0));

  // Real6
  DefNdArray<real, 6>(m, "Real6", "Fixed-size six-element floating-point array.")
      .def(
          nb::init<real, real, real, real, real, real>(),
          nb::arg("a") = real(0),
          nb::arg("b") = real(0),
          nb::arg("c") = real(0),
          nb::arg("d") = real(0),
          nb::arg("e") = real(0),
          nb::arg("f") = real(0));

  // Color
  DefNdArray<uint8_t, 4>(
      m, "Color", "RGBA color representation using 4 bytes (0-255 per channel) in RGBA order.")
      .def(
          nb::init<uint8_t, uint8_t, uint8_t, uint8_t>(),
          nb::arg("r") = uint8_t(0),
          nb::arg("g") = uint8_t(0),
          nb::arg("b") = uint8_t(0),
          nb::arg("a") = uint8_t(0));

  // Quaternion
  nb::class_<Quaternion>(m, "Quaternion", "Used for 3D rotations.")
      .def(nb::init<>())
      .def(
          nb::init<real, real, real, real>(),
          nb::arg("x") = real(0),
          nb::arg("y") = real(0),
          nb::arg("z") = real(0),
          nb::arg("w") = real(0))
      .def(
          "__init__",
          [](Quaternion* self, nb::sequence seq) {
            if (nb::len(seq) != 4) {
              throw std::runtime_error(
                  "Quaternion requires exactly 4 elements in order [x, y, z, w]");
            }
            new (self) Quaternion{
                nb::cast<real>(seq[0]),
                nb::cast<real>(seq[1]),
                nb::cast<real>(seq[2]),
                nb::cast<real>(seq[3])};
          })
      .def(
          "__getitem__",
          [](Quaternion const& self, size_t index) -> real {
            if (index >= 4) {
              throw nb::index_error();
            }
            return self.data[index];
          })
      .def(
          "__setitem__",
          [](Quaternion& self, size_t index, real value) {
            if (index >= 4) {
              throw nb::index_error();
            }
            self.data = Set(self.data, index, value);
          })
      .def("__len__", [](Quaternion const& /*self*/) { return size_t(4); })
      .def(
          "__array__",
          [](Quaternion const& self, nb::object dtype, nb::object /*copy*/) -> nb::object {
            real values[] = {self.data[0], self.data[1], self.data[2], self.data[3]};
            nb::object result = MakeOwningNumpy1D(values, std::size(values));
            if (!dtype.is_none()) {
              result = result.attr("astype")(dtype);
            }
            return result;
          },
          nb::arg("dtype") = nb::none(),
          nb::arg("copy") = nb::none())
      .def("__repr__", [](Quaternion const& self) { return ToPyReplString(self); })
      .def("__str__", [](Quaternion const& self) { return ToPyString(self); })
      .def(
          "__reduce__",
          [](Quaternion const& self) {
            return nb::make_tuple(
                nb::type<Quaternion>(),
                nb::make_tuple(self.data[0], self.data[1], self.data[2], self.data[3]));
          })
      .def(
          "tolist",
          [](Quaternion const& self) {
            return std::vector<real>{self.data[0], self.data[1], self.data[2], self.data[3]};
          })
      .def("to_rotation_vector", &Quaternion::ToRotationVector)
      .def("get_conjugate", &Quaternion::GetConjugate)
      .def_static("identity", &Quaternion::Identity)
      .def_static("zero", &Quaternion::Zero)
      .def_static(
          "from_axis_angle",
          static_cast<Quaternion (*)(Real3 const&, real)>(&Quaternion::FromAxisAngle),
          nb::arg("axis"),
          nb::arg("angle"))
      .def_static(
          "from_rotation_vector",
          static_cast<Quaternion (*)(Real3 const&)>(&Quaternion::FromRotationVector),
          nb::arg("rotation_vector"))
      .def_static("rotation_x", &Quaternion::RotationX, nb::arg("angle"))
      .def_static("rotation_y", &Quaternion::RotationY, nb::arg("angle"))
      .def_static("rotation_z", &Quaternion::RotationZ, nb::arg("angle"))
      .def(nb::self * nb::self)
      .def(nb::self + nb::self)
      // NOLINTNEXTLINE(misc-redundant-expression) nb::self is a binding marker, not a value
      .def(nb::self - nb::self)
      .def(nb::self * real())
      .def(nb::self / real())
      .def(real() * nb::self)
      .def(-nb::self)
      // NOLINTNEXTLINE(misc-redundant-expression) nb::self is a binding marker, not a value
      .def(nb::self == nb::self)
      // NOLINTNEXTLINE(misc-redundant-expression) nb::self is a binding marker, not a value
      .def(nb::self != nb::self)
      .def("__copy__", [](Quaternion const& self) { return Quaternion(self); })
      .def(
          "__deepcopy__",
          [](Quaternion const& self, nb::dict) { return Quaternion(self); },
          nb::arg("memo"));

  // Allow implicit conversion from Python sequences (e.g., [0, 0, 0, 1]) to Quaternion,
  // consistent with how NdArray types (Real3, Real2, etc.) support this.
  nb::implicitly_convertible<nb::sequence, Quaternion>();

  // TransformRT
  nb::class_<TransformRT>(
      m, "TransformRT", "A 3D affine transform (rotation and translation, but no scale).")
      .def(nb::init<>())
      .def(nb::init<Quaternion const&>(), nb::arg("rotation"))
      .def(nb::init<Real3 const&>(), nb::arg("translation"))
      .def(nb::init<Quaternion const&, Real3 const&>(), nb::arg("rotation"), nb::arg("translation"))
      .def(
          "__reduce__",
          [](TransformRT const& self) {
            return nb::make_tuple(
                nb::type<TransformRT>(), nb::make_tuple(self.GetRotation(), self.GetTranslation()));
          })
      .def("__repr__", [](TransformRT const& self) { return ToPyReplString(self); })
      .def("__str__", [](TransformRT const& self) { return ToPyString(self); })
      .def_prop_rw("rotation", &TransformRT::GetRotation, &TransformRT::SetRotation)
      .def_prop_rw(
          "translation",
          &TransformRT::GetTranslation,
          [](TransformRT& self, Real3 const& translation) { self.SetTranslation(translation); })
      .def_static("identity", &TransformRT::Identity)
      .def("inverse", [](TransformRT const& self) { return Invert(self); })
      .def(
          "transform_point_inverse",
          static_cast<Real3 (TransformRT::*)(Real3 const&) const>(
              &TransformRT::TransformPointInverse),
          nb::arg("point"))
      .def(nb::self * nb::self)
      .def(nb::self *= nb::self)
      // NOLINTNEXTLINE(misc-redundant-expression) nb::self is a binding marker, not a value
      .def(nb::self == nb::self)
      // NOLINTNEXTLINE(misc-redundant-expression) nb::self is a binding marker, not a value
      .def(nb::self != nb::self)
      .def("__copy__", [](TransformRT const& self) { return TransformRT(self); })
      .def(
          "__deepcopy__",
          [](TransformRT const& self, nb::dict) { return TransformRT(self); },
          nb::arg("memo"));

  // Aabb
  nb::class_<Aabb>(m, "Aabb", "Axis-aligned bounding box in 3D space.")
      .def(nb::init<>())
      .def(nb::init<Real3 const&, Real3 const&>(), nb::arg("min"), nb::arg("max"))
      .def(
          "__reduce__",
          [](Aabb const& self) {
            return nb::make_tuple(nb::type<Aabb>(), nb::make_tuple(self.GetMin(), self.GetMax()));
          })
      .def("__repr__", [](Aabb const& self) { return ToPyReplString(self); })
      .def("__str__", [](Aabb const& self) { return ToPyString(self); })
      .def_prop_rw(
          "min",
          &Aabb::GetMin,
          [](Aabb& self, nb::object const& min) {
            self = Aabb(nb::cast<Real3>(min), self.GetMax());
          })
      .def_prop_rw(
          "max",
          &Aabb::GetMax,
          [](Aabb& self, nb::object const& max) {
            self = Aabb(self.GetMin(), nb::cast<Real3>(max));
          })
      .def("get_center", &Aabb::GetCenter)
      .def("get_size", &Aabb::GetSize)
      // NOLINTNEXTLINE(misc-redundant-expression) nb::self is a binding marker, not a value
      .def(nb::self == nb::self)
      // NOLINTNEXTLINE(misc-redundant-expression) nb::self is a binding marker, not a value
      .def(nb::self != nb::self)
      .def("__copy__", [](Aabb const& self) { return Aabb(self); })
      .def("__deepcopy__", [](Aabb const& self, nb::dict) { return Aabb(self); }, nb::arg("memo"));

  // Obb
  nb::class_<Obb>(m, "Obb", "Oriented bounding box in 3D space.")
      .def(nb::init<>())
      .def(
          nb::init<TransformRT const&, Real3 const&>(),
          nb::arg("transform"),
          nb::arg("half_extents"))
      .def("__repr__", [](Obb const& self) { return ToPyReplString(self); })
      .def("__str__", [](Obb const& self) { return ToPyString(self); })
      .def("get_center", &Obb::GetCenter)
      .def("get_half_extents", &Obb::GetHalfExtents)
      .def("get_size", &Obb::GetSize)
      .def("__copy__", [](Obb const& self) { return Obb(self); })
      .def("__deepcopy__", [](Obb const& self, nb::dict) { return Obb(self); }, nb::arg("memo"));

  m.def(
      "is_initialized",
      []() { return GetContext() != nullptr; },
      "Return whether SuperDex Physics is currently initialized.");

  // initialize() performs no dependent teardown: InitGlobalContext throws if a context already
  // exists, so the only way to re-initialize is to call shutdown() first (which drains
  // dependents). A freshly created context therefore never has stale dependents bound to a
  // previous Context*.
  m.def(
      "initialize",
      &InitGlobalContext,
      nb::arg("num_worker_threads"),
      R"doc(Initialize SuperDex Physics by creating the process-wide context.

Only one context is needed for the entire process. It can be used to load shapes and create any
number of scenes. Call shutdown() to clean everything up when you are done.

Args:
    num_worker_threads: The number of asynchronous worker threads to initialize, or zero for
        single-threaded execution. Negative values (for example, -1) let SuperDex Physics choose a
        default based on the CPU hardware. Positive values are clamped to the number of logical
        processors.

Note:
    Simulation performance does not necessarily improve by using more threads. For simple scenes,
    using a small number of threads, or even single-threaded execution, may be fastest.

Raises:
    RuntimeError: If SuperDex Physics is already initialized.)doc");

  // shutdown() tears down the dependent contexts (bots/mpc) — in reverse registration order,
  // while the physics Context is still alive — before destroying it, so a dependent never
  // outlives the Context it was built against.
  m.def(
      "shutdown",
      []() {
        mochi::SetLogCallback(nullptr);
        ShutdownGlobalContext();
      },
      R"doc(Shut down SuperDex Physics by destroying the process-wide context.

Any remaining state is immediately destroyed, including all scenes, actors, and constraints.
Registered dependent contexts are destroyed first.

Warning:
    Must be called on the same thread that called initialize().

Raises:
    RuntimeError: If SuperDex Physics is not initialized.)doc");

  m.def(
      "uses_double_precision",
      []() { return bool(MOCHI_USE_DOUBLE_PRECISION); },
      "Return whether the loaded native library uses double-precision floating-point (FP64) values.");
  m.def(
      "uses_hdf5",
      []() { return bool(MOCHI_USE_HDF5); },
      "Return whether the loaded native library includes HDF5 support.");

  m.def(
      "normalize",
      static_cast<Real2 (*)(Real2 const&)>(&mochi::Normalize),
      nb::arg("v"),
      "Return a unit-length copy of the vector. A zero vector remains zero.");
  m.def(
      "normalize",
      static_cast<Real3 (*)(Real3 const&)>(&mochi::Normalize),
      nb::arg("v"),
      "Return a unit-length copy of the vector. A zero vector remains zero.");
  m.def(
      "normalize",
      static_cast<Quaternion (*)(Quaternion)>(&mochi::Normalize),
      nb::arg("q"),
      "Return a unit-length copy of the quaternion. A zero quaternion remains zero.");

  // Insert generated bindings here
  mochi::DefineAll(m);

  // Override release_shape to accept None (equivalent to default-constructed ShapeHandle)
  m.def(
      "release_shape",
      [](std::optional<ShapeHandle> shape) {
        CheckContext();
        GetContext()->ReleaseShape(shape.value_or(ShapeHandle{}));
      },
      nb::arg("shape"));

  // This must come after the generated code because it references generated type LogChannel.
  m.def(
      "log",
      [](std::string const& message, LogChannel channel = LogChannel::Info) {
        nb::gil_scoped_acquire gil;
        nb::module_ inspect = nb::module_::import_("inspect");
        nb::object stack = inspect.attr("stack")();
        nb::object frame_info = stack[nb::int_(0)];
        auto filename = nb::cast<std::string>(frame_info[nb::int_(1)]);
        auto lineno = nb::cast<int>(frame_info[nb::int_(2)]);
        MOCHI_LOG_IMPL(channel, filename.c_str(), lineno, "%s", message.c_str());
      },
      nb::arg("message"),
      nb::arg("channel") = LogChannel::Info);

  // Best-effort clean-up on module exit: tear down dependent contexts (bots/mpc) first,
  // then the physics Context. This is the single atexit for the whole context tree — bots
  // and mpc do not register their own, so ordering does not depend on atexit LIFO. Runs from
  // Python while the interpreter and GIL are alive (never from a C++ static destructor). If the
  // user already called shutdown(), there is no live context tree left to clean up; avoid running
  // dependent teardown callbacks again during interpreter shutdown, when extension unload order is
  // platform-sensitive.
  // The private name is a stable contract with worker bootstrap code that suppresses this callback
  // in distributed processes where interpreter finalization can race scheduler teardown.
  nb::object shutdownCallback = nb::cpp_function(
      []() {
        if (GetContext()) {
          mochi::SetLogCallback(nullptr);
          ShutdownGlobalContext();
        }
      },
      nb::scope(m),
      nb::name("_mochi_shutdown_global_context_atexit"));
  m.attr("_mochi_shutdown_global_context_atexit") = shutdownCallback;
  nb::module_::import_("atexit").attr("register")(shutdownCallback);

} // NB_MODULE
