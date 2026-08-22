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
namespace py = pybind11;

PYBIND11_MODULE(MOCHI_PHYSICS_MODULE_NAME, m) {
  // Build configuration
  m.def(
      "is_debug",
      []() { return bool(MOCHI_DEBUG); },
      "Return whether the loaded native library is a debug build.");

  // MochiErrorException
  py::register_exception<MochiErrorException>(m, "Error", PyExc_RuntimeError);

  // WARNING: If defining more NdArray types, please add them to `is_ndarray_type_alias`
  // in `emit_pybind.rs`.

  // Int3
  DefNdArray<int, 3>(m, "Int3", "Fixed-size three-element integer array.")
      .def(py::init<int, int, int>(), py::arg("x") = 0, py::arg("y") = 0, py::arg("z") = 0);

  // Real2
  DefNdArray<real, 2>(m, "Real2", "Fixed-size two-element floating-point array.")
      .def(py::init<real, real>(), py::arg("x") = real(0), py::arg("y") = real(0));

  // Real3
  DefNdArray<real, 3>(m, "Real3", "Fixed-size three-element floating-point array.")
      .def(
          py::init<real, real, real>(),
          py::arg("x") = real(0),
          py::arg("y") = real(0),
          py::arg("z") = real(0));

  // Real6
  DefNdArray<real, 6>(m, "Real6", "Fixed-size six-element floating-point array.")
      .def(
          py::init<real, real, real, real, real, real>(),
          py::arg("a") = real(0),
          py::arg("b") = real(0),
          py::arg("c") = real(0),
          py::arg("d") = real(0),
          py::arg("e") = real(0),
          py::arg("f") = real(0));

  // Color
  DefNdArray<uint8_t, 4>(
      m, "Color", "RGBA color representation using 4 bytes (0-255 per channel) in RGBA order.")
      .def(
          py::init<uint8_t, uint8_t, uint8_t, uint8_t>(),
          py::arg("r") = uint8_t(0),
          py::arg("g") = uint8_t(0),
          py::arg("b") = uint8_t(0),
          py::arg("a") = uint8_t(0));

  // Quaternion
  py::class_<Quaternion>(m, "Quaternion", "Used for 3D rotations.")
      .def(py::init<>())
      .def(
          py::init<real, real, real, real>(),
          py::arg("x") = real(0),
          py::arg("y") = real(0),
          py::arg("z") = real(0),
          py::arg("w") = real(0))
      .def(py::init([](py::sequence seq) {
        if (py::len(seq) != 4) {
          throw std::runtime_error("Quaternion requires exactly 4 elements in order [x, y, z, w]");
        }
        return new Quaternion{
            py::cast<real>(seq[0]),
            py::cast<real>(seq[1]),
            py::cast<real>(seq[2]),
            py::cast<real>(seq[3])};
      }))
      .def(
          "__getitem__",
          [](Quaternion const& self, size_t index) -> real {
            if (index >= 4) {
              throw py::index_error();
            }
            return self.data[index];
          })
      .def(
          "__setitem__",
          [](Quaternion& self, size_t index, real value) {
            if (index >= 4) {
              throw py::index_error();
            }
            self.data = Set(self.data, index, value);
          })
      .def("__len__", [](Quaternion const& /*self*/) { return size_t(4); })
      .def("__repr__", [](Quaternion const& self) { return ToPyReplString(self); })
      .def("__str__", [](Quaternion const& self) { return ToPyString(self); })
      .def(
          "__reduce__",
          [](Quaternion const& self) {
            return py::make_tuple(
                py::module::import(MOCHI_PHYSICS_MODULE_NAME_STR).attr("Quaternion"),
                py::make_tuple(self.data[0], self.data[1], self.data[2], self.data[3]));
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
          py::arg("axis"),
          py::arg("angle"))
      .def_static(
          "from_rotation_vector",
          static_cast<Quaternion (*)(Real3 const&)>(&Quaternion::FromRotationVector),
          py::arg("rotation_vector"))
      .def_static("rotation_x", &Quaternion::RotationX, py::arg("angle"))
      .def_static("rotation_y", &Quaternion::RotationY, py::arg("angle"))
      .def_static("rotation_z", &Quaternion::RotationZ, py::arg("angle"))
      .def(py::self * py::self)
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * real())
      .def(py::self / real())
      .def(real() * py::self)
      .def(-py::self)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__copy__", [](Quaternion const& self) { return Quaternion(self); })
      .def(
          "__deepcopy__",
          [](Quaternion const& self, py::dict) { return Quaternion(self); },
          py::arg("memo"));

  // Allow implicit conversion from Python sequences (e.g., [0, 0, 0, 1]) to Quaternion,
  // consistent with how NdArray types (Real3, Real2, etc.) support this.
  py::implicitly_convertible<py::sequence, Quaternion>();

  // TransformRT
  py::class_<TransformRT>(
      m, "TransformRT", "A 3D affine transform (rotation and translation, but no scale).")
      .def(py::init<>())
      .def(py::init<Quaternion const&>(), py::arg("rotation"))
      .def(py::init<Real3 const&>(), py::arg("translation"))
      .def(py::init<Quaternion const&, Real3 const&>(), py::arg("rotation"), py::arg("translation"))
      .def(
          "__reduce__",
          [](TransformRT const& self) {
            return py::make_tuple(
                py::module::import(MOCHI_PHYSICS_MODULE_NAME_STR).attr("TransformRT"),
                py::make_tuple(self.GetRotation(), self.GetTranslation()));
          })
      .def("__repr__", [](TransformRT const& self) { return ToPyReplString(self); })
      .def("__str__", [](TransformRT const& self) { return ToPyString(self); })
      .def_property("rotation", &TransformRT::GetRotation, &TransformRT::SetRotation)
      .def_property(
          "translation",
          &TransformRT::GetTranslation,
          [](TransformRT& self, Real3 const& translation) { self.SetTranslation(translation); })
      .def_static("identity", &TransformRT::Identity)
      .def("inverse", [](TransformRT const& self) { return Invert(self); })
      .def(
          "transform_point_inverse",
          static_cast<Real3 (TransformRT::*)(Real3 const&) const>(
              &TransformRT::TransformPointInverse),
          py::arg("point"))
      .def(py::self * py::self)
      .def(py::self *= py::self)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__copy__", [](TransformRT const& self) { return TransformRT(self); })
      .def(
          "__deepcopy__",
          [](TransformRT const& self, py::dict) { return TransformRT(self); },
          py::arg("memo"));

  // Aabb
  py::class_<Aabb>(m, "Aabb", "Axis-aligned bounding box in 3D space.")
      .def(py::init<>())
      .def(py::init<Real3 const&, Real3 const&>(), py::arg("min"), py::arg("max"))
      .def(
          "__reduce__",
          [](Aabb const& self) {
            return py::make_tuple(
                py::module::import(MOCHI_PHYSICS_MODULE_NAME_STR).attr("Aabb"),
                py::make_tuple(self.GetMin(), self.GetMax()));
          })
      .def("__repr__", [](Aabb const& self) { return ToPyReplString(self); })
      .def("__str__", [](Aabb const& self) { return ToPyString(self); })
      .def_property(
          "min",
          &Aabb::GetMin,
          [](Aabb& self, py::object const& min) {
            self = Aabb(py::cast<Real3>(min), self.GetMax());
          })
      .def_property(
          "max",
          &Aabb::GetMax,
          [](Aabb& self, py::object const& max) {
            self = Aabb(self.GetMin(), py::cast<Real3>(max));
          })
      .def("get_center", &Aabb::GetCenter)
      .def("get_size", &Aabb::GetSize)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__copy__", [](Aabb const& self) { return Aabb(self); })
      .def("__deepcopy__", [](Aabb const& self, py::dict) { return Aabb(self); }, py::arg("memo"));

  // Obb
  py::class_<Obb>(m, "Obb", "Oriented bounding box in 3D space.")
      .def(py::init<>())
      .def(
          py::init<TransformRT const&, Real3 const&>(),
          py::arg("transform"),
          py::arg("half_extents"))
      .def("__repr__", [](Obb const& self) { return ToPyReplString(self); })
      .def("__str__", [](Obb const& self) { return ToPyString(self); })
      .def("get_center", &Obb::GetCenter)
      .def("get_half_extents", &Obb::GetHalfExtents)
      .def("get_size", &Obb::GetSize)
      .def("__copy__", [](Obb const& self) { return Obb(self); })
      .def("__deepcopy__", [](Obb const& self, py::dict) { return Obb(self); }, py::arg("memo"));

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
      py::arg("num_worker_threads"),
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
      "Return whether the loaded native library uses double-precision floating-point values.");
  m.def(
      "uses_hdf5",
      []() { return bool(MOCHI_USE_HDF5); },
      "Return whether the loaded native library includes HDF5 support.");

  m.def(
      "normalize",
      static_cast<Real2 (*)(Real2 const&)>(&mochi::Normalize),
      py::arg("v"),
      "Return a unit-length copy of the vector. A zero vector remains zero.");
  m.def(
      "normalize",
      static_cast<Real3 (*)(Real3 const&)>(&mochi::Normalize),
      py::arg("v"),
      "Return a unit-length copy of the vector. A zero vector remains zero.");
  m.def(
      "normalize",
      static_cast<Quaternion (*)(Quaternion)>(&mochi::Normalize),
      py::arg("q"),
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
      py::arg("shape"));

  // This must come after the generated code because it references generated type LogChannel.
  m.def(
      "log",
      [](std::string const& message, LogChannel channel = LogChannel::Info) {
        py::gil_scoped_acquire gil;
        py::module inspect = py::module::import("inspect");
        py::object stack = inspect.attr("stack")();
        py::object frame_info = stack[py::int_(0)];
        std::string filename = py::str(frame_info[py::int_(1)]);
        auto lineno = py::cast<int>(frame_info[py::int_(2)]);
        MOCHI_LOG_IMPL(channel, filename.c_str(), lineno, "%s", message.c_str());
      },
      py::arg("message"),
      py::arg("channel") = LogChannel::Info);

  // Best-effort clean-up on module exit: tear down dependent contexts (bots/mpc) first,
  // then the physics Context. This is the single atexit for the whole context tree — bots
  // and mpc do not register their own, so ordering does not depend on atexit LIFO. Runs from
  // Python while the interpreter and GIL are alive (never from a C++ static destructor). If the
  // user already called shutdown(), there is no live context tree left to clean up; avoid running
  // dependent teardown callbacks again during interpreter shutdown, when extension unload order is
  // platform-sensitive.
  py::module_::import("atexit").attr("register")(py::cpp_function([]() {
    if (GetContext()) {
      mochi::SetLogCallback(nullptr);
      ShutdownGlobalContext();
    }
  }));

} // PYBIND11_MODULE
