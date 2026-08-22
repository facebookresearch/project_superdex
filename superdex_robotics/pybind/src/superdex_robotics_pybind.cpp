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

// @not-generated: This file is maintained by hand. The generated code goes into the
// "generated" folder and is invoked via mochi::DefineAll.

#include "pybind_include.h"

#include <superdex_robotics/core/context.h>

#include <mochi_core/utils/error.h>
#include <mochi_core/utils/log.h>

#include <string>
#include <string_view>
#include <utility>

using namespace mochi;
namespace py = pybind11;

namespace {

/* Adapter that lets a component be defined entirely in Python. It derives from the C++ base for
 * its kind (ControllerBase / SensorBase / ActuatorBase) and forwards the small, fixed set of
 * framework-invoked virtuals to a wrapped Python instance. A component's own API
 * (SetParams/ComputeOutput/ComputeSignal/...) is heterogeneous per type and lives in Python;
 * callers drive it directly on the Python object retrieved via get_python_{controller,sensor,
 * actuator}, so it is intentionally not part of this C++ surface.
 *
 * Composition rather than a pybind11 trampoline: the RoboticsContext owns every component and
 * destroys it through ComponentBase::Destroy, so the Python object cannot be the owner, and the
 * bases are bound (generated) as [no_init, no_destruct] types that Python does not subclass. The
 * adapter keeps the C++ side authoritative and treats the Python instance as held state.
 *
 * Ownership: the RoboticsContext holds the sole authoritative lifetime. The adapter holds one
 * strong reference to the Python instance and drops it in its destructor. Component destruction is
 * funnelled through RoboticsContext teardown, which the physics module always runs from Python with
 * the interpreter and GIL alive (mochi.shutdown() or its atexit hook), so manipulating Python
 * refcounts here is safe. The GIL is acquired defensively anyway (reentrant when already held). */
template <typename BaseT>
class PythonComponent : public BaseT {
 public:
  /* @p baseArgs are forwarded to BaseT's constructor archetype, which differs per component kind
   * ((prefab, actor, error) for controllers, (actor, error) for sensors and actuators), so they
   * trail the arguments this adapter consumes itself. */
  template <typename... BaseArgs>
  PythonComponent(py::object impl, std::string typeName, BaseArgs&&... baseArgs)
      : BaseT(std::forward<BaseArgs>(baseArgs)...),
        _impl(std::move(impl)),
        _typeName(std::move(typeName)) {}

  /* Owned by the RoboticsContext and holds a strong Python reference; non-copyable, non-movable. */
  PythonComponent(PythonComponent const&) = delete;
  PythonComponent& operator=(PythonComponent const&) = delete;
  PythonComponent(PythonComponent&&) = delete;
  PythonComponent& operator=(PythonComponent&&) = delete;

  /* No hasattr guard: the registration factories reject an instance without `reset`, so the
   * attribute is guaranteed. A raising reset() propagates to the caller rather than being
   * swallowed -- resetting is experiment hygiene, and a silent failure is the thing worth
   * avoiding. */
  void Reset() override {
    py::gil_scoped_acquire gil;
    _impl.attr("reset")();
  }

  [[nodiscard]] std::string_view GetTypeName() const override {
    return _typeName;
  }

  [[nodiscard]] py::object const& Impl() const {
    return _impl;
  }

  ~PythonComponent() override {
    /* Never let an exception escape a destructor (bugprone-exception-escape). */
    try {
      py::gil_scoped_acquire gil;
      _impl = py::object();
    } catch (...) {
    }
  }

 protected:
  [[nodiscard]] std::string const& TypeNameString() const {
    return _typeName;
  }

 private:
  py::object _impl;
  std::string _typeName;
};

/* Controllers add one virtual the other kinds do not have: they are configured after construction,
 * because the scene entry that names a controller also carries its params. Sensors and actuators
 * receive their paramArgs as a constructor argument instead, so their adapters need nothing beyond
 * the shared template. */
class PythonController final : public PythonComponent<superdex::robotics::ControllerBase> {
 public:
  PythonController(
      py::object impl,
      std::string typeName,
      superdex::robotics::BotPrefab const* prefab,
      mochi::Actor* actor,
      mochi::Error& error)
      : PythonComponent(std::move(impl), std::move(typeName), prefab, actor, error) {}

  void ConfigureFromSceneEntry(
      std::string_view paramArgs,
      std::string_view initArgs,
      mochi::Error& error) override {
    MOCHI_ERROR_RETURN(error);
    py::gil_scoped_acquire gil;
    if (!py::hasattr(Impl(), "configure_from_scene_entry")) {
      return;
    }
    /* The Python detail cannot be stored in Error (it holds a borrowed char const*), so log it and
     * set a static message the scene loader can surface. */
    try {
      Impl().attr("configure_from_scene_entry")(std::string(paramArgs), std::string(initArgs));
    } catch (py::error_already_set& e) {
      MOCHI_LOG_ERROR(
          "Python controller '%s' configure_from_scene_entry raised an exception: %s",
          TypeNameString().c_str(),
          e.what());
      MOCHI_ERROR_SET(
          error, "Python controller configure_from_scene_entry raised an exception (see log).");
    }
  }
};

using PythonSensor = PythonComponent<superdex::robotics::SensorBase>;
using PythonActuator = PythonComponent<superdex::robotics::ActuatorBase>;

/* Every component must implement reset(), matching the pure-virtual Reset() the C++ bases
 * require. A component is reset between episodes or trials, so a stateful one that quietly
 * inherited a no-op would carry stale state into the next run and only show it in the collected
 * data. Checked here, at creation, so the failure lands at scene load rather than mid-collection.
 * A component with no per-episode state writes `def reset(self): pass`. */
bool RequireResetMethod(py::object const& impl, std::string const& typeName, mochi::Error& error) {
  if (py::hasattr(impl, "reset")) {
    return true;
  }
  MOCHI_LOG_ERROR(
      "Python component type '%s' does not define reset(); every component must, even if the body "
      "is empty.",
      typeName.c_str());
  MOCHI_ERROR_SET(error, "Python component type does not define reset() (see log).");
  return false;
}

/* Build the (Actor*, paramArgs, Error&) factory that RegisterSensorType / RegisterActuatorType
 * take. The Python factory is called with (actor, param_args) -- unlike the controller factory,
 * which takes only the actor because a controller's params arrive later via
 * ConfigureFromSceneEntry. `actor` is None for an actor-less sensor. */
template <typename PythonComponentT>
auto MakePythonComponentFactory(std::string typeName, py::object factory) {
  return [factory = std::move(factory), typeName = std::move(typeName)](
             mochi::Actor* actor,
             std::string_view paramArgs,
             mochi::Error& error) -> PythonComponentT* {
    py::gil_scoped_acquire gil;
    try {
      py::object impl =
          factory(py::cast(actor, py::return_value_policy::reference), std::string(paramArgs));
      if (!RequireResetMethod(impl, typeName, error)) {
        return nullptr;
      }
      auto* component = new PythonComponentT(std::move(impl), typeName, actor, error);
      if (!error.IsOK()) {
        superdex::robotics::ComponentBase::Destroy(component);
        return nullptr;
      }
      return component;
    } catch (py::error_already_set& e) {
      MOCHI_LOG_ERROR(
          "Python factory for type '%s' raised an exception: %s", typeName.c_str(), e.what());
      MOCHI_ERROR_SET(error, "Python component factory raised an exception (see log).");
      return nullptr;
    }
  };
}

/* The live Python instance behind a handle, or None when the handle does not name a
 * Python-implemented component of that kind (an invalid handle, or a C++ implementation). */
template <typename PythonComponentT, typename ComponentT>
py::object PythonImplOf(ComponentT* component) {
  auto* const pythonComponent = dynamic_cast<PythonComponentT*>(component);
  if (pythonComponent == nullptr) {
    return py::none();
  }
  return pythonComponent->Impl();
}

} // namespace

PYBIND11_MODULE(SUPERDEX_ROBOTICS_MODULE_NAME, m) {
  // Import the physics module so that shared types (Real3, ShapeHandle, …) are
  // already registered in pybind11's global type registry, and so the shared pybind-core
  // library (which owns g_context and the MochiErrorException translator) is loaded.
  py::module_::import(MOCHI_PHYSICS_MODULE_NAME_STR);

  // Insert generated bindings
  mochi::DefineAll(m);

  // RoboticsContext lifecycle — defined on the "bots" submodule to preserve the
  // mochi.bots.create_context() access pattern from the old monolithic module.
  //
  // A function-local static (not an inline global): the bots generated bindings take the context as
  // an explicit argument (e.g. load_bot_scene(path, bots_ctx)), so nothing outside this entry point
  // needs it. Static storage means the create_context and teardown lambdas below reference it
  // without capturing, and it stays valid when the teardown runs after this function returns (at
  // shutdown/atexit).
  static superdex::robotics::RoboticsContext* g_botsContext = nullptr;
  auto m_bots = m.attr("bots").cast<py::module_>();

  m_bots.def(
      "create_context",
      []() -> superdex::robotics::RoboticsContext* {
        // The physics Context lives in the shared pybind-core library.
        mochi::CheckContext();
        if (!g_botsContext) {
          g_botsContext = superdex::robotics::CreateRoboticsContext();
        }
        return g_botsContext;
      },
      py::return_value_policy::reference,
      "Create a RoboticsContext. Returns the existing one if already created.");

  // Register a pure-C++ teardown with the shared pybind-core library: the physics module runs
  // it right before destroying the Context (during shutdown() or atexit), so the bots context
  // is always destroyed while the physics Context is still alive, regardless of atexit ordering.
  // Empty capture: g_botsContext has static storage. Null-guarded and idempotent, so it is safe
  // to run when no context was created or after a prior teardown.
  mochi::RegisterContextDependent([]() {
    if (g_botsContext) {
      superdex::robotics::DestroyRoboticsContext(g_botsContext);
      g_botsContext = nullptr;
    }
  });

  /* Pure-Python components.
   *
   * register_python_{controller,sensor,actuator} let a component be defined entirely in Python and
   * created by type string -- directly, from a bot prefab, or from a scene file -- with no C++
   * implementation. The Python `factory` must return the Python instance; the returned object is
   * wrapped in a C++ adapter that the RoboticsContext owns and that forwards the framework-invoked
   * virtuals to it. Retrieve the live Python instance (to drive its own API) with
   * get_python_{controller,sensor,actuator}.
   *
   * In each case the factory std::function captures the Python `factory` by value; RegisterXType
   * moves it into the registry under the GIL (these defs run from Python), and it is dropped at
   * RoboticsContext teardown, also under the GIL. Those are the only points the captured object is
   * (in/de)cref'd, so no explicit GIL guard is needed around them. */
  m_bots.def(
      "register_python_controller",
      [](superdex::robotics::RoboticsContext* botsCtx,
         std::string const& typeName,
         py::object factory) {
        botsCtx->RegisterControllerType(
            typeName,
            [factory = std::move(factory), typeName](
                superdex::robotics::BotPrefab const* prefab,
                mochi::Actor* actor,
                mochi::Error& error) -> superdex::robotics::ControllerBase* {
              py::gil_scoped_acquire gil;
              try {
                py::object impl = factory(py::cast(actor, py::return_value_policy::reference));
                if (!RequireResetMethod(impl, typeName, error)) {
                  return nullptr;
                }
                auto* controller =
                    new PythonController(std::move(impl), typeName, prefab, actor, error);
                if (!error.IsOK()) {
                  superdex::robotics::ComponentBase::Destroy(controller);
                  return nullptr;
                }
                return controller;
              } catch (py::error_already_set& e) {
                MOCHI_LOG_ERROR(
                    "Python factory for type '%s' raised an exception: %s",
                    typeName.c_str(),
                    e.what());
                MOCHI_ERROR_SET(error, "Python controller factory raised an exception (see log).");
                return nullptr;
              }
            });
      },
      py::arg("bots_ctx"),
      py::arg("type_name"),
      py::arg("factory"),
      "Register a controller type implemented in Python. `factory` is called with the robot Actor "
      "(None if the controller was created without one) and must return a Python controller "
      "instance. The instance must define `reset()` (an empty body is fine); its optional "
      "`configure_from_scene_entry(param_args, init_args)` is also invoked by the framework. "
      "Retrieve the live instance to drive it with "
      "get_python_controller.");

  m_bots.def(
      "register_python_sensor",
      [](superdex::robotics::RoboticsContext* botsCtx,
         std::string const& typeName,
         py::object factory) {
        botsCtx->RegisterSensorType(
            typeName, MakePythonComponentFactory<PythonSensor>(typeName, std::move(factory)));
      },
      py::arg("bots_ctx"),
      py::arg("type_name"),
      py::arg("factory"),
      "Register a sensor type implemented in Python. `factory` is called with (actor, param_args) "
      "-- the link Actor, or None for an actor-less sensor, and the sensor's params (a file path "
      "or inline JSON, empty for defaults) -- and must return a Python sensor instance, which must "
      "define `reset()` (an empty body is fine) -- the framework invokes it between episodes. "
      "Retrieve the live instance to drive it with get_python_sensor.");

  m_bots.def(
      "register_python_actuator",
      [](superdex::robotics::RoboticsContext* botsCtx,
         std::string const& typeName,
         py::object factory) {
        botsCtx->RegisterActuatorType(
            typeName, MakePythonComponentFactory<PythonActuator>(typeName, std::move(factory)));
      },
      py::arg("bots_ctx"),
      py::arg("type_name"),
      py::arg("factory"),
      "Register an actuator type implemented in Python. `factory` is called with (actor, "
      "param_args) -- the link Actor, which an actuator always has, and the actuator's params (a "
      "file path or inline JSON, empty for defaults) -- and must return a Python actuator "
      "instance, which must define `reset()` (an empty body is fine) -- the framework invokes it "
      "between episodes. Retrieve the live instance to drive it with get_python_actuator.");

  m_bots.def(
      "get_python_controller",
      [](superdex::robotics::RoboticsContext* botsCtx, superdex::robotics::ControllerHandle handle)
          -> py::object { return PythonImplOf<PythonController>(botsCtx->GetController(handle)); },
      py::arg("bots_ctx"),
      py::arg("handle"),
      "Return the Python controller instance behind a handle (created from a type registered via "
      "register_python_controller), or None if the handle is not a Python controller.");

  m_bots.def(
      "get_python_sensor",
      [](superdex::robotics::RoboticsContext* botsCtx, superdex::robotics::SensorHandle handle)
          -> py::object { return PythonImplOf<PythonSensor>(botsCtx->GetSensor(handle)); },
      py::arg("bots_ctx"),
      py::arg("handle"),
      "Return the Python sensor instance behind a handle (created from a type registered via "
      "register_python_sensor), or None if the handle is not a Python sensor.");

  m_bots.def(
      "get_python_actuator",
      [](superdex::robotics::RoboticsContext* botsCtx, superdex::robotics::ActuatorHandle handle)
          -> py::object { return PythonImplOf<PythonActuator>(botsCtx->GetActuator(handle)); },
      py::arg("bots_ctx"),
      py::arg("handle"),
      "Return the Python actuator instance behind a handle (created from a type registered via "
      "register_python_actuator), or None if the handle is not a Python actuator.");

} // PYBIND11_MODULE
