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

#include <pybind11/pybind11.h>

#include <functional>
#include <stdexcept>

#include <mochi_physics/mochi_physics.h>

namespace mochi {

// Exception to throw when mochi::Error is not OK.
//
// PYBIND11_EXPORT gives the type default (exported) visibility so its RTTI is unified across
// extension `.so`/`.dylib` boundaries. Combined with the out-of-line key function
// (~MochiErrorException, anchored in pybind_core.cpp), this makes the single exception
// translator registered by the physics module catch throws originating in the bots/mpc
// extensions — without per-module re-registration, which was previously needed because each
// `.so` otherwise had its own distinct RTTI for this type (notably on macOS).
class PYBIND11_EXPORT MochiErrorException : public std::runtime_error {
 public:
  explicit MochiErrorException(Error const& e) : std::runtime_error(e.ToString()) {}

  // Out-of-line key function: anchors the vtable and typeinfo to pybind_core.cpp.
  ~MochiErrorException() override;
};

// Creates the global context. Throws if one already exists.
PYBIND11_EXPORT void InitGlobalContext(int numWorkerThreads);

// Returns the global context pointer. May be null before initialization.
PYBIND11_EXPORT Context* GetContext();

// Destroys the global context. Throws if none exists.
PYBIND11_EXPORT void DestroyGlobalContext();

// Throws if the global context has not been initialized.
PYBIND11_EXPORT void CheckContext();

// --- Ordered teardown registry for dependent contexts (bots, mpc) -------------------------
// bots/mpc build their contexts against the physics Context returned by GetContext(), and
// their object destructors may touch it. So every dependent context MUST be destroyed BEFORE the
// physics Context. Dependent modules register a pure-C++ teardown at import time;
// ShutdownGlobalContext runs them (reverse registration order) immediately before destroying the
// global physics Context.
//
// Teardown is always triggered from Python (mochi.shutdown() or a single atexit) while the
// interpreter and GIL are alive — never from a C++ static destructor.

// Registers a teardown to run (in reverse registration order) immediately before the global
// context is destroyed. Callbacks must be null-guarded and idempotent: the registry is never
// cleared, so a context re-created after shutdown()+initialize() is torn down again next time.
PYBIND11_EXPORT void RegisterContextDependent(std::function<void()> teardown);

// Runs the registered dependent teardowns, most-recently-registered first. Best-effort: a
// throwing teardown is logged and does not block the remaining teardowns.
PYBIND11_EXPORT void RunContextDependentTeardowns();

// Runs the dependent teardowns, then destroys the global context. This is the single entry
// point for mochi.shutdown() and the physics atexit handler.
PYBIND11_EXPORT void ShutdownGlobalContext();

} // namespace mochi
