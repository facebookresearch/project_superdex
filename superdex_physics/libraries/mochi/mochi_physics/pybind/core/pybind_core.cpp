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

#include "pybind_core.h"

#include <mochi_core/utils/log.h>

#include <exception>
#include <utility>
#include <vector>

namespace mochi {

// Out-of-line key function: anchors MochiErrorException's vtable and typeinfo to this TU, so
// the exported RTTI lives in the shared pybind-core library rather than being duplicated per
// extension.
MochiErrorException::~MochiErrorException() = default;

namespace {
Context* g_context = nullptr;
} // namespace

void InitGlobalContext(int numWorkerThreads) {
  if (g_context) {
    throw std::runtime_error("Mochi has already been initialized.");
  }
  g_context = mochi::CreateContext(numWorkerThreads);
}

Context* GetContext() {
  return g_context;
}

void DestroyGlobalContext() {
  if (!g_context) {
    throw std::runtime_error("Mochi is not currently initialized.");
  }
  // Before destroying the context, clear the log callback. It may hold a reference to a
  // Python function that the user didn't release; otherwise, the interpreter can crash on exit
  // with: "Fatal Python error: gilstate_tss_set: failed to set current tstate (TSS)".
  g_context->SetLogCallback(nullptr);
  mochi::DestroyContext(g_context);
  g_context = nullptr;
}

void CheckContext() {
  if (!g_context) {
    throw std::runtime_error("Please call mochi.initialize(num_worker_threads) first.");
  }
}

namespace {
// Function-local static so registration works regardless of static-initialization order across
// the shared-library boundary. Intentionally never cleared (see RegisterContextDependent).
std::vector<std::function<void()>>& ContextDependents() {
  static std::vector<std::function<void()>> dependents;
  return dependents;
}
} // namespace

void RegisterContextDependent(std::function<void()> teardown) {
  ContextDependents().push_back(std::move(teardown));
}

void RunContextDependentTeardowns() {
  auto& dependents = ContextDependents();
  for (auto it = dependents.rbegin(); it != dependents.rend(); ++it) {
    try {
      (*it)();
    } catch (std::exception const& e) {
      MOCHI_LOG_ERROR("A context-dependent teardown raised an exception: %s", e.what());
    } catch (...) {
      MOCHI_LOG_ERROR("A context-dependent teardown raised an unknown exception.");
    }
  }
}

void ShutdownGlobalContext() {
  RunContextDependentTeardowns();
  DestroyGlobalContext();
}

} // namespace mochi
