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

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mochi {
namespace {

// The dependent-teardown registry is process-global and intentionally never cleared, so each
// test captures its own heap-allocated event log by value (shared_ptr). Callbacks left
// registered by earlier tests re-run on every RunContextDependentTeardowns() call, but they
// append to their own (still-alive) logs, so a test only ever observes its own callbacks.

TEST(PybindContextTeardownRegistry, RunsTeardownsInReverseRegistrationOrder) {
  auto events = std::make_shared<std::vector<std::string>>();
  RegisterContextDependent([events]() { events->push_back("a"); });
  RegisterContextDependent([events]() { events->push_back("b"); });

  RunContextDependentTeardowns();

  // Most-recently-registered first.
  EXPECT_EQ(*events, (std::vector<std::string>{"b", "a"}));
}

TEST(PybindContextTeardownRegistry, ThrowingTeardownDoesNotBlockOthers) {
  auto events = std::make_shared<std::vector<std::string>>();
  RegisterContextDependent([events]() { events->push_back("ok"); });
  // Registered last, so it runs first; its exception must not prevent the earlier teardown.
  RegisterContextDependent([]() { throw std::runtime_error("boom"); });

  RunContextDependentTeardowns();

  EXPECT_EQ(*events, (std::vector<std::string>{"ok"}));
}

} // namespace
} // namespace mochi
