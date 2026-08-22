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

#include <chrono>
#include <thread>

#include <gtest/gtest.h>

namespace mochi::test {

/**
 * @brief Block the calling thread until the predicate returns true, or the timeout is reached.
 *
 * @param pred Predicate function or lambda. Takes no arguments and returns bool.
 * @param timeoutSec Maximum wait time [s]
 * @param failOnTimeout If true, then a timeout will result in a test failure.
 *
 * @return True if the predicate returned true. False on timeout.
 */
template <typename Predicate>
bool WaitUntil(Predicate&& pred, float timeoutSec = 60.0f, bool failOnTimeout = true) {
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::duration<float>(timeoutSec);
  while (std::chrono::steady_clock::now() < deadline) {
    if (pred()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (failOnTimeout) {
    ADD_FAILURE() << "WaitUntil timed out after " << timeoutSec << "s";
  }
  return false;
}

} // namespace mochi::test
