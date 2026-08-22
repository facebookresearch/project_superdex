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

#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/log.h>

namespace mochi::test {

/**
 * @brief Disable logging on the specified channel for the duration of the scope.
 *
 * @param channel The channel to disable.
 * @return A Defer object that will restore the state when it is destroyed.
 */
inline auto SuppressLogChannel(LogChannel channel) {
  bool const wasEnabled = IsLogChannelEnabled(channel);
  EnableLogChannel(channel, false);
  return mochi::Defer([=]() { EnableLogChannel(channel, wasEnabled); });
}

/** @brief Disable LogChannel::Error for the duration of the scope. */
inline auto SuppressLogError() {
  return SuppressLogChannel(LogChannel::Error);
}

/** @brief Disable LogChannel::Warning for the duration of the scope. */
inline auto SuppressLogWarning() {
  return SuppressLogChannel(LogChannel::Warning);
}

} // namespace mochi::test
