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

#include <mochi_core/utils/reflection.h>

#if MOCHI_USE_REFLECTION

#include <mochi_core/utils/log.h>

#include <string>
#include <string_view>

namespace mochi {

inline void LogDeserializationIssues(
    LogChannel channel,
    std::string_view prefix,
    SReflect::IssueList const& issues) {
  if (issues.empty()) {
    return;
  }

  std::string message{prefix};
  for (SReflect::IssueReport const& issue : issues) {
    message += "\n  ";
    message += issue.ToString();
  }
  MOCHI_LOG_TO(channel, "%s", message.c_str());
}

inline void LogDeserializationWarnings(std::string_view prefix, SReflect::IssueList const& issues) {
  LogDeserializationIssues(LogChannel::Warning, prefix, issues);
}

} // namespace mochi

#endif // MOCHI_USE_REFLECTION
