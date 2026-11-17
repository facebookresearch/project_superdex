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

#include <iostream>
#include <string>

// Internal to the simple_reflection library. Not installed with the public headers.
namespace SReflect::detail {

std::string Format(char const* format, ...);

} // namespace SReflect::detail

// Logging is enabled by default
#ifndef SR_LOG_ENABLE
#define SR_LOG_ENABLE 1
#endif
#if SR_LOG_ENABLE
// Log using std::cout because it is easier to intercept/redirect if people want to do that.
// Every line ends with std::endl, which causes a flush which is considered to be appropriate
// because this macro is generally used for errors/warnings which should appear immediately.
#define SR_LOG(...) \
  std::cout << "[Simple Reflection] " << ::SReflect::detail::Format(__VA_ARGS__) << std::endl;
#else
#define SR_LOG(...)
#endif
