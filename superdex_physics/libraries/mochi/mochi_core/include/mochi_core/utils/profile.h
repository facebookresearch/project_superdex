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

#include <mochi_core/mochi_config.h>

#include <string_view>

namespace mochi {

// Place at the top of a function or {} block to measure its performance.
#define MOCHI_PROFILE_SCOPE() IMPL_MOCHI_PROFILE_SCOPE()

// MOCHI_PROFILE_SCOPE with a custom name (must be string literal).
// Useful for lambdas and other {} blocks within a function.
#define MOCHI_PROFILE_SCOPE_N(nameStringLiteral) IMPL_MOCHI_PROFILE_N(nameStringLiteral)

// Change the name of the current MOCHI_PROFILE_SCOPE, as seen in the profiler's timeline view.
// Converts to std::string_view and copies the bytes (does not need to be string literal).
#define MOCHI_PROFILE_LABEL(name) IMPL_MOCHI_PROFILE_LABEL(name)

// Like MOCHI_PROFILE_LABEL, but it joins to strings to form the name.
#define MOCHI_PROFILE_LABEL_2(prefix, suffix) IMPL_MOCHI_PROFILE_LABEL_2(prefix, suffix)

// Change the description text of the current MOCHI_PROFILE_SCOPE. Visible in the profiler when you
// click on the scope. Converts to std::string_view and copies the data (does not need to be a
// string literal).
#define MOCHI_PROFILE_DESCRIPTION(str) IMPL_MOCHI_PROFILE_DESCRIPTION(str)

// Like MOCHI_PROFILE_DESCRIPTION, but it takes a printf-style formatted string + arguments
#define MOCHI_PROFILE_DESCRIPTION_F(fmt, ...) IMPL_MOCHI_PROFILE_DESCRIPTION_F(fmt, __VA_ARGS__)

// Place at the end of the simulation step.
#define MOCHI_PROFILE_END_FRAME() IMPL_MOCHI_PROFILE_END_FRAME()

// Call once at the start of the application
void ProfilerInitialize();

// Call once at the end of the application
void ProfilerShutdown();

// Return true if ProfilerInitialize() has been called successfully
bool ProfilerIsInitialized();

// Return true if a profiler tool is currently connected and recording data
bool ProfilerIsConnected();

} // namespace mochi

#include "profile_inl.h"
