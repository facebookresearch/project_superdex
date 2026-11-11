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

// zenoh-pico's atomic.c, compiled as C++ (std::atomic) because MSVC only has C11 <stdatomic.h>
// behind a compiler flag modules can't set. Declaring atomic.h extern "C" first gives the
// definitions C linkage, matching the rest of zenoh-pico.

#include "HAL/Platform.h"

#include <atomic>

THIRD_PARTY_INCLUDES_START
extern "C" {
#include "zenoh-pico/collections/atomic.h"
}

#include "atomic.c.inl"
THIRD_PARTY_INCLUDES_END
