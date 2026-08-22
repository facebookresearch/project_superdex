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

/**************************************************************************************************
  Experimental Feature Availability
    - MOCHI_ENABLE_ROM_ACTORS         Enables ROM shape creation and actor creation. ROM-bearing
                                      shapes may still be loaded from assets when disabled.
    - MOCHI_ENABLE_DEEP_FLOW_ACTORS   Enables Deep Flow shape creation and actor creation.

  These features default to MOCHI_INTERNAL.
*/

#ifndef MOCHI_ENABLE_ROM_ACTORS
#define MOCHI_ENABLE_ROM_ACTORS MOCHI_INTERNAL
#endif

#ifndef MOCHI_ENABLE_DEEP_FLOW_ACTORS
#define MOCHI_ENABLE_DEEP_FLOW_ACTORS MOCHI_INTERNAL
#endif
