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

#ifndef MOCHI_PHYSICS_H
#define MOCHI_PHYSICS_H

#ifdef __cplusplus

#include "mochi_physics_config.h"

// C++ API
#include "cpp_api/mochi_actor.h"
#include "cpp_api/mochi_async_scene.h"
#include "cpp_api/mochi_constraint.h"
#include "cpp_api/mochi_context.h"
#include "cpp_api/mochi_debug_draw.h"
#include "cpp_api/mochi_debug_server.h"
#include "cpp_api/mochi_enums.h"
#include "cpp_api/mochi_handle.h"
#include "cpp_api/mochi_scene.h"
#include "cpp_api/mochi_structs.h"

// Optionally include reflection support for API types
#ifdef MOCHI_USE_REFLECTION
#if MOCHI_USE_REFLECTION
#include "utils/mochi_physics_reflection.generated.h"
#endif
#endif

#else

// The C API is coming soon.

#endif

#endif // MOCHI_PHYSICS_H
