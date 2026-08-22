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

// Re-use the physics module helpers (NdArray, Span, StringView, etc.).
#include <mochi_physics/pybind/src/pybind_dynamic_array.h>
#include <mochi_physics/pybind/src/pybind_helpers.h>
#include <mochi_physics/pybind/src/pybind_nd_array.h>
#include <mochi_physics/pybind/src/pybind_span.h>
#include <mochi_physics/pybind/src/pybind_string_view.h>

// Shared bindings-core context state (g_context, MochiErrorException, teardown registry).
#include <mochi_physics/pybind/core/pybind_core.h>

// Bots headers
#include <superdex_robotics/controllers/controller_basic_jsc_pd.h>
#include <superdex_robotics/core/context.h>
#include <superdex_robotics/sensors/camera_sensor.h>
#include <superdex_robotics/sensors/sensor_base.h>
#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/file_utils.h>

// Internal-only controllers. Included via the generic aggregator so shipped source
// never names the codename headers; dropped from open-source builds with the rest of internal/.
#if MOCHI_INTERNAL
#include <superdex_robotics/controllers/internal/controllers_internal.h>
#include <superdex_robotics/internal/bot_scene.h>
#include <superdex_robotics/internal/bot_task.h>
#include <superdex_robotics/internal/domain_randomization/derive.h>
#include <superdex_robotics/internal/domain_randomization/domain_randomization_episode.h>
#include <superdex_robotics/sensors/internal/sensors_internal.h>
#endif

// Module name for single- or double-precision
#if MOCHI_USE_DOUBLE_PRECISION
#define SUPERDEX_ROBOTICS_MODULE_NAME superdex_robotics_double
#else
#define SUPERDEX_ROBOTICS_MODULE_NAME superdex_robotics
#endif
