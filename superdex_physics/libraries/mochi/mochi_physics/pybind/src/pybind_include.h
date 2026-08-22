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

// NOTE: This file is included by mochi_physics_pybind.generated.h, so that every generated cpp file
// includes this list of headers.

#include "pybind_dynamic_array.h"
#include "pybind_helpers.h"
#include "pybind_span.h"
#include "pybind_string_view.h"

#include <mochi_core/geometry/model_utils.h>
#include <mochi_core/utils/coordinate_space.h>
#include <mochi_physics/diffsim/mochi_diffsim.h>
#include <mochi_physics/pybind/core/pybind_core.h>
