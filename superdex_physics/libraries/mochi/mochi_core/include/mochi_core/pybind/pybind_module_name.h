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
#include <mochi_core/mochi_platform.h>

#if MOCHI_USE_DOUBLE_PRECISION
#define MOCHI_PYBIND_MODULE_NAME_PRECISION(name) MOCHI_PP_CAT(name, _double)
#else
#define MOCHI_PYBIND_MODULE_NAME_PRECISION(name) name
#endif

#if MOCHI_ARCH_X64_AVX512
#define MOCHI_PYBIND_MODULE_NAME_ISA(name) MOCHI_PP_CAT(name, _avx512)
#else
#define MOCHI_PYBIND_MODULE_NAME_ISA(name) name
#endif

#define MOCHI_PYBIND_MODULE_NAME(name) \
  MOCHI_PYBIND_MODULE_NAME_ISA(MOCHI_PYBIND_MODULE_NAME_PRECISION(name))
