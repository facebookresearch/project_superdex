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
// PLEASE DO NOT ADD OTHER INCLUDES HERE. This header is included in the mochi_physics public API.

namespace mochi {
enum class NeuralComputeType { MochiCpu = 0, TorchCpu = 1, TorchGpu = 2, Count = 3 };
} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::NeuralComputeType);
MOCHI_ENUM_ITEM(MochiCpu);
MOCHI_ENUM_ITEM(TorchCpu);
MOCHI_ENUM_ITEM(TorchGpu);
MOCHI_ENUM_COUNT(Count);
MOCHI_ENUM_END();
