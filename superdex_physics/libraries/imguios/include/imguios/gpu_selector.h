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

#if defined(_MSC_VER)
#define GPUSELECTOR_API _declspec(dllexport)
#else
#define GPUSELECTOR_API __attribute__((visibility("default")))
#endif

// Magic variables used to select integrated vs discrete gpu

// http://developer.download.nvidia.com/devzone/devcenter/gamegraphics/files/OptimusRenderingPolicies.pdf
extern "C" GPUSELECTOR_API const unsigned long NvOptimusEnablement = 1;

// http://developer.amd.com/community/blog/2015/10/02/amd-enduro-system-for-developers/
extern "C" GPUSELECTOR_API const unsigned long AmdPowerXpressRequestHighPerformance = 1;
