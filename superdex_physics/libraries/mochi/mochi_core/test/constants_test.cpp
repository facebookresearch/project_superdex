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

#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/constants.h>

#include <limits>

using namespace mochi;

static_assert(NearEqual(kPI, Sqr(kSqrtPi), 2_r * std::numeric_limits<real>::epsilon()));
static_assert(NearEqual(2_r, Sqr(kSqrt2), 2_r * std::numeric_limits<real>::epsilon()));
static_assert(NearEqual(3_r, Sqr(2_r * kSqrt3Over2), 2_r * std::numeric_limits<real>::epsilon()));
