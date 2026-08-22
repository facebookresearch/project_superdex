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

#include <mochi_core/linear_algebra/krylov/ldlt_prec.h>

#include "exact_preconditioner_test_helpers.h"

#include <gtest/gtest.h>

using namespace mochi;

TEST(LDLtPrec, ExactPreconditioner) {
  test::TestExactPreconditioner<real, krylov::LDLtPrec<real>, true>();
}
