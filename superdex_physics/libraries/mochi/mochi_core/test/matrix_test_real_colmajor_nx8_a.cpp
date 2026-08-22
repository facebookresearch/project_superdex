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

/**
 * @file matrix_test_real_colmajor_nx8_a.cpp
 * @brief Instantiates the TestMatrixNx8_A test from matrix_test.h for the @ref real scalar
 * type with col-major storage order.
 *
 * @note Tests of matrix_test.h specializations are divided across multiple files to reduce build
 * time.
 */

#include "matrix_test.h"

using namespace mochi;

TEST(Matrix, Real_ColMajor_Nx8_A) {
  test::TestMatrixNx8_A<real, krylov::Direction::ColMajor>();
}
