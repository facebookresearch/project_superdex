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

#include "material_test_helpers.h"

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/vmatrix.h>

using namespace mochi;

Matrix<real, 3, 3> mochi::ComputeDmInv(
    ColumnVector<real, 3> const& V0,
    ColumnVector<real, 3> const& V1,
    ColumnVector<real, 3> const& V2,
    ColumnVector<real, 3> const& V3) {
  Matrix<real, 3, 3> dm;
  dm.Col(0) = V1 - V0;
  dm.Col(1) = V2 - V0;
  dm.Col(2) = V3 - V0;
  return Inverse(dm);
}

Matrix<real, 3, 3> mochi::ComputeF(
    ColumnVector<real, 3> const& v0,
    ColumnVector<real, 3> const& v1,
    ColumnVector<real, 3> const& v2,
    ColumnVector<real, 3> const& v3,
    Matrix<real, 3, 3> const& dmInv) {
  Matrix<real, 3, 3> ds;
  ds.Col(0) = v1 - v0;
  ds.Col(1) = v2 - v0;
  ds.Col(2) = v3 - v0;
  return ds * dmInv;
}

VMatrix3x3r mochi::ToSimdMatrix(RowMatrix<real, 3, 3> const& mat) {
  VMatrix3x3r out;
  LoadMatrix<3, 3>(out, mat.Data());
  return out;
}
