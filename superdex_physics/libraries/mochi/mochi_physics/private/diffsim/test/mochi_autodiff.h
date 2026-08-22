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
#include <gtest/gtest.h>
#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/src/mochi_rigid.h>

#if MOCHI_USE_EIGEN
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

namespace mochi::autodiff {

using Vec = Eigen::Matrix<mochi::real, Eigen::Dynamic, 1>;
using Mat = Eigen::Matrix<mochi::real, Eigen::Dynamic, Eigen::Dynamic>;
using Vec3 = Eigen::Matrix<mochi::real, 3, 1>;
using Mat3 = Eigen::Matrix<mochi::real, 3, 3>;
using Mat3x4 = Eigen::Matrix<mochi::real, 3, 4>;
using AD1 = Eigen::AutoDiffScalar<Eigen::Matrix<mochi::real, -1, 1>>;
using AD2 = Eigen::AutoDiffScalar<Eigen::Matrix<AD1, -1, 1>>;
using VecAD1 = Eigen::Matrix<AD1, Eigen::Dynamic, 1>;
using MatAD1 = Eigen::Matrix<AD1, Eigen::Dynamic, Eigen::Dynamic>;
using VecAD2 = Eigen::Matrix<AD2, Eigen::Dynamic, 1>;
using MatAD2 = Eigen::Matrix<AD2, Eigen::Dynamic, Eigen::Dynamic>;
using Vec3AD1 = Eigen::Matrix<AD1, 3, 1>;
using Mat3AD1 = Eigen::Matrix<AD1, 3, 3>;
using Vec3AD2 = Eigen::Matrix<AD2, 3, 1>;
using Mat3AD2 = Eigen::Matrix<AD2, 3, 3>;
using Mat3x4AD1 = Eigen::Matrix<AD1, 3, 4>;
using Mat3x4AD2 = Eigen::Matrix<AD2, 3, 4>;

// Skew-symmetric matrix
template <typename T>
Eigen::Matrix<T, 3, 3> Skew3(Eigen::Matrix<T, 3, 1> const& v) {
  Eigen::Matrix<T, 3, 3> result;
  result.setZero();
  result(2, 1) = v[0];
  result(1, 2) = -v[0];
  result(0, 2) = v[1];
  result(2, 0) = -v[1];
  result(1, 0) = v[2];
  result(0, 1) = -v[2];
  return result;
}

class AutoDiffAssembly {
 public:
  AutoDiffAssembly(Scene* scene, std::array<StateHandle, 3> states);

  void Assemble(mochi::real dt);
  mochi::real GetMerit() const;
  ColumnVector<mochi::real> GetRes() const;
  Matrix<mochi::real> GetDRes(int d) const;

 protected:
  Scene* _scene = nullptr;
  std::array<StateHandle, 3> _states;
  int _n = -1;
  VecAD2 _deltaState;
  AD2 _merits;
  Vec _grad;
  Mat _hess;
  NdArray<DynamicArray<Mat3x4AD2>, 3> _transforms;
};

} // namespace mochi::autodiff

#endif
