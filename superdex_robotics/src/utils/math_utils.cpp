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

#include <superdex_robotics/utils/math_utils.h>

#include <cmath>

using namespace mochi;
using namespace superdex::robotics;

Quaternion superdex::robotics::QuaternionFromRPY(Real3 const& rpy) {
  auto roll = Quaternion::RotationX(rpy[0]);
  auto pitch = Quaternion::RotationY(rpy[1]);
  auto yaw = Quaternion::RotationZ(rpy[2]);
  return yaw * pitch * roll;
}

Real3 superdex::robotics::RPYFromQuaternion(Quaternion const& q) {
  Real4 const v = q.ToReal4();
  real const x = v[0];
  real const y = v[1];
  real const z = v[2];
  real const w = v[3];

  real const sqx = x * x;
  real const sqy = y * y;
  real const sqz = z * z;
  real const sqw = w * w;

  // Sine of pitch
  real const sinPitch = -2_r * (x * z - w * y);

  // Handle gimbal lock
  constexpr real kGimbalLockThreshold = 0.99999_r;
  constexpr real kPiOver2 = 1.57079632679489661923_r;

  if (sinPitch <= -kGimbalLockThreshold) {
    return Real3{0_r, -kPiOver2, 2_r * std::atan2(x, -y)};
  }
  if (sinPitch >= kGimbalLockThreshold) {
    return Real3{0_r, kPiOver2, 2_r * std::atan2(-x, y)};
  }

  return Real3{
      std::atan2(2_r * (y * z + w * x), sqw - sqx - sqy + sqz), // roll
      std::asin(sinPitch), // pitch
      std::atan2(2_r * (x * y + w * z), sqw + sqx - sqy - sqz) // yaw
  };
}

real superdex::robotics::SafeDot(Real3 const& a, Real3 const& b) {
  real result = 0_r;
  for (int k = 0; k < 3; ++k) {
    if (a[k] != 0_r && b[k] != 0_r) {
      result += a[k] * b[k];
    }
  }
  return result;
}

Real3 superdex::robotics::SafeScale(Real3 const& v, real scalar) {
  return {
      v[0] != 0_r ? v[0] * scalar : 0_r,
      v[1] != 0_r ? v[1] * scalar : 0_r,
      v[2] != 0_r ? v[2] * scalar : 0_r,
  };
}
