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

#include <superdex_physics.h>
#include <superdex_robotics/superdex_robotics.h>

namespace superdex::robotics {

/**
 * @brief Construct a @ref Quaternion from roll-pitch-yaw (RPY) Euler angles.
 *
 * @param[in] rpy Roll (X), pitch (Y), and yaw (Z) angles [rad].
 * @return The resulting @ref Quaternion (ZYX intrinsic rotation order).
 *
 * @note This function only exists to make editing rotations in SuperDex Studio user-friendly and
 * to resolve imported RPY rotations (e.g. from URDF). Prefer quaternion math for all other
 * applications.
 */
[[nodiscard]] MOCHI_API Quaternion QuaternionFromRPY(Real3 const& rpy);

/**
 * @brief Extract roll-pitch-yaw (RPY) Euler angles from a @ref Quaternion.
 *
 * @param[in] q The quaternion to decompose.
 * @return Roll (X), pitch (Y), and yaw (Z) angles [rad].
 *
 * @note This function only exists to make editing rotations in SuperDex Studio user-friendly and
 * to resolve imported RPY rotations (e.g. from URDF). Prefer quaternion math for all other
 * applications.
 */
[[nodiscard]] MOCHI_API Real3 RPYFromQuaternion(Quaternion const& q);

/**
 * @brief Dot product that treats 0 * inf as 0, avoiding IEEE 754 NaN.
 *
 * @details Standard @ref Dot produces NaN when one operand has ±inf in a component where the other
 * operand is zero (e.g. Dot({-inf, 0, 0}, {0, 0, 1}) = NaN + 0 + 0 = NaN). This variant skips
 * terms where either factor is zero.
 *
 * @param[in] a First operand.
 * @param[in] b Second operand.
 * @return Dot product with 0 * inf treated as 0.
 */
[[nodiscard]] MOCHI_API real SafeDot(Real3 const& a, Real3 const& b);

/**
 * @brief Scale a vector by a scalar, treating 0 * inf as 0 to avoid IEEE 754 NaN.
 *
 * @details Standard multiplication produces NaN when a zero vector component is scaled by ±inf
 * (e.g. {0, 0, 1} * -inf = {NaN, NaN, -inf}). This variant preserves zero components.
 *
 * @param[in] v Vector to scale.
 * @param[in] scalar Scalar factor.
 * @return Scaled vector with zero components preserved.
 */
[[nodiscard]] MOCHI_API Real3 SafeScale(Real3 const& v, real scalar);

} // namespace superdex::robotics
