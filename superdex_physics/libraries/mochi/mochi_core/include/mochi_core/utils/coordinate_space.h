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
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/reflection.h>

#include <cstdint>

namespace mochi {

/**
 * @brief Combination of axis directions for a coordinate space convention.
 *
 * @details Each three-letter name gives the semantic direction of positive X,
 * positive Y, and positive Z, respectively. For example, @ref CoordinateSpaceAxes::FLU
 * means X-forward, Y-left, Z-up.
 */
enum class CoordinateSpaceAxes : uint16_t {
  // Coordinate space conventions are composed of three of these directions for X, Y, and Z.
  // These enumerators are not valid on their own.
  Right,
  Left,
  Up,
  Down,
  Forward,
  Backward,

  // X=right/left, Y=up/down, Z=forward/backward.
  RUF = Right << 0 | Up << 3 | Forward << 6,
  RUB = Right << 0 | Up << 3 | Backward << 6,
  RDF = Right << 0 | Down << 3 | Forward << 6,
  RDB = Right << 0 | Down << 3 | Backward << 6,
  LUF = Left << 0 | Up << 3 | Forward << 6,
  LUB = Left << 0 | Up << 3 | Backward << 6,
  LDF = Left << 0 | Down << 3 | Forward << 6,
  LDB = Left << 0 | Down << 3 | Backward << 6,

  // X=right/left, Y=forward/backward, Z=up/down.
  RFU = Right << 0 | Forward << 3 | Up << 6,
  RFD = Right << 0 | Forward << 3 | Down << 6,
  RBU = Right << 0 | Backward << 3 | Up << 6,
  RBD = Right << 0 | Backward << 3 | Down << 6,
  LFU = Left << 0 | Forward << 3 | Up << 6,
  LFD = Left << 0 | Forward << 3 | Down << 6,
  LBU = Left << 0 | Backward << 3 | Up << 6,
  LBD = Left << 0 | Backward << 3 | Down << 6,

  // X=up/down, Y=right/left, Z=forward/backward.
  URF = Up << 0 | Right << 3 | Forward << 6,
  URB = Up << 0 | Right << 3 | Backward << 6,
  ULF = Up << 0 | Left << 3 | Forward << 6,
  ULB = Up << 0 | Left << 3 | Backward << 6,
  DRF = Down << 0 | Right << 3 | Forward << 6,
  DRB = Down << 0 | Right << 3 | Backward << 6,
  DLF = Down << 0 | Left << 3 | Forward << 6,
  DLB = Down << 0 | Left << 3 | Backward << 6,

  // X=up/down, Y=forward/backward, Z=right/left.
  UFR = Up << 0 | Forward << 3 | Right << 6,
  UFL = Up << 0 | Forward << 3 | Left << 6,
  UBR = Up << 0 | Backward << 3 | Right << 6,
  UBL = Up << 0 | Backward << 3 | Left << 6,
  DFR = Down << 0 | Forward << 3 | Right << 6,
  DFL = Down << 0 | Forward << 3 | Left << 6,
  DBR = Down << 0 | Backward << 3 | Right << 6,
  DBL = Down << 0 | Backward << 3 | Left << 6,

  // X=forward/backward, Y=right/left, Z=up/down.
  FRU = Forward << 0 | Right << 3 | Up << 6,
  FRD = Forward << 0 | Right << 3 | Down << 6,
  FLU = Forward << 0 | Left << 3 | Up << 6,
  FLD = Forward << 0 | Left << 3 | Down << 6,
  BRU = Backward << 0 | Right << 3 | Up << 6,
  BRD = Backward << 0 | Right << 3 | Down << 6,
  BLU = Backward << 0 | Left << 3 | Up << 6,
  BLD = Backward << 0 | Left << 3 | Down << 6,

  // X=forward/backward, Y=up/down, Z=right/left.
  FUR = Forward << 0 | Up << 3 | Right << 6,
  FUL = Forward << 0 | Up << 3 | Left << 6,
  FDR = Forward << 0 | Down << 3 | Right << 6,
  FDL = Forward << 0 | Down << 3 | Left << 6,
  BUR = Backward << 0 | Up << 3 | Right << 6,
  BUL = Backward << 0 | Up << 3 | Left << 6,
  BDR = Backward << 0 | Down << 3 | Right << 6,
  BDL = Backward << 0 | Down << 3 | Left << 6,

  // Default convention for Mochi
  Default = FLU,
};

} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::CoordinateSpaceAxes)
MOCHI_ENUM_ITEM(RUF) MOCHI_ATTRIBUTE(DisplayName("X-right, Y-up, Z-forward (Unity)"));
MOCHI_ENUM_ITEM(RUB) MOCHI_ATTRIBUTE(DisplayName("X-right, Y-up, Z-backward (OpenGL, Filament)"));
MOCHI_ENUM_ITEM(RDF) MOCHI_ATTRIBUTE(DisplayName("X-right, Y-down, Z-forward"));
MOCHI_ENUM_ITEM(RDB) MOCHI_ATTRIBUTE(DisplayName("X-right, Y-down, Z-backward"));
MOCHI_ENUM_ITEM(LUF) MOCHI_ATTRIBUTE(DisplayName("X-left, Y-up, Z-forward"));
MOCHI_ENUM_ITEM(LUB) MOCHI_ATTRIBUTE(DisplayName("X-left, Y-up, Z-backward"));
MOCHI_ENUM_ITEM(LDF) MOCHI_ATTRIBUTE(DisplayName("X-left, Y-down, Z-forward"));
MOCHI_ENUM_ITEM(LDB) MOCHI_ATTRIBUTE(DisplayName("X-left, Y-down, Z-backward"));
MOCHI_ENUM_ITEM(RFU) MOCHI_ATTRIBUTE(DisplayName("X-right, Y-forward, Z-up"));
MOCHI_ENUM_ITEM(RFD) MOCHI_ATTRIBUTE(DisplayName("X-right, Y-forward, Z-down"));
MOCHI_ENUM_ITEM(RBU) MOCHI_ATTRIBUTE(DisplayName("X-right, Y-backward, Z-up"));
MOCHI_ENUM_ITEM(RBD) MOCHI_ATTRIBUTE(DisplayName("X-right, Y-backward, Z-down"));
MOCHI_ENUM_ITEM(LFU) MOCHI_ATTRIBUTE(DisplayName("X-left, Y-forward, Z-up"));
MOCHI_ENUM_ITEM(LFD) MOCHI_ATTRIBUTE(DisplayName("X-left, Y-forward, Z-down"));
MOCHI_ENUM_ITEM(LBU) MOCHI_ATTRIBUTE(DisplayName("X-left, Y-backward, Z-up"));
MOCHI_ENUM_ITEM(LBD) MOCHI_ATTRIBUTE(DisplayName("X-left, Y-backward, Z-down"));
MOCHI_ENUM_ITEM(URF) MOCHI_ATTRIBUTE(DisplayName("X-up, Y-right, Z-forward"));
MOCHI_ENUM_ITEM(URB) MOCHI_ATTRIBUTE(DisplayName("X-up, Y-right, Z-backward"));
MOCHI_ENUM_ITEM(ULF) MOCHI_ATTRIBUTE(DisplayName("X-up, Y-left, Z-forward"));
MOCHI_ENUM_ITEM(ULB) MOCHI_ATTRIBUTE(DisplayName("X-up, Y-left, Z-backward"));
MOCHI_ENUM_ITEM(DRF) MOCHI_ATTRIBUTE(DisplayName("X-down, Y-right, Z-forward"));
MOCHI_ENUM_ITEM(DRB) MOCHI_ATTRIBUTE(DisplayName("X-down, Y-right, Z-backward"));
MOCHI_ENUM_ITEM(DLF) MOCHI_ATTRIBUTE(DisplayName("X-down, Y-left, Z-forward"));
MOCHI_ENUM_ITEM(DLB) MOCHI_ATTRIBUTE(DisplayName("X-down, Y-left, Z-backward"));
MOCHI_ENUM_ITEM(UFR) MOCHI_ATTRIBUTE(DisplayName("X-up, Y-forward, Z-right"));
MOCHI_ENUM_ITEM(UFL) MOCHI_ATTRIBUTE(DisplayName("X-up, Y-forward, Z-left"));
MOCHI_ENUM_ITEM(UBR) MOCHI_ATTRIBUTE(DisplayName("X-up, Y-backward, Z-right"));
MOCHI_ENUM_ITEM(UBL) MOCHI_ATTRIBUTE(DisplayName("X-up, Y-backward, Z-left"));
MOCHI_ENUM_ITEM(DFR) MOCHI_ATTRIBUTE(DisplayName("X-down, Y-forward, Z-right"));
MOCHI_ENUM_ITEM(DFL) MOCHI_ATTRIBUTE(DisplayName("X-down, Y-forward, Z-left"));
MOCHI_ENUM_ITEM(DBR) MOCHI_ATTRIBUTE(DisplayName("X-down, Y-backward, Z-right"));
MOCHI_ENUM_ITEM(DBL) MOCHI_ATTRIBUTE(DisplayName("X-down, Y-backward, Z-left"));
MOCHI_ENUM_ITEM(FRU) MOCHI_ATTRIBUTE(DisplayName("X-forward, Y-right, Z-up (Unreal)"));
MOCHI_ENUM_ITEM(FRD) MOCHI_ATTRIBUTE(DisplayName("X-forward, Y-right, Z-down"));
MOCHI_ENUM_ITEM(FLU) MOCHI_ATTRIBUTE(DisplayName("X-forward, Y-left, Z-up (Default)"));
MOCHI_ENUM_ITEM(FLD) MOCHI_ATTRIBUTE(DisplayName("X-forward, Y-left, Z-down"));
MOCHI_ENUM_ITEM(BRU) MOCHI_ATTRIBUTE(DisplayName("X-backward, Y-right, Z-up"));
MOCHI_ENUM_ITEM(BRD) MOCHI_ATTRIBUTE(DisplayName("X-backward, Y-right, Z-down"));
MOCHI_ENUM_ITEM(BLU) MOCHI_ATTRIBUTE(DisplayName("X-backward, Y-left, Z-up"));
MOCHI_ENUM_ITEM(BLD) MOCHI_ATTRIBUTE(DisplayName("X-backward, Y-left, Z-down"));
MOCHI_ENUM_ITEM(FUR) MOCHI_ATTRIBUTE(DisplayName("X-forward, Y-up, Z-right"));
MOCHI_ENUM_ITEM(FUL) MOCHI_ATTRIBUTE(DisplayName("X-forward, Y-up, Z-left"));
MOCHI_ENUM_ITEM(FDR) MOCHI_ATTRIBUTE(DisplayName("X-forward, Y-down, Z-right"));
MOCHI_ENUM_ITEM(FDL) MOCHI_ATTRIBUTE(DisplayName("X-forward, Y-down, Z-left"));
MOCHI_ENUM_ITEM(BUR) MOCHI_ATTRIBUTE(DisplayName("X-backward, Y-up, Z-right"));
MOCHI_ENUM_ITEM(BUL) MOCHI_ATTRIBUTE(DisplayName("X-backward, Y-up, Z-left"));
MOCHI_ENUM_ITEM(BDR) MOCHI_ATTRIBUTE(DisplayName("X-backward, Y-down, Z-right"));
MOCHI_ENUM_ITEM(BDL) MOCHI_ATTRIBUTE(DisplayName("X-backward, Y-down, Z-left"));
MOCHI_ENUM_END()

namespace mochi {

/** @brief Coordinate-axis convention and linear unit scale. */
struct CoordinateSpace {
  CoordinateSpaceAxes axes = CoordinateSpaceAxes::Default;

  // Always double precision for precision-independent debugger protocol.
  double unitsPerMeter = 1.0;

  constexpr CoordinateSpace() = default;
  constexpr CoordinateSpace(CoordinateSpaceAxes inAxes, double inUnitsPerMeter)
      : axes(inAxes), unitsPerMeter(inUnitsPerMeter) {}

  /**
   * @brief Validate that @ref axes is a complete convention and
   * @ref unitsPerMeter is finite and positive.
   */
  void Validate(Error& error) const;

  /** @brief Return the unit vector pointing right, in this space's own coordinates. */
  [[nodiscard]] Real3 GetRight() const;

  /** @brief Return the unit vector pointing up, in this space's own coordinates. */
  [[nodiscard]] Real3 GetUp() const;

  /** @brief Return the unit vector pointing forward, in this space's own coordinates. */
  [[nodiscard]] Real3 GetForward() const;

  /** @brief Return the default coordinate space convention for Mochi. */
  [[nodiscard]] static CoordinateSpace Default() {
    return {CoordinateSpaceAxes::Default, 1.0};
  }

  /** @brief Return the coordinate space convention for Filament. */
  [[nodiscard]] static CoordinateSpace Filament() {
    return {CoordinateSpaceAxes::RUB, 1.0};
  }

  /** @brief Return the coordinate space convention for OpenGL. */
  [[nodiscard]] static CoordinateSpace OpenGL() {
    return {CoordinateSpaceAxes::RUB, 1.0};
  }

  /** @brief Return the coordinate space convention for ROS */
  [[nodiscard]] static CoordinateSpace ROS() {
    return {CoordinateSpaceAxes::FLU, 1.0}; // Same as Default
  }

  /** @brief Return the coordinate space convention for Unity. */
  [[nodiscard]] static CoordinateSpace Unity() {
    return {CoordinateSpaceAxes::RUF, 1.0};
  }

  /** @brief Return the coordinate space convention for Unreal Engine. */
  [[nodiscard]] static CoordinateSpace Unreal() {
    return {CoordinateSpaceAxes::FRU, 100.0};
  }

#if MOCHI_LANGUAGE_CPP20
  constexpr bool operator==(CoordinateSpace const& rhs) const = default;
  constexpr bool operator!=(CoordinateSpace const& rhs) const = default;
#endif

  MOCHI_STRUCT_BEGIN(mochi::CoordinateSpace)
  MOCHI_FIELD(axes)
  MOCHI_FIELD(unitsPerMeter)
  MOCHI_STRUCT_END()
};

} // namespace mochi
