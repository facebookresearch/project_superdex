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

#include <mochi_core/utils/nd_array.h>

namespace mochi::triangular {

template <int kNumQuadPoints>
struct TriangleQuadrature final {};

// --- 1-Point, Degree 1 Rule (Dunavant)
template <>
struct TriangleQuadrature<1> final {
  static constexpr size_t kNumQuadPoints = 1;
  static constexpr NdArray<real, 1, 2> points = {Real2{1_r / 3_r, 1_r / 3_r}};
  static constexpr NdArray<real, 1> weights = {1_r / 2_r};
};

// --- 3-Point, Degree 2 Rule (Dunavant)
template <>
struct TriangleQuadrature<3> final {
  static constexpr size_t kNumQuadPoints = 3;
  static constexpr NdArray<real, 3, 2> points = {
      Real2{2_r / 3_r, 1_r / 6_r},
      Real2{1_r / 6_r, 2_r / 3_r},
      Real2{1_r / 6_r, 1_r / 6_r}};
  static constexpr NdArray<real, 3> weights = {1_r / 6_r, 1_r / 6_r, 1_r / 6_r};
};

// --- 6-Point, Degree 4 Rule (Dunavant)
template <>
struct TriangleQuadrature<6> final {
  static constexpr size_t kNumQuadPoints = 6;
  static constexpr NdArray<real, 6, 2> points = {
      Real2{0.445948490915965_r, 0.445948490915965_r},
      Real2{0.445948490915965_r, 0.108103018168070_r},
      Real2{0.108103018168070_r, 0.445948490915965_r},
      Real2{0.091576213509771_r, 0.091576213509771_r},
      Real2{0.091576213509771_r, 0.816847572980459_r},
      Real2{0.816847572980459_r, 0.091576213509771_r}};
  static constexpr NdArray<real, 6> weights = {
      0.223381589678011_r / 2_r,
      0.223381589678011_r / 2_r,
      0.223381589678011_r / 2_r,
      0.109951743655322_r / 2_r,
      0.109951743655322_r / 2_r,
      0.109951743655322_r / 2_r};
};

// --- 7-Point, Degree 5 Rule (Lyness-Jespersen)
template <>
struct TriangleQuadrature<7> final {
  static constexpr size_t kNumQuadPoints = 7;
  static constexpr NdArray<real, 7, 2> points = {
      Real2{0.333333333333333_r, 0.333333333333333_r},
      Real2{0.797426985353087_r, 0.101286507323456_r},
      Real2{0.101286507323456_r, 0.797426985353087_r},
      Real2{0.101286507323456_r, 0.101286507323456_r},
      Real2{0.470142064105115_r, 0.059715871789770_r},
      Real2{0.059715871789770_r, 0.470142064105115_r},
      Real2{0.470142064105115_r, 0.470142064105115_r}};
  static constexpr NdArray<real, 7> weights = {
      0.1125_r,
      0.0629695902724135_r,
      0.0629695902724135_r,
      0.0629695902724135_r,
      0.066197076394253_r,
      0.066197076394253_r,
      0.066197076394253_r};
};

// --- 12-Point, Degree 6 Rule (Dunavant)
template <>
struct TriangleQuadrature<12> final {
  static constexpr size_t kNumQuadPoints = 12;
  static constexpr NdArray<real, 12, 2> points = {
      Real2{0.24928674517091_r, 0.24928674517091_r},
      Real2{0.50142650965818_r, 0.24928674517091_r},
      Real2{0.24928674517091_r, 0.50142650965818_r},
      Real2{0.06308901449150_r, 0.06308901449150_r},
      Real2{0.87382197101700_r, 0.06308901449150_r},
      Real2{0.06308901449150_r, 0.87382197101700_r},
      Real2{0.31035245103378_r, 0.05314504984482_r},
      Real2{0.63650249912140_r, 0.05314504984482_r},
      Real2{0.05314504984482_r, 0.31035245103378_r},
      Real2{0.31035245103378_r, 0.63650249912140_r},
      Real2{0.05314504984482_r, 0.63650249912140_r},
      Real2{0.63650249912140_r, 0.31035245103378_r}};
  static constexpr NdArray<real, 12> weights = {
      0.05839313786319_r,
      0.05839313786319_r,
      0.05839313786319_r,
      0.025422453185105_r,
      0.025422453185105_r,
      0.025422453185105_r,
      0.041425537809185_r,
      0.041425537809185_r,
      0.041425537809185_r,
      0.041425537809185_r,
      0.041425537809185_r,
      0.041425537809185_r};
};

// --- 16-Point, Degree 8 Rule (Lyness-Jespersen)
template <>
struct TriangleQuadrature<16> final {
  static constexpr size_t kNumQuadPoints = 16;
  static constexpr NdArray<real, 16, 2> points = {
      Real2{0.3333333333333333_r, 0.3333333333333333_r},
      Real2{0.4592925882927230_r, 0.4592925882927230_r},
      Real2{0.0814148234145540_r, 0.4592925882927230_r},
      Real2{0.4592925882927230_r, 0.0814148234145540_r},
      Real2{0.1705693077517600_r, 0.6588613844964800_r},
      Real2{0.6588613844964800_r, 0.1705693077517600_r},
      Real2{0.1705693077517600_r, 0.1705693077517600_r},
      Real2{0.0083947774099580_r, 0.2631128296346380_r},
      Real2{0.2631128296346380_r, 0.0083947774099580_r},
      Real2{0.7284923929554040_r, 0.0083947774099580_r},
      Real2{0.0083947774099580_r, 0.7284923929554040_r},
      Real2{0.2631128296346380_r, 0.7284923929554040_r},
      Real2{0.7284923929554040_r, 0.2631128296346380_r},
      Real2{0.0505472283170310_r, 0.8989055433659380_r},
      Real2{0.8989055433659380_r, 0.0505472283170310_r},
      Real2{0.0505472283170310_r, 0.0505472283170310_r}};
  static constexpr NdArray<real, 16> weights = {
      0.0721578038388935_r,
      0.0475458171336425_r,
      0.0475458171336425_r,
      0.0475458171336425_r,
      0.0516086852673590_r,
      0.0516086852673590_r,
      0.0516086852673590_r,
      0.0136151570872175_r,
      0.0136151570872175_r,
      0.0136151570872175_r,
      0.0136151570872175_r,
      0.0136151570872175_r,
      0.0136151570872175_r,
      0.0162292487415990_r,
      0.0162292487415990_r,
      0.0162292487415990_r};
};

} // namespace mochi::triangular
