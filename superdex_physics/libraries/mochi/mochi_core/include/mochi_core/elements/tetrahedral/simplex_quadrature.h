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

#include <mochi_core/elements/triangular/simplex_quadrature.h>
#include <mochi_core/utils/nd_array.h>

namespace mochi::tetrahedral {

// A quadrature is a a rule that has a set of points (Real3) and a set of weights (real). The
// quadrature rule structures are templatized by the kNumQuadPoints that refers to the number of
// quadrature points used in the rule
template <int kNumQuadPoints>
struct TetrahedralQuadrature final {
  NdArray<real, kNumQuadPoints, 3> points;
  NdArray<real, kNumQuadPoints> weights;
};

static constexpr TetrahedralQuadrature<1> kTetrahedralQuadrature1{
    {Real3{1_r / 4_r, 1_r / 4_r, 1_r / 4_r}},
    {1_r / 6_r}};

static constexpr TetrahedralQuadrature<4> kTetrahedralQuadrature4{
    {Real3{0.58541019662496852_r, 0.13819660112501051_r, 0.13819660112501051_r},
     Real3{0.13819660112501051_r, 0.58541019662496852_r, 0.13819660112501051_r},
     Real3{0.13819660112501051_r, 0.13819660112501051_r, 0.58541019662496852_r},
     Real3{0.13819660112501051_r, 0.13819660112501051_r, 0.13819660112501051_r}},
    {1_r / 24_r, 1_r / 24_r, 1_r / 24_r, 1_r / 24_r}};

/** Tetrahedra trace quadrature allows evaluation of basis functions and derivatives on the
    faces of the tetrahedron. For a tetrahedron with vertices labeled 0,1,2,3, the faces
    numbers are given by

    face | vertices
    0      0,1,2
    1      0,1,3
    2      0,2,3
    3      1,2,3
*/

namespace details {
// Helper function to compute face area scaling factor.
template <int kFaceId>
constexpr real GetFaceAreaScaling() {
  if constexpr (kFaceId == 0 || kFaceId == 1 || kFaceId == 2) {
    return 1_r;
  } else {
    static_assert(kFaceId == 3, "Unsupported face ID");
    return 1.73205080756887719318_r;
  }
};

// Helper function to map a point from the reference triangle to a tetrahedron face.
template <int kFaceId>
constexpr Real3 MapReferenceTriangleToTetrahedronFace(Real2 const& x) {
  if constexpr (kFaceId == 0) {
    return Real3{x[0], x[1], 0_r};
  } else if constexpr (kFaceId == 1) {
    return Real3{x[0], 0_r, x[1]};
  } else if constexpr (kFaceId == 2) {
    return Real3{0_r, x[0], x[1]};
  } else {
    static_assert(kFaceId == 3, "Unsupported face ID");
    return Real3{x[0], x[1], 1_r - x[0] - x[1]};
  }
};

// Helper function to map all quadrature points from the reference triangle to a tetrahedron face.
template <int kFaceId, typename TriangularQuadrature>
constexpr auto MapReferenceTriangleToTetrahedronFace() {
  constexpr size_t kNumPoints = TriangularQuadrature::kNumQuadPoints;
  NdArray<real, kNumPoints, 3> points{};
  for (size_t i = 0; i < kNumPoints; ++i) {
    points[i] = MapReferenceTriangleToTetrahedronFace<kFaceId>(TriangularQuadrature::points[i]);
  }
  return points;
};

// Helper function to generate a tetrahedral trace quadrature rule for a single face from a
// triangular quadrature rule.
template <int kFaceId, typename TriangularQuadrature>
constexpr auto MakeTetrahedralTraceQuadratureForFace() {
  return TetrahedralQuadrature<TriangularQuadrature::kNumQuadPoints>{
      MapReferenceTriangleToTetrahedronFace<kFaceId, TriangularQuadrature>(),
      TriangularQuadrature::weights * GetFaceAreaScaling<kFaceId>()};
};

// Helper function to generate a tetrahedral trace quadrature rule from a triangular quadrature
// rule.
template <typename TriangularQuadrature>
constexpr auto MakeTetrahedralTraceQuadrature() {
  return NdArray<TetrahedralQuadrature<TriangularQuadrature::kNumQuadPoints>, 4>{
      MakeTetrahedralTraceQuadratureForFace<0, TriangularQuadrature>(),
      MakeTetrahedralTraceQuadratureForFace<1, TriangularQuadrature>(),
      MakeTetrahedralTraceQuadratureForFace<2, TriangularQuadrature>(),
      MakeTetrahedralTraceQuadratureForFace<3, TriangularQuadrature>()};
};
} // namespace details

static constexpr auto kTetrahedralTraceQuadrature1 =
    details::MakeTetrahedralTraceQuadrature<triangular::TriangleQuadrature<1>>();
static constexpr auto kTetrahedralTraceQuadrature3 =
    details::MakeTetrahedralTraceQuadrature<triangular::TriangleQuadrature<3>>();
static constexpr auto kTetrahedralTraceQuadrature6 =
    details::MakeTetrahedralTraceQuadrature<triangular::TriangleQuadrature<6>>();
static constexpr auto kTetrahedralTraceQuadrature7 =
    details::MakeTetrahedralTraceQuadrature<triangular::TriangleQuadrature<7>>();
static constexpr auto kTetrahedralTraceQuadrature12 =
    details::MakeTetrahedralTraceQuadrature<triangular::TriangleQuadrature<12>>();
static constexpr auto kTetrahedralTraceQuadrature16 =
    details::MakeTetrahedralTraceQuadrature<triangular::TriangleQuadrature<16>>();

} // namespace mochi::tetrahedral
