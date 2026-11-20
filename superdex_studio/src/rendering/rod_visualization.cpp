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

#include "rendering/rod_visualization.h"

#include <mochi_core/contact/contact_params.h>
#include <mochi_core/elements/segment/simplex_quadrature.h>
#include <mochi_core/geometry/mesh_data_utils.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_physics/cpp_api/mochi_enums.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace superdex::studio {

using mochi::Int2;
using mochi::operator""_r;
using mochi::real;
using mochi::Real3;

bool UsesRodPointCloudCollider(mochi::ColliderType type) {
  return type == mochi::ColliderType::PointCloud || type == mochi::ColliderType::Auto;
}

bool HasUniformScaleMagnitude(mochi::Real3 const& scale) {
  auto const magnitudes = mochi::Abs(scale);
  return mochi::NearEqualRel(magnitudes[0], magnitudes[1]) &&
      mochi::NearEqualRel(magnitudes[0], magnitudes[2]);
}

std::vector<Int2> PolylineElements(int numNodes, mochi::Span<int const> connectivity) {
  if (!connectivity.empty()) {
    auto const elements = mochi::Unflatten<Int2 const>(connectivity);
    return {elements.begin(), elements.end()};
  }
  if (numNodes < 2) {
    return {};
  }
  mochi::DynamicArray<int> const sequential =
      mochi::MakeSequentialPolylineConnectivity(numNodes, /*isClosedLoop=*/false);
  auto const elements = mochi::Unflatten<Int2 const>(mochi::MakeConstSpan(sequential));
  return {elements.begin(), elements.end()};
}

std::vector<FrameAxisGlyph> ComputeFrameAxisGlyphs(
    mochi::Span<real const> nodeCoordinates,
    mochi::Span<int const> connectivity,
    mochi::Span<real const> frameAxes) {
  auto const nodes = mochi::Unflatten<Real3 const>(nodeCoordinates);
  std::vector<Int2> const elements = PolylineElements(mochi::isize(nodes), connectivity);
  if (elements.empty() || mochi::isize(frameAxes) != 3 * mochi::isize(elements)) {
    return {};
  }
  auto const axes = mochi::Unflatten<Real3 const>(frameAxes);
  real totalLength = 0_r;
  for (Int2 const& element : elements) {
    totalLength += mochi::Norm(nodes[element[1]] - nodes[element[0]]);
  }
  real const glyphLength =
      kFrameAxisGlyphLengthFraction * totalLength / static_cast<real>(elements.size());
  std::vector<FrameAxisGlyph> glyphs;
  glyphs.reserve(elements.size());
  for (int i = 0; i < mochi::isize(elements); ++i) {
    Real3 const midpoint = (nodes[elements[i][0]] + nodes[elements[i][1]]) * 0.5_r;
    glyphs.push_back({midpoint, axes[i] * glyphLength});
  }
  return glyphs;
}

std::vector<Real3> ComputeRodColliderSamples(
    mochi::Span<real const> nodeCoordinates,
    mochi::Span<int const> connectivity,
    std::optional<mochi::ActorSegmentElementType> segmentElementType) {
  auto const nodes = mochi::Unflatten<Real3 const>(nodeCoordinates);
  if (!segmentElementType.has_value()) {
    return {nodes.begin(), nodes.end()};
  }
  mochi::Span<mochi::Real1 const> points;
  switch (*segmentElementType) {
    case mochi::ActorSegmentElementType::P1Q1:
      points = mochi::MakeConstSpan(mochi::segment::SegmentQuadrature<1>::points);
      break;
    case mochi::ActorSegmentElementType::P1Q2:
      points = mochi::MakeConstSpan(mochi::segment::SegmentQuadrature<2>::points);
      break;
    case mochi::ActorSegmentElementType::P1Q3:
      points = mochi::MakeConstSpan(mochi::segment::SegmentQuadrature<3>::points);
      break;
    case mochi::ActorSegmentElementType::Count:
      break;
  }
  std::vector<Real3> samples;
  for (Int2 const& element : PolylineElements(mochi::isize(nodes), connectivity)) {
    for (auto const& point : points) {
      real const t = point[0];
      samples.push_back(nodes[element[0]] * (1_r - t) + nodes[element[1]] * t);
    }
  }
  return samples;
}

static constexpr int kSphereStacks = 5;
static constexpr int kSphereSlices = 8;

static std::vector<Real3> const& UnitSphereVertices() {
  static std::vector<Real3> const points = [] {
    std::vector<Real3> vertices;
    for (int stack = 0; stack <= kSphereStacks; ++stack) {
      real const theta = static_cast<real>(mochi::kPI) * static_cast<real>(stack) / kSphereStacks;
      for (int slice = 0; slice <= kSphereSlices; ++slice) {
        real const phi =
            2_r * static_cast<real>(mochi::kPI) * static_cast<real>(slice) / kSphereSlices;
        vertices.emplace_back(
            std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta));
      }
    }
    return vertices;
  }();
  return points;
}

std::vector<real> BuildSphereCloudPositions(mochi::Span<Real3 const> centers, real radius) {
  std::vector<real> positions;
  auto const& unitSphere = UnitSphereVertices();
  positions.reserve(centers.size() * unitSphere.size() * 3);
  for (Real3 const& center : centers) {
    for (Real3 const& normal : unitSphere) {
      Real3 const position = center + normal * radius;
      positions.insert(positions.end(), {position[0], position[1], position[2]});
    }
  }
  return positions;
}

SphereCloudMesh BuildSphereCloud(mochi::Span<Real3 const> centers, real radius) {
  auto const& unitSphere = UnitSphereVertices();
  int const verticesPerSphere = mochi::isize(unitSphere);
  SphereCloudMesh mesh;
  mesh.positions = BuildSphereCloudPositions(centers, radius);
  mesh.normals.reserve(mesh.positions.size());
  mesh.indices.reserve(centers.size() * kSphereStacks * kSphereSlices * 6);
  for (int sphere = 0; sphere < mochi::isize(centers); ++sphere) {
    for (Real3 const& normal : unitSphere) {
      mesh.normals.insert(mesh.normals.end(), {normal[0], normal[1], normal[2]});
    }
    int const base = sphere * verticesPerSphere;
    for (int stack = 0; stack < kSphereStacks; ++stack) {
      for (int slice = 0; slice < kSphereSlices; ++slice) {
        int const current = base + stack * (kSphereSlices + 1) + slice;
        int const below = current + kSphereSlices + 1;
        mesh.indices.insert(
            mesh.indices.end(), {current, below, current + 1, current + 1, below, below + 1});
      }
    }
  }
  return mesh;
}

std::vector<int> SelectGrabCluster(
    mochi::Span<Real3 const> nodes,
    Real3 const& grabPoint,
    real radiusFraction,
    int maxNodes) {
  if (nodes.empty()) {
    return {};
  }
  Real3 boundsMin = nodes[0];
  Real3 boundsMax = nodes[0];
  std::vector<std::pair<real, int>> byDistance;
  byDistance.reserve(nodes.size());
  for (int i = 0; i < mochi::isize(nodes); ++i) {
    boundsMin = mochi::Min(boundsMin, nodes[i]);
    boundsMax = mochi::Max(boundsMax, nodes[i]);
    byDistance.emplace_back(mochi::NormSqr(nodes[i] - grabPoint), i);
  }
  std::sort(byDistance.begin(), byDistance.end());
  real const radiusSq = mochi::Sqr(radiusFraction * mochi::Norm(boundsMax - boundsMin));
  // The nearest node always, then the rest in order while within the radius and the cap.
  std::vector<int> cluster = {byDistance[0].second};
  for (int k = 1; k < mochi::isize(byDistance) && mochi::isize(cluster) < maxNodes &&
       byDistance[k].first <= radiusSq;
       ++k) {
    cluster.push_back(byDistance[k].second);
  }
  return cluster;
}

static std::optional<mochi::Float2> ClipSegmentToNearPlane(ClipSegment const& segment) {
  float const distanceA = segment.a[3] + segment.a[2];
  float const distanceB = segment.b[3] + segment.b[2];
  if (distanceA < 0.0f && distanceB < 0.0f) {
    return {};
  }
  if (distanceA >= 0.0f && distanceB >= 0.0f) {
    return mochi::Float2{0.0f, 1.0f};
  }
  float const t = distanceA / (distanceA - distanceB);
  return distanceA < 0.0f ? mochi::Float2{t, 1.0f} : mochi::Float2{0.0f, t};
}

std::optional<ScreenSegmentHit> PickNearestScreenSegment(
    mochi::Span<ClipSegment const> segments,
    mochi::Float2 cursor,
    mochi::Float2 viewportSize,
    float maxDistancePx) {
  std::optional<ScreenSegmentHit> best;
  float bestDepth = std::numeric_limits<float>::max();
  auto toPixels = [&](mochi::Float4 const& clip, float inverseW) {
    return (mochi::Float2{clip[0], clip[1]} * inverseW * 0.5f + 0.5f) * viewportSize;
  };
  for (int i = 0; i < mochi::isize(segments); ++i) {
    ClipSegment const& segment = segments[i];
    auto const range = ClipSegmentToNearPlane(segment);
    if (!range) {
      continue;
    }
    mochi::Float4 const clipA = mochi::Lerp(segment.a, segment.b, (*range)[0]);
    mochi::Float4 const clipB = mochi::Lerp(segment.a, segment.b, (*range)[1]);
    if (clipA[3] <= 0.0f || clipB[3] <= 0.0f) {
      continue;
    }
    float const inverseWA = 1.0f / clipA[3];
    float const inverseWB = 1.0f / clipB[3];
    mochi::Float2 const a = toPixels(clipA, inverseWA);
    mochi::Float2 const b = toPixels(clipB, inverseWB);
    mochi::Float2 const ab = b - a;
    float const lengthSq = mochi::Dot(ab, ab);
    float const depthA = clipA[2] * inverseWA;
    float const depthB = clipB[2] * inverseWB;
    float const screenT = lengthSq > 0.0f
        ? std::clamp(mochi::Dot(cursor - a, ab) / lengthSq, 0.0f, 1.0f)
        : (depthA <= depthB ? 0.0f : 1.0f);
    float const distancePx = mochi::Norm(cursor - (a + ab * screenT));
    if (distancePx > maxDistancePx) {
      continue;
    }
    float const depth = mochi::Lerp(depthA, depthB, screenT);
    if (!best || distancePx < best->distancePx ||
        (distancePx == best->distancePx && depth < bestDepth)) {
      float const worldT = screenT * inverseWB / mochi::Lerp(inverseWA, inverseWB, screenT);
      float const t = mochi::Lerp((*range)[0], (*range)[1], worldT);
      best = {.segment = i, .t = t, .distancePx = distancePx};
      bestDepth = depth;
    }
  }
  return best;
}

} // namespace superdex::studio
