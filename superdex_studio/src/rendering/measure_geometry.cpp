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

#include "rendering/measure_geometry.h"

#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/math_utils.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <numbers>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace superdex::studio {

namespace {

using mochi::real;
using mochi::Real3;
using mochi::operator""_r; // real-valued literals (0_r, 0.5_r, ...)

constexpr real kRadToDeg = real(180) / std::numbers::pi_v<real>;

// Key identifying an undirected edge between two welded vertices
int64_t EdgeKey(int a, int b) {
  if (a > b) {
    std::swap(a, b);
  }
  return (static_cast<int64_t>(a) << 32) | static_cast<uint32_t>(b);
}

// Quantized position, used to merge the duplicate corners the de-indexing mesh readers emit.
struct WeldKey {
  int64_t x = 0;
  int64_t y = 0;
  int64_t z = 0;

  bool operator==(WeldKey const& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct WeldKeyHash {
  std::size_t operator()(WeldKey const& key) const {
    // Multiply-xor mix; the three components are small integers on a lattice, so a plain xor
    // would collide heavily along the axes.
    std::size_t h = static_cast<std::size_t>(key.x) * 0x9E3779B97F4A7C15ull;
    h ^= static_cast<std::size_t>(key.y) * 0xC2B2AE3D27D4EB4Full;
    h ^= static_cast<std::size_t>(key.z) * 0x165667B19E3779F9ull;
    return h;
  }
};

WeldKey MakeWeldKey(Real3 const& p, real tolerance) {
  real const inv = real(1) / tolerance;
  return WeldKey{
      static_cast<int64_t>(std::llround(p[0] * inv)),
      static_cast<int64_t>(std::llround(p[1] * inv)),
      static_cast<int64_t>(std::llround(p[2] * inv))};
}

// Merges @p rawPositions onto a lattice of @p tolerance, rewriting @p indices to the merged
// vertices. Positions that map to the same cell keep the first one seen.
void WeldVertices(
    std::vector<Real3> const& rawPositions,
    std::vector<int> const& rawIndices,
    real tolerance,
    std::vector<Real3>& outVertices,
    std::vector<int>& outIndices) {
  std::unordered_map<WeldKey, int, WeldKeyHash> lookup;
  lookup.reserve(rawPositions.size());
  std::vector<int> remap(rawPositions.size(), -1);
  outVertices.clear();
  outVertices.reserve(rawPositions.size());
  for (std::size_t i = 0; i < rawPositions.size(); ++i) {
    auto const [it, inserted] = lookup.try_emplace(
        MakeWeldKey(rawPositions[i], tolerance), static_cast<int>(outVertices.size()));
    if (inserted) {
      outVertices.push_back(rawPositions[i]);
    }
    remap[i] = it->second;
  }
  outIndices.clear();
  outIndices.reserve(rawIndices.size());
  for (int const index : rawIndices) {
    outIndices.push_back(remap[static_cast<std::size_t>(index)]);
  }
}

// Groups triangles into coplanar facets: a flood fill across shared edges, crossing an edge only
// when the two triangles' normals agree within @p angleToleranceDeg. Degenerate triangles get facet
// -1 and are not visited.
void BuildFacets(MeasureMesh& mesh, real angleToleranceDeg) {
  int const numTriangles = mesh.GetNumTriangles();
  mesh.triangleFacets.assign(static_cast<std::size_t>(numTriangles), -1);
  mesh.facets.clear();

  // (edge key, triangle) pairs sorted by key, so the triangles sharing an edge form a contiguous
  // run. Cheaper and far less allocation-heavy than a map of vectors on a large mesh.
  std::vector<std::pair<int64_t, int>> edges;
  edges.reserve(static_cast<std::size_t>(numTriangles) * 3);
  for (int t = 0; t < numTriangles; ++t) {
    int const base = t * 3;
    for (int e = 0; e < 3; ++e) {
      edges.emplace_back(EdgeKey(mesh.triangles[base + e], mesh.triangles[base + (e + 1) % 3]), t);
    }
  }
  std::ranges::sort(edges);

  real const cosTolerance = std::cos(angleToleranceDeg / kRadToDeg);
  std::vector<int> stack;
  for (int seed = 0; seed < numTriangles; ++seed) {
    if (mesh.triangleFacets[seed] != -1 || mesh.triangleAreas[seed] <= 0) {
      continue;
    }
    int const facet = static_cast<int>(mesh.facets.size());
    mesh.facets.emplace_back();
    stack.assign(1, seed);
    mesh.triangleFacets[seed] = facet;
    while (!stack.empty()) {
      int const t = stack.back();
      stack.pop_back();
      mesh.facets[static_cast<std::size_t>(facet)].push_back(t);
      int const base = t * 3;
      for (int e = 0; e < 3; ++e) {
        int64_t const key = EdgeKey(mesh.triangles[base + e], mesh.triangles[base + (e + 1) % 3]);
        auto it = std::ranges::lower_bound(
            edges, std::pair<int64_t, int>{key, std::numeric_limits<int>::min()});
        for (; it != edges.end() && it->first == key; ++it) {
          int const other = it->second;
          if (mesh.triangleFacets[other] != -1 || mesh.triangleAreas[other] <= 0) {
            continue;
          }
          real const alignment = mochi::Abs(
              mochi::Dot(
                  mesh.triangleNormals[static_cast<std::size_t>(t)],
                  mesh.triangleNormals[static_cast<std::size_t>(other)]));
          if (alignment < cosTolerance) {
            continue;
          }
          mesh.triangleFacets[other] = facet;
          stack.push_back(other);
        }
      }
    }
    std::sort(
        mesh.facets[static_cast<std::size_t>(facet)].begin(),
        mesh.facets[static_cast<std::size_t>(facet)].end());
  }
}

std::shared_ptr<MeasureMesh const> FinishMeasureMesh(
    std::vector<Real3> const& rawPositions,
    std::vector<int> const& rawIndices,
    MeasureMeshParams const& params) {
  auto mesh = std::make_shared<MeasureMesh>();
  if (rawIndices.size() < 3) {
    return mesh;
  }
  real const weldTolerance =
      params.weldToleranceMeters > 0 ? params.weldToleranceMeters : real(1e-6);
  WeldVertices(rawPositions, rawIndices, weldTolerance, mesh->vertices, mesh->triangles);

  int const numTriangles = mesh->GetNumTriangles();
  mesh->triangleNormals.resize(static_cast<std::size_t>(numTriangles));
  mesh->triangleAreas.resize(static_cast<std::size_t>(numTriangles));
  mesh->soup.triangles.reserve(static_cast<std::size_t>(numTriangles));
  for (int t = 0; t < numTriangles; ++t) {
    Real3 const& a = mesh->vertices[static_cast<std::size_t>(mesh->triangles[t * 3 + 0])];
    Real3 const& b = mesh->vertices[static_cast<std::size_t>(mesh->triangles[t * 3 + 1])];
    Real3 const& c = mesh->vertices[static_cast<std::size_t>(mesh->triangles[t * 3 + 2])];
    Real3 const cross = mochi::Cross(b - a, c - a);
    real const crossNorm = mochi::Norm(cross);
    mesh->triangleAreas[static_cast<std::size_t>(t)] = real(0.5) * crossNorm;
    mesh->triangleNormals[static_cast<std::size_t>(t)] =
        crossNorm > 0 ? cross / crossNorm : Real3{0_r, 0_r, 0_r};
    mesh->soup.triangles.emplace_back(a, b, c);
  }

  BuildFacets(*mesh, params.facetAngleToleranceDeg);
  mesh->bvh = std::make_unique<mochi::AabbTree>(&mesh->soup, mochi::BvhTreeParams{});
  return mesh;
}

// Ray-vs-triangle using the mesh's precomputed UNIT normal
std::optional<mochi::RayHit>
IntersectTriangle(MeasureMesh const& mesh, int triangle, mochi::Ray const& ray) {
  Real3 const& normal = mesh.triangleNormals[static_cast<std::size_t>(triangle)];
  if (mesh.triangleAreas[static_cast<std::size_t>(triangle)] <= 0) {
    return std::nullopt;
  }
  mochi::Vec4r const vNormal = mochi::ToSimd(normal, real(0));
  real const denominator = mochi::Dot<3>(ray.direction, vNormal);
  if (mochi::NearZero(denominator)) {
    return std::nullopt;
  }
  mochi::Triangle const& tri = mesh.soup.triangles[static_cast<std::size_t>(triangle)];
  real const t = mochi::Dot<3>(vNormal, tri.v1 - ray.origin) / denominator;
  if (t < 0) {
    return std::nullopt;
  }
  mochi::Vec4r const intersection = ray(t);
  if (mochi::HMin<3>(mochi::BarycentricCoordinates(tri, intersection)) < 0) {
    return std::nullopt;
  }
  mochi::RayHit hit{};
  hit.t = t;
  hit.intersection = intersection;
  hit.index = triangle;
  return hit;
}

//--------------------------------------------------------------------------------------------------
// FORMATTING
//--------------------------------------------------------------------------------------------------

int GetDecimals(MeasureUnit unit) {
  switch (unit) {
    case MeasureUnit::Meters:
      return 6;
    case MeasureUnit::Centimeters:
      return 4;
    case MeasureUnit::Millimeters:
      return 3;
  }
  return 3;
}

std::string FormatScalar(real value, int decimals) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(decimals) << value;
  return out.str();
}

std::string FormatLength(real meters, MeasureUnit unit) {
  return FormatScalar(meters * GetMeasureUnitScale(unit), GetDecimals(unit)) + " " +
      GetMeasureUnitSuffix(unit);
}

std::string FormatArea(real squareMeters, MeasureUnit unit) {
  real const scale = GetMeasureUnitScale(unit);
  return FormatScalar(squareMeters * scale * scale, GetDecimals(unit)) + " " +
      GetMeasureUnitSuffix(unit) + "^2";
}

// "X 1.000  Y 2.000  Z 3.000 mm"
std::string FormatVector(Real3 const& meters, MeasureUnit unit) {
  int const decimals = GetDecimals(unit);
  real const scale = GetMeasureUnitScale(unit);
  std::ostringstream out;
  out << "X " << FormatScalar(meters[0] * scale, decimals) << "  Y "
      << FormatScalar(meters[1] * scale, decimals) << "  Z "
      << FormatScalar(meters[2] * scale, decimals) << ' ' << GetMeasureUnitSuffix(unit);
  return out.str();
}

// Unitless triple, for a normal.
std::string FormatDirection(Real3 const& v) {
  std::ostringstream out;
  out << "X " << FormatScalar(v[0], 4) << "  Y " << FormatScalar(v[1], 4) << "  Z "
      << FormatScalar(v[2], 4);
  return out.str();
}

constexpr int kLabelWidth = 15;

void WriteRow(std::ostringstream& out, char const* label, std::string const& value) {
  out << "  " << std::left << std::setw(kLabelWidth) << label << value << '\n';
}

//--------------------------------------------------------------------------------------------------
// MEASUREMENT HELPERS
//--------------------------------------------------------------------------------------------------

struct Bounds {
  Real3 min{0_r, 0_r, 0_r};
  Real3 max{0_r, 0_r, 0_r};
  bool valid = false;

  void Add(Real3 const& p) {
    if (!valid) {
      min = p;
      max = p;
      valid = true;
      return;
    }
    for (int i = 0; i < 3; ++i) {
      min[i] = mochi::Min(min[i], p[i]);
      max[i] = mochi::Max(max[i], p[i]);
    }
  }
  Real3 GetExtents() const {
    return valid ? Real3{max[0] - min[0], max[1] - min[1], max[2] - min[2]} : Real3{0_r, 0_r, 0_r};
  }
};

Bounds ComputeBounds(std::vector<Real3> const& points) {
  Bounds bounds;
  for (Real3 const& p : points) {
    bounds.Add(p);
  }
  return bounds;
}

// Angle between two (unit) directions, in degrees, folded into [0, 180].
real AngleBetweenDeg(Real3 const& a, Real3 const& b) {
  real const cosine = mochi::Clamp(mochi::Dot(a, b), real(-1), real(1));
  return std::acos(cosine) * kRadToDeg;
}

// Closest point on a face entity's triangles to @p point, and its distance.
std::pair<Real3, real> ClosestPointOnFace(MeasureEntity const& face, Real3 const& point) {
  mochi::Vec4r const p = mochi::ToSimd(point, real(1));
  Real3 closest = face.centroid;
  real bestDistanceSqr = std::numeric_limits<real>::max();
  for (std::size_t i = 0; i + 2 < face.triangles.size(); i += 3) {
    Real3 const& a = face.points[static_cast<std::size_t>(face.triangles[i + 0])];
    Real3 const& b = face.points[static_cast<std::size_t>(face.triangles[i + 1])];
    Real3 const& c = face.points[static_cast<std::size_t>(face.triangles[i + 2])];
    mochi::VDistanceSignParams sign;
    mochi::Vec4r barycentric;
    real const distanceSqr = mochi::Get0(
        mochi::VDistancePointTriangleSqr(
            p,
            mochi::ToSimd(a, real(1)),
            mochi::ToSimd(b, real(1)),
            mochi::ToSimd(c, real(1)),
            sign,
            &barycentric));
    if (distanceSqr < bestDistanceSqr) {
      bestDistanceSqr = distanceSqr;
      Real3 const bary = mochi::ToReal3(barycentric);
      closest = a * bary[0] + b * bary[1] + c * bary[2];
    }
  }
  return {closest, std::sqrt(mochi::Max(bestDistanceSqr, real(0)))};
}

// The perimeter of a face entity's outline.
real ComputeBoundaryLength(MeasureEntity const& face) {
  real total = 0;
  for (std::size_t i = 0; i + 1 < face.boundaryEdges.size(); i += 2) {
    total += mochi::Norm(
        face.points[static_cast<std::size_t>(face.boundaryEdges[i + 1])] -
        face.points[static_cast<std::size_t>(face.boundaryEdges[i + 0])]);
  }
  return total;
}

// Every point of the selection, tagged with the entity it came from so the pairwise search can
// ignore pairs within one entity.
struct TaggedPoints {
  std::vector<Real3> points;
  std::vector<int> entity;
};

TaggedPoints CollectPoints(std::vector<MeasureEntity> const& entities) {
  TaggedPoints tagged;
  for (std::size_t e = 0; e < entities.size(); ++e) {
    for (Real3 const& p : entities[e].points) {
      tagged.points.push_back(p);
      tagged.entity.push_back(static_cast<int>(e));
    }
  }
  return tagged;
}

struct ExtremePair {
  Real3 a{0_r, 0_r, 0_r};
  Real3 b{0_r, 0_r, 0_r};
  real distance = 0;
  bool valid = false;
};

// Longest and shortest distance between points belonging to different entities.
void FindExtremePairs(TaggedPoints const& tagged, ExtremePair& outMax, ExtremePair& outMin) {
  real maxDistanceSqr = -1;
  real minDistanceSqr = std::numeric_limits<real>::max();
  for (std::size_t i = 0; i < tagged.points.size(); ++i) {
    for (std::size_t j = i + 1; j < tagged.points.size(); ++j) {
      if (tagged.entity[i] == tagged.entity[j]) {
        continue;
      }
      real const distanceSqr = mochi::NormSqr(tagged.points[j] - tagged.points[i]);
      if (distanceSqr > maxDistanceSqr) {
        maxDistanceSqr = distanceSqr;
        outMax = {tagged.points[i], tagged.points[j], std::sqrt(distanceSqr), true};
      }
      if (distanceSqr < minDistanceSqr) {
        minDistanceSqr = distanceSqr;
        outMin = {tagged.points[i], tagged.points[j], std::sqrt(distanceSqr), true};
      }
    }
  }
}

// The X, then Y, then Z legs from @p from to @p to, as labelled annotations. A leg of exactly zero
// length is dropped.
void AppendComponentLegs(
    Real3 const& from,
    Real3 const& to,
    MeasureUnit unit,
    std::vector<MeasureAnnotation>& out) {
  Real3 cursor = from;
  static constexpr char const* kAxisNames[3] = {"X", "Y", "Z"};
  for (int axis = 0; axis < 3; ++axis) {
    Real3 next = cursor;
    next[axis] = to[axis];
    real const delta = to[axis] - cursor[axis];
    if (mochi::Abs(delta) > 0) {
      out.push_back(
          {cursor,
           next,
           std::string(kAxisNames[axis]) + " " + FormatLength(delta, unit),
           MeasureAnnotation::Style::Component,
           axis});
    }
    cursor = next;
  }
}

void WriteEntityDetails(
    std::ostringstream& out,
    MeasureEntity const& entity,
    MeasureReportParams const& params) {
  MeasureUnit const unit = params.unit;
  if (!entity.IsFace()) {
    WriteRow(out, "Position", FormatVector(entity.points.front(), unit));
    return;
  }
  Bounds const bounds = ComputeBounds(entity.points);
  WriteRow(out, "Triangles", std::to_string(entity.triangles.size() / 3));
  WriteRow(out, "Corners", std::to_string(entity.points.size()));
  WriteRow(out, "Area", FormatArea(entity.area, unit));
  WriteRow(out, "Centroid", FormatVector(entity.centroid, unit));
  WriteRow(out, "Normal", FormatDirection(entity.normal));
  WriteRow(out, "Extents", FormatVector(bounds.GetExtents(), unit));
  WriteRow(out, "Perimeter", FormatLength(ComputeBoundaryLength(entity), unit));
}

void WritePairMeasurement(
    std::ostringstream& out,
    std::vector<MeasureAnnotation>& annotations,
    MeasureEntity const& a,
    MeasureEntity const& b,
    MeasureReportParams const& params) {
  MeasureUnit const unit = params.unit;
  if (!a.IsFace() && !b.IsFace()) {
    Real3 const from = a.points.front();
    Real3 const to = b.points.front();
    Real3 const delta = to - from;
    real const distance = mochi::Norm(delta);
    WriteRow(out, "Delta", FormatVector(delta, unit));
    WriteRow(out, "Distance", FormatLength(distance, unit));
    annotations.push_back({from, to, FormatLength(distance, unit), MeasureAnnotation::Style::Span});
    AppendComponentLegs(from, to, unit, annotations);
    return;
  }

  if (a.IsFace() != b.IsFace()) {
    MeasureEntity const& face = a.IsFace() ? a : b;
    MeasureEntity const& vertex = a.IsFace() ? b : a;
    Real3 const point = vertex.points.front();
    real const signedPlaneDistance = mochi::Dot(point - face.centroid, face.normal);
    Real3 const projected = point - face.normal * signedPlaneDistance;
    auto const [closest, closestDistance] = ClosestPointOnFace(face, point);
    WriteRow(out, "Plane dist", FormatLength(signedPlaneDistance, unit));
    WriteRow(out, "Face dist", FormatLength(closestDistance, unit));
    WriteRow(out, "Closest pt", FormatVector(closest, unit));
    annotations.push_back(
        {point,
         projected,
         FormatLength(signedPlaneDistance, unit),
         MeasureAnnotation::Style::Span});
    if (mochi::NormSqr(closest - projected) > 0) {
      annotations.push_back(
          {point, closest, FormatLength(closestDistance, unit), MeasureAnnotation::Style::Span});
    }
    return;
  }

  real const angleDeg = AngleBetweenDeg(a.normal, b.normal);
  real const offParallelDeg = mochi::Min(angleDeg, real(180) - angleDeg);
  WriteRow(out, "Normal angle", FormatScalar(angleDeg, 3) + " deg");
  if (offParallelDeg <= params.parallelAngleToleranceDeg) {
    real const signedDistance = mochi::Dot(b.centroid - a.centroid, a.normal);
    Real3 const projected = b.centroid - a.normal * signedDistance;
    WriteRow(out, "Parallel", "yes");
    WriteRow(out, "Plane gap", FormatLength(mochi::Abs(signedDistance), unit));
    annotations.push_back(
        {b.centroid,
         projected,
         FormatLength(mochi::Abs(signedDistance), unit),
         MeasureAnnotation::Style::Span});
  } else {
    WriteRow(out, "Parallel", "no");
    annotations.push_back(
        {a.centroid,
         b.centroid,
         FormatScalar(angleDeg, 3) + " deg",
         MeasureAnnotation::Style::Span});
  }
}

} // namespace

//--------------------------------------------------------------------------------------------------
// OPTIONS
//--------------------------------------------------------------------------------------------------

char const* GetMeasureUnitSuffix(MeasureUnit unit) {
  switch (unit) {
    case MeasureUnit::Meters:
      return "m";
    case MeasureUnit::Centimeters:
      return "cm";
    case MeasureUnit::Millimeters:
      return "mm";
  }
  return "m";
}

real GetMeasureUnitScale(MeasureUnit unit) {
  switch (unit) {
    case MeasureUnit::Meters:
      return real(1);
    case MeasureUnit::Centimeters:
      return real(100);
    case MeasureUnit::Millimeters:
      return real(1000);
  }
  return real(1);
}

char const* GetMeasureTargetKindLabel(MeasureTargetKind kind) {
  switch (kind) {
    case MeasureTargetKind::Render:
      return "Render";
    case MeasureTargetKind::Collision:
      return "Collision";
  }
  return "Render";
}

//--------------------------------------------------------------------------------------------------
// MESH
//--------------------------------------------------------------------------------------------------

std::shared_ptr<MeasureMesh const> BuildMeasureMesh(
    mochi::Span<MeasureMeshSection const> sections,
    MeasureMeshParams const& params) {
  std::vector<Real3> positions;
  std::vector<int> indices;
  for (MeasureMeshSection const& section : sections) {
    if (section.positions.size() < 9 || section.indices.size() < 3) {
      continue;
    }
    int const offset = static_cast<int>(positions.size());
    for (std::size_t i = 0; i + 2 < section.positions.size(); i += 3) {
      positions.emplace_back(
          real(section.positions[i]),
          real(section.positions[i + 1]),
          real(section.positions[i + 2]));
    }
    int const sectionVertices = static_cast<int>(positions.size()) - offset;
    for (std::size_t i = 0; i + 2 < section.indices.size(); i += 3) {
      // An out-of-range index would silently address another section's vertices once merged; drop
      // the triangle instead.
      if (section.indices[i] < 0 || section.indices[i + 1] < 0 || section.indices[i + 2] < 0 ||
          section.indices[i] >= sectionVertices || section.indices[i + 1] >= sectionVertices ||
          section.indices[i + 2] >= sectionVertices) {
        continue;
      }
      indices.push_back(offset + section.indices[i]);
      indices.push_back(offset + section.indices[i + 1]);
      indices.push_back(offset + section.indices[i + 2]);
    }
  }
  return FinishMeasureMesh(positions, indices, params);
}

std::shared_ptr<MeasureMesh const> BuildMeasureMesh(
    mochi::MeshDataView const& mesh,
    MeasureMeshParams const& params) {
  if (mesh.nodesPerElement != 3) {
    return std::make_shared<MeasureMesh>();
  }
  std::vector<Real3> positions;
  positions.reserve(static_cast<std::size_t>(mesh.GetNumNodes()));
  for (int node = 0; node < mesh.GetNumNodes(); ++node) {
    positions.emplace_back(
        mesh.coordinates[node * 3 + 0],
        mesh.coordinates[node * 3 + 1],
        mesh.coordinates[node * 3 + 2]);
  }
  std::vector<int> const indices(mesh.connectivity.begin(), mesh.connectivity.end());
  return FinishMeasureMesh(positions, indices, params);
}

std::optional<MeasureRayHit>
RayCastMeasureMesh(MeasureMesh const& mesh, Real3 const& origin, Real3 const& direction) {
  if (mesh.IsEmpty() || !mesh.bvh) {
    return std::nullopt;
  }
  mochi::Ray const ray(origin, direction);
  std::optional<mochi::RayHit> const hit =
      mochi::RayCast(ray, *mesh.bvh, [&mesh](mochi::Ray const& r, int index) {
        return IntersectTriangle(mesh, index, r);
      });
  if (!hit) {
    return std::nullopt;
  }

  MeasureRayHit result;
  result.triangle = hit->index;
  result.facet = mesh.triangleFacets[static_cast<std::size_t>(hit->index)];
  result.t = hit->t;
  result.point = mochi::ToReal3(hit->intersection);
  mochi::Vec4r const barycentric = mochi::BarycentricCoordinates(
      mesh.soup.triangles[static_cast<std::size_t>(hit->index)], hit->intersection);
  int const corner = static_cast<int>(mochi::ArgMax(mochi::ToReal3(barycentric)));
  result.nearestVertex = mesh.triangles[static_cast<std::size_t>(hit->index * 3 + corner)];
  return result;
}

//--------------------------------------------------------------------------------------------------
// ENTITIES
//--------------------------------------------------------------------------------------------------

Real3 MeasureEntity::GetAnchor() const {
  if (IsFace()) {
    return centroid;
  }
  return points.empty() ? Real3{0_r, 0_r, 0_r} : points.front();
}

std::string MeasureEntity::GetDescription() const {
  switch (kind) {
    case MeasureEntityKind::Vertex:
      return "Vertex";
    case MeasureEntityKind::Triangle:
      return "Triangle";
    case MeasureEntityKind::Facet:
      return "Facet (" + std::to_string(triangles.size() / 3) + " tris)";
  }
  return "Vertex";
}

MeasureEntity MakeMeasureEntity(
    MeasureMesh const& mesh,
    MeasureRayHit const& hit,
    MeasurePickMode pickMode,
    std::string targetLabel,
    MeasureTargetKind targetKind,
    MeasureLocalToMeasureFn const& localToMeasure) {
  MeasureEntity entity;
  entity.targetLabel = std::move(targetLabel);
  entity.targetKind = targetKind;

  if (pickMode == MeasurePickMode::Vertex) {
    entity.kind = MeasureEntityKind::Vertex;
    entity.points.push_back(
        localToMeasure(mesh.vertices[static_cast<std::size_t>(hit.nearestVertex)]));
    return entity;
  }

  bool const wholeFacet = pickMode == MeasurePickMode::Facet && hit.facet >= 0;
  entity.kind = wholeFacet ? MeasureEntityKind::Facet : MeasureEntityKind::Triangle;
  std::vector<int> const singleTriangle{hit.triangle};
  std::vector<int> const& sourceTriangles =
      wholeFacet ? mesh.facets[static_cast<std::size_t>(hit.facet)] : singleTriangle;

  // Re-index the selected triangles onto their own compact corner list, so the entity is
  // self-contained once the mesh it came from is gone.
  std::unordered_map<int, int> vertexRemap;
  auto localIndexOf = [&](int meshVertex) {
    auto const [it, inserted] =
        vertexRemap.try_emplace(meshVertex, static_cast<int>(entity.points.size()));
    if (inserted) {
      entity.points.push_back(localToMeasure(mesh.vertices[static_cast<std::size_t>(meshVertex)]));
    }
    return it->second;
  };
  for (int const triangle : sourceTriangles) {
    for (int corner = 0; corner < 3; ++corner) {
      entity.triangles.push_back(
          localIndexOf(mesh.triangles[static_cast<std::size_t>(triangle * 3 + corner)]));
    }
  }

  // Area, centroid and normal are recomputed in measurement space rather than transformed from the
  // mesh: the object's world transform may scale non-uniformly, which changes all three.
  Real3 weightedCentroid{0_r, 0_r, 0_r};
  Real3 weightedNormal{0_r, 0_r, 0_r};
  for (std::size_t i = 0; i + 2 < entity.triangles.size(); i += 3) {
    Real3 const& a = entity.points[static_cast<std::size_t>(entity.triangles[i + 0])];
    Real3 const& b = entity.points[static_cast<std::size_t>(entity.triangles[i + 1])];
    Real3 const& c = entity.points[static_cast<std::size_t>(entity.triangles[i + 2])];
    Real3 const cross = mochi::Cross(b - a, c - a);
    real const area = real(0.5) * mochi::Norm(cross);
    entity.area += area;
    weightedCentroid += (a + b + c) * (area / real(3));
    weightedNormal += cross * real(0.5);
  }
  if (entity.area > 0) {
    entity.centroid = weightedCentroid / entity.area;
  } else if (!entity.points.empty()) {
    entity.centroid = entity.points.front();
  }
  real const normalNorm = mochi::Norm(weightedNormal);
  entity.normal = normalNorm > 0 ? weightedNormal / normalNorm : Real3{0_r, 0_r, 1_r};

  // Outline: an edge used by exactly one of the selected triangles.
  std::unordered_map<int64_t, int> edgeUseCount;
  for (std::size_t i = 0; i + 2 < entity.triangles.size(); i += 3) {
    for (int e = 0; e < 3; ++e) {
      ++edgeUseCount[EdgeKey(
          entity.triangles[i + static_cast<std::size_t>(e)],
          entity.triangles[i + static_cast<std::size_t>((e + 1) % 3)])];
    }
  }
  for (std::size_t i = 0; i + 2 < entity.triangles.size(); i += 3) {
    for (int e = 0; e < 3; ++e) {
      int const from = entity.triangles[i + static_cast<std::size_t>(e)];
      int const to = entity.triangles[i + static_cast<std::size_t>((e + 1) % 3)];
      if (edgeUseCount[EdgeKey(from, to)] == 1) {
        entity.boundaryEdges.push_back(from);
        entity.boundaryEdges.push_back(to);
      }
    }
  }
  return entity;
}

//--------------------------------------------------------------------------------------------------
// MEASUREMENTS
//--------------------------------------------------------------------------------------------------

MeasurementReport ComputeMeasurements(
    std::vector<MeasureEntity> const& entities,
    MeasureReportParams const& params) {
  MeasurementReport report;
  if (entities.empty()) {
    return report;
  }
  MeasureUnit const unit = params.unit;
  std::ostringstream out;

  for (std::size_t i = 0; i < entities.size(); ++i) {
    MeasureEntity const& entity = entities[i];
    out << '[' << (i + 1) << "] " << entity.GetDescription() << "  " << entity.targetLabel << " ("
        << GetMeasureTargetKindLabel(entity.targetKind) << ")\n";
    WriteEntityDetails(out, entity, params);
    out << '\n';
  }

  if (entities.size() == 2) {
    out << "Pair [1] -> [2]\n";
    WritePairMeasurement(out, report.annotations, entities[0], entities[1], params);
    out << '\n';
  }

  if (entities.size() >= 2) {
    TaggedPoints const tagged = CollectPoints(entities);
    Bounds const bounds = ComputeBounds(tagged.points);
    Real3 const extents = bounds.GetExtents();
    out << "Selection (" << entities.size() << " items, " << tagged.points.size() << " points)\n";
    WriteRow(out, "Extents", FormatVector(extents, unit));
    WriteRow(out, "Diagonal", FormatLength(mochi::Norm(extents), unit));
    if (static_cast<int>(tagged.points.size()) <= params.maxPairwisePoints) {
      ExtremePair maxPair;
      ExtremePair minPair;
      FindExtremePairs(tagged, maxPair, minPair);
      if (maxPair.valid) {
        WriteRow(out, "Max distance", FormatLength(maxPair.distance, unit));
        WriteRow(out, "Min distance", FormatLength(minPair.distance, unit));
        // For exactly two entities the pair section above already draws the measurement.
        if (entities.size() > 2) {
          report.annotations.push_back(
              {maxPair.a,
               maxPair.b,
               FormatLength(maxPair.distance, unit),
               MeasureAnnotation::Style::Span});
        }
      }
    } else {
      WriteRow(out, "Max distance", "(skipped: too many points)");
      WriteRow(out, "Min distance", "(skipped: too many points)");
    }
  }

  report.text = out.str();
  return report;
}

} // namespace superdex::studio
