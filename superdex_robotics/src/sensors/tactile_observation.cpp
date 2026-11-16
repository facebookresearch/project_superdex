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

#include <superdex_robotics/sensors/tactile_observation.h>

#include <mochi_core/integration/integration_utils.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/math_utils.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>

using namespace mochi;
using namespace superdex::robotics;

namespace {

template <typename Scalar>
void FlattenTactileTaxelForces(Span<Real3 const> forces, Span<Scalar> out) {
  for (std::size_t vertex = 0; vertex < forces.size(); ++vertex) {
    for (std::size_t component = 0; component < 3; ++component) {
      out[3 * vertex + component] = static_cast<Scalar>(forces[vertex][component]);
    }
  }
}

void ValidateLayoutPositions(Span<Real3 const> positions, Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(positions.empty(), error, "A tactile taxel layout must not be empty.");
  MOCHI_ERROR_RETURN(error);
  for (Real3 const& position : positions) {
    MOCHI_ERROR_IF(!IsFinite(position), error, "Tactile taxel positions must be finite.");
    MOCHI_ERROR_RETURN(error);
  }
}

Real2 ConcentricDiskToSquare(real x, real y) {
  constexpr real kPi = std::numbers::pi_v<real>;
  real const radius = std::sqrt(x * x + y * y);
  if (radius == 0_r) {
    return {};
  }

  real angle = std::atan2(y, x);
  if (angle < 0_r) {
    angle += 2_r * kPi;
  }
  if (angle < kPi / 4_r || angle >= 7_r * kPi / 4_r) {
    real const adjustedAngle = angle < kPi / 4_r ? angle : angle - 2_r * kPi;
    real const squareX = radius;
    return {squareX, squareX * 4_r * adjustedAngle / kPi};
  }
  if (angle < 3_r * kPi / 4_r) {
    real const squareY = radius;
    return {squareY * (2_r - 4_r * angle / kPi), squareY};
  }
  if (angle < 5_r * kPi / 4_r) {
    real const squareX = -radius;
    return {squareX, squareX * (4_r * angle / kPi - 4_r)};
  }
  real const squareY = -radius;
  return {squareY * (6_r - 4_r * angle / kPi), squareY};
}

real NonzeroSign(real value) {
  return value < 0_r ? -1_r : 1_r;
}

Real2 ProjectDirection(
    Real3 const& direction,
    Real3 const& tip,
    Real3 const& zeroLongitude,
    Real3 const& positiveLongitude,
    real polarAngle,
    real radialScale,
    TactileHemisphereRectParams const& params) {
  real const x = Dot(direction, zeroLongitude);
  real const y = Dot(direction, positiveLongitude);
  real const z = Dot(direction, tip);
  switch (params.projection) {
    case TactileRectProjectionType::AzimuthalEqualArea: {
      real const normalizedRadius = std::sin(polarAngle / 2_r) * radialScale;
      real const longitude = std::atan2(y, x);
      return ConcentricDiskToSquare(
          normalizedRadius * std::cos(longitude), normalizedRadius * std::sin(longitude));
    }
    case TactileRectProjectionType::AzimuthalEquidistant: {
      real const normalizedRadius = polarAngle * radialScale;
      real const longitude = std::atan2(y, x);
      return ConcentricDiskToSquare(
          normalizedRadius * std::cos(longitude), normalizedRadius * std::sin(longitude));
    }
    case TactileRectProjectionType::Octahedral: {
      real const denominator = std::abs(x) + std::abs(y) + std::abs(z);
      if (denominator == 0_r) {
        return {};
      }
      real projectedX = x / denominator;
      real projectedY = y / denominator;
      if (z < 0_r) {
        real const upperX = projectedX;
        real const upperY = projectedY;
        projectedX = (1_r - std::abs(upperY)) * NonzeroSign(upperX);
        projectedY = (1_r - std::abs(upperX)) * NonzeroSign(upperY);
      }
      return {projectedX, projectedY};
    }
  }
  return {};
}

struct ProjectedTaxelNode {
  int taxel = -1;
  int left = -1;
  int right = -1;
  int axis = 0;
};

class ProjectedTaxelTree {
 public:
  ProjectedTaxelTree(
      DynamicArray<Real2> const& coordinates,
      DynamicArray<int32_t> const& taxelCells)
      : _coordinates(coordinates) {
    for (int iTaxel = 0; iTaxel < isize(taxelCells); ++iTaxel) {
      if (taxelCells[iTaxel] >= 0) {
        _indices.push_back(iTaxel);
      }
    }
    _nodes.reserve(_indices.size());
    _root = Build(0, static_cast<int>(_indices.size()), 0);
  }

  [[nodiscard]] int FindNearest(Real2 const& query) const {
    int bestTaxel = -1;
    real bestDistanceSquared = std::numeric_limits<real>::infinity();
    Search(_root, query, bestTaxel, bestDistanceSquared);
    return bestTaxel;
  }

 private:
  int Build(int begin, int end, int depth) {
    if (begin >= end) {
      return -1;
    }
    int const axis = depth % 2;
    int const middle = begin + (end - begin) / 2;
    std::nth_element(
        _indices.begin() + begin,
        _indices.begin() + middle,
        _indices.begin() + end,
        [this, axis](int left, int right) {
          Real2 const& leftCoordinate = _coordinates[left];
          Real2 const& rightCoordinate = _coordinates[right];
          if (leftCoordinate[axis] != rightCoordinate[axis]) {
            return leftCoordinate[axis] < rightCoordinate[axis];
          }
          if (leftCoordinate[1 - axis] != rightCoordinate[1 - axis]) {
            return leftCoordinate[1 - axis] < rightCoordinate[1 - axis];
          }
          return left < right;
        });
    int const node = static_cast<int>(_nodes.size());
    _nodes.push_back({.taxel = _indices[middle], .axis = axis});
    int const left = Build(begin, middle, depth + 1);
    int const right = Build(middle + 1, end, depth + 1);
    _nodes[node].left = left;
    _nodes[node].right = right;
    return node;
  }

  void Search(int nodeIndex, Real2 const& query, int& bestTaxel, real& bestDistanceSquared) const {
    if (nodeIndex < 0) {
      return;
    }
    ProjectedTaxelNode const& node = _nodes[nodeIndex];
    Real2 const& coordinate = _coordinates[node.taxel];
    Real2 const delta = coordinate - query;
    real const distanceSquared = Dot(delta, delta);
    if (bestTaxel < 0 || distanceSquared < bestDistanceSquared ||
        (distanceSquared == bestDistanceSquared && node.taxel < bestTaxel)) {
      bestTaxel = node.taxel;
      bestDistanceSquared = distanceSquared;
    }

    real const splitDelta = query[node.axis] - coordinate[node.axis];
    int const nearNode = splitDelta <= 0_r ? node.left : node.right;
    int const farNode = splitDelta <= 0_r ? node.right : node.left;
    Search(nearNode, query, bestTaxel, bestDistanceSquared);
    if (splitDelta * splitDelta <= bestDistanceSquared) {
      Search(farNode, query, bestTaxel, bestDistanceSquared);
    }
  }

  DynamicArray<Real2> const& _coordinates;
  DynamicArray<int32_t> _indices;
  DynamicArray<ProjectedTaxelNode> _nodes;
  int _root = -1;
};

void BuildNearestCellContributions(TactileRectProjectionMap& projection) {
  projection.taxelContributionOffsets.resize(projection.taxelCells.size() + 1);
  for (int iTaxel = 0; iTaxel < isize(projection.taxelCells); ++iTaxel) {
    projection.taxelContributionOffsets[iTaxel] = isize(projection.contributionCells);
    if (projection.taxelCells[iTaxel] >= 0) {
      projection.contributionCells.push_back(projection.taxelCells[iTaxel]);
      projection.contributionWeights.push_back(1_r);
    }
  }
  projection.taxelContributionOffsets.back() = isize(projection.contributionCells);
}

void BuildConservativeVoronoiContributions(TactileRectProjectionMap& projection, Error& error) {
  int64_t const fineWidth64 =
      static_cast<int64_t>(projection.width) * projection.voronoiOversampling;
  int64_t const fineHeight64 =
      static_cast<int64_t>(projection.height) * projection.voronoiOversampling;
  MOCHI_ERROR_IF(
      fineWidth64 > std::numeric_limits<int>::max() ||
          fineHeight64 > std::numeric_limits<int>::max(),
      error,
      "The tactile Voronoi support grid dimensions overflowed.");
  MOCHI_ERROR_RETURN(error);
  int64_t const fineCellCount64 = fineWidth64 * fineHeight64;
  MOCHI_ERROR_IF(
      fineCellCount64 > std::numeric_limits<int>::max(),
      error,
      "The tactile Voronoi support grid cell count overflowed.");
  MOCHI_ERROR_RETURN(error);
  int const fineWidth = static_cast<int>(fineWidth64);
  int const fineHeight = static_cast<int>(fineHeight64);

  ProjectedTaxelTree const tree(projection.coordinates, projection.taxelCells);
  DynamicArray<DynamicArray<int32_t>> ownedCells;
  ownedCells.resize(projection.taxelCells.size());
  for (int fineRow = 0; fineRow < fineHeight; ++fineRow) {
    for (int fineColumn = 0; fineColumn < fineWidth; ++fineColumn) {
      Real2 const query = {
          -1_r + 2_r * (static_cast<real>(fineColumn) + 0.5_r) / fineWidth,
          1_r - 2_r * (static_cast<real>(fineRow) + 0.5_r) / fineHeight};
      int const owner = tree.FindNearest(query);
      MOCHI_ASSERT(owner >= 0, "A tactile Voronoi support cell has no owner");
      int const row = fineRow / projection.voronoiOversampling;
      int const column = fineColumn / projection.voronoiOversampling;
      ownedCells[owner].push_back(row * projection.width + column);
    }
  }

  projection.taxelContributionOffsets.resize(projection.taxelCells.size() + 1);
  for (int iTaxel = 0; iTaxel < isize(projection.taxelCells); ++iTaxel) {
    projection.taxelContributionOffsets[iTaxel] = isize(projection.contributionCells);
    if (projection.taxelCells[iTaxel] < 0) {
      continue;
    }
    DynamicArray<int32_t>& cells = ownedCells[iTaxel];
    if (cells.empty()) {
      projection.contributionCells.push_back(projection.taxelCells[iTaxel]);
      projection.contributionWeights.push_back(1_r);
      continue;
    }
    std::ranges::sort(cells);
    int runBegin = 0;
    while (runBegin < static_cast<int>(cells.size())) {
      int runEnd = runBegin + 1;
      while (runEnd < static_cast<int>(cells.size()) && cells[runEnd] == cells[runBegin]) {
        ++runEnd;
      }
      projection.contributionCells.push_back(cells[runBegin]);
      projection.contributionWeights.push_back(
          static_cast<real>(runEnd - runBegin) / static_cast<real>(cells.size()));
      runBegin = runEnd;
    }
  }
  projection.taxelContributionOffsets.back() = isize(projection.contributionCells);
}

} // namespace

void superdex::robotics::FlattenTactileTaxelForcesInto(
    TactileTaxelForces const& forces,
    Span<float> out,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF_NOT(
      out.size() % 3 == 0 && out.size() / 3 == forces.forces.size(),
      error,
      "A flattened tactile force buffer must have shape [N, 3].");
  MOCHI_ERROR_RETURN(error);
  FlattenTactileTaxelForces(Span<Real3 const>{forces.forces}, out);
}

TactileTaxelLayout superdex::robotics::MakeTactileTaxelLayout(
    Span<Real3 const> positions,
    Error& error) {
  TactileTaxelLayout layout;
  ValidateLayoutPositions(positions, error);
  MOCHI_ERROR_RETURN(error, layout);
  layout.positions.assign(positions.begin(), positions.end());
  return layout;
}

TactileTaxelForces superdex::robotics::ScatterTactileTaxelForces(
    Span<NodeContactForce const> nodeForces,
    TransformRT const& worldFromSensor,
    TactileTaxelLayout const& layout,
    Error& error) {
  TactileTaxelForces result;
  result.forces.resize(layout.positions.size());
  ScatterTactileTaxelForcesInto(nodeForces, worldFromSensor, layout, result, error);
  if (!error.IsOK()) {
    return {};
  }
  return result;
}

void superdex::robotics::ScatterTactileTaxelForcesInto(
    Span<NodeContactForce const> nodeForces,
    TransformRT const& worldFromSensor,
    TactileTaxelLayout const& layout,
    TactileTaxelForces& out,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(layout.positions.empty(), error, "A valid tactile taxel layout is required.");
  ValidateLayoutPositions(layout.positions, error);
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF_NOT(
      out.forces.size() == layout.positions.size(),
      error,
      "A preallocated tactile observation must match its layout size.");
  MOCHI_ERROR_RETURN(error);

  std::ranges::fill(out.forces, Real3{});
  for (NodeContactForce const& nodeForce : nodeForces) {
    MOCHI_ERROR_IF(
        nodeForce.index < 0 || nodeForce.index >= isize(out.forces),
        error,
        "A tactile node-force index is outside the configured layout.");
    MOCHI_ERROR_RETURN(error);
    MOCHI_ERROR_IF(!IsFinite(nodeForce.force), error, "Tactile node forces must be finite.");
    MOCHI_ERROR_RETURN(error);
    Real3 const force = worldFromSensor.TransformDirectionInverse(nodeForce.force);
    MOCHI_ERROR_IF(!IsFinite(force), error, "Transformed tactile forces must be finite.");
    MOCHI_ERROR_RETURN(error);
    out.forces[nodeForce.index] += force;
    MOCHI_ERROR_IF(
        !IsFinite(out.forces[nodeForce.index]),
        error,
        "Accumulated tactile node forces overflowed.");
    MOCHI_ERROR_RETURN(error);
  }
}

TactileTaxelForces superdex::robotics::ReadTactileTaxelForces(
    Actor const* actor,
    TransformRT const& worldFromSensor,
    TactileTaxelLayout const& layout,
    Error& error) {
  TactileTaxelForces result;
  result.forces.resize(layout.positions.size());
  ReadTactileTaxelForcesInto(actor, worldFromSensor, layout, result, error);
  if (!error.IsOK()) {
    return {};
  }
  return result;
}

void superdex::robotics::ReadTactileTaxelForcesInto(
    Actor const* actor,
    TransformRT const& worldFromSensor,
    TactileTaxelLayout const& layout,
    TactileTaxelForces& out,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  if (actor == nullptr) {
    MOCHI_ERROR_SET(error, "A tactile taxel-force reader requires an actor.");
    return;
  }
  MOCHI_ERROR_IF(
      !actor->IsQuerySupported(QueryType::NodeContactForces),
      error,
      "The actor does not support node contact-force queries.");
  MOCHI_ERROR_RETURN(error);
  MeshDataView const contactSkin = actor->GetContactSkinMesh();
  MeshDataView const mesh = actor->GetMesh();
  int const layoutSize = isize(layout.positions);
  MeshDataView const& contactMesh = contactSkin.IsEmpty() ? mesh : contactSkin;
  MOCHI_ERROR_IF(
      contactMesh.IsEmpty() || contactMesh.GetNumNodes() != layoutSize,
      error,
      "The tactile taxel layout size does not match the actor contact mesh.");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF_NOT(
      EndsAtLastStage(actor->GetScene()->GetSolverParams().integrationMethod),
      error,
      "Tactile contact forces describe the last integration stage; select an integration method "
      "that ends its step at that stage.");
  MOCHI_ERROR_RETURN(error);
  Span<NodeContactForce const> const nodeForces = actor->GetNodeContactForcesWorld(error);
  ScatterTactileTaxelForcesInto(nodeForces, worldFromSensor, layout, out, error);
}

TactileAggregateForce superdex::robotics::AggregateTactileForce(
    TactileTaxelForces const& taxelForces,
    Error& error) {
  TactileAggregateForce result;
  MOCHI_ERROR_IF(
      taxelForces.forces.empty(), error, "A valid tactile force observation is required.");
  MOCHI_ERROR_RETURN(error, result);
  for (Real3 const& force : taxelForces.forces) {
    MOCHI_ERROR_IF(!IsFinite(force), error, "Tactile taxel forces must be finite.");
    MOCHI_ERROR_RETURN(error, {});
    result.force += force;
    MOCHI_ERROR_IF(!IsFinite(result.force), error, "The aggregate tactile force overflowed.");
    MOCHI_ERROR_RETURN(error, {});
  }
  return result;
}

TactileAggregateWrench superdex::robotics::AggregateTactileWrench(
    TactileTaxelLayout const& layout,
    TactileTaxelForces const& taxelForces,
    Real3 const& referencePoint,
    Error& error) {
  TactileAggregateWrench result;
  MOCHI_ERROR_IF(!IsFinite(referencePoint), error, "The wrench reference point must be finite.");
  MOCHI_ERROR_RETURN(error, result);
  MOCHI_ERROR_IF(layout.positions.empty(), error, "A valid tactile taxel layout is required.");
  MOCHI_ERROR_RETURN(error, result);
  MOCHI_ERROR_IF(
      taxelForces.forces.size() != layout.positions.size(),
      error,
      "The tactile layout and force observation do not match.");
  MOCHI_ERROR_RETURN(error, result);

  result.referencePoint = referencePoint;
  for (int iTaxel = 0; iTaxel < isize(layout.positions); ++iTaxel) {
    Real3 const position = layout.positions[iTaxel];
    Real3 const force = taxelForces.forces[iTaxel];
    MOCHI_ERROR_IF(!IsFinite(position), error, "Tactile taxel positions must be finite.");
    MOCHI_ERROR_RETURN(error, {});
    MOCHI_ERROR_IF(!IsFinite(force), error, "Tactile taxel forces must be finite.");
    MOCHI_ERROR_RETURN(error, {});
    result.force += force;
    result.torque += Cross(position - referencePoint, force);
    MOCHI_ERROR_IF(
        !IsFinite(result.force) || !IsFinite(result.torque),
        error,
        "The aggregate tactile wrench overflowed.");
    MOCHI_ERROR_RETURN(error, {});
  }
  return result;
}

TactileRectProjectionMap superdex::robotics::BuildTactileRectProjection(
    TactileTaxelLayout const& layout,
    TactileHemisphereRectParams const& params,
    Error& error) {
  TactileRectProjectionMap result;
  MOCHI_ERROR_IF(
      params.width < 1 || params.height < 1,
      error,
      "A tactile force grid must have positive width and height.");
  MOCHI_ERROR_RETURN(error, result);
  constexpr real kPi = std::numbers::pi_v<real>;
  MOCHI_ERROR_IF(
      !std::isfinite(params.maxPolarLatitude) || params.maxPolarLatitude <= 0_r ||
          params.maxPolarLatitude >= kPi,
      error,
      "The maximum polar angle must be in (0, pi).");
  MOCHI_ERROR_RETURN(error, result);
  MOCHI_ERROR_IF(
      params.projection != TactileRectProjectionType::AzimuthalEqualArea &&
          params.projection != TactileRectProjectionType::AzimuthalEquidistant &&
          params.projection != TactileRectProjectionType::Octahedral,
      error,
      "The tactile rectangular projection type is invalid.");
  MOCHI_ERROR_RETURN(error, result);
  MOCHI_ERROR_IF(
      params.rasterization != TactileRectRasterizationType::ConservativeVoronoi &&
          params.rasterization != TactileRectRasterizationType::NearestCell,
      error,
      "The tactile rectangular rasterization type is invalid.");
  MOCHI_ERROR_RETURN(error, result);
  MOCHI_ERROR_IF(
      params.voronoiOversampling < 1 || params.voronoiOversampling > 16,
      error,
      "Tactile Voronoi oversampling must be in [1, 16].");
  MOCHI_ERROR_RETURN(error, result);
  MOCHI_ERROR_IF(
      params.outsidePolicy != TactileRectOutsidePolicy::Reject &&
          params.outsidePolicy != TactileRectOutsidePolicy::Exclude,
      error,
      "The tactile projection outside policy is invalid.");
  MOCHI_ERROR_RETURN(error, result);
  MOCHI_ERROR_IF(
      !IsFinite(params.center) || !IsFinite(params.tipDirection) ||
          !IsFinite(params.zeroLongitudeDirection),
      error,
      "The tactile hemisphere frame must be finite.");
  MOCHI_ERROR_RETURN(error, result);
  MOCHI_ERROR_IF(layout.positions.empty(), error, "A valid tactile taxel layout is required.");
  MOCHI_ERROR_RETURN(error, result);

  real const tipNorm = Norm(params.tipDirection);
  MOCHI_ERROR_IF(tipNorm <= 0_r, error, "The tactile tip direction must be nonzero.");
  MOCHI_ERROR_RETURN(error, result);
  Real3 const tip = params.tipDirection * (1_r / tipNorm);
  Real3 zeroLongitude =
      params.zeroLongitudeDirection - Dot(params.zeroLongitudeDirection, tip) * tip;
  real const zeroLongitudeNorm = Norm(zeroLongitude);
  MOCHI_ERROR_IF(
      zeroLongitudeNorm <= 0_r,
      error,
      "The zero-longitude direction must not be parallel to the tip direction.");
  MOCHI_ERROR_RETURN(error, result);
  zeroLongitude *= 1_r / zeroLongitudeNorm;
  Real3 const positiveLongitude = Cross(tip, zeroLongitude);

  int64_t const numCells64 =
      static_cast<int64_t>(params.width) * static_cast<int64_t>(params.height);
  MOCHI_ERROR_IF(
      numCells64 > std::numeric_limits<int>::max(),
      error,
      "The tactile force-grid dimensions overflowed.");
  MOCHI_ERROR_RETURN(error, result);
  int const numCells = static_cast<int>(numCells64);
  result.coordinates.resize(layout.positions.size());
  result.taxelCells.resize(layout.positions.size(), -1);
  result.width = params.width;
  result.height = params.height;
  result.rasterization = params.rasterization;
  result.voronoiOversampling =
      params.rasterization == TactileRectRasterizationType::ConservativeVoronoi
      ? params.voronoiOversampling
      : 1;

  constexpr real kDirectionEpsilon = 1e-12_r;
  constexpr real kLatitudeTolerance = 1e-5_r;
  real radialScale = 1_r;
  if (params.projection == TactileRectProjectionType::AzimuthalEqualArea) {
    radialScale = 1_r / std::sin(params.maxPolarLatitude / 2_r);
  } else if (params.projection == TactileRectProjectionType::AzimuthalEquidistant) {
    radialScale = 1_r / params.maxPolarLatitude;
  }
  for (int iTaxel = 0; iTaxel < isize(layout.positions); ++iTaxel) {
    Real3 const position = layout.positions[iTaxel];
    MOCHI_ERROR_IF(!IsFinite(position), error, "Tactile taxel positions must be finite.");
    MOCHI_ERROR_RETURN(error, {});
    Real3 const radial = position - params.center;
    real const radius = Norm(radial);
    MOCHI_ERROR_IF(
        radius <= kDirectionEpsilon,
        error,
        "A tactile taxel cannot coincide with the hemisphere center.");
    MOCHI_ERROR_RETURN(error, {});
    Real3 const direction = radial * (1_r / radius);
    real const tipCosine = std::clamp(Dot(direction, tip), -1_r, 1_r);
    real const polarLatitude = std::acos(tipCosine);
    if (polarLatitude > params.maxPolarLatitude + kLatitudeTolerance) {
      MOCHI_ERROR_IF(
          params.outsidePolicy == TactileRectOutsidePolicy::Reject,
          error,
          "A tactile taxel lies outside the configured spherical cap.");
      MOCHI_ERROR_RETURN(error, {});
      continue;
    }

    Real2 coordinate = ProjectDirection(
        direction,
        tip,
        zeroLongitude,
        positiveLongitude,
        std::min(polarLatitude, params.maxPolarLatitude),
        radialScale,
        params);
    constexpr real kCoordinateTolerance = 1e-5_r;
    MOCHI_ERROR_IF(
        !std::isfinite(coordinate[0]) || !std::isfinite(coordinate[1]) ||
            std::abs(coordinate[0]) > 1_r + kCoordinateTolerance ||
            std::abs(coordinate[1]) > 1_r + kCoordinateTolerance,
        error,
        "A tactile projection produced invalid normalized coordinates.");
    MOCHI_ERROR_RETURN(error, {});
    coordinate[0] = std::clamp(coordinate[0], -1_r, 1_r);
    coordinate[1] = std::clamp(coordinate[1], -1_r, 1_r);
    real const continuousColumn =
        (coordinate[0] + 1_r) * 0.5_r * static_cast<real>(params.width - 1);
    real const continuousRow = (1_r - coordinate[1]) * 0.5_r * static_cast<real>(params.height - 1);
    int const column =
        std::clamp(static_cast<int>(std::floor(continuousColumn + 0.5_r)), 0, params.width - 1);
    int const row =
        std::clamp(static_cast<int>(std::floor(continuousRow + 0.5_r)), 0, params.height - 1);
    int64_t const cell64 = static_cast<int64_t>(row) * static_cast<int64_t>(params.width) + column;
    int const cell = static_cast<int>(cell64);
    MOCHI_ASSERT(cell >= 0 && cell < numCells, "Resolved tactile projection cell is invalid");
    result.coordinates[iTaxel] = coordinate;
    result.taxelCells[iTaxel] = cell;
    ++result.selectedTaxelCount;
  }
  MOCHI_ERROR_IF(
      result.selectedTaxelCount == 0, error, "The tactile projection selected no source taxels.");
  MOCHI_ERROR_RETURN(error, {});
  if (result.rasterization == TactileRectRasterizationType::ConservativeVoronoi) {
    BuildConservativeVoronoiContributions(result, error);
    MOCHI_ERROR_RETURN(error, {});
  } else {
    BuildNearestCellContributions(result);
  }
  return result;
}

TactileForceGrid superdex::robotics::RasterizeTactileRectProjection(
    TactileRectProjectionMap const& projection,
    TactileTaxelForces const& taxelForces,
    Error& error) {
  TactileForceGrid result;
  int64_t const numCells64 =
      static_cast<int64_t>(projection.width) * static_cast<int64_t>(projection.height);
  if (projection.width > 0 && projection.height > 0 &&
      numCells64 <= std::numeric_limits<int>::max()) {
    result.forces.resize(static_cast<std::size_t>(numCells64));
    result.taxelCounts.resize(static_cast<std::size_t>(numCells64));
  }
  RasterizeTactileRectProjectionInto(projection, taxelForces, result, error);
  if (!error.IsOK()) {
    return {};
  }
  return result;
}

void superdex::robotics::RasterizeTactileRectProjectionInto(
    TactileRectProjectionMap const& projection,
    TactileTaxelForces const& taxelForces,
    TactileForceGrid& out,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      projection.width < 1 || projection.height < 1 || projection.selectedTaxelCount < 1 ||
          projection.coordinates.size() != projection.taxelCells.size() ||
          projection.taxelContributionOffsets.size() != projection.taxelCells.size() + 1 ||
          projection.contributionCells.size() != projection.contributionWeights.size() ||
          projection.taxelContributionOffsets.empty() ||
          projection.taxelContributionOffsets[0] != 0 ||
          projection.taxelContributionOffsets.back() != isize(projection.contributionCells),
      error,
      "A valid tactile rectangular projection is required.");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      taxelForces.forces.size() != projection.taxelCells.size(),
      error,
      "The tactile projection and force observation do not match.");
  MOCHI_ERROR_RETURN(error);

  int64_t const numCells64 =
      static_cast<int64_t>(projection.width) * static_cast<int64_t>(projection.height);
  MOCHI_ERROR_IF(
      numCells64 > std::numeric_limits<int>::max(),
      error,
      "The tactile force-grid dimensions overflowed.");
  MOCHI_ERROR_RETURN(error);
  int const numCells = static_cast<int>(numCells64);
  MOCHI_ERROR_IF_NOT(
      out.forces.size() == static_cast<std::size_t>(numCells) &&
          out.taxelCounts.size() == static_cast<std::size_t>(numCells),
      error,
      "A preallocated tactile force grid must match its configured dimensions.");
  MOCHI_ERROR_RETURN(error);
  std::ranges::fill(out.forces, Real3{});
  std::ranges::fill(out.taxelCounts, 0);
  out.width = projection.width;
  out.height = projection.height;
  int selectedTaxelCount = 0;
  for (int iTaxel = 0; iTaxel < isize(taxelForces.forces); ++iTaxel) {
    Real3 const force = taxelForces.forces[iTaxel];
    MOCHI_ERROR_IF(!IsFinite(force), error, "Tactile taxel forces must be finite.");
    MOCHI_ERROR_RETURN(error);
    int const cell = projection.taxelCells[iTaxel];
    MOCHI_ERROR_IF(cell < -1 || cell >= numCells, error, "A tactile projection cell is invalid.");
    MOCHI_ERROR_RETURN(error);
    int const contributionBegin = projection.taxelContributionOffsets[iTaxel];
    int const contributionEnd = projection.taxelContributionOffsets[iTaxel + 1];
    MOCHI_ERROR_IF(
        contributionBegin < 0 || contributionEnd < contributionBegin ||
            contributionEnd > isize(projection.contributionCells),
        error,
        "A tactile projection contribution range is invalid.");
    MOCHI_ERROR_RETURN(error);
    if (cell < 0) {
      MOCHI_ERROR_IF(
          contributionBegin != contributionEnd,
          error,
          "An excluded tactile taxel has grid contributions.");
      MOCHI_ERROR_RETURN(error);
      continue;
    }
    MOCHI_ERROR_IF(
        contributionBegin == contributionEnd,
        error,
        "A selected tactile taxel has no grid contributions.");
    MOCHI_ERROR_RETURN(error);
    real weightSum = 0_r;
    for (int iContribution = contributionBegin; iContribution < contributionEnd; ++iContribution) {
      int const contributionCell = projection.contributionCells[iContribution];
      real const weight = projection.contributionWeights[iContribution];
      MOCHI_ERROR_IF(
          contributionCell < 0 || contributionCell >= numCells || !std::isfinite(weight) ||
              weight <= 0_r,
          error,
          "A tactile projection contribution is invalid.");
      MOCHI_ERROR_RETURN(error);
      out.forces[contributionCell] += force * weight;
      MOCHI_ERROR_IF(
          !IsFinite(out.forces[contributionCell]), error, "A tactile grid force overflowed.");
      MOCHI_ERROR_RETURN(error);
      ++out.taxelCounts[contributionCell];
      weightSum += weight;
    }
    MOCHI_ERROR_IF(
        !std::isfinite(weightSum) || std::abs(weightSum - 1_r) > 1e-5_r,
        error,
        "A tactile taxel's projection weights do not sum to one.");
    MOCHI_ERROR_RETURN(error);
    ++selectedTaxelCount;
  }
  MOCHI_ERROR_IF(
      selectedTaxelCount != projection.selectedTaxelCount,
      error,
      "The tactile projection selected-taxel count is inconsistent.");
  MOCHI_ERROR_RETURN(error);
}

TactileForceGrid superdex::robotics::MapTactileHemisphereToRect(
    TactileTaxelLayout const& layout,
    TactileTaxelForces const& taxelForces,
    TactileHemisphereRectParams const& params,
    Error& error) {
  TactileRectProjectionMap const projection = BuildTactileRectProjection(layout, params, error);
  return RasterizeTactileRectProjection(projection, taxelForces, error);
}

void superdex::robotics::MapTactileHemisphereToRectInto(
    TactileTaxelLayout const& layout,
    TactileTaxelForces const& taxelForces,
    TactileHemisphereRectParams const& params,
    TactileForceGrid& out,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  TactileRectProjectionMap const projection = BuildTactileRectProjection(layout, params, error);
  RasterizeTactileRectProjectionInto(projection, taxelForces, out, error);
}
