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

TactileForceGrid superdex::robotics::MapTactileHemisphereToRect(
    TactileTaxelLayout const& layout,
    TactileTaxelForces const& taxelForces,
    TactileHemisphereRectParams const& params,
    Error& error) {
  TactileForceGrid result;
  std::int64_t const numCells64 =
      static_cast<std::int64_t>(params.width) * static_cast<std::int64_t>(params.height);
  if (params.width > 0 && params.height > 0 && numCells64 <= std::numeric_limits<int>::max()) {
    result.forces.resize(static_cast<std::size_t>(numCells64));
    result.taxelCounts.resize(static_cast<std::size_t>(numCells64));
  }
  MapTactileHemisphereToRectInto(layout, taxelForces, params, result, error);
  if (!error.IsOK()) {
    return {};
  }
  return result;
}

void superdex::robotics::MapTactileHemisphereToRectInto(
    TactileTaxelLayout const& layout,
    TactileTaxelForces const& taxelForces,
    TactileHemisphereRectParams const& params,
    TactileForceGrid& out,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      params.width < 1 || params.height < 1,
      error,
      "A tactile force grid must have positive width and height.");
  MOCHI_ERROR_RETURN(error);
  constexpr real kPi = std::numbers::pi_v<real>;
  constexpr real kHalfPi = kPi / 2_r;
  MOCHI_ERROR_IF(
      !std::isfinite(params.maxPolarLatitude) || params.maxPolarLatitude <= 0_r ||
          params.maxPolarLatitude > kHalfPi,
      error,
      "The maximum polar latitude must be in (0, pi/2].");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      !IsFinite(params.center) || !IsFinite(params.tipDirection) ||
          !IsFinite(params.zeroLongitudeDirection),
      error,
      "The tactile hemisphere frame must be finite.");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(layout.positions.empty(), error, "A valid tactile taxel layout is required.");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      taxelForces.forces.size() != layout.positions.size(),
      error,
      "The tactile layout and force observation do not match.");
  MOCHI_ERROR_RETURN(error);

  real const tipNorm = Norm(params.tipDirection);
  MOCHI_ERROR_IF(tipNorm <= 0_r, error, "The tactile tip direction must be nonzero.");
  MOCHI_ERROR_RETURN(error);
  Real3 const tip = params.tipDirection * (1_r / tipNorm);
  Real3 zeroLongitude =
      params.zeroLongitudeDirection - Dot(params.zeroLongitudeDirection, tip) * tip;
  real const zeroLongitudeNorm = Norm(zeroLongitude);
  MOCHI_ERROR_IF(
      zeroLongitudeNorm <= 0_r,
      error,
      "The zero-longitude direction must not be parallel to the tip direction.");
  MOCHI_ERROR_RETURN(error);
  zeroLongitude *= 1_r / zeroLongitudeNorm;
  Real3 const positiveLongitude = Cross(tip, zeroLongitude);

  int64_t const numCells64 =
      static_cast<int64_t>(params.width) * static_cast<int64_t>(params.height);
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
  out.width = params.width;
  out.height = params.height;

  constexpr real kDirectionEpsilon = 1e-12_r;
  constexpr real kLatitudeTolerance = 1e-5_r;
  real const invTwoPi = 1_r / (2_r * kPi);
  real const invMaxPolarLatitude = 1_r / params.maxPolarLatitude;
  for (int iTaxel = 0; iTaxel < isize(layout.positions); ++iTaxel) {
    Real3 const position = layout.positions[iTaxel];
    MOCHI_ERROR_IF(!IsFinite(position), error, "Tactile taxel positions must be finite.");
    MOCHI_ERROR_RETURN(error);
    Real3 const radial = position - params.center;
    real const radius = Norm(radial);
    MOCHI_ERROR_IF(
        radius <= kDirectionEpsilon,
        error,
        "A tactile taxel cannot coincide with the hemisphere center.");
    MOCHI_ERROR_RETURN(error);
    Real3 const direction = radial * (1_r / radius);
    real const tipCosine = std::clamp(Dot(direction, tip), -1_r, 1_r);
    real const polarLatitude = std::acos(tipCosine);
    MOCHI_ERROR_IF(
        polarLatitude > params.maxPolarLatitude + kLatitudeTolerance,
        error,
        "A tactile taxel lies outside the configured hemisphere.");
    MOCHI_ERROR_RETURN(error);

    real longitude = 0_r;
    if (polarLatitude > kDirectionEpsilon) {
      longitude = std::atan2(Dot(direction, positiveLongitude), Dot(direction, zeroLongitude));
    }
    real normalizedLongitude = (longitude + kPi) * invTwoPi;
    constexpr real kLongitudeSeamTolerance = 8_r * std::numeric_limits<real>::epsilon();
    if (normalizedLongitude >= 1_r - kLongitudeSeamTolerance) {
      normalizedLongitude = 0_r;
    }
    real const normalizedLatitude =
        std::min(polarLatitude, params.maxPolarLatitude) * invMaxPolarLatitude;
    int const column =
        std::min(params.width - 1, static_cast<int>(normalizedLongitude * params.width));
    int const row =
        std::min(params.height - 1, static_cast<int>(normalizedLatitude * params.height));
    int64_t const cell64 = static_cast<int64_t>(row) * static_cast<int64_t>(params.width) + column;
    int const cell = static_cast<int>(cell64);

    Real3 const force = taxelForces.forces[iTaxel];
    MOCHI_ERROR_IF(!IsFinite(force), error, "Tactile taxel forces must be finite.");
    MOCHI_ERROR_RETURN(error);
    out.forces[cell] += force;
    MOCHI_ERROR_IF(!IsFinite(out.forces[cell]), error, "A tactile grid force overflowed.");
    MOCHI_ERROR_RETURN(error);
    ++out.taxelCounts[cell];
  }
}
