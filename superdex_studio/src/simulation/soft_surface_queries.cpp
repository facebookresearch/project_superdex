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

#include "simulation/soft_surface_queries.h"

#include <mochi_core/utils/basic_utils.h>

namespace superdex::studio {

std::pair<mochi::QueryHandle, mochi::QueryHandle>
RegisterSoftSurfaceQueries(mochi::Actor& actor, SoftSurface surface, mochi::Error& error) {
  using mochi::QueryType;
  auto const registerPair = [&](QueryType positions, QueryType normals) {
    return std::pair{actor.RegisterQuery(positions, error), actor.RegisterQuery(normals, error)};
  };
  switch (surface) {
    case SoftSurface::Visual:
      return registerPair(QueryType::VisualNodePositions, QueryType::VisualNodeNormals);
    case SoftSurface::ContactSkin:
      return registerPair(QueryType::ContactSkinNodePositions, QueryType::ContactSkinNodeNormals);
    case SoftSurface::Surface:
      return registerPair(QueryType::SurfaceNodePositions, QueryType::SurfaceNodeNormals);
  }
  return {};
}

void ReadSoftSurfaceNodes(
    mochi::Actor const& actor,
    SoftSurface surface,
    mochi::CoordinateSpaceConverter const& converter,
    std::vector<float>& outPositions,
    std::vector<float>& outNormals,
    mochi::Error& error) {
  using namespace mochi;
  Span<real const> positions;
  Span<real const> normals;
  switch (surface) {
    case SoftSurface::Visual:
      positions = actor.GetVisualMeshNodePositionsLocal(error);
      normals = actor.GetVisualMeshNodeNormalsLocal(error);
      break;
    case SoftSurface::ContactSkin:
      positions = actor.GetContactSkinMeshNodePositionsLocal(error);
      normals = actor.GetContactSkinMeshNodeNormalsLocal(error);
      break;
    case SoftSurface::Surface:
      positions = actor.GetSurfaceMeshNodePositionsLocal(error);
      normals = actor.GetSurfaceMeshNodeNormalsLocal(error);
      break;
  }
  int const numNodes = isize(positions) / 3;
  int const numNormals = isize(normals) / 3;
  outPositions.resize(static_cast<size_t>(numNodes) * 3);
  outNormals.resize(static_cast<size_t>(numNodes) * 3);
  for (int n = 0; n < numNodes; ++n) {
    auto const p = converter.TranslationToOutput(
        StaticCast<Float3>(
            Real3{positions[3 * n + 0], positions[3 * n + 1], positions[3 * n + 2]}));
    outPositions[3 * n + 0] = p[0];
    outPositions[3 * n + 1] = p[1];
    outPositions[3 * n + 2] = p[2];
    Float3 nm{0.0f, 1.0f, 0.0f};
    if (n < numNormals) {
      nm = {
          static_cast<float>(normals[3 * n + 0]),
          static_cast<float>(normals[3 * n + 1]),
          static_cast<float>(normals[3 * n + 2])};
    }
    nm = converter.DirectionToOutput(nm);
    outNormals[3 * n + 0] = nm[0];
    outNormals[3 * n + 1] = nm[1];
    outNormals[3 * n + 2] = nm[2];
  }
}

} // namespace superdex::studio
