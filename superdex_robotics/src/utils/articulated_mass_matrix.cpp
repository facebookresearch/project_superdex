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

#include <superdex_robotics/utils/articulated_mass_matrix.h>

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/utils/transform_rt_utils.h>

using namespace mochi;
using namespace superdex::robotics;

namespace {

/* Convert a 3x3 NdArray (Matrix3x3r) to a Matrix<real, 3, 3> for arithmetic. */
Matrix<real, 3, 3> ToMatrix(Matrix3x3r const& m) {
  Matrix<real, 3, 3> out;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      out(r, c) = m[r][c];
    }
  }
  return out;
}

} // namespace

void superdex::robotics::BuildArticulatedLinkInertias(
    Actor* articulatedActor,
    DynamicArray<ArticulatedLinkInertia>& outLinkInertias,
    Error& error) {
  MOCHI_ERROR_IF(articulatedActor == nullptr, error, "articulatedActor is null");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      articulatedActor->GetType() != ActorType::Articulated,
      error,
      "articulatedActor is not an articulated body");
  MOCHI_ERROR_RETURN(error);

  Scene* scene = articulatedActor->GetScene();
  MOCHI_ERROR_IF(scene == nullptr, error, "articulatedActor has no scene");
  MOCHI_ERROR_RETURN(error);

  Span<ActorHandle const> linkHandles = articulatedActor->GetNestedLinkActors(error);
  MOCHI_ERROR_RETURN(error);

  outLinkInertias.clear();
  outLinkInertias.reserve(linkHandles.size());

  for (ActorHandle const& handle : linkHandles) {
    Actor* link = scene->GetActor(handle);
    MOCHI_ERROR_IF(link == nullptr, error, "Failed to resolve articulated link actor handle");
    MOCHI_ERROR_RETURN(error);

    ArticulatedLinkInertia entry;
    // Static links (e.g., a fixed base) cannot be queried for mass/inertia and contribute zero
    // to M(q) anyway because their Jacobian rows are zero. Push a zero-inertia placeholder so
    // the per-link iteration in ComputeArticulatedMassMatrix stays index-aligned with
    // GetNestedLinkActors / GetArticulatedLinkTransforms.
    if (link->IsStatic()) {
      outLinkInertias.emplace_back(entry);
      continue;
    }

    entry.mass = link->GetMass(error);
    entry.comLocal = link->GetRigidCenterOfMassLocal(error);
    Real6 const moiPacked = link->GetRigidMomentOfInertiaLocal(error);
    MOCHI_ERROR_RETURN(error);

    // Unpack [ixx, ixy, ixz, iyy, iyz, izz] into a symmetric 3x3 matrix.
    entry.inertiaCom[0][0] = moiPacked[0];
    entry.inertiaCom[0][1] = moiPacked[1];
    entry.inertiaCom[0][2] = moiPacked[2];
    entry.inertiaCom[1][0] = moiPacked[1];
    entry.inertiaCom[1][1] = moiPacked[3];
    entry.inertiaCom[1][2] = moiPacked[4];
    entry.inertiaCom[2][0] = moiPacked[2];
    entry.inertiaCom[2][1] = moiPacked[4];
    entry.inertiaCom[2][2] = moiPacked[5];

    outLinkInertias.emplace_back(entry);
  }
}

void superdex::robotics::ComputeArticulatedMassMatrix(
    Actor* articulatedActor,
    Span<ArticulatedLinkInertia const> linkInertias,
    Span<TransformRT const> worldFromLinks,
    int numDofs,
    Span<real> outM,
    Error& error) {
  MOCHI_ERROR_IF(articulatedActor == nullptr, error, "articulatedActor is null");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      articulatedActor->GetType() != ActorType::Articulated,
      error,
      "articulatedActor is not an articulated body");
  MOCHI_ERROR_RETURN(error);

  int const numLinks = isize(linkInertias);
  MOCHI_ERROR_IF(numLinks == 0, error, "linkInertias is empty");
  MOCHI_ERROR_IF(numDofs <= 0, error, "numDofs must be positive");
  MOCHI_ERROR_IF(
      isize(worldFromLinks) != numLinks,
      error,
      "worldFromLinks size does not match numLinks (linkInertias.size())");
  MOCHI_ERROR_IF(
      isize(outM) != numDofs * numDofs, error, "outM size does not match numDofs * numDofs");
  MOCHI_ERROR_RETURN(error);

  Scene* scene = articulatedActor->GetScene();
  MOCHI_ERROR_IF(scene == nullptr, error, "articulatedActor has no scene");
  MOCHI_ERROR_RETURN(error);

  Span<ActorHandle const> linkHandles = articulatedActor->GetNestedLinkActors(error);
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      isize(linkHandles) != numLinks,
      error,
      "Number of nested link actors does not match linkInertias size");
  MOCHI_ERROR_RETURN(error);

  // Zero the output. RowMatrixView accumulates into it across links below.
  for (int i = 0; i < numDofs * numDofs; ++i) {
    outM[i] = 0_r;
  }
  RowMatrixView<real> M(outM.data(), numDofs, numDofs);

  for (int link = 0; link < numLinks; ++link) {
    ArticulatedLinkInertia const& inertia = linkInertias[link];
    real const m = inertia.mass;

    // World rotation R_i of link, used to rotate the body-frame inertia tensor into world frame.
    Matrix<real, 3, 3> const R = ToMatrix(GetRotationMatrix(worldFromLinks[link]));

    // World-frame inertia about COM: R * I_C * R^T (3x3, symmetric).
    Matrix<real, 3, 3> const I_C = ToMatrix(inertia.inertiaCom);
    Matrix<real, 3, 3> const I_world = R * I_C * R.Transpose();

    // Spatial inertia G_i at the link's COM expressed in world frame, [linear; angular] row
    // layout. Mochi's per-link Jacobian gives the linear velocity AT THE LINK'S COM -- the
    // articulation's internal "bone" frame is COM-centered, because CreateRestTransforms
    // (mochi_core/src/articulated_body/articulated_body.cpp) bakes comLocal into the rest
    // transforms the Jacobian differentiates. So the spatial inertia must also be taken at the
    // COM. With the reference point AT the COM, the off-diagonal blocks are zero and there is no
    // parallel-axis term.
    //   G = [ m * I_3            0        ]
    //       [   0           R * I_C * R^T ]
    // Note worldFromLinks is the link-ORIGIN frame, not the COM frame; only its rotation is read
    // below, and the two frames share that, so no conversion is needed here.
    Matrix<real, 6, 6> G;
    G.SetZero();
    for (int k = 0; k < 3; ++k) {
      G(k, k) = m;
    }
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        G(r + 3, c + 3) = I_world(r, c);
      }
    }

    // Fetch this link's 6 x numDofs spatial Jacobian (world-frame, row-major).
    Actor* linkActor = scene->GetActor(linkHandles[link]);
    MOCHI_ERROR_IF(linkActor == nullptr, error, "Failed to resolve articulated link actor handle");
    MOCHI_ERROR_RETURN(error);
    Span<real const> jacSpan = linkActor->GetArticulatedJacobian(error);
    MOCHI_ERROR_RETURN(error);
    MOCHI_ERROR_IF(
        isize(jacSpan) != 6 * numDofs, error, "Per-link Jacobian size does not match 6 * numDofs");
    MOCHI_ERROR_RETURN(error);

    RowMatrixView<real const> J(jacSpan.data(), 6, numDofs);

    // Accumulate M += J^T * G * J.
    M += J.Transpose() * G * J;
  }
}
