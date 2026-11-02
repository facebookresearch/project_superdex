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

#include "rendering/soft_surface.h"

#include <gtest/gtest.h>

using namespace superdex::studio;

namespace {

TEST(SoftSurfaceTest, TheVisualMeshIsDrawnAndTheUsedContactSkinShowsInTheCollisionView) {
  EXPECT_EQ(
      ChooseSoftSurfaces(/*hasVisualMesh=*/true, /*hasContactSkin=*/true, /*useContactSkin=*/true),
      (SoftSurfaces{.rendered = SoftSurface::Visual, .collision = SoftSurface::ContactSkin}));
}

TEST(SoftSurfaceTest, AnUnusedContactSkinLeavesTheBoundaryInTheCollisionView) {
  EXPECT_EQ(
      ChooseSoftSurfaces(/*hasVisualMesh=*/true, /*hasContactSkin=*/true, /*useContactSkin=*/false),
      (SoftSurfaces{.rendered = SoftSurface::Visual, .collision = SoftSurface::Surface}));
}

TEST(SoftSurfaceTest, AVisualMeshAloneIsDrawnOverTheBoundary) {
  EXPECT_EQ(
      ChooseSoftSurfaces(/*hasVisualMesh=*/true, /*hasContactSkin=*/false, /*useContactSkin=*/true),
      (SoftSurfaces{.rendered = SoftSurface::Visual, .collision = SoftSurface::Surface}));
}

TEST(SoftSurfaceTest, WithoutAVisualMeshTheContactSkinIsDrawnWhetherOrNotItIsUsed) {
  EXPECT_EQ(
      ChooseSoftSurfaces(/*hasVisualMesh=*/false, /*hasContactSkin=*/true, /*useContactSkin=*/true),
      (SoftSurfaces{.rendered = SoftSurface::ContactSkin, .collision = SoftSurface::ContactSkin}));
  EXPECT_EQ(
      ChooseSoftSurfaces(
          /*hasVisualMesh=*/false, /*hasContactSkin=*/true, /*useContactSkin=*/false),
      (SoftSurfaces{.rendered = SoftSurface::ContactSkin, .collision = SoftSurface::Surface}));
}

TEST(SoftSurfaceTest, WithNeitherTheBoundaryIsDrawn) {
  EXPECT_EQ(
      ChooseSoftSurfaces(
          /*hasVisualMesh=*/false, /*hasContactSkin=*/false, /*useContactSkin=*/true),
      SoftSurfaces{});
}

TEST(SoftSurfaceTest, AnUnembeddedVisualMeshFallsBack) {
  mochi::ModelData model;
  model.visualMesh.emplace();

  EXPECT_EQ(ChooseSoftSurfaces(model, /*useContactSkin=*/true), SoftSurfaces{});

  model.contactSkinMesh.emplace();
  EXPECT_EQ(
      ChooseSoftSurfaces(model, /*useContactSkin=*/true),
      (SoftSurfaces{.rendered = SoftSurface::ContactSkin, .collision = SoftSurface::ContactSkin}));

  model.visualMesh->skinning.emplace();
  EXPECT_EQ(
      ChooseSoftSurfaces(model, /*useContactSkin=*/true),
      (SoftSurfaces{.rendered = SoftSurface::Visual, .collision = SoftSurface::ContactSkin}));
}

} // namespace
