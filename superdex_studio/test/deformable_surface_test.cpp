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

#include "rendering/deformable_surface.h"

#include <gtest/gtest.h>

using namespace superdex::studio;

namespace {

TEST(DeformableSurfaceTest, TheVisualMeshIsDrawnAndTheUsedContactSkinShowsInTheCollisionView) {
  EXPECT_EQ(
      ChooseSoftSurfaces(/*hasVisualMesh=*/true, /*hasContactSkin=*/true, /*useContactSkin=*/true),
      (DeformableSurfaces{
          .rendered = DeformableSurface::Visual, .collision = DeformableSurface::ContactSkin}));
}

TEST(DeformableSurfaceTest, AnUnusedContactSkinLeavesTheBoundaryInTheCollisionView) {
  EXPECT_EQ(
      ChooseSoftSurfaces(/*hasVisualMesh=*/true, /*hasContactSkin=*/true, /*useContactSkin=*/false),
      (DeformableSurfaces{
          .rendered = DeformableSurface::Visual, .collision = DeformableSurface::Surface}));
}

TEST(DeformableSurfaceTest, AVisualMeshAloneIsDrawnOverTheBoundary) {
  EXPECT_EQ(
      ChooseSoftSurfaces(/*hasVisualMesh=*/true, /*hasContactSkin=*/false, /*useContactSkin=*/true),
      (DeformableSurfaces{
          .rendered = DeformableSurface::Visual, .collision = DeformableSurface::Surface}));
}

TEST(DeformableSurfaceTest, WithoutAVisualMeshTheContactSkinIsDrawnWhetherOrNotItIsUsed) {
  EXPECT_EQ(
      ChooseSoftSurfaces(/*hasVisualMesh=*/false, /*hasContactSkin=*/true, /*useContactSkin=*/true),
      (DeformableSurfaces{
          .rendered = DeformableSurface::ContactSkin,
          .collision = DeformableSurface::ContactSkin}));
  EXPECT_EQ(
      ChooseSoftSurfaces(
          /*hasVisualMesh=*/false, /*hasContactSkin=*/true, /*useContactSkin=*/false),
      (DeformableSurfaces{
          .rendered = DeformableSurface::ContactSkin, .collision = DeformableSurface::Surface}));
}

TEST(DeformableSurfaceTest, WithNeitherTheBoundaryIsDrawn) {
  EXPECT_EQ(
      ChooseSoftSurfaces(
          /*hasVisualMesh=*/false, /*hasContactSkin=*/false, /*useContactSkin=*/true),
      DeformableSurfaces{});
}

TEST(DeformableSurfaceTest, AnUnembeddedVisualMeshFallsBack) {
  mochi::ModelData model;
  model.visualMesh.emplace();

  EXPECT_EQ(ChooseSoftSurfaces(model, /*useContactSkin=*/true), DeformableSurfaces{});

  model.contactSkinMesh.emplace();
  EXPECT_EQ(
      ChooseSoftSurfaces(model, /*useContactSkin=*/true),
      (DeformableSurfaces{
          .rendered = DeformableSurface::ContactSkin,
          .collision = DeformableSurface::ContactSkin}));

  model.visualMesh->skinning.emplace();
  EXPECT_EQ(
      ChooseSoftSurfaces(model, /*useContactSkin=*/true),
      (DeformableSurfaces{
          .rendered = DeformableSurface::Visual, .collision = DeformableSurface::ContactSkin}));
}

} // namespace
