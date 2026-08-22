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

#include <superdex_physics.h>

#include <gtest/gtest.h>

using superdex::operator""_r;

namespace {

constexpr superdex::real kCoordinates[] = {
    -0.1_r, -0.1_r, -0.1_r, // 0
    +0.1_r, -0.1_r, -0.1_r, // 1
    -0.1_r, +0.1_r, -0.1_r, // 2
    +0.1_r, +0.1_r, -0.1_r, // 3
    -0.1_r, -0.1_r, +0.1_r, // 4
    +0.1_r, -0.1_r, +0.1_r, // 5
    -0.1_r, +0.1_r, +0.1_r, // 6
    +0.1_r, +0.1_r, +0.1_r, // 7
};

constexpr int kConnectivity[] = {
    0, 2, 1, 1, 2, 3, // -Z face
    4, 5, 6, 5, 7, 6, // +Z face
    0, 1, 5, 0, 5, 4, // -Y face
    2, 6, 3, 3, 6, 7, // +Y face
    0, 4, 2, 2, 4, 6, // -X face
    1, 3, 5, 3, 7, 5, // +X face
};

} // namespace

TEST(SuperDexPhysics, UsingDirectiveRunsRigidBodyScene) {
  constexpr int kNumWorkerThreads = 0;
  superdex::Context* context = superdex::CreateContext(kNumWorkerThreads);
  ASSERT_NE(nullptr, context);

  superdex::Error error;
  superdex::ShapeHandle cubeShape = context->CreateTriMeshShape(kCoordinates, kConnectivity, error);

  constexpr superdex::Real3 kUp{0_r, 1_r, 0_r};
  superdex::ShapeHandle planeShape = context->CreatePlaneShape(kUp, 0_r, error);

  superdex::Scene* scene = context->CreateScene("SuperDex Physics header smoke test");
  ASSERT_NE(nullptr, scene);
  scene->SetGravity(superdex::Real3{0_r, -9.81_r, 0_r});

  superdex::RigidActorParams planeParams;
  planeParams.name = "ground";
  planeParams.shape = planeShape;
  planeParams.isStatic = true;
  superdex::Actor* ground = scene->CreateRigidActor(planeParams, error);

  superdex::RigidActorParams cubeParams;
  cubeParams.name = "falling_cube";
  cubeParams.shape = cubeShape;
  cubeParams.worldFromLocal.SetTranslation(superdex::Real3{0_r, 1_r, 0_r});
  cubeParams.density = 1000_r;
  superdex::Actor* cube = scene->CreateRigidActor(cubeParams, error);

  ASSERT_TRUE(error.IsOK()) << error.ToString();
  ASSERT_NE(nullptr, ground);
  ASSERT_NE(nullptr, cube);

  superdex::real const initialHeight = cube->GetRootTransform().GetTranslation()[1];
  constexpr int kNumSteps = 10;
  constexpr double kTimeStep = 0.01;
  for (int step = 0; step < kNumSteps; ++step) {
    scene->Step(kTimeStep);
  }
  superdex::real const finalHeight = cube->GetRootTransform().GetTranslation()[1];

  EXPECT_LT(finalHeight, initialHeight);
  scene->DestroyActor(cube);
  scene->DestroyActor(ground);
  context->DestroyScene(scene);
  superdex::DestroyContext(context);
}
