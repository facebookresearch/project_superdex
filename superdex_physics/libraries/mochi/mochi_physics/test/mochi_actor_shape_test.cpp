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

#include "mochi_physics_test_fixture.h"

using namespace mochi;
using namespace mochi::test;

//----------------------------------------------------------------------
// Actor::GetReferenceShape tests
//----------------------------------------------------------------------

class GetReferenceShapeTest : public MochiSceneTestBase {
 protected:
  std::pair<ShapeHandle, Actor*> CreateShapeAndRigidActor() {
    auto unitCube = CreateMinimalTetMeshUnitCube();
    ShapeHandle shape = _mochiContext->CreateTetMeshShape(
        Flatten(MakeSpan(unitCube.first)), Flatten(MakeSpan(unitCube.second)), ExpectOK{});
    RigidActorParams params;
    params.shape = shape;
    return {shape, _scene->CreateRigidActor(params, ExpectOK{})};
  }
};

// Verify GetReferenceShape returns a valid handle with a different numeric value than the
// original, and that it increments the tracked shape count.
TEST_F(GetReferenceShapeTest, ReturnsValidHandleWithDifferentValue) {
  auto [originalShape, actor] = CreateShapeAndRigidActor();
  ASSERT_TRUE(originalShape.IsValid());
  EXPECT_EQ(1, _mochiContext->GetNumShapes());
  ASSERT_NE(nullptr, actor);

  ShapeHandle refShape = actor->GetReferenceShape(ExpectOK{});
  EXPECT_TRUE(refShape.IsValid());
  EXPECT_NE(originalShape.value, refShape.value);
  EXPECT_EQ(2, _mochiContext->GetNumShapes());
}

// Verify GetReferenceShape succeeds even after the original handle is released, because the
// actor's internal shared_ptr keeps the shape data alive.
TEST_F(GetReferenceShapeTest, ShapeSurvivesAfterOriginalHandleReleased) {
  auto [originalShape, actor] = CreateShapeAndRigidActor();
  ASSERT_TRUE(originalShape.IsValid());
  ASSERT_NE(nullptr, actor);

  _mochiContext->ReleaseShape(originalShape);
  EXPECT_EQ(0, _mochiContext->GetNumShapes());

  // GetReferenceShape still works because the actor holds the shared_ptr
  ShapeHandle refShape = actor->GetReferenceShape(ExpectOK{});
  EXPECT_TRUE(refShape.IsValid());
  EXPECT_EQ(1, _mochiContext->GetNumShapes());
}

// Verify that a reference shape obtained from one actor can be used to create a second actor
// with identical mesh data (same node and edge counts).
TEST_F(GetReferenceShapeTest, ReferenceShapeCanBeUsedToCreateNewActor) {
  auto [originalShape, actor1] = CreateShapeAndRigidActor();
  ASSERT_NE(nullptr, actor1);

  _mochiContext->ReleaseShape(originalShape);
  EXPECT_EQ(0, _mochiContext->GetNumShapes());

  ShapeHandle refShape = actor1->GetReferenceShape(ExpectOK{});
  EXPECT_TRUE(refShape.IsValid());
  EXPECT_EQ(1, _mochiContext->GetNumShapes());

  RigidActorParams params2;
  params2.shape = refShape;
  Actor* actor2 = _scene->CreateRigidActor(params2, ExpectOK{});
  ASSERT_NE(nullptr, actor2);

  auto mesh1 = _mochiContext->GetShapeMesh(refShape, ExpectOK{});
  auto shape2 = actor2->GetReferenceShape(ExpectOK{});
  auto mesh2 = _mochiContext->GetShapeMesh(shape2, ExpectOK{});
  EXPECT_EQ(mesh1.GetNumNodes(), mesh2.GetNumNodes());
  EXPECT_EQ(mesh1.GetNumElements(), mesh2.GetNumElements());
}

// Full lifecycle: create shape → release original → get reference → destroy actor → reuse
// reference for a new actor → clean up. The refShape handle is scoped to verify that
// auto-cleanup works for handles obtained via GetReferenceShape. Zero shape leaks at the end.
TEST_F(GetReferenceShapeTest, FullLifecycleNoLeaks) {
  EXPECT_EQ(0, _mochiContext->GetNumShapes());

  auto [originalShape, actor] = CreateShapeAndRigidActor();
  EXPECT_EQ(1, _mochiContext->GetNumShapes());

  _mochiContext->ReleaseShape(originalShape);
  EXPECT_EQ(0, _mochiContext->GetNumShapes());

  {
    ShapeHandle refShape = actor->GetReferenceShape(ExpectOK{});
    EXPECT_EQ(1, _mochiContext->GetNumShapes());

    _scene->DestroyActor(actor->GetHandle());

    // The reference shape handle still keeps the shape alive
    EXPECT_EQ(1, _mochiContext->GetNumShapes());

    RigidActorParams params2;
    params2.shape = refShape;
    Actor* actor2 = _scene->CreateRigidActor(params2, ExpectOK{});
    ASSERT_NE(nullptr, actor2);

    _scene->DestroyActor(actor2->GetHandle());
    _mochiContext->ReleaseShape(refShape);

    // Let refShape go out-of-scope
  }
  EXPECT_EQ(0, _mochiContext->GetNumShapes());
}

// Verify GetReferenceShape reports an error and returns an invalid handle when called on an
// articulated actor (which has no shape of its own — only its link children do).
TEST_F(GetReferenceShapeTest, ErrorWhenActorHasNoShape) {
  auto unitCube = CreateMinimalTetMeshUnitCube();
  ShapeHandle linkShape = _mochiContext->CreateTetMeshShape(
      Flatten(MakeSpan(unitCube.first)), Flatten(MakeSpan(unitCube.second)), ExpectOK{});

  ArticulatedActorParams articulatedParams;
  articulatedParams.joints = {{.type = ArticulatedJointType::Free}};
  articulatedParams.links = {
      {.parentLink = -1, .shape = linkShape, .colliderType = ColliderType::None}};
  Actor const* actor = _scene->CreateArticulatedActor(articulatedParams, ExpectOK{});
  ASSERT_NE(nullptr, actor);

  ShapeHandle result = actor->GetReferenceShape(ExpectNotOK{});
  EXPECT_FALSE(result.IsValid());
}

// Verify that multiple reference handles are independently reference-counted: each call
// increments the shape count, and releasing one does not affect the other.
TEST_F(GetReferenceShapeTest, MultipleReferencesAreIndependent) {
  auto [originalShape, actor] = CreateShapeAndRigidActor();
  _mochiContext->ReleaseShape(originalShape);

  ShapeHandle ref1 = actor->GetReferenceShape(ExpectOK{});
  ShapeHandle ref2 = actor->GetReferenceShape(ExpectOK{});
  EXPECT_TRUE(ref1.IsValid());
  EXPECT_TRUE(ref2.IsValid());
  EXPECT_NE(ref1.value, ref2.value);
  EXPECT_EQ(2, _mochiContext->GetNumShapes());

  // Releasing one does not affect the other
  _mochiContext->ReleaseShape(ref1);
  EXPECT_EQ(1, _mochiContext->GetNumShapes());
  _mochiContext->ReleaseShape(ref2);
  EXPECT_EQ(0, _mochiContext->GetNumShapes());
}
