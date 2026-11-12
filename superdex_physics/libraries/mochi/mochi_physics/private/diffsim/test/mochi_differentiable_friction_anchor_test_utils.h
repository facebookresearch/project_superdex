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

#pragma once

#include "mochi_physics_test_fixture.h"

#include <mochi_core/test/mochi_test_helpers.h>

#include <mochi_physics/mochi_physics.h>

// Scene setup shared by the tests of back-propagation through elasto-plastic friction anchors.
namespace mochi::test {

// Enables elasto-plastic friction, with the exact friction Hessian that back-propagation needs.
inline void EnableElastoPlasticFriction(Scene* scene) {
  auto params = scene->GetSolverParams();
  params.experimentalEval.elastoPlasticFriction = true;
  params.experimentalEval.fittedSaturationHessian.contactFriction = false;
  scene->SetSolverParams(params, ExpectOK{});
}

// A static ground plane y = 0.
inline Actor* CreateGroundPlane(Context* context, Scene* scene) {
  RigidActorParams params;
  params.name = "Plane";
  params.isStatic = true;
  params.shape = context->CreatePlaneShape(Real3{0_r, 1_r, 0_r}, 0_r, ExpectOK{});
  return scene->CreateRigidActor(params, ExpectOK{});
}

} // namespace mochi::test
