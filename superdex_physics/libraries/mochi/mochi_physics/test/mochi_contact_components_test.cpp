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

#include <mochi_physics/src/mochi_contact.h>

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/rigid_body_size.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/vmatrix.h>

#include <gtest/gtest.h>

#include <array>
#include <initializer_list>
#include <limits>
#include <optional>
#include <utility>

using namespace mochi;
using namespace mochi::test;

// ---------------------------------------------------------------------------------------
// Contact parameter validation and pair resolution
// ---------------------------------------------------------------------------------------

namespace {

ContactParams MakeCollidingContactParams() {
  ContactParams params;
  params.penaltyCoefficient = 4_r;
  params.frictionFalloffVel = 0.04_r;
  params.viscousFrictionCoefficient = 4_r;
  params.coulombFrictionCoefficient = 9_r;
  params.normalViscousDampingCoefficient = 16_r;
  params.collidingPenaltyLengthScale = 0.5_r;
  return params;
}

ContactParams MakeColliderContactParams() {
  ContactParams params;
  params.penaltyCoefficient = 9_r;
  params.frictionFalloffVel = 0.09_r;
  params.viscousFrictionCoefficient = 9_r;
  params.coulombFrictionCoefficient = 4_r;
  params.normalViscousDampingCoefficient = 25_r;
  params.penaltySmoothingHalfDistance = 0.123_r;
  params.penaltyThresholdDefault = -0.02_r;
  params.penaltyThresholdExtraPadding = 0.03_r;
  params.frictionWithColliderNormal = false;
  params.maxAlignmentNormals = 0.7_r;
  params.distanceErrorBound = 0.04_r;
  params.objScale = 3_r;
  return params;
}

void ExpectCombinedFields(
    ContactParams const& actual,
    real penalty,
    real falloff,
    real viscous,
    real coulomb,
    real damping) {
  EXPECT_NEAR_EQ(penalty, actual.penaltyCoefficient);
  EXPECT_NEAR_EQ(falloff, actual.frictionFalloffVel);
  EXPECT_NEAR_EQ(viscous, actual.viscousFrictionCoefficient);
  EXPECT_NEAR_EQ(coulomb, actual.coulombFrictionCoefficient);
  EXPECT_NEAR_EQ(damping, actual.normalViscousDampingCoefficient);
}

void ExpectColliderOwnedFieldsUnchanged(
    ContactParams const& actual,
    ContactParams const& collider) {
  EXPECT_EQ(collider.penaltySmoothingHalfDistance, actual.penaltySmoothingHalfDistance);
  EXPECT_EQ(collider.penaltyThresholdDefault, actual.penaltyThresholdDefault);
  EXPECT_EQ(collider.penaltyThresholdExtraPadding, actual.penaltyThresholdExtraPadding);
  EXPECT_EQ(collider.frictionWithColliderNormal, actual.frictionWithColliderNormal);
  EXPECT_EQ(collider.maxAlignmentNormals, actual.maxAlignmentNormals);
  EXPECT_EQ(collider.distanceErrorBound, actual.distanceErrorBound);
  EXPECT_EQ(collider.objScale, actual.objScale);
  EXPECT_EQ(collider.collidingPenaltyLengthScale, actual.collidingPenaltyLengthScale);
}

} // namespace

TEST(ContactParamsValidation, AcceptsFiniteValuesAtLegalBounds) {
  ContactParams params;
  params.penaltyThresholdDefault = -std::numeric_limits<real>::max();
  params.distanceErrorBound = -1_r;
  params.maxAlignmentNormals = -1_r;
  ValidateContactParams(params, ExpectOK{});

  params.maxAlignmentNormals = 1_r;
  ValidateContactParams(params, ExpectOK{});
}

TEST(ContactParamsValidation, RejectsNonFiniteValuesForEveryNumericField) {
  constexpr std::array numericFields = {
      &ContactParams::penaltyCoefficient,
      &ContactParams::penaltySmoothingHalfDistance,
      &ContactParams::penaltyThresholdDefault,
      &ContactParams::penaltyThresholdExtraPadding,
      &ContactParams::collidingPenaltyThresholdPadding,
      &ContactParams::maxAlignmentNormals,
      &ContactParams::viscousFrictionCoefficient,
      &ContactParams::coulombFrictionCoefficient,
      &ContactParams::frictionFalloffVel,
      &ContactParams::normalViscousDampingCoefficient,
      &ContactParams::distanceErrorBound,
      &ContactParams::objScale,
      &ContactParams::collidingPenaltyLengthScale,
  };
  constexpr std::array nonFiniteValues = {
      std::numeric_limits<real>::infinity(),
      -std::numeric_limits<real>::infinity(),
      std::numeric_limits<real>::quiet_NaN(),
  };

  ValidateContactParams(ContactParams{}, ExpectOK{});
  for (auto const field : numericFields) {
    for (real const value : nonFiniteValues) {
      ContactParams params;
      params.*field = value;
      ValidateContactParams(params, ExpectNotOK{});
    }
  }
}

TEST(ContactParamsValidation, RejectsNegativePaddings) {
  for (auto const field :
       {&ContactParams::penaltyThresholdExtraPadding,
        &ContactParams::collidingPenaltyThresholdPadding}) {
    ContactParams params;
    params.*field = 0_r;
    ValidateContactParams(params, ExpectOK{});
    params.*field = -1e-6_r;
    ValidateContactParams(params, ExpectNotOK{});
  }
}

TEST(ContactPairParams, CollidingPaddingAppliesToDetectionAndForceThresholdsAlike) {
  real constexpr kCollidingPadding = 0.25_r;
  ContactParams colliderParams = MakeColliderContactParams();
  // The collider's own colliding padding must not apply.
  colliderParams.collidingPenaltyThresholdPadding = 2_r * kCollidingPadding;

  struct ColliderCase {
    ColliderType type;
    bool hasSdfMapping;
    bool appliesCollidingPadding;
  };
  constexpr std::array colliderCases = {
      ColliderCase{
          .type = ColliderType::Sphere, .hasSdfMapping = false, .appliesCollidingPadding = true},
      ColliderCase{
          .type = ColliderType::Box, .hasSdfMapping = false, .appliesCollidingPadding = true},
      ColliderCase{
          .type = ColliderType::Mesh, .hasSdfMapping = false, .appliesCollidingPadding = true},
      ColliderCase{
          .type = ColliderType::Plane, .hasSdfMapping = false, .appliesCollidingPadding = true},
      ColliderCase{
          .type = ColliderType::Sdf, .hasSdfMapping = false, .appliesCollidingPadding = true},
      ColliderCase{
          .type = ColliderType::Sdf, .hasSdfMapping = true, .appliesCollidingPadding = false},
      ColliderCase{
          .type = ColliderType::PointCloud,
          .hasSdfMapping = false,
          .appliesCollidingPadding = false},
  };
  // The collider's extra padding applies only to these colliding types.
  constexpr std::array collidingTypes = {
      ColliderType::None, ColliderType::Sdf, ColliderType::PointCloud};

  for (ColliderCase const& colliderCase : colliderCases) {
    for (ColliderType const collidingType : collidingTypes) {
      for (real const collidingPadding : {0_r, kCollidingPadding}) {
        entt::registry reg;
        entt::entity const colliding = reg.create();
        entt::entity const collider = reg.create();
        ContactParams collidingParams = MakeCollidingContactParams();
        collidingParams.collidingPenaltyThresholdPadding = collidingPadding;
        reg.emplace<CContactParams>(colliding, collidingParams);
        reg.emplace<CContactParams>(collider, colliderParams);
        reg.emplace<CColliderInfo>(colliding).type = collidingType;
        reg.emplace<CColliderInfo>(collider).type = colliderCase.type;
        if (colliderCase.hasSdfMapping) {
          reg.emplace<CSdfMapping<TimeStep::Current>>(collider);
        }
        reg.set<CContactPairParamsOverrideTable>();

        bool const addPadding = ShouldAddPenaltyPadding(collidingType);
        real const expectedCollidingPadding =
            colliderCase.appliesCollidingPadding ? collidingPadding : 0_r;
        real const expectedThreshold =
            colliderParams.GetPenaltyThresholdDist(addPadding) + expectedCollidingPadding;
        EXPECT_EQ(expectedCollidingPadding, GetCollidingPadding(reg, colliding, collider));
        EXPECT_EQ(expectedThreshold, GetPairPenaltyThresholdDist(reg, colliding, collider));
        EXPECT_NEAR_EQ(
            expectedThreshold,
            GetContactPairParams(reg, colliding, collider).GetPenaltyThresholdDist(addPadding));
      }
    }
  }
}

TEST(ContactPairParams, NoOverridePreservesDynamicAndStaticCombination) {
  ContactParams const colliding = MakeCollidingContactParams();
  ContactParams const collider = MakeColliderContactParams();

  ContactParams expectedDynamic = collider;
  expectedDynamic.coulombFrictionCoefficient =
      Sqrt(colliding.coulombFrictionCoefficient * collider.coulombFrictionCoefficient);
  expectedDynamic.viscousFrictionCoefficient =
      Sqrt(colliding.viscousFrictionCoefficient * collider.viscousFrictionCoefficient);
  expectedDynamic.normalViscousDampingCoefficient =
      Sqrt(colliding.normalViscousDampingCoefficient * collider.normalViscousDampingCoefficient);
  expectedDynamic.penaltyCoefficient =
      Sqrt(colliding.penaltyCoefficient * collider.penaltyCoefficient);
  expectedDynamic.frictionFalloffVel =
      Sqrt(colliding.frictionFalloffVel * collider.frictionFalloffVel);

  ContactParams const dynamic = CombineContactParams(
      colliding,
      collider,
      nullptr,
      /*isStaticCollider=*/false,
      /*collidingIntegralDim=*/2,
      /*colliderIntegralDim=*/0,
      /*colliderPenaltyLengthScale=*/0_r,
      /*collidingPadding=*/0_r);
  EXPECT_EQ(expectedDynamic.penaltyCoefficient, dynamic.penaltyCoefficient);
  EXPECT_EQ(expectedDynamic.frictionFalloffVel, dynamic.frictionFalloffVel);
  EXPECT_EQ(expectedDynamic.viscousFrictionCoefficient, dynamic.viscousFrictionCoefficient);
  EXPECT_EQ(expectedDynamic.coulombFrictionCoefficient, dynamic.coulombFrictionCoefficient);
  EXPECT_EQ(
      expectedDynamic.normalViscousDampingCoefficient, dynamic.normalViscousDampingCoefficient);
  ExpectColliderOwnedFieldsUnchanged(dynamic, collider);

  ContactParams const staticCollider = CombineContactParams(
      colliding,
      collider,
      nullptr,
      /*isStaticCollider=*/true,
      /*collidingIntegralDim=*/2,
      /*colliderIntegralDim=*/0,
      /*colliderPenaltyLengthScale=*/0_r,
      /*collidingPadding=*/0_r);
  ExpectCombinedFields(staticCollider, 4_r, 0.04_r, 6_r, 6_r, 20_r);
  ExpectColliderOwnedFieldsUnchanged(staticCollider, collider);
}

TEST(ContactPairParams, EachOverrideFieldReplacesOnlyItsResolvedValue) {
  ContactParams const colliding = MakeCollidingContactParams();
  ContactParams const collider = MakeColliderContactParams();
  struct Case {
    std::optional<real> ContactPairParamsOverride::* overrideMember;
    real ContactParams::* resolvedMember;
  };
  constexpr std::array cases = {
      Case{&ContactPairParamsOverride::penaltyCoefficient, &ContactParams::penaltyCoefficient},
      Case{&ContactPairParamsOverride::frictionFalloffVel, &ContactParams::frictionFalloffVel},
      Case{
          &ContactPairParamsOverride::viscousFrictionCoefficient,
          &ContactParams::viscousFrictionCoefficient},
      Case{
          &ContactPairParamsOverride::coulombFrictionCoefficient,
          &ContactParams::coulombFrictionCoefficient},
      Case{
          &ContactPairParamsOverride::normalViscousDampingCoefficient,
          &ContactParams::normalViscousDampingCoefficient},
  };

  for (Case const& testCase : cases) {
    ContactPairParamsOverride paramsOverride;
    paramsOverride.*testCase.overrideMember = 7_r;
    ContactParams expected = CombineContactParams(
        colliding,
        collider,
        /*paramsOverride=*/nullptr,
        /*isStaticCollider=*/false,
        /*collidingIntegralDim=*/2,
        /*colliderIntegralDim=*/0,
        /*colliderPenaltyLengthScale=*/0_r,
        /*collidingPadding=*/0_r);
    expected.*testCase.resolvedMember = 7_r;

    ContactParams const actual = CombineContactParams(
        colliding,
        collider,
        &paramsOverride,
        /*isStaticCollider=*/false,
        /*collidingIntegralDim=*/2,
        /*colliderIntegralDim=*/0,
        /*colliderPenaltyLengthScale=*/0_r,
        /*collidingPadding=*/0_r);
    ExpectCombinedFields(
        actual,
        expected.penaltyCoefficient,
        expected.frictionFalloffVel,
        expected.viscousFrictionCoefficient,
        expected.coulombFrictionCoefficient,
        expected.normalViscousDampingCoefficient);
    ExpectColliderOwnedFieldsUnchanged(actual, collider);
  }
}

TEST(ContactPairParams, PartialPatchOverridesStaticSelectionAndRetainsOmittedFields) {
  ContactParams const colliding = MakeCollidingContactParams();
  ContactParams const collider = MakeColliderContactParams();
  ContactPairParamsOverride paramsOverride;
  paramsOverride.penaltyCoefficient = 12_r;
  paramsOverride.frictionFalloffVel = 0.25_r;
  paramsOverride.coulombFrictionCoefficient = 0_r;

  ContactParams const actual = CombineContactParams(
      colliding,
      collider,
      &paramsOverride,
      /*isStaticCollider=*/true,
      /*collidingIntegralDim=*/2,
      /*colliderIntegralDim=*/0,
      /*colliderPenaltyLengthScale=*/0_r,
      /*collidingPadding=*/0_r);
  ExpectCombinedFields(actual, 12_r, 0.25_r, 6_r, 0_r, 20_r);
  ExpectColliderOwnedFieldsUnchanged(actual, collider);
}

TEST(ContactPairParams, PenaltyOverridePrecedesBothDimensionalCorrections) {
  ContactParams const colliding = MakeCollidingContactParams();
  ContactParams const collider = MakeColliderContactParams();
  ContactPairParamsOverride paramsOverride;
  paramsOverride.penaltyCoefficient = 8_r;

  ContactParams const actual = CombineContactParams(
      colliding,
      collider,
      &paramsOverride,
      /*isStaticCollider=*/false,
      /*collidingIntegralDim=*/1,
      /*colliderIntegralDim=*/2,
      /*colliderPenaltyLengthScale=*/2_r,
      /*collidingPadding=*/0_r);
  EXPECT_EQ(1_r, actual.penaltyCoefficient);
}

TEST(ContactPairParams, RegistryLookupAppliesPairOverrideAndFallsBackForAbsentPair) {
  entt::registry reg;
  entt::entity const colliding = reg.create();
  entt::entity const otherColliding = reg.create();
  entt::entity const collider = reg.create();
  ContactParams const collidingParams = MakeCollidingContactParams();
  ContactParams const otherCollidingParams = MakeCollidingContactParams();
  ContactParams const colliderParams = MakeColliderContactParams();
  reg.emplace<CContactParams>(colliding, collidingParams);
  reg.emplace<CContactParams>(otherColliding, otherCollidingParams);
  reg.emplace<CContactParams>(collider, colliderParams);

  ContactPairParamsOverride paramsOverride;
  paramsOverride.penaltyCoefficient = 13_r;
  reg.set<CContactPairParamsOverrideTable>().Set(collider, colliding, paramsOverride);

  ContactParams const actualOverride = GetContactPairParams(reg, colliding, collider);
  EXPECT_EQ(13_r, actualOverride.penaltyCoefficient);

  ContactParams const expectedFallback = CombineContactParams(
      otherCollidingParams,
      colliderParams,
      /*paramsOverride=*/nullptr,
      /*isStaticCollider=*/false,
      /*collidingIntegralDim=*/2,
      /*colliderIntegralDim=*/0,
      /*colliderPenaltyLengthScale=*/0_r,
      /*collidingPadding=*/0_r);
  ContactParams const actualFallback = GetContactPairParams(reg, otherColliding, collider);
  ExpectCombinedFields(
      actualFallback,
      expectedFallback.penaltyCoefficient,
      expectedFallback.frictionFalloffVel,
      expectedFallback.viscousFrictionCoefficient,
      expectedFallback.coulombFrictionCoefficient,
      expectedFallback.normalViscousDampingCoefficient);
  ExpectColliderOwnedFieldsUnchanged(actualFallback, colliderParams);
}

// ---------------------------------------------------------------------------------------
// CDeformablePointAsyncCollisionsResponse
// ---------------------------------------------------------------------------------------

static constexpr int kInvalid = CDeformablePointAsyncCollisionsResponse::kInvalidIndex;

static void AddEnergyGradient(
    CDeformablePointAsyncCollisionsResponse& response,
    int sampleIndex,
    double energy,
    Real3 const& gradient) {
  response.AddContactSampleResponse(sampleIndex, &energy, &gradient, nullptr);
}

TEST(CDeformablePointAsyncCollisionsResponse, AppendsSamplesAndDedupsActiveElements) {
  CDeformablePointAsyncCollisionsResponse response;
  EXPECT_TRUE(response.Empty());

  response.Reset(/*numContactElements*/ 2, /*numQuadsPerContactElement*/ 2);
  // Samples 0 and 1 share contact element 0 (0/2 == 1/2); sample 3 is element 1.
  AddEnergyGradient(response, 0, 4.0, Real3{1_r, 0_r, 0_r});
  AddEnergyGradient(response, 1, 5.0, Real3{0_r, 1_r, 0_r});
  AddEnergyGradient(response, 3, 6.0, Real3{0_r, 0_r, 1_r});
  response.ValidateInvariants(MakeConstSpan(std::array<bool, 2>{true, true}));

  EXPECT_FALSE(response.Empty());
  EXPECT_EQ(response.GetResponseIndexFromSampleIndex(2), kInvalid); // produced no response
  EXPECT_NEAR_EQ(response.GetEnergy(response.GetResponseIndexFromSampleIndex(1)), 5.0);

  auto const subset = response.ViewActiveContactElementSubset();
  EXPECT_SPAN_EQ(subset.isElementActive, MakeConstSpan(std::array<bool, 2>{true, true}));
  EXPECT_EQ(isize(subset.activeElementIndices), 2); // element 0 recorded once despite two samples
}

TEST(CDeformablePointAsyncCollisionsResponse, AccumulatesRepeatedSample) {
  CDeformablePointAsyncCollisionsResponse response;
  response.Reset(2, 2);

  double const energy[] = {2.0, 5.0};
  Real3 const gradient[] = {Real3{1_r, 2_r, 3_r}, Real3{10_r, 20_r, 30_r}};
  VMatrix3x3r const hessian{Vec4r{1_r, 2_r, 3_r}, Vec4r{4_r, 5_r, 6_r}, Vec4r{7_r, 8_r, 9_r}};

  response.AddContactSampleResponse(1, &energy[0], &gradient[0], &hessian); // append
  int const idx = response.GetResponseIndexFromSampleIndex(1);
  response.AddContactSampleResponse(1, &energy[1], &gradient[1], &hessian); // accumulate
  response.ValidateInvariants(MakeConstSpan(std::array<bool, 2>{true, false}));

  EXPECT_EQ(response.GetResponseIndexFromSampleIndex(1), idx); // same slot reused
  EXPECT_NEAR_EQ(response.GetEnergy(idx), 7.0);
  EXPECT_NEAR_EQ(response.GetGradient(idx), gradient[0] + gradient[1]);
  EXPECT_NEAR_EQ(response.GetHessian(idx), ToNdArray3x3(hessian) + ToNdArray3x3(hessian));
}

TEST(CDeformablePointAsyncCollisionsResponse, PopulatesPresentArraysOnly) {
  CDeformablePointAsyncCollisionsResponse response;

  // Objective-only assembly.
  response.Reset(/*numContactElements*/ 2, /*numQuadsPerContactElement*/ 2);
  double const energy = 3.5;
  response.AddContactSampleResponse(/*sampleIndex*/ 2, &energy, nullptr, nullptr);
  response.ValidateInvariants(
      MakeConstSpan(std::array<bool, 2>{false, true})); // Empty gradient/hessian arrays is valid
  EXPECT_NEAR_EQ(response.GetEnergy(response.GetResponseIndexFromSampleIndex(2)), energy);

  // Residual-only assembly.
  response.Reset(/*numContactElements*/ 2, /*numQuadsPerContactElement*/ 2);
  Real3 const gradient{1_r, 2_r, 3_r};
  response.AddContactSampleResponse(/*sampleIndex*/ 2, nullptr, &gradient, nullptr);
  response.ValidateInvariants(
      MakeConstSpan(std::array<bool, 2>{false, true})); // Empty energy/hessian arrays is valid
  EXPECT_NEAR_EQ(response.GetGradient(response.GetResponseIndexFromSampleIndex(2)), gradient);
}

TEST(CDeformablePointAsyncCollisionsResponse, ResetSameSizeClearsStaleState) {
  CDeformablePointAsyncCollisionsResponse response;

  // Assembly 1: elements 0 and 2 active; sample 0 carries energy 10.
  response.Reset(/*numContactElements*/ 3, /*numQuadsPerContactElement*/ 2);
  AddEnergyGradient(response, 0, 10.0, Real3{1_r, 0_r, 0_r}); // element 0
  AddEnergyGradient(response, 4, 1.0, Real3{0_r, 0_r, 1_r}); // element 2
  response.ValidateInvariants(MakeConstSpan(std::array<bool, 3>{true, false, true}));

  // Assembly 2, same sizes -> capacity-preserving sparse-clear of both lookup arrays.
  response.Reset(3, 2);
  EXPECT_TRUE(response.Empty());
  EXPECT_EQ(response.GetResponseIndexFromSampleIndex(0), kInvalid); // stale response cleared

  AddEnergyGradient(response, 0, 5.0, Real3{1_r, 0_r, 0_r}); // re-add element 0
  response.ValidateInvariants(MakeConstSpan(std::array<bool, 3>{true, false, false}));
  EXPECT_NEAR_EQ(response.GetEnergy(response.GetResponseIndexFromSampleIndex(0)), 5.0); // not 15.0
  EXPECT_SPAN_EQ( // element 2 from the previous assembly must not linger
      response.ViewActiveContactElementSubset().isElementActive,
      MakeConstSpan(std::array<bool, 3>{true, false, false}));
}

TEST(CDeformablePointAsyncCollisionsResponse, ResetDifferentSizeReinitializes) {
  CDeformablePointAsyncCollisionsResponse response;

  response.Reset(2, 2);
  AddEnergyGradient(response, 3, 1.0, Real3{1_r, 0_r, 0_r});

  response.Reset(3, 2); // sizes changed -> clear + resize branch
  EXPECT_TRUE(response.Empty());
  for (int s = 0; s < 6; ++s) {
    EXPECT_EQ(response.GetResponseIndexFromSampleIndex(s), kInvalid);
  }

  AddEnergyGradient(response, 5, 1.0, Real3{0_r, 0_r, 1_r}); // element 2
  response.ValidateInvariants(MakeConstSpan(std::array<bool, 3>{false, false, true}));
  EXPECT_SPAN_EQ(
      response.ViewActiveContactElementSubset().isElementActive,
      MakeConstSpan(std::array<bool, 3>{false, false, true}));
}

TEST(CDeformablePointAsyncCollisionsResponse, ClearAllowsReuse) {
  CDeformablePointAsyncCollisionsResponse response;
  response.Reset(2, 2);
  AddEnergyGradient(response, 0, 1.0, Real3{1_r, 0_r, 0_r});

  response.Clear();
  EXPECT_TRUE(response.Empty());

  response.Reset(2, 2); // a fresh assembly after Clear() still works
  AddEnergyGradient(response, 1, 2.0, Real3{0_r, 1_r, 0_r});
  response.ValidateInvariants(MakeConstSpan(std::array<bool, 2>{true, false}));
  EXPECT_NEAR_EQ(response.GetEnergy(response.GetResponseIndexFromSampleIndex(1)), 2.0);
}

// ---------------------------------------------------------------------------------------
// CActiveCollisions
// ---------------------------------------------------------------------------------------

using CActiveAsync = CActiveCollisions</*kIsSync*/ false, TimeStep::Current>;
using CPotentialAsync = CPotentialColliders</*kIsSync*/ false>;

[[nodiscard]] static CPotentialAsync MakePotentialColliders(
    std::initializer_list<int> colliderEntities) {
  CPotentialAsync potentialColliders;
  potentialColliders.reserve(colliderEntities.size());
  for (int entity : colliderEntities) {
    potentialColliders.emplace_back(static_cast<entt::entity>(entity));
  }
  return potentialColliders;
}

// (colliderEntity, collidingPartitionId) pairs in storage order.
[[nodiscard]] static auto EntriesOf(CActiveAsync const& activeCollisions) {
  DynamicArray<std::pair<int, int>> entries;
  entries.reserve(activeCollisions.size());
  for (auto const& collision : activeCollisions) {
    entries.emplace_back(
        static_cast<int>(collision.colliderEntity), collision.collisionResult.collidingPartitionId);
  }
  return entries;
}

TEST(CActiveCollisions, ExposesOnlyActivatedPartitionsInSortedOrder) {
  CActiveAsync activeCollisions(/*numPartitions*/ 3);
  activeCollisions.SetUp(MakePotentialColliders({30, 10, 20}));
  EXPECT_TRUE(activeCollisions.empty());

  activeCollisions.ActivatePartitionResult(static_cast<entt::entity>(30), 1);
  activeCollisions.ActivatePartitionResult(static_cast<entt::entity>(10), 2);
  activeCollisions.ActivatePartitionResult(static_cast<entt::entity>(20), 0);
  activeCollisions.FinalizeActiveCollisions();

  DynamicArray<std::pair<int, int>> const expected{{10, 2}, {20, 0}, {30, 1}};
  EXPECT_EQ(EntriesOf(activeCollisions), expected);
}

TEST(CActiveCollisions, PreservesSinglePartitionBehavior) {
  CActiveAsync activeCollisions(/*numPartitions*/ 1);
  activeCollisions.SetUp(MakePotentialColliders({30, 10, 20}));
  activeCollisions.SetUp(MakePotentialColliders({30, 20, 10}));
  EXPECT_EQ(
      (DynamicArray<std::pair<int, int>>{{10, 0}, {20, 0}, {30, 0}}), EntriesOf(activeCollisions));

  // Removing without appending preserves sorted order regardless of input order.
  activeCollisions.SetUp(MakePotentialColliders({30, 10}));
  EXPECT_EQ((DynamicArray<std::pair<int, int>>{{10, 0}, {30, 0}}), EntriesOf(activeCollisions));

  // Verify that retained data is cleared and newly appended entries preserve sorted order.
  for (auto& collision : activeCollisions) {
    EXPECT_TRUE(collision.collisionResult.isSdfGradUnitary);
    collision.collisionResult.isSdfGradUnitary = false;
    collision.collisionResult.isNormalCollidingUnitary = false;
  }
  activeCollisions.SetUp(MakePotentialColliders({30, 20}));
  EXPECT_EQ((DynamicArray<std::pair<int, int>>{{20, 0}, {30, 0}}), EntriesOf(activeCollisions));
  for (auto const& collision : activeCollisions) {
    EXPECT_TRUE(collision.collisionResult.isSdfGradUnitary);
    EXPECT_TRUE(collision.collisionResult.isNormalCollidingUnitary);
  }
}

TEST(CActiveCollisions, ReusesInactivePairStorage) {
  auto const potentialColliders = MakePotentialColliders({10, 20});
  CActiveAsync activeCollisions(/*numPartitions*/ 3);
  activeCollisions.SetUp(potentialColliders);

  auto& firstResult =
      activeCollisions.ActivatePartitionResult(static_cast<entt::entity>(10), /*partitionId*/ 1);
  firstResult.sampleIndices.reserve(8);
  size_t const firstRetainedCapacity = firstResult.sampleIndices.capacity();
  auto& secondResult =
      activeCollisions.ActivatePartitionResult(static_cast<entt::entity>(20), /*partitionId*/ 2);
  secondResult.sampleIndices.reserve(16);
  secondResult.isSdfGradUnitary = false;
  secondResult.isNormalCollidingUnitary = false;
  size_t const secondRetainedCapacity = secondResult.sampleIndices.capacity();
  activeCollisions.FinalizeActiveCollisions();

  activeCollisions.SetUp(potentialColliders);
  EXPECT_TRUE(activeCollisions.empty());

  auto& reusedFirst =
      activeCollisions.ActivatePartitionResult(static_cast<entt::entity>(10), /*partitionId*/ 1);
  EXPECT_EQ(reusedFirst.sampleIndices.capacity(), firstRetainedCapacity);
  auto& reusedSecond =
      activeCollisions.ActivatePartitionResult(static_cast<entt::entity>(20), /*partitionId*/ 2);
  EXPECT_TRUE(reusedSecond.sampleIndices.empty());
  EXPECT_TRUE(reusedSecond.isSdfGradUnitary);
  EXPECT_TRUE(reusedSecond.isNormalCollidingUnitary);
  EXPECT_EQ(reusedSecond.sampleIndices.capacity(), secondRetainedCapacity);
}

TEST(CActiveCollisions, DropsPairStorageWhenColliderIsNoLongerPotential) {
  CActiveAsync activeCollisions(/*numPartitions*/ 2);
  activeCollisions.SetUp(MakePotentialColliders({10}));
  auto& result =
      activeCollisions.ActivatePartitionResult(static_cast<entt::entity>(10), /*partitionId*/ 1);
  result.sampleIndices.reserve(16);

  activeCollisions.SetUp(MakePotentialColliders({}));
  activeCollisions.SetUp(MakePotentialColliders({10}));
  auto& recreated =
      activeCollisions.ActivatePartitionResult(static_cast<entt::entity>(10), /*partitionId*/ 1);
  EXPECT_EQ(recreated.sampleIndices.capacity(), 0);
}

TEST(CActiveCollisions, ReusesRetainedPairAfterDroppingOtherCollider) {
  CActiveAsync activeCollisions(/*numPartitions*/ 2);
  activeCollisions.SetUp(MakePotentialColliders({10, 20}));
  activeCollisions.ActivatePartitionResult(static_cast<entt::entity>(10), /*partitionId*/ 1);
  auto& retainedResult =
      activeCollisions.ActivatePartitionResult(static_cast<entt::entity>(20), /*partitionId*/ 1);
  retainedResult.sampleIndices.reserve(16);
  size_t const retainedCapacity = retainedResult.sampleIndices.capacity();
  activeCollisions.FinalizeActiveCollisions();

  activeCollisions.SetUp(MakePotentialColliders({20}));
  auto& reusedResult =
      activeCollisions.ActivatePartitionResult(static_cast<entt::entity>(20), /*partitionId*/ 1);

  EXPECT_EQ(reusedResult.sampleIndices.capacity(), retainedCapacity);
}

TEST(CActiveCollisions, RestoresAggregateStorageAfterPartitionZeroSwap) {
  auto const potentialColliders = MakePotentialColliders({10});
  CActiveAsync activeCollisions(/*numPartitions*/ 2);
  activeCollisions.SetUp(potentialColliders);

  auto& aggregate = activeCollisions.GetNarrowPhaseResult(static_cast<entt::entity>(10));
  aggregate.sampleIndices.reserve(16);
  size_t const retainedCapacity = aggregate.sampleIndices.capacity();
  activeCollisions.ActivatePartitionResult(static_cast<entt::entity>(10), /*partitionId*/ 0);
  activeCollisions.FinalizeActiveCollisions();
  std::swap(
      activeCollisions.GetNarrowPhaseResult(static_cast<entt::entity>(10)),
      activeCollisions.GetActivePartitionResult(static_cast<entt::entity>(10), /*partitionId*/ 0));

  activeCollisions.SetUp(potentialColliders);

  EXPECT_EQ(
      activeCollisions.GetNarrowPhaseResult(static_cast<entt::entity>(10)).sampleIndices.capacity(),
      retainedCapacity);
  EXPECT_TRUE(activeCollisions.empty());
}

// ---------------------------------------------------------------------------------------
// JacData
// ---------------------------------------------------------------------------------------

static JacData MakeJacData(int otherEntity = 1) {
  return {/*isSync*/ true,
          nullptr,
          /*useContactSums*/ false,
          static_cast<entt::entity>(otherEntity),
          /*partitionId*/ 0};
}

static void PopulateJac(ContactJac& jac, int dofIndex, ContactJac::AuxView jacAux = {}) {
  if (jacAux.empty()) {
    jac.Resize(/*sharedDoFs*/ true, /*sharedJacs*/ true, 1, 1);
    jac.Inds(0)[0] = dofIndex;
  } else {
    jac.Resize(/*sharedDoFs*/ true, /*sharedJacs*/ true, RigidSize::kDAll, 1);
    std::array<int, 1> const dofs{dofIndex};
    jac.SetJacAux(jacAux, 0, dofs, /*internalDofOffset*/ 0);
  }
  jac.CompressIndices();
}

static DynamicArray<ContactJac const*> GetActiveJacs(JacData const& jacData) {
  DynamicArray<ContactJac const*> activeJacs;
  jacData.GetActiveJacs(activeJacs);
  return activeJacs;
}

TEST(JacData, ZeroDofJacobian) {
  ContactDetectionResult query;
  query.sampleIndices = {0, 1, 2};
  JacData jacData = MakeJacData();
  jacData.query = &query;
  auto const expectZeroDofJacobian = [&jacData]() {
    auto const jacs = jacData.GetJacs();
    ASSERT_EQ(jacs.size(), 1);
    EXPECT_EQ(jacs[0].nContacts, 3);
    EXPECT_EQ(jacs[0].nDoFsState, 0);
    EXPECT_TRUE(jacs[0].JacAux().empty());
    EXPECT_EQ(jacs[0].internalDofOffset, kSentinelIndex);
    EXPECT_FALSE(jacData.HasSolverDoFs());
  };

  // Fresh storage
  jacData.SetZeroDofJacobian();
  expectZeroDofJacobian();

  // Recycled slice with DoFs and an auxiliary Jacobian
  auto const jacAux = RowMatrix<real, RigidSize::kDAll>::Zero(RigidSize::kDAll, 1);
  auto& jac = jacData.PrepareJacs(1)[0];
  PopulateJac(jac, 4, jacAux);
  jacData.SetZeroDofJacobian();
  expectZeroDofJacobian();
}

TEST(JacData, InvalidateDeactivatesPreparedSlices) {
  ContactDetectionResult query;
  JacData jacData = MakeJacData();
  jacData.query = &query;

  // Only populated slices are active, in slice order.
  auto jacs = jacData.PrepareJacs(2);
  PopulateJac(jacs[1], 7);
  auto activeJacs = GetActiveJacs(jacData);
  ASSERT_EQ(activeJacs.size(), 1);
  EXPECT_EQ(activeJacs[0], &jacs[1]);

  PopulateJac(jacs[0], 4);
  jacData.Invalidate();
  EXPECT_EQ(jacData.query, nullptr);
  EXPECT_TRUE(GetActiveJacs(jacData).empty());

  // Re-preparing reuses the slices, and re-populating one reactivates only that one.
  EXPECT_EQ(jacData.PrepareJacs(2).data(), jacs.data());
  PopulateJac(jacs[1], 8);
  activeJacs = GetActiveJacs(jacData);
  ASSERT_EQ(activeJacs.size(), 1);
  EXPECT_EQ(activeJacs[0], &jacs[1]);
}
