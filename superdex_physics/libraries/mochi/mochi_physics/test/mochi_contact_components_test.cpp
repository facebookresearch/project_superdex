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

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/vmatrix.h>

#include <gtest/gtest.h>

#include <array>
#include <initializer_list>
#include <utility>

using namespace mochi;
using namespace mochi::test;

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

using CActiveAsync = CActiveCollisions<ContactType::Async, TimeStep::Current>;
using CPotentialAsync = CPotentialColliders<ContactType::Async>;

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

TEST(CActiveCollisions, SetUpBuildsSortedPartitionedEntries) {
  CActiveAsync activeCollisions;
  activeCollisions.SetUp(MakePotentialColliders({30, 10, 20}), /*numPartitions*/ 2);

  // Each collider is expanded to all partitions and sorted by (colliderEntity,
  // collidingPartitionId), regardless of input order.
  DynamicArray<std::pair<int, int>> const expected{
      {10, 0}, {10, 1}, {20, 0}, {20, 1}, {30, 0}, {30, 1}};
  EXPECT_EQ(EntriesOf(activeCollisions), expected);

  // A single partition expands each collider to exactly one entry.
  CActiveAsync singlePartition;
  singlePartition.SetUp(MakePotentialColliders({30, 10, 20}), /*numPartitions*/ 1);
  DynamicArray<std::pair<int, int>> const expectedSingle{{10, 0}, {20, 0}, {30, 0}};
  EXPECT_EQ(EntriesOf(singlePartition), expectedSingle);
}

TEST(CActiveCollisions, SetUpAddsRemovesAndKeepsAcrossCalls) {
  CActiveAsync activeCollisions;
  activeCollisions.SetUp(MakePotentialColliders({10, 20}), /*numPartitions*/ 2);

  // Flag every entry so we can verify SetUp clears the retained data of kept colliders.
  // isSdfGradUnitary is a convenient observable: ContactDetectionResult::Clear() resets it to true,
  // and (unlike the contact arrays) it needs no size-consistent setup.
  for (auto& collision : activeCollisions) {
    EXPECT_TRUE(collision.collisionResult.isSdfGradUnitary); // Just checking
    collision.collisionResult.isSdfGradUnitary = false;
  }

  // Re-run with a different collider set: 10 removed, 20 kept (not duplicated), 30 added.
  activeCollisions.SetUp(MakePotentialColliders({20, 30}), /*numPartitions*/ 2);

  DynamicArray<std::pair<int, int>> const expected{{20, 0}, {20, 1}, {30, 0}, {30, 1}};
  EXPECT_EQ(EntriesOf(activeCollisions), expected);

  // Kept collider 20 had its retained data cleared; added collider 30 starts fresh, so every
  // surviving entry's contact result must be reset. Catches removal of the SetUp clear loop.
  for (auto const& collision : activeCollisions) {
    EXPECT_TRUE(collision.collisionResult.isSdfGradUnitary);
  }

  // Re-running with no potential colliders removes everything.
  activeCollisions.SetUp(MakePotentialColliders({}), /*numPartitions*/ 2);
  EXPECT_TRUE(activeCollisions.empty());
}
