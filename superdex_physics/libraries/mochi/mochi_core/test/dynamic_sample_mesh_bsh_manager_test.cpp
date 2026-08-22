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

#include <mochi_core/rom/dynamic_sample_mesh_bsh_manager.h>

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/constants.h>

#include <array>

#include <gtest/gtest.h>

using namespace mochi;
using namespace mochi::rom::hyper;

static ContactSamplesBsh CreateFlatBsh() {
  ContactSamplesBsh bsh(ContactSamplesBsh::NodeData{.sampleIdx = -1});
  for (int sampleIdx = 0; sampleIdx < 4; ++sampleIdx) {
    bsh.AddNode(ContactSamplesBsh::NodeData{.sampleIdx = sampleIdx}, ContactSamplesBsh::RootIdx());
  }
  return bsh;
}

TEST(DynamicSampleMeshBshManager, UpdatesFiniteSamplesAndSkipsMissingSamples) {
  DynamicSampleMeshBshManager manager{CreateFlatBsh(), SdfLowerBoundAnchorSelection::Self};

  constexpr std::array<Real3, 4> kPositions = {
      Real3{0_r, 0_r, 0_r},
      Real3{1_r, 0_r, 0_r},
      Real3{0_r, 1_r, 0_r},
      Real3{1_r, 1_r, 0_r},
  };
  constexpr std::array<int, 4> kActiveSampleIndices = {0, 1, 2, 3};
  constexpr std::array<real, 4> kInitialDistances = {1_r, 2_r, 3_r, 4_r};

  manager.Update(
      kPositions,
      kActiveSampleIndices,
      kInitialDistances,
      /*currentTime*/ 0.0,
      /*maxVelocitySinceLastUpdate*/ 0_r,
      /*maxDistance*/ 100_r);

  // Second update: samples 0 and 2 receive fresh finite measurements; samples 1 and 3 report a
  // missing (inf) measurement and must keep their previous data.
  constexpr real kRefreshedDistance0 = 0.1_r;
  constexpr real kRefreshedDistance2 = 0.3_r;
  constexpr std::array<real, 4> kMixedDistances = {
      kRefreshedDistance0,
      kInf,
      kRefreshedDistance2,
      kInf,
  };
  constexpr double kUpdateDeltaTime = 0.25; // First update was at t = 0.
  constexpr real kMaxVelocity = 2_r;
  manager.Update(
      kPositions,
      kActiveSampleIndices,
      kMixedDistances,
      /*currentTime*/ kUpdateDeltaTime,
      /*maxVelocitySinceLastUpdate*/ kMaxVelocity,
      /*maxDistance*/ 100_r);

  // One update's worth of motion penalty accrues on every sample, then is reset to zero on the
  // samples that received a fresh measurement.
  constexpr real kAccruedPenalty = static_cast<real>(kUpdateDeltaTime) * kMaxVelocity;

  // Refreshed samples: distance == new measurement, penalty reset to zero.
  EXPECT_EQ(kRefreshedDistance0, manager.LastSampleData(0).distance);
  EXPECT_EQ(0_r, manager.LastSampleData(0).velocityPenalty);
  EXPECT_EQ(kRefreshedDistance2, manager.LastSampleData(2).distance);
  EXPECT_EQ(0_r, manager.LastSampleData(2).velocityPenalty);

  // Missing samples: distance preserved from the initial update, penalty accrued.
  EXPECT_EQ(kInitialDistances[1], manager.LastSampleData(1).distance);
  EXPECT_EQ(kAccruedPenalty, manager.LastSampleData(1).velocityPenalty);
  EXPECT_EQ(kInitialDistances[3], manager.LastSampleData(3).distance);
  EXPECT_EQ(kAccruedPenalty, manager.LastSampleData(3).velocityPenalty);
}

TEST(DynamicSampleMeshBshManager, MissingMeasurementsAccumulateVelocityPenalty) {
  DynamicSampleMeshBshManager manager{CreateFlatBsh(), SdfLowerBoundAnchorSelection::Self};

  constexpr std::array<Real3, 4> kPositions = {
      Real3{0_r, 0_r, 0_r},
      Real3{1_r, 0_r, 0_r},
      Real3{0_r, 1_r, 0_r},
      Real3{1_r, 1_r, 0_r},
  };
  constexpr std::array<int, 1> kActiveSampleIndices = {0};
  constexpr std::array<real, 4> kInitialDistances = {1_r, 2_r, 3_r, 4_r};

  manager.Update(
      kPositions,
      kActiveSampleIndices,
      kInitialDistances,
      /*currentTime*/ 0.0,
      /*maxVelocitySinceLastUpdate*/ 0_r,
      /*maxDistance*/ 100_r);

  constexpr std::array<real, 4> kMissingDistances = {
      kInf,
      kInf,
      kInf,
      kInf,
  };

  // Each missing update advances time by the same step and reports the same max velocity, so every
  // step adds the same motion penalty. Sample 0 keeps its last finite distance and, because its
  // position never changes, its lower bound is that distance minus the accumulated penalty.
  constexpr double kStepDeltaTime = 0.25;
  constexpr real kMaxVelocity = 1_r;
  constexpr real kPenaltyPerStep = static_cast<real>(kStepDeltaTime) * kMaxVelocity;
  constexpr real kKnownDistance = kInitialDistances[0];

  manager.Update(
      kPositions,
      kActiveSampleIndices,
      kMissingDistances,
      /*currentTime*/ kStepDeltaTime,
      /*maxVelocitySinceLastUpdate*/ kMaxVelocity,
      /*maxDistance*/ 100_r);

  int const sampleNodeIdx = manager.GetBsh().SampleIdxToNodeIdx(0);
  EXPECT_EQ(kPenaltyPerStep, manager.LastSampleData(0).velocityPenalty);
  EXPECT_EQ(
      kKnownDistance - kPenaltyPerStep,
      manager.EvaluateSdfLowerBound(sampleNodeIdx, kPositions[0]));

  manager.Update(
      kPositions,
      kActiveSampleIndices,
      kMissingDistances,
      /*currentTime*/ 2 * kStepDeltaTime,
      /*maxVelocitySinceLastUpdate*/ kMaxVelocity,
      /*maxDistance*/ 100_r);

  EXPECT_EQ(2_r * kPenaltyPerStep, manager.LastSampleData(0).velocityPenalty);
  EXPECT_EQ(
      kKnownDistance - 2_r * kPenaltyPerStep,
      manager.EvaluateSdfLowerBound(sampleNodeIdx, kPositions[0]));
}

TEST(DynamicSampleMeshBshManager, KeepsKnownLowerBoundWhenCurrentDistanceIsMissing) {
  DynamicSampleMeshBshManager manager{CreateFlatBsh(), SdfLowerBoundAnchorSelection::Self};

  constexpr std::array<Real3, 4> kPositions = {
      Real3{0_r, 0_r, 0_r},
      Real3{1_r, 0_r, 0_r},
      Real3{0_r, 1_r, 0_r},
      Real3{1_r, 1_r, 0_r},
  };
  constexpr std::array<int, 4> kActiveSampleIndices = {0, 1, 2, 3};
  constexpr std::array<real, 4> kNearDistances = {0.05_r, 10_r, 10_r, 10_r};

  manager.Update(
      kPositions,
      kActiveSampleIndices,
      kNearDistances,
      /*currentTime*/ 0.0,
      /*maxVelocitySinceLastUpdate*/ 0_r,
      /*maxDistance*/ 100_r);

  constexpr std::array<real, 4> kMissingDistances = {
      kInf,
      kInf,
      kInf,
      kInf,
  };
  manager.Update(
      kPositions,
      kActiveSampleIndices,
      kMissingDistances,
      /*currentTime*/ 0.01,
      /*maxVelocitySinceLastUpdate*/ 0_r,
      /*maxDistance*/ 100_r);

  manager.ComputeActiveBoundaryFaces(
      /*maxActiveSamples*/ 2, /*sampleActivationThreshold*/ 0.1_r);

  constexpr std::array<int, 1> kExpectedActiveFaces = {0};
  EXPECT_SPAN_EQ(kExpectedActiveFaces, manager.ActiveBoundaryFaceIndices());
  // Sample 0's distance is preserved across the missing update.
  EXPECT_EQ(kNearDistances[0], manager.LastSampleData(0).distance);
}

TEST(DynamicSampleMeshBshManager, CanReturnNoActiveSamplesWhenAllKnownBoundsAreFar) {
  DynamicSampleMeshBshManager manager{
      CreateFlatBsh(), SdfLowerBoundAnchorSelection::AncestorSibling};

  constexpr std::array<Real3, 4> kPositions = {
      Real3{0_r, 0_r, 0_r},
      Real3{1_r, 0_r, 0_r},
      Real3{0_r, 1_r, 0_r},
      Real3{1_r, 1_r, 0_r},
  };
  constexpr std::array<int, 4> kActiveSampleIndices = {0, 1, 2, 3};
  constexpr std::array<real, 4> kFarDistances = {10_r, 10_r, 10_r, 10_r};

  manager.Update(
      kPositions,
      kActiveSampleIndices,
      kFarDistances,
      /*currentTime*/ 0.0,
      /*maxVelocitySinceLastUpdate*/ 0_r,
      /*maxDistance*/ 100_r);

  manager.ComputeActiveBoundaryFaces(
      /*maxActiveSamples*/ 2, /*sampleActivationThreshold*/ 0.1_r);

  EXPECT_TRUE(manager.ActiveBoundaryFaceIndices().empty());
  EXPECT_TRUE(manager.ActiveBoundaryFaceWeightMultipliers().empty());
}
