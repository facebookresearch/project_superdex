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

#include <gtest/gtest.h>

#include <mochi_core/articulated_body/articulated_body.h>
#include <mochi_core/test/linear_algebra_test_helpers.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/constants.h>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

using namespace mochi;
using namespace mochi::articulated;
using namespace mochi::articulated::internal;
using namespace mochi::test;

/*************************************************************************************************/

namespace {
// Define constant values
constexpr real k0 = real(0_r / 180_r * kPI);
constexpr real k20 = real(20_r / 180_r * kPI);
constexpr real k120 = real(120_r / 180_r * kPI);

constexpr Real3 kv0 = {0_r, 0_r, 0_r};
constexpr Real3 kvx = {1_r, 0_r, 0_r};
constexpr Real3 kvy = {0_r, 1_r, 0_r};
constexpr Real3 kvz = {0_r, 0_r, 1_r};

// Bundles the per-joint spans that the articulated API consumes.
struct JointLayout {
  DynamicArray<ArticulatedJointType> types;
  DynamicArray<Real3> axes;
  DynamicArray<ArticulatedDofInfo> dofs;
  DynamicArray<ArticulatedPoseInfo> pose;
};

// Build a JointLayout from joint types and axes, computing the dof- and pose-space layouts.
JointLayout MakeLayout(DynamicArray<ArticulatedJointType>&& types, DynamicArray<Real3>&& axes) {
  DynamicArray<ArticulatedDofInfo> dofs = SetupJointDofs(types);
  DynamicArray<ArticulatedPoseInfo> pose = SetupJointPose(types);
  return JointLayout{std::move(types), std::move(axes), std::move(dofs), std::move(pose)};
}

// Common data structure to hold test data and expected result
struct TestConfig {
  ArticulatedJointType type;
  Real3 axis;
  ArticulatedPoseInfo poseInfo;
  ColumnVector<real> jointDofs;
  TransformRT jointTransform;

  static bool cmp(TestConfig const& a, TestConfig const& b) {
    return a.type < b.type;
  }
};

// Build a single-joint TestConfig, deriving the joint's pose-space layout from its type.
template <typename DofsT>
TestConfig MakeTestConfig(
    ArticulatedJointType type,
    Real3 const& axis,
    DofsT const& jointDofs,
    TransformRT const& jointTransform) {
  auto const size = GetJointTypeNumPose(type);
  return TestConfig{
      type, axis, ArticulatedPoseInfo{0, size.first, size.second}, jointDofs, jointTransform};
}

// Collect the per-joint types and axes from a list of test configs into a JointLayout.
JointLayout LayoutFromConfigs(Span<TestConfig const> configs) {
  DynamicArray<ArticulatedJointType> types;
  DynamicArray<Real3> axes;
  types.reserve(configs.size());
  axes.reserve(configs.size());
  for (auto const& tc : configs) {
    types.push_back(tc.type);
    axes.push_back(tc.axis);
  }
  return MakeLayout(std::move(types), std::move(axes));
}

void RunTests_JointTransformFromJointDofs(Span<TestConfig const> testConfigs) {
  for (auto const& tc : testConfigs) {
    TransformRT tx =
        ComputeJointTransform(AsConstView(tc.jointDofs), tc.type, tc.axis, tc.poseInfo);
    EXPECT_NEAR_EQ(tx.GetTranslation(), tc.jointTransform.GetTranslation());
    EXPECT_NEAR_EQ(tx.GetRotation(), tc.jointTransform.GetRotation());
  }
}
} // namespace

TEST(ArticulatedBody, JointTransformFromJointDofs_FreeJoint) {
  // Define test case combinations
  std::vector<Real3> kvs = {kvx, Normalize(kvx + kvy), Normalize(kvx + kvy + kvz)};
  std::vector<real> kAngles = {k0, k20, k120, -k120};

  // Create test config structures
  std::vector<TestConfig> testConfigs;
  size_t numKvs = int(kvs.size());
  size_t numAngles = int(kAngles.size());
  testConfigs.reserve(numKvs * numKvs * numAngles);
  for (auto kv : kvs) {
    for (auto qv : kvs) {
      for (auto angle : kAngles) {
        TransformRT tx(Quaternion::FromAxisAngle(qv, angle), kv);
        ColumnVector<real, RigidSize::kAll> pose;
        TransformToRawPose(tx, pose);

        // GCC is afraid that we may read 32-bytes (via Simd<float, 8>) from the "pose" vector which
        // is only 28 bytes. GCC is wrong. The read in question is in a branch that only runs if
        // there are >= Simd<float>::kSize values remaining in the buffer.
        MOCHI_WARNING_PUSH()
        MOCHI_WARNING_IGNORE_GCC(GCC diagnostic ignored "-Wstringop-overread")

        testConfigs.emplace_back(
            MakeTestConfig(ArticulatedJointType::Free, kv0, AsConstView(pose), tx));

        MOCHI_WARNING_POP()
      }
    }
  }

  // Compute joint transform from joint dofs and test expected value
  RunTests_JointTransformFromJointDofs(testConfigs);
}

TEST(ArticulatedBody, JointTransformFromJointDofs_SphericalJoint) {
  // Define test case combinations
  std::vector<Real3> kvs = {kvx, Normalize(kvx + kvy), Normalize(kvx + kvy + kvz)};
  std::vector<real> kAngles = {k0, k20, k120, -k120};

  // Create test config structures
  std::vector<TestConfig> testConfigs;
  size_t numKvs = int(kvs.size());
  size_t numAngles = int(kAngles.size());
  testConfigs.reserve(numKvs * numAngles);
  for (auto qv : kvs) {
    for (auto angle : kAngles) {
      TransformRT tx(Quaternion::FromAxisAngle(qv, angle), kv0);
      testConfigs.emplace_back(MakeTestConfig(
          ArticulatedJointType::Spherical, kv0, AsColumnVectorView(tx.GetRotation().data), tx));
    }
  }

  // Compute joint transform from joint dofs and test expected value
  RunTests_JointTransformFromJointDofs(testConfigs);
}

TEST(ArticulatedBody, JointTransformFromJointDofs_PrismaticJoint) {
  // Define test case combinations
  std::vector<Real3> kvs = {kvx, Normalize(kvx + kvy), Normalize(kvx + kvy + kvz)};
  std::vector<real> kValues = {0_r, 0.1_r, -0.1_r, -1.1_r};
  Quaternion qI = Quaternion::Identity();

  // Create test config structures
  std::vector<TestConfig> testConfigs;
  size_t numKvs = int(kvs.size());
  size_t numValues = int(kValues.size());
  testConfigs.reserve(numKvs * numValues);
  for (auto kv : kvs) {
    for (auto val : kValues) {
      TransformRT tx(qI, val * kv);
      ColumnVector<real> dofs(1);
      dofs.SetConstant(val);
      testConfigs.emplace_back(
          MakeTestConfig(ArticulatedJointType::Prismatic, kv, AsConstView(dofs), tx));
    }
  }

  // Compute joint transform from joint dofs and test expected value
  RunTests_JointTransformFromJointDofs(testConfigs);
}

TEST(ArticulatedBody, JointTransformFromJointDofs_RevoluteJoint) {
  // Define test case combinations
  std::vector<Real3> kvs = {kvx, Normalize(kvx + kvy), Normalize(kvx + kvy + kvz)};
  std::vector<real> kAngles = {k0, k20, k120, -k120};

  // Create test config structures
  std::vector<TestConfig> testConfigs;
  size_t numKvs = int(kvs.size());
  size_t numAngles = int(kAngles.size());
  testConfigs.reserve(numKvs * numAngles);
  for (auto kv : kvs) {
    for (auto angle : kAngles) {
      TransformRT tx(Quaternion::FromAxisAngle(kv, angle), kv0);
      ColumnVector<real> dofs(1);
      dofs.SetConstant(angle);
      testConfigs.emplace_back(
          MakeTestConfig(ArticulatedJointType::Revolute, kv, AsConstView(dofs), tx));
    }
  }

  // Compute joint transform from joint dofs and test expected value
  RunTests_JointTransformFromJointDofs(testConfigs);
}

/*************************************************************************************************/
namespace {
Quaternion q20y = Quaternion::FromAxisAngle(kvy, k20);
Quaternion q120z = Quaternion::FromAxisAngle(kvz, k120);

std::vector<TestConfig> CreateJointTransformsTestConfigs() {
  // Define a collection of test configs containing the 4 types of joint types
  Real3 const prismaticAxis = kvx;
  Real3 const revoluteAxis = kvx;

  // Define free joint type
  ColumnVector<real, RigidSize::kAll> freeDofs;
  TransformToRawPose(TransformRT{q20y, kvx}, freeDofs);
  TransformRT freeTx = TransformRT(q20y, kvx);
  TestConfig freeTestConfig =
      MakeTestConfig(ArticulatedJointType::Free, kv0, AsConstView(freeDofs), freeTx);

  // Define spherical joint type
  ColumnVector<real> sphericalDofs = AsColumnVectorView(q120z.data);
  TransformRT sphericalTx = TransformRT(q120z, Real3{});
  TestConfig sphericalTestConfig =
      MakeTestConfig(ArticulatedJointType::Spherical, kv0, AsConstView(sphericalDofs), sphericalTx);

  // Define prismatic joint type
  real prismaticDofVal = 0.1_r;
  ColumnVector<real> prismaticDofs(1);
  prismaticDofs.SetConstant(prismaticDofVal);
  TransformRT prismaticTx = TransformRT({}, prismaticAxis * prismaticDofVal);
  TestConfig prismaticTestConfig = MakeTestConfig(
      ArticulatedJointType::Prismatic, prismaticAxis, AsConstView(prismaticDofs), prismaticTx);

  // Define revolute joint type
  real revoluteDofVal = 0.1_r;
  ColumnVector<real> revoluteDofs(1);
  revoluteDofs.SetConstant(revoluteDofVal);
  TransformRT revoluteTx =
      TransformRT(Quaternion::FromRotationVector(revoluteAxis * revoluteDofVal), Real3{});
  TestConfig revoluteTestConfig = MakeTestConfig(
      ArticulatedJointType::Revolute, revoluteAxis, AsConstView(revoluteDofs), revoluteTx);

  // Define collection of test configs and compute reduced dofs size
  std::vector<TestConfig> testConfigs = {
      freeTestConfig, sphericalTestConfig, prismaticTestConfig, revoluteTestConfig};

  std::sort(testConfigs.begin(), testConfigs.end(), TestConfig::cmp);
  return testConfigs;
};

int GetNumReducedDofs(Span<TestConfig const> testConfigs) {
  int numReducedDofs = std::accumulate(
      testConfigs.begin(), testConfigs.end(), 0, [](int count, TestConfig const& tc) {
        return count + tc.jointDofs.Rows();
      });
  return numReducedDofs;
}

void AssembleReducedDofs(Span<TestConfig const> testConfigs, ColumnVectorView<real> reducedPose) {
  int offset = 0;
  for (auto const& tc : testConfigs) {
    int size = tc.poseInfo.GetSize();
    reducedPose.Slice(offset, size) = tc.jointDofs;
    offset += size;
  }
}
}; // namespace

TEST(ArticulatedBody, ComputeJointTransformsFromReducedDofs) {
  // Create test configs and sort
  std::vector<TestConfig> testConfigs = CreateJointTransformsTestConfigs();
  int numReducedDofs = GetNumReducedDofs(testConfigs);

  // Compute permutations of test configs collection
  ColumnVector<real> reducedPose(numReducedDofs);
  do {
    // Assemble reduced dofs from test configs permutation
    AssembleReducedDofs(testConfigs, reducedPose);

    // Compute joint transforms
    JointLayout const layout = LayoutFromConfigs(testConfigs);
    std::vector<TransformRT> jointTransforms(testConfigs.size());
    ComputeActiveJointTransforms(
        layout.types, layout.axes, layout.pose, reducedPose, MakeSpan(jointTransforms));

    // Compare to expected joint transforms
    for (auto i = 0; i < testConfigs.size(); ++i) {
      EXPECT_NEAR_EQ(jointTransforms[i], testConfigs[i].jointTransform);
    }
  } while (std::next_permutation(testConfigs.begin(), testConfigs.end(), TestConfig::cmp));
}

/*************************************************************************************************/
namespace {
void RunTests_JointDofsFromJointTransform(Span<TestConfig const> testConfigs) {
  for (auto tc : testConfigs) {
    ColumnVector<real> jointDofs(tc.poseInfo.GetSize());
    ComputeJointPose(tc.type, tc.axis, tc.poseInfo, tc.jointTransform, jointDofs);
    // TODO: check how to use EXPECT_NEAR_EQ with ColumnVector
    for (auto i = 0; i < jointDofs.Rows(); ++i) {
      EXPECT_NEAR_RTOL(jointDofs[i], tc.jointDofs[i], 1.0e-5);
    }
  }
}
}; // namespace

TEST(ArticulatedBody, JointDofsFromJointTransform_FreeJoint) {
  // Define test case combinations
  std::vector<Real3> kvs = {kvx, Normalize(kvx + kvy), Normalize(kvx + kvy + kvz)};
  std::vector<real> kAngles = {k0, k20, k120, -k120};

  // Create test config structures
  std::vector<TestConfig> testConfigs;
  size_t numKvs = int(kvs.size());
  size_t numAngles = int(kAngles.size());
  testConfigs.reserve(numKvs * numKvs * numAngles);
  for (auto kv : kvs) {
    for (auto qv : kvs) {
      for (auto angle : kAngles) {
        TransformRT tx(Quaternion::FromAxisAngle(qv, angle), kv);
        ColumnVector<real, RigidSize::kAll> pose;
        TransformToRawPose(tx, pose);

        // GCC is afraid that we may read 32-bytes (via Simd<float, 8>) from the "pose" vector which
        // is only 28 bytes. GCC is wrong. The read in question is in a branch that only runs if
        // there are >= Simd<float>::kSize values remaining in the buffer.
        MOCHI_WARNING_PUSH()
        MOCHI_WARNING_IGNORE_GCC(GCC diagnostic ignored "-Wstringop-overread")

        testConfigs.emplace_back(
            MakeTestConfig(ArticulatedJointType::Free, kv0, AsConstView(pose), tx));

        MOCHI_WARNING_POP()
      }
    }
  }

  // Compute joint transform from joint dofs and test expected value
  RunTests_JointDofsFromJointTransform(testConfigs);
}

TEST(ArticulatedBody, JointDofsFromJointTransform_SphericalJoint) {
  // Define test case combinations
  std::vector<Real3> kvs = {kvx, Normalize(kvx + kvy), Normalize(kvx + kvy + kvz)};
  std::vector<real> kAngles = {k0, k20, k120, -k120};

  // Create test config structures
  std::vector<TestConfig> testConfigs;
  size_t numKvs = int(kvs.size());
  size_t numAngles = int(kAngles.size());
  testConfigs.reserve(numKvs * numAngles);
  for (auto qv : kvs) {
    for (auto angle : kAngles) {
      TransformRT tx(Quaternion::FromAxisAngle(qv, angle), kv0);
      testConfigs.emplace_back(MakeTestConfig(
          ArticulatedJointType::Spherical, kv0, AsColumnVectorView(tx.GetRotation().data), tx));
    }
  }

  // Compute joint transform from joint dofs and test expected value
  RunTests_JointDofsFromJointTransform(testConfigs);
}

TEST(ArticulatedBody, JointDofsFromJointTransform_PrismaticJoint) {
  // Define test case combinations
  std::vector<Real3> kvs = {kvx, Normalize(kvx + kvy), Normalize(kvx + kvy + kvz)};
  std::vector<real> kValues = {0_r, 0.1_r, -0.1_r, -1.1_r};
  Quaternion qI = Quaternion::Identity();

  // Create test config structures
  std::vector<TestConfig> testConfigs;
  size_t numKvs = int(kvs.size());
  size_t numValues = int(kValues.size());
  testConfigs.reserve(numKvs * numValues);
  for (auto kv : kvs) {
    for (auto val : kValues) {
      TransformRT tx(qI, val * kv);
      ColumnVector<real> dofs(1);
      dofs.SetConstant(val);
      testConfigs.emplace_back(
          MakeTestConfig(ArticulatedJointType::Prismatic, kv, AsConstView(dofs), tx));
    }
  }

  // Compute joint transform from joint dofs and test expected value
  RunTests_JointDofsFromJointTransform(testConfigs);
}

TEST(ArticulatedBody, JointDofsFromJointTransform_RevoluteJoint) {
  // Define test case combinations
  std::vector<Real3> kvs = {kvx, Normalize(kvx + kvy), Normalize(kvx + kvy + kvz)};
  std::vector<real> kAngles = {k0, k20, k120, -k120};

  // Create test config structures
  std::vector<TestConfig> testConfigs;
  size_t numKvs = int(kvs.size());
  size_t numAngles = int(kAngles.size());
  testConfigs.reserve(numKvs * numAngles);
  for (auto kv : kvs) {
    for (auto angle : kAngles) {
      ColumnVector<real> dofs(1);
      dofs.SetConstant(angle);
      Quaternion q = Quaternion::FromAxisAngle(kv, angle);
      testConfigs.emplace_back(MakeTestConfig(
          ArticulatedJointType::Revolute, kv, AsConstView(dofs), TransformRT(q, kv0)));
    }
  }

  // Compute joint transform from joint dofs and test expected value
  RunTests_JointDofsFromJointTransform(testConfigs);
}

/*************************************************************************************************/
TEST(ArticulatedBody, ComputeReducedDofsFromJointTransforms) {
  // Create different configs with different joint types
  std::vector<TestConfig> testConfigs = CreateJointTransformsTestConfigs();
  int numReducedDofs = GetNumReducedDofs(testConfigs);

  // For each possible permutation of test configs
  ColumnVector<real> expectedReducedDofs(numReducedDofs);
  ColumnVector<real> reducedPose(numReducedDofs);
  do {
    // Compute expected reduced dofs
    AssembleReducedDofs(testConfigs, expectedReducedDofs);

    // Compute joint transforms
    JointLayout const layout = LayoutFromConfigs(testConfigs);
    std::vector<TransformRT> jointTransforms;
    jointTransforms.reserve(testConfigs.size());
    std::transform(
        testConfigs.begin(),
        testConfigs.end(),
        std::back_inserter(jointTransforms),
        [](TestConfig const& tc) { return tc.jointTransform; });
    ComputeReducedPose(layout.types, layout.axes, layout.pose, jointTransforms, reducedPose);

    // Compare to expected joint transforms
    for (auto j = 0; j < reducedPose.Rows(); ++j) {
      // TODO: Implement EXPECT_NEAR_EQ for krylov vectors
      EXPECT_NEAR_EQ(reducedPose[j], expectedReducedDofs[j]);
    }
  } while (std::next_permutation(testConfigs.begin(), testConfigs.end(), TestConfig::cmp));
}

/*************************************************************************************************/
namespace {
real constexpr k45 = 45 * kRadiansPerDegree;
real constexpr k90 = 90 * kRadiansPerDegree;

struct ArticulatedBodyTestConfig {
  ParentIndexArray parents;
  JointLayout layout;
  RestTransformArray restTransforms;
  std::vector<TransformRT> jointTransforms;
  std::vector<TransformRT> parentFromBoneTransforms;
  std::vector<TransformRT> worldFromBoneTransforms;
  ColumnVector<real> fullDofs;
  ColumnVector<real> reducedPose;
};

std::vector<ArticulatedBodyTestConfig> CreateArticulatedBodyTestConfigs() {
  std::vector<ArticulatedBodyTestConfig> testConfigs;
  {
    // Create test config with 1 link
    // o----o
    ParentIndexArray parents = {-1};
    JointLayout layout = MakeLayout({ArticulatedJointType::Free}, {kv0});
    RestTransformArray restTransforms = {{TransformRT::Identity(), TransformRT::Identity()}};
    Real3 tJ0 = {};
    Quaternion qJ0 = Quaternion::FromAxisAngle(kvz, k20);
    std::vector<TransformRT> jointTransforms = {TransformRT(qJ0, tJ0)};
    std::vector<TransformRT> parentFromBoneTransforms = {TransformRT(qJ0, tJ0)};
    std::vector<TransformRT> worldFromBoneTransforms = {TransformRT(qJ0, tJ0)};
    ColumnVector<real, RigidSize::kAll> pose; // Full and reduced pose match for this test
    TransformToRawPose(worldFromBoneTransforms[0], pose);
    testConfigs.push_back(
        {parents,
         layout,
         restTransforms,
         jointTransforms,
         parentFromBoneTransforms,
         worldFromBoneTransforms,
         pose,
         pose});
  }

  {
    // Create test config with 2 links arranged in series
    // o----o----o
    ParentIndexArray parents = {-1, 0};
    JointLayout layout =
        MakeLayout({ArticulatedJointType::Free, ArticulatedJointType::Spherical}, {kv0, kv0});
    Real3 tOF1_0 = {1_r, 0_r, 0_r};
    Quaternion qOF1_0 = Quaternion::Identity();
    RestTransformArray restTransforms = {
        {TransformRT::Identity(), TransformRT::Identity()},
        {TransformRT::Identity(), TransformRT(qOF1_0, tOF1_0)}};
    Real3 tJ0 = {};
    Quaternion qJ0 = Quaternion::Identity();
    Real3 tJ1 = {};
    Quaternion qJ1 = Quaternion::FromAxisAngle(kvz, k20);
    std::vector<TransformRT> jointTransforms = {TransformRT(qJ0, tJ0), TransformRT(qJ1, tJ1)};
    std::vector<TransformRT> parentFromBoneTransforms = {
        TransformRT::Identity(), TransformRT(Quaternion::FromAxisAngle(kvz, k20), tOF1_0)};
    std::vector<TransformRT> worldFromBoneTransforms = {
        TransformRT::Identity(), TransformRT(qOF1_0, tOF1_0) * TransformRT(qJ1, tJ1)};
    ColumnVector<real, 2 * RigidSize::kAll> fullDofs;
    TransformToRawPose(
        TransformRT::Identity(),
        fullDofs.TopRows<RigidSize::kAll>(RigidSize::kAll)); // First full dofs correspond to origin
    TransformToRawPose(
        worldFromBoneTransforms[1],
        fullDofs.Slice<RigidSize::kAll>(RigidSize::kAll, RigidSize::kAll));
    ColumnVector<real, RigidSize::kAll + RigidSize::kRot> reducedPose;
    TransformToRawPose(
        TransformRT::Identity(), reducedPose.TopRows<RigidSize::kAll>(RigidSize::kAll));
    reducedPose.BottomRows<RigidSize::kRot>(RigidSize::kRot) = AsColumnVectorView(qJ1.data);
    testConfigs.push_back(
        {parents,
         layout,
         restTransforms,
         jointTransforms,
         parentFromBoneTransforms,
         worldFromBoneTransforms,
         fullDofs,
         reducedPose});
  }

  {
    // Create test config with 3 links arranged in series
    // o----o----o----o
    ParentIndexArray parents = {-1, 0, 1};
    JointLayout layout = MakeLayout(
        {ArticulatedJointType::Free,
         ArticulatedJointType::Spherical,
         ArticulatedJointType::Revolute},
        {kv0, kv0, kvz});
    Real3 tOF1_0 = {2_r, 0_r, 0_r};
    Quaternion qOF1_0 = Quaternion::Identity();
    Real3 tOF2_1 = {1_r, 0_r, 0_r};
    Quaternion qOF2_1 = Quaternion::Identity();
    RestTransformArray restTransforms = {
        {TransformRT::Identity(), TransformRT::Identity()},
        {TransformRT::Identity(), TransformRT(qOF1_0, tOF1_0)},
        {TransformRT::Identity(), TransformRT(qOF2_1, tOF2_1)}};
    Real3 tJ0 = {};
    Quaternion qJ0 = Quaternion::Identity();
    Real3 tJ1 = {};
    Quaternion qJ1 = Quaternion::FromAxisAngle(kvz, k20);
    Real3 tJ2 = {};
    Quaternion qJ2 = Quaternion::FromAxisAngle(kvz, -k20);
    std::vector<TransformRT> jointTransforms = {
        TransformRT(qJ0, tJ0), TransformRT(qJ1, tJ1), TransformRT(qJ2, tJ2)};
    std::vector<TransformRT> parentFromBoneTransforms = {
        TransformRT::Identity(),
        TransformRT(Quaternion::FromAxisAngle(kvz, k20), tOF1_0),
        TransformRT(Quaternion::FromAxisAngle(kvz, -k20), tOF2_1)};
    std::vector<TransformRT> worldFromBoneTransforms = {
        TransformRT::Identity(),
        TransformRT(qOF1_0, tOF1_0) * TransformRT(qJ1, tJ1),
        TransformRT(qOF1_0, tOF1_0) * TransformRT(qJ1, tJ1) * TransformRT(qOF2_1, tOF2_1) *
            TransformRT(qJ2, tJ2)};
    ColumnVector<real, 3 * RigidSize::kAll> fullDofs;
    TransformToRawPose(TransformRT::Identity(), fullDofs.TopRows<RigidSize::kAll>(RigidSize::kAll));
    TransformToRawPose(
        worldFromBoneTransforms[1],
        fullDofs.Slice<RigidSize::kAll>(RigidSize::kAll, RigidSize::kAll));
    TransformToRawPose(
        worldFromBoneTransforms[2],
        fullDofs.Slice<RigidSize::kAll>(2 * RigidSize::kAll, RigidSize::kAll));
    ColumnVector<real, RigidSize::kAll + RigidSize::kRot + 1> reducedPose;
    TransformToRawPose(
        TransformRT::Identity(), reducedPose.TopRows<RigidSize::kAll>(RigidSize::kAll));
    reducedPose.Slice<RigidSize::kRot>(RigidSize::kAll, RigidSize::kRot) =
        AsColumnVectorView(qJ1.data);
    reducedPose[reducedPose.Rows() - 1] = -k20;
    testConfigs.push_back(
        {parents,
         layout,
         restTransforms,
         jointTransforms,
         parentFromBoneTransforms,
         worldFromBoneTransforms,
         fullDofs,
         reducedPose});
  }
  {
    // Create test config with 3 links, last two arranged in parallel:
    //        o
    //        |
    //   o----o----o
    ParentIndexArray parents = {-1, 0, 0};
    JointLayout layout = MakeLayout(
        {ArticulatedJointType::Free,
         ArticulatedJointType::Spherical,
         ArticulatedJointType::Spherical},
        {kv0, kv0, kv0});
    Real3 tOF1_0 = {1_r, 0_r, 0_r};
    Quaternion qOF1_0 = Quaternion::FromAxisAngle(kvz, k90);
    Real3 tOF2_0 = {2_r, 0_r, 0_r};
    Quaternion qOF2_0 = Quaternion::Identity();
    RestTransformArray restTransforms = {
        {TransformRT::Identity(), TransformRT::Identity()},
        {TransformRT::Identity(), TransformRT(qOF1_0, tOF1_0)},
        {TransformRT::Identity(), TransformRT(qOF2_0, tOF2_0)}};
    Real3 tJ0 = {};
    Quaternion qJ0 = Quaternion::Identity();
    Real3 tJ1 = {};
    Quaternion qJ1 = Quaternion::Identity();
    Real3 tJ2 = {};
    Quaternion qJ2 = Quaternion::FromAxisAngle(kvz, k20);
    std::vector<TransformRT> jointTransforms = {
        TransformRT(qJ0, tJ0), TransformRT(qJ1, tJ1), TransformRT(qJ2, tJ2)};
    std::vector<TransformRT> parentFromBoneTransforms = {
        TransformRT::Identity(),
        TransformRT(Quaternion::FromAxisAngle(kvz, k90), tOF1_0),
        TransformRT(Quaternion::FromAxisAngle(kvz, k20), tOF2_0)};
    std::vector<TransformRT> worldFromBoneTransforms = {
        TransformRT::Identity(),
        TransformRT(Quaternion::FromAxisAngle(kvz, k90), tOF1_0),
        TransformRT(Quaternion::Identity(), tOF2_0) * TransformRT(qJ2, tJ2)};
    ColumnVector<real, 3 * RigidSize::kAll> fullDofs;
    TransformToRawPose(TransformRT::Identity(), fullDofs.TopRows<RigidSize::kAll>(RigidSize::kAll));
    TransformToRawPose(
        worldFromBoneTransforms[1],
        fullDofs.Slice<RigidSize::kAll>(RigidSize::kAll, RigidSize::kAll));
    TransformToRawPose(
        worldFromBoneTransforms[2],
        fullDofs.Slice<RigidSize::kAll>(2 * RigidSize::kAll, RigidSize::kAll));
    ColumnVector<real, RigidSize::kAll + 2 * RigidSize::kRot> reducedPose;
    TransformToRawPose(
        TransformRT::Identity(), reducedPose.TopRows<RigidSize::kAll>(RigidSize::kAll));
    reducedPose.Slice<RigidSize::kRot>(RigidSize::kAll, RigidSize::kRot) =
        AsColumnVectorView(Quaternion::Identity().data);
    reducedPose.Slice<RigidSize::kRot>(RigidSize::kAll + RigidSize::kRot, RigidSize::kRot) =
        AsColumnVectorView(qJ2.data);
    testConfigs.push_back(
        {parents,
         layout,
         restTransforms,
         jointTransforms,
         parentFromBoneTransforms,
         worldFromBoneTransforms,
         fullDofs,
         reducedPose});
  }
  {
    // Create test config with 4 links
    //        o
    //        |
    //   o----o----o----o
    ParentIndexArray parents = {-1, 0, 0, 2};
    JointLayout layout = MakeLayout(
        {ArticulatedJointType::Free,
         ArticulatedJointType::Spherical,
         ArticulatedJointType::Spherical,
         ArticulatedJointType::Revolute},
        {kv0, kv0, kv0, kvz});
    Real3 tOF1_0 = {1_r, 0_r, 0_r};
    Quaternion qOF1_0 = Quaternion::FromAxisAngle(kvz, k90);
    Real3 tOF2_0 = {1_r, 0_r, 0_r};
    Quaternion qOF2_0 = Quaternion::Identity();
    Real3 tOF3_2 = {1_r, 0_r, 0_r};
    Quaternion qOF3_2 = Quaternion::Identity();
    RestTransformArray restTransforms = {
        {TransformRT::Identity(), TransformRT::Identity()},
        {TransformRT::Identity(), TransformRT(qOF1_0, tOF1_0)},
        {TransformRT::Identity(), TransformRT(qOF2_0, tOF2_0)},
        {TransformRT::Identity(), TransformRT(qOF3_2, tOF3_2)}};
    Real3 tJ0 = {};
    Quaternion qJ0 = Quaternion::Identity();
    Real3 tJ1 = {};
    Quaternion qJ1 = Quaternion::Identity();
    Real3 tJ2 = {};
    Quaternion qJ2 = Quaternion::Identity();
    Real3 tJ3 = {};
    Quaternion qJ3 = Quaternion::FromAxisAngle(kvz, k20);
    std::vector<TransformRT> jointTransforms = {
        TransformRT(qJ0, tJ0), TransformRT(qJ1, tJ1), TransformRT(qJ2, tJ2), TransformRT(qJ3, tJ3)};
    std::vector<TransformRT> parentFromBoneTransforms = {
        TransformRT::Identity(),
        TransformRT(Quaternion::FromAxisAngle(kvz, k90), tOF1_0),
        TransformRT(Quaternion::Identity(), tOF2_0),
        TransformRT(Quaternion::FromAxisAngle(kvz, k20), tOF3_2),
    };
    std::vector<TransformRT> worldFromBoneTransforms = {
        TransformRT::Identity(),
        TransformRT(Quaternion::FromAxisAngle(kvz, k90), tOF1_0),
        TransformRT(Quaternion::Identity(), tOF2_0) * TransformRT(qJ2, tJ2),
        TransformRT(Quaternion::Identity(), tOF2_0) * TransformRT(qJ2, tJ2) *
            TransformRT(qOF3_2, tOF3_2) * TransformRT(qJ3, tJ3)};
    ColumnVector<real, 4 * RigidSize::kAll> fullDofs;
    for (int i = 0; i < 4; ++i) {
      TransformToRawPose(
          worldFromBoneTransforms[i],
          fullDofs.Slice<RigidSize::kAll>(i * RigidSize::kAll, RigidSize::kAll));
    }
    ColumnVector<real, RigidSize::kAll + 2 * RigidSize::kRot + 1> reducedPose;
    TransformToRawPose(
        TransformRT::Identity(), reducedPose.TopRows<RigidSize::kAll>(RigidSize::kAll));
    reducedPose.Slice<RigidSize::kRot>(RigidSize::kAll, RigidSize::kRot) =
        AsColumnVectorView(Quaternion::Identity().data);
    reducedPose.Slice<RigidSize::kRot>(RigidSize::kAll + RigidSize::kRot, RigidSize::kRot) =
        AsColumnVectorView(Quaternion::Identity().data);
    reducedPose[reducedPose.Rows() - 1] = k20;
    testConfigs.push_back(
        {parents,
         layout,
         restTransforms,
         jointTransforms,
         parentFromBoneTransforms,
         worldFromBoneTransforms,
         fullDofs,
         reducedPose});
  }
  return testConfigs;
}
}; // namespace

TEST(ArticulatedBody, ComputeParentFromBoneFromJointTransforms) {
  // Create collection of test configs
  std::vector<ArticulatedBodyTestConfig> testConfigs = CreateArticulatedBodyTestConfigs();

  // For each test config
  for (auto tc : testConfigs) {
    // Compute parent from bone transforms
    std::vector<TransformRT> parentFromBoneTransforms(tc.parents.size());
    ComputeParentFromBone(
        MakeConstSpan(tc.restTransforms), tc.jointTransforms, parentFromBoneTransforms);

    // Compare expected and obtained parent from bone transforms
    EXPECT_EQ(tc.parentFromBoneTransforms.size(), parentFromBoneTransforms.size());
    for (auto i = 0; i < parentFromBoneTransforms.size(); ++i) {
      TransformRT expectedTx = tc.parentFromBoneTransforms[i];
      EXPECT_NEAR_EQ(expectedTx, parentFromBoneTransforms[i]);
    }
  }
}

/*************************************************************************************************/
TEST(ArticulatedBody, ComputeJointTransformsFromParentFromBone) {
  // Create collection of test configs
  std::vector<ArticulatedBodyTestConfig> testConfigs = CreateArticulatedBodyTestConfigs();

  // For each test config
  for (auto tc : testConfigs) {
    // Compute parent from bone transforms
    std::vector<TransformRT> jointTransforms(tc.parents.size());
    ComputeActiveJointTransforms(tc.restTransforms, tc.parentFromBoneTransforms, jointTransforms);

    // Compare expected and obtained parent from bone transforms
    EXPECT_EQ(tc.jointTransforms.size(), jointTransforms.size());
    for (auto i = 0; i < jointTransforms.size(); ++i) {
      TransformRT expectedTx = tc.jointTransforms[i];
      EXPECT_NEAR_EQ(expectedTx, jointTransforms[i]);
    }
  }
}

/*************************************************************************************************/
TEST(ArticulatedBody, ComputeWorldFromBoneFromParentFromBone) {
  // Create test configs
  std::vector<ArticulatedBodyTestConfig> testConfigs = CreateArticulatedBodyTestConfigs();

  // For each test config
  for (auto tc : testConfigs) {
    // Compute parent from bones
    std::vector<TransformRT> parentFromBoneTransforms(tc.parents.size());
    ComputeParentFromBone(
        MakeConstSpan(tc.restTransforms), tc.jointTransforms, parentFromBoneTransforms);

    // Compute world from bone transforms
    std::vector<TransformRT> worldFromBoneTransforms(tc.parents.size());
    ComputeWorldFromBone(
        tc.parents, parentFromBoneTransforms, TransformRT{}, worldFromBoneTransforms);
    // Compare expected and obtained world from bone transforms
    EXPECT_EQ(tc.worldFromBoneTransforms.size(), worldFromBoneTransforms.size());
    for (auto i = 0; i < worldFromBoneTransforms.size(); ++i) {
      TransformRT expectedTx = tc.worldFromBoneTransforms[i];
      EXPECT_NEAR_EQ(expectedTx, worldFromBoneTransforms[i]);
    }
  }
}

/*************************************************************************************************/
TEST(ArticulatedBody, ComputeParentFromBoneWorldFromBone) {
  // Create test configs
  std::vector<ArticulatedBodyTestConfig> testConfigs = CreateArticulatedBodyTestConfigs();

  // For each test config
  for (auto tc : testConfigs) {
    // Compute parent from bone transforms from expected world from bone transforms
    std::vector<TransformRT> parentFromBoneTransforms(tc.parents.size());
    ComputeParentFromBone(
        MakeConstSpan(tc.parents),
        tc.worldFromBoneTransforms,
        TransformRT{},
        parentFromBoneTransforms);

    // Compare to expected parent from bone transforms
    EXPECT_EQ(tc.parentFromBoneTransforms.size(), parentFromBoneTransforms.size());
    for (auto i = 0; i < parentFromBoneTransforms.size(); ++i) {
      TransformRT expectedTx = tc.parentFromBoneTransforms[i];
      EXPECT_NEAR_EQ(expectedTx, parentFromBoneTransforms[i]);
    }
  }
}

/*************************************************************************************************/
TEST(ArticulatedBody, ComputeFullDofsFromTransform) {
  // Assemble collection of world from bone transforms
  std::vector<Real3> kvs = {kvx, Normalize(kvx + kvy), Normalize(kvx + kvy + kvz)};
  std::vector<real> kAngles = {k0, k20, k120, -k120};

  size_t numKvs = int(kvs.size());
  size_t numAngles = int(kAngles.size());
  std::vector<TransformRT> transforms;
  transforms.reserve(numKvs * numKvs * numAngles);
  std::vector<ColumnVector<real, RigidSize::kAll>> expectedFullDofs;
  expectedFullDofs.reserve(numKvs * numKvs * numAngles);
  for (auto kv : kvs) {
    for (auto qv : kvs) {
      for (auto angle : kAngles) {
        TransformRT tx(Quaternion::FromAxisAngle(qv, angle), kv);
        transforms.emplace_back(tx);
        ColumnVector<real, RigidSize::kAll> fullDofs;
        TransformToRawPose(tx, fullDofs);
        expectedFullDofs.emplace_back(fullDofs);
      }
    }
  }

  // Traverse world from bone transforms and expected full dofs
  for (auto i = 0; i < transforms.size(); ++i) {
    // Compute full dofs
    TransformRT const& tx = transforms[i];
    ColumnVector<real, RigidSize::kAll> fullDofs;
    ComputeFullPose(MakeSingletonConstSpan(tx), fullDofs);

    // Compare to expected full dofs
    for (auto j = 0; j < fullDofs.Rows(); ++j) {
      EXPECT_NEAR_EQ(fullDofs[j], expectedFullDofs[i][j]);
    }
  }
}

/*************************************************************************************************/
TEST(ArticulatedBody, ComputeWorldFromBoneTransformFromFullDofs) {
  // Create collection of full dofs
  std::vector<Real3> kvs = {kvx, Normalize(kvx + kvy), Normalize(kvx + kvy + kvz)};
  std::vector<real> kAngles = {k0, k20, k120, -k120};

  size_t numKvs = int(kvs.size());
  size_t numAngles = int(kAngles.size());
  std::vector<TransformRT> expectedTxs;
  expectedTxs.reserve(numKvs * numKvs * numAngles);
  std::vector<ColumnVector<real, RigidSize::kAll>> fullDofs;
  fullDofs.reserve(numKvs * numKvs * numAngles);
  for (auto kv : kvs) {
    for (auto qv : kvs) {
      for (auto angle : kAngles) {
        TransformRT tx(Quaternion::FromAxisAngle(qv, angle), kv);
        ColumnVector<real, RigidSize::kAll> dofs;
        TransformToRawPose(tx, dofs);
        fullDofs.emplace_back(dofs);
        expectedTxs.emplace_back(tx);
      }
    }
  }

  // For each full dofs
  for (auto i = 0; i < fullDofs.size(); ++i) {
    // Compute full dofs
    std::vector<TransformRT> worldFromBoneTransform(1);
    ComputeWorldFromBone(fullDofs[i], worldFromBoneTransform);
    EXPECT_NEAR_EQ(worldFromBoneTransform[0], expectedTxs[i]);
  }
}

/*************************************************************************************************/
namespace {
void AssembleFullDofs(Span<TransformRT const> transforms, ColumnVectorView<real> fullDofs) {
  int offset = 0;
  for (auto tx : transforms) {
    TransformToRawPose(tx, fullDofs.Slice<RigidSize::kAll>(offset, RigidSize::kAll));
    offset += RigidSize::kAll;
  }
}

bool cmp_TransformRT(TransformRT const& a, TransformRT const& b) {
  Real3 ta = a.GetTranslation();
  Real3 tb = b.GetTranslation();
  for (int i = 0; i < 3; ++i) {
    if (ta[i] < tb[i]) {
      return true;
    } else if (tb[i] < ta[i]) {
      return false;
    }
  }
  Real3 rva = a.GetRotation().ToRotationVector();
  Real3 rvb = b.GetRotation().ToRotationVector();
  for (int i = 0; i < 3; ++i) {
    if (rva[i] < rvb[i]) {
      return true;
    } else if (rvb[i] < rva[i]) {
      return false;
    }
  }
  return false;
}
}; // namespace

TEST(ArticulatedBody, ComputeFullDofsFromWorldFromBone) {
  // Create transforms
  Quaternion kqI = Quaternion::Identity();
  std::vector<TransformRT> transforms = {
      TransformRT(kqI, kv0),
      TransformRT(kqI, kvy),
      TransformRT(Quaternion::FromAxisAngle(kvx, k20), Real3{}),
      TransformRT(Quaternion::FromAxisAngle(kvx, k20), kvy)};
  std::sort(transforms.begin(), transforms.end(), cmp_TransformRT);

  // Compute number of full dofs
  int numFullDofs = int(transforms.size()) * RigidSize::kAll;

  // For each possible permutation of test configs
  ColumnVector<real> expectedFullDofs(numFullDofs);
  ColumnVector<real> fullDofs(numFullDofs);
  do {
    // Compute expected full dofs
    AssembleFullDofs(transforms, expectedFullDofs);

    // Compute full dofs
    ComputeFullPose(transforms, fullDofs);

    // Compare to obtained and expected full dofs
    for (auto j = 0; j < fullDofs.Rows(); ++j) {
      // TODO: Implement EXPECT_NEAR_EQ for krylov vectors
      EXPECT_NEAR_EQ(fullDofs[j], expectedFullDofs[j]);
    }
  } while (std::next_permutation(transforms.begin(), transforms.end(), cmp_TransformRT));
}

/*************************************************************************************************/
TEST(ArticulatedBody, ComputeWorldFromBoneFromFullDofs) {
  // Create transforms
  Quaternion kqI = Quaternion::Identity();
  std::vector<TransformRT> transforms = {
      TransformRT(kqI, kv0),
      TransformRT(kqI, kvy),
      TransformRT(Quaternion::FromAxisAngle(kvx, k20), Real3{}),
      TransformRT(Quaternion::FromAxisAngle(kvx, k20), kvy)};
  std::sort(transforms.begin(), transforms.end(), cmp_TransformRT);

  // Compute number of full dofs
  int numFullDofs = int(transforms.size()) * RigidSize::kAll;

  // For each possible permutation of test configs
  ColumnVector<real> fullDofs(numFullDofs);

  // Traverse permutations and compare expected and obtained
  do {
    // Compute full dofs
    AssembleFullDofs(transforms, fullDofs);
    std::vector<TransformRT> worldFromBoneTransforms(transforms.size());
    ComputeWorldFromBone(fullDofs, worldFromBoneTransforms);

    for (auto i = 0; i < transforms.size(); ++i) {
      EXPECT_NEAR_EQ(transforms[i], worldFromBoneTransforms[i]);
    }
  } while (std::next_permutation(transforms.begin(), transforms.end(), cmp_TransformRT));
}

/*************************************************************************************************/
TEST(ArticulatedBody, ComputeFullDofsFromReducedDofs) {
  // Create test configs
  std::vector<ArticulatedBodyTestConfig> testConfigs = CreateArticulatedBodyTestConfigs();

  // For each test config
  for (auto tc : testConfigs) {
    std::vector<TransformRT> jointTransforms(tc.parents.size());
    std::vector<TransformRT> linkTransforms(tc.parents.size());
    ColumnVector<real> fullDofs(tc.fullDofs.Rows());
    ComputeFullPose(
        tc.layout.types,
        tc.layout.axes,
        tc.layout.pose,
        tc.parents,
        tc.restTransforms,
        TransformRT{},
        tc.reducedPose,
        jointTransforms,
        linkTransforms,
        fullDofs);
    for (auto i = 0; i < fullDofs.Rows(); ++i) {
      EXPECT_NEAR_RTOL(fullDofs[i], tc.fullDofs[i], 1.0e-5);
    }
  }
}

/*************************************************************************************************/
TEST(ArticulatedBody, ComputeReducedDofsFromFullDofs) {
  real rtol = 1.0e-3_r;

  // Create test configs
  std::vector<ArticulatedBodyTestConfig> testConfigs = CreateArticulatedBodyTestConfigs();

  // For each test config
  for (auto tc : testConfigs) {
    ColumnVector<real> reducedPose(tc.reducedPose.Rows());
    std::vector<TransformRT> jointTransforms(tc.parents.size());
    std::vector<TransformRT> linkTransforms(tc.parents.size());
    ComputeReducedPose(
        tc.layout.types,
        tc.layout.axes,
        tc.layout.pose,
        tc.parents,
        tc.restTransforms,
        TransformRT{},
        tc.fullDofs,
        jointTransforms,
        linkTransforms,
        reducedPose);
    for (auto i = 0; i < reducedPose.Rows(); ++i) {
      EXPECT_NEAR_RTOL(reducedPose[i], tc.reducedPose[i], rtol);
    }
  }
}

/*************************************************************************************************/
TEST(ArticulatedBody, ComputeReducedPoseFromTransforms) {
  real rtol = 1.0e-3_r;

  // Create test configs
  std::vector<ArticulatedBodyTestConfig> testConfigs = CreateArticulatedBodyTestConfigs();

  // For each test config
  for (auto tc : testConfigs) {
    ColumnVector<real> reducedPose(tc.reducedPose.Rows());
    std::vector<TransformRT> jointTransforms(tc.parents.size());
    ComputeReducedPoseFromTransforms(
        tc.layout.types,
        tc.layout.axes,
        tc.layout.pose,
        tc.parents,
        tc.restTransforms,
        TransformRT{},
        tc.worldFromBoneTransforms,
        jointTransforms,
        reducedPose);
    for (auto i = 0; i < reducedPose.Rows(); ++i) {
      EXPECT_NEAR_RTOL(reducedPose[i], tc.reducedPose[i], rtol);
    }
  }
}

static void JacobianFD(
    ColumnVectorView<real const> reducedDofsView,
    ArticulatedProperties const& props,
    Span<int const> parents,
    Span<ArticulatedRestTransform const> restTransforms,
    TransformRT const& worldFromRoot,
    JointLayout const& layout,
    RowMatrixView<real> outJacobian,
    real eps = 1.0e-3_r) {
  std::vector<TransformRT> jointTransforms(props.numLinks);
  std::vector<TransformRT> linkTransforms(props.numLinks);
  ColumnVector<real> fullDofsp(props.fullPoseDim);
  ColumnVector<real> fullDofsn(props.fullPoseDim);
  ColumnVector<real> reducedPose = reducedDofsView.Duplicate();
  ColumnVector<real> deltaReduced(props.reducedDofsDim);
  ColumnVector<real> deltaFull(props.fullDofsDim);

  for (uint32_t i = 0; i < props.reducedDofsDim; ++i) {
    deltaReduced.SetZero();
    deltaReduced[i] = eps;
    AddLieDeltaToReducedPose(
        layout.types, layout.dofs, layout.pose, reducedDofsView, deltaReduced, reducedPose);
    ComputeFullPose(
        layout.types,
        layout.axes,
        layout.pose,
        parents,
        restTransforms,
        worldFromRoot,
        reducedPose,
        jointTransforms,
        linkTransforms,
        fullDofsp);
    deltaReduced[i] = -eps;
    AddLieDeltaToReducedPose(
        layout.types, layout.dofs, layout.pose, reducedDofsView, deltaReduced, reducedPose);
    ComputeFullPose(
        layout.types,
        layout.axes,
        layout.pose,
        parents,
        restTransforms,
        worldFromRoot,
        reducedPose,
        jointTransforms,
        linkTransforms,
        fullDofsn);
    deltaReduced[i] = 0_r;
    ComputeLieDeltaFullPose(fullDofsn, fullDofsp, deltaFull);
    outJacobian.Block(0, i, props.fullDofsDim, 1) = deltaFull;
  }
  outJacobian /= (2_r * eps);
}

/*************************************************************************************************/
TEST(ArticulatedBody, Jacobian_OneLink) {
  // Create test config with 1 link
  // o----o
  ArticulatedProperties props = {
      .numLinks = 1,
      .fullDofsDim = RigidSize::kDAll,
      .fullPoseDim = RigidSize::kAll,
      .reducedDofsDim = RigidSize::kDAll,
      .reducedPoseDim = RigidSize::kAll};
  TransformRT worldFromRoot(
      Quaternion::FromRotationVector(Real3{0.5_r, -0.8_r, 0.7_r}), Real3{0.6_r, 0.1_r, -0.8_r});
  ParentIndexArray parents = {-1};
  JointLayout layout = MakeLayout({ArticulatedJointType::Free}, {kv0});
  Real3 tIF0 = {0.1_r, 0.2_r, 0.3_r};
  Quaternion qIF0 = Quaternion::FromAxisAngle(Real3{1_r, 0_r, 0_r}, 0.1_r);
  Real3 tOF0 = {0.4_r, 0.5_r, 0.6_r};
  Quaternion qOF0 = Quaternion::FromAxisAngle(Normalize(Real3{0.1_r, 0.3_r, 0.5_r}), 0.2_r);
  RestTransformArray restTransforms = {
      {TransformRT(qIF0, tIF0), TransformRT::Identity()},
      {TransformRT(qOF0, tOF0), TransformRT::Identity()}};
  Real3 tJ0 = {0.1_r, 0.3_r, 0.5_r};
  Quaternion qJ0 = Quaternion::FromAxisAngle(Normalize(Real3{0.1_r, -0.1_r, 0.3_r}), 0.2_r);
  std::vector<TransformRT> jointTransforms = {{TransformRT(qJ0, tJ0)}};
  ColumnVector<real, RigidSize::kAll> reducedPose;
  TransformToRawPose(jointTransforms[0], reducedPose);
  std::vector<TransformRT> linkTransforms(props.numLinks);
  ComputeParentFromBone(MakeConstSpan(restTransforms), jointTransforms, linkTransforms);
  ComputeWorldFromBone(parents, linkTransforms, worldFromRoot, linkTransforms);

  // Compute Jacobian by finite differences
  auto jac_fd = RowMatrix<real>::Zero(props.fullDofsDim, props.reducedDofsDim);
  JacobianFD(reducedPose, props, parents, restTransforms, worldFromRoot, layout, jac_fd);

  auto jac = RowMatrix<real>::Zero(props.fullDofsDim, props.reducedDofsDim);
  Jacobian(
      layout.types,
      parents,
      layout.axes,
      layout.dofs,
      restTransforms,
      worldFromRoot,
      jointTransforms,
      linkTransforms,
      jac);

  real rtol = 1.0e-3_r;
  Matrix<real> diff = jac - jac_fd;
  EXPECT_LE(diff.Norm(), rtol * jac_fd.Norm());
}

TEST(ArticulatedBody, Jacobian_TwoLink) {
  // Create test config with 2 links arranged in series
  // o----o----o
  ArticulatedProperties props = {
      .numLinks = 2,
      .fullDofsDim = 2 * RigidSize::kDAll,
      .fullPoseDim = 2 * RigidSize::kAll,
      .reducedDofsDim = RigidSize::kDAll + RigidSize::kDRot,
      .reducedPoseDim = RigidSize::kAll + RigidSize::kRot};
  TransformRT worldFromRoot(
      Quaternion::FromRotationVector(Real3{0.5_r, -0.8_r, 0.7_r}), Real3{0.6_r, 0.1_r, -0.8_r});
  ParentIndexArray parents = {-1, 0};
  JointLayout layout =
      MakeLayout({ArticulatedJointType::Free, ArticulatedJointType::Spherical}, {kv0, kv0});
  Real3 tIF0 = {0.1_r, 0.2_r, 0.3_r};
  Quaternion qIF0 = Quaternion::FromRotationVector(Real3{0.3_r, -0.1_r, 0.7_r});
  Real3 tIF1 = {0.2_r, 0.4_r, 0.6_r};
  Quaternion qIF1 = Quaternion::FromRotationVector(Real3{-0.1_r, 0.2_r, -0.3_r});
  Real3 tOF0 = {0.1_r, 0.3_r, 0.5_r};
  Quaternion qOF0 = Quaternion::FromRotationVector(Real3{0.3_r, 0.1_r, 0.3_r});
  Real3 tOF1 = {0.3_r, 0.6_r, 0.9_r};
  Quaternion qOF1 = Quaternion::FromRotationVector(Real3{0.2_r, 0.8_r, 0.1_r});
  RestTransformArray restTransforms = {
      {TransformRT(qIF0, tIF0), TransformRT(qIF1, tIF1)},
      {TransformRT(qOF0, tOF0), TransformRT(qOF1, tOF1)}};
  Real3 tJ0 = {0.2_r, 0.3_r, 0.5_r};
  Quaternion qJ0 = Quaternion::FromRotationVector(Real3{-0.1_r, -0.2_r, 0.3_r});
  Real3 tJ1 = {0_r, 0_r, 0_r}; // Spherical joint has no translational dofs
  Quaternion qJ1 = Quaternion::FromRotationVector(Real3{-0.2_r, 0.2_r, 0.1_r});
  std::vector<TransformRT> jointTransforms = {TransformRT(qJ0, tJ0), TransformRT(qJ1, tJ1)};
  ColumnVector<real, RigidSize::kAll + RigidSize::kRot> reducedPose;
  TransformToRawPose(TransformRT(qJ0, tJ0), reducedPose.TopRows<RigidSize::kAll>(RigidSize::kAll));
  reducedPose.BottomRows<RigidSize::kRot>(RigidSize::kRot) = AsColumnVectorView(qJ1.data);
  std::vector<TransformRT> linkTransforms(props.numLinks);
  ComputeParentFromBone(MakeConstSpan(restTransforms), jointTransforms, linkTransforms);
  ComputeWorldFromBone(parents, linkTransforms, worldFromRoot, linkTransforms);

  // Compute Jacobian by finite differences
  auto jac_fd = RowMatrix<real>::Zero(props.fullDofsDim, props.reducedDofsDim);
  JacobianFD(reducedPose, props, parents, restTransforms, worldFromRoot, layout, jac_fd);
  NdArray<real, 6, 6> dx0_dq0_fd = ToNdArray<6, 6>(jac_fd.Block(0, 0, 6, 6)); // For debugging
  NdArray<real, 6, 6> dx1_dq0_fd = ToNdArray<6, 6>(jac_fd.Block(6, 0, 6, 6)); // For debugging
  NdArray<real, 6, 3> dx1_dq1_fd = ToNdArray<6, 3>(jac_fd.Block(6, 6, 6, 3)); // For debugging

  auto jac = RowMatrix<real>::Zero(props.fullDofsDim, props.reducedDofsDim);
  Jacobian(
      layout.types,
      parents,
      layout.axes,
      layout.dofs,
      restTransforms,
      worldFromRoot,
      jointTransforms,
      linkTransforms,
      jac);

  NdArray<real, 6, 6> dx0_dq0 = ToNdArray<6, 6>(jac.Block(0, 0, 6, 6)); // For debugging
  NdArray<real, 6, 6> dx1_dq0 = ToNdArray<6, 6>(jac.Block(6, 0, 6, 6)); // For debugging
  NdArray<real, 6, 3> dx0_dq1 = ToNdArray<6, 3>(jac.Block(0, 6, 6, 3)); // For debugging
  NdArray<real, 6, 3> dx1_dq1 = ToNdArray<6, 3>(jac.Block(6, 6, 6, 3)); // For debugging

  real rtol = 1.0e-3_r;
  EXPECT_LE(Norm(dx0_dq0 - dx0_dq0_fd), rtol * Norm(dx0_dq0_fd));
  EXPECT_LE(Norm(dx1_dq0 - dx1_dq0_fd), rtol * Norm(dx1_dq0_fd));
  EXPECT_LE(Norm(dx0_dq1), 1e-6_r);
  EXPECT_LE(Norm(dx1_dq1 - dx1_dq1_fd), rtol * Norm(dx1_dq1_fd));

  Matrix<real> diff = jac - jac_fd;
  EXPECT_LE(diff.Norm(), rtol * jac_fd.Norm());
}

TEST(ArticulatedBody, Jacobian_TwoLink_RevoluteJoint) {
  // Create test config with 2 links arranged in series with a revolute joint
  // o----o----o
  ArticulatedProperties props = {
      .numLinks = 2,
      .fullDofsDim = 2 * RigidSize::kDAll,
      .fullPoseDim = 2 * RigidSize::kAll,
      .reducedDofsDim = RigidSize::kDAll + 1,
      .reducedPoseDim = RigidSize::kAll + 1};
  TransformRT worldFromRoot(
      Quaternion::FromRotationVector(Real3{0.5_r, -0.8_r, 0.7_r}), Real3{0.6_r, 0.1_r, -0.8_r});
  ParentIndexArray parents = {-1, 0};
  JointLayout layout =
      MakeLayout({ArticulatedJointType::Free, ArticulatedJointType::Revolute}, {kv0, kvz});
  Real3 tIF0 = {};
  Quaternion qIF0 = Quaternion::Identity();
  Real3 tIF1 = {};
  Quaternion qIF1 = Quaternion::Identity();
  Real3 tOF0 = {};
  Quaternion qOF0 = Quaternion::Identity();
  Real3 tOF1 = {1_r, 0_r, 0_r};
  Quaternion qOF1 = Quaternion::Identity();
  RestTransformArray restTransforms = {
      {TransformRT(qIF0, tIF0), TransformRT(qIF1, tIF1)},
      {TransformRT(qOF0, tOF0), TransformRT(qOF1, tOF1)}};
  Real3 tJ0 = {1_r, 2_r, 3_r};
  Quaternion qJ0 = Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, -0.1_r);
  Real3 tJ1 = {}; // Revolute joint has no translational dofs
  Quaternion qJ1 = Quaternion::FromAxisAngle(Real3{0_r, 0_r, 1_r}, 0.2_r);
  std::vector<TransformRT> jointTransforms = {TransformRT(qJ0, tJ0), TransformRT(qJ1, tJ1)};
  ColumnVector<real, RigidSize::kAll + 1> reducedPose;
  TransformToRawPose(TransformRT(qJ0, tJ0), reducedPose.TopRows<RigidSize::kAll>(RigidSize::kAll));
  reducedPose[reducedPose.Rows() - 1] = qJ1.ToRotationVector()[2];
  std::vector<TransformRT> linkTransforms(props.numLinks);
  ComputeParentFromBone(MakeConstSpan(restTransforms), jointTransforms, linkTransforms);
  ComputeWorldFromBone(parents, linkTransforms, worldFromRoot, linkTransforms);

  // Compute Jacobian by finite differences
  auto jac_fd = RowMatrix<real>::Zero(props.fullDofsDim, props.reducedDofsDim);
  JacobianFD(reducedPose, props, parents, restTransforms, worldFromRoot, layout, jac_fd);
  NdArray<real, 6, 6> dx0_dq0_fd = ToNdArray<6, 6>(jac_fd.Block(0, 0, 6, 6)); // For debugging
  NdArray<real, 6, 6> dx1_dq0_fd = ToNdArray<6, 6>(jac_fd.Block(6, 0, 6, 6)); // For debugging
  NdArray<real, 6, 1> dx1_dq1_fd = ToNdArray<6, 1>(jac_fd.Block(6, 6, 6, 1)); // For debugging

  auto jac = RowMatrix<real>::Zero(props.fullDofsDim, props.reducedDofsDim);
  Jacobian(
      layout.types,
      parents,
      layout.axes,
      layout.dofs,
      restTransforms,
      worldFromRoot,
      jointTransforms,
      linkTransforms,
      jac);

  NdArray<real, 6, 6> dx0_dq0 = ToNdArray<6, 6>(jac.Block(0, 0, 6, 6)); // For debugging
  NdArray<real, 6, 6> dx1_dq0 = ToNdArray<6, 6>(jac.Block(6, 0, 6, 6)); // For debugging
  NdArray<real, 6, 1> dx0_dq1 = ToNdArray<6, 1>(jac.Block(0, 6, 6, 1)); // For debugging
  NdArray<real, 6, 1> dx1_dq1 = ToNdArray<6, 1>(jac.Block(6, 6, 6, 1)); // For debugging

  real rtol = 1.0e-3_r;
  EXPECT_LE(Norm(dx0_dq0 - dx0_dq0_fd), rtol * Norm(dx0_dq0_fd));
  EXPECT_LE(Norm(dx1_dq0 - dx1_dq0_fd), rtol * Norm(dx1_dq0_fd));
  EXPECT_LE(Norm(dx0_dq1), 1e-6_r);
  EXPECT_LE(Norm(dx1_dq1 - dx1_dq1_fd), rtol * Norm(dx1_dq1_fd));

  Matrix<real> diff = jac - jac_fd;
  EXPECT_LE(diff.Norm(), rtol * jac_fd.Norm());
}

TEST(ArticulatedBody, Jacobian_ThreeLink) {
  // Create test config with 3 links arranged in series
  // o----o----o----o
  ArticulatedProperties props = {
      .numLinks = 3,
      .fullDofsDim = 3 * RigidSize::kDAll,
      .fullPoseDim = 3 * RigidSize::kAll,
      .reducedDofsDim = RigidSize::kDAll + 2 * RigidSize::kDRot,
      .reducedPoseDim = RigidSize::kAll + 2 * RigidSize::kRot};
  TransformRT worldFromRoot(
      Quaternion::FromRotationVector(Real3{0.5_r, -0.8_r, 0.7_r}), Real3{0.6_r, 0.1_r, -0.8_r});
  ParentIndexArray parents = {-1, 0, 1};
  JointLayout layout = MakeLayout(
      {ArticulatedJointType::Free,
       ArticulatedJointType::Spherical,
       ArticulatedJointType::Spherical},
      {kv0, kv0, kv0});
  Real3 tIF0 = {0.1_r, 0.2_r, 0.3_r};
  Quaternion qIF0 = Quaternion::FromAxisAngle(Real3{1_r, 0_r, 0_r}, 0.2_r);
  Real3 tIF1 = {0.2_r, 0.4_r, 0.6_r};
  Quaternion qIF1 = Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, -0.1_r);
  Real3 tIF2 = {0.2_r, 0.4_r, 0.6_r};
  Quaternion qIF2 = Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, -0.1_r);
  Real3 tOF0 = {0.1_r, 0.3_r, 0.5_r};
  Quaternion qOF0 = Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, -0.1_r);
  Real3 tOF1 = {0.3_r, 0.6_r, 0.9_r};
  Quaternion qOF1 = Quaternion::FromAxisAngle(Real3{0_r, 0_r, 1_r}, 0.3_r);
  Real3 tOF2 = {0.3_r, 0.6_r, 0.9_r};
  Quaternion qOF2 = Quaternion::FromAxisAngle(Real3{0_r, 0_r, 1_r}, 0.3_r);
  RestTransformArray restTransforms = {
      {TransformRT(qIF0, tIF0), TransformRT(qOF0, tOF0)},
      {TransformRT(qIF1, tIF1), TransformRT(qOF1, tOF1)},
      {TransformRT(qIF2, tIF2), TransformRT(qOF2, tOF2)}};
  Real3 tJ0 = {0.2_r, 0.3_r, 0.5_r};
  Quaternion qJ0 = Quaternion::FromAxisAngle(Normalize(Real3{1_r, 3_r, -2_r}), 0.1_r);
  Real3 tJ1 = {0_r, 0_r, 0_r}; // Spherical joint has no translational dofs
  Quaternion qJ1 = Quaternion::FromAxisAngle(Normalize(Real3{1_r, 2_r, 3_r}), -0.3_r);
  Real3 tJ2 = {0_r, 0_r, 0_r}; // Spherical joint has no translational dofs
  Quaternion qJ2 = Quaternion::FromAxisAngle(Normalize(Real3{-1_r, 1_r, -3_r}), 0.5_r);
  std::vector<TransformRT> jointTransforms = {
      TransformRT(qJ0, tJ0), TransformRT(qJ1, tJ1), TransformRT(qJ2, tJ2)};
  ColumnVector<real, RigidSize::kAll + 2 * RigidSize::kRot> reducedPose;
  TransformToRawPose(TransformRT(qJ0, tJ0), reducedPose.TopRows<RigidSize::kAll>(RigidSize::kAll));
  reducedPose.Slice<RigidSize::kRot>(RigidSize::kAll, RigidSize::kRot) =
      AsColumnVectorView(qJ1.data);
  reducedPose.BottomRows<RigidSize::kRot>(RigidSize::kRot) = AsColumnVectorView(qJ2.data);
  std::vector<TransformRT> linkTransforms(props.numLinks);
  ComputeParentFromBone(MakeConstSpan(restTransforms), jointTransforms, linkTransforms);
  ComputeWorldFromBone(parents, linkTransforms, worldFromRoot, linkTransforms);

  // Compute Jacobian by finite differences
  auto jac_fd = RowMatrix<real>::Zero(props.fullDofsDim, props.reducedDofsDim);
  JacobianFD(reducedPose, props, parents, restTransforms, worldFromRoot, layout, jac_fd);
  NdArray<real, 6, 6> dx0_dq0_fd = ToNdArray<6, 6>(jac_fd.Block(0, 0, 6, 6)); // For debugging
  NdArray<real, 6, 6> dx1_dq0_fd = ToNdArray<6, 6>(jac_fd.Block(6, 0, 6, 6)); // For debugging
  NdArray<real, 6, 6> dx2_dq0_fd = ToNdArray<6, 6>(jac_fd.Block(12, 0, 6, 6)); // For debugging
  NdArray<real, 6, 3> dx1_dq1_fd = ToNdArray<6, 3>(jac_fd.Block(6, 6, 6, 3)); // For debugging
  NdArray<real, 6, 3> dx2_dq1_fd = ToNdArray<6, 3>(jac_fd.Block(12, 6, 6, 3)); // For debugging
  NdArray<real, 6, 3> dx2_dq2_fd = ToNdArray<6, 3>(jac_fd.Block(12, 9, 6, 3)); // For debugging

  auto jac = RowMatrix<real>::Zero(props.fullDofsDim, props.reducedDofsDim);
  Jacobian(
      layout.types,
      parents,
      layout.axes,
      layout.dofs,
      restTransforms,
      worldFromRoot,
      jointTransforms,
      linkTransforms,
      jac);
  NdArray<real, 6, 6> dx0_dq0 = ToNdArray<6, 6>(jac.Block(0, 0, 6, 6)); // For debugging
  NdArray<real, 6, 6> dx1_dq0 = ToNdArray<6, 6>(jac.Block(6, 0, 6, 6)); // For debugging
  NdArray<real, 6, 6> dx2_dq0 = ToNdArray<6, 6>(jac.Block(12, 0, 6, 6)); // For debugging
  NdArray<real, 6, 3> dx0_dq1 = ToNdArray<6, 3>(jac.Block(0, 6, 6, 3)); // For debugging
  NdArray<real, 6, 3> dx1_dq1 = ToNdArray<6, 3>(jac.Block(6, 6, 6, 3)); // For debugging
  NdArray<real, 6, 3> dx2_dq1 = ToNdArray<6, 3>(jac.Block(12, 6, 6, 3)); // For debugging
  NdArray<real, 6, 3> dx0_dq2 = ToNdArray<6, 3>(jac.Block(0, 9, 6, 3)); // For debugging
  NdArray<real, 6, 3> dx1_dq2 = ToNdArray<6, 3>(jac.Block(6, 9, 6, 3)); // For debugging
  NdArray<real, 6, 3> dx2_dq2 = ToNdArray<6, 3>(jac.Block(12, 9, 6, 3)); // For debugging

  real rtol = 1.0e-3_r;
  EXPECT_LE(Norm(dx0_dq0 - dx0_dq0_fd), rtol * Norm(dx0_dq0_fd));
  EXPECT_LE(Norm(dx1_dq0 - dx1_dq0_fd), rtol * Norm(dx1_dq0_fd));
  EXPECT_LE(Norm(dx2_dq0 - dx2_dq0_fd), rtol * Norm(dx2_dq0_fd));
  EXPECT_LE(Norm(dx0_dq1), 1e-6_r);
  EXPECT_LE(Norm(dx1_dq1 - dx1_dq1_fd), rtol * Norm(dx1_dq1_fd));
  EXPECT_LE(Norm(dx2_dq1 - dx2_dq1_fd), rtol * Norm(dx2_dq1_fd));
  EXPECT_LE(Norm(dx0_dq2), 1e-6_r);
  EXPECT_LE(Norm(dx1_dq2), 1e-6_r);
  EXPECT_LE(Norm(dx2_dq2 - dx2_dq2_fd), rtol * Norm(dx2_dq2_fd));

  Matrix<real> diff = jac - jac_fd;
  EXPECT_LE(diff.Norm(), rtol * jac_fd.Norm());
}

TEST(ArticulatedBody, Jacobian_ThreeLink_Parallel) {
  // Create test config with 3 links, last two arranged in parallel:
  //        o
  //        |
  //   o----o----o
  ArticulatedProperties props = {
      .numLinks = 3,
      .fullDofsDim = 3 * RigidSize::kDAll,
      .fullPoseDim = 3 * RigidSize::kAll,
      .reducedDofsDim = RigidSize::kDAll + 2 * RigidSize::kDRot,
      .reducedPoseDim = RigidSize::kAll + 2 * RigidSize::kRot};
  TransformRT worldFromRoot(
      Quaternion::FromRotationVector(Real3{0.5_r, -0.8_r, 0.7_r}), Real3{0.6_r, 0.1_r, -0.8_r});
  ParentIndexArray parents = {-1, 0, 0};
  std::vector<int> dofOffsets = {0, 6, 9};
  JointLayout layout = MakeLayout(
      {ArticulatedJointType::Free,
       ArticulatedJointType::Spherical,
       ArticulatedJointType::Spherical},
      {kv0, kv0, kv0});
  Real3 tIF0 = {0.1_r, 0.2_r, 0.3_r};
  Quaternion qIF0 = Quaternion::FromAxisAngle(Real3{1_r, 0_r, 0_r}, 0.2_r);
  Real3 tIF1 = {0.2_r, 0.4_r, 0.6_r};
  Quaternion qIF1 = Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, -0.1_r);
  Real3 tIF2 = {0.2_r, 0.4_r, 0.6_r};
  Quaternion qIF2 = Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, -0.1_r);
  Real3 tOF0 = {0.1_r, 0.3_r, 0.5_r};
  Quaternion qOF0 = Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, -0.1_r);
  Real3 tOF1 = {0.3_r, 0.6_r, 0.9_r};
  Quaternion qOF1 = Quaternion::Identity();
  Real3 tOF2 = {0.3_r, 0.6_r, 0.9_r};
  Quaternion qOF2 = Quaternion::Identity();
  RestTransformArray restTransforms = {
      {TransformRT(qIF0, tIF0), TransformRT(qOF0, tOF0)},
      {TransformRT(qIF1, tIF1), TransformRT(qOF1, tOF1)},
      {TransformRT(qIF2, tIF2), TransformRT(qOF2, tOF2)}};
  Real3 tJ0 = {0.2_r, 0.3_r, 0.5_r};
  Quaternion qJ0 = Quaternion::FromAxisAngle(Real3{1_r, 0_r, 0_r}, -0.3_r);
  Real3 tJ1 = {0_r, 0_r, 0_r}; // Spherical joint has no translational dofs
  Quaternion qJ1 = Quaternion::FromAxisAngle(kvz, k20);
  Real3 tJ2 = {0_r, 0_r, 0_r}; // Spherical joint has no translational dofs
  Quaternion qJ2 = Quaternion::FromAxisAngle(kvz, k20);
  std::vector<TransformRT> jointTransforms = {
      TransformRT(qJ0, tJ0), TransformRT(qJ1, tJ1), TransformRT(qJ2, tJ2)};
  ColumnVector<real, RigidSize::kAll + 2 * RigidSize::kRot> reducedPose;
  TransformToRawPose(TransformRT(qJ0, tJ0), reducedPose.TopRows<RigidSize::kAll>(RigidSize::kAll));
  reducedPose.Slice<RigidSize::kRot>(RigidSize::kAll, RigidSize::kRot) =
      AsColumnVectorView(qJ1.data);
  reducedPose.BottomRows<RigidSize::kRot>(RigidSize::kRot) = AsColumnVectorView(qJ2.data);
  std::vector<TransformRT> linkTransforms(props.numLinks);
  ComputeParentFromBone(MakeConstSpan(restTransforms), jointTransforms, linkTransforms);
  ComputeWorldFromBone(parents, linkTransforms, worldFromRoot, linkTransforms);

  // Compute Jacobian by finite differences
  auto jac_fd = RowMatrix<real>::Zero(props.fullDofsDim, props.reducedDofsDim);
  JacobianFD(reducedPose, props, parents, restTransforms, worldFromRoot, layout, jac_fd);
  NdArray<real, 6, 6> dx0_dq0_fd = ToNdArray<6, 6>(jac_fd.Block(0, 0, 6, 6)); // For debugging
  NdArray<real, 6, 6> dx1_dq0_fd = ToNdArray<6, 6>(jac_fd.Block(6, 0, 6, 6)); // For debugging
  NdArray<real, 6, 6> dx2_dq0_fd = ToNdArray<6, 6>(jac_fd.Block(12, 0, 6, 6)); // For debugging
  NdArray<real, 6, 3> dx1_dq1_fd = ToNdArray<6, 3>(jac_fd.Block(6, 6, 6, 3)); // For debugging
  NdArray<real, 6, 3> dx2_dq2_fd = ToNdArray<6, 3>(jac_fd.Block(12, 9, 6, 3)); // For debugging

  auto jac = RowMatrix<real>::Zero(props.fullDofsDim, props.reducedDofsDim);
  Jacobian(
      layout.types,
      parents,
      layout.axes,
      layout.dofs,
      restTransforms,
      worldFromRoot,
      jointTransforms,
      linkTransforms,
      jac);
  NdArray<real, 6, 6> dx0_dq0 = ToNdArray<6, 6>(jac.Block(0, 0, 6, 6)); // For debugging
  NdArray<real, 6, 6> dx1_dq0 = ToNdArray<6, 6>(jac.Block(6, 0, 6, 6)); // For debugging
  NdArray<real, 6, 6> dx2_dq0 = ToNdArray<6, 6>(jac.Block(12, 0, 6, 6)); // For debugging
  NdArray<real, 6, 3> dx0_dq1 = ToNdArray<6, 3>(jac.Block(0, 6, 6, 3)); // For debugging
  NdArray<real, 6, 3> dx1_dq1 = ToNdArray<6, 3>(jac.Block(6, 6, 6, 3)); // For debugging
  NdArray<real, 6, 3> dx2_dq1 = ToNdArray<6, 3>(jac.Block(12, 6, 6, 3)); // For debugging
  NdArray<real, 6, 3> dx0_dq2 = ToNdArray<6, 3>(jac.Block(0, 9, 6, 3)); // For debugging
  NdArray<real, 6, 3> dx1_dq2 = ToNdArray<6, 3>(jac.Block(6, 9, 6, 3)); // For debugging
  NdArray<real, 6, 3> dx2_dq2 = ToNdArray<6, 3>(jac.Block(12, 9, 6, 3)); // For debugging

  real rtol = 1.0e-3_r;
  EXPECT_LE(Norm(dx0_dq0 - dx0_dq0_fd), rtol * Norm(dx0_dq0_fd));
  EXPECT_LE(Norm(dx1_dq0 - dx1_dq0_fd), rtol * Norm(dx1_dq0_fd));
  EXPECT_LE(Norm(dx2_dq0 - dx2_dq0_fd), rtol * Norm(dx2_dq0_fd));
  EXPECT_LE(Norm(dx0_dq1), 1e-6_r);
  EXPECT_LE(Norm(dx1_dq1 - dx1_dq1_fd), rtol * Norm(dx1_dq1_fd));
  EXPECT_LE(Norm(dx2_dq1), 1e-6_r);
  EXPECT_LE(Norm(dx0_dq2), 1e-6_r);
  EXPECT_LE(Norm(dx1_dq2), 1e-6_r);
  EXPECT_LE(Norm(dx2_dq2 - dx2_dq2_fd), rtol * Norm(dx2_dq2_fd));

  Matrix<real> diff = jac - jac_fd;
  EXPECT_LE(diff.Norm(), rtol * jac_fd.Norm());
}

TEST(ArticulatedBody, BoneToReducedDofsMap) {
  // Create test config with 4 bones in a mixed arrangement
  //        o
  //        |
  //   o----o----o
  ParentIndexArray parents = {-1, 0, 1, 1};
  JointLayout layout = MakeLayout(
      {ArticulatedJointType::Free,
       ArticulatedJointType::Spherical,
       ArticulatedJointType::Revolute,
       ArticulatedJointType::Prismatic},
      {kv0, kv0, kvx, kvy});
  ReducedDofsMap bonesToReducedDofsMap = CreateBonesToReducedDofsMap(parents, layout.dofs);
  std::vector<std::vector<int>> expectedBoneDofs = {
      {0, 1, 2, 3, 4, 5},
      {0, 1, 2, 3, 4, 5, 6, 7, 8},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 10}};
  for (auto i = 0; i < parents.size(); ++i) {
    EXPECT_SPAN_EQ(bonesToReducedDofsMap.dofs[i], expectedBoneDofs[i]);
  }
}

TEST(ArticulatedBody, ReducedPoseDistance) {
  JointLayout layout = MakeLayout(
      {ArticulatedJointType::Free,
       ArticulatedJointType::Spherical,
       ArticulatedJointType::Revolute,
       ArticulatedJointType::Prismatic},
      {kv0, kv0, kvx, kvy});

  ColumnVector<real, RigidSize::kAll + RigidSize::kRot + 1 + 1> poseA;
  TransformToRawPose(
      TransformRT(Quaternion::FromRotationVector(Real3{k45, 0_r, 0_r}), Real3{1_r, 1_r, 1_r}),
      poseA.TopRows<RigidSize::kAll>(RigidSize::kAll));
  poseA.Slice<RigidSize::kRot>(RigidSize::kAll, RigidSize::kRot) =
      AsColumnVectorView(Quaternion::FromRotationVector(Real3{0_r, k45, 0_r}).data);
  poseA(RigidSize::kAll + RigidSize::kRot) = k45;
  poseA(RigidSize::kAll + RigidSize::kRot + 1) = 0_r;
  ColumnVector<real, RigidSize::kAll + RigidSize::kRot + 1 + 1> poseB;
  TransformToRawPose(
      TransformRT(Quaternion::FromRotationVector(Real3{k90, 0_r, 0_r}), Real3{2_r, 2_r, 2_r}),
      poseB.TopRows<RigidSize::kAll>(RigidSize::kAll));
  poseB.Slice<RigidSize::kRot>(RigidSize::kAll, RigidSize::kRot) =
      AsColumnVectorView(Quaternion::FromRotationVector(Real3{0_r, k90, 0_r}).data);
  poseB(RigidSize::kAll + RigidSize::kRot) = k90;
  poseB(RigidSize::kAll + RigidSize::kRot + 1) = 1_r;
  std::vector<real> const expectedTransDistances{{Sqrt(3.0_r), 0.0_r, 0.0_r, 1.0_r}};
  std::vector<real> const expectedRotDistances{{k45, k45, k45, 0.0_r}};

  int const N = GetReducedPoseSize(layout.pose);
  int const M = isize(layout.types);
  ASSERT_EQ(N, poseA.Rows());
  ASSERT_EQ(N, poseB.Rows());
  std::vector<real> transDistances(M);
  std::vector<real> rotDistances(M);
  ReducedPoseDistance(layout.types, layout.pose, poseA, poseB, transDistances, rotDistances);

  for (int i = 0; i < M; ++i) {
    EXPECT_NEAR_EQ(transDistances[i], expectedTransDistances[i]);
    EXPECT_NEAR_EQ(rotDistances[i], expectedRotDistances[i]);
  }
}

TEST(ArticulatedBody, CreateRestTransforms) {
  // Two-link chain: fixed root (link 0) + child (link 1). CreateRestTransforms must produce rest
  // transforms that, when consumed by ComputeParentFromBone + ComputeWorldFromBone, reproduce the
  // parameter-space forward kinematics of the chain:
  //
  //   worldFromChildCoM = worldFromRoot * parentLinkFromJoint * jointTx * jointFromChildLink *
  //   {com1}
  //
  // The joint motion `jointTx` acts in the joint frame, i.e. between the parent-side anchor
  // (parentLinkFromJoint) and the child-side anchor (jointFromChildLink). The oracle below is built
  // purely from the input parameters and is independent of CreateRestTransforms' internals. The
  // child CoM in world does not depend on the parent CoM (com0); a correct implementation must
  // cancel it.
  ParentIndexArray const parents = {-1, 0};

  // World transform of link 1's CoM at joint pose `jointTx1`, computed through the actual
  // rest-transform consumption path.
  auto childComWorldFromRest = [&parents](
                                   Real3 const& com0,
                                   Real3 const& com1,
                                   TransformRT const& parentLinkFromJoint1,
                                   TransformRT const& jointFromChildLink1,
                                   TransformRT const& jointTx1) {
    std::vector<Real3> const comLocals = {com0, com1};
    std::vector<int> const jointsChildLinks = {0, 1};
    std::vector<int> const jointsParentLinks = {-1, 0};
    std::vector<TransformRT> const jointFromChildLink = {
        TransformRT::Identity(), jointFromChildLink1};
    std::vector<TransformRT> const parentLinkFromJoint = {
        TransformRT::Identity(), parentLinkFromJoint1};

    RestTransformArray const restTransforms = CreateRestTransforms(
        comLocals, jointsChildLinks, jointsParentLinks, jointFromChildLink, parentLinkFromJoint);

    std::vector<TransformRT> const jointTransforms = {TransformRT::Identity(), jointTx1};
    std::vector<TransformRT> linkTransforms(2);
    ComputeParentFromBone(MakeConstSpan(restTransforms), jointTransforms, linkTransforms);
    ComputeWorldFromBone(parents, linkTransforms, TransformRT::Identity(), linkTransforms);
    return linkTransforms[1];
  };

  // Independent forward-kinematics oracle for link 1's CoM in world (worldFromRoot is identity).
  auto childComWorldOracle = [](Real3 const& com1,
                                TransformRT const& parentLinkFromJoint1,
                                TransformRT const& jointFromChildLink1,
                                TransformRT const& jointTx1) {
    return parentLinkFromJoint1 * jointTx1 * jointFromChildLink1 * TransformRT{com1};
  };

  auto expectMatchesOracle = [&](Real3 const& com0,
                                 Real3 const& com1,
                                 TransformRT const& parentLinkFromJoint1,
                                 TransformRT const& jointFromChildLink1,
                                 TransformRT const& jointTx1) {
    TransformRT const expected =
        childComWorldOracle(com1, parentLinkFromJoint1, jointFromChildLink1, jointTx1);
    TransformRT const actual =
        childComWorldFromRest(com0, com1, parentLinkFromJoint1, jointFromChildLink1, jointTx1);
    EXPECT_NEAR_EQ(actual.GetRotation(), expected.GetRotation());
    EXPECT_NEAR_EQ(actual.GetTranslation(), expected.GetTranslation());
  };

  Quaternion const q = Quaternion::FromAxisAngle(kvz, k120);
  Quaternion const p = Quaternion::FromAxisAngle(kvx, k20);

  // Joint poses to exercise: zero pose plus actuated revolute poses about axes that do not commute
  // with the joint-frame rotation `q`. The actuated poses are what expose any joint-frame anchor
  // error in the rest-transform decomposition.
  std::vector<TransformRT> const jointPoses = {
      TransformRT::Identity(),
      TransformRT{Quaternion::FromAxisAngle(kvx, k20)},
      TransformRT{Quaternion::FromAxisAngle(kvy, k120)}};

  // (1) Joint anchor is a pure rotation (jointFromChildLink translation is zero) and the parent
  // anchor is a pure translation. Exercises the joint-orientation path.
  {
    Real3 const com0 = kv0;
    Real3 const com1 = {0.3_r, 0_r, 0_r};
    Real3 const linkOffset = {0.7_r, 0_r, 0_r};
    TransformRT const parentLinkFromJoint1{linkOffset};
    TransformRT const jointFromChildLink1{q};
    for (TransformRT const& jointTx : jointPoses) {
      expectMatchesOracle(com0, com1, parentLinkFromJoint1, jointFromChildLink1, jointTx);
    }
  }

  // (2) Hardened: jointFromChildLink carries BOTH a rotation and a non-zero translation,
  // parentLinkFromJoint carries both a rotation and a translation, and the parent CoM (com0) is
  // non-zero. This exercises the joint-frame offset and the parent-CoM cancellation under
  // actuation -- regimes left untested by configuration (1).
  {
    Real3 const com0 = {0.15_r, -0.2_r, 0.05_r};
    Real3 const com1 = {0.3_r, 0.1_r, -0.2_r};
    Real3 const linkOffset = {0.7_r, 0.1_r, 0_r};
    Real3 const jointOffset = {0.05_r, 0.2_r, -0.1_r};
    TransformRT const parentLinkFromJoint1{p, linkOffset};
    TransformRT const jointFromChildLink1{q, jointOffset};
    for (TransformRT const& jointTx : jointPoses) {
      expectMatchesOracle(com0, com1, parentLinkFromJoint1, jointFromChildLink1, jointTx);
    }
  }
}
