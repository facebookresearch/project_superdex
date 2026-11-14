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

#include "MochiArticulationTypes.h"
#include "MochiArticulation.h"

#include "Misc/AutomationTest.h"

#include <limits>

#if WITH_DEV_AUTOMATION_TESTS

#if ENGINE_MAJOR_VERSION >= 5
#define MOCHI_ARTICULATION_TYPES_TEST_FLAGS \
  (EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::SmokeFilter)
#else
#define MOCHI_ARTICULATION_TYPES_TEST_FLAGS \
  (EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::SmokeFilter)
#endif

namespace {

constexpr double kTolerance = 1e-5;

// SuperDex Studio space (right-handed, X forward, Y left, Z up, meters) to Unreal and back. The
// handedness flip is what makes rotational joint quantities interesting.
const FMochiToUnrealConverter kStudioToUnreal(kSuperDexSpace);
const FUnrealToMochiConverter kUnrealToStudio(kSuperDexSpace);

bool NearlyEqual(const mochi::Real3& A, const mochi::Real3& B) {
  for (int i = 0; i < 3; ++i) {
    if (!FMath::IsNearlyEqual(static_cast<double>(A[i]), static_cast<double>(B[i]), kTolerance)) {
      return false;
    }
  }
  return true;
}

bool NearlyEqual(const mochi::TransformRT& A, const mochi::TransformRT& B) {
  const mochi::Real3 offset{1_r, 2_r, 3_r};
  return NearlyEqual(A.GetTranslation(), B.GetTranslation()) &&
      NearlyEqual(A.GetRotation() * offset, B.GetRotation() * offset);
}

mochi::ArticulatedJointParams MakeJoint(
    mochi::ArticulatedJointType Type,
    const mochi::Real3& Axis) {
  mochi::ArticulatedJointParams joint;
  joint.name = "joint";
  joint.type = Type;
  joint.parentLinkFromJoint = mochi::TransformRT(
      mochi::Quaternion::FromAxisAngle(mochi::Real3{0.6_r, 0.0_r, 0.8_r}, 0.7_r),
      mochi::Real3{0.1_r, -0.2_r, 0.3_r});
  joint.axis = Axis;
  joint.friction.viscous = 0.5_r;
  joint.friction.coulomb = 0.25_r;
  joint.inertia = 0.125_r;
  joint.minLimit = mochi::Real3{-0.5_r, -0.25_r, -1.0_r};
  joint.maxLimit = mochi::Real3{0.75_r, 0.5_r, 1.5_r};
  joint.limitStiffness = 200;
  joint.limitDamping = 3;
  return joint;
}

} // namespace

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiArticulatedJointRoundTripTest,
    "SuperDexPhysics.Articulation.JointRoundTrip",
    MOCHI_ARTICULATION_TYPES_TEST_FLAGS)

bool FMochiArticulatedJointRoundTripTest::RunTest(const FString& Parameters) {
  for (const mochi::ArticulatedJointType type :
       {mochi::ArticulatedJointType::Revolute,
        mochi::ArticulatedJointType::Prismatic,
        mochi::ArticulatedJointType::Spherical}) {
    const mochi::ArticulatedJointParams expected = MakeJoint(type, mochi::Real3{0_r, 0.6_r, 0.8_r});
    FMochiArticulatedJoint joint;
    joint.FromMochi(expected, kStudioToUnreal);
    const mochi::ArticulatedJointParams actual = joint.ToMochi(kUnrealToStudio);
    const FString typeStr = FString::FromInt(static_cast<int>(type));
    TestTrue(TEXT("Type ") + typeStr, actual.type == expected.type);
    TestTrue(
        TEXT("ParentLinkFromJoint ") + typeStr,
        NearlyEqual(actual.parentLinkFromJoint, expected.parentLinkFromJoint));
    if (type != mochi::ArticulatedJointType::Spherical) {
      TestTrue(TEXT("Axis ") + typeStr, NearlyEqual(actual.axis, expected.axis));
    }
    TestTrue(TEXT("MinLimit ") + typeStr, NearlyEqual(*actual.minLimit, *expected.minLimit));
    TestTrue(TEXT("MaxLimit ") + typeStr, NearlyEqual(*actual.maxLimit, *expected.maxLimit));
    TestEqual(TEXT("Viscous ") + typeStr, actual.friction.viscous, expected.friction.viscous);
    TestEqual(TEXT("Coulomb ") + typeStr, actual.friction.coulomb, expected.friction.coulomb);
    TestEqual(TEXT("Inertia ") + typeStr, *actual.inertia, *expected.inertia);
    TestEqual(TEXT("LimitStiffness ") + typeStr, actual.limitStiffness, expected.limitStiffness);
    TestEqual(TEXT("LimitDamping ") + typeStr, actual.limitDamping, expected.limitDamping);
  }
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiArticulatedSphericalLimitsTest,
    "SuperDexPhysics.Articulation.SphericalLimits",
    MOCHI_ARTICULATION_TYPES_TEST_FLAGS)

bool FMochiArticulatedSphericalLimitsTest::RunTest(const FString& Parameters) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  auto testDofLimits = [&](const FString& What,
                           const FMochiArticulatedJoint& Joint,
                           const TArray<TPair<double, double>>& Expected) {
    for (int32 i = 0; i < 3; ++i) {
      double lower = 0;
      double upper = 0;
      Joint.GetDofLimits(i, lower, upper);
      // Exact comparisons, since unlimited bounds are infinite.
      TestTrue(FString::Printf(TEXT("%s lower %d"), *What, i), lower == Expected[i].Key);
      TestTrue(FString::Printf(TEXT("%s upper %d"), *What, i), upper == Expected[i].Value);
    }
  };

  // Studio to Unreal negates the X and Z rotation vector components, which swaps their bounds.
  mochi::ArticulatedJointParams twoSided =
      MakeJoint(mochi::ArticulatedJointType::Spherical, mochi::Real3{0_r, 0_r, 0_r});
  twoSided.minLimit = mochi::Real3{-0.5_r, -0.25_r, -1.0_r};
  twoSided.maxLimit = mochi::Real3{0.75_r, 0.5_r, 1.5_r};
  FMochiArticulatedJoint joint;
  joint.FromMochi(twoSided, kStudioToUnreal);
  testDofLimits(TEXT("Two-sided"), joint, {{-0.75, 0.5}, {-0.25, 0.5}, {-1.5, 1.0}});

  // A one-sided limit becomes two-sided per axis.
  mochi::ArticulatedJointParams oneSided = twoSided;
  oneSided.minLimit = std::nullopt;
  oneSided.maxLimit = mochi::Real3{0.5_r, 0.5_r, 0.5_r};
  joint.FromMochi(oneSided, kStudioToUnreal);
  testDofLimits(TEXT("One-sided"), joint, {{-0.5, kInf}, {-kInf, 0.5}, {-0.5, kInf}});
  const mochi::ArticulatedJointParams oneSidedRoundTrip = joint.ToMochi(kUnrealToStudio);
  TestFalse(TEXT("One-sided round trip min"), oneSidedRoundTrip.minLimit.has_value());
  TestTrue(
      TEXT("One-sided round trip max"),
      oneSidedRoundTrip.maxLimit && NearlyEqual(*oneSidedRoundTrip.maxLimit, *oneSided.maxLimit));

  // Limits authored in Unreal stay ordered in Studio space.
  FMochiArticulatedJoint authored;
  authored.Type = EMochiArticulatedJointType::Spherical;
  authored.bHasMinLimit = true;
  authored.MinLimit = FVector(-1, -2, -3);
  authored.bHasMaxLimit = true;
  authored.MaxLimit = FVector(1, 2, 3);
  const mochi::ArticulatedJointParams converted = authored.ToMochi(kUnrealToStudio);
  for (int32 i = 0; i < 3; ++i) {
    TestTrue(
        FString::Printf(TEXT("Authored limits ordered %d"), i),
        (*converted.minLimit)[i] <= (*converted.maxLimit)[i]);
  }
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiArticulatedDofConversionTest,
    "SuperDexPhysics.Articulation.DofConversionRoundTrip",
    MOCHI_ARTICULATION_TYPES_TEST_FLAGS)

bool FMochiArticulatedDofConversionTest::RunTest(const FString& Parameters) {
  const TArray<EMochiArticulatedJointType> types = {
      EMochiArticulatedJointType::Free,
      EMochiArticulatedJointType::Spherical,
      EMochiArticulatedJointType::Revolute,
      EMochiArticulatedJointType::Hard,
      EMochiArticulatedJointType::Prismatic};
  const TArray<double> expected = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1};
  TArray<double> dofs = expected;
  ConvertArticulatedDofsToUnreal(types, true, dofs, kStudioToUnreal);
  // The revolute and prismatic values are convention independent.
  TestEqual(TEXT("Revolute DOF"), dofs[9], expected[9]);
  TestEqual(TEXT("Prismatic DOF"), dofs[10], expected[10]);
  ConvertArticulatedDofsToMochi(types, true, dofs, kUnrealToStudio);
  for (int i = 0; i < expected.Num(); ++i) {
    TestTrue(
        FString::Printf(TEXT("DOF %d"), i), FMath::IsNearlyEqual(dofs[i], expected[i], kTolerance));
  }
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiArticulatedDofIndexConversionTest,
    "SuperDexPhysics.Articulation.DofIndexConversion",
    MOCHI_ARTICULATION_TYPES_TEST_FLAGS)

bool FMochiArticulatedDofIndexConversionTest::RunTest(const FString& Parameters) {
  for (const EMochiArticulatedJointType type :
       {EMochiArticulatedJointType::Free,
        EMochiArticulatedJointType::Spherical,
        EMochiArticulatedJointType::Revolute,
        EMochiArticulatedJointType::Prismatic}) {
    const int32 numDofs = GetArticulatedJointNumDofs(type);
    for (int32 dof = 0; dof < numDofs; ++dof) {
      const FString what = FString::Printf(TEXT("Joint type %d DOF %d"), int32(type), dof);
      // A value of the Unreal DOF must convert to the same value of the Mochi DOF as the whole
      // DOF vector does.
      int32 mochiDof = -1;
      double sign = 0.0;
      if (!TestTrue(
              what,
              ConvertArticulatedDofIndexToMochi(type, dof, kUnrealToStudio, mochiDof, sign))) {
        continue;
      }
      TArray<double> dofs;
      dofs.SetNumZeroed(numDofs);
      dofs[dof] = 0.5;
      ConvertArticulatedDofsToMochi(MakeArrayView(&type, 1), true, dofs, kUnrealToStudio);
      TestTrue(what, FMath::IsNearlyEqual(dofs[mochiDof], sign * 0.5, kTolerance));
      int32 roundTrip = -1;
      double roundTripSign = 0.0;
      ConvertArticulatedDofIndexToUnreal(type, mochiDof, kStudioToUnreal, roundTrip, roundTripSign);
      TestEqual(what, roundTrip, dof);
      TestEqual(what, roundTripSign, sign);
    }
    int32 outDof = 0;
    double sign = 0.0;
    TestFalse(
        TEXT("DOF out of range"),
        ConvertArticulatedDofIndexToMochi(type, numDofs, kUnrealToStudio, outDof, sign));
  }
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiRotationBoundsConversionTest,
    "SuperDexPhysics.Articulation.RotationBoundsConversion",
    MOCHI_ARTICULATION_TYPES_TEST_FLAGS)

bool FMochiRotationBoundsConversionTest::RunTest(const FString& Parameters) {
  // Studio to Unreal negates the X and Z rotation vector components, which swaps their bounds.
  const mochi::Real3 min{-1_r, -2_r, -3_r};
  const mochi::Real3 max{1_r, 2_r, 4_r};
  FVector unrealMin;
  FVector unrealMax;
  ConvertRotationBoundsToUnreal(min, max, kStudioToUnreal, unrealMin, unrealMax);
  TestTrue(TEXT("Unreal min"), unrealMin.Equals(FVector(-1, -2, -4), kTolerance));
  TestTrue(TEXT("Unreal max"), unrealMax.Equals(FVector(1, 2, 3), kTolerance));
  mochi::Real3 roundTripMin;
  mochi::Real3 roundTripMax;
  ConvertRotationBoundsToMochi(unrealMin, unrealMax, kUnrealToStudio, roundTripMin, roundTripMax);
  TestTrue(TEXT("Round trip min"), NearlyEqual(roundTripMin, min));
  TestTrue(TEXT("Round trip max"), NearlyEqual(roundTripMax, max));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiArticulationForwardKinematicsTest,
    "SuperDexPhysics.Articulation.ForwardKinematicsMatchesMochi",
    MOCHI_ARTICULATION_TYPES_TEST_FLAGS)

bool FMochiArticulationForwardKinematicsTest::RunTest(const FString& Parameters) {
  // A fixed root link with one revolute child, described in the Studio space.
  const mochi::Real3 axis{0_r, 0.6_r, 0.8_r};
  const mochi::ArticulatedJointParams childJoint =
      MakeJoint(mochi::ArticulatedJointType::Revolute, axis);
  const mochi::TransformRT childFromJointLink(
      mochi::Quaternion::FromAxisAngle(mochi::Real3{1_r, 0_r, 0_r}, 0.3_r),
      mochi::Real3{0.05_r, 0.0_r, -0.1_r});
  mochi::prefab::ArticulatedLinkPrefab root;
  root.name = "root";
  mochi::prefab::ArticulatedLinkPrefab child;
  child.name = "child";
  child.parentLink = 0;
  child.parentJointFromLink = childFromJointLink;

  TArray<FMochiArticulatedJoint> joints;
  joints.AddDefaulted(2);
  joints[0].Type = EMochiArticulatedJointType::Hard;
  joints[1].FromMochi(childJoint, kStudioToUnreal);
  TArray<FMochiArticulatedLink> links;
  links.AddDefaulted(2);
  links[0].FromMochi(root, kStudioToUnreal);
  links[1].FromMochi(child, kStudioToUnreal);

  UMochiArticulation* articulation = NewObject<UMochiArticulation>();
  articulation->SetDefinition(TEXT("test"), joints, links);
  if (!TestEqual(TEXT("NumDofs"), articulation->GetNumDofs(), 1)) {
    return false;
  }

  const double q = 0.9;
  TArray<FTransform> linkToRoot;
  if (!TestTrue(
          TEXT("FK"),
          articulation->ComputeLinkTransformsFromJointPositions(
              {q}, linkToRoot, EMochiArticulationLinkTransformSpace::LinkToRoot))) {
    return false;
  }

  // Mochi: rootFromChild = parentLinkFromJoint * rotation(axis, q) * parentJointFromLink.
  const mochi::TransformRT expected = childJoint.parentLinkFromJoint *
      mochi::TransformRT(mochi::Quaternion::FromAxisAngle(axis, static_cast<mochi::real>(q)),
                         mochi::Real3{0_r, 0_r, 0_r}) *
      childFromJointLink;
  TestTrue(
      TEXT("Child link transform"),
      NearlyEqual(kUnrealToStudio.TransformToMochi_RT(linkToRoot[1]), expected));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiArticulationSphericalForwardKinematicsTest,
    "SuperDexPhysics.Articulation.SphericalForwardKinematicsMatchesMochi",
    MOCHI_ARTICULATION_TYPES_TEST_FLAGS)

bool FMochiArticulationSphericalForwardKinematicsTest::RunTest(const FString& Parameters) {
  // A fixed root link with one spherical child, described in the Studio space.
  mochi::ArticulatedJointParams childJoint =
      MakeJoint(mochi::ArticulatedJointType::Spherical, mochi::Real3{0_r, 0_r, 0_r});
  childJoint.minLimit = std::nullopt;
  childJoint.maxLimit = std::nullopt;
  mochi::prefab::ArticulatedLinkPrefab root;
  root.name = "root";
  mochi::prefab::ArticulatedLinkPrefab child;
  child.name = "child";
  child.parentLink = 0;

  TArray<FMochiArticulatedJoint> joints;
  joints.AddDefaulted(2);
  joints[0].Type = EMochiArticulatedJointType::Hard;
  joints[1].FromMochi(childJoint, kStudioToUnreal);
  TArray<FMochiArticulatedLink> links;
  links.AddDefaulted(2);
  links[0].FromMochi(root, kStudioToUnreal);
  links[1].FromMochi(child, kStudioToUnreal);

  UMochiArticulation* articulation = NewObject<UMochiArticulation>();
  articulation->SetDefinition(TEXT("test"), joints, links);
  if (!TestEqual(TEXT("NumDofs"), articulation->GetNumDofs(), 3)) {
    return false;
  }

  const mochi::Real3 rotationVector{0.3_r, -0.2_r, 0.5_r};
  TArray<double> q = {rotationVector[0], rotationVector[1], rotationVector[2]};
  const TArray<EMochiArticulatedJointType> types = {
      EMochiArticulatedJointType::Hard, EMochiArticulatedJointType::Spherical};
  ConvertArticulatedDofsToUnreal(types, true, q, kStudioToUnreal);
  TArray<FTransform> linkToRoot;
  if (!TestTrue(
          TEXT("FK"),
          articulation->ComputeLinkTransformsFromJointPositions(
              q, linkToRoot, EMochiArticulationLinkTransformSpace::LinkToRoot))) {
    return false;
  }

  // Mochi: rootFromChild = parentLinkFromJoint * rotation(rotationVector).
  const mochi::TransformRT expected = childJoint.parentLinkFromJoint *
      mochi::TransformRT(mochi::Quaternion::FromRotationVector(rotationVector));
  TestTrue(
      TEXT("Child link transform"),
      NearlyEqual(kUnrealToStudio.TransformToMochi_RT(linkToRoot[1]), expected));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiArticulationJointPositionsFromLinkTransformsTest,
    "SuperDexPhysics.Articulation.JointPositionsFromLinkTransformsRoundTrip",
    MOCHI_ARTICULATION_TYPES_TEST_FLAGS)

bool FMochiArticulationJointPositionsFromLinkTransformsTest::RunTest(const FString& Parameters) {
  // A fixed root link with a revolute, a prismatic and a spherical link chained below it.
  const TArray<mochi::ArticulatedJointType> types = {
      mochi::ArticulatedJointType::Revolute,
      mochi::ArticulatedJointType::Prismatic,
      mochi::ArticulatedJointType::Spherical};
  TArray<FMochiArticulatedJoint> joints;
  joints.AddDefaulted(types.Num() + 1);
  joints[0].Type = EMochiArticulatedJointType::Hard;
  TArray<FMochiArticulatedLink> links;
  links.AddDefaulted(types.Num() + 1);
  mochi::prefab::ArticulatedLinkPrefab root;
  root.name = "root";
  links[0].FromMochi(root, kStudioToUnreal);
  const char* const linkNames[] = {"link1", "link2", "link3"};
  for (int i = 0; i < types.Num(); ++i) {
    mochi::ArticulatedJointParams joint = MakeJoint(types[i], mochi::Real3{0_r, 0.6_r, 0.8_r});
    joint.minLimit = std::nullopt;
    joint.maxLimit = std::nullopt;
    joints[i + 1].FromMochi(joint, kStudioToUnreal);
    mochi::prefab::ArticulatedLinkPrefab link;
    link.name = linkNames[i];
    link.parentLink = i;
    link.parentJointFromLink = mochi::TransformRT(
        mochi::Quaternion::FromAxisAngle(mochi::Real3{1_r, 0_r, 0_r}, 0.3_r),
        mochi::Real3{0.05_r, 0.0_r, -0.1_r});
    links[i + 1].FromMochi(link, kStudioToUnreal);
  }
  UMochiArticulation* articulation = NewObject<UMochiArticulation>();
  articulation->SetDefinition(TEXT("test"), joints, links);
  if (!TestEqual(TEXT("NumDofs"), articulation->GetNumDofs(), 5)) {
    return false;
  }

  const TArray<double> q = {0.9, 0.04, 0.3, -0.2, 0.5}; // [rad], [m], spherical rotation vector
  for (const EMochiArticulationLinkTransformSpace space :
       {EMochiArticulationLinkTransformSpace::LinkToParent,
        EMochiArticulationLinkTransformSpace::LinkToRoot}) {
    TArray<FTransform> linkTransforms;
    TArray<double> recovered;
    if (!TestTrue(
            TEXT("FK"),
            articulation->ComputeLinkTransformsFromJointPositions(q, linkTransforms, space)) ||
        !TestTrue(
            TEXT("Inverse"),
            articulation->ComputeJointPositionsFromLinkTransforms(
                linkTransforms, recovered, space)) ||
        !TestEqual(TEXT("Joint position count"), recovered.Num(), q.Num())) {
      return false;
    }
    for (int i = 0; i < q.Num(); ++i) {
      TestNearlyEqual(FString::Printf(TEXT("DOF %d"), i), recovered[i], q[i], kTolerance);
    }
  }
  return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
