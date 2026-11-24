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

#include "Engine/StaticMesh.h"
#include "Misc/AutomationTest.h"

#include <limits>

#if WITH_DEV_AUTOMATION_TESTS

#if ENGINE_MAJOR_VERSION >= 5
#define MOCHI_ARTICULATION_TYPES_TEST_FLAGS \
  (EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::SmokeFilter)
#define MOCHI_ARTICULATION_FIXTURE_TEST_FLAGS \
  (EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::EngineFilter)
#else
#define MOCHI_ARTICULATION_TYPES_TEST_FLAGS \
  (EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::SmokeFilter)
#define MOCHI_ARTICULATION_FIXTURE_TEST_FLAGS \
  (EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::EngineFilter)
#endif

namespace {

constexpr double ArticulationTestTolerance = 1e-5;

// SuperDex Studio space (right-handed, X forward, Y left, Z up, meters) to Unreal and back. The
// handedness flip is what makes rotational joint quantities interesting.
const FMochiToUnrealConverter StudioToUnreal(SuperDexSpace);
const FUnrealToMochiConverter UnrealToStudio(SuperDexSpace);

bool NearlyEqual(const mochi::Real3& A, const mochi::Real3& B) {
  for (int i = 0; i < 3; ++i) {
    if (!FMath::IsNearlyEqual(
            static_cast<double>(A[i]), static_cast<double>(B[i]), ArticulationTestTolerance)) {
      return false;
    }
  }
  return true;
}

bool NearlyEqual(const mochi::TransformRT& A, const mochi::TransformRT& B) {
  const mochi::Real3 Offset{1_r, 2_r, 3_r};
  return NearlyEqual(A.GetTranslation(), B.GetTranslation()) &&
      NearlyEqual(A.GetRotation() * Offset, B.GetRotation() * Offset);
}

mochi::ArticulatedJointParams MakeJoint(
    mochi::ArticulatedJointType Type,
    const mochi::Real3& Axis) {
  mochi::ArticulatedJointParams Joint;
  Joint.name = "joint";
  Joint.type = Type;
  Joint.parentLinkFromJoint = mochi::TransformRT(
      mochi::Quaternion::FromAxisAngle(mochi::Real3{0.6_r, 0.0_r, 0.8_r}, 0.7_r),
      mochi::Real3{0.1_r, -0.2_r, 0.3_r});
  Joint.axis = Axis;
  Joint.friction.viscous = 0.5_r;
  Joint.friction.coulomb = 0.25_r;
  Joint.inertia = 0.125_r;
  Joint.minLimit = mochi::Real3{-0.5_r, -0.25_r, -1.0_r};
  Joint.maxLimit = mochi::Real3{0.75_r, 0.5_r, 1.5_r};
  Joint.limitStiffness = 200;
  Joint.limitDamping = 3;
  return Joint;
}

} // namespace

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiArticulatedJointRoundTripTest,
    "SuperDexPhysics.Articulation.JointRoundTrip",
    MOCHI_ARTICULATION_TYPES_TEST_FLAGS)

bool FMochiArticulatedJointRoundTripTest::RunTest(const FString& Parameters) {
  for (const mochi::ArticulatedJointType Type :
       {mochi::ArticulatedJointType::Revolute,
        mochi::ArticulatedJointType::Prismatic,
        mochi::ArticulatedJointType::Spherical}) {
    const mochi::ArticulatedJointParams Expected = MakeJoint(Type, mochi::Real3{0_r, 0.6_r, 0.8_r});
    FMochiArticulatedJoint Joint;
    Joint.FromMochi(Expected, StudioToUnreal);
    const mochi::ArticulatedJointParams Actual = Joint.ToMochi(UnrealToStudio);
    const FString TypeStr = FString::FromInt(static_cast<int>(Type));
    TestTrue(TEXT("Type ") + TypeStr, Actual.type == Expected.type);
    TestTrue(
        TEXT("ParentLinkFromJoint ") + TypeStr,
        NearlyEqual(Actual.parentLinkFromJoint, Expected.parentLinkFromJoint));
    if (Type != mochi::ArticulatedJointType::Spherical) {
      TestTrue(TEXT("Axis ") + TypeStr, NearlyEqual(Actual.axis, Expected.axis));
    }
    TestTrue(TEXT("MinLimit ") + TypeStr, NearlyEqual(*Actual.minLimit, *Expected.minLimit));
    TestTrue(TEXT("MaxLimit ") + TypeStr, NearlyEqual(*Actual.maxLimit, *Expected.maxLimit));
    TestEqual(TEXT("Viscous ") + TypeStr, Actual.friction.viscous, Expected.friction.viscous);
    TestEqual(TEXT("Coulomb ") + TypeStr, Actual.friction.coulomb, Expected.friction.coulomb);
    TestEqual(TEXT("Inertia ") + TypeStr, *Actual.inertia, *Expected.inertia);
    TestEqual(TEXT("LimitStiffness ") + TypeStr, Actual.limitStiffness, Expected.limitStiffness);
    TestEqual(TEXT("LimitDamping ") + TypeStr, Actual.limitDamping, Expected.limitDamping);
  }
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiArticulatedSphericalLimitsTest,
    "SuperDexPhysics.Articulation.SphericalLimits",
    MOCHI_ARTICULATION_TYPES_TEST_FLAGS)

bool FMochiArticulatedSphericalLimitsTest::RunTest(const FString& Parameters) {
  constexpr double Inf = std::numeric_limits<double>::infinity();
  auto TestDofLimits = [&](const FString& What,
                           const FMochiArticulatedJoint& Joint,
                           const TArray<TPair<double, double>>& Expected) {
    for (int32 i = 0; i < 3; ++i) {
      double Lower = 0;
      double Upper = 0;
      Joint.GetDofLimits(i, Lower, Upper);
      // Exact comparisons, since unlimited bounds are infinite.
      TestTrue(FString::Printf(TEXT("%s lower %d"), *What, i), Lower == Expected[i].Key);
      TestTrue(FString::Printf(TEXT("%s upper %d"), *What, i), Upper == Expected[i].Value);
    }
  };

  // Studio to Unreal negates the X and Z rotation vector components, which swaps their bounds.
  mochi::ArticulatedJointParams TwoSided =
      MakeJoint(mochi::ArticulatedJointType::Spherical, mochi::Real3{0_r, 0_r, 0_r});
  TwoSided.minLimit = mochi::Real3{-0.5_r, -0.25_r, -1.0_r};
  TwoSided.maxLimit = mochi::Real3{0.75_r, 0.5_r, 1.5_r};
  FMochiArticulatedJoint Joint;
  Joint.FromMochi(TwoSided, StudioToUnreal);
  TestDofLimits(TEXT("Two-sided"), Joint, {{-0.75, 0.5}, {-0.25, 0.5}, {-1.5, 1.0}});

  // A one-sided limit becomes two-sided per axis.
  mochi::ArticulatedJointParams OneSided = TwoSided;
  OneSided.minLimit = std::nullopt;
  OneSided.maxLimit = mochi::Real3{0.5_r, 0.5_r, 0.5_r};
  Joint.FromMochi(OneSided, StudioToUnreal);
  TestDofLimits(TEXT("One-sided"), Joint, {{-0.5, Inf}, {-Inf, 0.5}, {-0.5, Inf}});
  const mochi::ArticulatedJointParams OneSidedRoundTrip = Joint.ToMochi(UnrealToStudio);
  TestFalse(TEXT("One-sided round trip min"), OneSidedRoundTrip.minLimit.has_value());
  TestTrue(
      TEXT("One-sided round trip max"),
      OneSidedRoundTrip.maxLimit && NearlyEqual(*OneSidedRoundTrip.maxLimit, *OneSided.maxLimit));

  // Limits authored in Unreal stay ordered in Studio space.
  FMochiArticulatedJoint Authored;
  Authored.Type = EMochiArticulatedJointType::Spherical;
  Authored.bHasMinLimit = true;
  Authored.MinLimit = FVector(-1, -2, -3);
  Authored.bHasMaxLimit = true;
  Authored.MaxLimit = FVector(1, 2, 3);
  const mochi::ArticulatedJointParams Converted = Authored.ToMochi(UnrealToStudio);
  for (int32 i = 0; i < 3; ++i) {
    TestTrue(
        FString::Printf(TEXT("Authored limits ordered %d"), i),
        (*Converted.minLimit)[i] <= (*Converted.maxLimit)[i]);
  }
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiArticulatedDofConversionTest,
    "SuperDexPhysics.Articulation.DofConversionRoundTrip",
    MOCHI_ARTICULATION_TYPES_TEST_FLAGS)

bool FMochiArticulatedDofConversionTest::RunTest(const FString& Parameters) {
  const TArray<EMochiArticulatedJointType> Types = {
      EMochiArticulatedJointType::Free,
      EMochiArticulatedJointType::Spherical,
      EMochiArticulatedJointType::Revolute,
      EMochiArticulatedJointType::Hard,
      EMochiArticulatedJointType::Prismatic};
  const TArray<double> Expected = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1};
  TArray<double> Dofs = Expected;
  ConvertArticulatedDofsToUnreal(Types, true, Dofs, StudioToUnreal);
  // The revolute and prismatic values are convention independent.
  TestEqual(TEXT("Revolute DOF"), Dofs[9], Expected[9]);
  TestEqual(TEXT("Prismatic DOF"), Dofs[10], Expected[10]);
  ConvertArticulatedDofsToMochi(Types, true, Dofs, UnrealToStudio);
  for (int i = 0; i < Expected.Num(); ++i) {
    TestTrue(
        FString::Printf(TEXT("DOF %d"), i),
        FMath::IsNearlyEqual(Dofs[i], Expected[i], ArticulationTestTolerance));
  }
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiArticulatedDofIndexConversionTest,
    "SuperDexPhysics.Articulation.DofIndexConversion",
    MOCHI_ARTICULATION_TYPES_TEST_FLAGS)

bool FMochiArticulatedDofIndexConversionTest::RunTest(const FString& Parameters) {
  for (const EMochiArticulatedJointType Type :
       {EMochiArticulatedJointType::Free,
        EMochiArticulatedJointType::Spherical,
        EMochiArticulatedJointType::Revolute,
        EMochiArticulatedJointType::Prismatic}) {
    const int32 NumDofs = GetArticulatedJointNumDofs(Type);
    for (int32 Dof = 0; Dof < NumDofs; ++Dof) {
      const FString What = FString::Printf(TEXT("Joint type %d DOF %d"), int32(Type), Dof);
      // A value of the Unreal DOF must convert to the same value of the Mochi DOF as the whole
      // DOF vector does.
      int32 MochiDof = -1;
      double Sign = 0.0;
      if (!TestTrue(
              What, ConvertArticulatedDofIndexToMochi(Type, Dof, UnrealToStudio, MochiDof, Sign))) {
        continue;
      }
      TArray<double> Dofs;
      Dofs.SetNumZeroed(NumDofs);
      Dofs[Dof] = 0.5;
      ConvertArticulatedDofsToMochi(MakeArrayView(&Type, 1), true, Dofs, UnrealToStudio);
      TestTrue(What, FMath::IsNearlyEqual(Dofs[MochiDof], Sign * 0.5, ArticulationTestTolerance));
      int32 RoundTrip = -1;
      double RoundTripSign = 0.0;
      ConvertArticulatedDofIndexToUnreal(Type, MochiDof, StudioToUnreal, RoundTrip, RoundTripSign);
      TestEqual(What, RoundTrip, Dof);
      TestEqual(What, RoundTripSign, Sign);
    }
    int32 OutDof = 0;
    double Sign = 0.0;
    TestFalse(
        TEXT("DOF out of range"),
        ConvertArticulatedDofIndexToMochi(Type, NumDofs, UnrealToStudio, OutDof, Sign));
  }
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiRotationBoundsConversionTest,
    "SuperDexPhysics.Articulation.RotationBoundsConversion",
    MOCHI_ARTICULATION_TYPES_TEST_FLAGS)

bool FMochiRotationBoundsConversionTest::RunTest(const FString& Parameters) {
  // Studio to Unreal negates the X and Z rotation vector components, which swaps their bounds.
  const mochi::Real3 Min{-1_r, -2_r, -3_r};
  const mochi::Real3 Max{1_r, 2_r, 4_r};
  FVector UnrealMin;
  FVector UnrealMax;
  ConvertRotationBoundsToUnreal(Min, Max, StudioToUnreal, UnrealMin, UnrealMax);
  TestTrue(TEXT("Unreal min"), UnrealMin.Equals(FVector(-1, -2, -4), ArticulationTestTolerance));
  TestTrue(TEXT("Unreal max"), UnrealMax.Equals(FVector(1, 2, 3), ArticulationTestTolerance));
  mochi::Real3 RoundTripMin;
  mochi::Real3 RoundTripMax;
  ConvertRotationBoundsToMochi(UnrealMin, UnrealMax, UnrealToStudio, RoundTripMin, RoundTripMax);
  TestTrue(TEXT("Round trip min"), NearlyEqual(RoundTripMin, Min));
  TestTrue(TEXT("Round trip max"), NearlyEqual(RoundTripMax, Max));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiArticulationForwardKinematicsTest,
    "SuperDexPhysics.Articulation.ForwardKinematicsMatchesMochi",
    MOCHI_ARTICULATION_FIXTURE_TEST_FLAGS)

bool FMochiArticulationForwardKinematicsTest::RunTest(const FString& Parameters) {
  // A fixed root link with one revolute child, described in the Studio space.
  const mochi::Real3 Axis{0_r, 0.6_r, 0.8_r};
  const mochi::ArticulatedJointParams ChildJoint =
      MakeJoint(mochi::ArticulatedJointType::Revolute, Axis);
  const mochi::TransformRT ChildFromJointLink(
      mochi::Quaternion::FromAxisAngle(mochi::Real3{1_r, 0_r, 0_r}, 0.3_r),
      mochi::Real3{0.05_r, 0.0_r, -0.1_r});
  mochi::prefab::ArticulatedLinkPrefab Root;
  Root.name = "root";
  mochi::prefab::ArticulatedLinkPrefab Child;
  Child.name = "child";
  Child.parentLink = 0;
  Child.parentJointFromLink = ChildFromJointLink;

  TArray<FMochiArticulatedJoint> Joints;
  Joints.AddDefaulted(2);
  Joints[0].Type = EMochiArticulatedJointType::Hard;
  Joints[1].FromMochi(ChildJoint, StudioToUnreal);
  TArray<FMochiArticulatedLink> Links;
  Links.AddDefaulted(2);
  Links[0].FromMochi(Root, StudioToUnreal);
  Links[1].FromMochi(Child, StudioToUnreal);

  UMochiArticulation* Articulation = NewObject<UMochiArticulation>();
  Articulation->SetDefinition(TEXT("test"), Joints, Links);
  if (!TestEqual(TEXT("NumDofs"), Articulation->GetNumDofs(), 1)) {
    return false;
  }

  const double Q = 0.9;
  TArray<FTransform> LinkToRoot;
  if (!TestTrue(
          TEXT("FK"), Articulation->ComputeLinkToArticulationRootTransforms({Q}, LinkToRoot))) {
    return false;
  }

  // Mochi: rootFromChild = parentLinkFromJoint * rotation(axis, q) * parentJointFromLink.
  const mochi::TransformRT Expected = ChildJoint.parentLinkFromJoint *
      mochi::TransformRT(mochi::Quaternion::FromAxisAngle(Axis, static_cast<mochi::real>(Q)),
                         mochi::Real3{0_r, 0_r, 0_r}) *
      ChildFromJointLink;
  TestTrue(
      TEXT("Child link transform"),
      NearlyEqual(UnrealToStudio.TransformToMochi_RT(LinkToRoot[1]), Expected));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiArticulationSphericalForwardKinematicsTest,
    "SuperDexPhysics.Articulation.SphericalForwardKinematicsMatchesMochi",
    MOCHI_ARTICULATION_FIXTURE_TEST_FLAGS)

bool FMochiArticulationSphericalForwardKinematicsTest::RunTest(const FString& Parameters) {
  // A fixed root link with one spherical child, described in the Studio space.
  mochi::ArticulatedJointParams ChildJoint =
      MakeJoint(mochi::ArticulatedJointType::Spherical, mochi::Real3{0_r, 0_r, 0_r});
  ChildJoint.minLimit = std::nullopt;
  ChildJoint.maxLimit = std::nullopt;
  mochi::prefab::ArticulatedLinkPrefab Root;
  Root.name = "root";
  mochi::prefab::ArticulatedLinkPrefab Child;
  Child.name = "child";
  Child.parentLink = 0;

  TArray<FMochiArticulatedJoint> Joints;
  Joints.AddDefaulted(2);
  Joints[0].Type = EMochiArticulatedJointType::Hard;
  Joints[1].FromMochi(ChildJoint, StudioToUnreal);
  TArray<FMochiArticulatedLink> Links;
  Links.AddDefaulted(2);
  Links[0].FromMochi(Root, StudioToUnreal);
  Links[1].FromMochi(Child, StudioToUnreal);

  UMochiArticulation* Articulation = NewObject<UMochiArticulation>();
  Articulation->SetDefinition(TEXT("test"), Joints, Links);
  if (!TestEqual(TEXT("NumDofs"), Articulation->GetNumDofs(), 3)) {
    return false;
  }

  const mochi::Real3 RotationVector{0.3_r, -0.2_r, 0.5_r};
  TArray<double> Q = {RotationVector[0], RotationVector[1], RotationVector[2]};
  const TArray<EMochiArticulatedJointType> Types = {
      EMochiArticulatedJointType::Hard, EMochiArticulatedJointType::Spherical};
  ConvertArticulatedDofsToUnreal(Types, true, Q, StudioToUnreal);
  TArray<FTransform> LinkToRoot;
  if (!TestTrue(TEXT("FK"), Articulation->ComputeLinkToArticulationRootTransforms(Q, LinkToRoot))) {
    return false;
  }

  // Mochi: rootFromChild = parentLinkFromJoint * rotation(rotationVector).
  const mochi::TransformRT Expected = ChildJoint.parentLinkFromJoint *
      mochi::TransformRT(mochi::Quaternion::FromRotationVector(RotationVector));
  TestTrue(
      TEXT("Child link transform"),
      NearlyEqual(UnrealToStudio.TransformToMochi_RT(LinkToRoot[1]), Expected));
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiArticulationJointPositionsFromLinkTransformsTest,
    "SuperDexPhysics.Articulation.JointPositionsFromLinkTransformsRoundTrip",
    MOCHI_ARTICULATION_TYPES_TEST_FLAGS)

bool FMochiArticulationJointPositionsFromLinkTransformsTest::RunTest(const FString& Parameters) {
  // A fixed root link with a revolute, a prismatic and a spherical link chained below it.
  const TArray<mochi::ArticulatedJointType> Types = {
      mochi::ArticulatedJointType::Revolute,
      mochi::ArticulatedJointType::Prismatic,
      mochi::ArticulatedJointType::Spherical};
  TArray<FMochiArticulatedJoint> Joints;
  Joints.AddDefaulted(Types.Num() + 1);
  Joints[0].Type = EMochiArticulatedJointType::Hard;
  TArray<FMochiArticulatedLink> Links;
  Links.AddDefaulted(Types.Num() + 1);
  mochi::prefab::ArticulatedLinkPrefab Root;
  Root.name = "root";
  Links[0].FromMochi(Root, StudioToUnreal);
  const char* const LinkNames[] = {"link1", "link2", "link3"};
  for (int i = 0; i < Types.Num(); ++i) {
    mochi::ArticulatedJointParams Joint = MakeJoint(Types[i], mochi::Real3{0_r, 0.6_r, 0.8_r});
    Joint.minLimit = std::nullopt;
    Joint.maxLimit = std::nullopt;
    Joints[i + 1].FromMochi(Joint, StudioToUnreal);
    mochi::prefab::ArticulatedLinkPrefab Link;
    Link.name = LinkNames[i];
    Link.parentLink = i;
    Link.parentJointFromLink = mochi::TransformRT(
        mochi::Quaternion::FromAxisAngle(mochi::Real3{1_r, 0_r, 0_r}, 0.3_r),
        mochi::Real3{0.05_r, 0.0_r, -0.1_r});
    Links[i + 1].FromMochi(Link, StudioToUnreal);
  }
  UMochiArticulation* Articulation = NewObject<UMochiArticulation>();
  Articulation->SetDefinition(TEXT("test"), Joints, Links);
  if (!TestEqual(TEXT("NumDofs"), Articulation->GetNumDofs(), 5)) {
    return false;
  }

  const TArray<double> Q = {0.9, 0.04, 0.3, -0.2, 0.5}; // [rad], [m], spherical rotation vector
  for (const bool bArticulationRoot : {false, true}) {
    TArray<FTransform> LinkTransforms;
    TArray<double> Recovered;
    const bool bForward = bArticulationRoot
        ? Articulation->ComputeLinkToArticulationRootTransforms(Q, LinkTransforms)
        : Articulation->ComputeLinkToParentTransforms(Q, LinkTransforms);
    if (!TestTrue(TEXT("FK"), bForward)) {
      return false;
    }
    const bool bInverse = bArticulationRoot
        ? Articulation->ComputeJointPositionsFromLinkToArticulationRootTransforms(
              LinkTransforms, Recovered)
        : Articulation->ComputeJointPositionsFromLinkToParentTransforms(LinkTransforms, Recovered);
    if (!TestTrue(TEXT("Inverse"), bInverse) ||
        !TestEqual(TEXT("Joint position count"), Recovered.Num(), Q.Num())) {
      return false;
    }
    for (int i = 0; i < Q.Num(); ++i) {
      TestNearlyEqual(
          FString::Printf(TEXT("DOF %d"), i), Recovered[i], Q[i], ArticulationTestTolerance);
    }
  }
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiArticulationNamedFramesTest,
    "SuperDexPhysics.Articulation.NamedFramesAndBounds",
    MOCHI_ARTICULATION_FIXTURE_TEST_FLAGS)

bool FMochiArticulationNamedFramesTest::RunTest(const FString& Parameters) {
  const mochi::ArticulatedJointParams RootJoint =
      MakeJoint(mochi::ArticulatedJointType::Hard, mochi::Real3{0_r, 0_r, 1_r});
  const mochi::Real3 Axis{0_r, 0.6_r, 0.8_r};
  const mochi::ArticulatedJointParams ChildJoint =
      MakeJoint(mochi::ArticulatedJointType::Revolute, Axis);
  mochi::prefab::ArticulatedLinkPrefab Root;
  Root.name = "root";
  mochi::prefab::ArticulatedLinkPrefab Child;
  Child.name = "child";
  Child.parentLink = 0;
  Child.parentJointFromLink = mochi::TransformRT(
      mochi::Quaternion::FromAxisAngle(mochi::Real3{1_r, 0_r, 0_r}, 0.3_r),
      mochi::Real3{0.05_r, 0.0_r, -0.1_r});
  TArray<FMochiArticulatedJoint> Joints;
  Joints.AddDefaulted(2);
  Joints[0].FromMochi(RootJoint, StudioToUnreal);
  Joints[1].FromMochi(ChildJoint, StudioToUnreal);
  TArray<FMochiArticulatedLink> Links;
  Links.AddDefaulted(2);
  Links[0].FromMochi(Root, StudioToUnreal);
  Links[1].FromMochi(Child, StudioToUnreal);
  // Transient meshes exercise visual offsets and aggregation without loading assets.
  UStaticMesh* Mesh = NewObject<UStaticMesh>();
  Links[0].RenderModel = Mesh;
  Links[0].RenderModelTranslation = FVector(5, -3, 2);
  Links[1].RenderModel = Mesh;
  Links[1].RenderModelTranslation = FVector(-7, 4, 9);
  UMochiArticulation* Articulation = NewObject<UMochiArticulation>();
  Articulation->SetDefinition(TEXT("named_frames"), Joints, Links);
  const UMochiArticulationBase* Base = Articulation;
  const double Q = 0.9;
  TArray<FTransform> Parents;
  TArray<FTransform> Roots;
  if (!TestTrue(TEXT("Parent FK"), Base->ComputeLinkToParentTransforms({Q}, Parents)) ||
      !TestTrue(TEXT("Root FK"), Base->ComputeLinkToArticulationRootTransforms({Q}, Roots)) ||
      !TestEqual(TEXT("Parent count"), Parents.Num(), 2) ||
      !TestEqual(TEXT("Root count"), Roots.Num(), 2)) {
    return false;
  }
  const mochi::TransformRT ChildToParent = ChildJoint.parentLinkFromJoint *
      mochi::TransformRT(mochi::Quaternion::FromAxisAngle(Axis, static_cast<mochi::real>(Q))) *
      Child.parentJointFromLink;
  const FTransform ExpectedRoot = StudioToUnreal.TransformToUnreal(RootJoint.parentLinkFromJoint);
  const FTransform ExpectedChild = StudioToUnreal.TransformToUnreal(ChildToParent) * ExpectedRoot;
  TestTrue(
      TEXT("Parent root preserves root joint"),
      Parents[0].Equals(ExpectedRoot, ArticulationTestTolerance));
  TestTrue(
      TEXT("Root is not reset to identity"),
      Roots[0].Equals(ExpectedRoot, ArticulationTestTolerance));
  TestTrue(
      TEXT("Child relative to parent"),
      NearlyEqual(UnrealToStudio.TransformToMochi_RT(Parents[1]), ChildToParent));
  TestTrue(
      TEXT("Root accumulates nonidentity root"),
      Roots[1].Equals(ExpectedChild, ArticulationTestTolerance));
  TestFalse(
      TEXT("Parent and root child differ"), Parents[1].Equals(Roots[1], ArticulationTestTolerance));
  TArray<FTransform> Prefix;
  TestTrue(TEXT("Parent prefix"), Base->ComputeLinkToParentTransforms({Q}, Prefix, 1.0, 100.0, 0));
  if (TestEqual(TEXT("Final root count"), Prefix.Num(), 1)) {
    TestTrue(
        TEXT("Prefix root preserved"), Prefix[0].Equals(ExpectedRoot, ArticulationTestTolerance));
  }
  TestTrue(
      TEXT("Explicit scales and final index"),
      Base->ComputeLinkToArticulationRootTransforms({Q * 2}, Prefix, 0.5, 100.0, 1));
  if (TestEqual(TEXT("Final child count"), Prefix.Num(), 2)) {
    TestTrue(TEXT("Angle scale preserved"), Prefix[1].Equals(Roots[1], ArticulationTestTolerance));
  }
  FBoxSphereBounds ParentBounds;
  FBoxSphereBounds RootBounds;
  FBoxSphereBounds JointBounds;
  if (!TestTrue(
          TEXT("Parent bounds"),
          Base->ComputeVisualBoundsFromLinkToParentTransforms(Parents, ParentBounds)) ||
      !TestTrue(
          TEXT("Root bounds"),
          Base->ComputeVisualBoundsFromLinkToArticulationRootTransforms(Roots, RootBounds)) ||
      !TestTrue(
          TEXT("Joint bounds"), Base->ComputeVisualBoundsFromJointPositions({Q}, JointBounds))) {
    return false;
  }
  FBox ExpectedBox(ForceInit);
  ExpectedBox +=
      Mesh->GetBounds().GetBox().TransformBy(Links[0].GetRenderModelTransform() * ExpectedRoot);
  ExpectedBox +=
      Mesh->GetBounds().GetBox().TransformBy(Links[1].GetRenderModelTransform() * ExpectedChild);
  const FBoxSphereBounds ExpectedBounds(ExpectedBox);
  auto CheckBounds = [&](const FBoxSphereBounds& Bounds) {
    TestTrue(
        TEXT("Bounds origin"),
        Bounds.Origin.Equals(ExpectedBounds.Origin, ArticulationTestTolerance));
    TestTrue(
        TEXT("Bounds extent"),
        Bounds.BoxExtent.Equals(ExpectedBounds.BoxExtent, ArticulationTestTolerance));
    TestTrue(
        TEXT("Bounds radius"),
        FMath::IsNearlyEqual(
            static_cast<double>(Bounds.SphereRadius),
            static_cast<double>(ExpectedBounds.SphereRadius),
            ArticulationTestTolerance));
  };
  CheckBounds(ParentBounds);
  CheckBounds(RootBounds);
  CheckBounds(JointBounds);
  TestTrue(TEXT("Nonempty aggregate bounds"), RootBounds.SphereRadius > 0);
  return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
