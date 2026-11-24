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

#include "MochiArticulation.h"
#include "MochiConversions.h"
#include "SuperDexBot.h"

#include "Misc/AutomationTest.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UnrealType.h"

#include <superdex_robotics/utils/bot_utils.h>
#include <superdex_robotics/utils/file_utils.h>

#include <limits>
#include <random>

#if WITH_DEV_AUTOMATION_TESTS

// Joint positions <-> link transforms on USuperDexBot and UMochiArticulation built from the same
// bot: they agree, transforms from joint positions round-trip, and transforms the joints can't
// produce (data is lost) give the nearest joint positions joint by joint.

namespace {

using mochi::operator""_r;

constexpr double AngleTolerance = 2e-4; // [rad]
constexpr double LengthTolerance = 1e-3; // [cm]
constexpr double PositionTolerance = 2e-4; // [rad or m]

// Angle of the rotation between two rotations [rad], in [0, pi]. Unlike FQuat::AngularDistance,
// whose acos argument can round past 1, this stays finite for nearly equal rotations.
double RotationDistance(const FQuat& A, const FQuat& B) {
  const FQuat Delta = A.GetNormalized().Inverse() * B.GetNormalized();
  return 2.0 * FMath::Atan2(FVector(Delta.X, Delta.Y, Delta.Z).Size(), FMath::Abs(Delta.W));
}

struct FChainJoint {
  mochi::ArticulatedJointType Type;
  int Parent;
  mochi::Real3 Axis;
};

// Every joint kind, a non-unit revolute axis, spherical links under revolute and prismatic links,
// and a revolute link under a spherical one.
const FChainJoint Chain[] = {
    {mochi::ArticulatedJointType::Hard, -1, {0_r, 0_r, 1_r}},
    {mochi::ArticulatedJointType::Revolute, 0, {0_r, 1.2_r, 1.6_r}},
    {mochi::ArticulatedJointType::Prismatic, 1, {0_r, 0.6_r, 0.8_r}},
    {mochi::ArticulatedJointType::Spherical, 2, {0_r, 0_r, 1_r}},
    {mochi::ArticulatedJointType::Spherical, 1, {0_r, 0_r, 1_r}},
    {mochi::ArticulatedJointType::Revolute, 4, {1_r, 0_r, 0_r}},
    {mochi::ArticulatedJointType::Hard, 5, {0_r, 0_r, 1_r}},
};
constexpr int NumLinks = UE_ARRAY_COUNT(Chain);

// Rest frames that rotate and translate on both sides of every non-root joint.
mochi::TransformRT ChainParentLinkFromJoint(int I) {
  const mochi::real S = static_cast<mochi::real>(I);
  return I == 0
      ? mochi::TransformRT::Identity()
      : mochi::TransformRT{
            mochi::Quaternion::FromRotationVector(mochi::Real3{0.1_r * S, -0.2_r, 0.15_r}),
            mochi::Real3{0.03_r, -0.02_r * S, 0.05_r}};
}

mochi::TransformRT ChainParentJointFromLink(int I) {
  const mochi::real S = static_cast<mochi::real>(I);
  return I == 0
      ? mochi::TransformRT::Identity()
      : mochi::TransformRT{
            mochi::Quaternion::FromRotationVector(mochi::Real3{-0.2_r, 0.05_r * S, 0.1_r}),
            mochi::Real3{0.01_r, -0.02_r, 0.03_r}};
}

#if !MOCHI_UNREAL_LINUX_EDITOR
// The chain as a bot.
superdex::robotics::BotPrefab MakeChainPrefab(mochi::Error& Error) {
  using namespace superdex::robotics;
  BotPrefab Prefab;
  Prefab.name = "kinematics_test";
  for (int i = 0; i < NumLinks; ++i) {
    BotLinkPrefab Link;
    Link.name = "link" + std::to_string(i);
    Link.parentLink = Chain[i].Parent;
    Link.mass = 1_r;
    Link.centerOfMass = mochi::Real3{};
    Link.momentOfInertia = {1_r, 0_r, 0_r, 1_r, 0_r, 1_r};
    Link.parentJointFromLink = ChainParentJointFromLink(i);
    Prefab.links.push_back(Link);
    BotJointPrefab Joint;
    Joint.name = "joint" + std::to_string(i);
    Joint.type = Chain[i].Type;
    Joint.axis = Chain[i].Axis;
    Joint.parentLinkFromJoint = ChainParentLinkFromJoint(i);
    Prefab.joints.push_back(Joint);
  }
  RebuildBotData(Prefab, Error);
  Prefab.defaultPose = mochi::DynamicArray<mochi::real>(Prefab._numDofs, 0_r);
  Validate(Prefab, nullptr, Error);
  return Prefab;
}

USuperDexBot* MakeBot(const superdex::robotics::BotPrefab& Prefab) {
  USuperDexBot* Bot = NewObject<USuperDexBot>(GetTransientPackage());
  mochi::Error Error;
  const mochi::DynamicString Json = superdex::robotics::WriteToJsonString(Prefab, Error);
  check(Error.IsOK());
  FindFProperty<FStrProperty>(USuperDexBot::StaticClass(), TEXT("BotPrefabJson"))
      ->SetPropertyValue_InContainer(Bot, FString(UTF8_TO_TCHAR(Json.c_str())));
  return Bot;
}
#endif // !MOCHI_UNREAL_LINUX_EDITOR

// The chain as an articulation, built without superdex_robotics.
UMochiArticulation* MakeArticulation() {
  const FMochiToUnrealConverter M2U(SuperDexSpace);
  TArray<FMochiArticulatedJoint> Joints;
  TArray<FMochiArticulatedLink> Links;
  Joints.AddDefaulted(NumLinks);
  Links.AddDefaulted(NumLinks);
  for (int i = 0; i < NumLinks; ++i) {
    mochi::ArticulatedJointParams Joint;
    Joint.name = "joint" + std::to_string(i);
    Joint.type = Chain[i].Type;
    Joint.parentLinkFromJoint = ChainParentLinkFromJoint(i);
    Joint.axis = Chain[i].Axis;
    Joints[i].FromMochi(Joint, M2U);
    mochi::prefab::ArticulatedLinkPrefab Link;
    Link.name = "link" + std::to_string(i);
    Link.parentLink = Chain[i].Parent;
    Link.parentJointFromLink = ChainParentJointFromLink(i);
    Links[i].FromMochi(Link, M2U);
  }
  UMochiArticulation* Articulation = NewObject<UMochiArticulation>(GetTransientPackage());
  Articulation->SetDefinition(TEXT("kinematics_test"), Joints, Links);
  return Articulation;
}

// Index of each link's first joint position, or -1 for links without any.
TArray<int> FirstDofIndices() {
  TArray<int> First;
  int DofIndex = 0;
  for (const FChainJoint& Joint : Chain) {
    const int NumDofs = Joint.Type == mochi::ArticulatedJointType::Spherical ? 3
        : Joint.Type == mochi::ArticulatedJointType::Revolute ||
            Joint.Type == mochi::ArticulatedJointType::Prismatic
        ? 1
        : 0;
    First.Add(NumDofs > 0 ? DofIndex : -1);
    DofIndex += NumDofs;
  }
  return First;
}

struct FKinematicsChecker {
  FAutomationTestBase& Test;
  const UMochiArticulationBase& Articulation;
  const TArray<int> FirstDof = FirstDofIndices();

  TArray<FTransform> Forward(const TArray<double>& JointPositions, bool bArticulationRoot) const {
    TArray<FTransform> Transforms;
    const bool bSucceeded = bArticulationRoot
        ? Articulation.ComputeLinkToArticulationRootTransforms(JointPositions, Transforms)
        : Articulation.ComputeLinkToParentTransforms(JointPositions, Transforms);
    Test.TestTrue(TEXT("Forward"), bSucceeded);
    return Transforms;
  }

  TArray<double> Inverse(const TArray<FTransform>& Transforms, bool bArticulationRoot) const {
    TArray<double> JointPositions;
    const bool bSucceeded = bArticulationRoot
        ? Articulation.ComputeJointPositionsFromLinkToArticulationRootTransforms(
              Transforms, JointPositions)
        : Articulation.ComputeJointPositionsFromLinkToParentTransforms(Transforms, JointPositions);
    Test.TestTrue(TEXT("Inverse"), bSucceeded);
    return JointPositions;
  }

  // Revolute angles and rotation-vector components in [-1, 1] rad, prismatic positions in
  // [-0.05, 0.05] m.
  TArray<double> RandomJointPositions(std::mt19937& Rng) const {
    std::uniform_real_distribution<double> Unit(-1.0, 1.0);
    TArray<double> JointPositions;
    for (const FChainJoint& Joint : Chain) {
      switch (Joint.Type) {
        case mochi::ArticulatedJointType::Revolute:
          JointPositions.Add(Unit(Rng));
          break;
        case mochi::ArticulatedJointType::Prismatic:
          JointPositions.Add(0.05 * Unit(Rng));
          break;
        case mochi::ArticulatedJointType::Spherical:
          for (int k = 0; k < 3; ++k) {
            JointPositions.Add(Unit(Rng));
          }
          break;
        default:
          break;
      }
    }
    return JointPositions;
  }

  // LinkToParent transforms of random joint positions with every non-root link moved by a random
  // rotation of up to about 1.2 rad and a random translation of up to about 2 cm, which in general
  // no joint can fully produce.
  TArray<FTransform> RandomOffAxisLinksToParent(std::mt19937& Rng) const {
    std::uniform_real_distribution<double> Unit(-1.0, 1.0);
    auto Random3 = [&](double Magnitude) {
      return FVector(Unit(Rng), Unit(Rng), Unit(Rng)) * Magnitude;
    };
    TArray<FTransform> Transforms = Forward(RandomJointPositions(Rng), false);
    for (int i = 1; i < Transforms.Num(); ++i) {
      Transforms[i] =
          FTransform(FQuat::MakeFromRotationVector(Random3(0.7)), Random3(1.2)) * Transforms[i];
    }
    return Transforms;
  }

  // How far link I posed by JointPositions is from Target relative to its parent, in what its joint
  // controls: the link rotation for rotational joints [rad], the joint frame's origin (the link's
  // transform without ParentJointFromLink) for prismatic ones [cm].
  double JointError(const TArray<double>& JointPositions, int I, const FTransform& Target) const {
    const FTransform Link = Forward(JointPositions, false)[I];
    if (Chain[I].Type != mochi::ArticulatedJointType::Prismatic) {
      return RotationDistance(Link.GetRotation(), Target.GetRotation());
    }
    const FTransform JointFromLink =
        FMochiToUnrealConverter(SuperDexSpace).TransformToUnreal(ChainParentJointFromLink(I));
    return FVector::Dist(
        (JointFromLink.Inverse() * Link).GetTranslation(),
        (JointFromLink.Inverse() * Target).GetTranslation());
  }

  // Each revolute and prismatic joint position is the best for its link against a dense sweep, and
  // each spherical joint matches its link's rotation exactly.
  void ExpectJointwiseNearest(
      const TArray<double>& JointPositions,
      const TArray<FTransform>& LinksToParent,
      const FString& What) const {
    for (int i = 1; i < NumLinks; ++i) {
      if (FirstDof[i] < 0) {
        continue;
      }
      const double Best = JointError(JointPositions, i, LinksToParent[i]);
      if (Chain[i].Type == mochi::ArticulatedJointType::Spherical) {
        Test.TestTrue(
            FString::Printf(TEXT("%s: spherical link %d matches its rotation"), *What, i),
            Best < AngleTolerance);
        continue;
      }
      const bool bRevolute = Chain[i].Type == mochi::ArticulatedJointType::Revolute;
      TArray<double> Trial = JointPositions;
      constexpr int Steps = 1000;
      double SweepBest = std::numeric_limits<double>::infinity();
      for (int k = 0; k <= Steps; ++k) {
        const double T = 2.0 * k / Steps - 1.0;
        Trial[FirstDof[i]] = bRevolute ? UE_DOUBLE_PI * T : JointPositions[FirstDof[i]] + 0.1 * T;
        SweepBest = FMath::Min(SweepBest, JointError(Trial, i, LinksToParent[i]));
      }
      Test.TestTrue(
          FString::Printf(
              TEXT("%s: link %d is nearest (%g vs sweep %g)"), *What, i, Best, SweepBest),
          Best <= SweepBest + (bRevolute ? 5e-5 : 1e-4));
    }
  }

  void ExpectSameJointPositions(
      const TArray<double>& Actual,
      const TArray<double>& Expected,
      const FString& What) const {
    if (!Test.TestEqual(What + TEXT(" count"), Actual.Num(), Expected.Num())) {
      return;
    }
    for (int i = 0; i < Actual.Num(); ++i) {
      Test.TestNearlyEqual(
          FString::Printf(TEXT("%s DOF %d"), *What, i), Actual[i], Expected[i], PositionTolerance);
    }
  }

  void ExpectSameLink(const FTransform& Actual, const FTransform& Expected, const FString& What)
      const {
    Test.TestTrue(
        What + TEXT(" rotation"),
        RotationDistance(Actual.GetRotation(), Expected.GetRotation()) < AngleTolerance);
    Test.TestTrue(
        What + TEXT(" location"),
        FVector::Dist(Actual.GetTranslation(), Expected.GetTranslation()) < LengthTolerance);
  }

  void Run() const {
    const bool bParentSpace = false;
    const bool bRootSpace = true;
    std::mt19937 Rng(17);

    // Transforms from joint positions round-trip.
    for (const bool bSpace : {bParentSpace, bRootSpace}) {
      const TArray<double> JointPositions = RandomJointPositions(Rng);
      ExpectSameJointPositions(
          Inverse(Forward(JointPositions, bSpace), bSpace), JointPositions, TEXT("Round trip"));
    }

    // Off-axis transforms lose data: they give the nearest joint positions joint by joint, the
    // transforms of those give them back, negated quaternions change nothing, and the same skeleton
    // in LinkToRoot space gives the same joint positions (an error stays in its own joint).
    for (int Trial = 0; Trial < 3; ++Trial) {
      const TArray<FTransform> LinksToParent = RandomOffAxisLinksToParent(Rng);
      const TArray<double> Projected = Inverse(LinksToParent, bParentSpace);
      ExpectJointwiseNearest(Projected, LinksToParent, TEXT("Off axis"));
      ExpectSameJointPositions(
          Inverse(Forward(Projected, bParentSpace), bParentSpace), Projected, TEXT("Reprojection"));
      TArray<FTransform> Negated = LinksToParent;
      for (FTransform& T : Negated) {
        const FQuat Q = T.GetRotation();
        T.SetRotation(FQuat(-Q.X, -Q.Y, -Q.Z, -Q.W));
      }
      ExpectSameJointPositions(
          Inverse(Negated, bParentSpace), Projected, TEXT("Negated quaternions"));
      TArray<FTransform> LinksToRoot = LinksToParent;
      for (int i = 1; i < NumLinks; ++i) {
        LinksToRoot[i] = LinksToParent[i] * LinksToRoot[Chain[i].Parent];
      }
      ExpectSameJointPositions(Inverse(LinksToRoot, bRootSpace), Projected, TEXT("LinkToRoot"));
    }

    TArray<double> Zero;
    Zero.Init(0.0, Articulation.GetNumDofs());
    const int Revolute = 1;
    const int Spherical = 3;
    const int RevoluteIndex = FirstDof[Revolute];

    // A half turn about an axis perpendicular to the revolute axis: no angle is nearer than
    // another, so the angle is 0.
    {
      // LinkToParent = ParentJointFromLink * Motion * ParentLinkFromJoint, with the turn as Motion.
      const FMochiToUnrealConverter M2U(SuperDexSpace);
      const FVector Axis = M2U.DirectionToUnreal(Chain[Revolute].Axis).GetSafeNormal();
      const FVector Perpendicular =
          FVector::CrossProduct(Axis, FVector::ForwardVector).GetSafeNormal();
      TArray<FTransform> Targets = Forward(Zero, bParentSpace);
      Targets[Revolute] = M2U.TransformToUnreal(ChainParentJointFromLink(Revolute)) *
          FTransform(FQuat(Perpendicular, UE_DOUBLE_PI)) *
          M2U.TransformToUnreal(ChainParentLinkFromJoint(Revolute));
      const TArray<double> Projected = Inverse(Targets, bParentSpace);
      Test.TestEqual(TEXT("Half turn: angle 0"), Projected[RevoluteIndex], 0.0);
      ExpectJointwiseNearest(Projected, Targets, TEXT("Half turn"));
    }

    // Revolute angles past +-pi wrap into [-pi, pi], to the same rotation.
    for (const double Angle :
         {UE_DOUBLE_PI - 1e-3,
          -UE_DOUBLE_PI + 1e-3,
          UE_DOUBLE_PI + 1e-2,
          -UE_DOUBLE_PI - 1e-2,
          7.0}) {
      TArray<double> JointPositions = Zero;
      JointPositions[RevoluteIndex] = Angle;
      const double Recovered =
          Inverse(Forward(JointPositions, bParentSpace), bParentSpace)[RevoluteIndex];
      Test.TestTrue(TEXT("Wrapped angle within pi"), FMath::Abs(Recovered) <= UE_DOUBLE_PI + 1e-6);
      Test.TestNearlyEqual(
          FString::Printf(TEXT("Revolute angle %g"), Angle),
          FMath::UnwindRadians(Recovered - Angle),
          0.0,
          PositionTolerance);
    }

    // Spherical rotations near a half turn: the rotation vector has norm <= pi and is the same
    // rotation.
    for (const double Angle : {UE_DOUBLE_PI - 1e-3, UE_DOUBLE_PI + 1e-3}) {
      TArray<double> JointPositions = Zero;
      const FVector RotationVector = FVector(0.3, -0.5, 0.8).GetSafeNormal() * Angle;
      for (int k = 0; k < 3; ++k) {
        JointPositions[FirstDof[Spherical] + k] = RotationVector[k];
      }
      const TArray<FTransform> Targets = Forward(JointPositions, bParentSpace);
      const TArray<double> Recovered = Inverse(Targets, bParentSpace);
      const FVector RecoveredVector(
          Recovered[FirstDof[Spherical]],
          Recovered[FirstDof[Spherical] + 1],
          Recovered[FirstDof[Spherical] + 2]);
      Test.TestTrue(
          TEXT("Rotation vector within pi"), RecoveredVector.Size() <= UE_DOUBLE_PI + 1e-6);
      ExpectSameLink(
          Forward(Recovered, bParentSpace)[Spherical],
          Targets[Spherical],
          FString::Printf(TEXT("Spherical link at %g"), Angle));
    }

    // A mismatched transform count is rejected without changing the output.
    const TArray<double> Sentinel = {1.0, 2.0, 3.0};
    const TArray<FTransform> Rest = Forward(Zero, bParentSpace);
    TArray<FTransform> ShortTransforms = Rest;
    ShortTransforms.Pop();
    TArray<double> Out = Sentinel;
    Test.TestFalse(
        TEXT("Wrong transform count rejected"),
        Articulation.ComputeJointPositionsFromLinkToParentTransforms(ShortTransforms, Out));
    Test.TestEqual(TEXT("Output unchanged"), Out, Sentinel);
  }
};

} // namespace

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexBotKinematicsTest,
    "SuperDexRobotics.Kinematics.JointPositionsAndLinkTransforms",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSuperDexBotKinematicsTest::RunTest(const FString& Parameters) {
  TStrongObjectPtr<UMochiArticulation> Articulation(MakeArticulation());
  const FKinematicsChecker ArticulationChecker{*this, *Articulation};
  AddInfo(TEXT("UMochiArticulation"));
  ArticulationChecker.Run();

#if MOCHI_UNREAL_LINUX_EDITOR
  // Mochi structs don't cross into superdex_robotics here: the Linux editor builds Mochi headers
  // without AVX2, so their layout differs from the library's (see SuperDexPhysics.Build.cs).
  AddInfo(TEXT("USuperDexBot skipped in the Linux editor"));
#else
  mochi::Error Error;
  const superdex::robotics::BotPrefab Prefab = MakeChainPrefab(Error);
  if (!TestTrue(
          FString::Printf(TEXT("Chain bot is valid: %hs"), Error.GetDescription()), Error.IsOK())) {
    return false;
  }
  TStrongObjectPtr<USuperDexBot> Bot(MakeBot(Prefab));
  if (!TestEqual(TEXT("Bot DOFs"), Bot->GetNumDofs(), Articulation->GetNumDofs())) {
    return false;
  }
  // USuperDexBot and UMochiArticulation agree, in both directions, on and off axis.
  std::mt19937 Rng(23);
  const FKinematicsChecker BotChecker{*this, *Bot};
  for (const bool bSpace : {false, true}) {
    const TArray<double> JointPositions = BotChecker.RandomJointPositions(Rng);
    const TArray<FTransform> BotLinks = BotChecker.Forward(JointPositions, bSpace);
    const TArray<FTransform> ArticulationLinks =
        ArticulationChecker.Forward(JointPositions, bSpace);
    if (!TestEqual(TEXT("Link count"), BotLinks.Num(), ArticulationLinks.Num())) {
      return false;
    }
    for (int i = 0; i < BotLinks.Num(); ++i) {
      BotChecker.ExpectSameLink(
          BotLinks[i], ArticulationLinks[i], FString::Printf(TEXT("Link %d"), i));
    }
    const TArray<FTransform> Targets = BotChecker.RandomOffAxisLinksToParent(Rng);
    BotChecker.ExpectSameJointPositions(
        BotChecker.Inverse(Targets, false),
        ArticulationChecker.Inverse(Targets, false),
        TEXT("Off-axis inverse"));
  }
  AddInfo(TEXT("USuperDexBot"));
  BotChecker.Run();
#endif
  return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
