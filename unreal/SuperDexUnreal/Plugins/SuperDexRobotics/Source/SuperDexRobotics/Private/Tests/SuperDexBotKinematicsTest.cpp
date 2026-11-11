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

constexpr double kAngleTolerance = 2e-4; // [rad]
constexpr double kLengthTolerance = 1e-3; // [cm]
constexpr double kPositionTolerance = 2e-4; // [rad or m]

// Angle of the rotation between two rotations [rad], in [0, pi]. Unlike FQuat::AngularDistance,
// whose acos argument can round past 1, this stays finite for nearly equal rotations.
double RotationDistance(const FQuat& A, const FQuat& B) {
  const FQuat delta = A.GetNormalized().Inverse() * B.GetNormalized();
  return 2.0 * FMath::Atan2(FVector(delta.X, delta.Y, delta.Z).Size(), FMath::Abs(delta.W));
}

struct FChainJoint {
  mochi::ArticulatedJointType Type;
  int Parent;
  mochi::Real3 Axis;
};

// Every joint kind, a non-unit revolute axis, spherical links under revolute and prismatic links,
// and a revolute link under a spherical one.
const FChainJoint kChain[] = {
    {mochi::ArticulatedJointType::Hard, -1, {0_r, 0_r, 1_r}},
    {mochi::ArticulatedJointType::Revolute, 0, {0_r, 1.2_r, 1.6_r}},
    {mochi::ArticulatedJointType::Prismatic, 1, {0_r, 0.6_r, 0.8_r}},
    {mochi::ArticulatedJointType::Spherical, 2, {0_r, 0_r, 1_r}},
    {mochi::ArticulatedJointType::Spherical, 1, {0_r, 0_r, 1_r}},
    {mochi::ArticulatedJointType::Revolute, 4, {1_r, 0_r, 0_r}},
    {mochi::ArticulatedJointType::Hard, 5, {0_r, 0_r, 1_r}},
};
constexpr int kNumLinks = UE_ARRAY_COUNT(kChain);

// Rest frames that rotate and translate on both sides of every non-root joint.
mochi::TransformRT ChainParentLinkFromJoint(int I) {
  const mochi::real s = static_cast<mochi::real>(I);
  return I == 0
      ? mochi::TransformRT::Identity()
      : mochi::TransformRT{
            mochi::Quaternion::FromRotationVector(mochi::Real3{0.1_r * s, -0.2_r, 0.15_r}),
            mochi::Real3{0.03_r, -0.02_r * s, 0.05_r}};
}

mochi::TransformRT ChainParentJointFromLink(int I) {
  const mochi::real s = static_cast<mochi::real>(I);
  return I == 0
      ? mochi::TransformRT::Identity()
      : mochi::TransformRT{
            mochi::Quaternion::FromRotationVector(mochi::Real3{-0.2_r, 0.05_r * s, 0.1_r}),
            mochi::Real3{0.01_r, -0.02_r, 0.03_r}};
}

#if !MOCHI_UNREAL_LINUX_EDITOR
// The chain as a bot.
superdex::robotics::BotPrefab MakeChainPrefab(mochi::Error& Error) {
  using namespace superdex::robotics;
  BotPrefab prefab;
  prefab.name = "kinematics_test";
  for (int i = 0; i < kNumLinks; ++i) {
    BotLinkPrefab link;
    link.name = "link" + std::to_string(i);
    link.parentLink = kChain[i].Parent;
    link.mass = 1_r;
    link.centerOfMass = mochi::Real3{};
    link.momentOfInertia = {1_r, 0_r, 0_r, 1_r, 0_r, 1_r};
    link.parentJointFromLink = ChainParentJointFromLink(i);
    prefab.links.push_back(link);
    BotJointPrefab joint;
    joint.name = "joint" + std::to_string(i);
    joint.type = kChain[i].Type;
    joint.axis = kChain[i].Axis;
    joint.parentLinkFromJoint = ChainParentLinkFromJoint(i);
    prefab.joints.push_back(joint);
  }
  RebuildBotData(prefab, Error);
  prefab.defaultPose = mochi::DynamicArray<mochi::real>(prefab._numDofs, 0_r);
  Validate(prefab, nullptr, Error);
  return prefab;
}

USuperDexBot* MakeBot(const superdex::robotics::BotPrefab& Prefab) {
  auto* bot = NewObject<USuperDexBot>(GetTransientPackage());
  mochi::Error error;
  const auto json = superdex::robotics::WriteToJsonString(Prefab, error);
  check(error.IsOK());
  FindFProperty<FStrProperty>(USuperDexBot::StaticClass(), TEXT("BotPrefabJson"))
      ->SetPropertyValue_InContainer(bot, FString(UTF8_TO_TCHAR(json.c_str())));
  return bot;
}
#endif // !MOCHI_UNREAL_LINUX_EDITOR

// The chain as an articulation, built without superdex_robotics.
UMochiArticulation* MakeArticulation() {
  const FMochiToUnrealConverter m2u(kSuperDexSpace);
  TArray<FMochiArticulatedJoint> joints;
  TArray<FMochiArticulatedLink> links;
  joints.AddDefaulted(kNumLinks);
  links.AddDefaulted(kNumLinks);
  for (int i = 0; i < kNumLinks; ++i) {
    mochi::ArticulatedJointParams joint;
    joint.name = "joint" + std::to_string(i);
    joint.type = kChain[i].Type;
    joint.parentLinkFromJoint = ChainParentLinkFromJoint(i);
    joint.axis = kChain[i].Axis;
    joints[i].FromMochi(joint, m2u);
    mochi::prefab::ArticulatedLinkPrefab link;
    link.name = "link" + std::to_string(i);
    link.parentLink = kChain[i].Parent;
    link.parentJointFromLink = ChainParentJointFromLink(i);
    links[i].FromMochi(link, m2u);
  }
  auto* articulation = NewObject<UMochiArticulation>(GetTransientPackage());
  articulation->SetDefinition(TEXT("kinematics_test"), joints, links);
  return articulation;
}

// Index of each link's first joint position, or -1 for links without any.
TArray<int> FirstDofIndices() {
  TArray<int> first;
  int iDof = 0;
  for (const FChainJoint& joint : kChain) {
    const int numDofs = joint.Type == mochi::ArticulatedJointType::Spherical ? 3
        : joint.Type == mochi::ArticulatedJointType::Revolute ||
            joint.Type == mochi::ArticulatedJointType::Prismatic
        ? 1
        : 0;
    first.Add(numDofs > 0 ? iDof : -1);
    iDof += numDofs;
  }
  return first;
}

struct FKinematicsChecker {
  FAutomationTestBase& Test;
  const UMochiArticulationBase& Articulation;
  const TArray<int> FirstDof = FirstDofIndices();

  TArray<FTransform> Forward(
      const TArray<double>& JointPositions,
      EMochiArticulationLinkTransformSpace Space) const {
    TArray<FTransform> transforms;
    Test.TestTrue(
        TEXT("Forward"),
        Articulation.ComputeLinkTransformsFromJointPositions(JointPositions, transforms, Space));
    return transforms;
  }

  TArray<double> Inverse(
      const TArray<FTransform>& Transforms,
      EMochiArticulationLinkTransformSpace Space) const {
    TArray<double> jointPositions;
    Test.TestTrue(
        TEXT("Inverse"),
        Articulation.ComputeJointPositionsFromLinkTransforms(Transforms, jointPositions, Space));
    return jointPositions;
  }

  // Revolute angles and rotation-vector components in [-1, 1] rad, prismatic positions in
  // [-0.05, 0.05] m.
  TArray<double> RandomJointPositions(std::mt19937& Rng) const {
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    TArray<double> jointPositions;
    for (const FChainJoint& joint : kChain) {
      switch (joint.Type) {
        case mochi::ArticulatedJointType::Revolute:
          jointPositions.Add(unit(Rng));
          break;
        case mochi::ArticulatedJointType::Prismatic:
          jointPositions.Add(0.05 * unit(Rng));
          break;
        case mochi::ArticulatedJointType::Spherical:
          for (int k = 0; k < 3; ++k) {
            jointPositions.Add(unit(Rng));
          }
          break;
        default:
          break;
      }
    }
    return jointPositions;
  }

  // LinkToParent transforms of random joint positions with every non-root link moved by a random
  // rotation of up to about 1.2 rad and a random translation of up to about 2 cm, which in general
  // no joint can fully produce.
  TArray<FTransform> RandomOffAxisLinksToParent(std::mt19937& Rng) const {
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    auto random3 = [&](double Magnitude) {
      return FVector(unit(Rng), unit(Rng), unit(Rng)) * Magnitude;
    };
    TArray<FTransform> transforms =
        Forward(RandomJointPositions(Rng), EMochiArticulationLinkTransformSpace::LinkToParent);
    for (int i = 1; i < transforms.Num(); ++i) {
      transforms[i] =
          FTransform(FQuat::MakeFromRotationVector(random3(0.7)), random3(1.2)) * transforms[i];
    }
    return transforms;
  }

  // How far link I posed by JointPositions is from Target relative to its parent, in what its joint
  // controls: the link rotation for rotational joints [rad], the joint frame's origin (the link's
  // transform without ParentJointFromLink) for prismatic ones [cm].
  double JointError(const TArray<double>& JointPositions, int I, const FTransform& Target) const {
    const FTransform link =
        Forward(JointPositions, EMochiArticulationLinkTransformSpace::LinkToParent)[I];
    if (kChain[I].Type != mochi::ArticulatedJointType::Prismatic) {
      return RotationDistance(link.GetRotation(), Target.GetRotation());
    }
    const FTransform jointFromLink =
        FMochiToUnrealConverter(kSuperDexSpace).TransformToUnreal(ChainParentJointFromLink(I));
    return FVector::Dist(
        (jointFromLink.Inverse() * link).GetTranslation(),
        (jointFromLink.Inverse() * Target).GetTranslation());
  }

  // Each revolute and prismatic joint position is the best for its link against a dense sweep, and
  // each spherical joint matches its link's rotation exactly.
  void ExpectJointwiseNearest(
      const TArray<double>& JointPositions,
      const TArray<FTransform>& LinksToParent,
      const FString& What) const {
    for (int i = 1; i < kNumLinks; ++i) {
      if (FirstDof[i] < 0) {
        continue;
      }
      const double best = JointError(JointPositions, i, LinksToParent[i]);
      if (kChain[i].Type == mochi::ArticulatedJointType::Spherical) {
        Test.TestTrue(
            FString::Printf(TEXT("%s: spherical link %d matches its rotation"), *What, i),
            best < kAngleTolerance);
        continue;
      }
      const bool bRevolute = kChain[i].Type == mochi::ArticulatedJointType::Revolute;
      TArray<double> trial = JointPositions;
      constexpr int kSteps = 1000;
      double sweepBest = std::numeric_limits<double>::infinity();
      for (int k = 0; k <= kSteps; ++k) {
        const double t = 2.0 * k / kSteps - 1.0;
        trial[FirstDof[i]] = bRevolute ? UE_DOUBLE_PI * t : JointPositions[FirstDof[i]] + 0.1 * t;
        sweepBest = FMath::Min(sweepBest, JointError(trial, i, LinksToParent[i]));
      }
      Test.TestTrue(
          FString::Printf(
              TEXT("%s: link %d is nearest (%g vs sweep %g)"), *What, i, best, sweepBest),
          best <= sweepBest + (bRevolute ? 5e-5 : 1e-4));
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
          FString::Printf(TEXT("%s DOF %d"), *What, i), Actual[i], Expected[i], kPositionTolerance);
    }
  }

  void ExpectSameLink(const FTransform& Actual, const FTransform& Expected, const FString& What)
      const {
    Test.TestTrue(
        What + TEXT(" rotation"),
        RotationDistance(Actual.GetRotation(), Expected.GetRotation()) < kAngleTolerance);
    Test.TestTrue(
        What + TEXT(" location"),
        FVector::Dist(Actual.GetTranslation(), Expected.GetTranslation()) < kLengthTolerance);
  }

  void Run() const {
    const auto parentSpace = EMochiArticulationLinkTransformSpace::LinkToParent;
    const auto rootSpace = EMochiArticulationLinkTransformSpace::LinkToRoot;
    std::mt19937 rng(17);

    // Transforms from joint positions round-trip.
    for (const auto space : {parentSpace, rootSpace}) {
      const TArray<double> jointPositions = RandomJointPositions(rng);
      ExpectSameJointPositions(
          Inverse(Forward(jointPositions, space), space), jointPositions, TEXT("Round trip"));
    }

    // Off-axis transforms lose data: they give the nearest joint positions joint by joint, the
    // transforms of those give them back, negated quaternions change nothing, and the same skeleton
    // in LinkToRoot space gives the same joint positions (an error stays in its own joint).
    for (int trial = 0; trial < 3; ++trial) {
      const TArray<FTransform> linksToParent = RandomOffAxisLinksToParent(rng);
      const TArray<double> projected = Inverse(linksToParent, parentSpace);
      ExpectJointwiseNearest(projected, linksToParent, TEXT("Off axis"));
      ExpectSameJointPositions(
          Inverse(Forward(projected, parentSpace), parentSpace), projected, TEXT("Reprojection"));
      TArray<FTransform> negated = linksToParent;
      for (FTransform& t : negated) {
        const FQuat q = t.GetRotation();
        t.SetRotation(FQuat(-q.X, -q.Y, -q.Z, -q.W));
      }
      ExpectSameJointPositions(
          Inverse(negated, parentSpace), projected, TEXT("Negated quaternions"));
      TArray<FTransform> linksToRoot = linksToParent;
      for (int i = 1; i < kNumLinks; ++i) {
        linksToRoot[i] = linksToParent[i] * linksToRoot[kChain[i].Parent];
      }
      ExpectSameJointPositions(Inverse(linksToRoot, rootSpace), projected, TEXT("LinkToRoot"));
    }

    TArray<double> zero;
    zero.Init(0.0, Articulation.GetNumDofs());
    const int revolute = 1;
    const int spherical = 3;
    const int iRevolute = FirstDof[revolute];

    // A half turn about an axis perpendicular to the revolute axis: no angle is nearer than
    // another, so the angle is 0.
    {
      // LinkToParent = ParentJointFromLink * Motion * ParentLinkFromJoint, with the turn as Motion.
      const FMochiToUnrealConverter m2u(kSuperDexSpace);
      const FVector axis = m2u.DirectionToUnreal(kChain[revolute].Axis).GetSafeNormal();
      const FVector perpendicular =
          FVector::CrossProduct(axis, FVector::ForwardVector).GetSafeNormal();
      TArray<FTransform> targets = Forward(zero, parentSpace);
      targets[revolute] = m2u.TransformToUnreal(ChainParentJointFromLink(revolute)) *
          FTransform(FQuat(perpendicular, UE_DOUBLE_PI)) *
          m2u.TransformToUnreal(ChainParentLinkFromJoint(revolute));
      const TArray<double> projected = Inverse(targets, parentSpace);
      Test.TestEqual(TEXT("Half turn: angle 0"), projected[iRevolute], 0.0);
      ExpectJointwiseNearest(projected, targets, TEXT("Half turn"));
    }

    // Revolute angles past +-pi wrap into [-pi, pi], to the same rotation.
    for (const double angle :
         {UE_DOUBLE_PI - 1e-3,
          -UE_DOUBLE_PI + 1e-3,
          UE_DOUBLE_PI + 1e-2,
          -UE_DOUBLE_PI - 1e-2,
          7.0}) {
      TArray<double> jointPositions = zero;
      jointPositions[iRevolute] = angle;
      const double recovered =
          Inverse(Forward(jointPositions, parentSpace), parentSpace)[iRevolute];
      Test.TestTrue(TEXT("Wrapped angle within pi"), FMath::Abs(recovered) <= UE_DOUBLE_PI + 1e-6);
      Test.TestNearlyEqual(
          FString::Printf(TEXT("Revolute angle %g"), angle),
          FMath::UnwindRadians(recovered - angle),
          0.0,
          kPositionTolerance);
    }

    // Spherical rotations near a half turn: the rotation vector has norm <= pi and is the same
    // rotation.
    for (const double angle : {UE_DOUBLE_PI - 1e-3, UE_DOUBLE_PI + 1e-3}) {
      TArray<double> jointPositions = zero;
      const FVector rotationVector = FVector(0.3, -0.5, 0.8).GetSafeNormal() * angle;
      for (int k = 0; k < 3; ++k) {
        jointPositions[FirstDof[spherical] + k] = rotationVector[k];
      }
      const TArray<FTransform> targets = Forward(jointPositions, parentSpace);
      const TArray<double> recovered = Inverse(targets, parentSpace);
      const FVector recoveredVector(
          recovered[FirstDof[spherical]],
          recovered[FirstDof[spherical] + 1],
          recovered[FirstDof[spherical] + 2]);
      Test.TestTrue(
          TEXT("Rotation vector within pi"), recoveredVector.Size() <= UE_DOUBLE_PI + 1e-6);
      ExpectSameLink(
          Forward(recovered, parentSpace)[spherical],
          targets[spherical],
          FString::Printf(TEXT("Spherical link at %g"), angle));
    }

    // A mismatched transform count is rejected without changing the output.
    const TArray<double> sentinel = {1.0, 2.0, 3.0};
    const TArray<FTransform> rest = Forward(zero, parentSpace);
    TArray<FTransform> shortTransforms = rest;
    shortTransforms.Pop();
    TArray<double> out = sentinel;
    Test.TestFalse(
        TEXT("Wrong transform count rejected"),
        Articulation.ComputeJointPositionsFromLinkTransforms(shortTransforms, out, parentSpace));
    Test.TestEqual(TEXT("Output unchanged"), out, sentinel);
  }
};

} // namespace

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSuperDexBotKinematicsTest,
    "SuperDexRobotics.Kinematics.JointPositionsAndLinkTransforms",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSuperDexBotKinematicsTest::RunTest(const FString& Parameters) {
  TStrongObjectPtr<UMochiArticulation> articulation(MakeArticulation());
  const FKinematicsChecker articulationChecker{*this, *articulation};
  AddInfo(TEXT("UMochiArticulation"));
  articulationChecker.Run();

#if MOCHI_UNREAL_LINUX_EDITOR
  // Mochi structs don't cross into superdex_robotics here: the Linux editor builds Mochi headers
  // without AVX2, so their layout differs from the library's (see SuperDexPhysics.Build.cs).
  AddInfo(TEXT("USuperDexBot skipped in the Linux editor"));
#else
  mochi::Error error;
  const superdex::robotics::BotPrefab prefab = MakeChainPrefab(error);
  if (!TestTrue(
          FString::Printf(TEXT("Chain bot is valid: %hs"), error.GetDescription()), error.IsOK())) {
    return false;
  }
  TStrongObjectPtr<USuperDexBot> bot(MakeBot(prefab));
  if (!TestEqual(TEXT("Bot DOFs"), bot->GetNumDofs(), articulation->GetNumDofs())) {
    return false;
  }
  // USuperDexBot and UMochiArticulation agree, in both directions, on and off axis.
  std::mt19937 rng(23);
  const FKinematicsChecker botChecker{*this, *bot};
  for (const auto space :
       {EMochiArticulationLinkTransformSpace::LinkToParent,
        EMochiArticulationLinkTransformSpace::LinkToRoot}) {
    const TArray<double> jointPositions = botChecker.RandomJointPositions(rng);
    const TArray<FTransform> botLinks = botChecker.Forward(jointPositions, space);
    const TArray<FTransform> articulationLinks = articulationChecker.Forward(jointPositions, space);
    if (!TestEqual(TEXT("Link count"), botLinks.Num(), articulationLinks.Num())) {
      return false;
    }
    for (int i = 0; i < botLinks.Num(); ++i) {
      botChecker.ExpectSameLink(
          botLinks[i], articulationLinks[i], FString::Printf(TEXT("Link %d"), i));
    }
    const TArray<FTransform> targets = botChecker.RandomOffAxisLinksToParent(rng);
    botChecker.ExpectSameJointPositions(
        botChecker.Inverse(targets, EMochiArticulationLinkTransformSpace::LinkToParent),
        articulationChecker.Inverse(targets, EMochiArticulationLinkTransformSpace::LinkToParent),
        TEXT("Off-axis inverse"));
  }
  AddInfo(TEXT("USuperDexBot"));
  botChecker.Run();
#endif
  return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
