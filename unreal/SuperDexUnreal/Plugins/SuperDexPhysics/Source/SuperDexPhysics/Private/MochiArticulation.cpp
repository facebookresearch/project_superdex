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

#include "Engine/StaticMesh.h"
#include "MochiAssets.h"
#include "MochiConversions.h"
#include "MochiModel.h"
#include "MochiPhysicsModule.h"

#include <random>

namespace {
inline FQuat FQuatFromRollPitchYaw(FVector const& RPY) {
  const FQuat Roll(FVector::XAxisVector, RPY.X);
  const FQuat Pitch(FVector::YAxisVector, RPY.Y);
  const FQuat Yaw(FVector::ZAxisVector, RPY.Z);
  return Yaw * Pitch * Roll;
}

template <typename TLink>
FTransform GetModelTransform(const TLink& Link) {
  const FQuat Orientation = FQuatFromRollPitchYaw(Link.ModelOriginRPY);
  return FTransform(Orientation, Link.ModelOriginXYZ);
}

// Motion of a joint in its own frame for the joint positions starting at InOutDofIndex, which is
// advanced past the joint's DOFs. Joints without DOFs in the joint position layout don't move.
FTransform GetJointMotion(
    const FMochiArticulatedJoint& Joint,
    const TArray<double>& JointPositions,
    int& InOutDofIndex,
    double AngleScale,
    double LengthScale) {
  switch (Joint.Type) {
    case EMochiArticulatedJointType::Revolute:
      return FTransform(
          FQuat(Joint.Axis.GetSafeNormal(), JointPositions[InOutDofIndex++] * AngleScale));
    case EMochiArticulatedJointType::Prismatic:
      return FTransform(Joint.Axis.GetSafeNormal() * JointPositions[InOutDofIndex++] * LengthScale);
    case EMochiArticulatedJointType::Spherical: {
      const FVector RotationVector(
          JointPositions[InOutDofIndex],
          JointPositions[InOutDofIndex + 1],
          JointPositions[InOutDofIndex + 2]);
      InOutDofIndex += 3;
      return FTransform(FQuat::MakeFromRotationVector(RotationVector * AngleScale));
    }
    case EMochiArticulatedJointType::Free:
    case EMochiArticulatedJointType::Hard:
    case EMochiArticulatedJointType::Cycle:
      break;
  }
  return FTransform::Identity;
}

// Number of joint positions GetJointMotion reads for a joint.
int GetNumMotionDofs(const FMochiArticulatedJoint& Joint) {
  switch (Joint.Type) {
    case EMochiArticulatedJointType::Revolute:
    case EMochiArticulatedJointType::Prismatic:
      return 1;
    case EMochiArticulatedJointType::Spherical:
      return 3;
    case EMochiArticulatedJointType::Free:
    case EMochiArticulatedJointType::Hard:
    case EMochiArticulatedJointType::Cycle:
      break;
  }
  return 0;
}

// Inverse of GetJointMotion: writes the joint positions nearest to Motion (in the joint's own
// frame) starting at InOutDofIndex, which is advanced past the joint's DOFs. Motion the joint can't
// produce is discarded.
void SetJointPositionsFromMotion(
    const FMochiArticulatedJoint& Joint,
    const FTransform& Motion,
    TArray<double>& InOutJointPositions,
    int& InOutDofIndex,
    double AngleScale,
    double LengthScale) {
  // Below this squared norm of the twist quaternion (a swing within about 2e-5 rad of a half turn
  // off a revolute axis), the revolute angle is undetermined.
  static constexpr double DegenerateTwistSq = 1e-10;
  switch (Joint.Type) {
    case EMochiArticulatedJointType::Revolute: {
      // Twist of the rotation about the axis: the nearest rotation about the axis. Both components
      // vanish when every angle is equally near (a half turn about a perpendicular axis); the angle
      // is then 0 rather than whatever rounding gives.
      const FQuat Q = Motion.GetRotation();
      const double SinHalfAngle = FVector(Q.X, Q.Y, Q.Z) | Joint.Axis.GetSafeNormal();
      const double Angle = FMath::Square(SinHalfAngle) + FMath::Square(Q.W) > DegenerateTwistSq
          ? 2.0 * FMath::Atan2(SinHalfAngle, Q.W)
          : 0.0;
      InOutJointPositions[InOutDofIndex++] = FMath::UnwindRadians(Angle) / AngleScale;
      break;
    }
    case EMochiArticulatedJointType::Prismatic:
      InOutJointPositions[InOutDofIndex++] =
          (Motion.GetTranslation() | Joint.Axis.GetSafeNormal()) / LengthScale;
      break;
    case EMochiArticulatedJointType::Spherical: {
      FQuat Q = Motion.GetRotation();
      if (Q.W < 0.0) {
        Q = FQuat(-Q.X, -Q.Y, -Q.Z, -Q.W); // same rotation, angle <= pi
      }
      const FVector RotationVector = Q.ToRotationVector() / AngleScale;
      InOutJointPositions[InOutDofIndex++] = RotationVector.X;
      InOutJointPositions[InOutDofIndex++] = RotationVector.Y;
      InOutJointPositions[InOutDofIndex++] = RotationVector.Z;
      break;
    }
    case EMochiArticulatedJointType::Free:
    case EMochiArticulatedJointType::Hard:
    case EMochiArticulatedJointType::Cycle:
      break;
  }
}

// Number of entries a joint contributes to the joint position layout. A Free root's DOFs belong to
// the base.
int GetNumLayoutDofs(const FMochiArticulatedJoint& Joint, bool bIsRoot) {
  return bIsRoot && Joint.Type == EMochiArticulatedJointType::Free ? 0 : Joint.GetNumDofs();
}

} // namespace

FMochiArticulatedLink* UMochiArticulation::FindLink(FName LinkName) {
  return Links.FindByPredicate([&](const FMochiArticulatedLink& L) { return L.Name == LinkName; });
}

const FMochiArticulatedLink* UMochiArticulation::FindLink(FName LinkName) const {
  return Links.FindByPredicate([&](const FMochiArticulatedLink& L) { return L.Name == LinkName; });
}

FMochiArticulatedJoint* UMochiArticulation::FindJoint(FName JointName) {
  return Joints.FindByPredicate(
      [&](const FMochiArticulatedJoint& J) { return J.Name == JointName; });
}

const FMochiArticulatedJoint* UMochiArticulation::FindJoint(FName JointName) const {
  return Joints.FindByPredicate(
      [&](const FMochiArticulatedJoint& J) { return J.Name == JointName; });
}

bool UMochiArticulation::ValidateLinkNames() const {
  TSet<FName> Names;
  for (const FMochiArticulatedLink& Link : Links) {
    if (Names.Contains(Link.Name)) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[UMochiArticulation] Detected duplicate link name [%s]!"),
          *Link.Name.ToString());
      return false;
    }
    Names.Add(Link.Name);
  }
  return true;
}

#define READY_WARN(Msg, ...)                    \
  if (bLogWarnings) {                           \
    UE_LOG(                                     \
        LogMochiPhysics,                        \
        Warning,                                \
        TEXT("[UMochiArticulation] (%s) " Msg), \
        *GetName(),                             \
        ##__VA_ARGS__);                         \
  }

#define READY_CHECK(Expr, Msg, ...)                                \
  if (!(Expr)) {                                                   \
    READY_WARN(Msg, ##__VA_ARGS__);                                \
    bReady = false;                                                \
    if (OutWarnings) {                                             \
      OutWarnings->Add(FString::Printf(TEXT(Msg), ##__VA_ARGS__)); \
    }                                                              \
  }

bool UMochiArticulation::IsReady(bool bLogWarnings, TArray<FString>* OutWarnings) const {
  bool bReady = true;
  if (OutWarnings) {
    OutWarnings->Reset();
  }
  READY_CHECK(!Links.IsEmpty(), "Links are empty!");
  READY_CHECK(
      Joints.Num() == Links.Num(), "Num joints [%d] != num links [%d]", Joints.Num(), Links.Num());
  for (int LinkIndex = 0; LinkIndex < Links.Num(); ++LinkIndex) {
    const int Parent = Links[LinkIndex].ParentLink;
    READY_CHECK(
        LinkIndex == 0 ? Parent == INDEX_NONE : Parent >= 0 && Parent < LinkIndex,
        "Link [%s] has invalid parent link index [%d]",
        *Links[LinkIndex].Name.ToString(),
        Parent);
  }
  for (int JointIndex = 1; JointIndex < Joints.Num(); ++JointIndex) {
    READY_CHECK(
        Joints[JointIndex].Type != EMochiArticulatedJointType::Free &&
            Joints[JointIndex].Type != EMochiArticulatedJointType::Cycle,
        "Joint [%s] is a Free or Cycle joint, which is only supported as the root joint or in Cycles",
        *Joints[JointIndex].Name.ToString());
  }
  READY_CHECK(
      DefaultJointPositions.Num() == GetNumDofs(),
      "Num default joint positions [%d] != num DOFs [%d]",
      DefaultJointPositions.Num(),
      GetNumDofs());
  READY_CHECK(SkeletalMesh, "SkeletalMesh is null!");
  READY_CHECK(!bNeedGenerateSkeletalMesh, "SkeletalMesh needs regeneration!");
  return bReady;
}

#undef READY_CHECK
#undef READY_WARN

void UMochiArticulation::SetDefinition(
    const FString& InArticulationName,
    TArray<FMochiArticulatedJoint> InJoints,
    TArray<FMochiArticulatedLink> InLinks,
    TArray<FMochiArticulatedCycleJoint> InCycles,
    const FMochiArticulatedSkin* InSkin) {
  ArticulationName = InArticulationName;
  Joints = MoveTemp(InJoints);
  Links = MoveTemp(InLinks);
  Cycles = MoveTemp(InCycles);
  bHasSkin = InSkin != nullptr;
  Skin = InSkin ? *InSkin : FMochiArticulatedSkin{};
  BuildDerivedData();
  bNeedGenerateSkeletalMesh = true;
}

const FString& UMochiArticulation::GetArticulationName() const {
  return ArticulationName;
}

const TArray<FMochiArticulatedLink>& UMochiArticulation::GetLinks() const {
  return Links;
}

const TArray<FMochiArticulatedJoint>& UMochiArticulation::GetJoints() const {
  return Joints;
}

const TArray<FMochiArticulatedCycleJoint>& UMochiArticulation::GetCycles() const {
  return Cycles;
}

bool UMochiArticulation::HasSkin() const {
  return bHasSkin;
}

const FMochiArticulatedSkin& UMochiArticulation::GetSkin() const {
  return Skin;
}

bool UMochiArticulation::HasSoftLinks() const {
  return SoftLinks.Num() > 0;
}

const TArray<FMochiArticulationSoftLink>& UMochiArticulation::GetSoftLinks() const {
  return SoftLinks;
}

bool UMochiArticulation::GetEnableCollidingLinks() const {
  return bEnableCollidingLinks;
}

const TArray<FMochiArticulationContactOverride>& UMochiArticulation::GetContactOverrides() const {
  return ContactOverrides;
}

void UMochiArticulation::SetContactOverrides(
    TArray<FMochiArticulationContactOverride> InContactOverrides) {
  ContactOverrides = MoveTemp(InContactOverrides);
}

const TArray<FMochiArticulationConstraint>& UMochiArticulation::GetAdditionalConstraints() const {
  return AdditionalConstraints;
}

const TArray<double>& UMochiArticulation::GetDefaultJointPositions() const {
  return DefaultJointPositions;
}

bool UMochiArticulation::DefaultFixedBase() const {
  return !Joints.IsEmpty() && Joints[0].Type != EMochiArticulatedJointType::Free;
}

EMochiArticulatedJointType UMochiArticulation::GetEffectiveRootJointType(bool bFixedBase) const {
  if (Joints.IsEmpty() || Joints[0].Type == EMochiArticulatedJointType::Free ||
      Joints[0].Type == EMochiArticulatedJointType::Hard) {
    return bFixedBase ? EMochiArticulatedJointType::Hard : EMochiArticulatedJointType::Free;
  }
  return Joints[0].Type;
}

int UMochiArticulation::GetLinkIndex(FName LinkName) const {
  return Links.IndexOfByPredicate(
      [&](const FMochiArticulatedLink& L) { return L.Name == LinkName; });
}

bool UMochiArticulation::GetLink(FName LinkName, FMochiArticulatedLink& OutLink) const {
  if (const FMochiArticulatedLink* const Link = FindLink(LinkName)) {
    OutLink = *Link;
    return true;
  }
  return false;
}

int UMochiArticulation::GetJointIndex(FName JointName) const {
  return Joints.IndexOfByPredicate(
      [&](const FMochiArticulatedJoint& J) { return J.Name == JointName; });
}

bool UMochiArticulation::GetJoint(FName JointName, FMochiArticulatedJoint& OutJoint) const {
  if (const FMochiArticulatedJoint* const Joint = FindJoint(JointName)) {
    OutJoint = *Joint;
    return true;
  }
  return false;
}

bool UMochiArticulation::GetLinkJoint(int LinkIndex, FMochiArticulatedJoint& OutJoint) const {
  if (Joints.IsValidIndex(LinkIndex)) {
    OutJoint = Joints[LinkIndex];
    return true;
  }
  return false;
}

TArray<int> UMochiArticulation::GetChildLinkIndices(int LinkIndex) const {
  return ChildLinkIndices.IsValidIndex(LinkIndex) ? ChildLinkIndices[LinkIndex] : TArray<int>{};
}

TArray<int> UMochiArticulation::GetLeafLinkIndices() const {
  TArray<int> Indices;
  for (int i = 0; i < ChildLinkIndices.Num(); ++i) {
    if (ChildLinkIndices[i].Num() == 0) {
      Indices.Add(i);
    }
  }
  return Indices;
}

FMochiArticulatedLink UMochiArticulation::GetLastOrFirstForkingLink() const {
  for (int i = 0; i < ChildLinkIndices.Num(); ++i) {
    if (ChildLinkIndices[i].Num() != 1) {
      return Links[i];
    }
  }
  return {};
}

int UMochiArticulation::GetNumDofs() const {
  return DofIndices.Num();
}

const TArray<int>& UMochiArticulation::GetDofIndices() const {
  return DofIndices;
}

int UMochiArticulation::GetDofIndex(FName JointName) const {
  const int JointIndex = GetJointIndex(JointName);
  if (JointIndex == INDEX_NONE) {
    return INDEX_NONE;
  }
  return DofIndices.Find(JointIndex);
}

bool UMochiArticulation::GetDof(int DofIndex, FMochiArticulatedJoint& OutJoint) const {
  if (DofIndices.IsValidIndex(DofIndex) && Joints.IsValidIndex(DofIndices[DofIndex])) {
    OutJoint = Joints[DofIndices[DofIndex]];
    return true;
  }
  return false;
}

bool UMochiArticulation::GetDofLimits(int DofIndex, double& OutLower, double& OutUpper) const {
  if (!DofIndices.IsValidIndex(DofIndex) || !Joints.IsValidIndex(DofIndices[DofIndex])) {
    return false;
  }
  const int JointIndex = DofIndices[DofIndex];
  const int FirstDof = DofIndices.Find(JointIndex);
  Joints[JointIndex].GetDofLimits(DofIndex - FirstDof, OutLower, OutUpper);
  return true;
}

FTransform UMochiArticulation::GetRestLinkToParent(int LinkIndex) const {
  return RestLinksToParent.IsValidIndex(LinkIndex) ? RestLinksToParent[LinkIndex]
                                                   : FTransform::Identity;
}

FTransform UMochiArticulation::GetRestLinkToRoot(int LinkIndex) const {
  return RestLinksToRoot.IsValidIndex(LinkIndex) ? RestLinksToRoot[LinkIndex]
                                                 : FTransform::Identity;
}

TArray<FName> UMochiArticulation::GetLinkNames() const {
  TArray<FName> Names;
  for (const FMochiArticulatedLink& Link : Links) {
    Names.AddUnique(Link.Name);
  }
  return Names;
}

TArray<FName> UMochiArticulation::GetJointNames() const {
  TArray<FName> Names;
  for (const FMochiArticulatedJoint& Joint : Joints) {
    Names.AddUnique(Joint.Name);
  }
  return Names;
}

USkeletalMesh* UMochiArticulation::GetSkeletalMesh() const {
  return SkeletalMesh;
}

bool UMochiArticulation::ComputeLinkTransformsInternal(
    const TArray<double>& JointPositions,
    TArray<FTransform>& OutLinkTransforms,
    bool bToArticulationRoot,
    double AngleScale,
    double LengthScale,
    int FinalLinkIndex) const {
  if (JointPositions.Num() != GetNumDofs()) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT(
            "[UMochiArticulation] Number of provided JointPositions is not equal to number of DOFs!"));
    return false;
  }
  if (Links.Num() == 0 || Joints.Num() != Links.Num() || RestLinksToParent.Num() != Links.Num()) {
    UE_LOG(LogMochiPhysics, Warning, TEXT("[UMochiArticulation] Articulation data is invalid!!"));
    return false;
  }
  const int FinalIndex = FinalLinkIndex < 0 ? Links.Num() : FinalLinkIndex + 1;
  OutLinkTransforms.SetNum(FinalIndex);
  int DofIndex = 0;
  for (int LinkIndex = 0; LinkIndex < FinalIndex; ++LinkIndex) {
    const FMochiArticulatedLink& Link = Links[LinkIndex];
    const FMochiArticulatedJoint& Joint = Joints[LinkIndex];
    const FTransform Motion =
        GetJointMotion(Joint, JointPositions, DofIndex, AngleScale, LengthScale);
    OutLinkTransforms[LinkIndex] = Link.ParentJointFromLink * Motion * Joint.ParentLinkFromJoint;
    if (bToArticulationRoot && Link.ParentLink != INDEX_NONE) {
      OutLinkTransforms[LinkIndex] =
          OutLinkTransforms[LinkIndex] * OutLinkTransforms[Link.ParentLink];
    }
  }
  return true;
}

bool UMochiArticulation::ComputeJointPositionsInternal(
    const TArray<FTransform>& LinkTransforms,
    TArray<double>& OutJointPositions,
    bool bFromArticulationRoot,
    double AngleScale,
    double LengthScale) const {
  if (Links.Num() == 0 || Joints.Num() != Links.Num() || RestLinksToParent.Num() != Links.Num()) {
    UE_LOG(LogMochiPhysics, Warning, TEXT("[UMochiArticulation] Articulation data is invalid!!"));
    return false;
  }
  if (LinkTransforms.Num() != Links.Num()) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT(
            "[UMochiArticulation] Number of provided LinkTransforms is not equal to number of links!"));
    return false;
  }
  // The motion DOFs must fit the joint position layout.
  int NumMotionDofs = 0;
  for (const FMochiArticulatedJoint& Joint : Joints) {
    NumMotionDofs += GetNumMotionDofs(Joint);
  }
  if (NumMotionDofs > GetNumDofs()) {
    UE_LOG(LogMochiPhysics, Warning, TEXT("[UMochiArticulation] Joint DOFs exceed GetNumDofs()!"));
    return false;
  }
  OutJointPositions.Init(0.0, GetNumDofs());
  int DofIndex = 0;
  for (int LinkIndex = 0; LinkIndex < Links.Num(); ++LinkIndex) {
    const FMochiArticulatedLink& Link = Links[LinkIndex];
    FTransform LinkToParent = LinkTransforms[LinkIndex];
    if (bFromArticulationRoot && Link.ParentLink != INDEX_NONE) {
      LinkToParent = LinkTransforms[LinkIndex] * LinkTransforms[Link.ParentLink].Inverse();
    }
    // LinkToParent = ParentJointFromLink * Motion * ParentLinkFromJoint.
    const FTransform Motion = Link.ParentJointFromLink.Inverse() * LinkToParent *
        Joints[LinkIndex].ParentLinkFromJoint.Inverse();
    SetJointPositionsFromMotion(
        Joints[LinkIndex], Motion, OutJointPositions, DofIndex, AngleScale, LengthScale);
  }
  return true;
}

bool UMochiArticulation::ForwardKinematics(
    FName LinkName,
    const TArray<double>& JointPositions,
    FTransform& OutLinkTransform,
    double AngleScale,
    double LengthScale) const {
  const int LinkIndex = GetLinkIndex(LinkName);
  if (LinkIndex == -1) {
    return false;
  }
  thread_local TArray<FTransform> LinkTransforms;
  if (ComputeLinkToArticulationRootTransforms(
          JointPositions, LinkTransforms, AngleScale, LengthScale, LinkIndex)) {
    OutLinkTransform = LinkTransforms[LinkIndex];
    return true;
  }
  return false;
}

bool UMochiArticulation::ComputeVisualBoundsInternal(
    const TArray<FTransform>& LinksToTransformSpace,
    bool bToArticulationRoot,
    FBoxSphereBounds& OutBounds) const {
  if (LinksToTransformSpace.Num() != Links.Num()) {
    return false;
  }
  TArray<FTransform> LinksToRoot;
  if (!bToArticulationRoot) {
    LinksToRoot.SetNum(Links.Num());
    for (int i = 0; i < Links.Num(); ++i) {
      const int ParentIndex = Links[i].ParentLink;
      if (ParentIndex == INDEX_NONE) {
        LinksToRoot[i] = LinksToTransformSpace[i];
      } else {
        LinksToRoot[i] = LinksToTransformSpace[i] * LinksToRoot[ParentIndex];
      }
    }
  }
  const TArray<FTransform>& LinkToRootTransforms =
      bToArticulationRoot ? LinksToTransformSpace : LinksToRoot;
  FBox CompositeBox(EForceInit::ForceInit);
  for (int i = 0; i < Links.Num(); ++i) {
    if (Links[i].RenderModel) {
      FBoxSphereBounds LinkLocalBounds = Links[i].RenderModel->GetBounds();
      FTransform MeshToRoot = Links[i].GetRenderModelTransform() * LinkToRootTransforms[i];
      FBox TransformedBox = LinkLocalBounds.GetBox().TransformBy(MeshToRoot);
      CompositeBox += TransformedBox;
    }
  }
  OutBounds = FBoxSphereBounds(CompositeBox);
  return true;
}

void UMochiArticulation::ClampJointPositions(TArray<double>& JointPositions) const {
  const int MaxIndex = FMath::Min(JointPositions.Num(), GetNumDofs());
  for (int DofIndex = 0; DofIndex < MaxIndex; ++DofIndex) {
    double Lower = 0;
    double Upper = 0;
    GetDofLimits(DofIndex, Lower, Upper);
    JointPositions[DofIndex] = FMath::Clamp(JointPositions[DofIndex], Lower, Upper);
  }
}

void UMochiArticulation::RandomJointPositions(
    TArray<double>& JointPositions,
    bool NormalDistribution) const {
  // for generating normally distributed random numbers
  std::random_device Rd{};
  std::mt19937 Gen{Rd()};
  std::normal_distribution D{0.0, 1.0}; // zero mean, unit variance

  const int MaxIndex = FMath::Min(JointPositions.Num(), GetNumDofs());
  for (int DofIndex = 0; DofIndex < MaxIndex; ++DofIndex) {
    double Lower = 0;
    double Upper = 0;
    GetDofLimits(DofIndex, Lower, Upper);
    // Unlimited DOFs sample one revolution, or a meter for translations.
    const bool bTranslation =
        Joints[DofIndices[DofIndex]].Type == EMochiArticulatedJointType::Prismatic;
    const double Fallback = bTranslation ? 1.0 : UE_DOUBLE_PI;
    Lower = FMath::IsFinite(Lower) ? Lower : -Fallback;
    Upper = FMath::IsFinite(Upper) ? Upper : +Fallback;
    if (NormalDistribution) {
      JointPositions[DofIndex] = FMath::GetMappedRangeValueClamped(
          TRange<double>{-2.5, 2.5}, TRange<double>{Lower, Upper}, D(Gen));
    } else {
      JointPositions[DofIndex] = FMath::RandRange(Lower, Upper);
    }
  }
}

TArray<mochi::ShapeHandle> UMochiArticulation::GetLinkShapes() const {
  MOCHI_PROFILE_SCOPE();
  if (!IsInGameThread()) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT(
            "[UMochiArticulation] Attempting to load UMochiArticulation %s shape on thread that is not the UE game thread. "
            "It is highly recommended that all shape loading be performed on the UE game thread."),
        *GetName());
  }
  TArray<mochi::ShapeHandle> Shapes;
  for (const FMochiArticulatedLink& Link : Links) {
    Shapes.Add(
        Link.Shape ? Link.Shape->GetMochiShape(Link.ShapeScale, Link.GetShapeTransform())
                   : mochi::ShapeHandle{});
  }
  return Shapes;
}

TArray<mochi::ShapeHandle> UMochiArticulation::GetSoftShapes() const {
  int const NumSoftLinks = SoftLinks.Num();
  if (NumSoftLinks == 0) {
    return {};
  }
  TArray<mochi::ShapeHandle> Result;
  Result.SetNum(NumSoftLinks);
  for (int i = 0; i < NumSoftLinks; ++i) {
    const FMochiArticulationSoftLink& SoftLink = SoftLinks[i];
    const int ParentLinkIndex = GetLinkIndex(SoftLink.ParentName);
    if (SoftLink.Model == nullptr) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[UMochiArticulation] SoftLinks[%d] %s Model is null! Returning invalid shape."),
          i,
          *SoftLink.Name.ToString());
      Result[i] = {};
    } else if (ParentLinkIndex != INDEX_NONE) {
      // Reuse the transform of the parent link's shape.
      const FTransform SoftModelToParentModel = GetModelTransform(SoftLink);
      const FTransform ParentModelToParentLink = Links[ParentLinkIndex].GetShapeTransform();
      const FTransform ParentLinkToRoot = GetRestLinkToRoot(ParentLinkIndex);
      const FTransform ParentLinkToArticulationRoot(
          ParentLinkToRoot.GetRotation(), ParentLinkToRoot.GetTranslation());
      const FTransform SoftModelToArticulationRoot =
          SoftModelToParentModel * ParentModelToParentLink * ParentLinkToArticulationRoot;
      Result[i] = SoftLink.Model->GetMochiShape(SoftLink.ModelScale, SoftModelToArticulationRoot);
    } else {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT(
              "[UMochiArticulation] Failed to find parent link %s for SoftLinks[%d] %s; using identity transform."),
          *SoftLink.ParentName.ToString(),
          i,
          *SoftLink.Name.ToString());
      Result[i] = SoftLink.Model->GetMochiShape(FVector::OneVector, FTransform::Identity);
    }
  }
  return Result;
}

mochi::ShapeHandle UMochiArticulation::GetSkinShape() const {
  if (bHasSkin && Skin.Shape) {
    return Skin.Shape->GetMochiShape(FVector::OneVector, FTransform::Identity);
  }
  return {};
}

mochi::ArticulatedActorParams UMochiArticulation::GetArticulatedActorParams(
    bool bInFixedBase,
    const UMochiContactParams* InContactParams,
    FName InLayer,
    bool bInHasGravity,
    mochi::Span<mochi::ShapeHandle const> InLinkShapes,
    mochi::ShapeHandle InSkinShape,
    const TArray<double>& InitialJointVelocities) const {
  // This dereferences UObjects (e.g. link.ContactParams) and reads UMochiArticulation state, so it
  // MUST run on the UE game thread. Warn (rather than assert) to mirror GetLinkShapes() and catch
  // a future off-thread caller early.
  if (!IsInGameThread()) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT(
            "[UMochiArticulation] GetArticulatedActorParams for %s called on a thread that is not "
            "the UE game thread. This must be called on the UE game thread."),
        *GetName());
  }
  // Caller is expected to provide one ShapeHandle per link; otherwise mochi::Validate would
  // silently treat missing entries as "dummy" links (no volume, no mass). Fail loudly here so
  // a callsite bug (e.g. forgetting to call GetLinkShapes()) is caught immediately.
  check(InLinkShapes.size() == static_cast<size_t>(Links.Num()));
  check(Joints.Num() == Links.Num());
  FUnrealToMochiConverter U2m;
  mochi::ContactParams DefaultContact;
  if (InContactParams) {
    DefaultContact = InContactParams->ToMochi();
  }
  const EMochiArticulatedJointType RootType = GetEffectiveRootJointType(bInFixedBase);
  mochi::ArticulatedActorParams Params;
  Params.joints.reserve(Joints.Num());
  Params.links.reserve(Links.Num());
  for (int i = 0; i < Links.Num(); ++i) {
    mochi::ArticulatedJointParams JointParams = Joints[i].ToMochi(U2m);
    if (i == 0) {
      JointParams.type = ::ToMochi(RootType);
    }
    Params.joints.push_back(std::move(JointParams));
    Params.links.push_back(
        Links[i].ToMochi(U2m, InLinkShapes[i], InLayer, DefaultContact, bInHasGravity));
  }
  for (const FMochiArticulatedCycleJoint& Cycle : Cycles) {
    Params.cycles.push_back(Cycle.ToMochi(U2m));
  }
  if (bHasSkin && InSkinShape.IsValid()) {
    Params.skin = Skin.ToMochi(InSkinShape, InLayer, DefaultContact);
  }
  if (!InitialJointVelocities.IsEmpty()) {
    if (InitialJointVelocities.Num() == GetNumDofs()) {
      TArray<double> Velocities = InitialJointVelocities;
      TArray<EMochiArticulatedJointType> Types;
      for (const FMochiArticulatedJoint& Joint : Joints) {
        Types.Add(Joint.Type);
      }
      ConvertArticulatedDofsToMochi(Types, false, Velocities, U2m);
      // Mochi's velocities also cover a Free root's DOFs, which start at rest.
      mochi::DynamicArray<mochi::real> MochiVelocities;
      if (RootType == EMochiArticulatedJointType::Free) {
        MochiVelocities.resize(GetArticulatedJointNumDofs(EMochiArticulatedJointType::Free), 0);
      }
      MochiVelocities.reserve(MochiVelocities.size() + Velocities.Num());
      for (const double V : Velocities) {
        MochiVelocities.push_back(static_cast<mochi::real>(V));
      }
      Params.jointVelocities = std::move(MochiVelocities);
    } else {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT(
              "[UMochiArticulation] Ignoring %d initial joint velocities for %s, which has %d DOFs."),
          InitialJointVelocities.Num(),
          *GetName(),
          GetNumDofs());
    }
  }
  return Params;
}

void UMochiArticulation::PostLoad() {
  Super::PostLoad();
  BuildDerivedData();
}

FString UMochiArticulation::RemoveAssetTypePrefix(const FString& AssetName, bool* bOutHadPrefix) {
  for (const TCHAR* Prefix : {TEXT("MA_"), TEXT("URDF_")}) {
    if (AssetName.StartsWith(Prefix)) {
      if (bOutHadPrefix) {
        *bOutHadPrefix = true;
      }
      return AssetName.RightChop(FCString::Strlen(Prefix));
    }
  }
  if (bOutHadPrefix) {
    *bOutHadPrefix = false;
  }
  return AssetName;
}

bool UMochiArticulation::PreGenValidateOrUpdateDerivedData() {
  // this function logs errors for us
  if (!ValidateLinkNames()) {
    return false;
  }
  // this function logs errors for us
  if (!BuildDerivedData()) {
    return false;
  }
  Modify();
  return true;
}

bool UMochiArticulation::BuildDerivedData() {
  const int NumLinks = Links.Num();
  auto ResetDerivedData = [&] {
    DofIndices.Reset();
    RestLinksToParent.Init(FTransform::Identity, NumLinks);
    RestLinksToRoot.Init(FTransform::Identity, NumLinks);
    ChildLinkIndices.Init(TArray<int>(), NumLinks);
  };
  ResetDerivedData();
  if (Joints.Num() != NumLinks) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiArticulation] %s has %d joints but %d links!"),
        *GetName(),
        Joints.Num(),
        NumLinks);
    return false;
  }
  for (int LinkIndex = 0; LinkIndex < NumLinks; ++LinkIndex) {
    const FMochiArticulatedLink& Link = Links[LinkIndex];
    const FMochiArticulatedJoint& Joint = Joints[LinkIndex];
    const int ParentIndex = Link.ParentLink;
    if (LinkIndex == 0 ? ParentIndex != INDEX_NONE : ParentIndex < 0 || ParentIndex >= LinkIndex) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[UMochiArticulation] Link %s of %s has invalid parent link index %d!"),
          *Link.Name.ToString(),
          *GetName(),
          ParentIndex);
      ResetDerivedData();
      return false;
    }
    RestLinksToParent[LinkIndex] = Link.ParentJointFromLink * Joint.ParentLinkFromJoint;
    if (ParentIndex == INDEX_NONE) {
      RestLinksToRoot[LinkIndex] = RestLinksToParent[LinkIndex];
    } else {
      RestLinksToRoot[LinkIndex] = RestLinksToParent[LinkIndex] * RestLinksToRoot[ParentIndex];
      ChildLinkIndices[ParentIndex].Add(LinkIndex);
    }
    for (int i = 0; i < GetNumLayoutDofs(Joint, LinkIndex == 0); ++i) {
      DofIndices.Add(LinkIndex);
    }
  }
  if (DefaultJointPositions.Num() != DofIndices.Num()) {
    DefaultJointPositions.SetNumZeroed(DofIndices.Num());
  }
  return true;
}

#if WITH_EDITOR

void UMochiArticulation::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) {
  Super::PostEditChangeProperty(PropertyChangedEvent);
  const FName PropName = PropertyChangedEvent.GetMemberPropertyName();
  const bool bTopologyEdit = PropName == GET_MEMBER_NAME_CHECKED(UMochiArticulation, Links) ||
      PropName == GET_MEMBER_NAME_CHECKED(UMochiArticulation, Joints);
  if (bTopologyEdit) {
    BuildDerivedData();
    bNeedGenerateSkeletalMesh = true;
    Modify(true);
  }
  if (PropName == GET_MEMBER_NAME_CHECKED(UMochiArticulation, DefaultJointPositions)) {
    if (DefaultJointPositions.Num() != GetNumDofs()) {
      DefaultJointPositions.SetNumZeroed(GetNumDofs());
      Modify(true);
    }
  }
}

void UMochiArticulation::PostEditUndo() {
  Super::PostEditUndo();
  BuildDerivedData();
}

TArray<FName> UMochiArticulation::GetLinkNames_EditorOnly() const {
  return GetLinkNames();
}

TArray<FName> UMochiArticulation::GetJointNames_EditorOnly() const {
  return GetJointNames();
}

#endif // WITH_EDITOR
