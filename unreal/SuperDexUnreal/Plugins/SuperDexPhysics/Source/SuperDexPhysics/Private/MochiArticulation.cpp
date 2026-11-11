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
  const FQuat roll(FVector::XAxisVector, RPY.X);
  const FQuat pitch(FVector::YAxisVector, RPY.Y);
  const FQuat yaw(FVector::ZAxisVector, RPY.Z);
  return yaw * pitch * roll;
}

template <typename TLink>
FTransform GetModelTransform(const TLink& Link) {
  const FQuat orientation = FQuatFromRollPitchYaw(Link.ModelOriginRPY);
  return FTransform(orientation, Link.ModelOriginXYZ);
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
      const FVector rotationVector(
          JointPositions[InOutDofIndex],
          JointPositions[InOutDofIndex + 1],
          JointPositions[InOutDofIndex + 2]);
      InOutDofIndex += 3;
      return FTransform(FQuat::MakeFromRotationVector(rotationVector * AngleScale));
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
  static constexpr double kDegenerateTwistSq = 1e-10;
  switch (Joint.Type) {
    case EMochiArticulatedJointType::Revolute: {
      // Twist of the rotation about the axis: the nearest rotation about the axis. Both components
      // vanish when every angle is equally near (a half turn about a perpendicular axis); the angle
      // is then 0 rather than whatever rounding gives.
      const FQuat q = Motion.GetRotation();
      const double sinHalfAngle = FVector(q.X, q.Y, q.Z) | Joint.Axis.GetSafeNormal();
      const double angle = FMath::Square(sinHalfAngle) + FMath::Square(q.W) > kDegenerateTwistSq
          ? 2.0 * FMath::Atan2(sinHalfAngle, q.W)
          : 0.0;
      InOutJointPositions[InOutDofIndex++] = FMath::UnwindRadians(angle) / AngleScale;
      break;
    }
    case EMochiArticulatedJointType::Prismatic:
      InOutJointPositions[InOutDofIndex++] =
          (Motion.GetTranslation() | Joint.Axis.GetSafeNormal()) / LengthScale;
      break;
    case EMochiArticulatedJointType::Spherical: {
      FQuat q = Motion.GetRotation();
      if (q.W < 0.0) {
        q = FQuat(-q.X, -q.Y, -q.Z, -q.W); // same rotation, angle <= pi
      }
      const FVector rotationVector = q.ToRotationVector() / AngleScale;
      InOutJointPositions[InOutDofIndex++] = rotationVector.X;
      InOutJointPositions[InOutDofIndex++] = rotationVector.Y;
      InOutJointPositions[InOutDofIndex++] = rotationVector.Z;
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
  TSet<FName> names;
  for (const auto& link : Links) {
    if (names.Contains(link.Name)) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[UMochiArticulation] Detected duplicate link name [%s]!"),
          *link.Name.ToString());
      return false;
    }
    names.Add(link.Name);
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
  for (int iLink = 0; iLink < Links.Num(); ++iLink) {
    const int parent = Links[iLink].ParentLink;
    READY_CHECK(
        iLink == 0 ? parent == INDEX_NONE : parent >= 0 && parent < iLink,
        "Link [%s] has invalid parent link index [%d]",
        *Links[iLink].Name.ToString(),
        parent);
  }
  for (int iJoint = 1; iJoint < Joints.Num(); ++iJoint) {
    READY_CHECK(
        Joints[iJoint].Type != EMochiArticulatedJointType::Free &&
            Joints[iJoint].Type != EMochiArticulatedJointType::Cycle,
        "Joint [%s] is a Free or Cycle joint, which is only supported as the root joint or in Cycles",
        *Joints[iJoint].Name.ToString());
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
  if (const auto link = FindLink(LinkName)) {
    OutLink = *link;
    return true;
  }
  return false;
}

int UMochiArticulation::GetJointIndex(FName JointName) const {
  return Joints.IndexOfByPredicate(
      [&](const FMochiArticulatedJoint& J) { return J.Name == JointName; });
}

bool UMochiArticulation::GetJoint(FName JointName, FMochiArticulatedJoint& OutJoint) const {
  if (const auto joint = FindJoint(JointName)) {
    OutJoint = *joint;
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
  TArray<int> indices;
  for (int i = 0; i < ChildLinkIndices.Num(); ++i) {
    if (ChildLinkIndices[i].Num() == 0) {
      indices.Add(i);
    }
  }
  return indices;
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
  const int iJoint = GetJointIndex(JointName);
  if (iJoint == INDEX_NONE) {
    return INDEX_NONE;
  }
  return DofIndices.Find(iJoint);
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
  const int iJoint = DofIndices[DofIndex];
  const int firstDof = DofIndices.Find(iJoint);
  Joints[iJoint].GetDofLimits(DofIndex - firstDof, OutLower, OutUpper);
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
  TArray<FName> names;
  for (auto& link : Links) {
    names.AddUnique(link.Name);
  }
  return names;
}

TArray<FName> UMochiArticulation::GetJointNames() const {
  TArray<FName> names;
  for (auto& joint : Joints) {
    names.AddUnique(joint.Name);
  }
  return names;
}

USkeletalMesh* UMochiArticulation::GetSkeletalMesh() const {
  return SkeletalMesh;
}

bool UMochiArticulation::ComputeLinkTransformsFromJointPositions(
    const TArray<double>& JointPositions,
    TArray<FTransform>& OutLinkTransforms,
    EMochiArticulationLinkTransformSpace TransformSpace,
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
  const int iFinal = FinalLinkIndex < 0 ? Links.Num() : FinalLinkIndex + 1;
  OutLinkTransforms.SetNum(iFinal);
  int iDof = 0;
  for (int iLink = 0; iLink < iFinal; ++iLink) {
    const auto& link = Links[iLink];
    const auto& joint = Joints[iLink];
    const FTransform motion = GetJointMotion(joint, JointPositions, iDof, AngleScale, LengthScale);
    OutLinkTransforms[iLink] = link.ParentJointFromLink * motion * joint.ParentLinkFromJoint;
    if (TransformSpace == EMochiArticulationLinkTransformSpace::LinkToRoot &&
        link.ParentLink != INDEX_NONE) {
      OutLinkTransforms[iLink] = OutLinkTransforms[iLink] * OutLinkTransforms[link.ParentLink];
    }
  }
  return true;
}

bool UMochiArticulation::ComputeJointPositionsFromLinkTransforms(
    const TArray<FTransform>& LinkTransforms,
    TArray<double>& OutJointPositions,
    EMochiArticulationLinkTransformSpace TransformSpace,
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
  // The DOFs ComputeLinkTransformsFromJointPositions reads, which must fit the joint position
  // layout.
  int numMotionDofs = 0;
  for (const auto& joint : Joints) {
    numMotionDofs += GetNumMotionDofs(joint);
  }
  if (numMotionDofs > GetNumDofs()) {
    UE_LOG(LogMochiPhysics, Warning, TEXT("[UMochiArticulation] Joint DOFs exceed GetNumDofs()!"));
    return false;
  }
  OutJointPositions.Init(0.0, GetNumDofs());
  int iDof = 0;
  for (int iLink = 0; iLink < Links.Num(); ++iLink) {
    const auto& link = Links[iLink];
    FTransform linkToParent = LinkTransforms[iLink];
    if (TransformSpace == EMochiArticulationLinkTransformSpace::LinkToRoot &&
        link.ParentLink != INDEX_NONE) {
      linkToParent = LinkTransforms[iLink] * LinkTransforms[link.ParentLink].Inverse();
    }
    // ComputeLinkTransformsFromJointPositions: LinkToParent = ParentJointFromLink * Motion *
    // ParentLinkFromJoint.
    const FTransform motion = link.ParentJointFromLink.Inverse() * linkToParent *
        Joints[iLink].ParentLinkFromJoint.Inverse();
    SetJointPositionsFromMotion(
        Joints[iLink], motion, OutJointPositions, iDof, AngleScale, LengthScale);
  }
  return true;
}

bool UMochiArticulation::ForwardKinematics(
    FName LinkName,
    const TArray<double>& JointPositions,
    FTransform& OutLinkTransform,
    double AngleScale,
    double LengthScale) const {
  const int iLink = GetLinkIndex(LinkName);
  if (iLink == -1) {
    return false;
  }
  thread_local TArray<FTransform> sLinkTransforms;
  if (ComputeLinkTransformsFromJointPositions(
          JointPositions,
          sLinkTransforms,
          EMochiArticulationLinkTransformSpace::LinkToRoot,
          AngleScale,
          LengthScale,
          iLink)) {
    OutLinkTransform = sLinkTransforms[iLink];
    return true;
  }
  return false;
}

bool UMochiArticulation::ComputeVisualBoundsFromLinkTransforms(
    const TArray<FTransform>& LinksToTransformSpace,
    EMochiArticulationLinkTransformSpace TransformSpace,
    FBoxSphereBounds& OutBounds) const {
  if (LinksToTransformSpace.Num() != Links.Num()) {
    return false;
  }
  TArray<FTransform> linksToRoot;
  if (TransformSpace == EMochiArticulationLinkTransformSpace::LinkToParent) {
    linksToRoot.SetNum(Links.Num());
    for (int i = 0; i < Links.Num(); ++i) {
      const int iParent = Links[i].ParentLink;
      if (iParent == INDEX_NONE) {
        linksToRoot[i] = LinksToTransformSpace[i];
      } else {
        linksToRoot[i] = LinksToTransformSpace[i] * linksToRoot[iParent];
      }
    }
  }
  const TArray<FTransform>& linkToRootTransforms =
      TransformSpace == EMochiArticulationLinkTransformSpace::LinkToRoot ? LinksToTransformSpace
                                                                         : linksToRoot;
  FBox compositeBox(EForceInit::ForceInit);
  for (int i = 0; i < Links.Num(); ++i) {
    if (Links[i].RenderModel) {
      FBoxSphereBounds linkLocalBounds = Links[i].RenderModel->GetBounds();
      FTransform meshToRoot = Links[i].GetRenderModelTransform() * linkToRootTransforms[i];
      FBox transformedBox = linkLocalBounds.GetBox().TransformBy(meshToRoot);
      compositeBox += transformedBox;
    }
  }
  OutBounds = FBoxSphereBounds(compositeBox);
  return true;
}

void UMochiArticulation::ClampJointPositions(TArray<double>& JointPositions) const {
  const int iMax = FMath::Min(JointPositions.Num(), GetNumDofs());
  for (int iDof = 0; iDof < iMax; ++iDof) {
    double lower = 0;
    double upper = 0;
    GetDofLimits(iDof, lower, upper);
    JointPositions[iDof] = FMath::Clamp(JointPositions[iDof], lower, upper);
  }
}

void UMochiArticulation::RandomJointPositions(
    TArray<double>& JointPositions,
    bool NormalDistribution) const {
  // for generating normally distributed random numbers
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution d{0.0, 1.0}; // zero mean, unit variance

  const int iMax = FMath::Min(JointPositions.Num(), GetNumDofs());
  for (int iDof = 0; iDof < iMax; ++iDof) {
    double lower = 0;
    double upper = 0;
    GetDofLimits(iDof, lower, upper);
    // Unlimited DOFs sample one revolution, or a meter for translations.
    const bool bTranslation =
        Joints[DofIndices[iDof]].Type == EMochiArticulatedJointType::Prismatic;
    const double fallback = bTranslation ? 1.0 : UE_DOUBLE_PI;
    lower = FMath::IsFinite(lower) ? lower : -fallback;
    upper = FMath::IsFinite(upper) ? upper : +fallback;
    if (NormalDistribution) {
      JointPositions[iDof] = FMath::GetMappedRangeValueClamped(
          TRange<double>{-2.5, 2.5}, TRange<double>{lower, upper}, d(gen));
    } else {
      JointPositions[iDof] = FMath::RandRange(lower, upper);
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
  TArray<mochi::ShapeHandle> shapes;
  for (auto& link : Links) {
    shapes.Add(
        link.Shape ? link.Shape->GetMochiShape(link.ShapeScale, link.GetShapeTransform())
                   : mochi::ShapeHandle{});
  }
  return shapes;
}

TArray<mochi::ShapeHandle> UMochiArticulation::GetSoftShapes() const {
  int const numSoftLinks = SoftLinks.Num();
  if (numSoftLinks == 0) {
    return {};
  }
  TArray<mochi::ShapeHandle> result;
  result.SetNum(numSoftLinks);
  for (int i = 0; i < numSoftLinks; ++i) {
    auto& softLink = SoftLinks[i];
    const int parentLinkIndex = GetLinkIndex(softLink.ParentName);
    if (softLink.Model == nullptr) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[UMochiArticulation] SoftLinks[%d] %s Model is null! Returning invalid shape."),
          i,
          *softLink.Name.ToString());
      result[i] = {};
    } else if (parentLinkIndex != INDEX_NONE) {
      // Reuse the transform of the parent link's shape.
      const FTransform softModelToParentModel = GetModelTransform(softLink);
      const FTransform parentModelToParentLink = Links[parentLinkIndex].GetShapeTransform();
      const FTransform parentLinkToRoot = GetRestLinkToRoot(parentLinkIndex);
      const FTransform parentLinkToArticulationRoot(
          parentLinkToRoot.GetRotation(), parentLinkToRoot.GetTranslation());
      const FTransform softModelToArticulationRoot =
          softModelToParentModel * parentModelToParentLink * parentLinkToArticulationRoot;
      result[i] = softLink.Model->GetMochiShape(softLink.ModelScale, softModelToArticulationRoot);
    } else {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT(
              "[UMochiArticulation] Failed to find parent link %s for SoftLinks[%d] %s; using identity transform."),
          *softLink.ParentName.ToString(),
          i,
          *softLink.Name.ToString());
      result[i] = softLink.Model->GetMochiShape(FVector::OneVector, FTransform::Identity);
    }
  }
  return result;
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
  FUnrealToMochiConverter u2m;
  mochi::ContactParams defaultContact;
  if (InContactParams) {
    defaultContact = InContactParams->ToMochi();
  }
  const EMochiArticulatedJointType rootType = GetEffectiveRootJointType(bInFixedBase);
  mochi::ArticulatedActorParams params;
  params.joints.reserve(Joints.Num());
  params.links.reserve(Links.Num());
  for (int i = 0; i < Links.Num(); ++i) {
    mochi::ArticulatedJointParams jointParams = Joints[i].ToMochi(u2m);
    if (i == 0) {
      jointParams.type = static_cast<mochi::ArticulatedJointType>(rootType);
    }
    params.joints.push_back(std::move(jointParams));
    params.links.push_back(
        Links[i].ToMochi(u2m, InLinkShapes[i], InLayer, defaultContact, bInHasGravity));
  }
  for (const FMochiArticulatedCycleJoint& cycle : Cycles) {
    params.cycles.push_back(cycle.ToMochi(u2m));
  }
  if (bHasSkin && InSkinShape.IsValid()) {
    params.skin = Skin.ToMochi(InSkinShape, InLayer, defaultContact);
  }
  if (!InitialJointVelocities.IsEmpty()) {
    if (InitialJointVelocities.Num() == GetNumDofs()) {
      TArray<double> velocities = InitialJointVelocities;
      TArray<EMochiArticulatedJointType> types;
      for (const FMochiArticulatedJoint& joint : Joints) {
        types.Add(joint.Type);
      }
      ConvertArticulatedDofsToMochi(types, false, velocities, u2m);
      // Mochi's velocities also cover a Free root's DOFs, which start at rest.
      mochi::DynamicArray<mochi::real> mochiVelocities;
      if (rootType == EMochiArticulatedJointType::Free) {
        mochiVelocities.resize(GetArticulatedJointNumDofs(EMochiArticulatedJointType::Free), 0);
      }
      mochiVelocities.reserve(mochiVelocities.size() + velocities.Num());
      for (const double v : velocities) {
        mochiVelocities.push_back(static_cast<mochi::real>(v));
      }
      params.jointVelocities = std::move(mochiVelocities);
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
  return params;
}

void UMochiArticulation::PostLoad() {
  Super::PostLoad();
  BuildDerivedData();
}

FString UMochiArticulation::RemoveAssetTypePrefix(const FString& AssetName, bool* bOutHadPrefix) {
  for (const TCHAR* prefix : {TEXT("MA_"), TEXT("URDF_")}) {
    if (AssetName.StartsWith(prefix)) {
      if (bOutHadPrefix) {
        *bOutHadPrefix = true;
      }
      return AssetName.RightChop(FCString::Strlen(prefix));
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
  const int numLinks = Links.Num();
  auto resetDerivedData = [&] {
    DofIndices.Reset();
    RestLinksToParent.Init(FTransform::Identity, numLinks);
    RestLinksToRoot.Init(FTransform::Identity, numLinks);
    ChildLinkIndices.Init(TArray<int>(), numLinks);
  };
  resetDerivedData();
  if (Joints.Num() != numLinks) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiArticulation] %s has %d joints but %d links!"),
        *GetName(),
        Joints.Num(),
        numLinks);
    return false;
  }
  for (int iLink = 0; iLink < numLinks; ++iLink) {
    const FMochiArticulatedLink& link = Links[iLink];
    const FMochiArticulatedJoint& joint = Joints[iLink];
    const int iParent = link.ParentLink;
    if (iLink == 0 ? iParent != INDEX_NONE : iParent < 0 || iParent >= iLink) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT("[UMochiArticulation] Link %s of %s has invalid parent link index %d!"),
          *link.Name.ToString(),
          *GetName(),
          iParent);
      resetDerivedData();
      return false;
    }
    RestLinksToParent[iLink] = link.ParentJointFromLink * joint.ParentLinkFromJoint;
    if (iParent == INDEX_NONE) {
      RestLinksToRoot[iLink] = RestLinksToParent[iLink];
    } else {
      RestLinksToRoot[iLink] = RestLinksToParent[iLink] * RestLinksToRoot[iParent];
      ChildLinkIndices[iParent].Add(iLink);
    }
    for (int i = 0; i < GetNumLayoutDofs(joint, iLink == 0); ++i) {
      DofIndices.Add(iLink);
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
