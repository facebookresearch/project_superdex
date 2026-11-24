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

// Prefab constraints to the SuperDexPhysics constraint components. Each component names its bodies
// by their component path in the generated Blueprint or map, and holds its values in the units and
// frames the component documents: what Mochi holds in a prefab actor's frame moves to the frame of
// the component that actor became, and what it holds in the prefab frame becomes the constraint
// component's own transform.

#include "MochiArticulationJointConstraintComponent.h"
#include "MochiArticulationTypes.h"
#include "MochiPhysicsEditorModule.h"
#include "MochiRigidJointConstraintComponent.h"
#include "MochiRigidPivotConstraintComponent.h"
#include "MochiSoftNodeConstraintComponent.h"
#include "SuperDexImportUtils.h"
#include "SuperDexPrefabImporter.h"
#include "SuperDexPrefabImporterInternal.h"

using namespace SuperDexPrefabImporterInternal;

namespace {

// The prefab actor a constraint names, and the path of the component it was imported as.
struct FConstraintActor {
  const mochi::prefab::ArticulatedActorPrefab* Articulated = nullptr;
  // The link, when the name is "<articulated actor>/<link>".
  int Link = -1;
  // Where a rigid or soft actor's component sits in the actor's frame.
  FTransform ShapeToActor;
  FString Path;
};

// Finds the actors constraints name in a prefab and the prefabs it nests, loading those as needed.
class FConstraintActorFinder {
 public:
  FConstraintActorFinder(
      const mochi::prefab::ScenePrefab& InPrefab,
      const FString& InSourcePath,
      const FString& InRootDir,
      const FMochiToUnrealConverter& InM2U)
      : Prefab(InPrefab), SourcePath(InSourcePath), RootDir(InRootDir), M2U(InM2U) {}

  // Finds the actor Name names, or sets Error.
  bool Find(const mochi::DynamicString& Name, FConstraintActor& OutActor) {
    TArray<FString> Segments;
    ToFString(Name).ParseIntoArray(Segments, TEXT("/"));
    OutActor = {};
    Error = FString::Printf(TEXT("names %s, which the prefab does not have"), *ToFString(Name));
    return !Segments.IsEmpty() && FindIn(Prefab, SourcePath, Segments, 0, OutActor);
  }

  FString Error;

 private:
  bool FindIn(
      const mochi::prefab::ScenePrefab& In,
      const FString& InSourcePath,
      const TArray<FString>& Segments,
      int Index,
      FConstraintActor& OutActor) {
    const FString& SegmentName = Segments[Index];
    const bool bLast = Index == Segments.Num() - 1;
    // Appends the component an actor or nested prefab of In was imported as. Its name is made
    // unique only against the names of the other components, so it must not collide with them.
    auto AppendComponent = [&](const FString& Name) {
      const FString Component = SuperDexImportUtils::SanitizeForComponentName(Name);
      if (CountComponentsNamed(In, InSourcePath, Component) > 1) {
        Error = FString::Printf(
            TEXT("names %s, whose component name %s is not unique in its prefab"),
            *Name,
            *Component);
        return false;
      }
      OutActor.Path = OutActor.Path.IsEmpty() ? Component : OutActor.Path + TEXT("/") + Component;
      return true;
    };
    for (const mochi::prefab::RigidActorPrefab& Actor : In.actors.rigid) {
      if (bLast && ToFString(Actor.name) == SegmentName) {
        OutActor.ShapeToActor = GetShapeToActor(Actor, M2U);
        return AppendComponent(SegmentName);
      }
    }
    for (const mochi::prefab::SoftActorPrefab& Actor : In.actors.soft) {
      if (bLast && ToFString(Actor.name) == SegmentName) {
        OutActor.ShapeToActor = GetShapeToActor(Actor, M2U);
        return AppendComponent(SegmentName);
      }
    }
    for (const mochi::prefab::RodActorPrefab& Actor : In.actors.rod) {
      if (bLast && ToFString(Actor.name) == SegmentName) {
        OutActor.ShapeToActor = GetShapeToActor(Actor, M2U);
        return AppendComponent(SegmentName);
      }
    }
    for (const mochi::prefab::ArticulatedActorPrefab& Actor : In.actors.articulated) {
      if (ToFString(Actor.name) != SegmentName || Index < Segments.Num() - 2) {
        continue;
      }
      OutActor.Articulated = &Actor;
      if (!AppendComponent(SegmentName)) {
        return false;
      }
      if (bLast) {
        return true;
      }
      // Links keep their names in the articulation asset.
      for (int i = 0; i < static_cast<int>(Actor.links.size()); ++i) {
        if (ToFString(Actor.links[i].name) == Segments[Index + 1]) {
          OutActor.Link = i;
          OutActor.Path += TEXT("/") + Segments[Index + 1];
          return true;
        }
      }
      return false;
    }
    for (const mochi::prefab::PrefabReference& Reference : In.prefabs) {
      // Mochi gives the actors of an unnamed nested prefab no namespace, while the importer names
      // its component after the prefab.
      const bool bNamed = !Reference.name.empty();
      if (bNamed && (bLast || ToFString(Reference.name) != SegmentName)) {
        continue;
      }
      const FString Path = ResolvePrefabPath(RootDir, InSourcePath, Reference.path);
      const mochi::prefab::ScenePrefab* Nested = Load(Path);
      const FString PathBefore = OutActor.Path;
      if (Nested == nullptr ||
          !AppendComponent(bNamed ? SegmentName : FSuperDexPrefabImporter::GetPrefabName(Path))) {
        return false;
      }
      if (FindIn(*Nested, Path, Segments, bNamed ? Index + 1 : Index, OutActor)) {
        return true;
      }
      if (bNamed) {
        return false;
      }
      OutActor.Path = PathBefore;
    }
    return false;
  }

  // The number of actors and nested prefabs of In whose components would be named Component.
  int CountComponentsNamed(
      const mochi::prefab::ScenePrefab& In,
      const FString& InSourcePath,
      const FString& Component) {
    int Count = 0;
    auto Add = [&](const FString& Name) {
      Count += SuperDexImportUtils::SanitizeForComponentName(Name) == Component ? 1 : 0;
    };
    for (const mochi::prefab::RigidActorPrefab& Actor : In.actors.rigid) {
      Add(ToFString(Actor.name));
    }
    for (const mochi::prefab::SoftActorPrefab& Actor : In.actors.soft) {
      Add(ToFString(Actor.name));
    }
    for (const mochi::prefab::RodActorPrefab& Actor : In.actors.rod) {
      Add(ToFString(Actor.name));
    }
    for (const mochi::prefab::ArticulatedActorPrefab& Actor : In.actors.articulated) {
      Add(ToFString(Actor.name));
    }
    for (const mochi::prefab::PrefabReference& Reference : In.prefabs) {
      Add(Reference.name.empty() ? FSuperDexPrefabImporter::GetPrefabName(
                                       ResolvePrefabPath(RootDir, InSourcePath, Reference.path))
                                 : ToFString(Reference.name));
    }
    return Count;
  }

  const mochi::prefab::ScenePrefab* Load(const FString& Path) {
    if (const TUniquePtr<mochi::prefab::ScenePrefab>* LoadedPrefab = Loaded.Find(Path)) {
      return LoadedPrefab->Get();
    }
    bool bLoaded = false;
    TUniquePtr<mochi::prefab::ScenePrefab> PrefabData =
        MakeUnique<mochi::prefab::ScenePrefab>(LoadPrefab(Path, bLoaded));
    return bLoaded ? Loaded.Add(Path, MoveTemp(PrefabData)).Get() : nullptr;
  }

  const mochi::prefab::ScenePrefab& Prefab;
  FString SourcePath;
  FString RootDir;
  const FMochiToUnrealConverter& M2U;
  TMap<FString, TUniquePtr<mochi::prefab::ScenePrefab>> Loaded;
};

} // namespace

void FSuperDexPrefabImporter::AddConstraints(FPrefabContext& Context, FOutput& Output) {
  const mochi::prefab::ConstraintLists& Constraints = Context.Data.constraints;
  FConstraintActorFinder Finder(Context.Data, Context.SourcePath, Context.RootDir, M2U);
  int NumSkipped = 0;
  // Finds the actor Name names, or counts the constraint as skipped.
  auto Find = [&](const mochi::DynamicString& Name, FConstraintActor& OutActor) {
    if (Finder.Find(Name, OutActor)) {
      return true;
    }
    UE_LOG(
        LogMochiPhysicsEditor,
        Warning,
        TEXT("[%s] A constraint %s; it was skipped."),
        *Context.PrefabName,
        *Finder.Error);
    ++NumSkipped;
    return false;
  };
  // A position [m] in a prefab actor's frame, in the frame of the component it was imported as
  // [cm]. A rigid, soft or rod component sits at the actor's shape frame, scale included; a link
  // carries the articulation's scale, which Mochi applies to constraint positions too.
  auto LocalPosition = [&](const FConstraintActor& Actor, const mochi::Real3& Position) {
    const FVector UnrealPosition = M2U.TranslationToUnreal(Position);
    return Actor.Articulated ? UnrealPosition * Actor.Articulated->scale
                             : Actor.ShapeToActor.InverseTransformPosition(UnrealPosition);
  };
  auto LocalRotation = [&](const FConstraintActor& Actor, const mochi::Real3& RotationVector) {
    const FQuat Rotation =
        M2U.RotationToUnreal(mochi::Quaternion::FromRotationVector(RotationVector));
    return Actor.ShapeToActor.GetRotation().Inverse() * Rotation;
  };
  auto DistanceSaturation = [&](mochi::real Saturation) {
    return Saturation < 0 ? -1.0f : static_cast<float>(M2U.MeasureToUnreal(Saturation));
  };
  auto AngleSaturation = [](mochi::real Saturation) {
    return Saturation < 0 ? -1.0f : static_cast<float>(FMath::RadiansToDegrees(Saturation));
  };
  // Joint constraints.
  auto AddJoint = [&](const auto& C,
                      EMochiArticulationJointConstraintType Type,
                      const TCHAR* Label,
                      FConstraintActor& OutActor) -> UMochiArticulationJointConstraintComponent* {
    if (!Find(C.actorName, OutActor)) {
      return static_cast<UMochiArticulationJointConstraintComponent*>(nullptr);
    }
    if (!OutActor.Articulated || OutActor.Link >= 0 || C.jointIndex < 0 ||
        C.jointIndex >= static_cast<int>(OutActor.Articulated->joints.size())) {
      UE_LOG(
          LogMochiPhysicsEditor,
          Warning,
          TEXT(
              "[%s] A joint constraint names joint %d of %s, which has no such joint; it was "
              "skipped."),
          *Context.PrefabName,
          C.jointIndex,
          *ToFString(C.actorName));
      ++NumSkipped;
      return static_cast<UMochiArticulationJointConstraintComponent*>(nullptr);
    }
    const mochi::prefab::ArticulatedJointPrefab& Joint = OutActor.Articulated->joints[C.jointIndex];
    const FString JointName = Joint.name.empty() ? FString::Printf(TEXT("joint_%d"), C.jointIndex)
                                                 : ToFString(Joint.name);
    UMochiArticulationJointConstraintComponent* Component =
        CastChecked<UMochiArticulationJointConstraintComponent>(Output.AddActor(
            UMochiArticulationJointConstraintComponent::StaticClass(),
            FString::Printf(TEXT("%s_%s"), Label, *JointName)));
    Component->Type = Type;
    Component->Articulation = OutActor.Path;
    Component->Joint = FName(JointName);
    Component->Stiffness = C.stiffness;
    Component->Damping = C.damping;
    Component->Saturation = C.saturation;
    return Component;
  };
  // A single-DOF value in the joint units of the imported articulation, which carry its scale.
  auto SingleDof = [&](const auto& C,
                       const FConstraintActor& Actor,
                       int& OutDof,
                       double& OutSign,
                       double& OutScale) {
    const mochi::prefab::ArticulatedJointPrefab& Joint = Actor.Articulated->joints[C.jointIndex];
    OutDof = C.dofIndex;
    OutSign = 1.0;
    ConvertArticulatedDofIndexToUnreal(
        static_cast<EMochiArticulatedJointType>(Joint.type), C.dofIndex, M2U, OutDof, OutSign);
    const bool bTranslational = Joint.type == mochi::ArticulatedJointType::Prismatic ||
        (Joint.type == mochi::ArticulatedJointType::Free && C.dofIndex < 3);
    OutScale = bTranslational ? Actor.Articulated->scale : 1.0;
  };
  for (const mochi::prefab::ArticulatedSingleDofTargetConstraintPrefab& C :
       Constraints.articulatedSingleDofTarget) {
    FConstraintActor Actor;
    if (UMochiArticulationJointConstraintComponent* Component = AddJoint(
            C, EMochiArticulationJointConstraintType::Target, TEXT("JointTarget"), Actor)) {
      double Sign = 1.0;
      double Scale = 1.0;
      SingleDof(C, Actor, Component->Dof, Sign, Scale);
      Component->TargetValue = Sign * Scale * C.targetValue;
    }
  }
  for (const mochi::prefab::ArticulatedSingleDofRangeConstraintPrefab& C :
       Constraints.articulatedSingleDofRange) {
    FConstraintActor Actor;
    if (UMochiArticulationJointConstraintComponent* Component =
            AddJoint(C, EMochiArticulationJointConstraintType::Range, TEXT("JointRange"), Actor)) {
      double Sign = 1.0;
      double Scale = 1.0;
      SingleDof(C, Actor, Component->Dof, Sign, Scale);
      Component->MinValue = FMath::Min(Sign * C.minValue, Sign * C.maxValue) * Scale;
      Component->MaxValue = FMath::Max(Sign * C.minValue, Sign * C.maxValue) * Scale;
    }
  }
  for (const mochi::prefab::Articulated3dRotationTargetConstraintPrefab& C :
       Constraints.articulated3dRotationTarget) {
    FConstraintActor Actor;
    if (UMochiArticulationJointConstraintComponent* Component = AddJoint(
            C,
            EMochiArticulationJointConstraintType::RotationTarget,
            TEXT("JointRotationTarget"),
            Actor)) {
      Component->TargetRotation = M2U.RotationToUnreal(C.target).Rotator();
    }
  }
  for (const mochi::prefab::Articulated3dRotationRangeConstraintPrefab& C :
       Constraints.articulated3dRotationRange) {
    FConstraintActor Actor;
    if (UMochiArticulationJointConstraintComponent* Component = AddJoint(
            C,
            EMochiArticulationJointConstraintType::RotationRange,
            TEXT("JointRotationRange"),
            Actor)) {
      ConvertRotationBoundsToUnreal(
          C.minValues, C.maxValues, M2U, Component->MinRotation, Component->MaxRotation);
    }
  }

  // Joints between two rigid bodies.
  auto AddRigidJoint = [&](const auto& C,
                           EMochiRigidJointConstraintType Type,
                           const TCHAR* Label,
                           FConstraintActor& OutA,
                           FConstraintActor& OutB) -> UMochiRigidJointConstraintComponent* {
    if (!Find(C.actorNameA, OutA) || !Find(C.actorNameB, OutB)) {
      return static_cast<UMochiRigidJointConstraintComponent*>(nullptr);
    }
    UMochiRigidJointConstraintComponent* Component =
        CastChecked<UMochiRigidJointConstraintComponent>(Output.AddActor(
            UMochiRigidJointConstraintComponent::StaticClass(),
            FString::Printf(
                TEXT("%s_%s_%s"), Label, *ToFString(C.actorNameA), *ToFString(C.actorNameB))));
    Component->Type = Type;
    Component->BodyA = OutA.Path;
    Component->BodyB = OutB.Path;
    // Mochi leaves contact between constrained actors to the contact filter.
    Component->bAllowContact = true;
    Component->Stiffness = C.stiffness;
    Component->Damping = C.damping;
    return Component;
  };
  for (const mochi::prefab::RigidSphericalJointConstraintPrefab& C :
       Constraints.rigidSphericalJoint) {
    FConstraintActor A;
    FConstraintActor B;
    if (UMochiRigidJointConstraintComponent* Component = AddRigidJoint(
            C, EMochiRigidJointConstraintType::SphericalJoint, TEXT("SphericalJoint"), A, B)) {
      Component->LocalPositionA = LocalPosition(A, C.localPosA);
      Component->LocalPositionB = LocalPosition(B, C.localPosB);
      Component->Saturation = DistanceSaturation(C.saturation);
    }
  }
  for (const mochi::prefab::RigidPrismaticJointConstraintPrefab& C :
       Constraints.rigidPrismaticJoint) {
    FConstraintActor A;
    FConstraintActor B;
    if (UMochiRigidJointConstraintComponent* Component = AddRigidJoint(
            C, EMochiRigidJointConstraintType::PrismaticJoint, TEXT("PrismaticJoint"), A, B)) {
      // The joint slides along the component's X axis.
      Component->SetRelativeRotation(
          FQuat::FindBetweenNormals(
              FVector::XAxisVector, M2U.DirectionToUnreal(C.freeAxis).GetSafeNormal()));
      Component->bHasMinLimit = C.min.has_value();
      Component->MinLimit = C.min ? M2U.MeasureToUnreal(*C.min) : 0.0;
      Component->bHasMaxLimit = C.max.has_value();
      Component->MaxLimit = C.max ? M2U.MeasureToUnreal(*C.max) : 0.0;
      Component->Saturation = DistanceSaturation(C.saturation);
    }
  }
  for (const mochi::prefab::JointRotationTrackingConstraintPrefab& C :
       Constraints.jointRotationTracking) {
    FConstraintActor A;
    FConstraintActor B;
    if (UMochiRigidJointConstraintComponent* Component = AddRigidJoint(
            C, EMochiRigidJointConstraintType::RotationTracking, TEXT("RotationTracking"), A, B)) {
      Component->SetRelativeRotation(
          M2U.RotationToUnreal(mochi::Quaternion::FromRotationVector(C.refFrameRotVec)));
      Component->Saturation = AngleSaturation(C.saturation);
    }
  }
  for (const mochi::prefab::JointRotationRangeConstraintPrefab& C :
       Constraints.jointRotationRange) {
    FConstraintActor A;
    FConstraintActor B;
    // Absolute limits bound the relative rotation of the bodies' frames. The components sit at the
    // shape frames, which only preserves it if both are rotated alike.
    if (!C.rangeAroundRest && Finder.Find(C.actorNameA, A) && Finder.Find(C.actorNameB, B) &&
        !A.ShapeToActor.GetRotation().Equals(B.ShapeToActor.GetRotation())) {
      UE_LOG(
          LogMochiPhysicsEditor,
          Warning,
          TEXT(
              "[%s] A rotation range between %s and %s has absolute limits, which the different "
              "shape rotations of the two actors change; it was skipped."),
          *Context.PrefabName,
          *ToFString(C.actorNameA),
          *ToFString(C.actorNameB));
      ++NumSkipped;
      continue;
    }
    if (UMochiRigidJointConstraintComponent* Component = AddRigidJoint(
            C, EMochiRigidJointConstraintType::RotationRange, TEXT("RotationRange"), A, B)) {
      Component->SetRelativeRotation(
          M2U.RotationToUnreal(mochi::Quaternion::FromRotationVector(C.refFrameRotVec)));
      FVector MinAngles;
      FVector MaxAngles;
      ConvertRotationBoundsToUnreal(
          {C.angleRangeX[0], C.angleRangeY[0], C.angleRangeZ[0]},
          {C.angleRangeX[1], C.angleRangeY[1], C.angleRangeZ[1]},
          M2U,
          MinAngles,
          MaxAngles);
      Component->MinAngles = FMath::RadiansToDegrees(MinAngles);
      Component->MaxAngles = FMath::RadiansToDegrees(MaxAngles);
      Component->bRangeAroundRest = C.rangeAroundRest;
      Component->Saturation = AngleSaturation(C.saturation);
    }
  }

  // Pivots: the target is the component's own transform.
  auto AddPivot = [&](const auto& C,
                      EMochiRigidPivotConstraintType Type,
                      const TCHAR* Label,
                      FConstraintActor& OutActor) -> UMochiRigidPivotConstraintComponent* {
    if (!Find(C.actorName, OutActor)) {
      return static_cast<UMochiRigidPivotConstraintComponent*>(nullptr);
    }
    UMochiRigidPivotConstraintComponent* Component =
        CastChecked<UMochiRigidPivotConstraintComponent>(Output.AddActor(
            UMochiRigidPivotConstraintComponent::StaticClass(),
            FString::Printf(TEXT("%s_%s"), Label, *ToFString(C.actorName))));
    Component->Type = Type;
    Component->Body = OutActor.Path;
    Component->Stiffness = C.stiffness;
    Component->Damping = C.damping;
    return Component;
  };
  for (const mochi::prefab::RigidPivotPositionConstraintPrefab& C :
       Constraints.rigidPivotPosition) {
    FConstraintActor Actor;
    if (UMochiRigidPivotConstraintComponent* Component =
            AddPivot(C, EMochiRigidPivotConstraintType::Position, TEXT("PivotPosition"), Actor)) {
      Component->SetRelativeLocation(M2U.TranslationToUnreal(C.targetPosition));
      Component->LocalPosition = LocalPosition(Actor, C.localPosition);
      Component->Saturation = DistanceSaturation(C.saturation);
    }
  }
  for (const mochi::prefab::RigidPivotToRigidTargetConstraintPrefab& C :
       Constraints.rigidPivotToRigidTarget) {
    FConstraintActor Actor;
    if (UMochiRigidPivotConstraintComponent* Component = AddPivot(
            C, EMochiRigidPivotConstraintType::ToRigidTarget, TEXT("PivotToRigidTarget"), Actor)) {
      // The target is the actor's center of mass and orientation; the component's orientation is
      // its shape frame's.
      FTransform Target = M2U.TransformToUnreal(C.targetTransform);
      Target.SetRotation(Target.GetRotation() * Actor.ShapeToActor.GetRotation());
      Component->SetRelativeTransform(Target);
      Component->LocalPosition = LocalPosition(Actor, C.localPosition);
      Component->Saturation = DistanceSaturation(C.saturation);
    }
  }
  for (const mochi::prefab::RigidPivotRotationConstraintPrefab& C :
       Constraints.rigidPivotRotation) {
    FConstraintActor Actor;
    if (UMochiRigidPivotConstraintComponent* Component =
            AddPivot(C, EMochiRigidPivotConstraintType::Rotation, TEXT("PivotRotation"), Actor)) {
      Component->SetRelativeRotation(
          M2U.RotationToUnreal(mochi::Quaternion::FromRotationVector(C.targetRotation)));
      Component->LocalRotation = LocalRotation(Actor, C.localRotation).Rotator();
      Component->Saturation = AngleSaturation(C.saturation);
    }
  }

  // Deformable nodes.
  auto AddNode = [&](const auto& C,
                     EMochiSoftNodeConstraintType Type,
                     const TCHAR* Label,
                     const FConstraintActor& Actor) -> UMochiSoftNodeConstraintComponent* {
    UMochiSoftNodeConstraintComponent* Component =
        CastChecked<UMochiSoftNodeConstraintComponent>(Output.AddActor(
            UMochiSoftNodeConstraintComponent::StaticClass(),
            FString::Printf(TEXT("%s_%s"), Label, *Actor.Path.Replace(TEXT("/"), TEXT("_")))));
    Component->Type = Type;
    Component->Body = Actor.Path;
    Component->bAllowContact = true;
    Component->Stiffness = C.stiffness;
    Component->Damping = C.damping;
    Component->Saturation = DistanceSaturation(C.saturation);
    return Component;
  };
  for (const mochi::prefab::DeformableNodePositionConstraintPrefab& C :
       Constraints.deformableNodePosition) {
    FConstraintActor Actor;
    if (Find(C.actorName, Actor)) {
      UMochiSoftNodeConstraintComponent* Component =
          AddNode(C, EMochiSoftNodeConstraintType::NodePosition, TEXT("NodePosition"), Actor);
      Component->NodeIndex = C.nodeIndex;
      Component->SetRelativeLocation(M2U.TranslationToUnreal(C.position));
    }
  }
  for (const mochi::prefab::DeformableNodeToRigidConstraintPrefab& C :
       Constraints.deformableNodeToRigid) {
    FConstraintActor Deformable;
    FConstraintActor Rigid;
    if (Find(C.deformableActorName, Deformable) && Find(C.rigidActorName, Rigid)) {
      UMochiSoftNodeConstraintComponent* Component =
          AddNode(C, EMochiSoftNodeConstraintType::NodeToRigid, TEXT("NodeToRigid"), Deformable);
      Component->OtherBody = Rigid.Path;
      Component->NodeSelection =
          C.findClosest ? EMochiSoftNodeSelection::Closest : EMochiSoftNodeSelection::Index;
      Component->NodeIndex = FMath::Max(C.deformableNodeIndex, 0);
      Component->bFixToDeformablePosition = C.fixToDeformablePos;
      Component->RigidLocalPosition = LocalPosition(Rigid, C.rigidLocalPos);
    }
  }
  for (const mochi::prefab::DeformableNodeToDeformableNodeConstraintPrefab& C :
       Constraints.deformableNodeToDeformableNode) {
    FConstraintActor A;
    FConstraintActor B;
    if (Find(C.actorNameA, A) && Find(C.actorNameB, B)) {
      UMochiSoftNodeConstraintComponent* Component =
          AddNode(C, EMochiSoftNodeConstraintType::NodeToNode, TEXT("NodeToNode"), A);
      Component->OtherBody = B.Path;
      Component->NodeSelection =
          C.findClosest ? EMochiSoftNodeSelection::Closest : EMochiSoftNodeSelection::Index;
      Component->NodeIndex = FMath::Max(C.nodeIndexA, 0);
      Component->OtherNodeIndex = FMath::Max(C.nodeIndexB, 0);
    }
  }

  if (NumSkipped > 0) {
    UE_LOG(
        LogMochiPhysicsEditor,
        Warning,
        TEXT("[%s] %d constraints were skipped."),
        *Context.PrefabName,
        NumSkipped);
  }
}
