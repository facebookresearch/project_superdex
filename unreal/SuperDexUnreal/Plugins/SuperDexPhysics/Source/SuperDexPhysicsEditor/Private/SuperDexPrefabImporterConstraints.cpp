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
    TArray<FString> segments;
    ToFString(Name).ParseIntoArray(segments, TEXT("/"));
    OutActor = {};
    Error = FString::Printf(TEXT("names %s, which the prefab does not have"), *ToFString(Name));
    return !segments.IsEmpty() && FindIn(Prefab, SourcePath, segments, 0, OutActor);
  }

  FString Error;

 private:
  bool FindIn(
      const mochi::prefab::ScenePrefab& In,
      const FString& InSourcePath,
      const TArray<FString>& Segments,
      int Index,
      FConstraintActor& OutActor) {
    const FString& name = Segments[Index];
    const bool bLast = Index == Segments.Num() - 1;
    // Appends the component an actor or nested prefab of In was imported as. Its name is made
    // unique only against the names of the other components, so it must not collide with them.
    auto appendComponent = [&](const FString& Name) {
      const FString component = SuperDexImportUtils::SanitizeForComponentName(Name);
      if (CountComponentsNamed(In, InSourcePath, component) > 1) {
        Error = FString::Printf(
            TEXT("names %s, whose component name %s is not unique in its prefab"),
            *Name,
            *component);
        return false;
      }
      OutActor.Path = OutActor.Path.IsEmpty() ? component : OutActor.Path + TEXT("/") + component;
      return true;
    };
    for (const auto& actor : In.actors.rigid) {
      if (bLast && ToFString(actor.name) == name) {
        OutActor.ShapeToActor = GetShapeToActor(actor, M2U);
        return appendComponent(name);
      }
    }
    for (const auto& actor : In.actors.soft) {
      if (bLast && ToFString(actor.name) == name) {
        OutActor.ShapeToActor = GetShapeToActor(actor, M2U);
        return appendComponent(name);
      }
    }
    for (const auto& actor : In.actors.articulated) {
      if (ToFString(actor.name) != name || Index < Segments.Num() - 2) {
        continue;
      }
      OutActor.Articulated = &actor;
      if (!appendComponent(name)) {
        return false;
      }
      if (bLast) {
        return true;
      }
      // Links keep their names in the articulation asset.
      for (int i = 0; i < static_cast<int>(actor.links.size()); ++i) {
        if (ToFString(actor.links[i].name) == Segments[Index + 1]) {
          OutActor.Link = i;
          OutActor.Path += TEXT("/") + Segments[Index + 1];
          return true;
        }
      }
      return false;
    }
    for (const auto& reference : In.prefabs) {
      // Mochi gives the actors of an unnamed nested prefab no namespace, while the importer names
      // its component after the prefab.
      const bool bNamed = !reference.name.empty();
      if (bNamed && (bLast || ToFString(reference.name) != name)) {
        continue;
      }
      const FString path = ResolvePrefabPath(RootDir, InSourcePath, reference.path);
      const mochi::prefab::ScenePrefab* nested = Load(path);
      const FString pathBefore = OutActor.Path;
      if (nested == nullptr ||
          !appendComponent(bNamed ? name : FSuperDexPrefabImporter::GetPrefabName(path))) {
        return false;
      }
      if (FindIn(*nested, path, Segments, bNamed ? Index + 1 : Index, OutActor)) {
        return true;
      }
      if (bNamed) {
        return false;
      }
      OutActor.Path = pathBefore;
    }
    return false;
  }

  // The number of actors and nested prefabs of In whose components would be named Component.
  int CountComponentsNamed(
      const mochi::prefab::ScenePrefab& In,
      const FString& InSourcePath,
      const FString& Component) {
    int count = 0;
    auto add = [&](const FString& Name) {
      count += SuperDexImportUtils::SanitizeForComponentName(Name) == Component ? 1 : 0;
    };
    for (const auto& actor : In.actors.rigid) {
      add(ToFString(actor.name));
    }
    for (const auto& actor : In.actors.soft) {
      add(ToFString(actor.name));
    }
    for (const auto& actor : In.actors.articulated) {
      add(ToFString(actor.name));
    }
    for (const auto& reference : In.prefabs) {
      add(reference.name.empty() ? FSuperDexPrefabImporter::GetPrefabName(
                                       ResolvePrefabPath(RootDir, InSourcePath, reference.path))
                                 : ToFString(reference.name));
    }
    return count;
  }

  const mochi::prefab::ScenePrefab* Load(const FString& Path) {
    if (const TUniquePtr<mochi::prefab::ScenePrefab>* loaded = Loaded.Find(Path)) {
      return loaded->Get();
    }
    bool bLoaded = false;
    auto prefab = MakeUnique<mochi::prefab::ScenePrefab>(LoadPrefab(Path, bLoaded));
    return bLoaded ? Loaded.Add(Path, MoveTemp(prefab)).Get() : nullptr;
  }

  const mochi::prefab::ScenePrefab& Prefab;
  FString SourcePath;
  FString RootDir;
  const FMochiToUnrealConverter& M2U;
  TMap<FString, TUniquePtr<mochi::prefab::ScenePrefab>> Loaded;
};

} // namespace

void FSuperDexPrefabImporter::AddConstraints(FPrefabContext& Context, FOutput& Output) {
  const mochi::prefab::ConstraintLists& constraints = Context.Data.constraints;
  FConstraintActorFinder finder(Context.Data, Context.SourcePath, Context.RootDir, M2U);
  int numSkipped = 0;
  // Finds the actor Name names, or counts the constraint as skipped.
  auto find = [&](const mochi::DynamicString& Name, FConstraintActor& OutActor) {
    if (finder.Find(Name, OutActor)) {
      return true;
    }
    UE_LOG(
        LogMochiPhysicsEditor,
        Warning,
        TEXT("[%s] A constraint %s; it was skipped."),
        *Context.PrefabName,
        *finder.Error);
    ++numSkipped;
    return false;
  };
  // A position [m] in a prefab actor's frame, in the frame of the component it was imported as
  // [cm]. A rigid or soft component sits at the actor's shape frame, scale included; a link frame
  // carries the articulation's scale, which Mochi applies to constraint positions too.
  auto localPosition = [&](const FConstraintActor& Actor, const mochi::Real3& Position) {
    const FVector position = M2U.TranslationToUnreal(Position);
    return Actor.Articulated ? position * Actor.Articulated->scale
                             : Actor.ShapeToActor.InverseTransformPosition(position);
  };
  auto localRotation = [&](const FConstraintActor& Actor, const mochi::Real3& RotationVector) {
    const FQuat rotation =
        M2U.RotationToUnreal(mochi::Quaternion::FromRotationVector(RotationVector));
    return Actor.ShapeToActor.GetRotation().Inverse() * rotation;
  };
  auto distanceSaturation = [&](mochi::real Saturation) {
    return Saturation < 0 ? -1.0f : static_cast<float>(M2U.MeasureToUnreal(Saturation));
  };
  auto angleSaturation = [](mochi::real Saturation) {
    return Saturation < 0 ? -1.0f : static_cast<float>(FMath::RadiansToDegrees(Saturation));
  };
  // Joint constraints.
  auto addJoint = [&](const auto& C,
                      EMochiArticulationJointConstraintType Type,
                      const TCHAR* Label,
                      FConstraintActor& OutActor) {
    if (!find(C.actorName, OutActor)) {
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
      ++numSkipped;
      return static_cast<UMochiArticulationJointConstraintComponent*>(nullptr);
    }
    const auto& joint = OutActor.Articulated->joints[C.jointIndex];
    const FString jointName = joint.name.empty() ? FString::Printf(TEXT("joint_%d"), C.jointIndex)
                                                 : ToFString(joint.name);
    auto* component = CastChecked<UMochiArticulationJointConstraintComponent>(Output.AddActor(
        UMochiArticulationJointConstraintComponent::StaticClass(),
        FString::Printf(TEXT("%s_%s"), Label, *jointName)));
    component->Type = Type;
    component->Articulation = OutActor.Path;
    component->Joint = FName(jointName);
    component->Stiffness = C.stiffness;
    component->Damping = C.damping;
    component->Saturation = C.saturation;
    return component;
  };
  // A single-DOF value in the joint units of the imported articulation, which carry its scale.
  auto singleDof = [&](const auto& C,
                       const FConstraintActor& Actor,
                       int& OutDof,
                       double& OutSign,
                       double& OutScale) {
    const auto& joint = Actor.Articulated->joints[C.jointIndex];
    OutDof = C.dofIndex;
    OutSign = 1.0;
    ConvertArticulatedDofIndexToUnreal(
        static_cast<EMochiArticulatedJointType>(joint.type), C.dofIndex, M2U, OutDof, OutSign);
    const bool bTranslational = joint.type == mochi::ArticulatedJointType::Prismatic ||
        (joint.type == mochi::ArticulatedJointType::Free && C.dofIndex < 3);
    OutScale = bTranslational ? Actor.Articulated->scale : 1.0;
  };
  for (const auto& c : constraints.articulatedSingleDofTarget) {
    FConstraintActor actor;
    if (auto* component = addJoint(
            c, EMochiArticulationJointConstraintType::Target, TEXT("JointTarget"), actor)) {
      double sign = 1.0;
      double scale = 1.0;
      singleDof(c, actor, component->Dof, sign, scale);
      component->TargetValue = sign * scale * c.targetValue;
    }
  }
  for (const auto& c : constraints.articulatedSingleDofRange) {
    FConstraintActor actor;
    if (auto* component =
            addJoint(c, EMochiArticulationJointConstraintType::Range, TEXT("JointRange"), actor)) {
      double sign = 1.0;
      double scale = 1.0;
      singleDof(c, actor, component->Dof, sign, scale);
      component->MinValue = FMath::Min(sign * c.minValue, sign * c.maxValue) * scale;
      component->MaxValue = FMath::Max(sign * c.minValue, sign * c.maxValue) * scale;
    }
  }
  for (const auto& c : constraints.articulated3dRotationTarget) {
    FConstraintActor actor;
    if (auto* component = addJoint(
            c,
            EMochiArticulationJointConstraintType::RotationTarget,
            TEXT("JointRotationTarget"),
            actor)) {
      component->TargetRotation = M2U.RotationToUnreal(c.target).Rotator();
    }
  }
  for (const auto& c : constraints.articulated3dRotationRange) {
    FConstraintActor actor;
    if (auto* component = addJoint(
            c,
            EMochiArticulationJointConstraintType::RotationRange,
            TEXT("JointRotationRange"),
            actor)) {
      ConvertRotationBoundsToUnreal(
          c.minValues, c.maxValues, M2U, component->MinRotation, component->MaxRotation);
    }
  }

  // Joints between two rigid bodies.
  auto addRigidJoint = [&](const auto& C,
                           EMochiRigidJointConstraintType Type,
                           const TCHAR* Label,
                           FConstraintActor& OutA,
                           FConstraintActor& OutB) {
    if (!find(C.actorNameA, OutA) || !find(C.actorNameB, OutB)) {
      return static_cast<UMochiRigidJointConstraintComponent*>(nullptr);
    }
    auto* component = CastChecked<UMochiRigidJointConstraintComponent>(Output.AddActor(
        UMochiRigidJointConstraintComponent::StaticClass(),
        FString::Printf(
            TEXT("%s_%s_%s"), Label, *ToFString(C.actorNameA), *ToFString(C.actorNameB))));
    component->Type = Type;
    component->BodyA = OutA.Path;
    component->BodyB = OutB.Path;
    // Mochi leaves contact between constrained actors to the contact filter.
    component->bAllowContact = true;
    component->Stiffness = C.stiffness;
    component->Damping = C.damping;
    return component;
  };
  for (const auto& c : constraints.rigidSphericalJoint) {
    FConstraintActor a;
    FConstraintActor b;
    if (auto* component = addRigidJoint(
            c, EMochiRigidJointConstraintType::SphericalJoint, TEXT("SphericalJoint"), a, b)) {
      component->LocalPositionA = localPosition(a, c.localPosA);
      component->LocalPositionB = localPosition(b, c.localPosB);
      component->Saturation = distanceSaturation(c.saturation);
    }
  }
  for (const auto& c : constraints.rigidPrismaticJoint) {
    FConstraintActor a;
    FConstraintActor b;
    if (auto* component = addRigidJoint(
            c, EMochiRigidJointConstraintType::PrismaticJoint, TEXT("PrismaticJoint"), a, b)) {
      // The joint slides along the component's X axis.
      component->SetRelativeRotation(
          FQuat::FindBetweenNormals(
              FVector::XAxisVector, M2U.DirectionToUnreal(c.freeAxis).GetSafeNormal()));
      component->bHasMinLimit = c.min.has_value();
      component->MinLimit = c.min ? M2U.MeasureToUnreal(*c.min) : 0.0;
      component->bHasMaxLimit = c.max.has_value();
      component->MaxLimit = c.max ? M2U.MeasureToUnreal(*c.max) : 0.0;
      component->Saturation = distanceSaturation(c.saturation);
    }
  }
  for (const auto& c : constraints.jointRotationTracking) {
    FConstraintActor a;
    FConstraintActor b;
    if (auto* component = addRigidJoint(
            c, EMochiRigidJointConstraintType::RotationTracking, TEXT("RotationTracking"), a, b)) {
      component->SetRelativeRotation(
          M2U.RotationToUnreal(mochi::Quaternion::FromRotationVector(c.refFrameRotVec)));
      component->Saturation = angleSaturation(c.saturation);
    }
  }
  for (const auto& c : constraints.jointRotationRange) {
    FConstraintActor a;
    FConstraintActor b;
    // Absolute limits bound the relative rotation of the bodies' frames. The components sit at the
    // shape frames, which only preserves it if both are rotated alike.
    if (!c.rangeAroundRest && finder.Find(c.actorNameA, a) && finder.Find(c.actorNameB, b) &&
        !a.ShapeToActor.GetRotation().Equals(b.ShapeToActor.GetRotation())) {
      UE_LOG(
          LogMochiPhysicsEditor,
          Warning,
          TEXT(
              "[%s] A rotation range between %s and %s has absolute limits, which the different "
              "shape rotations of the two actors change; it was skipped."),
          *Context.PrefabName,
          *ToFString(c.actorNameA),
          *ToFString(c.actorNameB));
      ++numSkipped;
      continue;
    }
    if (auto* component = addRigidJoint(
            c, EMochiRigidJointConstraintType::RotationRange, TEXT("RotationRange"), a, b)) {
      component->SetRelativeRotation(
          M2U.RotationToUnreal(mochi::Quaternion::FromRotationVector(c.refFrameRotVec)));
      FVector minAngles;
      FVector maxAngles;
      ConvertRotationBoundsToUnreal(
          {c.angleRangeX[0], c.angleRangeY[0], c.angleRangeZ[0]},
          {c.angleRangeX[1], c.angleRangeY[1], c.angleRangeZ[1]},
          M2U,
          minAngles,
          maxAngles);
      component->MinAngles = FMath::RadiansToDegrees(minAngles);
      component->MaxAngles = FMath::RadiansToDegrees(maxAngles);
      component->bRangeAroundRest = c.rangeAroundRest;
      component->Saturation = angleSaturation(c.saturation);
    }
  }

  // Pivots: the target is the component's own transform.
  auto addPivot = [&](const auto& C,
                      EMochiRigidPivotConstraintType Type,
                      const TCHAR* Label,
                      FConstraintActor& OutActor) {
    if (!find(C.actorName, OutActor)) {
      return static_cast<UMochiRigidPivotConstraintComponent*>(nullptr);
    }
    auto* component = CastChecked<UMochiRigidPivotConstraintComponent>(Output.AddActor(
        UMochiRigidPivotConstraintComponent::StaticClass(),
        FString::Printf(TEXT("%s_%s"), Label, *ToFString(C.actorName))));
    component->Type = Type;
    component->Body = OutActor.Path;
    component->Stiffness = C.stiffness;
    component->Damping = C.damping;
    return component;
  };
  for (const auto& c : constraints.rigidPivotPosition) {
    FConstraintActor actor;
    if (auto* component =
            addPivot(c, EMochiRigidPivotConstraintType::Position, TEXT("PivotPosition"), actor)) {
      component->SetRelativeLocation(M2U.TranslationToUnreal(c.targetPosition));
      component->LocalPosition = localPosition(actor, c.localPosition);
      component->Saturation = distanceSaturation(c.saturation);
    }
  }
  for (const auto& c : constraints.rigidPivotToRigidTarget) {
    FConstraintActor actor;
    if (auto* component = addPivot(
            c, EMochiRigidPivotConstraintType::ToRigidTarget, TEXT("PivotToRigidTarget"), actor)) {
      // The target is the actor's center of mass and orientation; the component's orientation is
      // its shape frame's.
      FTransform target = M2U.TransformToUnreal(c.targetTransform);
      target.SetRotation(target.GetRotation() * actor.ShapeToActor.GetRotation());
      component->SetRelativeTransform(target);
      component->LocalPosition = localPosition(actor, c.localPosition);
      component->Saturation = distanceSaturation(c.saturation);
    }
  }
  for (const auto& c : constraints.rigidPivotRotation) {
    FConstraintActor actor;
    if (auto* component =
            addPivot(c, EMochiRigidPivotConstraintType::Rotation, TEXT("PivotRotation"), actor)) {
      component->SetRelativeRotation(
          M2U.RotationToUnreal(mochi::Quaternion::FromRotationVector(c.targetRotation)));
      component->LocalRotation = localRotation(actor, c.localRotation).Rotator();
      component->Saturation = angleSaturation(c.saturation);
    }
  }

  // Deformable nodes.
  auto addNode = [&](const auto& C,
                     EMochiSoftNodeConstraintType Type,
                     const TCHAR* Label,
                     const FConstraintActor& Actor) {
    auto* component = CastChecked<UMochiSoftNodeConstraintComponent>(Output.AddActor(
        UMochiSoftNodeConstraintComponent::StaticClass(),
        FString::Printf(TEXT("%s_%s"), Label, *Actor.Path.Replace(TEXT("/"), TEXT("_")))));
    component->Type = Type;
    component->Body = Actor.Path;
    component->bAllowContact = true;
    component->Stiffness = C.stiffness;
    component->Damping = C.damping;
    component->Saturation = distanceSaturation(C.saturation);
    return component;
  };
  for (const auto& c : constraints.deformableNodePosition) {
    FConstraintActor actor;
    if (find(c.actorName, actor)) {
      auto* component =
          addNode(c, EMochiSoftNodeConstraintType::NodePosition, TEXT("NodePosition"), actor);
      component->NodeIndex = c.nodeIndex;
      component->SetRelativeLocation(M2U.TranslationToUnreal(c.position));
    }
  }
  for (const auto& c : constraints.deformableNodeToRigid) {
    FConstraintActor deformable;
    FConstraintActor rigid;
    if (find(c.deformableActorName, deformable) && find(c.rigidActorName, rigid)) {
      auto* component =
          addNode(c, EMochiSoftNodeConstraintType::NodeToRigid, TEXT("NodeToRigid"), deformable);
      component->OtherBody = rigid.Path;
      component->NodeSelection =
          c.findClosest ? EMochiSoftNodeSelection::Closest : EMochiSoftNodeSelection::Index;
      component->NodeIndex = FMath::Max(c.deformableNodeIndex, 0);
      component->bFixToDeformablePosition = c.fixToDeformablePos;
      component->RigidLocalPosition = localPosition(rigid, c.rigidLocalPos);
    }
  }
  for (const auto& c : constraints.deformableNodeToDeformableNode) {
    FConstraintActor a;
    FConstraintActor b;
    if (find(c.actorNameA, a) && find(c.actorNameB, b)) {
      auto* component = addNode(c, EMochiSoftNodeConstraintType::NodeToNode, TEXT("NodeToNode"), a);
      component->OtherBody = b.Path;
      component->NodeSelection =
          c.findClosest ? EMochiSoftNodeSelection::Closest : EMochiSoftNodeSelection::Index;
      component->NodeIndex = FMath::Max(c.nodeIndexA, 0);
      component->OtherNodeIndex = FMath::Max(c.nodeIndexB, 0);
    }
  }

  if (numSkipped > 0) {
    UE_LOG(
        LogMochiPhysicsEditor,
        Warning,
        TEXT("[%s] %d constraints were skipped."),
        *Context.PrefabName,
        numSkipped);
  }
}
