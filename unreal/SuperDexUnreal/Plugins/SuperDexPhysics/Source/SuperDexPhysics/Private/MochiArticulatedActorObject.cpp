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

#include "MochiArticulatedActorObject.h"

#include "MochiArticulationUtil.h"
#include "MochiAssets.h"
#include "MochiConversions.h"
#include "MochiGameInstanceSubsystem.h"
#include "MochiPhysicsModule.h"

#include "MochiSceneUtils.h"

#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/profile.h>
#include <cstddef>

namespace MochiArticulatedActorObjectInternal {

inline void ApplyContactOverrides(
    mochi::Scene* Scene,
    const TArray<mochi::ActorHandle>& LinkActorHandles,
    const UMochiArticulation* Articulation,
    mochi::ErrorLog& E) {
  MOCHI_ERROR_RETURN(E);
  for (const FMochiArticulationContactOverride& Contact : Articulation->GetContactOverrides()) {
    const int AIndex = Articulation->GetLinkIndex(Contact.LinkAName);
    const int BIndex = Articulation->GetLinkIndex(Contact.LinkBName);
    if (AIndex == -1 || BIndex == -1) {
      continue;
    }
    const mochi::ActorHandle LinkA = LinkActorHandles[AIndex];
    const mochi::ActorHandle LinkB = LinkActorHandles[BIndex];
    Scene->EnableActorContactSymmetric(
        LinkA, LinkB, Contact.bEnable, mochi::IncludeNestedActors::No, E);
    if (!E.IsOK()) {
      UE_LOG(
          LogTemp,
          Error,
          TEXT("[UMochiArticulatedActorObject] Failed to set link contact : %hs"),
          E._error.GetDescription());
    }
  }
}

inline void CreateAdditionalConstraints(
    mochi::Scene* Scene,
    const TArray<mochi::ActorHandle>& LinkActorHandles,
    const UMochiArticulation* Articulation,
    mochi::ErrorLog& E) {
  MOCHI_ERROR_RETURN(E);
  FUnrealToMochiConverter U2m;
  const TArray<FMochiArticulationConstraint>& AdditionalConstraints =
      Articulation->GetAdditionalConstraints();
  for (const FMochiArticulationConstraint& Constraint : AdditionalConstraints) {
    const int AIndex = Articulation->GetLinkIndex(Constraint.LinkAName);
    const int BIndex = Articulation->GetLinkIndex(Constraint.LinkBName);
    if (AIndex == -1 || BIndex == -1) {
      continue;
    }
    const mochi::ActorHandle LinkA = LinkActorHandles[AIndex];
    const mochi::ActorHandle LinkB = LinkActorHandles[BIndex];
    const mochi::Real3 PosA = U2m.TranslationToMochi(Constraint.LinkAPosition);
    const mochi::Real3 PosB = U2m.TranslationToMochi(Constraint.LinkBPosition);
    mochi::RigidSphericalJointConstraintParams Params;
    Params.actorA = LinkA;
    Params.actorB = LinkB;
    Params.localPosA = PosA;
    Params.localPosB = PosB;
    Params.stiffness = Constraint.Stiffness;
    Params.damping = Constraint.Damping;
    Params.saturation = Constraint.Saturation;
    Scene->CreateRigidSphericalJointConstraint(Params, E);
    Scene->EnableActorContactSymmetric(
        LinkA, LinkB, /*enable*/ false, mochi::IncludeNestedActors::No, E);
    if (!E.IsOK()) {
      UE_LOG(
          LogTemp,
          Error,
          TEXT("[UMochiArticulatedActorObject] Failed to create additional constraint : %hs"),
          E._error.GetDescription());
    }
  }
}

inline mochi::CallbackHandle RegisterGetPoseCallback(
    mochi::Scene* Scene,
    mochi::Actor* ArticulatedActor,
    FOnMochiArticulatedActorGetPose GetPose) {
  return Scene->RegisterPostStepCallback(
      "UMochiArticulatedActorObject::PostStepCallback",
      [Scene, ArticulatedActor, GetPose](mochi::StepInfo const&) {
        mochi::ErrorLog E{};
        thread_local TArray<mochi::TransformRT> LinkTransforms;
        thread_local TArray<mochi::real> JointPositions;
        const int NumLinks = ArticulatedActor->GetNestedLinkActors(E).size();
        LinkTransforms.SetNum(NumLinks);
        const int NumDofs = ArticulatedActor->GetNumDofs();
        JointPositions.SetNum(NumDofs);
        ArticulatedActor->GetArticulatedLinkTransforms(ToMochiSpan(LinkTransforms), E);
        ArticulatedActor->GetArticulatedPose(ToMochiSpan(JointPositions), E);
        GetPose(LinkTransforms, JointPositions, ArticulatedActor, Scene);
        MOCHI_ERROR_RETURN(E);
      });
}

static bool SetupController(
    mochi::Scene* Scene,
    mochi::Actor* ArticulatedActor,
    bool bAddPoseController,
    const FMochiPoseControllerGains& PoseControllerGains,
    FOnMochiArticulatedActorControl SetPoseTarget,
    const std::vector<mochi::TransformRT>& InitialWorldFromLink,
    mochi::CallbackHandle& OutPreStepCallbackHandle,
    mochi::QueryHandle& OutArticulatedControllerForceQueryHandle,
    mochi::ErrorLog& E) {
  MOCHI_ERROR_RETURN(E, false);
  if (bAddPoseController) {
    const size_t NumLinks = ArticulatedActor->GetNestedLinkActors(E).size();
    const int NumPosGains = PoseControllerGains.PositionTrackingGains.Num();
    const int NumRotGains = PoseControllerGains.RotationTrackingGains.Num();
    const int NumJointGains = PoseControllerGains.JointTrackingGains.Num();
    check(NumPosGains == 0 || NumPosGains == NumLinks);
    check(NumRotGains == 0 || NumRotGains == NumLinks);
    check(NumJointGains == 0 || NumJointGains == NumLinks);
    mochi::PoseControllerParams PoseControllerParams;
    PoseControllerParams.linkPosTracking = ToMochiSpan(PoseControllerGains.PositionTrackingGains);
    PoseControllerParams.linkRotTracking = ToMochiSpan(PoseControllerGains.RotationTrackingGains);
    PoseControllerParams.jointTracking = ToMochiSpan(PoseControllerGains.JointTrackingGains);

    ArticulatedActor->AddArticulatedPoseController(PoseControllerParams, E);
    MOCHI_ERROR_RETURN(E, false);
    const bool bPoseControllerCreated = ArticulatedActor->HasArticulatedPoseController(E);
    MOCHI_ERROR_RETURN(E, false)
    if (!bPoseControllerCreated) {
      return false;
    }
    if (!OutArticulatedControllerForceQueryHandle.IsValid()) {
      OutArticulatedControllerForceQueryHandle =
          ArticulatedActor->RegisterQuery(mochi::QueryType::ArticulatedControllerForce, E);
    }
    MOCHI_ERROR_RETURN(E, false);
    // init controller targets
    ArticulatedActor->ResetArticulatedTargetLinkTransforms(InitialWorldFromLink, E);
  }
  // register pre step pose target callback
  if (SetPoseTarget) {
    OutPreStepCallbackHandle = Scene->RegisterPreStepCallback(
        "UMochiArticulatedActorObject::PreStepCallback",
        [Scene, ArticulatedActor, SetPoseTarget, bAddPoseController](mochi::StepInfo const& Info) {
          mochi::ErrorLog E{};
          // get target transforms from user
          FMochiArticulatedActorPoseVariant Pose;
          if (SetPoseTarget(Pose, ArticulatedActor, Scene, Info.timeStepSec)) {
            if (Pose.IsType<TArray<mochi::TransformRT>>() && bAddPoseController) {
              TArray<mochi::TransformRT>& LinkTransforms = Pose.Get<TArray<mochi::TransformRT>>();
              const int NumLinks = ArticulatedActor->GetNestedLinkActors(E).size();
              if (LinkTransforms.Num() == NumLinks) {
                ArticulatedActor->SetArticulatedTargetLinkTransforms(
                    ToMochiSpan(LinkTransforms), E);
              } else {
                UE_LOG(
                    LogMochiPhysics,
                    Error,
                    TEXT(
                        "[UMochiArticulatedActorObject] Failed to set articulated target link transforms due to num links mismatch. Expected %d, received %d!"),
                    NumLinks,
                    LinkTransforms.Num())
              }
            } else if (Pose.IsType<TArray<mochi::real>>() && bAddPoseController) {
              TArray<mochi::real>& JointPositions = Pose.Get<TArray<mochi::real>>();
              const int NumDofs = ArticulatedActor->GetNumDofs();
              if (JointPositions.Num() == NumDofs) {
                ArticulatedActor->SetArticulatedTargetPose(ToMochiSpan(JointPositions), E);
              } else {
                UE_LOG(
                    LogMochiPhysics,
                    Error,
                    TEXT(
                        "[UMochiArticulatedActorObject] Failed to set articulated target joint positions due "
                        "to num DOFs mismatch. Expected %d, received %d!"),
                    NumDofs,
                    JointPositions.Num())
              }
            } else if (bAddPoseController) {
              UE_LOG(
                  LogMochiPhysics,
                  Error,
                  TEXT(
                      "[UMochiArticulatedActorObject] Failed to set articulated pose target because the pose "
                      "variant does not contain a TArray<mochi::TransformRT> or a TArray<mochi::real>!"));
            }
          }
          MOCHI_ERROR_RETURN(E);
        });
  }
  return true;
}

} // namespace MochiArticulatedActorObjectInternal
using namespace MochiArticulatedActorObjectInternal;

UMochiArticulatedActorObject* UMochiArticulatedActorObject::CreateArticulation(
    UObject* Outer,
    FString ArticulationName,
    const UMochiArticulation* Articulation,
    const FTransform& WorldFromRoot,
    const TArray<FTransform>& RootFromLinks,
    const UMochiContactParams* ContactParams,
    FName Layer,
    bool bFixedBase,
    bool bHasGravity,
    const TArray<double>& InitialJointVelocities,
    const FOnMochiArticulatedActorGetPose& GetPoseCallback,
    bool bAddPoseController,
    const FMochiPoseControllerGains& PoseControllerGains,
    const FOnMochiArticulatedActorControl& ControlCallback,
    const TArray<FName>& LinksThatQueryContactPoints,
    bool bSkinQueriesContactPoints,
    UMochiSceneObject* Scene) {
  MOCHI_PROFILE_SCOPE();
  if (!VerifyOuterAndScene(Outer, Scene, TEXT("UMochiArticulatedActorObject"))) {
    return nullptr;
  }
  if (!VerifySceneAcceptsNewActors(Scene, TEXT("UMochiArticulatedActorObject"), ArticulationName)) {
    return nullptr;
  }
  // get context pointers
  mochi::Context* MochiContext = IMochiPhysicsModule::Get()->GetMochiContext();
  FUnrealToMochiConverter U2m;
  // checks
  check(Scene);
  check(MochiContext);
  check(Articulation);
  // confirm we have links
  const int NumLinks = Articulation->GetLinks().Num();
  if (NumLinks == 0) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "[UMochiArticulatedActorObject] Failed to create UMochiArticulation because params does not contain links!"))
    return nullptr;
  }
  // build per-link shapes; soft shapes are still loaded separately for soft-skinned actors
  const TArray<mochi::ShapeHandle> LinkShapes = Articulation->GetLinkShapes();
  const TArray<mochi::ShapeHandle> SoftShapes = Articulation->GetSoftShapes();
  const mochi::ShapeHandle SkinShape = Articulation->GetSkinShape();
  // data we will pull off the mochi thread
  FActorSetupData SetupData;
  const int NumSoftLinks = Articulation->GetSoftLinks().Num();
  bool bHasArticulatedPoseController = false;
  mochi::ErrorLog E{};
  // Build the shared articulated-actor params (topology + per-link/per-joint properties) on
  // the UE game thread, BEFORE queuing the scene command. The helper dereferences UObject
  // pointers (e.g. link.ContactParams) and reads UMochiArticulation state, both of which
  // require game-thread access; matches the legacy code which also called shape/param
  // construction outside Scene->QueueOrExecuteCommand.
  const FString NameStr = ArticulationName + "___Articulation";
  const FString LayerStr = Layer.ToString();
  mochi::ArticulatedActorParams SkeletonParams = Articulation->GetArticulatedActorParams(
      bFixedBase,
      ContactParams,
      Layer,
      bHasGravity,
      ToMochiSpan(LinkShapes),
      SkinShape,
      InitialJointVelocities);
  SkeletonParams.name = TCHAR_TO_ANSI(*NameStr);
  SkeletonParams.worldFromRoot = U2m.TransformToMochi_RT(WorldFromRoot);
  TArray<mochi::SoftMaterialParams> SoftMaterials;
  for (const FMochiArticulationSoftLink& SoftLink : Articulation->GetSoftLinks()) {
    SoftMaterials.Add(
        SoftLink.MaterialParams ? SoftLink.MaterialParams->ToMochi() : mochi::SoftMaterialParams{});
  }
  const bool bEnableCollidingLinks = Articulation->GetEnableCollidingLinks();
  // queue command to create actor
  Scene->QueueOrExecuteCommand([&](mochi::Scene* SyncScene) {
    mochi::Actor* ArticulatedActor = nullptr;
    // we create a different type of actor depending on presence of soft links
    if (NumSoftLinks == 0) {
      ArticulatedActor = SyncScene->CreateArticulatedActor(SkeletonParams, E);
      MOCHI_ERROR_RETURN(E);
    } else {
      mochi::SoftSkinnedActorParams SoftSkinnedParams;
      SoftSkinnedParams.skeletonParams = std::move(SkeletonParams);
      // create soft params
      mochi::SoftActorParams SoftParamsCommon;
      SoftParamsCommon.layer = TCHAR_TO_ANSI(*LayerStr);
      // Mochi takes the gravity of soft-skinned actors from SoftSkinnedActorParams::hasGravity.
      SoftParamsCommon.hasGravity = false;
      TArray<mochi::SoftActorParams> SoftParams;
      SoftParams.Init(SoftParamsCommon, NumSoftLinks);
      for (int i = 0; i < NumSoftLinks; ++i) {
        SoftParams[i].shape = SoftShapes[i];
        SoftParams[i].name = TCHAR_TO_ANSI(*Articulation->GetSoftLinks()[i].Name.ToString());
        SoftParams[i].material = SoftMaterials[i];
      }
      SoftSkinnedParams.softParams = ToMochiSpan(SoftParams);
      // initialize soft attachments
      TArray<std::string> SoftAttachLinks;
      SoftAttachLinks.SetNum(NumSoftLinks);
      for (int i = 0; i < NumSoftLinks; ++i) {
        SoftAttachLinks[i] = TCHAR_TO_ANSI(*Articulation->GetSoftLinks()[i].ParentName.ToString());
      }
      SoftSkinnedParams.softAttachLinks = ToMochiSpan(SoftAttachLinks);
      SoftSkinnedParams.enableCollidingLinks = bEnableCollidingLinks;
      // create articulated actor
      ArticulatedActor = SyncScene->CreateSoftSkinnedActor(SoftSkinnedParams, E);
    }
    if (ArticulatedActor == nullptr) {
      MOCHI_ERROR_SET(E, "Failed to create articulated actor!");
      MOCHI_ERROR_RETURN(E);
    }
    // shared setup: extract handles/names, register contact queries and callbacks
    SetupActorHandlesAndCallbacks(
        SyncScene,
        ArticulatedActor,
        Outer,
        GetPoseCallback,
        {},
        LinksThatQueryContactPoints,
        bSkinQueriesContactPoints,
        SetupData,
        E);
    MOCHI_ERROR_RETURN(E);
    check(SetupData.LinkActorHandles.Num() == NumLinks);
    check(SetupData.SoftActorHandles.Num() == NumSoftLinks);
    // UMochiArticulation-specific: link contact overrides
    if (Articulation->GetContactOverrides().Num() > 0) {
      ApplyContactOverrides(SyncScene, SetupData.LinkActorHandles, Articulation, E);
      MOCHI_ERROR_RETURN(E);
    }
    // UMochiArticulation-specific: additional constraints
    if (Articulation->GetAdditionalConstraints().Num() > 0) {
      CreateAdditionalConstraints(SyncScene, SetupData.LinkActorHandles, Articulation, E);
      MOCHI_ERROR_RETURN(E);
    }
    // init pose
    const mochi::TransformSRT WorldFromRootMochi = U2m.TransformToMochi_SRT(WorldFromRoot);
    std::vector<mochi::TransformRT> WorldFromLinks(RootFromLinks.Num());
    for (int LinkIndex = 0; LinkIndex < RootFromLinks.Num(); ++LinkIndex) {
      const mochi::TransformRT RootFromLinkRT = U2m.TransformToMochi_RT(RootFromLinks[LinkIndex]);
      const mochi::TransformSRT WorldFromLinkSRT = WorldFromRootMochi * RootFromLinkRT;
      WorldFromLinks[LinkIndex] = WorldFromLinkSRT.GetTransformRT();
    }
    ArticulatedActor->SetArticulatedPoseFromLinks(WorldFromLinks, E);
    MOCHI_ERROR_RETURN(E);
    // create pose controller (UMochiArticulation-specific)
    if (bAddPoseController || ControlCallback) {
      SetupController(
          SyncScene,
          ArticulatedActor,
          bAddPoseController,
          PoseControllerGains,
          ControlCallback,
          WorldFromLinks,
          SetupData.PreStepCallback,
          SetupData.ArticulatedControllerForceQueryHandle,
          E);
      MOCHI_ERROR_RETURN(E);
    }
    bHasArticulatedPoseController = ArticulatedActor->HasArticulatedPoseController(E);
    MOCHI_ERROR_RETURN(E);
  });
  // wait for creation to finish
  Scene->WaitForQueuedCommands();
  if (!E.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "[UMochiArticulatedActorObject] Failed to create UMochiArticulatedActorObject because an error was encountered: %hs!"),
        E._error.GetDescription())
    return nullptr;
  }
  SetupData.bHasPoseController = bHasArticulatedPoseController;
  return CreateFromSetupData(Outer, Scene, SetupData);
}

UMochiArticulatedActorObject* UMochiArticulatedActorObject::CreateFromExistingArticulatedActor(
    UObject* Outer,
    FString Name,
    mochi::ActorHandle ArticulatedActorHandle,
    const FOnMochiArticulatedActorGetPose& GetPoseCallback,
    const FOnMochiArticulatedActorControl& ControlCallback,
    const TArray<FName>& LinksThatQueryContactPoints,
    bool bSkinQueriesContactPoints,
    UMochiSceneObject* Scene) {
  MOCHI_PROFILE_SCOPE();
  if (!VerifyOuterAndScene(Outer, Scene, TEXT("UMochiArticulatedActorObject"))) {
    return nullptr;
  }
  if (!ArticulatedActorHandle.IsValid()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "[UMochiArticulatedActorObject] CreateFromExistingArticulatedActor: invalid actor handle!"))
    return nullptr;
  }
  FActorSetupData SetupData;
  mochi::ErrorLog E{};
  Scene->QueueOrExecuteCommand([&](mochi::Scene* SyncScene) {
    mochi::Actor* ArticulatedActor = SyncScene->GetActor(ArticulatedActorHandle);
    if (!ArticulatedActor) {
      MOCHI_ERROR_SET(E, "CreateFromExistingArticulatedActor: actor not found in scene");
      return;
    }
    SetupActorHandlesAndCallbacks(
        SyncScene,
        ArticulatedActor,
        Outer,
        GetPoseCallback,
        ControlCallback,
        LinksThatQueryContactPoints,
        bSkinQueriesContactPoints,
        SetupData,
        E);
  });
  Scene->WaitForQueuedCommands();
  if (!E.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "[UMochiArticulatedActorObject] Failed to create UMochiArticulatedActorObject from existing actor: %hs!"),
        E._error.GetDescription())
    return nullptr;
  }
  return CreateFromSetupData(Outer, Scene, SetupData);
}

void UMochiArticulatedActorObject::SetupActorHandlesAndCallbacks(
    mochi::Scene* SyncScene,
    mochi::Actor* ArticulatedActor,
    UObject* Outer,
    const FOnMochiArticulatedActorGetPose& GetPoseCallback,
    const FOnMochiArticulatedActorControl& ControlCallback,
    const TArray<FName>& LinksThatQueryContactPoints,
    bool bSkinQueriesContactPoints,
    FActorSetupData& OutData,
    mochi::ErrorLog& E) {
  MOCHI_ERROR_RETURN(E);
  OutData.ActorName = ArticulatedActor->GetName();
  ArticulatedActor->SetUserData(Outer);
  OutData.ActorHandle = ArticulatedActor->GetHandle();
  OutData.NumDofs = ArticulatedActor->GetNumDofs();
  // strip the articulated actor name path from nested actor names
  const FString ToStripFromName = OutData.ActorName + TEXT("/");
  auto GetNestedName = [&SyncScene, &ToStripFromName](mochi::ActorHandle ActorHandle) {
    mochi::Actor* Actor = SyncScene->GetActor(ActorHandle);
    FString NestedActorName = Actor->GetName();
    NestedActorName.ReplaceInline(*ToStripFromName, TEXT(""));
    return FName(NestedActorName);
  };
  // extract link actors
  const mochi::Span<mochi::ActorHandle const> LinkActors = ArticulatedActor->GetNestedLinkActors(E);
  MOCHI_ERROR_RETURN(E);
  const int NumLinks = LinkActors.size();
  OutData.LinkActorHandles.Init({}, NumLinks);
  OutData.LinkActorNames.Init(NAME_None, NumLinks);
  for (int i = 0; i < NumLinks; ++i) {
    OutData.LinkActorHandles[i] = LinkActors[i];
    OutData.LinkActorNames[i] = GetNestedName(LinkActors[i]);
    SyncScene->GetActor(LinkActors[i])->SetUserData(Outer);
  }
  // extract soft actors
  const mochi::Span<mochi::ActorHandle const> SoftActors = ArticulatedActor->GetNestedSoftActors(E);
  const int NumSoftLinks = SoftActors.size();
  if (NumSoftLinks > 0) {
    OutData.SoftActorHandles.Init({}, NumSoftLinks);
    OutData.SoftActorNames.Init(NAME_None, NumSoftLinks);
    for (int i = 0; i < NumSoftLinks; ++i) {
      OutData.SoftActorHandles[i] = SoftActors[i];
      OutData.SoftActorNames[i] = GetNestedName(SoftActors[i]);
      SyncScene->GetActor(SoftActors[i])->SetUserData(Outer);
    }
  }
  // register contact queries
  if (LinksThatQueryContactPoints.Num() > 0) {
    auto AddContactableActors =
        [&LinksThatQueryContactPoints, &SyncScene, &OutData, &GetNestedName, &E](
            const TArray<mochi::ActorHandle>& ActorHandles) {
          for (int HandleIndex = 0; HandleIndex < ActorHandles.Num(); ++HandleIndex) {
            const mochi::ActorHandle Handle = ActorHandles[HandleIndex];
            const FName Name = GetNestedName(Handle);
            if (LinksThatQueryContactPoints.Contains(Name)) {
              mochi::Actor* Actor = SyncScene->GetActor(Handle);
              // A link the skin covers is not a colliding actor and carries no contact samples, so
              // registering the query on it would fail and abort setup for the whole actor.
              if (!Actor->IsStatic() && Actor->IsQuerySupported(mochi::QueryType::ContactPoints)) {
                Actor->RegisterQuery(mochi::QueryType::ContactPoints, E);
                OutData.ContactableActorHandles.Add({Name, Handle});
              }
            }
          }
        };
    AddContactableActors(OutData.LinkActorHandles);
    AddContactableActors(OutData.SoftActorHandles);
    MOCHI_ERROR_RETURN(E);
  }
  // The skin has no actor of its own -- its contact samples live on the top-level articulated
  // actor -- so it is registered separately and reported under that actor's name.
  if (bSkinQueriesContactPoints) {
    if (ArticulatedActor->IsQuerySupported(mochi::QueryType::ContactPoints)) {
      ArticulatedActor->RegisterQuery(mochi::QueryType::ContactPoints, E);
      MOCHI_ERROR_RETURN(E);
      OutData.ContactableActorHandles.Add({FName(OutData.ActorName), OutData.ActorHandle});
    } else {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT(
              "[UMochiArticulatedActorObject] Skin contact queries were requested on '%s', which has no colliding skin; no skin contacts will be reported."),
          *OutData.ActorName);
    }
  }
  // register GetPose callback
  if (GetPoseCallback) {
    OutData.PostStepCallback =
        RegisterGetPoseCallback(SyncScene, ArticulatedActor, GetPoseCallback);
  }
  // register Control callback (no pose controller)
  if (ControlCallback) {
    SetupController(
        SyncScene,
        ArticulatedActor,
        false,
        {},
        ControlCallback,
        {},
        OutData.PreStepCallback,
        OutData.ArticulatedControllerForceQueryHandle,
        E);
    MOCHI_ERROR_RETURN(E);
  }
}

UMochiArticulatedActorObject* UMochiArticulatedActorObject::CreateFromSetupData(
    UObject* Outer,
    UMochiSceneObject* Scene,
    const FActorSetupData& Data) {
  UMochiArticulatedActorObject* MochiActorObj =
      NewObject<UMochiArticulatedActorObject>(Outer, FName(*Data.ActorName));
  MochiActorObj->MochiSceneObject = Scene;
  MochiActorObj->MochiActorHandle = Data.ActorHandle;
  MochiActorObj->LinkActorHandles = Data.LinkActorHandles;
  MochiActorObj->LinkActorNames = Data.LinkActorNames;
  MochiActorObj->SoftActorHandles = Data.SoftActorHandles;
  MochiActorObj->SoftActorNames = Data.SoftActorNames;
  MochiActorObj->PreStepCallbackHandle = Data.PreStepCallback;
  MochiActorObj->PostStepCallbackHandle = Data.PostStepCallback;
  MochiActorObj->NumDofs = Data.NumDofs;
  MochiActorObj->bHasArticulatedPoseController = Data.bHasPoseController;
  MochiActorObj->bQueryContactPoints = Data.ContactableActorHandles.Num() > 0;
  MochiActorObj->ContactableActorHandles = Data.ContactableActorHandles;
  MochiActorObj->ArticulatedControllerForceQueryHandle = Data.ArticulatedControllerForceQueryHandle;
  MochiActorObj->ExtractQueryDataCallbackHandle = Scene->RegisterPostStepCallback(
      "UMochiArticulatedActorObject::ExtractQueryData_Internal",
      [MochiActorObj](mochi::StepInfo const& Info) {
        mochi::Actor* Actor = Info.scene->GetActor(MochiActorObj->MochiActorHandle);
        MochiActorObj->ExtractQueryData_Internal(Actor, Info.scene);
      });
  return MochiActorObj;
}

bool UMochiArticulatedActorObject::IsValid() const {
  return MochiSceneObject && MochiSceneObject->IsValid() && MochiActorHandle.IsValid();
}

UMochiSceneObject* UMochiArticulatedActorObject::GetMochiScene() const {
  return MochiSceneObject;
}

int UMochiArticulatedActorObject::GetNumDofs() const {
  return NumDofs;
}

bool UMochiArticulatedActorObject::HasPoseController() const {
  return bHasArticulatedPoseController;
}

void UMochiArticulatedActorObject::AddPoseController(
    const FMochiPoseControllerGains& PoseControllerGains,
    bool bBlocking) {
  MOCHI_PROFILE_SCOPE();
  if (!IsValid()) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("[UMochiArticulatedActorObject] UMochiArticulatedActorObject is invalid!"))
    return;
  }
  const int NumLinks = LinkActorHandles.Num();
  const int NumPosGains = PoseControllerGains.PositionTrackingGains.Num();
  const int NumRotGains = PoseControllerGains.RotationTrackingGains.Num();
  const int NumJointGains = PoseControllerGains.JointTrackingGains.Num();
  check(NumPosGains == 0 || NumPosGains == NumLinks);
  check(NumRotGains == 0 || NumRotGains == NumLinks);
  check(NumJointGains == 0 || NumJointGains == NumLinks);
  mochi::PoseControllerParams PoseControllerParams;
  PoseControllerParams.linkPosTracking = ToMochiSpan(PoseControllerGains.PositionTrackingGains);
  PoseControllerParams.linkRotTracking = ToMochiSpan(PoseControllerGains.RotationTrackingGains);
  PoseControllerParams.jointTracking = ToMochiSpan(PoseControllerGains.JointTrackingGains);
  MochiSceneObject->QueueOrExecuteActorCommand(
      MochiActorHandle, [this, PoseControllerParams](mochi::Actor* ArticulatedActor) {
        mochi::ErrorLog E;
        bHasArticulatedPoseController = ArticulatedActor->HasArticulatedPoseController(E);
        MOCHI_ERROR_RETURN(E);
        if (bHasArticulatedPoseController) {
          ArticulatedActor->RemoveArticulatedPoseController(E);
          MOCHI_ERROR_RETURN(E);
          bHasArticulatedPoseController = ArticulatedActor->HasArticulatedPoseController(E);
          MOCHI_ERROR_RETURN(E);
          if (bHasArticulatedPoseController) {
            UE_LOG(
                LogMochiPhysics,
                Warning,
                TEXT("[UMochiArticulatedActorObject] Failed to remove existing pose controller!"));
            return;
          }
          if (ArticulatedControllerForceQueryHandle.IsValid()) {
            ArticulatedActor->CancelQuery(ArticulatedControllerForceQueryHandle);
            ArticulatedControllerForceQueryHandle = {};
          }
          MOCHI_ERROR_RETURN(E);
          UE_LOG(
              LogMochiPhysics,
              Display,
              TEXT(
                  "[UMochiArticulatedActorObject] Removed existing pose controller when adding a new one."));
          MOCHI_ERROR_RETURN(E);
        }
        ArticulatedActor->AddArticulatedPoseController(PoseControllerParams, E);
        MOCHI_ERROR_RETURN(E);
        bHasArticulatedPoseController = ArticulatedActor->HasArticulatedPoseController(E);
        MOCHI_ERROR_RETURN(E);
        if (bHasArticulatedPoseController) {
          if (!ArticulatedControllerForceQueryHandle.IsValid()) {
            ArticulatedControllerForceQueryHandle =
                ArticulatedActor->RegisterQuery(mochi::QueryType::ArticulatedControllerForce, E);
          }
          MOCHI_ERROR_RETURN(E);
        } else {
          UE_LOG(
              LogMochiPhysics,
              Warning,
              TEXT("[UMochiArticulatedActorObject] Failed to add pose controller!"))
        }
      });
  if (bBlocking) {
    MochiSceneObject->WaitForQueuedCommands();
  }
}

void UMochiArticulatedActorObject::RemovePoseController(bool bBlocking) {
  MOCHI_PROFILE_SCOPE();
  if (!IsValid()) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("[UMochiArticulatedActorObject] UMochiArticulatedActorObject is invalid!"))
    return;
  }
  MochiSceneObject->QueueOrExecuteActorCommand(MochiActorHandle, [this](mochi::Actor* ArticulatedActor) {
    mochi::ErrorLog E;
    bHasArticulatedPoseController = ArticulatedActor->HasArticulatedPoseController(E);
    MOCHI_ERROR_RETURN(E);
    if (bHasArticulatedPoseController) {
      ArticulatedActor->RemoveArticulatedPoseController(E);
      MOCHI_ERROR_RETURN(E);
      bHasArticulatedPoseController = ArticulatedActor->HasArticulatedPoseController(E);
      if (bHasArticulatedPoseController) {
        UE_LOG(
            LogMochiPhysics,
            Warning,
            TEXT("[UMochiArticulatedActorObject] Failed to remove pose controller!"));
        return;
      }
      if (ArticulatedControllerForceQueryHandle.IsValid()) {
        ArticulatedActor->CancelQuery(ArticulatedControllerForceQueryHandle);
        ArticulatedControllerForceQueryHandle = {};
      }
      MOCHI_ERROR_RETURN(E);
    } else {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT(
              "[UMochiArticulatedActorObject] Articulated actor has no pose controller to remove!"));
      MOCHI_ERROR_RETURN(E);
    }
  });
  if (bBlocking) {
    MochiSceneObject->WaitForQueuedCommands();
  }
}

mochi::ActorHandle UMochiArticulatedActorObject::GetArticulatedActorHandle() const {
  return MochiActorHandle;
}

const TArray<mochi::ActorHandle>& UMochiArticulatedActorObject::GetNestedLinkActorHandles() const {
  return LinkActorHandles;
}

const TArray<FName>& UMochiArticulatedActorObject::GetNestedLinkActorNames() const {
  return LinkActorNames;
}

int UMochiArticulatedActorObject::GetNestedLinkActorIndex(FName Name) const {
  return LinkActorNames.Find(Name);
}

const TArray<mochi::ActorHandle>& UMochiArticulatedActorObject::GetNestedSoftActorHandles() const {
  return SoftActorHandles;
}

const TArray<FName>& UMochiArticulatedActorObject::GetNestedSoftActorNames() const {
  return SoftActorNames;
}

int UMochiArticulatedActorObject::GetNestedSoftActorIndex(FName Name) const {
  return SoftActorNames.Find(Name);
}

void UMochiArticulatedActorObject::TryGetQueryData(
    TFunctionRef<void(const FMochiArticulatedActorQueryData& QueryData)> Callback) {
  MOCHI_PROFILE_SCOPE();
  if (bHasPendingQueryData.load()) {
    // Sanity check
    const bool bWasIsUpdatingUnrealQueryData = bIsUpdatingUnrealQueryData.exchange(true);
    if (bWasIsUpdatingUnrealQueryData) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT(
              "[UMochiArticulatedActorObject] TryGetQueryData only supports single-threaded access!"))
      return;
    }
    MOCHI_DEFER(bIsUpdatingUnrealQueryData.store(false));

    // Synchronize
    bool bHasNewData = false;
    {
      FScopeLock Lock(&PendingQueryDataLock);
      if (bHasPendingQueryData) {
        std::swap(PendingQueryData, UnrealThreadQueryData);
        bHasPendingQueryData = false;
        bHasNewData = true;
      }
    }

    if (bHasNewData) {
      // Fix up data here on the Unreal (render) thread. We don't do this part on the Mochi
      // simulation thread because it requires access to the Unreal components and actors.
      if (UnrealThreadQueryData->ContactPoints.Num() > 0) {
        UObject* const Outer = GetOuter();
        for (FMochiContactPoint& Contact : UnrealThreadQueryData->ContactPoints) {
          Contact.MochiActor =
              Contact.MochiComponent ? Contact.MochiComponent->GetOwner() : nullptr;
          Contact.OtherMochiActor =
              Contact.OtherMochiComponent ? Contact.OtherMochiComponent->GetOwner() : nullptr;
          // There's no guarantee that OtherMochiActor is registered.
          // ensure(contact.MochiActor && contact.OtherMochiActor);
          constexpr bool bVerboseContactWarnings = false;
          if (bVerboseContactWarnings) {
            if (!Contact.MochiActor || !Contact.OtherMochiActor || !Contact.MochiComponent ||
                !Contact.MochiComponent) {
              FString AA = Contact.MochiActor ? *Contact.MochiActor->GetName() : TEXT("NULL");
              FString CA =
                  Contact.MochiComponent ? *Contact.MochiComponent->GetName() : TEXT("NULL");
              FString MA = Contact.InternalMochiActor.ToString();
              FString AB =
                  Contact.OtherMochiActor ? *Contact.OtherMochiActor->GetName() : TEXT("NULL");
              FString CB = Contact.OtherMochiComponent ? *Contact.OtherMochiComponent->GetName()
                                                       : TEXT("NULL");
              FString MB = Contact.OtherInternalMochiActor.ToString();
              UE_LOG(
                  LogMochiPhysics,
                  Warning,
                  TEXT(
                      "[UMochiArticulatedActorObject] Untraceable contact detected: "
                      "AActorA=%s : USceneComponentA=%s : MochiActorA=%s  <---> AActorB=%s : USceneComponentB=%s : MochiActorB=%s"),
                  *AA,
                  *CA,
                  *MA,
                  *AB,
                  *CB,
                  *MB);
              continue;
            }
            if (Contact.MochiComponent && Contact.MochiComponent == Contact.OtherMochiComponent) {
              UE_LOG(
                  LogMochiPhysics,
                  Warning,
                  TEXT("[UMochiArticulatedActorObject] Self-contact detected on component %s"),
                  *Contact.MochiComponent->GetName());
            }
          }
          if (Contact.MochiComponent != Outer) {
            // Swap the direction of this contact (by convention)
            ensure(Contact.OtherMochiComponent == Outer);
            std::swap(Contact.InternalMochiActor, Contact.OtherInternalMochiActor);
            std::swap(Contact.MochiComponent, Contact.OtherMochiComponent);
            std::swap(Contact.MochiActor, Contact.OtherMochiActor);
            std::swap(Contact.Location, Contact.OtherLocation);
            std::swap(Contact.Velocity, Contact.OtherVelocity);
            Contact.Normal = -Contact.Normal;
            Contact.Force = -Contact.Force;

            // Unambiguously record that this is a swapped (other->this) contact,
            // for clients that need to process contacts in an asymmetric manner
            Contact.ContactDirection = EMochiContactDirection::OtherContactsThis;
          }
        }
      }

      // Finally, pass the data back to the caller
      Callback(*UnrealThreadQueryData);
    }
  }
}

// TODO: Only the root link's transform is replaced; the other links keep their current
// transforms, so this re-poses the articulation rather than teleporting it. To teleport it
// rigidly, apply the root link's change in transform to every link (or have Mochi support setting
// an articulation's root transform directly).
void UMochiArticulatedActorObject::SetRootTransform(
    const FTransform& WorldFromLocal,
    bool bBlocking) const {
  MOCHI_PROFILE_SCOPE();
  if (!IsValid()) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("[UMochiArticulatedActorObject] UMochiArticulatedActorObject is invalid!"))
    return;
  }
  MochiSceneObject->QueueOrExecuteCommand([WorldFromLocal, this](mochi::Scene* Scene) {
    FUnrealToMochiConverter U2m;
    mochi::Actor* ArticulatedActor = Scene->GetActor(MochiActorHandle);
    if (ArticulatedActor == nullptr) {
      return; // actor no longer exists; UE component must be shutting down
    }
    mochi::ErrorLog E{};
    const int NumLinks = ArticulatedActor->GetNestedLinkActors(E).size();
    MOCHI_ERROR_RETURN(E);
    std::vector<mochi::TransformRT> Transforms(NumLinks);
    ArticulatedActor->GetArticulatedLinkTransforms(Transforms, E);
    MOCHI_ERROR_RETURN(E);
    Transforms[0] = U2m.TransformToMochi_RT(WorldFromLocal);
    TeleportArticulatedActor(ArticulatedActor, Transforms, E);
  });
  if (bBlocking) {
    MochiSceneObject->WaitForQueuedCommands();
  }
}

// TODO: This function is untested with robots that do not have a fixed base. It is possible that it
// does not work as intended in this case. Intent would be to leave the base link in place, but move
// all other links.
void UMochiArticulatedActorObject::SetArticulatedPoseFromJoints(
    TArray<double> const& JointPositions,
    bool bBlocking) const {
  MOCHI_PROFILE_SCOPE();
  if (!IsValid()) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("[UMochiArticulatedActorObject] UMochiArticulatedActorObject is invalid!"))
    return;
  }
  if (JointPositions.Num() != NumDofs) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT(
            "[UMochiArticulatedActorObject] JointPositions array size does not match number of joints!"))
    return;
  }
  MochiSceneObject->QueueOrExecuteCommand([JointPositions, this](mochi::Scene* Scene) {
    mochi::Actor* ArticulatedActor = Scene->GetActor(MochiActorHandle);
    if (ArticulatedActor == nullptr) {
      return; // actor no longer exists; UE component must be shutting down
    }
    const int ActorNumDofs = ArticulatedActor->GetNumDofs();
    std::vector<mochi::real> NewJointPose(ActorNumDofs, 0);
    for (int i = 0; i < JointPositions.Num(); ++i) {
      NewJointPose[i] = static_cast<mochi::real>(JointPositions[i]);
    }
    mochi::ErrorLog E{};
    ArticulatedActor->SetArticulatedPoseFromJoints(NewJointPose, E);
    MOCHI_ERROR_RETURN(E);

    std::vector<mochi::real> ZeroVel(ActorNumDofs, 0);
    ArticulatedActor->SetArticulatedJointVelocities(ZeroVel, E);
    MOCHI_ERROR_RETURN(E);
  });
  if (bBlocking) {
    MochiSceneObject->WaitForQueuedCommands();
  }
}

TArray<UMochiContactParams*> UMochiArticulatedActorObject::GetLinkContactParams() const {
  MOCHI_PROFILE_SCOPE();
  TArray<UMochiContactParams*> ContactParams;
  if (!IsValid()) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("[UMochiArticulatedActorObject] UMochiArticulatedActorObject is invalid!"))
    return ContactParams;
  }
  ContactParams.SetNum(LinkActorHandles.Num());
  for (int i = 0; i < LinkActorHandles.Num(); ++i) {
    MochiSceneObject->QueueOrExecuteActorCommand(
        LinkActorHandles[i], [&ContactParams, i](mochi::Actor* Actor) {
          mochi::ErrorLog E{};
          const mochi::ContactParams MochiContactParams = Actor->GetContactParams(E);
          if (E.IsOK()) {
            // Convert mochi params to UE params
            UMochiContactParams* UeContactParams =
                NewObject<UMochiContactParams>(GetTransientPackage());
            UeContactParams->FromMochi(MochiContactParams);
            ContactParams[i] = UeContactParams;
          } else {
            // Link doesn't have contact params (dummy link, static actor, etc.)
            ContactParams[i] = nullptr;
          }
        });
  }
  MochiSceneObject->WaitForQueuedCommands();
  return ContactParams;
}

void UMochiArticulatedActorObject::SetLinkContactParams(
    int LinkIndex,
    const UMochiContactParams* ContactParams,
    bool bBlocking) const {
  MOCHI_PROFILE_SCOPE();
  if (!IsValid()) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("[UMochiArticulatedActorObject] UMochiArticulatedActorObject is invalid!"))
    return;
  }
  if (LinkIndex < 0 || LinkIndex >= LinkActorHandles.Num()) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("[UMochiArticulatedActorObject] LinkIndex %d is out of bounds (num links: %d)!"),
        LinkIndex,
        LinkActorHandles.Num())
    return;
  }
  if (!ContactParams) {
    return;
  }
  const mochi::ContactParams MochiContactParams = ContactParams->ToMochi();
  MochiSceneObject->QueueOrExecuteActorCommand(
      LinkActorHandles[LinkIndex], [MochiContactParams](mochi::Actor* Actor) {
        mochi::ErrorLog E{};
        Actor->SetContactParams(MochiContactParams, E);
      });
  if (bBlocking) {
    MochiSceneObject->WaitForQueuedCommands();
  }
}

TArray<float> UMochiArticulatedActorObject::GetLinkDensities() const {
  MOCHI_PROFILE_SCOPE();
  TArray<float> Densities;
  if (!IsValid()) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("[UMochiArticulatedActorObject] UMochiArticulatedActorObject is invalid!"))
    return Densities;
  }
  Densities.SetNum(LinkActorHandles.Num());
  for (int i = 0; i < LinkActorHandles.Num(); ++i) {
    MochiSceneObject->QueueOrExecuteActorCommand(
        LinkActorHandles[i], [&Densities, i](mochi::Actor* Actor) {
          mochi::ErrorLog E{};
          Densities[i] = static_cast<float>(Actor->GetDensity(E));
        });
  }
  MochiSceneObject->WaitForQueuedCommands();
  return Densities;
}

void UMochiArticulatedActorObject::SetLinkDensity(int LinkIndex, float Density, bool bBlocking)
    const {
  MOCHI_PROFILE_SCOPE();
  if (!IsValid()) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("[UMochiArticulatedActorObject] UMochiArticulatedActorObject is invalid!"))
    return;
  }
  if (LinkIndex < 0 || LinkIndex >= LinkActorHandles.Num()) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("[UMochiArticulatedActorObject] LinkIndex %d is out of bounds (num links: %d)!"),
        LinkIndex,
        LinkActorHandles.Num())
    return;
  }
  const mochi::real MochiDensity = static_cast<mochi::real>(Density);
  MochiSceneObject->QueueOrExecuteActorCommand(
      LinkActorHandles[LinkIndex], [MochiDensity](mochi::Actor* Actor) {
        mochi::ErrorLog E{};
        Actor->SetDensity(MochiDensity, E);
      });
  if (bBlocking) {
    MochiSceneObject->WaitForQueuedCommands();
  }
}

void UMochiArticulatedActorObject::DestroyMochiResources() {
  MOCHI_PROFILE_SCOPE();
  if (MochiSceneObject) {
    MochiSceneObject->QueueOrExecuteCommand([&](mochi::Scene* Scene) {
      mochi::ErrorLog E{};
      Scene->CancelCallback(ExtractQueryDataCallbackHandle);
      if (PreStepCallbackHandle.IsValid()) {
        Scene->CancelCallback(PreStepCallbackHandle);
      }
      if (PostStepCallbackHandle.IsValid()) {
        Scene->CancelCallback(PostStepCallbackHandle);
      }
      if (bDestroyMochiActorOnObjectDestruction) {
        Scene->DestroyActor(MochiActorHandle);
      }
      MOCHI_ERROR_RETURN(E);
    });
    MochiSceneObject->WaitForQueuedCommands();
    MochiSceneObject = nullptr;
  }
  MochiActorHandle = {};
  LinkActorHandles = {};
  LinkActorNames = {};
  SoftActorHandles = {};
  SoftActorNames = {};
  PreStepCallbackHandle = {};
  PostStepCallbackHandle = {};
  NumDofs = 0;
  bHasArticulatedPoseController = false;
  bQueryContactPoints = false;
  ContactableActorHandles = {};
}

void UMochiArticulatedActorObject::BeginDestroy() {
  MOCHI_PROFILE_SCOPE();
  UObject::BeginDestroy();
  DestroyMochiResources();
}

void UMochiArticulatedActorObject::ExtractQueryData_Internal(
    mochi::Actor* Actor,
    mochi::Scene* SyncScene) {
  MOCHI_PROFILE_SCOPE();
  mochi::ErrorLog E;
  if (Actor == nullptr || SyncScene == nullptr) {
    return;
  }
  FMochiToUnrealConverter M2u;
  // extract root transform
  MochiThreadQueryData->RootToWorld = M2u.TransformToUnreal(Actor->GetRootTransform());
  // extract link transforms
  thread_local TArray<mochi::TransformRT> LinkTransforms;
  const int NumLinks = Actor->GetNestedLinkActors(E).size();
  LinkTransforms.SetNum(NumLinks);
  Actor->GetArticulatedLinkTransforms(ToMochiSpan(LinkTransforms), E);
  MochiThreadQueryData->LinksToWorld.SetNum(NumLinks);
  for (int i = 0; i < NumLinks; i++) {
    MochiThreadQueryData->LinksToWorld[i] = M2u.TransformToUnreal(LinkTransforms[i]);
  }
  // extract joint state
  FMochiArticulationState State = ExtractArticulatedActorState(Actor);
  ConvertArticulatedActorDofsToUnreal(Actor, State.JointPositions);
  ConvertArticulatedActorDofsToUnreal(Actor, State.JointVelocities);
  MochiThreadQueryData->JointPositions.Reset();
  MochiThreadQueryData->JointPositions.Append(State.JointPositions);
  MochiThreadQueryData->JointVelocities.Reset();
  MochiThreadQueryData->JointVelocities.Append(State.JointVelocities);
  // extract contact points
  if (bQueryContactPoints) {
    TArray<FMochiContactPoint>& Contacts = MochiThreadQueryData->ContactPoints;
    Contacts.Reset();
    for (const TPair<FName, mochi::ActorHandle>& Pair : ContactableActorHandles) {
      const FName Name = Pair.Get<0>();
      const mochi::ActorHandle Handle = Pair.Get<1>();
      mochi::Actor* const ContactActor = SyncScene->GetActor(Handle);
      const mochi::Span<const mochi::ContactPoint> MochiContactPoints =
          ContactActor->GetContactPointsWorld(E);
      if (E.IsOK()) {
        MOCHI_PROFILE_SCOPE_N("Convert Contact Points For Unreal");
        // TODO: Convert on the game thread instead; see
        // UMochiActorObject::ExtractQueryData_Internal.
        for (int MochiContactIndex = 0; MochiContactIndex < MochiContactPoints.size();
             ++MochiContactIndex) {
          const mochi::ContactPoint& MochiContact = MochiContactPoints[MochiContactIndex];
          FMochiContactPoint UnrealContact = {};
          ContactPointToUnreal(MochiContact, SyncScene, Name, M2u, UnrealContact);
          Contacts.Add(UnrealContact);
        }
      }
    }
  }
  // synchronize
  {
    FScopeLock Lock(&PendingQueryDataLock);
    std::swap(MochiThreadQueryData, PendingQueryData);
    bHasPendingQueryData = true;
  }
}
