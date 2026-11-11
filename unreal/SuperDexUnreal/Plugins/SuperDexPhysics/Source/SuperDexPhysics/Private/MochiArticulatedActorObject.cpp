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

namespace MochiArticulatedActorObjectInternal {

inline void ApplyContactOverrides(
    mochi::Scene* Scene,
    const TArray<mochi::ActorHandle>& LinkActorHandles,
    const UMochiArticulation* Articulation,
    mochi::ErrorLog& e) {
  MOCHI_ERROR_RETURN(e);
  for (const FMochiArticulationContactOverride& contact : Articulation->GetContactOverrides()) {
    const int iA = Articulation->GetLinkIndex(contact.LinkAName);
    const int iB = Articulation->GetLinkIndex(contact.LinkBName);
    if (iA == -1 || iB == -1) {
      continue;
    }
    const mochi::ActorHandle linkA = LinkActorHandles[iA];
    const mochi::ActorHandle linkB = LinkActorHandles[iB];
    Scene->EnableActorContactSymmetric(
        linkA, linkB, contact.bEnable, mochi::IncludeNestedActors::No, e);
    if (!e.IsOK()) {
      UE_LOG(
          LogTemp,
          Error,
          TEXT("[UMochiArticulatedActorObject] Failed to set link contact : %hs"),
          e._error.GetDescription());
    }
  }
}

inline void CreateAdditionalConstraints(
    mochi::Scene* Scene,
    const TArray<mochi::ActorHandle>& LinkActorHandles,
    const UMochiArticulation* Articulation,
    mochi::ErrorLog& e) {
  MOCHI_ERROR_RETURN(e);
  FUnrealToMochiConverter u2m;
  auto& additionalConstraints = Articulation->GetAdditionalConstraints();
  for (const FMochiArticulationConstraint& constraint : additionalConstraints) {
    const int iA = Articulation->GetLinkIndex(constraint.LinkAName);
    const int iB = Articulation->GetLinkIndex(constraint.LinkBName);
    if (iA == -1 || iB == -1) {
      continue;
    }
    const mochi::ActorHandle linkA = LinkActorHandles[iA];
    const mochi::ActorHandle linkB = LinkActorHandles[iB];
    const mochi::Real3 posA = u2m.TranslationToMochi(constraint.LinkAPosition);
    const mochi::Real3 posB = u2m.TranslationToMochi(constraint.LinkBPosition);
    mochi::RigidSphericalJointConstraintParams params;
    params.actorA = linkA;
    params.actorB = linkB;
    params.localPosA = posA;
    params.localPosB = posB;
    params.stiffness = constraint.Stiffness;
    params.damping = constraint.Damping;
    params.saturation = constraint.Saturation;
    Scene->CreateRigidSphericalJointConstraint(params, e);
    Scene->EnableActorContactSymmetric(
        linkA, linkB, /*enable*/ false, mochi::IncludeNestedActors::No, e);
    if (!e.IsOK()) {
      UE_LOG(
          LogTemp,
          Error,
          TEXT("[UMochiArticulatedActorObject] Failed to create additional constraint : %hs"),
          e._error.GetDescription());
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
        mochi::ErrorLog e{};
        thread_local TArray<mochi::TransformRT> linkTransforms;
        thread_local TArray<mochi::real> jointPositions;
        const int numLinks = ArticulatedActor->GetNestedLinkActors(e).size();
        linkTransforms.SetNum(numLinks);
        const int numDofs = ArticulatedActor->GetNumDofs();
        jointPositions.SetNum(numDofs);
        ArticulatedActor->GetArticulatedLinkTransforms(ToMochiSpan(linkTransforms), e);
        ArticulatedActor->GetArticulatedPose(ToMochiSpan(jointPositions), e);
        GetPose(linkTransforms, jointPositions, ArticulatedActor, Scene);
        MOCHI_ERROR_RETURN(e);
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
    mochi::ErrorLog& e) {
  MOCHI_ERROR_RETURN(e, false);
  if (bAddPoseController) {
    const auto numLinks = ArticulatedActor->GetNestedLinkActors(e).size();
    const int numPosGains = PoseControllerGains.PositionTrackingGains.Num();
    const int numRotGains = PoseControllerGains.RotationTrackingGains.Num();
    const int numJointGains = PoseControllerGains.JointTrackingGains.Num();
    check(numPosGains == 0 || numPosGains == numLinks);
    check(numRotGains == 0 || numRotGains == numLinks);
    check(numJointGains == 0 || numJointGains == numLinks);
    mochi::PoseControllerParams poseControllerParams;
    poseControllerParams.linkPosTracking = ToMochiSpan(PoseControllerGains.PositionTrackingGains);
    poseControllerParams.linkRotTracking = ToMochiSpan(PoseControllerGains.RotationTrackingGains);
    poseControllerParams.jointTracking = ToMochiSpan(PoseControllerGains.JointTrackingGains);

    ArticulatedActor->AddArticulatedPoseController(poseControllerParams, e);
    MOCHI_ERROR_RETURN(e, false);
    const bool bPoseControllerCreated = ArticulatedActor->HasArticulatedPoseController(e);
    MOCHI_ERROR_RETURN(e, false)
    if (!bPoseControllerCreated) {
      return false;
    }
    if (!OutArticulatedControllerForceQueryHandle.IsValid()) {
      OutArticulatedControllerForceQueryHandle =
          ArticulatedActor->RegisterQuery(mochi::QueryType::ArticulatedControllerForce, e);
    }
    MOCHI_ERROR_RETURN(e, false);
    // init controller targets
    ArticulatedActor->ResetArticulatedTargetLinkTransforms(InitialWorldFromLink, e);
  }
  // register pre step pose target callback
  if (SetPoseTarget) {
    OutPreStepCallbackHandle = Scene->RegisterPreStepCallback(
        "UMochiArticulatedActorObject::PreStepCallback",
        [Scene, ArticulatedActor, SetPoseTarget, bAddPoseController](mochi::StepInfo const& info) {
          mochi::ErrorLog e{};
          // get target transforms from user
          FMochiArticulatedActorPoseVariant pose;
          if (SetPoseTarget(pose, ArticulatedActor, Scene, info.timeStepSec)) {
            if (pose.IsType<TArray<mochi::TransformRT>>() && bAddPoseController) {
              auto& linkTransforms = pose.Get<TArray<mochi::TransformRT>>();
              const int numLinks = ArticulatedActor->GetNestedLinkActors(e).size();
              if (linkTransforms.Num() == numLinks) {
                ArticulatedActor->SetArticulatedTargetLinkTransforms(
                    ToMochiSpan(linkTransforms), e);
              } else {
                UE_LOG(
                    LogMochiPhysics,
                    Error,
                    TEXT(
                        "[UMochiArticulatedActorObject] Failed to set articulated target link transforms due to num links mismatch. Expected %d, received %d!"),
                    numLinks,
                    linkTransforms.Num())
              }
            } else if (pose.IsType<TArray<mochi::real>>() && bAddPoseController) {
              auto& jointPositions = pose.Get<TArray<mochi::real>>();
              const int numDofs = ArticulatedActor->GetNumDofs();
              if (jointPositions.Num() == numDofs) {
                ArticulatedActor->SetArticulatedTargetPose(ToMochiSpan(jointPositions), e);
              } else {
                UE_LOG(
                    LogMochiPhysics,
                    Error,
                    TEXT(
                        "[UMochiArticulatedActorObject] Failed to set articulated target joint positions due "
                        "to num DOFs mismatch. Expected %d, received %d!"),
                    numDofs,
                    jointPositions.Num())
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
          MOCHI_ERROR_RETURN(e);
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
  mochi::Context* mochiContext = IMochiPhysicsModule::Get()->GetMochiContext();
  FUnrealToMochiConverter u2m;
  // checks
  check(Scene);
  check(mochiContext);
  check(Articulation);
  // confirm we have links
  const int numLinks = Articulation->GetLinks().Num();
  if (numLinks == 0) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "[UMochiArticulatedActorObject] Failed to create UMochiArticulation because params does not contain links!"))
    return nullptr;
  }
  // build per-link shapes; soft shapes are still loaded separately for soft-skinned actors
  const TArray<mochi::ShapeHandle> linkShapes = Articulation->GetLinkShapes();
  const TArray<mochi::ShapeHandle> softShapes = Articulation->GetSoftShapes();
  const mochi::ShapeHandle skinShape = Articulation->GetSkinShape();
  // data we will pull off the mochi thread
  FActorSetupData setupData;
  const int numSoftLinks = Articulation->GetSoftLinks().Num();
  bool bHasArticulatedPoseController = false;
  mochi::ErrorLog e{};
  // Build the shared articulated-actor params (topology + per-link/per-joint properties) on
  // the UE game thread, BEFORE queuing the scene command. The helper dereferences UObject
  // pointers (e.g. link.ContactParams) and reads UMochiArticulation state, both of which
  // require game-thread access; matches the legacy code which also called shape/param
  // construction outside Scene->QueueOrExecuteCommand.
  const FString nameStr = ArticulationName + "___Articulation";
  const FString layerStr = Layer.ToString();
  mochi::ArticulatedActorParams skeletonParams = Articulation->GetArticulatedActorParams(
      bFixedBase,
      ContactParams,
      Layer,
      bHasGravity,
      ToMochiSpan(linkShapes),
      skinShape,
      InitialJointVelocities);
  skeletonParams.name = TCHAR_TO_ANSI(*nameStr);
  skeletonParams.worldFromRoot = u2m.TransformToMochi_RT(WorldFromRoot);
  // queue command to create actor
  Scene->QueueOrExecuteCommand([&](mochi::Scene* SyncScene) {
    mochi::Actor* articulatedActor = nullptr;
    // we create a different type of actor depending on presence of soft links
    if (numSoftLinks == 0) {
      articulatedActor = SyncScene->CreateArticulatedActor(skeletonParams, e);
      MOCHI_ERROR_RETURN(e);
    } else {
      mochi::SoftSkinnedActorParams softSkinnedParams;
      softSkinnedParams.skeletonParams = std::move(skeletonParams);
      // create soft params
      mochi::SoftActorParams softParamsCommon;
      softParamsCommon.layer = TCHAR_TO_ANSI(*layerStr);
      softParamsCommon.material.type = mochi::SoftMaterialType::NeoHookean;
      softParamsCommon.material.neoHookean.youngsModulus = 3e5_r;
      softParamsCommon.hasGravity = false;
      softParamsCommon.hasInertia = true;
      softParamsCommon.hasStress = true;
      TArray<mochi::SoftActorParams> softParams;
      softParams.Init(softParamsCommon, numSoftLinks);
      for (int i = 0; i < numSoftLinks; ++i) {
        softParams[i].shape = softShapes[i];
        softParams[i].name = TCHAR_TO_ANSI(*Articulation->GetSoftLinks()[i].Name.ToString());
      }
      softSkinnedParams.softParams = ToMochiSpan(softParams);
      // initialize soft attachments
      TArray<std::string> softAttachLinks;
      softAttachLinks.SetNum(numSoftLinks);
      for (int i = 0; i < numSoftLinks; ++i) {
        softAttachLinks[i] = TCHAR_TO_ANSI(*Articulation->GetSoftLinks()[i].ParentName.ToString());
      }
      softSkinnedParams.softAttachLinks = ToMochiSpan(softAttachLinks);
      // initialize other global params
      softSkinnedParams.enableCollidingLinks = true;
      softSkinnedParams.hasGravity = false;
      softSkinnedParams.hasInertia = false;
      softSkinnedParams.hasStress = false;
      // create articulated actor
      articulatedActor = SyncScene->CreateSoftSkinnedActor(softSkinnedParams, e);
    }
    if (articulatedActor == nullptr) {
      MOCHI_ERROR_SET(e, "Failed to create articulated actor!");
      MOCHI_ERROR_RETURN(e);
    }
    // shared setup: extract handles/names, register contact queries and callbacks
    SetupActorHandlesAndCallbacks(
        SyncScene,
        articulatedActor,
        Outer,
        GetPoseCallback,
        {},
        LinksThatQueryContactPoints,
        bSkinQueriesContactPoints,
        setupData,
        e);
    MOCHI_ERROR_RETURN(e);
    check(setupData.LinkActorHandles.Num() == numLinks);
    check(setupData.SoftActorHandles.Num() == numSoftLinks);
    // UMochiArticulation-specific: link contact overrides
    if (Articulation->GetContactOverrides().Num() > 0) {
      ApplyContactOverrides(SyncScene, setupData.LinkActorHandles, Articulation, e);
      MOCHI_ERROR_RETURN(e);
    }
    // UMochiArticulation-specific: additional constraints
    if (Articulation->GetAdditionalConstraints().Num() > 0) {
      CreateAdditionalConstraints(SyncScene, setupData.LinkActorHandles, Articulation, e);
      MOCHI_ERROR_RETURN(e);
    }
    // init pose
    const mochi::TransformSRT worldFromRoot = u2m.TransformToMochi_SRT(WorldFromRoot);
    std::vector<mochi::TransformRT> worldFromLinks(RootFromLinks.Num());
    for (int iLink = 0; iLink < RootFromLinks.Num(); ++iLink) {
      const mochi::TransformRT rootFromLinkRT = u2m.TransformToMochi_RT(RootFromLinks[iLink]);
      const mochi::TransformSRT worldFromLinkSRT = worldFromRoot * rootFromLinkRT;
      worldFromLinks[iLink] = worldFromLinkSRT.GetTransformRT();
    }
    articulatedActor->SetArticulatedPoseFromLinks(worldFromLinks, e);
    MOCHI_ERROR_RETURN(e);
    // create pose controller (UMochiArticulation-specific)
    if (bAddPoseController || ControlCallback) {
      SetupController(
          SyncScene,
          articulatedActor,
          bAddPoseController,
          PoseControllerGains,
          ControlCallback,
          worldFromLinks,
          setupData.PreStepCallback,
          setupData.ArticulatedControllerForceQueryHandle,
          e);
      MOCHI_ERROR_RETURN(e);
    }
    bHasArticulatedPoseController = articulatedActor->HasArticulatedPoseController(e);
    MOCHI_ERROR_RETURN(e);
  });
  // wait for creation to finish
  Scene->WaitForQueuedCommands();
  if (!e.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "[UMochiArticulatedActorObject] Failed to create UMochiArticulatedActorObject because an error was encountered: %hs!"),
        e._error.GetDescription())
    return nullptr;
  }
  setupData.bHasPoseController = bHasArticulatedPoseController;
  return CreateFromSetupData(Outer, Scene, setupData);
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
  FActorSetupData setupData;
  mochi::ErrorLog e{};
  Scene->QueueOrExecuteCommand([&](mochi::Scene* SyncScene) {
    mochi::Actor* articulatedActor = SyncScene->GetActor(ArticulatedActorHandle);
    if (!articulatedActor) {
      MOCHI_ERROR_SET(e, "CreateFromExistingArticulatedActor: actor not found in scene");
      return;
    }
    SetupActorHandlesAndCallbacks(
        SyncScene,
        articulatedActor,
        Outer,
        GetPoseCallback,
        ControlCallback,
        LinksThatQueryContactPoints,
        bSkinQueriesContactPoints,
        setupData,
        e);
  });
  Scene->WaitForQueuedCommands();
  if (!e.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "[UMochiArticulatedActorObject] Failed to create UMochiArticulatedActorObject from existing actor: %hs!"),
        e._error.GetDescription())
    return nullptr;
  }
  return CreateFromSetupData(Outer, Scene, setupData);
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
    mochi::ErrorLog& e) {
  MOCHI_ERROR_RETURN(e);
  OutData.ActorName = ArticulatedActor->GetName();
  ArticulatedActor->SetUserData(Outer);
  OutData.ActorHandle = ArticulatedActor->GetHandle();
  OutData.NumDofs = ArticulatedActor->GetNumDofs();
  // strip the articulated actor name path from nested actor names
  const FString toStripFromName = OutData.ActorName + TEXT("/");
  auto GetNestedName = [&SyncScene, &toStripFromName](mochi::ActorHandle ActorHandle) {
    mochi::Actor* actor = SyncScene->GetActor(ActorHandle);
    FString nestedActorName = actor->GetName();
    nestedActorName.ReplaceInline(*toStripFromName, TEXT(""));
    return FName(nestedActorName);
  };
  // extract link actors
  const mochi::Span<mochi::ActorHandle const> linkActors = ArticulatedActor->GetNestedLinkActors(e);
  MOCHI_ERROR_RETURN(e);
  const int numLinks = linkActors.size();
  OutData.LinkActorHandles.Init({}, numLinks);
  OutData.LinkActorNames.Init(NAME_None, numLinks);
  for (int i = 0; i < numLinks; ++i) {
    OutData.LinkActorHandles[i] = linkActors[i];
    OutData.LinkActorNames[i] = GetNestedName(linkActors[i]);
    SyncScene->GetActor(linkActors[i])->SetUserData(Outer);
  }
  // extract soft actors
  const mochi::Span<mochi::ActorHandle const> softActors = ArticulatedActor->GetNestedSoftActors(e);
  const int numSoftLinks = softActors.size();
  if (numSoftLinks > 0) {
    OutData.SoftActorHandles.Init({}, numSoftLinks);
    OutData.SoftActorNames.Init(NAME_None, numSoftLinks);
    for (int i = 0; i < numSoftLinks; ++i) {
      OutData.SoftActorHandles[i] = softActors[i];
      OutData.SoftActorNames[i] = GetNestedName(softActors[i]);
      SyncScene->GetActor(softActors[i])->SetUserData(Outer);
    }
  }
  // register contact queries
  if (LinksThatQueryContactPoints.Num() > 0) {
    auto AddContactableActors =
        [&LinksThatQueryContactPoints, &SyncScene, &OutData, &GetNestedName, &e](
            const TArray<mochi::ActorHandle>& ActorHandles) {
          for (int iHandle = 0; iHandle < ActorHandles.Num(); ++iHandle) {
            const auto handle = ActorHandles[iHandle];
            const auto name = GetNestedName(handle);
            if (LinksThatQueryContactPoints.Contains(name)) {
              mochi::Actor* actor = SyncScene->GetActor(handle);
              // A link the skin covers is not a colliding actor and carries no contact samples, so
              // registering the query on it would fail and abort setup for the whole actor.
              if (!actor->IsStatic() && actor->IsQuerySupported(mochi::QueryType::ContactPoints)) {
                actor->RegisterQuery(mochi::QueryType::ContactPoints, e);
                OutData.ContactableActorHandles.Add({name, handle});
              }
            }
          }
        };
    AddContactableActors(OutData.LinkActorHandles);
    AddContactableActors(OutData.SoftActorHandles);
    MOCHI_ERROR_RETURN(e);
  }
  // The skin has no actor of its own -- its contact samples live on the top-level articulated
  // actor -- so it is registered separately and reported under that actor's name.
  if (bSkinQueriesContactPoints) {
    if (ArticulatedActor->IsQuerySupported(mochi::QueryType::ContactPoints)) {
      ArticulatedActor->RegisterQuery(mochi::QueryType::ContactPoints, e);
      MOCHI_ERROR_RETURN(e);
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
        e);
    MOCHI_ERROR_RETURN(e);
  }
}

UMochiArticulatedActorObject* UMochiArticulatedActorObject::CreateFromSetupData(
    UObject* Outer,
    UMochiSceneObject* Scene,
    const FActorSetupData& Data) {
  UMochiArticulatedActorObject* mochiActorObj =
      NewObject<UMochiArticulatedActorObject>(Outer, FName(*Data.ActorName));
  mochiActorObj->MochiSceneObject = Scene;
  mochiActorObj->MochiActorHandle = Data.ActorHandle;
  mochiActorObj->LinkActorHandles = Data.LinkActorHandles;
  mochiActorObj->LinkActorNames = Data.LinkActorNames;
  mochiActorObj->SoftActorHandles = Data.SoftActorHandles;
  mochiActorObj->SoftActorNames = Data.SoftActorNames;
  mochiActorObj->PreStepCallbackHandle = Data.PreStepCallback;
  mochiActorObj->PostStepCallbackHandle = Data.PostStepCallback;
  mochiActorObj->NumDofs = Data.NumDofs;
  mochiActorObj->bHasArticulatedPoseController = Data.bHasPoseController;
  mochiActorObj->bQueryContactPoints = Data.ContactableActorHandles.Num() > 0;
  mochiActorObj->ContactableActorHandles = Data.ContactableActorHandles;
  mochiActorObj->ArticulatedControllerForceQueryHandle = Data.ArticulatedControllerForceQueryHandle;
  mochiActorObj->ExtractQueryDataCallbackHandle = Scene->RegisterPostStepCallback(
      "UMochiArticulatedActorObject::ExtractQueryData_Internal",
      [mochiActorObj](mochi::StepInfo const& info) {
        mochi::Actor* actor = info.scene->GetActor(mochiActorObj->MochiActorHandle);
        mochiActorObj->ExtractQueryData_Internal(actor, info.scene);
      });
  return mochiActorObj;
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
  const auto numLinks = LinkActorHandles.Num();
  const int numPosGains = PoseControllerGains.PositionTrackingGains.Num();
  const int numRotGains = PoseControllerGains.RotationTrackingGains.Num();
  const int numJointGains = PoseControllerGains.JointTrackingGains.Num();
  check(numPosGains == 0 || numPosGains == numLinks);
  check(numRotGains == 0 || numRotGains == numLinks);
  check(numJointGains == 0 || numJointGains == numLinks);
  mochi::PoseControllerParams poseControllerParams;
  poseControllerParams.linkPosTracking = ToMochiSpan(PoseControllerGains.PositionTrackingGains);
  poseControllerParams.linkRotTracking = ToMochiSpan(PoseControllerGains.RotationTrackingGains);
  poseControllerParams.jointTracking = ToMochiSpan(PoseControllerGains.JointTrackingGains);
  MochiSceneObject->QueueOrExecuteActorCommand(
      MochiActorHandle, [this, poseControllerParams](mochi::Actor* ArticulatedActor) {
        mochi::ErrorLog e;
        bHasArticulatedPoseController = ArticulatedActor->HasArticulatedPoseController(e);
        MOCHI_ERROR_RETURN(e);
        if (bHasArticulatedPoseController) {
          ArticulatedActor->RemoveArticulatedPoseController(e);
          MOCHI_ERROR_RETURN(e);
          bHasArticulatedPoseController = ArticulatedActor->HasArticulatedPoseController(e);
          MOCHI_ERROR_RETURN(e);
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
          MOCHI_ERROR_RETURN(e);
          UE_LOG(
              LogMochiPhysics,
              Display,
              TEXT(
                  "[UMochiArticulatedActorObject] Removed existing pose controller when adding a new one."));
          MOCHI_ERROR_RETURN(e);
        }
        ArticulatedActor->AddArticulatedPoseController(poseControllerParams, e);
        MOCHI_ERROR_RETURN(e);
        bHasArticulatedPoseController = ArticulatedActor->HasArticulatedPoseController(e);
        MOCHI_ERROR_RETURN(e);
        if (bHasArticulatedPoseController) {
          if (!ArticulatedControllerForceQueryHandle.IsValid()) {
            ArticulatedControllerForceQueryHandle =
                ArticulatedActor->RegisterQuery(mochi::QueryType::ArticulatedControllerForce, e);
          }
          MOCHI_ERROR_RETURN(e);
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
    mochi::ErrorLog e;
    bHasArticulatedPoseController = ArticulatedActor->HasArticulatedPoseController(e);
    MOCHI_ERROR_RETURN(e);
    if (bHasArticulatedPoseController) {
      ArticulatedActor->RemoveArticulatedPoseController(e);
      MOCHI_ERROR_RETURN(e);
      bHasArticulatedPoseController = ArticulatedActor->HasArticulatedPoseController(e);
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
      MOCHI_ERROR_RETURN(e);
    } else {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT(
              "[UMochiArticulatedActorObject] Articulated actor has no pose controller to remove!"));
      MOCHI_ERROR_RETURN(e);
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
    const bool wasIsUpdatingUnrealQueryData = bIsUpdatingUnrealQueryData.exchange(true);
    if (wasIsUpdatingUnrealQueryData) {
      UE_LOG(
          LogMochiPhysics,
          Error,
          TEXT(
              "[UMochiArticulatedActorObject] TryGetQueryData only supports single-threaded access!"))
      return;
    }
    MOCHI_DEFER(bIsUpdatingUnrealQueryData.store(false));

    // Synchronize
    bool hasNewData = false;
    {
      FScopeLock lock(&PendingQueryDataLock);
      if (bHasPendingQueryData) {
        std::swap(PendingQueryData, UnrealThreadQueryData);
        bHasPendingQueryData = false;
        hasNewData = true;
      }
    }

    if (hasNewData) {
      // Fix up data here on the Unreal (render) thread. We don't do this part on the Mochi
      // simulation thread because it requires access to the Unreal components and actors.
      if (UnrealThreadQueryData->ContactPoints.Num() > 0) {
        const auto outer = GetOuter();
        for (auto& contact : UnrealThreadQueryData->ContactPoints) {
          contact.MochiActor =
              contact.MochiComponent ? contact.MochiComponent->GetOwner() : nullptr;
          contact.OtherMochiActor =
              contact.OtherMochiComponent ? contact.OtherMochiComponent->GetOwner() : nullptr;
          // There's no guarantee that OtherMochiActor is registered.
          // ensure(contact.MochiActor && contact.OtherMochiActor);
          constexpr bool bVerboseContactWarnings = false;
          if (bVerboseContactWarnings) {
            if (!contact.MochiActor || !contact.OtherMochiActor || !contact.MochiComponent ||
                !contact.MochiComponent) {
              FString aA = contact.MochiActor ? *contact.MochiActor->GetName() : TEXT("NULL");
              FString cA =
                  contact.MochiComponent ? *contact.MochiComponent->GetName() : TEXT("NULL");
              FString mA = contact.InternalMochiActor.ToString();
              FString aB =
                  contact.OtherMochiActor ? *contact.OtherMochiActor->GetName() : TEXT("NULL");
              FString cB = contact.OtherMochiComponent ? *contact.OtherMochiComponent->GetName()
                                                       : TEXT("NULL");
              FString mB = contact.OtherInternalMochiActor.ToString();
              UE_LOG(
                  LogMochiPhysics,
                  Warning,
                  TEXT(
                      "[UMochiArticulatedActorObject] Untraceable contact detected: "
                      "AActorA=%s : USceneComponentA=%s : MochiActorA=%s  <---> AActorB=%s : USceneComponentB=%s : MochiActorB=%s"),
                  *aA,
                  *cA,
                  *mA,
                  *aB,
                  *cB,
                  *mB);
              continue;
            }
            if (contact.MochiComponent && contact.MochiComponent == contact.OtherMochiComponent) {
              UE_LOG(
                  LogMochiPhysics,
                  Warning,
                  TEXT("[UMochiArticulatedActorObject] Self-contact detected on component %s"),
                  *contact.MochiComponent->GetName());
            }
          }
          if (contact.MochiComponent != outer) {
            // Swap the direction of this contact (by convention)
            ensure(contact.OtherMochiComponent == outer);
            std::swap(contact.InternalMochiActor, contact.OtherInternalMochiActor);
            std::swap(contact.MochiComponent, contact.OtherMochiComponent);
            std::swap(contact.MochiActor, contact.OtherMochiActor);
            std::swap(contact.Location, contact.OtherLocation);
            std::swap(contact.Velocity, contact.OtherVelocity);
            contact.Normal = -contact.Normal;
            contact.Force = -contact.Force;

            // Unambiguously record that this is a swapped (other->this) contact,
            // for clients that need to process contacts in an asymmetric manner
            contact.ContactDirection = EMochiContactDirection::OtherContactsThis;
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
  MochiSceneObject->QueueOrExecuteCommand([WorldFromLocal, this](mochi::Scene* scene) {
    FUnrealToMochiConverter u2m;
    auto articulatedActor = scene->GetActor(MochiActorHandle);
    if (articulatedActor == nullptr) {
      return; // actor no longer exists; UE component must be shutting down
    }
    mochi::ErrorLog e{};
    const int numLinks = articulatedActor->GetNestedLinkActors(e).size();
    MOCHI_ERROR_RETURN(e);
    std::vector<mochi::TransformRT> transforms(numLinks);
    articulatedActor->GetArticulatedLinkTransforms(transforms, e);
    MOCHI_ERROR_RETURN(e);
    transforms[0] = u2m.TransformToMochi_RT(WorldFromLocal);
    articulatedActor->SetArticulatedPoseFromLinks(transforms, e);
    MOCHI_ERROR_RETURN(e);
    const int numDofs = articulatedActor->GetNumDofs();
    std::vector<mochi::real> zeroVel(numDofs, 0);
    articulatedActor->SetArticulatedJointVelocities(zeroVel, e);
    MOCHI_ERROR_RETURN(e);
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
  MochiSceneObject->QueueOrExecuteCommand([JointPositions, this](mochi::Scene* scene) {
    auto articulatedActor = scene->GetActor(MochiActorHandle);
    if (articulatedActor == nullptr) {
      return; // actor no longer exists; UE component must be shutting down
    }
    const int numDofs = articulatedActor->GetNumDofs();
    std::vector<mochi::real> newJointPose(numDofs, 0);
    for (int i = 0; i < JointPositions.Num(); ++i) {
      newJointPose[i] = static_cast<mochi::real>(JointPositions[i]);
    }
    mochi::ErrorLog e{};
    articulatedActor->SetArticulatedPoseFromJoints(newJointPose, e);
    MOCHI_ERROR_RETURN(e);

    std::vector<mochi::real> zeroVel(numDofs, 0);
    articulatedActor->SetArticulatedJointVelocities(zeroVel, e);
    MOCHI_ERROR_RETURN(e);
  });
  if (bBlocking) {
    MochiSceneObject->WaitForQueuedCommands();
  }
}

TArray<UMochiContactParams*> UMochiArticulatedActorObject::GetLinkContactParams() const {
  MOCHI_PROFILE_SCOPE();
  TArray<UMochiContactParams*> contactParams;
  if (!IsValid()) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("[UMochiArticulatedActorObject] UMochiArticulatedActorObject is invalid!"))
    return contactParams;
  }
  contactParams.SetNum(LinkActorHandles.Num());
  for (int i = 0; i < LinkActorHandles.Num(); ++i) {
    MochiSceneObject->QueueOrExecuteActorCommand(
        LinkActorHandles[i], [&contactParams, i](mochi::Actor* actor) {
          mochi::ErrorLog e{};
          const auto mochiContactParams = actor->GetContactParams(e);
          if (e.IsOK()) {
            // Convert mochi params to UE params
            auto ueContactParams = NewObject<UMochiContactParams>(GetTransientPackage());
            ueContactParams->FromMochi(mochiContactParams);
            contactParams[i] = ueContactParams;
          } else {
            // Link doesn't have contact params (dummy link, static actor, etc.)
            contactParams[i] = nullptr;
          }
        });
  }
  MochiSceneObject->WaitForQueuedCommands();
  return contactParams;
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
  const auto mochiContactParams = ContactParams->ToMochi();
  MochiSceneObject->QueueOrExecuteActorCommand(
      LinkActorHandles[LinkIndex], [mochiContactParams](mochi::Actor* actor) {
        mochi::ErrorLog e{};
        actor->SetContactParams(mochiContactParams, e);
      });
  if (bBlocking) {
    MochiSceneObject->WaitForQueuedCommands();
  }
}

TArray<float> UMochiArticulatedActorObject::GetLinkDensities() const {
  MOCHI_PROFILE_SCOPE();
  TArray<float> densities;
  if (!IsValid()) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("[UMochiArticulatedActorObject] UMochiArticulatedActorObject is invalid!"))
    return densities;
  }
  densities.SetNum(LinkActorHandles.Num());
  for (int i = 0; i < LinkActorHandles.Num(); ++i) {
    MochiSceneObject->QueueOrExecuteActorCommand(
        LinkActorHandles[i], [&densities, i](mochi::Actor* actor) {
          mochi::ErrorLog e{};
          densities[i] = static_cast<float>(actor->GetDensity(e));
        });
  }
  MochiSceneObject->WaitForQueuedCommands();
  return densities;
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
  const auto mochiDensity = static_cast<mochi::real>(Density);
  MochiSceneObject->QueueOrExecuteActorCommand(
      LinkActorHandles[LinkIndex], [mochiDensity](mochi::Actor* actor) {
        mochi::ErrorLog e{};
        actor->SetDensity(mochiDensity, e);
      });
  if (bBlocking) {
    MochiSceneObject->WaitForQueuedCommands();
  }
}

void UMochiArticulatedActorObject::DestroyMochiResources() {
  MOCHI_PROFILE_SCOPE();
  if (MochiSceneObject) {
    MochiSceneObject->QueueOrExecuteCommand([&](mochi::Scene* scene) {
      mochi::ErrorLog e{};
      scene->CancelCallback(ExtractQueryDataCallbackHandle);
      if (PreStepCallbackHandle.IsValid()) {
        scene->CancelCallback(PreStepCallbackHandle);
      }
      if (PostStepCallbackHandle.IsValid()) {
        scene->CancelCallback(PostStepCallbackHandle);
      }
      if (bDestroyMochiActorOnObjectDestruction) {
        scene->DestroyActor(MochiActorHandle);
      }
      MOCHI_ERROR_RETURN(e);
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
  mochi::ErrorLog e;
  if (Actor == nullptr || SyncScene == nullptr) {
    return;
  }
  FMochiToUnrealConverter m2u;
  // extract root transform
  MochiThreadQueryData->RootToWorld = m2u.TransformToUnreal(Actor->GetRootTransform());
  // extract link transforms
  thread_local TArray<mochi::TransformRT> linkTransforms;
  const int numLinks = Actor->GetNestedLinkActors(e).size();
  linkTransforms.SetNum(numLinks);
  Actor->GetArticulatedLinkTransforms(ToMochiSpan(linkTransforms), e);
  MochiThreadQueryData->LinksToWorld.SetNum(numLinks);
  for (int i = 0; i < numLinks; i++) {
    MochiThreadQueryData->LinksToWorld[i] = m2u.TransformToUnreal(linkTransforms[i]);
  }
  // extract joint state
  FMochiArticulationState state = ExtractArticulatedActorState(Actor);
  ConvertArticulatedActorDofsToUnreal(Actor, state.JointPositions);
  ConvertArticulatedActorDofsToUnreal(Actor, state.JointVelocities);
  MochiThreadQueryData->JointPositions.Reset();
  MochiThreadQueryData->JointPositions.Append(state.JointPositions);
  MochiThreadQueryData->JointVelocities.Reset();
  MochiThreadQueryData->JointVelocities.Append(state.JointVelocities);
  // extract contact points
  if (bQueryContactPoints) {
    auto& contacts = MochiThreadQueryData->ContactPoints;
    contacts.Reset();
    for (const auto& pair : ContactableActorHandles) {
      const auto name = pair.Get<0>();
      const auto handle = pair.Get<1>();
      const auto actor = SyncScene->GetActor(handle);
      const mochi::Span<const mochi::ContactPoint> mochiContactPoints =
          actor->GetContactPointsWorld(e);
      if (e.IsOK()) {
        MOCHI_PROFILE_SCOPE_N("Convert Contact Points For Unreal");
        // TODO: Convert on the game thread instead; see
        // UMochiActorObject::ExtractQueryData_Internal.
        for (int iMochiContact = 0; iMochiContact < mochiContactPoints.size(); ++iMochiContact) {
          const auto& mochiContact = mochiContactPoints[iMochiContact];
          FMochiContactPoint unrealContact = {};
          ContactPointToUnreal(mochiContact, SyncScene, name, m2u, unrealContact);
          contacts.Add(unrealContact);
        }
      }
    }
  }
  // synchronize
  {
    FScopeLock lock(&PendingQueryDataLock);
    std::swap(MochiThreadQueryData, PendingQueryData);
    bHasPendingQueryData = true;
  }
}
