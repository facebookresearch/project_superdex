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

#pragma once

#include "CoreMinimal.h"
#include "MochiArticulatedActorObject.h"
#include "MochiArticulationController.h"
#include "MochiArticulationPreviewComponent.h"
#include "MochiAssets.h"

#include "MochiArticulationComponent.generated.h"

class UMochiSoftMeshComponent;

UCLASS(
    ClassGroup = (Custom),
    HideCategories = (Physics, Collision, Shelling, Navigation),
    meta = (BlueprintSpawnableComponent))
class SUPERDEXPHYSICS_API UMochiArticulationComponent : public UPoseableMeshComponent {
  GENERATED_BODY()

 protected:
  // The MochiArticulation asset from which the articulation will be generated.
  UPROPERTY(EditAnywhere, Category = "Mochi Articulation")
  UMochiArticulation* Articulation = nullptr;
  // The initial joint positions with which the articulation will be spawned.
  UPROPERTY(
      EditAnywhere,
      Category = "Mochi Articulation",
      BlueprintReadOnly,
      EditFixedSize,
      meta = (EditFixedOrder))
  TArray<double> InitialJointPositions;
  // Optional initial joint velocities [rad/s or m/s], laid out like InitialJointPositions.
  UPROPERTY(EditAnywhere, Category = "Mochi Articulation", BlueprintReadOnly, AdvancedDisplay)
  TArray<double> InitialJointVelocities;
  // If true, the articulation will be spawned with a fixed root joint, making the articulation
  // fixed in space at the location it is spawned. Only overrides Free and Hard root joints.
  UPROPERTY(EditAnywhere, Category = "Mochi Articulation", BlueprintReadOnly)
  bool bFixedBase = false;
  // The layer the articulation will be simulated on.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Mochi Articulation")
  FName Layer = FMochiLayer::GenericObject;
  // Parameters controlling contacts of the links.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Mochi Articulation")
  UMochiContactParams* ContactParams = nullptr;
  // Whether the articulation should be simulated under the influence of gravity.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Mochi Articulation")
  bool bGravityEnabled = true;
  // Enable contact queries and events.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Mochi Articulation")
  bool bEnableContacts = false;
  // The links that will be included in contact queries and events. Defaults to all links;
  // can be pruned of uninteresting links for performance gains. Note that fixed base links
  // will never generate contacts; this is a Mochi limitation.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Articulation",
      meta = (EditCondition = "bEnableContacts"))
  TArray<FName> ContactGeneratingLinks;
  // Filter contacts against self.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Mochi Articulation",
      meta = (EditCondition = "bEnableContacts"))
  bool bFilterSelfContacts = true;
  // The MochiArticulationController class to use.
  UPROPERTY(EditAnywhere, Category = "Mochi Articulation", BlueprintReadOnly)
  TSubclassOf<UMochiArticulationController> ArticulationControllerClass;

 public:
  DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(
      FOnArticulationSpawned,
      UMochiArticulationComponent*,
      Articulation);
  DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(
      FOnSoftLinksSpawned,
      UMochiArticulationComponent*,
      Articulation,
      const TArray<UMochiSoftMeshComponent*>&,
      SoftLinks);

  // Emitted when the articulation has been spawned.
  UPROPERTY(BlueprintAssignable, Category = "Mochi Articulation")
  FOnArticulationSpawned OnArticulationSpawned;
  // Emitted when the articulation soft links have been spawned.
  UPROPERTY(BlueprintAssignable, Category = "Mochi Articulation")
  FOnSoftLinksSpawned OnSoftLinksSpawned;

  // Emitted when a nested element in this component begins contact with another Mochi component.
  // bEnableContacts must be enabled. Occurs before the non-nested equivalent.
  UPROPERTY(BlueprintAssignable, Category = "Mochi Articulation")
  FOnMochiComponentBeginNestedContact OnMochiComponentBeginNestedContact;
  // Emitted when this component begins contact with another Mochi component.
  // bEnableContacts must be enabled. Occurs after the nested equivalent.
  UPROPERTY(BlueprintAssignable, Category = "Mochi Articulation")
  FOnMochiComponentBeginContact OnMochiComponentBeginContact;
  // Emitted when a nested element in this component ends contact with another Mochi component.
  // bEnableContacts must be enabled. Occurs before the non-nested equivalent.
  UPROPERTY(BlueprintAssignable, Category = "Mochi Articulation")
  FOnMochiComponentEndNestedContact OnMochiComponentEndNestedContact;
  // Emitted when this component ends contact with another Mochi component.
  // bEnableContacts must be enabled. Occurs after the nested equivalent.
  UPROPERTY(BlueprintAssignable, Category = "Mochi Articulation")
  FOnMochiComponentEndContact OnMochiComponentEndContact;
  // Emitted when a nested element in this component is contacting any other Mochi components.
  // bEnableContacts must be enabled. Occurs before the non-nested equivalent.
  UPROPERTY(BlueprintAssignable, Category = "Mochi Articulation")
  FOnMochiComponentNestedContact OnMochiComponentNestedContact;
  // Emitted when this component is contacting any other Mochi components.
  // bEnableContacts must be enabled. Occurs after the nested equivalent.
  UPROPERTY(BlueprintAssignable, Category = "Mochi Articulation")
  FOnMochiComponentContact OnMochiComponentContact;

  // Emitted before any contact events are invoked.
  UPROPERTY(BlueprintAssignable, Category = "Mochi Articulation")
  FOnMochiComponentContactProcessing OnMochiComponentBeginContactProcessing;
  // Emitted after all contact events have been invoked, even if none actually were.
  UPROPERTY(BlueprintAssignable, Category = "Mochi Articulation")
  FOnMochiComponentContactProcessing OnMochiComponentEndContactProcessing;

 public:
  // Set the articulation asset. This has no effect if the articulation has already been spawned.
  UFUNCTION(BlueprintCallable, Category = "Mochi Articulation")
  void SetArticulationAsset(UMochiArticulation* InArticulation);
  // Get the articulation asset.
  UFUNCTION(BlueprintPure, Category = "Mochi Articulation")
  UMochiArticulation* GetArticulationAsset() const;
  // Set the initial pose (configuration) of the articulation as an array of joint positions [rad or
  // m]. Typically used from BP constructors since setting after BeginPlay has no effect. Fails if
  // array size is not equal to the number of articulation DOFs of if the articulation has already
  // been spawned.
  UFUNCTION(BlueprintCallable, Category = "Mochi Articulation")
  bool SetInitialJointPositions(const TArray<double>& NewInitialJointPositions);
  // Get the initial pose (configuration) of the articulation as an array of joint positions [rad or
  // m].
  UFUNCTION(BlueprintPure, Category = "Mochi Articulation")
  const TArray<double>& GetInitialJointPositions() const;
  // Returns true if the articulation is spawned with a fixed base.
  UFUNCTION(BlueprintPure, Category = "Mochi Articulation")
  bool IsFixedBase() const;
  // Returns true if gravity is enabled on this articulation.
  UFUNCTION(BlueprintPure, Category = "Mochi Articulation")
  bool GetMochiGravityEnabled() const;
  // Get the UMochiSoftMeshComponent associated with a soft link.
  UFUNCTION(BlueprintPure, Category = "Mochi Articulation")
  const UMochiSoftMeshComponent* GetSoftLink(int SoftLinIndex);
  // Set the world transform of the articulation base to a new location (non-fixed articulations
  // only)
  // TODO: This function is untested with robots that do not have a fixed base. It is possible
  // that it does not work as intended in this case. Intent would be to leave the base link in
  // place, but move all other links.
  UFUNCTION(BlueprintCallable, Category = "Mochi Articulation")
  void SetArticulationWorldTransform(const FTransform& WorldTransform);
  // Teleport the articulation to a new pose defined by joint positions
  UFUNCTION(BlueprintCallable, Category = "Mochi Articulation")
  void SetArticulatedPoseFromJoints(TArray<double> const& JointPositions);
  // Get the contact params configured on this articulation (editor/initial value).
  // Note: does not reflect runtime changes made via SetLinkContactParams.
  UFUNCTION(BlueprintPure, Category = "Mochi Articulation")
  UMochiContactParams* GetContactParams() const {
    return ContactParams;
  }
  // Get the underlying Mochi articulated actor object.
  UFUNCTION(BlueprintPure, Category = "Mochi Articulation")
  UMochiArticulatedActorObject* GetMochiArticulatedActor() const;

  // Get the world transform of a link by name. These transforms are updated each
  // component tick if new data is available from Mochi in the same block that queries
  // contacts and triggers contact events. This serves as an alternative to
  // USkinnedMeshComponent::GetBoneTransform when more precise synchronization is
  // required.
  UFUNCTION(BlueprintPure, Category = "Mochi Articulation")
  FTransform GetLinkToWorldTransform(FName LinkName) const;

  // Set the MochiArticulationController to a new one (nullptr to remove controller).
  UFUNCTION(BlueprintCallable, Category = "Mochi Articulation")
  bool SetArticulationController(TSubclassOf<UMochiArticulationController> NewControllerClass);
  // Get the MochiArticulationController if it exists.
  UFUNCTION(BlueprintPure, Category = "Mochi Articulation")
  UMochiArticulationController* GetArticulationController() const;

 public:
  UMochiArticulationComponent();

 protected:
  virtual void BeginPlay() override;
  virtual void TickComponent(
      float DeltaTime,
      ELevelTick TickType,
      FActorComponentTickFunction* ThisTickFunction) override;
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
  virtual void OnRegister() override;
  virtual FBoxSphereBounds CalcBounds(const FTransform& LocalToWorld) const override;

 protected:
  virtual bool WantsPoseController() const;
  virtual FMochiPoseControllerGains GetPoseControllerGains() const;
  virtual void CreateArticulatedActor();
  virtual void DestroyArticulatedActor();
  virtual void OnArticulatedActorGetPose(
      const TArray<mochi::TransformRT>& InWorldFromLinks,
      const TArray<mochi::real>& InJointDofs,
      mochi::Actor* ArticulatedActor,
      mochi::Scene* Scene);
  virtual bool OnArticulatedActorControl(
      FMochiArticulatedActorPoseVariant& OutPoseTarget,
      mochi::Actor* ArticulatedActor,
      mochi::Scene* Scene);

 protected:
  bool EnsureArrayProperties();
  void PoseMeshFromJointPositions(const TArray<double>& InJointPositions);
  void PoseMeshFromLinkWorldTransforms(const TArray<FTransform>& InLinkWorldTransforms);
  void CreateSoftLinksDynamicMeshComponents(const TArray<mochi::ActorHandle>& SoftActors);

#if WITH_EDITOR
  virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
  virtual void PostLoad() override;
  friend class FMochiArticulationComponentDetailsCustomization;
#endif

 protected:
  // The UMochiArticulatedActorObject owning the articulated actor
  UPROPERTY() UMochiArticulatedActorObject* ArticulatedActorObject = nullptr;
  // Set to true once all requirements for articulated actor callbacks have been met.
  std::atomic<bool> bArticulatedActorReady = false;

  // Contact event handler
  FMochiContactEventHandler ContactEventHandler;
  // Nested contact event handle
  FMochiNestedContactEventHandler NestedContactEventHandler;
  // Dynamic mesh components create for each soft shape
  UPROPERTY()
  TArray<UMochiSoftMeshComponent*> SoftLinkDynamicMeshComponents;

  // Link transforms queried at the same time of contacts and their events.
  TArray<FTransform> QueriedLinkTransforms;

  // DOF Bookeeping
  int NumMochiDofs = 0;
  int NumMochiBaseDofs = 0;
  int NumMochiJointDofs = 0;
  int NumArticulationDofs = 0;

 private:
  UPROPERTY()
  TObjectPtr<UMochiArticulationController> ArticulationController = nullptr;
  mutable FCriticalSection ArticulationControllerMutex;
  UFUNCTION()
  void HandleArticulationControllerOnArticulationSpawned(
      UMochiArticulationComponent* InArticulation);

  friend class FSuperDexPrefabImporter;
};
