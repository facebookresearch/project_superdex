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
#include "MochiArticulationPreviewComponent.h"
#include "SuperDexBot.h"
#include "SuperDexBotController.h"

#include <superdex_robotics/superdex_robotics.h>

#include "SuperDexBotComponent.generated.h"

UCLASS(
    ClassGroup = (Custom),
    HideCategories = (Physics, Collision, Shelling, Navigation),
    meta = (BlueprintSpawnableComponent))
class SUPERDEXROBOTICS_API USuperDexBotComponent : public UPoseableMeshComponent {
  GENERATED_BODY()

 protected:
  // The SuperDexBot asset from which the bot will be generated.
  UPROPERTY(EditAnywhere, Category = "SuperDex Bot")
  TObjectPtr<USuperDexBot> Bot = nullptr;
  // The initial pose in which the bot will be spawned as an array of joint
  // positions [rad or m] equal to the number of DOFs in the bot.
  UPROPERTY(
      EditAnywhere,
      Category = "SuperDex Bot",
      BlueprintReadOnly,
      EditFixedSize,
      meta = (EditFixedOrder))
  TArray<double> InitialPose;
  // Override the mochi contact layer for every collidable the bot spawns -- all links and
  // the skin, if it has one. If empty, the layers from the bot asset are used.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDex Bot")
  FString LayerOverride;
  // Whether the bot should be simulated under the influence of gravity.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDex Bot")
  bool bHasGravity = true;
  // Enable contact queries and events.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDex Bot")
  bool bEnableContacts = false;
  // The links that will be included in contact queries and events.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "SuperDex Bot",
      meta = (EditCondition = "bEnableContacts"))
  TArray<FName> ContactGeneratingLinks;
  // Whether the bot's skin, if it has one, is included in contact queries and events. Skin
  // contacts are reported under the bot's name rather than a link name.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "SuperDex Bot",
      meta = (EditCondition = "bEnableContacts"))
  bool bSkinGeneratesContacts = false;
  // Filter contacts against self.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "SuperDex Bot",
      meta = (EditCondition = "bEnableContacts"))
  bool bFilterSelfContacts = true;
  // The SuperDexBotController class to use.
  UPROPERTY(EditAnywhere, Category = "SuperDex Bot", BlueprintReadOnly)
  TSubclassOf<USuperDexBotController> BotControllerClass;
  // Create the bot on the first game tick. Disable to create it later with SpawnBot, e.g. once a
  // tracked hand has been seen.
  // The bot is created at this component's world transform, scale included: a uniform world scale
  // other than 1 creates a scaled bot ([Experimental] BotPrefab::scale). Bots only scale uniformly,
  // so a non-uniform scale is replaced by the mean magnitude of its axes. The scale is fixed once
  // the bot exists.
  UPROPERTY(EditAnywhere, Category = "SuperDex Bot", BlueprintReadOnly)
  bool bAutoSpawn = true;

 public:
  DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnBotSpawned, USuperDexBotComponent*, BotComponent);

  // Emitted when the bot has been spawned.
  UPROPERTY(BlueprintAssignable, Category = "SuperDex Bot")
  FOnBotSpawned OnBotSpawned;

  // Contact events.
  UPROPERTY(BlueprintAssignable, Category = "SuperDex Bot")
  FOnMochiComponentBeginNestedContact OnMochiComponentBeginNestedContact;
  UPROPERTY(BlueprintAssignable, Category = "SuperDex Bot")
  FOnMochiComponentBeginContact OnMochiComponentBeginContact;
  UPROPERTY(BlueprintAssignable, Category = "SuperDex Bot")
  FOnMochiComponentEndNestedContact OnMochiComponentEndNestedContact;
  UPROPERTY(BlueprintAssignable, Category = "SuperDex Bot")
  FOnMochiComponentEndContact OnMochiComponentEndContact;
  UPROPERTY(BlueprintAssignable, Category = "SuperDex Bot")
  FOnMochiComponentNestedContact OnMochiComponentNestedContact;
  UPROPERTY(BlueprintAssignable, Category = "SuperDex Bot")
  FOnMochiComponentContact OnMochiComponentContact;
  UPROPERTY(BlueprintAssignable, Category = "SuperDex Bot")
  FOnMochiComponentContactProcessing OnMochiComponentBeginContactProcessing;
  UPROPERTY(BlueprintAssignable, Category = "SuperDex Bot")
  FOnMochiComponentContactProcessing OnMochiComponentEndContactProcessing;

 public:
  /**
   * Drive this component from a bot the loaded .superdex_scene already created, instead of
   * creating one of its own.
   *
   * Borrows the scene bot's articulated actor (with
   * bDestroyMochiActorOnObjectDestruction = false, since the BotScene owns it) so the normal tick
   * path poses this actor's skeletal mesh from it. The component remains a passive view: it does
   * not attach its level-authored controller to the scene-owned articulation.
   *
   * That posing is load-bearing for teleop, not just cosmetic. The clutch initializes by reading
   * bone transforms off the operator's arm poseable
   * (UClutchSkeletonRootToVrRootRetargeter::InitializeClutchBasedOnLeafBone), and only once that
   * poseable reports a cached pose. With nothing posing the mesh the clutch never initializes,
   * SetClutchEnabled stays a no-op, and the whole retargeter chain downstream of it goes silent.
   *
   * @return false if SceneBot is null or has no articulated actor, this component already has a
   * bot (bound or owned), there is no valid Mochi scene, or wrapping the articulated actor fails.
   * On failure the component is left as it was.
   */
  bool BindToExistingBot(superdex::robotics::Bot* SceneBot);

  /**
   * True once BindToExistingBot succeeded: this component poses a bot the BotScene owns rather
   * than one it created. Consumers that enumerate bots from both the level and the scene use this
   * to avoid counting the same physical bot twice.
   */
  UFUNCTION(BlueprintPure, Category = "SuperDex Bot")
  bool IsBorrowedBot() const {
    return bIsBorrowedBot;
  }

  /**
   * When bound to a scene bot, stop drawing this component's own skeletal mesh and let the
   * BotScene visualization render the bot from its Mochi collision geometry instead. The mesh
   * still poses while hidden -- the teleop clutch reads bone transforms off the retargeter
   * poseables, so posing must keep running regardless of visibility.
   */
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDex Bot")
  bool bHideMeshWhenBoundToSceneBot = true;

  // Create the bot now with its root at WorldFromRoot (scale ignored) and the given uniform scale,
  // which becomes the component's world scale. For components with bAutoSpawn disabled. Returns
  // false if the bot already exists or creation fails.
  UFUNCTION(BlueprintCallable, Category = "SuperDex Bot")
  bool SpawnBot(const FTransform& WorldFromRoot, float Scale = 1.f);
  // Whether the bot is created on the next game tick while it does not exist (see bAutoSpawn).
  UFUNCTION(BlueprintCallable, Category = "SuperDex Bot")
  void SetAutoSpawn(bool bInAutoSpawn) {
    bAutoSpawn = bInAutoSpawn;
  }
  // True once the bot exists in the Mochi scene.
  UFUNCTION(BlueprintPure, Category = "SuperDex Bot")
  bool IsBotSpawned() const;
  // The uniform scale the bot was spawned at, or 1 before it spawns.
  UFUNCTION(BlueprintPure, Category = "SuperDex Bot")
  float GetBotScale() const {
    return BotScale;
  }
  // Set the bot asset. This has no effect if the bot has already been spawned.
  UFUNCTION(BlueprintCallable, Category = "SuperDex Bot")
  void SetBotAsset(USuperDexBot* InBot);
  // Get the bot asset.
  UFUNCTION(BlueprintPure, Category = "SuperDex Bot")
  USuperDexBot* GetBotAsset() const;
  // Set the initial pose of the bot as an array of joint positions [rad or m].
  // Fails if the bot has already been spawned.
  UFUNCTION(BlueprintCallable, Category = "SuperDex Bot")
  bool SetInitialPose(const TArray<double>& NewInitialPose);
  // Get the initial pose of the bot as an array of joint positions [rad or m].
  UFUNCTION(BlueprintPure, Category = "SuperDex Bot")
  const TArray<double>& GetInitialPose() const;
  // Returns true if gravity is enabled on this bot.
  UFUNCTION(BlueprintPure, Category = "SuperDex Bot")
  bool HasGravity() const;
  // Get the underlying Mochi articulated actor object.
  UFUNCTION(BlueprintPure, Category = "SuperDex Bot")
  UMochiArticulatedActorObject* GetMochiArticulatedActor() const;
  // Set the world transform of the bot base.
  UFUNCTION(BlueprintCallable, Category = "SuperDex Bot")
  void SetWorldTransform(const FTransform& WorldTransform);
  // Teleport the bot to a new pose defined by joint positions.
  UFUNCTION(BlueprintCallable, Category = "SuperDex Bot")
  void SetPose(TArray<double> const& Pose);
  // Get the world transform of a link by name.
  UFUNCTION(BlueprintPure, Category = "SuperDex Bot")
  FTransform GetLinkToWorldTransform(FName LinkName) const;
  // Get the joint positions [rad or m], one per DOF, as of the latest simulation step. Empty until
  // the bot has been simulated.
  UFUNCTION(BlueprintPure, Category = "SuperDex Bot")
  const TArray<double>& GetCurrentJointPositions() const;
  // Get the joint velocities [rad/s or m/s], one per DOF, as of the latest simulation step. Empty
  // until the bot has been simulated.
  UFUNCTION(BlueprintPure, Category = "SuperDex Bot")
  const TArray<double>& GetCurrentJointVelocities() const;
  // Set the SuperDexBotController to a new one (nullptr to remove controller).
  UFUNCTION(BlueprintCallable, Category = "SuperDex Bot")
  bool SetController(TSubclassOf<USuperDexBotController> NewControllerClass);
  // Get the SuperDexBotController if it exists.
  UFUNCTION(BlueprintPure, Category = "SuperDex Bot")
  USuperDexBotController* GetController() const;
  // Returns true if contact queries and events are enabled on this bot.
  UFUNCTION(BlueprintPure, Category = "SuperDex Bot")
  bool GetEnableContacts() const;
  // Returns the links included in contact queries and events. Populated with all
  // links during spawn when left empty in the editor.
  UFUNCTION(BlueprintPure, Category = "SuperDex Bot")
  const TArray<FName>& GetContactGeneratingLinks() const;
  // Returns true if the bot's skin is included in contact queries and events.
  UFUNCTION(BlueprintPure, Category = "SuperDex Bot")
  bool GetSkinGeneratesContacts() const;
  // Returns true if contacts against the bot's own links should be filtered out.
  UFUNCTION(BlueprintPure, Category = "SuperDex Bot")
  bool GetFilterSelfContacts() const;

 public:
  // How a component world scale maps to a uniform bot scale (see bAutoSpawn).
  enum class EBotScaleFit : uint8 {
    Uniform, // Equal positive axes: the bot is created at that scale.
    NonUniform, // Unequal or mirrored axes: the bot is created at their mean magnitude.
    Degenerate, // A zero or invalid axis: no bot can be created.
  };
  // The uniform bot scale for a component world scale (the mean magnitude of its axes), and how
  // well it fits.
  static EBotScaleFit ResolveUniformBotScale(const FVector& ComponentScale, double& OutScale);

 public:
  USuperDexBotComponent();

 protected:
  virtual void BeginPlay() override;
  virtual void TickComponent(
      float DeltaTime,
      ELevelTick TickType,
      FActorComponentTickFunction* ThisTickFunction) override;
  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
  virtual void OnRegister() override;
  virtual void OnUnregister() override;
  virtual FBoxSphereBounds CalcBounds(const FTransform& LocalToWorld) const override;

 protected:
  void CreateBotAndArticulatedActor();
  void DestroyBotAndArticulatedActor();
  // Stamps this component's per-instance overrides onto the prefab copy that is handed to
  // CreateBot. Anything the prefab can spawn that carries a contact layer must be covered
  // here, or it silently keeps the layer baked into the bot asset.
  void ApplyPrefabOverrides(superdex::robotics::BotPrefab& InOutPrefab) const;
  bool OnArticulatedActorControl(
      FMochiArticulatedActorPoseVariant& OutPoseTarget,
      mochi::Actor* ArticulatedActor,
      mochi::Scene* Scene,
      double TimeStepSec) const;
  bool EnsureArrayProperties();
  void PoseMeshFromJointPositions(const TArray<double>& InJointPositions);
  void PoseMeshFromLinkWorldTransforms(const TArray<FTransform>& InLinkWorldTransforms);
  // Logs a warning listing any bot link names that have no matching bone on the
  // current skeletal mesh. Mochi drives the mesh purely by bone name, so links
  // without a matching bone are silently skipped and never posed. This enforces
  // the rule that an override mesh must name its bones after the bot's links.
  void WarnOnUnmatchedLinkBones(const TArray<FName>& LinkNames) const;
  // Puts back the Mochi user data that wrapping the borrowed bot overwrote. The scene outlives
  // the borrow and its readers reinterpret that pointer as a USceneComponent, so leaving this
  // component stamped on scene-owned actors outlives this UObject.
  void RestoreBorrowedActorUserData();

#if WITH_EDITOR
  virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
  void OnAssetReimported(UObject* ReimportedAsset);
  FDelegateHandle ReimportDelegateHandle;
#endif

 protected:
  superdex::robotics::Bot* BotPtr = nullptr;
  // True when BotPtr belongs to a BotScene: this component poses from it but must not destroy it.
  bool bIsBorrowedBot = false;
  // True when binding hid the mesh, so unbinding restores only what it changed.
  bool bHidMeshForSceneBot = false;
  // The uniform scale the bot was spawned at (see bAutoSpawn).
  float BotScale = 1.f;
  // Scene-owned actors whose Mochi user data the borrow overwrote, with the value to put back.
  TArray<TPair<mochi::ActorHandle, void*>> BorrowedActorUserData;
  UPROPERTY()
  TObjectPtr<UMochiArticulatedActorObject> ArticulatedActorObject = nullptr;
  std::atomic<bool> bArticulatedActorReady = false;
  FString LastBotCreationFailure;

  FMochiContactEventHandler ContactEventHandler;
  FMochiNestedContactEventHandler NestedContactEventHandler;
  TArray<FTransform> QueriedLinkTransforms;
  TArray<double> CurrentJointPositions;
  TArray<double> CurrentJointVelocities;

  UPROPERTY()
  TObjectPtr<USuperDexBotController> BotController = nullptr;
  mutable FCriticalSection BotControllerMutex;
  UFUNCTION()
  void HandleBotControllerOnBotSpawned(USuperDexBotComponent* InBot);
};
