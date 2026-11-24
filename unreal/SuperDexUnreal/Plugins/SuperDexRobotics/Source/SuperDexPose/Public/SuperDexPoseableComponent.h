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

#include <atomic>

#include "Components/PoseableMeshComponent.h"
#include "SuperDexPoseSource.h"
#include "SuperDexPoseTypes.h"

#include "SuperDexPoseableComponent.generated.h"

class FSuperDexPoseCacheWriteScope;
class UMochiArticulationBase;
class USuperDexPoseSubsystem;

/**
 * Component for poseable objects in SuperDexPose.
 *
 * A PoseableMeshComponent that interacts with, and is posed by, the pose pipeline. Usually part of
 * a SuperDexPoseableActor, from which this component inherits some properties (e.g. handedness) and
 * gets pose, though could be used elsewhere if overrides are set. Usually used for hands, but could
 * be anything representable by skeletal mesh (e.g. robots).
 */
UCLASS(Blueprintable, meta = (BlueprintSpawnableComponent), HideCategories = ("HiddenVariables"))
class SUPERDEXPOSE_API USuperDexPoseableComponent : public UPoseableMeshComponent {
  GENERATED_BODY()

 public:
  /**
   * Overrides the PoseSource for this component. If set, the USuperDexPoseSubsystem will be asked
   * directly for a pose from this source. If the source is not valid, it will provide invalid
   * poses.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDexPoseableComponent")
  TSubclassOf<USuperDexPoseSource> PoseSourceOverride = nullptr;

  /**
   * SourcePoseable override for this component. If set, indicates that this component should
   * harvest its input pose from the provided USuperDexPoseableComponent if possible. If the
   * PoseableSource is set but not valid, it will provide invalid poses. If the
   * SourcePoseableOverride is nullptr this component will continue down the priority list looking
   * for a valid source.
   *
   * WARNING: The base SuperDexPoseableComponent does not cache its pose and directly pipes it
   * through. This means if you form a loop by setting this override to an ancestor/parent
   * component, you may enter an infinite loop if one of the components in the loop does not handle
   * caching pose each frame.
   */
  UPROPERTY(BlueprintReadWrite, Category = "SuperDexPoseableComponent")
  USuperDexPoseableComponent* SourcePoseableOverride = nullptr;

  /**
   * A substring used for disambiguation when multiple children sources are present. Can safely be
   * left blank if there is 0-1 children. Otherwise, when this component needs to GetPose from its
   * children it will do so using the first child matching the substring.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDexPoseableComponent")
  FString ChildSourceNameContainsSubstring = "";

  /**
   * If true, the component will be moved each tick such that its componentToRoot transform is
   * equal to the SkeletonRootToVrRoot of its output pose. This functionally ignores the parent
   * relative attachments of the component. If false, no updates to the component scene transform
   * will be made. The internal CachedVrRootToWorld will be updated with the correct current pose
   * and OutputPoseScaleOverride will be respected. Note that CachedOutputPose.SkeletonRootToVrRoot
   * may no longer be equal to this component's componentToRoot transform.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDexPoseableComponent")
  bool bUpdateComponentToRootTransformOnTick = true;

  /** Handedness override for this component. */
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "SuperDexPoseableComponent")
  ESuperDexHandedness HandednessOverride = ESuperDexHandedness::None;

  /**
   * If true, override the scale of the input SuperDexPose on recipt before doing anything with it.
   */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDexPoseableComponent")
  bool bOverrideInputPoseScale = false;

  /** Input Pose scale override value. */
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "SuperDexPoseableComponent",
      meta = (EditCondition = "bOverrideInputPoseScale == true", EditConditionHides))
  float InputPoseScaleOverride = 1.0f; // 0.74f

  /** If true, override the scale reported in the SuperDexPose output by this component. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDexPoseableComponent")
  bool bOverrideOutputPoseScale = true;

  /** Output Pose scale override value. */
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "SuperDexPoseableComponent",
      meta = (EditCondition = "bOverrideOutputPoseScale == true", EditConditionHides))
  float OutputPoseScaleOverride = 1.0f; // 0.74f

  /** If false, always return false and early out when GetPose() is called */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDexPoseableComponent")
  bool bEnableGetPose = true;

  /**
   * Only used while in editor, not play. If set and the SuperDexPoseAsset holds joint positions of
   * its bot, the cache will be set allowing the mesh to be posed. Otherwise, the mesh will remain
   * in its rest pose. Derived classes may override this pose based on their needs.
   */
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "SuperDexPoseableComponent",
      meta = (EditCondition = "bShow_EditorViewportPose == true", EditConditionHides))
  USuperDexPoseAsset* EditorViewportPose = nullptr;

  /**
   * This bool allows derived classes to hide the EditorViewportPose variable in the details panel.
   * It needs to be a UPROPERTY for the EditCondition to function properly.
   */
  UPROPERTY(VisibleAnywhere, Category = "HiddenVariables")
  bool bShow_EditorViewportPose = true;

  /**
   * Handedness of this component.
   * Determined by parent poseable actor unless override is set.
   * Should not be set in the details panel.
   */
  UPROPERTY(BlueprintReadOnly, Category = "SuperDexPoseableComponent")
  ESuperDexHandedness Handedness = ESuperDexHandedness::None;

 protected:
  /** Guards access to the local cache. */
  mutable FRWLock CacheLock;

  /** bool indicating that at least one valid pose has been put into the cache */
  bool bCacheInitialized = false;

  /**
   * Local cache for the output pose. The cache is by default updated each Tick() as part of
   * OnTickManageCache(). The PoseTransformer variant will update the cache when it invokes its
   * action to compute a transformed pose. The Mochi driven variants will update the cache on tick
   * based on the pose collected in the Mochi post step.
   */
  UPROPERTY(Replicated)
  FSuperDexPose CachedOutputPose;

  /**
   * A copy of the owning actor's localToWorld transform. This is done so that we have a thread
   * safe way to get the transform from a mochi thread.
   */
  FTransform CachedVrRootToWorld = FTransform::Identity;

 private:
  /** Transform from each bone to the root in the skinned asset's rest pose. */
  mutable TArray<FTransform> BoneToRestPoseRootTransforms;

  /** The bot and mesh MeshBonesOfLinks maps between. */
  TWeakObjectPtr<const UMochiArticulationBase> MeshBonesOfLinksBot;
  TWeakObjectPtr<const USkinnedAsset> MeshBonesOfLinksMesh;

  /** The mesh bone each link of MeshBonesOfLinksBot poses, or INDEX_NONE. */
  TArray<int32> MeshBonesOfLinks;

 public:
  /** Ticks in the editor and in TG_StartPhysics, and disables gravity. */
  USuperDexPoseableComponent();
  ~USuperDexPoseableComponent() override;

  /**
   * Get pose from parent poseable component, or directly from pose subsystem if override is set.
   *
   * @param OutPose The output pose. Unchanged if returning false.
   * @return true if successful, false otherwise.
   */
  UFUNCTION(
      BlueprintCallable,
      Category = "SuperDexPoseableComponent",
      meta = (ReturnDisplayName = "Success"))
  virtual bool GetPose(FSuperDexPose& OutPose);

  /**
   * Searches all children for a SuperDexPoseableComponent whose name contains
   * SourceChildNameContainsSubstring and returns the result. If the component exists and a
   * PoseSourceOverride and a SourcePoseableOverride are not set, then this will be the source used
   * by this component when it calls GetPose(). If the return value is nullptr then there is not a
   * valid ChildSource and this component will continue checking other input options.
   *
   * @return a pointer to a child that is providing the pose input to this class if it exists
   */
  USuperDexPoseableComponent* GetChildSourcePoseable() const;

  /**
   * Checks the different poseable source input options in priority order and returns the first one
   * that is valid. FSuperDexPoseSourcePtrTypePair contains both the type and a generic UObject
   * pointer to the source. The source must be cast to the correct type before calling GetPose().
   *
   * Priority order:
   * SuperDexPoseSubsystem
   * SuperDexPoseableComponent override
   * SuperDexPoseableComponent child
   * ASuperDexPoseableActor this owner
   * ASuperDexPlayerController from the pawn
   * no pose source found
   *
   * @return The source that this class uses for GetPose() inputs. It can be of various types
   * including nullptr.
   */
  FSuperDexPoseSourcePtrTypePair GetSourcePoseable() const;

  /**
   * Gets the current pose stored in the cache.
   * Note that this may not be the most up to date pose possible.
   *
   * @param OutPose The output pose.
   * @return true if successful, false otherwise.
   */
  UFUNCTION(
      BlueprintPure,
      Category = "SuperDexPoseableComponent",
      meta = (ReturnDisplayName = "Success"))
  virtual bool GetCachedPose(FSuperDexPose& OutPose) const;

  /**
   * Get the VrRoot pose in world space. The component updates its relative pose each tick so that
   * this is also the parent actor's world transform.
   *
   * @return The cached VrRootToWorld which should be equal to the parent actor's world transform at
   * time of capture.
   */
  UFUNCTION(BlueprintPure, Category = "SuperDexPoseableComponent")
  FTransform GetVrRootToWorld() const;

  /**
   * Get the transform from a bone to the root in the skinned asset's rest pose.
   *
   * @param BoneIndex An index into the UE skeleton
   * @return The transform from the chosen bone space to the rest pose root bone space
   */
  UFUNCTION(BlueprintPure, Category = "SuperDexPoseableComponent")
  FTransform GetBoneToRestPoseRootTransform(int BoneIndex) const;

  /**
   * Transform a local bone location to the root in the skinned asset's rest pose.
   *
   * @param BoneIndex An index into the UE skeleton
   * @param BoneLocalLocation A location in the bone space of the bone indicated by BoneIndex
   * @return The location relative to the root bone in rest pose that corresponds with the location
   * in the chosen bone space while in rest pose.
   */
  UFUNCTION(BlueprintPure, Category = "SuperDexPoseableComponent")
  FVector TransformRestPoseBoneLocationToRoot(int BoneIndex, const FVector& BoneLocalLocation)
      const;

  /**
   * Get the world transform of a link of the cached pose's bot: its transform relative to the
   * skeleton root at the cached pose's joint positions, placed by this component's transform (where
   * the mesh shows the skeleton root). Game thread only.
   *
   * @param LinkName The name of a link of the cached pose's bot
   * @param OutLinkToWorld The link's world transform. Unchanged if returning false.
   * @return false if there is no cached pose on a bot, or its bot has no such link
   */
  UFUNCTION(
      BlueprintCallable,
      Category = "SuperDexPoseableComponent",
      meta = (ReturnDisplayName = "Success"))
  bool GetCachedPoseLinkTransform(FName LinkName, FTransform& OutLinkToWorld) const;

  /**
   * Get the transform of very tip of the finger(this is not the fingerpad): the finger's tip link
   * on the OpenXR hand skeleton (xr_05_thumb_tip, ...) in the cached pose
   * (GetCachedPoseLinkTransform). Poses without that link use the mesh's bone or socket of that
   * name, and meshes without it return the component transform.
   *
   * @param Finger the finger to get the tip from
   * @return The transform of the tip of the corresponding finger
   */
  UFUNCTION(BlueprintPure, Category = "SuperDex")
  FTransform GetFingerTipTransform(ESuperDexFinger Finger) const;

 protected:
  virtual void OnRegister() override;
  virtual void BeginPlay() override;
#if WITH_EDITOR
  virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
#endif
  /**
   * Outside a game world, caches EditorViewportPose as the output pose for the editor preview when
   * its bone count matches the mesh.
   */
  void WhenNotPlayingInitPoseCache();

  /**
   * Applies the cached output pose to the mesh's BoneSpaceTransforms, directly posing the
   * UPoseableMeshComponent bones. Called each tick after OnTickManageCache updates the cache.
   *
   * Each link of the pose's bot poses the mesh bone of the same name (or, if no link names a bone
   * of the mesh and the mesh has one bone per link, the bone of the same index) with its transform
   * relative to its parent link at the pose's joint positions. The bones take the link rotations,
   * and also the link translations if the mesh is the bot's own mesh (other meshes keep their
   * proportions). A pose that is not on a bot leaves the mesh unchanged.
   */
  void ApplyCachedPoseToMesh();

  /**
   * TickComponent by default will call GetPose() to ensure a recent pose and possibly cache have
   * been generated. Derived classes should have a minimalistic, pass through, or appropriately
   * cached GetPose() function. It then calls OnTickManageCache() passing in the results of
   * GetPose() to ensure the cache is properly updated as needed. Then ApplyCachedPoseToMesh()
   * applies the cached pose to BoneSpaceTransforms. Finally it calls
   * OnTickUpdateComponentActorRelativeTransformFromCache() to update the component transform.
   * For proper visual animations it is important that derived classes update their
   * CachedOutputPose each tick either through GetPose or OnTickManageCache().
   */
  virtual void TickComponent(
      float DeltaTime,
      enum ELevelTick TickType,
      FActorComponentTickFunction* ThisTickFunction) override;

  /**
   * Provides general handling of cache state on tick and should likely be overridden by any derived
   * class that wants to manage its own cache differently. This function is called on UE Tick(). By
   * default, this function updates the cache with the provided pose if it was valid. The
   * PoseTransformer variant does nothing in this function because the cache is updated by
   * GetPose(). The Mochi variants ignore the input pose (because GetPose returns the cache) and
   * utilize a Async buffer variable instead.
   *
   * @param Pose The pose to cache.
   * @param bIsValid Whether the pose is valid.
   */
  virtual void OnTickManageCache(FSuperDexPose& Pose, bool bIsValid);

  /**
   * Updates the component's local transform such that its transform relative to its owning actor
   * (ASuperDexPoseableActor) matches the cached SkeletonRootToVrRoot. The owning actor world space
   * transform is considered the VrRoot. When the ASuperDexPoseableActor is attached to a pawn it is
   * placed at the pawn's VrRoot component. This ensures skeleton bones are placed correctly in the
   * world and to prevent multiple poseable components, attached in a chain, having their transforms
   * "stack" (apply additively to each other). This function is called in the tick function and is
   * not thread safe and must be called on the UE main thread.
   */
  void OnTickUpdateComponentActorRelativeTransformFromCache();

  /**
   * Checks to see if GetInputPoseFromSource() has been reentered on this thread, aka. called again
   * while within the scope of a previous call. This indicates we have formed an infinite loop and
   * need to return false. If a loop is detected, the function will set Pose.bIsInInfiniteLoop to
   * true, Log an error, and return true. If no loop is detected, the function will return false.
   * This function uses a thread_local variable to track the current thread's call stack.
   */
  bool CheckIfInInfiniteLoop(FSuperDexPose& Pose);

  /**
   * Whether the calling thread holds CacheLock for writing through an FSuperDexPoseCacheWriteScope.
   * Reads of the cache on that thread take no lock: CacheLock is not reentrant, and the thread
   * already has the cache to itself.
   */
  bool IsCacheWriterThread() const;

  /**
   * An inner function that figures out by connection what the source of the pose should be and
   * calls GetPose() on it. Derived classes that are not modifying the pose between the input and
   * output should be calling Super::GetPose().
   *
   * @param InputPose The pose provided from the input source. Unchanged if returning false.
   * @return true if successful, false otherwise.
   */
  bool GetInputPoseFromSource(FSuperDexPose& InputPose);

  /**
   * An internal list of child poseable components to exclude from the GetChildSourcePoseable().
   * This is predominantly expected to be used by derived classes that want to spawn additional
   * children for debug visualization or other reasons. These added components are not added by the
   * user in the hierarchy and so we should not expect users to know they are added and compensate.
   */
  TSet<USuperDexPoseableComponent*> ChildSourcePoseableExcludeList;

 private:
  friend class FSuperDexPoseCacheWriteScope;

  /** The thread holding CacheLock through an FSuperDexPoseCacheWriteScope, or 0. */
  mutable std::atomic<uint32> CacheWriterThreadId{0};

  /**
   * Initialize BoneToRestPoseRootTransforms if not already initialized.
   *
   * @return true if already initialized or successfully initialized, false otherwise.
   */
  bool InitBoneToRestPoseRootTransforms() const;

  /** The mesh bone each link of Bot poses (see ApplyCachedPoseToMesh), cached in MeshBonesOfLinks.
   */
  const TArray<int32>& GetMeshBonesOfLinks(const UMochiArticulationBase& Bot);
};

/**
 * Holds a poseable component's CacheLock for writing, and marks the calling thread as its writer,
 * while in scope. Use it to write the cache while computing the pose that goes in it: that can run
 * code that reads the cache, such as Blueprint or C++ overrides and delegates, and those reads
 * (GetCachedPose and everything built on it, such as GetCachedPoseLinkTransform) then read the
 * cache directly instead of waiting on the lock their own thread holds.
 */
class SUPERDEXPOSE_API FSuperDexPoseCacheWriteScope {
 public:
  explicit FSuperDexPoseCacheWriteScope(const USuperDexPoseableComponent& InComponent);
  ~FSuperDexPoseCacheWriteScope();
  UE_NONCOPYABLE(FSuperDexPoseCacheWriteScope);

 private:
  const USuperDexPoseableComponent& Component;
};
