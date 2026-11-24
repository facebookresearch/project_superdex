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
#include "MochiSceneObject.h"
#include "SuperDexBotController.h"

#include <superdex_recorder/physics_recorder.h>
#include <superdex_recorder/recorder.h>
#include <superdex_recorder/robot_shadow.h>
#include <superdex_recorder/robotics_recorder.h>
#include <superdex_recorder/skeleton_shadow.h>
#include <superdex_recorder/source_recorder.h>

#include "HAL/CriticalSection.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class UObject;
class USuperDexBotComponent;

/**
 * A keypoint skeleton that the application provides, such as an operator's tracked hand. Robots
 * themselves are recorded from their bots and need no source.
 */
struct FSuperDexSkeletonSource {
  /**
   * Optional. When set, the skeleton is recorded as a shadow of that robot (grouped with it under
   * "world/shadows/", e.g. the hand that drives it); a robot that isn't recorded, or no longer
   * exists, skips the skeleton with a warning. When never set, the skeleton is recorded on its own
   * under "world/sources/<Id>".
   */
  TWeakObjectPtr<const USuperDexBotComponent> Robot;
  /**
   * Identifies this shadow among the robot's shadows (the recorder already uses "command"), or,
   * without a Robot, this source among the episode's sources.
   */
  FString Id;
  /** Labels the skeleton in the viewer; empty uses Id. */
  FString Label;
  /** Names each keypoint, in the order SampleWorldKeypoints writes them. */
  TArray<FString> KeypointNames;
  /** Each keypoint's parent keypoint index, which must be lower than its own, or -1 for none. */
  TArray<int32> ParentIndices;
  /** Draws the skeleton in the viewer. */
  FColor Color = [] {
    const auto Rgba = superdex::recorder::SkeletonShadowConfig{}.colorRgba;
    return FColor(Rgba[0], Rgba[1], Rgba[2], Rgba[3]);
  }();
  /**
   * Called on the Mochi thread after every step until EndCapture returns. Writes one finite Mochi
   * world position per keypoint and returns true, or returns false while the skeleton is untracked,
   * which hides it.
   */
  TFunction<bool(TArrayView<mochi::Real3>)> SampleWorldKeypoints;
};

/**
 * A world transform, such as a headset or an object that has no Mochi physics, recorded under
 * "world/sources/<Id>".
 */
struct FSuperDexTransformSource {
  /** Identifies this source among the episode's sources. */
  FString Id;
  /**
   * Called on the Mochi thread after every step until EndCapture returns. Writes the Mochi world
   * transform and returns true, or returns false while the source is untracked, which records
   * nothing for that step.
   */
  TFunction<bool(mochi::TransformRT&)> SampleWorldTransform;
};

/**
 * A robot's most recent command target, for a robot that no USuperDexBotController drives. Read on
 * the Mochi thread after each step and recorded as that robot's "command" shadow, like
 * USuperDexBotController::GetRecordingTarget.
 */
class ISuperDexRecordingTargetSource {
 public:
  virtual ~ISuperDexRecordingTargetSource() = default;

  /**
   * Gets the robot's most recent command target.
   *
   * @param OutAnchorLinkIndex Set to the index of the link the target is anchored to.
   * @param OutWorldFromAnchor Set to the anchor link's target world transform.
   * @param OutJointPositions Set to the target joint positions.
   * @return False while there is no target to record.
   */
  virtual bool GetRecordingTarget(
      int32& OutAnchorLinkIndex,
      mochi::TransformRT& OutWorldFromAnchor,
      TArray<mochi::real>& OutJointPositions) const = 0;
};

/** Thread-safe shared pointer to an ISuperDexRecordingTargetSource. */
using FSuperDexRecordingTargetSourcePtr =
    TSharedPtr<const ISuperDexRecordingTargetSource, ESPMode::ThreadSafe>;

/** What an FSuperDexSceneRecorder episode records, and where. */
struct FSuperDexSceneRecorderConfig {
  /**
   * The Mochi scene to record. Never set, it's the SuperDexPhysics game-instance scene. Only robots
   * whose articulation lives in this scene are recorded. Recording several scenes means one
   * recorder, and so one episode, per scene.
   */
  TWeakObjectPtr<UMochiSceneObject> Scene;
  /** Where the episode is written; see ResolveOutputDirectory. */
  FString OutputDirectory;
  /**
   * The scene archive whose filename and hash the episode stores, so the scene can be rebuilt
   * offline. Required unless bRequireSceneArchive is false.
   */
  FString SceneArchivePath;
  /** Refuse to record without SceneArchivePath. */
  bool bRequireSceneArchive = true;
  /** The scene task the episode records, stored with it. */
  FString TaskPath;
  /** Stored with the episode, e.g. the participant, condition or task. */
  TMap<FString, FString> Properties;
  /** Records Mochi's contact forces directly, at a cost per contact point in every step. */
  bool bRecordContactForces = superdex::recorder::PhysicsRecorderConfig{}.recordContactForces;
  /** The task spawn name of each Mochi actor that a scene task spawned. */
  TArray<TPair<mochi::ActorHandle, FString>> SpawnNames;
  /**
   * Skeletons recorded as shadows of recorded robots, or on their own when they have no robot.
   * Sources of robots that aren't recorded are ignored.
   */
  TArray<FSuperDexSkeletonSource> SkeletonSources;
  /** World transforms recorded as sources. */
  TArray<FSuperDexTransformSource> TransformSources;
  /** The command-target source of each robot that no USuperDexBotController drives. */
  TArray<TPair<mochi::ActorHandle, FSuperDexRecordingTargetSourcePtr>> RecordingTargetSources;
};

/** Records one Mochi scene episode without buffering through Unreal types. */
class SUPERDEXRECORDER_API FSuperDexSceneRecorder final {
 public:
  FSuperDexSceneRecorder();
  /**
   * Aborts an episode that has not been committed.
   * Waits for an earlier asynchronous commit to finish.
   */
  ~FSuperDexSceneRecorder();

  /** Begins an episode of Config. Returns false if one is already active or it cannot start. */
  bool BeginEpisode(UObject* WorldContext, const FSuperDexSceneRecorderConfig& Config);
  /**
   * Stops capturing Mochi steps. Queued events and scalars are written at the last recorded step
   * while the scene and capture are still valid, and are dropped otherwise.
   */
  void EndCapture();
  /**
   * Waits for an earlier CommitEpisodeAsync, then writes this episode's file before returning.
   * Ends capture and writes the episode. Returns false if no episode is active, capture failed or
   * writing fails.
   *
   * @return True if this episode was committed; false otherwise.
   */
  bool CommitEpisodeSync();
  /**
   * Writes the file in the background; the next commit and the recorder's destruction wait for
   * it.
   * Ends capture before starting the commit. Later write failures are logged by the native
   * recorder; a true result does not guarantee that the file will be published.
   *
   * @return True if the commit started; false if no episode is active, capture failed or the
   * native commit reported an error before returning.
   */
  bool CommitEpisodeAsync();
  /** Ends capture and discards the episode, recording Reason. */
  void AbortEpisode(FStringView Reason);

  /** Whether an episode is in progress, and its ID and output file. */
  bool IsRecording() const;
  FString GetEpisodeId() const;
  FString GetOutputPath() const;

  /**
   * The absolute directory episodes are written to for a config's OutputDirectory: empty means
   * <Project>/Saved/SuperDexRecorder.
   */
  static FString ResolveOutputDirectory(const FString& OutputDirectory);

  /**
   * Records Text under "data/events/<Stream>" at the next Mochi step. Callable from any thread
   * while capturing; ignored otherwise. Texts queued for one stream within one step share its
   * sample. Texts and scalars still queued when capture ends are written at the last recorded step.
   */
  void RecordEvent(FStringView Stream, FStringView Text);
  /**
   * Records Value under "data/scalars/<Name>" at the next Mochi step. Callable from any thread
   * while capturing; ignored otherwise. When a name is queued more than once in one step, the last
   * call wins.
   */
  void RecordScalar(FStringView Name, double Value);
  /**
   * Records Values as one sample of "data/scalars/<Name>", like RecordScalar. The recording doesn't
   * fix the list's length: keep it consistent per name and interpret the values when reading.
   */
  void RecordScalars(FStringView Name, TConstArrayView<double> Values);

 private:
  struct FRecordedSkeleton {
    size_t RobotIndex = 0;
    superdex::recorder::SkeletonShadowConfig Config;
    TFunction<bool(TArrayView<mochi::Real3>)> SampleWorldKeypoints;
    TArray<mochi::Real3> Keypoints;
    std::unique_ptr<superdex::recorder::SkeletonShadow> Shadow;
  };

  struct FRecordedSourceSkeleton {
    std::string Id;
    superdex::recorder::SourceSkeletonConfig Config;
    TFunction<bool(TArrayView<mochi::Real3>)> SampleWorldKeypoints;
    TArray<mochi::Real3> Keypoints;
  };

  struct FRecordedTransform {
    std::string Id;
    TFunction<bool(mochi::TransformRT&)> SampleWorldTransform;
  };

  void InitializeOnMochiThread(mochi::Scene& Scene);
  void InitializeRoboticsRecorders(mochi::Scene& Scene, mochi::Error& Error);
  void InitializeSourceRecorder(mochi::Error& Error);
  void RecordPostStep(const mochi::StepInfo& StepInfo);
  void RecordSources(const superdex::recorder::SampleTime& Time, mochi::Error& Error);
  void RecordQueuedValues(const superdex::recorder::SampleTime& Time, mochi::Error& Error);
  void SetCaptureError(const mochi::Error& Error);
  bool CommitEpisode(bool bAsync);

  TWeakObjectPtr<UMochiSceneObject> SceneObject;
  mochi::CallbackHandle PostStepCallback;
  superdex::robotics::RoboticsContext* RoboticsContext = nullptr;

  /** Kept across episodes, so that later episodes reuse the chunk buffers of earlier ones. */
  std::unique_ptr<superdex::recorder::Recorder> Recorder;
  FString RecorderOutputDirectory;
  std::unique_ptr<superdex::recorder::Episode> Episode;
  std::unique_ptr<superdex::recorder::Episode> CommittingEpisode;
  superdex::recorder::PhysicsRecorderConfig PhysicsConfig;
  std::unique_ptr<superdex::recorder::PhysicsRecorder> PhysicsRecorder;
  std::vector<std::unique_ptr<superdex::recorder::RoboticsRecorder>> RoboticsRecorders;
  std::vector<superdex::robotics::BotHandle> BotHandles;
  std::vector<std::unique_ptr<superdex::recorder::RobotShadow>> RobotShadows;
  std::vector<FRecordedSkeleton> RecordedSkeletons;
  std::vector<FRecordedSourceSkeleton> RecordedSourceSkeletons;
  std::vector<FRecordedTransform> RecordedTransforms;
  std::unique_ptr<superdex::recorder::SourceRecorder> SourceRecorder;
  /** Guards the queues and bAcceptingQueuedValues, which any thread may use. */
  FCriticalSection QueueLock;
  TArray<TPair<FString, FString>> QueuedEvents;
  TMap<FString, TArray<double>> QueuedScalars;
  bool bAcceptingQueuedValues = false;
  TArray<mochi::ActorHandle> RobotActorHandles;
  TArray<TWeakObjectPtr<USuperDexBotController>> RobotControllers;
  TArray<FSuperDexRecordingTargetSourcePtr> RobotTargetSources;
  TArray<mochi::real> SourceJointPositions;
  uint64_t NextStateFrame = 1;

  /**
   * Frame, simulation time and wall-clock time of the last recorded physics state. What is sampled
   * after the next step (commands, skeletons, sources, events) is recorded at this time, so each
   * frame has one wall time in every stream.
   */
  superdex::recorder::SampleTime LastStateTime{};
  bool bCaptureActive = false;
  bool bRoboticsInitialized = false;
  FString CaptureError;
  FString EpisodeId;
  FString OutputPath;
};
