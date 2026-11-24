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
#include "Curves/CurveFloat.h"
#include "Engine/LatentActionManager.h"
#include "LatentActions.h"
#include "SuperDexPoseUtils.h"
#include "UObject/WeakObjectPtr.h"

#include "SuperDexLatentActions.generated.h"

/**
 * Output execution pin of the fire-and-forget latent actions: Update on each update and Finished
 * once at the end.
 */
UENUM(BlueprintType)
enum class ESuperDexFireAndForgetExec : uint8 { Update, Finished };

/**
 * Output execution pin of the iteration latent action: Update on each update, IterationFinished at
 * the end of each iteration and Finished once all iterations are done.
 */
UENUM(BlueprintType)
enum class ESuperDexFireAndForgetIterationExec : uint8 { Update, IterationFinished, Finished };

/** Latent action of USuperDexPoseFunctionLibrary::FireAndForgetSeconds. */
class FSuperDexFireAndForgetSecondsAction : public FPendingLatentAction {
 public:
  /** Run time [s]. */
  float Duration;

  /** The Blueprint outputs, written on each update. */
  float& Time;
  float& PercentComplete;
  ESuperDexFireAndForgetExec& Exec;

  /** The Blueprint callback, from the call's FLatentActionInfo. */
  FName ExecutionFunction;
  int32 OutputLink;
  FWeakObjectPtr CallbackTarget;

  /** Whether the final Update has fired; Finished fires on the next update. */
  bool bLastUpdatedTriggered = false;

  /**
   * Keeps references to the Blueprint outputs, which must outlive the action, and resets Time and
   * PercentComplete to 0.
   */
  FSuperDexFireAndForgetSecondsAction(
      const float InDuration,
      float& InTime,
      float& InPercentComplete,
      ESuperDexFireAndForgetExec& InExec,
      const FLatentActionInfo& LatentInfo)
      : Duration(InDuration),
        Time(InTime),
        PercentComplete(InPercentComplete),
        Exec(InExec),
        ExecutionFunction(LatentInfo.ExecutionFunction),
        OutputLink(LatentInfo.Linkage),
        CallbackTarget(LatentInfo.CallbackTarget) {
    Time = 0;
    PercentComplete = 0;
  }

  virtual void UpdateOperation(FLatentResponse& Response) override {
    Time += Response.ElapsedTime();
    const bool bAtTime = (Time >= Duration);
    const bool bFinished = (bAtTime && bLastUpdatedTriggered);
    if (bFinished) {
      Time = Duration;
      PercentComplete = 1.0f;
      OnFinished();
      Exec = ESuperDexFireAndForgetExec::Finished;
      Response.FinishAndTriggerIf(true, ExecutionFunction, OutputLink, CallbackTarget);
    } else if (bAtTime) {
      bLastUpdatedTriggered = true;
      Time = Duration;
      PercentComplete = 1.0f;
      OnFinalUpdate();
      Exec = ESuperDexFireAndForgetExec::Update;
      Response.TriggerLink(ExecutionFunction, OutputLink, CallbackTarget);
    } else {
      PercentComplete = Time / Duration;
      OnUpdate();
      Exec = ESuperDexFireAndForgetExec::Update;
      Response.TriggerLink(ExecutionFunction, OutputLink, CallbackTarget);
    }
  }

  /** Hooks for subclasses, called before an Update, the final Update and Finished fire. */
  virtual void OnUpdate() {}
  virtual void OnFinalUpdate() {}
  virtual void OnFinished() {}

#if WITH_EDITOR
  /** Returns a human readable description of the latent operation's current state */
  virtual FString GetDescription() const override {
    static const FNumberFormattingOptions TimeFormatOptions =
        FNumberFormattingOptions().SetMinimumFractionalDigits(3).SetMaximumFractionalDigits(3);
    return FText::Format(
               NSLOCTEXT(
                   "FSuperDexFireAndForgetSecondsAction",
                   "FireAndForgetSecondsActionTimeFmt",
                   "FireAndForget ({0} percent complete)"),
               FText::AsNumber(PercentComplete, &TimeFormatOptions))
        .ToString();
  }
#endif
};

/** Latent action of USuperDexPoseFunctionLibrary::FireAndForgetFrames. */
class FSuperDexFireAndForgetFramesAction : public FPendingLatentAction {
 public:
  /** Number of frames to run. */
  const int Frames;

  /** The Blueprint outputs, written on each update. */
  int& Frame;
  ESuperDexFireAndForgetExec& Exec;

  /** The Blueprint callback, from the call's FLatentActionInfo. */
  FName ExecutionFunction;
  int32 OutputLink;
  FWeakObjectPtr CallbackTarget;

  /**
   * Keeps references to the Blueprint outputs, which must outlive the action, and sets Frame to -1
   * until the first update.
   */
  FSuperDexFireAndForgetFramesAction(
      int InFrames,
      int& InFrame,
      ESuperDexFireAndForgetExec& InExec,
      const FLatentActionInfo& LatentInfo)
      : Frames(InFrames),
        Frame(InFrame),
        Exec(InExec),
        ExecutionFunction(LatentInfo.ExecutionFunction),
        OutputLink(LatentInfo.Linkage),
        CallbackTarget(LatentInfo.CallbackTarget) {
    Frame = -1;
  }

  virtual void UpdateOperation(FLatentResponse& Response) override {
    ++Frame;
    bool bComplete = Frame == Frames;
    if (bComplete) {
      Frame = Frames - 1;
      Exec = ESuperDexFireAndForgetExec::Finished;
      Response.FinishAndTriggerIf(true, ExecutionFunction, OutputLink, CallbackTarget);
    } else {
      Exec = ESuperDexFireAndForgetExec::Update;
      Response.TriggerLink(ExecutionFunction, OutputLink, CallbackTarget);
    }
  }

#if WITH_EDITOR
  /** Returns a human readable description of the latent operation's current state */
  virtual FString GetDescription() const override {
    return FText::Format(
               NSLOCTEXT(
                   "FSuperDexFireAndForgetFramesAction",
                   "FireAndForgetFramesAction",
                   "FireAndForget (Frame {0} of {1})"),
               FText::AsNumber(Frame + 1),
               FText::AsNumber(Frames))
        .ToString();
  }
#endif
};

/** Latent action of USuperDexPoseFunctionLibrary::FireAndForgetIterations. */
class FSuperDexFireAndForgetIterationsAction : public FPendingLatentAction {
 public:
  /** Number of iterations and the length of each [s]. */
  const int Iterations;
  const float IterationDuration;

  /** The Blueprint outputs, written on each update. */
  int& Iteration;
  float& IterationTime;
  float& IterationPercentComplete;
  float& Time;
  float& PercentComplete;
  ESuperDexFireAndForgetIterationExec& Exec;

  /** The Blueprint callback, from the call's FLatentActionInfo. */
  FName ExecutionFunction;
  int32 OutputLink;
  FWeakObjectPtr CallbackTarget;

  /** Iteration and finish bookkeeping between updates. */
  float NextIterationTime;
  bool bLastIterationUpdateTriggered = false;
  bool bTriggerFinish = false;
  bool bNextIteration = false;
  float Duration;

  /**
   * Keeps references to the Blueprint outputs, which must outlive the action, and resets the
   * numeric progress outputs (iteration, times and fractions) to 0; Exec is not assigned until the
   * first update.
   */
  FSuperDexFireAndForgetIterationsAction(
      int InIterations,
      float InIterationDuration,
      int& InIteration,
      float& InIterationTime,
      float& InIterationPercentComplete,
      float& InTime,
      float& InPercentComplete,
      ESuperDexFireAndForgetIterationExec& InExec,
      const FLatentActionInfo& LatentInfo)
      : Iterations(InIterations),
        IterationDuration(InIterationDuration),
        Iteration(InIteration),
        IterationTime(InIterationTime),
        IterationPercentComplete(InIterationPercentComplete),
        Time(InTime),
        PercentComplete(InPercentComplete),
        Exec(InExec),
        ExecutionFunction(LatentInfo.ExecutionFunction),
        OutputLink(LatentInfo.Linkage),
        CallbackTarget(LatentInfo.CallbackTarget) {
    Iteration = 0;
    IterationTime = 0;
    IterationPercentComplete = 0;
    Time = 0;
    PercentComplete = 0;
    NextIterationTime = InIterationDuration;
    Duration = IterationDuration * Iterations;
  }

  virtual void UpdateOperation(FLatentResponse& Response) override {
    if (bTriggerFinish) {
      Exec = ESuperDexFireAndForgetIterationExec::Finished;
      Response.FinishAndTriggerIf(true, ExecutionFunction, OutputLink, CallbackTarget);
      return;
    }
    if (bNextIteration) {
      ++Iteration;
      IterationTime = 0;
      IterationPercentComplete = 0;
      bNextIteration = false;
    }
    Time += Response.ElapsedTime();
    IterationTime += Response.ElapsedTime();
    const bool bIterationAtTime = (IterationTime >= IterationDuration);
    const bool bIterationFinished = (bIterationAtTime && bLastIterationUpdateTriggered);
    if (bIterationFinished) {
      IterationTime = IterationDuration;
      IterationPercentComplete = 1.0f;
      Time = NextIterationTime;
      PercentComplete = Time / Duration;
      NextIterationTime += IterationDuration;
      bNextIteration = true;
      bLastIterationUpdateTriggered = false;
      IterationTime = IterationDuration;
      IterationPercentComplete = 1.0;
      bTriggerFinish = (Iteration == Iterations - 1);
      Exec = ESuperDexFireAndForgetIterationExec::IterationFinished;
      Response.TriggerLink(ExecutionFunction, OutputLink, CallbackTarget);
    } else if (bIterationAtTime) {
      bLastIterationUpdateTriggered = true;
      IterationTime = IterationDuration;
      IterationPercentComplete = 1.0f;
      Time = NextIterationTime;
      PercentComplete = Time / Duration;
      Exec = ESuperDexFireAndForgetIterationExec::Update;
      Response.TriggerLink(ExecutionFunction, OutputLink, CallbackTarget);
    } else {
      IterationPercentComplete = IterationTime / IterationDuration;
      PercentComplete = Time / Duration;
      Exec = ESuperDexFireAndForgetIterationExec::Update;
      Response.TriggerLink(ExecutionFunction, OutputLink, CallbackTarget);
    }
  }

#if WITH_EDITOR
  /** Returns a human readable description of the latent operation's current state */
  virtual FString GetDescription() const override {
    static const FNumberFormattingOptions TimeFormatOptions =
        FNumberFormattingOptions().SetMinimumFractionalDigits(3).SetMaximumFractionalDigits(3);
    return FText::Format(
               NSLOCTEXT(
                   "FSuperDexFireAndForgetIterationsAction",
                   "FireAndForgetIterationsActionTimeFmt",
                   "FireAndForget ({0} percent complete)"),
               FText::AsNumber(PercentComplete, &TimeFormatOptions))
        .ToString();
  }
#endif
};

/** Latent action of USuperDexPoseFunctionLibrary::FireAndForgetCurve. */
class FSuperDexFireAndForgetCurveAction : public FSuperDexFireAndForgetSecondsAction {
 public:
  /** The sampled curve. */
  UCurveFloat* Curve;

  /** The Blueprint output, set to the curve value on each update. */
  float& Value;

  /** The curve's time range. */
  float MinTime;
  float MaxTime;

  /**
   * Keeps references to the Blueprint outputs, which must outlive the action, and runs for the
   * curve's time range divided by InSpeed. InCurve must not be null.
   */
  FSuperDexFireAndForgetCurveAction(
      UCurveFloat* InCurve,
      float InSpeed,
      float& InValue,
      float& InTime,
      float& InPercentComplete,
      ESuperDexFireAndForgetExec& InExec,
      const FLatentActionInfo& LatentInfo)
      : FSuperDexFireAndForgetSecondsAction(0, InTime, InPercentComplete, InExec, LatentInfo),
        Curve(InCurve),
        Value(InValue) {
    Curve->GetTimeRange(MinTime, MaxTime);
    Duration = (MaxTime - MinTime) / InSpeed;
  }

  virtual void OnUpdate() override {
    Value = Curve->GetFloatValue(MinTime + PercentComplete * (MaxTime - MinTime));
  }

  virtual void OnFinalUpdate() override {
    Value = Curve->GetFloatValue(MaxTime);
  }

  virtual void OnFinished() override {
    Value = Curve->GetFloatValue(MaxTime);
  }
};
