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

UENUM(BlueprintType)
enum class ESuperDexFireAndForgetExec : uint8 { Update, Finished };

UENUM(BlueprintType)
enum class ESuperDexFireAndForgetIterationExec : uint8 { Update, IterationFinished, Finished };

class FSuperDexFireAndForgetSecondsAction : public FPendingLatentAction {
 public:
  float Duration;

  float& Time;
  float& PercentComplete;
  ESuperDexFireAndForgetExec& Exec;

  FName ExecutionFunction;
  int32 OutputLink;
  FWeakObjectPtr CallbackTarget;

  bool bLastUpdatedTriggered = false;

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

  virtual void OnUpdate() {}
  virtual void OnFinalUpdate() {}
  virtual void OnFinished() {}

#if WITH_EDITOR
  // Returns a human readable description of the latent operation's current state
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

class FSuperDexFireAndForgetFramesAction : public FPendingLatentAction {
 public:
  const int Frames;

  int& Frame;
  ESuperDexFireAndForgetExec& Exec;

  FName ExecutionFunction;
  int32 OutputLink;
  FWeakObjectPtr CallbackTarget;

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
  // Returns a human readable description of the latent operation's current state
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

class FSuperDexFireAndForgetIterationsAction : public FPendingLatentAction {
 public:
  const int Iterations;
  const float IterationDuration;

  int& Iteration;
  float& IterationTime;
  float& IterationPercentComplete;
  float& Time;
  float& PercentComplete;
  ESuperDexFireAndForgetIterationExec& Exec;

  FName ExecutionFunction;
  int32 OutputLink;
  FWeakObjectPtr CallbackTarget;

  float NextIterationTime;
  bool bLastIterationUpdateTriggered = false;
  bool bTriggerFinish = false;
  bool bNextIteration = false;
  float Duration;

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
  // Returns a human readable description of the latent operation's current state
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

class FSuperDexFireAndForgetCurveAction : public FSuperDexFireAndForgetSecondsAction {
 public:
  UCurveFloat* Curve;
  float& Value;
  float MinTime;
  float MaxTime;

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
