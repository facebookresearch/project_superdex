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

#include "Questionnaire/SuperDexStudyQuestionData.h"

#include "Components/TextRenderComponent.h"
#include "CoreMinimal.h"
#include "GameFramework/Actor.h"

#include "SuperDexStudyQuestionPrompt.generated.h"

/** Broadcast with the participant's response to the question. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(
    FSuperDexStudyQuestionResponseDelegate,
    float,
    Response);

/**
 * Actor that presents a questionnaire question and broadcasts the participant's response.
 * Blueprints implement UpdateInterface to show QuestionData.
 */
UCLASS(Blueprintable)
class SUPERDEXSTUDY_API ASuperDexStudyQuestionPrompt : public AActor {
  GENERATED_BODY()

 public:
  /** The question to present. Setting it through SetQuestionData also calls UpdateInterface. */
  UPROPERTY(EditAnywhere, BlueprintSetter = SetQuestionData)
  USuperDexStudyQuestionData* QuestionData = nullptr;

  /** Broadcast with the participant's response. */
  UPROPERTY(BlueprintAssignable, BlueprintCallable, Category = "Events")
  FSuperDexStudyQuestionResponseDelegate OnResponse;

  /**
   * Sets QuestionData and calls UpdateInterface.
   *
   * @param Data The question to present.
   */
  UFUNCTION(BlueprintSetter)
  void SetQuestionData(USuperDexStudyQuestionData* Data);

  /** Enables ticking, which prompt Blueprints use for Event Tick. */
  ASuperDexStudyQuestionPrompt();

 protected:
  /** Refreshes the prompt for QuestionData. The native implementation does nothing. */
  UFUNCTION(BlueprintCallable, BlueprintNativeEvent)
  void UpdateInterface();
};
