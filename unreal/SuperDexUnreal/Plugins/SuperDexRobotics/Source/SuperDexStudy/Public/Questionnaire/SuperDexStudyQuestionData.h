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
#include "Engine/DataAsset.h"

#include "SuperDexStudyQuestionData.generated.h"

/**
 * How a questionnaire question is answered: a binary response, a point on a discrete scale
 * (NumPoints points) or a value on a continuous scale.
 */
UENUM(BlueprintType)
enum class ESuperDexStudyQuestionResponseType : uint8 {
  BinaryResponse,
  DiscreteScale,
  ContinuousScale
};

/** A questionnaire question: its prompt, how it's answered, and its scale's end descriptors. */
UCLASS()
class SUPERDEXSTUDY_API USuperDexStudyQuestionData : public UPrimaryDataAsset {
  GENERATED_BODY()

 public:
  /** The question's prompt text. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex Robotics|Study")
  FString Prompt;

  /** How the question is answered. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex Robotics|Study")
  ESuperDexStudyQuestionResponseType ResponseType;

  /** Number of points on the discrete scale. */
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "SuperDex Robotics|Study",
      Meta =
          (EditCondition = "ResponseType == ESuperDexStudyQuestionResponseType::DiscreteScale",
           EditConditionHides))
  int32 NumPoints = 7;

  /** Descriptor for the scale's left end. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex Robotics|Study")
  FString LeftDescriptor;

  /** Descriptor for the scale's right end. */
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SuperDex Robotics|Study")
  FString RightDescriptor;
};
