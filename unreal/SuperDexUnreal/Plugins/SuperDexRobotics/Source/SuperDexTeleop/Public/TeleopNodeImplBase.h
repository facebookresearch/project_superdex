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
#include "TeleopTypes.h"
#include "UObject/Object.h"

#include "TeleopNodeImplBase.generated.h"

//------------------------------------------------------------------------------
// CALLBACK TYPES
//------------------------------------------------------------------------------

using FDoubleArrayCallback = TFunction<void(const TArray<double>&)>;
using FTransformArrayCallback = TFunction<void(const TArray<FTransform>&)>;
using FStringCallback = TFunction<void(const FString&)>;
using FByteArrayCallback = TFunction<void(const TArray<uint8>&)>;

//------------------------------------------------------------------------------
// BASE IMPLEMENTATION CLASS
//------------------------------------------------------------------------------

UCLASS(Abstract)
class SUPERDEXTELEOP_API UTeleopNodeImplBase : public UObject {
  GENERATED_BODY()

 public:
  // Lifecycle
  virtual void SetName(const FString& Name);
  virtual bool Initialize();
  virtual void Deinitialize();

  // Called each tick to process incoming messages
  virtual void Pump();

  // Publisher creation
  virtual FTeleopPublisherHandle CreatePublisher_DoubleArray(const FString& TopicName);
  virtual FTeleopPublisherHandle CreatePublisher_TransformArray(const FString& TopicName);
  virtual FTeleopPublisherHandle CreatePublisher_String(const FString& TopicName);
  virtual FTeleopPublisherHandle CreatePublisher_ByteArray(const FString& TopicName);

  // Subscriber creation
  virtual FTeleopSubscriberHandle CreateSubscriber_DoubleArray(
      const FString& TopicName,
      FDoubleArrayCallback Callback);
  virtual FTeleopSubscriberHandle CreateSubscriber_TransformArray(
      const FString& TopicName,
      FTransformArrayCallback Callback);
  virtual FTeleopSubscriberHandle CreateSubscriber_String(
      const FString& TopicName,
      FStringCallback Callback);
  virtual FTeleopSubscriberHandle CreateSubscriber_ByteArray(
      const FString& TopicName,
      FByteArrayCallback Callback);

  // Publishing
  virtual void Publish_DoubleArray(FTeleopPublisherHandle Handle, const TArray<double>& Data);
  virtual void Publish_TransformArray(
      FTeleopPublisherHandle Handle,
      const TArray<FTransform>& Data);
  virtual void Publish_String(FTeleopPublisherHandle Handle, const FString& Data);
  virtual void Publish_ByteArray(FTeleopPublisherHandle Handle, const TArray<uint8>& Data);

 protected:
  FString NodeName;
};
