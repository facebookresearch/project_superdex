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
#include "TeleopNodeImplBase.h"

#include "TeleopNodeImplZenoh.generated.h"

class FZenohPublisher;
class FZenohSession;
class FZenohSubscriber;

/**
 * Zenoh communication backend for teleop. Select it with
 * CommunicationClassName = /Script/TeleopNodeZenoh.TeleopNodeImplZenoh.
 *
 * Topic "/teleop/<robot>/<topic>" maps to the zenoh key "teleop/<robot>/<topic>".
 * Double arrays are superdex.teleop.v1.DoubleArray
 * Transform arrays are superdex.teleop.v1.PoseArray
 * Strings are UTF-8 and byte arrays are raw
 * Poses enforce a FRU<->FLU and cm<->m conversion
 */
UCLASS()
class TELEOPNODEZENOH_API UTeleopNodeImplZenoh : public UTeleopNodeImplBase {
  GENERATED_BODY()

 public:
  virtual bool Initialize() override;
  virtual void Deinitialize() override;
  virtual void Pump() override;

  virtual FTeleopPublisherHandle CreatePublisher_DoubleArray(const FString& TopicName) override;
  virtual FTeleopPublisherHandle CreatePublisher_TransformArray(const FString& TopicName) override;
  virtual FTeleopPublisherHandle CreatePublisher_String(const FString& TopicName) override;
  virtual FTeleopPublisherHandle CreatePublisher_ByteArray(const FString& TopicName) override;

  virtual FTeleopSubscriberHandle CreateSubscriber_DoubleArray(
      const FString& TopicName,
      FDoubleArrayCallback Callback) override;
  virtual FTeleopSubscriberHandle CreateSubscriber_TransformArray(
      const FString& TopicName,
      FTransformArrayCallback Callback) override;
  virtual FTeleopSubscriberHandle CreateSubscriber_String(
      const FString& TopicName,
      FStringCallback Callback) override;
  virtual FTeleopSubscriberHandle CreateSubscriber_ByteArray(
      const FString& TopicName,
      FByteArrayCallback Callback) override;

  virtual void Publish_DoubleArray(FTeleopPublisherHandle Handle, const TArray<double>& Data)
      override;
  virtual void Publish_TransformArray(FTeleopPublisherHandle Handle, const TArray<FTransform>& Data)
      override;
  virtual void Publish_String(FTeleopPublisherHandle Handle, const FString& Data) override;
  virtual void Publish_ByteArray(FTeleopPublisherHandle Handle, const TArray<uint8>& Data) override;

 private:
  struct FPublisher {
    TSharedPtr<FZenohPublisher> Publisher;
    uint32 Seq = 0;
  };

  // Decodes one payload and calls the user callback; false if the payload didn't decode.
  using FDispatch = TFunction<bool(TConstArrayView<uint8>)>;

  struct FSubscriber {
    TSharedPtr<FZenohSubscriber> Subscriber;
    FDispatch Dispatch;
  };

  TSharedPtr<FZenohSession> Session;
  TMap<int32, FPublisher> Publishers;
  TMap<int32, FSubscriber> Subscribers;
  int32 NextHandleId = 0;
  TArray<uint8> EncodeBuffer;

  FTeleopPublisherHandle AddPublisher(const FString& TopicName, const FString& Encoding);
  FTeleopSubscriberHandle AddSubscriber(const FString& TopicName, FDispatch Dispatch);
  FPublisher* FindPublisher(FTeleopPublisherHandle Handle, const TCHAR* Caller);
};
