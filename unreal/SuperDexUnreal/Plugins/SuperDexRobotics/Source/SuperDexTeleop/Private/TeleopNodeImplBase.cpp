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

#include "TeleopNodeImplBase.h"

void UTeleopNodeImplBase::SetName(const FString& Name) {
  NodeName = Name;
}

bool UTeleopNodeImplBase::Initialize() {
  unimplemented();
  return false;
}

void UTeleopNodeImplBase::Deinitialize() {
  unimplemented();
}

void UTeleopNodeImplBase::Pump() {
  unimplemented();
}

FTeleopPublisherHandle UTeleopNodeImplBase::CreatePublisher_DoubleArray(const FString& TopicName) {
  unimplemented();
  return FTeleopPublisherHandle();
}

FTeleopPublisherHandle UTeleopNodeImplBase::CreatePublisher_TransformArray(
    const FString& TopicName) {
  unimplemented();
  return FTeleopPublisherHandle();
}

FTeleopPublisherHandle UTeleopNodeImplBase::CreatePublisher_String(const FString& TopicName) {
  unimplemented();
  return FTeleopPublisherHandle();
}

FTeleopPublisherHandle UTeleopNodeImplBase::CreatePublisher_ByteArray(const FString& TopicName) {
  unimplemented();
  return FTeleopPublisherHandle();
}

FTeleopSubscriberHandle UTeleopNodeImplBase::CreateSubscriber_DoubleArray(
    const FString& TopicName,
    FDoubleArrayCallback Callback) {
  unimplemented();
  return FTeleopSubscriberHandle();
}

FTeleopSubscriberHandle UTeleopNodeImplBase::CreateSubscriber_TransformArray(
    const FString& TopicName,
    FTransformArrayCallback Callback) {
  unimplemented();
  return FTeleopSubscriberHandle();
}

FTeleopSubscriberHandle UTeleopNodeImplBase::CreateSubscriber_String(
    const FString& TopicName,
    FStringCallback Callback) {
  unimplemented();
  return FTeleopSubscriberHandle();
}

FTeleopSubscriberHandle UTeleopNodeImplBase::CreateSubscriber_ByteArray(
    const FString& TopicName,
    FByteArrayCallback Callback) {
  unimplemented();
  return FTeleopSubscriberHandle();
}

void UTeleopNodeImplBase::Publish_DoubleArray(
    FTeleopPublisherHandle Handle,
    const TArray<double>& Data) {
  unimplemented();
}

void UTeleopNodeImplBase::Publish_TransformArray(
    FTeleopPublisherHandle Handle,
    const TArray<FTransform>& Data) {
  unimplemented();
}

void UTeleopNodeImplBase::Publish_String(FTeleopPublisherHandle Handle, const FString& Data) {
  unimplemented();
}

void UTeleopNodeImplBase::Publish_ByteArray(
    FTeleopPublisherHandle Handle,
    const TArray<uint8>& Data) {
  unimplemented();
}
