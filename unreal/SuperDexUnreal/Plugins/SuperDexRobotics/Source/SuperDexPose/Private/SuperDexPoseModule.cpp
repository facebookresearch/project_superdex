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

#include "SuperDexPoseModule.h"
#include "Skeletons/SuperDexSkeletonBotsLifetime.h"

#include "JsonObjectConverter.h"

DEFINE_LOG_CATEGORY(LogSuperDexPose);

#define LOCTEXT_NAMESPACE "FSuperDexPoseModule"

void FSuperDexPoseModule::SetTrackingSpaceToHandParentSpaceTransform(const FTransform& Transform) {
  TrackingSpaceToHandParentSpaceTransform = Transform;
}

FTransform FSuperDexPoseModule::GetTrackingSpaceToHandParentSpaceTransform() const {
  return TrackingSpaceToHandParentSpaceTransform;
}

bool FSuperDexPoseModule::LoadFromJsonFile(
    const UStruct* StructDefinition,
    void* StructData,
    TCHAR const* FilePath) {
  // Load file
  FString JsonStr;
  if (!FFileHelper::LoadFileToString(JsonStr, FilePath)) {
    UE_LOG(
        LogSuperDexPose, Warning, TEXT("[%hs] Failed to load file [%s]"), __FUNCTION__, FilePath);
    return false;
  }

  // Parse json
  TSharedPtr<FJsonObject> JsonObj = MakeShareable(new FJsonObject());
  TSharedRef<TJsonReader<>> JsonReader = TJsonReaderFactory<>::Create(JsonStr);
  if (!FJsonSerializer::Deserialize(JsonReader, JsonObj) || !JsonObj.IsValid()) {
    UE_LOG(
        LogSuperDexPose,
        Warning,
        TEXT("[%hs] Failed while loading [%s]: unable to parse json [%s]"),
        __FUNCTION__,
        FilePath,
        *JsonStr);
    return false;
  }
  if (!FJsonObjectConverter::JsonObjectToUStruct(
          JsonObj.ToSharedRef(), StructDefinition, StructData, 0, 0)) {
    UE_LOG(
        LogSuperDexPose,
        Warning,
        TEXT("[%hs] Failed while loading [%s]: unable to convert to struct"),
        __FUNCTION__,
        FilePath);
    return false;
  }

  return true;
}

bool FSuperDexPoseModule::SaveToJsonFile(
    const UStruct* StructDefinition,
    const void* StructData,
    TCHAR const* FilePath) {
  TSharedRef<FJsonObject> JsonObj = MakeShareable(new FJsonObject());
  if (!FJsonObjectConverter::UStructToJsonObject(StructDefinition, StructData, JsonObj, 0, 0)) {
    UE_LOG(
        LogSuperDexPose,
        Warning,
        TEXT("[%hs] Failed to create json object for file [%s]"),
        __FUNCTION__,
        FilePath);
    return false;
  }

  FString JsonStr;
  TSharedRef<TJsonWriter<>> JsonWriter = TJsonWriterFactory<>::Create(&JsonStr, 0);
  if (!FJsonSerializer::Serialize(JsonObj, JsonWriter)) {
    UE_LOG(
        LogSuperDexPose,
        Warning,
        TEXT("[%hs] Failed to create json string for file [%s]"),
        __FUNCTION__,
        FilePath);
    return false;
  }

  if (!FFileHelper::SaveStringToFile(JsonStr, FilePath)) {
    UE_LOG(
        LogSuperDexPose,
        Warning,
        TEXT("[%hs] Failed to write json file [%s]"),
        __FUNCTION__,
        FilePath);
    return false;
  }

  return true;
}

void FSuperDexPoseModule::StartupModule() {
  UE_LOG(LogSuperDexPose, Display, TEXT("[%hs]"), __FUNCTION__);
  CreateSkeletonBots();
}

void FSuperDexPoseModule::ShutdownModule() {
  UE_LOG(LogSuperDexPose, Display, TEXT("[%hs]"), __FUNCTION__);
  DestroySkeletonBots();
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FSuperDexPoseModule, SuperDexPose)
