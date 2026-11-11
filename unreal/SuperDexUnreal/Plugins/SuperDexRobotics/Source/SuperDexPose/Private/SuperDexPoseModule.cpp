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
    const UStruct* structDefinition,
    void* structData,
    TCHAR const* filePath) {
  // Load file
  FString jsonStr;
  if (!FFileHelper::LoadFileToString(jsonStr, filePath)) {
    UE_LOG(
        LogSuperDexPose, Warning, TEXT("[%hs] Failed to load file [%s]"), __FUNCTION__, filePath);
    return false;
  }

  // Parse json
  TSharedPtr<FJsonObject> jsonObj = MakeShareable(new FJsonObject());
  TSharedRef<TJsonReader<>> jsonReader = TJsonReaderFactory<>::Create(jsonStr);
  if (!FJsonSerializer::Deserialize(jsonReader, jsonObj) || !jsonObj.IsValid()) {
    UE_LOG(
        LogSuperDexPose,
        Warning,
        TEXT("[%hs] Failed while loading [%s]: unable to parse json [%s]"),
        __FUNCTION__,
        filePath,
        *jsonStr);
    return false;
  }
  if (!FJsonObjectConverter::JsonObjectToUStruct(
          jsonObj.ToSharedRef(), structDefinition, structData, 0, 0)) {
    UE_LOG(
        LogSuperDexPose,
        Warning,
        TEXT("[%hs] Failed while loading [%s]: unable to convert to struct"),
        __FUNCTION__,
        filePath);
    return false;
  }

  return true;
}

bool FSuperDexPoseModule::SaveToJsonFile(
    const UStruct* structDefinition,
    const void* structData,
    TCHAR const* filePath) {
  TSharedRef<FJsonObject> jsonObj = MakeShareable(new FJsonObject());
  if (!FJsonObjectConverter::UStructToJsonObject(structDefinition, structData, jsonObj, 0, 0)) {
    UE_LOG(
        LogSuperDexPose,
        Warning,
        TEXT("[%hs] Failed to create json object for file [%s]"),
        __FUNCTION__,
        filePath);
    return false;
  }

  FString jsonStr;
  TSharedRef<TJsonWriter<>> jsonWriter = TJsonWriterFactory<>::Create(&jsonStr, 0);
  if (!FJsonSerializer::Serialize(jsonObj, jsonWriter)) {
    UE_LOG(
        LogSuperDexPose,
        Warning,
        TEXT("[%hs] Failed to create json string for file [%s]"),
        __FUNCTION__,
        filePath);
    return false;
  }

  if (!FFileHelper::SaveStringToFile(jsonStr, filePath)) {
    UE_LOG(
        LogSuperDexPose,
        Warning,
        TEXT("[%hs] Failed to write json file [%s]"),
        __FUNCTION__,
        filePath);
    return false;
  }

  return true;
}

void FSuperDexPoseModule::StartupModule() {
  UE_LOG(LogSuperDexPose, Display, TEXT("[%hs]"), __FUNCTION__);
}

void FSuperDexPoseModule::ShutdownModule() {
  UE_LOG(LogSuperDexPose, Display, TEXT("[%hs]"), __FUNCTION__);
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FSuperDexPoseModule, SuperDexPose)
