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

#include "SuperDexRecorderWorldSubsystem.h"

#include "SuperDexRecorder.h"
#include "SuperDexRecorderModule.h"

bool USuperDexRecorderWorldSubsystem::Register(ASuperDexRecorder* Recorder) {
  Recorders.RemoveAll(
      [](const TWeakObjectPtr<ASuperDexRecorder>& Registered) { return !Registered.IsValid(); });
  Recorders.AddUnique(Recorder);
  if (Recorders.Num() == 1) {
    ActiveRecorder = Recorder;
    return true;
  }

  TArray<FString> Others;
  for (const TWeakObjectPtr<ASuperDexRecorder>& Registered : Recorders) {
    if (Registered.Get() != ActiveRecorder.Get()) {
      Others.Add(Registered->GetActorNameOrLabel());
    }
  }
  UE_LOG(
      LogSuperDexRecorder,
      Error,
      TEXT("A world records with one SuperDex Recorder, but %s has %d: %s records; %s won't."),
      *GetWorld()->GetMapName(),
      Recorders.Num(),
      ActiveRecorder.IsValid() ? *ActiveRecorder->GetActorNameOrLabel() : TEXT("none"),
      *FString::Join(Others, TEXT(", ")));
  return false;
}

void USuperDexRecorderWorldSubsystem::Unregister(const ASuperDexRecorder* Recorder) {
  Recorders.RemoveAll([Recorder](const TWeakObjectPtr<ASuperDexRecorder>& Registered) {
    return !Registered.IsValid() || Registered.Get() == Recorder;
  });
  if (ActiveRecorder.Get() == Recorder) {
    ActiveRecorder.Reset();
  }
}

ASuperDexRecorder* USuperDexRecorderWorldSubsystem::GetActiveRecorder() const {
  return ActiveRecorder.Get();
}

bool USuperDexRecorderWorldSubsystem::DoesSupportWorldType(const EWorldType::Type WorldType) const {
  return WorldType == EWorldType::Game || WorldType == EWorldType::PIE;
}
