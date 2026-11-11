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

#include "SuperDexBotActorFactory.h"

#include "SuperDexBot.h"
#include "SuperDexBotActor.h"

#define LOCTEXT_NAMESPACE "SuperDexBotActorFactory"

//------------------------------------------------------------------------------
// SUPERDEX BOT ACTOR FACTORY
//------------------------------------------------------------------------------

USuperDexBotActorFactory::USuperDexBotActorFactory(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer) {
  DisplayName = LOCTEXT("SuperDexBotActorDisplayName", "SuperDex Bot Actor");
  NewActorClass = ASuperDexBotActor::StaticClass();
  bUseSurfaceOrientation = true;
}

bool USuperDexBotActorFactory::CanCreateActorFrom(const FAssetData& AssetData, FText& OutErrorMsg) {
  if (!AssetData.IsValid() || !AssetData.IsInstanceOf(USuperDexBot::StaticClass())) {
    OutErrorMsg =
        NSLOCTEXT("CanCreateActor", "NoSuperDexBot", "A valid SuperDexBot must be specified.");
    return false;
  }
  return true;
}

void USuperDexBotActorFactory::PostSpawnActor(UObject* Asset, AActor* NewActor) {
  Super::PostSpawnActor(Asset, NewActor);
  USuperDexBot* Bot = CastChecked<USuperDexBot>(Asset);
  ASuperDexBotActor* Actor = CastChecked<ASuperDexBotActor>(NewActor);
  Actor->SetBotAsset(Bot);
}

UObject* USuperDexBotActorFactory::GetAssetFromActorInstance(AActor* Instance) {
  check(Instance->IsA(NewActorClass));
  ASuperDexBotActor* Actor = CastChecked<ASuperDexBotActor>(Instance);
  return Actor->GetBotAsset();
}

FQuat USuperDexBotActorFactory::AlignObjectToSurfaceNormal(
    const FVector& InSurfaceNormal,
    const FQuat& ActorRotation) const {
  return FindActorAlignmentRotation(ActorRotation, FVector(0.f, 0.f, 1.f), InSurfaceNormal);
}

#undef LOCTEXT_NAMESPACE
