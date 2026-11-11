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

#include "SuperDexBotActor.h"

#include "Logging/MessageLog.h"
#include "Misc/MapErrors.h"
#include "Misc/UObjectToken.h"

#include <mochi_core/utils/profile.h>

#define LOCTEXT_NAMESPACE "SuperDexBotActor"

//------------------------------------------------------------------------------
// SUPERDEX BOT ACTOR
//------------------------------------------------------------------------------

FName ASuperDexBotActor::SuperDexBotComponentName(TEXT("SuperDexBotComponent0"));

ASuperDexBotActor::ASuperDexBotActor() {
  PrimaryActorTick.bCanEverTick = true;
  BotComponent = CreateDefaultSubobject<USuperDexBotComponent>(SuperDexBotComponentName);
  RootComponent = BotComponent;
  bCanBeInCluster = (GetClass() == StaticClass());
}

void ASuperDexBotActor::BeginPlay() {
  Super::BeginPlay();
}

void ASuperDexBotActor::Tick(float DeltaTime) {
  Super::Tick(DeltaTime);
}

void ASuperDexBotActor::SetBotAsset(USuperDexBot* NewBot) {
  BotComponent->SetBotAsset(NewBot);
}

USuperDexBot* ASuperDexBotActor::GetBotAsset() const {
  return BotComponent->GetBotAsset();
}

USuperDexBotComponent* ASuperDexBotActor::GetBotComponent() const {
  return BotComponent;
}

#if WITH_EDITOR

void ASuperDexBotActor::CheckForErrors() {
  Super::CheckForErrors();
  FMessageLog MapCheck("MapCheck");
  if (!BotComponent) {
    FFormatNamedArguments Arguments;
    Arguments.Add(TEXT("ActorName"), FText::FromString(GetName()));
    MapCheck.Warning()
        ->AddToken(FUObjectToken::Create(this))
        ->AddToken(
            FTextToken::Create(
                FText::Format(
                    LOCTEXT(
                        "MapCheck_Message_ASuperDexBotActor",
                        "SuperDexBot actor {ActorName} has NULL BotComponent property - please delete"),
                    Arguments)));
  }
}

bool ASuperDexBotActor::GetReferencedContentObjects(TArray<UObject*>& Objects) const {
  Super::GetReferencedContentObjects(Objects);
  if (BotComponent) {
    Objects.Add(BotComponent);
    if (BotComponent->GetBotAsset()) {
      if (BotComponent->GetBotAsset()->GetSkeletalMesh()) {
        Objects.Add(BotComponent->GetBotAsset()->GetSkeletalMesh());
      }
    }
  }
  return true;
}

#endif

#undef LOCTEXT_NAMESPACE
