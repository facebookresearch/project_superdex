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

#include "ActorFactories/ActorFactory.h"
#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "UObject/ObjectMacros.h"

#include "SuperDexBotActorFactory.generated.h"

//------------------------------------------------------------------------------
// SUPERDEX BOT ACTOR FACTORY
//------------------------------------------------------------------------------

UCLASS(MinimalAPI, Config = Editor, CollapseCategories, HideCategories = Object)
class USuperDexBotActorFactory : public UActorFactory {
  GENERATED_UCLASS_BODY()
  virtual bool CanCreateActorFrom(const FAssetData& AssetData, FText& OutErrorMsg) override;
  virtual void PostSpawnActor(UObject* Asset, AActor* NewActor) override;
  virtual UObject* GetAssetFromActorInstance(AActor* ActorInstance) override;
  virtual FQuat AlignObjectToSurfaceNormal(
      const FVector& InSurfaceNormal,
      const FQuat& ActorRotation) const override;
};
