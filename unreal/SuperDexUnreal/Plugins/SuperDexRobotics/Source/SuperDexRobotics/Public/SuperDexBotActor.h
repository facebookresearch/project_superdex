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
#include "GameFramework/Actor.h"
#include "SuperDexBotComponent.h"

#include "SuperDexBotActor.generated.h"

//------------------------------------------------------------------------------
// SUPERDEX BOT ACTOR
//------------------------------------------------------------------------------

UCLASS(
    Blueprintable,
    HideCategories = (Input),
    ShowCategories = ("Input|MouseInput", "Input|TouchInput"),
    ConversionRoot,
    ComponentWrapperClass,
    meta = (ChildCanTick))
class SUPERDEXROBOTICS_API ASuperDexBotActor : public AActor {
  GENERATED_BODY()

 private:
  UPROPERTY(
      Category = SuperDexBotActor,
      VisibleAnywhere,
      BlueprintReadOnly,
      meta =
          (ExposeFunctionCategories = "Mesh,Rendering,Components|SuperDexBotComponent",
           AllowPrivateAccess = "true"))
  TObjectPtr<class USuperDexBotComponent> BotComponent = nullptr;

 public:
  UFUNCTION(BlueprintCallable, Category = "SuperDexBotActor")
  void SetBotAsset(USuperDexBot* NewBot);

  UFUNCTION(BlueprintPure, Category = "SuperDexBotActor")
  USuperDexBot* GetBotAsset() const;

  UFUNCTION(BlueprintPure, Category = "SuperDexBotActor")
  USuperDexBotComponent* GetBotComponent() const;

 public:
  ASuperDexBotActor();
  virtual void Tick(float DeltaTime) override;

 protected:
  virtual void BeginPlay() override;

#if WITH_EDITOR
  virtual void CheckForErrors() override;
  virtual bool GetReferencedContentObjects(TArray<UObject*>& Objects) const override;
#endif

 private:
  static FName SuperDexBotComponentName;
};
