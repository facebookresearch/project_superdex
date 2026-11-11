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

#include "Commandlets/Commandlet.h"
#include "CoreMinimal.h"

#include "SuperDexBotSynchronization.generated.h"

class USuperDexBot;
struct FAssetData;

/// Static utility class for checking and resolving sync status of imported SuperDexBot assets.
/// Used by the Mochi menu "Reimport Bots" action and the ReimportSuperDexBots commandlet.
class FSuperDexBotSyncUtils {
 public:
  /// Discovers all USuperDexBot assets in the project and plugins via the Asset Registry.
  static TArray<FAssetData> FindAllBotAssets();

  /// Loads all USuperDexBot assets and returns those whose source files have changed since
  /// last import. Calls USuperDexBot::IsOutOfSync() on each, which computes SHA1 hashes of
  /// the source .superdex_bot file and all referenced assets (.glb, .mochi.h5) and compares
  /// against the stored hash.
  static TArray<USuperDexBot*> FindOutOfSyncBots();

  /// Reimports the given bots via FReimportManager. Returns the number successfully
  /// reimported. The reimport pipeline has a hash-skip optimization, so calling this on
  /// an already-synced bot is safe (no-op).
  static int32 ReimportOutOfSyncBots(const TArray<USuperDexBot*>& Bots);

  /// Displays a modal dialog listing out-of-sync bot names and source paths, with
  /// "Reimport All" and "Dismiss" buttons. If the user clicks "Reimport All", calls
  /// ReimportOutOfSyncBots() and returns the count reimported. Returns 0 if dismissed.
  static int32 ShowOutOfSyncDialog(const TArray<USuperDexBot*>& OutOfSyncBots);
};

/// Commandlet that reimports all out-of-sync SuperDexBot assets without opening the editor GUI.
/// Invoke via: UnrealEditor-Cmd.exe <project.uproject>
/// -run=SuperDexRoboticsEditor.ReimportSuperDexBots
UCLASS()
class UReimportSuperDexBotsCommandlet : public UCommandlet {
  GENERATED_BODY()
 public:
  UReimportSuperDexBotsCommandlet();
  virtual int32 Main(const FString& Params) override;
};
