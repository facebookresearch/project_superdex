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
#include "Framework/Docking/TabManager.h"

class FNotifyHook;
class IDetailsView;

// The tab setup the Mochi asset editors share: a viewport with a details panel to its right.
namespace MochiAssetEditorTabs {

// The default layout: the viewport tab takes 70% of the width, the details tab the rest.
TSharedRef<FTabManager::FLayout>
MakeDefaultLayout(FName LayoutName, FName ViewportTab, FName DetailsTab);

// Registers the viewport and details tab spawners in Group.
void RegisterTabSpawners(
    FTabManager& TabManager,
    const TSharedRef<FWorkspaceItem>& Group,
    FName ViewportTab,
    const FOnSpawnTab& SpawnViewport,
    FName DetailsTab,
    const FOnSpawnTab& SpawnDetails);

// A details view of Asset without a name area, reporting property edits to NotifyHook.
TSharedRef<IDetailsView> MakeDetailsView(UObject* Asset, FNotifyHook* NotifyHook);

} // namespace MochiAssetEditorTabs
