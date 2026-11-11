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

#include "MochiAssetEditorTabs.h"

#include "IDetailsView.h"
#include "Modules/ModuleManager.h"
#include "PropertyEditorModule.h"
#include "Styling/AppStyle.h"

#define LOCTEXT_NAMESPACE "MochiAssetEditorTabs"

namespace MochiAssetEditorTabs {

TSharedRef<FTabManager::FLayout>
MakeDefaultLayout(FName LayoutName, FName ViewportTab, FName DetailsTab) {
  auto MakeSingleTabStack = [](FName Tab, float WidthFraction) {
    return FTabManager::NewStack()
        ->AddTab(Tab, ETabState::OpenedTab)
        ->SetHideTabWell(true)
        ->SetSizeCoefficient(WidthFraction);
  };
  return FTabManager::NewLayout(LayoutName)
      ->AddArea(
          FTabManager::NewPrimaryArea()
              ->SetOrientation(Orient_Horizontal)
              ->Split(MakeSingleTabStack(ViewportTab, 0.7f))
              ->Split(MakeSingleTabStack(DetailsTab, 0.3f)));
}

void RegisterTabSpawners(
    FTabManager& TabManager,
    const TSharedRef<FWorkspaceItem>& Group,
    FName ViewportTab,
    const FOnSpawnTab& SpawnViewport,
    FName DetailsTab,
    const FOnSpawnTab& SpawnDetails) {
  const FName StyleSet = FAppStyle::GetAppStyleSetName();
  TabManager.RegisterTabSpawner(ViewportTab, SpawnViewport)
      .SetDisplayName(LOCTEXT("ViewportTab", "Viewport"))
      .SetIcon(FSlateIcon(StyleSet, "LevelEditor.Tabs.Viewports"))
      .SetGroup(Group);
  TabManager.RegisterTabSpawner(DetailsTab, SpawnDetails)
      .SetDisplayName(LOCTEXT("DetailsTab", "Details"))
      .SetIcon(FSlateIcon(StyleSet, "LevelEditor.Tabs.Details"))
      .SetGroup(Group);
}

TSharedRef<IDetailsView> MakeDetailsView(UObject* Asset, FNotifyHook* NotifyHook) {
  FDetailsViewArgs Args;
  Args.NameAreaSettings = FDetailsViewArgs::HideNameArea;
  Args.bHideSelectionTip = true;
  Args.NotifyHook = NotifyHook;
  TSharedRef<IDetailsView> DetailsView =
      FModuleManager::GetModuleChecked<FPropertyEditorModule>("PropertyEditor")
          .CreateDetailView(Args);
  DetailsView->SetObject(Asset);
  return DetailsView;
}

} // namespace MochiAssetEditorTabs

#undef LOCTEXT_NAMESPACE
