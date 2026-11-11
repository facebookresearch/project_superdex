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

#include "MochiForceVisualizerComponent.h"

#include "MochiPhysicsModule.h"
#include "MochiRigidMeshComponent.h"
#include "MochiSoftMeshComponent.h"

#include "Materials/Material.h"
#include "UObject/ConstructorHelpers.h"

#include <mochi_core/utils/profile.h>

UMochiForceVisualizerComponent::UMochiForceVisualizerComponent() {
  PrimaryComponentTick.bCanEverTick = true;
  PrimaryComponentTick.TickGroup = TG_PostPhysics;
  // initialize material
  static ConstructorHelpers::FObjectFinder<UMaterial> VizMaterialAsset(TEXT(
      "/Script/Engine.Material'/SuperDexPhysics/Materials/M_MochiForceVisualizer.M_MochiForceVisualizer'"));
  if (VizMaterialAsset.Succeeded()) {
    // NB: virtual function call in constructor; explicitly call ULineSetComponent impl
    ULineSetComponent::SetMaterial(0, VizMaterialAsset.Object);
  }
}

void UMochiForceVisualizerComponent::BeginPlay() {
  MOCHI_PROFILE_SCOPE();
  Super::BeginPlay();
  // TODO: generalize using IMochiMeshComponentInterface if possible
  const auto parent = GetAttachParent();
  // UMochiSoftMeshComponent parent
  if (const auto dynMeshComp = Cast<UMochiSoftMeshComponent>(parent)) {
    if (!dynMeshComp->GetContactsEnabled()) {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT(
              "[%hs] Force visualizer attached to %s but contact queries are not enabled; visualization will not render!"),
          __FUNCTION__,
          *dynMeshComp->GetName());
      return;
    }
    dynMeshComp->OnMochiComponentBeginContact.AddDynamic(
        this, &UMochiForceVisualizerComponent::OnMochiComponentBeginContact);
    dynMeshComp->OnMochiComponentEndContact.AddDynamic(
        this, &UMochiForceVisualizerComponent::OnMochiComponentEndContact);
    dynMeshComp->OnMochiComponentContact.AddDynamic(
        this, &UMochiForceVisualizerComponent::OnMochiComponentContact);
  }
  // UMochiRigidMeshComponent parent
  else if (const auto statMeshComp = Cast<UMochiRigidMeshComponent>(parent)) {
    if (!statMeshComp->GetContactsEnabled()) {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT(
              "[%hs] Force visualizer attached to %s but contact queries are not enabled; visualization will not render!"),
          __FUNCTION__,
          *UMochiRigidMeshComponent::StaticClass()->GetName());
      return;
    }
    statMeshComp->OnMochiComponentBeginContact.AddDynamic(
        this, &UMochiForceVisualizerComponent::OnMochiComponentBeginContact);
    statMeshComp->OnMochiComponentEndContact.AddDynamic(
        this, &UMochiForceVisualizerComponent::OnMochiComponentEndContact);
    statMeshComp->OnMochiComponentContact.AddDynamic(
        this, &UMochiForceVisualizerComponent::OnMochiComponentContact);
  }
  // try using reflection on everything else, assuming the component copies
  // the patterns set forth by the components above
  else {
    const TCHAR* onBeginContactPropName = TEXT("OnMochiComponentBeginContact");
    const TCHAR* onEndContactPropName = TEXT("OnMochiComponentEndContact");
    const TCHAR* onContactPropName = TEXT("OnMochiComponentContact");
    const TCHAR* enableContactsPropName = TEXT("bEnableContacts");

    const UClass* parentClass = parent->GetClass();
    const FMulticastDelegateProperty* onBeginContactProp = CastField<FMulticastDelegateProperty>(
        parentClass->FindPropertyByName(FName(onBeginContactPropName)));
    const FMulticastDelegateProperty* onEndContactProp = CastField<FMulticastDelegateProperty>(
        parentClass->FindPropertyByName(FName(onEndContactPropName)));
    const FMulticastDelegateProperty* onContactProp = CastField<FMulticastDelegateProperty>(
        parentClass->FindPropertyByName(FName(onContactPropName)));
    if (onBeginContactProp && onEndContactProp && onContactProp) {
      const FBoolProperty* bEnableContactsProp =
          CastField<FBoolProperty>(parentClass->FindPropertyByName(FName(enableContactsPropName)));
      bool parentEnableContacts = false;
      if (bEnableContactsProp) {
        const uint8* address =
            reinterpret_cast<const uint8*>(parent) + bEnableContactsProp->GetOffset_ForInternal();
        parentEnableContacts = bEnableContactsProp->GetPropertyValue(address);
      }
      if (bEnableContactsProp && parentEnableContacts == false) {
        UE_LOG(
            LogMochiPhysics,
            Warning,
            TEXT(
                "[%hs] Force visualizer attached to compatible component but contact queries are not enabled; visualization will not render!"),
            __FUNCTION__);
        return;
      }
      auto Bind = [this, parent](
                      const FMulticastDelegateProperty* PropToBindTo, FName ForceVizCompFuncName) {
        FScriptDelegate binding;
        binding.BindUFunction(this, ForceVizCompFuncName);
        PropToBindTo->AddDelegate(MoveTemp(binding), parent);
      };
      Bind(onBeginContactProp, FName(onBeginContactPropName));
      Bind(onEndContactProp, FName(onEndContactPropName));
      Bind(onContactProp, FName(onContactPropName));
    }
    // The parent exposes no compatible contact events.
    else {
      UE_LOG(
          LogMochiPhysics,
          Warning,
          TEXT(
              "[%hs] Component not attached to a compatible component exposing the Mochi contact events named: "
              "%s, %s, and %s, or is missing a bool property named %s. You can make your custom Mochi component "
              "compatible by copying the patterns used in %s, %s, etc."),
          __FUNCTION__,
          onBeginContactPropName,
          onEndContactPropName,
          onContactPropName,
          enableContactsPropName,
          *UStaticMeshComponent::StaticClass()->GetName(),
          *UMochiSoftMeshComponent::StaticClass()->GetName());
    }
  }
}

void UMochiForceVisualizerComponent::TickComponent(
    float DeltaTime,
    ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction) {
  MOCHI_PROFILE_SCOPE();
  Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
  // clear lines
  Clear();
  if (bEnabled) {
    // iterate map of mochi components
    for (auto& pair : ContactsMap) {
      auto& contacts = pair.Value;
      // add a line for each contact force
      for (int iLine = 0; iLine < contacts.Num(); ++iLine) {
        const auto force = contacts[iLine].Force.Size();
        const auto forceNormal = contacts[iLine].Force.GetSafeNormal();
        const auto forceLength =
            FMath::GetMappedRangeValueUnclamped(ForceMinMax, LineLengthMinMax, force);
        const auto worldStart = contacts[iLine].Location;
        const auto worldEnd = worldStart + forceNormal * forceLength;
        const auto localStart = GetComponentTransform().InverseTransformPosition(worldStart);
        const auto localEnd = GetComponentTransform().InverseTransformPosition(worldEnd);
        const auto color = GetColorFromForce(force);
        AddLine(localStart, localEnd, color, 0);
      }
    }
  } else {
    ContactsMap.Reset();
  }
}

void UMochiForceVisualizerComponent::OnMochiComponentBeginContact(
    USceneComponent* MochiComponent,
    AActor* OtherMochiActor,
    USceneComponent* OtherMochiComponent,
    const TArray<FMochiContactPoint>& Contacts) {
  if (bEnabled) {
    if (ContactsMap.Contains(MochiComponent)) {
      ContactsMap[MochiComponent] = Contacts;
    } else {
      ContactsMap.Add(MochiComponent, Contacts);
    }
  }
}

void UMochiForceVisualizerComponent::OnMochiComponentEndContact(
    USceneComponent* MochiComponent,
    AActor* OtherMochiActor,
    USceneComponent* OtherMochiComponent) {
  if (bEnabled) {
    if (ContactsMap.Contains(MochiComponent)) {
      ContactsMap[MochiComponent].Reset();
    }
  }
}

void UMochiForceVisualizerComponent::OnMochiComponentContact(
    USceneComponent* MochiComponent,
    const TArray<FMochiContactPoint>& Contacts) {
  if (bEnabled) {
    if (ContactsMap.Contains(MochiComponent)) {
      ContactsMap[MochiComponent] = Contacts;
    } else {
      ContactsMap.Add(MochiComponent, Contacts);
    }
  }
}

FColor UMochiForceVisualizerComponent::GetColorFromForce(float Force) const {
  // Remap min/max pressure to [0,1]
  Force = FMath::GetMappedRangeValueClamped(ForceMinMax, {0.0f, 1.0f}, Force);
  // Optionally quantize to discrete color bands
  float t = Force;
  if (NumColorBands) {
    const int band = int(t * (NumColorBands - 0.0001f));
    t = band / (NumColorBands - 1.0f);
  }

  if (ColorMode == EMochiForceVisualizerColorMode::TwoTone) {
    return FMath::Lerp(LowColor, HighColor, t).ToFColor(true);
  } else /* EMochiForceVisualizerColorMode::Rainbow */ {
    // The Turbo colormap (sRGB bytes) by Anton Mikhailov, Google LLC. Copyright 2019 Google LLC,
    // licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
    // Source: https://gist.github.com/mikhailov-work/6a308c20e494d9e0ccc29036b28faa7a
    // See thirdparty_licenses/NOTICE.md in this plugin.
    // clang-format off
    static unsigned char turbo_srgb[256][3] = {{48,18,59},{50,21,67},{51,24,74},{52,27,81},{53,30,88},{54,33,95},{55,36,102},{56,39,109},{57,42,115},{58,45,121},{59,47,128},{60,50,134},{61,53,139},{62,56,145},{63,59,151},{63,62,156},{64,64,162},{65,67,167},{65,70,172},{66,73,177},{66,75,181},{67,78,186},{68,81,191},{68,84,195},{68,86,199},{69,89,203},{69,92,207},{69,94,211},{70,97,214},{70,100,218},{70,102,221},{70,105,224},{70,107,227},{71,110,230},{71,113,233},{71,115,235},{71,118,238},{71,120,240},{71,123,242},{70,125,244},{70,128,246},{70,130,248},{70,133,250},{70,135,251},{69,138,252},{69,140,253},{68,143,254},{67,145,254},{66,148,255},{65,150,255},{64,153,255},{62,155,254},{61,158,254},{59,160,253},{58,163,252},{56,165,251},{55,168,250},{53,171,248},{51,173,247},{49,175,245},{47,178,244},{46,180,242},{44,183,240},{42,185,238},{40,188,235},{39,190,233},{37,192,231},{35,195,228},{34,197,226},{32,199,223},{31,201,221},{30,203,218},{28,205,216},{27,208,213},{26,210,210},{26,212,208},{25,213,205},{24,215,202},{24,217,200},{24,219,197},{24,221,194},{24,222,192},{24,224,189},{25,226,187},{25,227,185},{26,228,182},{28,230,180},{29,231,178},{31,233,175},{32,234,172},{34,235,170},{37,236,167},{39,238,164},{42,239,161},{44,240,158},{47,241,155},{50,242,152},{53,243,148},{56,244,145},{60,245,142},{63,246,138},{67,247,135},{70,248,132},{74,248,128},{78,249,125},{82,250,122},{85,250,118},{89,251,115},{93,252,111},{97,252,108},{101,253,105},{105,253,102},{109,254,98},{113,254,95},{117,254,92},{121,254,89},{125,255,86},{128,255,83},{132,255,81},{136,255,78},{139,255,75},{143,255,73},{146,255,71},{150,254,68},{153,254,66},{156,254,64},{159,253,63},{161,253,61},{164,252,60},{167,252,58},{169,251,57},{172,251,56},{175,250,55},{177,249,54},{180,248,54},{183,247,53},{185,246,53},{188,245,52},{190,244,52},{193,243,52},{195,241,52},{198,240,52},{200,239,52},{203,237,52},{205,236,52},{208,234,52},{210,233,53},{212,231,53},{215,229,53},{217,228,54},{219,226,54},{221,224,55},{223,223,55},{225,221,55},{227,219,56},{229,217,56},{231,215,57},{233,213,57},{235,211,57},{236,209,58},{238,207,58},{239,205,58},{241,203,58},{242,201,58},{244,199,58},{245,197,58},{246,195,58},{247,193,58},{248,190,57},{249,188,57},{250,186,57},{251,184,56},{251,182,55},{252,179,54},{252,177,54},{253,174,53},{253,172,52},{254,169,51},{254,167,50},{254,164,49},{254,161,48},{254,158,47},{254,155,45},{254,153,44},{254,150,43},{254,147,42},{254,144,41},{253,141,39},{253,138,38},{252,135,37},{252,132,35},{251,129,34},{251,126,33},{250,123,31},{249,120,30},{249,117,29},{248,114,28},{247,111,26},{246,108,25},{245,105,24},{244,102,23},{243,99,21},{242,96,20},{241,93,19},{240,91,18},{239,88,17},{237,85,16},{236,83,15},{235,80,14},{234,78,13},{232,75,12},{231,73,12},{229,71,11},{228,69,10},{226,67,10},{225,65,9},{223,63,8},{221,61,8},{220,59,7},{218,57,7},{216,55,6},{214,53,6},{212,51,5},{210,49,5},{208,47,5},{206,45,4},{204,43,4},{202,42,4},{200,40,3},{197,38,3},{195,37,3},{193,35,2},{190,33,2},{188,32,2},{185,30,2},{183,29,2},{180,27,1},{178,26,1},{175,24,1},{172,23,1},{169,22,1},{167,20,1},{164,19,1},{161,18,1},{158,16,1},{155,15,1},{152,14,1},{149,13,1},{146,11,1},{142,10,1},{139,9,2},{136,8,2},{133,7,2},{129,6,2},{126,5,2},{122,4,3}};
    // clang-format on
    if (Force == 0.0f) {
      return MinColor;
    }
    if (Force == 1.0f) {
      return MaxColor;
    }
    // Remap to the sub-range of the turbo color palette that we are using
    t = FMath::GetMappedRangeValueClamped(FVector2D(0.0f, 1.0f), RainbowRange, t);
    const int index = int(t * (256 - 0.0001f));
    return FColor(turbo_srgb[index][0], turbo_srgb[index][1], turbo_srgb[index][2]);
  }
}
