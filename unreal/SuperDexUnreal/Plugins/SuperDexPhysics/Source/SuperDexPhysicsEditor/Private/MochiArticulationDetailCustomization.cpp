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

#include "MochiArticulationDetailCustomization.h"

#include "DetailCategoryBuilder.h"
#include "DetailLayoutBuilder.h"
#include "DetailWidgetRow.h"
#include "IDetailChildrenBuilder.h"
#include "MochiArticulationAssetUtils.h"
#include "MochiArticulationComponent.h"
#include "MochiSkeletonHeaderExport.h"
#include "PropertyCustomizationHelpers.h"
#include "PropertyHandle.h"
#include "SWarningOrErrorBox.h"
#include "Widgets/Input/NumericTypeInterface.h"
#include "Widgets/Input/NumericUnitTypeInterface.inl"
#include "Widgets/Input/SNumericEntryBox.h"
#include "Widgets/Text/STextBlock.h"

#define LOCTEXT_NAMESPACE "MochiArticulationEditor"

// clang-format off
namespace {

// A small icon button for a details row. The icon can show a state (e.g. whether an asset needs
// regenerating) and the tooltip says what clicking does.
TSharedRef<SWidget> MakeIconButton(
  const FText& ToolTip,
  const TAttribute<const FSlateBrush*>& Icon,
  const TAttribute<FSlateColor>& IconColor,
  FSimpleDelegate OnClicked,
  bool bIsFocusable)
{
  return SNew(SBox)
      .HAlign(HAlign_Center)
      .VAlign(VAlign_Top)
      .WidthOverride(22.0f)
      .HeightOverride(22.0f)
      .ToolTipText(ToolTip)
      [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "SimpleButton")
        .ContentPadding(0.0f)
        .IsFocusable(bIsFocusable)
        .OnClicked_Lambda([OnClicked]() -> FReply {
          OnClicked.ExecuteIfBound();
          return FReply::Handled();
        })
        [
          SNew(SImage).Image(Icon).ColorAndOpacity(IconColor)
        ]
      ];
}

void AddFancyAsset(
  const IDetailLayoutBuilder& DetailLayout,
  IDetailCategoryBuilder& AssCat, FName PropertyName,
  const TFunction<bool(void)> &GetWarn,
  FSimpleDelegate OnGenerate)
{
  // The icon tracks whether the asset is out of date: a warning while it needs regenerating, a
  // check mark once it is current.
  TSharedRef<SWidget> Buttons = MakeIconButton(
      LOCTEXT("GenerateButtonToolTipText", "Generate Asset"),
      TAttribute<const FSlateBrush*>::CreateLambda([GetWarn]() -> const FSlateBrush* {
        return FAppStyle::GetBrush(GetWarn() ? "Icons.Warning" : "Icons.Check");
      }),
      TAttribute<FSlateColor>::CreateLambda([GetWarn]() {
        return FSlateColor(GetWarn() ? FLinearColor(1.0f, 0.6f, 0.0f) : FLinearColor::Green);
      }),
      OnGenerate,
      /*bIsFocusable=*/false);

  TSharedRef<IPropertyHandle> AssProp = DetailLayout.GetProperty(PropertyName);
  IDetailPropertyRow& Row = AssCat.AddProperty(AssProp);
  Row.ShowPropertyButtons(false);
  Row.CustomWidget(false)
      .NameContent()
      [
        AssProp->CreatePropertyNameWidget() // default widget
      ]
      .ValueContent()
      .MinDesiredWidth(250)
      .MaxDesiredWidth(TOptional<float>())
      [
      SNew(SHorizontalBox) +
        // The original asset picker
        SHorizontalBox::Slot()
        .MinWidth(250)
        .MaxWidth(250)
        .FillWidth(1.0f)
        [
          AssProp->CreatePropertyValueWidget()
        ] +
        // Buttons shown after the property value
        SHorizontalBox::Slot()
        .AutoWidth()
        .Padding(2.f, 5.f)
        [
            // PropertyCustomizationHelpers::MakeEditButton(FSimpleDelegate::CreateLambda([]{}))
            Buttons
        ]
        // AssProp->CreatePropertyValueWidget() // default widget
      ];
}

// Creates the text on the left-hand size of each row.
TSharedRef<SWidget> CreateDefaultJointPositionsNameWidget(
    const TSharedPtr<IPropertyHandle> &PropertyHandle,
    const FText& JointNameText) {
  return
  SNew(SHorizontalBox)
  +
  SHorizontalBox::Slot()
  .AutoWidth()
  .HAlign(HAlign_Left)
  [
    SNew(STextBlock)
    .Font(IDetailLayoutBuilder::GetDetailFont())
    .RenderOpacity(1.f)
    .Text(PropertyHandle->GetPropertyDisplayName())
  ] +
  SHorizontalBox::Slot()
  .Padding(FMargin(4.0f,0.0f,0.0f,0.0f))
  .AutoWidth()
  .HAlign(HAlign_Left)
  [
    SNew(STextBlock)
    .Font(IDetailLayoutBuilder::GetDetailFont())
    .RenderOpacity(0.5f)
    .Text(JointNameText)
  ];
}

// Creates the slider on the right-hand side of each row.
TSharedRef<SWidget> CreateDefaultJointPositionsSliderWidget(
    TSharedPtr<IPropertyHandle> PropertyHandle,
    double LimitLower,
    double LimitUpper,
    EMochiArticulatedJointType Type,
    bool& bIsUsingSlider) {
  const bool bRadial = Type != EMochiArticulatedJointType::Prismatic;
  const FString UnitStr = bRadial ? TEXT("radians") : TEXT("meters");
  const EUnit Unit = bRadial ? EUnit::Radians : EUnit::Meters;
  const TSharedRef<TNumericUnitTypeInterface<double>> UnitTypeInterface = MakeShared<TNumericUnitTypeInterface<double>>(Unit);
  // Calling SetupFixedDisplay will make UE use localized units according to project settings.
  // This means radians will probably be displayed as degrees, and meters as centimeters.
  // If we want to use meters for prismatic joints, we can condition this function call.
  UnitTypeInterface->SetupFixedDisplay(bRadial ? 2 * PI : 1.0);
  // limits optionals will dictate slider or not slider
  TOptional<double> MinValue;
  TOptional<double> MaxValue;
  if (FMath::IsFinite(LimitLower) && FMath::IsFinite(LimitUpper)) {
    MinValue = LimitLower;
    MaxValue = LimitUpper;
  }
  return
  SNew(SNumericEntryBox<double>)
  .MinValue(MinValue)
  .MaxValue(MaxValue)
  .MinSliderValue(MinValue)
  .MaxSliderValue(MaxValue)
  .AllowSpin(true)
  .SliderExponent(1.0f)
  .TypeInterface(UnitTypeInterface)
  .MinFractionalDigits(0)
  .MaxFractionalDigits(3)
  .Font(IDetailLayoutBuilder::GetDetailFont())
  .Value_Lambda([PropertyHandle]() {
    double Val = 0.0;
    PropertyHandle->GetValue(Val);
    return Val;
  })
  .OnValueChanged_Lambda([&bIsUsingSlider, LimitLower, LimitUpper, PropertyHandle](double NewValue) {
    if (bIsUsingSlider) {
      const double Clamped = FMath::Clamp(NewValue, LimitLower, LimitUpper);
      PropertyHandle->SetValue(Clamped, EPropertyValueSetFlags::InteractiveChange);
    }
  })
  .OnValueCommitted_Lambda([LimitLower, LimitUpper, PropertyHandle](double Value, ETextCommit::Type)
  {
    const double Clamped = FMath::Clamp(Value, LimitLower, LimitUpper);
    PropertyHandle->SetValue(Clamped, EPropertyValueSetFlags::DefaultFlags);
  })
  .OnBeginSliderMovement_Lambda([&bIsUsingSlider]()    {
    bIsUsingSlider = true;
    GEditor->BeginTransaction(LOCTEXT("SetJointPositionsProperty", "Set Property"));
  })
  .OnEndSliderMovement_Lambda([&bIsUsingSlider](float FinalValue) {
    bIsUsingSlider = false;
    GEditor->EndTransaction();
  });
}

// Creates each row.
void GenerateDefaultJointPositionsWidget(
    const TSharedRef<IPropertyHandle> &PropertyHandle,
    int32 ArrayIndex,
    const UMochiArticulation* Articulation,
    IDetailChildrenBuilder& ChildrenBuilder,
    bool& bIsUsingSlider) {
  FMochiArticulatedJoint Joint;
  double LimitLower = 0;
  double LimitUpper = 0;
  if (!Articulation->GetDof(ArrayIndex, Joint) ||
      !Articulation->GetDofLimits(ArrayIndex, LimitLower, LimitUpper)) {
    return;
  }
  IDetailPropertyRow& Row = ChildrenBuilder.AddProperty(PropertyHandle);
  FNumberFormattingOptions NoCommas;
  NoCommas.UseGrouping = false;
  const FString DofTypeStr = Joint.Type == EMochiArticulatedJointType::Revolute ? TEXT("R") :
    Joint.Type == EMochiArticulatedJointType::Prismatic ? TEXT("P") :
    Joint.Type == EMochiArticulatedJointType::Spherical ? TEXT("S") :  TEXT("?");
  const FText DofIndexText = FText::Format(LOCTEXT("DefaultJointPositionsIndex", "[ {0} ] ( {1} )"),
    FText::AsNumber(ArrayIndex, &NoCommas),
    FText::FromString(DofTypeStr));
  const FText DofNameText = FText::FromString(Joint.Name.ToString());
  Row.DisplayName(DofIndexText);
  Row.ShowPropertyButtons(false);
  Row.CustomWidget(false)
      .NameContent()
      [
        // PropertyHandle->CreatePropertyNameWidget() // default widget
        CreateDefaultJointPositionsNameWidget(PropertyHandle, DofNameText)
      ]
      .ValueContent()
      .MinDesiredWidth(250)
      .MaxDesiredWidth(TOptional<float>())
      [
        // PropertyHandle->CreatePropertyValueWidget() // default widget
        CreateDefaultJointPositionsSliderWidget(PropertyHandle, LimitLower, LimitUpper, Joint.Type, bIsUsingSlider)
      ];
}
} // namespace
// clang-format on

//------------------------------------------------------------------------------
// FMochiArticulationDetailsCustomization
//------------------------------------------------------------------------------

TSharedRef<IDetailCustomization> FMochiArticulationDetailsCustomization::MakeInstance() {
  return MakeShareable(new FMochiArticulationDetailsCustomization());
}

void FMochiArticulationDetailsCustomization::CustomizeDetails(IDetailLayoutBuilder& DetailLayout) {
  TArray<TWeakObjectPtr<UObject>> Objects;
  DetailLayout.GetObjectsBeingCustomized(Objects);
  for (int ObjIndex = 0; ObjIndex < Objects.Num(); ++ObjIndex) {
    Articulation = Cast<UMochiArticulation>(Objects[ObjIndex]);
    if (Articulation) {
      // Warnings category
      TArray<FString> Warnings;
      if (!Articulation->IsReady(true, &Warnings)) {
        IDetailCategoryBuilder& WarnCat = DetailLayout.EditCategory("Warnings");
        for (const FString& Msg : Warnings) {
          WarnCat.AddCustomRow(FText::FromString(Msg))
              .WholeRowContent()[SNew(SWarningOrErrorBox)
                                     .MessageStyle(EMessageStyle::Warning)
                                     .IconSize({16, 16})
                                     .Padding(8.0f)
                                     .Message(FText::FromString(Msg))]
              .ResetToDefaultContent()[MakeIconButton(
                  LOCTEXT("RefreshWarningsTooltipText", "Refresh Warnings"),
                  FAppStyle::GetBrush("PropertyWindow.DiffersFromDefault"),
                  FSlateColor::UseForeground(),
                  FSimpleDelegate::CreateLambda(
                      [&DetailLayout] { DetailLayout.ForceRefreshDetails(); }),
                  /*bIsFocusable=*/true)];
        }
      }
      // Robot category
      IDetailCategoryBuilder& ArtCat = DetailLayout.EditCategory("Articulation");
      ArtCat.AddProperty(
          DetailLayout.GetProperty(GET_MEMBER_NAME_CHECKED(UMochiArticulation, ArticulationName)));
      const TSharedPtr<IPropertyHandle> PosProp = DetailLayout.GetProperty(
          GET_MEMBER_NAME_CHECKED(UMochiArticulation, DefaultJointPositions));
      const TSharedRef<FDetailArrayBuilder> Builder =
          MakeShareable(new FDetailArrayBuilder(PosProp.ToSharedRef(), true, false, true));
      Builder->OnGenerateArrayElementWidget(
          FOnGenerateArrayElementWidget::CreateSP(
              this, &FMochiArticulationDetailsCustomization::GenerateDefaultJointPositionsWidget));
      ArtCat.AddCustomBuilder(Builder);
      ArtCat.AddProperty(
          DetailLayout.GetProperty(GET_MEMBER_NAME_CHECKED(UMochiArticulation, Joints)));
      ArtCat.AddProperty(
          DetailLayout.GetProperty(GET_MEMBER_NAME_CHECKED(UMochiArticulation, Links)));
      ArtCat.AddProperty(
          DetailLayout.GetProperty(GET_MEMBER_NAME_CHECKED(UMochiArticulation, Cycles)));
      ArtCat.AddProperty(
          DetailLayout.GetProperty(GET_MEMBER_NAME_CHECKED(UMochiArticulation, Skin)));
      ArtCat.AddProperty(
          DetailLayout.GetProperty(GET_MEMBER_NAME_CHECKED(UMochiArticulation, ContactOverrides)));
      ArtCat.AddProperty(DetailLayout.GetProperty(
          GET_MEMBER_NAME_CHECKED(UMochiArticulation, AdditionalConstraints)));
      ArtCat.AddProperty(
          DetailLayout.GetProperty(GET_MEMBER_NAME_CHECKED(UMochiArticulation, SoftLinks)));
      // Assets category
      IDetailCategoryBuilder& AssCat = DetailLayout.EditCategory("Assets");
      AddFancyAsset(
          DetailLayout,
          AssCat,
          GET_MEMBER_NAME_CHECKED(UMochiArticulation, SkeletalMesh),
          [this] {
            return Articulation->bNeedGenerateSkeletalMesh || Articulation->SkeletalMesh == nullptr;
          },
          FSimpleDelegate::CreateLambda(
              [this] { FMochiArticulationAssetUtils::GenerateSkeletalMesh(Articulation); }));
      MochiSkeletonHeaderExport::AddDetailsRow(DetailLayout, AssCat, Articulation);
      // Internal category
      IDetailCategoryBuilder& IntCat = DetailLayout.EditCategory("Internal");
      IntCat.SetDisplayName(FText::FromString(TEXT("Internal (Advanced Use Only)")));
      IntCat.InitiallyCollapsed(true);
    }
  }
}

void FMochiArticulationDetailsCustomization::GenerateDefaultJointPositionsWidget(
    TSharedRef<IPropertyHandle> PropertyHandle,
    int32 ArrayIndex,
    IDetailChildrenBuilder& ChildrenBuilder) {
  ::GenerateDefaultJointPositionsWidget(
      PropertyHandle, ArrayIndex, Articulation, ChildrenBuilder, bIsUsingSlider);
}

//------------------------------------------------------------------------------
// FMochiArticulationComponentDetailsCustomization
//------------------------------------------------------------------------------

TSharedRef<IDetailCustomization> FMochiArticulationComponentDetailsCustomization::MakeInstance() {
  return MakeShareable(new FMochiArticulationComponentDetailsCustomization());
}

void FMochiArticulationComponentDetailsCustomization::CustomizeDetails(
    IDetailLayoutBuilder& DetailLayout) {
  TArray<TWeakObjectPtr<UObject>> Objects;
  DetailLayout.GetObjectsBeingCustomized(Objects);
  for (int ObjIndex = 0; ObjIndex < Objects.Num(); ++ObjIndex) {
    UMochiArticulationComponent* const Component =
        Cast<UMochiArticulationComponent>(Objects[ObjIndex]);
    Articulation = Component->GetArticulationAsset();
    if (Articulation) {
      IDetailCategoryBuilder& ArticulationCat = DetailLayout.EditCategory(
          "Mochi Articulation", FText::GetEmpty(), ECategoryPriority::Important);
      ArticulationCat.AddProperty(DetailLayout.GetProperty(
          GET_MEMBER_NAME_CHECKED(UMochiArticulationComponent, Articulation)));
      const TSharedPtr<IPropertyHandle> PosProp = DetailLayout.GetProperty(
          GET_MEMBER_NAME_CHECKED(UMochiArticulationComponent, InitialJointPositions));
      const TSharedRef<FDetailArrayBuilder> Builder =
          MakeShareable(new FDetailArrayBuilder(PosProp.ToSharedRef(), true, false, true));
      Builder->OnGenerateArrayElementWidget(
          FOnGenerateArrayElementWidget::CreateSP(
              this,
              &FMochiArticulationComponentDetailsCustomization::
                  GenerateDefaultJointPositionsWidget));
      ArticulationCat.AddCustomBuilder(Builder);
    }
  }
}

void FMochiArticulationComponentDetailsCustomization::GenerateDefaultJointPositionsWidget(
    TSharedRef<IPropertyHandle> PropertyHandle,
    int32 ArrayIndex,
    IDetailChildrenBuilder& ChildrenBuilder) {
  ::GenerateDefaultJointPositionsWidget(
      PropertyHandle, ArrayIndex, Articulation, ChildrenBuilder, bIsUsingSlider);
}

#undef LOCTEXT_NAMESPACE
