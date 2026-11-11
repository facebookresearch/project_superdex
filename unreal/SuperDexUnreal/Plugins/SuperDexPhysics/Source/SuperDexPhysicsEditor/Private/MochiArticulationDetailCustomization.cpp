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
        .OnClicked_Lambda([OnClicked]() {
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
  TSharedRef<SWidget> buttons = MakeIconButton(
      LOCTEXT("GenerateButtonToolTipText", "Generate Asset"),
      TAttribute<const FSlateBrush*>::CreateLambda([GetWarn]() {
        return FAppStyle::GetBrush(GetWarn() ? "Icons.Warning" : "Icons.Check");
      }),
      TAttribute<FSlateColor>::CreateLambda([GetWarn]() {
        return FSlateColor(GetWarn() ? FLinearColor(1.0f, 0.6f, 0.0f) : FLinearColor::Green);
      }),
      OnGenerate,
      /*bIsFocusable=*/false);

  auto assProp = DetailLayout.GetProperty(PropertyName);
  auto& row = AssCat.AddProperty(assProp);
  row.ShowPropertyButtons(false);
  row.CustomWidget(false)
      .NameContent()
      [
        assProp->CreatePropertyNameWidget() // default widget
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
          assProp->CreatePropertyValueWidget()
        ] +
        // Buttons shown after the property value
        SHorizontalBox::Slot()
        .AutoWidth()
        .Padding(2.f, 5.f)
        [
            // PropertyCustomizationHelpers::MakeEditButton(FSimpleDelegate::CreateLambda([]{}))
            buttons
        ]
        // assProp->CreatePropertyValueWidget() // default widget
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
  const FString unitStr = bRadial ? TEXT("radians") : TEXT("meters");
  const EUnit unit = bRadial ? EUnit::Radians : EUnit::Meters;
  const auto unitTypeInterface = MakeShared<TNumericUnitTypeInterface<double>>(unit);
  // Calling SetupFixedDisplay will make UE use localized units according to project settings.
  // This means radians will probably be displayed as degrees, and meters as centimeters.
  // If we want to use meters for prismatic joints, we can condition this function call.
  unitTypeInterface->SetupFixedDisplay(bRadial ? 2 * PI : 1.0);
  // limits optionals will dictate slider or not slider
  TOptional<double> minValue;
  TOptional<double> maxValue;
  if (FMath::IsFinite(LimitLower) && FMath::IsFinite(LimitUpper)) {
    minValue = LimitLower;
    maxValue = LimitUpper;
  }
  return
  SNew(SNumericEntryBox<double>)
  .MinValue(minValue)
  .MaxValue(maxValue)
  .MinSliderValue(minValue)
  .MaxSliderValue(maxValue)
  .AllowSpin(true)
  .SliderExponent(1.0f)
  .TypeInterface(unitTypeInterface)
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
      const double clamped = FMath::Clamp(NewValue, LimitLower, LimitUpper);
      PropertyHandle->SetValue(clamped, EPropertyValueSetFlags::InteractiveChange);
    }
  })
  .OnValueCommitted_Lambda([LimitLower, LimitUpper, PropertyHandle](double Value, ETextCommit::Type)
  {
    const double clamped = FMath::Clamp(Value, LimitLower, LimitUpper);
    PropertyHandle->SetValue(clamped, EPropertyValueSetFlags::DefaultFlags);
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
  FMochiArticulatedJoint joint;
  double limitLower = 0;
  double limitUpper = 0;
  if (!Articulation->GetDof(ArrayIndex, joint) ||
      !Articulation->GetDofLimits(ArrayIndex, limitLower, limitUpper)) {
    return;
  }
  IDetailPropertyRow& Row = ChildrenBuilder.AddProperty(PropertyHandle);
  FNumberFormattingOptions NoCommas;
  NoCommas.UseGrouping = false;
  const FString dofTypeStr = joint.Type == EMochiArticulatedJointType::Revolute ? TEXT("R") :
    joint.Type == EMochiArticulatedJointType::Prismatic ? TEXT("P") :
    joint.Type == EMochiArticulatedJointType::Spherical ? TEXT("S") :  TEXT("?");
  const FText dofIndexText = FText::Format(LOCTEXT("DefaultJointPositionsIndex", "[ {0} ] ( {1} )"),
    FText::AsNumber(ArrayIndex, &NoCommas),
    FText::FromString(dofTypeStr));
  const FText dofNameText = FText::FromString(joint.Name.ToString());
  Row.DisplayName(dofIndexText);
  Row.ShowPropertyButtons(false);
  Row.CustomWidget(false)
      .NameContent()
      [
        // PropertyHandle->CreatePropertyNameWidget() // default widget
        CreateDefaultJointPositionsNameWidget(PropertyHandle, dofNameText)
      ]
      .ValueContent()
      .MinDesiredWidth(250)
      .MaxDesiredWidth(TOptional<float>())
      [
        // PropertyHandle->CreatePropertyValueWidget() // default widget
        CreateDefaultJointPositionsSliderWidget(PropertyHandle, limitLower, limitUpper, joint.Type, bIsUsingSlider)
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
  TArray<TWeakObjectPtr<UObject>> objects;
  DetailLayout.GetObjectsBeingCustomized(objects);
  for (int iObj = 0; iObj < objects.Num(); ++iObj) {
    Articulation = Cast<UMochiArticulation>(objects[iObj]);
    if (Articulation) {
      // Warnings category
      TArray<FString> warnings;
      if (!Articulation->IsReady(true, &warnings)) {
        auto& warnCat = DetailLayout.EditCategory("Warnings");
        for (const auto& msg : warnings) {
          warnCat.AddCustomRow(FText::FromString(msg))
              .WholeRowContent()[SNew(SWarningOrErrorBox)
                                     .MessageStyle(EMessageStyle::Warning)
                                     .IconSize({16, 16})
                                     .Padding(8.0f)
                                     .Message(FText::FromString(msg))]
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
      auto& artCat = DetailLayout.EditCategory("Articulation");
      artCat.AddProperty(
          DetailLayout.GetProperty(GET_MEMBER_NAME_CHECKED(UMochiArticulation, ArticulationName)));
      const TSharedPtr<IPropertyHandle> posProp = DetailLayout.GetProperty(
          GET_MEMBER_NAME_CHECKED(UMochiArticulation, DefaultJointPositions));
      const TSharedRef<FDetailArrayBuilder> builder =
          MakeShareable(new FDetailArrayBuilder(posProp.ToSharedRef(), true, false, true));
      builder->OnGenerateArrayElementWidget(
          FOnGenerateArrayElementWidget::CreateSP(
              this, &FMochiArticulationDetailsCustomization::GenerateDefaultJointPositionsWidget));
      artCat.AddCustomBuilder(builder);
      artCat.AddProperty(
          DetailLayout.GetProperty(GET_MEMBER_NAME_CHECKED(UMochiArticulation, Joints)));
      artCat.AddProperty(
          DetailLayout.GetProperty(GET_MEMBER_NAME_CHECKED(UMochiArticulation, Links)));
      artCat.AddProperty(
          DetailLayout.GetProperty(GET_MEMBER_NAME_CHECKED(UMochiArticulation, Cycles)));
      artCat.AddProperty(
          DetailLayout.GetProperty(GET_MEMBER_NAME_CHECKED(UMochiArticulation, Skin)));
      artCat.AddProperty(
          DetailLayout.GetProperty(GET_MEMBER_NAME_CHECKED(UMochiArticulation, ContactOverrides)));
      artCat.AddProperty(DetailLayout.GetProperty(
          GET_MEMBER_NAME_CHECKED(UMochiArticulation, AdditionalConstraints)));
      artCat.AddProperty(
          DetailLayout.GetProperty(GET_MEMBER_NAME_CHECKED(UMochiArticulation, SoftLinks)));
      // Assets category
      auto& assCat = DetailLayout.EditCategory("Assets");
      AddFancyAsset(
          DetailLayout,
          assCat,
          GET_MEMBER_NAME_CHECKED(UMochiArticulation, SkeletalMesh),
          [this] {
            return Articulation->bNeedGenerateSkeletalMesh || Articulation->SkeletalMesh == nullptr;
          },
          FSimpleDelegate::CreateLambda([this, &DetailLayout] {
            FMochiArticulationAssetUtils::GenerateSkeletalMesh(Articulation);
          }));
      MochiSkeletonHeaderExport::AddDetailsRow(DetailLayout, assCat, Articulation);
      // Internal category
      auto& intCat = DetailLayout.EditCategory("Internal");
      intCat.SetDisplayName(FText::FromString(TEXT("Internal (Advanced Use Only)")));
      intCat.InitiallyCollapsed(true);
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
  TArray<TWeakObjectPtr<UObject>> objects;
  DetailLayout.GetObjectsBeingCustomized(objects);
  for (int iObj = 0; iObj < objects.Num(); ++iObj) {
    const auto component = Cast<UMochiArticulationComponent>(objects[iObj]);
    Articulation = component->GetArticulationAsset();
    if (Articulation) {
      auto& articulationCat = DetailLayout.EditCategory(
          "Mochi Articulation", FText::GetEmpty(), ECategoryPriority::Important);
      articulationCat.AddProperty(DetailLayout.GetProperty(
          GET_MEMBER_NAME_CHECKED(UMochiArticulationComponent, Articulation)));
      const TSharedPtr<IPropertyHandle> posProp = DetailLayout.GetProperty(
          GET_MEMBER_NAME_CHECKED(UMochiArticulationComponent, InitialJointPositions));
      const TSharedRef<FDetailArrayBuilder> builder =
          MakeShareable(new FDetailArrayBuilder(posProp.ToSharedRef(), true, false, true));
      builder->OnGenerateArrayElementWidget(
          FOnGenerateArrayElementWidget::CreateSP(
              this,
              &FMochiArticulationComponentDetailsCustomization::
                  GenerateDefaultJointPositionsWidget));
      articulationCat.AddCustomBuilder(builder);
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
