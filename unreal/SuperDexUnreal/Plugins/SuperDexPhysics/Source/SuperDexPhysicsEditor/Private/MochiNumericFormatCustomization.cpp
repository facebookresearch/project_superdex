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

#include "MochiNumericFormatCustomization.h"

#include "DetailLayoutBuilder.h"
#include "DetailWidgetRow.h"
#include "IDetailPropertyRow.h"
#include "PropertyHandle.h"
#include "UObject/UnrealType.h"
#include "Widgets/Input/NumericTypeInterface.h"
#include "Widgets/Input/SNumericEntryBox.h"
#include "Widgets/Text/STextBlock.h"

#define LOCTEXT_NAMESPACE "MochiNumericFormatEditor"

namespace {

template <typename T>
struct TMochiScientificTypeInterface : public TDefaultNumericTypeInterface<T> {
  int32 GetMaxFractionalDigits() const override {
    return 18;
  }
  void SetMaxFractionalDigits(const TAttribute<TOptional<int32>>&) override {}

  FString ToString(const T& Value) const override {
    return FString::Printf(TEXT("%.7g"), static_cast<double>(Value));
  }

  bool IsCharacterValid(TCHAR InChar) const override {
    return InChar == TEXT('e') || InChar == TEXT('E') ||
        TDefaultNumericTypeInterface<T>::IsCharacterValid(InChar);
  }

  TOptional<T> FromString(const FString& InString, const T& Existing) override {
    return static_cast<T>(FCString::Atod(*InString));
  }
};

} // namespace

TSharedRef<IDetailCustomization> FMochiScientificNotationCustomization::MakeInstance() {
  return MakeShareable(new FMochiScientificNotationCustomization);
}

void FMochiScientificNotationCustomization::CustomizeDetails(IDetailLayoutBuilder& DetailLayout) {
  TArray<TWeakObjectPtr<UObject>> Objects;
  DetailLayout.GetObjectsBeingCustomized(Objects);
  if (Objects.IsEmpty()) {
    return;
  }

  UClass* Class = Objects[0]->GetClass();

  auto FloatInterface = MakeShared<TMochiScientificTypeInterface<float>>();
  auto DoubleInterface = MakeShared<TMochiScientificTypeInterface<double>>();

  // clang-format off
  for (TFieldIterator<FProperty> It(Class); It; ++It) {
    FProperty* Prop = *It;
    if (Prop->GetOwnerClass() != Class) {
      continue;
    }
    TSharedRef<IPropertyHandle> Handle = DetailLayout.GetProperty(Prop->GetFName());
    FString Units = Prop->GetMetaData(TEXT("ForceUnits"));
    IDetailPropertyRow* Row = DetailLayout.EditDefaultProperty(Handle);
    if (!Row) {
      continue;
    }

    if (CastField<FFloatProperty>(Prop)) {
      Row->CustomWidget()
        .NameContent()[ Handle->CreatePropertyNameWidget() ]
        .ValueContent()
        [
          SNew(SHorizontalBox)
          + SHorizontalBox::Slot().FillWidth(1.0f)
          [
            SNew(SNumericEntryBox<float>)
            .TypeInterface(FloatInterface)
            .Font(IDetailLayoutBuilder::GetDetailFont())
            .Value_Lambda([Handle]() -> TOptional<float> {
              float Val = 0.0f;
              Handle->GetValue(Val);
              return Val;
            })
            .OnValueCommitted_Lambda([Handle](float Value, ETextCommit::Type) {
              Handle->SetValueFromFormattedString(FString::Printf(TEXT("%.15g"), Value));
            })
          ]
          + SHorizontalBox::Slot().AutoWidth().Padding(4, 0, 0, 0).VAlign(VAlign_Center)
          [
            SNew(STextBlock)
            .Font(IDetailLayoutBuilder::GetDetailFont())
            .Text(FText::FromString(Units))
          ]
        ];
    } else if (auto* StructProp = CastField<FStructProperty>(Prop);
               StructProp && StructProp->Struct == TBaseStructure<FVector2D>::Get()) {
      TSharedPtr<IPropertyHandle> XHandle = Handle->GetChildHandle(0);
      TSharedPtr<IPropertyHandle> YHandle = Handle->GetChildHandle(1);
      Row->CustomWidget()
        .NameContent()[ Handle->CreatePropertyNameWidget() ]
        .ValueContent().MinDesiredWidth(300)
        [
          SNew(SHorizontalBox)
          + SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(0, 0, 2, 0)
          [ SNew(STextBlock).Font(IDetailLayoutBuilder::GetDetailFont()).Text(LOCTEXT("XLabel", "X")) ]
          + SHorizontalBox::Slot().FillWidth(1.0f).Padding(0, 0, 4, 0)
          [
            SNew(SNumericEntryBox<double>)
            .TypeInterface(DoubleInterface)
            .Font(IDetailLayoutBuilder::GetDetailFont())
            .Value_Lambda([XHandle]() -> TOptional<double> {
              double Val = 0.0;
              if (XHandle.IsValid()) { XHandle->GetValue(Val); }
              return Val;
            })
            .OnValueCommitted_Lambda([XHandle](double Value, ETextCommit::Type) {
              if (XHandle.IsValid()) { XHandle->SetValueFromFormattedString(FString::Printf(TEXT("%.15g"), Value)); }
            })
          ]
          + SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(0, 0, 2, 0)
          [ SNew(STextBlock).Font(IDetailLayoutBuilder::GetDetailFont()).Text(LOCTEXT("YLabel", "Y")) ]
          + SHorizontalBox::Slot().FillWidth(1.0f)
          [
            SNew(SNumericEntryBox<double>)
            .TypeInterface(DoubleInterface)
            .Font(IDetailLayoutBuilder::GetDetailFont())
            .Value_Lambda([YHandle]() -> TOptional<double> {
              double Val = 0.0;
              if (YHandle.IsValid()) { YHandle->GetValue(Val); }
              return Val;
            })
            .OnValueCommitted_Lambda([YHandle](double Value, ETextCommit::Type) {
              if (YHandle.IsValid()) { YHandle->SetValueFromFormattedString(FString::Printf(TEXT("%.15g"), Value)); }
            })
          ]
          + SHorizontalBox::Slot().AutoWidth().Padding(4, 0, 0, 0).VAlign(VAlign_Center)
          [ SNew(STextBlock).Font(IDetailLayoutBuilder::GetDetailFont()).Text(FText::FromString(Units)) ]
        ];
    }
  }
  // clang-format on
}

#undef LOCTEXT_NAMESPACE
