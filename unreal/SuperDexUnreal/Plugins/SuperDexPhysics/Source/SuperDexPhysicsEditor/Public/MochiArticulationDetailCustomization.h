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
#include "IDetailCustomization.h"
#include "IPropertyTypeCustomization.h"
#include "MochiArticulation.h"
#include "UObject/Object.h"

/** Details customization for articulation assets. */
class FMochiArticulationDetailsCustomization : public IDetailCustomization {
 public:
  /**
   * Create a details customization for articulation assets.
   *
   * @return The new details customization.
   */
  static TSharedRef<IDetailCustomization> MakeInstance();
  virtual void CustomizeDetails(IDetailLayoutBuilder& DetailLayout) override;

  /**
   * Build a joint-position editor for a details-array entry.
   *
   * @param PropertyHandle Handle of the single joint-position element being edited.
   * @param ArrayIndex Index of the joint-position entry.
   * @param ChildrenBuilder Details builder to populate.
   */
  void GenerateDefaultJointPositionsWidget(
      TSharedRef<IPropertyHandle> PropertyHandle,
      int32 ArrayIndex,
      IDetailChildrenBuilder& ChildrenBuilder);

 private:
  UMochiArticulation* Articulation = nullptr;
  bool bIsUsingSlider = false;
};

/** Details customization for articulation components. */
class FMochiArticulationComponentDetailsCustomization : public IDetailCustomization {
 public:
  /**
   * Create a details customization for articulation components.
   *
   * @return The new details customization.
   */
  static TSharedRef<IDetailCustomization> MakeInstance();
  virtual void CustomizeDetails(IDetailLayoutBuilder& DetailLayout) override;

  /**
   * Build a joint-position editor for a component details-array entry.
   *
   * @param PropertyHandle Handle of the single joint-position element being edited.
   * @param ArrayIndex Index of the joint-position entry.
   * @param ChildrenBuilder Details builder to populate.
   */
  void GenerateDefaultJointPositionsWidget(
      TSharedRef<IPropertyHandle> PropertyHandle,
      int32 ArrayIndex,
      IDetailChildrenBuilder& ChildrenBuilder);

 private:
  UMochiArticulation* Articulation = nullptr;
  bool bIsUsingSlider = false;
};
