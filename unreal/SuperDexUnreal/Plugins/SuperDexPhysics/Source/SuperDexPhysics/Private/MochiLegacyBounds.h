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

#include "MochiLegacyBounds.generated.h"

// TODO: Resave affected UMochiModel packages in the native-generated layout, validate
// them without the legacy reader, then remove this private schema and its load conversion.
USTRUCT()
struct FMochiLegacyBounds {
  GENERATED_BODY()

  UPROPERTY()
  FVector3f Min = FVector3f::ZeroVector;

  UPROPERTY()
  FVector3f Max = FVector3f::ZeroVector;
};
