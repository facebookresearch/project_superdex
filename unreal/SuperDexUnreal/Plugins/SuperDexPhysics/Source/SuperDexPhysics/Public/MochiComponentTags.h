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

#include "UObject/NameTypes.h"

class AActor;

namespace MochiComponentTags {

extern SUPERDEXPHYSICS_API const FName DisableComponents;

/**
 * Get whether actor tags disable its Mochi components.
 *
 * @param Actor Unreal actor whose component configuration is inspected.
 * @return Whether the actor tags disable its Mochi components.
 */
SUPERDEXPHYSICS_API bool AreComponentsDisabled(const AActor* Actor);

} // namespace MochiComponentTags
