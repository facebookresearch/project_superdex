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

class UBlueprint;

namespace SuperDexSplitBlueprintChildren {

// Returns true if ParentBP's SimpleConstructionScript contains at least one
// top-level child that either:
//  - is (or contains) a Mochi component (UMochiPhysics module), or
//  - is a UChildActorComponent.
// Used to gate the "Split SuperDex Children..." context-menu entry.
bool IsBlueprintSplittable(UBlueprint const* ParentBP);

// Splits each top-level SuperDex Physics-bearing child subtree of ParentBP into its own
// new Blueprint asset under <ParentBPDir>/<ParentBPName>_Children/, and writes
// a .superdex_task JSON next to the new BPs describing the spawn list +
// relative transforms (translation in meters, rotation as a quaternion in
// SuperDexRobotics space). ChildActorComponents are not re-extracted into new BPs -
// their existing ChildActorClass is referenced directly in the task file.
//
// Pops a results dialog with counts + "Open Location" on success. Idempotent
// per-asset: pre-existing child BPs at the destination path are skipped (the
// task file is still rewritten to include them).
void ExecuteSplitBlueprintChildren(UBlueprint* ParentBP);

} // namespace SuperDexSplitBlueprintChildren
