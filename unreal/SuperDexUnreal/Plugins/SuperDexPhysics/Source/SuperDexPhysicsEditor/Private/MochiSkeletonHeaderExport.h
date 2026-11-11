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

class IDetailCategoryBuilder;
class IDetailLayoutBuilder;
class UMochiArticulationBase;

// Writes a SuperDexSkeleton header for an articulation's skeletal mesh (a UMochiArticulation or a
// USuperDexBot): the skeleton name, an ESuperDexBone_<Name> enum in bone order, the bone names and
// each bone's parent. Skeleton headers are kept side by side in
// SuperDexRobotics/Source/SuperDexPose/Public/Skeletons, which the save dialog opens by default.
namespace MochiSkeletonHeaderExport {

// Whether `Articulation` has a skeletal mesh to export.
bool CanGenerate(const UMochiArticulationBase* Articulation);

// Asks for the header's location and writes it.
void Generate(const UMochiArticulationBase* Articulation);

// Adds a "generate skeleton header" row to `Category`.
void AddDetailsRow(
    IDetailLayoutBuilder& DetailLayout,
    IDetailCategoryBuilder& Category,
    const UMochiArticulationBase* Articulation);

} // namespace MochiSkeletonHeaderExport
