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

using UnrealBuildTool;

public class SuperDexTeleop : ModuleRules
{
    public SuperDexTeleop(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[] {
            "Core",
            "DeveloperSettings",
            "SuperDexPose",
            "SuperDexPhysics",
            "SuperDexRobotics",
        });

        PrivateDependencyModuleNames.AddRange(new string[] {
            "CoreUObject",
            "Engine",
            "Projects",
            "Json",
            "JsonUtilities"
        });

        if (Target.bBuildEditor)
        {
            // Only this module's own .cpp files use them, so they don't propagate to dependents.
            PrivateDependencyModuleNames.AddRange(new string[] { "Kismet", "Slate", "UnrealEd", });

            // NOTE: The skeleton-rebase utility (UTeleopSkeletonUtils::ClearReferenceRootBoneRotation)
            // is disabled by default and its editor-only module dependencies have been removed. To
            // re-enable it, set TELEOP_ENABLE_SKELETON_REBASE to 1 in TeleopSkeletonUtils.cpp and add
            // these back here:
            //   "MeshDescription", "StaticMeshDescription", "SkeletalMeshDescription"
        }

        // superdex_robotics controller headers include mochi_core, which throws (std::bad_alloc, ...).
        // MSVC tolerates throw with exceptions disabled; Android clang does not.
        if (Target.Platform == UnrealTargetPlatform.Android)
        {
            bEnableExceptions = true;
        }
    }
}
