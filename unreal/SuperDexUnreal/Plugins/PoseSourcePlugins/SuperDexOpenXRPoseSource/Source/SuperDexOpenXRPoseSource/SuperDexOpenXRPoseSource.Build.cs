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

public class SuperDexOpenXRPoseSource : ModuleRules
{
    public SuperDexOpenXRPoseSource(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[] {
            "Core",
            "SuperDexPose",
        });

        // Only engine modules that every Unreal Engine has: hand tracking is read through
        // IXRTrackingSystem (HeadMountedDisplay), so this module builds without the OpenXR plugin.
        // SuperDexPhysics' headers come through SuperDexPose, but its symbols (the Mochi space
        // conversions in the tests) only link with a direct dependency.
        PrivateDependencyModuleNames.AddRange(new string[] {
            "CoreUObject",
            "Engine",
            "HeadMountedDisplay",
            "Projects",
            "SuperDexPhysics",
        });
    }
}
