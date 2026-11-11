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

using System.IO;
using UnrealBuildTool;

public class TeleopNodeZenoh : ModuleRules
{
    public TeleopNodeZenoh(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(
            new string[] { "Core", "CoreUObject", "DeveloperSettings", "SuperDexTeleop" });

        string privatePath = Path.Combine(ModuleDirectory, "Private");

        // zenoh-pico, compiled from source as part of this module
        string zenohPicoPath = Path.Combine(privatePath, "ZenohPico");
        PrivateIncludePaths.Add(Path.Combine(zenohPicoPath, "include"));
        // For atomic.c.inl, included by ZenohPicoAtomic.cpp.
        PrivateIncludePaths.Add(Path.Combine(zenohPicoPath, "src"));
        PrivateDefinitions.AddRange(new string[] { "ZENOH_C_STANDARD=11", "Z_BUILD_LOG=0" });

        // The teleop messages (teleop.proto, its nanopb output and teleop_codec.h), compiled against
        // nanopb's C runtime in Private/Nanopb.
        string teleopMsgsPath = Path.Combine(privatePath, "TeleopMsgs");
        PrivateIncludePaths.Add(teleopMsgsPath);
        PrivateIncludePaths.Add(Path.Combine(teleopMsgsPath, "generated"));
        PrivateIncludePaths.Add(Path.Combine(privatePath, "Nanopb"));

        // zenoh-pico's and nanopb's .c files share static helper names, so they must not be merged
        // into unity files.
        bUseUnity = false;

        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            PrivateDefinitions.Add("ZENOH_WINDOWS");
            PublicSystemLibraries.AddRange(new string[] { "ws2_32.lib", "iphlpapi.lib", "advapi32.lib" });
        }
        else if (Target.Platform == UnrealTargetPlatform.Android)
        {
            PrivateDefinitions.Add("ZENOH_LINUX");
        }
    }
}
