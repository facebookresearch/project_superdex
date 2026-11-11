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
using System.IO;

namespace UnrealBuildTool.Rules
{
    public class SuperDexRecorder : ModuleRules
    {
        public SuperDexRecorder(ReadOnlyTargetRules Target) : base(Target)
        {
            CppStandard = CppStandardVersion.Cpp20;
            // Mochi's headers use `throw`, which clang rejects when exceptions are disabled.
            bEnableExceptions = true;

            PublicDependencyModuleNames.AddRange(new string[] {
                "Core",
                "CoreUObject",
                "Engine",
                "SuperDexPhysics",
                "SuperDexRobotics",
            });

            PrivateDependencyModuleNames.Add("Projects");

            string pluginRootPath = Path.GetFullPath(Path.Combine(ModuleDirectory, "../.."));
            string recorderPath = Path.Combine(pluginRootPath, "ThirdParty/superdex_recorder");
            string sourceThirdPartyPath = Path.Combine(ModuleDirectory, "ThirdParty");
            PublicIncludePaths.Add(Path.Combine(sourceThirdPartyPath, "superdex_recorder/include"));
            PublicDefinitions.Add("SUPERDEX_RECORDER_SHARED=1");

            if (Target.Platform == UnrealTargetPlatform.Win64)
            {
                string libraryPath = Path.Combine(recorderPath, "lib/Win64");
                string libraryName = "superdex_recorder_shared.dll";
                PublicAdditionalLibraries.Add(Path.Combine(libraryPath, libraryName + ".imp.lib"));
                RuntimeDependencies.Add(Path.Combine(libraryPath, libraryName));
                PublicDelayLoadDLLs.Add(libraryName);
            }
        }
    }
}
