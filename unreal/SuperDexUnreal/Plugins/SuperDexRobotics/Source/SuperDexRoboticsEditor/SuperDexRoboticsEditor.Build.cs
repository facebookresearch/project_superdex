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
    public class SuperDexRoboticsEditor : ModuleRules
    {
        public SuperDexRoboticsEditor(ReadOnlyTargetRules Target) : base(Target)
        {
            // SuperDexRoboticsEditor mirrors MochiPhysicsEditor: Win64-only editor infrastructure.
            if (Target.Platform == UnrealTargetPlatform.Linux)
            {
                Type = ModuleType.External;
                return;
            }

            CppStandard = CppStandardVersion.Cpp20;

            PublicDependencyModuleNames.AddRange(new string[] {
                "Core",
                "CoreUObject",
                "Engine",
                "SuperDexPhysics",
                "SuperDexRobotics",
                "SuperDexPhysicsEditor",
                "SuperDexPose",
                "SuperDexStudy",
                "SuperDexTeleop",
                "Slate",
                "SlateCore",
            });

            PrivateDependencyModuleNames.AddRange(new string[] {
                "Projects",
                "AssetRegistry",
                "AssetTools",
                "ContentBrowser",
                "DesktopPlatform",
                "EditorFramework",
                "EditorScriptingUtilities",
                "ToolWidgets",
                "InputCore",
                "InterchangeEngine",
                "InterchangePipelines",
                "Json",
                "JsonUtilities",
                "Kismet",
                "MeshDescription",
                "StaticMeshDescription",
                "SkeletalMeshDescription",
                "ToolMenus",
                "UnrealEd",
            });

            // Path to the root of this plugin (with the .uplugin file)
            string pluginRootPath = Path.GetFullPath(Path.Combine(ModuleDirectory, "../.."));
            string mochiPhysicsPluginRootPath =
                Path.GetFullPath(Path.Combine(pluginRootPath, "../SuperDexPhysics"));

            // link mochi libraries. Editor-side bot import/export/sync code calls directly
            // into superdex_robotics (and, transitively, mochi_physics) native APIs.
            if (Target.Platform == UnrealTargetPlatform.Win64)
            {
                string superDexRoboticsLibsPath = Path.Combine(pluginRootPath, "ThirdParty/superdex_robotics/lib/Win64");
                string mochiPhysicsLibsPath =
                    Path.Combine(mochiPhysicsPluginRootPath, "ThirdParty/mochi_physics/lib/Win64");

                PublicAdditionalLibraries.Add(Path.Combine(superDexRoboticsLibsPath, "superdex_robotics.dll.imp.lib"));
                RuntimeDependencies.Add(Path.Combine(superDexRoboticsLibsPath, "superdex_robotics.dll"));
                PublicDelayLoadDLLs.Add("superdex_robotics.dll");

                // Only the import lib is referenced here; RuntimeDependencies/staging for
                // mochi_physics.dll is intentionally left to SuperDexPhysics.Build.cs.
                PublicAdditionalLibraries.Add(Path.Combine(mochiPhysicsLibsPath, "mochi_physics.dll.imp.lib"));
                PublicDelayLoadDLLs.Add("mochi_physics.dll");
            }
        }
    }
}
