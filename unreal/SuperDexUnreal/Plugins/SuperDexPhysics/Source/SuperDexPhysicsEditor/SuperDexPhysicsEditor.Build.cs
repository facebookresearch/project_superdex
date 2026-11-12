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
    public class SuperDexPhysicsEditor : ModuleRules
    {
        public SuperDexPhysicsEditor(ReadOnlyTargetRules Target) : base(Target)
        {
            // SuperDexPhysicsEditor depends on FBX SDK and Win64-only editor infrastructure.
            // On Linux, mark as external to skip compilation entirely.
            if (Target.Platform == UnrealTargetPlatform.Linux)
            {
                Type = ModuleType.External;
                return;
            }

            CppStandard = CppStandardVersion.Cpp20;

            // Unreal Module dependencies
            PublicDependencyModuleNames.AddRange(new string[] {
                "Core",
                "CoreUObject",
                "Engine",
                "EditorStyle",
                "ModelingComponents",
                "DynamicMesh",
                "DeveloperSettings",
                "GeometryCore",
                "GeometryFramework",
                "MeshDescription",
                "SuperDexPhysics",
                "PropertyEditor",
                "Slate",
                "SlateCore"
            });

            PrivateDependencyModuleNames.AddRange(new string[] {
                "Projects",
                "EditorScriptingUtilities",
                "GLTFCore",
                "InterchangeCore",
                "InterchangeEngine",
                "InterchangePipelines",
                "StaticMeshDescription",
                "SkeletalMeshDescription",
            });

            if (Target.bBuildEditor)
            {
                PrivateDependencyModuleNames.AddRange(new string[]
                {
                    "AdvancedPreviewScene",
                    "AssetRegistry",
                    "ContentBrowser",
                    "DesktopPlatform",
                    "EditorFramework",
                    "Eigen",
                    "InputCore",
                    "Json",
                    "JsonUtilities",
                    "Kismet",
                    "MeshConversion",
                    "MeshConversionEngineTypes",
                    "RenderCore",
                    "SkeletalMeshUtilitiesCommon",
                    "SlateCore",
                    "Slate",
                    "ToolMenus",
                    "ToolWidgets",
                    "UnrealEd",
                    "GLTFExporter",
                });

                PublicDependencyModuleNames.AddRange(new string[]
                {
                    "AssetTools",
                });
            }

            // FBX SDK dependency
            string fbxInc = EngineDirectory + "/Source/ThirdParty/FBX/2020.2/include";
            string fbxLib = EngineDirectory + "/Source/ThirdParty/FBX/2020.2/lib/vs2017/x64/release/libfbxsdk.lib";
            string fbxDll = EngineDirectory + "/Binaries/ThirdParty/FBX/2020.2/Win64/libfbxsdk.dll";
            PublicDefinitions.Add("FBXSDK_SHARED"); // We are using DLL versions of the FBX libraries
            PublicAdditionalLibraries.Add(fbxLib);
            RuntimeDependencies.Add(fbxDll);
            PrivateIncludePaths.Add(Path.GetFullPath(fbxInc));

            // Path to the root of the plugin (with the .uplugin file)
            string pluginRootPath = Path.GetFullPath(Path.Combine(ModuleDirectory, "../.."));

            // link mochi libraries
            if (Target.Platform == UnrealTargetPlatform.Win64)
            {
                string win64LibsPath = Path.Combine(pluginRootPath, "ThirdParty/mochi_physics/lib/Win64");

                // List of DLLs we want to access via C APIs on Windows. The superdex_robotics
                // library is linked by the SuperDexRoboticsEditor module, which owns the bot
                // editor features.
                string[] dllsWithImportLibs = {
                    "mochi_physics.dll",
                };

                // Other needed DLLs (whose APIs we do not access via importlibs)
                string[] dllsWithoutImportLibs = {
                    "marl.dll",
                };

                foreach (string dllName in dllsWithImportLibs)
                {
                    // Statically link against the import lib
                    PublicAdditionalLibraries.Add(Path.Combine(win64LibsPath, dllName + ".imp.lib"));

                    // Mark DLL as a runtime dependency (causing it to get packaged)
                    RuntimeDependencies.Add(Path.Combine(win64LibsPath, dllName));

                    // Set up DLL for delay load
                    PublicDelayLoadDLLs.Add(dllName);
                }

                foreach (string dllName in dllsWithoutImportLibs)
                {
                    // Mark DLL as a runtime dependency (causing it to get packaged)
                    RuntimeDependencies.Add(Path.Combine(win64LibsPath, dllName));

                    // Set up DLL for delay load
                    PublicDelayLoadDLLs.Add(dllName);
                }
            }
        }
    }
}
