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
    public class SuperDexPhysics : ModuleRules
    {
        public SuperDexPhysics(ReadOnlyTargetRules Target) : base(Target)
        {
            CppStandard = CppStandardVersion.Cpp20;

            // Unreal Module dependencies
            PublicDependencyModuleNames.AddRange(new string[] {
                "Core",
                "CoreUObject",
                "Engine",
                "ModelingComponents",
                "DynamicMesh",
                "GeometryAlgorithms",
                "DeveloperSettings",
                "ProceduralMeshComponent",
#if UE_5_0_OR_LATER
                "GeometryCore",
                "GeometryFramework",
#else
                "GeometricObjects",
#endif
            });

            PrivateDependencyModuleNames.AddRange(new string[] {
                "Projects",
                // USuperDexRuntimeMeshSubsystem: runtime .glb import.
                "ImageCore",
                "InterchangeCore",
                "InterchangeEngine",
                "RenderCore",
                // USuperDexRuntimeMeshSubsystem: runtime skinned meshes.
                "MeshDescription",
                "RHI",
                "SkeletalMeshDescription",
                "StaticMeshDescription",
            });

            // The Interchange plugin (pipelines, glTF translator) only supports desktop platforms. Elsewhere
            // (Android) USuperDexRuntimeMeshSubsystem reports every import as failed, so callers fall back to
            // drawing the Mochi shapes.
            bool bWithRuntimeMeshImport = Target.Platform == UnrealTargetPlatform.Win64 ||
                Target.Platform == UnrealTargetPlatform.Linux ||
                Target.Platform == UnrealTargetPlatform.Mac;
            if (bWithRuntimeMeshImport)
            {
                PrivateDependencyModuleNames.AddRange(new string[] { "GLTFCore", "InterchangePipelines" });
            }
            PrivateDefinitions.Add("SUPERDEX_WITH_RUNTIME_MESH_IMPORT=" + (bWithRuntimeMeshImport ? "1" : "0"));

            if (Target.bBuildEditor)
            {
                PrivateDependencyModuleNames.AddRange(new string[]
                {
                    "DesktopPlatform",
                    "EditorScriptingUtilities",
                    "Json",
                    "JsonUtilities",
                    "SlateCore",
                    "Slate",
                    "UnrealEd",
                    "Kismet"
                });
            }

            // Path to the root of the plugin (with the .uplugin file)
            string pluginRootPath = Path.GetFullPath(Path.Combine(ModuleDirectory, "../.."));

            // Path to the folder with all our includes/libs
            string mochiThirdPartyPath = Path.GetFullPath(Path.Combine(ModuleDirectory, "ThirdParty"));

            // Add includes folder to PublicIncludes
            // This allows users of this plugin to #include headers needed to use the C APIs of the plugins
            // NOTE: superdex_robotics/include is intentionally NOT added here: this module does not use
            // superdex_robotics (SuperDexRobotics.Build.cs stages its own copy of these headers).
            PublicIncludePaths.Add(Path.Combine(mochiThirdPartyPath, "mochi_physics/include"));
            // superdex_physics.h re-exports the mochi_physics API under the superdex namespace and
            // is transitively included by superdex_robotics/superdex_robotics.h; SuperDexRobotics inherits
            // this path since it depends on this module.
            PublicIncludePaths.Add(Path.Combine(mochiThirdPartyPath, "superdex_physics/include"));
            PublicIncludePaths.Add(Path.Combine(mochiThirdPartyPath, "mochi_core/include"));

            // link mochi libraries
            if (Target.Platform == UnrealTargetPlatform.Win64)
            {
                string win64LibsPath = Path.Combine(pluginRootPath, "ThirdParty/mochi_physics/lib/Win64");

                // List of DLLs we want to access via C APIs on Windows. The superdex_robotics
                // library is linked by the SuperDexRobotics plugin (see SuperDexRobotics.Build.cs).
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

                PublicDefinitions.Add("MOCHI_UNREAL_LINUX=0");

                PublicDefinitions.Add("MOCHI_UNREAL_LINUX_EDITOR=0");
            }
            else if (Target.Platform == UnrealTargetPlatform.Android)
            {
                // Path to folder with all the Android libs
                string androidLibsPath = Path.Combine(pluginRootPath, "ThirdParty/mochi_physics/lib/Android");

                // Tell compiler about .so's that we'll be dynamically linking with. superdex_robotics.so
                // is no longer linked here -- see the superdex_robotics.dll NOTE above.
                PublicAdditionalLibraries.Add(Path.Combine(androidLibsPath, "libmochi_physics.so"));
                PublicAdditionalLibraries.Add(Path.Combine(androidLibsPath, "libmarl.so"));

                // Set up deploying of .so's to Android devices
                AdditionalPropertiesForReceipt.Add("AndroidPlugin", Path.Combine(androidLibsPath, "mochi_core_apl.xml"));
                AdditionalPropertiesForReceipt.Add("AndroidPlugin", Path.Combine(androidLibsPath, "mochi_physics_apl.xml"));

                PublicDefinitions.Add("MOCHI_UNREAL_LINUX=0");

                PublicDefinitions.Add("MOCHI_UNREAL_LINUX_EDITOR=0");
            }
            else if (Target.Platform == UnrealTargetPlatform.Linux)
            {
                // MOCHI_UNREAL_LINUX=1: UE5's libc++ lacks <memory_resource>, so Mochi
                // headers need this to disable PMR at the preprocessor level.
                PublicDefinitions.Add("MOCHI_UNREAL_LINUX=1");
                // MOCHI_UNREAL_LINUX_EDITOR: The editor target shares its build environment
                // with UnrealEditor, so we can't set MinCpuArchX64=AVX2 per-module (PCH
                // AST mismatch). Mochi headers use this to disable AVX2 inline SIMD and
                // fall back to SSE/scalar paths. The Game target has AVX2 enabled at the
                // target level, so this is 0 there.
                PublicDefinitions.Add(Target.bBuildEditor
                    ? "MOCHI_UNREAL_LINUX_EDITOR=1"
                    : "MOCHI_UNREAL_LINUX_EDITOR=0");

                string linuxLibsPath = Path.Combine(pluginRootPath, "ThirdParty/mochi_physics/lib/Linux");

                // Add library search path so the linker can resolve -l flags
                PublicSystemLibraryPaths.Add(linuxLibsPath);

                // libsuperdex_robotics.so is no longer linked here -- see the superdex_robotics.dll NOTE above.
                string[] sharedLibs = {
                    "libmochi_physics.so",
                    "libmarl.so",
                };

                foreach (string libName in sharedLibs)
                {
                    PublicAdditionalLibraries.Add(Path.Combine(linuxLibsPath, libName));
                    RuntimeDependencies.Add("$(BinaryOutputDir)/" + libName, Path.Combine(linuxLibsPath, libName));
                }
            }

        }
    }
}
