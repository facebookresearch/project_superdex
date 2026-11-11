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

public class SuperDexPose : ModuleRules
{
    public SuperDexPose(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(
            new string[]
            {
          "Core",
          "SuperDexPhysics",
          "ModelingComponents",
          "DynamicMesh",
          "GeometryAlgorithms",
          "DeveloperSettings",
          "ProceduralMeshComponent",
          "GeometryCore",
          "GeometryFramework"
            }
            );

        PrivateDependencyModuleNames.AddRange(
            new string[]
            {
          "CoreUObject",
          "Engine",
          "Slate",
          "SlateCore",
          "Json",
          "JsonUtilities",
          "Projects"
            }
            );

        // HeadMountedDisplay is a cross-platform UE module (not OculusXR-specific).
        // Needed on all platforms for IMotionController.h used by PlayerPawnPoseSource.
        PrivateDependencyModuleNames.Add("HeadMountedDisplay");

        if (Target.bBuildEditor)
        {
            PrivateDependencyModuleNames.AddRange(
              new string[]
              {
          "DesktopPlatform",
          "EditorScriptingUtilities",
          "Json",
          "JsonUtilities",
          "SlateCore",
          "Slate",
          "UnrealEd"
              }
              );
        }

        // Path to the root of the plugin (with the .uplugin file)
        string PluginRootPath = Path.GetFullPath(Path.Combine(ModuleDirectory, "../.."));

        ////////// Mochi DLL Loading

        // Add Mochi includes folder to PublicIncludes
        // This allows users of this plugin to #include headers needed to use the C APIs of the plugins
        // mochi_physics/mochi_core are owned by the sibling SuperDexPhysics plugin.
        PublicIncludePaths.Add(Path.Combine(PluginRootPath, "../SuperDexPhysics/Source/SuperDexPhysics/ThirdParty/mochi_physics/include"));
        PublicIncludePaths.Add(Path.Combine(PluginRootPath, "../SuperDexPhysics/Source/SuperDexPhysics/ThirdParty/mochi_core/include"));

        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            string Win64LibsPath = Path.Combine(PluginRootPath, "../SuperDexPhysics/ThirdParty/mochi_physics/lib/Win64");

            // List of DLLs we want to access via C APIs on Windows:
            string[] DllsWithImportLibs = {
                    "mochi_physics.dll",
                };

            // Other needed DLLs (whose APIs we do not access via importlibs)
            string[] DllsWithoutImportLibs = {
                    "marl.dll",
                };

            foreach (string DllName in DllsWithImportLibs)
            {
                // Statically link against the import lib
                PublicAdditionalLibraries.Add(Path.Combine(Win64LibsPath, DllName + ".imp.lib"));

                // Mark DLL as a runtime dependency (causing it to get packaged)
                RuntimeDependencies.Add(Path.Combine(Win64LibsPath, DllName));

                // Set up DLL for delay load
                PublicDelayLoadDLLs.Add(DllName);
            }

            foreach (string DllName in DllsWithoutImportLibs)
            {
                // Mark DLL as a runtime dependency (causing it to get packaged)
                RuntimeDependencies.Add(Path.Combine(Win64LibsPath, DllName));

                // Set up DLL for delay load
                PublicDelayLoadDLLs.Add(DllName);
            }
        }
        else if (Target.Platform == UnrealTargetPlatform.Android)
        {
            // Path to folder with all the Android libs
            string AndroidLibsPath = Path.Combine(PluginRootPath, "../SuperDexPhysics/ThirdParty/mochi_physics/lib/Android");

            // Tell compiler about .so's that we'll be dynamically linking with
            PublicAdditionalLibraries.Add(Path.Combine(AndroidLibsPath, "libmochi_physics.so"));
            PublicAdditionalLibraries.Add(Path.Combine(AndroidLibsPath, "libmarl.so"));

            // Set up deploying of .so's to Android devices
            AdditionalPropertiesForReceipt.Add("AndroidPlugin", Path.Combine(AndroidLibsPath, "mochi_core_apl.xml"));
            AdditionalPropertiesForReceipt.Add("AndroidPlugin", Path.Combine(AndroidLibsPath, "mochi_physics_apl.xml"));
        }
        else if (Target.Platform == UnrealTargetPlatform.Linux)
        {
            string LinuxLibsPath = Path.Combine(PluginRootPath, "../SuperDexPhysics/ThirdParty/mochi_physics/lib/Linux");

            // These are already staged to the binary output dir by SuperDexPhysics.
            // Only link them here; don't stage duplicates.
            string[] SharedLibsLinkOnly = {
                "libmochi_physics.so",
                "libmarl.so",
            };

            foreach (string LibName in SharedLibsLinkOnly)
            {
                PublicAdditionalLibraries.Add(Path.Combine(LinuxLibsPath, LibName));
            }
        }
    }
}
