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
    public class SuperDexRobotics : ModuleRules
    {
        public SuperDexRobotics(ReadOnlyTargetRules Target) : base(Target)
        {
            CppStandard = CppStandardVersion.Cpp20;

            // Unreal Module dependencies
            PublicDependencyModuleNames.AddRange(new string[] {
                "Core",
                "CoreUObject",
                "Engine",
                "DeveloperSettings",
                "GeometryCore",
                "GeometryFramework",
                "DynamicMesh",
                "ModelingComponents",
                // For UMochiArticulationBase, IMochiPhysicsModule, MochiTypes, URDF/
                // MochiUrdfConversions.h (TMochiToUnrealConverter, kUrdfSpace), etc.
                "SuperDexPhysics",
            });

            PrivateDependencyModuleNames.AddRange(new string[] {
                "Projects",
            });

            // USuperDexBotComponent::OnRegister() calls GEditor->GetEditorSubsystem<UImportSubsystem>()
            // under #if WITH_EDITOR; GEditor is exported by UnrealEd.
            if (Target.bBuildEditor)
            {
                PrivateDependencyModuleNames.Add("UnrealEd");
            }

            // Path to the root of this plugin (with the .uplugin file)
            string pluginRootPath = Path.GetFullPath(Path.Combine(ModuleDirectory, "../.."));

            // Path to the sibling MochiPhysics plugin. mochi_physics.dll is staged exclusively
            // into that plugin's ThirdParty tree; SuperDexRobotics links against that single copy
            // rather than duplicating the binary into its own, to avoid two copies drifting out
            // of sync.
            string mochiPhysicsPluginRootPath =
                Path.GetFullPath(Path.Combine(pluginRootPath, "../SuperDexPhysics"));

            // Path to the folder with our own (superdex_robotics) includes/libs. mochi_physics and
            // mochi_core public include paths are inherited transitively from the
            // "SuperDexPhysics" module dependency above (its own PublicIncludePaths propagate
            // to any module that depends on it), so only superdex_robotics's own include path is
            // added here.
            string mochiThirdPartyPath = Path.GetFullPath(Path.Combine(ModuleDirectory, "ThirdParty"));

            // Add includes folder to PublicIncludes
            // This allows users of this plugin to #include headers needed to use the C APIs of superdex_robotics
            PublicIncludePaths.Add(Path.Combine(mochiThirdPartyPath, "superdex_robotics/include"));

            // link mochi libraries. SuperDexRobotics classes (USuperDexBotComponent,
            // USuperDexIKSolverObject, FSuperDexBotLoader, etc.) call directly into mochi:: namespace
            // physics types (mochi::Scene, mochi::Actor, mochi::Context, ...) in addition to
            // superdex::robotics::, so both superdex_robotics.dll and mochi_physics.dll are needed here.
            if (Target.Platform == UnrealTargetPlatform.Win64)
            {
                string superDexRoboticsLibsPath = Path.Combine(pluginRootPath, "ThirdParty/superdex_robotics/lib/Win64");
                string mochiPhysicsLibsPath =
                    Path.Combine(mochiPhysicsPluginRootPath, "ThirdParty/mochi_physics/lib/Win64");

                // DLLs staged into this plugin's own ThirdParty tree
                string[] ownDllsWithImportLibs = {
                    "superdex_robotics.dll",
                };
                foreach (string dllName in ownDllsWithImportLibs)
                {
                    PublicAdditionalLibraries.Add(Path.Combine(superDexRoboticsLibsPath, dllName + ".imp.lib"));
                    RuntimeDependencies.Add(Path.Combine(superDexRoboticsLibsPath, dllName));
                    PublicDelayLoadDLLs.Add(dllName);
                }

                // DLLs owned by (and already packaged/staged by) the sibling MochiPhysics plugin.
                // Only the import lib is referenced here for linking; RuntimeDependencies/staging
                // is intentionally left to SuperDexPhysics.Build.cs to avoid double-packaging.
                PublicAdditionalLibraries.Add(Path.Combine(mochiPhysicsLibsPath, "mochi_physics.dll.imp.lib"));
                PublicDelayLoadDLLs.Add("mochi_physics.dll");
            }
            else if (Target.Platform == UnrealTargetPlatform.Android)
            {
                string superDexRoboticsLibsPath = Path.Combine(pluginRootPath, "ThirdParty/superdex_robotics/lib/Android");
                string mochiPhysicsLibsPath =
                    Path.Combine(mochiPhysicsPluginRootPath, "ThirdParty/mochi_physics/lib/Android");

                PublicAdditionalLibraries.Add(Path.Combine(superDexRoboticsLibsPath, "libsuperdex_robotics.so"));
                PublicAdditionalLibraries.Add(Path.Combine(mochiPhysicsLibsPath, "libmochi_physics.so"));

                AdditionalPropertiesForReceipt.Add(
                    "AndroidPlugin", Path.Combine(superDexRoboticsLibsPath, "superdex_robotics_apl.xml"));
            }
            else if (Target.Platform == UnrealTargetPlatform.Linux)
            {
                // See matching comments in SuperDexPhysics.Build.cs for why these are needed.
                PublicDefinitions.Add("MOCHI_UNREAL_LINUX=1");
                PublicDefinitions.Add(Target.bBuildEditor
                    ? "MOCHI_UNREAL_LINUX_EDITOR=1"
                    : "MOCHI_UNREAL_LINUX_EDITOR=0");

                string superDexRoboticsLibsPath = Path.Combine(pluginRootPath, "ThirdParty/superdex_robotics/lib/Linux");
                string mochiPhysicsLibsPath =
                    Path.Combine(mochiPhysicsPluginRootPath, "ThirdParty/mochi_physics/lib/Linux");

                PublicSystemLibraryPaths.Add(superDexRoboticsLibsPath);
                PublicSystemLibraryPaths.Add(mochiPhysicsLibsPath);

                PublicAdditionalLibraries.Add(Path.Combine(superDexRoboticsLibsPath, "libsuperdex_robotics.so"));
                RuntimeDependencies.Add(
                    "$(BinaryOutputDir)/libsuperdex_robotics.so", Path.Combine(superDexRoboticsLibsPath, "libsuperdex_robotics.so"));

                PublicAdditionalLibraries.Add(Path.Combine(mochiPhysicsLibsPath, "libmochi_physics.so"));
            }
        }
    }
}
