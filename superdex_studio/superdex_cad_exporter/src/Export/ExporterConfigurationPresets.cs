/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System.Collections.Generic;

namespace CADRobotExporter.Export
{
    public class ExporterConfigurationPreset
    {
        public string Name { get; set; }
        public ExporterConfiguration Configuration { get; set; }
        public bool IsUserPreset { get; set; }
        public override string ToString() => Name;
    }

    public static class ExporterConfigurationPresets
    {
        public static readonly IReadOnlyList<ExporterConfigurationPreset> Predefined =
            new List<ExporterConfigurationPreset>
        {
            new ExporterConfigurationPreset
            {
                Name = "Default",
                Configuration = ExporterConfiguration.CreateDefault(),
                IsUserPreset = false
            },
            new ExporterConfigurationPreset
            {
#if NX
                Name = "MuJoCo/ROS (obj) - Native, widest compat",
#elif SOLIDWORKS
                Name = "MuJoCo/ROS (stl) - Native, widest compat",
#endif
                Configuration = new ExporterConfiguration
                {
                    visualMeshingOptions = new MeshingOptions
                    {
                        linearDeflection = 0.5,
                        angularDeflection = 0.5,
                        scale = 1.0
                    },
                    collisionMeshingOptions = new MeshingOptions
                    {
                        linearDeflection = 1.0,
                        angularDeflection = 0.75,
                        scale = 1.0
                    },
                    enableCollisionMeshing = true,
#if NX
                    meshFormats = new List<MeshFormat> { MeshFormat.objCAD },
                    collisionMeshFormats = new List<MeshFormat> { MeshFormat.objCAD },
#elif SOLIDWORKS
                    meshFormats = new List<MeshFormat> { MeshFormat.stlSolidworks },
                    collisionMeshFormats = new List<MeshFormat> { MeshFormat.stlSolidworks },
#endif
#if NX
                    meshTagExtension = "obj",
                    collisionMeshTagExtension = "obj",
#elif SOLIDWORKS
                    meshTagExtension = "stl",
                    collisionMeshTagExtension = "stl",
#endif
                    hasCollisionMeshFormats = true,
                    folderStructure = FolderStructure.ROS,
                },
                IsUserPreset = false
            },
            new ExporterConfigurationPreset
            {
#if NX
                Name = "MuJoCo/ROS (stl) - Native, widest compat",
#elif SOLIDWORKS
                Name = "MuJoCo (obj) - SuperDex, widest compat",
#endif
                Configuration = new ExporterConfiguration
                {
                    visualMeshingOptions = new MeshingOptions
                    {
                        linearDeflection = 0.5,
                        angularDeflection = 0.5,
                        scale = 1.0
                    },
                    collisionMeshingOptions = new MeshingOptions
                    {
                        linearDeflection = 1.0,
                        angularDeflection = 0.75,
                        scale = 1.0
                    },
                    enableCollisionMeshing = true,
#if NX
                    meshFormats = new List<MeshFormat> { MeshFormat.stlSolidworks },
                    collisionMeshFormats = new List<MeshFormat> { MeshFormat.stlSolidworks },
#elif SOLIDWORKS
                    meshFormats = new List<MeshFormat> { MeshFormat.objOpenCascade },
                    collisionMeshFormats = new List<MeshFormat> { MeshFormat.objOpenCascade },
#endif
#if NX
                    meshTagExtension = "stl",
                    collisionMeshTagExtension = "stl",
#elif SOLIDWORKS
                    meshTagExtension = "obj",
                    collisionMeshTagExtension = "obj",
#endif
                    hasCollisionMeshFormats = true,
                    folderStructure = FolderStructure.ROS,
                },
                IsUserPreset = false
            },
            new ExporterConfigurationPreset
            {
#if NX
                Name = "SuperDex (glb) - Quick preview",
#elif SOLIDWORKS
                Name = "SuperDex (glb, stl) - Quick preview",
#endif
                Configuration = new ExporterConfiguration
                {
                    visualMeshingOptions = new MeshingOptions
                    {
                        linearDeflection = 0.1,
                        angularDeflection = 0.5,
                        scale = 1.0
                    },
                    collisionMeshingOptions = new MeshingOptions
                    {
                        linearDeflection = 0.5,
                        angularDeflection = 0.75,
                        scale = 1.0
                    },
                    enableCollisionMeshing = true,
#if NX
                    meshFormats = new List<MeshFormat> { MeshFormat.glbCAD },
                    collisionMeshFormats = new List<MeshFormat> { MeshFormat.glbCAD },
#elif SOLIDWORKS
                    meshFormats = new List<MeshFormat> { MeshFormat.glbOpenCascade },
                    collisionMeshFormats = new List<MeshFormat> { MeshFormat.stlSolidworks },
#endif
                    meshTagExtension = "glb",
#if NX
                    collisionMeshTagExtension = "glb",
#elif SOLIDWORKS
                    collisionMeshTagExtension = "stl",
#endif
                    hasCollisionMeshFormats = true,
                    folderStructure = FolderStructure.SuperDex,
                },
                IsUserPreset = false
            },
            new ExporterConfigurationPreset
            {
#if NX
                Name = "SuperDex (glb, stp) - Studio import",
#elif SOLIDWORKS
                Name = "SuperDex (glb, stl, stp) - Studio import",
#endif
                Configuration = new ExporterConfiguration
                {
                    visualMeshingOptions = new MeshingOptions
                    {
                        linearDeflection = 0.1,
                        angularDeflection = 0.5,
                        scale = 1.0
                    },
                    collisionMeshingOptions = new MeshingOptions
                    {
                        linearDeflection = 0.5,
                        angularDeflection = 0.75,
                        scale = 1.0
                    },
                    enableCollisionMeshing = true,
#if NX
                    meshFormats = new List<MeshFormat> { MeshFormat.glbCAD },
                    collisionMeshFormats = new List<MeshFormat> { MeshFormat.glbCAD, MeshFormat.stepSolidworks },
#elif SOLIDWORKS
                    meshFormats = new List<MeshFormat> { MeshFormat.glbOpenCascade },
                    collisionMeshFormats = new List<MeshFormat> { MeshFormat.stlSolidworks, MeshFormat.stepSolidworks },
#endif
                    meshTagExtension = "glb",
                    collisionMeshTagExtension = "mochi.h5",
                    hasCollisionMeshFormats = true,
                    folderStructure = FolderStructure.SuperDex,
                },
                IsUserPreset = false
            },
        };
    }
}
