/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Collections.Generic;
using System.IO;

using CADRobotExporter.Utilities;

using Newtonsoft.Json;

namespace CADRobotExporter.Export
{
    [Serializable]
    public class ExporterConfiguration
    {
        /// <summary>
        /// Optional file next to the add-in assembly, in the same format as a saved preset, whose
        /// values replace the built-in defaults for new configurations.
        /// </summary>
        public const string DefaultsFileName = "ExporterDefaults.json";

        public MeshingOptions visualMeshingOptions;
        public MeshingOptions collisionMeshingOptions;
        public List<MeshFormat> meshFormats;
        public List<MeshFormat> collisionMeshFormats;
        public bool hasCollisionMeshFormats;
        public string meshTagExtension;
        public bool enableCollisionMeshing;
        public string collisionMeshTagExtension;
        public string robotName;
        public FolderStructure folderStructure;

        public ExporterConfiguration()
        {
            visualMeshingOptions = new MeshingOptions
            {
                linearDeflection = 0.1,
                angularDeflection = 0.5,
                scale = 1.0
            };

            collisionMeshingOptions = new MeshingOptions
            {
                linearDeflection = 0.5,
                angularDeflection = 0.75,
                scale = 1.0
            };

            meshFormats = new List<MeshFormat>
            {
#if NX
                MeshFormat.glbCAD,
#elif SOLIDWORKS
                MeshFormat.glbOpenCascade,
#endif
            };

            collisionMeshFormats = new List<MeshFormat>
            {
#if NX
                MeshFormat.objCAD,
#elif SOLIDWORKS
                MeshFormat.stlSolidworks,
#endif
            };

            hasCollisionMeshFormats = true;

#if NX
            meshTagExtension = "glb";
            collisionMeshTagExtension = "obj";
#elif SOLIDWORKS
            meshTagExtension = "glb";
            collisionMeshTagExtension = "stl";
#endif

            enableCollisionMeshing = true;

            robotName = "";

            folderStructure = FolderStructure.SuperDex;
        }

        /// <summary>
        /// The configuration a new robot starts with: the built-in defaults, overlaid with
        /// <see cref="DefaultsFileName"/> when it exists next to the add-in assembly. Fields the file
        /// leaves out keep their built-in values; a meshing options object it includes replaces the
        /// built-in one as a whole.
        /// </summary>
        public static ExporterConfiguration CreateDefault()
        {
            string path = GetDefaultsFilePath();
            if (path == null || !File.Exists(path))
            {
                return new ExporterConfiguration();
            }

            try
            {
                return JsonConvert.DeserializeObject<ExporterConfiguration>(
                    File.ReadAllText(path), PresetStore.GetJsonSettings()) ?? new ExporterConfiguration();
            }
            catch (Exception ex) when (ex is IOException || ex is UnauthorizedAccessException || ex is JsonException)
            {
                Logger.GetLogger().Warning(ex, "Ignoring unreadable defaults file {Path}", path);
                return new ExporterConfiguration();
            }
        }

        private static string GetDefaultsFilePath()
        {
            string assemblyPath = typeof(ExporterConfiguration).Assembly.Location;
            if (string.IsNullOrEmpty(assemblyPath))
            {
                return null;
            }
            return Path.Combine(Path.GetDirectoryName(assemblyPath), DefaultsFileName);
        }
    }
}
