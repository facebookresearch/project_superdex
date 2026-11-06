/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System.Collections.Generic;
using System.IO;

using CADRobotExporter.RobotDescription;
using CADRobotExporter.Utilities;

namespace CADRobotExporter.Export
{
    /// <summary>
    /// Writes the robot as a floating actor: rigid for a single link, articulated otherwise.
    /// Effort limits, the default pose and tendons have no prefab equivalent and are dropped.
    /// </summary>
    public class SuperDexPrefabFormatWriter : SuperDexBotFormatWriter
    {
        private static readonly Serilog.ILogger logger = Logger.GetLogger();

        public SuperDexPrefabFormatWriter(string savePath, FolderStructure folderStructure = FolderStructure.SuperDex)
            : base(savePath, folderStructure)
        {
        }

        protected override string RootJointType => "Free";

        protected override bool WriteEffortLimit => false;

        // Prefab paths without a "./" prefix resolve from the asset root, not the prefab's folder.
        protected override string MeshPath(string filename)
        {
            string path = StripPackagePrefix(filename);
            if (Path.IsPathRooted(path) || path.StartsWith("./") || path.StartsWith("../"))
            {
                return path;
            }
            return "./" + path;
        }

        public override void WriteRobot(Robot robot)
        {
            Articulation articulation = BuildArticulation(robot);

            if (robot.Tendons != null && robot.Tendons.Count > 0)
            {
                logger.Warning("Tendons are not supported by .superdex_prefab and were not written to " + _savePath);
            }

            var actors = new Dictionary<string, object>();
            if (IsRigid(articulation))
            {
                actors["rigid"] = new List<Dictionary<string, object>> { articulation.JsonLinks[0] };
            }
            else
            {
                actors["articulated"] = new List<Dictionary<string, object>>
                {
                    new Dictionary<string, object>
                    {
                        ["joints"] = articulation.JsonJoints,
                        ["links"] = articulation.JsonLinks,
                        ["name"] = robot.Name
                    }
                };
            }

            var root = new Dictionary<string, object>
            {
                ["_comment"] = "Exported using SuperDex CAD Exporter " + Versioning.VersionString.Get(),
                ["actors"] = actors
            };

            WriteJson(root);
        }

        // A rigid actor has no joints and needs a collision shape.
        private static bool IsRigid(Articulation articulation)
        {
            return articulation.Links.Count == 1 && articulation.JsonLinks[0].ContainsKey("shape");
        }
    }
}
