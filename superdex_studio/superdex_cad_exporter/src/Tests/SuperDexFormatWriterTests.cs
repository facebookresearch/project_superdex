/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System.IO;

using CADRobotExporter.Export;
using CADRobotExporter.RobotDescription;
using Newtonsoft.Json.Linq;
using Xunit;

namespace CADRobotExporter.Tests
{
    public class SuperDexFormatWriterTests
    {
        private static Robot SingleLinkRobot(bool withCollision)
        {
            var robot = new Robot { Name = "part" };
            robot.BaseLink.Name = "base";
            robot.BaseLink.Visual.Geometry.Mesh.Filename = "package://part/render/base.glb";
            if (withCollision)
            {
                robot.BaseLink.Collision.Geometry.Mesh.Filename = "package://part/collision/base.stl";
            }
            return robot;
        }

        private static Robot TwoLinkRobot()
        {
            Robot robot = SingleLinkRobot(true);
            var child = new Link { Name = "arm" };
            child.Joint.Name = "pivot";
            child.Joint.Type = "revolute";
            child.Joint.Axis.SetXYZ(new double[] { 0, 0, 1 });
            child.Joint.Limit.Lower = -1;
            child.Joint.Limit.Upper = 1;
            child.Joint.Limit.Effort = 5;
            child.Collision.Geometry.Mesh.Filename = "package://part/collision/arm.stl";
            robot.BaseLink.AddChild(child);
            return robot;
        }

        private static JObject Write(IFormatWriter writer, string path, Robot robot)
        {
            writer.WriteRobot(robot);
            try
            {
                return JObject.Parse(File.ReadAllText(path));
            }
            finally
            {
                File.Delete(path);
            }
        }

        private static JObject WritePrefab(Robot robot)
        {
            string path = Path.GetTempFileName();
            return Write(new SuperDexPrefabFormatWriter(path), path, robot);
        }

        private static JObject WriteBot(Robot robot)
        {
            string path = Path.GetTempFileName();
            return Write(new SuperDexBotFormatWriter(path), path, robot);
        }

        [Fact]
        public void Prefab_SingleLinkWithCollision_IsRigidWithPrefabRelativePaths()
        {
            JObject prefab = WritePrefab(SingleLinkRobot(withCollision: true));

            JToken rigid = prefab["actors"]["rigid"][0];
            Assert.Null(prefab["actors"]["articulated"]);
            Assert.Equal("base", (string)rigid["name"]);
            Assert.Equal("./collision/base.stl", (string)rigid["shape"]);
            Assert.Equal("./render/base.glb", (string)rigid["renderModel"]);
        }

        [Fact]
        public void Prefab_SingleLinkWithoutCollision_IsArticulated()
        {
            JObject prefab = WritePrefab(SingleLinkRobot(withCollision: false));

            Assert.Null(prefab["actors"]["rigid"]);
            Assert.Equal("Free", (string)prefab["actors"]["articulated"][0]["joints"][0]["type"]);
        }

        [Fact]
        public void Prefab_MultipleLinks_IsFloatingArticulationWithoutBotFields()
        {
            JObject prefab = WritePrefab(TwoLinkRobot());

            JToken actor = prefab["actors"]["articulated"][0];
            Assert.Equal("part", (string)actor["name"]);
            Assert.Equal("Free", (string)actor["joints"][0]["type"]);
            Assert.Equal("Revolute", (string)actor["joints"][1]["type"]);
            Assert.Null(actor["joints"][1]["effortLimit"]);
            Assert.Equal(0, (int)actor["links"][1]["parentLink"]);
            Assert.Equal("./collision/arm.stl", (string)actor["links"][1]["shape"]);
            Assert.Null(prefab["defaultPose"]);
        }

        [Fact]
        public void Bot_KeepsFixedRootEffortLimitAndPlainPaths()
        {
            JObject bot = WriteBot(TwoLinkRobot());

            Assert.Equal("Hard", (string)bot["joints"][0]["type"]);
            Assert.Equal(5, (double)bot["joints"][1]["effortLimit"]);
            Assert.Equal("collision/arm.stl", (string)bot["links"][1]["shape"]);
        }

        [Fact]
        public void Bot_WritesZeroCenterOfMassAlongsideInertia()
        {
            Robot robot = SingleLinkRobot(withCollision: true);
            robot.BaseLink.Inertial.Mass.Value = 2;
            robot.BaseLink.Inertial.Inertia.Ixx = 1;
            robot.BaseLink.Inertial.Inertia.Iyy = 1;
            robot.BaseLink.Inertial.Inertia.Izz = 1;

            JToken link = WriteBot(robot)["links"][0];

            Assert.Equal(2, (double)link["mass"]);
            Assert.Equal(new double[] { 0, 0, 0 }, link["centerOfMass"].ToObject<double[]>());
            Assert.NotNull(link["momentOfInertia"]);
        }

        [Fact]
        public void Bot_WithoutMass_WritesNoInertialFields()
        {
            Robot robot = SingleLinkRobot(withCollision: true);
            robot.BaseLink.Inertial.Inertia.Ixx = 1;

            JToken link = WriteBot(robot)["links"][0];

            Assert.Null(link["mass"]);
            Assert.Null(link["centerOfMass"]);
            Assert.Null(link["momentOfInertia"]);
        }
    }
}
