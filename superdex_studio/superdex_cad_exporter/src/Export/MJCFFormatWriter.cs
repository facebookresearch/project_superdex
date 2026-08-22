/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System.Collections.Generic;
using System.Xml;

using CADRobotExporter.RobotDescription;
using CADRobotExporter.UI;

namespace CADRobotExporter.Export
{
    /// <summary>
    /// Writes robot models in MJCF (MuJoCo XML Format).
    /// </summary>
    public class MJCFFormatWriter : FormatWriterBase
    {
        private readonly List<string> visualMeshAssets = new List<string>();
        private readonly List<string> collisionMeshAssets = new List<string>();
        private readonly List<string> actuatorJoints = new List<string>();
        private readonly Dictionary<string, string> linkToJointName = new Dictionary<string, string>();
        private readonly FolderStructure _folderStructure;

        private Dictionary<string, List<(string siteName, double[] pos)>> tendonSitesByLink;
        private Dictionary<RoutingElement, string> tendonSiteNames;
        private HashSet<string> siteLinks;

        public MJCFFormatWriter(string savePath, FolderStructure folderStructure = FolderStructure.ROS) : base(savePath)
        {
            _folderStructure = folderStructure;
        }

        public MJCFFormatWriter(XmlWriter writer, FolderStructure folderStructure = FolderStructure.ROS) : base(writer)
        {
            _folderStructure = folderStructure;
        }

        public override void WriteRobot(RobotDescription.Robot robot)
        {
            CollectAssets(robot.BaseLink);
            siteLinks = CollectSiteLinks(robot.BaseLink);
            BuildTendonSiteMap(robot.Tendons);

            Writer.WriteStartDocument();
#if NX
            Writer.WriteComment(" Exported from NX using SuperDex CAD Exporter, https://github.com/facebookresearch/project_superdex ");
#elif SOLIDWORKS
            Writer.WriteComment(" Exported from SolidWorks using SuperDex CAD Exporter, https://github.com/facebookresearch/project_superdex ");
#endif

            Writer.WriteStartElement("mujoco");
            Writer.WriteAttributeString("model", robot.Name);

            WriteCompilerSection();
            WriteDefaultsSection();
            WriteAssetSection();
            WriteWorldbody(robot.BaseLink);
            WriteTendons(robot.Tendons);
            WriteActuatorSection();

            Writer.WriteEndElement();
            Writer.WriteEndDocument();
        }

        private void CollectAssets(Link link)
        {
            if (link.Visual?.Geometry?.Mesh?.Filename != null && !string.IsNullOrEmpty(link.Visual.Geometry.Mesh.Filename))
            {
                string visualMesh = link.Visual.Geometry.Mesh.Filename;
                if (!visualMeshAssets.Contains(visualMesh))
                {
                    visualMeshAssets.Add(visualMesh);
                }
            }

            if (link.Collision?.Geometry?.Mesh?.Filename != null && !string.IsNullOrEmpty(link.Collision.Geometry.Mesh.Filename))
            {
                string collisionMesh = link.Collision.Geometry.Mesh.Filename;
                if (!collisionMeshAssets.Contains(collisionMesh))
                {
                    collisionMeshAssets.Add(collisionMesh);
                }
            }

            if (link.Joint != null && link.Joint.ElementContainsData() && link.Joint.Type != "fixed" && !link.isSite)
            {
                actuatorJoints.Add(link.Joint.Name);
            }

            if (link.Joint != null && !string.IsNullOrEmpty(link.Joint.Name))
            {
                linkToJointName[link.Name] = link.Joint.Name;
            }

            foreach (Link child in link.Children)
            {
                CollectAssets(child);
            }
        }

        private void WriteCompilerSection()
        {
            Writer.WriteStartElement("compiler");
            Writer.WriteAttributeString("angle", "radian");

            switch (_folderStructure)
            {
                case FolderStructure.MuJoCo:
                    Writer.WriteAttributeString("meshdir", "assets");
                    break;
                case FolderStructure.ROS:
                    Writer.WriteAttributeString("meshdir", "meshes");
                    break;
                case FolderStructure.SuperDex:
                    // No common parent directory; paths are specified fully in asset references
                    break;
            }

            Writer.WriteEndElement();
        }

        private void WriteDefaultsSection()
        {
            Writer.WriteStartElement("default");

            Writer.WriteStartElement("joint");
            Writer.WriteAttributeString("damping", "0.1");
            Writer.WriteEndElement();

            Writer.WriteStartElement("default");
            Writer.WriteAttributeString("class", "visual");
            Writer.WriteStartElement("geom");
            Writer.WriteAttributeString("group", "2");
            Writer.WriteAttributeString("type", "mesh");
            Writer.WriteAttributeString("density", "0");
            Writer.WriteAttributeString("contype", "0");
            Writer.WriteAttributeString("conaffinity", "0");
            Writer.WriteEndElement();
            Writer.WriteEndElement();

            Writer.WriteStartElement("default");
            Writer.WriteAttributeString("class", "collision");
            Writer.WriteStartElement("geom");
            Writer.WriteAttributeString("group", "1");
            Writer.WriteAttributeString("type", "mesh");
            Writer.WriteAttributeString("density", "0");
            Writer.WriteEndElement();
            Writer.WriteEndElement();

            Writer.WriteEndElement();
        }

        private void WriteAssetSection()
        {
            if (visualMeshAssets.Count == 0 && collisionMeshAssets.Count == 0)
            {
                return;
            }

            Writer.WriteStartElement("asset");

            foreach (string meshPath in visualMeshAssets)
            {
                Writer.WriteStartElement("mesh");
                string meshName = System.IO.Path.GetFileNameWithoutExtension(meshPath);
                Writer.WriteAttributeString("name", meshName);
                string meshFilename = System.IO.Path.GetFileName(meshPath);

                switch (_folderStructure)
                {
                    case FolderStructure.ROS:
                        Writer.WriteAttributeString("file", "visual/" + meshFilename);
                        break;
                    case FolderStructure.SuperDex:
                        Writer.WriteAttributeString("file", "_render/" + meshFilename);
                        break;
                    case FolderStructure.Legacy:
                        Writer.WriteAttributeString("file", meshFilename);
                        break;
                    case FolderStructure.MuJoCo:
                    default:
                        Writer.WriteAttributeString("file", meshFilename);
                        break;
                }

                Writer.WriteEndElement();
            }

            foreach (string meshPath in collisionMeshAssets)
            {
                Writer.WriteStartElement("mesh");
                string meshName = System.IO.Path.GetFileNameWithoutExtension(meshPath);
                Writer.WriteAttributeString("name", meshName);
                string meshFilename = System.IO.Path.GetFileName(meshPath);

                switch (_folderStructure)
                {
                    case FolderStructure.ROS:
                        Writer.WriteAttributeString("file", "collision/" + meshFilename);
                        break;
                    case FolderStructure.SuperDex:
                        Writer.WriteAttributeString("file", "collision/" + meshFilename);
                        break;
                    case FolderStructure.Legacy:
                        Writer.WriteAttributeString("file", "collision/" + meshFilename);
                        break;
                    case FolderStructure.MuJoCo:
                    default:
                        Writer.WriteAttributeString("file", meshFilename);
                        break;
                }

                Writer.WriteEndElement();
            }

            Writer.WriteEndElement();
        }

        private string GetVisualPostfix()
        {
            switch (_folderStructure)
            {
                case FolderStructure.SuperDex:
                    return "_render";
                case FolderStructure.Legacy:
                    return "";
                case FolderStructure.ROS:
                case FolderStructure.MuJoCo:
                default:
                    return "_visual";
            }
        }

        private string GetCollisionPostfix()
        {
            switch (_folderStructure)
            {
                case FolderStructure.ROS:
                case FolderStructure.SuperDex:
                case FolderStructure.Legacy:
                case FolderStructure.MuJoCo:
                default:
                    return "_collision";
            }
        }

        private void WriteWorldbody(Link baseLink)
        {
            Writer.WriteStartElement("worldbody");
            WriteBody(baseLink, isRoot: true);
            Writer.WriteEndElement();
        }

        private void BuildTendonSiteMap(List<Tendon> tendons)
        {
            tendonSitesByLink = new Dictionary<string, List<(string siteName, double[] pos)>>();
            tendonSiteNames = new Dictionary<RoutingElement, string>();

            if (tendons == null)
                return;

            foreach (var tendon in tendons)
            {
                int waypointIndex = 0;
                foreach (var element in tendon.RoutingElements)
                {
                    if (element.Type != RoutingElement.TypeWaypoint)
                        continue;

                    string linkName = element.Link;

                    if (!string.IsNullOrEmpty(linkName) && siteLinks.Contains(linkName))
                    {
                        tendonSiteNames[element] = linkName;
                        continue;
                    }

                    waypointIndex++;
                    string siteName = $"{tendon.Name}_waypoint_{waypointIndex}";
                    tendonSiteNames[element] = siteName;

                    if (string.IsNullOrEmpty(linkName))
                        continue;

                    if (!tendonSitesByLink.ContainsKey(linkName))
                        tendonSitesByLink[linkName] = new List<(string, double[])>();

                    tendonSitesByLink[linkName].Add((siteName, element.GetPosition()));
                }
            }
        }

        private HashSet<string> CollectSiteLinks(Link link)
        {
            var result = new HashSet<string>();
            CollectSiteLinksRecursive(link, result);
            return result;
        }

        private void CollectSiteLinksRecursive(Link link, HashSet<string> result)
        {
            if (link.isSite)
                result.Add(link.Name);

            foreach (Link child in link.Children)
                CollectSiteLinksRecursive(child, result);
        }

        private void WriteBody(Link link, bool isRoot = false)
        {
            Writer.WriteStartElement("body");
            Writer.WriteAttributeString("name", link.Name);

            string visualPostfix = GetVisualPostfix();
            string collisionPostfix = GetCollisionPostfix();

            if (!isRoot && link.Joint != null && link.Joint.Origin.ElementContainsData())
            {
                double[] xyz = link.Joint.Origin.GetXYZ();
                double[] rpy = link.Joint.Origin.GetRPY();
                WriteAttributeIfNotNull("pos", xyz);
                double[] quat = RpyToQuat(rpy);
                WriteAttributeIfNotNull("quat", quat);
            }

            if (!isRoot && link.Joint != null && link.Joint.ElementContainsData())
            {
                WriteJoint(link.Joint);
            }

            if (link.Inertial != null && link.Inertial.ElementContainsData() && !link.Inertial.IsZero())
            {
                WriteInertial(link.Inertial);
            }

            if (link.Visual != null && !string.IsNullOrEmpty(link.Visual.Geometry?.Mesh?.Filename))
            {
                WriteGeomFromVisual(link.Visual, link.Name, visualPostfix);
            }

            if (link.Collision != null && !string.IsNullOrEmpty(link.Collision.Geometry?.Mesh?.Filename))
            {
                WriteGeomFromCollision(link.Collision, link.Name, collisionPostfix);
            }

            foreach (Link child in link.Children)
            {
                if (child.isSite)
                {
                    WriteSite(child);
                }
                else
                {
                    WriteBody(child);
                }
            }

            // Write auto-generated tendon waypoint sites for this body
            if (tendonSitesByLink != null && tendonSitesByLink.TryGetValue(link.Name, out var sites))
            {
                foreach (var (siteName, pos) in sites)
                {
                    Writer.WriteStartElement("site");
                    Writer.WriteAttributeString("name", siteName);
                    if (pos != null)
                        WriteAttributeIfNotNull("pos", pos);
                    Writer.WriteEndElement();
                }
            }

            Writer.WriteEndElement();
        }

        private void WriteSite(Link link)
        {
            Writer.WriteStartElement("site");
            Writer.WriteAttributeString("name", link.Name);

            if (link.Joint != null && link.Joint.Origin.ElementContainsData())
            {
                double[] xyz = link.Joint.Origin.GetXYZ();
                double[] rpy = link.Joint.Origin.GetRPY();
                WriteAttributeIfNotNull("pos", xyz);
                double[] quat = RpyToQuat(rpy);
                WriteAttributeIfNotNull("quat", quat);
            }

            Writer.WriteEndElement();
        }

        public override void WriteJoint(Joint joint)
        {
            if (joint.Type == "fixed")
            {
                return;
            }

            Writer.WriteStartElement("joint");
            Writer.WriteAttributeString("name", joint.Name);
            Writer.WriteAttributeString("type", MapJointType(joint.Type));

            if (joint.Axis.ElementContainsData())
            {
                WriteAttributeIfNotNull("axis", joint.Axis.GetXYZ());
            }

            if (joint.Limit.ElementContainsData())
            {
                Writer.WriteAttributeString("range",
                    FormatDouble(joint.Limit.Lower) + " " + FormatDouble(joint.Limit.Upper));
            }

            if (joint.Dynamics.ElementContainsData())
            {
                WriteAttributeIfNotNull("damping", joint.Dynamics.Damping);
                WriteAttributeIfNotNull("frictionloss", joint.Dynamics.Friction);
            }

            Writer.WriteEndElement();
        }

        private string MapJointType(string urdfType)
        {
            switch (urdfType)
            {
                case "revolute":
                    return "hinge";
                case "continuous":
                    return "hinge";
                case "prismatic":
                    return "slide";
                case "fixed":
                    return "fixed";
                case "floating":
                    return "free";
                case "planar":
                    return "slide";
                default:
                    return "hinge";
            }
        }

        public override void WriteInertial(Inertial inertial)
        {
            Writer.WriteStartElement("inertial");

            if (inertial.Origin.ElementContainsData())
            {
                WriteAttributeIfNotNull("pos", inertial.Origin.GetXYZ());
            }

            Writer.WriteAttributeString("mass", FormatDouble(inertial.Mass.Value));

            Inertia inertia = inertial.Inertia;
            string fullInertia = FormatDouble(inertia.Ixx) + " " +
                                 FormatDouble(inertia.Iyy) + " " +
                                 FormatDouble(inertia.Izz) + " " +
                                 FormatDouble(inertia.Ixy) + " " +
                                 FormatDouble(inertia.Ixz) + " " +
                                 FormatDouble(inertia.Iyz);
            Writer.WriteAttributeString("fullinertia", fullInertia);

            Writer.WriteEndElement();
        }

        private void WriteGeomFromVisual(Visual visual, string linkName, string postfix)
        {
            Writer.WriteStartElement("geom");
            Writer.WriteAttributeString("name", linkName + postfix);
            Writer.WriteAttributeString("type", "mesh");

            string meshName = System.IO.Path.GetFileNameWithoutExtension(visual.Geometry.Mesh.Filename);
            Writer.WriteAttributeString("mesh", meshName);

            Writer.WriteAttributeString("class", "visual");

            if (visual.Material != null && visual.Material.Color.ElementContainsData())
            {
                double[] rgba = visual.Material.Color.GetColor();
                WriteAttributeIfNotNull("rgba", rgba);
            }

            Writer.WriteEndElement();
        }

        private void WriteGeomFromCollision(Collision collision, string linkName, string postfix)
        {
            Writer.WriteStartElement("geom");
            Writer.WriteAttributeString("name", linkName + postfix);
            Writer.WriteAttributeString("type", "mesh");

            Writer.WriteAttributeString("class", "collision");

            string meshName = System.IO.Path.GetFileNameWithoutExtension(collision.Geometry.Mesh.Filename);
            Writer.WriteAttributeString("mesh", meshName);

            Writer.WriteEndElement();
        }

        private void WriteActuatorSection()
        {
            if (actuatorJoints.Count == 0)
            {
                return;
            }

            Writer.WriteStartElement("actuator");

            foreach (string jointName in actuatorJoints)
            {
                Writer.WriteStartElement("motor");
                Writer.WriteAttributeString("name", jointName);
                Writer.WriteAttributeString("joint", jointName);
                Writer.WriteEndElement();
            }

            Writer.WriteEndElement();
        }

        private static double[] RpyToQuat(double[] rpy)
        {
            if (rpy == null || rpy.Length != 3)
            {
                return new double[] { 1.0, 0.0, 0.0, 0.0 };
            }

            double roll = rpy[0];
            double pitch = rpy[1];
            double yaw = rpy[2];

            double cy = System.Math.Cos(yaw * 0.5);
            double sy = System.Math.Sin(yaw * 0.5);
            double cp = System.Math.Cos(pitch * 0.5);
            double sp = System.Math.Sin(pitch * 0.5);
            double cr = System.Math.Cos(roll * 0.5);
            double sr = System.Math.Sin(roll * 0.5);

            double w = cr * cp * cy + sr * sp * sy;
            double x = sr * cp * cy - cr * sp * sy;
            double y = cr * sp * cy + sr * cp * sy;
            double z = cr * cp * sy - sr * sp * cy;

            return new double[] { w, x, y, z };
        }

        #region Interface methods mapped to MJCF structure

        public override void WriteTendons(List<Tendon> tendons)
        {
            if (tendons == null || tendons.Count == 0)
            {
                return;
            }

            Writer.WriteStartElement("tendon");

            foreach (Tendon tendon in tendons)
            {
                bool hasSpatial = false;
                bool hasFixed = false;

                foreach (RoutingElement element in tendon.RoutingElements)
                {
                    if (element.Type == RoutingElement.TypeWaypoint)
                        hasSpatial = true;
                    else if (element.Type == RoutingElement.TypeLinearJoint)
                        hasFixed = true;
                }

                if (hasSpatial)
                {
                    Writer.WriteStartElement("spatial");
                    Writer.WriteAttributeString("name", tendon.Name);

                    foreach (RoutingElement element in tendon.RoutingElements)
                    {
                        if (element.Type == RoutingElement.TypeWaypoint &&
                            tendonSiteNames != null &&
                            tendonSiteNames.TryGetValue(element, out string siteName))
                        {
                            Writer.WriteStartElement("site");
                            Writer.WriteAttributeString("site", siteName);
                            Writer.WriteEndElement();
                        }
                    }

                    Writer.WriteEndElement();
                }

                if (hasFixed)
                {
                    Writer.WriteStartElement("fixed");
                    Writer.WriteAttributeString("name", tendon.Name);

                    foreach (RoutingElement element in tendon.RoutingElements)
                    {
                        if (element.Type == RoutingElement.TypeLinearJoint)
                        {
                            Writer.WriteStartElement("joint");
                            string jointName = linkToJointName.ContainsKey(element.Link)
                                ? linkToJointName[element.Link]
                                : element.Link;
                            Writer.WriteAttributeString("joint", jointName);
                            WriteAttributeIfNotNull("coef", element.Coefficient);
                            Writer.WriteEndElement();
                        }
                    }

                    Writer.WriteEndElement();
                }
            }

            Writer.WriteEndElement();
        }

        public override void WriteLink(Link link)
        {
            WriteBody(link);
        }

        public override void WriteVisual(Visual visual)
        {
        }

        public override void WriteCollision(Collision collision)
        {
        }

        public override void WriteOrigin(Origin origin)
        {
        }

        public override void WriteGeometry(Geometry geometry)
        {
        }

        public override void WriteMesh(Mesh mesh)
        {
        }

        public override void WriteMaterial(Material material)
        {
        }

        public override void WriteMass(Mass mass)
        {
        }

        public override void WriteInertia(Inertia inertia)
        {
        }

        public override void WriteAxis(Axis axis)
        {
        }

        public override void WriteLimit(Limit limit)
        {
        }

        public override void WriteDynamics(Dynamics dynamics)
        {
        }

        public override void WriteMimic(Mimic mimic)
        {
        }

        public override void WriteSafetyController(SafetyController safety)
        {
        }

        public override void WriteCalibration(Calibration calibration)
        {
        }

        public override void WriteColor(Color color)
        {
        }

        public override void WriteTexture(Texture texture)
        {
        }

        #endregion
    }
}
