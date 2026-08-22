/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;

using Newtonsoft.Json;

using CADRobotExporter.RobotDescription;
using CADRobotExporter.UI;

namespace CADRobotExporter.Export
{
    public class SuperDexBotFormatWriter : IFormatWriter
    {
        private readonly string _savePath;
        private readonly FolderStructure _folderStructure;

        public SuperDexBotFormatWriter(string savePath, FolderStructure folderStructure = FolderStructure.SuperDex)
        {
            _savePath = savePath;
            _folderStructure = folderStructure;
        }

        public void WriteRobot(Robot robot)
        {
            var linkIndices = new Dictionary<string, int>();
            var orderedLinks = new List<Link>();
            var parentMap = new Dictionary<string, int>();

            CollectLinks(robot.BaseLink, orderedLinks, parentMap);

            for (int i = 0; i < orderedLinks.Count; i++)
            {
                linkIndices[orderedLinks[i].Name] = i;
            }

            var jsonJoints = new List<Dictionary<string, object>>();
            var jsonLinks = new List<Dictionary<string, object>>();
            int dofCount = 0;

            for (int i = 0; i < orderedLinks.Count; i++)
            {
                var link = orderedLinks[i];
                var jointObj = BuildJoint(link, i == 0, linkIndices);
                jsonJoints.Add(jointObj);

                if (i > 0 && link.Joint != null && !IsFixedJoint(link))
                {
                    dofCount++;
                }

                var linkObj = BuildLink(link, i, parentMap);
                jsonLinks.Add(linkObj);
            }

            var defaultPose = new List<double>();
            for (int i = 0; i < dofCount; i++)
            {
                defaultPose.Add(0);
            }

            var root = new Dictionary<string, object>
            {
                ["defaultPose"] = defaultPose,
                ["joints"] = jsonJoints,
                ["links"] = jsonLinks,
                ["name"] = robot.Name
            };

            var spatialTendons = BuildSpatialTendons(robot.Tendons, linkIndices);
            if (spatialTendons.Count > 0)
            {
                root["spatialTendons"] = spatialTendons;
            }

            var settings = new JsonSerializerSettings
            {
                Formatting = Formatting.Indented,
                NullValueHandling = NullValueHandling.Ignore
            };

            string json = JsonConvert.SerializeObject(root, settings).Replace("\r\n", "\n");
            File.WriteAllText(_savePath, json + "\n", new System.Text.UTF8Encoding(false));
        }

        private void CollectLinks(Link link, List<Link> result, Dictionary<string, int> parentMap)
        {
            int myIndex = result.Count;
            result.Add(link);

            var children = new List<Link>();
            foreach (var child in link.Children)
            {
                if (!child.isFixedFrame)
                {
                    children.Add(child);
                }
            }
            children.Sort((a, b) => NaturalCompare(a.Name, b.Name));

            foreach (var child in children)
            {
                parentMap[child.Name] = myIndex;
                CollectLinks(child, result, parentMap);
            }
        }

        private static int NaturalCompare(string a, string b)
        {
            int ia = 0, ib = 0;
            while (ia < a.Length && ib < b.Length)
            {
                bool aDigit = char.IsDigit(a[ia]);
                bool bDigit = char.IsDigit(b[ib]);

                if (aDigit && bDigit)
                {
                    int startA = ia, startB = ib;
                    while (ia < a.Length && char.IsDigit(a[ia])) ia++;
                    while (ib < b.Length && char.IsDigit(b[ib])) ib++;
                    long numA = long.Parse(a.Substring(startA, ia - startA));
                    long numB = long.Parse(b.Substring(startB, ib - startB));
                    if (numA != numB) return numA.CompareTo(numB);
                }
                else if (aDigit != bDigit)
                {
                    return aDigit ? -1 : 1;
                }
                else
                {
                    if (a[ia] != b[ib]) return a[ia].CompareTo(b[ib]);
                    ia++;
                    ib++;
                }
            }
            return a.Length.CompareTo(b.Length);
        }

        private bool IsFixedJoint(Link link)
        {
            if (link.isSite) return true;
            if (link.Joint == null) return true;
            string type = link.Joint.Type;
            return type == "fixed";
        }

        private Dictionary<string, object> BuildJoint(Link link, bool isRoot, Dictionary<string, int> linkIndices)
        {
            var joint = new Dictionary<string, object>();

            if (isRoot)
            {
                joint["name"] = "world_joint";
                joint["type"] = "Hard";
                return joint;
            }

            var j = link.Joint;
            if (j == null)
            {
                joint["name"] = link.Name + "_joint";
                joint["type"] = "Hard";
                return joint;
            }

            if (link.isSite)
            {
                string name = !string.IsNullOrEmpty(j.Name) ? j.Name : link.Name + "_joint";
                joint["name"] = name;

                var siteTransform = BuildParentLinkFromJoint(j);
                if (siteTransform.Count > 0)
                {
                    joint["parentLinkFromJoint"] = siteTransform;
                }

                joint["type"] = "Hard";
                return joint;
            }

            string superdexType = MapJointType(j.Type, link.isSite);

            if (superdexType != "Hard" && j.Axis != null && j.Axis.ElementContainsData())
            {
                double[] axis = j.Axis.GetXYZ();
                joint["axis"] = new List<double> { axis[0], axis[1], axis[2] };
            }

            if (superdexType != "Hard" && j.Limit != null && j.Limit.ElementContainsData())
            {
                if (j.Limit.IsEffortSet() && j.Limit.Effort > 0)
                {
                    joint["effortLimit"] = j.Limit.Effort;
                }

                if (j.Dynamics != null && j.Dynamics.IsFrictionSet() && j.Dynamics.Friction > 0)
                {
                    joint["friction"] = new Dictionary<string, object>
                    {
                        ["coulomb"] = j.Dynamics.Friction
                    };
                }

                if (j.Dynamics != null && j.Dynamics.IsDampingSet() && j.Dynamics.Damping > 0)
                {
                    joint["inertia"] = j.Dynamics.Damping;
                }

                double[] ax = j.Axis != null && j.Axis.ElementContainsData() ? j.Axis.GetXYZ() : new double[] { 0, 0, 1 };
                double upper = j.Limit.IsUpperSet() ? j.Limit.Upper : 0;
                double lower = j.Limit.IsLowerSet() ? j.Limit.Lower : 0;
                joint["maxLimit"] = new List<double> { ax[0] * upper, ax[1] * upper, ax[2] * upper };
                joint["minLimit"] = new List<double> { ax[0] * lower, ax[1] * lower, ax[2] * lower };
            }

            joint["name"] = j.Name;

            var parentLinkFromJoint = BuildParentLinkFromJoint(j);
            if (parentLinkFromJoint.Count > 0)
            {
                joint["parentLinkFromJoint"] = parentLinkFromJoint;
            }

            joint["type"] = superdexType;

            return joint;
        }

        private Dictionary<string, object> BuildParentLinkFromJoint(Joint joint)
        {
            var transform = new Dictionary<string, object>();

            if (joint.Origin != null && joint.Origin.ElementContainsData())
            {
                double[] rpy = joint.Origin.GetRPY();
                if (rpy != null && (rpy[0] != 0 || rpy[1] != 0 || rpy[2] != 0))
                {
                    double[] quat = RpyToQuatXYZW(rpy);
                    transform["rotation"] = new List<double> { quat[0], quat[1], quat[2], quat[3] };
                }

                double[] xyz = joint.Origin.GetXYZ();
                if (xyz != null && (xyz[0] != 0 || xyz[1] != 0 || xyz[2] != 0))
                {
                    transform["translation"] = new List<double> { xyz[0], xyz[1], xyz[2] };
                }
            }

            return transform;
        }

        private Dictionary<string, object> BuildLink(Link link, int index, Dictionary<string, int> parentMap)
        {
            var obj = new Dictionary<string, object>();

            bool hasInertial = link.Inertial != null && link.Inertial.ElementContainsData() && !link.Inertial.IsZero();
            if (hasInertial)
            {
                double[] com = link.Inertial.Origin.GetXYZ();
                if (com != null && (com[0] != 0 || com[1] != 0 || com[2] != 0))
                {
                    obj["centerOfMass"] = new List<double> { com[0], com[1], com[2] };
                }

                if (link.Inertial.Mass.Value > 0)
                {
                    obj["mass"] = link.Inertial.Mass.Value;
                }

                var inertia = link.Inertial.Inertia;
                obj["momentOfInertia"] = new List<double>
                {
                    inertia.Ixx, inertia.Ixy, inertia.Ixz,
                    inertia.Iyy, inertia.Iyz, inertia.Izz
                };
            }

            obj["name"] = link.Name;

            if (index > 0 && parentMap.ContainsKey(link.Name))
            {
                obj["parentLink"] = parentMap[link.Name];
            }

            string visualPath = GetVisualMeshPath(link);
            if (!string.IsNullOrEmpty(visualPath))
            {
                obj["renderModel"] = visualPath;

                if (link.Visual?.Origin != null && link.Visual.Origin.ElementContainsData())
                {
                    double[] rpy = link.Visual.Origin.GetRPY();
                    if (rpy != null && (rpy[0] != 0 || rpy[1] != 0 || rpy[2] != 0))
                    {
                        obj["renderModelRotation"] = QuatAsList(RpyToQuatXYZW(rpy));
                    }
                    double[] xyz = link.Visual.Origin.GetXYZ();
                    if (xyz != null && (xyz[0] != 0 || xyz[1] != 0 || xyz[2] != 0))
                    {
                        obj["renderModelTranslation"] = new List<double> { xyz[0], xyz[1], xyz[2] };
                    }
                }
            }

            string collisionPath = GetCollisionMeshPath(link);
            if (!string.IsNullOrEmpty(collisionPath))
            {
                obj["shape"] = collisionPath;

                if (link.Collision?.Origin != null && link.Collision.Origin.ElementContainsData())
                {
                    double[] rpy = link.Collision.Origin.GetRPY();
                    if (rpy != null && (rpy[0] != 0 || rpy[1] != 0 || rpy[2] != 0))
                    {
                        obj["shapeRotation"] = QuatAsList(RpyToQuatXYZW(rpy));
                    }
                    double[] xyz = link.Collision.Origin.GetXYZ();
                    if (xyz != null && (xyz[0] != 0 || xyz[1] != 0 || xyz[2] != 0))
                    {
                        obj["shapeTranslation"] = new List<double> { xyz[0], xyz[1], xyz[2] };
                    }
                }
            }

            return obj;
        }

        private string GetVisualMeshPath(Link link)
        {
            if (link.Visual?.Geometry?.Mesh?.Filename == null)
                return null;
            string filename = link.Visual.Geometry.Mesh.Filename;
            if (string.IsNullOrEmpty(filename))
                return null;
            return StripPackagePrefix(filename);
        }

        private string GetCollisionMeshPath(Link link)
        {
            if (link.Collision?.Geometry?.Mesh?.Filename == null)
                return null;
            string filename = link.Collision.Geometry.Mesh.Filename;
            if (string.IsNullOrEmpty(filename))
                return null;
            return StripPackagePrefix(filename);
        }

        private static string StripPackagePrefix(string path)
        {
            var match = Regex.Match(path, @"package://[^/]+/(.+)$");
            if (match.Success)
            {
                return match.Groups[1].Value;
            }
            return path;
        }

        private List<Dictionary<string, object>> BuildSpatialTendons(
            List<Tendon> tendons,
            Dictionary<string, int> linkIndices)
        {
            var result = new List<Dictionary<string, object>>();
            if (tendons == null)
                return result;

            foreach (var tendon in tendons)
            {
                var tendonObj = new Dictionary<string, object>();
                var routingElements = new List<Dictionary<string, object>>();

                double defaultStiffness = 1000;
                double defaultDamping = 200;
                bool hasWaypoints = false;

                foreach (var element in tendon.RoutingElements)
                {
                    if (element.Type == RoutingElement.TypeWaypoint)
                    {
                        hasWaypoints = true;
                        var routeObj = new Dictionary<string, object>();
                        double[] pos = element.GetPosition();

                        if (!string.IsNullOrEmpty(element.Link) && linkIndices.ContainsKey(element.Link))
                        {
                            int idx = linkIndices[element.Link];
                            if (idx != 0)
                            {
                                routeObj["index"] = idx;
                            }
                        }

                        routeObj["localPosition"] = new List<double> { pos[0], pos[1], pos[2] };
                        routingElements.Add(routeObj);
                    }
                    else if (element.Type == RoutingElement.TypeLinearJoint)
                    {
                        var routeObj = new Dictionary<string, object>();
                        routeObj["coefficient"] = element.Coefficient;

                        if (!string.IsNullOrEmpty(element.Link) && linkIndices.ContainsKey(element.Link))
                        {
                            routeObj["index"] = linkIndices[element.Link];
                        }

                        routeObj["type"] = "LinearJoint";
                        routingElements.Add(routeObj);
                    }
                }

                if (routingElements.Count == 0)
                    continue;

                if (hasWaypoints)
                {
                    tendonObj["damping"] = defaultDamping;
                }

                tendonObj["name"] = tendon.Name;
                tendonObj["routingElements"] = routingElements;

                if (hasWaypoints)
                {
                    tendonObj["stiffness"] = defaultStiffness;
                }

                result.Add(tendonObj);
            }

            return result;
        }

        private static string MapJointType(string urdfType, bool isSite)
        {
            if (isSite) return "Hard";
            switch (urdfType)
            {
                case "revolute":
                case "continuous":
                    return "Revolute";
                case "prismatic":
                    return "Prismatic";
                case "fixed":
                    return "Hard";
                default:
                    return "Hard";
            }
        }

        private static double[] RpyToQuatXYZW(double[] rpy)
        {
            if (rpy == null || rpy.Length != 3)
                return new double[] { 0, 0, 0, 1 };

            double roll = rpy[0];
            double pitch = rpy[1];
            double yaw = rpy[2];

            double cy = Math.Cos(yaw * 0.5);
            double sy = Math.Sin(yaw * 0.5);
            double cp = Math.Cos(pitch * 0.5);
            double sp = Math.Sin(pitch * 0.5);
            double cr = Math.Cos(roll * 0.5);
            double sr = Math.Sin(roll * 0.5);

            double w = cr * cp * cy + sr * sp * sy;
            double x = sr * cp * cy - cr * sp * sy;
            double y = cr * sp * cy + sr * cp * sy;
            double z = cr * cp * sy - sr * sp * cy;

            return new double[] { x, y, z, w };
        }

        private static List<double> QuatAsList(double[] q)
        {
            return new List<double> { q[0], q[1], q[2], q[3] };
        }

        #region IFormatWriter interface stubs (not used for JSON output)

        public void WriteLink(Link link) { }
        public void WriteJoint(Joint joint) { }
        public void WriteInertial(Inertial inertial) { }
        public void WriteVisual(Visual visual) { }
        public void WriteCollision(Collision collision) { }
        public void WriteOrigin(Origin origin) { }
        public void WriteGeometry(Geometry geometry) { }
        public void WriteMesh(Mesh mesh) { }
        public void WriteMaterial(Material material) { }
        public void WriteInertia(Inertia inertia) { }
        public void WriteMass(Mass mass) { }
        public void WriteAxis(Axis axis) { }
        public void WriteLimit(Limit limit) { }
        public void WriteDynamics(Dynamics dynamics) { }
        public void WriteMimic(Mimic mimic) { }
        public void WriteSafetyController(SafetyController safety) { }
        public void WriteCalibration(Calibration calibration) { }
        public void WriteColor(Color color) { }
        public void WriteTexture(Texture texture) { }
        public void WriteTendons(List<Tendon> tendons) { }

        public void Dispose() { }

        #endregion
    }
}
