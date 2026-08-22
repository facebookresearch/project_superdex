/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System.Collections.Generic;
using System.Xml;

using CADRobotExporter.RobotDescription;

namespace CADRobotExporter.Export
{
    /// <summary>
    /// Writes robot models in URDF (Unified Robot Description Format).
    /// </summary>
    public class URDFFormatWriter : FormatWriterBase
    {
        public URDFFormatWriter(string savePath) : base(savePath)
        {
        }

        public URDFFormatWriter(XmlWriter writer) : base(writer)
        {
        }

        public override void WriteRobot(RobotDescription.Robot robot)
        {
            Writer.WriteStartDocument();
#if NX
            Writer.WriteComment(" Exported from NX using SuperDex CAD Exporter: https://github.com/facebookresearch/project_superdex ");
#elif SOLIDWORKS
            Writer.WriteComment(" Exported from SolidWorks using SuperDex CAD Exporter: https://github.com/facebookresearch/project_superdex ");
#endif

            Writer.WriteStartElement("robot");
            Writer.WriteAttributeString("name", robot.Name);

            Writer.WriteComment("Links");
            WriteLink(robot.BaseLink);

            Writer.WriteComment("Joints");
            WriteJoints(robot.BaseLink);

            WriteTendons(robot.Tendons);

            Writer.WriteEndElement();
            Writer.WriteEndDocument();
        }

        public override void WriteLink(Link link)
        {
            Writer.WriteStartElement("link");
            Writer.WriteAttributeString("name", link.Name);

            if (link.Inertial != null && link.Inertial.ElementContainsData())
            {
                WriteInertial(link.Inertial);
            }

            if (link.Visual != null && !string.IsNullOrEmpty(link.Visual.Geometry?.Mesh?.Filename))
            {
                WriteVisual(link.Visual);
            }

            if (link.Collision != null && !string.IsNullOrEmpty(link.Collision.Geometry?.Mesh?.Filename))
            {
                WriteCollision(link.Collision);
            }

            Writer.WriteEndElement();

            foreach (Link child in link.Children)
            {
                WriteLink(child);
            }
        }

        public void WriteJoints(Link link)
        {
            if (link.Joint != null && link.Joint.ElementContainsData())
            {
                if (link.isSite)
                {
                    WriteSiteJoint(link);
                }
                else
                {
                    WriteJoint(link.Joint);
                }
            }
            else if (link.isSite)
            {
                WriteSiteJoint(link);
            }

            foreach (Link child in link.Children)
            {
                WriteJoints(child);
            }
        }

        public override void WriteTendons(List<Tendon> tendons)
        {
            if (tendons == null || tendons.Count == 0)
            {
                return;
            }

            Writer.WriteComment("Tendons");
            foreach (Tendon tendon in tendons)
            {
                Writer.WriteStartElement("tendon");
                Writer.WriteAttributeString("name", tendon.Name);

                foreach (RoutingElement element in tendon.RoutingElements)
                {
                    Writer.WriteStartElement("routing_element");
                    Writer.WriteAttributeString("link", element.Link);
                    Writer.WriteAttributeString("type", element.Type);
                    if (element.Type == RoutingElement.TypeWaypoint)
                    {
                        WriteAttributeIfNotNull("xyz", element.GetPosition());
                    }
                    else if (element.Type == RoutingElement.TypeLinearJoint)
                    {
                        WriteAttributeIfNotNull("coef", element.Coefficient);
                    }
                    Writer.WriteEndElement();
                }

                Writer.WriteEndElement();
            }
        }

        private void WriteSiteJoint(Link link)
        {
            Joint joint = link.Joint;
            string jointName = (joint != null && !string.IsNullOrEmpty(joint.Name))
                ? joint.Name
                : link.Name + "_joint";

            Writer.WriteStartElement("joint");
            Writer.WriteAttributeString("name", jointName);
            Writer.WriteAttributeString("type", "fixed");

            if (joint != null && joint.Origin.ElementContainsData())
            {
                WriteOrigin(joint.Origin);
            }

            if (joint?.Parent != null && joint.Parent.ElementContainsData())
            {
                Writer.WriteStartElement("parent");
                Writer.WriteAttributeString("link", joint.Parent.Name);
                Writer.WriteEndElement();
            }

            if (joint?.Child != null && joint.Child.ElementContainsData())
            {
                Writer.WriteStartElement("child");
                Writer.WriteAttributeString("link", joint.Child.Name);
                Writer.WriteEndElement();
            }

            Writer.WriteEndElement();
        }

        public override void WriteJoint(Joint joint)
        {
            if (!joint.ElementContainsData())
            {
                return;
            }

            Writer.WriteStartElement("joint");
            Writer.WriteAttributeString("name", joint.Name);
            Writer.WriteAttributeString("type", joint.Type);

            if (joint.Origin.ElementContainsData())
            {
                WriteOrigin(joint.Origin);
            }

            if (joint.Parent != null && joint.Parent.ElementContainsData())
            {
                Writer.WriteStartElement("parent");
                Writer.WriteAttributeString("link", joint.Parent.Name);
                Writer.WriteEndElement();
            }

            if (joint.Child != null && joint.Child.ElementContainsData())
            {
                Writer.WriteStartElement("child");
                Writer.WriteAttributeString("link", joint.Child.Name);
                Writer.WriteEndElement();
            }

            if (joint.Type != "fixed")
            {
                if (joint.Axis.ElementContainsData())
                {
                    WriteAxis(joint.Axis);
                }

                if (joint.Limit.ElementContainsData())
                {
                    WriteLimit(joint.Limit);
                }

                if (joint.Dynamics.ElementContainsData())
                {
                    WriteDynamics(joint.Dynamics);
                }
            }

            Writer.WriteEndElement();
        }

        public override void WriteInertial(Inertial inertial)
        {
            if (!inertial.ElementContainsData())
            {
                return;
            }

            Writer.WriteStartElement("inertial");

            if (inertial.Origin.ElementContainsData())
            {
                WriteOrigin(inertial.Origin);
            }

            WriteMass(inertial.Mass);
            WriteInertia(inertial.Inertia);

            Writer.WriteEndElement();
        }

        public override void WriteVisual(Visual visual)
        {
            Writer.WriteStartElement("visual");

            if (visual.Origin.ElementContainsData())
            {
                WriteOrigin(visual.Origin);
            }

            WriteGeometry(visual.Geometry);

            if (visual.Material.ElementContainsData())
            {
                WriteMaterial(visual.Material);
            }

            Writer.WriteEndElement();
        }

        public override void WriteCollision(Collision collision)
        {
            Writer.WriteStartElement("collision");

            if (collision.Origin.ElementContainsData())
            {
                WriteOrigin(collision.Origin);
            }

            WriteGeometry(collision.Geometry);

            Writer.WriteEndElement();
        }

        public override void WriteOrigin(Origin origin)
        {
            Writer.WriteStartElement("origin");
            WriteAttributeIfNotNull("xyz", origin.GetXYZ());
            WriteAttributeIfNotNull("rpy", origin.GetRPY());
            Writer.WriteEndElement();
        }

        public override void WriteGeometry(Geometry geometry)
        {
            Writer.WriteStartElement("geometry");
            WriteMesh(geometry.Mesh);
            Writer.WriteEndElement();
        }

        public override void WriteMesh(Mesh mesh)
        {
            if (string.IsNullOrEmpty(mesh.Filename))
            {
                return;
            }

            Writer.WriteStartElement("mesh");
            Writer.WriteAttributeString("filename", mesh.Filename);
            Writer.WriteEndElement();
        }

        public override void WriteMaterial(Material material)
        {
            Writer.WriteStartElement("material");
            WriteAttributeIfNotNull("name", material.Name);

            if (material.Color.ElementContainsData())
            {
                WriteColor(material.Color);
            }

            if (material.Texture.ElementContainsData())
            {
                WriteTexture(material.Texture);
            }

            Writer.WriteEndElement();
        }

        public override void WriteMass(Mass mass)
        {
            Writer.WriteStartElement("mass");
            Writer.WriteAttributeString("value", FormatDouble(mass.Value));
            Writer.WriteEndElement();
        }

        public override void WriteInertia(Inertia inertia)
        {
            Writer.WriteStartElement("inertia");
            Writer.WriteAttributeString("ixx", FormatDouble(inertia.Ixx));
            Writer.WriteAttributeString("ixy", FormatDouble(inertia.Ixy));
            Writer.WriteAttributeString("ixz", FormatDouble(inertia.Ixz));
            Writer.WriteAttributeString("iyy", FormatDouble(inertia.Iyy));
            Writer.WriteAttributeString("iyz", FormatDouble(inertia.Iyz));
            Writer.WriteAttributeString("izz", FormatDouble(inertia.Izz));
            Writer.WriteEndElement();
        }

        public override void WriteAxis(Axis axis)
        {
            Writer.WriteStartElement("axis");
            WriteAttributeIfNotNull("xyz", axis.GetXYZ());
            Writer.WriteEndElement();
        }

        public override void WriteLimit(Limit limit)
        {
            Writer.WriteStartElement("limit");
            Writer.WriteAttributeString("lower", FormatDouble(limit.Lower));
            Writer.WriteAttributeString("upper", FormatDouble(limit.Upper));
            Writer.WriteAttributeString("effort", FormatDouble(limit.Effort));
            Writer.WriteAttributeString("velocity", FormatDouble(limit.Velocity));
            Writer.WriteEndElement();
        }

        public override void WriteDynamics(Dynamics dynamics)
        {
            if (!dynamics.ElementContainsData())
            {
                return;
            }

            Writer.WriteStartElement("dynamics");
            WriteAttributeIfNotNull("damping", dynamics.Damping);
            WriteAttributeIfNotNull("friction", dynamics.Friction);
            Writer.WriteEndElement();
        }

        public override void WriteMimic(Mimic mimic)
        {
            if (!mimic.ElementContainsData())
            {
                return;
            }

            Writer.WriteStartElement("mimic");
            WriteAttributeIfNotNull("joint", mimic.JointName);
            WriteAttributeIfNotNull("multiplier", mimic.Multiplier);
            WriteAttributeIfNotNull("offset", mimic.Offset);
            Writer.WriteEndElement();
        }

        public override void WriteSafetyController(SafetyController safety)
        {
            if (!safety.ElementContainsData())
            {
                return;
            }

            Writer.WriteStartElement("safety_controller");
            WriteAttributeIfNotNull("soft_upper_limit", safety.SoftUpper);
            WriteAttributeIfNotNull("soft_lower_limit", safety.SoftLower);
            WriteAttributeIfNotNull("k_position", safety.KPosition);
            WriteAttributeIfNotNull("k_velocity", safety.KVelocity);
            Writer.WriteEndElement();
        }

        public override void WriteCalibration(Calibration calibration)
        {
            if (!calibration.ElementContainsData())
            {
                return;
            }

            Writer.WriteStartElement("calibration");
            WriteAttributeIfNotNull("rising", calibration.Rising);
            WriteAttributeIfNotNull("falling", calibration.Falling);
            Writer.WriteEndElement();
        }

        public override void WriteColor(Color color)
        {
            if (!color.ElementContainsData())
            {
                return;
            }

            Writer.WriteStartElement("color");
            WriteAttributeIfNotNull("rgba", color.GetColor());
            Writer.WriteEndElement();
        }

        public override void WriteTexture(Texture texture)
        {
            if (!texture.ElementContainsData())
            {
                return;
            }

            Writer.WriteStartElement("texture");
            WriteAttributeIfNotNull("filename", texture.Filename);
            Writer.WriteEndElement();
        }
    }
}
