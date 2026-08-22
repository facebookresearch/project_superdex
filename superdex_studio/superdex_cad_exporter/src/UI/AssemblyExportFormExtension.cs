/*
Copyright (c) 2015 Stephen Brawner
Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

using CADRobotExporter.RobotDescription;

namespace CADRobotExporter.UI
{
    //This source file contains all the non-handler methods for the assembly export form,
    // the ones that are helpers.
    public partial class AssemblyExportForm : Form
    {
        //From the link, this method fills the property boxes on the Link Properties page
        public void FillLinkPropertyBoxes(Link Link)
        {
            AutoUpdatingForm = true;
            FillBlank(linkBoxes);
            if (Link != null && !Link.isFixedFrame)
            {
                const string format = "R";

                // Visual Origin
                if (Link.Visual?.Origin != null)
                {
                    try
                    {
                        textBoxVisualOriginX.Text = Link.Visual.Origin.X.ToString(format);
                        textBoxVisualOriginY.Text = Link.Visual.Origin.Y.ToString(format);
                        textBoxVisualOriginZ.Text = Link.Visual.Origin.Z.ToString(format);
                        textBoxVisualOriginRoll.Text = Link.Visual.Origin.Roll.ToString(format);
                        textBoxVisualOriginPitch.Text = Link.Visual.Origin.Pitch.ToString(format);
                        textBoxVisualOriginYaw.Text = Link.Visual.Origin.Yaw.ToString(format);
                    }
                    catch (NullReferenceException)
                    {
                        // Attribute values are null, leave text boxes empty
                    }
                }

                // Inertial Origin
                if (Link.Inertial?.Origin != null)
                {
                    try
                    {
                        textBoxInertialOriginX.Text = Link.Inertial.Origin.X.ToString(format);
                        textBoxInertialOriginY.Text = Link.Inertial.Origin.Y.ToString(format);
                        textBoxInertialOriginZ.Text = Link.Inertial.Origin.Z.ToString(format);
                        textBoxInertialOriginRoll.Text = Link.Inertial.Origin.Roll.ToString(format);
                        textBoxInertialOriginPitch.Text = Link.Inertial.Origin.Pitch.ToString(format);
                        textBoxInertialOriginYaw.Text = Link.Inertial.Origin.Yaw.ToString(format);
                    }
                    catch (NullReferenceException)
                    {
                        // Attribute values are null, leave text boxes empty
                    }
                }

                // Mass
                if (Link.Inertial?.Mass != null)
                {
                    try
                    {
                        textBoxMass.Text = Link.Inertial.Mass.Value.ToString(format);
                    }
                    catch (NullReferenceException)
                    {
                        // Attribute value is null, leave text box empty
                    }
                }

                // Inertia
                if (Link.Inertial?.Inertia != null)
                {
                    try
                    {
                        textBoxIxx.Text = Link.Inertial.Inertia.Ixx.ToString(format);
                        textBoxIxy.Text = Link.Inertial.Inertia.Ixy.ToString(format);
                        textBoxIxz.Text = Link.Inertial.Inertia.Ixz.ToString(format);
                        textBoxIyy.Text = Link.Inertial.Inertia.Iyy.ToString(format);
                        textBoxIyz.Text = Link.Inertial.Inertia.Iyz.ToString(format);
                        textBoxIzz.Text = Link.Inertial.Inertia.Izz.ToString(format);
                    }
                    catch (NullReferenceException)
                    {
                        // Attribute values are null, leave text boxes empty
                    }
                }

                // Color
                if (Link.Visual?.Material?.Color != null)
                {
                    try
                    {
                        domainUpDownRed.Text = Link.Visual.Material.Color.Red.ToString(format);
                        domainUpDownGreen.Text = Link.Visual.Material.Color.Green.ToString(format);
                        domainUpDownBlue.Text = Link.Visual.Material.Color.Blue.ToString(format);
                        domainUpDownAlpha.Text = Link.Visual.Material.Color.Alpha.ToString(format);
                    }
                    catch (NullReferenceException)
                    {
                        // Attribute values are null, leave text boxes empty
                    }
                }

                if (checkBoxPerLinkMeshing.Checked)
                {
                    LoadMeshingOptionsFromLink(Link);
                }
            }
            AutoUpdatingForm = false;
        }

        //Fills the property boxes on the joint properties page
        public void FillJointPropertyBoxes(Joint joint)
        {
            AutoUpdatingForm = true;
            FillBlank(jointBoxes);
            if (joint != null) //For the base_link or if none is selected
            {
                const string format = "R";
                bool useDegrees = checkBoxUseDegrees.Checked;

                // Joint name and type
                textBoxJointName.Text = joint.Name ?? "";
                comboBoxJointType.Text = joint.Type ?? "";

                // Parent and child links
                textBoxParentLink.Text = joint.Parent?.Name ?? "";
                textBoxChildLink.Text = joint.Child?.Name ?? "";

                // Joint Origin
                if (joint.Origin != null)
                {
                    textBoxJointX.Text = joint.Origin.X.ToString(format);
                    textBoxJointY.Text = joint.Origin.Y.ToString(format);
                    textBoxJointZ.Text = joint.Origin.Z.ToString(format);

                    // Convert rotation values if using degrees
                    double roll = joint.Origin.Roll;
                    double pitch = joint.Origin.Pitch;
                    double yaw = joint.Origin.Yaw;
                    if (useDegrees)
                    {
                        roll = RadiansToDegrees(roll);
                        pitch = RadiansToDegrees(pitch);
                        yaw = RadiansToDegrees(yaw);
                    }
                    textBoxJointRoll.Text = roll.ToString(format);
                    textBoxJointPitch.Text = pitch.ToString(format);
                    textBoxJointYaw.Text = yaw.ToString(format);
                }

                if (joint.Type != "fixed" && joint.Axis != null)
                {
                    // Axis
                    textBoxAxisX.Text = joint.Axis.X.ToString(format);
                    textBoxAxisY.Text = joint.Axis.Y.ToString(format);
                    textBoxAxisZ.Text = joint.Axis.Z.ToString(format);
                }

                if (joint.Limit != null && joint.Type != "fixed" && joint.Limit.ElementContainsData())
                {
                    if (joint.Type == "revolute")
                    {
                        numericUpDownLimitLower.Increment = useDegrees ? 5M : 0.0872665M;
                        numericUpDownLimitUpper.Increment = useDegrees ? 5M : 0.0872665M;
                    }
                    else
                    {
                        numericUpDownLimitLower.Increment = 0.1M;
                        numericUpDownLimitUpper.Increment = 0.1M;
                    }

                    // Limit - safely handle potentially null attribute values
                    try
                    {
                        double lower = joint.Limit.Lower;
                        double upper = joint.Limit.Upper;
                        if (useDegrees && joint.Type == "revolute")
                        {
                            lower = RadiansToDegrees(lower);
                            upper = RadiansToDegrees(upper);
                        }
                        numericUpDownLimitLower.Value = Convert.ToDecimal(lower);
                        numericUpDownLimitUpper.Value = Convert.ToDecimal(upper);
                        textBoxLimitEffort.Text = joint.Limit.Effort.ToString(format);
                        double velocity = joint.Limit.Velocity;
                        if (useDegrees && joint.Type == "revolute")
                        {
                            velocity = RadiansToDegrees(velocity);
                        }
                        textBoxLimitVelocity.Text = velocity.ToString(format);
                    }
                    catch (NullReferenceException)
                    {
                        // Attribute values are null, leave text boxes empty
                    }
                }

                if (joint.Dynamics != null && joint.Dynamics.ElementContainsData())
                {
                    // Dynamics - safely handle potentially null attribute values
                    try
                    {
                        textBoxDamping.Text = joint.Dynamics.Damping.ToString(format);
                        textBoxFriction.Text = joint.Dynamics.Friction.ToString(format);
                    }
                    catch (NullReferenceException)
                    {
                        // Attribute values are null, leave text boxes empty
                    }
                }

                string angleUnit = useDegrees ? "deg" : "rad";
                if (joint.Type == "revolute" || joint.Type == "continuous")
                {
                    labelLowerLimit.Text = $"Lower ({angleUnit})";
                    labelLimitUpper.Text = $"Upper ({angleUnit})";
                    labelEffort.Text = "Effort (N-m)";
                    labelVelocity.Text = $"Velocity ({angleUnit}/s)";
                    labelFriction.Text = "Friction (N-m)";
                    labelDamping.Text = $"Damping (N-m-s/{angleUnit})";
                }
                else if (joint.Type == "prismatic")
                {
                    labelLowerLimit.Text = "Lower (m)";
                    labelLimitUpper.Text = "Upper (m)";
                    labelEffort.Text = "Effort (N)";
                    labelVelocity.Text = "Velocity (m/s)";
                    labelFriction.Text = "Friction (N)";
                    labelDamping.Text = "Damping (N-s/m)";
                }
                else
                {
                    labelLowerLimit.Text = "Lower";
                    labelLimitUpper.Text = "Upper";
                    labelEffort.Text = "Effort";
                    labelVelocity.Text = "Velocity";
                    labelFriction.Text = "Friction";
                    labelDamping.Text = "Damping";
                }

                if (joint.Type == "fixed")
                {
                    textBoxAxisX.Enabled = false;
                    textBoxAxisY.Enabled = false;
                    textBoxAxisZ.Enabled = false;
                    numericUpDownLimitLower.Enabled = false;
                    numericUpDownLimitUpper.Enabled = false;
                    textBoxLimitEffort.Enabled = false;
                    textBoxLimitVelocity.Enabled = false;
                    textBoxDamping.Enabled = false;
                    textBoxFriction.Enabled = false;
                }
                else
                {
                    textBoxAxisX.Enabled = true;
                    textBoxAxisY.Enabled = true;
                    textBoxAxisZ.Enabled = true;
                    numericUpDownLimitLower.Enabled = true;
                    numericUpDownLimitUpper.Enabled = true;
                    textBoxLimitEffort.Enabled = true;
                    textBoxLimitVelocity.Enabled = true;
                    textBoxDamping.Enabled = true;
                    textBoxFriction.Enabled = true;
                }

                textBoxCoordSys.Text = joint.CoordinateSystemName ?? "";
                textBoxRefAxis.Text = joint.AxisName ?? "";

                // Updating Mimic Element Fields
                List<string> jointNames = Exporter.GetJointNames();
            }
            else
            {
                // joint is null - set default labels
                labelLowerLimit.Text = "Lower";
                labelLimitUpper.Text = "Upper";
                labelEffort.Text = "Effort";
                labelVelocity.Text = "Velocity";
                labelFriction.Text = "Friction";
                labelDamping.Text = "Damping";
            }

            AutoUpdatingForm = false;
        }

        public static void FillBlank(Control[] boxes)
        {
            foreach (Control box in boxes)
            {
                box.Text = "";
            }
        }

        //Converts the text boxes back into values for the link
        public void SaveLinkDataFromPropertyBoxes(Link Link)
        {
            if (!Link.isFixedFrame)
            {
                // Inertial Origin
                if (double.TryParse(textBoxInertialOriginX.Text, out var inertialX)) Link.Inertial.Origin.X = inertialX;
                if (double.TryParse(textBoxInertialOriginY.Text, out var inertialY)) Link.Inertial.Origin.Y = inertialY;
                if (double.TryParse(textBoxInertialOriginZ.Text, out var inertialZ)) Link.Inertial.Origin.Z = inertialZ;
                if (double.TryParse(textBoxInertialOriginRoll.Text, out var inertialRoll)) Link.Inertial.Origin.Roll = inertialRoll;
                if (double.TryParse(textBoxInertialOriginPitch.Text, out var inertialPitch)) Link.Inertial.Origin.Pitch = inertialPitch;
                if (double.TryParse(textBoxInertialOriginYaw.Text, out var inertialYaw)) Link.Inertial.Origin.Yaw = inertialYaw;

                // Visual Origin
                if (double.TryParse(textBoxVisualOriginX.Text, out var visualX)) Link.Visual.Origin.X = visualX;
                if (double.TryParse(textBoxVisualOriginY.Text, out var visualY)) Link.Visual.Origin.Y = visualY;
                if (double.TryParse(textBoxVisualOriginZ.Text, out var visualZ)) Link.Visual.Origin.Z = visualZ;
                if (double.TryParse(textBoxVisualOriginRoll.Text, out var visualRoll)) Link.Visual.Origin.Roll = visualRoll;
                if (double.TryParse(textBoxVisualOriginPitch.Text, out var visualPitch)) Link.Visual.Origin.Pitch = visualPitch;
                if (double.TryParse(textBoxVisualOriginYaw.Text, out var visualYaw)) Link.Visual.Origin.Yaw = visualYaw;

                // Mass
                if (double.TryParse(textBoxMass.Text, out var mass)) Link.Inertial.Mass.Value = mass;

                // Inertia
                if (double.TryParse(textBoxIxx.Text, out var ixx)) Link.Inertial.Inertia.Ixx = ixx;
                if (double.TryParse(textBoxIxy.Text, out var ixy)) Link.Inertial.Inertia.Ixy = ixy;
                if (double.TryParse(textBoxIxz.Text, out var ixz)) Link.Inertial.Inertia.Ixz = ixz;
                if (double.TryParse(textBoxIyy.Text, out var iyy)) Link.Inertial.Inertia.Iyy = iyy;
                if (double.TryParse(textBoxIyz.Text, out var iyz)) Link.Inertial.Inertia.Iyz = iyz;
                if (double.TryParse(textBoxIzz.Text, out var izz)) Link.Inertial.Inertia.Izz = izz;

                // Color
                if (double.TryParse(domainUpDownRed.Text, out var red)) Link.Visual.Material.Color.Red = red;
                if (double.TryParse(domainUpDownGreen.Text, out var green)) Link.Visual.Material.Color.Green = green;
                if (double.TryParse(domainUpDownBlue.Text, out var blue)) Link.Visual.Material.Color.Blue = blue;
                if (double.TryParse(domainUpDownAlpha.Text, out var alpha)) Link.Visual.Material.Color.Alpha = alpha;

                if (checkBoxPerLinkMeshing.Checked)
                {
                    SaveMeshingOptionsToLink(Link);
                }
            }
        }

        //Saves data from text boxes back into a joint
        public void SaveJointDataFromPropertyBoxes(Joint Joint)
        {
            bool useDegrees = checkBoxUseDegrees.Checked;

            // Joint name and type
            Joint.Name = textBoxJointName.Text;
            Joint.Type = comboBoxJointType.Text;

            // Parent and child links
            Joint.Parent.Name = textBoxParentLink.Text;
            Joint.Child.Name = textBoxChildLink.Text;

            // Joint Origin
            if (double.TryParse(textBoxJointX.Text, out var jointX)) Joint.Origin.X = jointX;
            if (double.TryParse(textBoxJointY.Text, out var jointY)) Joint.Origin.Y = jointY;
            if (double.TryParse(textBoxJointZ.Text, out var jointZ)) Joint.Origin.Z = jointZ;

            // Convert rotation values if using degrees
            if (double.TryParse(textBoxJointRoll.Text, out var jointRoll))
                Joint.Origin.Roll = useDegrees ? DegreesToRadians(jointRoll) : jointRoll;
            if (double.TryParse(textBoxJointPitch.Text, out var jointPitch))
                Joint.Origin.Pitch = useDegrees ? DegreesToRadians(jointPitch) : jointPitch;
            if (double.TryParse(textBoxJointYaw.Text, out var jointYaw))
                Joint.Origin.Yaw = useDegrees ? DegreesToRadians(jointYaw) : jointYaw;

            // Axis
            if (double.TryParse(textBoxAxisX.Text, out var axisX)) Joint.Axis.X = axisX;
            if (double.TryParse(textBoxAxisY.Text, out var axisY)) Joint.Axis.Y = axisY;
            if (double.TryParse(textBoxAxisZ.Text, out var axisZ)) Joint.Axis.Z = axisZ;

            // Limit
            Joint.Limit.SetRequired(Joint.Type == "revolute" || Joint.Type == "prismatic");
            if (string.IsNullOrWhiteSpace(textBoxLimitEffort.Text) &&
                string.IsNullOrWhiteSpace(textBoxLimitVelocity.Text) &&
                !Joint.Limit.IsRequired())
            {
                // If all text boxes are empty and this element isn't required, then leave blank
            }
            else
            {
                double lower = (double)numericUpDownLimitLower.Value;
                double upper = (double)numericUpDownLimitUpper.Value;
                if (useDegrees && Joint.Type == "revolute")
                {
                    lower = DegreesToRadians(lower);
                    upper = DegreesToRadians(upper);
                }
                Joint.Limit.Lower = lower;
                Joint.Limit.Upper = upper;
                if (double.TryParse(textBoxLimitEffort.Text, out var effort)) Joint.Limit.Effort = effort;
                if (double.TryParse(textBoxLimitVelocity.Text, out var velocity))
                {
                    if (useDegrees && Joint.Type == "revolute")
                        velocity = DegreesToRadians(velocity);
                    Joint.Limit.Velocity = velocity;
                }
            }

            // Dynamics
            if (String.IsNullOrWhiteSpace(textBoxFriction.Text) &&
                String.IsNullOrWhiteSpace(textBoxDamping.Text))
            {
                Joint.Dynamics.Unset();
            }
            else
            {
                if (double.TryParse(textBoxDamping.Text, out var damping)) Joint.Dynamics.Damping = damping;
                if (double.TryParse(textBoxFriction.Text, out var friction)) Joint.Dynamics.Friction = friction;
            }
        }

        //Fills specifically the joint TreeView
        public void FillJointTree()
        {
            treeViewJointTree.Nodes.Clear();

            while (BaseNode.Nodes.Count > 0)
            {
                LinkNode node = (LinkNode)BaseNode.FirstNode;
                BaseNode.Nodes.Remove(node);
                treeViewJointTree.Nodes.Add(node);
                UpdateNodeText(node, true);
            }
            treeViewJointTree.ExpandAll();

            if (treeViewJointTree.Nodes.Count > 0)
            {
                treeViewJointTree.SelectedNode = treeViewJointTree.Nodes[0];
            }
        }

        public void FillLinkTree()
        {
            treeViewLinkProperties.Nodes.Clear();
            treeViewLinkProperties.Nodes.Add(BaseNode);
            UpdateNodeText(BaseNode, false);
            treeViewLinkProperties.ExpandAll();

            if (treeViewLinkProperties.Nodes.Count > 0)
            {
                treeViewLinkProperties.SelectedNode = treeViewLinkProperties.Nodes[0];
            }
        }

        public void UpdateNodeText(LinkNode node, bool useJointName)
        {
            if (useJointName)
            {
                node.Text = node.Link.Joint.Name;
            }
            else
            {
                node.Text = node.Link.Name;
            }
            node.NodeFont = new Font(treeViewJointTree.Font, FontStyle.Regular);
            foreach (LinkNode child in node.Nodes)
            {
                UpdateNodeText(child, useJointName);
            }
        }

        //Converts a Link to a LinkNode
        public static LinkNode CreateLinkNodeFromLink(Link Link)
        {
            LinkNode node = new LinkNode(Link);
            node.Link.Children.Clear();
            return node;
        }

        //Converts a TreeView back into a robot
        public RobotDescription.Robot CreateRobotFromTreeView(TreeView tree)
        {
            RobotDescription.Robot Robot = Exporter.Robot;
            Link baseLink = CreateLinkFromLinkNode((LinkNode)tree.Nodes[0]);
            Robot.SetBaseLink(baseLink);
            Robot.Name = textBoxRobotName.Text;
            return Robot;
        }

        //Converts a LinkNode into a Link
        public Link CreateLinkFromLinkNode(LinkNode node)
        {
            Link Link = node.Link;
            Link.Children.Clear();
            foreach (LinkNode child in node.Nodes)
            {
                Link childLink = CreateLinkFromLinkNode(child);
                Link.Children.Add(childLink); // Recreates the children of each embedded link
            }
            return Link;
        }

        private void CheckLinksForWarnings(Link node, StringBuilder builder)
        {
            string msg = "";

            if (!string.IsNullOrWhiteSpace(msg))
            {
                builder.Append(node.Name + " - " + msg + "\r\n");
            }
            foreach (Link child in node.Children)
            {
                CheckLinksForWarnings(child, builder);
            }
        }

        private string CheckLinksForWarnings(Link baseNode)
        {
            StringBuilder builder = new StringBuilder();
            CheckLinksForWarnings(baseNode, builder);
            return builder.ToString();
        }

        // this SaveConfigTree is called when the robot may have a name already
        public void SaveConfigurationFromTree(LinkNode BaseNode, bool warnUser)
        {
            ExporterConfiguration config = SaveFormToExporterConfiguration();
            if (checkBoxPerLinkMeshing.Checked)
            {
                // small ugly hack to grab cached non-per link meshing meshing options
                config.visualMeshingOptions = cachedExporterConfig.visualMeshingOptions;
                config.collisionMeshingOptions = cachedExporterConfig.collisionMeshingOptions;
            }

            CadBridge.SaveConfigurationFromTree(config, BaseNode, warnUser);
        }

        public void ChangeAllNodeFont(LinkNode node, Font font)
        {
            node.NodeFont = font;
            foreach (LinkNode child in node.Nodes)
            {
                ChangeAllNodeFont(child, font);
            }
        }
    }
}
