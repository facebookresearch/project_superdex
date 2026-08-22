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
using System.Collections.Specialized;
using System.Runtime.Serialization;

#if SOLIDWORKS
using SolidWorks.Interop.sldworks;
#endif

#if NX
using NXOpen;
#endif

using CADRobotExporter.Model;

namespace CADRobotExporter.RobotDescription
{
    //The joint class. There is one for every link but the base link
    [DataContract(IsReference = true, Namespace = "http://schemas.datacontract.org/2004/07/SW2URDF")]
    public class Joint : RobotElement, IExtensibleDataObject
    {
        public ExtensionDataObject ExtensionData { get; set; }
        public static readonly List<string> AvailableTypes = new List<string>
        {
            "revolute", "continuous", "prismatic", "fixed", "floating", "planar"
        };

        /// <summary>
        /// Magic keyword indicating the joint axis should be derived from the X axis of the coordinate system.
        /// </summary>
        public const string AxisFromCsysX = "$CSYS_X_AXIS";

        /// <summary>
        /// Magic keyword indicating the joint axis should be derived from the Y axis of the coordinate system.
        /// </summary>
        public const string AxisFromCsysY = "$CSYS_Y_AXIS";

        /// <summary>
        /// Magic keyword indicating the joint axis should be derived from the Z axis of the coordinate system.
        /// </summary>
        public const string AxisFromCsysZ = "$CSYS_Z_AXIS";

        /// <summary>
        /// Checks if the given axis name is a magic keyword indicating the axis should be derived from the CSYS.
        /// </summary>
        public static bool IsAxisFromCsys(string axisName)
        {
            return axisName == AxisFromCsysX || axisName == AxisFromCsysY || axisName == AxisFromCsysZ;
        }

        [DataMember]
        private readonly ElementAttribute NameAttribute;

        public string Name
        {
            get => (string)NameAttribute.Value;
            set => NameAttribute.Value = value;
        }

        [DataMember]
        private readonly ElementAttribute TypeAttribute;

        public string Type
        {
            get => (string)TypeAttribute.Value;
            set => TypeAttribute.Value = value;
        }

        [DataMember]
        public readonly Origin Origin;

        [DataMember]
        public readonly ParentLink Parent;

        [DataMember]
        public readonly ChildLink Child;

        [DataMember]
        public readonly Axis Axis;

        [DataMember]
        public readonly Limit Limit;

        [DataMember]
        public readonly Dynamics Dynamics;

        public double[] originInGlobalSpace;
        public double[] axisInGlobalSpace;

        // For NX, these are Handles that persist across sessions already
        // For Solidworks, these are fully qualified names
        // and require PID below to retreive as Features, then the names
        // are used thereafter
        [DataMember]
        public string CoordinateSystemName;
        [DataMember]
        public string AxisName;

        [DataMember]
        public byte[] CoordinateSystemPID;
        [DataMember]
        public byte[] AxisPID;

#if SOLIDWORKS
        public Feature SWRefAxisFeature;
        public Feature SWCoordinateSystemFeature;
#endif

        public Joint() : base("joint", false)
        {
            Origin = new Origin(false);
            Parent = new ParentLink();
            Child = new ChildLink();
            Axis = new Axis();

            Limit = new Limit();
            Dynamics = new Dynamics();

            NameAttribute = new ElementAttribute("name", true, "");
            TypeAttribute = new ElementAttribute("type", true, "");

            Attributes.Add(NameAttribute);
            Attributes.Add(TypeAttribute);

            ChildElements.Add(Origin);
            ChildElements.Add(Parent);
            ChildElements.Add(Child);
            ChildElements.Add(Axis);

            ChildElements.Add(Limit);
            ChildElements.Add(Dynamics);
        }

        public override bool ElementContainsData()
        {
            return !string.IsNullOrWhiteSpace(Name) && !string.IsNullOrWhiteSpace(Type);
        }

        public override bool AreRequiredFieldsSatisfied()
        {
            Limit.SetRequired(Type == "prismatic" || Type == "revolute");
            return base.AreRequiredFieldsSatisfied();
        }

        public override void AppendToCSVDictionary(List<string> context, OrderedDictionary dictionary)
        {
            string contextString = string.Join(".", context);

            string coordSysContext = contextString + ".CoordSysName";
            dictionary.Add(coordSysContext, CoordinateSystemName);

            string axisContext = contextString + ".AxisName";
            dictionary.Add(axisContext, AxisName);

            base.AppendToCSVDictionary(context, dictionary);
        }

        public override void SetElement(RobotElement externalElement)
        {
            base.SetElement(externalElement);

            // The base method already performs the type check, so we don't have to for this cast
            Joint joint = (Joint)externalElement;

            // These strings aren't kept as ElementAttribute objects and so they are tracked separately
            CoordinateSystemName = joint.CoordinateSystemName;
            AxisName = joint.AxisName;

            if (joint.CoordinateSystemPID != null)
            {
                CoordinateSystemPID = joint.CoordinateSystemPID;
            }

            if (joint.AxisPID != null)
            {
                AxisPID = joint.AxisPID;
            }
        }

        public override void SetElementFromData(List<string> context, StringDictionary dictionary)
        {
            string contextString = string.Join(".", context);

            string coordSysContext = contextString + ".CoordSysName";
            CoordinateSystemName = dictionary[coordSysContext];

            string axisContext = contextString + ".AxisName";
            AxisName = dictionary[axisContext];

            base.SetElementFromData(context, dictionary);
        }

        public void SetJointLimitDefaultsIfNotSet()
        {
            bool hasNonZeroUpper = Limit.IsUpperSet();
            bool hasNonZeroLower = Limit.IsLowerSet();
            bool hasEffort = Limit.IsEffortSet();
            bool hasVelocity = Limit.IsVelocitySet();

            if (hasNonZeroUpper && hasNonZeroLower && hasEffort && hasVelocity)
            {
                return;
            }

            switch (Type)
            {
                case "revolute":
                    Limit.Upper = Math.PI;
                    Limit.Lower = -Math.PI;
                    Limit.Effort = 1.0;
                    Limit.Velocity = 1.0;
                    break;
                case "prismatic":
                    Limit.Upper = 1.0;
                    Limit.Lower = -1.0;
                    Limit.Effort = 1.0;
                    Limit.Velocity = 1.0;
                    break;
                case "continuous":
                    Limit.Effort = 1.0;
                    Limit.Velocity = 1.0;
                    break;
                default:
                    break;
            }
        }
    }
}
