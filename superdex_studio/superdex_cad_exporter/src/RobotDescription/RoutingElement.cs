/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System.Runtime.Serialization;

using CADRobotExporter.Model;

namespace CADRobotExporter.RobotDescription
{
    [DataContract(IsReference = true, Namespace = "http://schemas.datacontract.org/2004/07/SW2URDF")]
    public class RoutingElement : RobotElement
    {
        public static readonly string TypeWaypoint = "waypoint";
        public static readonly string TypeLinearJoint = "linear_joint";

        [DataMember]
        private readonly ElementAttribute LinkAttribute;

        public string Link
        {
            get => (string)LinkAttribute.Value;
            set => LinkAttribute.Value = value;
        }

        [DataMember]
        private readonly ElementAttribute TypeAttribute;

        public string Type
        {
            get => (string)TypeAttribute.Value;
            set => TypeAttribute.Value = value;
        }

        [DataMember]
        private readonly ElementAttribute PositionAttribute;

        private double[] Position
        {
            get => (double[])PositionAttribute.Value;
            set => PositionAttribute.Value = value;
        }

        public double X
        {
            get => Position != null ? Position[0] : 0;
            set { EnsurePosition(); Position[0] = value; }
        }

        public double Y
        {
            get => Position != null ? Position[1] : 0;
            set { EnsurePosition(); Position[1] = value; }
        }

        public double Z
        {
            get => Position != null ? Position[2] : 0;
            set { EnsurePosition(); Position[2] = value; }
        }

        public double[] GetPosition()
        {
            return Position != null ? (double[])Position.Clone() : new double[3];
        }

        public void SetPosition(double[] xyz)
        {
            Position = xyz != null ? (double[])xyz.Clone() : new double[3];
        }

        [DataMember]
        private readonly ElementAttribute CoefficientAttribute;

        public double Coefficient
        {
            get => CoefficientAttribute.Value != null ? (double)CoefficientAttribute.Value : 0;
            set => CoefficientAttribute.Value = value;
        }

        [DataMember]
        public string PointKey { get; set; }

        public RoutingElement() : base("routing_element", true)
        {
            LinkAttribute = new ElementAttribute("link", true, "");
            TypeAttribute = new ElementAttribute("type", true, "");
            PositionAttribute = new ElementAttribute("xyz", false, null);
            CoefficientAttribute = new ElementAttribute("coef", false, null);

            Attributes.Add(LinkAttribute);
            Attributes.Add(TypeAttribute);
            Attributes.Add(PositionAttribute);
            Attributes.Add(CoefficientAttribute);
        }

        private void EnsurePosition()
        {
            if (Position == null)
            {
                Position = new double[3];
            }
        }
    }
}
