/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System.Collections.Generic;
using System.Runtime.Serialization;

using CADRobotExporter.Model;

namespace CADRobotExporter.RobotDescription
{
    [DataContract(IsReference = true, Namespace = "http://schemas.datacontract.org/2004/07/SW2URDF")]
    public class Tendon : RobotElement
    {
        [DataMember]
        private readonly ElementAttribute NameAttribute;

        public string Name
        {
            get => (string)NameAttribute.Value;
            set => NameAttribute.Value = value;
        }

        [DataMember]
        public List<RoutingElement> RoutingElements { get; private set; }

        public Tendon() : base("tendon", true)
        {
            NameAttribute = new ElementAttribute("name", true, "");
            RoutingElements = new List<RoutingElement>();

            Attributes.Add(NameAttribute);
        }

        public void AddRoutingElement(RoutingElement element)
        {
            RoutingElements.Add(element);
            ChildElements.Add(element);
        }

        public void RemoveRoutingElement(RoutingElement element)
        {
            RoutingElements.Remove(element);
            ChildElements.Remove(element);
        }
    }
}
