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

using System.Runtime.Serialization;

using CADRobotExporter.Model;

namespace CADRobotExporter.RobotDescription
{
    //The dynamics element of a joint.
    [DataContract(IsReference = true, Namespace = "http://schemas.datacontract.org/2004/07/SW2URDF")]
    public class Dynamics : RobotElement
    {
        [DataMember]
        private readonly ElementAttribute DampingAttribute;

        public double Damping
        {
            get => (double)DampingAttribute.Value;
            set => DampingAttribute.Value = value;
        }

        [DataMember]
        private readonly ElementAttribute FrictionAttribute;

        public double Friction
        {
            get => (double)FrictionAttribute.Value;
            set => FrictionAttribute.Value = value;
        }

        public Dynamics() : base("dynamics", false)
        {
            DampingAttribute = new ElementAttribute("damping", false, null);
            FrictionAttribute = new ElementAttribute("friction", false, null);

            Attributes.Add(DampingAttribute);
            Attributes.Add(FrictionAttribute);
        }

        public bool IsDampingSet() => DampingAttribute.Value != null;
        public bool IsFrictionSet() => FrictionAttribute.Value != null;
    }
}
