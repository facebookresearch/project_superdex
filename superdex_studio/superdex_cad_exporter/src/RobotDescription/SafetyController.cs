/*
Copyright (c) 2015 Stephen Brawner

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
    //The safety_controller element of a joint.
    [DataContract(IsReference = true, Namespace = "http://schemas.datacontract.org/2004/07/SW2URDF")]
    public class SafetyController : RobotElement
    {
        [DataMember]
        private readonly ElementAttribute SoftLowerAttribute;

        [DataMember]
        private readonly ElementAttribute SoftUpperAttribute;

        [DataMember]
        private readonly ElementAttribute KPositionAttribute;

        [DataMember]
        private readonly ElementAttribute KVelocityAttribute;

        public double SoftLower
        {
            get => (double)SoftLowerAttribute.Value;
            set => SoftLowerAttribute.Value = value;
        }

        public double SoftUpper
        {
            get => (double)SoftUpperAttribute.Value;
            set => SoftUpperAttribute.Value = value;
        }

        public double KPosition
        {
            get => (double)KPositionAttribute.Value;
            set => KPositionAttribute.Value = value;
        }

        public double KVelocity
        {
            get => (double)KVelocityAttribute.Value;
            set => KVelocityAttribute.Value = value;
        }

        public SafetyController() : base("safety_controller", false)
        {
            SoftUpperAttribute = new ElementAttribute("soft_upper_limit", false, null);
            SoftLowerAttribute = new ElementAttribute("soft_lower_limit", false, null);
            KPositionAttribute = new ElementAttribute("k_position", false, null);
            KVelocityAttribute = new ElementAttribute("k_velocity", true, null);

            Attributes.Add(SoftUpperAttribute);
            Attributes.Add(SoftLowerAttribute);
            Attributes.Add(KPositionAttribute);
            Attributes.Add(KVelocityAttribute);
        }
    }
}
