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
    [DataContract(Name = "Mimic", Namespace = "http://schemas.datacontract.org/2004/07/SW2URDF")]
    public class Mimic : RobotElement
    {
        [DataMember]
        private readonly ElementAttribute JointNameAttribute;

        public string JointName
        {
            get => (string)JointNameAttribute.Value;
            set
            {
                if (value.GetType() == typeof(string))
                {
                    JointNameAttribute.Value = value;
                }
            }
        }

        [DataMember]
        private readonly ElementAttribute MultiplierAttribute;

        public double Multiplier
        {
            get => (double)MultiplierAttribute.Value;
            set
            {
                if (value.GetType() == typeof(double))
                {
                    MultiplierAttribute.Value = value;
                }
            }
        }

        [DataMember]
        private readonly ElementAttribute OffsetAttribute;

        public double Offset
        {
            get => (double)OffsetAttribute.Value;
            set
            {
                if (value.GetType() == typeof(double))
                {
                    OffsetAttribute.Value = value;
                }
            }
        }

        public Mimic() : base("mimic", false)
        {
            JointNameAttribute = new ElementAttribute("joint", true, null);
            MultiplierAttribute = new ElementAttribute("multiplier", false, null);
            OffsetAttribute = new ElementAttribute("offset", false, null);

            Attributes.Add(JointNameAttribute);
            Attributes.Add(MultiplierAttribute);
            Attributes.Add(OffsetAttribute);
        }
    }
}
