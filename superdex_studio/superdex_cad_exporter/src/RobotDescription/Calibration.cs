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
    //The calibration element of a joint.
    [DataContract(IsReference = true, Namespace = "http://schemas.datacontract.org/2004/07/SW2URDF")]
    public class Calibration : RobotElement
    {
        [DataMember]
        private readonly ElementAttribute RisingAttribute;

        public double Rising
        {
            get => (double)RisingAttribute.Value;
            set => RisingAttribute.Value = value;
        }

        [DataMember]
        private readonly ElementAttribute FallingAttribute;

        public double Falling
        {
            get => (double)FallingAttribute.Value;
            set => FallingAttribute.Value = value;
        }

        public Calibration() : base("calibration", false)
        {
            RisingAttribute = new ElementAttribute("rising", false, null);
            FallingAttribute = new ElementAttribute("falling", false, null);
            Attributes.Add(RisingAttribute);
            Attributes.Add(FallingAttribute);
        }
    }
}
