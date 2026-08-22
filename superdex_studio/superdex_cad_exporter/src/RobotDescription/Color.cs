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
    //The color element of the material element. Contains a single RGBA.
    [DataContract(Namespace = "http://schemas.datacontract.org/2004/07/SW2URDF")]
    public class Color : RobotElement
    {
        [DataMember]
        private readonly ElementAttribute RGBAAttribute;

        private double[] RGBA
        {
            get => (double[])RGBAAttribute.Value;
            set => RGBAAttribute.Value = value;
        }

        public double Red
        {
            get => RGBA[0];
            set => RGBA[0] = value;
        }

        public double Green
        {
            get => RGBA[1];
            set => RGBA[1] = value;
        }

        public double Blue
        {
            get => RGBA[2];
            set => RGBA[2] = value;
        }

        public double Alpha
        {
            get => RGBA[3];
            set => RGBA[3] = value;
        }

        public Color() : base("color", false)
        {
            RGBAAttribute = new ElementAttribute("rgba", true, new double[] { 1, 1, 1, 1 });

            Attributes.Add(RGBAAttribute);
        }

        public void SetColor(double[] rgba)
        {
            Red = rgba[0];
            Green = rgba[1];
            Blue = rgba[2];
            Alpha = rgba[3];
        }

        public double[] GetColor()
        {
            return new double[] { Red, Green, Blue, Alpha };
        }
    }
}
