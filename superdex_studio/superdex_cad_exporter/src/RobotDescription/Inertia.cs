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
    //Inertia element, which means moment of inertia. In the inertial element
    [DataContract(IsReference = true, Namespace = "http://schemas.datacontract.org/2004/07/SW2URDF")]
    public class Inertia : RobotElement
    {
        [DataMember]
        private readonly ElementAttribute IxxAttribute;

        [DataMember]
        private readonly ElementAttribute IxyAttribute;

        [DataMember]
        private readonly ElementAttribute IxzAttribute;

        [DataMember]
        private readonly ElementAttribute IyyAttribute;

        [DataMember]
        private readonly ElementAttribute IyzAttribute;

        [DataMember]
        private readonly ElementAttribute IzzAttribute;

        public double Ixx
        {
            get => (double)IxxAttribute.Value;
            set => IxxAttribute.Value = value;
        }

        public double Ixy
        {
            get => (double)IxyAttribute.Value;
            set => IxyAttribute.Value = value;
        }

        public double Ixz
        {
            get => (double)IxzAttribute.Value;
            set => IxzAttribute.Value = value;
        }

        public double Iyy
        {
            get => (double)IyyAttribute.Value;
            set => IyyAttribute.Value = value;
        }

        public double Iyz
        {
            get => (double)IyzAttribute.Value;
            set => IyzAttribute.Value = value;
        }

        public double Izz
        {
            get => (double)IzzAttribute.Value;
            set => IzzAttribute.Value = value;
        }

        public Inertia() : base("inertia", false)
        {
            IxxAttribute = new ElementAttribute("ixx", true, 0.0);
            IxyAttribute = new ElementAttribute("ixy", true, 0.0);
            IxzAttribute = new ElementAttribute("ixz", true, 0.0);
            IyyAttribute = new ElementAttribute("iyy", true, 0.0);
            IyzAttribute = new ElementAttribute("iyz", true, 0.0);
            IzzAttribute = new ElementAttribute("izz", true, 0.0);

            Attributes.Add(IxxAttribute);
            Attributes.Add(IxyAttribute);
            Attributes.Add(IxzAttribute);
            Attributes.Add(IyyAttribute);
            Attributes.Add(IyzAttribute);
            Attributes.Add(IzzAttribute);
        }

        public void SetMomentMatrix(double[] array)
        {
            Ixx = array[0];
            Ixy = array[1];
            Ixz = array[2];
            Iyy = array[4];
            Iyz = array[5];
            Izz = array[8];
        }

        public bool IsZero()
        {
            return Ixx == 0 && Ixy == 0 && Ixz == 0 && Iyy == 0 && Iyz == 0 && Izz == 0;
        }

        internal double[] GetMoment()
        {
            return new double[] { Ixx, Ixy, Ixz, Ixy, Iyy, Iyz, Ixz, Iyz, Izz };
        }
    }
}
