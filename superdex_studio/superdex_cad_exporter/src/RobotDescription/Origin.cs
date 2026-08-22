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

using System.Collections.Generic;
using System.Collections.Specialized;
using System.Runtime.Serialization;

using CADRobotExporter.Model;

namespace CADRobotExporter.RobotDescription
{
    [DataContract(Name = "Origin", Namespace = "http://schemas.datacontract.org/2004/07/SW2URDF")]
    public class Origin : RobotElement
    {
        [DataMember]
        private readonly ElementAttribute XYZAttribute;

        [DataMember]
        private readonly ElementAttribute RPYAttribute;

        private double[] XYZ
        {
            get => (double[])XYZAttribute.Value;
            set => XYZAttribute.Value = value;
        }

        public double[] GetXYZ()
        {
            return (double[])XYZ.Clone();
        }

        public void SetXYZ(double[] xyz)
        {
            XYZ = xyz;
        }

        public double X
        {
            get => XYZ[0];
            set => XYZ[0] = value;
        }

        public double Y
        {
            get => XYZ[1];
            set => XYZ[1] = value;
        }

        public double Z
        {
            get => XYZ[2];
            set => XYZ[2] = value;
        }

        private double[] RPY
        {
            get => (double[])RPYAttribute.Value;
            set => RPYAttribute.Value = value;
        }

        public double[] GetRPY()
        {
            return (double[])RPY.Clone();
        }

        public void SetRPY(double[] rpy)
        {
            RPY = rpy;
        }

        public double Roll
        {
            get => RPY[0];
            set => RPY[0] = value;
        }

        public double Pitch
        {
            get => RPY[1];
            set => RPY[1] = value;
        }

        public double Yaw
        {
            get => RPY[2];
            set => RPY[2] = value;
        }

        [DataMember]
        public bool isCustomized;

        public Origin(bool isRequired) : base("origin", isRequired)
        {
            isCustomized = false;

            XYZAttribute = new ElementAttribute("xyz", true, new double[] { 0, 0, 0 });
            RPYAttribute = new ElementAttribute("rpy", true, new double[] { 0, 0, 0 });

            Attributes.Add(XYZAttribute);
            Attributes.Add(RPYAttribute);
        }

        /// <summary>
        /// Origin is a unique case in that its attributes are stored as double arrays.
        /// </summary>
        /// <param name="context"></param>
        /// <param name="dictionary"></param>
        public override void SetElementFromData(List<string> context, StringDictionary dictionary)
        {
            string typeName = GetType().Name;
            List<string> updatedContext = new List<string>(context) { typeName };

            double[] xyz = new double[3];
            double[] rpy = new double[3];

            string contextString = string.Join(".", updatedContext) + ".xyz";
            for (int i = 0; i < 3; i++)
            {
                string lookupString = contextString + "." + "xyz"[i];
                if (!dictionary.ContainsKey(lookupString))
                {
                    logger.Information("CSV file does not contain column for " + lookupString);
                    continue;
                }

                object value = ElementAttribute.GetValueFromString(dictionary[lookupString]);
                if (value != null && value.GetType() == typeof(double))
                {
                    xyz[i] = (double)value;
                }
            }

            contextString = string.Join(".", updatedContext) + ".rpy";
            for (int i = 0; i < 3; i++)
            {
                string lookupString = contextString + "." + "rpy"[i];
                if (!dictionary.ContainsKey(lookupString))
                {
                    logger.Information("CSV file does not contain column for " + lookupString);
                    continue;
                }

                object value = ElementAttribute.GetValueFromString(dictionary[lookupString]);
                if (value != null && value.GetType() == typeof(double))
                {
                    rpy[i] = (double)value;
                }
            }

            XYZAttribute.Value = xyz;
            RPYAttribute.Value = rpy;
        }
    }
}
