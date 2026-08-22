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
using System.Runtime.Serialization;

using CADRobotExporter.Export;
using CADRobotExporter.Model;

namespace CADRobotExporter.RobotDescription
{
    /// <summary>
    /// The base URDF element representing a robot model.
    /// </summary>
    [DataContract(IsReference = true, Namespace = "http://schemas.datacontract.org/2004/07/SW2URDF")]
#pragma warning disable CS0618 // Type or member is obsolete
    public class Robot : RobotElement
#pragma warning restore CS0618 // Type or member is obsolete
    {
        [DataMember]
        public Link BaseLink { get; private set; }

        [DataMember]
        public List<Tendon> Tendons { get; private set; }

        [DataMember]
        private readonly ElementAttribute NameAttribute;

        public string Name
        {
            get => (string)NameAttribute.Value;
            set => NameAttribute.Value = value;
        }

#pragma warning disable CS0618 // Type or member is obsolete
        public Robot() : base("robot", true)
        {
            BaseLink = new Link(null);
            Tendons = new List<Tendon>();
            NameAttribute = new ElementAttribute("name", true, "");

            ChildElements.Add(BaseLink);
            Attributes.Add(NameAttribute);
        }
#pragma warning restore CS0618 // Type or member is obsolete

        /// <summary>
        /// Exports the robot model using the specified format writer.
        /// This is the preferred method for exporting.
        /// </summary>
        /// <param name="writer">The format writer to use for export.</param>
        public void Export(IFormatWriter writer)
        {
            if (writer == null)
            {
                throw new ArgumentNullException(nameof(writer));
            }
            writer.WriteRobot(this);
        }

        /// <summary>
        /// Exports the robot model to the specified path in the given format.
        /// </summary>
        /// <param name="savePath">The output file path.</param>
        /// <param name="format">The export format (defaults to URDF).</param>
        public void Export(string savePath, ExportFormat format = ExportFormat.URDF)
        {
            if (string.IsNullOrEmpty(savePath))
            {
                throw new ArgumentNullException(nameof(savePath));
            }

            using (IFormatWriter writer = FormatWriterFactory.Create(format, savePath))
            {
                Export(writer);
            }
        }

        public void SetBaseLink(Link link)
        {
            BaseLink = link;
            ChildElements.Clear();
            ChildElements.Add(link);
        }

        internal string[] GetJointNames(bool includeFixed)
        {
            return BaseLink.GetJointNames(includeFixed);
        }
    }
}
