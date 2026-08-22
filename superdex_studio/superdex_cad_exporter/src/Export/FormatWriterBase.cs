/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using System.Xml;

namespace CADRobotExporter.Export
{
    /// <summary>
    /// Abstract base class providing common XML writing utilities for format writers.
    /// </summary>
    public abstract class FormatWriterBase : IFormatWriter
    {
        protected XmlWriter Writer { get; private set; }
        protected bool IsDisposed { get; private set; }
        private readonly string _savePath;

        public static readonly NumberFormatInfo NumberFormat =
            CultureInfo.CreateSpecificCulture("en-US").NumberFormat;

        protected FormatWriterBase(string savePath)
        {
            _savePath = savePath;
            XmlWriterSettings settings = new XmlWriterSettings
            {
                Encoding = new UTF8Encoding(false),
                Indent = true,
                NewLineOnAttributes = false,
                NewLineChars = "\n",
                NewLineHandling = NewLineHandling.Replace,
            };
            Writer = XmlWriter.Create(savePath, settings);
        }

        protected FormatWriterBase(XmlWriter writer)
        {
            Writer = writer ?? throw new ArgumentNullException(nameof(writer));
        }

        #region Utility Methods

        protected static string FormatDouble(double value)
        {
            return value.ToString(NumberFormat);
        }

        protected static string FormatDoubleArray(double[] values)
        {
            if (values == null)
            {
                return null;
            }

            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < values.Length; i++)
            {
                if (i > 0)
                {
                    sb.Append(' ');
                }
                sb.Append(FormatDouble(values[i]));
            }
            return sb.ToString();
        }

        protected void WriteAttributeIfNotNull(string name, string value)
        {
            if (value != null)
            {
                Writer.WriteAttributeString(name, value);
            }
        }

        protected void WriteAttributeIfNotNull(string name, double? value)
        {
            if (value.HasValue)
            {
                Writer.WriteAttributeString(name, FormatDouble(value.Value));
            }
        }

        protected void WriteAttributeIfNotNull(string name, double[] values)
        {
            if (values != null)
            {
                Writer.WriteAttributeString(name, FormatDoubleArray(values));
            }
        }

        #endregion

        #region Abstract Methods (must be implemented by derived classes)

        public abstract void WriteRobot(RobotDescription.Robot robot);
        public abstract void WriteLink(RobotDescription.Link link);
        public abstract void WriteJoint(RobotDescription.Joint joint);
        public abstract void WriteInertial(RobotDescription.Inertial inertial);
        public abstract void WriteVisual(RobotDescription.Visual visual);
        public abstract void WriteCollision(RobotDescription.Collision collision);
        public abstract void WriteOrigin(RobotDescription.Origin origin);
        public abstract void WriteGeometry(RobotDescription.Geometry geometry);
        public abstract void WriteMesh(RobotDescription.Mesh mesh);
        public abstract void WriteMaterial(RobotDescription.Material material);
        public abstract void WriteInertia(RobotDescription.Inertia inertia);
        public abstract void WriteMass(RobotDescription.Mass mass);
        public abstract void WriteAxis(RobotDescription.Axis axis);
        public abstract void WriteLimit(RobotDescription.Limit limit);
        public abstract void WriteDynamics(RobotDescription.Dynamics dynamics);
        public abstract void WriteMimic(RobotDescription.Mimic mimic);
        public abstract void WriteSafetyController(RobotDescription.SafetyController safety);
        public abstract void WriteCalibration(RobotDescription.Calibration calibration);
        public abstract void WriteColor(RobotDescription.Color color);
        public abstract void WriteTexture(RobotDescription.Texture texture);
        public abstract void WriteTendons(List<RobotDescription.Tendon> tendons);

        #endregion

        #region IDisposable

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!IsDisposed)
            {
                if (disposing)
                {
                    Writer?.Close();
                    Writer?.Dispose();

                    if (_savePath != null)
                    {
                        System.IO.File.AppendAllText(_savePath, "\n");
                    }
                }
                IsDisposed = true;
            }
        }

        #endregion
    }
}
