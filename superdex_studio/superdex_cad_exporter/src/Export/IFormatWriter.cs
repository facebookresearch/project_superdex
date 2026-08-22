/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Collections.Generic;

namespace CADRobotExporter.Export
{
    /// <summary>
    /// Interface for writing robot models to various formats (URDF, MJCF, etc.)
    /// Implements the Visitor pattern to separate serialization from data model.
    /// </summary>
    public interface IFormatWriter : IDisposable
    {
        /// <summary>
        /// Writes the complete robot model to the output.
        /// </summary>
        void WriteRobot(RobotDescription.Robot robot);

        /// <summary>
        /// Writes a link element.
        /// </summary>
        void WriteLink(RobotDescription.Link link);

        /// <summary>
        /// Writes a joint element.
        /// </summary>
        void WriteJoint(RobotDescription.Joint joint);

        /// <summary>
        /// Writes an inertial element.
        /// </summary>
        void WriteInertial(RobotDescription.Inertial inertial);

        /// <summary>
        /// Writes a visual element.
        /// </summary>
        void WriteVisual(RobotDescription.Visual visual);

        /// <summary>
        /// Writes a collision element.
        /// </summary>
        void WriteCollision(RobotDescription.Collision collision);

        /// <summary>
        /// Writes an origin (pose) element.
        /// </summary>
        void WriteOrigin(RobotDescription.Origin origin);

        /// <summary>
        /// Writes a geometry element.
        /// </summary>
        void WriteGeometry(RobotDescription.Geometry geometry);

        /// <summary>
        /// Writes a mesh element.
        /// </summary>
        void WriteMesh(RobotDescription.Mesh mesh);

        /// <summary>
        /// Writes a material element.
        /// </summary>
        void WriteMaterial(RobotDescription.Material material);

        /// <summary>
        /// Writes an inertia (moment of inertia) element.
        /// </summary>
        void WriteInertia(RobotDescription.Inertia inertia);

        /// <summary>
        /// Writes a mass element.
        /// </summary>
        void WriteMass(RobotDescription.Mass mass);

        /// <summary>
        /// Writes an axis element.
        /// </summary>
        void WriteAxis(RobotDescription.Axis axis);

        /// <summary>
        /// Writes joint limits.
        /// </summary>
        void WriteLimit(RobotDescription.Limit limit);

        /// <summary>
        /// Writes joint dynamics (damping, friction).
        /// </summary>
        void WriteDynamics(RobotDescription.Dynamics dynamics);

        /// <summary>
        /// Writes joint mimic configuration.
        /// </summary>
        void WriteMimic(RobotDescription.Mimic mimic);

        /// <summary>
        /// Writes safety controller configuration.
        /// </summary>
        void WriteSafetyController(RobotDescription.SafetyController safety);

        /// <summary>
        /// Writes calibration data.
        /// </summary>
        void WriteCalibration(RobotDescription.Calibration calibration);

        /// <summary>
        /// Writes color element.
        /// </summary>
        void WriteColor(RobotDescription.Color color);

        /// <summary>
        /// Writes texture element.
        /// </summary>
        void WriteTexture(RobotDescription.Texture texture);

        /// <summary>
        /// Writes tendon elements.
        /// </summary>
        void WriteTendons(List<RobotDescription.Tendon> tendons);
    }
}
