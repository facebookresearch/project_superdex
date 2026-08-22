/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;

namespace CADRobotExporter.CAD
{
    public static class SolidworksJointVisualizer
    {
        /// <summary>
        /// Draws a semicircle arc representing joint limits
        /// </summary>
        /// <param name="center">Center point of the arc (joint position)</param>
        /// <param name="axis">Axis vector (will be normalized)</param>
        /// <param name="startAngle">Start angle in radians</param>
        /// <param name="endAngle">End angle in radians</param>
        /// <param name="radius">Radius of the arc</param>
        /// <param name="segments">Number of segments for smoothness</param>
        /// <param name="filled">Whether to fill the arc or just draw outline</param>
        public static void DrawJointLimitArc(
            double[] center,
            double[] axis,
            double startAngle,
            double endAngle,
            double radius = 1.0,
            int segments = 32,
            bool filled = true,
            double[] referenceDirection = null)
        {
            // Normalize the axis
            double[] axisNorm = NormalizeVector(axis);
            // Create two perpendicular vectors to the axis
            double[] perpVec1 = GetOrientedPerpendicularVector(axisNorm, referenceDirection);
            double[] perpVec2 = CrossProduct(axisNorm, perpVec1);
            // Convert angles to radians
            double angleRange = endAngle - startAngle;
            double angleStep = angleRange / segments;
            if (filled)
            {
                OpenGL.glDisable(OpenGL.GL_CULL_FACE);
                // Draw filled arc using triangle fan
                OpenGL.glBegin(OpenGL.GL_TRIANGLE_FAN);

                // Center vertex
                OpenGL.glVertex3d(center[0], center[1], center[2]);
                // Arc vertices
                for (int i = 0; i <= segments; i++)
                {
                    double angle = startAngle + (i * angleStep);
                    double x = Math.Cos(angle);
                    double y = Math.Sin(angle);
                    // Calculate point on arc using the perpendicular vectors
                    double px = center[0] + radius * (x * perpVec1[0] + y * perpVec2[0]);
                    double py = center[1] + radius * (x * perpVec1[1] + y * perpVec2[1]);
                    double pz = center[2] + radius * (x * perpVec1[2] + y * perpVec2[2]);
                    OpenGL.glVertex3d(px, py, pz);
                }
                OpenGL.glEnd();
                OpenGL.glEnable(OpenGL.GL_CULL_FACE);
            }
            else
            {
                // Draw arc outline
                OpenGL.glDisable(OpenGL.GL_DEPTH_TEST);
                OpenGL.glBegin(OpenGL.GL_LINE_STRIP);
                for (int i = 0; i <= segments; i++)
                {
                    double angle = startAngle + (i * angleStep);
                    double x = Math.Cos(angle);
                    double y = Math.Sin(angle);
                    double px = center[0] + radius * (x * perpVec1[0] + y * perpVec2[0]);
                    double py = center[1] + radius * (x * perpVec1[1] + y * perpVec2[1]);
                    double pz = center[2] + radius * (x * perpVec1[2] + y * perpVec2[2]);
                    OpenGL.glVertex3d(px, py, pz);
                }
                OpenGL.glEnd();
                OpenGL.glEnable(OpenGL.GL_DEPTH_TEST);
            }
        }

        /// <summary>
        /// Draws a radial line from center to arc edge at a given angle in radians
        /// </summary>
        private static void DrawRadialLine(double[] center, double[] axis, double angle, double radius, double[] referenceDirection = null)
        {
            double[] axisNorm = NormalizeVector(axis);
            double[] perpVec1 = GetOrientedPerpendicularVector(axisNorm, referenceDirection);
            double[] perpVec2 = CrossProduct(axisNorm, perpVec1);
            double x = Math.Cos(angle);
            double y = Math.Sin(angle);
            double px = center[0] + radius * (x * perpVec1[0] + y * perpVec2[0]);
            double py = center[1] + radius * (x * perpVec1[1] + y * perpVec2[1]);
            double pz = center[2] + radius * (x * perpVec1[2] + y * perpVec2[2]);
            OpenGL.glDisable(OpenGL.GL_DEPTH_TEST);
            OpenGL.glBegin(OpenGL.GL_LINES);
            OpenGL.glVertex3d(center[0], center[1], center[2]);
            OpenGL.glVertex3d(px, py, pz);
            OpenGL.glEnd();
            OpenGL.glEnable(OpenGL.GL_DEPTH_TEST);
        }

        /// <summary>
        /// Draw a simple 2D-style arrowhead (just two wings)
        /// </summary>
        private static void DrawStraightArrowhead2D(
            double[] position,
            double[] direction,
            double arrowSize)
        {
            // Get one perpendicular vector
            double[] perp = GetPerpendicularVector(direction);
            perp = NormalizeVector(perp);
            double backDist = arrowSize;
            double spreadDist = arrowSize * 0.5;
            // Wing 1
            double[] wing1 = new double[3];
            for (int i = 0; i < 3; i++)
            {
                wing1[i] = position[i] - backDist * direction[i] + spreadDist * perp[i];
            }
            // Wing 2
            double[] wing2 = new double[3];
            for (int i = 0; i < 3; i++)
            {
                wing2[i] = position[i] - backDist * direction[i] - spreadDist * perp[i];
            }
            // Draw V-shaped arrowhead
            OpenGL.glBegin(OpenGL.GL_LINES);

            OpenGL.glVertex3d(position[0], position[1], position[2]);
            OpenGL.glVertex3d(wing1[0], wing1[1], wing1[2]);

            OpenGL.glVertex3d(position[0], position[1], position[2]);
            OpenGL.glVertex3d(wing2[0], wing2[1], wing2[2]);

            OpenGL.glEnd();
        }

        /// <summary>
        /// Draw a line with arrow representing the joint axis
        /// </summary>
        /// <param name="center">Center/origin of the axis</param>
        /// <param name="axis">Axis direction vector</param>
        /// <param name="length">Length of the axis line</param>
        /// <param name="arrowSize">Size of the arrowhead</param>
        /// <param name="use3DArrow">Use 4-wing 3D arrow or 2-wing flat arrow</param>
        public static void DrawAxisArrow(
            double[] center,
            double[] axis,
            double length = 1.0,
            double arrowSize = 0.1)
        {
            // Normalize axis
            double[] axisNorm = NormalizeVector(axis);
            // Calculate end point
            double[] endPoint = new double[3];
            for (int i = 0; i < 3; i++)
            {
                endPoint[i] = center[i] + length * axisNorm[i];
            }
            OpenGL.glDisable(OpenGL.GL_DEPTH_TEST);
            OpenGL.glLineWidth(5.0f);
            // Draw the line
            OpenGL.glBegin(OpenGL.GL_LINES);
            OpenGL.glVertex3d(center[0], center[1], center[2]);
            OpenGL.glVertex3d(endPoint[0], endPoint[1], endPoint[2]);
            OpenGL.glEnd();

            DrawStraightArrowhead2D(endPoint, axisNorm, arrowSize);
            OpenGL.glEnable(OpenGL.GL_DEPTH_TEST);
        }

        /// <summary>
        /// Draws a cylinder representing the joint axis
        /// </summary>
        /// <param name="center">Center point of the joint</param>
        /// <param name="axis">Axis direction vector</param>
        /// <param name="length">Length of the cylinder</param>
        /// <param name="radius">Radius of the cylinder</param>
        /// <param name="segments">Number of segments around the cylinder</param>
        public static void DrawAxisCylinder(
            double[] center,
            double[] axis,
            double length = 1.0,
            double radius = 0.05,
            int segments = 16)
        {
            // Normalize the axis
            double[] axisNorm = NormalizeVector(axis);

            // Create two perpendicular vectors to the axis
            double[] perpVec1 = GetPerpendicularVector(axisNorm);
            double[] perpVec2 = CrossProduct(axisNorm, perpVec1);

            // Calculate start and end points
            double[] startPoint = new double[3];
            double[] endPoint = new double[3];

            for (int i = 0; i < 3; i++)
            {
                startPoint[i] = center[i] - (axisNorm[i] * length / 2.0);
                endPoint[i] = center[i] + (axisNorm[i] * length / 2.0);
            }

            // Draw cylinder sides
            OpenGL.glBegin(OpenGL.GL_QUADS);

            for (int i = 0; i < segments; i++)
            {
                double angle1 = (2.0 * Math.PI * i) / segments;
                double angle2 = (2.0 * Math.PI * (i + 1)) / segments;

                double x1 = Math.Cos(angle1);
                double y1 = Math.Sin(angle1);
                double x2 = Math.Cos(angle2);
                double y2 = Math.Sin(angle2);

                // First vertex (bottom of current segment)
                double p1x = startPoint[0] + radius * (x1 * perpVec1[0] + y1 * perpVec2[0]);
                double p1y = startPoint[1] + radius * (x1 * perpVec1[1] + y1 * perpVec2[1]);
                double p1z = startPoint[2] + radius * (x1 * perpVec1[2] + y1 * perpVec2[2]);

                // Second vertex (bottom of next segment)
                double p2x = startPoint[0] + radius * (x2 * perpVec1[0] + y2 * perpVec2[0]);
                double p2y = startPoint[1] + radius * (x2 * perpVec1[1] + y2 * perpVec2[1]);
                double p2z = startPoint[2] + radius * (x2 * perpVec1[2] + y2 * perpVec2[2]);

                // Third vertex (top of next segment)
                double p3x = endPoint[0] + radius * (x2 * perpVec1[0] + y2 * perpVec2[0]);
                double p3y = endPoint[1] + radius * (x2 * perpVec1[1] + y2 * perpVec2[1]);
                double p3z = endPoint[2] + radius * (x2 * perpVec1[2] + y2 * perpVec2[2]);

                // Fourth vertex (top of current segment)
                double p4x = endPoint[0] + radius * (x1 * perpVec1[0] + y1 * perpVec2[0]);
                double p4y = endPoint[1] + radius * (x1 * perpVec1[1] + y1 * perpVec2[1]);
                double p4z = endPoint[2] + radius * (x1 * perpVec1[2] + y1 * perpVec2[2]);

                // Calculate normal for this face
                double nx = x1 + x2;
                double ny = y1 + y2;
                double normalX = nx * perpVec1[0] + ny * perpVec2[0];
                double normalY = nx * perpVec1[1] + ny * perpVec2[1];
                double normalZ = nx * perpVec1[2] + ny * perpVec2[2];

                OpenGL.glNormal3f((float)normalX, (float)normalY, (float)normalZ);

                OpenGL.glVertex3d(p1x, p1y, p1z);
                OpenGL.glVertex3d(p2x, p2y, p2z);
                OpenGL.glVertex3d(p3x, p3y, p3z);
                OpenGL.glVertex3d(p4x, p4y, p4z);
            }

            OpenGL.glEnd();

            // Draw end caps (optional but looks better)
            DrawCylinderCap(startPoint, axisNorm, perpVec1, perpVec2, radius, segments, false);
            DrawCylinderCap(endPoint, axisNorm, perpVec1, perpVec2, radius, segments, false);
        }

        /// <summary>
        /// Draws a cap (circle) at the end of the cylinder
        /// </summary>
        private static void DrawCylinderCap(
            double[] center,
            double[] normal,
            double[] perpVec1,
            double[] perpVec2,
            double radius,
            int segments,
            bool flipNormal)
        {
            OpenGL.glDisable(OpenGL.GL_CULL_FACE);
            OpenGL.glBegin(OpenGL.GL_TRIANGLE_FAN);

            // Normal for the cap
            if (flipNormal)
                OpenGL.glNormal3f((float)normal[0], (float)normal[1], (float)normal[2]);
            else
                OpenGL.glNormal3f((float)-normal[0], (float)-normal[1], (float)-normal[2]);

            // Center vertex
            OpenGL.glVertex3d(center[0], center[1], center[2]);

            // Circle vertices
            for (int i = 0; i <= segments; i++)
            {
                double angle = (2.0 * Math.PI * i) / segments;
                double x = Math.Cos(angle);
                double y = Math.Sin(angle);

                double px = center[0] + radius * (x * perpVec1[0] + y * perpVec2[0]);
                double py = center[1] + radius * (x * perpVec1[1] + y * perpVec2[1]);
                double pz = center[2] + radius * (x * perpVec1[2] + y * perpVec2[2]);

                OpenGL.glVertex3d(px, py, pz);
            }

            OpenGL.glEnd();
            OpenGL.glEnable(OpenGL.GL_CULL_FACE);
        }

        /// <summary>
        /// Draw a plus sign (+) at a specific position
        /// </summary>
        private static void DrawPlusSign(
            double[] position,
            double[] axis,
            double size = 0.1)
        {
            // Get perpendicular vectors for the symbol orientation
            double[] axisNorm = NormalizeVector(axis);
            double[] perpVec1 = GetPerpendicularVector(axisNorm);
            double[] perpVec2 = CrossProduct(axisNorm, perpVec1);
            // Draw horizontal line
            OpenGL.glDisable(OpenGL.GL_DEPTH_TEST);
            OpenGL.glBegin(OpenGL.GL_LINES);

            double hx1 = position[0] - size * perpVec1[0];
            double hy1 = position[1] - size * perpVec1[1];
            double hz1 = position[2] - size * perpVec1[2];

            double hx2 = position[0] + size * perpVec1[0];
            double hy2 = position[1] + size * perpVec1[1];
            double hz2 = position[2] + size * perpVec1[2];

            OpenGL.glVertex3d(hx1, hy1, hz1);
            OpenGL.glVertex3d(hx2, hy2, hz2);

            // Draw vertical line
            double vx1 = position[0] - size * perpVec2[0];
            double vy1 = position[1] - size * perpVec2[1];
            double vz1 = position[2] - size * perpVec2[2];

            double vx2 = position[0] + size * perpVec2[0];
            double vy2 = position[1] + size * perpVec2[1];
            double vz2 = position[2] + size * perpVec2[2];

            OpenGL.glVertex3d(vx1, vy1, vz1);
            OpenGL.glVertex3d(vx2, vy2, vz2);

            OpenGL.glEnd();
            OpenGL.glEnable(OpenGL.GL_DEPTH_TEST);
        }

        /// <summary>
        /// Draw a minus sign (-) at a specific position
        /// </summary>
        private static void DrawMinusSign(
            double[] position,
            double[] axis,
            double size = 0.1)
        {
            // Get perpendicular vectors for the symbol orientation
            double[] axisNorm = NormalizeVector(axis);
            double[] perpVec1 = GetPerpendicularVector(axisNorm);
            // Draw horizontal line only
            OpenGL.glDisable(OpenGL.GL_DEPTH_TEST);
            OpenGL.glBegin(OpenGL.GL_LINES);

            double hx1 = position[0] - size * perpVec1[0];
            double hy1 = position[1] - size * perpVec1[1];
            double hz1 = position[2] - size * perpVec1[2];

            double hx2 = position[0] + size * perpVec1[0];
            double hy2 = position[1] + size * perpVec1[1];
            double hz2 = position[2] + size * perpVec1[2];

            OpenGL.glVertex3d(hx1, hy1, hz1);
            OpenGL.glVertex3d(hx2, hy2, hz2);

            OpenGL.glEnd();
            OpenGL.glEnable(OpenGL.GL_DEPTH_TEST);
        }

        /// <summary>
        /// Draws complete joint visualization with axis cylinder and limit arc
        /// </summary>
        public static void DrawCompleteRevoluteJoint(
            double[] center,
            double[] axis,
            double minAngle,
            double maxAngle,
            double arcRadius = 0.2,
            double axisLength = 0.2,
            double[] referenceDirection = null)
        {
            double symbolSize = arcRadius * 0.05;

            OpenGL.glDisable(OpenGL.GL_LIGHTING);
            OpenGL.glDisable(OpenGL.GL_BLEND);

            // Draw arc outline
            OpenGL.glLineWidth(2.0f);
            OpenGL.glColor3f(0.0f, 0.0f, 1.0f); // Blue outline
            DrawJointLimitArc(center, axis, minAngle, maxAngle, arcRadius, 32, false, referenceDirection);

            // Lower limit - Red
            OpenGL.glLineWidth(10.0f);
            OpenGL.glColor3f(181f / 255f, 36f / 255f, 0.0f);
            DrawRadialLine(center, axis, minAngle, arcRadius, referenceDirection);

            double[] minEndpoint = GetRadialEndpoint(center, axis, minAngle, arcRadius * 1.1, referenceDirection);
            OpenGL.glLineWidth(2.0f);
            DrawMinusSign(minEndpoint, axis, symbolSize);

            // Upper limit - Green
            OpenGL.glLineWidth(10.0f);
            OpenGL.glColor3f(51f / 255f, 181f / 255f, 0.0f);
            DrawRadialLine(center, axis, maxAngle, arcRadius, referenceDirection);

            // Center line - Blue
            OpenGL.glLineWidth(16.0f);
            OpenGL.glColor3f(28f / 255f, 94f / 255f, 235f / 255f);
            DrawRadialLine(center, axis, 0, arcRadius * 0.66f, referenceDirection);

            double[] maxEndpoint = GetRadialEndpoint(center, axis, maxAngle, arcRadius * 1.1, referenceDirection);
            OpenGL.glLineWidth(2.0f);
            OpenGL.glColor3f(51f / 255f, 181f / 255f, 0.0f);
            DrawPlusSign(maxEndpoint, axis, symbolSize);

            // Draw axis arrow
            OpenGL.glColor3f(0.0f, 0.0f, 1.0f);
            DrawAxisArrow(center, axis, axisLength, axisLength * 0.2f);

            // Draw joint limit arc
            OpenGL.glEnable(OpenGL.GL_BLEND);
            OpenGL.glBlendFunc(OpenGL.GL_SRC_ALPHA, OpenGL.GL_ONE_MINUS_SRC_ALPHA);
            OpenGL.glColor4f(0.0f, 0.5f, 1.0f, 0.3f); // Light blue, semi-transparent

            int numSegments = (int)Math.Round(64.0 * (Math.Abs(maxAngle - minAngle) / Math.PI));
            numSegments = Math.Max(numSegments, 64);
            DrawJointLimitArc(center, axis, minAngle, maxAngle, arcRadius, numSegments, true, referenceDirection);

            OpenGL.glColor3f(0.0f, 0.0f, 1.0f); // Blue outline
            DrawRotationDirectionArrow(center, axis, arcRadius * 0.2f, 270.0f, 20.0f, referenceDirection);
        }

        /// <summary>
        /// Draw a filled circle perpendicular to an axis
        /// </summary>
        private static void DrawFilledCircle(
            double[] center,
            double[] axis,
            double radius,
            int segments = 32)
        {
            // Normalize the axis
            double[] axisNorm = NormalizeVector(axis);

            // Create two perpendicular vectors to the axis
            double[] perpVec1 = GetPerpendicularVector(axisNorm);
            double[] perpVec2 = CrossProduct(axisNorm, perpVec1);

            OpenGL.glBegin(OpenGL.GL_TRIANGLE_FAN);

            // Set normal pointing along axis
            OpenGL.glNormal3f((float)axisNorm[0], (float)axisNorm[1], (float)axisNorm[2]);

            // Center vertex
            OpenGL.glVertex3d(center[0], center[1], center[2]);

            // Circle vertices
            for (int i = 0; i <= segments; i++)
            {
                double angle = (2.0 * Math.PI * i) / segments;
                double x = Math.Cos(angle);
                double y = Math.Sin(angle);

                double px = center[0] + radius * (x * perpVec1[0] + y * perpVec2[0]);
                double py = center[1] + radius * (x * perpVec1[1] + y * perpVec2[1]);
                double pz = center[2] + radius * (x * perpVec1[2] + y * perpVec2[2]);

                OpenGL.glVertex3d(px, py, pz);
            }

            OpenGL.glEnd();
        }

        /// <summary>
        /// Draw a circle outline perpendicular to an axis
        /// </summary>
        private static void DrawCircleOutline(
            double[] center,
            double[] axis,
            double radius,
            int segments = 32)
        {
            // Normalize the axis
            double[] axisNorm = NormalizeVector(axis);

            // Create two perpendicular vectors to the axis
            double[] perpVec1 = GetPerpendicularVector(axisNorm);
            double[] perpVec2 = CrossProduct(axisNorm, perpVec1);

            OpenGL.glBegin(OpenGL.GL_LINE_LOOP);

            // Circle vertices
            for (int i = 0; i < segments; i++)
            {
                double angle = (2.0 * Math.PI * i) / segments;
                double x = Math.Cos(angle);
                double y = Math.Sin(angle);

                double px = center[0] + radius * (x * perpVec1[0] + y * perpVec2[0]);
                double py = center[1] + radius * (x * perpVec1[1] + y * perpVec2[1]);
                double pz = center[2] + radius * (x * perpVec1[2] + y * perpVec2[2]);

                OpenGL.glVertex3d(px, py, pz);
            }

            OpenGL.glEnd();
        }

        /// <summary>
        /// Draw a prismatic joint showing linear limits
        /// </summary>
        /// <param name="jointPosition">Base position of the joint</param>
        /// <param name="axis">Direction of linear motion</param>
        /// <param name="minLimit">Minimum distance along axis</param>
        /// <param name="maxLimit">Maximum distance along axis</param>
        /// <param name="circleRadius">Radius of the limit circles</param>
        public static void DrawPrismaticJoint(
            double[] jointPosition,
            double[] axis,
            double minLimit,
            double maxLimit,
            double circleRadius = 0.2)
        {
            OpenGL.glDisable(OpenGL.GL_LIGHTING);

            // Normalize axis
            double[] axisNorm = NormalizeVector(axis);

            // Calculate positions of min and max limits
            double[] minPosition = new double[3];
            double[] maxPosition = new double[3];

            for (int i = 0; i < 3; i++)
            {
                minPosition[i] = jointPosition[i] + minLimit * axisNorm[i];
                maxPosition[i] = jointPosition[i] + maxLimit * axisNorm[i];
            }
            // Disable depth test so circles render on top
            OpenGL.glDisable(OpenGL.GL_DEPTH_TEST);

            // Draw connecting line
            OpenGL.glLineWidth(2.0f);
            OpenGL.glColor3f(28f / 255f, 94f / 255f, 235f / 255f);

            OpenGL.glBegin(OpenGL.GL_LINES);
            OpenGL.glVertex3d(minPosition[0], minPosition[1], minPosition[2]);
            OpenGL.glVertex3d(maxPosition[0], maxPosition[1], maxPosition[2]);
            OpenGL.glEnd();

            // Draw minimum limit circle (Red)
            OpenGL.glColor4f(181f / 255f, 36f / 255f, 0.0f, 0.7f);
            OpenGL.glEnable(OpenGL.GL_BLEND);
            OpenGL.glBlendFunc(OpenGL.GL_SRC_ALPHA, OpenGL.GL_ONE_MINUS_SRC_ALPHA);

            DrawFilledCircle(minPosition, axis, circleRadius);

            // Draw minimum limit outline
            OpenGL.glDisable(OpenGL.GL_BLEND);
            OpenGL.glLineWidth(2.0f);
            OpenGL.glColor3f(0.8f, 0.0f, 0.0f); // Darker red
            DrawCircleOutline(minPosition, axis, circleRadius);

            // Draw maximum limit circle (Green)
            OpenGL.glColor4f(51f / 255f, 181f / 255f, 0.0f, 0.7f);
            OpenGL.glEnable(OpenGL.GL_BLEND);
            OpenGL.glBlendFunc(OpenGL.GL_SRC_ALPHA, OpenGL.GL_ONE_MINUS_SRC_ALPHA);

            DrawFilledCircle(maxPosition, axis, circleRadius);

            // Draw maximum limit outline
            OpenGL.glDisable(OpenGL.GL_BLEND);
            OpenGL.glLineWidth(2.0f);
            OpenGL.glColor3f(0.0f, 0.8f, 0.0f); // Darker green
            DrawCircleOutline(maxPosition, axis, circleRadius);

            // Re-enable depth test
            OpenGL.glEnable(OpenGL.GL_DEPTH_TEST);

            OpenGL.glLineWidth(1.0f);
        }

        /// <summary>
        /// Draw a prismatic joint with axis visualization
        /// </summary>
        public static void DrawCompletePrismaticJoint(
            double[] jointPosition,
            double[] axis,
            double minLimit,
            double maxLimit,
            double circleRadius = 0.2,
            double axisLength = 0.2)
        {
            OpenGL.glDisable(OpenGL.GL_LIGHTING);

            // Normalize axis
            double[] axisNorm = NormalizeVector(axis);

            // Draw axis arrow
            OpenGL.glColor3f(0.0f, 0.0f, 1.0f);
            DrawAxisArrow(jointPosition, axis, axisLength, axisLength * 0.2f);

            OpenGL.glEnable(OpenGL.GL_BLEND);
            OpenGL.glBlendFunc(OpenGL.GL_SRC_ALPHA, OpenGL.GL_ONE_MINUS_SRC_ALPHA);
            OpenGL.glColor4f(0.0f, 0.5f, 1.0f, 0.3f); // Light blue, semi-transparent

            // Draw the joint limits
            DrawPrismaticJoint(jointPosition, axis, minLimit, maxLimit, circleRadius);
        }

        /// <summary>
        /// Draw an arrowhead at the end of an arc (simplified version)
        /// </summary>
        private static void DrawArcArrowheadSimple(
            double[] center,
            double[] axis,
            double angle,
            double radius,
            double arrowSize,
            double[] referenceDirection = null)
        {
            double[] axisNorm = NormalizeVector(axis);
            double[] perpVec1 = GetOrientedPerpendicularVector(axisNorm, referenceDirection);
            double[] perpVec2 = CrossProduct(axisNorm, perpVec1);
            double angleRad = angle * Math.PI / 180.0;

            // Arrow tip position
            double x0 = Math.Cos(angleRad);
            double y0 = Math.Sin(angleRad);
            double px = center[0] + radius * (x0 * perpVec1[0] + y0 * perpVec2[0]);
            double py = center[1] + radius * (x0 * perpVec1[1] + y0 * perpVec2[1]);
            double pz = center[2] + radius * (x0 * perpVec1[2] + y0 * perpVec2[2]);
            // Convert arc length to angle
            double backAngle = arrowSize / radius;
            double spreadAngle = backAngle * 0.5;
            // Wing 1: back and inward
            double angle1Rad = angleRad - backAngle;
            double x1 = Math.Cos(angle1Rad);
            double y1 = Math.Sin(angle1Rad);
            double r1 = radius - arrowSize * 0.3;
            double p1x = center[0] + r1 * (x1 * perpVec1[0] + y1 * perpVec2[0]);
            double p1y = center[1] + r1 * (x1 * perpVec1[1] + y1 * perpVec2[1]);
            double p1z = center[2] + r1 * (x1 * perpVec1[2] + y1 * perpVec2[2]);
            // Wing 2: back and outward
            double angle2Rad = angleRad - backAngle;
            double x2 = Math.Cos(angle2Rad);
            double y2 = Math.Sin(angle2Rad);
            double r2 = radius + arrowSize * 0.3;
            double p2x = center[0] + r2 * (x2 * perpVec1[0] + y2 * perpVec2[0]);
            double p2y = center[1] + r2 * (x2 * perpVec1[1] + y2 * perpVec2[1]);
            double p2z = center[2] + r2 * (x2 * perpVec1[2] + y2 * perpVec2[2]);
            // Draw two line wings forming a V
            OpenGL.glBegin(OpenGL.GL_LINES);
            // Wing 1 (inner)
            OpenGL.glVertex3d(px, py, pz);
            OpenGL.glVertex3d(p1x, p1y, p1z);
            // Wing 2 (outer)
            OpenGL.glVertex3d(px, py, pz);
            OpenGL.glVertex3d(p2x, p2y, p2z);
            OpenGL.glEnd();
        }

        /// <summary>
        /// Draw a small arc with arrow showing rotation direction
        /// </summary>
        /// <param name="center">Center of rotation</param>
        /// <param name="axis">Rotation axis (right-hand rule)</param>
        /// <param name="radius">Radius of the direction arrow arc</param>
        /// <param name="arcAngle">Arc span in degrees (default 120)</param>
        /// <param name="startOffset">Starting angle offset in degrees (default 20)</param>
        public static void DrawRotationDirectionArrow(
            double[] center,
            double[] axis,
            double radius = 0.3,
            double arcAngle = 120.0,
            double startOffset = 20.0,
            double[] referenceDirection = null)
        {
            OpenGL.glDisable(OpenGL.GL_DEPTH_TEST);
            // Draw the arc
            double startAngle = startOffset;
            double endAngle = startOffset + arcAngle;
            OpenGL.glLineWidth(5.0f);
            OpenGL.glBegin(OpenGL.GL_LINE_STRIP);
            int segments = 24;
            for (int i = 0; i <= segments; i++)
            {
                double t = (double)i / segments;
                double angle = startAngle + t * arcAngle;
                double angleRad = angle * Math.PI / 180.0;
                double[] axisNorm = NormalizeVector(axis);
                double[] perpVec1 = GetOrientedPerpendicularVector(axisNorm, referenceDirection);
                double[] perpVec2 = CrossProduct(axisNorm, perpVec1);
                double x = Math.Cos(angleRad);
                double y = Math.Sin(angleRad);
                double px = center[0] + radius * (x * perpVec1[0] + y * perpVec2[0]);
                double py = center[1] + radius * (x * perpVec1[1] + y * perpVec2[1]);
                double pz = center[2] + radius * (x * perpVec1[2] + y * perpVec2[2]);
                OpenGL.glVertex3d(px, py, pz);
            }
            OpenGL.glEnd();
            // Draw arrowhead at the end
            double arrowSize = radius * 0.25f;
            DrawArcArrowheadSimple(center, axis, endAngle, radius, arrowSize, referenceDirection);
            OpenGL.glLineWidth(1.0f);
            OpenGL.glEnable(OpenGL.GL_DEPTH_TEST);
        }

        /// <summary>
        /// Normalize a vector
        /// </summary>
        private static double[] NormalizeVector(double[] v)
        {
            double length = Math.Sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            if (length < 1e-10) return new double[] { 0, 0, 1 }; // Default to Z-axis

            return new double[] { v[0] / length, v[1] / length, v[2] / length };
        }
        /// <summary>
        /// Get a vector perpendicular to the input vector, optionally oriented toward a reference direction.
        /// </summary>
        private static double[] GetOrientedPerpendicularVector(double[] axis, double[] referenceDirection)
        {
            if (referenceDirection != null)
            {
                double dot = referenceDirection[0] * axis[0] + referenceDirection[1] * axis[1] + referenceDirection[2] * axis[2];
                double[] projected = new double[] {
                    referenceDirection[0] - dot * axis[0],
                    referenceDirection[1] - dot * axis[1],
                    referenceDirection[2] - dot * axis[2]
                };
                double len = Math.Sqrt(projected[0] * projected[0] + projected[1] * projected[1] + projected[2] * projected[2]);
                if (len > 1e-10)
                    return new double[] { projected[0] / len, projected[1] / len, projected[2] / len };
            }
            return GetPerpendicularVector(axis);
        }

        /// <summary>
        /// Get a vector perpendicular to the input vector
        /// </summary>
        private static double[] GetPerpendicularVector(double[] v)
        {
            double[] result;

            // Choose the axis that is least aligned with v
            if (Math.Abs(v[0]) < Math.Abs(v[1]) && Math.Abs(v[0]) < Math.Abs(v[2]))
            {
                result = new double[] { 1, 0, 0 };
            }
            else if (Math.Abs(v[1]) < Math.Abs(v[2]))
            {
                result = new double[] { 0, 1, 0 };
            }
            else
            {
                result = new double[] { 0, 0, 1 };
            }
            // Get perpendicular using cross product
            double[] perp = CrossProduct(v, result);
            return NormalizeVector(perp);
        }

        /// <summary>
        /// Get the endpoint of a radial line at a given angle in radians
        /// </summary>
        private static double[] GetRadialEndpoint(
            double[] center,
            double[] axis,
            double angle,
            double radius,
            double[] referenceDirection = null)
        {
            double[] axisNorm = NormalizeVector(axis);
            double[] perpVec1 = GetOrientedPerpendicularVector(axisNorm, referenceDirection);
            double[] perpVec2 = CrossProduct(axisNorm, perpVec1);
            double x = Math.Cos(angle);
            double y = Math.Sin(angle);
            return new double[]
            {
                center[0] + radius * (x * perpVec1[0] + y * perpVec2[0]),
                center[1] + radius * (x * perpVec1[1] + y * perpVec2[1]),
                center[2] + radius * (x * perpVec1[2] + y * perpVec2[2])
            };
        }

        /// <summary>
        /// Calculate scale factor to maintain constant screen size
        /// </summary>
        /// <param name="worldPosition">Position in world space</param>
        /// <param name="pixelSize">Desired size in pixels (default 100)</param>
        /// <returns>Scale factor to apply</returns>
        public static double GetScreenSizeScale(double[] worldPosition, double pixelSize = 100.0)
        {
            // Get viewport dimensions
            int[] viewport = new int[4];
            OpenGL.glGetIntegerv(OpenGL.GL_VIEWPORT, viewport);
            int screenHeight = viewport[3];
            // Get modelview and projection matrices
            double[] modelview = new double[16];
            double[] projection = new double[16];
            OpenGL.glGetDoublev(OpenGL.GL_MODELVIEW_MATRIX, modelview);
            OpenGL.glGetDoublev(OpenGL.GL_PROJECTION_MATRIX, projection);
            // Transform point to eye space
            double[] eyePos = MultiplyMatrixVector(modelview, worldPosition);
            // Get the distance from camera (magnitude of eye space position)
            double distance = Math.Sqrt(eyePos[0] * eyePos[0] + eyePos[1] * eyePos[1] + eyePos[2] * eyePos[2]);
            // Calculate field of view from projection matrix
            // For perspective: fov relates to projection[5] (m[1][1])
            double fovFactor = projection[5]; // This is cot(fov/2) for perspective
            if (Math.Abs(fovFactor) < 0.0001)
            {
                // Orthographic projection
                // Scale is based on orthographic height
                double orthoHeight = 2.0 / projection[5];
                return (pixelSize / screenHeight) * orthoHeight;
            }
            else
            {
                // Perspective projection
                // Object size in world units = (pixel size / screen height) * (distance / fovFactor)
                double scale = (pixelSize / screenHeight) * (distance / fovFactor);
                return scale;
            }
        }
        /// <summary>
        /// Helper: Multiply 4x4 matrix by a 3D point (assumes w=1)
        /// </summary>
        private static double[] MultiplyMatrixVector(double[] matrix, double[] point)
        {
            double[] result = new double[4];

            result[0] = matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12];
            result[1] = matrix[1] * point[0] + matrix[5] * point[1] + matrix[9] * point[2] + matrix[13];
            result[2] = matrix[2] * point[0] + matrix[6] * point[1] + matrix[10] * point[2] + matrix[14];
            result[3] = matrix[3] * point[0] + matrix[7] * point[1] + matrix[11] * point[2] + matrix[15];
            return result;
        }
        /// <summary>
        /// Calculate cross product of two vectors
        /// </summary>
        private static double[] CrossProduct(double[] a, double[] b)
        {
            return new double[]
            {
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]
            };
        }
    }
}
