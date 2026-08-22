/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if NX

using NXOpen;
using System;
using System.Collections.Generic;

namespace CADRobotExporter.CAD.NX
{
    public class NXJointVisualizer
    {
        private List<NXObject> createdObjects = new List<NXObject>();

        private const int COLOR_BLUE = 211;
        private const int COLOR_RED = 186;
        private const int COLOR_GREEN = 36;

        private const double M_TO_MM = 1000;

        public void DrawRevoluteJoint(Part workPart, double[] center, double[] axis,
            double lowerLimit, double upperLimit, double radius, double[] referenceDirection = null)
        {
            Clear(workPart);

            double[] axisNorm = Normalize(axis);
            double[] perpVec1 = GetOrientedPerpendicular(axisNorm, referenceDirection);
            double[] perpVec2 = Cross(axisNorm, perpVec1);

            Point3d centerPt = new Point3d(center[0], center[1], center[2]);
            Vector3d xDir = new Vector3d(perpVec1[0], perpVec1[1], perpVec1[2]);
            Vector3d yDir = new Vector3d(perpVec2[0], perpVec2[1], perpVec2[2]);

            // Limit arc
            Arc arc = workPart.Curves.CreateArc(centerPt, xDir, yDir, radius, lowerLimit, upperLimit);
            arc.Color = COLOR_BLUE;
            arc.LineWidth = DisplayableObject.ObjectWidth.Thick;
            arc.RedisplayObject();
            createdObjects.Add(arc);

            // Lower limit line (red)
            Point3d lowerEnd = RadialPoint(center, perpVec1, perpVec2, lowerLimit, radius);
            Line lowerLine = workPart.Curves.CreateLine(centerPt, lowerEnd);
            lowerLine.Color = COLOR_RED;
            lowerLine.LineWidth = DisplayableObject.ObjectWidth.Thick;
            lowerLine.RedisplayObject();
            createdObjects.Add(lowerLine);

            // Upper limit line (green)
            Point3d upperEnd = RadialPoint(center, perpVec1, perpVec2, upperLimit, radius);
            Line upperLine = workPart.Curves.CreateLine(centerPt, upperEnd);
            upperLine.Color = COLOR_GREEN;
            upperLine.LineWidth = DisplayableObject.ObjectWidth.Thick;
            upperLine.RedisplayObject();
            createdObjects.Add(upperLine);

            // Center reference line (blue, shorter)
            Point3d centerEnd = RadialPoint(center, perpVec1, perpVec2, 0, radius * 0.66);
            Line centerLine = workPart.Curves.CreateLine(centerPt, centerEnd);
            centerLine.Color = COLOR_BLUE;
            centerLine.LineWidth = DisplayableObject.ObjectWidth.Thick;
            centerLine.RedisplayObject();
            createdObjects.Add(centerLine);

            // Axis arrow
            DrawAxisArrow(workPart, center, axisNorm, radius);
        }

        public void DrawPrismaticJoint(Part workPart, double[] center, double[] axis,
            double lowerLimit, double upperLimit, double radius)
        {
            Clear(workPart);

            double[] axisNorm = Normalize(axis);
            double[] perpVec1 = GetPerpendicular(axisNorm);
            double[] perpVec2 = Cross(axisNorm, perpVec1);

            Point3d centerPt = new Point3d(center[0], center[1], center[2]);
            double[] lowerPos = AxisPoint(center, axisNorm, lowerLimit * M_TO_MM);
            double[] upperPos = AxisPoint(center, axisNorm, upperLimit * M_TO_MM);
            Point3d lowerPt = new Point3d(lowerPos[0], lowerPos[1], lowerPos[2]);
            Point3d upperPt = new Point3d(upperPos[0], upperPos[1], upperPos[2]);

            // Axis line between limits
            Line axisLine = workPart.Curves.CreateLine(lowerPt, upperPt);
            axisLine.Color = COLOR_BLUE;
            axisLine.LineWidth = DisplayableObject.ObjectWidth.Thick;
            axisLine.RedisplayObject();
            createdObjects.Add(axisLine);

            // Circle at lower limit (red)
            DrawCircle(workPart, lowerPos, perpVec1, perpVec2, radius * 0.3, COLOR_RED);

            // Circle at upper limit (green)
            DrawCircle(workPart, upperPos, perpVec1, perpVec2, radius * 0.3, COLOR_GREEN);

            // Center marker (blue)
            DrawCircle(workPart, center, perpVec1, perpVec2, radius * 0.15, COLOR_BLUE);

            // Axis arrow
            DrawAxisArrow(workPart, center, axisNorm, radius);
        }

        public void Clear(Part workPart)
        {
            if (createdObjects.Count == 0)
                return;

            Session session = Session.GetSession();
            session.UpdateManager.ClearErrorList();
            NXObject[] toDelete = createdObjects.ToArray();
            createdObjects.Clear();

            try
            {
                session.UpdateManager.AddObjectsToDeleteList(toDelete);
                session.UpdateManager.DoUpdate(new Session.UndoMarkId());
            }
            catch
            {
            }
        }

        private void DrawAxisArrow(Part workPart, double[] center, double[] axisNorm, double length)
        {
            double[] tipPos = AxisPoint(center, axisNorm, length);
            Point3d basePt = new Point3d(center[0], center[1], center[2]);
            Point3d tipPt = new Point3d(tipPos[0], tipPos[1], tipPos[2]);

            Line axisLine = workPart.Curves.CreateLine(basePt, tipPt);
            axisLine.Color = COLOR_BLUE;
            axisLine.LineWidth = DisplayableObject.ObjectWidth.Thick;
            axisLine.RedisplayObject();
            createdObjects.Add(axisLine);

            // Arrowhead
            double[] perpVec1 = GetPerpendicular(axisNorm);
            double arrowSize = length * 0.15;
            double arrowBack = length * 0.85;
            double[] arrowBase = AxisPoint(center, axisNorm, arrowBack);

            Point3d arrow1 = new Point3d(
                arrowBase[0] + arrowSize * perpVec1[0],
                arrowBase[1] + arrowSize * perpVec1[1],
                arrowBase[2] + arrowSize * perpVec1[2]);
            Point3d arrow2 = new Point3d(
                arrowBase[0] - arrowSize * perpVec1[0],
                arrowBase[1] - arrowSize * perpVec1[1],
                arrowBase[2] - arrowSize * perpVec1[2]);

            Line arrowLine1 = workPart.Curves.CreateLine(tipPt, arrow1);
            arrowLine1.Color = COLOR_BLUE;
            arrowLine1.LineWidth = DisplayableObject.ObjectWidth.Thick;
            arrowLine1.RedisplayObject();
            createdObjects.Add(arrowLine1);

            Line arrowLine2 = workPart.Curves.CreateLine(tipPt, arrow2);
            arrowLine2.Color = COLOR_BLUE;
            arrowLine2.LineWidth = DisplayableObject.ObjectWidth.Thick;
            arrowLine2.RedisplayObject();
            createdObjects.Add(arrowLine2);
        }

        private void DrawCircle(Part workPart, double[] center, double[] perpVec1, double[] perpVec2, double radius, int color)
        {
            Point3d centerPt = new Point3d(center[0], center[1], center[2]);
            Vector3d xDir = new Vector3d(perpVec1[0], perpVec1[1], perpVec1[2]);
            Vector3d yDir = new Vector3d(perpVec2[0], perpVec2[1], perpVec2[2]);

            Arc circle = workPart.Curves.CreateArc(centerPt, xDir, yDir, radius, 0, 2 * Math.PI);
            circle.Color = color;
            circle.LineWidth = DisplayableObject.ObjectWidth.Thick;
            circle.RedisplayObject();
            createdObjects.Add(circle);
        }

        private static Point3d RadialPoint(double[] center, double[] perpVec1, double[] perpVec2, double angle, double radius)
        {
            double x = Math.Cos(angle);
            double y = Math.Sin(angle);
            return new Point3d(
                center[0] + radius * (x * perpVec1[0] + y * perpVec2[0]),
                center[1] + radius * (x * perpVec1[1] + y * perpVec2[1]),
                center[2] + radius * (x * perpVec1[2] + y * perpVec2[2]));
        }

        private static double[] AxisPoint(double[] center, double[] axis, double distance)
        {
            return new double[] {
                center[0] + axis[0] * distance,
                center[1] + axis[1] * distance,
                center[2] + axis[2] * distance
            };
        }

        private static double[] Normalize(double[] v)
        {
            double len = Math.Sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            if (len < 1e-10) return new double[] { 0, 0, 1 };
            return new double[] { v[0] / len, v[1] / len, v[2] / len };
        }

        private static double[] GetOrientedPerpendicular(double[] axis, double[] referenceDirection)
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
            return GetPerpendicular(axis);
        }

        private static double[] GetPerpendicular(double[] v)
        {
            double[] seed;
            if (Math.Abs(v[0]) < Math.Abs(v[1]) && Math.Abs(v[0]) < Math.Abs(v[2]))
                seed = new double[] { 1, 0, 0 };
            else if (Math.Abs(v[1]) < Math.Abs(v[2]))
                seed = new double[] { 0, 1, 0 };
            else
                seed = new double[] { 0, 0, 1 };
            return Normalize(Cross(v, seed));
        }

        private static double[] Cross(double[] a, double[] b)
        {
            return new double[] {
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]
            };
        }
    }
}

#endif
