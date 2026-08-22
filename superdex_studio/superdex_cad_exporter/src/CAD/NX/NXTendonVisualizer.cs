/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if NX

using NXOpen;
using System.Collections.Generic;

namespace CADRobotExporter.CAD.NX
{
    public class NXTendonVisualizer
    {
        private List<NXObject> createdObjects = new List<NXObject>();

        private const int COLOR_ORANGE = 114;
        private const int COLOR_PURPLE = 199;

        private static readonly int[] Palette = { 199, 36, 186, 6, 211, 41, 169, 114 };

        private const double CROSS_SIZE = 3.0; // mm
        private const double CROSS_SIZE_HIGHLIGHT = 4.0; // mm

        public static int GetPaletteColor(int index)
        {
            return Palette[index % Palette.Length];
        }

        public void DrawTendon(Part workPart, List<double[]> points, int highlightIndex)
        {
            DrawTendon(workPart, points, highlightIndex, COLOR_PURPLE);
        }

        public void DrawTendon(Part workPart, List<double[]> points, int highlightIndex, int color)
        {
            if (points == null || points.Count < 2)
                return;

            // Draw lines connecting consecutive points
            for (int i = 0; i < points.Count - 1; i++)
            {
                Point3d from = new Point3d(points[i][0], points[i][1], points[i][2]);
                Point3d to = new Point3d(points[i + 1][0], points[i + 1][1], points[i + 1][2]);

                Line line = workPart.Curves.CreateLine(from, to);
                line.Color = color;
                line.LineWidth = DisplayableObject.ObjectWidth.Thick;
                line.RedisplayObject();
                createdObjects.Add(line);
            }

            // Draw cross markers at each point
            for (int i = 0; i < points.Count; i++)
            {
                double size = (i == highlightIndex) ? CROSS_SIZE_HIGHLIGHT : CROSS_SIZE;
                int markerColor = (i == highlightIndex) ? COLOR_ORANGE : color;
                DrawCross(workPart, points[i], size, markerColor);
            }
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

        private void DrawCross(Part workPart, double[] center, double size, int color)
        {
            // Draw a 3D cross (+) at center along X, Y, Z
            DrawCrossArm(workPart, center, new double[] { 1, 0, 0 }, size, color);
            DrawCrossArm(workPart, center, new double[] { 0, 1, 0 }, size, color);
            DrawCrossArm(workPart, center, new double[] { 0, 0, 1 }, size, color);
        }

        private void DrawCrossArm(Part workPart, double[] center, double[] direction, double size, int color)
        {
            Point3d from = new Point3d(
                center[0] - direction[0] * size,
                center[1] - direction[1] * size,
                center[2] - direction[2] * size);
            Point3d to = new Point3d(
                center[0] + direction[0] * size,
                center[1] + direction[1] * size,
                center[2] + direction[2] * size);

            Line line = workPart.Curves.CreateLine(from, to);
            line.Color = color;
            line.LineWidth = DisplayableObject.ObjectWidth.Thick;
            line.RedisplayObject();
            createdObjects.Add(line);
        }
    }
}

#endif
