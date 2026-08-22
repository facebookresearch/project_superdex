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
    public class NXInertialVisualizer
    {
        private List<NXObject> createdObjects = new List<NXObject>();

        private const int COLOR_RED = 186;
        private const int COLOR_GREEN = 36;

        public void DrawInertialGizmo(Part workPart, double[] comGlobal, double[,] rotation, double[] boxHalfExtents, double crossSize)
        {
            Clear(workPart);

            // Red cross at CoM
            DrawCross(workPart, comGlobal, crossSize, COLOR_RED);

            // Green wireframe box
            DrawWireframeBox(workPart, comGlobal, rotation, boxHalfExtents);
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

        private void DrawWireframeBox(Part workPart, double[] center, double[,] rot, double[] half)
        {
            double[][] corners = new double[8][];
            int idx = 0;
            for (int sx = -1; sx <= 1; sx += 2)
            {
                for (int sy = -1; sy <= 1; sy += 2)
                {
                    for (int sz = -1; sz <= 1; sz += 2)
                    {
                        double lx = sx * half[0];
                        double ly = sy * half[1];
                        double lz = sz * half[2];
                        corners[idx] = new double[]
                        {
                            center[0] + rot[0, 0] * lx + rot[0, 1] * ly + rot[0, 2] * lz,
                            center[1] + rot[1, 0] * lx + rot[1, 1] * ly + rot[1, 2] * lz,
                            center[2] + rot[2, 0] * lx + rot[2, 1] * ly + rot[2, 2] * lz,
                        };
                        idx++;
                    }
                }
            }

            int[,] edges = new int[,]
            {
                {0,4}, {1,5}, {2,6}, {3,7},
                {0,2}, {1,3}, {4,6}, {5,7},
                {0,1}, {2,3}, {4,5}, {6,7},
            };

            for (int i = 0; i < 12; i++)
            {
                int a = edges[i, 0];
                int b = edges[i, 1];
                Point3d from = new Point3d(corners[a][0], corners[a][1], corners[a][2]);
                Point3d to = new Point3d(corners[b][0], corners[b][1], corners[b][2]);

                Line line = workPart.Curves.CreateLine(from, to);
                line.Color = COLOR_GREEN;
                line.LineWidth = DisplayableObject.ObjectWidth.Normal;
                line.RedisplayObject();
                createdObjects.Add(line);
            }
        }
    }
}

#endif
