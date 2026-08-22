/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace CADRobotExporter.CAD
{
    public static class SolidworksInertialVisualizer
    {
        private const float COLOR_RED_R = 0.9f;
        private const float COLOR_RED_G = 0.1f;
        private const float COLOR_RED_B = 0.1f;

        private const float COLOR_GREEN_R = 0.1f;
        private const float COLOR_GREEN_G = 0.8f;
        private const float COLOR_GREEN_B = 0.2f;

        public static void DrawInertialGizmo(double[] comGlobal, double[,] rotation3x3, double[] boxHalfExtents, double crossSize)
        {
            OpenGL.glDisable(OpenGL.GL_LIGHTING);
            OpenGL.glDisable(OpenGL.GL_DEPTH_TEST);

            // Red cross at CoM
            DrawCross3D(comGlobal, crossSize, COLOR_RED_R, COLOR_RED_G, COLOR_RED_B);

            // Green wireframe box
            DrawWireframeBox(comGlobal, rotation3x3, boxHalfExtents);

            OpenGL.glEnable(OpenGL.GL_DEPTH_TEST);
        }

        private static void DrawCross3D(double[] center, double size, float r, float g, float b)
        {
            OpenGL.glLineWidth(5.0f);
            OpenGL.glColor3f(r, g, b);
            OpenGL.glBegin(OpenGL.GL_LINES);

            OpenGL.glVertex3d(center[0] - size, center[1], center[2]);
            OpenGL.glVertex3d(center[0] + size, center[1], center[2]);

            OpenGL.glVertex3d(center[0], center[1] - size, center[2]);
            OpenGL.glVertex3d(center[0], center[1] + size, center[2]);

            OpenGL.glVertex3d(center[0], center[1], center[2] - size);
            OpenGL.glVertex3d(center[0], center[1], center[2] + size);

            OpenGL.glEnd();
        }

        private static void DrawWireframeBox(double[] center, double[,] rot, double[] half)
        {
            // 8 corners of the box in local frame, transformed by rotation + center
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

            // 12 edges: corners indexed as (sx,sy,sz) → idx = (sx+1)/2*4 + (sy+1)/2*2 + (sz+1)/2
            // Edges along X (differ in sx): (0,1,2,3) paired with (4,5,6,7)
            // Edges along Y (differ in sy): (0,1) with (2,3), (4,5) with (6,7)
            // Edges along Z (differ in sz): (0) with (1), (2) with (3), (4) with (5), (6) with (7)
            int[,] edges = new int[,]
            {
                {0,4}, {1,5}, {2,6}, {3,7}, // along X
                {0,2}, {1,3}, {4,6}, {5,7}, // along Y
                {0,1}, {2,3}, {4,5}, {6,7}, // along Z
            };

            OpenGL.glLineWidth(2.0f);
            OpenGL.glColor3f(COLOR_GREEN_R, COLOR_GREEN_G, COLOR_GREEN_B);
            OpenGL.glBegin(OpenGL.GL_LINES);

            for (int i = 0; i < 12; i++)
            {
                int a = edges[i, 0];
                int b = edges[i, 1];
                OpenGL.glVertex3d(corners[a][0], corners[a][1], corners[a][2]);
                OpenGL.glVertex3d(corners[b][0], corners[b][1], corners[b][2]);
            }

            OpenGL.glEnd();
            OpenGL.glLineWidth(1.0f);
        }
    }
}
