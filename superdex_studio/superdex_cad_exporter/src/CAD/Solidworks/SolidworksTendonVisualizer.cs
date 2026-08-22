/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System.Collections.Generic;

namespace CADRobotExporter.CAD
{
    public static class SolidworksTendonVisualizer
    {
        private const float COLOR_PURPLE_R = 0.6f;
        private const float COLOR_PURPLE_G = 0.0f;
        private const float COLOR_PURPLE_B = 0.8f;

        private const float COLOR_ORANGE_R = 1.0f;
        private const float COLOR_ORANGE_G = 0.5f;
        private const float COLOR_ORANGE_B = 0.0f;

        private const float HIGHLIGHT_SCALE = 1.5f;

        private static readonly float[][] Palette =
        {
            new[] { 0.6f, 0.0f, 0.8f },  // purple
            new[] { 0.0f, 0.8f, 0.2f },  // green
            new[] { 0.2f, 0.4f, 1.0f },  // blue
            new[] { 1.0f, 0.2f, 0.2f },  // red
            new[] { 0.0f, 0.8f, 0.8f },  // cyan
            new[] { 0.9f, 0.9f, 0.0f },  // yellow
            new[] { 1.0f, 0.4f, 0.7f },  // pink
            new[] { 1.0f, 0.5f, 0.0f },  // orange
        };

        public static float[] GetPaletteColor(int index)
        {
            return Palette[index % Palette.Length];
        }

        public static void DrawTendon(List<double[]> points, int highlightIndex, double crossSize)
        {
            DrawTendon(points, highlightIndex, crossSize, COLOR_PURPLE_R, COLOR_PURPLE_G, COLOR_PURPLE_B);
        }

        public static void DrawTendon(List<double[]> points, int highlightIndex, double crossSize, float r, float g, float b)
        {
            if (points == null || points.Count < 2)
                return;

            OpenGL.glDisable(OpenGL.GL_LIGHTING);
            OpenGL.glDisable(OpenGL.GL_DEPTH_TEST);

            // Draw lines connecting consecutive points
            OpenGL.glLineWidth(3.0f);
            OpenGL.glColor3f(r, g, b);
            OpenGL.glBegin(OpenGL.GL_LINE_STRIP);
            for (int i = 0; i < points.Count; i++)
            {
                OpenGL.glVertex3d(points[i][0], points[i][1], points[i][2]);
            }
            OpenGL.glEnd();

            // Draw cross markers at each point
            for (int i = 0; i < points.Count; i++)
            {
                if (i == highlightIndex)
                {
                    DrawCross3D(points[i], crossSize * HIGHLIGHT_SCALE,
                        COLOR_ORANGE_R, COLOR_ORANGE_G, COLOR_ORANGE_B);
                }
                else
                {
                    DrawCross3D(points[i], crossSize, r, g, b);
                }
            }

            OpenGL.glEnable(OpenGL.GL_DEPTH_TEST);
            OpenGL.glLineWidth(1.0f);
        }

        private static void DrawCross3D(double[] center, double size, float r, float g, float b)
        {
            OpenGL.glLineWidth(4.0f);
            OpenGL.glColor3f(r, g, b);
            OpenGL.glBegin(OpenGL.GL_LINES);

            // X arm
            OpenGL.glVertex3d(center[0] - size, center[1], center[2]);
            OpenGL.glVertex3d(center[0] + size, center[1], center[2]);

            // Y arm
            OpenGL.glVertex3d(center[0], center[1] - size, center[2]);
            OpenGL.glVertex3d(center[0], center[1] + size, center[2]);

            // Z arm
            OpenGL.glVertex3d(center[0], center[1], center[2] - size);
            OpenGL.glVertex3d(center[0], center[1], center[2] + size);

            OpenGL.glEnd();
        }
    }
}
