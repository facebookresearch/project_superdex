/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System.Runtime.InteropServices;

namespace CADRobotExporter.CAD
{
    public static class OpenGL
    {
        private const string OPENGL_DLL = "opengl32.dll";

#pragma warning disable CA1707 // disabled to keep GL_FOO_BAR parity between C#/C
        // Constants
        public const uint GL_POINTS = 0x0000;
        public const uint GL_LINES = 0x0001;
        public const uint GL_LINE_LOOP = 0x0002;
        public const uint GL_LINE_STRIP = 0x0003;
        public const uint GL_TRIANGLES = 0x0004;
        public const uint GL_TRIANGLE_STRIP = 0x0005;
        public const uint GL_TRIANGLE_FAN = 0x0006;
        public const uint GL_QUADS = 0x0007;
        public const uint GL_QUAD_STRIP = 0x0008;
        public const uint GL_POLYGON = 0x0009;

        public const uint GL_MODELVIEW = 0x1700;
        public const uint GL_PROJECTION = 0x1701;
        public const uint GL_TEXTURE = 0x1702;

        public const uint GL_COLOR_BUFFER_BIT = 0x4000;
        public const uint GL_DEPTH_BUFFER_BIT = 0x0100;
        public const uint GL_STENCIL_BUFFER_BIT = 0x0400;

        public const uint GL_DEPTH_TEST = 0x0B71;
        public const uint GL_LIGHTING = 0x0B50;
        public const uint GL_LIGHT0 = 0x4000;
        public const uint GL_LIGHT1 = 0x4001;
        public const uint GL_LIGHT2 = 0x4002;
        public const uint GL_LIGHT3 = 0x4003;
        public const uint GL_LIGHT4 = 0x4004;
        public const uint GL_LIGHT5 = 0x4005;
        public const uint GL_LIGHT6 = 0x4006;
        public const uint GL_LIGHT7 = 0x4007;

        public const uint GL_BLEND = 0x0BE2;
        public const uint GL_CULL_FACE = 0x0B44;
        public const uint GL_TEXTURE_2D = 0x0DE1;
        public const uint GL_NORMALIZE = 0x0BA1;

        public const uint GL_FRONT = 0x0404;
        public const uint GL_BACK = 0x0405;
        public const uint GL_FRONT_AND_BACK = 0x0408;

        public const uint GL_AMBIENT = 0x1200;
        public const uint GL_DIFFUSE = 0x1201;
        public const uint GL_SPECULAR = 0x1202;
        public const uint GL_POSITION = 0x1203;
        public const uint GL_SHININESS = 0x1601;
        public const uint GL_EMISSION = 0x1600;

        // Blend Function Constants
        public const uint GL_ZERO = 0;
        public const uint GL_ONE = 1;
        public const uint GL_SRC_COLOR = 0x0300;
        public const uint GL_ONE_MINUS_SRC_COLOR = 0x0301;
        public const uint GL_SRC_ALPHA = 0x0302;
        public const uint GL_ONE_MINUS_SRC_ALPHA = 0x0303;
        public const uint GL_DST_ALPHA = 0x0304;
        public const uint GL_ONE_MINUS_DST_ALPHA = 0x0305;
        public const uint GL_DST_COLOR = 0x0306;
        public const uint GL_ONE_MINUS_DST_COLOR = 0x0307;
        public const uint GL_SRC_ALPHA_SATURATE = 0x0308;

        // Matrix/state constants
        public const uint GL_MODELVIEW_MATRIX = 0x0BA6;
        public const uint GL_PROJECTION_MATRIX = 0x0BA7;
        public const uint GL_VIEWPORT = 0x0BA2;
#pragma warning restore CA1707

#pragma warning disable CA1401
        // Drawing Functions
        [DllImport(OPENGL_DLL)]
        public static extern void glBegin(uint mode);

        [DllImport(OPENGL_DLL)]
        public static extern void glEnd();

        [DllImport(OPENGL_DLL)]
        public static extern void glVertex2f(float x, float y);

        [DllImport(OPENGL_DLL)]
        public static extern void glVertex3f(float x, float y, float z);

        [DllImport(OPENGL_DLL)]
        public static extern void glVertex3d(double x, double y, double z);

        [DllImport(OPENGL_DLL)]
        public static extern void glVertex3fv(float[] v);

        [DllImport(OPENGL_DLL)]
        public static extern void glNormal3f(float nx, float ny, float nz);

        [DllImport(OPENGL_DLL)]
        public static extern void glNormal3fv(float[] v);

        // Color Functions
        [DllImport(OPENGL_DLL)]
        public static extern void glColor3f(float red, float green, float blue);

        [DllImport(OPENGL_DLL)]
        public static extern void glColor4f(float red, float green, float blue, float alpha);

        [DllImport(OPENGL_DLL)]
        public static extern void glColor3fv(float[] v);

        [DllImport(OPENGL_DLL)]
        public static extern void glColor4fv(float[] v);

        // Matrix Functions
        [DllImport(OPENGL_DLL)]
        public static extern void glMatrixMode(uint mode);

        [DllImport(OPENGL_DLL)]
        public static extern void glLoadIdentity();

        [DllImport(OPENGL_DLL)]
        public static extern void glLoadMatrixf(float[] m);

        [DllImport(OPENGL_DLL)]
        public static extern void glLoadMatrixd(double[] m);

        [DllImport(OPENGL_DLL)]
        public static extern void glMultMatrixf(float[] m);

        [DllImport(OPENGL_DLL)]
        public static extern void glMultMatrixd(double[] m);

        [DllImport(OPENGL_DLL)]
        public static extern void glPushMatrix();

        [DllImport(OPENGL_DLL)]
        public static extern void glPopMatrix();

        // Transformation Functions
        [DllImport(OPENGL_DLL)]
        public static extern void glTranslatef(float x, float y, float z);

        [DllImport(OPENGL_DLL)]
        public static extern void glTranslated(double x, double y, double z);

        [DllImport(OPENGL_DLL)]
        public static extern void glRotatef(float angle, float x, float y, float z);

        [DllImport(OPENGL_DLL)]
        public static extern void glRotated(double angle, double x, double y, double z);

        [DllImport(OPENGL_DLL)]
        public static extern void glScalef(float x, float y, float z);

        [DllImport(OPENGL_DLL)]
        public static extern void glScaled(double x, double y, double z);

        // Buffer Functions
        [DllImport(OPENGL_DLL)]
        public static extern void glClear(uint mask);

        [DllImport(OPENGL_DLL)]
        public static extern void glClearColor(float red, float green, float blue, float alpha);

        [DllImport(OPENGL_DLL)]
        public static extern void glClearDepth(double depth);

        [DllImport(OPENGL_DLL)]
        public static extern void glFlush();

        [DllImport(OPENGL_DLL)]
        public static extern void glFinish();

        // Viewport and Perspective
        [DllImport(OPENGL_DLL)]
        public static extern void glViewport(int x, int y, int width, int height);

        [DllImport(OPENGL_DLL)]
        public static extern void glOrtho(double left, double right, double bottom, double top, double nearVal, double farVal);

        [DllImport(OPENGL_DLL)]
        public static extern void glFrustum(double left, double right, double bottom, double top, double nearVal, double farVal);

        // State Management
        [DllImport(OPENGL_DLL)]
        public static extern void glEnable(uint cap);

        [DllImport(OPENGL_DLL)]
        public static extern void glDisable(uint cap);

        [DllImport(OPENGL_DLL)]
        public static extern void glShadeModel(uint mode);

        [DllImport(OPENGL_DLL)]
        public static extern void glCullFace(uint mode);

        [DllImport(OPENGL_DLL)]
        public static extern void glFrontFace(uint mode);

        // Lighting
        [DllImport(OPENGL_DLL)]
        public static extern void glLightfv(uint light, uint pname, float[] parameters);

        [DllImport(OPENGL_DLL)]
        public static extern void glMaterialfv(uint face, uint pname, float[] parameters);

        [DllImport(OPENGL_DLL)]
        public static extern void glMaterialf(uint face, uint pname, float param);

        // Texture Coordinates
        [DllImport(OPENGL_DLL)]
        public static extern void glTexCoord2f(float s, float t);

        [DllImport(OPENGL_DLL)]
        public static extern void glTexCoord2fv(float[] v);

        // Get Functions
        [DllImport(OPENGL_DLL)]
        public static extern void glGetFloatv(uint pname, float[] parameters);

        [DllImport(OPENGL_DLL)]
        public static extern void glGetIntegerv(uint pname, int[] parameters);

        [DllImport(OPENGL_DLL)]
        public static extern uint glGetError();

        // Point and Line Width
        [DllImport(OPENGL_DLL)]
        public static extern void glPointSize(float size);

        [DllImport(OPENGL_DLL)]
        public static extern void glLineWidth(float width);

        // Blend Functions
        [DllImport(OPENGL_DLL)]
        public static extern void glBlendFunc(uint sfactor, uint dfactor);

        [DllImport(OPENGL_DLL)]
        public static extern void glGetDoublev(uint pname, double[] parameters);

#pragma warning restore CA1401
    }
}
