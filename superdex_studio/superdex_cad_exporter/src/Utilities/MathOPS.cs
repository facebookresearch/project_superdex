/*
Copyright (c) 2015 Stephen Brawner

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

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace CADRobotExporter.Utilities
{
    public struct DecomposedTransform
    {
        public Vector3 Rotation;    // Euler angles in radians (X, Y, Z)
        public Vector3 Translation; // Translation vector
    }

    public struct Vector3
    {
        public double X, Y, Z;

        public Vector3(double x, double y, double z)
        {
            X = x; Y = y; Z = z;
        }
    }

    public static class MathOps
    {
        public static double epsilon = 1e-15;

        public static T Max<T>(T d1, T d2, T d3) where T : IComparable<T>
        {
            return Max(new T[] { d1, d2, d3 });
        }

        public static T Max<T>(T[] array) where T : IComparable<T>
        {
            T result = default;
            if (array.Length > 0)
            {
                result = array[0];
                foreach (T t in array)
                {
                    result = Comparer<T>.Default.Compare(t, result) > 0 ? t : result;
                }
            }
            return result;
        }

        public static T Min<T>(T d1, T d2, T d3) where T : IComparable<T>
        {
            return Min(new T[] { d1, d2, d3 });
        }

        public static T Min<T>(T[] array) where T : IComparable<T>
        {
            T result = default;
            if (array.Length > 0)
            {
                result = array[0];
                foreach (T t in array)
                {
                    result = Comparer<T>.Default.Compare(t, result) < 0 ? t : result;
                }
            }
            return result;
        }

        public static T Envelope<T>(T value, T min, T max) where T : IComparable<T>
        {
            if (Comparer<T>.Default.Compare(value, max) > 0)
            {
                return max;
            }
            else if (Comparer<T>.Default.Compare(value, min) < 0)
            {
                return min;
            }
            else
            {
                return value;
            }
        }

        public static double[] ClosestPointOnLineToPoint(double[] point, double[] line, double[] pointOnLine)
        {
            if (point.Length != line.Length || point.Length != pointOnLine.Length)
            {
                throw new Exception("Points and line vectors are not the same length");
            }

            double denominator = 0;
            double numerator = 0;
            for (int i = 0; i < point.Length; i++)
            {
                denominator += line[i] * line[i];
                numerator += line[i] * (point[i] - pointOnLine[i]);
            }
            double k = numerator / denominator;
            double[] result = new double[point.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = pointOnLine[i] + k * line[i];
            }
            return result;
        }

        public static double[] ClosestPointOnLineWithinBox(
            double xMin, double xMax, double yMin, double yMax, double zMin, double zMax,
            double[] line, double[] pointOnLine)
        {
            if (pointOnLine[0] > xMin &&
                pointOnLine[0] < xMax &&
                pointOnLine[1] > yMin &&
                pointOnLine[1] < yMax &&
                pointOnLine[2] > zMin &&
                pointOnLine[2] < zMax)
            {
                return pointOnLine;
            }
            double[] point1 =
                ClosestPointOnLineToPoint(new double[] { xMax, yMax, zMax }, line, pointOnLine);
            double[] point2 =
                ClosestPointOnLineToPoint(new double[] { xMin, yMin, zMin }, line, pointOnLine);

            if (Distance2(pointOnLine, point1) < Distance2(pointOnLine, point2))
            {
                return point1;
            }
            else
            {
                return point2;
            }
        }

        public static double[] GetXYZ(Matrix<double> m)
        {
            double[] XYZ = new double[3];
            XYZ[0] = m[0, 3]; XYZ[1] = m[1, 3]; XYZ[2] = m[2, 3];
            return XYZ;
        }

        public static double[] GetRPY(Matrix<double> m)
        {
            double roll, pitch, yaw;
            if (Math.Abs(m[2, 0]) >= 1.0)
            {
                // Gimbal Lock
                pitch = -Math.Asin(Math.Sign(m[2, 0]) * 1.0);
                roll = Math.Atan2(-m[1, 2], m[1, 1]);
                yaw = 0;
            }
            else
            {
                pitch = -Math.Asin(m[2, 0]);
                roll = Math.Atan2(m[2, 1], m[2, 2]);
                yaw = Math.Atan2(m[1, 0], m[0, 0]);
            }

            return new double[] { roll, pitch, yaw };
        }

        public static Matrix<double> GetRotation(double[] RPY)
        {
            Matrix<double> RX = DenseMatrix.CreateIdentity(4);
            Matrix<double> RY = DenseMatrix.CreateIdentity(4);
            Matrix<double> RZ = DenseMatrix.CreateIdentity(4);

            RX[1, 1] = Math.Cos(RPY[0]);
            RX[1, 2] = -Math.Sin(RPY[0]);
            RX[2, 1] = Math.Sin(RPY[0]);
            RX[2, 2] = Math.Cos(RPY[0]);

            RY[0, 0] = Math.Cos(RPY[1]);
            RY[0, 2] = Math.Sin(RPY[1]);
            RY[2, 0] = -Math.Sin(RPY[1]);
            RY[2, 2] = Math.Cos(RPY[1]);

            RZ[0, 0] = Math.Cos(RPY[2]);
            RZ[0, 1] = -Math.Sin(RPY[2]);
            RZ[1, 0] = Math.Sin(RPY[2]);
            RZ[1, 1] = Math.Cos(RPY[2]);

            return RZ * RY * RX;
        }

        public static Matrix<double> GetTranslation(double[] XYZ)
        {
            Matrix<double> m = DenseMatrix.CreateIdentity(4);
            m[0, 3] = XYZ[0]; m[1, 3] = XYZ[1]; m[2, 3] = XYZ[2];
            return m;
        }

        public static Matrix<double> GetTransformation(double[] XYZ, double[] RPY)
        {
            Matrix<double> translation = GetTranslation(XYZ);
            Matrix<double> rotation = GetRotation(RPY);
            return translation * rotation;
        }

        public static double[] PNorm(double[] array, double power)
        {
            double magnitude = 0;
            for (int i = 0; i < array.Length; i++)
            {
                magnitude += Math.Pow(array[i], power);
            }
            if (magnitude != 0)
            {
                magnitude = Math.Pow(magnitude, 1 / power);
                for (int i = 0; i < array.Length; i++)
                {
                    array[i] /= magnitude;
                }
            }
            return array;
        }

        public static double[] Negate(double[] array)
        {
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = -array[i];
            }
            return array;
        }

        public static double Distance2(double[] array1, double[] array2)
        {
            double sqrdmag = 0;
            for (int i = 0; i < array1.Length; i++)
            {
                double d = array1[i] - array2[i];
                sqrdmag += d * d;
            }
            return sqrdmag;
        }

        public static double[] Threshold(double[] array, double minValue)
        {
            double[] result = (double[])array.Clone();
            for (int i = 0; i < array.Length; i++)
            {
                result[i] = (Math.Abs(array[i]) >= minValue) ? array[i] : 0;
            }
            return result;
        }

        // Originally generated by MATLAB Coder from "rotm2eul(foo, "XYZ")" and converted to C#
        public static DecomposedTransform DecomposeTransformationMatrixToXYZEuler(double[] matrix)
        {
            if (matrix == null || matrix.Length != 16)
                throw new ArgumentException("Matrix must be an array of 16 elements");
            var result = new DecomposedTransform();
            // Extract translation (elements 9-11)
            result.Translation = new Vector3(matrix[9], matrix[10], matrix[11]);
            // Extract rotation matrix (elements 0-8)
            // Apply MATLAB's rotm2eul algorithm
            double[] eul = new double[3];

            double cySq = matrix[8] * matrix[8] + matrix[7] * matrix[7];
            double cy = Math.Sqrt(cySq);

            eul[0] = Math.Atan2(matrix[3], matrix[0]);
            eul[1] = Math.Atan2(-matrix[6], cy);
            eul[2] = Math.Atan2(matrix[7], matrix[8]);

            // Handle gimbal lock
            if (cySq < 2.2204460492503131E-15)
            {
                eul[0] = Math.Atan2(-matrix[1], matrix[4]);
                eul[1] = Math.Atan2(-matrix[6], cy);
                eul[2] = 0.0;
            }

            // Negate all angles
            eul[0] = -eul[0];
            eul[1] = -eul[1];
            eul[2] = -eul[2];

            // Swap eul[0] and eul[2]
            double temp = eul[0];
            eul[0] = eul[2];
            eul[2] = temp;

            result.Rotation = new Vector3(eul[0], eul[1], eul[2]);
            return result;
        }

        public static bool ComputePrincipalInertia(
            double mass, double Ixx, double Iyy, double Izz,
            double Ixy, double Ixz, double Iyz,
            out double[] boxHalfExtents, out double[,] principalRotation)
        {
            boxHalfExtents = new double[3];
            principalRotation = new double[3, 3] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };

            if (mass <= 1e-10)
                return false;

            var inertiaMatrix = Matrix<double>.Build.DenseOfArray(new double[,]
            {
                { Ixx, Ixy, Ixz },
                { Ixy, Iyy, Iyz },
                { Ixz, Iyz, Izz }
            });

            var evd = inertiaMatrix.Evd();
            var eigenvalues = evd.EigenValues;
            var eigenvectors = evd.EigenVectors;

            double l1 = eigenvalues[0].Real;
            double l2 = eigenvalues[1].Real;
            double l3 = eigenvalues[2].Real;

            // Equivalent box: a² = 6(l2+l3-l1)/m, etc.
            double a2 = 6.0 * (l2 + l3 - l1) / mass;
            double b2 = 6.0 * (l1 + l3 - l2) / mass;
            double c2 = 6.0 * (l1 + l2 - l3) / mass;

            const double MIN_DIM = 1e-6;
            boxHalfExtents[0] = Math.Sqrt(Math.Max(a2, MIN_DIM)) / 2.0;
            boxHalfExtents[1] = Math.Sqrt(Math.Max(b2, MIN_DIM)) / 2.0;
            boxHalfExtents[2] = Math.Sqrt(Math.Max(c2, MIN_DIM)) / 2.0;

            // Ensure right-handed rotation (det > 0)
            double det = eigenvectors.Determinant();
            for (int r = 0; r < 3; r++)
            {
                for (int c = 0; c < 3; c++)
                {
                    principalRotation[r, c] = eigenvectors[r, c];
                }
            }

            if (det < 0)
            {
                for (int r = 0; r < 3; r++)
                    principalRotation[r, 2] = -principalRotation[r, 2];
            }

            return true;
        }

        public static double[,] ComposeRotation(double[,] rCsys, double[] rpy, double[,] rPrincipal)
        {
            double[,] rRpy = RpyToRotationMatrix(rpy[0], rpy[1], rpy[2]);
            double[,] rCsysRpy = Multiply3x3(rCsys, rRpy);
            return Multiply3x3(rCsysRpy, rPrincipal);
        }

        private static double[,] RpyToRotationMatrix(double roll, double pitch, double yaw)
        {
            double cr = Math.Cos(roll), sr = Math.Sin(roll);
            double cp = Math.Cos(pitch), sp = Math.Sin(pitch);
            double cy = Math.Cos(yaw), sy = Math.Sin(yaw);

            return new double[,]
            {
                { cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr },
                { sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr },
                { -sp,   cp*sr,            cp*cr }
            };
        }

        private static double[,] Multiply3x3(double[,] a, double[,] b)
        {
            double[,] result = new double[3, 3];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    result[i, j] = a[i, 0] * b[0, j] + a[i, 1] * b[1, j] + a[i, 2] * b[2, j];
            return result;
        }
    }
}
