/*
Copyright (c) 2015 Stephen Brawner
Copyright (c) Meta Platforms, Inc. and affiliates.

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

using System.IO;

using CADRobotExporter.UI;

namespace CADRobotExporter.RobotExport
{
    public class RobotPackage
    {
        public static IMessageBox MessageBox = new MessageBoxHelper();
        public string PackageName { get; }

        public string PackageDirectory { get; }
        public string MeshesDirectory { get; }
        public string CollisionMeshesDirectory { get; }
        public string BackupDirectory { get; }
        public string RobotsDirectory { get; }

        public string WindowsPackageDirectory { get; }
        public string WindowsMeshesDirectory { get; }
        public string WindowsCollisionMeshesDirectory { get; }
        public string WindowsCadDirectory { get; }
        public string WindowsBackupDirectory { get; }
        public string WindowsRobotsDirectory { get; }

        public string VisualMeshPostfix { get; }
        public string CollisionMeshPostfix { get; }
        public string StepVisualPostfix { get; }
        public string StepCollisionPostfix { get; }

        public RobotPackage(string name, string dir, FolderStructure folderStructure = FolderStructure.ROS)
        {
            PackageName = name;
            PackageDirectory = @"package://" + name + @"/";
            RobotsDirectory = PackageDirectory + @"urdf/";
            BackupDirectory = PackageDirectory + @"backup/";

            char last = string.IsNullOrEmpty(dir) ? '\0' : dir[dir.Length - 1];
            dir = (last == '\\') ? dir : dir + @"\";
            WindowsPackageDirectory = dir + name + @"\";
            WindowsRobotsDirectory = WindowsPackageDirectory + @"urdf\";
            WindowsBackupDirectory = WindowsPackageDirectory + @"backup\";

            switch (folderStructure)
            {
                case FolderStructure.SuperDex:
                    MeshesDirectory = PackageDirectory + @"render/";
                    CollisionMeshesDirectory = PackageDirectory + @"collision/";
                    WindowsMeshesDirectory = WindowsPackageDirectory + @"render\";
                    WindowsCollisionMeshesDirectory = WindowsPackageDirectory + @"collision\";
                    WindowsCadDirectory = WindowsPackageDirectory + @"cad\";
                    VisualMeshPostfix = "_render";
                    CollisionMeshPostfix = "_collision";
                    StepVisualPostfix = "_render";
                    StepCollisionPostfix = "_collision";
                    break;
                case FolderStructure.MuJoCo:
                    MeshesDirectory = PackageDirectory + @"assets/";
                    CollisionMeshesDirectory = PackageDirectory + @"assets/";
                    WindowsMeshesDirectory = WindowsPackageDirectory + @"assets\";
                    WindowsCollisionMeshesDirectory = WindowsPackageDirectory + @"assets\";
                    WindowsCadDirectory = null;
                    VisualMeshPostfix = "_visual";
                    CollisionMeshPostfix = "_collision";
                    StepVisualPostfix = null;
                    StepCollisionPostfix = null;
                    break;
                case FolderStructure.Legacy:
                    MeshesDirectory = PackageDirectory + @"meshes/";
                    CollisionMeshesDirectory = PackageDirectory + @"meshes/collision/";
                    WindowsMeshesDirectory = WindowsPackageDirectory + @"meshes\";
                    WindowsCollisionMeshesDirectory = WindowsPackageDirectory + @"meshes\collision\";
                    WindowsCadDirectory = null;
                    VisualMeshPostfix = "";
                    CollisionMeshPostfix = "_collision";
                    StepVisualPostfix = null;
                    StepCollisionPostfix = null;
                    break;
                case FolderStructure.ROS:
                default:
                    MeshesDirectory = PackageDirectory + @"meshes/visual/";
                    CollisionMeshesDirectory = PackageDirectory + @"meshes/collision/";
                    WindowsMeshesDirectory = WindowsPackageDirectory + @"meshes\visual\";
                    WindowsCollisionMeshesDirectory = WindowsPackageDirectory + @"meshes\collision\";
                    WindowsCadDirectory = null;
                    VisualMeshPostfix = "_visual";
                    CollisionMeshPostfix = "_collision";
                    StepVisualPostfix = null;
                    StepCollisionPostfix = null;
                    break;
            }
        }

        public void CreateDirectories()
        {
            if (!Directory.Exists(WindowsPackageDirectory))
            {
                Directory.CreateDirectory(WindowsPackageDirectory);
            }
            if (!Directory.Exists(WindowsMeshesDirectory))
            {
                Directory.CreateDirectory(WindowsMeshesDirectory);
            }
            if (!Directory.Exists(WindowsCollisionMeshesDirectory))
            {
                Directory.CreateDirectory(WindowsCollisionMeshesDirectory);
            }
            if (WindowsCadDirectory != null && !Directory.Exists(WindowsCadDirectory))
            {
                Directory.CreateDirectory(WindowsCadDirectory);
            }
            if (!Directory.Exists(WindowsRobotsDirectory))
            {
                Directory.CreateDirectory(WindowsRobotsDirectory);
            }
            if (!Directory.Exists(WindowsBackupDirectory))
            {
                Directory.CreateDirectory(WindowsBackupDirectory);
            }
        }
    }
}
