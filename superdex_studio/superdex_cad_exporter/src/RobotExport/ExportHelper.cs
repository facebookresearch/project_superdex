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

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Windows;
using System.Xml;
using System.Xml.Linq;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

using Meshing;

using CADRobotExporter.CAD;
using CADRobotExporter.Export;
using CADRobotExporter.RobotDescription;
using CADRobotExporter.UI;
using CADRobotExporter.Utilities;

namespace CADRobotExporter.RobotExport
{
    // This class contains a long list of methods that are used throughout the export process.
    // Methods for building links and joints are contained in here.
    // Many of the methods are overloaded, but seek to reduce repeated code as much as possible
    // (i.e. the overloaded methods call eachother).
    // These methods are used by the PartExportForm, the AssemblyExportForm and the PropertyManager Page
    public partial class ExportHelper
    {
        private const string SaveStpStep = "SaveSTEP";

        private const string SaveGlbCadStep = "SaveGlbCad";
        private const string SaveStlCadStep = "SaveStlCad";
        private const string SaveObjCadStep = "SaveObjCad";

        private const string SaveStlOccStep = "SaveSTLSuperDex";
        private const string SaveObjOccStep = "SaveOBJSuperDex";
        private const string SaveGlbOccStep = "SaveGLBSuperDex";

        private const string SaveStpCollisionStep = "SaveSTEPCollision";

        private const string SaveGlbCadCollisionStep = "SaveGlbCollisionCad";
        private const string SaveStlCadCollisionStep = "SaveStlCadCollision";
        private const string SaveObjCadCollisionStep = "SaveObjCollisionCad";

        private const string SaveStlOccCollisionStep = "SaveSTLSuperDexCollision";
        private const string SaveObjOccCollisionStep = "SaveOBJSuperDexCollision";
        private const string SaveGlbOccCollisionStep = "SaveGLBSuperDexCollision";

        private static Serilog.ILogger logger = Logger.GetLogger();

        public Robot Robot { get; set; }

        public string PackageName { get; set; }

        public string SavePath { get; set; }

        public string URDFFileName { get; private set; }

        public string MJCFFileName { get; private set; }

        public string SuperDexBotFileName { get; private set; }

        public CADBridge CadBridge { get; private set; }

        private ProgressIndicatorManager _progressManager;

        private string _exportErrorWhy;

        // Constructor for SW2URDF Exporter class
        public ExportHelper(CADBridge cadBridge)
        {
            // todo: null check
            CadBridge = cadBridge;

            SavePath = System.Environment.ExpandEnvironmentVariables("%HOMEDRIVE%%HOMEPATH%");
            PackageName = CadBridge.GetDocumentTitle();

            CadBridge.IsShowingJointGizmo = false;
            CadBridge.CurrentNodeShown = null;
            CadBridge.JointGizmoScale = 0.5;
        }

        private static string ConvertMeshFilename(string original)
        {
            var match = Regex.Match(original, @"package://[^/]+/(.+)$");

            if (match.Success)
            {
                var relativePath = match.Groups[1].Value;
                return $"../{relativePath}";
            }

            // Return original if pattern doesn't match
            return original;
        }

        // Beginning method for exporting the full package
        public void ExportRobot(bool exportLinkMesh, List<MeshFormat> meshFormats, List<MeshFormat> collisionMeshFormats, ExporterMeshingOptions meshingOptions, ExporterConfiguration exporterConfig)
        {
            var allLinks = GetAllLinkDescendants(Robot.BaseLink);

            _progressManager = new ProgressIndicatorManager();
            _progressManager.Show();

            _progressManager.AddStep(Robot.BaseLink.Name, Robot.BaseLink.Name);
            foreach (var link in allLinks)
            {
                _progressManager.AddStep(link.Name, link.Name);
            }

            bool hasOpenCascade = meshFormats.Contains(MeshFormat.stlOpenCascade) || meshFormats.Contains(MeshFormat.objOpenCascade) || meshFormats.Contains(MeshFormat.glbOpenCascade);
            bool hasSTEP = meshFormats.Contains(MeshFormat.stepSolidworks);

            if (hasSTEP)
            {
                _progressManager.AddSubStep(SaveStpStep, "Export STEP file");
            }

            if (hasOpenCascade && !hasSTEP)
            {
                _progressManager.AddSubStep(SaveStpStep, "Export temporary STEP file");
            }

            foreach (var meshFormat in meshFormats)
            {
                switch (meshFormat)
                {
                    case MeshFormat.stlSolidworks:
                        _progressManager.AddSubStep(SaveStlCadStep, "Export CAD Native STL");
                        break;
                    case MeshFormat.glbCAD:
                        _progressManager.AddSubStep(SaveGlbCadStep, "Export CAD Native GLB");
                        break;
                    case MeshFormat.objCAD:
                        _progressManager.AddSubStep(SaveObjCadStep, "Export CAD Native OBJ");
                        break;
                    case MeshFormat.stlOpenCascade:
                        _progressManager.AddSubStep(SaveStlOccStep, "Export SuperDex STL");
                        break;
                    case MeshFormat.objOpenCascade:
                        _progressManager.AddSubStep(SaveObjOccStep, "Export SuperDex OBJ");
                        break;
                    case MeshFormat.stepSolidworks:
                        continue;
                    case MeshFormat.glbOpenCascade:
                        _progressManager.AddSubStep(SaveGlbOccStep, "Export SuperDex GLTF");
                        break;
                    default:
                        break;
                }
            }

            if (meshingOptions.exportCollision)
            {
                bool hasCollisionOpenCascade = collisionMeshFormats.Contains(MeshFormat.stlOpenCascade) || collisionMeshFormats.Contains(MeshFormat.objOpenCascade) || collisionMeshFormats.Contains(MeshFormat.glbOpenCascade);
                bool hasCollisionSTEP = collisionMeshFormats.Contains(MeshFormat.stepSolidworks);

                if (hasCollisionSTEP)
                {
                    _progressManager.AddSubStep(SaveStpCollisionStep, "Export STEP file (collision)");
                }

                if (hasCollisionOpenCascade && !hasCollisionSTEP)
                {
                    _progressManager.AddSubStep(SaveStpCollisionStep, "Export temporary STEP file (collision)");
                }

                foreach (var meshFormat in collisionMeshFormats)
                {
                    switch (meshFormat)
                    {
                        case MeshFormat.stlSolidworks:
                            _progressManager.AddSubStep(SaveStlCadCollisionStep, "Export CAD Native STL (collision)");
                            break;
                        case MeshFormat.glbCAD:
                            _progressManager.AddSubStep(SaveGlbCadCollisionStep, "Export CAD Native GLB (collision)");
                            break;
                        case MeshFormat.objCAD:
                            _progressManager.AddSubStep(SaveObjCadCollisionStep, "Export CAD Native OBJ (collision)");
                            break;
                        case MeshFormat.stlOpenCascade:
                            _progressManager.AddSubStep(SaveStlOccCollisionStep, "Export SuperDex STL (collision)");
                            break;
                        case MeshFormat.objOpenCascade:
                            _progressManager.AddSubStep(SaveObjOccCollisionStep, "Export SuperDex OBJ (collision)");
                            break;
                        case MeshFormat.stepSolidworks:
                            continue;
                        case MeshFormat.glbOpenCascade:
                            _progressManager.AddSubStep(SaveGlbOccCollisionStep, "Export SuperDex GLTF (collision)");
                            break;
                        default:
                            break;
                    }
                }
            }

            //Setting up the progress bar
            logger.Information("Beginning the export process");
            int progressBarBound = RobotDescription.Utilities.GetCount(Robot.BaseLink);
            CadBridge.SetProgressBarStart(progressBarBound, "Creating package directories");

            //Creating package directories
            logger.Information("Creating package directories with name " + PackageName + " and save path " + SavePath);
            RobotPackage package = new RobotPackage(PackageName, SavePath, exporterConfig.folderStructure);
            package.CreateDirectories();
            string windowsURDFFileName = package.WindowsRobotsDirectory + Robot.Name + ".urdf";
            string windowsMJCFFileName = package.WindowsPackageDirectory + Robot.Name + ".xml";
            string windowsCSVFileName = package.WindowsRobotsDirectory + Robot.Name + ".csv";
            string windowsPackageXmlFileName = package.WindowsPackageDirectory + "package.xml";

#if SOLIDWORKS
            if (ConfigurationSerialization.GetRawStringData(
                Robot.BaseLink,
                exporterConfig,
                out string robotName,
                out string urdfConfiguration,
                out string exporterConfiguration))
            {
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");

                string xmlPath = Path.Combine(package.WindowsBackupDirectory, $"{robotName}.{timestamp}.urdfConfiguration.xml");
                File.WriteAllText(xmlPath, urdfConfiguration);
                string jsonPath = Path.Combine(package.WindowsBackupDirectory, $"{robotName}.{timestamp}.exporterConfiguration.json");
                File.WriteAllText(jsonPath, exporterConfiguration);

                if (Robot.Tendons != null && Robot.Tendons.Count > 0)
                {
                    string tendonData = ConfigurationSerialization.SerializeTendons(Robot.Tendons);
                    if (!string.IsNullOrEmpty(tendonData))
                    {
                        string tendonPath = Path.Combine(package.WindowsBackupDirectory, $"{robotName}.{timestamp}.tendons.xml");
                        File.WriteAllText(tendonPath, tendonData);
                    }
                }
            }
#endif

#if NX
            if (CADRobotExporter.CAD.NX.NXConfigurationSerialization.GetRawStringData(
                Robot.BaseLink,
                exporterConfig,
                Robot.Tendons,
                out string robotName,
                out string urdfConfiguration,
                out string exporterConfiguration,
                out string tendonData))
            {
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");

                string xmlPath = Path.Combine(package.WindowsBackupDirectory, $"{robotName}.{timestamp}.urdfConfiguration.xml");
                File.WriteAllText(xmlPath, urdfConfiguration);
                string jsonPath = Path.Combine(package.WindowsBackupDirectory, $"{robotName}.{timestamp}.exporterConfiguration.json");
                File.WriteAllText(jsonPath, exporterConfiguration);
                if (!string.IsNullOrEmpty(tendonData))
                {
                    string tendonPath = Path.Combine(package.WindowsBackupDirectory, $"{robotName}.{timestamp}.tendons.xml");
                    File.WriteAllText(tendonPath, tendonData);
                }
            }
#endif

            // Customizing STL preferences
            logger.Information("Saving existing STL preferences");
            SaveSTLExportUserPreferences();

            logger.Information("Modifying STL preferences");
            SetSTLExportPreferences();

            // Hide components before export
            CadBridge.HideAllComponents();

            bool success = false;
            string exception = "";
            string meshingErrors = "";
            try
            {
                logger.Information("Beginning individual files export");
                ExportFiles(Robot.BaseLink, package, 0, exportLinkMesh, meshFormats, collisionMeshFormats, meshingOptions, ref meshingErrors);
                success = true;
            }
            catch (Exception e)
            {
                exception = e.Message;
                logger.Error($"An exception was thrown attempting to export the Robot: {exception}");
            }
            finally
            {
                logger.Information("Showing all components except previously hidden components");
                CadBridge.ShowHiddenComponents();

                logger.Information("Resetting STL preferences");
                ResetSTLExportPreferences();
            }

            if (!success)
            {
                _progressManager.Close();
                MessageBox.Show($"Exporting the Robot was cancelled or failed unexpectedly. Exception thrown: {exception}");
                return;
            }

            _progressManager.Close();

            if (!string.IsNullOrEmpty(meshingErrors))
            {
                MessageBox.Show("The following links had meshing errors:\n"
                    + meshingErrors
                    + "\n\nIf all are partial failures, some faces may be missing, but visual models are generally fine. You may need to adjust meshing parameters and re-export."
                    + "\n\nComplete failure meshes will be missing.",
                    "Robot Configuration Exporter - Meshing Errors",
                    MessageBoxButton.OK,
                    MessageBoxImage.None,
                    MessageBoxResult.OK,
                    MessageBoxOptions.DefaultDesktopOnly);
                logger.Warning("The following links had meshing errors:\n" + meshingErrors + "\n\n");
            }

            logger.Information("Writing URDF file to " + windowsURDFFileName);

            using (var writer = FormatWriterFactory.Create(ExportFormat.URDF, windowsURDFFileName))
            {
                writer.WriteRobot(Robot);
            }

            logger.Information("Writing MJCF file to " + windowsMJCFFileName);

            using (var writer = FormatWriterFactory.Create(ExportFormat.MJCF, windowsMJCFFileName, exporterConfig.folderStructure))
            {
                writer.WriteRobot(Robot);
            }

            string windowsSuperDexBotFileName = package.WindowsPackageDirectory + Robot.Name + ".superdex_bot";
            logger.Information("Writing SuperDex Bot file to " + windowsSuperDexBotFileName);

            using (var writer = FormatWriterFactory.Create(ExportFormat.SuperDexBot, windowsSuperDexBotFileName, exporterConfig.folderStructure))
            {
                writer.WriteRobot(Robot);
            }

            logger.Information("Writing package.xml file to " + windowsPackageXmlFileName);
            WritePackageXml(windowsPackageXmlFileName, Robot.Name);

            logger.Information("Resetting STL preferences");
            ResetSTLExportPreferences();

            CadBridge.SetProgressBarEnd();

            URDFFileName = windowsURDFFileName;
            MJCFFileName = windowsMJCFFileName;
            SuperDexBotFileName = windowsSuperDexBotFileName;
        }

        public List<string> GetJointNames()
        {
            List<string> jointNames = new List<string>();

            Queue<Link> queue = new Queue<Link>();
            queue.Enqueue(Robot.BaseLink);
            while (queue.Count > 0)
            {
                Link current = queue.Dequeue();
                if (current.Parent != null)
                {
                    jointNames.Add(current.Joint.Name);
                }

                foreach (Link child in current.Children)
                {
                    queue.Enqueue(child);
                }
            }

            return jointNames;
        }

        //Recursive method for exporting each link (and writing it to the URDF)
        private void ExportFiles(Link link, RobotPackage package, int count, bool exportLinkMesh, List<MeshFormat> meshFormats, List<MeshFormat> collisionMeshFormats, ExporterMeshingOptions meshingOptions, ref string meshingErrors)
        {
            CadBridge.SetProgressBarProgress(count);
            CadBridge.SetProgressBarTitle("Exporting mesh: " + link.Name);
            logger.Information("Exporting link: " + link.Name);
            _progressManager.SetCurrentStep(link.Name);
            // Iterate through each child and export its files
            logger.Information("Link " + link.Name + " has " + link.Children.Count + " children");

            Link.ComponentType visualComponentType = Link.ComponentType.Visual;
            Link.ComponentType collisionComponentType = Link.ComponentType.Collision;

            // if there are any visual components, use them
            if (link.HasVisualComponents())
            {
                visualComponentType = Link.ComponentType.Visual;
            }

            // if there are no visual components use inertials
            if (!link.HasVisualComponents() && link.HasInertialComponents())
            {
                // but only if we've decided to do so
                if (!link.inertialsOnly)
                {
                    logger.Information($"link: {link.Name}, will use inertials for visuals");
                    visualComponentType = Link.ComponentType.Inertial;
                }
            }

            // if there are collision components, use them
            if (link.HasCollisionComponents())
            {
                collisionComponentType = Link.ComponentType.Collision;
            }

            // if there are no collision compoments, and there are visual components, use them
            if (!link.HasCollisionComponents())
            {
                // but only if visual components exists (as defined above)
                if (!link.visualsOnly && link.HasComponents(visualComponentType))
                {
                    logger.Information($"link: {link.Name}, will use visuals for collision");
                    collisionComponentType = visualComponentType;
                }
            }

            // Only export meshes if the link has components tied to it
            if (link.HasComponents(visualComponentType) || link.HasComponents(collisionComponentType))
            {
                // Create the mesh filenames. SolidWorks likes to use / but that will get messy in filenames so use _ instead
                string linkName = link.Name.Replace('/', '_');
                string meshFilename = package.MeshesDirectory + linkName + package.VisualMeshPostfix + "." + meshingOptions.meshTagExtension;
                string windowsMeshFileName = package.WindowsMeshesDirectory + linkName + package.VisualMeshPostfix;

                string collisionMeshFilename = meshFilename;
                string windowsCollisionMeshFileName = windowsMeshFileName;

                if (meshingOptions.exportCollision)
                {
                    collisionMeshFilename = package.CollisionMeshesDirectory + linkName + package.CollisionMeshPostfix + "." + meshingOptions.collisionMeshTagExtension;
                    windowsCollisionMeshFileName = package.WindowsCollisionMeshesDirectory + linkName + package.CollisionMeshPostfix;
                }

                ExporterMeshingOptions linkMeshingOptions = meshingOptions;

                if (meshingOptions.perLinkMeshing)
                {
                    linkMeshingOptions.visualMeshingOptions = link.visualMeshingOptions;
                    linkMeshingOptions.collisionMeshingOptions = link.collisionMeshingOptions;
                }

                if (link.HasComponents(visualComponentType))
                {
                    if (exportLinkMesh)
                    {
                        string stepBasePath = (package.WindowsCadDirectory != null)
                            ? package.WindowsCadDirectory + linkName + package.StepVisualPostfix
                            : null;
                        SaveMesh(link, visualComponentType, meshFormats, linkMeshingOptions, windowsMeshFileName, false, ref meshingErrors, stepBasePath);
                    }
                    link.Visual.Geometry.Mesh.Filename = meshFilename;
                }

                if (link.HasComponents(collisionComponentType) && meshingOptions.exportCollision)
                {
                    if (exportLinkMesh)
                    {
                        string stepBasePath = (package.WindowsCadDirectory != null)
                            ? package.WindowsCadDirectory + linkName + package.StepCollisionPostfix
                            : null;
                        SaveMesh(link, collisionComponentType, collisionMeshFormats, linkMeshingOptions, windowsCollisionMeshFileName, true, ref meshingErrors, stepBasePath);
                    }
                    link.Collision.Geometry.Mesh.Filename = collisionMeshFilename;
                }
            }

            if (!link.HasComponents(visualComponentType))
            {
                link.Visual.Geometry.Mesh.Filename = "";
            }

            if (!link.HasComponents(collisionComponentType))
            {
                link.Collision.Geometry.Mesh.Filename = "";
            }

            foreach (Link child in link.Children)
            {
                count += 1;
                if (!child.isFixedFrame)
                {
                    if (_progressManager.IsCancelled)
                    {
                        logger.Information("Cancel invoked by user");
                        throw new Exception("Cancelled by user.");
                    }
                    ExportFiles(child, package, count, exportLinkMesh, meshFormats, collisionMeshFormats, meshingOptions, ref meshingErrors);
                }
            }
        }

        private void SaveMesh(Link link, Link.ComponentType componentType, List<MeshFormat> formats, ExporterMeshingOptions options, string windowsMeshFileName, bool exportingCollision, ref string meshingErrors, string windowsStepBasePath = null)
        {
            bool hasOpenCascade = formats.Contains(MeshFormat.stlOpenCascade) || formats.Contains(MeshFormat.objOpenCascade) || formats.Contains(MeshFormat.glbOpenCascade);
            bool hasSTEP = formats.Contains(MeshFormat.stepSolidworks);

            // If we're already exporting STEP, we don't want to do it twice, and use the generated STEP
            // to feed the OpenCascade mesh generation instead of creating a temporary one.

            string stepPath = (windowsStepBasePath != null)
                ? windowsStepBasePath + ".stp"
                : windowsMeshFileName + ".stp";

            if (hasOpenCascade && !hasSTEP)
            {
                _progressManager.SetCurrentSubStep(exportingCollision ? SaveStpCollisionStep : SaveStpStep);
                stepPath = Path.Combine(Path.GetTempPath(), $"sw2urdf_{Guid.NewGuid()}.step");
                SaveStp(link, componentType, stepPath);
            }

            string meshingError = "";
            string meshingErrorHeader = link.Name + (exportingCollision ? " (collision): " : ": ");

            if (hasSTEP)
            {
                _progressManager.SetCurrentSubStep(exportingCollision ? SaveStpCollisionStep : SaveStpStep);
                SaveStp(link, componentType, stepPath);
            }

            if (formats.Contains(MeshFormat.stlSolidworks))
            {
                _progressManager.SetCurrentSubStep(exportingCollision ? SaveStlCadCollisionStep : SaveStlCadStep);
                SaveStl(link, componentType, windowsMeshFileName + ".stl", options, exportingCollision);
            }

            if (formats.Contains(MeshFormat.glbCAD))
            {
                _progressManager.SetCurrentSubStep(exportingCollision ? SaveGlbCadCollisionStep : SaveGlbCadStep);
                SaveGlb(link, componentType, windowsMeshFileName + ".glb", options, exportingCollision);
            }

            if (formats.Contains(MeshFormat.objCAD))
            {
                _progressManager.SetCurrentSubStep(exportingCollision ? SaveObjCadCollisionStep : SaveObjCadStep);
                SaveObj(link, componentType, windowsMeshFileName + ".obj", options, exportingCollision);
            }

            // All OpenCascade formats come from one tessellation of the STEP, so they are gathered
            // first and written in a single call.
            var occOutputs = new List<MeshExportOutput>();
            if (formats.Contains(MeshFormat.objOpenCascade))
            {
                occOutputs.Add(new MeshExportOutput(MeshExportFormat.Obj, windowsMeshFileName + ".obj"));
            }
            if (formats.Contains(MeshFormat.stlOpenCascade))
            {
                occOutputs.Add(new MeshExportOutput(MeshExportFormat.Stl, windowsMeshFileName + ".stl"));
            }
            if (formats.Contains(MeshFormat.glbOpenCascade))
            {
                occOutputs.Add(new MeshExportOutput(MeshExportFormat.Glb, windowsMeshFileName + ".glb"));
            }

            if (occOutputs.Count > 0)
            {
                _progressManager.SetCurrentSubStep(
                    OpenCascadeSubStepFor(occOutputs[0].Format, exportingCollision));
                SaveMeshesOpenCascade(
                    occOutputs,
                    options,
                    exportingCollision,
                    stepPath,
                    out meshingError,
                    () => _progressManager.IsCancelled);
                if (!string.IsNullOrEmpty(meshingError))
                {
                    meshingErrors += meshingErrorHeader + meshingError + "\n";
                }
            }

            if (!hasSTEP)
            {
                try
                {
                    File.Delete(stepPath);
                }
                catch (Exception ex)
                {
                    logger.Warning($"Failed to delete temporary STEP file: {ex.Message}");
                }
            }
        }

        private static string OpenCascadeSubStepFor(MeshExportFormat format, bool exportingCollision)
        {
            switch (format)
            {
                case MeshExportFormat.Obj:
                    return exportingCollision ? SaveObjOccCollisionStep : SaveObjOccStep;
                case MeshExportFormat.Stl:
                    return exportingCollision ? SaveStlOccCollisionStep : SaveStlOccStep;
                default:
                    return exportingCollision ? SaveGlbOccCollisionStep : SaveGlbOccStep;
            }
        }

        public void ExportSingleLinkMeshes(
            Link link,
            RobotPackage package,
            string outputFolder,
            List<MeshFormat> visualFormats,
            List<MeshFormat> collisionFormats,
            ExporterMeshingOptions meshingOptions,
            Func<string, bool> confirmOverwrite)
        {
            Link.ComponentType visualComponentType = Link.ComponentType.Visual;
            Link.ComponentType collisionComponentType = Link.ComponentType.Collision;

            if (link.HasVisualComponents())
            {
                visualComponentType = Link.ComponentType.Visual;
            }
            else if (!link.inertialsOnly && link.HasInertialComponents())
            {
                visualComponentType = Link.ComponentType.Inertial;
            }

            if (link.HasCollisionComponents())
            {
                collisionComponentType = Link.ComponentType.Collision;
            }
            else if (!link.visualsOnly && link.HasComponents(visualComponentType))
            {
                collisionComponentType = visualComponentType;
            }

            _progressManager = new ProgressIndicatorManager();
            _progressManager.Show();

            _progressManager.AddStep(link.Name, link.Name);

            AddMeshFormatSubSteps(visualFormats, false);
            if (meshingOptions.exportCollision)
            {
                AddMeshFormatSubSteps(collisionFormats, true);
            }

            string meshingErrors = "";

            SaveSTLExportUserPreferences();
            SetSTLExportPreferences();
            CadBridge.HideAllComponents();

            try
            {
                _progressManager.SetCurrentStep(link.Name);

                if (visualFormats.Count > 0 && link.HasComponents(visualComponentType))
                {
                    string baseName = Path.Combine(outputFolder, link.Name + package.VisualMeshPostfix);
                    ExportSingleMesh(link, visualComponentType, visualFormats, meshingOptions,
                        baseName, false, confirmOverwrite, ref meshingErrors);
                }

                if (collisionFormats.Count > 0 && meshingOptions.exportCollision && link.HasComponents(collisionComponentType))
                {
                    string baseName = Path.Combine(outputFolder, link.Name + package.CollisionMeshPostfix);
                    ExportSingleMesh(link, collisionComponentType, collisionFormats, meshingOptions,
                        baseName, true, confirmOverwrite, ref meshingErrors);
                }
            }
            finally
            {
                CadBridge.ShowHiddenComponents();
                ResetSTLExportPreferences();
                _progressManager.Close();
            }

            if (!string.IsNullOrEmpty(meshingErrors))
            {
                System.Windows.Forms.MessageBox.Show(
                    "Meshing warnings:\n" + meshingErrors,
                    "Export Link Mesh",
                    System.Windows.Forms.MessageBoxButtons.OK,
                    System.Windows.Forms.MessageBoxIcon.Warning,
                    System.Windows.Forms.MessageBoxDefaultButton.Button1,
                    System.Windows.Forms.MessageBoxOptions.DefaultDesktopOnly);
            }
        }

        private void AddMeshFormatSubSteps(List<MeshFormat> formats, bool collision)
        {
            bool hasOpenCascade = formats.Contains(MeshFormat.stlOpenCascade)
                || formats.Contains(MeshFormat.objOpenCascade)
                || formats.Contains(MeshFormat.glbOpenCascade);
            bool hasSTEP = formats.Contains(MeshFormat.stepSolidworks);

            if (hasSTEP)
                _progressManager.AddSubStep(collision ? SaveStpCollisionStep : SaveStpStep,
                    collision ? "Export STEP file (collision)" : "Export STEP file");
            if (hasOpenCascade && !hasSTEP)
                _progressManager.AddSubStep(collision ? SaveStpCollisionStep : SaveStpStep,
                    collision ? "Export temporary STEP file (collision)" : "Export temporary STEP file");

            foreach (var meshFormat in formats)
            {
                switch (meshFormat)
                {
                    case MeshFormat.stlSolidworks:
                        _progressManager.AddSubStep(collision ? SaveStlCadCollisionStep : SaveStlCadStep,
                            collision ? "Export CAD Native STL (collision)" : "Export CAD Native STL");
                        break;
                    case MeshFormat.glbCAD:
                        _progressManager.AddSubStep(collision ? SaveGlbCadCollisionStep : SaveGlbCadStep,
                            collision ? "Export CAD Native GLB (collision)" : "Export CAD Native GLB");
                        break;
                    case MeshFormat.objCAD:
                        _progressManager.AddSubStep(collision ? SaveObjCadCollisionStep : SaveObjCadStep,
                            collision ? "Export CAD Native OBJ (collision)" : "Export CAD Native OBJ");
                        break;
                    case MeshFormat.stlOpenCascade:
                        _progressManager.AddSubStep(collision ? SaveStlOccCollisionStep : SaveStlOccStep,
                            collision ? "Export SuperDex STL (collision)" : "Export SuperDex STL");
                        break;
                    case MeshFormat.objOpenCascade:
                        _progressManager.AddSubStep(collision ? SaveObjOccCollisionStep : SaveObjOccStep,
                            collision ? "Export SuperDex OBJ (collision)" : "Export SuperDex OBJ");
                        break;
                    case MeshFormat.stepSolidworks:
                        continue;
                    case MeshFormat.glbOpenCascade:
                        _progressManager.AddSubStep(collision ? SaveGlbOccCollisionStep : SaveGlbOccStep,
                            collision ? "Export SuperDex GLTF (collision)" : "Export SuperDex GLTF");
                        break;
                }
            }
        }

        private void ExportSingleMesh(
            Link link,
            Link.ComponentType componentType,
            List<MeshFormat> formats,
            ExporterMeshingOptions options,
            string baseFileName,
            bool exportingCollision,
            Func<string, bool> confirmOverwrite,
            ref string meshingErrors,
            string windowsStepBasePath = null)
        {
            bool hasOpenCascade = formats.Contains(MeshFormat.stlOpenCascade)
                || formats.Contains(MeshFormat.objOpenCascade)
                || formats.Contains(MeshFormat.glbOpenCascade);
            bool hasSTEP = formats.Contains(MeshFormat.stepSolidworks);

            string stepPath = (windowsStepBasePath != null)
                ? windowsStepBasePath + ".stp"
                : baseFileName + ".stp";
            bool tempStep = false;

            if (hasOpenCascade && !hasSTEP)
            {
                _progressManager.SetCurrentSubStep(exportingCollision ? SaveStpCollisionStep : SaveStpStep);
                stepPath = Path.Combine(Path.GetTempPath(), $"sw2urdf_{Guid.NewGuid()}.step");
                SaveStp(link, componentType, stepPath);
                tempStep = true;
            }

            string meshingError = "";
            string meshingErrorHeader = link.Name + (exportingCollision ? " (collision): " : ": ");

            if (hasSTEP)
            {
                _progressManager.SetCurrentSubStep(exportingCollision ? SaveStpCollisionStep : SaveStpStep);
                if (!File.Exists(stepPath) || confirmOverwrite(stepPath))
                    SaveStp(link, componentType, stepPath);
            }

            if (formats.Contains(MeshFormat.stlSolidworks))
            {
                _progressManager.SetCurrentSubStep(exportingCollision ? SaveStlCadCollisionStep : SaveStlCadStep);
                string file = baseFileName + ".stl";
                if (!File.Exists(file) || confirmOverwrite(file))
                    SaveStl(link, componentType, file, options, exportingCollision);
            }

            if (formats.Contains(MeshFormat.glbCAD))
            {
                _progressManager.SetCurrentSubStep(exportingCollision ? SaveGlbCadCollisionStep : SaveGlbCadStep);
                string file = baseFileName + ".glb";
                if (!File.Exists(file) || confirmOverwrite(file))
                    SaveGlb(link, componentType, file, options, exportingCollision);
            }

            if (formats.Contains(MeshFormat.objCAD))
            {
                _progressManager.SetCurrentSubStep(exportingCollision ? SaveObjCadCollisionStep : SaveObjCadStep);
                string file = baseFileName + ".obj";
                if (!File.Exists(file) || confirmOverwrite(file))
                    SaveObj(link, componentType, file, options, exportingCollision);
            }

            // All OpenCascade formats come from one tessellation of the STEP, so they are gathered
            // first and written in a single call.
            var occOutputs = new List<MeshExportOutput>();
            if (formats.Contains(MeshFormat.objOpenCascade))
            {
                string file = baseFileName + ".obj";
                if (!File.Exists(file) || confirmOverwrite(file))
                {
                    occOutputs.Add(new MeshExportOutput(MeshExportFormat.Obj, file));
                }
            }

            if (formats.Contains(MeshFormat.stlOpenCascade))
            {
                string file = baseFileName + ".stl";
                if (!File.Exists(file) || confirmOverwrite(file))
                {
                    occOutputs.Add(new MeshExportOutput(MeshExportFormat.Stl, file));
                }
            }

            if (formats.Contains(MeshFormat.glbOpenCascade))
            {
                string file = baseFileName + ".glb";
                if (!File.Exists(file) || confirmOverwrite(file))
                {
                    occOutputs.Add(new MeshExportOutput(MeshExportFormat.Glb, file));
                }
            }

            if (occOutputs.Count > 0)
            {
                _progressManager.SetCurrentSubStep(
                    OpenCascadeSubStepFor(occOutputs[0].Format, exportingCollision));
                SaveMeshesOpenCascade(
                    occOutputs,
                    options,
                    exportingCollision,
                    stepPath,
                    out meshingError,
                    () => _progressManager.IsCancelled);
                if (!string.IsNullOrEmpty(meshingError))
                {
                    meshingErrors += meshingErrorHeader + meshingError + "\n";
                }
            }

            if (tempStep)
            {
                try { File.Delete(stepPath); }
                catch (Exception ex) { logger.Warning($"Failed to delete temporary STEP file: {ex.Message}"); }
            }
        }

        /// <summary>
        /// Builds the shared tessellation settings for one export.
        /// </summary>
        /// <param name="meshingOptions">Options from the export dialog.</param>
        /// <param name="exportingCollision">
        /// Selects the collision deflection/scale set rather than the visual one.
        /// </param>
        public static MeshExportOptions CreateMeshExportOptions(
            ExporterMeshingOptions meshingOptions,
            bool exportingCollision)
        {
            MeshingOptions source = exportingCollision
                ? meshingOptions.collisionMeshingOptions
                : meshingOptions.visualMeshingOptions;

            return new MeshExportOptions
            {
                LinearDeflection = source.linearDeflection,
                AngularDeflection = source.angularDeflection,
                Scale = MeshExportOptions.ScaleFromExporterUnits(source.scale),
                Backend = source.backend,
                EdgeSampling = source.edgeSampling,
                TargetEdgeLength = source.targetEdgeLength,
                TargetEdgeLengthFraction = source.targetEdgeLengthFraction,
                // Always keep a partial mesh and tell the user, rather than losing the whole file
                // to a handful of bad faces.
                AllowPartialFailure = true,
            };
        }

        private static string MeshExportStatusToMeshingError(MeshExportStatus status)
        {
            switch (status)
            {
                case MeshExportStatus.Written:
                    return "";
                case MeshExportStatus.WrittenPartial:
                    return "Partial meshing failure";
                case MeshExportStatus.Failed:
                    return "Complete meshing failure";
                default:
                    return "Unknown error";
            }
        }

        /// <summary>
        /// Tessellates <paramref name="stepFileName"/> once and writes every requested output from
        /// that single tessellation.
        /// </summary>
        /// <param name="outputs">Formats and destination paths to write.</param>
        /// <param name="meshingErrors">
        /// Per-output problems, one line each, suffixed with the format. Empty when all is well.
        /// </param>
        /// <param name="shouldCancel">Polled while the helper runs; true kills it.</param>
        public static void SaveMeshesOpenCascade(
            IList<MeshExportOutput> outputs,
            ExporterMeshingOptions meshingOptions,
            bool exportingCollision,
            string stepFileName,
            out string meshingErrors,
            Func<bool> shouldCancel = null)
        {
            meshingErrors = "";

            if (!File.Exists(stepFileName))
            {
                meshingErrors = "No mesh created. STEP file missing. Were all bodies hidden?";
                return;
            }

            IList<MeshExportStatus> statuses;
            try
            {
                statuses = MeshExporter.Export(
                    stepFileName,
                    outputs,
                    CreateMeshExportOptions(meshingOptions, exportingCollision),
                    shouldCancel);
            }
            catch (MeshCliException ex)
            {
                // A failure here covers the whole batch: the STEP could not be read or tessellated,
                // or the helper could not be run at all.
                logger.Warning($"Mesh export failed for {stepFileName}: {ex.Message}");
                meshingErrors = ex.Message;
                return;
            }

            var errors = new List<string>();
            for (int i = 0; i < statuses.Count; ++i)
            {
                string error = MeshExportStatusToMeshingError(statuses[i]);
                if (!string.IsNullOrEmpty(error))
                {
                    errors.Add($"{error} ({FormatSuffix(outputs[i].Format)})");
                }
            }
            meshingErrors = string.Join("\n", errors);
        }

        private static string FormatSuffix(MeshExportFormat format)
        {
            switch (format)
            {
                case MeshExportFormat.Obj:
                    return "obj";
                case MeshExportFormat.Stl:
                    return "stl";
                case MeshExportFormat.Glb:
                    return "glb";
                case MeshExportFormat.Gltf:
                    return "gltf";
                default:
                    return "mesh";
            }
        }

        private bool SaveGlb(Link link, Link.ComponentType componentType, string windowsMeshFilename, ExporterMeshingOptions meshingOptions, bool exportingCollision)
        {
            return CadBridge.SaveGlb(link, componentType, windowsMeshFilename, meshingOptions, exportingCollision);
        }

        private bool SaveStl(Link link, Link.ComponentType componentType, string windowsMeshFilename, ExporterMeshingOptions meshingOptions, bool exportingCollision)
        {
            return CadBridge.SaveStl(link, componentType, windowsMeshFilename, meshingOptions, exportingCollision);
        }

        private bool SaveObj(Link link, Link.ComponentType componentType, string windowsMeshFilename, ExporterMeshingOptions meshingOptions, bool exportingCollision)
        {
            return CadBridge.SaveObj(link, componentType, windowsMeshFilename, meshingOptions, exportingCollision);
        }

        private bool SaveStp(Link link, Link.ComponentType componentType, string windowsMeshFilename)
        {
            return CadBridge.SaveStpFile(link, componentType, windowsMeshFilename);
        }

        //Saves the preferences that the user had setup so that I can change them and revert back to their configuration
        private void SaveSTLExportUserPreferences()
        {
            logger.Information("Saving users preferences");
            CadBridge.SaveSTLExportUserPreferences();
        }

        //This is how the STL export preferences need to be to properly export
        private void SetSTLExportPreferences()
        {
            logger.Information("Setting STL preferences");
            CadBridge.SetSTLExportUserPreferences();
        }

        //This resets the user preferences back to what they were.
        private void ResetSTLExportPreferences()
        {
            logger.Information("Returning STL preferences to user preferences");
            CadBridge.ResetSTLExportUserPreferences();
        }

        public void BoostPerformance(bool enable)
        {
            CadBridge.BoostPerformance(enable);
        }

        public List<Link> GetAllLinkDescendants(Link root)
        {
            var result = new List<Link>();

            foreach (var child in root.Children)
            {
                result.Add(child);
                result.AddRange(GetAllLinkDescendants(child));
            }

            return result;
        }

        private List<LinkNode> GetAllNodeDescendants(LinkNode root)
        {
            var result = new List<LinkNode>();

            foreach (LinkNode child in root.Nodes)
            {
                result.Add(child);
                result.AddRange(GetAllNodeDescendants(child));
            }

            return result;
        }

        // The one used by the Assembly Exporter
        public bool CreateRobotFromTreeView(LinkNode baseNode)
        {
            var allNodes = GetAllNodeDescendants(baseNode);

            _progressManager = new ProgressIndicatorManager();

            _progressManager.Show();

            _progressManager.AddStep("base_link", "Base link");
            foreach (var node in allNodes)
            {
                _progressManager.AddStep(node.Link.Name, node.Link.Joint.Name + " of " + node.Link.Name);
            }

            _progressManager.AddSubStep("CreateLinkFromComponents", "Gather components");
            _progressManager.AddSubStep("GetTopLevelCoordinateSystem", "Creating temporary coordinate axis (if needed)");
            _progressManager.AddSubStep("GetRefAxisInGlobalSpace", "Calculating joint axis (global)");
            _progressManager.AddSubStep("LocalizeJoint", "Calculating joint axis (local)");
            _progressManager.AddSubStep("GetComponentsInertialProperties", "Calculating inertial properties");
            _progressManager.AddSubStep("ComputeVisualCollisionProperties", "Extracing color properties");

            // BoostSolidworksPerformance(true);

            _exportErrorWhy = "";
            Robot = new Robot();

            CadBridge.SetProgressBarStart(RobotDescription.Utilities.GetCount(baseNode.Nodes) + 1, "Building links");
            int count = 0;

            CadBridge.SetProgressBarProgress(count);
            CadBridge.SetProgressBarTitle("Building link: " + baseNode.Name);

            Link baseLink = CreateLink(baseNode, 1);
            if (baseLink == null || !string.IsNullOrWhiteSpace(_exportErrorWhy))
            {
                MessageBox.Show(_exportErrorWhy);
                logger.Warning(_exportErrorWhy);
                CadBridge.SetProgressBarEnd();
                // BoostSolidworksPerformance(false);
                return false;
            }
            Robot.SetBaseLink(baseLink);
            baseNode.Link = baseLink;

            CadBridge.SetProgressBarEnd();
            _progressManager.Close();

            BoostPerformance(false);

            return true;
        }

        private Link CreateBaseLinkFromComponents(LinkNode node)
        {
            // Build the link from the partdoc
            _progressManager.SetCurrentStep("base_link");
            Link link = CreateLinkFromComponents(null, node);
            link.Joint.CoordinateSystemName = node.Link.Joint.CoordinateSystemName;
            return link;
        }

        //Method which builds an entire link and iterates through.
        private Link CreateLink(LinkNode node, int count)
        {
            CadBridge.SetProgressBarTitle("Building link: " + node.Name);
            CadBridge.SetProgressBarProgress(count);
            Link link;
            if (node.IsBaseNode)
            {
                _progressManager.SetCurrentStep("base_link");
                link = CreateBaseLinkFromComponents(node);
                Robot.SetBaseLink(link);
            }
            else
            {
                _progressManager.SetCurrentStep(node.Link.Name);
                if (_progressManager.IsCancelled)
                {
                    logger.Information("Cancel invoked by user");
                    CleanUpTemporaryFeatures();
                    _exportErrorWhy = "Cancelled export";
                    _progressManager.Close();
                    return null;
                }
                LinkNode parentNode = (LinkNode)node.Parent;
                link = CreateLinkFromComponents(parentNode.Link, node);
            }
            node.Link = link;
            if (!string.IsNullOrWhiteSpace(_exportErrorWhy))
            {
                return null;
            }

            // Reset list of children, don't worry the links that were saved are still attached to the child nodes
            link.Children.Clear();
            foreach (LinkNode child in node.Nodes)
            {
                Link childLink = CreateLink(child, count + 1);

                if (!string.IsNullOrWhiteSpace(_exportErrorWhy))
                {
                    return null;
                }
                else
                {
                    link.Children.Add(childLink);
                }
            }
            return link;
        }

        public void ComputeInertialProperties(Link link)
        {
            _progressManager.SetCurrentSubStep("GetComponentsInertialProperties");
            logger.Information("Start ComputeInertialProperties");
            if (link.visualsOnly)
            {
                link.Inertial.Mass.Value = 0.0;
                link.Inertial.Inertia.SetMomentMatrix(new double[9]);
                link.Inertial.Origin.SetXYZ(new double[3]);
                link.Inertial.Origin.SetRPY(new double[3]);
                return;
            }

            Link.ComponentType componentType =
                link.HasInertialComponents() ? Link.ComponentType.Inertial : Link.ComponentType.Visual;

            CadBridge.GetComponentsInertialProperties(
                link,
                componentType,
                link.Joint.CoordinateSystemName,
                out double mass,
                out double[] centerOfMass,
                out double[] momentOfInertiaAroundCoM);

            link.Inertial.Mass.Value = mass;
            link.Inertial.Inertia.SetMomentMatrix(momentOfInertiaAroundCoM);
            link.Inertial.Origin.SetXYZ(centerOfMass);
            link.Inertial.Origin.SetRPY(new double[3] { 0, 0, 0 });
            logger.Information("End ComputeInertialProperties");
        }

        private void ComputeVisualCollisionProperties(Link link)
        {
            link.Visual.Origin.SetXYZ(new double[3] { 0, 0, 0 });
            link.Visual.Origin.SetRPY(new double[3] { 0, 0, 0 });
            link.Collision.Origin.SetXYZ(new double[3] { 0, 0, 0 });
            link.Collision.Origin.SetRPY(new double[3] { 0, 0, 0 });

            Link.ComponentType componentsToGetPropertiesFrom;

            if (!link.HasVisualComponents() && link.HasInertialComponents())
            {
                componentsToGetPropertiesFrom = Link.ComponentType.Inertial;
            }
            else if (link.HasVisualComponents())
            {
                componentsToGetPropertiesFrom = Link.ComponentType.Visual;
            }
            else
            {
                return;
            }

            // [ R, G, B, Ambient, Diffuse, Specular, Shininess, Transparency, Emission ]
            double[] values = CadBridge.GetVisualProperties(link, componentsToGetPropertiesFrom);
            link.Visual.Material.Color.Red = values[0];
            link.Visual.Material.Color.Green = values[1];
            link.Visual.Material.Color.Blue = values[2];
            link.Visual.Material.Color.Alpha = values[7];
        }

        // Method which builds a single link
        private Link CreateLinkFromComponents(Link parent, LinkNode node)
        {
            _progressManager.SetCurrentSubStep("CreateLinkFromComponents");

            if (parent != null)
            {
                logger.Information("Creating joint from link:" + node.Link.Name);
                bool success = CreateJoint(parent, node.Link);
                if (!success)
                {
                    logger.Warning(
                        string.Format("Creating joint from parent {0} to child {1} failed",
                            parent.Name, node.Link.Name));
                }
            }

            ComputeInertialProperties(node.Link);

            _progressManager.SetCurrentSubStep("ComputeVisualCollisionProperties");
            ComputeVisualCollisionProperties(node.Link);

            return node.Link;
        }

        // Base method for constructing a joint from a parent link and child link.
        private bool CreateJoint(Link parent, Link child)
        {
            child.Joint.Parent.Name = parent.Name;
            child.Joint.Child.Name = child.Name;

            // First we get the axis in global space
            EstimateGlobalJointFromRefGeometry(child);

            string parentCoordSysName = parent.Joint.CoordinateSystemName;

            // Then we localize the joint to its parent's space
            LocalizeJoint(child.Joint, parentCoordSysName);

            child.Joint.SetJointLimitDefaultsIfNotSet();

            return true;
        }

        // Takes a links joint and calculates the local transform from the global transforms of
        // the parent and child. It also converts the axis to local values
        private void LocalizeJoint(Joint joint, string parentCoordsysName)
        {
            _progressManager.SetCurrentSubStep("LocalizeJoint");
            Matrix<double> parentJointGlobalTransform = CadBridge.GetCoordinateSystemTransform(parentCoordsysName);
            Matrix<double> childJointGlobalTransform = CadBridge.GetCoordinateSystemTransform(joint.CoordinateSystemName);

            // Transform from global origin to child joint
            Matrix<double> childJointOrigin =
                parentJointGlobalTransform.Inverse() * childJointGlobalTransform;

            // Localize the axis to the Link's coordinate system.
            joint.Axis.SetXYZ(LocalizeAxis(joint.Axis.GetXYZ(), joint.CoordinateSystemName));

            // Get the array values and threshold them so small values are set to 0.
            joint.Origin.SetXYZ(MathOps.GetXYZ(childJointOrigin));
            joint.Origin.SetXYZ(MathOps.Threshold(joint.Origin.GetXYZ(), 0.00001));
            joint.Origin.SetRPY(MathOps.GetRPY(childJointOrigin));
            joint.Origin.SetRPY(MathOps.Threshold(joint.Origin.GetRPY(), 0.00001));
            logger.Information("End LocalizeJoint");
        }

        /// <summary>
        /// Transforms tendon routing element positions from assembly-global (meters)
        /// to each element's parent link's local coordinate frame.
        /// </summary>
        public void LocalizeTendonPositions()
        {
            if (Robot == null || Robot.Tendons == null || Robot.Tendons.Count == 0)
                return;

            // Build link name → CSYS name lookup from the robot tree
            var linkCsysMap = new Dictionary<string, string>();
            CollectLinkCsysNames(Robot.BaseLink, linkCsysMap);

            foreach (var tendon in Robot.Tendons)
            {
                foreach (var element in tendon.RoutingElements)
                {
                    if (element.Type != RoutingElement.TypeWaypoint)
                        continue;

                    if (string.IsNullOrEmpty(element.PointKey) || string.IsNullOrEmpty(element.Link))
                        continue;

                    double[] globalPos = CadBridge.GetTopLevelPointCoordinates(element.PointKey);
                    if (globalPos == null)
                        continue;

                    if (!linkCsysMap.TryGetValue(element.Link, out string csysName))
                        continue;

                    Matrix<double> linkTransform = CadBridge.GetCoordinateSystemTransform(csysName);
                    Matrix<double> linkInverse = linkTransform.Inverse();

                    // Transform global point to link-local: linkInverse * [x, y, z, 1]
                    double[] localPos = new double[]
                    {
                        linkInverse[0, 0] * globalPos[0] + linkInverse[0, 1] * globalPos[1] + linkInverse[0, 2] * globalPos[2] + linkInverse[0, 3],
                        linkInverse[1, 0] * globalPos[0] + linkInverse[1, 1] * globalPos[1] + linkInverse[1, 2] * globalPos[2] + linkInverse[1, 3],
                        linkInverse[2, 0] * globalPos[0] + linkInverse[2, 1] * globalPos[1] + linkInverse[2, 2] * globalPos[2] + linkInverse[2, 3]
                    };

                    localPos = MathOps.Threshold(localPos, 0.00001);
                    element.SetPosition(localPos);
                }
            }
        }

        private void CollectLinkCsysNames(Link link, Dictionary<string, string> map)
        {
            if (link == null)
                return;

            if (!string.IsNullOrEmpty(link.Name) && !string.IsNullOrEmpty(link.Joint?.CoordinateSystemName))
                map[link.Name] = link.Joint.CoordinateSystemName;

            if (link.Children != null)
            {
                foreach (var child in link.Children)
                    CollectLinkCsysNames(child, map);
            }
        }

        private void EstimateGlobalJointFromRefGeometry(Link child)
        {
            logger.Information("Start EstimateGlobalJointFromRefGeometry");

            Matrix<double> jointCoordinateTransform = CadBridge.GetCoordinateSystemTransform(child.Joint.CoordinateSystemName);

            child.Joint.originInGlobalSpace = MathOps.GetXYZ(jointCoordinateTransform);
            child.Joint.Origin.SetXYZ(MathOps.GetXYZ(jointCoordinateTransform));
            child.Joint.Origin.SetRPY(MathOps.GetRPY(jointCoordinateTransform));
            if (child.Joint.Type != "fixed")
            {
                double[] refAxisInGlobalSpace = GetRefAxisInGlobalSpace(
                    child.Joint.AxisName,
                    child.Joint.CoordinateSystemName,
                    jointCoordinateTransform,
                    child.shouldFlipAxis);
                child.Joint.Axis.SetXYZ(refAxisInGlobalSpace);
                child.Joint.axisInGlobalSpace = refAxisInGlobalSpace;
            }
            logger.Information("End EstimateGlobalJointFromRefGeometry");
        }

        private double[] GetRefAxisInGlobalSpace(string axisName, string csysName, Matrix<double> csysTransform, bool flipAxis)
        {
            _progressManager.SetCurrentSubStep("GetRefAxisInGlobalSpace");
            double[] axisVector = new double[3];

            logger.Information("Start GetRefAxisInGlobalSpace");

            // Check if the axis should be derived from the coordinate system
            if (Joint.IsAxisFromCsys(axisName))
            {
                axisVector = GetAxisFromCsysTransform(axisName, csysTransform, flipAxis);
            }
            else
            {
                axisVector = CadBridge.GetAxisInGlobalSpace(axisName, flipAxis);
            }

            logger.Information("End GetRefAxisInGlobalSpace");

            return axisVector;
        }

        /// <summary>
        /// Extracts the X, Y, or Z axis from a coordinate system transform matrix.
        /// </summary>
        private static double[] GetAxisFromCsysTransform(string axisKeyword, Matrix<double> csysTransform, bool flipAxis)
        {
            double[] axisVector = new double[3];

            // The transform matrix is 4x4:
            // [Xx Yx Zx Tx]
            // [Xy Yy Zy Ty]
            // [Xz Yz Zz Tz]
            // [0  0  0  1 ]
            // Column 0 = X axis, Column 1 = Y axis, Column 2 = Z axis

            if (axisKeyword == Joint.AxisFromCsysX)
            {
                axisVector[0] = csysTransform[0, 0];
                axisVector[1] = csysTransform[1, 0];
                axisVector[2] = csysTransform[2, 0];
            }
            else if (axisKeyword == Joint.AxisFromCsysY)
            {
                axisVector[0] = csysTransform[0, 1];
                axisVector[1] = csysTransform[1, 1];
                axisVector[2] = csysTransform[2, 1];
            }
            else if (axisKeyword == Joint.AxisFromCsysZ)
            {
                axisVector[0] = csysTransform[0, 2];
                axisVector[1] = csysTransform[1, 2];
                axisVector[2] = csysTransform[2, 2];
            }
            else
            {
                // Default to Z axis if unknown
                axisVector[0] = 0;
                axisVector[1] = 0;
                axisVector[2] = 1;
            }

            // Normalize the axis vector
            double length = Math.Sqrt(axisVector[0] * axisVector[0] + axisVector[1] * axisVector[1] + axisVector[2] * axisVector[2]);
            if (length > 0)
            {
                axisVector[0] /= length;
                axisVector[1] /= length;
                axisVector[2] /= length;
            }

            // Apply flip if needed
            if (flipAxis)
            {
                axisVector[0] = -axisVector[0];
                axisVector[1] = -axisVector[1];
                axisVector[2] = -axisVector[2];
            }

            return MathOps.Threshold(axisVector, 0.00001);
        }

        // This is called whenever the pull down menu is changed and the axis needs to be
        // recalculated in reference to the coordinate system
        public double[] LocalizeAxis(double[] Axis, string coordsys)
        {
            Matrix<double> coordsysTransform = CadBridge.GetCoordinateSystemTransform(coordsys);
            return LocalizeAxis(Axis, coordsysTransform);
        }

        // This is called by the above method and the getRefAxis method
        private static double[] LocalizeAxis(double[] Axis, Matrix<double> coordsysTransform)
        {
            if (coordsysTransform != null)
            {
                Vector<double> vec = new DenseVector(new double[] { Axis[0], Axis[1], Axis[2], 0 }); ;
                vec = coordsysTransform.Inverse() * vec;
                Axis[0] = vec[0]; Axis[1] = vec[1]; Axis[2] = vec[2];
            }
            return MathOps.Threshold(Axis, 0.00001);
        }

        public string GetTopLevelCoordinateSystem(string someCoordinateSystemName)
        {
            _progressManager.SetCurrentSubStep("GetTopLevelCoordinateSystem");
            return CadBridge.GetTopLevelCoordinateSystem(someCoordinateSystemName);
        }

        public void CleanUpTemporaryFeatures()
        {
            CadBridge.CleanUpTemporaryFeatures();
        }

        public void GetAssemblyBoundingBox(out double[] boundingBox)
        {
            CadBridge.GetModelBoundingBox(out boundingBox);
        }

        /// <summary>
        /// Writes a minimal ROS package.xml file for the robot package.
        /// </summary>
        /// <param name="filePath">Full path to the package.xml file</param>
        /// <param name="robotName">Name of the robot (used as package name)</param>
        private void WritePackageXml(string filePath, string robotName)
        {
            try
            {
                string packageXml =
                    "<?xml version=\"1.0\"?>\n" +
                    "<package format=\"3\">\n" +
                    $"  <name>{robotName}</name>\n" +
                    "  <version>0.0.0</version>\n" +
                    $"  <description>URDF description package for {robotName}</description>\n" +
                    "  <maintainer email=\"todo@todo.com\">TODO</maintainer>\n" +
                    "  <license>TODO</license>\n" +
                    "</package>\n";
                File.WriteAllText(filePath, packageXml);
                logger.Information($"Successfully wrote package.xml to {filePath}");
            }
            catch (Exception ex)
            {
                logger.Error($"Failed to write package.xml: {ex.Message}");
            }
        }
    }
}
