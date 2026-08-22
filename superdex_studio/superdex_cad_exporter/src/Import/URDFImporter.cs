/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Xml.Linq;

using MathNet.Numerics.LinearAlgebra;

#if NX
using NXOpen;
using NXOpen.Facet;
using NXOpen.Features;
using CADRobotExporter.CAD.NX;
#endif

#if SOLIDWORKS
using SolidWorks.Interop.sldworks;
#endif

using CADRobotExporter.CAD;
using CADRobotExporter.RobotDescription;
using CADRobotExporter.UI;
using CADRobotExporter.Utilities;

using Axis = CADRobotExporter.RobotDescription.Axis;
using Joint = CADRobotExporter.RobotDescription.Joint;
using Material = CADRobotExporter.RobotDescription.Material;
using Collision = CADRobotExporter.RobotDescription.Collision;

namespace CADRobotExporter.Import
{
    public enum MeshImportMode
    {
        None,
        VisualOnly,
        CollisionOnly,
        Both
    }

    /// <summary>
    /// Configuration options for URDF import.
    /// </summary>
    public class URDFImportConfiguration
    {
        /// <summary>
        /// Path to the URDF file to import.
        /// </summary>
        public string UrdfFilePath { get; set; }

        /// <summary>
        /// Whether to create NX coordinate systems at each joint origin.
        /// </summary>
        public bool CreateCoordinateSystems { get; set; } = true;

        /// <summary>
        /// Whether to create a Robot Configuration CustomFeature from the imported data.
        /// </summary>
        public bool CreateRobotConfigurationFeature { get; set; } = true;

        /// <summary>
        /// Which meshes to import into the CAD model (Visual, Collision, Both, or None).
        /// </summary>
        public MeshImportMode ImportMeshes { get; set; } = MeshImportMode.Both;

        /// <summary>
        /// Optional base path for resolving relative mesh file paths.
        /// If not specified, mesh paths from the URDF will be stored as-is.
        /// </summary>
        public string MeshBasePath { get; set; }

        /// <summary>
        /// Robot name override. If null, uses the name from the URDF file.
        /// </summary>
        public string RobotNameOverride { get; set; }

        /// <summary>
        /// Optional name of the coordinate system to use as the base/world frame.
        /// If null or empty, uses the assembly origin (identity transform).
        /// </summary>
        public string BaseCoordinateSystemName { get; set; }

        /// <summary>
        /// Optional 4x4 base/world frame transform (in meters). Takes precedence over
        /// <see cref="BaseCoordinateSystemName"/>. If null, the current WCS is used.
        /// </summary>
        public Matrix<double> BaseTransform { get; set; }
    }

    /// <summary>
    /// Result of a URDF import operation.
    /// </summary>
    public class URDFImportResult
    {
        public Link RootLink { get; set; }

        public List<string> Warnings { get; set; } = new List<string>();

        public string RobotName { get; set; }

#if NX
        public CustomFeature ConfigurationFeature { get; set; }
#endif

        public Dictionary<string, string> CreatedCoordinateSystems { get; set; } = new Dictionary<string, string>();

        public bool Success { get; set; }

        public string ErrorMessage { get; set; }
    }

    /// <summary>
    /// Imports URDF (Unified Robot Description Format) files into NX.
    ///
    /// Supports:
    /// - Parsing full URDF XML structure (links, joints, origins, axes, limits, inertials)
    /// - Creating NX coordinate systems at joint origins
    /// - Creating Robot Configuration CustomFeatures for use with the Robot Exporter
    /// </summary>
    public static class URDFImporter
    {
        private static readonly Serilog.ILogger logger = Logger.GetLogger();

#if NX
        private static Session Session => Session.GetSession();
#endif

        private const double METERS_TO_MM = 1000.0;

        #region Public API

#if NX
        /// <summary>
        /// Imports a URDF file into the specified NX part.
        /// </summary>
        /// <param name="workPart">The NX part to import into.</param>
        /// <param name="config">Import configuration options.</param>
        /// <returns>Result of the import operation.</returns>
        public static URDFImportResult Import(Part workPart, URDFImportConfiguration config)
        {
            var result = new URDFImportResult();

            try
            {
                if (workPart == null)
                    throw new ArgumentNullException(nameof(workPart), "Work part cannot be null");

                if (config == null)
                    throw new ArgumentNullException(nameof(config), "Configuration cannot be null");

                if (string.IsNullOrEmpty(config.UrdfFilePath))
                    throw new ArgumentException("URDF file path must be specified", nameof(config));

                if (!File.Exists(config.UrdfFilePath))
                    throw new FileNotFoundException($"URDF file not found: {config.UrdfFilePath}");

                logger.Information($"Starting URDF import from: {config.UrdfFilePath}");

                // Parse the URDF file
                var (rootLink, robotName) = ParseUrdfFile(config.UrdfFilePath, config.MeshBasePath, result.Warnings);

                if (rootLink == null)
                    throw new InvalidOperationException("Failed to parse URDF file - no root link found");

                result.RootLink = rootLink;
                result.RobotName = config.RobotNameOverride ?? robotName ?? "imported_robot";

                logger.Information($"Parsed robot '{result.RobotName}' with {CountLinks(rootLink)} links");

                var nxBridge = new NXBridge(workPart);
                Matrix<double> baseTransform = config.BaseTransform;

                // Create coordinate systems at joint origins if requested
                if (config.CreateCoordinateSystems)
                {
                    result.CreatedCoordinateSystems = CreateCoordinateSystemsForJoints(nxBridge, rootLink, baseTransform);
                    logger.Information($"Created {result.CreatedCoordinateSystems.Count} coordinate systems");
                }

                // Import meshes if requested
                if (config.ImportMeshes != MeshImportMode.None)
                {
                    string urdfDirectory = Path.GetDirectoryName(Path.GetFullPath(config.UrdfFilePath));
                    ImportMeshesForLinks(workPart, nxBridge, rootLink, config.ImportMeshes, urdfDirectory, config.MeshBasePath, baseTransform, result.Warnings);
                }

                // Create Robot Configuration feature if requested
                if (config.CreateRobotConfigurationFeature)
                {
                    var exporterConfig = new ExporterConfiguration
                    {
                        robotName = result.RobotName
                    };

                    result.ConfigurationFeature = CADRobotExporter.CAD.NX.NXConfigurationSerialization.SaveConfiguration(
                        workPart,
                        rootLink,
                        exporterConfig,
                        null,
                        $"Robot Configuration ({result.RobotName})");

                    if (result.ConfigurationFeature != null)
                    {
                        logger.Information($"Created Robot Configuration feature: {result.ConfigurationFeature.Name}");
                    }
                    else
                    {
                        result.Warnings.Add("Failed to create Robot Configuration feature");
                    }
                }

                result.Success = true;
            }
            catch (Exception ex)
            {
                logger.Error($"URDF import failed: {ex.Message}");
                result.Success = false;
                result.ErrorMessage = ex.Message;
            }

            return result;
        }

        /// <summary>
        /// Imports mesh files for all links in the tree, positioning them at the correct world transforms.
        /// </summary>
        private static void ImportMeshesForLinks(
            Part workPart,
            NXBridge nxBridge,
            Link rootLink,
            MeshImportMode mode,
            string urdfDirectory,
            string meshBasePath,
            Matrix<double> baseTransformOverride,
            List<string> warnings)
        {
            // Must match the base frame used for the joint coordinate systems
            Matrix<double> baseMatrix = baseTransformOverride ?? nxBridge.GetWorkCoordinateSystemTransform();
            Transform4x4 rootTransform = Transform4x4.FromMatrix(baseMatrix);

            logger.Information($"Starting mesh import (mode: {mode})");
            int importCount = 0;

            ImportMeshesRecursive(workPart, rootLink, rootTransform, mode, urdfDirectory, meshBasePath, warnings, ref importCount);

            logger.Information($"Imported {importCount} mesh bodies");
        }

        private static void ImportMeshesRecursive(
            Part workPart,
            Link link,
            Transform4x4 parentWorldTransform,
            MeshImportMode mode,
            string urdfDirectory,
            string meshBasePath,
            List<string> warnings,
            ref int importCount)
        {
            // Compute this link's world transform (same logic as CreateCoordinateSystemsRecursive)
            Transform4x4 linkWorldTransform = parentWorldTransform;

            if (!link.IsBaseLink && link.Joint != null && !string.IsNullOrEmpty(link.Joint.Name))
            {
                double[] xyz = link.Joint.Origin.GetXYZ();
                double[] rpy = link.Joint.Origin.GetRPY();
                Transform4x4 jointLocalTransform = Transform4x4.FromXyzRpy(
                    xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2]);
                linkWorldTransform = parentWorldTransform.Multiply(jointLocalTransform);
            }

            // Import visual mesh
            if (mode == MeshImportMode.VisualOnly || mode == MeshImportMode.Both)
            {
                string visualMeshPath = ResolveMeshPath(link.Visual?.Geometry?.Mesh?.Filename, urdfDirectory, meshBasePath);
                if (!string.IsNullOrEmpty(visualMeshPath) && File.Exists(visualMeshPath))
                {
                    // Compose link world transform with visual origin offset
                    Transform4x4 visualTransform = linkWorldTransform;
                    if (link.Visual?.Origin != null)
                    {
                        double[] vXyz = link.Visual.Origin.GetXYZ();
                        double[] vRpy = link.Visual.Origin.GetRPY();
                        if (vXyz != null && vRpy != null)
                        {
                            Transform4x4 visualOffset = Transform4x4.FromXyzRpy(
                                vXyz[0], vXyz[1], vXyz[2], vRpy[0], vRpy[1], vRpy[2]);
                            visualTransform = linkWorldTransform.Multiply(visualOffset);
                        }
                    }

                    double scaleFactor = link.Visual?.Geometry?.MeshScaleFactor ?? 1.0;
                    List<Body> visualBodies = ImportAndPositionMesh(workPart, visualMeshPath, visualTransform, scaleFactor, warnings);
                    if (visualBodies.Count > 0)
                    {
                        if (link.NXVisualBodiesHandles == null)
                            link.NXVisualBodiesHandles = new List<string>();
                        foreach (var body in visualBodies)
                        {
                            link.NXVisualBodiesHandles.Add(NXPersistentId.GetOrCreateBodyKey(body));
                        }
                        importCount += visualBodies.Count;
                        logger.Debug($"Imported {visualBodies.Count} visual body(ies) for link '{link.Name}': {Path.GetFileName(visualMeshPath)}");
                    }
                }
            }

            // Import collision mesh
            if (mode == MeshImportMode.CollisionOnly || mode == MeshImportMode.Both)
            {
                string collisionMeshPath = ResolveMeshPath(link.Collision?.Geometry?.Mesh?.Filename, urdfDirectory, meshBasePath);
                if (!string.IsNullOrEmpty(collisionMeshPath) && File.Exists(collisionMeshPath))
                {
                    // Compose link world transform with collision origin offset
                    Transform4x4 collisionTransform = linkWorldTransform;
                    if (link.Collision?.Origin != null)
                    {
                        double[] cXyz = link.Collision.Origin.GetXYZ();
                        double[] cRpy = link.Collision.Origin.GetRPY();
                        if (cXyz != null && cRpy != null)
                        {
                            Transform4x4 collisionOffset = Transform4x4.FromXyzRpy(
                                cXyz[0], cXyz[1], cXyz[2], cRpy[0], cRpy[1], cRpy[2]);
                            collisionTransform = linkWorldTransform.Multiply(collisionOffset);
                        }
                    }

                    double scaleFactor = link.Collision?.Geometry?.MeshScaleFactor ?? 1.0;
                    List<Body> collisionBodies = ImportAndPositionMesh(workPart, collisionMeshPath, collisionTransform, scaleFactor, warnings);
                    if (collisionBodies.Count > 0)
                    {
                        if (link.NXCollisionBodiesHandles == null)
                            link.NXCollisionBodiesHandles = new List<string>();
                        foreach (var body in collisionBodies)
                        {
                            link.NXCollisionBodiesHandles.Add(NXPersistentId.GetOrCreateBodyKey(body));
                        }
                        importCount += collisionBodies.Count;
                        logger.Debug($"Imported {collisionBodies.Count} collision body(ies) for link '{link.Name}': {Path.GetFileName(collisionMeshPath)}");
                    }
                }
            }

            // Recurse to children
            foreach (var child in link.Children)
            {
                ImportMeshesRecursive(workPart, child, linkWorldTransform, mode, urdfDirectory, meshBasePath, warnings, ref importCount);
            }
        }

        /// <summary>
        /// Imports a mesh file (STL or OBJ) and moves the resulting bodies to the specified world transform.
        /// </summary>
        private static List<Body> ImportAndPositionMesh(Part workPart, string meshPath, Transform4x4 worldTransform, double scaleFactor, List<string> warnings)
        {
            Session theSession = Session.GetSession();
            string extension = Path.GetExtension(meshPath).ToLowerInvariant();

            List<Body> importedBodies = null;

            try
            {
                switch (extension)
                {
                    case ".stl":
                        importedBodies = ImportStlFile(workPart, theSession, meshPath, scaleFactor);
                        break;
                    case ".obj":
                        importedBodies = ImportObjFile(workPart, theSession, meshPath, scaleFactor);
                        break;
                    default:
                        warnings.Add($"Unsupported mesh format '{extension}' for file: {meshPath}");
                        return new List<Body>();
                }
            }
            catch (Exception ex)
            {
                warnings.Add($"Failed to import mesh '{Path.GetFileName(meshPath)}': {ex.Message}");
                return new List<Body>();
            }

            if (importedBodies == null || importedBodies.Count == 0)
            {
                warnings.Add($"No bodies produced from mesh import: {Path.GetFileName(meshPath)}");
                return new List<Body>();
            }

            // Move all bodies to the correct world position
            try
            {
                MoveBodiesToTransform(workPart, theSession, importedBodies, worldTransform);
            }
            catch (Exception ex)
            {
                warnings.Add($"Failed to position mesh '{Path.GetFileName(meshPath)}': {ex.Message}");
            }

            return importedBodies;
        }

        /// <summary>
        /// Determines the NX STL import units from the URDF mesh scale factor.
        /// scale=0.001 → mesh is in mm, scale=0.01 → cm, scale=1 → meters.
        /// </summary>
        private static STLImportBuilder.STLFileUnitsTypes GetStlUnitsFromScale(double scaleFactor)
        {
            if (Math.Abs(scaleFactor - 0.001) < 1e-6)
                return STLImportBuilder.STLFileUnitsTypes.Millimeters;
            return STLImportBuilder.STLFileUnitsTypes.Meters;
        }

        private static WavefrontObjImporter.UnitsEnum GetObjUnitsFromScale(double scaleFactor)
        {
            if (Math.Abs(scaleFactor - 0.001) < 1e-6)
                return WavefrontObjImporter.UnitsEnum.Millimeters;
            if (Math.Abs(scaleFactor - 0.01) < 1e-6)
                return WavefrontObjImporter.UnitsEnum.Centimeters;
            return WavefrontObjImporter.UnitsEnum.Meters;
        }

        private static List<Body> ImportStlFile(Part workPart, Session theSession, string filePath, double scaleFactor)
        {
            // Snapshot existing bodies before import
            var bodiesBefore = new HashSet<Tag>(GetAllBodyTags(workPart));

            Session.UndoMarkId markId = theSession.SetUndoMark(Session.MarkVisibility.Invisible, "STL Import");

            STLImportBuilder stlBuilder = workPart.FacetedBodies.CreateSTLImportBuilder();
            try
            {
                stlBuilder.File = filePath;
                stlBuilder.STLFileUnits = GetStlUnitsFromScale(scaleFactor);
                stlBuilder.AngularTolerance = STLImportBuilder.AngularToleranceTypes.Medium;

                stlBuilder.Commit();
            }
            finally
            {
                stlBuilder.Destroy();
            }

            return FindNewlyCreatedBodies(workPart, bodiesBefore);
        }

        private static List<Body> ImportObjFile(Part workPart, Session theSession, string filePath, double scaleFactor)
        {
            // Snapshot existing bodies before import
            var bodiesBefore = new HashSet<Tag>(GetAllBodyTags(workPart));

            Session.UndoMarkId markId = theSession.SetUndoMark(Session.MarkVisibility.Invisible, "OBJ Import");

            WavefrontObjImporter objImporter = theSession.DexManager.CreateWavefrontObjImporter();
            try
            {
                objImporter.SetMode(BaseImporter.Mode.NativeFileSystem);
                objImporter.InputFile = filePath;
                objImporter.ImportUnits = GetObjUnitsFromScale(scaleFactor);
                objImporter.ImportGroups = WavefrontObjImporter.GroupsEnum.Off;
                objImporter.ImportTo = WavefrontObjImporter.ImportToOption.WorkPart;
                objImporter.FileOpenFlag = false;

                objImporter.Commit();
            }
            finally
            {
                objImporter.Destroy();
            }

            return FindNewlyCreatedBodies(workPart, bodiesBefore);
        }

        private static List<Tag> GetAllBodyTags(Part workPart)
        {
            var tags = new List<Tag>();
            foreach (Body body in workPart.Bodies)
            {
                tags.Add(body.Tag);
            }
            return tags;
        }

        private static List<Body> FindNewlyCreatedBodies(Part workPart, HashSet<Tag> bodiesBefore)
        {
            var newBodies = new List<Body>();
            foreach (Body body in workPart.Bodies)
            {
                if (!bodiesBefore.Contains(body.Tag))
                    newBodies.Add(body);
            }
            return newBodies;
        }

        private static void MoveBodiesToTransform(Part workPart, Session theSession, List<Body> bodies, Transform4x4 transform)
        {
            // Check if the transform is identity (no move needed)
            bool isIdentity =
                Math.Abs(transform.R00 - 1) < 1e-10 && Math.Abs(transform.R11 - 1) < 1e-10 && Math.Abs(transform.R22 - 1) < 1e-10 &&
                Math.Abs(transform.R01) < 1e-10 && Math.Abs(transform.R02) < 1e-10 &&
                Math.Abs(transform.R10) < 1e-10 && Math.Abs(transform.R12) < 1e-10 &&
                Math.Abs(transform.R20) < 1e-10 && Math.Abs(transform.R21) < 1e-10 &&
                Math.Abs(transform.Tx) < 1e-10 && Math.Abs(transform.Ty) < 1e-10 && Math.Abs(transform.Tz) < 1e-10;

            if (isIdentity)
                return;

            // Build the 4x4 homogeneous transform for NX (translation in mm)
            Matrix4x4 nxMatrix = new Matrix4x4();
            nxMatrix.Rxx = transform.R00; nxMatrix.Rxy = transform.R01; nxMatrix.Rxz = transform.R02;
            nxMatrix.Ryx = transform.R10; nxMatrix.Ryy = transform.R11; nxMatrix.Ryz = transform.R12;
            nxMatrix.Rzx = transform.R20; nxMatrix.Rzy = transform.R21; nxMatrix.Rzz = transform.R22;
            nxMatrix.Ss = 1.0;
            nxMatrix.Sx = 1.0; nxMatrix.Sy = 1.0; nxMatrix.Sz = 1.0;
            nxMatrix.Xt = transform.Tx * METERS_TO_MM;
            nxMatrix.Yt = transform.Ty * METERS_TO_MM;
            nxMatrix.Zt = transform.Tz * METERS_TO_MM;

            // Use MoveObjectBuilder with PreMultiplicationTransform for positioning
            MoveObjectBuilder moveBuilder = workPart.BaseFeatures.CreateMoveObjectBuilder(null);
            try
            {
                moveBuilder.PreMultiplicationTransform = nxMatrix;
                moveBuilder.TransformMotion.Option = NXOpen.GeometricUtilities.ModlMotion.Options.DeltaXyz;
                moveBuilder.TransformMotion.DeltaEnum = NXOpen.GeometricUtilities.ModlMotion.Delta.ReferenceAcsWorkPart;
                moveBuilder.TransformMotion.DeltaXc.Value = 0;
                moveBuilder.TransformMotion.DeltaYc.Value = 0;
                moveBuilder.TransformMotion.DeltaZc.Value = 0;
                moveBuilder.MoveObjectResult = MoveObjectBuilder.MoveObjectResultOptions.MoveOriginal;

                foreach (var body in bodies)
                {
                    moveBuilder.ObjectToMoveObject.Add(body);
                }
                moveBuilder.Commit();
            }
            finally
            {
                moveBuilder.Destroy();
            }
        }
#endif

#if SOLIDWORKS
        /// <summary>
        /// Imports a URDF file into the specified SolidWorks model.
        /// </summary>
        /// <param name="swApp">The SolidWorks application instance.</param>
        /// <param name="model">The SolidWorks model to import into.</param>
        /// <param name="config">Import configuration options.</param>
        /// <returns>Result of the import operation.</returns>
        public static URDFImportResult Import(
            SolidWorks.Interop.sldworks.ISldWorks swApp,
            SolidWorks.Interop.sldworks.ModelDoc2 model,
            URDFImportConfiguration config)
        {
            var result = new URDFImportResult();

            try
            {
                if (swApp == null)
                    throw new ArgumentNullException(nameof(swApp), "SolidWorks application cannot be null");

                if (model == null)
                    throw new ArgumentNullException(nameof(model), "Model cannot be null");

                if (config == null)
                    throw new ArgumentNullException(nameof(config), "Configuration cannot be null");

                if (string.IsNullOrEmpty(config.UrdfFilePath))
                    throw new ArgumentException("URDF file path must be specified", nameof(config));

                if (!File.Exists(config.UrdfFilePath))
                    throw new FileNotFoundException($"URDF file not found: {config.UrdfFilePath}");

                logger.Information($"Starting URDF import from: {config.UrdfFilePath}");

                // Parse the URDF file
                var (rootLink, robotName) = ParseUrdfFile(config.UrdfFilePath, config.MeshBasePath, result.Warnings);

                if (rootLink == null)
                    throw new InvalidOperationException("Failed to parse URDF file - no root link found");

                result.RootLink = rootLink;
                result.RobotName = config.RobotNameOverride ?? robotName ?? "imported_robot";

                logger.Information($"Parsed robot '{result.RobotName}' with {CountLinks(rootLink)} links");

                // Create coordinate systems at joint origins if requested
                if (config.CreateCoordinateSystems)
                {
                    var swBridge = new SolidworksBridge(swApp, model);
                    Matrix<double> baseTransform = config.BaseTransform;
                    if (baseTransform == null && !string.IsNullOrEmpty(config.BaseCoordinateSystemName))
                    {
                        baseTransform = swBridge.GetCoordinateSystemTransform(config.BaseCoordinateSystemName);
                    }
                    result.CreatedCoordinateSystems = CreateCoordinateSystemsForJoints(swBridge, rootLink, baseTransform);
                    logger.Information($"Created {result.CreatedCoordinateSystems.Count} coordinate systems");
                }

                // Create Robot Configuration feature if requested
                if (config.CreateRobotConfigurationFeature)
                {
                    var exporterConfig = new ExporterConfiguration
                    {
                        robotName = result.RobotName
                    };

                    // Serialize configuration and create feature
                    ConfigurationSerialization.GetRawStringData(
                        rootLink,
                        exporterConfig,
                        out string robotNameOut,
                        out string xmlConfiguration,
                        out string exporterConfiguration);

                    bool created = ConfigurationSerialization.CreateNewExporterFeatureFromRawData(
                        (SolidWorks.Interop.sldworks.SldWorks)swApp,
                        result.RobotName,
                        xmlConfiguration,
                        exporterConfiguration);

                    if (created)
                    {
                        logger.Information($"Created Robot Configuration feature for '{result.RobotName}'");
                    }
                    else
                    {
                        result.Warnings.Add("Failed to create Robot Configuration feature");
                    }
                }

                result.Success = true;
            }
            catch (Exception ex)
            {
                logger.Error($"URDF import failed: {ex.Message}");
                result.Success = false;
                result.ErrorMessage = ex.Message;
            }

            return result;
        }

        /// <summary>
        /// Creates SolidWorks coordinate systems at each joint origin in the link tree (SolidWorks-specific overload).
        /// </summary>
        private static Dictionary<string, string> CreateCoordinateSystemsForJoints(
            SolidWorks.Interop.sldworks.ISldWorks swApp,
            SolidWorks.Interop.sldworks.ModelDoc2 model,
            Link rootLink)
        {
            // Create SolidworksBridge instance for coordinate system creation
            var swBridge = new SolidworksBridge(swApp, model);
            return CreateCoordinateSystemsForJoints(swBridge, rootLink);
        }
#endif

        /// <summary>
        /// Parses a URDF file and returns the robot link tree without creating any NX features.
        /// Useful for previewing URDF content before importing.
        /// </summary>
        /// <param name="urdfFilePath">Path to the URDF file.</param>
        /// <returns>Tuple of (root link, robot name).</returns>
        public static (Link RootLink, string RobotName) ParseUrdfFile(string urdfFilePath)
        {
            var warnings = new List<string>();
            return ParseUrdfFile(urdfFilePath, null, warnings);
        }

        #endregion

        #region URDF Parsing

        private static (Link RootLink, string RobotName) ParseUrdfFile(
            string urdfFilePath,
            string meshBasePath,
            List<string> warnings)
        {
            XDocument doc = XDocument.Load(urdfFilePath);
            XElement robotElement = doc.Root;

            if (robotElement == null || robotElement.Name.LocalName != "robot")
            {
                throw new InvalidOperationException("Invalid URDF file: root element must be <robot>");
            }

            string robotName = robotElement.Attribute("name")?.Value ?? "robot";

            // Parse all links into a dictionary
            var linksByName = new Dictionary<string, Link>();
            foreach (var linkElement in robotElement.Elements("link"))
            {
                Link link = ParseLink(linkElement, meshBasePath, warnings);
                if (link != null && !string.IsNullOrEmpty(link.Name))
                {
                    linksByName[link.Name] = link;
                }
            }

            if (linksByName.Count == 0)
            {
                throw new InvalidOperationException("No links found in URDF file");
            }

            // Parse all joints and build parent-child relationships
            var childLinkNames = new HashSet<string>();
            foreach (var jointElement in robotElement.Elements("joint"))
            {
                ParseJointAndBuildRelationship(jointElement, linksByName, childLinkNames, warnings);
            }

            // Find the root link (the one that is not a child of any joint)
            Link rootLink = null;
            foreach (var link in linksByName.Values)
            {
                if (!childLinkNames.Contains(link.Name))
                {
                    if (rootLink != null)
                    {
                        warnings.Add($"Multiple root links found: '{rootLink.Name}' and '{link.Name}'. Using '{link.Name}'.");
                    }
                    rootLink = link;
                }
            }

            if (rootLink == null)
            {
                throw new InvalidOperationException("Could not determine root link - possible circular joint references");
            }

            rootLink.IsBaseLink = true;

            return (rootLink, robotName);
        }

        private static Link ParseLink(XElement linkElement, string meshBasePath, List<string> warnings)
        {
            string linkName = linkElement.Attribute("name")?.Value;
            if (string.IsNullOrEmpty(linkName))
            {
                warnings.Add("Skipping link without name attribute");
                return null;
            }

            var link = new Link
            {
                Name = linkName
            };

            // Parse inertial
            var inertialElement = linkElement.Element("inertial");
            if (inertialElement != null)
            {
                ParseInertial(inertialElement, link.Inertial, warnings);
            }

            // Parse visual
            var visualElement = linkElement.Element("visual");
            if (visualElement != null)
            {
                ParseVisual(visualElement, link.Visual, meshBasePath, warnings);
            }

            // Parse collision
            var collisionElement = linkElement.Element("collision");
            if (collisionElement != null)
            {
                ParseCollision(collisionElement, link.Collision, meshBasePath, warnings);
            }

            return link;
        }

        private static void ParseJointAndBuildRelationship(
            XElement jointElement,
            Dictionary<string, Link> linksByName,
            HashSet<string> childLinkNames,
            List<string> warnings)
        {
            string jointName = jointElement.Attribute("name")?.Value;
            string jointType = jointElement.Attribute("type")?.Value ?? "fixed";

            var parentElement = jointElement.Element("parent");
            var childElement = jointElement.Element("child");

            string parentLinkName = parentElement?.Attribute("link")?.Value;
            string childLinkName = childElement?.Attribute("link")?.Value;

            if (string.IsNullOrEmpty(parentLinkName) || string.IsNullOrEmpty(childLinkName))
            {
                warnings.Add($"Joint '{jointName}' missing parent or child link reference");
                return;
            }

            if (!linksByName.TryGetValue(parentLinkName, out Link parentLink))
            {
                warnings.Add($"Joint '{jointName}' references unknown parent link '{parentLinkName}'");
                return;
            }

            if (!linksByName.TryGetValue(childLinkName, out Link childLink))
            {
                warnings.Add($"Joint '{jointName}' references unknown child link '{childLinkName}'");
                return;
            }

            // Set up the joint on the child link
            childLink.Joint.Name = jointName;
            childLink.Joint.Type = jointType.ToLower();
            childLink.Joint.Parent.Name = parentLinkName;
            childLink.Joint.Child.Name = childLinkName;

            // Parse joint origin
            var originElement = jointElement.Element("origin");
            if (originElement != null)
            {
                ParseOrigin(originElement, childLink.Joint.Origin);
            }

            // Parse joint axis
            var axisElement = jointElement.Element("axis");
            if (axisElement != null)
            {
                ParseAxis(axisElement, childLink.Joint.Axis);
            }

            // Parse joint limits
            var limitElement = jointElement.Element("limit");
            if (limitElement != null)
            {
                ParseLimit(limitElement, childLink.Joint.Limit);
            }

            // Parse joint dynamics
            var dynamicsElement = jointElement.Element("dynamics");
            if (dynamicsElement != null)
            {
                ParseDynamics(dynamicsElement, childLink.Joint.Dynamics);
            }

            // Build the parent-child relationship
            parentLink.AddChild(childLink);
            childLinkNames.Add(childLinkName);
        }

        private static void ParseOrigin(XElement originElement, Origin origin)
        {
            string xyzAttr = originElement.Attribute("xyz")?.Value;
            if (!string.IsNullOrEmpty(xyzAttr))
            {
                double[] xyz = ParseDoubleArray(xyzAttr, 3);
                if (xyz != null)
                {
                    origin.SetXYZ(xyz);
                }
            }

            string rpyAttr = originElement.Attribute("rpy")?.Value;
            if (!string.IsNullOrEmpty(rpyAttr))
            {
                double[] rpy = ParseDoubleArray(rpyAttr, 3);
                if (rpy != null)
                {
                    origin.SetRPY(rpy);
                }
            }
        }

        private static void ParseAxis(XElement axisElement, Axis axis)
        {
            string xyzAttr = axisElement.Attribute("xyz")?.Value;
            if (!string.IsNullOrEmpty(xyzAttr))
            {
                double[] xyz = ParseDoubleArray(xyzAttr, 3);
                if (xyz != null)
                {
                    axis.SetXYZ(xyz);
                }
            }
        }

        private static void ParseLimit(XElement limitElement, Limit limit)
        {
            string lowerAttr = limitElement.Attribute("lower")?.Value;
            if (!string.IsNullOrEmpty(lowerAttr) && double.TryParse(lowerAttr, out double lower))
            {
                limit.Lower = lower;
            }

            string upperAttr = limitElement.Attribute("upper")?.Value;
            if (!string.IsNullOrEmpty(upperAttr) && double.TryParse(upperAttr, out double upper))
            {
                limit.Upper = upper;
            }

            string effortAttr = limitElement.Attribute("effort")?.Value;
            if (!string.IsNullOrEmpty(effortAttr) && double.TryParse(effortAttr, out double effort))
            {
                limit.Effort = effort;
            }

            string velocityAttr = limitElement.Attribute("velocity")?.Value;
            if (!string.IsNullOrEmpty(velocityAttr) && double.TryParse(velocityAttr, out double velocity))
            {
                limit.Velocity = velocity;
            }
        }

        private static void ParseDynamics(XElement dynamicsElement, Dynamics dynamics)
        {
            string dampingAttr = dynamicsElement.Attribute("damping")?.Value;
            if (!string.IsNullOrEmpty(dampingAttr) && double.TryParse(dampingAttr, out double damping))
            {
                dynamics.Damping = damping;
            }

            string frictionAttr = dynamicsElement.Attribute("friction")?.Value;
            if (!string.IsNullOrEmpty(frictionAttr) && double.TryParse(frictionAttr, out double friction))
            {
                dynamics.Friction = friction;
            }
        }

        private static void ParseInertial(XElement inertialElement, Inertial inertial, List<string> warnings)
        {
            var originElement = inertialElement.Element("origin");
            if (originElement != null)
            {
                ParseOrigin(originElement, inertial.Origin);
            }

            var massElement = inertialElement.Element("mass");
            if (massElement != null)
            {
                string valueAttr = massElement.Attribute("value")?.Value;
                if (!string.IsNullOrEmpty(valueAttr) && double.TryParse(valueAttr, out double mass))
                {
                    inertial.Mass.Value = mass;
                }
            }

            var inertiaElement = inertialElement.Element("inertia");
            if (inertiaElement != null)
            {
                ParseInertia(inertiaElement, inertial.Inertia);
            }
        }

        private static void ParseInertia(XElement inertiaElement, Inertia inertia)
        {
            string ixxAttr = inertiaElement.Attribute("ixx")?.Value;
            if (!string.IsNullOrEmpty(ixxAttr) && double.TryParse(ixxAttr, out double ixx))
            {
                inertia.Ixx = ixx;
            }

            string ixyAttr = inertiaElement.Attribute("ixy")?.Value;
            if (!string.IsNullOrEmpty(ixyAttr) && double.TryParse(ixyAttr, out double ixy))
            {
                inertia.Ixy = ixy;
            }

            string ixzAttr = inertiaElement.Attribute("ixz")?.Value;
            if (!string.IsNullOrEmpty(ixzAttr) && double.TryParse(ixzAttr, out double ixz))
            {
                inertia.Ixz = ixz;
            }

            string iyyAttr = inertiaElement.Attribute("iyy")?.Value;
            if (!string.IsNullOrEmpty(iyyAttr) && double.TryParse(iyyAttr, out double iyy))
            {
                inertia.Iyy = iyy;
            }

            string iyzAttr = inertiaElement.Attribute("iyz")?.Value;
            if (!string.IsNullOrEmpty(iyzAttr) && double.TryParse(iyzAttr, out double iyz))
            {
                inertia.Iyz = iyz;
            }

            string izzAttr = inertiaElement.Attribute("izz")?.Value;
            if (!string.IsNullOrEmpty(izzAttr) && double.TryParse(izzAttr, out double izz))
            {
                inertia.Izz = izz;
            }
        }

        private static void ParseVisual(XElement visualElement, Visual visual, string meshBasePath, List<string> warnings)
        {
            var originElement = visualElement.Element("origin");
            if (originElement != null)
            {
                ParseOrigin(originElement, visual.Origin);
            }

            var geometryElement = visualElement.Element("geometry");
            if (geometryElement != null)
            {
                ParseGeometry(geometryElement, visual.Geometry, meshBasePath);
            }

            var materialElement = visualElement.Element("material");
            if (materialElement != null)
            {
                ParseMaterial(materialElement, visual.Material);
            }
        }

        private static void ParseCollision(XElement collisionElement, Collision collision, string meshBasePath, List<string> warnings)
        {
            var originElement = collisionElement.Element("origin");
            if (originElement != null)
            {
                ParseOrigin(originElement, collision.Origin);
            }

            var geometryElement = collisionElement.Element("geometry");
            if (geometryElement != null)
            {
                ParseGeometry(geometryElement, collision.Geometry, meshBasePath);
            }
        }

        private static void ParseGeometry(XElement geometryElement, Geometry geometry, string meshBasePath)
        {
            var meshElement = geometryElement.Element("mesh");
            if (meshElement != null)
            {
                string filename = meshElement.Attribute("filename")?.Value;
                if (!string.IsNullOrEmpty(filename))
                {
                    // Resolve relative paths if meshBasePath is provided
                    if (!string.IsNullOrEmpty(meshBasePath) && !Path.IsPathRooted(filename) && !filename.StartsWith("package://"))
                    {
                        filename = Path.Combine(meshBasePath, filename);
                    }
                    geometry.Mesh.Filename = filename;
                }

                string scaleAttr = meshElement.Attribute("scale")?.Value;
                if (!string.IsNullOrEmpty(scaleAttr))
                {
                    double[] scale = ParseDoubleArray(scaleAttr, 3);
                    if (scale != null)
                    {
                        // Use first component as uniform scale factor
                        geometry.MeshScaleFactor = scale[0];
                    }
                }
            }

            // Handle primitive geometries (box, cylinder, sphere)
            // These are stored for reference but won't map directly to NX bodies
            var boxElement = geometryElement.Element("box");
            if (boxElement != null)
            {
                // Box geometry - size attribute contains x, y, z dimensions
                // Not directly supported, would need to be created as NX features
            }

            var cylinderElement = geometryElement.Element("cylinder");
            if (cylinderElement != null)
            {
                // Cylinder geometry - radius and length attributes
            }

            var sphereElement = geometryElement.Element("sphere");
            if (sphereElement != null)
            {
                // Sphere geometry - radius attribute
            }
        }

        private static void ParseMaterial(XElement materialElement, Material material)
        {
            material.Name = materialElement.Attribute("name")?.Value ?? "";

            var colorElement = materialElement.Element("color");
            if (colorElement != null)
            {
                string rgbaAttr = colorElement.Attribute("rgba")?.Value;
                if (!string.IsNullOrEmpty(rgbaAttr))
                {
                    double[] rgba = ParseDoubleArray(rgbaAttr, 4);
                    if (rgba != null)
                    {
                        material.Color.SetColor(rgba);
                    }
                }
            }

            var textureElement = materialElement.Element("texture");
            if (textureElement != null)
            {
                string filename = textureElement.Attribute("filename")?.Value;
                if (!string.IsNullOrEmpty(filename))
                {
                    material.Texture.Filename = filename;
                }
            }
        }

        private static double[] ParseDoubleArray(string value, int expectedLength)
        {
            if (string.IsNullOrWhiteSpace(value))
                return null;

            string[] parts = value.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length != expectedLength)
                return null;

            double[] result = new double[expectedLength];
            for (int i = 0; i < expectedLength; i++)
            {
                if (!double.TryParse(parts[i], out result[i]))
                    return null;
            }

            return result;
        }

        #endregion

        #region Coordinate System Creation

        /// <summary>
        /// Represents a 4x4 homogeneous transformation matrix.
        /// </summary>
        private class Transform4x4
        {
            // Rotation matrix (3x3) stored as columns
            public double R00, R01, R02;  // First row
            public double R10, R11, R12;  // Second row
            public double R20, R21, R22;  // Third row

            // Translation vector
            public double Tx, Ty, Tz;

            public static Transform4x4 Identity()
            {
                return new Transform4x4
                {
                    R00 = 1,
                    R01 = 0,
                    R02 = 0,
                    R10 = 0,
                    R11 = 1,
                    R12 = 0,
                    R20 = 0,
                    R21 = 0,
                    R22 = 1,
                    Tx = 0,
                    Ty = 0,
                    Tz = 0
                };
            }

            /// <summary>
            /// Creates a Transform4x4 from a MathNet Matrix<double> (4x4 homogeneous transform).
            /// </summary>
            public static Transform4x4 FromMatrix(Matrix<double> matrix)
            {
                if (matrix == null || matrix.RowCount != 4 || matrix.ColumnCount != 4)
                {
                    return Identity();
                }

                return new Transform4x4
                {
                    R00 = matrix[0, 0],
                    R01 = matrix[0, 1],
                    R02 = matrix[0, 2],
                    R10 = matrix[1, 0],
                    R11 = matrix[1, 1],
                    R12 = matrix[1, 2],
                    R20 = matrix[2, 0],
                    R21 = matrix[2, 1],
                    R22 = matrix[2, 2],
                    Tx = matrix[0, 3],
                    Ty = matrix[1, 3],
                    Tz = matrix[2, 3]
                };
            }

            /// <summary>
            /// Creates a transform from XYZ position and RPY angles.
            /// URDF convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
            /// </summary>
            public static Transform4x4 FromXyzRpy(double x, double y, double z, double roll, double pitch, double yaw)
            {
                double cr = Math.Cos(roll);
                double sr = Math.Sin(roll);
                double cp = Math.Cos(pitch);
                double sp = Math.Sin(pitch);
                double cy = Math.Cos(yaw);
                double sy = Math.Sin(yaw);

                return new Transform4x4
                {
                    // R = Rz(yaw) * Ry(pitch) * Rx(roll)
                    R00 = cy * cp,
                    R01 = cy * sp * sr - sy * cr,
                    R02 = cy * sp * cr + sy * sr,

                    R10 = sy * cp,
                    R11 = sy * sp * sr + cy * cr,
                    R12 = sy * sp * cr - cy * sr,

                    R20 = -sp,
                    R21 = cp * sr,
                    R22 = cp * cr,

                    Tx = x,
                    Ty = y,
                    Tz = z
                };
            }

            /// <summary>
            /// Multiplies this transform by another: result = this * other
            /// </summary>
            public Transform4x4 Multiply(Transform4x4 other)
            {
                var result = new Transform4x4();

                // Rotation part: R_result = R_this * R_other
                result.R00 = R00 * other.R00 + R01 * other.R10 + R02 * other.R20;
                result.R01 = R00 * other.R01 + R01 * other.R11 + R02 * other.R21;
                result.R02 = R00 * other.R02 + R01 * other.R12 + R02 * other.R22;

                result.R10 = R10 * other.R00 + R11 * other.R10 + R12 * other.R20;
                result.R11 = R10 * other.R01 + R11 * other.R11 + R12 * other.R21;
                result.R12 = R10 * other.R02 + R11 * other.R12 + R12 * other.R22;

                result.R20 = R20 * other.R00 + R21 * other.R10 + R22 * other.R20;
                result.R21 = R20 * other.R01 + R21 * other.R11 + R22 * other.R21;
                result.R22 = R20 * other.R02 + R21 * other.R12 + R22 * other.R22;

                // Translation part: T_result = R_this * T_other + T_this
                result.Tx = R00 * other.Tx + R01 * other.Ty + R02 * other.Tz + Tx;
                result.Ty = R10 * other.Tx + R11 * other.Ty + R12 * other.Tz + Ty;
                result.Tz = R20 * other.Tx + R21 * other.Ty + R22 * other.Tz + Tz;

                return result;
            }

#if NX
            /// <summary>
            /// Gets the position as a Point3d (with unit conversion to mm).
            /// </summary>
            public Point3d GetPositionMm()
            {
                return new Point3d(Tx * METERS_TO_MM, Ty * METERS_TO_MM, Tz * METERS_TO_MM);
            }

            /// <summary>
            /// Gets the X direction vector.
            /// </summary>
            public Vector3d GetXDirection()
            {
                return new Vector3d(R00, R10, R20);
            }

            /// <summary>
            /// Gets the Y direction vector.
            /// </summary>
            public Vector3d GetYDirection()
            {
                return new Vector3d(R01, R11, R21);
            }
#endif

            /// <summary>
            /// Converts this Transform4x4 to a MathNet Matrix<double>.
            /// The matrix is in meters (URDF convention).
            /// </summary>
            public Matrix<double> ToMatrix()
            {
                var matrix = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.OfArray(new double[,]
                {
                    { R00, R01, R02, Tx },
                    { R10, R11, R12, Ty },
                    { R20, R21, R22, Tz },
                    { 0,   0,   0,   1  }
                });
                return matrix;
            }

            /// <summary>
            /// Transforms a direction vector (without translation) using this transform's rotation.
            /// </summary>
            public double[] TransformDirection(double[] direction)
            {
                if (direction == null || direction.Length != 3)
                    return direction;

                // Apply only the rotation part (3x3) to the direction vector
                double x = R00 * direction[0] + R01 * direction[1] + R02 * direction[2];
                double y = R10 * direction[0] + R11 * direction[1] + R12 * direction[2];
                double z = R20 * direction[0] + R21 * direction[1] + R22 * direction[2];

                // Normalize the result
                double length = Math.Sqrt(x * x + y * y + z * z);
                if (length > 1e-10)
                {
                    x /= length;
                    y /= length;
                    z /= length;
                }

                return new double[] { x, y, z };
            }
        }

        /// <summary>
        /// Creates coordinate systems at each joint origin in the link tree.
        /// Joint origins in URDF are relative to parent link frames, so we compute
        /// cumulative transforms from the root.
        /// Uses the current WCS as the base transform for all created coordinate systems.
        /// </summary>
        /// <param name="cadBridge">The CAD bridge to use for creating coordinate systems and axes.</param>
        /// <param name="rootLink">The root link of the robot tree.</param>
        /// <returns>Dictionary mapping joint names to CSYS keys.</returns>
        public static Dictionary<string, string> CreateCoordinateSystemsForJoints(CADBridge cadBridge, Link rootLink)
        {
            return CreateCoordinateSystemsForJoints(cadBridge, rootLink, null);
        }

        public static Dictionary<string, string> CreateCoordinateSystemsForJoints(CADBridge cadBridge, Link rootLink, Matrix<double> baseTransformOverride)
        {
            var createdCsys = new Dictionary<string, string>();

            // Base frame for the whole robot: caller-supplied override, else the current WCS
            Matrix<double> wcsMatrix = baseTransformOverride ?? cadBridge.GetWorkCoordinateSystemTransform();

            // Convert Matrix<double> to Transform4x4
            Transform4x4 rootTransform = Transform4x4.FromMatrix(wcsMatrix);

            logger.Information($"Using {(baseTransformOverride != null ? "specified CSYS" : "WCS")} as root transform - Origin: ({rootTransform.Tx:F4}, {rootTransform.Ty:F4}, {rootTransform.Tz:F4}) m");

            // Create the base/world CSYS at the root link position
            try
            {
                string baseCsysName = rootLink.Name;
                string baseCsysKey = cadBridge.CreateCoordinateSystemFromTransform(wcsMatrix, baseCsysName);
                if (!string.IsNullOrEmpty(baseCsysKey))
                {
                    createdCsys[baseCsysName] = baseCsysKey;
                    rootLink.Joint.CoordinateSystemName = baseCsysKey;
                    logger.Information($"Created base/world CSYS '{baseCsysName}' with key {baseCsysKey}");
                }
            }
            catch (Exception ex)
            {
                logger.Warning($"Failed to create base/world CSYS for '{rootLink.Name}': {ex.Message}");
            }

            CreateCoordinateSystemsRecursive(cadBridge, rootLink, rootTransform, createdCsys);
            return createdCsys;
        }

        private static void CreateCoordinateSystemsRecursive(
            CADBridge cadBridge,
            Link link,
            Transform4x4 parentWorldTransform,
            Dictionary<string, string> createdCsys)
        {
            // Compute this link's world transform
            Transform4x4 linkWorldTransform = parentWorldTransform;

            // If this link has a joint, the joint origin defines the transform from parent to this link
            if (!link.IsBaseLink && link.Joint != null && !string.IsNullOrEmpty(link.Joint.Name))
            {
                // Get the joint's local transform (relative to parent)
                double[] xyz = link.Joint.Origin.GetXYZ();
                double[] rpy = link.Joint.Origin.GetRPY();

                Transform4x4 jointLocalTransform = Transform4x4.FromXyzRpy(
                    xyz[0], xyz[1], xyz[2],
                    rpy[0], rpy[1], rpy[2]);

                // Compute world transform: parent * local
                linkWorldTransform = parentWorldTransform.Multiply(jointLocalTransform);

                // Create the CSYS at the world position/orientation using CADBridge
                try
                {
                    // Convert Transform4x4 to Matrix<double> for the bridge
                    var transformMatrix = linkWorldTransform.ToMatrix();
                    string csysKey = cadBridge.CreateCoordinateSystemFromTransform(transformMatrix, link.Joint.Name);

                    if (!string.IsNullOrEmpty(csysKey))
                    {
                        createdCsys[link.Joint.Name] = csysKey;

                        // Store the CSYS reference in the joint
                        link.Joint.CoordinateSystemName = csysKey;

                        logger.Debug($"Created CSYS for joint '{link.Joint.Name}' with key {csysKey}");

                        // Handle joint axis for revolute/prismatic/continuous joints
                        double[] jointAxis = link.Joint.Axis.GetXYZ();
                        if (jointAxis != null && (link.Joint.Type == "revolute" || link.Joint.Type == "prismatic" || link.Joint.Type == "continuous"))
                        {
                            // Check if the LOCAL axis (in joint frame) is aligned with a principal axis
                            // The magic keywords ($CSYS_X_AXIS, etc.) refer to the LOCAL coordinate system axes,
                            // not world axes, so we check the original jointAxis, not a world-transformed version
                            var (axisKeyword, flipAxis) = GetPrincipalAxisAlignment(jointAxis);

                            if (axisKeyword != null)
                            {
                                // Axis is colinear with a local principal axis - use magic keyword
                                link.Joint.AxisName = axisKeyword;
                                link.shouldFlipAxis = flipAxis;
                                logger.Debug($"Joint '{link.Joint.Name}' axis [{jointAxis[0]:F3}, {jointAxis[1]:F3}, {jointAxis[2]:F3}] is colinear with {axisKeyword}, FlipAxis={flipAxis}");
                            }
                            else
                            {
                                // Axis is not colinear with any principal axis - create a datum axis
                                // For datum axes, we DO need to transform to world space
                                double[] worldAxis = linkWorldTransform.TransformDirection(jointAxis);
                                double[] worldOrigin = new double[] {
                                    linkWorldTransform.Tx,
                                    linkWorldTransform.Ty,
                                    linkWorldTransform.Tz
                                };

                                string axisKey = cadBridge.CreateAxisFromDirection(worldOrigin, worldAxis, link.Joint.Name);
                                if (!string.IsNullOrEmpty(axisKey))
                                {
                                    link.Joint.AxisName = axisKey;
                                    link.shouldFlipAxis = false;
                                    logger.Debug($"Created Axis for joint '{link.Joint.Name}' with key {axisKey}");
                                }
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    logger.Warning($"Failed to create CSYS for joint '{link.Joint.Name}': {ex.Message}");
                }
            }

            // Recurse to children, passing this link's world transform
            foreach (var child in link.Children)
            {
                CreateCoordinateSystemsRecursive(cadBridge, child, linkWorldTransform, createdCsys);
            }
        }

        #endregion

        #region Utility Methods

        /// <summary>
        /// Determines if a direction vector is aligned with a principal axis (X, Y, or Z).
        /// Returns the appropriate magic keyword and whether the axis should be flipped.
        /// </summary>
        /// <param name="direction">Normalized direction vector [x, y, z].</param>
        /// <param name="tolerance">Tolerance for considering alignment (default 0.001).</param>
        /// <returns>Tuple of (axisKeyword, flipAxis) where axisKeyword is null if not aligned with any principal axis.</returns>
        private static (string axisKeyword, bool flipAxis) GetPrincipalAxisAlignment(double[] direction, double tolerance = 0.001)
        {
            if (direction == null || direction.Length != 3)
                return (null, false);

            double absX = Math.Abs(direction[0]);
            double absY = Math.Abs(direction[1]);
            double absZ = Math.Abs(direction[2]);

            // Check if aligned with X axis
            if (absX > 1.0 - tolerance && absY < tolerance && absZ < tolerance)
            {
                bool flip = direction[0] < 0;
                return (Joint.AxisFromCsysX, flip);
            }

            // Check if aligned with Y axis
            if (absY > 1.0 - tolerance && absX < tolerance && absZ < tolerance)
            {
                bool flip = direction[1] < 0;
                return (Joint.AxisFromCsysY, flip);
            }

            // Check if aligned with Z axis
            if (absZ > 1.0 - tolerance && absX < tolerance && absY < tolerance)
            {
                bool flip = direction[2] < 0;
                return (Joint.AxisFromCsysZ, flip);
            }

            // Not aligned with any principal axis
            return (null, false);
        }

        /// <summary>
        /// Counts the total number of links in the tree.
        /// </summary>
        private static int CountLinks(Link rootLink)
        {
            if (rootLink == null)
                return 0;

            int count = 1;
            foreach (var child in rootLink.Children)
            {
                count += CountLinks(child);
            }
            return count;
        }

        /// <summary>
        /// Gets a flat list of all links in the tree.
        /// </summary>
        public static List<Link> GetAllLinks(Link rootLink)
        {
            var links = new List<Link>();
            CollectLinksRecursive(rootLink, links);
            return links;
        }

        private static void CollectLinksRecursive(Link link, List<Link> links)
        {
            if (link == null)
                return;

            links.Add(link);
            foreach (var child in link.Children)
            {
                CollectLinksRecursive(child, links);
            }
        }

        /// <summary>
        /// Gets a flat list of all joints in the tree.
        /// </summary>
        public static List<Joint> GetAllJoints(Link rootLink)
        {
            var joints = new List<Joint>();
            CollectJointsRecursive(rootLink, joints);
            return joints;
        }

        private static void CollectJointsRecursive(Link link, List<Joint> joints)
        {
            if (link == null)
                return;

            if (!link.IsBaseLink && link.Joint != null && !string.IsNullOrEmpty(link.Joint.Name))
            {
                joints.Add(link.Joint);
            }

            foreach (var child in link.Children)
            {
                CollectJointsRecursive(child, joints);
            }
        }

        /// <summary>
        /// Validates a URDF file and returns any errors/warnings.
        /// </summary>
        public static List<string> ValidateUrdfFile(string urdfFilePath)
        {
            var issues = new List<string>();

            try
            {
                if (!File.Exists(urdfFilePath))
                {
                    issues.Add($"File not found: {urdfFilePath}");
                    return issues;
                }

                XDocument doc;
                try
                {
                    doc = XDocument.Load(urdfFilePath);
                }
                catch (Exception ex)
                {
                    issues.Add($"Failed to parse XML: {ex.Message}");
                    return issues;
                }

                if (doc.Root == null || doc.Root.Name.LocalName != "robot")
                {
                    issues.Add("Root element must be <robot>");
                    return issues;
                }

                var links = doc.Root.Elements("link").ToList();
                var joints = doc.Root.Elements("joint").ToList();

                if (links.Count == 0)
                {
                    issues.Add("No <link> elements found");
                }

                var linkNames = new HashSet<string>();
                foreach (var link in links)
                {
                    string name = link.Attribute("name")?.Value;
                    if (string.IsNullOrEmpty(name))
                    {
                        issues.Add("Found <link> without name attribute");
                    }
                    else if (!linkNames.Add(name))
                    {
                        issues.Add($"Duplicate link name: '{name}'");
                    }
                }

                var jointNames = new HashSet<string>();
                foreach (var joint in joints)
                {
                    string name = joint.Attribute("name")?.Value;
                    if (string.IsNullOrEmpty(name))
                    {
                        issues.Add("Found <joint> without name attribute");
                    }
                    else if (!jointNames.Add(name))
                    {
                        issues.Add($"Duplicate joint name: '{name}'");
                    }

                    string parentLink = joint.Element("parent")?.Attribute("link")?.Value;
                    string childLink = joint.Element("child")?.Attribute("link")?.Value;

                    if (string.IsNullOrEmpty(parentLink))
                    {
                        issues.Add($"Joint '{name}' missing parent link");
                    }
                    else if (!linkNames.Contains(parentLink))
                    {
                        issues.Add($"Joint '{name}' references unknown parent link '{parentLink}'");
                    }

                    if (string.IsNullOrEmpty(childLink))
                    {
                        issues.Add($"Joint '{name}' missing child link");
                    }
                    else if (!linkNames.Contains(childLink))
                    {
                        issues.Add($"Joint '{name}' references unknown child link '{childLink}'");
                    }
                }
            }
            catch (Exception ex)
            {
                issues.Add($"Validation error: {ex.Message}");
            }

            return issues;
        }

        /// <summary>
        /// Resolves a mesh filename from URDF to an absolute file path.
        /// Handles package:// URIs, relative paths, and absolute paths.
        /// </summary>
        private static string ResolveMeshPath(string meshFilename, string urdfDirectory, string meshBasePath)
        {
            if (string.IsNullOrEmpty(meshFilename))
                return null;

            string urdfParentDirectory = !string.IsNullOrEmpty(urdfDirectory)
                ? Path.GetDirectoryName(urdfDirectory)
                : null;

            // Handle package:// URIs
            if (meshFilename.StartsWith("package://"))
            {
                string afterScheme = meshFilename.Substring("package://".Length);
                // Extract package name and relative path within package
                int slashIndex = afterScheme.IndexOf('/');
                string packageName = slashIndex >= 0 ? afterScheme.Substring(0, slashIndex) : afterScheme;
                string packageRelative = slashIndex >= 0 ? afterScheme.Substring(slashIndex + 1) : "";

                // Try meshBasePath first
                if (!string.IsNullOrEmpty(meshBasePath))
                {
                    string resolved = Path.Combine(meshBasePath, packageRelative);
                    if (File.Exists(resolved))
                        return resolved;
                }

                // Walk up from URDF directory to find the package root (directory matching package name)
                if (!string.IsNullOrEmpty(urdfDirectory))
                {
                    string searchDir = urdfDirectory;
                    while (!string.IsNullOrEmpty(searchDir))
                    {
                        string dirName = Path.GetFileName(searchDir);
                        if (string.Equals(dirName, packageName, StringComparison.OrdinalIgnoreCase))
                        {
                            // Found the package root - resolve relative to it
                            string resolved = Path.Combine(searchDir, packageRelative);
                            if (File.Exists(resolved))
                                return resolved;
                            break;
                        }
                        string parent = Path.GetDirectoryName(searchDir);
                        if (parent == searchDir) break;
                        searchDir = parent;
                    }
                }

                // Fallback: try URDF directory and parent with just the relative path
                if (!string.IsNullOrEmpty(urdfDirectory))
                {
                    string resolved = Path.Combine(urdfDirectory, packageRelative);
                    if (File.Exists(resolved))
                        return resolved;
                }

                if (!string.IsNullOrEmpty(urdfParentDirectory))
                {
                    string resolved = Path.Combine(urdfParentDirectory, packageRelative);
                    if (File.Exists(resolved))
                        return resolved;
                }

                return null;
            }

            // Absolute path
            if (Path.IsPathRooted(meshFilename))
                return meshFilename;

            // Relative path - resolve from meshBasePath, URDF directory, then parent directory
            if (!string.IsNullOrEmpty(meshBasePath))
            {
                string resolved = Path.Combine(meshBasePath, meshFilename);
                if (File.Exists(resolved))
                    return resolved;
            }

            if (!string.IsNullOrEmpty(urdfDirectory))
            {
                string resolved = Path.Combine(urdfDirectory, meshFilename);
                if (File.Exists(resolved))
                    return resolved;
            }

            if (!string.IsNullOrEmpty(urdfParentDirectory))
            {
                string resolved = Path.Combine(urdfParentDirectory, meshFilename);
                if (File.Exists(resolved))
                    return resolved;
            }

            return meshFilename;
        }

        #endregion
    }
}
