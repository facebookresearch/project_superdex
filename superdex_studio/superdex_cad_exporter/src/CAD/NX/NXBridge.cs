/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if NX

using MathNet.Numerics.LinearAlgebra;
using NXOpen;
using NXOpen.Assemblies;
using NXOpen.Features;
using NXOpen.UF;
using CADRobotExporter.Export;
using CADRobotExporter.RobotDescription;
using CADRobotExporter.UI;
using CADRobotExporter.Utilities;
using System;
using System.Collections.Generic;
using System.IO;
using Joint = CADRobotExporter.RobotDescription.Joint;

namespace CADRobotExporter.CAD.NX
{
    /// <summary>
    /// NX implementation of the CADBridge abstract class.
    /// Provides NX-specific functionality for robot export operations.
    /// </summary>
    public class NXBridge : CADBridge
    {
        private static readonly Serilog.ILogger logger = Logger.GetLogger();

        // NX models are typically in millimeters, URDF expects meters
        private const double MM_TO_M = 0.001;
        private const double M_TO_MM = 1.0/MM_TO_M;
        // NX preferences for visualization are in inches and degrees
        private const double MM_TO_IN = 0.0393701;

        private Session session;
        private UFSession ufSession;
        private Part workPart;

        private ExporterConfiguration exporterConfiguration;
        private CustomFeature configurationFeature;
        private string latestExportLocation;

        private Dictionary<string, string> topLevelCoordinateSystems;
        private Dictionary<string, string> topLevelCoordinateSystemsForCadMeshExport;
        private Dictionary<string, double[]> topLevelPointCoordinates;
        private List<Tag> tempTopLevelCoordinateSystems;
        private List<Tag> tempTopLevelCoordinateSystemsForCadMeshExport;

        private List<Body> highlightedBodies;
        private CartesianCoordinateSystem highlightedCsys;
        private DatumAxis highlightedAxis;

        private NXJointVisualizer jointVisualizer = new NXJointVisualizer();
        private NXTendonVisualizer tendonVisualizer = new NXTendonVisualizer();
        private NXInertialVisualizer inertialVisualizer = new NXInertialVisualizer();
        private double[] cachedBoundingBox;

        private NXOpen.Preferences.PartVisualizationShade.ShadedViewToleranceType originalShadedViewToleranceType;
        private double originalEdgeTolerance;
        private double originalFaceTolerance;
        private double originalAngleTolerance;
        private bool originalAlignFacetAlongEdges;
        private NXOpen.Display.FacetSettingsBuilder.FacetScale originalFacetScale;

        public NXBridge()
        {
            session = Session.GetSession();
            ufSession = UFSession.GetUFSession();
            workPart = session.Parts.Work;

            exporterConfiguration = new ExporterConfiguration();
            configurationFeature = null;
            latestExportLocation = "";

            topLevelCoordinateSystems = new Dictionary<string, string>();
            topLevelCoordinateSystemsForCadMeshExport = new Dictionary<string, string>();
            topLevelPointCoordinates = new Dictionary<string, double[]>();
            tempTopLevelCoordinateSystems = new List<Tag>();
            tempTopLevelCoordinateSystemsForCadMeshExport = new List<Tag>();

            highlightedBodies = new List<Body>();
        }

        public NXBridge(Part part) : this()
        {
            workPart = part;
        }

        /// <summary>
        /// Gets the appropriate body handles list from a Link based on component type.
        /// </summary>
        private List<string> GetHandlesForComponentType(Link link, Link.ComponentType type)
        {
            switch (type)
            {
                case Link.ComponentType.Visual:
                    return link.NXVisualBodiesHandles ?? new List<string>();
                case Link.ComponentType.Collision:
                    return link.NXCollisionBodiesHandles ?? new List<string>();
                case Link.ComponentType.Inertial:
                    return link.NXInertialBodiesHandles ?? new List<string>();
                default:
                    return new List<string>();
            }
        }

        /// <summary>
        /// Resolves body/component keys to Body objects using persistent GUID attributes.
        /// Component keys are expanded to their contained bodies.
        /// </summary>
        private List<Body> ResolveBodiesToList(List<string> keys)
        {
            return NXPersistentId.ResolveKeysToBodiesExpandingComponents(workPart, keys, true);
        }

        private CartesianCoordinateSystem ResolveCSYS(string path)
        {
            return NXPersistentId.FindCoordinateSystemByKey(workPart, path);
        }

        private DatumAxis ResolveAxis(string path)
        {
            return NXPersistentId.FindAxisByKey(workPart, path);
        }

        /// <summary>
        /// Gets Tag array from body list.
        /// </summary>
        private Tag[] GetBodyTags(List<Body> bodies)
        {
            Tag[] tags = new Tag[bodies.Count];
            for (int i = 0; i < bodies.Count; i++)
            {
                tags[i] = bodies[i].Tag;
            }
            return tags;
        }

        /// <summary>
        /// Builds a 4x4 transformation matrix from NX origin and orientation.
        /// </summary>
        private Matrix<double> BuildTransformMatrix(Point3d origin, Matrix3x3 orientation)
        {
            var matrix = Matrix<double>.Build.Dense(4, 4);

            // Rotation part (3x3)
            matrix[0, 0] = orientation.Xx; matrix[1, 0] = orientation.Xy; matrix[2, 0] = orientation.Xz;
            matrix[0, 1] = orientation.Yx; matrix[1, 1] = orientation.Yy; matrix[2, 1] = orientation.Yz;
            matrix[0, 2] = orientation.Zx; matrix[1, 2] = orientation.Zy; matrix[2, 2] = orientation.Zz;

            // Translation part
            matrix[0, 3] = origin.X;
            matrix[1, 3] = origin.Y;
            matrix[2, 3] = origin.Z;

            // Homogeneous row
            matrix[3, 3] = 1.0;

            return matrix;
        }

        /// <summary>
        /// Resolves a Handle to a CartesianCoordinateSystem
        /// </summary>
        private CartesianCoordinateSystem ResolveCoordinateSystemByHandle(string handle)
        {
            Tag csysTag = ufSession.Tag.AskTagOfHandle(handle);

            if (csysTag == Tag.Null)
            {
                logger.Error($"Could not resolve coordinate system with Handle: {handle}");
                return null;
            }

            var obj = NXOpen.Utilities.NXObjectManager.Get(csysTag);
            if (obj is CartesianCoordinateSystem csys)
            {
                return csys;
            }
            else
            {
                logger.Error($"Could not retrive coordinate system with Handle: {handle}");
                return null;
            }
        }

        /// <summary>
        /// Gets the component that owns this object, if any.
        /// </summary>
        private Component GetOwningComponent(NXObject obj)
        {
            try
            {
                if (obj.OwningPart != workPart && obj.OwningPart != null)
                {
                    foreach (Component comp in workPart.ComponentAssembly.RootComponent.GetChildren())
                    {
                        if (comp.Prototype.OwningPart == obj.OwningPart)
                        {
                            return comp;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                logger.Warning($"Failed to get owning component: {ex.Message}");
            }

            return null;
        }

        public override string GetDocumentTitle()
        {
            if (workPart == null)
                return "Untitled";

            return workPart.Name ?? "Untitled";
        }

        public override ExporterConfiguration GetExporterConfiguration()
        {
            return exporterConfiguration;
        }

        public void SetExporterConfiguration(ExporterConfiguration config)
        {
            exporterConfiguration = config ?? new ExporterConfiguration();
        }

        public override void SetLatestExportLocation(string location)
        {
            latestExportLocation = location ?? "";
        }

        public override string GetLatestExportLocation()
        {
            return latestExportLocation ?? "";
        }

        /// <summary>
        /// Sets the existing configuration feature for updates.
        /// </summary>
        public void SetConfigurationFeature(CustomFeature feature)
        {
            configurationFeature = feature;
        }

        /// <summary>
        /// Gets the current configuration feature.
        /// </summary>
        public CustomFeature GetConfigurationFeature()
        {
            return configurationFeature;
        }

        public override void SaveConfigurationFromTree(
            ExporterConfiguration config,
            LinkNode baseNode,
            bool warnUser,
            string featureNameOverride = null)
        {
            if (workPart == null)
            {
                logger.Warning("Cannot save configuration - no work part");
                return;
            }

            Link baseLink = baseNode?.Link;
            if (baseLink == null)
            {
                logger.Warning("Cannot save configuration - no base link");
                return;
            }

            string featureName = featureNameOverride;
            if (string.IsNullOrEmpty(featureName))
            {
                if (config != null && !string.IsNullOrEmpty(config.robotName))
                {
                    featureName = $"Robot Configuration ({config.robotName})";
                }
                else
                {
                    featureName = $"Robot Configuration";
                }
            }

            try
            {
                // Temporary hack to save tendons
                List<Tendon> existingTendons = null;
                if (configurationFeature != null)
                {
                    NXConfigurationSerialization.LoadConfiguration(configurationFeature, out _, out existingTendons);
                }

                CustomFeature newFeature = NXConfigurationSerialization.SaveConfiguration(
                    workPart,
                    baseLink,
                    config,
                    configurationFeature,
                    featureName,
                    existingTendons);

                if (newFeature != null)
                {
                    configurationFeature = newFeature;
                    exporterConfiguration = config;
                    logger.Information($"Saved configuration to feature: {newFeature.Name}");
                }
                else
                {
                    logger.Error("Failed to save configuration feature");
                }
            }
            catch (Exception ex)
            {
                logger.Error($"Error saving configuration: {ex.Message}");
            }
        }

        /// <summary>
        /// Saves the configuration from a Link directly (for NX which uses NXLinkNode instead of LinkNode).
        /// </summary>
        public void SaveConfigurationFromLink(
            ExporterConfiguration config,
            Link baseLink,
            bool warnUser,
            string featureNameOverride = null,
            List<Tendon> tendons = null)
        {
            if (workPart == null)
            {
                logger.Warning("Cannot save configuration - no work part");
                return;
            }

            if (baseLink == null)
            {
                logger.Warning("Cannot save configuration - no base link");
                return;
            }

            string featureName = featureNameOverride;
            if (string.IsNullOrEmpty(featureName))
            {
                if (config != null && !string.IsNullOrEmpty(config.robotName))
                {
                    featureName = $"Robot Configuration ({config.robotName})";
                }
                else
                {
                    featureName = $"Robot Configuration";
                }
            }

            try
            {
                CustomFeature newFeature = NXConfigurationSerialization.SaveConfiguration(
                    workPart,
                    baseLink,
                    config,
                    configurationFeature,
                    featureName,
                    tendons);

                if (newFeature != null)
                {
                    configurationFeature = newFeature;
                    exporterConfiguration = config;
                    logger.Information($"Saved configuration to feature: {newFeature.Name}");
                }
                else
                {
                    logger.Error("Failed to save configuration feature");
                }
            }
            catch (Exception ex)
            {
                logger.Error($"Error saving configuration: {ex.Message}");
            }
        }


        public override void SelectLinkComponents(LinkNode node)
        {
            if (node?.Link == null)
                return;

            try
            {
                foreach (var body in highlightedBodies)
                {
                    body.Unhighlight();
                }

                List<string> handles = node.Link.NXVisualBodiesHandles;
                List<Body> bodies = ResolveBodiesToList(handles);

                handles = node.Link.NXInertialBodiesHandles;
                bodies.AddRange(ResolveBodiesToList(handles));

                foreach (var body in bodies)
                {
                    body.Highlight();
                }

                highlightedBodies = bodies;

                HighlightJointAxisAndCsys(node.Link);

                workPart.Views.Refresh();
            }
            catch (Exception ex)
            {
                logger.Warning($"SelectLinkComponents failed: {ex.Message}");
            }
        }

        public override void SelectJointComponents(LinkNode node)
        {
            SelectLinkComponents(node);
        }

        public void HighlightJointAxisAndCsys(Link link)
        {

            if (highlightedCsys != null)
            {
                highlightedCsys.Unhighlight();
            }

            var csys = ResolveCSYS(link.Joint.CoordinateSystemName);
            if (csys != null)
            {
                csys.Highlight();
                highlightedCsys = csys;
            }

            if (highlightedAxis != null)
            {
                highlightedAxis.Unhighlight();
            }

            var axis = ResolveAxis(link.Joint.AxisName);
            if (axis != null)
            {
                axis.Highlight();
                highlightedAxis = axis;
            }
        }

        public override void UnselectAll()
        {
            foreach (var body in highlightedBodies)
            {
                body.Unhighlight();
            }

            highlightedCsys?.Unhighlight();
            highlightedAxis?.Unhighlight();
        }

        public override void RestoreHostForeground()
        {
            // NX manages its own window activation via ExportRobotDialog; nothing to do here.
        }

        public override void HideAllComponents()
        {
            // Not required for NX
        }

        public override void ShowHiddenComponents()
        {
            // Not required for NX
        }

        public override void ShowHideVisualizations(bool show)
        {
            TriggerGraphicsRedraw();
        }

        public override void TriggerGraphicsRedraw()
        {
            try
            {
                if (workPart == null)
                    return;

                jointVisualizer.Clear(workPart);
                tendonVisualizer.Clear(workPart);
                inertialVisualizer.Clear(workPart);

                if (cachedBoundingBox == null)
                {
                    GetModelBoundingBox(out cachedBoundingBox);
                }
                if (IsShowingJointGizmo && CurrentNodeShown != null &&
                    CurrentNodeShown.Link.Joint != null)
                {
                    DrawJointGizmo_Internal();
                }

                if (IsShowingTendonVisualization && CurrentTendonShown != null)
                {
                    DrawTendonVisualization_Internal();
                }

                if (IsShowingAllTendons && AllTendons != null && AllTendons.Count > 0)
                {
                    DrawAllTendonsVisualization_Internal();
                }

                if (IsShowingInertialGizmo && CurrentLinkNodeShown != null)
                {
                    DrawInertialGizmo_Internal();
                }

                workPart.Views.Refresh();
            }
            catch (Exception ex)
            {
                logger.Warning($"Failed to refresh graphics: {ex.Message}");
            }
        }

        private void DrawJointGizmo_Internal()
        {
            var joint = CurrentNodeShown.Link.Joint;

            double[] center = GetJointOriginInModelSpace(joint);
            double[] axis = GetJointAxisInModelSpace(joint);

            double lowerLimit = joint.Limit.Lower;
            double upperLimit = joint.Limit.Upper;

            if (cachedBoundingBox == null)
            {
                GetModelBoundingBox(out cachedBoundingBox);
            }

            double xLen = Math.Abs(cachedBoundingBox[3] - cachedBoundingBox[0]);
            double yLen = Math.Abs(cachedBoundingBox[4] - cachedBoundingBox[1]);
            double zLen = Math.Abs(cachedBoundingBox[5] - cachedBoundingBox[2]);
            double averageLength = (xLen + yLen + zLen) / 3.0;

            double scale = JointGizmoScale > 0 ? JointGizmoScale : 0.5;
            double radius = scale * averageLength;

            double[] referenceDirection = GetCoMReferenceDirection(CurrentNodeShown.Link, center);

            if (joint.Type == "revolute")
            {
                jointVisualizer.DrawRevoluteJoint(workPart, center, axis,
                    lowerLimit, upperLimit, radius, referenceDirection);
            }
            else if (joint.Type == "prismatic")
            {
                jointVisualizer.DrawPrismaticJoint(workPart, center, axis,
                    lowerLimit, upperLimit, radius);
            }
        }

        private void DrawTendonVisualization_Internal()
        {
            var tendon = CurrentTendonShown;
            if (tendon.RoutingElements == null || tendon.RoutingElements.Count == 0)
                return;

            List<double[]> points = new List<double[]>();
            int highlightPointIndex = -1;

            for (int elemIdx = 0; elemIdx < tendon.RoutingElements.Count; elemIdx++)
            {
                var elem = tendon.RoutingElements[elemIdx];
                double[] pos = null;

                if (elem.Type == RobotDescription.RoutingElement.TypeWaypoint)
                {
                    if (!string.IsNullOrEmpty(elem.PointKey))
                    {
                        double[] globalMeters = GetTopLevelPointCoordinates(elem.PointKey);
                        if (globalMeters != null)
                        {
                            pos = new double[] {
                                globalMeters[0] * M_TO_MM,
                                globalMeters[1] * M_TO_MM,
                                globalMeters[2] * M_TO_MM
                            };
                        }
                    }
                }
                else if (elem.Type == RobotDescription.RoutingElement.TypeLinearJoint)
                {
                    if (TendonLinkCsysMap != null &&
                        !string.IsNullOrEmpty(elem.Link) &&
                        TendonLinkCsysMap.TryGetValue(elem.Link, out string csysName))
                    {
                        pos = GetLinkCsysOriginInModelSpace(csysName);
                    }
                }

                if (pos != null)
                {
                    if (elemIdx == TendonHighlightElementIndex)
                    {
                        highlightPointIndex = points.Count;
                    }
                    points.Add(pos);
                }
            }

            tendonVisualizer.DrawTendon(workPart, points, highlightPointIndex);
        }

        private void DrawAllTendonsVisualization_Internal()
        {
            for (int t = 0; t < AllTendons.Count; t++)
            {
                var tendon = AllTendons[t];
                if (tendon.RoutingElements == null || tendon.RoutingElements.Count == 0)
                    continue;

                List<double[]> points = new List<double[]>();

                for (int elemIdx = 0; elemIdx < tendon.RoutingElements.Count; elemIdx++)
                {
                    var elem = tendon.RoutingElements[elemIdx];
                    double[] pos = null;

                    if (elem.Type == RobotDescription.RoutingElement.TypeWaypoint)
                    {
                        if (!string.IsNullOrEmpty(elem.PointKey))
                        {
                            double[] globalMeters = GetTopLevelPointCoordinates(elem.PointKey);
                            if (globalMeters != null)
                            {
                                pos = new double[] {
                                    globalMeters[0] * M_TO_MM,
                                    globalMeters[1] * M_TO_MM,
                                    globalMeters[2] * M_TO_MM
                                };
                            }
                        }
                    }
                    else if (elem.Type == RobotDescription.RoutingElement.TypeLinearJoint)
                    {
                        if (TendonLinkCsysMap != null &&
                            !string.IsNullOrEmpty(elem.Link) &&
                            TendonLinkCsysMap.TryGetValue(elem.Link, out string csysName))
                        {
                            pos = GetLinkCsysOriginInModelSpace(csysName);
                        }
                    }

                    if (pos != null)
                    {
                        points.Add(pos);
                    }
                }

                int color = NXTendonVisualizer.GetPaletteColor(t);
                tendonVisualizer.DrawTendon(workPart, points, -1, color);
            }
        }

        private void DrawInertialGizmo_Internal()
        {
            var link = CurrentLinkNodeShown.Link;
            if (link.Inertial == null || link.Inertial.Mass.Value <= 1e-10)
                return;
            if (link.Joint == null || string.IsNullOrEmpty(link.Joint.CoordinateSystemName))
                return;

            double[] comLocal = link.Inertial.Origin.GetXYZ();
            double[] rpy = link.Inertial.Origin.GetRPY();

            var csys = ResolveCSYS(link.Joint.CoordinateSystemName);
            if (csys == null)
                return;

            Point3d origin = csys.Origin;
            Matrix3x3 orient = csys.Orientation.Element;

            // Transform CoM from link-local (meters) → model space (mm)
            double comMm0 = comLocal[0] * M_TO_MM;
            double comMm1 = comLocal[1] * M_TO_MM;
            double comMm2 = comLocal[2] * M_TO_MM;

            double comGlobalX = origin.X + orient.Xx * comMm0 + orient.Yx * comMm1 + orient.Zx * comMm2;
            double comGlobalY = origin.Y + orient.Xy * comMm0 + orient.Yy * comMm1 + orient.Zy * comMm2;
            double comGlobalZ = origin.Z + orient.Xz * comMm0 + orient.Yz * comMm1 + orient.Zz * comMm2;
            double[] comGlobal = new double[] { comGlobalX, comGlobalY, comGlobalZ };

            // CSYS rotation matrix (NX column-major orientation)
            double[,] rCsys = new double[,]
            {
                { orient.Xx, orient.Yx, orient.Zx },
                { orient.Xy, orient.Yy, orient.Zy },
                { orient.Xz, orient.Yz, orient.Zz }
            };

            // Compute principal inertia
            if (!MathOps.ComputePrincipalInertia(
                link.Inertial.Mass.Value,
                link.Inertial.Inertia.Ixx, link.Inertial.Inertia.Iyy, link.Inertial.Inertia.Izz,
                link.Inertial.Inertia.Ixy, link.Inertial.Inertia.Ixz, link.Inertial.Inertia.Iyz,
                out double[] boxHalfExtents, out double[,] principalRotation))
                return;

            // Convert box extents from meters to mm
            boxHalfExtents[0] *= M_TO_MM;
            boxHalfExtents[1] *= M_TO_MM;
            boxHalfExtents[2] *= M_TO_MM;

            double[,] composedRotation = MathOps.ComposeRotation(rCsys, rpy, principalRotation);

            if (cachedBoundingBox == null)
                GetModelBoundingBox(out cachedBoundingBox);

            double xLen = Math.Abs(cachedBoundingBox[3] - cachedBoundingBox[0]);
            double yLen = Math.Abs(cachedBoundingBox[4] - cachedBoundingBox[1]);
            double zLen = Math.Abs(cachedBoundingBox[5] - cachedBoundingBox[2]);
            double scale = LinkGizmoScale > 0 ? LinkGizmoScale : 1.0;
            double crossSize = (xLen + yLen + zLen) / 3.0 * 0.015 * scale;

            inertialVisualizer.DrawInertialGizmo(workPart, comGlobal, composedRotation, boxHalfExtents, crossSize);
        }

        private double[] GetLinkCsysOriginInModelSpace(string coordinateSystemName)
        {
            if (string.IsNullOrEmpty(coordinateSystemName))
                return new double[] { 0, 0, 0 };

            try
            {
                var csys = ResolveCSYS(coordinateSystemName);
                if (csys != null)
                {
                    Point3d origin = csys.Origin;
                    return new double[] { origin.X, origin.Y, origin.Z };
                }
            }
            catch (Exception ex)
            {
                logger.Warning($"GetLinkCsysOriginInModelSpace failed: {ex.Message}");
            }

            return new double[] { 0, 0, 0 };
        }

        private double[] GetCoMReferenceDirection(Link link, double[] jointCenter)
        {
            if (link.Inertial == null)
                return null;

            double[] comLocal = link.Inertial.Origin.GetXYZ();
            if (Math.Abs(comLocal[0]) < 1e-10 && Math.Abs(comLocal[1]) < 1e-10 && Math.Abs(comLocal[2]) < 1e-10)
                return null;

            try
            {
                var csys = ResolveCSYS(link.Joint.CoordinateSystemName);
                if (csys == null)
                    return null;

                Point3d origin = csys.Origin;
                Matrix3x3 orient = csys.Orientation.Element;

                // CoM is in meters (link-local frame), convert to mm and transform to model space
                double comMm0 = comLocal[0] * 1000.0;
                double comMm1 = comLocal[1] * 1000.0;
                double comMm2 = comLocal[2] * 1000.0;

                double comGlobalX = origin.X + orient.Xx * comMm0 + orient.Yx * comMm1 + orient.Zx * comMm2;
                double comGlobalY = origin.Y + orient.Xy * comMm0 + orient.Yy * comMm1 + orient.Zy * comMm2;
                double comGlobalZ = origin.Z + orient.Xz * comMm0 + orient.Yz * comMm1 + orient.Zz * comMm2;

                return new double[] {
                    comGlobalX - jointCenter[0],
                    comGlobalY - jointCenter[1],
                    comGlobalZ - jointCenter[2]
                };
            }
            catch
            {
                return null;
            }
        }

        private double[] GetJointAxisInModelSpace(Joint joint)
        {
            if (Joint.IsAxisFromCsys(joint.AxisName))
            {
                var csys = ResolveCSYS(joint.CoordinateSystemName);
                if (csys != null)
                {
                    Matrix3x3 orientation = csys.Orientation.Element;
                    if (joint.AxisName == Joint.AxisFromCsysX)
                        return new double[] { orientation.Xx, orientation.Xy, orientation.Xz };
                    else if (joint.AxisName == Joint.AxisFromCsysY)
                        return new double[] { orientation.Yx, orientation.Yy, orientation.Yz };
                    else
                        return new double[] { orientation.Zx, orientation.Zy, orientation.Zz };
                }
            }

            if (!string.IsNullOrEmpty(joint.AxisName))
                return GetAxisInGlobalSpace(joint.AxisName, false);

            return new double[] { 0, 0, 1 };
        }

        private double[] GetJointOriginInModelSpace(Joint joint)
        {
            if (string.IsNullOrEmpty(joint.CoordinateSystemName))
                return new double[] { 0, 0, 0 };

            try
            {
                var csys = ResolveCSYS(joint.CoordinateSystemName);
                if (csys != null)
                {
                    Point3d origin = csys.Origin;
                    return new double[] { origin.X, origin.Y, origin.Z };
                }
            }
            catch (Exception ex)
            {
                logger.Warning($"GetJointOriginInModelSpace failed: {ex.Message}");
            }

            return new double[] { 0, 0, 0 };
        }

        public override void BoostPerformance(bool enable)
        {
            // Not required for NX
        }

        public override void SetProgressBarStart(int maxProgress, string title)
        {
            logger.Information($"Progress: {title} (0/{maxProgress})");
        }

        public override void SetProgressBarEnd()
        {
            logger.Information("Progress: Complete");
        }

        public override void SetProgressBarTitle(string title)
        {
            logger.Information($"Progress: {title}");
        }

        public override void SetProgressBarProgress(int progress)
        {
        }


        public override bool GetComponentsInertialProperties(
            Link link,
            Link.ComponentType componentType,
            string coordinateSystemName,
            out double mass,
            out double[] centerOfMass,
            out double[] momentOfInertia)
        {
            mass = 0.0;
            centerOfMass = new double[] { 0, 0, 0 };
            momentOfInertia = new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

            List<string> handles = GetHandlesForComponentType(link, componentType);
            List<Body> bodies = ResolveBodiesToList(handles);

            if (bodies.Count == 0)
            {
                return true;
            }

            try
            {
                Tag[] bodyTags = GetBodyTags(bodies);
                int numBodies = bodyTags.Length;

                // Save current WCS so we can restore it later
                Tag savedWcsTag = Tag.Null;
                ufSession.Csys.AskWcs(out savedWcsTag);

                // If a coordinate system is specified, set the WCS to it
                // AskMassProps3d outputs values in WCS
                if (!string.IsNullOrEmpty(coordinateSystemName))
                {
                    // Get the top-level coordinate system (creates one in work part if needed)
                    // This is required because SetWcs cannot work with occurrence csys from lightweight components
                    string topLevlCsysHandle = GetTopLevelCoordinateSystem(coordinateSystemName);
                    CartesianCoordinateSystem csys = ResolveCoordinateSystemByHandle(topLevlCsysHandle);
                    if (csys != null)
                    {
                        ufSession.Csys.SetWcs(csys.Tag);
                    }
                }

                // AskMassProps3d parameters:
                // type: 1 = Solid Bodies, 2 = Thin Shell, 3 = Bounded by Sheet Bodies
                // units: 1 = lb/in, 2 = lb/ft, 3 = g/cm, 4 = kg/m
                // density: Not used for solid bodies (they use their own density)
                // accuracy: 1 = Use Accuracy, 2 = Use Relative Tolerances
                int type = 1;        // Solid bodies
                int units = 4;       // kg/m (SI units)
                double density = 1.0; // Not used for solid bodies
                int accuracyMode = 1; // Use accuracy value

                // acc_value[11]: [0] = accuracy (0.0 to 1.0), [1-10] for relative tolerances
                double[] accValue = new double[11];
                accValue[0] = 0.99; // High accuracy

                // Output arrays
                double[] massProps = new double[47];
                double[] statistics = new double[13];

                ufSession.Modl.AskMassProps3d(
                    bodyTags,
                    numBodies,
                    type,
                    units,
                    density,
                    accuracyMode,
                    accValue,
                    massProps,
                    statistics);

                // Restore original WCS
                if (savedWcsTag != Tag.Null)
                {
                    ufSession.Csys.SetWcs(savedWcsTag);
                }

                // 0   Surface Area
                // 1   Volume
                ////////// 2   Mass
                ////////// 3   CoM X, WCS
                ////////// 4   CoM Y, WCS
                ////////// 5   CoM Z, WCS
                // 6   First moments, WCS, Mxc
                // 7   First moments, WCS, Myc
                // 8   First moments, WCS, Mzc
                // 9   Moments of inertia, WCS, Ixxw
                // 10  Moments of inertia, WCS, Iyyw
                // 11  Moments of Inertia, WCS, Izzw
                ////////// 12  Moments of inertia, Centroidal, Ixx
                ////////// 13  Moments of inertia, Centroidal, Iyy
                ////////// 14  Moments of inertia, Centroidal, Izz
                // 15  Spherical moments of inertia
                // 16  Products of Inertia (WCS), Pyzw
                // 17  Products of Inertia (WCS), Pxzw
                // 18  Products of Inertia (WCS), Pxyw
                ////////// 19  Products of Inertia (Centroidal), Pyz
                ////////// 20  Products of Inertia (Centroidal), Pxz
                ////////// 21  Products of Inertia (Centroidal), Pxy
                // 22  Principal Axes, WCS, Xp(X)
                // 23  Principal Axes, WCS, Xp(Y)
                // 24  Principal Axes, WCS, Xp(Z)
                // 25  Principal Axes, WCS, Yp(X)
                // 26  Principal Axes, WCS, Yp(Y)
                // 27  Principal Axes, WCS, Yp(Z)
                // 28  Principal Axes, WCS, Zp(X)
                // 29  Principal Axes, WCS, Zp(Y)
                // 30  Principal Axes, WCS, Zp(Z)
                // 31  Principal Moments, Centroidal, Ixxp
                // 32  Principal Moments, Centroidal, Iyyp
                // 33  Principal Moments, Centroidal, Izzp
                // 34  Radii of gyration, WCS, Rgxw
                // 35  Radii of gyration, WCS, Rgyw
                // 36  Radii of gyration, WCS, Rgzw
                // 37  Radii of gyration, Centroidal, Rgxw
                // 38  Radii of gyration, Centroidal, Rgyw
                // 39  Radii of gyration, Centroidal, Rgzw
                // 40  Spherical radius of gyration
                // 41  Unused
                // 42  Unused
                // 43  Unused
                // 44  Unused
                // 45  Unused
                // 46  Density

                mass = massProps[2];

                centerOfMass[0] = massProps[3];
                centerOfMass[1] = massProps[4];
                centerOfMass[2] = massProps[5];

                // The system calculates the product of inertia of the specified region relative to the XC and YC axes using the formula:
                // Pxy = Integral(xy)dA
                // Where X and Y are distances from the YC and XC axes. The product of inertia is calculated relative to the centroidal axes.
                // This means positive convention/notation

                // Centroidal moments of inertia (diagonal)
                double Ixx = massProps[12];
                double Iyy = massProps[13];
                double Izz = massProps[14];
                // Centroidal products of inertia (off-diagonal)
                // Order in array: Pyz, Pxz, Pxy
                double Iyz = massProps[19];
                double Ixz = massProps[20];
                double Ixy = massProps[21];

                // Full 3x3 tensor - negative convention (row-major)
                // | Ixx  -Ixy  -Ixz |
                // |-Ixy   Iyy  -Iyz |
                // |-Ixz  -Iyz   Izz |
                momentOfInertia[0] = Ixx;  // [0,0]
                momentOfInertia[1] = -Ixy; // [0,1]
                momentOfInertia[2] = -Ixz; // [0,2]
                momentOfInertia[3] = -Ixy; // [1,0]
                momentOfInertia[4] = Iyy;  // [1,1]
                momentOfInertia[5] = -Iyz; // [1,2]
                momentOfInertia[6] = -Ixz; // [2,0]
                momentOfInertia[7] = -Iyz; // [2,1]
                momentOfInertia[8] = Izz;  // [2,2]

                logger.Information($"Calculated mass properties for {bodies.Count} bodies: mass={mass:F6}");
                return true;
            }
            catch (Exception ex)
            {
                logger.Error($"GetComponentsInertialProperties failed: {ex.Message}");
                return false;
            }
        }

        public override double[] GetVisualProperties(Link link, Link.ComponentType componentType)
        {
            double[] values = new double[] { 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 0.5, 1.0, 0.0 };

            try
            {
                List<string> handles = GetHandlesForComponentType(link, componentType);
                List<Body> bodies = ResolveBodiesToList(handles);

                if (bodies.Count > 0)
                {
                    Body firstBody = bodies[0];
                    Face[] faces = firstBody.GetFaces();

                    if (faces.Length > 0)
                    {
                        int colorIndex = faces[0].Color;

                        double[] rgb = new double[3];
                        ufSession.Disp.AskColor(colorIndex, UFConstants.UF_DISP_rgb_model, out string colorName, rgb);

                        values[0] = rgb[0];
                        values[1] = rgb[1];
                        values[2] = rgb[2];
                    }
                }
            }
            catch (Exception ex)
            {
                logger.Warning($"GetVisualProperties failed: {ex.Message}");
            }

            return values;
        }


        public override double[] GetAxisInGlobalSpace(string axisGuid, bool flipAxis)
        {
            double[] axisVector = new double[] { 0, 0, 1 };

            if (string.IsNullOrEmpty(axisGuid))
                return axisVector;

            try
            {
                DatumAxis datumAxis = NXPersistentId.FindAxisByKey(workPart, axisGuid);
                if (datumAxis == null)
                {
                    logger.Warning($"Could not resolve axis GUID: {axisGuid}");
                    return axisVector;
                }

                Point3d origin = datumAxis.Origin;
                Vector3d direction = datumAxis.Direction;

                axisVector[0] = direction.X;
                axisVector[1] = direction.Y;
                axisVector[2] = direction.Z;

                Component owningComponent = GetOwningComponent(datumAxis);
                if (owningComponent != null)
                {
                    double[] compOrigin = new double[3];
                    double[] compCsysMatrix = new double[9];
                    double[,] compTransform = new double[4, 4];
                    ufSession.Assem.AskComponentData(
                        owningComponent.Tag,
                        out string partName,
                        out string refsetName,
                        out string instanceName,
                        compOrigin,
                        compCsysMatrix,
                        compTransform);

                    // Use the 3x3 rotation matrix to transform the direction vector
                    // csys_matrix is row-major: [Xx, Xy, Xz, Yx, Yy, Yz, Zx, Zy, Zz]
                    double x = compCsysMatrix[0] * axisVector[0] + compCsysMatrix[1] * axisVector[1] + compCsysMatrix[2] * axisVector[2];
                    double y = compCsysMatrix[3] * axisVector[0] + compCsysMatrix[4] * axisVector[1] + compCsysMatrix[5] * axisVector[2];
                    double z = compCsysMatrix[6] * axisVector[0] + compCsysMatrix[7] * axisVector[1] + compCsysMatrix[8] * axisVector[2];

                    axisVector[0] = x;
                    axisVector[1] = y;
                    axisVector[2] = z;
                }

                double length = Math.Sqrt(axisVector[0] * axisVector[0] + axisVector[1] * axisVector[1] + axisVector[2] * axisVector[2]);
                if (length > 0)
                {
                    axisVector[0] /= length;
                    axisVector[1] /= length;
                    axisVector[2] /= length;
                }

                if (flipAxis)
                {
                    axisVector[0] = -axisVector[0];
                    axisVector[1] = -axisVector[1];
                    axisVector[2] = -axisVector[2];
                }
            }
            catch (Exception ex)
            {
                logger.Warning($"GetAxisInGlobalSpace failed: {ex.Message}");
            }

            return axisVector;
        }

        /// <summary>
        /// Gets the Work Coordinate System (WCS) transformation matrix.
        /// </summary>
        /// <returns>The 4x4 transformation matrix of the WCS (in meters).</returns>
        public override Matrix<double> GetWorkCoordinateSystemTransform()
        {
            try
            {
                Tag wcsTag = Tag.Null;
                ufSession.Csys.AskWcs(out wcsTag);

                if (wcsTag == Tag.Null)
                {
                    logger.Warning("No WCS found, returning identity transform");
                    return Matrix<double>.Build.DenseIdentity(4, 4);
                }

                // Get the WCS csys info
                double[] csysOrigin = new double[3];
                double[] csysMatrix = new double[9];
                ufSession.Csys.AskCsysInfo(wcsTag, out Tag matrixTag, csysOrigin);

                if (matrixTag != Tag.Null)
                {
                    ufSession.Csys.AskMatrixValues(matrixTag, csysMatrix);
                }
                else
                {
                    // Default to identity rotation
                    csysMatrix[0] = 1; csysMatrix[1] = 0; csysMatrix[2] = 0;
                    csysMatrix[3] = 0; csysMatrix[4] = 1; csysMatrix[5] = 0;
                    csysMatrix[6] = 0; csysMatrix[7] = 0; csysMatrix[8] = 1;
                }

                // Build transformation matrix (origin is in mm, convert to meters)
                // csysMatrix is row-major: [Xx, Xy, Xz, Yx, Yy, Yz, Zx, Zy, Zz]
                var matrix = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix.OfArray(new double[,]
                {
                    { csysMatrix[0], csysMatrix[3], csysMatrix[6], csysOrigin[0] * MM_TO_M },
                    { csysMatrix[1], csysMatrix[4], csysMatrix[7], csysOrigin[1] * MM_TO_M },
                    { csysMatrix[2], csysMatrix[5], csysMatrix[8], csysOrigin[2] * MM_TO_M },
                    { 0,             0,             0,             1 }
                });

                return matrix;
            }
            catch (Exception ex)
            {
                logger.Error($"GetWorkCoordinateSystemTransform failed: {ex.Message}");
                return Matrix<double>.Build.DenseIdentity(4, 4);
            }
        }

        public override Matrix<double> GetCoordinateSystemTransform(string coordinateSystemGuid)
        {
            if (string.IsNullOrEmpty(coordinateSystemGuid))
            {
                return Matrix<double>.Build.DenseIdentity(4, 4);
            }

            try
            {
                string topLevelCsysHandle = GetTopLevelCoordinateSystem(coordinateSystemGuid);
                Tag csysTag = ufSession.Tag.AskTagOfHandle(topLevelCsysHandle);

                if (csysTag == Tag.Null)
                {
                    logger.Error($"Could not resolve coordinate system with Handle: {topLevelCsysHandle} from key {coordinateSystemGuid}");
                    return Matrix<double>.Build.DenseIdentity(4, 4);
                }

                var obj = NXOpen.Utilities.NXObjectManager.Get(csysTag);
                if (obj is CartesianCoordinateSystem csys)
                {
                    return GetCoordinateSystemTransform(csys);
                }
                else
                {
                    logger.Error($"Could not retrieve CSYS from Handle: {topLevelCsysHandle} from key {coordinateSystemGuid}");
                    return Matrix<double>.Build.DenseIdentity(4, 4);
                }
            }
            catch (Exception ex)
            {
                logger.Error($"GetCoordinateSystemTransform failed: {ex.Message}");
            }

            return Matrix<double>.Build.DenseIdentity(4, 4);
        }

        /// <summary>
        /// Gets the transformation matrix (in meters) for an already-resolved coordinate system.
        /// </summary>
        public Matrix<double> GetCoordinateSystemTransform(CartesianCoordinateSystem csys)
        {
            if (csys == null)
            {
                return Matrix<double>.Build.DenseIdentity(4, 4);
            }

            Point3d origin = csys.Origin;

            // Convert origin from mm to m for URDF
            Point3d originMeters = new Point3d(
                origin.X * MM_TO_M,
                origin.Y * MM_TO_M,
                origin.Z * MM_TO_M);

            return BuildTransformMatrix(originMeters, csys.Orientation.Element);
        }

        // NB: GetTopLevelCoordinateSystem returns a Handle rather than GUID because
        // setting a User Attribute to the temporary CSYS seems to have no effect
        // or the retreival path doesn't work
        public override string GetTopLevelCoordinateSystem(string someCoordinateSystemGuid)
        {
            if (string.IsNullOrEmpty(someCoordinateSystemGuid))
                return someCoordinateSystemGuid;

            // Return cached top-level csys if we've already created one for this GUID
            if (topLevelCoordinateSystems.ContainsKey(someCoordinateSystemGuid))
            {
                return topLevelCoordinateSystems[someCoordinateSystemGuid];
            }

            // Always create a new coordinate system in the work part for WCS/STEP compatibility.
            // NX gives us top-level measurements directly from coordinate systems,
            // but the original csys cannot be used for setting the WCS (needed for
            // inertia calculations) or STEP export if it's an occurrence.
            // Using the NXOpen API (not UF) to create a proper csys in the work part.

            try
            {
                string topLevelCsysHandle = "";

                CartesianCoordinateSystem csys = NXPersistentId.FindCoordinateSystemByKey(workPart, someCoordinateSystemGuid);
                if (csys == null)
                {
                    logger.Warning($"Could not resolve coordinate system GUID: {someCoordinateSystemGuid}");
                    return someCoordinateSystemGuid;
                }

                // Get the origin and orientation - NX already provides these in top-level coordinates
                Point3d origin = csys.Origin;
                Matrix3x3 orientation = csys.Orientation.Element;

                // Extract X and Y direction vectors from the orientation matrix
                Vector3d xDirection = new Vector3d(orientation.Xx, orientation.Xy, orientation.Xz);
                Vector3d yDirection = new Vector3d(orientation.Yx, orientation.Yy, orientation.Yz);

                // Create an Xform from the origin point and direction vectors
                Xform xform = workPart.Xforms.CreateXform(
                    origin,
                    xDirection,
                    yDirection,
                    SmartObject.UpdateOption.WithinModeling,
                    1.0);

                // Create a coordinate system in the work part using the Xform
                CartesianCoordinateSystem newCsys = workPart.CoordinateSystems.CreateCoordinateSystem(
                    xform,
                    SmartObject.UpdateOption.WithinModeling);

                if (newCsys != null)
                {
                    ufSession.Tag.AskHandleFromTag(newCsys.Tag, out topLevelCsysHandle);
                    topLevelCoordinateSystems.Add(someCoordinateSystemGuid, topLevelCsysHandle);
                    tempTopLevelCoordinateSystems.Add(newCsys.Tag);

                    logger.Information($"Created work part CSYS for {someCoordinateSystemGuid} with Handle {topLevelCsysHandle}");
                }

                // For NX GLB export, NX erroneously exports these as Z-up, where GLB standard is Y-up
                xDirection = new Vector3d(orientation.Xx, orientation.Xy, orientation.Xz);
                yDirection = new Vector3d(orientation.Zx, orientation.Zy, orientation.Zz);

                xform = workPart.Xforms.CreateXform(
                    origin,
                    xDirection,
                    yDirection,
                    SmartObject.UpdateOption.WithinModeling,
                    1.0);

                CartesianCoordinateSystem meshCsys = workPart.CoordinateSystems.CreateCoordinateSystem(
                    xform,
                    SmartObject.UpdateOption.WithinModeling);

                if (meshCsys != null)
                {
                    ufSession.Tag.AskHandleFromTag(meshCsys.Tag, out string newMeshCsysHandle);
                    topLevelCoordinateSystemsForCadMeshExport.Add(someCoordinateSystemGuid, newMeshCsysHandle);
                    tempTopLevelCoordinateSystemsForCadMeshExport.Add(newCsys.Tag);

                    logger.Information($"Created work part CSYS for mesh export for {someCoordinateSystemGuid} with Handle {newMeshCsysHandle}");
                }

                return topLevelCsysHandle;
            }
            catch (Exception ex)
            {
                logger.Warning($"GetTopLevelCoordinateSystem failed: {ex.Message}");
            }

            return someCoordinateSystemGuid;
        }

        public override double[] GetTopLevelPointCoordinates(string pointKey)
        {
            if (string.IsNullOrEmpty(pointKey))
                return null;

            if (topLevelPointCoordinates.TryGetValue(pointKey, out double[] cached))
                return cached;

            try
            {
                Point point = NXPersistentId.FindPointByKey(workPart, pointKey);
                if (point == null)
                {
                    logger.Warning($"GetTopLevelPointCoordinates: Could not resolve point key: {pointKey}");
                    return null;
                }

                Point3d coords = point.Coordinates;
                double[] result = new double[]
                {
                    coords.X * MM_TO_M,
                    coords.Y * MM_TO_M,
                    coords.Z * MM_TO_M
                };

                topLevelPointCoordinates[pointKey] = result;
                return result;
            }
            catch (Exception ex)
            {
                logger.Warning($"GetTopLevelPointCoordinates failed for key {pointKey}: {ex.Message}");
                return null;
            }
        }

        private string GetTopLevelCoordinateSystemForMeshExport(string someCoordinateSystemGuid)
        {
            if (string.IsNullOrEmpty(someCoordinateSystemGuid))
            {
                logger.Error("Empty guid passed to GetTopLevelCoordinateSystemForMeshExport");
                return null;
            }

            if (string.IsNullOrEmpty(GetTopLevelCoordinateSystem(someCoordinateSystemGuid)))
            {
                logger.Error($"Could not create top level CSYS for mesh export from {someCoordinateSystemGuid}");
                return null;
            }

            if (!topLevelCoordinateSystemsForCadMeshExport.ContainsKey(someCoordinateSystemGuid))
            {
                logger.Error($"Could not get top level CSYS for mesh export from {someCoordinateSystemGuid}");
                return null;
            }

            return topLevelCoordinateSystemsForCadMeshExport[someCoordinateSystemGuid];
        }

        /// <summary>
        /// Creates a coordinate system from a 4x4 transformation matrix.
        /// </summary>
        /// <param name="transform">4x4 homogeneous transformation matrix (in meters).</param>
        /// <param name="name">Name for the coordinate system.</param>
        /// <returns>A key that can be used to reference the created coordinate system.</returns>
        public override string CreateCoordinateSystemFromTransform(Matrix<double> transform, string name)
        {
            if (workPart == null)
            {
                logger.Error("Cannot create coordinate system - no work part");
                return null;
            }

            try
            {
                // Extract origin from matrix column 3 (rows 0-2), convert meters to mm
                Point3d origin = new Point3d(
                    transform[0, 3] * M_TO_MM,
                    transform[1, 3] * M_TO_MM,
                    transform[2, 3] * M_TO_MM);

                // Extract X direction from matrix column 0 (rows 0-2)
                Vector3d xDirection = new Vector3d(
                    transform[0, 0],
                    transform[1, 0],
                    transform[2, 0]);

                // Extract Y direction from matrix column 1 (rows 0-2)
                Vector3d yDirection = new Vector3d(
                    transform[0, 1],
                    transform[1, 1],
                    transform[2, 1]);

                // Create an Xform from the origin and direction vectors
                Xform xform = workPart.Xforms.CreateXform(
                    origin,
                    xDirection,
                    yDirection,
                    SmartObject.UpdateOption.WithinModeling,
                    1.0);

                // Create the coordinate system
                CartesianCoordinateSystem csys = workPart.CoordinateSystems.CreateCoordinateSystem(
                    xform,
                    SmartObject.UpdateOption.WithinModeling);

                DatumCsysBuilder builder = workPart.Features.CreateDatumCsysBuilder(null);
                builder.Csys = csys;
                NXObject nxObj = builder.Commit();
                builder.Destroy();

                // csys.SetVisibility(SmartObject.VisibilityOption.Visible);

                if (nxObj is DatumCsys datumCsys)
                {
                    // Name the coordinate system for easy identification
                    if (datumCsys != null && !string.IsNullOrEmpty(name))
                    {
                        try
                        {
                            datumCsys.SetName($"URDF_{name}");
                        }
                        catch
                        {
                            // Name setting may fail in some contexts, ignore
                        }
                    }
                }

                // Return persistent key
                string csysKey = NXPersistentId.GetOrCreateCoordinateSystemKey(csys);
                logger.Information($"Created CSYS '{name}' at ({origin.X:F2}, {origin.Y:F2}, {origin.Z:F2}) mm with key {csysKey}");
                return csysKey;
            }
            catch (Exception ex)
            {
                logger.Error($"CreateCoordinateSystemFromTransform failed: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Creates a datum axis from a direction vector at an origin point.
        /// Only creates an axis if the direction is NOT colinear with principal axes (X, Y, Z).
        /// </summary>
        /// <param name="origin">Origin point [x, y, z] in meters.</param>
        /// <param name="direction">Direction vector [x, y, z] (normalized).</param>
        /// <param name="name">Name for the axis.</param>
        /// <returns>A key that can be used to reference the created axis, or null if not created.</returns>
        public override string CreateAxisFromDirection(double[] origin, double[] direction, string name)
        {
            if (workPart == null)
            {
                logger.Error("Cannot create datum axis - no work part");
                return null;
            }

            // Check if direction is colinear with principal axes
            if (IsColinearWithPrincipalAxis(direction))
            {
                logger.Information($"Axis '{name}' is colinear with principal axis, skipping datum axis creation");
                return null;
            }

            try
            {
                // Convert origin from meters to mm
                Point3d originPt = new Point3d(
                    origin[0] * M_TO_MM,
                    origin[1] * M_TO_MM,
                    origin[2] * M_TO_MM);

                Vector3d directionVec = new Vector3d(direction[0], direction[1], direction[2]);

                // Create datum axis using DatumAxisBuilder
                DatumAxisBuilder builder = workPart.Features.CreateDatumAxisBuilder(null);
                builder.Type = DatumAxisBuilder.Types.PointAndDir;
                builder.IsAssociative = false;

                // Set the point (origin)
                Point point = workPart.Points.CreatePoint(originPt);
                builder.Point = point;

                // Set the direction
                Direction dir = workPart.Directions.CreateDirection(originPt, directionVec, SmartObject.UpdateOption.WithinModeling);
                builder.Vector = dir;

                // Commit the feature
                NXObject nxObj = builder.Commit();
                builder.Destroy();

                if (nxObj is DatumAxisFeature datumAxisFeature)
                {
                    DatumAxis datumAxis = datumAxisFeature.DatumAxis;

                    // Name the axis
                    if (datumAxis != null && !string.IsNullOrEmpty(name))
                    {
                        try
                        {
                            datumAxisFeature.SetName($"URDF_{name}_axis");
                        }
                        catch
                        {
                            // Name setting may fail in some contexts, ignore
                        }
                    }

                    string axisKey = NXPersistentId.GetOrCreateAxisKey(datumAxis);
                    logger.Information($"Created DatumAxis '{name}' with direction ({direction[0]:F3}, {direction[1]:F3}, {direction[2]:F3}) with key {axisKey}");
                    return axisKey;
                }

                return null;
            }
            catch (Exception ex)
            {
                logger.Error($"CreateAxisFromDirection failed: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Checks if a direction vector is colinear with a principal axis (X, Y, or Z).
        /// </summary>
        private static bool IsColinearWithPrincipalAxis(double[] direction, double tolerance = 0.001)
        {
            double[] absDir = { Math.Abs(direction[0]), Math.Abs(direction[1]), Math.Abs(direction[2]) };
            // Check if one component is ~1 and others are ~0
            return (absDir[0] > 1.0 - tolerance && absDir[1] < tolerance && absDir[2] < tolerance) ||
                   (absDir[1] > 1.0 - tolerance && absDir[0] < tolerance && absDir[2] < tolerance) ||
                   (absDir[2] > 1.0 - tolerance && absDir[0] < tolerance && absDir[1] < tolerance);
        }

        public override void CleanUpTemporaryFeatures()
        {
            try
            {
                foreach (Tag featureTag in tempTopLevelCoordinateSystems)
                {
                    if (featureTag != Tag.Null)
                    {
                        var feature = NXOpen.Utilities.NXObjectManager.Get(featureTag) as NXOpen.Features.Feature;
                        if (feature != null)
                        {
                            session.UpdateManager.AddToDeleteList(feature);
                        }
                    }
                }

                foreach (Tag featureTag in tempTopLevelCoordinateSystemsForCadMeshExport)
                {
                    if (featureTag != Tag.Null)
                    {
                        var feature = NXOpen.Utilities.NXObjectManager.Get(featureTag) as NXOpen.Features.Feature;
                        if (feature != null)
                        {
                            session.UpdateManager.AddToDeleteList(feature);
                        }
                    }
                }

                if (tempTopLevelCoordinateSystems.Count > 0 || tempTopLevelCoordinateSystemsForCadMeshExport.Count > 0)
                {
                    session.UpdateManager.DoUpdate(new Session.UndoMarkId());
                    logger.Information($"Cleaned up {tempTopLevelCoordinateSystems.Count + tempTopLevelCoordinateSystemsForCadMeshExport.Count} temporary features");
                }

                tempTopLevelCoordinateSystems.Clear();
                tempTopLevelCoordinateSystemsForCadMeshExport.Clear();

                topLevelCoordinateSystems.Clear();
                topLevelCoordinateSystemsForCadMeshExport.Clear();
                topLevelPointCoordinates.Clear();
            }
            catch (Exception ex)
            {
                logger.Warning($"CleanUpTemporaryFeatures failed: {ex.Message}");
            }
        }

        public override void GetModelBoundingBox(out double[] boundingBox)
        {
            boundingBox = new double[] { -250, -250, -250, 250, 250, 250 };

            try
            {
                double minX = double.MaxValue, minY = double.MaxValue, minZ = double.MaxValue;
                double maxX = double.MinValue, maxY = double.MinValue, maxZ = double.MinValue;

                bool found = false;

                // NX's bounding boxes require querying the assembly tree and collecting bodies,
                // which is slow. Thus we do a crude approximation using CSYS from the part.
                // This works on the assumption that temporary + existing top level CSYS has been constructed.
                foreach (string handle in topLevelCoordinateSystems.Values)
                {
                    CartesianCoordinateSystem csys = ResolveCoordinateSystemByHandle(handle);
                    if (csys == null)
                        continue;

                    Point3d origin = csys.Origin;
                    minX = Math.Min(minX, origin.X);
                    minY = Math.Min(minY, origin.Y);
                    minZ = Math.Min(minZ, origin.Z);
                    maxX = Math.Max(maxX, origin.X);
                    maxY = Math.Max(maxY, origin.Y);
                    maxZ = Math.Max(maxZ, origin.Z);
                    found = true;
                }

                if (found)
                {
                    boundingBox = new double[] { minX, minY, minZ, maxX, maxY, maxZ };
                }
            }
            catch (Exception ex)
            {
                logger.Warning($"GetModelBoundingBox failed: {ex.Message}");
            }
        }

        public override bool SaveStpFile(Link link, Link.ComponentType componentType, string windowsMeshFilename)
        {
            List<string> handles = GetHandlesForComponentType(link, componentType);
            List<Body> bodies = ResolveBodiesToList(handles);

            if (bodies.Count == 0)
            {
                logger.Warning($"No bodies to export for link {link.Name}");
                return false;
            }

            string coordinateSystemName = link.Joint.CoordinateSystemName;
            string topLevelCsysHandle = GetTopLevelCoordinateSystem(coordinateSystemName);
            CartesianCoordinateSystem coordSys = ResolveCoordinateSystemByHandle(topLevelCsysHandle);

            try
            {
                logger.Information($"{link.Name}: Exporting STEP with {bodies.Count} bodies to {windowsMeshFilename}");

                StepCreator stepCreator = session.DexManager.CreateStepCreator();

                stepCreator.ExportAs = StepCreator.ExportAsOption.Ap242;
                stepCreator.ExportFrom = StepCreator.ExportFromOption.DisplayPart;

                stepCreator.ObjectTypes.Csys = true;
                stepCreator.ObjectTypes.Solids = true;
                stepCreator.ObjectTypes.Surfaces = true;

                stepCreator.InputFile = workPart.FullPath;
                stepCreator.OutputFile = windowsMeshFilename;

                if (bodies != null && bodies.Count > 0)
                {
                    stepCreator.ExportSelectionBlock.SelectionScope = ObjectSelector.Scope.SelectedObjects;
                    foreach (NXObject obj in bodies)
                    {
                        bool added = stepCreator.ExportSelectionBlock.SelectionComp.Add(obj);
                    }
                }
                else
                {
                    stepCreator.ExportSelectionBlock.SelectionScope = ObjectSelector.Scope.EntireAssembly;
                }
                stepCreator.FileSaveFlag = false;
                stepCreator.LayerMask = "1-256";
                stepCreator.ColorAndLayers = true;

                // NB: this causes NX to wait for the export process to finish first before returning
                // otherwise the export is async
                stepCreator.ProcessHoldFlag = true;

                if (coordSys != null)
                {
                    stepCreator.ReferenceType = StepCreator.CsysrefEnum.SpecifiedCsys;
                    stepCreator.Csys = coordSys;
                }

                NXObject stepObject = stepCreator.Commit();
                stepCreator.Destroy();

                bool fileExists = File.Exists(windowsMeshFilename);
                if (fileExists)
                {
                    logger.Information($"Successfully exported STEP file: {windowsMeshFilename}");
                }
                else
                {
                    logger.Warning($"STEP export completed but file not found: {windowsMeshFilename}");
                }

                return fileExists;
            }
            catch (Exception ex)
            {
                logger.Error($"SaveSTEPFile failed: {ex.Message}");
                return false;
            }
        }

        public override bool SaveObj(
            Link link,
            Link.ComponentType componentType,
            string windowsMeshFilename,
            ExporterMeshingOptions meshingOptions,
            bool exportingCollision)
        {
            List<string> handles = GetHandlesForComponentType(link, componentType);
            List<Body> bodies = ResolveBodiesToList(handles);

            if (bodies.Count == 0)
            {
                logger.Warning($"No bodies to export for link {link.Name}");
                return false;
            }

            string coordinateSystemName = link.Joint.CoordinateSystemName;
            string topLevelCsysHandle = GetTopLevelCoordinateSystem(coordinateSystemName);
            CartesianCoordinateSystem coordSys = ResolveCoordinateSystemByHandle(topLevelCsysHandle);

            Tag savedWcsTag = Tag.Null;
            ufSession.Csys.AskWcs(out savedWcsTag);

            if (coordSys != null)
            {
                ufSession.Csys.SetWcs(coordSys.Tag);
            }

            try
            {
                logger.Information($"{link.Name}: Exporting OBJ with {bodies.Count} bodies to {windowsMeshFilename}");

                WavefrontObjCreator objCreator = session.DexManager.CreateWavefrontObjCreator();

                objCreator.OutputFile = windowsMeshFilename;
                objCreator.ExportFrom = WavefrontObjCreator.ExportFromOption.DisplayPart;
                objCreator.ExportSelectionBlock.SelectionScope = ObjectSelector.Scope.SelectedObjects;
                foreach (NXObject obj in bodies)
                {
                    bool added = objCreator.ExportSelectionBlock.SelectionComp.Add(obj);
                }
                objCreator.ExportPositionReference = WavefrontObjCreator.ExportPositionReferenceOption.WorkCSYS;
                objCreator.ExportUnits = WavefrontObjCreator.UnitsEnum.Meters;
                objCreator.ProcessHoldFlag = true;

                double chordalTolerance = exportingCollision ? meshingOptions.collisionMeshingOptions.linearDeflection : meshingOptions.visualMeshingOptions.linearDeflection;
                chordalTolerance = chordalTolerance * MM_TO_M; // this depends on ExportUnits
                double angularTolerance = exportingCollision ? meshingOptions.collisionMeshingOptions.angularDeflection : meshingOptions.visualMeshingOptions.angularDeflection;
                angularTolerance = angularTolerance / Math.PI * 180.0;

                objCreator.ChordalTolerance = chordalTolerance;
                objCreator.AngularTolerance = angularTolerance;

                objCreator.Commit();
                objCreator.Destroy();

                // Restore original WCS
                if (savedWcsTag != Tag.Null)
                {
                    ufSession.Csys.SetWcs(savedWcsTag);
                }

                bool fileExists = File.Exists(windowsMeshFilename);

                if (fileExists)
                {
                    logger.Information($"Successfully exported OBJ file: {windowsMeshFilename}");
                }
                else
                {
                    logger.Warning($"OBJ export completed but file not found: {windowsMeshFilename}");
                }

                return fileExists;
            }
            catch (Exception ex)
            {
                logger.Error($"SaveOBJ failed: {ex.Message}");

                // Restore original WCS
                if (savedWcsTag != Tag.Null)
                {
                    ufSession.Csys.SetWcs(savedWcsTag);
                }

                return false;
            }


        }

        public override bool SaveGlb(
            Link link,
            Link.ComponentType componentType,
            string windowsMeshFilename,
            ExporterMeshingOptions meshingOptions,
            bool exportingCollision)
        {
            List<string> handles = GetHandlesForComponentType(link, componentType);
            List<Body> bodies = ResolveBodiesToList(handles);

            if (bodies.Count == 0)
            {
                logger.Warning($"No bodies to export for link {link.Name}");
                return false;
            }

            string coordinateSystemName = link.Joint.CoordinateSystemName;
            // NB: NX's GLB export is Z-up, we need Y-up
            string topLevelCsysHandle = GetTopLevelCoordinateSystemForMeshExport(coordinateSystemName);
            CartesianCoordinateSystem coordSys = ResolveCoordinateSystemByHandle(topLevelCsysHandle);

            try
            {
                logger.Information($"{link.Name}: Exporting GLB with {bodies.Count} bodies to {windowsMeshFilename}");

                double chordalTolerance = exportingCollision ? meshingOptions.collisionMeshingOptions.linearDeflection : meshingOptions.visualMeshingOptions.linearDeflection;
                chordalTolerance = chordalTolerance * MM_TO_IN; // NX Visualization preferences is in inches
                double angularTolerance = exportingCollision ? meshingOptions.collisionMeshingOptions.angularDeflection : meshingOptions.visualMeshingOptions.angularDeflection;
                angularTolerance = angularTolerance / Math.PI * 180.0;

                workPart.Preferences.ShadeVisualization.ShadedViewTolerance = NXOpen.Preferences.PartVisualizationShade.ShadedViewToleranceType.Customize;
                workPart.Preferences.ShadeVisualization.AlignShadedViewFacetsAlongEdges = true;
                workPart.Preferences.ShadeVisualization.SetShadedViewFacetTolerances(
                    NXOpen.Preferences.PartVisualizationShade.ShadedViewToleranceType.Customize,
                    chordalTolerance,
                    chordalTolerance,
                    angularTolerance);

                ExtendedRealityFileCreator xrCreator = session.DexManager.CreateExtendedRealityFileCreator();

                xrCreator.FileLocation = windowsMeshFilename;
                xrCreator.Format = ExtendedRealityFileCreator.OutputFormat.Glb;
                xrCreator.ExportEntity = ExtendedRealityFileCreator.ObjectToExport.SelectedObjects;
                xrCreator.ExportPositionType = ExtendedRealityFileCreator.ExportedPositionCSYS.UserDefined;
                xrCreator.CsysOfexportPosition = coordSys;
                xrCreator.DataCompression = false;

                foreach (NXObject obj in bodies)
                {
                    bool added = xrCreator.BodiesToExport.Add(obj as NXOpen.DisplayableObject);
                }

                xrCreator.Commit();
                xrCreator.Destroy();

                bool fileExists = File.Exists(windowsMeshFilename);
                if (fileExists)
                {
                    logger.Information($"Successfully exported GLB file: {windowsMeshFilename}");
                }
                else
                {
                    logger.Warning($"GLB export completed but file not found: {windowsMeshFilename}");
                }

                return fileExists;
            }
            catch (Exception ex)
            {
                logger.Error($"SaveGLB failed: {ex.Message}");
                return false;
            }
        }

        public override bool SaveStl(
            Link link,
            Link.ComponentType componentType,
            string windowsMeshFilename,
            ExporterMeshingOptions meshingOptions,
            bool exportingCollision)
        {
            List<string> handles = GetHandlesForComponentType(link, componentType);
            List<Body> bodies = ResolveBodiesToList(handles);

            if (bodies.Count == 0)
            {
                logger.Warning($"No bodies to export for link {link.Name}");
                return false;
            }

            string coordinateSystemName = link.Joint.CoordinateSystemName;
            string topLevelCsysHandle = GetTopLevelCoordinateSystem(coordinateSystemName);
            CartesianCoordinateSystem coordSys = ResolveCoordinateSystemByHandle(topLevelCsysHandle);

            Tag savedWcsTag = Tag.Null;
            ufSession.Csys.AskWcs(out savedWcsTag);

            if (coordSys != null)
            {
                ufSession.Csys.SetWcs(coordSys.Tag);
            }

            try
            {
                logger.Information($"{link.Name}: Exporting STL with {bodies.Count} bodies to {windowsMeshFilename}");

                STLCreator stlCreator = session.DexManager.CreateStlCreator();

                stlCreator.OutputFile = windowsMeshFilename;
                stlCreator.ExportDestination = BaseCreator.ExportDestinationOption.NativeFileSystem;
                SelectNXObjectList objectList = stlCreator.ExportSelectionBlock;
                foreach (NXObject obj in bodies)
                {
                    bool added = objectList.Add(obj);
                }

                double chordalTolerance = exportingCollision ? meshingOptions.collisionMeshingOptions.linearDeflection : meshingOptions.visualMeshingOptions.linearDeflection;
                double angularTolerance = exportingCollision ? meshingOptions.collisionMeshingOptions.angularDeflection : meshingOptions.visualMeshingOptions.angularDeflection;
                angularTolerance = angularTolerance / Math.PI * 180.0;

                stlCreator.ChordalTol = chordalTolerance;
                stlCreator.AngularTol = angularTolerance;
                stlCreator.OutputType = STLCreator.OutputTypeEnum.Binary;
                stlCreator.TriangleDisplay = false;
                stlCreator.ProcessHoldFlag = true;

                stlCreator.Commit();
                stlCreator.Destroy();

                if (savedWcsTag != Tag.Null)
                {
                    ufSession.Csys.SetWcs(savedWcsTag);
                }

                bool fileExists = File.Exists(windowsMeshFilename);
                if (fileExists)
                {
                    // NX exports STL in mm, scale to meters
                    Utilities.StlScaler.ScaleInPlace(windowsMeshFilename, (float)MM_TO_M);
                    logger.Information($"Successfully exported STL file: {windowsMeshFilename}");
                }
                else
                {
                    logger.Warning($"STL export completed but file not found: {windowsMeshFilename}");
                }

                return fileExists;
            }
            catch (Exception ex)
            {
                logger.Error($"SaveSTL failed: {ex.Message}");

                if (savedWcsTag != Tag.Null)
                {
                    ufSession.Csys.SetWcs(savedWcsTag);
                }

                return false;
            }
        }

        protected override void SetLinkSpecificSTEPPreferences(string coordinateSystemName)
        {
        }

        protected override void SetLinkSpecificSTLPreferences(string coordinateSystemName, double linearDeviation, double angularDeviation)
        {
        }

        public override void SaveSTLExportUserPreferences()
        {
            // This is actually for GLB export which does not support tesselation options

            originalShadedViewToleranceType = workPart.Preferences.ShadeVisualization.ShadedViewTolerance;
            originalAlignFacetAlongEdges = workPart.Preferences.ShadeVisualization.AlignShadedViewFacetsAlongEdges;
            workPart.Preferences.ShadeVisualization.GetShadedViewFacetTolerances(
                NXOpen.Preferences.PartVisualizationShade.ShadedViewToleranceType.Customize,
                out originalEdgeTolerance,
                out originalFaceTolerance,
                out originalAngleTolerance);

            NXOpen.Display.FacetSettingsBuilder facetSettingsBuilder;
            facetSettingsBuilder = workPart.CreateFacetSettingsBuilder();
            originalFacetScale = facetSettingsBuilder.ShadedFacetScale;
            facetSettingsBuilder.ShadedFacetScale = NXOpen.Display.FacetSettingsBuilder.FacetScale.Fixed;
            facetSettingsBuilder.Commit();
            facetSettingsBuilder.Destroy();
        }

        public override void SetSTLExportUserPreferences()
        {
            // Handled in GLB export
        }

        public override void ResetSTLExportUserPreferences()
        {
            // This is actually for GLB export which does not support tesselation options

            workPart.Preferences.ShadeVisualization.SetShadedViewFacetTolerances(
                NXOpen.Preferences.PartVisualizationShade.ShadedViewToleranceType.Customize,
                originalEdgeTolerance,
                originalFaceTolerance,
                originalAngleTolerance);
            workPart.Preferences.ShadeVisualization.ShadedViewTolerance = originalShadedViewToleranceType;
            workPart.Preferences.ShadeVisualization.AlignShadedViewFacetsAlongEdges = originalAlignFacetAlongEdges;

            NXOpen.Display.FacetSettingsBuilder facetSettingsBuilder;
            facetSettingsBuilder = workPart.CreateFacetSettingsBuilder();
            facetSettingsBuilder.ShadedFacetScale = originalFacetScale;
            facetSettingsBuilder.Commit();
            facetSettingsBuilder.Destroy();
        }
    }
}

#endif
