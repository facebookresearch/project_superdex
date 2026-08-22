/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if SOLIDWORKS

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Linq;

using MathNet.Numerics.LinearAlgebra;

using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;

using CADRobotExporter.Export;
using CADRobotExporter.RobotDescription;
using CADRobotExporter.UI;
using CADRobotExporter.Utilities;
using CADRobotExporter.SW;

namespace CADRobotExporter.CAD
{
    public class SolidworksBridge : CADBridge
    {
        private static readonly Serilog.ILogger logger = Logger.GetLogger();

        private ISldWorks _app;
        private ModelDoc2 _model;

        public Feature ExporterFeature { get; set; }

        private UserProgressBar _swProgressBar;

        private bool _userPreferenceSTLBinary;
        private bool _userPreferenceSTLShowInfo;
        private bool _userPreferenceSTLPreview;
        private bool _userPreferenceSTLTranslate;
        private bool _userPreferenceSTLCombine;
        private int _userPreferenceSTLUnits;
        private int _userPreferenceSTLQuality;
        private double _userPreferenceHideShowTransition;
        private double _userPreferenceSTLDeviation;
        private double _userPreferenceSTLAngularTolerance;

        private List<Component2> _hiddenComponents;

        private Dictionary<string, string> _topLevelCoordinateAxes;
        private List<string> _tempTopLevelCoordinateAxes;

        public SolidworksBridge(ISldWorks app, ModelDoc2 model)
        {
            _app = app;
            _model = model;

            _topLevelCoordinateAxes = new Dictionary<string, string>();
            _tempTopLevelCoordinateAxes = new List<string>();
        }

        public override string GetDocumentTitle()
        {
            return _model.GetTitle();
        }

        public override ExporterConfiguration GetExporterConfiguration()
        {
            return ConfigurationSerialization.LoadExporterConfiguration(ExporterFeature);
        }

        public override void SetLatestExportLocation(string location)
        {
            SwAddin.SetLatestExportPath(location);
        }

        public override string GetLatestExportLocation()
        {
            return SwAddin.LatestExportPath;
        }

        public override void SaveConfigurationFromTree(ExporterConfiguration config, LinkNode baseNode, bool warnUser, string featureNameOverride = null)
        {
            CommonSwOperations.RetrieveSWComponentPIDs(_model, baseNode);

            string featureName = "";
            if (featureNameOverride != null)
            {
                featureName = featureNameOverride;
            }
            else
            {
                featureName = $"Robot Configuration ({config.robotName})";
            }

            Feature exporterFeature = ConfigurationSerialization.SaveConfigTreeXML(_model, baseNode, warnUser, ExporterFeature, featureName);

            if (exporterFeature != null)
            {
                ExporterFeature = exporterFeature;
                if (config != null)
                {
                    ConfigurationSerialization.SaveExporterConfiguration(config, exporterFeature);
                }
            }
        }

        public void SaveTendons(List<Tendon> tendons)
        {
            if (ExporterFeature != null)
            {
                ConfigurationSerialization.SaveTendonData(tendons, ExporterFeature);
            }
        }

        public List<Tendon> LoadTendons()
        {
            return ConfigurationSerialization.LoadTendonData(ExporterFeature);
        }

        public override void SelectLinkComponents(LinkNode node)
        {
            _model.ClearSelection2(true);
            SelectionMgr manager = _model.SelectionManager;

            SelectData data = manager.CreateSelectData();
            data.Mark = -1;
            foreach (Component2 component in node.Link.SWVisualComponents)
            {
                component.Select4(true, data, false);
            }
            if (node.Link.SWVisualComponents.Count == 0)
            {
                foreach (Component2 component in node.Link.SWInertialComponents)
                {
                    component.Select4(true, data, false);
                }
            }
            if (node.Link.Joint != null)
            {
                if (node.Link.Joint.SWCoordinateSystemFeature != null)
                {
                    node.Link.Joint.SWCoordinateSystemFeature.Select2(true, -1);
                }
                if (node.Link.Joint.SWRefAxisFeature != null)
                {
                    node.Link.Joint.SWRefAxisFeature.Select2(true, -1);
                }
            }
        }

        public override void SelectJointComponents(LinkNode node)
        {
            _model.ClearSelection2(true);
            SelectionMgr manager = _model.SelectionManager;

            SelectData data = manager.CreateSelectData();
            data.Mark = -1;
            foreach (Component2 component in node.Link.SWVisualComponents)
            {
                component.Select4(true, data, false);
            }
            if (node.Link.SWVisualComponents.Count == 0)
            {
                foreach (Component2 component in node.Link.SWInertialComponents)
                {
                    component.Select4(true, data, false);
                }
            }
            if (node.Link.Joint != null)
            {
                if (node.Link.Joint.SWCoordinateSystemFeature != null)
                {
                    node.Link.Joint.SWCoordinateSystemFeature.Select2(true, -1);
                }
                if (node.Link.Joint.SWRefAxisFeature != null)
                {
                    node.Link.Joint.SWRefAxisFeature.Select2(true, -1);
                }
            }
        }

        public override void HideAllComponents()
        {
            AssemblyDoc assyDoc = (AssemblyDoc)_model;
            _hiddenComponents = CommonSwOperations.FindHiddenComponents(assyDoc.GetComponents(false));
            logger.Information("Found " + _hiddenComponents.Count + " hidden components " + String.Join(", ", _hiddenComponents));
            logger.Information("Hiding all components");
            _model.Extension.SelectAll();
            _model.HideComponent2();
        }

        public override void ShowHiddenComponents()
        {
            CommonSwOperations.ShowAllComponents(_model, _hiddenComponents);
        }

        public override void ShowHideVisualizations(bool show)
        {
            ModelView view = _model.IActiveView;
            if (show)
            {
                view.BufferSwapNotify += View_BufferSwapNotify_Internal;
            }
            else
            {
                view.BufferSwapNotify -= View_BufferSwapNotify_Internal;
            }
        }

        private int View_BufferSwapNotify_Internal()
        {
            double[] boundingBox = new double[6];
            GetModelBoundingBox(out boundingBox);

            double xLength = Math.Abs(boundingBox[0] - boundingBox[3]);
            double yLength = Math.Abs(boundingBox[1] - boundingBox[4]);
            double zLength = Math.Abs(boundingBox[2] - boundingBox[5]);

            double averageLength = (xLength + yLength + zLength) / 3.0;

            if (IsShowingJointGizmo && CurrentNodeShown != null &&
                CurrentNodeShown.Link.Joint.originInGlobalSpace != null &&
                CurrentNodeShown.Link.Joint.axisInGlobalSpace != null)
            {
                double upperLimit = CurrentNodeShown.Link.Joint.Limit.Upper;
                double lowerLimit = CurrentNodeShown.Link.Joint.Limit.Lower;

                double[] referenceDirection = GetCoMReferenceDirection(CurrentNodeShown.Link);

                OpenGL.glDisable(OpenGL.GL_LIGHTING);

                if (CurrentNodeShown.Link.Joint.Type == "revolute")
                {
                    SolidworksJointVisualizer.DrawCompleteRevoluteJoint(
                        CurrentNodeShown.Link.Joint.originInGlobalSpace,
                        CurrentNodeShown.Link.Joint.axisInGlobalSpace,
                        lowerLimit,
                        upperLimit,
                        JointGizmoScale * averageLength,
                        JointGizmoScale * 0.5 * 1.25 * averageLength,
                        referenceDirection);
                }

                if (CurrentNodeShown.Link.Joint.Type == "prismatic")
                {
                    SolidworksJointVisualizer.DrawCompletePrismaticJoint(
                        CurrentNodeShown.Link.Joint.originInGlobalSpace,
                        CurrentNodeShown.Link.Joint.axisInGlobalSpace,
                        lowerLimit,
                        upperLimit,
                        JointGizmoScale * 0.5 * 0.5 * averageLength,
                        JointGizmoScale * 0.5 * 1.25 * averageLength);
                }
            }

            if (IsShowingTendonVisualization && CurrentTendonShown != null)
            {
                DrawTendonVisualization(averageLength);
            }

            if (IsShowingAllTendons && AllTendons != null && AllTendons.Count > 0)
            {
                DrawAllTendonsVisualization(averageLength);
            }

            if (IsShowingInertialGizmo && CurrentLinkNodeShown != null)
            {
                DrawInertialGizmo_Internal(averageLength);
            }

            return 0;
        }

        private void DrawTendonVisualization(double averageLength)
        {
            var tendon = CurrentTendonShown;
            if (tendon.RoutingElements == null || tendon.RoutingElements.Count == 0)
                return;

            var points = new System.Collections.Generic.List<double[]>();
            int highlightPointIndex = -1;

            for (int elemIdx = 0; elemIdx < tendon.RoutingElements.Count; elemIdx++)
            {
                var elem = tendon.RoutingElements[elemIdx];
                double[] pos = null;

                if (elem.Type == RobotDescription.RoutingElement.TypeWaypoint)
                {
                    if (!string.IsNullOrEmpty(elem.PointKey))
                    {
                        pos = GetTopLevelPointCoordinates(elem.PointKey);
                    }
                }
                else if (elem.Type == RobotDescription.RoutingElement.TypeLinearJoint)
                {
                    if (TendonLinkCsysMap != null &&
                        !string.IsNullOrEmpty(elem.Link) &&
                        TendonLinkCsysMap.TryGetValue(elem.Link, out string csysName))
                    {
                        var tf = GetCoordinateSystemTransform(csysName);
                        if (tf != null)
                        {
                            pos = new double[] { tf[0, 3], tf[1, 3], tf[2, 3] };
                        }
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

            double crossSize = averageLength * 0.01;
            SolidworksTendonVisualizer.DrawTendon(points, highlightPointIndex, crossSize);
        }

        private void DrawAllTendonsVisualization(double averageLength)
        {
            double crossSize = averageLength * 0.01;

            for (int t = 0; t < AllTendons.Count; t++)
            {
                var tendon = AllTendons[t];
                if (tendon.RoutingElements == null || tendon.RoutingElements.Count == 0)
                    continue;

                var points = new System.Collections.Generic.List<double[]>();

                for (int elemIdx = 0; elemIdx < tendon.RoutingElements.Count; elemIdx++)
                {
                    var elem = tendon.RoutingElements[elemIdx];
                    double[] pos = null;

                    if (elem.Type == RobotDescription.RoutingElement.TypeWaypoint)
                    {
                        if (!string.IsNullOrEmpty(elem.PointKey))
                        {
                            pos = GetTopLevelPointCoordinates(elem.PointKey);
                        }
                    }
                    else if (elem.Type == RobotDescription.RoutingElement.TypeLinearJoint)
                    {
                        if (TendonLinkCsysMap != null &&
                            !string.IsNullOrEmpty(elem.Link) &&
                            TendonLinkCsysMap.TryGetValue(elem.Link, out string csysName))
                        {
                            var tf = GetCoordinateSystemTransform(csysName);
                            if (tf != null)
                            {
                                pos = new double[] { tf[0, 3], tf[1, 3], tf[2, 3] };
                            }
                        }
                    }

                    if (pos != null)
                    {
                        points.Add(pos);
                    }
                }

                float[] color = SolidworksTendonVisualizer.GetPaletteColor(t);
                SolidworksTendonVisualizer.DrawTendon(points, -1, crossSize, color[0], color[1], color[2]);
            }
        }

        private void DrawInertialGizmo_Internal(double averageLength)
        {
            var link = CurrentLinkNodeShown.Link;
            if (link.Inertial == null || link.Inertial.Mass.Value <= 1e-10)
                return;
            if (link.Joint == null || string.IsNullOrEmpty(link.Joint.CoordinateSystemName))
                return;

            double[] comLocal = link.Inertial.Origin.GetXYZ();
            double[] rpy = link.Inertial.Origin.GetRPY();

            var tf = GetCoordinateSystemTransform(link.Joint.CoordinateSystemName);
            if (tf == null)
                return;

            // Transform CoM to global (meters)
            double comGlobalX = tf[0, 0] * comLocal[0] + tf[0, 1] * comLocal[1] + tf[0, 2] * comLocal[2] + tf[0, 3];
            double comGlobalY = tf[1, 0] * comLocal[0] + tf[1, 1] * comLocal[1] + tf[1, 2] * comLocal[2] + tf[1, 3];
            double comGlobalZ = tf[2, 0] * comLocal[0] + tf[2, 1] * comLocal[1] + tf[2, 2] * comLocal[2] + tf[2, 3];
            double[] comGlobal = new double[] { comGlobalX, comGlobalY, comGlobalZ };

            // Extract CSYS rotation (upper-left 3x3)
            double[,] rCsys = new double[3, 3];
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    rCsys[r, c] = tf[r, c];

            // Compute principal inertia
            if (!MathOps.ComputePrincipalInertia(
                link.Inertial.Mass.Value,
                link.Inertial.Inertia.Ixx, link.Inertial.Inertia.Iyy, link.Inertial.Inertia.Izz,
                link.Inertial.Inertia.Ixy, link.Inertial.Inertia.Ixz, link.Inertial.Inertia.Iyz,
                out double[] boxHalfExtents, out double[,] principalRotation))
                return;

            double[,] composedRotation = MathOps.ComposeRotation(rCsys, rpy, principalRotation);

            double scale = LinkGizmoScale > 0 ? LinkGizmoScale : 1.0;
            double crossSize = averageLength * 0.015 * scale;
            SolidworksInertialVisualizer.DrawInertialGizmo(comGlobal, composedRotation, boxHalfExtents, crossSize);
        }

        private double[] GetCoMReferenceDirection(Link link)
        {
            if (link.Inertial == null)
                return null;

            double[] comLocal = link.Inertial.Origin.GetXYZ();
            if (Math.Abs(comLocal[0]) < 1e-10 && Math.Abs(comLocal[1]) < 1e-10 && Math.Abs(comLocal[2]) < 1e-10)
                return null;

            try
            {
                var transform = GetCoordinateSystemTransform(link.Joint.CoordinateSystemName);
                if (transform == null)
                    return null;

                double comGlobalX = transform[0, 0] * comLocal[0] + transform[0, 1] * comLocal[1] + transform[0, 2] * comLocal[2] + transform[0, 3];
                double comGlobalY = transform[1, 0] * comLocal[0] + transform[1, 1] * comLocal[1] + transform[1, 2] * comLocal[2] + transform[1, 3];
                double comGlobalZ = transform[2, 0] * comLocal[0] + transform[2, 1] * comLocal[1] + transform[2, 2] * comLocal[2] + transform[2, 3];

                double[] jointOrigin = link.Joint.originInGlobalSpace;
                return new double[] {
                    comGlobalX - jointOrigin[0],
                    comGlobalY - jointOrigin[1],
                    comGlobalZ - jointOrigin[2]
                };
            }
            catch
            {
                return null;
            }
        }

        public override void TriggerGraphicsRedraw()
        {
            _model.GraphicsRedraw2();
        }

        public override void BoostPerformance(bool enable)
        {
            ModelView modelView = _model.ActiveView;
            FeatureManager featureManager = _model.FeatureManager;

            modelView.EnableGraphicsUpdate = !enable;
            featureManager.EnableFeatureTree = !enable;
            _app.CommandInProgress = !enable;
        }

        public override void SetProgressBarStart(int maxProgress, string title)
        {
            if (_swProgressBar == null)
            {
                _app.GetUserProgressBar(out _swProgressBar);
            }

            _swProgressBar?.Start(0, maxProgress, title);
        }

        public override void SetProgressBarEnd()
        {
            _swProgressBar?.End();
        }

        public override void SetProgressBarTitle(string title)
        {
            _swProgressBar?.UpdateTitle(title);
        }

        public override void SetProgressBarProgress(int progress)
        {
            _swProgressBar?.UpdateProgress(progress);
        }

        public override bool GetComponentsInertialProperties(
            Link link,
            Link.ComponentType componentType,
            string coordinateSystemName,
            out double mass,
            out double[] centerOfMass,
            out double[] momentOfInertia)
        {
            _model.ClearSelection2(true);

            List<Component2> components = link.GetComponents(componentType);
            MathTransform coordinateSystemTransform = GetCoordinateSystemTransform_Internal(coordinateSystemName);

            if (components.Count == 0)
            {
                mass = 0;
                centerOfMass = new double[3];
                momentOfInertia = new double[9];
                return true;
            }

            logger.Information("Start GetComponentsInertialProperties " + components[0].Name);

            bool originalPreference = _model.GetUserPreferenceToggle((int)swUserPreferenceToggle_e.swUsePositiveInertiaTensorNotation);

            _model.SetUserPreferenceToggle((int)swUserPreferenceToggle_e.swUsePositiveInertiaTensorNotation, false);

            SelectionMgr manager = _model.SelectionManager;
            foreach (var component in components)
            {
                SelectData data = manager.CreateSelectData();
                data.Mark = -1;
                component.Select4(true, data, false);
            }
            MassProperty2 swMass = _model.Extension.CreateMassProperty2();
            swMass.SetCoordinateSystem(coordinateSystemTransform);
            // swMass.AccuracyLevel = (int)swMassPropertyAccuracyLevel_e.swMassPropertyAccuracyLevel_Lower;
            swMass.Recalculate();

            mass = swMass.Mass;
            centerOfMass = (double[])swMass.CenterOfMass;
            momentOfInertia = (double[])swMass.GetMomentOfInertia((int)swMomentsOfInertiaReferenceFrame_e.swMomentsOfInertiaReferenceFrame_CenterOfMass);

            _model.SetUserPreferenceToggle((int)swUserPreferenceToggle_e.swUsePositiveInertiaTensorNotation, originalPreference);
            logger.Information("End GetComponentsInertialProperties");
            return true;
        }

        // [ R, G, B, Ambient, Diffuse, Specular, Shininess, Transparency, Emission ]
        public override double[] GetVisualProperties(Link link, Link.ComponentType componentType)
        {
            double[] values = new double[9];
            List<Component2> components = link.GetComponents(componentType);

            if (components != null && components.Count > 0)
            {
                ModelDoc2 componentModelDoc = components[0].GetModelDoc2();
                values = componentModelDoc.MaterialPropertyValues;
                values[7] = 1.0 - values[7];
            }

            return values;
        }

        public override double[] GetAxisInGlobalSpace(string axisName, bool flipAxis)
        {
            double[] axisParams;
            double[] axisVector = new double[3];

            _model.ClearSelection2(true);
            bool selected =
                _model.Extension.SelectByID2(axisName, "AXIS", 0, 0, 0, false, 0, null, 0);
            if (selected)
            {
                Feature feature = _model.SelectionManager.GetSelectedObject6(1, 0);
                RefAxis axis = (RefAxis)feature.GetSpecificFeature2();

                // GetRefAxisParams returns {startX, startY, startZ, endX, endY, endZ}
                axisParams = axis.GetRefAxisParams();
                axisParams = TransformAxisToGlobalSpace_Internal(feature, axisParams);
                axisVector[0] = axisParams[0] - axisParams[3];
                axisVector[1] = axisParams[1] - axisParams[4];
                axisVector[2] = axisParams[2] - axisParams[5];

                // Normalize and cleanup
                axisVector = MathOps.PNorm(axisVector, 2);

                if (flipAxis)
                {
                    axisVector = MathOps.Negate(axisVector);
                }
            }

            return axisVector;
        }

        private double[] TransformAxisToGlobalSpace_Internal(Feature feature, double[] axisParams)
        {
            Entity entity = (Entity)feature;
            Component2 component = (Component2)entity.GetComponent();

            // Most likely this is a top level feature, so abort.
            if (component == null)
            {
                return axisParams;
            }

            MathUtility mathUtil = _app.GetMathUtility();
            // NB: Transform2 returns the transform from the component to the root component of
            // the opened document. NOT THE LOCAL TRANSFORM WRT TO PARENT.
            MathTransform transform = component.Transform2;

            // Transform start point
            double[] startPoint = new double[] { axisParams[0], axisParams[1], axisParams[2] };
            MathPoint startMathPt = (MathPoint)mathUtil.CreatePoint(startPoint);
            MathPoint transformedStart = (MathPoint)startMathPt.MultiplyTransform(transform);
            double[] transformedStartArray = (double[])transformedStart.ArrayData;
            // Transform end point
            double[] endPoint = new double[] { axisParams[3], axisParams[4], axisParams[5] };
            MathPoint endMathPt = (MathPoint)mathUtil.CreatePoint(endPoint);
            MathPoint transformedEnd = (MathPoint)endMathPt.MultiplyTransform(transform);
            double[] transformedEndArray = (double[])transformedEnd.ArrayData;
            // Return transformed parameters
            return new double[]
            {
                transformedStartArray[0], transformedStartArray[1], transformedStartArray[2],
                transformedEndArray[0], transformedEndArray[1], transformedEndArray[2]
            };
        }

        public override Matrix<double> GetCoordinateSystemTransform(string coordinateSystemName)
        {
            MathTransform transform = GetCoordinateSystemTransform_Internal(coordinateSystemName);
            return CommonSwOperations.GetTransformation(transform);
        }

        private MathTransform GetCoordinateSystemTransform_Internal(string coordinateSystemName)
        {
            string topLevelCsysName = GetTopLevelCoordinateSystem(coordinateSystemName);
            return _model.Extension.GetCoordinateSystemTransformByName(topLevelCsysName);
        }

        public override string GetTopLevelCoordinateSystem(string someCoordinateSystemName)
        {
            if (_topLevelCoordinateAxes.ContainsKey(someCoordinateSystemName))
            {
                return _topLevelCoordinateAxes[someCoordinateSystemName];
            }
            else
            {
                _model.ClearSelection2(true);
                if (!_model.Extension.SelectByID2(someCoordinateSystemName, "COORDSYS", 0, 0, 0, false, 0, null, 0))
                {
                    throw new Exception("Could not find csys named: " + someCoordinateSystemName);
                }

                Feature feature = _model.SelectionManager.GetSelectedObject6(1, 0);
                Entity entity = (Entity)feature;
                Component2 component = (Component2)entity.GetComponent();

                // This is already a top level csys, creating a temp top level is useless
                if (component == null)
                {
                    _topLevelCoordinateAxes.Add(someCoordinateSystemName, someCoordinateSystemName);
                    return someCoordinateSystemName;
                }
                CreateTempTopLevelCoordinateSystem_Internal(someCoordinateSystemName, out string newTopLevelCoordinateSystem);
                _topLevelCoordinateAxes.Add(someCoordinateSystemName, newTopLevelCoordinateSystem);
                _tempTopLevelCoordinateAxes.Add(newTopLevelCoordinateSystem);
                return newTopLevelCoordinateSystem;
            }
        }

        public override double[] GetTopLevelPointCoordinates(string pointKey)
        {
            if (string.IsNullOrEmpty(pointKey) || _model == null)
                return null;

            try
            {
                byte[] pid = Convert.FromBase64String(pointKey);
                int errorCode;
                object entity = _model.Extension.GetObjectByPersistReference3(pid, out errorCode);
                if (entity == null)
                    return null;

                if (entity is Feature feat)
                {
                    IRefPoint refPt = (IRefPoint)feat.GetSpecificFeature2();
                    if (refPt != null)
                    {
                        IMathPoint pt = refPt.GetRefPoint();
                        if (pt != null)
                        {
                            Component2 component = ((Entity)feat).GetComponent();
                            if (component != null)
                            {
                                MathTransform componentToRootTransform = component.Transform2;
                                pt = (IMathPoint)pt.MultiplyTransform(componentToRootTransform);
                            }
                            return (double[])pt.ArrayData;
                        }
                    }
                }

                logger.Warning("GetTopLevelPointCoordinates: could not extract coordinates from resolved entity");
            }
            catch (Exception ex)
            {
                logger.Warning("GetTopLevelPointCoordinates failed for key: " + ex.Message);
            }

            return null;
        }

        /// <summary>
        /// Creates a coordinate system from a 4x4 transformation matrix.
        /// </summary>
        /// <param name="transform">4x4 homogeneous transformation matrix (in meters).</param>
        /// <param name="name">Name for the coordinate system.</param>
        /// <returns>A key/name that can be used to reference the created coordinate system.</returns>
        public override string CreateCoordinateSystemFromTransform(Matrix<double> transform, string name)
        {
            if (_model == null)
            {
                logger.Error("Cannot create coordinate system - no active model");
                return null;
            }

            try
            {
                // Convert transform matrix to SolidWorks 16-element format:
                // SolidWorks transformation matrix layout:
                //   |a b c . n |
                //   |d e f . o |
                //   |g h i . p |
                //   |j k l . m |
                //
                // Elements 0-8 (a-i): 3x3 rotation matrix stored COLUMN-MAJOR
                // Elements 9-11 (j,k,l): Translation vector (converted to mm)
                // Element 12 (m): Scaling factor (1.0)
                // Elements 13-15 (n,o,p): Unused (0.0)
                //
                // MathNet Matrix is row-major:
                //   | R00 R01 R02 Tx |
                //   | R10 R11 R12 Ty |
                //   | R20 R21 R22 Tz |
                //   |  0   0   0   1 |

                double[] matrixArray = new double[16];

                // Rotation 3x3 - stored column-major for SolidWorks
                // Column 0: elements 0, 1, 2
                matrixArray[0] = transform[0, 0];  // a = R00
                matrixArray[1] = transform[1, 0];  // d = R10
                matrixArray[2] = transform[2, 0];  // g = R20

                // Column 1: elements 3, 4, 5
                matrixArray[3] = transform[0, 1];  // b = R01
                matrixArray[4] = transform[1, 1];  // e = R11
                matrixArray[5] = transform[2, 1];  // h = R21

                // Column 2: elements 6, 7, 8
                matrixArray[6] = transform[0, 2];  // c = R02
                matrixArray[7] = transform[1, 2];  // f = R12
                matrixArray[8] = transform[2, 2];  // i = R22

                // Translation: elements 9, 10, 11
                matrixArray[9] = transform[0, 3];   // j = Tx
                matrixArray[10] = transform[1, 3];  // k = Ty
                matrixArray[11] = transform[2, 3];  // l = Tz

                // Scale factor
                matrixArray[12] = 1.0;  // m = scale

                // Unused
                matrixArray[13] = 0.0;  // n
                matrixArray[14] = 0.0;  // o
                matrixArray[15] = 0.0;  // p

                // Decompose to XYZ and Euler angles
                DecomposedTransform tf = MathOps.DecomposeTransformationMatrixToXYZEuler(matrixArray);

                Feature tempCoordinateSystemFeature = _model.FeatureManager.CreateCoordinateSystemUsingNumericalValues(
                    true, tf.Translation.X, tf.Translation.Y, tf.Translation.Z,
                    true, tf.Rotation.X, tf.Rotation.Y, tf.Rotation.Z);

                if (tempCoordinateSystemFeature == null)
                {
                    logger.Error("CreateCoordinateSystemUsingNumericalValues returned null");
                    return null;
                }

                // Rename the feature
                if (!string.IsNullOrEmpty(name))
                {
                    try
                    {
                        tempCoordinateSystemFeature.Name = $"URDF_{name}";
                    }
                    catch
                    {
                        // Name setting may fail, ignore
                    }
                }

                string csysName = tempCoordinateSystemFeature.GetNameForSelection(out _);
                logger.Information($"Created CSYS '{name}' at ({tf.Translation.X:F2}, {tf.Translation.Y:F2}, {tf.Translation.Z:F2}) mm with name {csysName}");
                return csysName;
            }
            catch (Exception ex)
            {
                logger.Error($"CreateCoordinateSystemFromTransform failed: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Creates a reference axis from a direction vector at an origin point.
        /// </summary>
        /// <param name="origin">Origin point [x, y, z] in meters.</param>
        /// <param name="direction">Direction vector [x, y, z] (normalized).</param>
        /// <param name="name">Name for the axis.</param>
        /// <returns>The feature name for selection, or null if creation failed.</returns>
        public override string CreateAxisFromDirection(double[] origin, double[] direction, string name)
        {
            if (_model == null)
            {
                logger.Error("Cannot create reference axis - no active model");
                return null;
            }

            // Keep this check for future use - currently we create axes regardless of colinearity
            // bool isColinear = IsColinearWithPrincipalAxis(direction);

            try
            {
                double unitScale = 1000.0; // meters to mm

                // Convert origin to mm
                double originX = origin[0] * unitScale;
                double originY = origin[1] * unitScale;
                double originZ = origin[2] * unitScale;

                // Create a line 0.1m (100mm) long centered on the origin
                double halfLength = 0.05 * unitScale; // 50mm each side
                double startX = originX + halfLength * direction[0];
                double startY = originY + halfLength * direction[1];
                double startZ = originZ + halfLength * direction[2];
                double endX = originX - halfLength * direction[0];
                double endY = originY - halfLength * direction[1];
                double endZ = originZ - halfLength * direction[2];

                // Set up or use a 3D sketch for reference geometry
                string sketchName = SetupURDFReferenceSketch();

                // Open the sketch for editing
                if (_model.SketchManager.ActiveSketch == null)
                {
                    _model.Extension.SelectByID2(sketchName, "SKETCH", 0, 0, 0, false, 0, null, 0);
                    _model.SketchManager.Insert3DSketch(true);
                }

                // Create the sketch line
                SketchSegment rotAxis = _model.SketchManager.CreateLine(
                    startX, startY, startZ,
                    endX, endY, endZ);

                if (rotAxis == null)
                {
                    logger.Error("Failed to create sketch line for axis");
                    if (_model.SketchManager.ActiveSketch != null)
                    {
                        _model.SketchManager.Insert3DSketch(true);
                    }
                    return null;
                }

                rotAxis.ConstructionGeometry = true;
                rotAxis.Width = 2;

                // Close the sketch
                if (_model.SketchManager.ActiveSketch != null)
                {
                    _model.SketchManager.Insert3DSketch(true);
                }

                // Select the sketch segment and create the axis
                Feature axisFeature = InsertAxisFromSketchSegment(rotAxis);

                if (axisFeature == null)
                {
                    logger.Error("InsertAxis2 failed to create axis");
                    return null;
                }

                // Rename the feature
                if (!string.IsNullOrEmpty(name))
                {
                    try
                    {
                        axisFeature.Name = $"URDF_{name}_axis";
                    }
                    catch
                    {
                        // Name setting may fail, ignore
                    }
                }

                string axisName = axisFeature.GetNameForSelection(out _);
                logger.Information($"Created RefAxis '{name}' with direction ({direction[0]:F3}, {direction[1]:F3}, {direction[2]:F3}) with name {axisName}");
                return axisName;
            }
            catch (Exception ex)
            {
                logger.Error($"CreateAxisFromDirection failed: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Sets up or retrieves the URDF Reference 3D sketch for creating reference geometry.
        /// </summary>
        private string SetupURDFReferenceSketch()
        {
            string sketchName = "URDF Reference";

            bool sketchExists = _model.Extension.SelectByID2(sketchName, "SKETCH", 0, 0, 0, false, 0, null, 0);
            _model.SketchManager.Insert3DSketch(true);
            _model.SketchManager.CreatePoint(0, 0, 0);
            IFeature sketch = (IFeature)_model.SketchManager.ActiveSketch;
            _model.SketchManager.Insert3DSketch(true);

            if (!sketchExists && sketch != null)
            {
                sketch.Name = sketchName;
            }

            return sketch?.Name ?? sketchName;
        }

        /// <summary>
        /// Creates a reference axis from a selected sketch segment.
        /// </summary>
        private Feature InsertAxisFromSketchSegment(SketchSegment segment)
        {
            // Select the sketch segment
            SelectionMgr selMgr = _model.SelectionManager;
            SelectData data = selMgr.CreateSelectData();
            segment.Select4(false, data);

            // Get features before axis creation
            object[] featuresBefore = _model.FeatureManager.GetFeatures(true);

            // Create the axis from the selected sketch segment
            bool success = _model.InsertAxis2(true);
            if (!success)
            {
                logger.Warning("InsertAxis2 returned false");
                return null;
            }

            // Get features after axis creation
            object[] featuresAfter = _model.FeatureManager.GetFeatures(true);

            // Find the newly created feature
            if (featuresBefore.Length < featuresAfter.Length)
            {
                // The new feature is probably at the end
                foreach (Feature feat in featuresAfter.Reverse())
                {
                    if (!featuresBefore.Contains(feat))
                    {
                        return feat;
                    }
                }
            }

            return null;
        }

        /// <summary>
        /// Checks if a direction vector is colinear with a principal axis (X, Y, or Z).
        /// Kept for future use.
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
            _model.ClearSelection2(true);

            foreach (string tempCsysName in _tempTopLevelCoordinateAxes)
            {
                _model.Extension.SelectByID2(tempCsysName, "COORDSYS", 0, 0, 0, true, 0, null, 0);
            }
            _model.Extension.DeleteSelection2(0);
            _model.ClearSelection2(true);
        }

        public override void GetModelBoundingBox(out double[] boundingBox)
        {
            AssemblyDoc assemblyDoc = (AssemblyDoc)_model;

            boundingBox = assemblyDoc.GetBox(0); // 0 = do not include planes or sketches
        }

        private static string AddElementNameToCoordinateSystemName_Internal(string coordinateSystemName, string element)
        {
            string output = coordinateSystemName;
            int delimiterIdx = coordinateSystemName.IndexOf('@');

            if (delimiterIdx >= 0)
            {
                output = coordinateSystemName.Substring(0, delimiterIdx) + "\\" + element + coordinateSystemName.Substring(delimiterIdx);
            }
            else
            {
                output += "\\" + element;
            }

            return output;
        }

        public void CreateTempTopLevelCoordinateSystem_Internal(string coordinateSystemName, out string topLevelCoordinateSystemName)
        {
            logger.Information("Begin CreateTempTopLevelCoordinateSystem:" + coordinateSystemName);
            MathUtility mathUtil = _app.GetMathUtility();

            // get the coordinate system's MathTransform in the most roundabout way possible
            _model.Extension.SelectByID2(AddElementNameToCoordinateSystemName_Internal(coordinateSystemName, "Point"), "COORDSYS", 0, 0, 0, false, 0, null, 0);
            Feature coordSysFeature = (Feature)_model.SelectionManager.GetSelectedObject6(1, -1);
            _model.ClearSelection2(true);

            CoordinateSystemFeatureData csData = coordSysFeature.GetDefinition() as CoordinateSystemFeatureData;
            MathTransform coordsysLocalTransform = csData.Transform;
            if (coordsysLocalTransform == null)
            {
                throw new Exception("Couldn't find coordinate system name: " + coordinateSystemName);
            }

            // get the owning component
            Entity entity = (Entity)coordSysFeature;
            Component2 component = (Component2)entity.GetComponent(); // this shouldn't be null
            // get the component to root transform
            MathTransform componentToRootTransform = component.Transform2;

            MathTransform coordsysGlobalTransform = coordsysLocalTransform.Multiply(componentToRootTransform);

            DecomposedTransform tf = MathOps.DecomposeTransformationMatrixToXYZEuler((double[])coordsysGlobalTransform.ArrayData);

            double[] matrix = (double[])coordsysGlobalTransform.ArrayData;

            Feature tempCoordinateSystemFeature = _model.FeatureManager.CreateCoordinateSystemUsingNumericalValues(
                true, tf.Translation.X, tf.Translation.Y, tf.Translation.Z,
                true, tf.Rotation.X, tf.Rotation.Y, tf.Rotation.Z);

            topLevelCoordinateSystemName = tempCoordinateSystemFeature.GetNameForSelection(out _);
            logger.Information("End CreateTempTopLevelCoordinateSystem:" + coordinateSystemName);
        }

        public override bool SaveStpFile(Link link, Link.ComponentType componentType, string windowsMeshFilename)
        {
            int errors = 0;
            int warnings = 0;

            string coordsysName = link.Joint.CoordinateSystemName;

            logger.Information(link.Name + ": Exporting STEP with coordinate frame " + coordsysName);

            List<Component2> components = link.GetComponents(componentType);

            CommonSwOperations.ShowComponentsWithDependents(_app, components);
            CommonSwOperations.HideComponents(_hiddenComponents);
            CommonSwOperations.SelectComponents(_model, components);

            int saveOptions = (int)swSaveAsOptions_e.swSaveAsOptions_Silent |
                (int)swSaveAsOptions_e.swSaveAsOptions_Copy;
            _app.SetUserPreferenceIntegerValue((int)swUserPreferenceIntegerValue_e.swStepAP, 214);
            _app.SetUserPreferenceToggle((int)swUserPreferenceToggle_e.swStepExportAppearances, true);
            string topLevelCoordSysName = GetTopLevelCoordinateSystem(link.Joint.CoordinateSystemName);
            SetLinkSpecificSTEPPreferences(topLevelCoordSysName);

            logger.Information("Saving STEP to " + windowsMeshFilename);
            _model.Extension.SaveAs(windowsMeshFilename,
                (int)swSaveAsVersion_e.swSaveAsCurrentVersion, saveOptions, null, ref errors, ref warnings);
            if (errors + warnings != 0)
            {
                logger.Warning("Exporting STEP for link " + link.Name + " failed with error " + errors +
                    " or warnings " + warnings);
            }

            CommonSwOperations.HideComponents(components);

            return true;
        }

        public override bool SaveStl(Link link, Link.ComponentType componentType, string windowsMeshFilename, ExporterMeshingOptions meshingOptions, bool exportingCollision)
        {
            int errors = 0;
            int warnings = 0;

            string coordsysName = link.Joint.CoordinateSystemName;

            logger.Information(link.Name + ": Exporting STL with coordinate frame " + coordsysName);

            List<Component2> components = link.GetComponents(componentType);

            CommonSwOperations.ShowComponentsWithDependents(_app, components);
            CommonSwOperations.HideComponents(_hiddenComponents);
            CommonSwOperations.SelectComponents(_model, components);

            int saveOptions = (int)swSaveAsOptions_e.swSaveAsOptions_Silent |
                (int)swSaveAsOptions_e.swSaveAsOptions_Copy;
            string topLevelCoordSysName = GetTopLevelCoordinateSystem(link.Joint.CoordinateSystemName);
            double linearDeflection = exportingCollision ? meshingOptions.collisionMeshingOptions.linearDeflection : meshingOptions.visualMeshingOptions.linearDeflection;
            double angularDeflection = exportingCollision ? meshingOptions.collisionMeshingOptions.angularDeflection : meshingOptions.visualMeshingOptions.angularDeflection;
            SetLinkSpecificSTLPreferences(topLevelCoordSysName, linearDeflection, angularDeflection);

            logger.Information("Saving STL to " + windowsMeshFilename);
            _model.Extension.SaveAs(windowsMeshFilename,
                (int)swSaveAsVersion_e.swSaveAsCurrentVersion, saveOptions, null, ref errors, ref warnings);
            if (errors + warnings != 0)
            {
                logger.Warning("Exporting STL for link " + link.Name + " failed with error " + errors +
                    " or warnings " + warnings);
            }

            CommonSwOperations.HideComponents(components);

            return CorrectSTLExtension(windowsMeshFilename);
        }

        public static bool CorrectSTLExtension(string filename)
        {
            string previousFilename = Path.ChangeExtension(filename, ".STL");
            if (!File.Exists(previousFilename))
            {
                return false;
            }
            string newFilename = Path.ChangeExtension(filename, ".stl");

            // Wait up to 10 seconds for the file to become available
            const int timeoutMs = 10000;
            const int pollIntervalMs = 100;
            var stopwatch = Stopwatch.StartNew();

            while (stopwatch.ElapsedMilliseconds < timeoutMs)
            {
                if (!IsFileLocked(previousFilename))
                {
                    try
                    {
                        File.Move(previousFilename, newFilename);
                        return true;
                    }
                    catch (IOException)
                    {
                        // File became locked between check and move, continue waiting
                    }
                }

                Task.Delay(pollIntervalMs).Wait();
            }

            // Timeout reached - fall back to copy
            try
            {
                File.Copy(previousFilename, newFilename, overwrite: true);
                return true;
            }
            catch (IOException)
            {
                return false;
            }
        }

        private static bool IsFileLocked(string filePath)
        {
            try
            {
                using (FileStream stream = new FileStream(filePath, FileMode.Open, FileAccess.ReadWrite, FileShare.None))
                {
                    return false;
                }
            }
            catch (IOException)
            {
                return true;
            }
        }

        protected override void SetLinkSpecificSTEPPreferences(string coordinateSystemName)
        {
            _model.Extension.SetUserPreferenceString((int)swUserPreferenceStringValue_e.swFileSaveAsCoordinateSystem,
                (int)swUserPreferenceOption_e.swDetailingNoOptionSpecified, coordinateSystemName);
        }

        protected override void SetLinkSpecificSTLPreferences(string coordinateSystemName, double linearDeviation, double angularDeviation)
        {
            const double kMMtoMeters = 0.001;
            _model.Extension.SetUserPreferenceString((int)swUserPreferenceStringValue_e.swFileSaveAsCoordinateSystem,
                (int)swUserPreferenceOption_e.swDetailingNoOptionSpecified, coordinateSystemName);
            _app.SetUserPreferenceDoubleValue((int)swUserPreferenceDoubleValue_e.swSTLDeviation, linearDeviation * kMMtoMeters);
            _app.SetUserPreferenceDoubleValue((int)swUserPreferenceDoubleValue_e.swSTLAngleTolerance, angularDeviation * (180.0 / Math.PI));
        }

        public override void SaveSTLExportUserPreferences()
        {
            _userPreferenceSTLBinary = _app.GetUserPreferenceToggle((int)swUserPreferenceToggle_e.swSTLBinaryFormat);
            _userPreferenceSTLTranslate = _app.GetUserPreferenceToggle((int)swUserPreferenceToggle_e.swSTLDontTranslateToPositive);
            _userPreferenceSTLUnits = _app.GetUserPreferenceIntegerValue((int)swUserPreferenceIntegerValue_e.swExportStlUnits);
            _userPreferenceSTLQuality = _app.GetUserPreferenceIntegerValue((int)swUserPreferenceIntegerValue_e.swSTLQuality);
            _userPreferenceSTLShowInfo = _app.GetUserPreferenceToggle((int)swUserPreferenceToggle_e.swSTLShowInfoOnSave);
            _userPreferenceSTLPreview = _app.GetUserPreferenceToggle((int)swUserPreferenceToggle_e.swSTLPreview);
            _userPreferenceHideShowTransition = _app.GetUserPreferenceDoubleValue((int)swUserPreferenceDoubleValue_e.swViewTransitionHideShowComponent);
            _userPreferenceSTLCombine = _app.GetUserPreferenceToggle((int)swUserPreferenceToggle_e.swSTLComponentsIntoOneFile);

            _userPreferenceSTLDeviation = _app.GetUserPreferenceDoubleValue((int)swUserPreferenceDoubleValue_e.swSTLDeviation);
            _userPreferenceSTLAngularTolerance = _app.GetUserPreferenceDoubleValue((int)swUserPreferenceDoubleValue_e.swSTLAngleTolerance);
        }

        public override void SetSTLExportUserPreferences()
        {
            _app.SetUserPreferenceToggle((int)swUserPreferenceToggle_e.swSTLBinaryFormat, true);
            _app.SetUserPreferenceToggle((int)swUserPreferenceToggle_e.swSTLDontTranslateToPositive, true);
            _app.SetUserPreferenceIntegerValue((int)swUserPreferenceIntegerValue_e.swExportStlUnits, 2);
            _app.SetUserPreferenceIntegerValue((int)swUserPreferenceIntegerValue_e.swSTLQuality, (int)swSTLQuality_e.swSTLQuality_Custom);
            _app.SetUserPreferenceToggle((int)swUserPreferenceToggle_e.swSTLShowInfoOnSave, false);
            _app.SetUserPreferenceToggle((int)swUserPreferenceToggle_e.swSTLPreview, false);
            _app.SetUserPreferenceDoubleValue((int)swUserPreferenceDoubleValue_e.swViewTransitionHideShowComponent, 0);
            _app.SetUserPreferenceToggle((int)swUserPreferenceToggle_e.swSTLComponentsIntoOneFile, true);
        }

        public override void ResetSTLExportUserPreferences()
        {
            _app.SetUserPreferenceToggle((int)swUserPreferenceToggle_e.swSTLBinaryFormat, _userPreferenceSTLBinary);
            _app.SetUserPreferenceToggle((int)swUserPreferenceToggle_e.swSTLDontTranslateToPositive, _userPreferenceSTLTranslate);
            _app.SetUserPreferenceIntegerValue((int)swUserPreferenceIntegerValue_e.swExportStlUnits, _userPreferenceSTLUnits);
            _app.SetUserPreferenceIntegerValue((int)swUserPreferenceIntegerValue_e.swSTLQuality, _userPreferenceSTLQuality);
            _app.SetUserPreferenceToggle((int)swUserPreferenceToggle_e.swSTLShowInfoOnSave, _userPreferenceSTLShowInfo);
            _app.SetUserPreferenceToggle((int)swUserPreferenceToggle_e.swSTLPreview, _userPreferenceSTLPreview);
            _app.SetUserPreferenceDoubleValue((int)swUserPreferenceDoubleValue_e.swViewTransitionHideShowComponent, _userPreferenceHideShowTransition);
            _app.SetUserPreferenceToggle((int)swUserPreferenceToggle_e.swSTLComponentsIntoOneFile, _userPreferenceSTLCombine);

            _app.SetUserPreferenceDoubleValue((int)swUserPreferenceDoubleValue_e.swSTLDeviation, _userPreferenceSTLDeviation);
            _app.SetUserPreferenceDoubleValue((int)swUserPreferenceDoubleValue_e.swSTLAngleTolerance, _userPreferenceSTLAngularTolerance);
        }

        public override bool SaveObj(Link link, Link.ComponentType componentType, string windowsMeshFilename, ExporterMeshingOptions meshingOptions, bool exportingCollision)
        {
            throw new NotSupportedException("Solidworks does not support native OBJ export");
        }

        public override bool SaveGlb(Link link, Link.ComponentType componentType, string windowsMeshFilename, ExporterMeshingOptions meshingOptions, bool exportingCollision)
        {
            throw new NotSupportedException("Solidworks does not support native GLB export that we can use");
        }

        public override void UnselectAll()
        {
            _model.ClearSelection2(true);
        }

        [DllImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool SetForegroundWindow(IntPtr hWnd);

        [DllImport("user32.dll")]
        private static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

        [DllImport("user32.dll")]
        private static extern bool IsIconic(IntPtr hWnd);

        private const int SW_RESTORE = 9;

        public override void RestoreHostForeground()
        {
            try
            {
                IntPtr frameHwnd = new IntPtr(((Frame)_app.Frame()).GetHWnd());
                if (frameHwnd == IntPtr.Zero)
                {
                    return;
                }

                // Only restore if actually minimized so we don't clobber a maximized frame.
                if (IsIconic(frameHwnd))
                {
                    ShowWindow(frameHwnd, SW_RESTORE);
                }
                SetForegroundWindow(frameHwnd);
            }
            catch (Exception e)
            {
                logger.Warning("Failed to restore SolidWorks foreground window", e);
            }
        }
    }
}

#endif
