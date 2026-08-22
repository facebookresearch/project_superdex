/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using MathNet.Numerics.LinearAlgebra;

using CADRobotExporter.Export;
using CADRobotExporter.RobotDescription;
using CADRobotExporter.UI;
using System.Collections.Generic;

namespace CADRobotExporter.CAD
{
    /// <summary>
    /// Abstract base class for CAD software bridges.
    /// Provides a common interface for interacting with different CAD systems.
    /// </summary>
    public abstract class CADBridge
    {
        /// <summary>
        /// Whether the joint visualization gizmo is currently being shown.
        /// </summary>
        public bool IsShowingJointGizmo { get; set; }

        /// <summary>
        /// The current LinkNode being visualized in the joint gizmo.
        /// </summary>
        public LinkNode CurrentNodeShown { get; set; }

        /// <summary>
        /// Scale factor for the joint visualization gizmo.
        /// </summary>
        public double JointGizmoScale { get; set; }

        /// <summary>
        /// Scale factor for the link inertial visualization gizmo.
        /// </summary>
        public double LinkGizmoScale { get; set; }

        /// <summary>
        /// Whether the tendon visualization is currently being shown.
        /// </summary>
        public bool IsShowingTendonVisualization { get; set; }

        /// <summary>
        /// Whether all tendons should be drawn simultaneously with distinct colors.
        /// </summary>
        public bool IsShowingAllTendons { get; set; }

        /// <summary>
        /// The full list of tendons to draw when IsShowingAllTendons is true.
        /// </summary>
        public List<Tendon> AllTendons { get; set; }

        /// <summary>
        /// The current tendon being visualized.
        /// </summary>
        public Tendon CurrentTendonShown { get; set; }

        /// <summary>
        /// The routing element index within the current tendon to highlight.
        /// </summary>
        public int TendonHighlightElementIndex { get; set; } = -1;

        /// <summary>
        /// Maps link name to CSYS name for tendon visualization of linear_joint elements.
        /// </summary>
        public Dictionary<string, string> TendonLinkCsysMap { get; set; }

        /// <summary>
        /// Whether the inertial debug visualization (CoM + inertia box) is currently shown.
        /// </summary>
        public bool IsShowingInertialGizmo { get; set; }

        /// <summary>
        /// The current LinkNode being visualized for inertial properties (Link Properties tab).
        /// </summary>
        public LinkNode CurrentLinkNodeShown { get; set; }

        /// <summary>
        /// Gets the title of the current document.
        /// </summary>
        public abstract string GetDocumentTitle();

        /// <summary>
        /// Gets the exporter configuration from the current document.
        /// </summary>
        public abstract ExporterConfiguration GetExporterConfiguration();

        /// <summary>
        /// Sets the latest export location path.
        /// </summary>
        public abstract void SetLatestExportLocation(string location);

        /// <summary>
        /// Gets the latest export location path.
        /// </summary>
        public abstract string GetLatestExportLocation();

        /// <summary>
        /// Saves the configuration from the tree to the document.
        /// </summary>
        /// <param name="config">The exporter configuration.</param>
        /// <param name="baseNode">The root link node.</param>
        /// <param name="warnUser">Whether to warn the user about overwriting.</param>
        /// <param name="featureNameOverride">Optional override for the feature name.</param>
        public abstract void SaveConfigurationFromTree(
            ExporterConfiguration config,
            LinkNode baseNode,
            bool warnUser,
            string featureNameOverride = null);

        /// <summary>
        /// Selects all components associated with a link.
        /// </summary>
        public abstract void SelectLinkComponents(LinkNode node);

        /// <summary>
        /// Selects all joint-related components for a link.
        /// </summary>
        public abstract void SelectJointComponents(LinkNode node);

        /// <summary>
        /// Unselects all objects
        /// </summary>
        public abstract void UnselectAll();

        /// <summary>
        /// Re-asserts the host CAD application's main window as the foreground window.
        /// Called after a modeless form closes so the host does not lose focus or
        /// minimize when Windows transfers activation away from the destroyed owned window.
        /// </summary>
        public abstract void RestoreHostForeground();

        /// <summary>
        /// Hides all components in the assembly.
        /// </summary>
        public abstract void HideAllComponents();

        /// <summary>
        /// Shows all components that were previously hidden.
        /// </summary>
        public abstract void ShowHiddenComponents();

        /// <summary>
        /// Shows or hides the joint visualization overlay.
        /// </summary>
        /// <param name="show">True to show, false to hide.</param>
        public abstract void ShowHideVisualizations(bool show);

        /// <summary>
        /// Triggers a graphics redraw of the model view.
        /// </summary>
        public abstract void TriggerGraphicsRedraw();

        /// <summary>
        /// Enables or disables performance boost mode.
        /// When enabled, disables graphics updates and feature tree updates.
        /// </summary>
        /// <param name="enable">True to enable performance boost, false to disable.</param>
        public abstract void BoostPerformance(bool enable);

        /// <summary>
        /// Starts the progress bar with a maximum value and title.
        /// </summary>
        public abstract void SetProgressBarStart(int maxProgress, string title);

        /// <summary>
        /// Ends the progress bar.
        /// </summary>
        public abstract void SetProgressBarEnd();

        /// <summary>
        /// Updates the progress bar title.
        /// </summary>
        public abstract void SetProgressBarTitle(string title);

        /// <summary>
        /// Updates the progress bar progress value.
        /// </summary>
        public abstract void SetProgressBarProgress(int progress);

        /// <summary>
        /// Gets the inertial properties (mass, center of mass, moment of inertia) for components.
        /// </summary>
        /// <param name="link">The link containing the components.</param>
        /// <param name="componentType">The type of components to analyze.</param>
        /// <param name="coordinateSystemName">The coordinate system to use for calculations.</param>
        /// <param name="mass">Output: the total mass.</param>
        /// <param name="centerOfMass">Output: the center of mass [x, y, z].</param>
        /// <param name="momentOfInertia">Output: the moment of inertia tensor [Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz].</param>
        /// <returns>True if successful, false otherwise.</returns>
        public abstract bool GetComponentsInertialProperties(
            Link link,
            Link.ComponentType componentType,
            string coordinateSystemName,
            out double mass,
            out double[] centerOfMass,
            out double[] momentOfInertia);

        /// <summary>
        /// Gets the visual/material properties for a link's components.
        /// Returns array: [R, G, B, Ambient, Diffuse, Specular, Shininess, Transparency, Emission]
        /// </summary>
        /// <param name="link">The link containing the components.</param>
        /// <param name="componentType">The type of components to get properties for.</param>
        /// <returns>Array of visual property values.</returns>
        public abstract double[] GetVisualProperties(Link link, Link.ComponentType componentType);

        /// <summary>
        /// Gets the axis vector in global space for a named axis.
        /// </summary>
        /// <param name="axisName">The name of the axis.</param>
        /// <param name="flipAxis">Whether to flip the axis direction.</param>
        /// <returns>The normalized axis vector [x, y, z].</returns>
        public abstract double[] GetAxisInGlobalSpace(string axisName, bool flipAxis);

        /// <summary>
        /// Gets the transformation matrix for a named coordinate system.
        /// </summary>
        /// <param name="coordinateSystemName">The name of the coordinate system.</param>
        /// <returns>The 4x4 transformation matrix.</returns>
        public abstract Matrix<double> GetCoordinateSystemTransform(string coordinateSystemName);

        /// <summary>
        /// Gets or creates the top-level coordinate system name for a given coordinate system.
        /// </summary>
        /// <param name="someCoordinateSystemName">The coordinate system name to resolve.</param>
        /// <returns>The top-level coordinate system name.</returns>
        public abstract string GetTopLevelCoordinateSystem(string someCoordinateSystemName);

        /// <summary>
        /// Gets the assembly-global coordinates of a point identified by its persistent key.
        /// Handles points at any level of the assembly hierarchy.
        /// </summary>
        /// <param name="pointKey">The persistent point key (composite key with component path and GUID).</param>
        /// <returns>The point coordinates [x, y, z] in meters, assembly-global frame. Null if not found.</returns>
        public abstract double[] GetTopLevelPointCoordinates(string pointKey);

        /// <summary>
        /// Creates a coordinate system from a 4x4 transformation matrix.
        /// </summary>
        /// <param name="transform">4x4 homogeneous transformation matrix (in meters).</param>
        /// <param name="name">Name for the coordinate system.</param>
        /// <returns>A key/name that can be used to reference the created coordinate system.</returns>
        public abstract string CreateCoordinateSystemFromTransform(Matrix<double> transform, string name);

        /// <summary>
        /// Creates an axis (datum axis) from a direction vector at an origin point.
        /// Only creates an axis if the direction is NOT colinear with principal axes (X, Y, Z).
        /// </summary>
        /// <param name="origin">Origin point [x, y, z] in meters.</param>
        /// <param name="direction">Direction vector [x, y, z] (normalized).</param>
        /// <param name="name">Name for the axis.</param>
        /// <returns>A key/name that can be used to reference the created axis, or null if not created/supported.</returns>
        public abstract string CreateAxisFromDirection(double[] origin, double[] direction, string name);

        /// <summary>
        /// Cleans up any temporary features created during export.
        /// </summary>
        public abstract void CleanUpTemporaryFeatures();

        /// <summary>
        /// Gets the bounding box of the model.
        /// </summary>
        /// <param name="boundingBox">Output: [minX, minY, minZ, maxX, maxY, maxZ].</param>
        public abstract void GetModelBoundingBox(out double[] boundingBox);

        /// <summary>
        /// Saves the link's components as a STEP file.
        /// </summary>
        /// <param name="link">The link containing the components to export.</param>
        /// <param name="componentType">The type of components to export.</param>
        /// <param name="windowsMeshFilename">The full path for the output file.</param>
        /// <returns>True if successful, false otherwise.</returns>
        public abstract bool SaveStpFile(Link link, Link.ComponentType componentType, string windowsMeshFilename);

        /// <summary>
        /// Saves the link's components as an STL file, if supported by the CAD software's native API
        /// </summary>
        /// <param name="link">The link containing the components to export.</param>
        /// <param name="componentType">The type of components to export.</param>
        /// <param name="windowsMeshFilename">The full path for the output file.</param>
        /// <param name="meshingOptions">The meshing options for export quality.</param>
        /// <param name="exportingCollision">Whether this is for collision geometry (uses different mesh settings).</param>
        /// <param name="coordinateSystemName">Coordinate system used for export</param>
        /// <returns>True if successful, false otherwise.</returns>
        public abstract bool SaveStl(
            Link link,
            Link.ComponentType componentType,
            string windowsMeshFilename,
            ExporterMeshingOptions meshingOptions,
            bool exportingCollision);

        /// <summary>
        /// Saves the link's components as an OBJ file, if supported by the CAD software's native API
        /// </summary>
        /// <param name="link">The link containing the components to export.</param>
        /// <param name="componentType">The type of components to export.</param>
        /// <param name="windowsMeshFilename">The full path for the output file.</param>
        /// <param name="meshingOptions">The meshing options for export quality.</param>
        /// <param name="exportingCollision">Whether this is for collision geometry (uses different mesh settings).</param>
        /// <param name="coordinateSystemName">Coordinate system used for export</param>
        /// <returns>True if successful, false otherwise.</returns>
        public abstract bool SaveObj(
            Link link,
            Link.ComponentType componentType,
            string windowsMeshFilename,
            ExporterMeshingOptions meshingOptions,
            bool exportingCollision);

        /// <summary>
        /// Saves the link's components as an OBJ file, if supported by the CAD software's native API
        /// </summary>
        /// <param name="link">The link containing the components to export.</param>
        /// <param name="componentType">The type of components to export.</param>
        /// <param name="windowsMeshFilename">The full path for the output file.</param>
        /// <param name="meshingOptions">The meshing options for export quality.</param>
        /// <param name="exportingCollision">Whether this is for collision geometry (uses different mesh settings).</param>
        /// <param name="coordinateSystemName">Coordinate system used for export</param>
        /// <returns>True if successful, false otherwise.</returns>
        public abstract bool SaveGlb(
            Link link,
            Link.ComponentType componentType,
            string windowsMeshFilename,
            ExporterMeshingOptions meshingOptions,
            bool exportingCollision);

        /// <summary>
        /// Sets link-specific STEP export preferences, if required by the CAD software's native API
        /// </summary>
        /// <param name="coordinateSystemName">The coordinate system to use for export.</param>
        protected abstract void SetLinkSpecificSTEPPreferences(string coordinateSystemName);

        /// <summary>
        /// Sets link-specific STL export preferences, if required by the CAD software's native API
        /// </summary>
        /// <param name="coordinateSystemName">The coordinate system to use for export.</param>
        /// <param name="linearDeviation">The linear deviation tolerance.</param>
        /// <param name="angularDeviation">The angular deviation tolerance.</param>
        protected abstract void SetLinkSpecificSTLPreferences(string coordinateSystemName, double linearDeviation, double angularDeviation);

        /// <summary>
        /// Saves the current STL export user preferences so they can be restored later, if required by the CAD software's native API
        /// </summary>
        public abstract void SaveSTLExportUserPreferences();

        /// <summary>
        /// Sets the STL export preferences to optimal values for URDF export, if required by the CAD software's native API
        /// </summary>
        public abstract void SetSTLExportUserPreferences();

        /// <summary>
        /// Resets the STL export preferences to the previously saved values, if required by the CAD software's native API
        /// </summary>
        public abstract void ResetSTLExportUserPreferences();

        /// <summary>
        /// Gets the Work Coordinate System (WCS) transformation matrix.
        /// This is the current user-defined coordinate system in the CAD environment.
        /// </summary>
        /// <returns>The 4x4 transformation matrix of the WCS (in meters), or identity if not applicable.</returns>
        public virtual Matrix<double> GetWorkCoordinateSystemTransform()
        {
            return Matrix<double>.Build.DenseIdentity(4, 4);
        }
    }
}
