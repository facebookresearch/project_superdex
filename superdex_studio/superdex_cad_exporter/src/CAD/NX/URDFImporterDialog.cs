/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.IO;

using MathNet.Numerics.LinearAlgebra;

using NXOpen;
using NXOpen.BlockStyler;

using CADRobotExporter.Import;
using CADRobotExporter.Utilities;

namespace CADRobotExporter.CAD.NX
{
    public class URDFImporterDialog
    {
        private static readonly Serilog.ILogger logger = Logger.GetLogger();

        private static Session session = null;
        private static NXOpen.UI nxUI = null;
        private Part workPart = null;

        private string dlxFileName = "URDFImporterDialog.dlx";
        private NXOpen.BlockStyler.BlockDialog dialog;

        private NXOpen.BlockStyler.Group group0;
        private NXOpen.BlockStyler.Enumeration enumCsysMode;
        private NXOpen.BlockStyler.SpecifyCSYS selectionCsys;
        private NXOpen.BlockStyler.Toggle toggleCreateCsys;
        private NXOpen.BlockStyler.Toggle toggleCreateRobotConfiguration;
        private NXOpen.BlockStyler.Toggle toggleImportVisualMeshes;
        private NXOpen.BlockStyler.Toggle toggleImportCollisionMeshes;
        private NXOpen.BlockStyler.Group group;
        private NXOpen.BlockStyler.FileSelection nativeFileBrowser;

        public URDFImporterDialog()
        {
            session = Session.GetSession();
            nxUI = NXOpen.UI.GetUI();
            workPart = session.Parts.Work;

            dialog = nxUI.CreateDialog(dlxFileName);
            dialog.AddApplyHandler(new NXOpen.BlockStyler.BlockDialog.Apply(ApplyCallback));
            dialog.AddOkHandler(new NXOpen.BlockStyler.BlockDialog.Ok(OkCallback));
            dialog.AddUpdateHandler(new NXOpen.BlockStyler.BlockDialog.Update(UpdateCallback));
            dialog.AddInitializeHandler(new NXOpen.BlockStyler.BlockDialog.Initialize(InitializeCallback));
            dialog.AddDialogShownHandler(new NXOpen.BlockStyler.BlockDialog.DialogShown(DialogShownCallback));
        }

        public static int GetUnloadOption(string arg)
        {
            return System.Convert.ToInt32(Session.LibraryUnloadOption.AtTermination);
        }

        public static void UnloadLibrary(string arg)
        {
        }

        public NXOpen.BlockStyler.BlockDialog.DialogResponse Launch()
        {
            NXOpen.BlockStyler.BlockDialog.DialogResponse dialogResponse =
                NXOpen.BlockStyler.BlockDialog.DialogResponse.Invalid;
            try
            {
                dialogResponse = dialog.Launch();
            }
            catch (Exception ex)
            {
                nxUI.NXMessageBox.Show("URDF Import", NXMessageBox.DialogType.Error, ex.ToString());
            }
            return dialogResponse;
        }

        public void Dispose()
        {
            if (dialog != null)
            {
                dialog.Dispose();
                dialog = null;
            }
        }

        public void InitializeCallback()
        {
            try
            {
                group0 = (NXOpen.BlockStyler.Group)dialog.TopBlock.FindBlock("group0");
                enumCsysMode = (NXOpen.BlockStyler.Enumeration)dialog.TopBlock.FindBlock("enumCsysMode");
                selectionCsys = (NXOpen.BlockStyler.SpecifyCSYS)dialog.TopBlock.FindBlock("selectionCsys");
                toggleCreateCsys = (NXOpen.BlockStyler.Toggle)dialog.TopBlock.FindBlock("toggleCreateCsys");
                toggleCreateRobotConfiguration = (NXOpen.BlockStyler.Toggle)dialog.TopBlock.FindBlock("toggleCreateRobotConfiguration");
                toggleImportVisualMeshes = (NXOpen.BlockStyler.Toggle)dialog.TopBlock.FindBlock("toggleImportVisualMeshes");
                toggleImportCollisionMeshes = (NXOpen.BlockStyler.Toggle)dialog.TopBlock.FindBlock("toggleImportCollisionMeshes");
                group = (NXOpen.BlockStyler.Group)dialog.TopBlock.FindBlock("group");
                nativeFileBrowser = (NXOpen.BlockStyler.FileSelection)dialog.TopBlock.FindBlock("nativeFileBrowser");

                nativeFileBrowser.Filter = ".urdf,.*";
            }
            catch (Exception ex)
            {
                nxUI.NXMessageBox.Show("URDF Import", NXMessageBox.DialogType.Error, ex.ToString());
            }
        }

        public void DialogShownCallback()
        {
            try
            {
                toggleCreateCsys.Value = true;
                toggleCreateRobotConfiguration.Value = true;
                toggleImportVisualMeshes.Value = true;
                toggleImportCollisionMeshes.Value = true;

                UpdateCsysVisibility();
            }
            catch (Exception ex)
            {
                nxUI.NXMessageBox.Show("URDF Import", NXMessageBox.DialogType.Error, ex.ToString());
            }
        }

        public int UpdateCallback(NXOpen.BlockStyler.UIBlock block)
        {
            try
            {
                if (block == enumCsysMode)
                {
                    UpdateCsysVisibility();
                }
            }
            catch (Exception ex)
            {
                nxUI.NXMessageBox.Show("URDF Import", NXMessageBox.DialogType.Error, ex.ToString());
            }
            return 0;
        }

        public int OkCallback()
        {
            return ApplyCallback();
        }

        public int ApplyCallback()
        {
            int errorCode = 0;
            try
            {
                string urdfPath = nativeFileBrowser.Path;

                if (string.IsNullOrEmpty(urdfPath) || !File.Exists(urdfPath))
                {
                    nxUI.NXMessageBox.Show("URDF Import", NXMessageBox.DialogType.Error,
                        "Please select a valid URDF file.");
                    return 1;
                }

                var issues = URDFImporter.ValidateUrdfFile(urdfPath);
                if (issues.Count > 0)
                {
                    string issueList = string.Join("\n", issues);
                    int proceed = nxUI.NXMessageBox.Show("URDF Validation",
                        NXMessageBox.DialogType.Question,
                        $"URDF file has issues:\n\n{issueList}\n\nProceed anyway?");
                    if (proceed != 1)
                        return 1;
                }

                MeshImportMode meshMode = MeshImportMode.None;
                if (toggleImportVisualMeshes.Value && toggleImportCollisionMeshes.Value)
                    meshMode = MeshImportMode.Both;
                else if (toggleImportVisualMeshes.Value)
                    meshMode = MeshImportMode.VisualOnly;
                else if (toggleImportCollisionMeshes.Value)
                    meshMode = MeshImportMode.CollisionOnly;

                if (!TryGetBaseTransform(out Matrix<double> baseTransform))
                    return 1;

                var config = new URDFImportConfiguration
                {
                    UrdfFilePath = urdfPath,
                    CreateCoordinateSystems = toggleCreateCsys.Value,
                    CreateRobotConfigurationFeature = toggleCreateRobotConfiguration.Value,
                    MeshBasePath = Path.GetDirectoryName(urdfPath),
                    ImportMeshes = meshMode,
                    BaseTransform = baseTransform,
                };

                var result = URDFImporter.Import(workPart, config);

                if (result.Success)
                {
                    string message = $"URDF imported successfully!\n\n" +
                        $"Robot: {result.RobotName}\n" +
                        $"Coordinate systems created: {result.CreatedCoordinateSystems.Count}";

                    if (result.Warnings.Count > 0)
                    {
                        message += $"\n\nWarnings ({result.Warnings.Count}):\n" +
                            string.Join("\n", result.Warnings);
                    }

                    nxUI.NXMessageBox.Show("URDF Import", NXMessageBox.DialogType.Information, message);
                }
                else
                {
                    nxUI.NXMessageBox.Show("URDF Import", NXMessageBox.DialogType.Error,
                        $"Failed to import URDF:\n{result.ErrorMessage}");
                    errorCode = 1;
                }
            }
            catch (Exception ex)
            {
                logger.Error($"URDF import failed: {ex.Message}");
                nxUI.NXMessageBox.Show("URDF Import", NXMessageBox.DialogType.Error, ex.ToString());
                errorCode = 1;
            }
            return errorCode;
        }

        private void UpdateCsysVisibility()
        {
            string mode = enumCsysMode.ValueAsString;
            bool showCsysSelector = (mode == "Select CSYS");

            PropertyList csysProps = selectionCsys.GetProperties();
            csysProps.SetLogical("Show", showCsysSelector);
            csysProps.Dispose();
        }

        /// <summary>
        /// Resolves the base/world frame the robot is imported into, based on the CSYS mode.
        /// A null transform means "use the current WCS". Returns false if the user picked
        /// "Select CSYS" without selecting one.
        /// </summary>
        private bool TryGetBaseTransform(out Matrix<double> baseTransform)
        {
            baseTransform = null;

            switch (enumCsysMode.ValueAsString)
            {
                case "Absolute":
                    baseTransform = Matrix<double>.Build.DenseIdentity(4, 4);
                    return true;

                case "WCS":
                    return true;

                case "Select CSYS":
                    CartesianCoordinateSystem csys = GetSelectedCoordinateSystem();
                    if (csys == null)
                    {
                        nxUI.NXMessageBox.Show("URDF Import", NXMessageBox.DialogType.Error,
                            "Please specify a coordinate system, or change Base/World CSYS to Absolute or WCS.");
                        return false;
                    }
                    baseTransform = new NXBridge(workPart).GetCoordinateSystemTransform(csys);
                    return true;

                default:
                    logger.Warning($"Unknown CSYS mode '{enumCsysMode.ValueAsString}', falling back to WCS");
                    return true;
            }
        }

        private CartesianCoordinateSystem GetSelectedCoordinateSystem()
        {
            if (selectionCsys == null)
                return null;

            try
            {
                TaggedObject[] selectedObjects = selectionCsys.GetSelectedObjects();
                foreach (TaggedObject taggedObj in selectedObjects)
                {
                    if (taggedObj is CartesianCoordinateSystem csys)
                        return csys;
                }
            }
            catch (Exception ex)
            {
                logger.Error($"Failed to read the selected coordinate system: {ex.Message}");
                return null;
            }

            return null;
        }
    }
}
