/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Collections.Generic;
using System.IO;

using Meshing;

using NXOpen;
using NXOpen.BlockStyler;
using NXOpen.UF;

namespace CADRobotExporter.CAD.NX
{
    /// <summary>
    /// Block Styler dialog that tessellates the selected bodies via a temporary STEP export,
    /// writing STL/OBJ/GLB with superdex_mesh_cli.
    /// </summary>
    public class SuperDexMeshExporterDialog
    {
        private static Session session = null;
        private static NXOpen.UI nxUI = null;
        private static UFSession ufSession = null;
        private Part workPart = null;
        private string meshFilePath = "";
        private readonly string dlxFileName = "SuperDexMeshExporter.dlx";

        private BlockDialog dialog;
        private NXOpen.BlockStyler.Group group0;
        private BodyCollector bodySelection;
        private SpecifyCSYS csysSelection;
        private NXOpen.BlockStyler.Group group1;
        private DoubleBlock doubleLinearDeflection;
        private DoubleBlock doubleAngularDeflection;
        private DoubleBlock doubleScale;
        private NXOpen.BlockStyler.Group group;
        private Toggle toggleGlb;
        private Toggle toggleStl;
        private Toggle toggleObj;
        private Toggle toggleStep;
        private NXOpen.BlockStyler.Group group2;
        private Toggle toggleIsotropicMesher;
        private Toggle toggleAdaptiveSampling;
        private DoubleBlock doubleTargetEdgeLength;
        private DoubleBlock doubleTargetEdgeLengthFraction;
        private Button buttonClearSelections;
        private Button buttonRestoreDefaults;
        private Button buttonSave;

        public static readonly int UF_UI_OK = 2;
        public static readonly int UF_UI_CANCEL = 3;

        public SuperDexMeshExporterDialog()
        {
            session = Session.GetSession();
            nxUI = NXOpen.UI.GetUI();
            ufSession = UFSession.GetUFSession();
            workPart = session.Parts.Work;
            dialog = nxUI.CreateDialog(dlxFileName);
            dialog.AddApplyHandler(new BlockDialog.Apply(ApplyCallback));
            dialog.AddOkHandler(new BlockDialog.Ok(OkCallback));
            dialog.AddUpdateHandler(new BlockDialog.Update(UpdateCallback));
            dialog.AddInitializeHandler(new BlockDialog.Initialize(InitializeCallback));
            dialog.AddDialogShownHandler(new BlockDialog.DialogShown(DiaglogShownCallback));
        }

        /// <summary>Menu action callback, registered by ExportRobotDialog.Startup.</summary>
        public static NXOpen.MenuBar.MenuBarManager.CallbackStatus LaunchSuperDexMeshExporter(
            NXOpen.MenuBar.MenuButtonEvent buttonEvent)
        {
            if (Session.GetSession().Parts.Work == null)
            {
                return NXOpen.MenuBar.MenuBarManager.CallbackStatus.Cancel;
            }

            SuperDexMeshExporterDialog exporter = null;
            try
            {
                exporter = new SuperDexMeshExporterDialog();
                exporter.Launch();
            }
            catch (Exception ex)
            {
                nxUI.NXMessageBox.Show(
                    "SuperDex Mesh Exporter", NXMessageBox.DialogType.Error, ex.ToString());
            }
            finally
            {
                exporter?.Dispose();
            }
            return NXOpen.MenuBar.MenuBarManager.CallbackStatus.Continue;
        }

        public BlockDialog.DialogResponse Launch()
        {
            BlockDialog.DialogResponse dialogResponse = BlockDialog.DialogResponse.Invalid;
            try
            {
                dialogResponse = dialog.Launch();
            }
            catch (Exception ex)
            {
                nxUI.NXMessageBox.Show("Block Styler", NXMessageBox.DialogType.Error, ex.ToString());
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
                bodySelection = (BodyCollector)dialog.TopBlock.FindBlock("bodySelection");
                csysSelection = (SpecifyCSYS)dialog.TopBlock.FindBlock("csysSelection");
                group1 = (NXOpen.BlockStyler.Group)dialog.TopBlock.FindBlock("group1");
                doubleLinearDeflection = (DoubleBlock)dialog.TopBlock.FindBlock("doubleLinearDeflection");
                doubleAngularDeflection = (DoubleBlock)dialog.TopBlock.FindBlock("doubleAngularDeflection");
                doubleScale = (DoubleBlock)dialog.TopBlock.FindBlock("doubleScale");
                group = (NXOpen.BlockStyler.Group)dialog.TopBlock.FindBlock("group");
                toggleGlb = (Toggle)dialog.TopBlock.FindBlock("toggleGlb");
                toggleStl = (Toggle)dialog.TopBlock.FindBlock("toggleStl");
                toggleObj = (Toggle)dialog.TopBlock.FindBlock("toggleObj");
                toggleStep = (Toggle)dialog.TopBlock.FindBlock("toggleStep");
                group2 = (NXOpen.BlockStyler.Group)dialog.TopBlock.FindBlock("group2");
                buttonSave = (Button)dialog.TopBlock.FindBlock("buttonSave");
                buttonRestoreDefaults = (Button)dialog.TopBlock.FindBlock("buttonRestoreDefaults");
                buttonClearSelections = (Button)dialog.TopBlock.FindBlock("buttonClearSelections");

                toggleIsotropicMesher = TryFindBlock<Toggle>("toggleIsotropicMesher");
                toggleAdaptiveSampling = TryFindBlock<Toggle>("toggleAdaptiveSampling");
                doubleTargetEdgeLength = TryFindBlock<DoubleBlock>("doubleTargetEdgeLength");
                doubleTargetEdgeLengthFraction =
                    TryFindBlock<DoubleBlock>("doubleTargetEdgeLengthFraction");

                // Default the output path alongside the part.
                if (workPart != null)
                {
                    string partPath = workPart.FullPath;
                    meshFilePath = Path.Combine(
                        Path.GetDirectoryName(partPath),
                        Path.GetFileNameWithoutExtension(partPath));
                }
            }
            catch (Exception ex)
            {
                nxUI.NXMessageBox.Show("Block Styler", NXMessageBox.DialogType.Error, ex.ToString());
            }
        }

        private T TryFindBlock<T>(string blockId)
            where T : UIBlock
        {
            try
            {
                return dialog.TopBlock.FindBlock(blockId) as T;
            }
            catch (Exception)
            {
                return null;
            }
        }

        public void DiaglogShownCallback()
        {
        }

        public int ApplyCallback()
        {
            return 0;
        }

        public int UpdateCallback(UIBlock block)
        {
            try
            {
                if (block == buttonSave)
                {
                    if (bodySelection.GetSelectedObjects().Length == 0)
                    {
                        nxUI.NXMessageBox.Show(
                            "Save Mesh", NXMessageBox.DialogType.Error, "No bodies selected.");
                        return 0;
                    }

                    if (!toggleGlb.Value && !toggleStl.Value && !toggleObj.Value && !toggleStep.Value)
                    {
                        nxUI.NXMessageBox.Show(
                            "Mesh Exporter",
                            NXMessageBox.DialogType.Error,
                            "No mesh types to export, please select at least one.");
                        return 0;
                    }

                    string selectedFile = ShowFileSaveDialog();
                    if (!string.IsNullOrEmpty(selectedFile))
                    {
                        meshFilePath = selectedFile;
                        if (SaveMesh())
                        {
                            nxUI.NXMessageBox.Show(
                                "Save Mesh", NXMessageBox.DialogType.Information, "Export complete!");
                        }
                    }
                }

                if (block == buttonRestoreDefaults)
                {
                    doubleLinearDeflection.Value = 0.1;
                    doubleAngularDeflection.Value = 0.5;
                    doubleScale.Value = 1.0;

                    var defaults = new MeshExportOptions();
                    if (toggleIsotropicMesher != null)
                    {
                        toggleIsotropicMesher.Value = defaults.Backend == MeshingBackend.Isotropic;
                    }
                    if (toggleAdaptiveSampling != null)
                    {
                        toggleAdaptiveSampling.Value = defaults.EdgeSampling == EdgeSampling.Adaptive;
                    }
                    if (doubleTargetEdgeLength != null)
                    {
                        doubleTargetEdgeLength.Value = defaults.TargetEdgeLength;
                    }
                    if (doubleTargetEdgeLengthFraction != null)
                    {
                        doubleTargetEdgeLengthFraction.Value = defaults.TargetEdgeLengthFraction;
                    }
                }

                if (block == buttonClearSelections)
                {
                    bodySelection.SetSelectedObjects(new TaggedObject[] { });
                    csysSelection.SetSelectedObjects(new TaggedObject[] { });
                }
            }
            catch (Exception ex)
            {
                nxUI.NXMessageBox.Show("Block Styler", NXMessageBox.DialogType.Error, ex.ToString());
            }
            return 0;
        }

        public int OkCallback()
        {
            int errorCode = 0;
            try
            {
                errorCode = ApplyCallback();
            }
            catch (Exception ex)
            {
                errorCode = 1;
                nxUI.NXMessageBox.Show("Block Styler", NXMessageBox.DialogType.Error, ex.ToString());
            }
            return errorCode;
        }

        public PropertyList GetBlockProperties(string blockID)
        {
            PropertyList plist = null;
            try
            {
                plist = dialog.GetBlockProperties(blockID);
            }
            catch (Exception ex)
            {
                nxUI.NXMessageBox.Show("Block Styler", NXMessageBox.DialogType.Error, ex.ToString());
            }
            return plist;
        }

        private bool SaveMesh()
        {
            if (workPart == null)
            {
                nxUI.NXMessageBox.Show(
                    "Save Mesh", NXMessageBox.DialogType.Error, "No part is currently open.");
                return false;
            }

            string meshFilename = meshFilePath;
            if (string.IsNullOrEmpty(meshFilename))
            {
                nxUI.NXMessageBox.Show(
                    "Save Mesh", NXMessageBox.DialogType.Error, "Please specify an output file path.");
                return false;
            }

            List<Body> selectedBodies = GetSelectedBodies();
            CoordinateSystem coordSys = GetSelectedCoordinateSystem();

            // Every format is written from one tessellation, so all overwrite prompts happen here,
            // before anything is exported.
            var outputs = new List<MeshExportOutput>();
            if (toggleStl.Value && ConfirmOverwrite(meshFilename + ".stl"))
            {
                outputs.Add(new MeshExportOutput(MeshExportFormat.Stl, meshFilename + ".stl"));
            }
            if (toggleObj.Value && ConfirmOverwrite(meshFilename + ".obj"))
            {
                outputs.Add(new MeshExportOutput(MeshExportFormat.Obj, meshFilename + ".obj"));
            }
            if (toggleGlb.Value && ConfirmOverwrite(meshFilename + ".glb"))
            {
                outputs.Add(new MeshExportOutput(MeshExportFormat.Glb, meshFilename + ".glb"));
            }

            bool exportStep = toggleStep.Value;
            if (outputs.Count == 0 && !exportStep)
            {
                nxUI.NXMessageBox.Show(
                    "Save Mesh",
                    NXMessageBox.DialogType.Warning,
                    "Please select at least one export format.");
                return false;
            }

            string stepPath;
            bool keepStepFile = exportStep;

            if (keepStepFile)
            {
                stepPath = meshFilename + ".step";
                if (!ConfirmOverwrite(stepPath))
                {
                    keepStepFile = false;
                    // The user declined to replace the STEP but may still want the meshes, which
                    // need a STEP to come from.
                    if (outputs.Count == 0)
                    {
                        return false;
                    }
                    stepPath = Path.Combine(Path.GetTempPath(), $"nxopen_{Guid.NewGuid()}.step");
                }
            }
            else
            {
                stepPath = Path.Combine(Path.GetTempPath(), $"nxopen_{Guid.NewGuid()}.step");
            }

            if (!SaveSTEP(coordSys, stepPath, selectedBodies))
            {
                nxUI.NXMessageBox.Show(
                    "Save Mesh", NXMessageBox.DialogType.Error, "Failed to export STEP file.");
                return false;
            }

            bool exported = true;
            if (outputs.Count > 0)
            {
                var options = new MeshExportOptions
                {
                    LinearDeflection = doubleLinearDeflection.Value,
                    AngularDeflection = doubleAngularDeflection.Value,
                    Scale = MeshExportOptions.ScaleFromExporterUnits(doubleScale.Value),
                    AllowPartialFailure = true,
                };

                if (toggleIsotropicMesher != null)
                {
                    options.Backend = toggleIsotropicMesher.Value
                        ? MeshingBackend.Isotropic
                        : MeshingBackend.Delabella;
                }
                if (toggleAdaptiveSampling != null)
                {
                    options.EdgeSampling = toggleAdaptiveSampling.Value
                        ? EdgeSampling.Adaptive
                        : EdgeSampling.Uniform;
                }
                if (doubleTargetEdgeLength != null)
                {
                    options.TargetEdgeLength = doubleTargetEdgeLength.Value;
                }
                if (doubleTargetEdgeLengthFraction != null)
                {
                    options.TargetEdgeLengthFraction = doubleTargetEdgeLengthFraction.Value;
                }

                try
                {
                    IList<MeshExportStatus> statuses =
                        MeshExporter.Export(stepPath, outputs, options);

                    var problems = new List<string>();
                    for (int i = 0; i < statuses.Count; ++i)
                    {
                        string name = Path.GetFileName(outputs[i].Path);
                        if (statuses[i] == MeshExportStatus.Failed)
                        {
                            problems.Add($"{name}: not written");
                        }
                        else if (statuses[i] == MeshExportStatus.WrittenPartial)
                        {
                            problems.Add($"{name}: written, but some faces failed to mesh");
                        }
                    }

                    if (problems.Count > 0)
                    {
                        exported = false;
                        nxUI.NXMessageBox.Show(
                            "Save Mesh",
                            NXMessageBox.DialogType.Warning,
                            "Mesh export finished with problems:\n" + string.Join("\n", problems));
                    }
                }
                catch (MeshCliException ex)
                {
                    exported = false;
                    nxUI.NXMessageBox.Show(
                        "Save Mesh",
                        NXMessageBox.DialogType.Error,
                        "Mesh export failed:\n" + ex.Message);
                }
            }

            if (!keepStepFile)
            {
                try
                {
                    File.Delete(stepPath);
                }
                catch (Exception ex)
                {
                    nxUI.NXMessageBox.Show(
                        "Save Mesh",
                        NXMessageBox.DialogType.Warning,
                        "Couldn't delete temporary STEP file: \n" + ex.Message);
                }
            }

            return exported;
        }

        private bool SaveSTEP(CoordinateSystem coordSys, string stepFilePath, List<Body> selectedBodies)
        {
            try
            {
                StepCreator stepCreator = session.DexManager.CreateStepCreator();

                stepCreator.ExportAs = StepCreator.ExportAsOption.Ap242;
                stepCreator.ExportFrom = StepCreator.ExportFromOption.DisplayPart;

                stepCreator.ObjectTypes.Csys = true;
                stepCreator.ObjectTypes.Solids = true;
                stepCreator.ObjectTypes.Surfaces = true;

                stepCreator.InputFile = workPart.FullPath;
                stepCreator.OutputFile = stepFilePath;

                if (selectedBodies != null && selectedBodies.Count > 0)
                {
                    stepCreator.ExportSelectionBlock.SelectionScope = ObjectSelector.Scope.SelectedObjects;
                    foreach (NXObject obj in selectedBodies)
                    {
                        stepCreator.ExportSelectionBlock.SelectionComp.Add(obj);
                    }
                }
                else
                {
                    stepCreator.ExportSelectionBlock.SelectionScope = ObjectSelector.Scope.EntireAssembly;
                }
                stepCreator.FileSaveFlag = false;
                stepCreator.LayerMask = "1-256";
                stepCreator.ColorAndLayers = true;

                // NB: this causes NX to wait for the export process to finish first before
                // returning, otherwise the export is async.
                stepCreator.ProcessHoldFlag = true;

                if (coordSys != null)
                {
                    stepCreator.ReferenceType = StepCreator.CsysrefEnum.SpecifiedCsys;
                    stepCreator.Csys = coordSys;
                }

                stepCreator.Commit();
                stepCreator.Destroy();

                return File.Exists(stepFilePath);
            }
            catch (Exception ex)
            {
                nxUI.NXMessageBox.Show(
                    "Save Mesh", NXMessageBox.DialogType.Error, "Error exporting STEP: " + ex.Message);
                return false;
            }
        }

        private List<Body> GetSelectedBodies()
        {
            var selectedBodies = new List<Body>();

            if (workPart == null || bodySelection == null)
            {
                return selectedBodies;
            }

            try
            {
                foreach (TaggedObject taggedObj in bodySelection.GetSelectedObjects())
                {
                    if (taggedObj is Body body)
                    {
                        selectedBodies.Add(body);
                    }
                }
            }
            catch (Exception)
            {
                // Return an empty list if the selection cannot be read.
            }

            return selectedBodies;
        }

        private CoordinateSystem GetSelectedCoordinateSystem()
        {
            if (csysSelection == null)
            {
                return null;
            }

            try
            {
                foreach (TaggedObject taggedObj in csysSelection.GetSelectedObjects())
                {
                    if (taggedObj is CoordinateSystem csys)
                    {
                        return csys;
                    }
                }
            }
            catch (Exception)
            {
                return null;
            }

            return null;
        }

        private static bool ConfirmOverwrite(string filePath)
        {
            if (!File.Exists(filePath))
            {
                return true;
            }

            int response = nxUI.NXMessageBox.Show(
                "Confirm Save As",
                NXMessageBox.DialogType.Question,
                $"{Path.GetFileName(filePath)} already exists.\nDo you want to replace it?");

            // Show with Question returns 1 for Yes/OK.
            return response == 1;
        }

        private string ShowFileSaveDialog()
        {
            string defaultName = meshFilePath;
            if (string.IsNullOrEmpty(defaultName) && workPart != null)
            {
                string partPath = workPart.FullPath;
                defaultName = Path.Combine(
                    Path.GetDirectoryName(partPath),
                    Path.GetFileNameWithoutExtension(partPath));
            }

            string filterString = "*.*";
            ufSession.Ui.CreateFilebox(
                "Select output file location",
                "Save Mesh As",
                ref filterString,
                defaultName,
                out string fileName,
                out int response);

            if (response == UF_UI_OK)
            {
                // Drop any extension; one is added per selected format.
                return Path.Combine(
                    Path.GetDirectoryName(fileName),
                    Path.GetFileNameWithoutExtension(fileName));
            }

            return null;
        }
    }
}
