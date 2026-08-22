/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using System.Windows.Forms;

using Meshing;

using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;

using CADRobotExporter.RobotExport;
using CADRobotExporter.CAD;

namespace CADRobotExporter.UI
{
    public partial class MeshSaveForm : Form
    {
        public ModelDoc2 ActiveSWModel;
        public SldWorks SWApp = null;
        public string MeshFilename = "";

        public static string DefaultCoordSysName = "-- default --";

        public MeshSaveForm(SldWorks iSldWorksApp, string solidworksSaveAsString)
        {
            InitializeComponent();

            SWApp = iSldWorksApp;
            ActiveSWModel = (ModelDoc2)SWApp.ActiveDoc;

            buttonSave.Click += ButtonSave_Click;
            buttonCancel.Click += ButtonCancel_Click;

            checkBoxGlb.CheckedChanged += CheckBox_CheckedChanged;
            checkBoxObj.CheckedChanged += CheckBox_CheckedChanged;
            checkBoxStl.CheckedChanged += CheckBox_CheckedChanged;
            checkBoxStep.CheckedChanged += CheckBox_CheckedChanged;

            comboBoxBackend.SelectedIndex = 0; // Isotropic
            comboBoxEdgeSampling.SelectedIndex = 0; // Adaptive
            comboBoxBackend.SelectedIndexChanged += MesherSettingChanged;
            numericUpDownTargetEdgeLength.ValueChanged += MesherSettingChanged;
            UpdateMesherControlEnabledState();

            AddCoordinateSystems();
            MeshFilename = ExtractFilePathWithoutExtension(solidworksSaveAsString);
        }

        private void MesherSettingChanged(object sender, EventArgs e)
        {
            UpdateMesherControlEnabledState();
        }

        /// <summary>
        /// Delabella ignores the sizing controls, and the bounding-box fraction only applies while
        /// no absolute edge length is set, so both are greyed out rather than left to mislead.
        /// </summary>
        private void UpdateMesherControlEnabledState()
        {
            bool isotropic = comboBoxBackend.SelectedIndex != 1;

            comboBoxEdgeSampling.Enabled = isotropic;
            numericUpDownTargetEdgeLength.Enabled = isotropic;
            numericUpDownEdgeLengthFraction.Enabled =
                isotropic && numericUpDownTargetEdgeLength.Value == 0M;
        }

        private void CheckBox_CheckedChanged(object sender, EventArgs e)
        {
            buttonSave.Enabled = (checkBoxGlb.Checked || checkBoxObj.Checked || checkBoxStl.Checked
                || checkBoxStep.Checked);
        }

        private void ButtonSave_Click(object sender, EventArgs e)
        {
            if (SaveMesh())
            {
                MessageBox.Show("Export complete");
            }
            Close();
        }

        private void ButtonCancel_Click(object sender, EventArgs e)
        {
            Close();
        }

        public void AddCoordinateSystems()
        {
            comboBoxCoordinateSystem.Items.Clear();
            comboBoxCoordinateSystem.Items.Add(DefaultCoordSysName);
            comboBoxCoordinateSystem.SelectedIndex = 0;

            object[] features = ActiveSWModel.FeatureManager.GetFeatures(true) as object[];

            if (features != null)
            {
                foreach (Feature feat in features)
                {
                    if (feat.GetTypeName2() == "CoordSys")
                    {
                        comboBoxCoordinateSystem.Items.Add(feat.Name);
                    }
                }
            }
        }

        private bool SaveMesh()
        {
            int modelType = ActiveSWModel.GetType();
            bool isPart = modelType == (int)swDocumentTypes_e.swDocPART;

            SelectionMgr selMgr = ActiveSWModel.SelectionManager;
            int selCount = selMgr.GetSelectedObjectCount2(-1);
            bool hasExportableSelection = false;

            if (selCount > 0)
            {
                if (isPart)
                {
                    hasExportableSelection = CommonSwOperations.SelectionHasFaceOrBody(ActiveSWModel);
                }
                else
                {
                    var components = new List<Component2>();
                    CommonSwOperations.GetSelectedComponents(ActiveSWModel, components);
                    hasExportableSelection = components.Count > 0;
                }
            }

            if (hasExportableSelection)
            {
                var result = MessageBox.Show(
                    isPart ?
                    "Do you want to export only the selected faces or bodies?" +
                    "\n\nChoosing No will export the entire visible part." :
                    "Do you want to export only the selected Parts or Subassemblies?" +
                    "\n(Individual faces or bodies are always not exported)" +
                    "\n\nChoosing No will export the entire visible assembly.",
                    "SOLIDWORKS",
                    MessageBoxButtons.YesNoCancel);

                switch (result)
                {
                    case DialogResult.Cancel:
                        return false;
                    case DialogResult.Yes:
                        break;
                    case DialogResult.No:
                        ActiveSWModel.ClearSelection2(true);
                        break;
                    default:
                        break;
                }
            }

            string coordSysName = comboBoxCoordinateSystem.Text;

            List<MeshFormat> exportFormats = new List<MeshFormat>();

            if (checkBoxStl.Checked)
            {
                exportFormats.Add(MeshFormat.stlOpenCascade);
            }
            if (checkBoxObj.Checked)
            {
                exportFormats.Add(MeshFormat.objOpenCascade);
            }
            if (checkBoxGlb.Checked)
            {
                exportFormats.Add(MeshFormat.glbOpenCascade);
            }

            float linearDeflection = (float)numericUpDownLinearDeflection.Value;
            float angularDeflection = (float)numericUpDownAngularDeflection.Value;
            float scale = (float)numericUpDownScale.Value;
            bool overwrite = checkBoxAlwaysOverwrite.Checked;

            // Every requested format is written from one tessellation, so the overwrite prompts all
            // happen up front rather than between exports.
            var outputs = new List<MeshExportOutput>();
            if (exportFormats.Contains(MeshFormat.stlOpenCascade)
                && (overwrite || ConfirmOverwrite(MeshFilename + ".stl")))
            {
                outputs.Add(new MeshExportOutput(MeshExportFormat.Stl, MeshFilename + ".stl"));
            }
            if (exportFormats.Contains(MeshFormat.objOpenCascade)
                && (overwrite || ConfirmOverwrite(MeshFilename + ".obj")))
            {
                outputs.Add(new MeshExportOutput(MeshExportFormat.Obj, MeshFilename + ".obj"));
            }
            if (exportFormats.Contains(MeshFormat.glbOpenCascade)
                && (overwrite || ConfirmOverwrite(MeshFilename + ".glb")))
            {
                outputs.Add(new MeshExportOutput(MeshExportFormat.Glb, MeshFilename + ".glb"));
            }

            string stepOutputPath = MeshFilename + ".stp";
            bool keepStep = checkBoxStep.Checked
                && (overwrite || ConfirmOverwrite(stepOutputPath));

            if (outputs.Count == 0 && !keepStep)
            {
                return false;
            }

            string stepPath = keepStep
                ? stepOutputPath
                : Path.Combine(Path.GetTempPath(), $"sw2urdf_{Guid.NewGuid()}.stp");
            SaveSTEP(SWApp, ActiveSWModel, coordSysName, stepPath);

            var options = new MeshExportOptions
            {
                LinearDeflection = linearDeflection,
                AngularDeflection = angularDeflection,
                Scale = MeshExportOptions.ScaleFromExporterUnits(scale),
                Backend = comboBoxBackend.SelectedIndex == 1
                    ? MeshingBackend.Delabella
                    : MeshingBackend.Isotropic,
                EdgeSampling = comboBoxEdgeSampling.SelectedIndex == 1
                    ? EdgeSampling.Uniform
                    : EdgeSampling.Adaptive,
                TargetEdgeLength = (double)numericUpDownTargetEdgeLength.Value,
                TargetEdgeLengthFraction = (double)numericUpDownEdgeLengthFraction.Value,
                AllowPartialFailure = true,
            };

            bool exported = true;
            if (outputs.Count > 0)
            {
                try
                {
                    IList<MeshExportStatus> statuses =
                        MeshExporter.Export(stepPath, outputs, options);

                    var problems = new List<string>();
                    for (int i = 0; i < statuses.Count; ++i)
                    {
                        if (statuses[i] == MeshExportStatus.Failed)
                        {
                            problems.Add($"{Path.GetFileName(outputs[i].Path)}: not written");
                        }
                        else if (statuses[i] == MeshExportStatus.WrittenPartial)
                        {
                            problems.Add(
                                $"{Path.GetFileName(outputs[i].Path)}: written, but some faces failed to mesh");
                        }
                    }

                    if (problems.Count > 0)
                    {
                        exported = false;
                        MessageBox.Show(
                            "Mesh export finished with problems:\n" + string.Join("\n", problems),
                            "Export mesh",
                            MessageBoxButtons.OK,
                            MessageBoxIcon.Warning);
                    }
                }
                catch (MeshCliException ex)
                {
                    exported = false;
                    MessageBox.Show(
                        "Mesh export failed:\n" + ex.Message,
                        "Export mesh",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Error);
                }
            }

            if (!keepStep)
            {
                try
                {
                    File.Delete(stepPath);
                }
                catch (Exception ex)
                {
                    MessageBox.Show("Couldn't delete temporary STEP file: \n" + ex.Message);
                }
            }

            return exported;
        }

        public static bool SaveSTEP(ISldWorks iSwApp, ModelDoc2 ActiveSWModel, string coordSysName, string windowsMeshFilename)
        {
            int errors = 0;
            int warnings = 0;

            ModelDoc2 ActiveDoc = ActiveSWModel;

            int saveOptions = (int)swSaveAsOptions_e.swSaveAsOptions_Silent |
                (int)swSaveAsOptions_e.swSaveAsOptions_Copy;
            iSwApp.SetUserPreferenceIntegerValue((int)swUserPreferenceIntegerValue_e.swStepAP, 214);
            iSwApp.SetUserPreferenceToggle((int)swUserPreferenceToggle_e.swStepExportAppearances, true);
            ActiveSWModel.Extension.SetUserPreferenceString((int)swUserPreferenceStringValue_e.swFileSaveAsCoordinateSystem,
                (int)swUserPreferenceOption_e.swDetailingNoOptionSpecified, coordSysName);

            ActiveDoc.Extension.SaveAs(windowsMeshFilename,
                (int)swSaveAsVersion_e.swSaveAsCurrentVersion, saveOptions, null, ref errors, ref warnings);

            // TODO: implement error/warning handling

            return true;
        }

        public static string ExtractFilePathWithoutExtension(string solidworksString)
        {
            if (string.IsNullOrWhiteSpace(solidworksString))
            {
                return string.Empty;
            }

            // Pattern: ^(.+\.\w+)\s
            // The .+ is GREEDY, so it captures everything including path separators
            // until it reaches .extension followed by space
            Match match = Regex.Match(solidworksString, @"^(.+\.\w+)\s");

            string fullPath;

            if (match.Success)
            {
                fullPath = match.Groups[1].Value;
            }
            else
            {
                fullPath = solidworksString.Trim();
            }

            string directory = Path.GetDirectoryName(fullPath);
            string fileNameWithoutExt = Path.GetFileNameWithoutExtension(fullPath);

            if (string.IsNullOrEmpty(directory))
            {
                return fileNameWithoutExt;
            }

            return Path.Combine(directory, fileNameWithoutExt);
        }

        public static bool ConfirmOverwrite(string filePath)
        {
            // Check if file exists
            if (File.Exists(filePath))
            {
                string fileName = Path.GetFileName(filePath);
                string message = $"{fileName} already exists.\nDo you want to replace it?";
                string caption = "Confirm Save As";

                DialogResult result = MessageBox.Show(
                    message,
                    caption,
                    MessageBoxButtons.YesNo,
                    MessageBoxIcon.Warning
                );

                return result == DialogResult.Yes;
            }

            // File doesn't exist, OK to proceed
            return true;
        }
    }
}
