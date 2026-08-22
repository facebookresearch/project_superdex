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
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Windows.Forms;

using Microsoft.Win32;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

using Meshing;

using CADRobotExporter.Export;
using CADRobotExporter.Model;
using CADRobotExporter.RobotDescription;
using CADRobotExporter.RobotExport;
using CADRobotExporter.CAD;
using CADRobotExporter.Utilities;

using SaveFileDialog = System.Windows.Forms.SaveFileDialog;

namespace CADRobotExporter.UI
{
    public partial class AssemblyExportForm : Form
    {
        public static bool FormIsOpen = false;
        public static AssemblyExportForm Instance { get; private set; }

        private static readonly Serilog.ILogger logger = Logger.GetLogger();

        public ExportHelper Exporter;
        public bool AutoUpdatingForm;

        private LinkNode previouslySelectedNode;
        private readonly Control[] jointBoxes;
        private readonly Control[] linkBoxes;
        private readonly LinkNode BaseNode;

        private ExporterConfiguration cachedExporterConfig;

        private int _currentTabIndex;
        private bool _navigating;
        private bool _isClosing;
        private bool _isUpdatingTree;

        // Solidworks
        private CADBridge CadBridge { get; set; }

        public AssemblyExportForm(LinkNode node, ExportHelper exporter)
        {
            Application.ThreadException +=
                new ThreadExceptionEventHandler(ExceptionHandler);
            AppDomain.CurrentDomain.UnhandledException +=
                new UnhandledExceptionEventHandler(UnhandledException);
            InitializeComponent();

            AutoUpdatingForm = false;

            BaseNode = node;
            Exporter = exporter;
            CadBridge = Exporter.CadBridge;

            jointBoxes = new Control[] {
                textBoxJointName, textBoxRefAxis, comboBoxJointType,
                textBoxAxisX, textBoxAxisY, textBoxAxisZ,
                textBoxJointX, textBoxJointY, textBoxJointZ,
                textBoxJointPitch, textBoxJointRoll, textBoxJointYaw,
                textBoxLimitEffort, textBoxLimitVelocity,
                textBoxDamping, textBoxFriction,
            };
            linkBoxes = new Control[] {
                textBoxInertialOriginX, textBoxInertialOriginY, textBoxInertialOriginZ,
                textBoxInertialOriginRoll, textBoxInertialOriginPitch, textBoxInertialOriginYaw,
                textBoxVisualOriginX, textBoxVisualOriginY, textBoxVisualOriginZ,
                textBoxVisualOriginRoll, textBoxVisualOriginPitch, textBoxVisualOriginYaw,
                textBoxIxx, textBoxIxy, textBoxIxz, textBoxIyy, textBoxIyz, textBoxIzz,
                textBoxMass,
                domainUpDownRed, domainUpDownGreen, domainUpDownBlue, domainUpDownAlpha,
            };

            List<TextBox> numericTextBoxes = new List<TextBox>() {
                textBoxAxisX, textBoxAxisY, textBoxAxisZ,
                textBoxJointX, textBoxJointY, textBoxJointZ,
                textBoxJointPitch, textBoxJointRoll, textBoxJointYaw,
                textBoxLimitEffort, textBoxLimitVelocity,
                textBoxDamping, textBoxFriction,
                textBoxInertialOriginX, textBoxInertialOriginY, textBoxInertialOriginZ,
                textBoxInertialOriginRoll, textBoxInertialOriginPitch, textBoxInertialOriginYaw,
                textBoxVisualOriginX, textBoxVisualOriginY, textBoxVisualOriginZ,
                textBoxVisualOriginRoll, textBoxVisualOriginPitch, textBoxVisualOriginYaw,
                textBoxIxx, textBoxIxy, textBoxIxz, textBoxIyy, textBoxIyz, textBoxIzz,
                textBoxMass,
            };

            foreach (TextBox textBox in numericTextBoxes)
            {
                textBox.KeyPress += NumericalTextBoxKeyPress;
            }

            comboBoxFolderStructure.Items.AddRange(Enum.GetNames(typeof(FolderStructure)));

            cachedExporterConfig = CadBridge.GetExporterConfiguration();
            LoadExporterConfigurationToForm(cachedExporterConfig);

            treeViewJointTree.AfterSelect += new TreeViewEventHandler(TreeViewJointtreeAfterSelect);
            treeViewLinkProperties.AfterSelect += new TreeViewEventHandler(TreeViewLinkPropertiesAfterSelect);
            buttonPrevious.Click += new EventHandler(ButtonPreviousClick);
            buttonNext.Click += new EventHandler(ButtonNextClick);
            buttonClose.Click += new EventHandler(ButtonCancelClick);
            buttonLinksFinish.Click += new EventHandler(ButtonLinksFinishClick);
            buttonLinksExportUrdfOnly.Click += new EventHandler(ButtonLinksExportUrdfOnlyClick);
            tabControl.Selected += TabControlSelected;
            FormClosing += OnCloseForm;
            FormClosed += OnFormClosed;
            checkBoxObjOpenCascadeVisual.CheckedChanged += new EventHandler(MeshExportCheckChanged);
            checkBoxStepVisual.CheckedChanged += new EventHandler(MeshExportCheckChanged);
            checkBoxStlCADVisual.CheckedChanged += new EventHandler(MeshExportCheckChanged);
            checkBoxStlOpenCascadeVisual.CheckedChanged += new EventHandler(MeshExportCheckChanged);
            checkBoxGlbOpenCascadeVisual.CheckedChanged += new EventHandler(MeshExportCheckChanged);
            checkBoxGlbCADVisual.CheckedChanged += new EventHandler(MeshExportCheckChanged);
            checkBoxObjCADVisual.CheckedChanged += new EventHandler(MeshExportCheckChanged);
            checkBoxObjOpenCascadeCollision.CheckedChanged += new EventHandler(MeshExportCollisionCheckChanged);
            checkBoxStepCollision.CheckedChanged += new EventHandler(MeshExportCollisionCheckChanged);
            checkBoxStlCADCollision.CheckedChanged += new EventHandler(MeshExportCollisionCheckChanged);
            checkBoxStlOpenCascadeCollision.CheckedChanged += new EventHandler(MeshExportCollisionCheckChanged);
            checkBoxGlbOpenCascadeCollision.CheckedChanged += new EventHandler(MeshExportCollisionCheckChanged);
            checkBoxGlbCADCollision.CheckedChanged += new EventHandler(MeshExportCollisionCheckChanged);
            checkBoxObjCADCollision.CheckedChanged += new EventHandler(MeshExportCollisionCheckChanged);
            buttonExportLinkMesh.Click += new EventHandler(ButtonExportLinkMeshClick);
            buttonRecalculateInertial.Click += new EventHandler(RecalculateInertialProperties);
            numericUpDownLimitUpper.ValueChanged += new EventHandler(UpdateJointLimits);
            numericUpDownLimitLower.ValueChanged += new EventHandler(UpdateJointLimits);
            checkBoxShowJointVisualization.CheckedChanged += new EventHandler(ShowJointVizCheckChanged);
            checkBoxShowLinkVisualization.CheckedChanged += new EventHandler(ShowLinkVizCheckChanged);
            checkBoxShowTendonVisualization.CheckedChanged += new EventHandler(ShowTendonVizCheckChanged);
            checkBoxShowAllTendons.CheckedChanged += new EventHandler(ShowAllTendonsCheckChanged);
            trackBarJointGizmoSize.ValueChanged += new EventHandler(UpdateJointGizmoSize);
            checkBoxShowJointHighlights.CheckedChanged += new EventHandler(ShowJointHighlightsCheckChanged);
            checkBoxLinkHighlights.CheckedChanged += new EventHandler(ShowLinkHighlightsCheckChanged);
            trackBarLinkGizmoSize.ValueChanged += new EventHandler(UpdateLinkGizmoSize);
            checkBoxCollisionMeshing.CheckedChanged += new EventHandler(CollisionCheckboxChanged);
            comboBoxBackend.SelectedIndexChanged += new EventHandler(MesherSettingChanged);
            comboBoxBackendCollision.SelectedIndexChanged += new EventHandler(MesherSettingChanged);
            numericUpDownTargetEdgeLength.ValueChanged += new EventHandler(MesherSettingChanged);
            numericUpDownTargetEdgeLengthCollision.ValueChanged += new EventHandler(MesherSettingChanged);
            buttonResetMeshingToDefaults.Click += new EventHandler(ResetMeshingToDefaults);
            checkBoxPerLinkMeshing.CheckedChanged += new EventHandler(PerLinkMeshingCheckboxChanged);
            checkBoxUseDegrees.CheckedChanged += new EventHandler(UseDegreesCheckboxChanged);
            checkBoxSummaryDegrees.CheckedChanged += (s, ev) => PopulateSummaryGrid();
            dataGridViewKinematicsSummary.CellDoubleClick += DataGridViewSummaryCellDoubleClick;
            dataGridViewTendonsSummary.CellClick += DataGridViewTendonsSummaryCellClick;

            linkLabelHelpMeChoose.LinkClicked += LinkLabelHelpMeChoose_LinkClicked;

            SetCADSpecificCheckboxProperties();

            PopulatePresetComboBox();
            comboBoxExporterConfigurationPreset.SelectedIndexChanged += ComboBoxExporterConfigurationPreset_SelectedIndexChanged;

            _currentTabIndex = 0;
            _navigating = false;
            _isUpdatingTree = false;

            FormIsOpen = true;
            Instance = this;

            UseDegreesCheckboxChanged(null, null);

            this.Text = "SuperDex CAD Exporter " + Versioning.VersionString.Get() + " - Export";
        }

        private void LinkLabelHelpMeChoose_LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
        {
            Process.Start(new ProcessStartInfo
            {
                FileName = "https://facebookresearch.github.io/project_superdex/studio/docs/cad_exporter/#mesh-export-guide",
                UseShellExecute = true
            });
        }

        private void SetCADSpecificCheckboxProperties()
        {
            const string tag = "<cad>";
            string cad = "";
#if NX
            cad = "NX";
#endif
#if SOLIDWORKS
            cad = "Solidworks";
            checkBoxObjCADVisual.Enabled = false;
            checkBoxGlbCADVisual.Enabled = false;
            checkBoxObjCADCollision.Enabled = false;
            checkBoxGlbCADCollision.Enabled = false;
            labelObjCAD.Enabled = false;
            labelGlbCAD.Enabled = false;
#endif
            labelStlCAD.Text = labelStlCAD.Text.Replace(tag, cad);
            labelObjCAD.Text = labelObjCAD.Text.Replace(tag, cad);
            labelStep.Text = labelStep.Text.Replace(tag, cad);
            labelGlbCAD.Text = labelGlbCAD.Text.Replace(tag, cad);

#if NX
            labelObjCAD.Text = labelObjCAD.Text + " ⚠️";
            toolTips.SetToolTip(checkBoxObjCADVisual,
                "NX OBJ export does not produce smooth normals for visual models. " +
                "Use GLB or SuperDex options. MuJoCo also does not support multi-body OBJs, the meshes may need post-processing.");
            toolTips.SetToolTip(checkBoxObjCADCollision,
                "NX OBJ export does not produce smooth normals. " +
                "Use GLB or SuperDex options.");

            //labelJointVisualization.Text = "Hint: Enable \"Show Through Curves\" if joint limit visualization is occluded.";
#endif
        }

        private void SantizeCADSpecificCheckboxes()
        {
#if SOLIDWORKS
            checkBoxObjCADVisual.Checked = false;
            checkBoxGlbCADVisual.Checked = false;
            checkBoxObjCADCollision.Checked = false;
            checkBoxGlbCADCollision.Checked = false;
#endif
        }

        private void PerLinkMeshingCheckboxChanged(object sender, EventArgs e)
        {
            CheckBox checkBox = (CheckBox)sender;
            bool isChecked = checkBox.Checked;

            if (isChecked)
            {
                cachedExporterConfig = SaveFormToExporterConfiguration();

                LinkNode currentNode = (LinkNode)treeViewLinkProperties.SelectedNode;
                if (currentNode != null)
                {
                    LoadMeshingOptionsFromLink(currentNode.Link);
                }
            }
            else
            {
                if (previouslySelectedNode != null)
                {
                    SaveMeshingOptionsToLink(previouslySelectedNode.Link);
                }

                LoadMeshingOptionsToForm(cachedExporterConfig);
            }
        }

        private void UseDegreesCheckboxChanged(object sender, EventArgs e)
        {
            AutoUpdatingForm = true;

            string jointType = comboBoxJointType.Text;
            bool isRevolute = jointType == "revolute" || jointType == "continuous";

            // Convert currently displayed values when checkbox changes
            if (checkBoxUseDegrees.Checked)
            {
                // Converting from radians to degrees
                ConvertTextBoxValue(textBoxJointRoll, RadiansToDegrees);
                ConvertTextBoxValue(textBoxJointPitch, RadiansToDegrees);
                ConvertTextBoxValue(textBoxJointYaw, RadiansToDegrees);
                if (isRevolute)
                {
                    ConvertTextBoxValue(textBoxLimitVelocity, RadiansToDegrees);
                    ConvertNumericUpDownValue(numericUpDownLimitLower, RadiansToDegrees);
                    ConvertNumericUpDownValue(numericUpDownLimitUpper, RadiansToDegrees);
                }

                // Update increment for degrees (5 degrees)
                numericUpDownLimitLower.Increment = 5M;
                numericUpDownLimitUpper.Increment = 5M;
            }
            else
            {
                // Converting from degrees to radians
                ConvertTextBoxValue(textBoxJointRoll, DegreesToRadians);
                ConvertTextBoxValue(textBoxJointPitch, DegreesToRadians);
                ConvertTextBoxValue(textBoxJointYaw, DegreesToRadians);
                if (isRevolute)
                {
                    ConvertTextBoxValue(textBoxLimitVelocity, DegreesToRadians);
                    ConvertNumericUpDownValue(numericUpDownLimitLower, DegreesToRadians);
                    ConvertNumericUpDownValue(numericUpDownLimitUpper, DegreesToRadians);
                }

                // Update increment for radians (~5 degrees in radians)
                numericUpDownLimitLower.Increment = 0.0872665M;
                numericUpDownLimitUpper.Increment = 0.0872665M;
            }

            // Update labels to reflect current unit
            UpdateJointLabelsForUnit();

            AutoUpdatingForm = false;
        }

        private static double RadiansToDegrees(double radians) => radians * (180.0 / Math.PI);
        private static double DegreesToRadians(double degrees) => degrees * (Math.PI / 180.0);

        private void ConvertNumericUpDownValue(NumericUpDown control, Func<double, double> converter)
        {
            try
            {
                double currentValue = (double)control.Value;
                double newValue = converter(currentValue);
                // Clamp to control's min/max
                decimal clampedValue = Math.Max(control.Minimum, Math.Min(control.Maximum, Convert.ToDecimal(newValue)));
                control.Value = clampedValue;
            }
            catch { }
        }

        private void ConvertTextBoxValue(TextBox control, Func<double, double> converter)
        {
            if (double.TryParse(control.Text, out double currentValue))
            {
                double newValue = converter(currentValue);
                control.Text = newValue.ToString("R");
            }
        }

        private void UpdateJointLabelsForUnit()
        {
            string angleUnit = checkBoxUseDegrees.Checked ? "deg" : "rad";
            string jointType = comboBoxJointType.Text;

            if (jointType == "revolute" || jointType == "continuous")
            {
                labelLowerLimit.Text = $"Lower ({angleUnit})";
                labelLimitUpper.Text = $"Upper ({angleUnit})";
                labelVelocity.Text = $"Velocity ({angleUnit}/s)";
            }

            groupBoxJointOrigin.Text = $"Origin* (m, {angleUnit})";
        }

        private void ResetMeshingToDefaults(object sender, EventArgs e)
        {
            LoadMeshingOptionsToForm(new ExporterConfiguration());
        }

        private void CollisionCheckboxChanged(object sender, EventArgs e)
        {
            CheckBox checkBox = (CheckBox)sender;
            bool isChecked = checkBox.Checked;

            numericUpDownLinearDeflectionCollision.Enabled = isChecked;
            numericUpDownAngularDeflectionCollision.Enabled = isChecked;
            numericUpDownScaleCollision.Enabled = isChecked;

            UpdateMeshingControlEnabledState();
        }

        private void ShowJointVizCheckChanged(object sender, EventArgs e)
        {
            CheckBox checkBox = (CheckBox)sender;
            bool isChecked = checkBox.Checked;

            CadBridge.IsShowingJointGizmo = isChecked;
            CadBridge.TriggerGraphicsRedraw();
        }

        private void ShowLinkVizCheckChanged(object sender, EventArgs e)
        {
            CheckBox checkBox = (CheckBox)sender;
            bool isChecked = checkBox.Checked;

            CadBridge.IsShowingInertialGizmo = isChecked;
            CadBridge.TriggerGraphicsRedraw();
        }

        private void ShowTendonVizCheckChanged(object sender, EventArgs e)
        {
            CheckBox checkBox = (CheckBox)sender;
            bool isChecked = checkBox.Checked;

            CadBridge.IsShowingTendonVisualization = isChecked;
            CadBridge.TriggerGraphicsRedraw();
        }

        private void ShowAllTendonsCheckChanged(object sender, EventArgs e)
        {
            CheckBox checkBox = (CheckBox)sender;
            CadBridge.IsShowingAllTendons = checkBox.Checked;
            if (checkBox.Checked)
            {
                CadBridge.AllTendons = Exporter.Robot.Tendons;
                if (CadBridge.TendonLinkCsysMap == null)
                {
                    var map = new Dictionary<string, string>();
                    CollectLinkCsysNames(BaseNode.Link, map);
                    CadBridge.TendonLinkCsysMap = map;
                }
            }
            CadBridge.ShowHideVisualizations(checkBox.Checked || checkBoxShowTendonVisualization.Checked);
            CadBridge.TriggerGraphicsRedraw();
        }

        private void UpdateJointGizmoSize(object sender, EventArgs e)
        {
            TrackBar trackBar = (TrackBar)sender;
            double normalized = (double)trackBar.Value / trackBar.Maximum;
            CadBridge.JointGizmoScale = Math.Pow(8.0, (normalized - 0.5) / 0.5);
            CadBridge.TriggerGraphicsRedraw();
        }

        private void UpdateLinkGizmoSize(object sender, EventArgs e)
        {
            TrackBar trackBar = (TrackBar)sender;
            double normalized = (double)trackBar.Value / trackBar.Maximum;
            CadBridge.LinkGizmoScale = Math.Pow(8.0, (normalized - 0.5) / 0.5);
            CadBridge.TriggerGraphicsRedraw();
        }

        private void ShowJointHighlightsCheckChanged(object sender, EventArgs e)
        {
            CheckBox checkBox = (CheckBox)sender;
            if (!checkBox.Checked)
            {
                CadBridge.UnselectAll();
            }
            else
            {
                LinkNode node = (LinkNode)treeViewJointTree.SelectedNode;
                if (node != null)
                {
                    CadBridge.SelectJointComponents(node);
                }
            }
        }

        private void ShowLinkHighlightsCheckChanged(object sender, EventArgs e)
        {
            CheckBox checkBox = (CheckBox)sender;
            if (!checkBox.Checked)
            {
                CadBridge.UnselectAll();
            }
            else
            {
                LinkNode node = (LinkNode)treeViewLinkProperties.SelectedNode;
                if (node != null)
                {
                    CadBridge.SelectLinkComponents(node);
                }
            }
        }

        private void UpdateJointLimits(object sender, EventArgs e)
        {
            if (AutoUpdatingForm)
            {
                return;
            }

            LinkNode node = (LinkNode)treeViewJointTree.SelectedNode;
            if (node != null && node.Link.Joint != null)
            {
                bool useDegrees = checkBoxUseDegrees.Checked;
                double upper = Convert.ToDouble(numericUpDownLimitUpper.Value);
                double lower = Convert.ToDouble(numericUpDownLimitLower.Value);

                if (useDegrees && node.Link.Joint.Type == "revolute")
                {
                    upper = DegreesToRadians(upper);
                    lower = DegreesToRadians(lower);
                }

                node.Link.Joint.Limit.Upper = upper;
                node.Link.Joint.Limit.Lower = lower;
            }

            CadBridge.TriggerGraphicsRedraw();
        }

        private void RecalculateInertialProperties(object sender, EventArgs e)
        {
            LinkNode node = (LinkNode)treeViewLinkProperties.SelectedNode;
            if (node != null && node.Link != null)
            {
                Exporter.ComputeInertialProperties(node.Link);
                FillLinkPropertyBoxes(node.Link);
            }
        }

        private void MeshExportCheckChanged(object sender, EventArgs e)
        {
            CheckBox checkBox = (CheckBox)sender;
            bool isChecked = checkBox.Checked;

            if (!isChecked)
            {
                if (checkBoxGlbOpenCascadeVisual.Checked || checkBoxGlbCADVisual.Checked)
                {
                    textBoxMeshFileExtension.Text = "glb";
                    return;
                }
                if (checkBoxObjOpenCascadeVisual.Checked || checkBoxObjCADVisual.Checked)
                {
                    textBoxMeshFileExtension.Text = "obj";
                    return;
                }
                if (checkBoxStlCADVisual.Checked || checkBoxStlOpenCascadeVisual.Checked)
                {
                    textBoxMeshFileExtension.Text = "stl";
                    return;
                }
                if (checkBoxStepVisual.Checked)
                {
                    textBoxMeshFileExtension.Text = "stp";
                    return;
                }
                return;
            }

            if (checkBox == checkBoxStepVisual)
            {
                textBoxMeshFileExtension.Text = "stp";
            }
            if (checkBox == checkBoxObjOpenCascadeVisual || checkBox == checkBoxObjCADVisual)
            {
                textBoxMeshFileExtension.Text = "obj";

                if (checkBox == checkBoxObjOpenCascadeVisual && checkBoxObjCADVisual.Checked)
                {
                    checkBoxObjCADVisual.Checked = false;
                }
                if (checkBox == checkBoxObjCADVisual && checkBoxObjOpenCascadeVisual.Checked)
                {
                    checkBoxObjOpenCascadeVisual.Checked = false;
                }
            }
            if (checkBox == checkBoxStlCADVisual || checkBox == checkBoxStlOpenCascadeVisual)
            {
                textBoxMeshFileExtension.Text = "stl";

                if (checkBox == checkBoxStlOpenCascadeVisual && checkBoxStlCADVisual.Checked)
                {
                    checkBoxStlCADVisual.Checked = false;
                }
                if (checkBox == checkBoxStlCADVisual && checkBoxStlOpenCascadeVisual.Checked)
                {
                    checkBoxStlOpenCascadeVisual.Checked = false;
                }
            }
            if (checkBox == checkBoxGlbOpenCascadeVisual || checkBox == checkBoxGlbCADVisual)
            {
                textBoxMeshFileExtension.Text = "glb";

                if (checkBox == checkBoxGlbOpenCascadeVisual && checkBoxGlbCADVisual.Checked)
                {
                    checkBoxGlbCADVisual.Checked = false;
                }
                if (checkBox == checkBoxGlbCADVisual && checkBoxGlbOpenCascadeVisual.Checked)
                {
                    checkBoxGlbOpenCascadeVisual.Checked = false;
                }
            }
        }

        private void MeshExportCollisionCheckChanged(object sender, EventArgs e)
        {
            CheckBox checkBox = (CheckBox)sender;
            bool isChecked = checkBox.Checked;

            if (!isChecked)
            {
                if (checkBoxGlbOpenCascadeCollision.Checked || checkBoxGlbCADCollision.Checked)
                {
                    textBoxMeshFileExtensionCollision.Text = "glb";
                    return;
                }
                if (checkBoxObjOpenCascadeCollision.Checked || checkBoxObjCADCollision.Checked)
                {
                    textBoxMeshFileExtensionCollision.Text = "obj";
                    return;
                }
                if (checkBoxStlCADCollision.Checked || checkBoxStlOpenCascadeCollision.Checked)
                {
                    textBoxMeshFileExtensionCollision.Text = "stl";
                    return;
                }
                if (checkBoxStepCollision.Checked)
                {
                    textBoxMeshFileExtensionCollision.Text = "stp";
                    return;
                }
                return;
            }

            if (checkBox == checkBoxStepCollision)
            {
                textBoxMeshFileExtensionCollision.Text = "stp";
            }
            if (checkBox == checkBoxObjOpenCascadeCollision || checkBox == checkBoxObjCADCollision)
            {
                textBoxMeshFileExtensionCollision.Text = "obj";

                if (checkBox == checkBoxObjOpenCascadeCollision && checkBoxObjCADCollision.Checked)
                {
                    checkBoxObjCADCollision.Checked = false;
                }
                if (checkBox == checkBoxObjCADCollision && checkBoxObjOpenCascadeCollision.Checked)
                {
                    checkBoxObjOpenCascadeCollision.Checked = false;
                }
            }
            if (checkBox == checkBoxStlCADCollision || checkBox == checkBoxStlOpenCascadeCollision)
            {
                textBoxMeshFileExtensionCollision.Text = "stl";

                if (checkBox == checkBoxStlOpenCascadeCollision && checkBoxStlCADCollision.Checked)
                {
                    checkBoxStlCADCollision.Checked = false;
                }
                if (checkBox == checkBoxStlCADCollision && checkBoxStlOpenCascadeCollision.Checked)
                {
                    checkBoxStlOpenCascadeCollision.Checked = false;
                }
            }
            if (checkBox == checkBoxGlbOpenCascadeCollision || checkBox == checkBoxGlbCADCollision)
            {
                textBoxMeshFileExtensionCollision.Text = "glb";

                if (checkBox == checkBoxGlbOpenCascadeCollision && checkBoxGlbCADCollision.Checked)
                {
                    checkBoxGlbCADCollision.Checked = false;
                }
                if (checkBox == checkBoxGlbCADCollision && checkBoxGlbOpenCascadeCollision.Checked)
                {
                    checkBoxGlbOpenCascadeCollision.Checked = false;
                }
            }
        }

        private void TabControlSelected(object sender, TabControlEventArgs e)
        {
            if (_navigating)
                return;
            NavigateToTab(e.TabPageIndex);
        }

        private void NavigateToTab(int targetTab)
        {
            if (targetTab == _currentTabIndex)
                return;

            if (!OnExitTab(_currentTabIndex))
                return;

            _currentTabIndex = targetTab;
            OnEnterTab(_currentTabIndex);

            _navigating = true;
            tabControl.SelectTab(_currentTabIndex);
            _navigating = false;

            Focus();
            CadBridge.TriggerGraphicsRedraw();
        }

        private bool OnExitTab(int tabIndex)
        {
            CadBridge.UnselectAll();

            switch (tabIndex)
            {
                case 0: return OnExitJointTab();
                case 1: OnExitLinkTab(); return true;
                case 2: return true;
                case 3:
                    CadBridge.IsShowingTendonVisualization = false;
                    CadBridge.IsShowingAllTendons = false;
                    CadBridge.CurrentTendonShown = null;
                    CadBridge.ShowHideVisualizations(false);
                    CadBridge.TriggerGraphicsRedraw();
                    return true;
                default: return true;
            }
        }

        private void OnEnterTab(int tabIndex)
        {
            switch (tabIndex)
            {
                case 0: OnEnterJointTab(); break;
                case 1: OnEnterLinkTab(); break;
                case 2: OnEnterSummaryTab(); break;
                case 3: OnEnterTendonsSummaryTab(); break;
            }
        }

        private bool OnExitJointTab()
        {
            if (!(previouslySelectedNode == null || previouslySelectedNode.Link.Joint == null))
            {
                SaveJointDataFromPropertyBoxes(previouslySelectedNode.Link.Joint);
            }
            previouslySelectedNode = null;

            string errors = CheckJointsForErrors();
            if (!string.IsNullOrWhiteSpace(errors))
            {
                string message = "The following joints are missing required fields, please " +
                    "address them before continuing\r\n\r\n" + errors;
                MessageBox.Show(message, "Robot Configuration Exporter - Joint Errors");
                return false;
            }

            CadBridge.IsShowingJointGizmo = false;
            CadBridge.ShowHideVisualizations(false);
            CadBridge.TriggerGraphicsRedraw();

            _isUpdatingTree = true;
            while (treeViewJointTree.Nodes.Count > 0)
            {
                LinkNode node = (LinkNode)treeViewJointTree.Nodes[0];
                treeViewJointTree.Nodes.Remove(node);
                BaseNode.Nodes.Add(node);
            }
            _isUpdatingTree = false;

            return true;
        }

        private void OnEnterJointTab()
        {
            treeViewLinkProperties.Nodes.Clear();
            FillJointTree();
            buttonPrevious.Enabled = false;
            buttonNext.Enabled = true;
            buttonLinksExportUrdfOnly.Enabled = false;
            buttonLinksFinish.Enabled = false;

            CadBridge.ShowHideVisualizations(checkBoxShowJointVisualization.Checked);
        }

        private void OnExitLinkTab()
        {
            LinkNode node = (LinkNode)treeViewLinkProperties.SelectedNode;
            if (node != null)
            {
                SaveLinkDataFromPropertyBoxes(node.Link);
            }
            previouslySelectedNode = null;

            CadBridge.IsShowingInertialGizmo = false;
            CadBridge.CurrentLinkNodeShown = null;
            CadBridge.ShowHideVisualizations(false);
            CadBridge.TriggerGraphicsRedraw();
        }

        private void OnEnterLinkTab()
        {
            FillLinkTree();
            buttonPrevious.Enabled = true;
            buttonNext.Enabled = true;
            buttonLinksExportUrdfOnly.Enabled = true;
            buttonLinksFinish.Enabled = true;

            CadBridge.IsShowingInertialGizmo = checkBoxShowLinkVisualization.Checked;
            CadBridge.ShowHideVisualizations(true);
        }

        private void OnEnterSummaryTab()
        {
            buttonPrevious.Enabled = true;
            buttonNext.Enabled = true;
            PopulateSummaryGrid();
        }

        private void OnEnterTendonsSummaryTab()
        {
            buttonPrevious.Enabled = true;
            buttonNext.Enabled = false;

            // Build link CSYS map for tendon visualization
            var map = new Dictionary<string, string>();
            CollectLinkCsysNames(BaseNode.Link, map);
            CadBridge.TendonLinkCsysMap = map;
            CadBridge.IsShowingTendonVisualization = false;
            CadBridge.CurrentTendonShown = null;

            CadBridge.ShowHideVisualizations(true);

            PopulateTendonsSummaryGrid();
        }

        private void ButtonNextClick(object sender, EventArgs e)
        {
            if (_currentTabIndex < 3)
                NavigateToTab(_currentTabIndex + 1);
        }

        private void ButtonPreviousClick(object sender, EventArgs e)
        {
            if (_currentTabIndex > 0)
                NavigateToTab(_currentTabIndex - 1);
        }

        private void PopulateSummaryGrid()
        {
            dataGridViewKinematicsSummary.ReadOnly = true;
            dataGridViewKinematicsSummary.AllowUserToAddRows = false;
            dataGridViewKinematicsSummary.AllowUserToDeleteRows = false;

            dataGridViewKinematicsSummary.Columns.Clear();
            dataGridViewKinematicsSummary.Rows.Clear();

            dataGridViewKinematicsSummary.Columns.Add("Index", "Idx");
            dataGridViewKinematicsSummary.Columns.Add("LinkName", "Link Name");
            dataGridViewKinematicsSummary.Columns.Add("JointName", "Joint Name");
            dataGridViewKinematicsSummary.Columns.Add("JointType", "Joint Type");
            dataGridViewKinematicsSummary.Columns.Add("Axis", "Axis");
            string limitUnit = checkBoxSummaryDegrees.Checked ? "deg" : "rad";
            dataGridViewKinematicsSummary.Columns.Add("LowerLimit", $"Lower ({limitUnit})");
            dataGridViewKinematicsSummary.Columns.Add("UpperLimit", $"Upper ({limitUnit})");
            dataGridViewKinematicsSummary.Columns.Add("EffortLimit", "Effort (N-m, N)");
            dataGridViewKinematicsSummary.Columns.Add("VelocityLimit", $"Velocity ({limitUnit}/s, m/s)");
            dataGridViewKinematicsSummary.Columns.Add("Friction", "Friction");
            dataGridViewKinematicsSummary.Columns.Add("Damping", "Damping");
            dataGridViewKinematicsSummary.Columns.Add("Mass", "Mass (kg)");
            dataGridViewKinematicsSummary.Columns.Add("Inertia", "Inertia (diag)");
            dataGridViewKinematicsSummary.Columns.Add("InertialBodies", "# Ine");
            dataGridViewKinematicsSummary.Columns.Add("CollisionBodies", "# Col");
            dataGridViewKinematicsSummary.Columns.Add("VisualBodies", "# Vis");

            int index = 0;
            AddLinkToSummaryGrid(BaseNode.Link, true, 0, ref index);

            for (int i = 0; i < dataGridViewKinematicsSummary.Columns.Count; i++)
            {
                dataGridViewKinematicsSummary.Columns[i].AutoSizeMode = DataGridViewAutoSizeColumnMode.AllCells;
            }
        }

        private void AddLinkToSummaryGrid(Link link, bool isBase, int depth, ref int index)
        {
            string indent = new string(' ', depth * 2);
            string linkName = indent + link.Name;

            string jointName = "";
            string jointType = "";
            string axis = "";
            string lowerLimit = "";
            string upperLimit = "";
            string effortLimit = "";
            string velocityLimit = "";
            string friction = "";
            string damping = "";

            if (!isBase && link.Joint != null && !string.IsNullOrWhiteSpace(link.Joint.Name))
            {
                jointName = link.Joint.Name;
                jointType = link.Joint.Type ?? "";

                if (link.isSite)
                {
                    jointType = "(site)";
                }

                bool showDegrees = checkBoxSummaryDegrees.Checked && jointType == "revolute";

                if (link.Joint.Limit.IsLowerSet())
                {
                    double lower = link.Joint.Limit.Lower;
                    lowerLimit = (showDegrees ? RadiansToDegrees(lower) : lower).ToString("G4");
                }
                if (link.Joint.Limit.IsUpperSet())
                {
                    double upper = link.Joint.Limit.Upper;
                    upperLimit = (showDegrees ? RadiansToDegrees(upper) : upper).ToString("G4");
                }
                if (link.Joint.Limit.IsEffortSet())
                {
                    double effort = link.Joint.Limit.Effort;
                    effortLimit = effort.ToString("G4");
                }
                if (link.Joint.Limit.IsVelocitySet())
                {
                    double velocity = link.Joint.Limit.Velocity;
                    if (jointType == "revolute")
                    {
                        velocityLimit = (showDegrees ? RadiansToDegrees(velocity) : velocity).ToString("G4");
                    }
                }
                if (link.Joint.Dynamics.IsFrictionSet())
                {
                    friction = link.Joint.Dynamics.Friction.ToString("G4");
                }
                if (link.Joint.Dynamics.IsDampingSet())
                {
                    damping = link.Joint.Dynamics.Damping.ToString("G4");
                }

                if (jointType != "fixed" && !string.IsNullOrWhiteSpace(jointType))
                {
                    double[] xyz = link.Joint.Axis.GetXYZ();
                    axis = $"{xyz[0]:G3}, {xyz[1]:G3}, {xyz[2]:G3}";
                }
            }

            string mass = link.Inertial.Mass.Value.ToString("G4");
            string inertia = $"{link.Inertial.Inertia.Ixx:G4}, {link.Inertial.Inertia.Iyy:G4}, {link.Inertial.Inertia.Izz:G4}";

            int inertialCount = 0;
            int collisionCount = 0;
            int visualCount = 0;

#if SOLIDWORKS
            if (link.SWInertialComponents != null) inertialCount = link.SWInertialComponents.Count;
            if (link.SWCollisionComponents != null) collisionCount = link.SWCollisionComponents.Count;
            if (link.SWVisualComponents != null) visualCount = link.SWVisualComponents.Count;
#endif
#if NX
            if (link.NXInertialBodiesHandles != null) inertialCount = link.NXInertialBodiesHandles.Count;
            if (link.NXCollisionBodiesHandles != null) collisionCount = link.NXCollisionBodiesHandles.Count;
            if (link.NXVisualBodiesHandles != null) visualCount = link.NXVisualBodiesHandles.Count;
#endif

            dataGridViewKinematicsSummary.Rows.Add(
                index,
                linkName, jointName, jointType,
                axis,
                lowerLimit, upperLimit,
                effortLimit, velocityLimit,
                friction, damping,
                mass, inertia,
                inertialCount.ToString(), collisionCount.ToString(), visualCount.ToString());

            foreach (Link child in link.Children)
            {
                index += 1;
                AddLinkToSummaryGrid(child, false, depth + 1, ref index);
            }
        }

        private void PopulateTendonsSummaryGrid()
        {
            dataGridViewTendonsSummary.ReadOnly = true;
            dataGridViewTendonsSummary.AllowUserToAddRows = false;
            dataGridViewTendonsSummary.AllowUserToDeleteRows = false;
            dataGridViewTendonsSummary.SelectionMode = DataGridViewSelectionMode.FullRowSelect;

            dataGridViewTendonsSummary.Columns.Clear();
            dataGridViewTendonsSummary.Rows.Clear();

            dataGridViewTendonsSummary.Columns.Add("Index", "Idx");
            dataGridViewTendonsSummary.Columns.Add("TendonName", "Tendon");
            dataGridViewTendonsSummary.Columns.Add("Elem", "Elem");
            dataGridViewTendonsSummary.Columns.Add("Link", "Link");
            dataGridViewTendonsSummary.Columns.Add("Type", "Type");
            dataGridViewTendonsSummary.Columns.Add("X", "X");
            dataGridViewTendonsSummary.Columns.Add("Y", "Y");
            dataGridViewTendonsSummary.Columns.Add("Z", "Z");
            dataGridViewTendonsSummary.Columns.Add("Coefficient", "Coefficient");

            if (Exporter.Robot?.Tendons == null)
                return;

            int index = 0;
            foreach (var tendon in Exporter.Robot.Tendons)
            {
                for (int i = 0; i < tendon.RoutingElements.Count; i++)
                {
                    var elem = tendon.RoutingElements[i];
                    string tendonName = (i == 0) ? tendon.Name : "";
                    string coef = elem.Coefficient != 0 ? elem.Coefficient.ToString("G4") : "";
                    bool isLinearJoint = elem.Type == RoutingElement.TypeLinearJoint;

                    dataGridViewTendonsSummary.Rows.Add(
                        index,
                        tendonName,
                        i,
                        elem.Link,
                        elem.Type,
                        isLinearJoint ? "-" : elem.X.ToString("G5"),
                        isLinearJoint ? "-" : elem.Y.ToString("G5"),
                        isLinearJoint ? "-" : elem.Z.ToString("G5"),
                        coef);

                    index++;
                }
            }

            for (int i = 0; i < dataGridViewTendonsSummary.Columns.Count; i++)
            {
                dataGridViewTendonsSummary.Columns[i].AutoSizeMode = DataGridViewAutoSizeColumnMode.AllCells;
            }
        }

        private void DataGridViewTendonsSummaryCellClick(object sender, DataGridViewCellEventArgs e)
        {
            if (e.RowIndex < 0)
                return;

            // Find the tendon name for this row (walk backwards to find the group header)
            string tendonName = null;
            for (int row = e.RowIndex; row >= 0; row--)
            {
                var val = dataGridViewTendonsSummary.Rows[row].Cells["TendonName"].Value?.ToString();
                if (!string.IsNullOrEmpty(val))
                {
                    tendonName = val;
                    break;
                }
            }

            if (string.IsNullOrEmpty(tendonName) || Exporter.Robot?.Tendons == null)
                return;

            Tendon tendon = null;
            foreach (var t in Exporter.Robot.Tendons)
            {
                if (t.Name == tendonName)
                {
                    tendon = t;
                    break;
                }
            }

            if (tendon == null)
                return;

            int elemIndex = -1;
            var elemVal = dataGridViewTendonsSummary.Rows[e.RowIndex].Cells["Elem"].Value;
            if (elemVal != null && int.TryParse(elemVal.ToString(), out int parsed))
            {
                elemIndex = parsed;
            }

            CadBridge.CurrentTendonShown = tendon;
            CadBridge.TendonHighlightElementIndex = elemIndex;
            CadBridge.IsShowingTendonVisualization = checkBoxShowTendonVisualization.Checked;
            CadBridge.TriggerGraphicsRedraw();
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

        private void DataGridViewSummaryCellDoubleClick(object sender, DataGridViewCellEventArgs e)
        {
            if (e.RowIndex < 0)
            {
                return;
            }

            string linkName = dataGridViewKinematicsSummary.Rows[e.RowIndex].Cells["LinkName"].Value?.ToString()?.Trim();
            if (string.IsNullOrEmpty(linkName))
            {
                return;
            }

            string columnName = dataGridViewKinematicsSummary.Columns[e.ColumnIndex].Name;

            bool isJointColumn = columnName == "JointName" || columnName == "JointType"
                || columnName == "Axis" || columnName == "LowerLimit" || columnName == "UpperLimit"
                || columnName == "EffortLimit" || columnName == "VelocityLimit"
                || columnName == "Friction" || columnName == "Damping";

            if (isJointColumn)
            {
                // Base link has no joint - skip
                string jointName = dataGridViewKinematicsSummary.Rows[e.RowIndex].Cells["JointName"].Value?.ToString();
                if (string.IsNullOrWhiteSpace(jointName))
                {
                    return;
                }

                NavigateToTab(0);
                LinkNode node = FindNodeByLinkName(treeViewJointTree.Nodes, linkName);
                if (node != null)
                {
                    treeViewJointTree.SelectedNode = node;
                }
            }
            else
            {
                NavigateToTab(1);
                LinkNode node = FindNodeByLinkName(treeViewLinkProperties.Nodes, linkName);
                if (node != null)
                {
                    treeViewLinkProperties.SelectedNode = node;
                }
            }
        }

        private static LinkNode FindNodeByLinkName(TreeNodeCollection nodes, string linkName)
        {
            foreach (TreeNode treeNode in nodes)
            {
                LinkNode node = (LinkNode)treeNode;
                if (node.Link.Name == linkName)
                {
                    return node;
                }
                LinkNode found = FindNodeByLinkName(node.Nodes, linkName);
                if (found != null)
                {
                    return found;
                }
            }
            return null;
        }

        private void ExceptionHandler(object sender, ThreadExceptionEventArgs e)
        {
            logger.Error("Exception encountered in Assembly export form", e.Exception);
            MessageBox.Show("There was a problem with the export form: \n\"" +
                e.Exception.Message + "\"\nContact your maintainer with the log file found at " +
                Logger.GetLogFolder());
        }

        private void UnhandledException(object sender, UnhandledExceptionEventArgs e)
        {
            Exception ex = (Exception)e.ExceptionObject;
            logger.Error("Unhandled exception in Assembly Export form", ex);
            MessageBox.Show("There was a problem with the export form: \n\"" +
                ex.Message + "\"\nContact your maintainer with the log file found at " +
                Logger.GetLogFolder());
        }

        //Joint form configuration controls
        private void AssemblyExportFormLoad(object sender, EventArgs e)
        {
            // Exporter.UpdateReferenceGeometries();
            FillJointTree();
            CadBridge.ShowHideVisualizations(checkBoxShowJointVisualization.Checked);
        }

        private string CheckJointsForErrors()
        {
            StringBuilder builder = new StringBuilder();
            foreach (LinkNode child in treeViewJointTree.Nodes)
            {
                CheckJointsForErrors(child, builder);
            }
            return builder.ToString();
        }

        private StringBuilder CheckJointsForErrors(LinkNode node, StringBuilder builder)
        {
            if (!node.Link.Joint.AreRequiredFieldsSatisfied())
            {
                builder.Append(node.Link.Joint.Name).Append("\r\n");
            }

            foreach (LinkNode child in node.Nodes)
            {
                CheckJointsForErrors(child, builder);
            }
            return builder;
        }

        private void ButtonCancelClick(object sender, EventArgs e)
        {
            if (!_isClosing)
            {
                switch (_currentTabIndex)
                {
                    case 0:
                        ButtonJointCancelClick(sender, e);
                        break;
                    case 1:
                    case 2:
                    case 3:
                        ButtonLinksCancelClick(sender, e);
                        break;
                }
                CadBridge.IsShowingJointGizmo = false;
                CadBridge.IsShowingTendonVisualization = false;
                CadBridge.IsShowingInertialGizmo = false;
                CadBridge.ShowHideVisualizations(false);
                Exporter.CleanUpTemporaryFeatures();
                CadBridge.UnselectAll();
                _isClosing = true;
            }
            Close();
        }

        private void OnCloseForm(object sender, FormClosingEventArgs e)
        {
            if (!_isClosing)
            {
                ButtonCancelClick(sender, e);
            }

            FormIsOpen = false;
            Instance = null;
        }

        private void OnFormClosed(object sender, FormClosedEventArgs e)
        {
            // When this owned modeless form is destroyed, Windows transfers activation
            // away from the host and it ends up behind another window. Re-assert the
            // host CAD window as foreground so it stays focused.
            CadBridge?.RestoreHostForeground();
        }

        private void ButtonJointCancelClick(object sender, EventArgs e)
        {
            if (previouslySelectedNode != null)
            {
                SaveJointDataFromPropertyBoxes(previouslySelectedNode.Link.Joint);
            }
            _isUpdatingTree = true;
            while (treeViewJointTree.Nodes.Count > 0)
            {
                LinkNode node = (LinkNode)treeViewJointTree.Nodes[0];
                treeViewJointTree.Nodes.Remove(node);
                BaseNode.Nodes.Add(node);
            }
            _isUpdatingTree = false;
            SaveConfigurationFromTree(BaseNode, true);
        }

        private void ButtonLinksCancelClick(object sender, EventArgs e)
        {
            if (previouslySelectedNode != null)
            {
                SaveLinkDataFromPropertyBoxes(previouslySelectedNode.Link);
            }
            SaveConfigurationFromTree(BaseNode, true);
        }

        private void ButtonLinksFinishClick(object sender, EventArgs e)
        {
            FinishExport(true);
        }

        private void ButtonLinksExportUrdfOnlyClick(object sender, EventArgs e)
        {
            FinishExport(false);
        }

        private void ButtonExportLinkMeshClick(object sender, EventArgs e)
        {
            LinkNode selectedNode = (LinkNode)treeViewLinkProperties.SelectedNode;
            if (selectedNode == null)
            {
                MessageBox.Show("Please select a link in the tree first.", "Export Link Mesh",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning, MessageBoxDefaultButton.Button1, MessageBoxOptions.DefaultDesktopOnly);
                return;
            }

            Link link = selectedNode.Link;

            List<MeshFormat> visualFormats = BuildVisualFormatList();
            List<MeshFormat> collisionFormats = BuildCollisionFormatList();

            if (visualFormats.Count == 0 && (!checkBoxCollisionMeshing.Checked || collisionFormats.Count == 0))
            {
                MessageBox.Show("At least one mesh export format must be selected.", "Export Link Mesh",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning, MessageBoxDefaultButton.Button1, MessageBoxOptions.DefaultDesktopOnly);
                return;
            }

            using (FolderBrowserDialog folderDialog = new FolderBrowserDialog())
            {
                folderDialog.Description = "Select output folder for link meshes";
                folderDialog.SelectedPath = CadBridge.GetLatestExportLocation();
                if (folderDialog.ShowDialog() != DialogResult.OK)
                    return;

                ExporterMeshingOptions meshingOptions = BuildMeshingOptionsFromForm();
                ExporterConfiguration config = SaveFormToExporterConfiguration();
                RobotPackage package = new RobotPackage("", "", config.folderStructure);

                Func<string, bool> confirmOverwrite = (filePath) =>
                {
                    string fileName = Path.GetFileName(filePath);
                    DialogResult result = MessageBox.Show(
                        $"File \"{fileName}\" already exists. Replace?",
                        "Export Link Mesh",
                        MessageBoxButtons.YesNo,
                        MessageBoxIcon.Question,
                        MessageBoxDefaultButton.Button1,
                        MessageBoxOptions.DefaultDesktopOnly);
                    return result == DialogResult.Yes;
                };

                this.Hide();

                try
                {
                    Exporter.ExportSingleLinkMeshes(
                        link,
                        package,
                        folderDialog.SelectedPath,
                        visualFormats,
                        checkBoxCollisionMeshing.Checked ? collisionFormats : new List<MeshFormat>(),
                        meshingOptions,
                        confirmOverwrite);

                    MessageBox.Show($"Mesh export complete for \"{link.Name}\".", "Export Link Mesh",
                        MessageBoxButtons.OK, MessageBoxIcon.Information, MessageBoxDefaultButton.Button1, MessageBoxOptions.DefaultDesktopOnly);
                }
                finally
                {
                    this.Show();
                }
            }
        }

        private List<MeshFormat> BuildVisualFormatList()
        {
            List<MeshFormat> formats = new List<MeshFormat>();
            if (checkBoxStepVisual.Checked) formats.Add(MeshFormat.stepSolidworks);
            if (checkBoxStlCADVisual.Checked) formats.Add(MeshFormat.stlSolidworks);
            if (checkBoxGlbCADVisual.Checked) formats.Add(MeshFormat.glbCAD);
            if (checkBoxObjCADVisual.Checked) formats.Add(MeshFormat.objCAD);
            if (checkBoxObjOpenCascadeVisual.Checked) formats.Add(MeshFormat.objOpenCascade);
            if (checkBoxStlOpenCascadeVisual.Checked) formats.Add(MeshFormat.stlOpenCascade);
            if (checkBoxGlbOpenCascadeVisual.Checked) formats.Add(MeshFormat.glbOpenCascade);
            return formats;
        }

        private List<MeshFormat> BuildCollisionFormatList()
        {
            List<MeshFormat> formats = new List<MeshFormat>();
            if (checkBoxStepCollision.Checked) formats.Add(MeshFormat.stepSolidworks);
            if (checkBoxStlCADCollision.Checked) formats.Add(MeshFormat.stlSolidworks);
            if (checkBoxGlbCADCollision.Checked) formats.Add(MeshFormat.glbCAD);
            if (checkBoxObjCADCollision.Checked) formats.Add(MeshFormat.objCAD);
            if (checkBoxObjOpenCascadeCollision.Checked) formats.Add(MeshFormat.objOpenCascade);
            if (checkBoxStlOpenCascadeCollision.Checked) formats.Add(MeshFormat.stlOpenCascade);
            if (checkBoxGlbOpenCascadeCollision.Checked) formats.Add(MeshFormat.glbOpenCascade);
            return formats;
        }

        private ExporterMeshingOptions BuildMeshingOptionsFromForm()
        {
            ExporterMeshingOptions meshingOptions = new ExporterMeshingOptions();
            meshingOptions.visualMeshingOptions = new Export.MeshingOptions();
            meshingOptions.collisionMeshingOptions = new Export.MeshingOptions();
            meshingOptions.visualMeshingOptions.linearDeflection = (float)numericUpDownLinearDeflection.Value;
            meshingOptions.visualMeshingOptions.angularDeflection = (float)numericUpDownAngularDeflection.Value;
            meshingOptions.visualMeshingOptions.scale = (float)numericUpDownScale.Value;
            ReadMesherSettings(meshingOptions.visualMeshingOptions, exportingCollision: false);
            meshingOptions.collisionMeshingOptions.linearDeflection = (float)numericUpDownLinearDeflectionCollision.Value;
            meshingOptions.collisionMeshingOptions.angularDeflection = (float)numericUpDownAngularDeflectionCollision.Value;
            meshingOptions.collisionMeshingOptions.scale = (float)numericUpDownScaleCollision.Value;
            ReadMesherSettings(meshingOptions.collisionMeshingOptions, exportingCollision: true);
            meshingOptions.meshTagExtension = textBoxMeshFileExtension.Text;
            meshingOptions.collisionMeshTagExtension = textBoxMeshFileExtensionCollision.Text;
            meshingOptions.exportCollision = checkBoxCollisionMeshing.Checked;
            meshingOptions.perLinkMeshing = checkBoxPerLinkMeshing.Checked;
            return meshingOptions;
        }

        // The four mesher settings are read and written in five places (export, config load/save,
        // per-link load/save), so the control-to-field mapping lives here once rather than being
        // repeated -- and repeated twice over, for the visual and collision columns.
        private void ReadMesherSettings(Export.MeshingOptions target, bool exportingCollision)
        {
            target.backend = BackendFromIndex(
                (exportingCollision ? comboBoxBackendCollision : comboBoxBackend).SelectedIndex);
            target.edgeSampling = EdgeSamplingFromIndex(
                (exportingCollision ? comboBoxEdgeSamplingCollision : comboBoxEdgeSampling).SelectedIndex);
            target.targetEdgeLength = (double)(exportingCollision
                ? numericUpDownTargetEdgeLengthCollision
                : numericUpDownTargetEdgeLength).Value;
            target.targetEdgeLengthFraction = (double)(exportingCollision
                ? numericUpDownEdgeLengthFractionCollision
                : numericUpDownEdgeLengthFraction).Value;
        }

        private void WriteMesherSettings(Export.MeshingOptions source, bool exportingCollision)
        {
            (exportingCollision ? comboBoxBackendCollision : comboBoxBackend).SelectedIndex =
                IndexFromBackend(source.backend);
            (exportingCollision ? comboBoxEdgeSamplingCollision : comboBoxEdgeSampling).SelectedIndex =
                IndexFromEdgeSampling(source.edgeSampling);
            SetClamped(
                exportingCollision
                    ? numericUpDownTargetEdgeLengthCollision
                    : numericUpDownTargetEdgeLength,
                source.targetEdgeLength);
            SetClamped(
                exportingCollision
                    ? numericUpDownEdgeLengthFractionCollision
                    : numericUpDownEdgeLengthFraction,
                source.targetEdgeLengthFraction);
        }

        // These values arrive from persisted JSON, which nothing validates against the control's
        // range -- and NumericUpDown throws rather than clamping when assigned out of range, which
        // inside a CAD host means taking the session down over a stale configuration.
        private static void SetClamped(NumericUpDown control, double value)
        {
            decimal clamped = (decimal)value;
            if (clamped < control.Minimum)
            {
                clamped = control.Minimum;
            }
            else if (clamped > control.Maximum)
            {
                clamped = control.Maximum;
            }
            control.Value = clamped;
        }

        // Mapped explicitly rather than cast, so the combo item order stays a presentation choice
        // and cannot silently change which backend a saved configuration means.
        private const int BackendIndexIsotropic = 0;
        private const int BackendIndexDelabella = 1;
        private const int EdgeSamplingIndexAdaptive = 0;
        private const int EdgeSamplingIndexUniform = 1;

        private static MeshingBackend BackendFromIndex(int index) =>
            index == BackendIndexDelabella ? MeshingBackend.Delabella : MeshingBackend.Isotropic;

        private static int IndexFromBackend(MeshingBackend backend) =>
            backend == MeshingBackend.Delabella ? BackendIndexDelabella : BackendIndexIsotropic;

        private static EdgeSampling EdgeSamplingFromIndex(int index) =>
            index == EdgeSamplingIndexUniform ? EdgeSampling.Uniform : EdgeSampling.Adaptive;

        private static int IndexFromEdgeSampling(EdgeSampling sampling) =>
            sampling == EdgeSampling.Uniform ? EdgeSamplingIndexUniform : EdgeSamplingIndexAdaptive;

        /// <summary>
        /// Applies the three interacting enable rules across both columns: the collision column
        /// follows its enable checkbox, the isotropic-only settings follow that column's backend,
        /// and the edge-length fraction is only meaningful while the absolute length is zero.
        /// </summary>
        private void UpdateMeshingControlEnabledState()
        {
            UpdateMeshingColumnEnabledState(exportingCollision: false, columnEnabled: true);
            UpdateMeshingColumnEnabledState(
                exportingCollision: true, columnEnabled: checkBoxCollisionMeshing.Checked);
        }

        private void UpdateMeshingColumnEnabledState(bool exportingCollision, bool columnEnabled)
        {
            ComboBox backend = exportingCollision ? comboBoxBackendCollision : comboBoxBackend;
            ComboBox edgeSampling =
                exportingCollision ? comboBoxEdgeSamplingCollision : comboBoxEdgeSampling;
            NumericUpDown targetEdgeLength = exportingCollision
                ? numericUpDownTargetEdgeLengthCollision
                : numericUpDownTargetEdgeLength;
            NumericUpDown edgeLengthFraction = exportingCollision
                ? numericUpDownEdgeLengthFractionCollision
                : numericUpDownEdgeLengthFraction;

            bool isotropic = BackendFromIndex(backend.SelectedIndex) == MeshingBackend.Isotropic;

            backend.Enabled = columnEnabled;
            edgeSampling.Enabled = columnEnabled && isotropic;
            targetEdgeLength.Enabled = columnEnabled && isotropic;
            edgeLengthFraction.Enabled = columnEnabled && isotropic && targetEdgeLength.Value == 0M;
        }

        private void MesherSettingChanged(object sender, EventArgs e)
        {
            UpdateMeshingControlEnabledState();
        }

        private void FinishExport(bool exportLinkMesh)
        {
            if (exportLinkMesh
                && !checkBoxStlCADVisual.Checked
                && !checkBoxStlOpenCascadeVisual.Checked
                && !checkBoxStepVisual.Checked
                && !checkBoxObjOpenCascadeVisual.Checked
                && !checkBoxGlbOpenCascadeVisual.Checked
                && !checkBoxGlbCADVisual.Checked
                && !checkBoxObjCADVisual.Checked)
            {
                MessageBox.Show("At least one Visual Link Export Format must be chosen!");
                return;
            }

            if (exportLinkMesh && checkBoxCollisionMeshing.Checked
                && !checkBoxStlCADCollision.Checked
                && !checkBoxStlOpenCascadeCollision.Checked
                && !checkBoxStepCollision.Checked
                && !checkBoxObjOpenCascadeCollision.Checked
                && !checkBoxGlbOpenCascadeCollision.Checked
                && !checkBoxGlbCADCollision.Checked
                && !checkBoxObjCADCollision.Checked)
            {
                MessageBox.Show("At least one Collision Link Export Format must be chosen when collision meshing is enabled!");
                return;
            }

            if (exportLinkMesh &&
                ((checkBoxStlOpenCascadeVisual.Checked && checkBoxStlCADVisual.Checked)
                || (checkBoxGlbOpenCascadeVisual.Checked && checkBoxGlbCADVisual.Checked)
                || (checkBoxObjOpenCascadeVisual.Checked && checkBoxObjCADVisual.Checked)))
            {
                MessageBox.Show("Only one method of export per format is supported at a time (visual)!");
                return;
            }

            if (exportLinkMesh && checkBoxCollisionMeshing.Checked &&
                ((checkBoxStlOpenCascadeCollision.Checked && checkBoxStlCADCollision.Checked)
                || (checkBoxGlbOpenCascadeCollision.Checked && checkBoxGlbCADCollision.Checked)
                || (checkBoxObjOpenCascadeCollision.Checked && checkBoxObjCADCollision.Checked)))
            {
                MessageBox.Show("Only one method of export per format is supported at a time (collision)!");
                return;
            }

            ExporterMeshingOptions meshingOptions = BuildMeshingOptionsFromForm();

            logger.Information("Completing Robot export");
            SaveConfigurationFromTree(BaseNode, false);

            // Saving selected node
            LinkNode node = (LinkNode)treeViewLinkProperties.SelectedNode;
            if (node != null)
            {
                SaveLinkDataFromPropertyBoxes(node.Link);
            }

            Exporter.Robot = CreateRobotFromTreeView(treeViewLinkProperties);
            Exporter.PackageName = Exporter.Robot.Name;

            // The UI should prevent these sorts of errors, but just in case
            string errors = CheckLinksForErrors(Exporter.Robot.BaseLink);
            if (!string.IsNullOrWhiteSpace(errors))
            {
                logger.Information("Link errors encountered:\n " + errors);

                string message = "The following links contained errors in either their link or joint " +
                    "properties. Please address before continuing\r\n\r\n" + errors;
                MessageBox.Show(message, "Robot Configuration Exporter - Link Errors");
                return;
            }

            string linksWithInvalidInertials = CheckLinksForValidInertials(Exporter.Robot.BaseLink);
            if (!string.IsNullOrEmpty(linksWithInvalidInertials))
            {
                logger.Information("Links without valid inertials encountered:\r\n" + linksWithInvalidInertials);

                string message = "The following links have possibly missing inertial properties. You may need to recalculate them." +
                "\r\n\r\nDo you want to proceed anyways?\r\n\r\n" + linksWithInvalidInertials;
                DialogResult result =
                    MessageBox.Show(message, "Links with invalid inertials detected", MessageBoxButtons.YesNo);

                if (result == DialogResult.No)
                {
                    logger.Information("Export canceled for user to review warnings");
                    return;
                }
            }

            string warnings = CheckLinksForWarnings(Exporter.Robot.BaseLink);

            if (!string.IsNullOrWhiteSpace(warnings))
            {
                logger.Information("Link warnings encountered:\r\n" + warnings);

                string message = "The following links contained issues that may cause problems. " +
                "Do you wish to proceed?\r\n\r\n" + warnings;
                DialogResult result =
                    MessageBox.Show(message, "Robot Configuration Exporter - Link Warnings", MessageBoxButtons.YesNo);

                if (result == DialogResult.No)
                {
                    logger.Information("Export canceled for user to review warnings");
                    return;
                }
            }

            string fileName = Path.GetFileNameWithoutExtension(Exporter.PackageName);

            bool retrievedLastSavePath = true;

            string initialDir = GetMruDirectory();
            if (string.IsNullOrEmpty(initialDir))
            {
                initialDir = CadBridge.GetLatestExportLocation();
            }

            if (string.IsNullOrEmpty(initialDir))
            {
                retrievedLastSavePath = false;
            }
            else
            {
                string candidate = Path.Combine(initialDir, fileName);
                if (Directory.Exists(candidate))
                {
                    int suffix = 1;
                    while (Directory.Exists($"{candidate} ({suffix})"))
                        suffix++;
                    fileName = $"{fileName} ({suffix})";
                }
            }

            SaveFileDialog saveFileDialog1 = new SaveFileDialog
            {
                InitialDirectory = initialDir ?? "",
                RestoreDirectory = !retrievedLastSavePath, // let Windows restore if we could't retrieve it ourselves
                FileName = fileName
            };

            bool saveResult = DialogResult.OK == saveFileDialog1.ShowDialog();
            saveFileDialog1.Dispose();
            if (saveResult)
            {
                string parentFolder = Path.GetDirectoryName(saveFileDialog1.FileName);
                string baseName = Path.GetFileName(saveFileDialog1.FileName);

                Exporter.SavePath = parentFolder;
                Exporter.PackageName = baseName;
                CadBridge.SetLatestExportLocation(Exporter.SavePath);

                string targetFolder = Path.Combine(parentFolder, baseName);
                logger.Information("Saving Robot package to " + targetFolder);

                List<MeshFormat> exportFormats = BuildVisualFormatList();
                List<MeshFormat> collisionExportFormats = BuildCollisionFormatList();

                this.WindowState = FormWindowState.Minimized;

                ExporterConfiguration config = SaveFormToExporterConfiguration();
                Exporter.ExportRobot(exportLinkMesh, exportFormats, collisionExportFormats, meshingOptions, config);

                string urdfPath = Exporter.URDFFileName;
                string mjcfPath = Exporter.MJCFFileName;
                string superdexBotPath = Exporter.SuperDexBotFileName;

                using (var dialog = new ExportCompleteDialog(urdfPath, mjcfPath, superdexBotPath))
                {
                    if (dialog.ShowDialog(this) == DialogResult.Yes)
                    {
                        Process.Start("explorer.exe", $"/select,\"{targetFolder}\"");
                        this.WindowState = FormWindowState.Minimized;
                    }
                    else
                    {
                        this.WindowState = FormWindowState.Normal;
                        Focus();
                    }
                }
            }
        }

        private string CheckLinksForErrors(Link baseLink)
        {
            StringBuilder builder = new StringBuilder();
            CheckLinkForErrors(baseLink, builder);
            return builder.ToString();
        }

        private StringBuilder CheckLinkForErrors(Link link, StringBuilder builder)
        {
            if (!link.AreRequiredFieldsSatisfied())
            {
                builder.Append(link.Name).Append("\r\n");
            }
            foreach (Link child in link.Children)
            {
                CheckLinkForErrors(child, builder);
            }
            return builder;
        }

        private string CheckLinksForValidInertials(Link baseLink)
        {
            StringBuilder builder = new StringBuilder();
            CheckLinkForValidInertials(baseLink, builder);
            return builder.ToString();
        }

        private StringBuilder CheckLinkForValidInertials(Link link, StringBuilder builder)
        {
            if (!link.HasValidInertialProperties())
            {
                builder.Append(link.Name).Append("\r\n");
            }
            foreach (Link child in link.Children)
            {
                CheckLinkForValidInertials(child, builder);
            }
            return builder;
        }

        private void TreeViewLinkPropertiesAfterSelect(object sender, TreeViewEventArgs e)
        {
            if (_isUpdatingTree)
            {
                return;
            }

            Font fontRegular = new Font(treeViewJointTree.Font, FontStyle.Regular);
            Font fontBold = new Font(treeViewJointTree.Font, FontStyle.Bold);
            if (previouslySelectedNode != null)
            {
                SaveLinkDataFromPropertyBoxes(previouslySelectedNode.Link);
                previouslySelectedNode.NodeFont = fontRegular;
            }
            LinkNode node = (LinkNode)e.Node;
            node.NodeFont = fontBold;
            node.Text = node.Text;

            if (checkBoxLinkHighlights.Checked)
            {
                CadBridge.SelectLinkComponents(node);
            }

            FillLinkPropertyBoxes(node.Link);
            treeViewLinkProperties.Focus();
            previouslySelectedNode = node;

            CadBridge.CurrentLinkNodeShown = node;
            CadBridge.TriggerGraphicsRedraw();
        }

        /// <summary>
        /// Validates text entry for numerical text boxes to limit improper input. It's not perfect
        /// because you can still copy and paste bad input into the fields, but that's addressed
        /// elsewhere
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void NumericalTextBoxKeyPress(object sender, KeyPressEventArgs e)
        {
            // In most cases, if we can't parse what they are trying to type, then it's not
            // valid input.
            TextBox textBox = (TextBox)sender;
            string potentialText = textBox.Text + e.KeyChar;
            bool parseSuccess =
                double.TryParse(potentialText,
                    ElementAttribute.NumberStyle,
                    ElementAttribute.NumberFormat,
                    out _);

            // If the key pressed is not a digit, +/- sign or the decimal separator than ignore it (e.Handled = true)
            e.Handled = (!parseSuccess &&
                         !char.IsControl(e.KeyChar) &&
                         !char.IsDigit(e.KeyChar) &&
                         potentialText != "-" &&
                         potentialText != "+");
        }

        private void TreeViewJointtreeAfterSelect(object sender, TreeViewEventArgs e)
        {
            if (_isUpdatingTree)
            {
                return;
            }

            Font fontRegular = new Font(treeViewJointTree.Font, FontStyle.Regular);
            Font fontBold = new Font(treeViewJointTree.Font, FontStyle.Bold);
            if (previouslySelectedNode != null && !previouslySelectedNode.IsBaseNode)
            {
                SaveJointDataFromPropertyBoxes(previouslySelectedNode.Link.Joint);
            }
            if (previouslySelectedNode != null)
            {
                previouslySelectedNode.NodeFont = fontRegular;
            }
            LinkNode node = (LinkNode)e.Node;

            if (checkBoxShowJointHighlights.Checked)
            {
                CadBridge.SelectJointComponents(node);
            }

            node.NodeFont = fontBold;
            node.Text = node.Text;
            FillJointPropertyBoxes(node.Link.Joint);
            previouslySelectedNode = node;

            CadBridge.IsShowingJointGizmo = true;
            CadBridge.CurrentNodeShown = node;

            CadBridge.TriggerGraphicsRedraw();
        }

        public void LoadMeshingOptionsFromLink(Link link)
        {
            numericUpDownLinearDeflection.Value = (decimal)link.visualMeshingOptions.linearDeflection;
            numericUpDownAngularDeflection.Value = (decimal)link.visualMeshingOptions.angularDeflection;
            numericUpDownScale.Value = (decimal)link.visualMeshingOptions.scale;
            WriteMesherSettings(link.visualMeshingOptions, exportingCollision: false);

            numericUpDownLinearDeflectionCollision.Value = (decimal)link.collisionMeshingOptions.linearDeflection;
            numericUpDownAngularDeflectionCollision.Value = (decimal)link.collisionMeshingOptions.angularDeflection;
            numericUpDownScaleCollision.Value = (decimal)link.collisionMeshingOptions.scale;
            WriteMesherSettings(link.collisionMeshingOptions, exportingCollision: true);

            UpdateMeshingControlEnabledState();
        }
        public void SaveMeshingOptionsToLink(Link link)
        {
            link.visualMeshingOptions.linearDeflection = (double)numericUpDownLinearDeflection.Value;
            link.visualMeshingOptions.angularDeflection = (double)numericUpDownAngularDeflection.Value;
            link.visualMeshingOptions.scale = (double)numericUpDownScale.Value;
            ReadMesherSettings(link.visualMeshingOptions, exportingCollision: false);

            link.collisionMeshingOptions.linearDeflection = (double)numericUpDownLinearDeflectionCollision.Value;
            link.collisionMeshingOptions.angularDeflection = (double)numericUpDownAngularDeflectionCollision.Value;
            link.collisionMeshingOptions.scale = (double)numericUpDownScaleCollision.Value;
            ReadMesherSettings(link.collisionMeshingOptions, exportingCollision: true);
        }

        public void LoadMeshingOptionsToForm(ExporterConfiguration config)
        {
            if (config == null)
            {
                config = new ExporterConfiguration(); // Use defaults
            }

            numericUpDownLinearDeflection.Value = (decimal)config.visualMeshingOptions.linearDeflection;
            numericUpDownAngularDeflection.Value = (decimal)config.visualMeshingOptions.angularDeflection;
            numericUpDownScale.Value = (decimal)config.visualMeshingOptions.scale;
            WriteMesherSettings(config.visualMeshingOptions, exportingCollision: false);

            numericUpDownLinearDeflectionCollision.Value = (decimal)config.collisionMeshingOptions.linearDeflection;
            numericUpDownAngularDeflectionCollision.Value = (decimal)config.collisionMeshingOptions.angularDeflection;
            numericUpDownScaleCollision.Value = (decimal)config.collisionMeshingOptions.scale;
            WriteMesherSettings(config.collisionMeshingOptions, exportingCollision: true);

            UpdateMeshingControlEnabledState();
        }

        public void LoadExporterConfigurationToForm(ExporterConfiguration config)
        {
            if (config == null)
            {
                config = new ExporterConfiguration();
            }

            LoadMeshingOptionsToForm(config);

            checkBoxGlbOpenCascadeVisual.Checked = config.meshFormats.Contains(MeshFormat.glbOpenCascade);
            checkBoxObjOpenCascadeVisual.Checked = config.meshFormats.Contains(MeshFormat.objOpenCascade);
            checkBoxStlOpenCascadeVisual.Checked = config.meshFormats.Contains(MeshFormat.stlOpenCascade);
            checkBoxStlCADVisual.Checked = config.meshFormats.Contains(MeshFormat.stlSolidworks);
            checkBoxStepVisual.Checked = config.meshFormats.Contains(MeshFormat.stepSolidworks);
            checkBoxGlbCADVisual.Checked = config.meshFormats.Contains(MeshFormat.glbCAD);
            checkBoxObjCADVisual.Checked = config.meshFormats.Contains(MeshFormat.objCAD);

            if (config.hasCollisionMeshFormats)
            {
                checkBoxGlbOpenCascadeCollision.Checked = config.collisionMeshFormats.Contains(MeshFormat.glbOpenCascade);
                checkBoxObjOpenCascadeCollision.Checked = config.collisionMeshFormats.Contains(MeshFormat.objOpenCascade);
                checkBoxStlOpenCascadeCollision.Checked = config.collisionMeshFormats.Contains(MeshFormat.stlOpenCascade);
                checkBoxStlCADCollision.Checked = config.collisionMeshFormats.Contains(MeshFormat.stlSolidworks);
                checkBoxStepCollision.Checked = config.collisionMeshFormats.Contains(MeshFormat.stepSolidworks);
                checkBoxGlbCADCollision.Checked = config.collisionMeshFormats.Contains(MeshFormat.glbCAD);
                checkBoxObjCADCollision.Checked = config.collisionMeshFormats.Contains(MeshFormat.objCAD);
            }
            else
            {
                checkBoxGlbOpenCascadeCollision.Checked = config.meshFormats.Contains(MeshFormat.glbOpenCascade);
                checkBoxObjOpenCascadeCollision.Checked = config.meshFormats.Contains(MeshFormat.objOpenCascade);
                checkBoxStlOpenCascadeCollision.Checked = config.meshFormats.Contains(MeshFormat.stlOpenCascade);
                checkBoxStlCADCollision.Checked = config.meshFormats.Contains(MeshFormat.stlSolidworks);
                checkBoxStepCollision.Checked = config.meshFormats.Contains(MeshFormat.stepSolidworks);
                checkBoxGlbCADCollision.Checked = config.meshFormats.Contains(MeshFormat.glbCAD);
                checkBoxObjCADCollision.Checked = config.meshFormats.Contains(MeshFormat.objCAD);
            }

            SantizeCADSpecificCheckboxes();

            textBoxMeshFileExtension.Text = config.meshTagExtension;
            textBoxMeshFileExtensionCollision.Text = config.collisionMeshTagExtension;
            checkBoxCollisionMeshing.Checked = config.enableCollisionMeshing;
            CollisionCheckboxChanged(checkBoxCollisionMeshing, null);
            checkBoxPerLinkMeshing.Checked = config.perLinkMeshing;

            string documentTitle = CadBridge.GetDocumentTitle();

            textBoxRobotName.Text = string.IsNullOrEmpty(config.robotName) ? Path.GetFileNameWithoutExtension(documentTitle) : config.robotName;

            comboBoxFolderStructure.SelectedIndex = (int)config.folderStructure;
        }

        public ExporterConfiguration SaveFormToExporterConfiguration()
        {
            ExporterConfiguration config = new ExporterConfiguration();

            config.visualMeshingOptions.linearDeflection = (double)numericUpDownLinearDeflection.Value;
            config.visualMeshingOptions.angularDeflection = (double)numericUpDownAngularDeflection.Value;
            config.visualMeshingOptions.scale = (double)numericUpDownScale.Value;
            ReadMesherSettings(config.visualMeshingOptions, exportingCollision: false);

            config.collisionMeshingOptions.linearDeflection = (double)numericUpDownLinearDeflectionCollision.Value;
            config.collisionMeshingOptions.angularDeflection = (double)numericUpDownAngularDeflectionCollision.Value;
            config.collisionMeshingOptions.scale = (double)numericUpDownScaleCollision.Value;
            ReadMesherSettings(config.collisionMeshingOptions, exportingCollision: true);

            config.meshFormats = new List<MeshFormat>();

            if (checkBoxGlbOpenCascadeVisual.Checked)
                config.meshFormats.Add(MeshFormat.glbOpenCascade);
            if (checkBoxObjOpenCascadeVisual.Checked)
                config.meshFormats.Add(MeshFormat.objOpenCascade);
            if (checkBoxStlOpenCascadeVisual.Checked)
                config.meshFormats.Add(MeshFormat.stlOpenCascade);
            if (checkBoxStlCADVisual.Checked)
                config.meshFormats.Add(MeshFormat.stlSolidworks);
            if (checkBoxStepVisual.Checked)
                config.meshFormats.Add(MeshFormat.stepSolidworks);
            if (checkBoxGlbCADVisual.Checked)
                config.meshFormats.Add(MeshFormat.glbCAD);
            if (checkBoxObjCADVisual.Checked)
                config.meshFormats.Add(MeshFormat.objCAD);

            config.meshTagExtension = textBoxMeshFileExtension.Text;
            config.collisionMeshTagExtension = textBoxMeshFileExtensionCollision.Text;
            config.enableCollisionMeshing = checkBoxCollisionMeshing.Checked;
            config.perLinkMeshing = checkBoxPerLinkMeshing.Checked;

            config.collisionMeshFormats = new List<MeshFormat>();

            if (checkBoxGlbOpenCascadeCollision.Checked)
                config.collisionMeshFormats.Add(MeshFormat.glbOpenCascade);
            if (checkBoxObjOpenCascadeCollision.Checked)
                config.collisionMeshFormats.Add(MeshFormat.objOpenCascade);
            if (checkBoxStlOpenCascadeCollision.Checked)
                config.collisionMeshFormats.Add(MeshFormat.stlOpenCascade);
            if (checkBoxStlCADCollision.Checked)
                config.collisionMeshFormats.Add(MeshFormat.stlSolidworks);
            if (checkBoxStepCollision.Checked)
                config.collisionMeshFormats.Add(MeshFormat.stepSolidworks);
            if (checkBoxGlbCADCollision.Checked)
                config.collisionMeshFormats.Add(MeshFormat.glbCAD);
            if (checkBoxObjCADCollision.Checked)
                config.collisionMeshFormats.Add(MeshFormat.objCAD);

            config.hasCollisionMeshFormats = true;

            config.robotName = textBoxRobotName.Text;

            config.folderStructure = (FolderStructure)comboBoxFolderStructure.SelectedIndex;

            return config;
        }

        private void ComboBoxExporterConfigurationPreset_SelectedIndexChanged(object sender, EventArgs e)
        {
            var selected = comboBoxExporterConfigurationPreset.SelectedItem;
            // Ignore separator
            if (selected is string str)
            {
                if (str == SavePresetOption)
                {
                    SaveCurrentAsPreset();
                    // Reset selection to previous or default
                    comboBoxExporterConfigurationPreset.SelectedIndex = 0;
                }
                return;
            }
            if (selected is ExporterConfigurationPreset preset)
            {
                ExporterConfiguration newConfig = preset.Configuration;
                newConfig.robotName = textBoxRobotName.Text;
                LoadExporterConfigurationToForm(newConfig);
            }
        }

        private void SaveCurrentAsPreset()
        {
            // Get current config from form
            ExporterConfiguration config = SaveFormToExporterConfiguration();

            using (var dialog = new InputDialog("Save Preset", "Enter preset name:"))
            {
                if (dialog.ShowDialog(this) != DialogResult.OK || string.IsNullOrWhiteSpace(dialog.InputText))
                    return;
                string presetName = dialog.InputText.Trim();
                string invalidChars = new string(Path.GetInvalidFileNameChars());

                foreach (char c in invalidChars)
                    presetName = presetName.Replace(c.ToString(), "");
                string filePath = Path.Combine(GetPresetsDirectory(), $"{presetName}.json");
                string json = JsonConvert.SerializeObject(config, GetJsonSettings());
                File.WriteAllText(filePath, json);
                // Refresh the comboBox
                PopulatePresetComboBox();
                MessageBox.Show($"Preset '{presetName}' saved.\n\nYou can share your presets from  {GetPresetsDirectory()}", "Success",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
        }

        private static readonly List<ExporterConfigurationPreset> PredefinedPresets = new List<ExporterConfigurationPreset>
        {
            new ExporterConfigurationPreset
            {
                Name = "Default",
                Configuration = new ExporterConfiguration(),
                IsUserPreset = false
            },
            new ExporterConfigurationPreset
            {
#if NX
                Name = "MuJoCo/ROS (obj) - Native, widest compat",
#elif SOLIDWORKS
                Name = "MuJoCo/ROS (stl) - Native, widest compat",
#endif
                Configuration = new ExporterConfiguration
                {
                    visualMeshingOptions = new Export.MeshingOptions
                    {
                        linearDeflection = 0.5,
                        angularDeflection = 0.5,
                        scale = 1.0
                    },
                    collisionMeshingOptions = new Export.MeshingOptions
                    {
                        linearDeflection = 1.0,
                        angularDeflection = 0.75,
                        scale = 1.0
                    },
                    enableCollisionMeshing = true,
#if NX
                    meshFormats = new List<MeshFormat> { MeshFormat.objCAD },
                    collisionMeshFormats = new List<MeshFormat> { MeshFormat.objCAD },
#elif SOLIDWORKS
                    meshFormats = new List<MeshFormat> { MeshFormat.stlSolidworks },
                    collisionMeshFormats = new List<MeshFormat> { MeshFormat.stlSolidworks },
#endif
#if NX
                    meshTagExtension = "obj",
                    collisionMeshTagExtension = "obj",
#elif SOLIDWORKS
                    meshTagExtension = "stl",
                    collisionMeshTagExtension = "stl",
#endif
                    hasCollisionMeshFormats = true,
                    folderStructure = FolderStructure.ROS,
                },
                IsUserPreset = false
            },
            new ExporterConfigurationPreset
            {
#if NX
                Name = "MuJoCo/ROS (stl) - Native, widest compat",
#elif SOLIDWORKS
                Name = "MuJoCo (obj) - SuperDex, widest compat",
#endif
                Configuration = new ExporterConfiguration
                {
                    visualMeshingOptions = new Export.MeshingOptions
                    {
                        linearDeflection = 0.5,
                        angularDeflection = 0.5,
                        scale = 1.0
                    },
                    collisionMeshingOptions = new Export.MeshingOptions
                    {
                        linearDeflection = 1.0,
                        angularDeflection = 0.75,
                        scale = 1.0
                    },
                    enableCollisionMeshing = true,
#if NX
                    meshFormats = new List<MeshFormat> { MeshFormat.stlSolidworks },
                    collisionMeshFormats = new List<MeshFormat> { MeshFormat.stlSolidworks },
#elif SOLIDWORKS
                    meshFormats = new List<MeshFormat> { MeshFormat.objOpenCascade },
                    collisionMeshFormats = new List<MeshFormat> { MeshFormat.objOpenCascade },
#endif
#if NX
                    meshTagExtension = "stl",
                    collisionMeshTagExtension = "stl",
#elif SOLIDWORKS
                    meshTagExtension = "obj",
                    collisionMeshTagExtension = "obj",
#endif
                    hasCollisionMeshFormats = true,
                    folderStructure = FolderStructure.ROS,
                },
                IsUserPreset = false
            },
            new ExporterConfigurationPreset
            {
#if NX
                Name = "SuperDex (glb) - Quick preview",
#elif SOLIDWORKS
                Name = "SuperDex (glb, stl) - Quick preview",
#endif
                Configuration = new ExporterConfiguration
                {
                    visualMeshingOptions = new Export.MeshingOptions
                    {
                        linearDeflection = 0.1,
                        angularDeflection = 0.5,
                        scale = 1.0
                    },
                    collisionMeshingOptions = new Export.MeshingOptions
                    {
                        linearDeflection = 0.5,
                        angularDeflection = 0.75,
                        scale = 1.0
                    },
                    enableCollisionMeshing = true,
#if NX
                    meshFormats = new List<MeshFormat> { MeshFormat.glbCAD },
                    collisionMeshFormats = new List<MeshFormat> { MeshFormat.glbCAD },
#elif SOLIDWORKS
                    meshFormats = new List<MeshFormat> { MeshFormat.glbOpenCascade },
                    collisionMeshFormats = new List<MeshFormat> { MeshFormat.stlSolidworks },
#endif
                    meshTagExtension = "glb",
#if NX
                    collisionMeshTagExtension = "glb",
#elif SOLIDWORKS
                    collisionMeshTagExtension = "stl",
#endif
                    hasCollisionMeshFormats = true,
                    folderStructure = FolderStructure.SuperDex,
                },
                IsUserPreset = false
            },
            new ExporterConfigurationPreset
            {
#if NX
                Name = "SuperDex (glb, stp) - Studio import",
#elif SOLIDWORKS
                Name = "SuperDex (glb, stl, stp) - Studio import",
#endif
                Configuration = new ExporterConfiguration
                {
                    visualMeshingOptions = new Export.MeshingOptions
                    {
                        linearDeflection = 0.1,
                        angularDeflection = 0.5,
                        scale = 1.0
                    },
                    collisionMeshingOptions = new Export.MeshingOptions
                    {
                        linearDeflection = 0.5,
                        angularDeflection = 0.75,
                        scale = 1.0
                    },
                    enableCollisionMeshing = true,
#if NX
                    meshFormats = new List<MeshFormat> { MeshFormat.glbCAD },
                    collisionMeshFormats = new List<MeshFormat> { MeshFormat.glbCAD, MeshFormat.stepSolidworks },
#elif SOLIDWORKS
                    meshFormats = new List<MeshFormat> { MeshFormat.glbOpenCascade },
                    collisionMeshFormats = new List<MeshFormat> { MeshFormat.stlSolidworks, MeshFormat.stepSolidworks },
#endif
                    meshTagExtension = "glb",
                    collisionMeshTagExtension = "mochi.h5",
                    hasCollisionMeshFormats = true,
                    folderStructure = FolderStructure.SuperDex,
                },
                IsUserPreset = false
            },
        };

        private const string SavePresetOption = "(Save Preset...)";

        private static string GetPresetsDirectory()
        {
            string path = Path.Combine(
                System.Environment.GetFolderPath(System.Environment.SpecialFolder.UserProfile),
#if NX
                "NxRobotExporter",
#elif SOLIDWORKS
                "sw2urdf",
#endif
                "presets");

            if (!Directory.Exists(path))
                Directory.CreateDirectory(path);

            return path;
        }

        private static JsonSerializerSettings GetJsonSettings()
        {
            return new JsonSerializerSettings
            {
                ObjectCreationHandling = ObjectCreationHandling.Replace,
                Converters = new List<JsonConverter> { new StringEnumConverter() },
                Formatting = Formatting.Indented
            };
        }

        private static List<ExporterConfigurationPreset> LoadUserPresets()
        {
            var presets = new List<ExporterConfigurationPreset>();
            string presetsDir = GetPresetsDirectory();
            foreach (string file in Directory.GetFiles(presetsDir, "*.json"))
            {
                try
                {
                    string json = File.ReadAllText(file);
                    var config = JsonConvert.DeserializeObject<ExporterConfiguration>(json, GetJsonSettings());
                    presets.Add(new ExporterConfigurationPreset
                    {
                        Name = Path.GetFileNameWithoutExtension(file),
                        Configuration = config,
                        IsUserPreset = true
                    });
                }
                catch
                {
                    // Skip invalid files
                }
            }
            return presets;
        }

        private void PopulatePresetComboBox()
        {
            comboBoxExporterConfigurationPreset.Items.Clear();
            comboBoxExporterConfigurationPreset.Items.Add("");
            // Add predefined presets
            foreach (var preset in PredefinedPresets)
            {
                comboBoxExporterConfigurationPreset.Items.Add(preset);
            }
            // Add user presets
            var userPresets = LoadUserPresets();
            if (userPresets.Count > 0)
            {
                comboBoxExporterConfigurationPreset.Items.Add("─── User Presets ───"); // Separator
                foreach (var preset in userPresets)
                {
                    comboBoxExporterConfigurationPreset.Items.Add(preset);
                }
            }
            // Add save option at the end
            comboBoxExporterConfigurationPreset.Items.Add(SavePresetOption);
            // Select default
            comboBoxExporterConfigurationPreset.SelectedIndex = 0;
        }

        // This essentially pulls the "RestoreDirectory = true" directory
        private const string LastVisitedMruPath =
            @"Software\Microsoft\Windows\CurrentVersion\Explorer\ComDlg32\LastVisitedPidlMRU";

        [DllImport("shell32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool SHGetPathFromIDListW(
            IntPtr pidl, [MarshalAs(UnmanagedType.LPTStr)] StringBuilder pszPath);

        private static string GetMruDirectory()
        {
            try
            {
                using (RegistryKey rk = Registry.CurrentUser.OpenSubKey(LastVisitedMruPath))
                {
                    if (rk == null)
                        return null;

                    byte[] mruList = rk.GetValue("MRUListEx") as byte[];
                    if (mruList == null || mruList.Length < 4)
                        return null;

                    // Walk the MRU list looking for ugraf.exe entries
                    for (int i = 0; i < mruList.Length - 4; i += 4)
                    {
                        int index = BitConverter.ToInt32(mruList, i);
                        if (index == -1)
                            break;

                        byte[] entry = rk.GetValue(index.ToString()) as byte[];
                        if (entry == null || entry.Length < 4)
                            continue;

                        // Entry format: null-terminated Unicode exe name, followed by PIDL
                        string exeName = Encoding.Unicode.GetString(entry, 0, entry.Length);
                        int nullIdx = exeName.IndexOf('\0');
                        if (nullIdx < 0)
                            continue;

                        string app = exeName.Substring(0, nullIdx);
#if NX
                        if (!app.Equals("ugraf.exe", StringComparison.OrdinalIgnoreCase))
#elif SOLIDWORKS
                        if (!app.Equals("SLDWORKS.exe", StringComparison.OrdinalIgnoreCase))
#endif
                        continue;

                        // PIDL starts after the null-terminated exe name (double-null in Unicode)
                        int pidlOffset = (nullIdx + 1) * 2;
                        if (pidlOffset >= entry.Length)
                            continue;

                        byte[] pidlBytes = new byte[entry.Length - pidlOffset];
                        Array.Copy(entry, pidlOffset, pidlBytes, 0, pidlBytes.Length);

                        GCHandle handle = GCHandle.Alloc(pidlBytes, GCHandleType.Pinned);
                        try
                        {
                            StringBuilder path = new StringBuilder(260);
                            if (SHGetPathFromIDListW(handle.AddrOfPinnedObject(), path))
                            {
                                string dir = path.ToString();
                                if (Directory.Exists(dir))
                                    return dir;
                            }
                        }
                        finally
                        {
                            handle.Free();
                        }
                    }
                }
            }
            catch
            {
            }

            return null;
        }
    }

    public class ExporterConfigurationPreset
    {
        public string Name { get; set; }
        public ExporterConfiguration Configuration { get; set; }
        public bool IsUserPreset { get; set; }
        public override string ToString() => Name;
    }

    public enum MeshFormat
    {
        glbOpenCascade,
        objOpenCascade,
        stlOpenCascade,
        stlSolidworks,
        stepSolidworks,
        objCAD,
        glbCAD
    }

    public enum FolderStructure
    {
        Legacy,
        ROS,
        MuJoCo,
        SuperDex,
    }

    [Serializable]
    public class ExporterConfiguration
    {
        public Export.MeshingOptions visualMeshingOptions;
        public Export.MeshingOptions collisionMeshingOptions;
        public List<MeshFormat> meshFormats;
        public List<MeshFormat> collisionMeshFormats;
        public bool hasCollisionMeshFormats;
        public string meshTagExtension;
        public bool enableCollisionMeshing;
        public bool perLinkMeshing;
        public string collisionMeshTagExtension;
        public string robotName;
        public FolderStructure folderStructure;

        public ExporterConfiguration()
        {
            visualMeshingOptions = new Export.MeshingOptions
            {
                linearDeflection = 0.1,
                angularDeflection = 0.5,
                scale = 1.0
            };

            collisionMeshingOptions = new Export.MeshingOptions
            {
                linearDeflection = 0.5,
                angularDeflection = 0.75,
                scale = 1.0
            };

            meshFormats = new List<MeshFormat>
            {
#if NX
                MeshFormat.glbCAD,
#elif SOLIDWORKS
                MeshFormat.glbOpenCascade,
#endif
            };

            collisionMeshFormats = new List<MeshFormat>
            {
#if NX
                MeshFormat.objCAD,
#elif SOLIDWORKS
                MeshFormat.stlSolidworks,
#endif
            };

            hasCollisionMeshFormats = true;

#if NX
            meshTagExtension = "glb";
            collisionMeshTagExtension = "obj";
#elif SOLIDWORKS
            meshTagExtension = "glb";
            collisionMeshTagExtension = "stl";
#endif

            enableCollisionMeshing = true;

            perLinkMeshing = false;

            robotName = "";

            folderStructure = FolderStructure.SuperDex;
        }
    }

}
