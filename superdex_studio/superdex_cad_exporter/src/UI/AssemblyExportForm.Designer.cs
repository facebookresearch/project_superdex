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

namespace CADRobotExporter.UI
{
    partial class AssemblyExportForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed;
        /// otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.saveFileDialog1 = new System.Windows.Forms.SaveFileDialog();
            this.buttonLinksFinish = new System.Windows.Forms.Button();
            this.buttonLinksExportUrdfOnly = new System.Windows.Forms.Button();
            this.buttonPrevious = new System.Windows.Forms.Button();
            this.toolTips = new System.Windows.Forms.ToolTip(this.components);
            this.label60 = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.label28 = new System.Windows.Forms.Label();
            this.label34 = new System.Windows.Forms.Label();
            this.label42 = new System.Windows.Forms.Label();
            this.labelBackend = new System.Windows.Forms.Label();
            this.labelEdgeSampling = new System.Windows.Forms.Label();
            this.labelTargetEdgeLength = new System.Windows.Forms.Label();
            this.labelEdgeLengthFraction = new System.Windows.Forms.Label();
            this.comboBoxBackend = new System.Windows.Forms.ComboBox();
            this.comboBoxBackendCollision = new System.Windows.Forms.ComboBox();
            this.comboBoxEdgeSampling = new System.Windows.Forms.ComboBox();
            this.comboBoxEdgeSamplingCollision = new System.Windows.Forms.ComboBox();
            this.numericUpDownTargetEdgeLength = new System.Windows.Forms.NumericUpDown();
            this.numericUpDownTargetEdgeLengthCollision = new System.Windows.Forms.NumericUpDown();
            this.numericUpDownEdgeLengthFraction = new System.Windows.Forms.NumericUpDown();
            this.numericUpDownEdgeLengthFractionCollision = new System.Windows.Forms.NumericUpDown();
            this.buttonNext = new System.Windows.Forms.Button();
            this.buttonClose = new System.Windows.Forms.Button();
            this.label29 = new System.Windows.Forms.Label();
            this.textBoxRobotName = new System.Windows.Forms.TextBox();
            this.tabPageLinkProperties = new System.Windows.Forms.TabPage();
            this.groupBox11 = new System.Windows.Forms.GroupBox();
            this.label70 = new System.Windows.Forms.Label();
            this.trackBarLinkGizmoSize = new System.Windows.Forms.TrackBar();
            this.checkBoxLinkHighlights = new System.Windows.Forms.CheckBox();
            this.checkBoxShowLinkVisualization = new System.Windows.Forms.CheckBox();
            this.label2 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.groupBox5 = new System.Windows.Forms.GroupBox();
            this.buttonRecalculateInertial = new System.Windows.Forms.Button();
            this.groupBox9 = new System.Windows.Forms.GroupBox();
            this.textBoxIxx = new System.Windows.Forms.TextBox();
            this.label49 = new System.Windows.Forms.Label();
            this.label48 = new System.Windows.Forms.Label();
            this.textBoxIxy = new System.Windows.Forms.TextBox();
            this.textBoxIyz = new System.Windows.Forms.TextBox();
            this.textBoxIxz = new System.Windows.Forms.TextBox();
            this.textBoxIzz = new System.Windows.Forms.TextBox();
            this.label18 = new System.Windows.Forms.Label();
            this.label11 = new System.Windows.Forms.Label();
            this.label50 = new System.Windows.Forms.Label();
            this.label14 = new System.Windows.Forms.Label();
            this.textBoxIyy = new System.Windows.Forms.TextBox();
            this.groupBox8 = new System.Windows.Forms.GroupBox();
            this.textBoxInertialOriginX = new System.Windows.Forms.TextBox();
            this.label16 = new System.Windows.Forms.Label();
            this.textBoxInertialOriginYaw = new System.Windows.Forms.TextBox();
            this.textBoxInertialOriginPitch = new System.Windows.Forms.TextBox();
            this.label13 = new System.Windows.Forms.Label();
            this.textBoxInertialOriginRoll = new System.Windows.Forms.TextBox();
            this.textBoxInertialOriginY = new System.Windows.Forms.TextBox();
            this.label17 = new System.Windows.Forms.Label();
            this.label45 = new System.Windows.Forms.Label();
            this.label46 = new System.Windows.Forms.Label();
            this.textBoxInertialOriginZ = new System.Windows.Forms.TextBox();
            this.label47 = new System.Windows.Forms.Label();
            this.label15 = new System.Windows.Forms.Label();
            this.label12 = new System.Windows.Forms.Label();
            this.textBoxMass = new System.Windows.Forms.TextBox();
            this.groupBox4 = new System.Windows.Forms.GroupBox();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.comboBoxExporterConfigurationPreset = new System.Windows.Forms.ComboBox();
            this.groupBox7 = new System.Windows.Forms.GroupBox();
            this.groupBox16 = new System.Windows.Forms.GroupBox();
            this.numericUpDownScaleCollision = new System.Windows.Forms.NumericUpDown();
            this.numericUpDownScale = new System.Windows.Forms.NumericUpDown();
            this.buttonExportLinkMesh = new System.Windows.Forms.Button();
            this.checkBoxPerLinkMeshing = new System.Windows.Forms.CheckBox();
            this.buttonResetMeshingToDefaults = new System.Windows.Forms.Button();
            this.checkBoxCollisionMeshing = new System.Windows.Forms.CheckBox();
            this.numericUpDownAngularDeflectionCollision = new System.Windows.Forms.NumericUpDown();
            this.numericUpDownLinearDeflectionCollision = new System.Windows.Forms.NumericUpDown();
            this.numericUpDownAngularDeflection = new System.Windows.Forms.NumericUpDown();
            this.numericUpDownLinearDeflection = new System.Windows.Forms.NumericUpDown();
            this.groupBox6 = new System.Windows.Forms.GroupBox();
            this.domainUpDownAlpha = new System.Windows.Forms.DomainUpDown();
            this.label33 = new System.Windows.Forms.Label();
            this.domainUpDownRed = new System.Windows.Forms.DomainUpDown();
            this.label32 = new System.Windows.Forms.Label();
            this.domainUpDownGreen = new System.Windows.Forms.DomainUpDown();
            this.label31 = new System.Windows.Forms.Label();
            this.domainUpDownBlue = new System.Windows.Forms.DomainUpDown();
            this.label30 = new System.Windows.Forms.Label();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.label20 = new System.Windows.Forms.Label();
            this.label21 = new System.Windows.Forms.Label();
            this.label24 = new System.Windows.Forms.Label();
            this.label22 = new System.Windows.Forms.Label();
            this.label25 = new System.Windows.Forms.Label();
            this.label26 = new System.Windows.Forms.Label();
            this.textBoxVisualOriginZ = new System.Windows.Forms.TextBox();
            this.textBoxVisualOriginYaw = new System.Windows.Forms.TextBox();
            this.textBoxVisualOriginPitch = new System.Windows.Forms.TextBox();
            this.textBoxVisualOriginRoll = new System.Windows.Forms.TextBox();
            this.textBoxVisualOriginX = new System.Windows.Forms.TextBox();
            this.textBoxVisualOriginY = new System.Windows.Forms.TextBox();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.comboBoxFolderStructure = new System.Windows.Forms.ComboBox();
            this.tableLayoutPanel1 = new System.Windows.Forms.TableLayoutPanel();
            this.labelStep = new System.Windows.Forms.Label();
            this.labelObjCAD = new System.Windows.Forms.Label();
            this.labelGlbCAD = new System.Windows.Forms.Label();
            this.labelStlCAD = new System.Windows.Forms.Label();
            this.label40 = new System.Windows.Forms.Label();
            this.checkBoxStepVisual = new System.Windows.Forms.CheckBox();
            this.checkBoxObjCADVisual = new System.Windows.Forms.CheckBox();
            this.checkBoxStepCollision = new System.Windows.Forms.CheckBox();
            this.checkBoxObjCADCollision = new System.Windows.Forms.CheckBox();
            this.checkBoxGlbCADVisual = new System.Windows.Forms.CheckBox();
            this.checkBoxGlbCADCollision = new System.Windows.Forms.CheckBox();
            this.checkBoxStlCADVisual = new System.Windows.Forms.CheckBox();
            this.checkBoxStlCADCollision = new System.Windows.Forms.CheckBox();
            this.label23 = new System.Windows.Forms.Label();
            this.label38 = new System.Windows.Forms.Label();
            this.label39 = new System.Windows.Forms.Label();
            this.label37 = new System.Windows.Forms.Label();
            this.textBoxMeshFileExtension = new System.Windows.Forms.TextBox();
            this.textBoxMeshFileExtensionCollision = new System.Windows.Forms.TextBox();
            this.label41 = new System.Windows.Forms.Label();
            this.checkBoxObjOpenCascadeCollision = new System.Windows.Forms.CheckBox();
            this.checkBoxGlbOpenCascadeCollision = new System.Windows.Forms.CheckBox();
            this.checkBoxStlOpenCascadeCollision = new System.Windows.Forms.CheckBox();
            this.checkBoxObjOpenCascadeVisual = new System.Windows.Forms.CheckBox();
            this.checkBoxGlbOpenCascadeVisual = new System.Windows.Forms.CheckBox();
            this.checkBoxStlOpenCascadeVisual = new System.Windows.Forms.CheckBox();
            this.linkLabelHelpMeChoose = new System.Windows.Forms.LinkLabel();
            this.label19 = new System.Windows.Forms.Label();
            this.treeViewLinkProperties = new System.Windows.Forms.TreeView();
            this.tabPageJointProperties = new System.Windows.Forms.TabPage();
            this.checkBoxUseDegrees = new System.Windows.Forms.CheckBox();
            this.label35 = new System.Windows.Forms.Label();
            this.groupBox18 = new System.Windows.Forms.GroupBox();
            this.checkBoxShowJointHighlights = new System.Windows.Forms.CheckBox();
            this.label10 = new System.Windows.Forms.Label();
            this.checkBoxShowJointVisualization = new System.Windows.Forms.CheckBox();
            this.trackBarJointGizmoSize = new System.Windows.Forms.TrackBar();
            this.label7 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.label27 = new System.Windows.Forms.Label();
            this.treeViewJointTree = new System.Windows.Forms.TreeView();
            this.label69 = new System.Windows.Forms.Label();
            this.groupBox10 = new System.Windows.Forms.GroupBox();
            this.textBoxParentLink = new System.Windows.Forms.TextBox();
            this.textBoxChildLink = new System.Windows.Forms.TextBox();
            this.textBoxRefAxis = new System.Windows.Forms.TextBox();
            this.label62 = new System.Windows.Forms.Label();
            this.textBoxCoordSys = new System.Windows.Forms.TextBox();
            this.comboBoxJointType = new System.Windows.Forms.ComboBox();
            this.textBoxJointName = new System.Windows.Forms.TextBox();
            this.label63 = new System.Windows.Forms.Label();
            this.label64 = new System.Windows.Forms.Label();
            this.label65 = new System.Windows.Forms.Label();
            this.label67 = new System.Windows.Forms.Label();
            this.label66 = new System.Windows.Forms.Label();
            this.groupBoxJointOrigin = new System.Windows.Forms.GroupBox();
            this.textBoxJointY = new System.Windows.Forms.TextBox();
            this.label57 = new System.Windows.Forms.Label();
            this.textBoxJointZ = new System.Windows.Forms.TextBox();
            this.label56 = new System.Windows.Forms.Label();
            this.label55 = new System.Windows.Forms.Label();
            this.label53 = new System.Windows.Forms.Label();
            this.textBoxJointRoll = new System.Windows.Forms.TextBox();
            this.label52 = new System.Windows.Forms.Label();
            this.textBoxJointPitch = new System.Windows.Forms.TextBox();
            this.textBoxJointYaw = new System.Windows.Forms.TextBox();
            this.label51 = new System.Windows.Forms.Label();
            this.textBoxJointX = new System.Windows.Forms.TextBox();
            this.groupBox12 = new System.Windows.Forms.GroupBox();
            this.textBoxAxisZ = new System.Windows.Forms.TextBox();
            this.label61 = new System.Windows.Forms.Label();
            this.label59 = new System.Windows.Forms.Label();
            this.textBoxAxisY = new System.Windows.Forms.TextBox();
            this.label58 = new System.Windows.Forms.Label();
            this.textBoxAxisX = new System.Windows.Forms.TextBox();
            this.groupBox13 = new System.Windows.Forms.GroupBox();
            this.numericUpDownLimitUpper = new System.Windows.Forms.NumericUpDown();
            this.numericUpDownLimitLower = new System.Windows.Forms.NumericUpDown();
            this.labelEffort = new System.Windows.Forms.Label();
            this.textBoxLimitEffort = new System.Windows.Forms.TextBox();
            this.labelLimitUpper = new System.Windows.Forms.Label();
            this.labelLowerLimit = new System.Windows.Forms.Label();
            this.labelVelocity = new System.Windows.Forms.Label();
            this.textBoxLimitVelocity = new System.Windows.Forms.TextBox();
            this.groupBox14 = new System.Windows.Forms.GroupBox();
            this.textBoxFriction = new System.Windows.Forms.TextBox();
            this.labelDamping = new System.Windows.Forms.Label();
            this.textBoxDamping = new System.Windows.Forms.TextBox();
            this.labelFriction = new System.Windows.Forms.Label();
            this.tabControl = new System.Windows.Forms.TabControl();
            this.tabPageKinematicsSummary = new System.Windows.Forms.TabPage();
            this.checkBoxSummaryDegrees = new System.Windows.Forms.CheckBox();
            this.dataGridViewKinematicsSummary = new System.Windows.Forms.DataGridView();
            this.label43 = new System.Windows.Forms.Label();
            this.label44 = new System.Windows.Forms.Label();
            this.tabPageTendonsSummary = new System.Windows.Forms.TabPage();
            this.groupBox15 = new System.Windows.Forms.GroupBox();
            this.checkBoxShowAllTendons = new System.Windows.Forms.CheckBox();
            this.checkBoxShowTendonVisualization = new System.Windows.Forms.CheckBox();
            this.dataGridViewTendonsSummary = new System.Windows.Forms.DataGridView();
            this.label54 = new System.Windows.Forms.Label();
            this.label68 = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownTargetEdgeLength)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownTargetEdgeLengthCollision)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownEdgeLengthFraction)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownEdgeLengthFractionCollision)).BeginInit();
            this.tabPageLinkProperties.SuspendLayout();
            this.groupBox11.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLinkGizmoSize)).BeginInit();
            this.groupBox5.SuspendLayout();
            this.groupBox9.SuspendLayout();
            this.groupBox8.SuspendLayout();
            this.groupBox4.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.groupBox7.SuspendLayout();
            this.groupBox16.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownScaleCollision)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownScale)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownAngularDeflectionCollision)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownLinearDeflectionCollision)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownAngularDeflection)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownLinearDeflection)).BeginInit();
            this.groupBox6.SuspendLayout();
            this.groupBox3.SuspendLayout();
            this.groupBox1.SuspendLayout();
            this.tableLayoutPanel1.SuspendLayout();
            this.tabPageJointProperties.SuspendLayout();
            this.groupBox18.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarJointGizmoSize)).BeginInit();
            this.groupBox10.SuspendLayout();
            this.groupBoxJointOrigin.SuspendLayout();
            this.groupBox12.SuspendLayout();
            this.groupBox13.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownLimitUpper)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownLimitLower)).BeginInit();
            this.groupBox14.SuspendLayout();
            this.tabControl.SuspendLayout();
            this.tabPageKinematicsSummary.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.dataGridViewKinematicsSummary)).BeginInit();
            this.tabPageTendonsSummary.SuspendLayout();
            this.groupBox15.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.dataGridViewTendonsSummary)).BeginInit();
            this.SuspendLayout();
            // 
            // buttonLinksFinish
            // 
            this.buttonLinksFinish.Enabled = false;
            this.buttonLinksFinish.Location = new System.Drawing.Point(1242, 1104);
            this.buttonLinksFinish.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.buttonLinksFinish.Name = "buttonLinksFinish";
            this.buttonLinksFinish.Size = new System.Drawing.Size(225, 35);
            this.buttonLinksFinish.TabIndex = 100;
            this.buttonLinksFinish.Text = "Export Robot and Meshes...";
            this.toolTips.SetToolTip(this.buttonLinksFinish, "Exports .superdex_bot, .urdf and mjcf .xml and kick of the meshing process");
            this.buttonLinksFinish.UseVisualStyleBackColor = true;
            // 
            // buttonLinksExportUrdfOnly
            // 
            this.buttonLinksExportUrdfOnly.Enabled = false;
            this.buttonLinksExportUrdfOnly.Location = new System.Drawing.Point(1010, 1104);
            this.buttonLinksExportUrdfOnly.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.buttonLinksExportUrdfOnly.Name = "buttonLinksExportUrdfOnly";
            this.buttonLinksExportUrdfOnly.Size = new System.Drawing.Size(225, 35);
            this.buttonLinksExportUrdfOnly.TabIndex = 103;
            this.buttonLinksExportUrdfOnly.Text = "Export Robot Only...";
            this.toolTips.SetToolTip(this.buttonLinksExportUrdfOnly, "Exports .superdex_bot, .urdf and mjcf .xmx files only");
            this.buttonLinksExportUrdfOnly.UseVisualStyleBackColor = true;
            // 
            // buttonPrevious
            // 
            this.buttonPrevious.Enabled = false;
            this.buttonPrevious.Location = new System.Drawing.Point(484, 1104);
            this.buttonPrevious.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.buttonPrevious.Name = "buttonPrevious";
            this.buttonPrevious.Size = new System.Drawing.Size(112, 35);
            this.buttonPrevious.TabIndex = 31;
            this.buttonPrevious.Text = "Previous";
            this.buttonPrevious.UseVisualStyleBackColor = true;
            // 
            // label60
            // 
            this.label60.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.label60.AutoSize = true;
            this.label60.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label60.Location = new System.Drawing.Point(179, 335);
            this.label60.Name = "label60";
            this.label60.Padding = new System.Windows.Forms.Padding(0, 5, 0, 0);
            this.label60.Size = new System.Drawing.Size(138, 39);
            this.label60.TabIndex = 79;
            this.label60.Text = "<mesh> tag extension";
            this.label60.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            this.toolTips.SetToolTip(this.label60, "Since we support multiple exports at the same time, we need to choose which exten" +
        "sion the <mesh> tag will use.");
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.Location = new System.Drawing.Point(21, 76);
            this.label1.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(172, 24);
            this.label1.TabIndex = 50;
            this.label1.Text = "Linear Deflection (mm)";
            this.toolTips.SetToolTip(this.label1, "Allowed linear deflection (mm) for meshing (default: 0.1), see https://fburl.com/" +
        "occt_meshing");
            this.label1.UseCompatibleTextRendering = true;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label3.Location = new System.Drawing.Point(21, 119);
            this.label3.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(180, 24);
            this.label3.TabIndex = 52;
            this.label3.Text = "Angular Deflection (rad)";
            this.toolTips.SetToolTip(this.label3, "Allowed angular deflection (radians) for meshing (default: 0.5), see https://fbur" +
        "l.com/occt_meshing");
            this.label3.UseCompatibleTextRendering = true;
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label6.Location = new System.Drawing.Point(10, 42);
            this.label6.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(129, 24);
            this.label6.TabIndex = 56;
            this.label6.Text = "Scale (multiplier)";
            this.toolTips.SetToolTip(this.label6, "Scales the mesh by this as a multiplier (for OBJ for Unreal Engine use 100)");
            this.label6.UseCompatibleTextRendering = true;
            // 
            // label28
            // 
            this.label28.AutoSize = true;
            this.label28.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label28.Location = new System.Drawing.Point(307, 38);
            this.label28.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label28.Name = "label28";
            this.label28.Size = new System.Drawing.Size(51, 24);
            this.label28.TabIndex = 58;
            this.label28.Text = "Visual";
            this.label28.TextAlign = System.Drawing.ContentAlignment.TopRight;
            this.toolTips.SetToolTip(this.label28, "Allowed linear deflection (mm) for meshing (default: 0.1), see https://fburl.com/" +
        "occt_meshing");
            this.label28.UseCompatibleTextRendering = true;
            // 
            // label34
            // 
            this.label34.AutoSize = true;
            this.label34.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label34.Location = new System.Drawing.Point(16, 36);
            this.label34.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label34.Name = "label34";
            this.label34.Size = new System.Drawing.Size(104, 24);
            this.label34.TabIndex = 51;
            this.label34.Text = "Select Preset";
            this.label34.UseCompatibleTextRendering = true;
            // 
            // label42
            // 
            this.label42.AutoSize = true;
            this.label42.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label42.Location = new System.Drawing.Point(11, 441);
            this.label42.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label42.Name = "label42";
            this.label42.Size = new System.Drawing.Size(124, 24);
            this.label42.TabIndex = 52;
            this.label42.Text = "Folder Structure";
            this.toolTips.SetToolTip(this.label42, "Folder structure for meshes on export");
            this.label42.UseCompatibleTextRendering = true;
            // 
            // labelBackend
            // 
            this.labelBackend.AutoSize = true;
            this.labelBackend.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelBackend.Location = new System.Drawing.Point(10, 82);
            this.labelBackend.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelBackend.Name = "labelBackend";
            this.labelBackend.Size = new System.Drawing.Size(61, 24);
            this.labelBackend.TabIndex = 120;
            this.labelBackend.Text = "Mesher";
            this.toolTips.SetToolTip(this.labelBackend, "Isotropic = CGAL per-face remesher, targeting an even triangle size. Delabella = " +
        "OpenCascade\'s stock mesher, driven only by the deflections above.");
            this.labelBackend.UseCompatibleTextRendering = true;
            // 
            // labelEdgeSampling
            // 
            this.labelEdgeSampling.AutoSize = true;
            this.labelEdgeSampling.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelEdgeSampling.Location = new System.Drawing.Point(10, 122);
            this.labelEdgeSampling.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelEdgeSampling.Name = "labelEdgeSampling";
            this.labelEdgeSampling.Size = new System.Drawing.Size(118, 24);
            this.labelEdgeSampling.TabIndex = 123;
            this.labelEdgeSampling.Text = "Edge Sampling";
            this.toolTips.SetToolTip(this.labelEdgeSampling, "How face boundary curves are sampled. Both space samples evenly; Adaptive adds mo" +
        "re wherever the deflection tolerance demands it, which resolves small curved fea" +
        "tures.");
            this.labelEdgeSampling.UseCompatibleTextRendering = true;
            // 
            // labelTargetEdgeLength
            // 
            this.labelTargetEdgeLength.AutoSize = true;
            this.labelTargetEdgeLength.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelTargetEdgeLength.Location = new System.Drawing.Point(10, 162);
            this.labelTargetEdgeLength.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelTargetEdgeLength.Name = "labelTargetEdgeLength";
            this.labelTargetEdgeLength.Size = new System.Drawing.Size(197, 24);
            this.labelTargetEdgeLength.TabIndex = 126;
            this.labelTargetEdgeLength.Text = "Edge Length mm (0=auto)";
            this.toolTips.SetToolTip(this.labelTargetEdgeLength, "Desired triangle edge length in mm. Leave at 0 to derive it from the bounding-box" +
        " fraction below.");
            this.labelTargetEdgeLength.UseCompatibleTextRendering = true;
            // 
            // labelEdgeLengthFraction
            // 
            this.labelEdgeLengthFraction.AutoSize = true;
            this.labelEdgeLengthFraction.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelEdgeLengthFraction.Location = new System.Drawing.Point(10, 202);
            this.labelEdgeLengthFraction.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelEdgeLengthFraction.Name = "labelEdgeLengthFraction";
            this.labelEdgeLengthFraction.Size = new System.Drawing.Size(163, 24);
            this.labelEdgeLengthFraction.TabIndex = 129;
            this.labelEdgeLengthFraction.Text = "Edge Length Fraction";
            this.toolTips.SetToolTip(this.labelEdgeLengthFraction, "Used when the edge length above is 0: the target becomes this fraction of the mod" +
        "el\'s bounding-box diagonal, so parts of different sizes get comparable triangles" +
        ".");
            this.labelEdgeLengthFraction.UseCompatibleTextRendering = true;
            // 
            // comboBoxBackend
            // 
            this.comboBoxBackend.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxBackend.FormattingEnabled = true;
            this.comboBoxBackend.Items.AddRange(new object[] {
            "Isotropic",
            "Delabella"});
            this.comboBoxBackend.Location = new System.Drawing.Point(227, 76);
            this.comboBoxBackend.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.comboBoxBackend.Name = "comboBoxBackend";
            this.comboBoxBackend.Size = new System.Drawing.Size(120, 33);
            this.comboBoxBackend.TabIndex = 121;
            // 
            // comboBoxBackendCollision
            // 
            this.comboBoxBackendCollision.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxBackendCollision.FormattingEnabled = true;
            this.comboBoxBackendCollision.Items.AddRange(new object[] {
            "Isotropic",
            "Delabella"});
            this.comboBoxBackendCollision.Location = new System.Drawing.Point(367, 76);
            this.comboBoxBackendCollision.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.comboBoxBackendCollision.Name = "comboBoxBackendCollision";
            this.comboBoxBackendCollision.Size = new System.Drawing.Size(120, 33);
            this.comboBoxBackendCollision.TabIndex = 122;
            // 
            // comboBoxEdgeSampling
            // 
            this.comboBoxEdgeSampling.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxEdgeSampling.FormattingEnabled = true;
            this.comboBoxEdgeSampling.Items.AddRange(new object[] {
            "Adaptive",
            "Uniform"});
            this.comboBoxEdgeSampling.Location = new System.Drawing.Point(227, 118);
            this.comboBoxEdgeSampling.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.comboBoxEdgeSampling.Name = "comboBoxEdgeSampling";
            this.comboBoxEdgeSampling.Size = new System.Drawing.Size(120, 33);
            this.comboBoxEdgeSampling.TabIndex = 124;
            // 
            // comboBoxEdgeSamplingCollision
            // 
            this.comboBoxEdgeSamplingCollision.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxEdgeSamplingCollision.FormattingEnabled = true;
            this.comboBoxEdgeSamplingCollision.Items.AddRange(new object[] {
            "Adaptive",
            "Uniform"});
            this.comboBoxEdgeSamplingCollision.Location = new System.Drawing.Point(367, 118);
            this.comboBoxEdgeSamplingCollision.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.comboBoxEdgeSamplingCollision.Name = "comboBoxEdgeSamplingCollision";
            this.comboBoxEdgeSamplingCollision.Size = new System.Drawing.Size(120, 33);
            this.comboBoxEdgeSamplingCollision.TabIndex = 125;
            // 
            // numericUpDownTargetEdgeLength
            // 
            this.numericUpDownTargetEdgeLength.DecimalPlaces = 3;
            this.numericUpDownTargetEdgeLength.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.numericUpDownTargetEdgeLength.Increment = new decimal(new int[] {
            1,
            0,
            0,
            65536});
            this.numericUpDownTargetEdgeLength.Location = new System.Drawing.Point(227, 160);
            this.numericUpDownTargetEdgeLength.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.numericUpDownTargetEdgeLength.Maximum = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            this.numericUpDownTargetEdgeLength.Name = "numericUpDownTargetEdgeLength";
            this.numericUpDownTargetEdgeLength.Size = new System.Drawing.Size(120, 29);
            this.numericUpDownTargetEdgeLength.TabIndex = 127;
            // 
            // numericUpDownTargetEdgeLengthCollision
            // 
            this.numericUpDownTargetEdgeLengthCollision.DecimalPlaces = 3;
            this.numericUpDownTargetEdgeLengthCollision.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.numericUpDownTargetEdgeLengthCollision.Increment = new decimal(new int[] {
            1,
            0,
            0,
            65536});
            this.numericUpDownTargetEdgeLengthCollision.Location = new System.Drawing.Point(367, 160);
            this.numericUpDownTargetEdgeLengthCollision.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.numericUpDownTargetEdgeLengthCollision.Maximum = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            this.numericUpDownTargetEdgeLengthCollision.Name = "numericUpDownTargetEdgeLengthCollision";
            this.numericUpDownTargetEdgeLengthCollision.Size = new System.Drawing.Size(120, 29);
            this.numericUpDownTargetEdgeLengthCollision.TabIndex = 128;
            // 
            // numericUpDownEdgeLengthFraction
            // 
            this.numericUpDownEdgeLengthFraction.DecimalPlaces = 4;
            this.numericUpDownEdgeLengthFraction.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.numericUpDownEdgeLengthFraction.Increment = new decimal(new int[] {
            5,
            0,
            0,
            196608});
            this.numericUpDownEdgeLengthFraction.Location = new System.Drawing.Point(227, 198);
            this.numericUpDownEdgeLengthFraction.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.numericUpDownEdgeLengthFraction.Maximum = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            this.numericUpDownEdgeLengthFraction.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            196608});
            this.numericUpDownEdgeLengthFraction.Name = "numericUpDownEdgeLengthFraction";
            this.numericUpDownEdgeLengthFraction.Size = new System.Drawing.Size(120, 29);
            this.numericUpDownEdgeLengthFraction.TabIndex = 130;
            this.numericUpDownEdgeLengthFraction.Value = new decimal(new int[] {
            2,
            0,
            0,
            131072});
            // 
            // numericUpDownEdgeLengthFractionCollision
            // 
            this.numericUpDownEdgeLengthFractionCollision.DecimalPlaces = 4;
            this.numericUpDownEdgeLengthFractionCollision.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.numericUpDownEdgeLengthFractionCollision.Increment = new decimal(new int[] {
            5,
            0,
            0,
            196608});
            this.numericUpDownEdgeLengthFractionCollision.Location = new System.Drawing.Point(367, 198);
            this.numericUpDownEdgeLengthFractionCollision.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.numericUpDownEdgeLengthFractionCollision.Maximum = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            this.numericUpDownEdgeLengthFractionCollision.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            196608});
            this.numericUpDownEdgeLengthFractionCollision.Name = "numericUpDownEdgeLengthFractionCollision";
            this.numericUpDownEdgeLengthFractionCollision.Size = new System.Drawing.Size(120, 29);
            this.numericUpDownEdgeLengthFractionCollision.TabIndex = 131;
            this.numericUpDownEdgeLengthFractionCollision.Value = new decimal(new int[] {
            2,
            0,
            0,
            131072});
            // 
            // buttonNext
            // 
            this.buttonNext.Location = new System.Drawing.Point(604, 1104);
            this.buttonNext.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.buttonNext.Name = "buttonNext";
            this.buttonNext.Size = new System.Drawing.Size(112, 35);
            this.buttonNext.TabIndex = 30;
            this.buttonNext.Text = "Next";
            this.buttonNext.UseVisualStyleBackColor = true;
            // 
            // buttonClose
            // 
            this.buttonClose.Location = new System.Drawing.Point(1475, 1104);
            this.buttonClose.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.buttonClose.Name = "buttonClose";
            this.buttonClose.Size = new System.Drawing.Size(112, 35);
            this.buttonClose.TabIndex = 32;
            this.buttonClose.Text = "Close";
            this.buttonClose.UseVisualStyleBackColor = true;
            // 
            // label29
            // 
            this.label29.AutoSize = true;
            this.label29.Location = new System.Drawing.Point(754, 22);
            this.label29.Name = "label29";
            this.label29.Size = new System.Drawing.Size(99, 20);
            this.label29.TabIndex = 108;
            this.label29.Text = "Robot Name";
            // 
            // textBoxRobotName
            // 
            this.textBoxRobotName.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxRobotName.Location = new System.Drawing.Point(860, 19);
            this.textBoxRobotName.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxRobotName.Name = "textBoxRobotName";
            this.textBoxRobotName.Size = new System.Drawing.Size(727, 29);
            this.textBoxRobotName.TabIndex = 109;
            this.textBoxRobotName.Text = "default";
            // 
            // tabPageLinkProperties
            // 
            this.tabPageLinkProperties.Controls.Add(this.groupBox11);
            this.tabPageLinkProperties.Controls.Add(this.label2);
            this.tabPageLinkProperties.Controls.Add(this.label5);
            this.tabPageLinkProperties.Controls.Add(this.groupBox5);
            this.tabPageLinkProperties.Controls.Add(this.groupBox4);
            this.tabPageLinkProperties.Controls.Add(this.treeViewLinkProperties);
            this.tabPageLinkProperties.Location = new System.Drawing.Point(4, 30);
            this.tabPageLinkProperties.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.tabPageLinkProperties.Name = "tabPageLinkProperties";
            this.tabPageLinkProperties.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.tabPageLinkProperties.Size = new System.Drawing.Size(1579, 1007);
            this.tabPageLinkProperties.TabIndex = 1;
            this.tabPageLinkProperties.Text = "Link Properties";
            this.tabPageLinkProperties.UseVisualStyleBackColor = true;
            // 
            // groupBox11
            // 
            this.groupBox11.Controls.Add(this.label70);
            this.groupBox11.Controls.Add(this.trackBarLinkGizmoSize);
            this.groupBox11.Controls.Add(this.checkBoxLinkHighlights);
            this.groupBox11.Controls.Add(this.checkBoxShowLinkVisualization);
            this.groupBox11.Font = new System.Drawing.Font("Segoe UI Variable Display", 9F, System.Drawing.FontStyle.Bold);
            this.groupBox11.Location = new System.Drawing.Point(14, 872);
            this.groupBox11.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox11.Name = "groupBox11";
            this.groupBox11.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox11.Size = new System.Drawing.Size(686, 115);
            this.groupBox11.TabIndex = 312;
            this.groupBox11.TabStop = false;
            this.groupBox11.Text = "Visualization";
            // 
            // label70
            // 
            this.label70.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label70.Location = new System.Drawing.Point(351, 58);
            this.label70.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label70.Name = "label70";
            this.label70.Size = new System.Drawing.Size(122, 24);
            this.label70.TabIndex = 320;
            this.label70.Text = "Gizmo Size";
            this.label70.TextAlign = System.Drawing.ContentAlignment.TopRight;
            this.label70.UseCompatibleTextRendering = true;
            // 
            // trackBarLinkGizmoSize
            // 
            this.trackBarLinkGizmoSize.AutoSize = false;
            this.trackBarLinkGizmoSize.Location = new System.Drawing.Point(480, 39);
            this.trackBarLinkGizmoSize.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.trackBarLinkGizmoSize.Maximum = 20;
            this.trackBarLinkGizmoSize.Minimum = 1;
            this.trackBarLinkGizmoSize.Name = "trackBarLinkGizmoSize";
            this.trackBarLinkGizmoSize.Size = new System.Drawing.Size(198, 46);
            this.trackBarLinkGizmoSize.TabIndex = 319;
            this.trackBarLinkGizmoSize.Value = 10;
            // 
            // checkBoxLinkHighlights
            // 
            this.checkBoxLinkHighlights.AutoSize = true;
            this.checkBoxLinkHighlights.Checked = true;
            this.checkBoxLinkHighlights.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkBoxLinkHighlights.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxLinkHighlights.Location = new System.Drawing.Point(15, 35);
            this.checkBoxLinkHighlights.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxLinkHighlights.Name = "checkBoxLinkHighlights";
            this.checkBoxLinkHighlights.Size = new System.Drawing.Size(217, 24);
            this.checkBoxLinkHighlights.TabIndex = 318;
            this.checkBoxLinkHighlights.Text = "Highlight Selected Bodies";
            this.checkBoxLinkHighlights.UseVisualStyleBackColor = true;
            // 
            // checkBoxShowLinkVisualization
            // 
            this.checkBoxShowLinkVisualization.AutoSize = true;
            this.checkBoxShowLinkVisualization.Checked = true;
            this.checkBoxShowLinkVisualization.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkBoxShowLinkVisualization.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxShowLinkVisualization.Location = new System.Drawing.Point(15, 70);
            this.checkBoxShowLinkVisualization.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxShowLinkVisualization.Name = "checkBoxShowLinkVisualization";
            this.checkBoxShowLinkVisualization.Size = new System.Drawing.Size(267, 24);
            this.checkBoxShowLinkVisualization.TabIndex = 317;
            this.checkBoxShowLinkVisualization.Text = "Show Inertia and Center of Mass";
            this.checkBoxShowLinkVisualization.UseVisualStyleBackColor = true;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.Location = new System.Drawing.Point(10, 45);
            this.label2.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(431, 24);
            this.label2.TabIndex = 102;
            this.label2.Text = "Use this page to make any changes to the links\' properties";
            this.label2.UseCompatibleTextRendering = true;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Font = new System.Drawing.Font("Microsoft Sans Serif", 10F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label5.Location = new System.Drawing.Point(9, 18);
            this.label5.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(249, 29);
            this.label5.TabIndex = 101;
            this.label5.Text = "Configure Link Properties";
            this.label5.UseCompatibleTextRendering = true;
            // 
            // groupBox5
            // 
            this.groupBox5.Controls.Add(this.buttonRecalculateInertial);
            this.groupBox5.Controls.Add(this.groupBox9);
            this.groupBox5.Controls.Add(this.groupBox8);
            this.groupBox5.Controls.Add(this.label15);
            this.groupBox5.Controls.Add(this.label12);
            this.groupBox5.Controls.Add(this.textBoxMass);
            this.groupBox5.Font = new System.Drawing.Font("Segoe UI Variable Display", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.groupBox5.Location = new System.Drawing.Point(708, 18);
            this.groupBox5.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.groupBox5.Name = "groupBox5";
            this.groupBox5.Padding = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.groupBox5.Size = new System.Drawing.Size(858, 258);
            this.groupBox5.TabIndex = 95;
            this.groupBox5.TabStop = false;
            // 
            // buttonRecalculateInertial
            // 
            this.buttonRecalculateInertial.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F);
            this.buttonRecalculateInertial.Location = new System.Drawing.Point(721, 212);
            this.buttonRecalculateInertial.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.buttonRecalculateInertial.Name = "buttonRecalculateInertial";
            this.buttonRecalculateInertial.Size = new System.Drawing.Size(118, 31);
            this.buttonRecalculateInertial.TabIndex = 36;
            this.buttonRecalculateInertial.Text = "Recalculate";
            this.buttonRecalculateInertial.UseVisualStyleBackColor = true;
            // 
            // groupBox9
            // 
            this.groupBox9.Controls.Add(this.textBoxIxx);
            this.groupBox9.Controls.Add(this.label49);
            this.groupBox9.Controls.Add(this.label48);
            this.groupBox9.Controls.Add(this.textBoxIxy);
            this.groupBox9.Controls.Add(this.textBoxIyz);
            this.groupBox9.Controls.Add(this.textBoxIxz);
            this.groupBox9.Controls.Add(this.textBoxIzz);
            this.groupBox9.Controls.Add(this.label18);
            this.groupBox9.Controls.Add(this.label11);
            this.groupBox9.Controls.Add(this.label50);
            this.groupBox9.Controls.Add(this.label14);
            this.groupBox9.Controls.Add(this.textBoxIyy);
            this.groupBox9.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.groupBox9.Location = new System.Drawing.Point(327, 32);
            this.groupBox9.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox9.Name = "groupBox9";
            this.groupBox9.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox9.Size = new System.Drawing.Size(515, 172);
            this.groupBox9.TabIndex = 35;
            this.groupBox9.TabStop = false;
            this.groupBox9.Text = "Moment of Inertia (Kg * m^2)";
            // 
            // textBoxIxx
            // 
            this.textBoxIxx.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxIxx.Location = new System.Drawing.Point(58, 38);
            this.textBoxIxx.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxIxx.Name = "textBoxIxx";
            this.textBoxIxx.Size = new System.Drawing.Size(97, 29);
            this.textBoxIxx.TabIndex = 8;
            // 
            // label49
            // 
            this.label49.AutoSize = true;
            this.label49.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label49.Location = new System.Drawing.Point(163, 81);
            this.label49.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label49.Name = "label49";
            this.label49.Size = new System.Drawing.Size(26, 20);
            this.label49.TabIndex = 17;
            this.label49.Text = "iyy";
            // 
            // label48
            // 
            this.label48.AutoSize = true;
            this.label48.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label48.Location = new System.Drawing.Point(318, 81);
            this.label48.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label48.Name = "label48";
            this.label48.Size = new System.Drawing.Size(27, 20);
            this.label48.TabIndex = 18;
            this.label48.Text = "iyz";
            // 
            // textBoxIxy
            // 
            this.textBoxIxy.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxIxy.Location = new System.Drawing.Point(202, 38);
            this.textBoxIxy.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxIxy.Name = "textBoxIxy";
            this.textBoxIxy.Size = new System.Drawing.Size(97, 29);
            this.textBoxIxy.TabIndex = 9;
            // 
            // textBoxIyz
            // 
            this.textBoxIyz.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxIyz.Location = new System.Drawing.Point(357, 78);
            this.textBoxIyz.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxIyz.Name = "textBoxIyz";
            this.textBoxIyz.Size = new System.Drawing.Size(97, 29);
            this.textBoxIyz.TabIndex = 12;
            // 
            // textBoxIxz
            // 
            this.textBoxIxz.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxIxz.Location = new System.Drawing.Point(357, 38);
            this.textBoxIxz.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxIxz.Name = "textBoxIxz";
            this.textBoxIxz.Size = new System.Drawing.Size(97, 29);
            this.textBoxIxz.TabIndex = 10;
            // 
            // textBoxIzz
            // 
            this.textBoxIzz.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxIzz.Location = new System.Drawing.Point(357, 118);
            this.textBoxIzz.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxIzz.Name = "textBoxIzz";
            this.textBoxIzz.Size = new System.Drawing.Size(97, 29);
            this.textBoxIzz.TabIndex = 13;
            // 
            // label18
            // 
            this.label18.AutoSize = true;
            this.label18.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label18.Location = new System.Drawing.Point(320, 121);
            this.label18.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label18.Name = "label18";
            this.label18.Size = new System.Drawing.Size(28, 20);
            this.label18.TabIndex = 19;
            this.label18.Text = "izz";
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label11.Location = new System.Drawing.Point(320, 41);
            this.label11.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(27, 20);
            this.label11.TabIndex = 16;
            this.label11.Text = "ixz";
            // 
            // label50
            // 
            this.label50.AutoSize = true;
            this.label50.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label50.Location = new System.Drawing.Point(164, 41);
            this.label50.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label50.Name = "label50";
            this.label50.Size = new System.Drawing.Size(26, 20);
            this.label50.TabIndex = 15;
            this.label50.Text = "ixy";
            // 
            // label14
            // 
            this.label14.AutoSize = true;
            this.label14.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label14.Location = new System.Drawing.Point(21, 41);
            this.label14.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(26, 20);
            this.label14.TabIndex = 14;
            this.label14.Text = "ixx";
            // 
            // textBoxIyy
            // 
            this.textBoxIyy.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxIyy.Location = new System.Drawing.Point(202, 78);
            this.textBoxIyy.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxIyy.Name = "textBoxIyy";
            this.textBoxIyy.Size = new System.Drawing.Size(97, 29);
            this.textBoxIyy.TabIndex = 11;
            // 
            // groupBox8
            // 
            this.groupBox8.Controls.Add(this.textBoxInertialOriginX);
            this.groupBox8.Controls.Add(this.label16);
            this.groupBox8.Controls.Add(this.textBoxInertialOriginYaw);
            this.groupBox8.Controls.Add(this.textBoxInertialOriginPitch);
            this.groupBox8.Controls.Add(this.label13);
            this.groupBox8.Controls.Add(this.textBoxInertialOriginRoll);
            this.groupBox8.Controls.Add(this.textBoxInertialOriginY);
            this.groupBox8.Controls.Add(this.label17);
            this.groupBox8.Controls.Add(this.label45);
            this.groupBox8.Controls.Add(this.label46);
            this.groupBox8.Controls.Add(this.textBoxInertialOriginZ);
            this.groupBox8.Controls.Add(this.label47);
            this.groupBox8.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.groupBox8.Location = new System.Drawing.Point(14, 32);
            this.groupBox8.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox8.Name = "groupBox8";
            this.groupBox8.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox8.Size = new System.Drawing.Size(304, 171);
            this.groupBox8.TabIndex = 34;
            this.groupBox8.TabStop = false;
            this.groupBox8.Text = "Origin (m, rad)";
            // 
            // textBoxInertialOriginX
            // 
            this.textBoxInertialOriginX.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxInertialOriginX.Location = new System.Drawing.Point(39, 36);
            this.textBoxInertialOriginX.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxInertialOriginX.Name = "textBoxInertialOriginX";
            this.textBoxInertialOriginX.Size = new System.Drawing.Size(97, 29);
            this.textBoxInertialOriginX.TabIndex = 1;
            // 
            // label16
            // 
            this.label16.AutoSize = true;
            this.label16.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label16.Location = new System.Drawing.Point(153, 38);
            this.label16.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label16.Name = "label16";
            this.label16.Size = new System.Drawing.Size(36, 20);
            this.label16.TabIndex = 39;
            this.label16.Text = "Roll";
            this.label16.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // textBoxInertialOriginYaw
            // 
            this.textBoxInertialOriginYaw.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxInertialOriginYaw.Location = new System.Drawing.Point(195, 112);
            this.textBoxInertialOriginYaw.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxInertialOriginYaw.Name = "textBoxInertialOriginYaw";
            this.textBoxInertialOriginYaw.Size = new System.Drawing.Size(97, 29);
            this.textBoxInertialOriginYaw.TabIndex = 6;
            // 
            // textBoxInertialOriginPitch
            // 
            this.textBoxInertialOriginPitch.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxInertialOriginPitch.Location = new System.Drawing.Point(195, 72);
            this.textBoxInertialOriginPitch.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxInertialOriginPitch.Name = "textBoxInertialOriginPitch";
            this.textBoxInertialOriginPitch.Size = new System.Drawing.Size(97, 29);
            this.textBoxInertialOriginPitch.TabIndex = 5;
            // 
            // label13
            // 
            this.label13.AutoSize = true;
            this.label13.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label13.Location = new System.Drawing.Point(14, 41);
            this.label13.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(20, 20);
            this.label13.TabIndex = 36;
            this.label13.Text = "X";
            // 
            // textBoxInertialOriginRoll
            // 
            this.textBoxInertialOriginRoll.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxInertialOriginRoll.Location = new System.Drawing.Point(195, 32);
            this.textBoxInertialOriginRoll.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxInertialOriginRoll.Name = "textBoxInertialOriginRoll";
            this.textBoxInertialOriginRoll.Size = new System.Drawing.Size(97, 29);
            this.textBoxInertialOriginRoll.TabIndex = 4;
            // 
            // textBoxInertialOriginY
            // 
            this.textBoxInertialOriginY.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxInertialOriginY.Location = new System.Drawing.Point(39, 76);
            this.textBoxInertialOriginY.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxInertialOriginY.Name = "textBoxInertialOriginY";
            this.textBoxInertialOriginY.Size = new System.Drawing.Size(97, 29);
            this.textBoxInertialOriginY.TabIndex = 2;
            // 
            // label17
            // 
            this.label17.AutoSize = true;
            this.label17.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label17.Location = new System.Drawing.Point(14, 81);
            this.label17.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label17.Name = "label17";
            this.label17.Size = new System.Drawing.Size(20, 20);
            this.label17.TabIndex = 37;
            this.label17.Text = "Y";
            // 
            // label45
            // 
            this.label45.AutoSize = true;
            this.label45.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label45.Location = new System.Drawing.Point(145, 78);
            this.label45.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label45.Name = "label45";
            this.label45.Size = new System.Drawing.Size(44, 20);
            this.label45.TabIndex = 40;
            this.label45.Text = "Pitch";
            this.label45.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label46
            // 
            this.label46.AutoSize = true;
            this.label46.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label46.Location = new System.Drawing.Point(148, 118);
            this.label46.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label46.Name = "label46";
            this.label46.Size = new System.Drawing.Size(40, 20);
            this.label46.TabIndex = 41;
            this.label46.Text = "Yaw";
            this.label46.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // textBoxInertialOriginZ
            // 
            this.textBoxInertialOriginZ.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxInertialOriginZ.Location = new System.Drawing.Point(39, 116);
            this.textBoxInertialOriginZ.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxInertialOriginZ.Name = "textBoxInertialOriginZ";
            this.textBoxInertialOriginZ.Size = new System.Drawing.Size(97, 29);
            this.textBoxInertialOriginZ.TabIndex = 3;
            // 
            // label47
            // 
            this.label47.AutoSize = true;
            this.label47.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label47.Location = new System.Drawing.Point(14, 121);
            this.label47.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label47.Name = "label47";
            this.label47.Size = new System.Drawing.Size(19, 20);
            this.label47.TabIndex = 38;
            this.label47.Text = "Z";
            // 
            // label15
            // 
            this.label15.AutoSize = true;
            this.label15.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label15.Location = new System.Drawing.Point(9, 0);
            this.label15.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label15.Name = "label15";
            this.label15.Size = new System.Drawing.Size(175, 25);
            this.label15.TabIndex = 33;
            this.label15.Text = "Inertial Properties*";
            // 
            // label12
            // 
            this.label12.AutoSize = true;
            this.label12.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label12.Location = new System.Drawing.Point(119, 215);
            this.label12.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(78, 20);
            this.label12.TabIndex = 13;
            this.label12.Text = "Mass (kg)";
            // 
            // textBoxMass
            // 
            this.textBoxMass.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxMass.Location = new System.Drawing.Point(209, 212);
            this.textBoxMass.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxMass.Name = "textBoxMass";
            this.textBoxMass.Size = new System.Drawing.Size(97, 29);
            this.textBoxMass.TabIndex = 7;
            // 
            // groupBox4
            // 
            this.groupBox4.Controls.Add(this.groupBox2);
            this.groupBox4.Controls.Add(this.groupBox7);
            this.groupBox4.Controls.Add(this.groupBox6);
            this.groupBox4.Controls.Add(this.groupBox3);
            this.groupBox4.Controls.Add(this.groupBox1);
            this.groupBox4.Controls.Add(this.label19);
            this.groupBox4.Font = new System.Drawing.Font("Segoe UI Variable Display", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.groupBox4.Location = new System.Drawing.Point(708, 286);
            this.groupBox4.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.groupBox4.Name = "groupBox4";
            this.groupBox4.Padding = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.groupBox4.Size = new System.Drawing.Size(858, 698);
            this.groupBox4.TabIndex = 96;
            this.groupBox4.TabStop = false;
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.label34);
            this.groupBox2.Controls.Add(this.comboBoxExporterConfigurationPreset);
            this.groupBox2.Location = new System.Drawing.Point(340, 36);
            this.groupBox2.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox2.Size = new System.Drawing.Size(511, 75);
            this.groupBox2.TabIndex = 79;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Presets";
            // 
            // comboBoxExporterConfigurationPreset
            // 
            this.comboBoxExporterConfigurationPreset.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxExporterConfigurationPreset.FormattingEnabled = true;
            this.comboBoxExporterConfigurationPreset.Location = new System.Drawing.Point(127, 30);
            this.comboBoxExporterConfigurationPreset.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.comboBoxExporterConfigurationPreset.Name = "comboBoxExporterConfigurationPreset";
            this.comboBoxExporterConfigurationPreset.Size = new System.Drawing.Size(385, 32);
            this.comboBoxExporterConfigurationPreset.TabIndex = 0;
            // 
            // groupBox7
            // 
            this.groupBox7.Controls.Add(this.groupBox16);
            this.groupBox7.Controls.Add(this.buttonExportLinkMesh);
            this.groupBox7.Controls.Add(this.checkBoxPerLinkMeshing);
            this.groupBox7.Controls.Add(this.buttonResetMeshingToDefaults);
            this.groupBox7.Controls.Add(this.checkBoxCollisionMeshing);
            this.groupBox7.Controls.Add(this.numericUpDownAngularDeflectionCollision);
            this.groupBox7.Controls.Add(this.numericUpDownLinearDeflectionCollision);
            this.groupBox7.Controls.Add(this.label28);
            this.groupBox7.Controls.Add(this.numericUpDownAngularDeflection);
            this.groupBox7.Controls.Add(this.numericUpDownLinearDeflection);
            this.groupBox7.Controls.Add(this.label3);
            this.groupBox7.Controls.Add(this.label1);
            this.groupBox7.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.groupBox7.Location = new System.Drawing.Point(340, 193);
            this.groupBox7.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox7.Name = "groupBox7";
            this.groupBox7.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox7.Size = new System.Drawing.Size(513, 495);
            this.groupBox7.TabIndex = 78;
            this.groupBox7.TabStop = false;
            this.groupBox7.Text = "Meshing Options";
            // 
            // groupBox16
            // 
            this.groupBox16.Controls.Add(this.numericUpDownScaleCollision);
            this.groupBox16.Controls.Add(this.numericUpDownScale);
            this.groupBox16.Controls.Add(this.label6);
            this.groupBox16.Controls.Add(this.labelBackend);
            this.groupBox16.Controls.Add(this.comboBoxBackend);
            this.groupBox16.Controls.Add(this.comboBoxBackendCollision);
            this.groupBox16.Controls.Add(this.labelEdgeSampling);
            this.groupBox16.Controls.Add(this.comboBoxEdgeSampling);
            this.groupBox16.Controls.Add(this.comboBoxEdgeSamplingCollision);
            this.groupBox16.Controls.Add(this.labelTargetEdgeLength);
            this.groupBox16.Controls.Add(this.numericUpDownTargetEdgeLength);
            this.groupBox16.Controls.Add(this.numericUpDownTargetEdgeLengthCollision);
            this.groupBox16.Controls.Add(this.labelEdgeLengthFraction);
            this.groupBox16.Controls.Add(this.numericUpDownEdgeLengthFraction);
            this.groupBox16.Controls.Add(this.numericUpDownEdgeLengthFractionCollision);
            this.groupBox16.Location = new System.Drawing.Point(11, 200);
            this.groupBox16.Name = "groupBox16";
            this.groupBox16.Size = new System.Drawing.Size(495, 243);
            this.groupBox16.TabIndex = 132;
            this.groupBox16.TabStop = false;
            this.groupBox16.Text = "SuperDex Meshing Options";
            // 
            // numericUpDownScaleCollision
            // 
            this.numericUpDownScaleCollision.DecimalPlaces = 3;
            this.numericUpDownScaleCollision.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.numericUpDownScaleCollision.Increment = new decimal(new int[] {
            10,
            0,
            0,
            0});
            this.numericUpDownScaleCollision.Location = new System.Drawing.Point(367, 38);
            this.numericUpDownScaleCollision.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.numericUpDownScaleCollision.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numericUpDownScaleCollision.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            131072});
            this.numericUpDownScaleCollision.Name = "numericUpDownScaleCollision";
            this.numericUpDownScaleCollision.Size = new System.Drawing.Size(120, 29);
            this.numericUpDownScaleCollision.TabIndex = 61;
            this.numericUpDownScaleCollision.Value = new decimal(new int[] {
            10,
            0,
            0,
            65536});
            // 
            // numericUpDownScale
            // 
            this.numericUpDownScale.DecimalPlaces = 3;
            this.numericUpDownScale.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.numericUpDownScale.Increment = new decimal(new int[] {
            10,
            0,
            0,
            0});
            this.numericUpDownScale.Location = new System.Drawing.Point(227, 38);
            this.numericUpDownScale.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.numericUpDownScale.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numericUpDownScale.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            131072});
            this.numericUpDownScale.Name = "numericUpDownScale";
            this.numericUpDownScale.Size = new System.Drawing.Size(120, 29);
            this.numericUpDownScale.TabIndex = 57;
            this.numericUpDownScale.Value = new decimal(new int[] {
            10,
            0,
            0,
            65536});
            // 
            // buttonExportLinkMesh
            // 
            this.buttonExportLinkMesh.Location = new System.Drawing.Point(305, 454);
            this.buttonExportLinkMesh.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.buttonExportLinkMesh.Name = "buttonExportLinkMesh";
            this.buttonExportLinkMesh.Size = new System.Drawing.Size(201, 35);
            this.buttonExportLinkMesh.TabIndex = 108;
            this.buttonExportLinkMesh.Text = "Export Link Mesh...";
            this.toolTips.SetToolTip(this.buttonExportLinkMesh, "Export only the visual and collision meshes of the selected link");
            this.buttonExportLinkMesh.UseVisualStyleBackColor = true;
            // 
            // checkBoxPerLinkMeshing
            // 
            this.checkBoxPerLinkMeshing.AutoSize = true;
            this.checkBoxPerLinkMeshing.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxPerLinkMeshing.Location = new System.Drawing.Point(21, 37);
            this.checkBoxPerLinkMeshing.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxPerLinkMeshing.Name = "checkBoxPerLinkMeshing";
            this.checkBoxPerLinkMeshing.Size = new System.Drawing.Size(215, 24);
            this.checkBoxPerLinkMeshing.TabIndex = 106;
            this.checkBoxPerLinkMeshing.Text = "Per Link Meshing Options";
            this.checkBoxPerLinkMeshing.UseVisualStyleBackColor = true;
            // 
            // buttonResetMeshingToDefaults
            // 
            this.buttonResetMeshingToDefaults.Location = new System.Drawing.Point(12, 451);
            this.buttonResetMeshingToDefaults.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.buttonResetMeshingToDefaults.Name = "buttonResetMeshingToDefaults";
            this.buttonResetMeshingToDefaults.Size = new System.Drawing.Size(201, 35);
            this.buttonResetMeshingToDefaults.TabIndex = 104;
            this.buttonResetMeshingToDefaults.Text = "Reset to Default";
            this.buttonResetMeshingToDefaults.UseVisualStyleBackColor = true;
            // 
            // checkBoxCollisionMeshing
            // 
            this.checkBoxCollisionMeshing.AutoSize = true;
            this.checkBoxCollisionMeshing.Checked = true;
            this.checkBoxCollisionMeshing.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkBoxCollisionMeshing.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxCollisionMeshing.Location = new System.Drawing.Point(409, 37);
            this.checkBoxCollisionMeshing.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxCollisionMeshing.Name = "checkBoxCollisionMeshing";
            this.checkBoxCollisionMeshing.Size = new System.Drawing.Size(93, 24);
            this.checkBoxCollisionMeshing.TabIndex = 63;
            this.checkBoxCollisionMeshing.Text = "Collision";
            this.checkBoxCollisionMeshing.UseVisualStyleBackColor = true;
            // 
            // numericUpDownAngularDeflectionCollision
            // 
            this.numericUpDownAngularDeflectionCollision.DecimalPlaces = 3;
            this.numericUpDownAngularDeflectionCollision.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.numericUpDownAngularDeflectionCollision.Increment = new decimal(new int[] {
            1,
            0,
            0,
            65536});
            this.numericUpDownAngularDeflectionCollision.Location = new System.Drawing.Point(378, 112);
            this.numericUpDownAngularDeflectionCollision.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.numericUpDownAngularDeflectionCollision.Maximum = new decimal(new int[] {
            180,
            0,
            0,
            0});
            this.numericUpDownAngularDeflectionCollision.Name = "numericUpDownAngularDeflectionCollision";
            this.numericUpDownAngularDeflectionCollision.Size = new System.Drawing.Size(120, 29);
            this.numericUpDownAngularDeflectionCollision.TabIndex = 60;
            this.numericUpDownAngularDeflectionCollision.Value = new decimal(new int[] {
            75,
            0,
            0,
            131072});
            // 
            // numericUpDownLinearDeflectionCollision
            // 
            this.numericUpDownLinearDeflectionCollision.DecimalPlaces = 3;
            this.numericUpDownLinearDeflectionCollision.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.numericUpDownLinearDeflectionCollision.Increment = new decimal(new int[] {
            1,
            0,
            0,
            65536});
            this.numericUpDownLinearDeflectionCollision.Location = new System.Drawing.Point(378, 72);
            this.numericUpDownLinearDeflectionCollision.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.numericUpDownLinearDeflectionCollision.Name = "numericUpDownLinearDeflectionCollision";
            this.numericUpDownLinearDeflectionCollision.Size = new System.Drawing.Size(120, 29);
            this.numericUpDownLinearDeflectionCollision.TabIndex = 59;
            this.numericUpDownLinearDeflectionCollision.Value = new decimal(new int[] {
            3,
            0,
            0,
            65536});
            // 
            // numericUpDownAngularDeflection
            // 
            this.numericUpDownAngularDeflection.DecimalPlaces = 3;
            this.numericUpDownAngularDeflection.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.numericUpDownAngularDeflection.Increment = new decimal(new int[] {
            1,
            0,
            0,
            65536});
            this.numericUpDownAngularDeflection.Location = new System.Drawing.Point(238, 112);
            this.numericUpDownAngularDeflection.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.numericUpDownAngularDeflection.Maximum = new decimal(new int[] {
            180,
            0,
            0,
            0});
            this.numericUpDownAngularDeflection.Name = "numericUpDownAngularDeflection";
            this.numericUpDownAngularDeflection.Size = new System.Drawing.Size(120, 29);
            this.numericUpDownAngularDeflection.TabIndex = 55;
            this.numericUpDownAngularDeflection.Value = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            // 
            // numericUpDownLinearDeflection
            // 
            this.numericUpDownLinearDeflection.DecimalPlaces = 3;
            this.numericUpDownLinearDeflection.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.numericUpDownLinearDeflection.Increment = new decimal(new int[] {
            1,
            0,
            0,
            65536});
            this.numericUpDownLinearDeflection.Location = new System.Drawing.Point(238, 72);
            this.numericUpDownLinearDeflection.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.numericUpDownLinearDeflection.Name = "numericUpDownLinearDeflection";
            this.numericUpDownLinearDeflection.Size = new System.Drawing.Size(120, 29);
            this.numericUpDownLinearDeflection.TabIndex = 54;
            this.numericUpDownLinearDeflection.Value = new decimal(new int[] {
            1,
            0,
            0,
            65536});
            // 
            // groupBox6
            // 
            this.groupBox6.Controls.Add(this.domainUpDownAlpha);
            this.groupBox6.Controls.Add(this.label33);
            this.groupBox6.Controls.Add(this.domainUpDownRed);
            this.groupBox6.Controls.Add(this.label32);
            this.groupBox6.Controls.Add(this.domainUpDownGreen);
            this.groupBox6.Controls.Add(this.label31);
            this.groupBox6.Controls.Add(this.domainUpDownBlue);
            this.groupBox6.Controls.Add(this.label30);
            this.groupBox6.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.groupBox6.Location = new System.Drawing.Point(340, 114);
            this.groupBox6.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox6.Name = "groupBox6";
            this.groupBox6.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox6.Size = new System.Drawing.Size(513, 75);
            this.groupBox6.TabIndex = 77;
            this.groupBox6.TabStop = false;
            this.groupBox6.Text = "Per-Link Color*";
            // 
            // domainUpDownAlpha
            // 
            this.domainUpDownAlpha.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.domainUpDownAlpha.Location = new System.Drawing.Point(368, 30);
            this.domainUpDownAlpha.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.domainUpDownAlpha.Name = "domainUpDownAlpha";
            this.domainUpDownAlpha.Size = new System.Drawing.Size(76, 29);
            this.domainUpDownAlpha.TabIndex = 23;
            this.domainUpDownAlpha.Text = "1";
            // 
            // label33
            // 
            this.label33.AutoSize = true;
            this.label33.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label33.Location = new System.Drawing.Point(343, 32);
            this.label33.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label33.Name = "label33";
            this.label33.Size = new System.Drawing.Size(17, 24);
            this.label33.TabIndex = 64;
            this.label33.Text = "A";
            this.label33.UseCompatibleTextRendering = true;
            // 
            // domainUpDownRed
            // 
            this.domainUpDownRed.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.domainUpDownRed.Location = new System.Drawing.Point(40, 30);
            this.domainUpDownRed.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.domainUpDownRed.Name = "domainUpDownRed";
            this.domainUpDownRed.Size = new System.Drawing.Size(76, 29);
            this.domainUpDownRed.TabIndex = 20;
            this.domainUpDownRed.Text = "1";
            // 
            // label32
            // 
            this.label32.AutoSize = true;
            this.label32.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label32.Location = new System.Drawing.Point(234, 32);
            this.label32.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label32.Name = "label32";
            this.label32.Size = new System.Drawing.Size(17, 24);
            this.label32.TabIndex = 63;
            this.label32.Text = "B";
            this.label32.UseCompatibleTextRendering = true;
            // 
            // domainUpDownGreen
            // 
            this.domainUpDownGreen.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.domainUpDownGreen.Location = new System.Drawing.Point(150, 30);
            this.domainUpDownGreen.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.domainUpDownGreen.Name = "domainUpDownGreen";
            this.domainUpDownGreen.Size = new System.Drawing.Size(76, 29);
            this.domainUpDownGreen.TabIndex = 21;
            this.domainUpDownGreen.Text = "1";
            // 
            // label31
            // 
            this.label31.AutoSize = true;
            this.label31.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label31.Location = new System.Drawing.Point(123, 32);
            this.label31.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label31.Name = "label31";
            this.label31.Size = new System.Drawing.Size(19, 24);
            this.label31.TabIndex = 62;
            this.label31.Text = "G";
            this.label31.UseCompatibleTextRendering = true;
            // 
            // domainUpDownBlue
            // 
            this.domainUpDownBlue.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.domainUpDownBlue.Location = new System.Drawing.Point(259, 30);
            this.domainUpDownBlue.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.domainUpDownBlue.Name = "domainUpDownBlue";
            this.domainUpDownBlue.Size = new System.Drawing.Size(76, 29);
            this.domainUpDownBlue.TabIndex = 22;
            this.domainUpDownBlue.Text = "1";
            // 
            // label30
            // 
            this.label30.AutoSize = true;
            this.label30.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label30.Location = new System.Drawing.Point(14, 32);
            this.label30.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label30.Name = "label30";
            this.label30.Size = new System.Drawing.Size(18, 24);
            this.label30.TabIndex = 61;
            this.label30.Text = "R";
            this.label30.UseCompatibleTextRendering = true;
            // 
            // groupBox3
            // 
            this.groupBox3.Controls.Add(this.label20);
            this.groupBox3.Controls.Add(this.label21);
            this.groupBox3.Controls.Add(this.label24);
            this.groupBox3.Controls.Add(this.label22);
            this.groupBox3.Controls.Add(this.label25);
            this.groupBox3.Controls.Add(this.label26);
            this.groupBox3.Controls.Add(this.textBoxVisualOriginZ);
            this.groupBox3.Controls.Add(this.textBoxVisualOriginYaw);
            this.groupBox3.Controls.Add(this.textBoxVisualOriginPitch);
            this.groupBox3.Controls.Add(this.textBoxVisualOriginRoll);
            this.groupBox3.Controls.Add(this.textBoxVisualOriginX);
            this.groupBox3.Controls.Add(this.textBoxVisualOriginY);
            this.groupBox3.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.groupBox3.Location = new System.Drawing.Point(8, 34);
            this.groupBox3.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox3.Size = new System.Drawing.Size(324, 170);
            this.groupBox3.TabIndex = 76;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "Origin (m, rad)";
            this.groupBox3.UseCompatibleTextRendering = true;
            // 
            // label20
            // 
            this.label20.AutoSize = true;
            this.label20.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label20.Location = new System.Drawing.Point(153, 52);
            this.label20.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label20.Name = "label20";
            this.label20.Size = new System.Drawing.Size(34, 24);
            this.label20.TabIndex = 47;
            this.label20.Text = "Roll";
            this.label20.TextAlign = System.Drawing.ContentAlignment.TopRight;
            this.label20.UseCompatibleTextRendering = true;
            // 
            // label21
            // 
            this.label21.AutoSize = true;
            this.label21.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label21.Location = new System.Drawing.Point(144, 92);
            this.label21.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label21.Name = "label21";
            this.label21.Size = new System.Drawing.Size(43, 24);
            this.label21.TabIndex = 50;
            this.label21.Text = "Pitch";
            this.label21.TextAlign = System.Drawing.ContentAlignment.TopRight;
            this.label21.UseCompatibleTextRendering = true;
            // 
            // label24
            // 
            this.label24.AutoSize = true;
            this.label24.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label24.Location = new System.Drawing.Point(14, 51);
            this.label24.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label24.Name = "label24";
            this.label24.Size = new System.Drawing.Size(17, 24);
            this.label24.TabIndex = 48;
            this.label24.Text = "X";
            this.label24.UseCompatibleTextRendering = true;
            // 
            // label22
            // 
            this.label22.AutoSize = true;
            this.label22.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label22.Location = new System.Drawing.Point(148, 132);
            this.label22.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label22.Name = "label22";
            this.label22.Size = new System.Drawing.Size(38, 24);
            this.label22.TabIndex = 53;
            this.label22.Text = "Yaw";
            this.label22.TextAlign = System.Drawing.ContentAlignment.TopRight;
            this.label22.UseCompatibleTextRendering = true;
            // 
            // label25
            // 
            this.label25.AutoSize = true;
            this.label25.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label25.Location = new System.Drawing.Point(14, 91);
            this.label25.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label25.Name = "label25";
            this.label25.Size = new System.Drawing.Size(17, 24);
            this.label25.TabIndex = 51;
            this.label25.Text = "Y";
            this.label25.UseCompatibleTextRendering = true;
            // 
            // label26
            // 
            this.label26.AutoSize = true;
            this.label26.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label26.Location = new System.Drawing.Point(14, 131);
            this.label26.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label26.Name = "label26";
            this.label26.Size = new System.Drawing.Size(16, 24);
            this.label26.TabIndex = 54;
            this.label26.Text = "Z";
            this.label26.UseCompatibleTextRendering = true;
            // 
            // textBoxVisualOriginZ
            // 
            this.textBoxVisualOriginZ.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxVisualOriginZ.Location = new System.Drawing.Point(39, 128);
            this.textBoxVisualOriginZ.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxVisualOriginZ.Name = "textBoxVisualOriginZ";
            this.textBoxVisualOriginZ.Size = new System.Drawing.Size(97, 29);
            this.textBoxVisualOriginZ.TabIndex = 16;
            // 
            // textBoxVisualOriginYaw
            // 
            this.textBoxVisualOriginYaw.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxVisualOriginYaw.Location = new System.Drawing.Point(195, 128);
            this.textBoxVisualOriginYaw.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxVisualOriginYaw.Name = "textBoxVisualOriginYaw";
            this.textBoxVisualOriginYaw.Size = new System.Drawing.Size(97, 29);
            this.textBoxVisualOriginYaw.TabIndex = 19;
            // 
            // textBoxVisualOriginPitch
            // 
            this.textBoxVisualOriginPitch.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxVisualOriginPitch.Location = new System.Drawing.Point(195, 88);
            this.textBoxVisualOriginPitch.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxVisualOriginPitch.Name = "textBoxVisualOriginPitch";
            this.textBoxVisualOriginPitch.Size = new System.Drawing.Size(97, 29);
            this.textBoxVisualOriginPitch.TabIndex = 18;
            // 
            // textBoxVisualOriginRoll
            // 
            this.textBoxVisualOriginRoll.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxVisualOriginRoll.Location = new System.Drawing.Point(195, 48);
            this.textBoxVisualOriginRoll.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxVisualOriginRoll.Name = "textBoxVisualOriginRoll";
            this.textBoxVisualOriginRoll.Size = new System.Drawing.Size(97, 29);
            this.textBoxVisualOriginRoll.TabIndex = 17;
            // 
            // textBoxVisualOriginX
            // 
            this.textBoxVisualOriginX.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxVisualOriginX.Location = new System.Drawing.Point(39, 48);
            this.textBoxVisualOriginX.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxVisualOriginX.Name = "textBoxVisualOriginX";
            this.textBoxVisualOriginX.Size = new System.Drawing.Size(97, 29);
            this.textBoxVisualOriginX.TabIndex = 14;
            // 
            // textBoxVisualOriginY
            // 
            this.textBoxVisualOriginY.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxVisualOriginY.Location = new System.Drawing.Point(39, 88);
            this.textBoxVisualOriginY.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxVisualOriginY.Name = "textBoxVisualOriginY";
            this.textBoxVisualOriginY.Size = new System.Drawing.Size(97, 29);
            this.textBoxVisualOriginY.TabIndex = 15;
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.comboBoxFolderStructure);
            this.groupBox1.Controls.Add(this.label42);
            this.groupBox1.Controls.Add(this.tableLayoutPanel1);
            this.groupBox1.Controls.Add(this.linkLabelHelpMeChoose);
            this.groupBox1.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.groupBox1.Location = new System.Drawing.Point(8, 212);
            this.groupBox1.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Padding = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.groupBox1.Size = new System.Drawing.Size(324, 476);
            this.groupBox1.TabIndex = 74;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Mesh Format";
            // 
            // comboBoxFolderStructure
            // 
            this.comboBoxFolderStructure.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxFolderStructure.FormattingEnabled = true;
            this.comboBoxFolderStructure.Location = new System.Drawing.Point(138, 435);
            this.comboBoxFolderStructure.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.comboBoxFolderStructure.Name = "comboBoxFolderStructure";
            this.comboBoxFolderStructure.Size = new System.Drawing.Size(173, 33);
            this.comboBoxFolderStructure.TabIndex = 81;
            // 
            // tableLayoutPanel1
            // 
            this.tableLayoutPanel1.ColumnCount = 3;
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 27.77778F));
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 27.77778F));
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 44.44444F));
            this.tableLayoutPanel1.Controls.Add(this.labelStep, 2, 7);
            this.tableLayoutPanel1.Controls.Add(this.labelObjCAD, 2, 6);
            this.tableLayoutPanel1.Controls.Add(this.labelGlbCAD, 2, 5);
            this.tableLayoutPanel1.Controls.Add(this.labelStlCAD, 2, 4);
            this.tableLayoutPanel1.Controls.Add(this.label40, 2, 2);
            this.tableLayoutPanel1.Controls.Add(this.checkBoxStepVisual, 0, 7);
            this.tableLayoutPanel1.Controls.Add(this.checkBoxObjCADVisual, 0, 6);
            this.tableLayoutPanel1.Controls.Add(this.checkBoxStepCollision, 1, 7);
            this.tableLayoutPanel1.Controls.Add(this.checkBoxObjCADCollision, 1, 6);
            this.tableLayoutPanel1.Controls.Add(this.checkBoxGlbCADVisual, 0, 5);
            this.tableLayoutPanel1.Controls.Add(this.checkBoxGlbCADCollision, 1, 5);
            this.tableLayoutPanel1.Controls.Add(this.checkBoxStlCADVisual, 0, 4);
            this.tableLayoutPanel1.Controls.Add(this.checkBoxStlCADCollision, 1, 4);
            this.tableLayoutPanel1.Controls.Add(this.label23, 0, 0);
            this.tableLayoutPanel1.Controls.Add(this.label38, 2, 0);
            this.tableLayoutPanel1.Controls.Add(this.label39, 2, 1);
            this.tableLayoutPanel1.Controls.Add(this.label37, 1, 0);
            this.tableLayoutPanel1.Controls.Add(this.textBoxMeshFileExtension, 0, 9);
            this.tableLayoutPanel1.Controls.Add(this.textBoxMeshFileExtensionCollision, 1, 9);
            this.tableLayoutPanel1.Controls.Add(this.label60, 2, 9);
            this.tableLayoutPanel1.Controls.Add(this.label41, 2, 3);
            this.tableLayoutPanel1.Controls.Add(this.checkBoxObjOpenCascadeCollision, 1, 3);
            this.tableLayoutPanel1.Controls.Add(this.checkBoxGlbOpenCascadeCollision, 1, 2);
            this.tableLayoutPanel1.Controls.Add(this.checkBoxStlOpenCascadeCollision, 1, 1);
            this.tableLayoutPanel1.Controls.Add(this.checkBoxObjOpenCascadeVisual, 0, 3);
            this.tableLayoutPanel1.Controls.Add(this.checkBoxGlbOpenCascadeVisual, 0, 2);
            this.tableLayoutPanel1.Controls.Add(this.checkBoxStlOpenCascadeVisual, 0, 1);
            this.tableLayoutPanel1.Location = new System.Drawing.Point(7, 32);
            this.tableLayoutPanel1.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.tableLayoutPanel1.Name = "tableLayoutPanel1";
            this.tableLayoutPanel1.RowCount = 10;
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 9.697078F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 9.697078F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 9.697078F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 12.72728F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 9.697078F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 9.697078F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 9.697078F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 9.697078F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 9.697078F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 9.696108F));
            this.tableLayoutPanel1.Size = new System.Drawing.Size(320, 374);
            this.tableLayoutPanel1.TabIndex = 69;
            // 
            // labelStep
            // 
            this.labelStep.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.labelStep.AutoSize = true;
            this.labelStep.Location = new System.Drawing.Point(179, 263);
            this.labelStep.Name = "labelStep";
            this.labelStep.Padding = new System.Windows.Forms.Padding(0, 0, 0, 5);
            this.labelStep.Size = new System.Drawing.Size(138, 36);
            this.labelStep.TabIndex = 78;
            this.labelStep.Text = "<cad>";
            this.labelStep.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // labelObjCAD
            // 
            this.labelObjCAD.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.labelObjCAD.AutoSize = true;
            this.labelObjCAD.Location = new System.Drawing.Point(179, 227);
            this.labelObjCAD.Name = "labelObjCAD";
            this.labelObjCAD.Padding = new System.Windows.Forms.Padding(0, 0, 0, 5);
            this.labelObjCAD.Size = new System.Drawing.Size(138, 36);
            this.labelObjCAD.TabIndex = 77;
            this.labelObjCAD.Text = "<cad>";
            this.labelObjCAD.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // labelGlbCAD
            // 
            this.labelGlbCAD.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.labelGlbCAD.AutoSize = true;
            this.labelGlbCAD.Location = new System.Drawing.Point(179, 191);
            this.labelGlbCAD.Name = "labelGlbCAD";
            this.labelGlbCAD.Padding = new System.Windows.Forms.Padding(0, 0, 0, 5);
            this.labelGlbCAD.Size = new System.Drawing.Size(138, 36);
            this.labelGlbCAD.TabIndex = 76;
            this.labelGlbCAD.Text = "<cad>";
            this.labelGlbCAD.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // labelStlCAD
            // 
            this.labelStlCAD.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.labelStlCAD.AutoSize = true;
            this.labelStlCAD.Location = new System.Drawing.Point(179, 155);
            this.labelStlCAD.Name = "labelStlCAD";
            this.labelStlCAD.Padding = new System.Windows.Forms.Padding(0, 0, 0, 5);
            this.labelStlCAD.Size = new System.Drawing.Size(138, 36);
            this.labelStlCAD.TabIndex = 75;
            this.labelStlCAD.Text = "<cad>";
            this.labelStlCAD.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // label40
            // 
            this.label40.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.label40.AutoSize = true;
            this.label40.Location = new System.Drawing.Point(179, 72);
            this.label40.Name = "label40";
            this.label40.Padding = new System.Windows.Forms.Padding(0, 0, 0, 5);
            this.label40.Size = new System.Drawing.Size(138, 36);
            this.label40.TabIndex = 73;
            this.label40.Text = "SuperDex";
            this.label40.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // checkBoxStepVisual
            // 
            this.checkBoxStepVisual.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left)));
            this.checkBoxStepVisual.AutoSize = true;
            this.checkBoxStepVisual.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxStepVisual.Location = new System.Drawing.Point(3, 265);
            this.checkBoxStepVisual.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxStepVisual.Name = "checkBoxStepVisual";
            this.checkBoxStepVisual.Size = new System.Drawing.Size(76, 32);
            this.checkBoxStepVisual.TabIndex = 65;
            this.checkBoxStepVisual.Text = "STEP";
            this.checkBoxStepVisual.UseVisualStyleBackColor = true;
            // 
            // checkBoxObjCADVisual
            // 
            this.checkBoxObjCADVisual.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left)));
            this.checkBoxObjCADVisual.AutoSize = true;
            this.checkBoxObjCADVisual.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxObjCADVisual.Location = new System.Drawing.Point(3, 229);
            this.checkBoxObjCADVisual.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxObjCADVisual.Name = "checkBoxObjCADVisual";
            this.checkBoxObjCADVisual.Size = new System.Drawing.Size(66, 32);
            this.checkBoxObjCADVisual.TabIndex = 68;
            this.checkBoxObjCADVisual.Text = "OBJ";
            this.checkBoxObjCADVisual.UseVisualStyleBackColor = true;
            // 
            // checkBoxStepCollision
            // 
            this.checkBoxStepCollision.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left)));
            this.checkBoxStepCollision.AutoSize = true;
            this.checkBoxStepCollision.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxStepCollision.Location = new System.Drawing.Point(91, 265);
            this.checkBoxStepCollision.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxStepCollision.Name = "checkBoxStepCollision";
            this.checkBoxStepCollision.Size = new System.Drawing.Size(76, 32);
            this.checkBoxStepCollision.TabIndex = 30;
            this.checkBoxStepCollision.Text = "STEP";
            this.checkBoxStepCollision.UseVisualStyleBackColor = true;
            // 
            // checkBoxObjCADCollision
            // 
            this.checkBoxObjCADCollision.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left)));
            this.checkBoxObjCADCollision.AutoSize = true;
            this.checkBoxObjCADCollision.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxObjCADCollision.Location = new System.Drawing.Point(91, 229);
            this.checkBoxObjCADCollision.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxObjCADCollision.Name = "checkBoxObjCADCollision";
            this.checkBoxObjCADCollision.Size = new System.Drawing.Size(66, 32);
            this.checkBoxObjCADCollision.TabIndex = 60;
            this.checkBoxObjCADCollision.Text = "OBJ";
            this.checkBoxObjCADCollision.UseVisualStyleBackColor = true;
            // 
            // checkBoxGlbCADVisual
            // 
            this.checkBoxGlbCADVisual.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left)));
            this.checkBoxGlbCADVisual.AutoSize = true;
            this.checkBoxGlbCADVisual.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxGlbCADVisual.Location = new System.Drawing.Point(3, 193);
            this.checkBoxGlbCADVisual.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxGlbCADVisual.Name = "checkBoxGlbCADVisual";
            this.checkBoxGlbCADVisual.Size = new System.Drawing.Size(68, 32);
            this.checkBoxGlbCADVisual.TabIndex = 67;
            this.checkBoxGlbCADVisual.Text = "GLB";
            this.checkBoxGlbCADVisual.UseVisualStyleBackColor = true;
            // 
            // checkBoxGlbCADCollision
            // 
            this.checkBoxGlbCADCollision.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left)));
            this.checkBoxGlbCADCollision.AutoSize = true;
            this.checkBoxGlbCADCollision.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxGlbCADCollision.Location = new System.Drawing.Point(91, 193);
            this.checkBoxGlbCADCollision.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxGlbCADCollision.Name = "checkBoxGlbCADCollision";
            this.checkBoxGlbCADCollision.Size = new System.Drawing.Size(68, 32);
            this.checkBoxGlbCADCollision.TabIndex = 59;
            this.checkBoxGlbCADCollision.Text = "GLB";
            this.checkBoxGlbCADCollision.UseVisualStyleBackColor = true;
            // 
            // checkBoxStlCADVisual
            // 
            this.checkBoxStlCADVisual.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left)));
            this.checkBoxStlCADVisual.AutoSize = true;
            this.checkBoxStlCADVisual.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxStlCADVisual.Location = new System.Drawing.Point(3, 157);
            this.checkBoxStlCADVisual.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxStlCADVisual.Name = "checkBoxStlCADVisual";
            this.checkBoxStlCADVisual.Size = new System.Drawing.Size(64, 32);
            this.checkBoxStlCADVisual.TabIndex = 64;
            this.checkBoxStlCADVisual.Text = "STL";
            this.checkBoxStlCADVisual.UseVisualStyleBackColor = true;
            // 
            // checkBoxStlCADCollision
            // 
            this.checkBoxStlCADCollision.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left)));
            this.checkBoxStlCADCollision.AutoSize = true;
            this.checkBoxStlCADCollision.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxStlCADCollision.Location = new System.Drawing.Point(91, 157);
            this.checkBoxStlCADCollision.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxStlCADCollision.Name = "checkBoxStlCADCollision";
            this.checkBoxStlCADCollision.Size = new System.Drawing.Size(64, 32);
            this.checkBoxStlCADCollision.TabIndex = 29;
            this.checkBoxStlCADCollision.Text = "STL";
            this.checkBoxStlCADCollision.UseVisualStyleBackColor = true;
            // 
            // label23
            // 
            this.label23.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.label23.AutoSize = true;
            this.label23.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label23.Location = new System.Drawing.Point(0, 0);
            this.label23.Margin = new System.Windows.Forms.Padding(0);
            this.label23.Name = "label23";
            this.label23.Size = new System.Drawing.Size(88, 36);
            this.label23.TabIndex = 69;
            this.label23.Text = "Visual";
            this.label23.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // label38
            // 
            this.label38.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.label38.AutoSize = true;
            this.label38.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label38.Location = new System.Drawing.Point(176, 0);
            this.label38.Margin = new System.Windows.Forms.Padding(0);
            this.label38.Name = "label38";
            this.label38.Size = new System.Drawing.Size(144, 36);
            this.label38.TabIndex = 71;
            this.label38.Text = "Export Method";
            this.label38.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // label39
            // 
            this.label39.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.label39.AutoSize = true;
            this.label39.Location = new System.Drawing.Point(179, 36);
            this.label39.Name = "label39";
            this.label39.Padding = new System.Windows.Forms.Padding(0, 0, 0, 5);
            this.label39.Size = new System.Drawing.Size(138, 36);
            this.label39.TabIndex = 72;
            this.label39.Text = "SuperDex";
            this.label39.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // label37
            // 
            this.label37.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.label37.AutoSize = true;
            this.label37.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label37.Location = new System.Drawing.Point(88, 0);
            this.label37.Margin = new System.Windows.Forms.Padding(0);
            this.label37.Name = "label37";
            this.label37.Size = new System.Drawing.Size(88, 36);
            this.label37.TabIndex = 70;
            this.label37.Text = "Collision";
            this.label37.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // textBoxMeshFileExtension
            // 
            this.textBoxMeshFileExtension.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxMeshFileExtension.Location = new System.Drawing.Point(4, 340);
            this.textBoxMeshFileExtension.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxMeshFileExtension.Name = "textBoxMeshFileExtension";
            this.textBoxMeshFileExtension.Size = new System.Drawing.Size(72, 29);
            this.textBoxMeshFileExtension.TabIndex = 55;
            this.textBoxMeshFileExtension.Text = "glb";
            // 
            // textBoxMeshFileExtensionCollision
            // 
            this.textBoxMeshFileExtensionCollision.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxMeshFileExtensionCollision.Location = new System.Drawing.Point(92, 340);
            this.textBoxMeshFileExtensionCollision.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxMeshFileExtensionCollision.Name = "textBoxMeshFileExtensionCollision";
            this.textBoxMeshFileExtensionCollision.Size = new System.Drawing.Size(72, 29);
            this.textBoxMeshFileExtensionCollision.TabIndex = 58;
            this.textBoxMeshFileExtensionCollision.Text = "glb";
            // 
            // label41
            // 
            this.label41.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.label41.AutoSize = true;
            this.label41.Location = new System.Drawing.Point(179, 108);
            this.label41.Name = "label41";
            this.label41.Padding = new System.Windows.Forms.Padding(0, 5, 0, 5);
            this.label41.Size = new System.Drawing.Size(138, 35);
            this.label41.TabIndex = 74;
            this.label41.Text = "SuperDex";
            this.label41.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // checkBoxObjOpenCascadeCollision
            // 
            this.checkBoxObjOpenCascadeCollision.AutoSize = true;
            this.checkBoxObjOpenCascadeCollision.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxObjOpenCascadeCollision.Location = new System.Drawing.Point(91, 110);
            this.checkBoxObjOpenCascadeCollision.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxObjOpenCascadeCollision.Name = "checkBoxObjOpenCascadeCollision";
            this.checkBoxObjOpenCascadeCollision.Padding = new System.Windows.Forms.Padding(0, 5, 0, 0);
            this.checkBoxObjOpenCascadeCollision.Size = new System.Drawing.Size(66, 29);
            this.checkBoxObjOpenCascadeCollision.TabIndex = 27;
            this.checkBoxObjOpenCascadeCollision.Text = "OBJ";
            this.checkBoxObjOpenCascadeCollision.UseVisualStyleBackColor = true;
            // 
            // checkBoxGlbOpenCascadeCollision
            // 
            this.checkBoxGlbOpenCascadeCollision.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left)));
            this.checkBoxGlbOpenCascadeCollision.AutoSize = true;
            this.checkBoxGlbOpenCascadeCollision.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxGlbOpenCascadeCollision.Location = new System.Drawing.Point(91, 74);
            this.checkBoxGlbOpenCascadeCollision.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxGlbOpenCascadeCollision.Name = "checkBoxGlbOpenCascadeCollision";
            this.checkBoxGlbOpenCascadeCollision.Size = new System.Drawing.Size(68, 32);
            this.checkBoxGlbOpenCascadeCollision.TabIndex = 56;
            this.checkBoxGlbOpenCascadeCollision.Text = "GLB";
            this.checkBoxGlbOpenCascadeCollision.UseVisualStyleBackColor = true;
            // 
            // checkBoxStlOpenCascadeCollision
            // 
            this.checkBoxStlOpenCascadeCollision.AutoSize = true;
            this.checkBoxStlOpenCascadeCollision.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxStlOpenCascadeCollision.Location = new System.Drawing.Point(91, 38);
            this.checkBoxStlOpenCascadeCollision.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxStlOpenCascadeCollision.Name = "checkBoxStlOpenCascadeCollision";
            this.checkBoxStlOpenCascadeCollision.Size = new System.Drawing.Size(64, 24);
            this.checkBoxStlOpenCascadeCollision.TabIndex = 28;
            this.checkBoxStlOpenCascadeCollision.Text = "STL";
            this.checkBoxStlOpenCascadeCollision.UseVisualStyleBackColor = true;
            // 
            // checkBoxObjOpenCascadeVisual
            // 
            this.checkBoxObjOpenCascadeVisual.AutoSize = true;
            this.checkBoxObjOpenCascadeVisual.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxObjOpenCascadeVisual.Location = new System.Drawing.Point(3, 110);
            this.checkBoxObjOpenCascadeVisual.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxObjOpenCascadeVisual.Name = "checkBoxObjOpenCascadeVisual";
            this.checkBoxObjOpenCascadeVisual.Padding = new System.Windows.Forms.Padding(0, 5, 0, 0);
            this.checkBoxObjOpenCascadeVisual.Size = new System.Drawing.Size(66, 29);
            this.checkBoxObjOpenCascadeVisual.TabIndex = 62;
            this.checkBoxObjOpenCascadeVisual.Text = "OBJ";
            this.checkBoxObjOpenCascadeVisual.UseVisualStyleBackColor = true;
            // 
            // checkBoxGlbOpenCascadeVisual
            // 
            this.checkBoxGlbOpenCascadeVisual.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left)));
            this.checkBoxGlbOpenCascadeVisual.AutoSize = true;
            this.checkBoxGlbOpenCascadeVisual.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxGlbOpenCascadeVisual.Location = new System.Drawing.Point(3, 74);
            this.checkBoxGlbOpenCascadeVisual.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxGlbOpenCascadeVisual.Name = "checkBoxGlbOpenCascadeVisual";
            this.checkBoxGlbOpenCascadeVisual.Size = new System.Drawing.Size(68, 32);
            this.checkBoxGlbOpenCascadeVisual.TabIndex = 66;
            this.checkBoxGlbOpenCascadeVisual.Text = "GLB";
            this.checkBoxGlbOpenCascadeVisual.UseVisualStyleBackColor = true;
            // 
            // checkBoxStlOpenCascadeVisual
            // 
            this.checkBoxStlOpenCascadeVisual.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left)));
            this.checkBoxStlOpenCascadeVisual.AutoSize = true;
            this.checkBoxStlOpenCascadeVisual.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxStlOpenCascadeVisual.Location = new System.Drawing.Point(3, 38);
            this.checkBoxStlOpenCascadeVisual.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxStlOpenCascadeVisual.Name = "checkBoxStlOpenCascadeVisual";
            this.checkBoxStlOpenCascadeVisual.Size = new System.Drawing.Size(64, 32);
            this.checkBoxStlOpenCascadeVisual.TabIndex = 63;
            this.checkBoxStlOpenCascadeVisual.Text = "STL";
            this.checkBoxStlOpenCascadeVisual.UseVisualStyleBackColor = true;
            // 
            // linkLabelHelpMeChoose
            // 
            this.linkLabelHelpMeChoose.AutoSize = true;
            this.linkLabelHelpMeChoose.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(249)))), ((int)(((byte)(249)))), ((int)(((byte)(249)))));
            this.linkLabelHelpMeChoose.Location = new System.Drawing.Point(179, 0);
            this.linkLabelHelpMeChoose.Name = "linkLabelHelpMeChoose";
            this.linkLabelHelpMeChoose.Size = new System.Drawing.Size(141, 25);
            this.linkLabelHelpMeChoose.TabIndex = 61;
            this.linkLabelHelpMeChoose.TabStop = true;
            this.linkLabelHelpMeChoose.Text = "Help me choose";
            this.linkLabelHelpMeChoose.VisitedLinkColor = System.Drawing.Color.Blue;
            // 
            // label19
            // 
            this.label19.AutoSize = true;
            this.label19.Font = new System.Drawing.Font("Segoe UI Variable Display", 9F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label19.Location = new System.Drawing.Point(4, 0);
            this.label19.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label19.Name = "label19";
            this.label19.Size = new System.Drawing.Size(375, 24);
            this.label19.TabIndex = 62;
            this.label19.Text = "Visual and Collision Mesh Export Properties";
            // 
            // treeViewLinkProperties
            // 
            this.treeViewLinkProperties.AllowDrop = true;
            this.treeViewLinkProperties.Font = new System.Drawing.Font("Segoe UI", 10F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.treeViewLinkProperties.Location = new System.Drawing.Point(14, 91);
            this.treeViewLinkProperties.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.treeViewLinkProperties.Name = "treeViewLinkProperties";
            this.treeViewLinkProperties.Size = new System.Drawing.Size(686, 774);
            this.treeViewLinkProperties.TabIndex = 97;
            // 
            // tabPageJointProperties
            // 
            this.tabPageJointProperties.Controls.Add(this.checkBoxUseDegrees);
            this.tabPageJointProperties.Controls.Add(this.label35);
            this.tabPageJointProperties.Controls.Add(this.groupBox18);
            this.tabPageJointProperties.Controls.Add(this.label7);
            this.tabPageJointProperties.Controls.Add(this.label4);
            this.tabPageJointProperties.Controls.Add(this.label27);
            this.tabPageJointProperties.Controls.Add(this.treeViewJointTree);
            this.tabPageJointProperties.Controls.Add(this.label69);
            this.tabPageJointProperties.Controls.Add(this.groupBox10);
            this.tabPageJointProperties.Controls.Add(this.groupBoxJointOrigin);
            this.tabPageJointProperties.Controls.Add(this.groupBox12);
            this.tabPageJointProperties.Controls.Add(this.groupBox13);
            this.tabPageJointProperties.Controls.Add(this.groupBox14);
            this.tabPageJointProperties.Location = new System.Drawing.Point(4, 30);
            this.tabPageJointProperties.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.tabPageJointProperties.Name = "tabPageJointProperties";
            this.tabPageJointProperties.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.tabPageJointProperties.Size = new System.Drawing.Size(1579, 1007);
            this.tabPageJointProperties.TabIndex = 0;
            this.tabPageJointProperties.Text = "Joint Properties";
            this.tabPageJointProperties.UseVisualStyleBackColor = true;
            // 
            // checkBoxUseDegrees
            // 
            this.checkBoxUseDegrees.AutoSize = true;
            this.checkBoxUseDegrees.Checked = true;
            this.checkBoxUseDegrees.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkBoxUseDegrees.Location = new System.Drawing.Point(1354, 959);
            this.checkBoxUseDegrees.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxUseDegrees.Name = "checkBoxUseDegrees";
            this.checkBoxUseDegrees.Size = new System.Drawing.Size(199, 25);
            this.checkBoxUseDegrees.TabIndex = 309;
            this.checkBoxUseDegrees.Text = "Use Degrees for Editing";
            this.checkBoxUseDegrees.UseVisualStyleBackColor = true;
            // 
            // label35
            // 
            this.label35.AutoSize = true;
            this.label35.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label35.Location = new System.Drawing.Point(1198, 45);
            this.label35.Name = "label35";
            this.label35.Size = new System.Drawing.Size(355, 20);
            this.label35.TabIndex = 308;
            this.label35.Text = "* Field will be overridden from CAD on next export";
            this.label35.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // groupBox18
            // 
            this.groupBox18.Controls.Add(this.checkBoxShowJointHighlights);
            this.groupBox18.Controls.Add(this.label10);
            this.groupBox18.Controls.Add(this.checkBoxShowJointVisualization);
            this.groupBox18.Controls.Add(this.trackBarJointGizmoSize);
            this.groupBox18.Font = new System.Drawing.Font("Segoe UI Variable Display", 9F, System.Drawing.FontStyle.Bold);
            this.groupBox18.Location = new System.Drawing.Point(14, 872);
            this.groupBox18.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox18.Name = "groupBox18";
            this.groupBox18.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox18.Size = new System.Drawing.Size(686, 112);
            this.groupBox18.TabIndex = 307;
            this.groupBox18.TabStop = false;
            this.groupBox18.Text = "Visualization";
            // 
            // checkBoxShowJointHighlights
            // 
            this.checkBoxShowJointHighlights.AutoSize = true;
            this.checkBoxShowJointHighlights.Checked = true;
            this.checkBoxShowJointHighlights.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkBoxShowJointHighlights.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxShowJointHighlights.Location = new System.Drawing.Point(15, 35);
            this.checkBoxShowJointHighlights.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxShowJointHighlights.Name = "checkBoxShowJointHighlights";
            this.checkBoxShowJointHighlights.Size = new System.Drawing.Size(217, 24);
            this.checkBoxShowJointHighlights.TabIndex = 318;
            this.checkBoxShowJointHighlights.Text = "Highlight Selected Bodies";
            this.checkBoxShowJointHighlights.UseVisualStyleBackColor = true;
            // 
            // label10
            // 
            this.label10.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label10.Location = new System.Drawing.Point(351, 58);
            this.label10.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(122, 24);
            this.label10.TabIndex = 290;
            this.label10.Text = "Gizmo Size";
            this.label10.TextAlign = System.Drawing.ContentAlignment.TopRight;
            this.label10.UseCompatibleTextRendering = true;
            // 
            // checkBoxShowJointVisualization
            // 
            this.checkBoxShowJointVisualization.AutoSize = true;
            this.checkBoxShowJointVisualization.Checked = true;
            this.checkBoxShowJointVisualization.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkBoxShowJointVisualization.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxShowJointVisualization.Location = new System.Drawing.Point(15, 70);
            this.checkBoxShowJointVisualization.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxShowJointVisualization.Name = "checkBoxShowJointVisualization";
            this.checkBoxShowJointVisualization.Size = new System.Drawing.Size(222, 24);
            this.checkBoxShowJointVisualization.TabIndex = 317;
            this.checkBoxShowJointVisualization.Text = "Show Joint Axis and Limits";
            this.checkBoxShowJointVisualization.UseVisualStyleBackColor = true;
            // 
            // trackBarJointGizmoSize
            // 
            this.trackBarJointGizmoSize.AutoSize = false;
            this.trackBarJointGizmoSize.Location = new System.Drawing.Point(480, 39);
            this.trackBarJointGizmoSize.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.trackBarJointGizmoSize.Maximum = 20;
            this.trackBarJointGizmoSize.Minimum = 1;
            this.trackBarJointGizmoSize.Name = "trackBarJointGizmoSize";
            this.trackBarJointGizmoSize.Size = new System.Drawing.Size(198, 46);
            this.trackBarJointGizmoSize.TabIndex = 0;
            this.trackBarJointGizmoSize.Value = 10;
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Font = new System.Drawing.Font("Microsoft Sans Serif", 10F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label7.Location = new System.Drawing.Point(9, 18);
            this.label7.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(255, 29);
            this.label7.TabIndex = 302;
            this.label7.Text = "Configure Joint Properties";
            this.label7.UseCompatibleTextRendering = true;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label4.Location = new System.Drawing.Point(1155, 18);
            this.label4.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(398, 20);
            this.label4.TabIndex = 230;
            this.label4.Text = "Entries that are blank will not be written to URDF/MJCF";
            this.label4.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label27
            // 
            this.label27.AutoSize = true;
            this.label27.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label27.Location = new System.Drawing.Point(1372, 72);
            this.label27.Name = "label27";
            this.label27.Size = new System.Drawing.Size(181, 20);
            this.label27.TabIndex = 234;
            this.label27.Text = "** Field group is required";
            this.label27.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // treeViewJointTree
            // 
            this.treeViewJointTree.Font = new System.Drawing.Font("Segoe UI", 10F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.treeViewJointTree.Location = new System.Drawing.Point(14, 91);
            this.treeViewJointTree.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.treeViewJointTree.Name = "treeViewJointTree";
            this.treeViewJointTree.Size = new System.Drawing.Size(686, 774);
            this.treeViewJointTree.TabIndex = 0;
            // 
            // label69
            // 
            this.label69.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label69.Location = new System.Drawing.Point(10, 45);
            this.label69.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label69.Name = "label69";
            this.label69.Size = new System.Drawing.Size(698, 48);
            this.label69.TabIndex = 305;
            this.label69.Text = "Customize the joint properties. If you want to adjust the coordinate systems and " +
    "axes in the model, click cancel and restart the export. The tool will recognize " +
    "your changes on the next run.";
            this.label69.UseCompatibleTextRendering = true;
            // 
            // groupBox10
            // 
            this.groupBox10.Controls.Add(this.textBoxParentLink);
            this.groupBox10.Controls.Add(this.textBoxChildLink);
            this.groupBox10.Controls.Add(this.textBoxRefAxis);
            this.groupBox10.Controls.Add(this.label62);
            this.groupBox10.Controls.Add(this.textBoxCoordSys);
            this.groupBox10.Controls.Add(this.comboBoxJointType);
            this.groupBox10.Controls.Add(this.textBoxJointName);
            this.groupBox10.Controls.Add(this.label63);
            this.groupBox10.Controls.Add(this.label64);
            this.groupBox10.Controls.Add(this.label65);
            this.groupBox10.Controls.Add(this.label67);
            this.groupBox10.Controls.Add(this.label66);
            this.groupBox10.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.groupBox10.Location = new System.Drawing.Point(714, 91);
            this.groupBox10.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox10.Name = "groupBox10";
            this.groupBox10.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox10.Size = new System.Drawing.Size(863, 296);
            this.groupBox10.TabIndex = 1;
            this.groupBox10.TabStop = false;
            this.groupBox10.Text = "Basic Properties";
            // 
            // textBoxParentLink
            // 
            this.textBoxParentLink.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxParentLink.Location = new System.Drawing.Point(125, 39);
            this.textBoxParentLink.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxParentLink.Name = "textBoxParentLink";
            this.textBoxParentLink.ReadOnly = true;
            this.textBoxParentLink.Size = new System.Drawing.Size(717, 31);
            this.textBoxParentLink.TabIndex = 2;
            // 
            // textBoxChildLink
            // 
            this.textBoxChildLink.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxChildLink.Location = new System.Drawing.Point(125, 84);
            this.textBoxChildLink.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxChildLink.Name = "textBoxChildLink";
            this.textBoxChildLink.ReadOnly = true;
            this.textBoxChildLink.Size = new System.Drawing.Size(717, 31);
            this.textBoxChildLink.TabIndex = 3;
            // 
            // textBoxRefAxis
            // 
            this.textBoxRefAxis.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxRefAxis.Location = new System.Drawing.Point(125, 239);
            this.textBoxRefAxis.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxRefAxis.Name = "textBoxRefAxis";
            this.textBoxRefAxis.ReadOnly = true;
            this.textBoxRefAxis.Size = new System.Drawing.Size(717, 31);
            this.textBoxRefAxis.TabIndex = 5;
            // 
            // label62
            // 
            this.label62.AutoSize = true;
            this.label62.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label62.Location = new System.Drawing.Point(572, 138);
            this.label62.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label62.Name = "label62";
            this.label62.Size = new System.Drawing.Size(82, 24);
            this.label62.TabIndex = 267;
            this.label62.Text = "Joint Type";
            this.label62.UseCompatibleTextRendering = true;
            // 
            // textBoxCoordSys
            // 
            this.textBoxCoordSys.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxCoordSys.Location = new System.Drawing.Point(125, 186);
            this.textBoxCoordSys.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxCoordSys.Name = "textBoxCoordSys";
            this.textBoxCoordSys.ReadOnly = true;
            this.textBoxCoordSys.Size = new System.Drawing.Size(717, 31);
            this.textBoxCoordSys.TabIndex = 4;
            // 
            // comboBoxJointType
            // 
            this.comboBoxJointType.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.comboBoxJointType.FormattingEnabled = true;
            this.comboBoxJointType.Items.AddRange(new object[] {
            "revolute",
            "prismatic",
            "fixed"});
            this.comboBoxJointType.Location = new System.Drawing.Point(662, 134);
            this.comboBoxJointType.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.comboBoxJointType.Name = "comboBoxJointType";
            this.comboBoxJointType.Size = new System.Drawing.Size(180, 29);
            this.comboBoxJointType.TabIndex = 1;
            // 
            // textBoxJointName
            // 
            this.textBoxJointName.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxJointName.Location = new System.Drawing.Point(125, 134);
            this.textBoxJointName.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxJointName.Name = "textBoxJointName";
            this.textBoxJointName.Size = new System.Drawing.Size(435, 29);
            this.textBoxJointName.TabIndex = 0;
            // 
            // label63
            // 
            this.label63.AutoSize = true;
            this.label63.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label63.Location = new System.Drawing.Point(22, 138);
            this.label63.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label63.Name = "label63";
            this.label63.Size = new System.Drawing.Size(89, 24);
            this.label63.TabIndex = 269;
            this.label63.Text = "Joint Name";
            this.label63.UseCompatibleTextRendering = true;
            // 
            // label64
            // 
            this.label64.AutoSize = true;
            this.label64.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label64.Location = new System.Drawing.Point(22, 42);
            this.label64.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label64.Name = "label64";
            this.label64.Size = new System.Drawing.Size(89, 24);
            this.label64.TabIndex = 270;
            this.label64.Text = "Parent Link";
            this.label64.UseCompatibleTextRendering = true;
            // 
            // label65
            // 
            this.label65.AutoSize = true;
            this.label65.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label65.Location = new System.Drawing.Point(33, 88);
            this.label65.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label65.Name = "label65";
            this.label65.Size = new System.Drawing.Size(78, 24);
            this.label65.TabIndex = 271;
            this.label65.Text = "Child Link";
            this.label65.UseCompatibleTextRendering = true;
            // 
            // label67
            // 
            this.label67.AutoSize = true;
            this.label67.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label67.Location = new System.Drawing.Point(35, 242);
            this.label67.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label67.Name = "label67";
            this.label67.Size = new System.Drawing.Size(76, 24);
            this.label67.TabIndex = 304;
            this.label67.Text = "Joint Axis";
            this.label67.TextAlign = System.Drawing.ContentAlignment.TopRight;
            this.label67.UseCompatibleTextRendering = true;
            // 
            // label66
            // 
            this.label66.AutoSize = true;
            this.label66.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label66.Location = new System.Drawing.Point(16, 178);
            this.label66.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label66.Name = "label66";
            this.label66.Size = new System.Drawing.Size(95, 42);
            this.label66.TabIndex = 303;
            this.label66.Text = "Coordinate\r\nSystem";
            this.label66.TextAlign = System.Drawing.ContentAlignment.TopRight;
            this.label66.UseCompatibleTextRendering = true;
            // 
            // groupBoxJointOrigin
            // 
            this.groupBoxJointOrigin.Controls.Add(this.textBoxJointY);
            this.groupBoxJointOrigin.Controls.Add(this.label57);
            this.groupBoxJointOrigin.Controls.Add(this.textBoxJointZ);
            this.groupBoxJointOrigin.Controls.Add(this.label56);
            this.groupBoxJointOrigin.Controls.Add(this.label55);
            this.groupBoxJointOrigin.Controls.Add(this.label53);
            this.groupBoxJointOrigin.Controls.Add(this.textBoxJointRoll);
            this.groupBoxJointOrigin.Controls.Add(this.label52);
            this.groupBoxJointOrigin.Controls.Add(this.textBoxJointPitch);
            this.groupBoxJointOrigin.Controls.Add(this.textBoxJointYaw);
            this.groupBoxJointOrigin.Controls.Add(this.label51);
            this.groupBoxJointOrigin.Controls.Add(this.textBoxJointX);
            this.groupBoxJointOrigin.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.groupBoxJointOrigin.Location = new System.Drawing.Point(714, 419);
            this.groupBoxJointOrigin.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBoxJointOrigin.Name = "groupBoxJointOrigin";
            this.groupBoxJointOrigin.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBoxJointOrigin.Size = new System.Drawing.Size(351, 208);
            this.groupBoxJointOrigin.TabIndex = 2;
            this.groupBoxJointOrigin.TabStop = false;
            this.groupBoxJointOrigin.Text = "Origin* (m, rad)";
            // 
            // textBoxJointY
            // 
            this.textBoxJointY.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxJointY.Location = new System.Drawing.Point(45, 81);
            this.textBoxJointY.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxJointY.Name = "textBoxJointY";
            this.textBoxJointY.Size = new System.Drawing.Size(97, 29);
            this.textBoxJointY.TabIndex = 1;
            // 
            // label57
            // 
            this.label57.AutoSize = true;
            this.label57.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label57.Location = new System.Drawing.Point(16, 124);
            this.label57.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label57.Name = "label57";
            this.label57.Size = new System.Drawing.Size(16, 24);
            this.label57.TabIndex = 251;
            this.label57.Text = "Z";
            this.label57.UseCompatibleTextRendering = true;
            // 
            // textBoxJointZ
            // 
            this.textBoxJointZ.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxJointZ.Location = new System.Drawing.Point(45, 121);
            this.textBoxJointZ.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxJointZ.Name = "textBoxJointZ";
            this.textBoxJointZ.Size = new System.Drawing.Size(97, 29);
            this.textBoxJointZ.TabIndex = 2;
            // 
            // label56
            // 
            this.label56.AutoSize = true;
            this.label56.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label56.Location = new System.Drawing.Point(169, 124);
            this.label56.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label56.Name = "label56";
            this.label56.Size = new System.Drawing.Size(38, 24);
            this.label56.TabIndex = 255;
            this.label56.Text = "Yaw";
            this.label56.UseCompatibleTextRendering = true;
            // 
            // label55
            // 
            this.label55.AutoSize = true;
            this.label55.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label55.Location = new System.Drawing.Point(164, 85);
            this.label55.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label55.Name = "label55";
            this.label55.Size = new System.Drawing.Size(43, 24);
            this.label55.TabIndex = 254;
            this.label55.Text = "Pitch";
            this.label55.UseCompatibleTextRendering = true;
            // 
            // label53
            // 
            this.label53.AutoSize = true;
            this.label53.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label53.Location = new System.Drawing.Point(16, 86);
            this.label53.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label53.Name = "label53";
            this.label53.Size = new System.Drawing.Size(17, 24);
            this.label53.TabIndex = 250;
            this.label53.Text = "Y";
            this.label53.UseCompatibleTextRendering = true;
            // 
            // textBoxJointRoll
            // 
            this.textBoxJointRoll.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxJointRoll.Location = new System.Drawing.Point(217, 41);
            this.textBoxJointRoll.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxJointRoll.Name = "textBoxJointRoll";
            this.textBoxJointRoll.Size = new System.Drawing.Size(97, 29);
            this.textBoxJointRoll.TabIndex = 3;
            // 
            // label52
            // 
            this.label52.AutoSize = true;
            this.label52.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label52.Location = new System.Drawing.Point(16, 44);
            this.label52.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label52.Name = "label52";
            this.label52.Size = new System.Drawing.Size(17, 24);
            this.label52.TabIndex = 249;
            this.label52.Text = "X";
            this.label52.UseCompatibleTextRendering = true;
            // 
            // textBoxJointPitch
            // 
            this.textBoxJointPitch.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxJointPitch.Location = new System.Drawing.Point(217, 81);
            this.textBoxJointPitch.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxJointPitch.Name = "textBoxJointPitch";
            this.textBoxJointPitch.Size = new System.Drawing.Size(97, 29);
            this.textBoxJointPitch.TabIndex = 4;
            // 
            // textBoxJointYaw
            // 
            this.textBoxJointYaw.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxJointYaw.Location = new System.Drawing.Point(217, 121);
            this.textBoxJointYaw.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxJointYaw.Name = "textBoxJointYaw";
            this.textBoxJointYaw.Size = new System.Drawing.Size(97, 29);
            this.textBoxJointYaw.TabIndex = 5;
            // 
            // label51
            // 
            this.label51.AutoSize = true;
            this.label51.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label51.Location = new System.Drawing.Point(173, 44);
            this.label51.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label51.Name = "label51";
            this.label51.Size = new System.Drawing.Size(34, 24);
            this.label51.TabIndex = 253;
            this.label51.Text = "Roll";
            this.label51.UseCompatibleTextRendering = true;
            // 
            // textBoxJointX
            // 
            this.textBoxJointX.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxJointX.Location = new System.Drawing.Point(45, 41);
            this.textBoxJointX.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxJointX.Name = "textBoxJointX";
            this.textBoxJointX.Size = new System.Drawing.Size(97, 29);
            this.textBoxJointX.TabIndex = 0;
            // 
            // groupBox12
            // 
            this.groupBox12.Controls.Add(this.textBoxAxisZ);
            this.groupBox12.Controls.Add(this.label61);
            this.groupBox12.Controls.Add(this.label59);
            this.groupBox12.Controls.Add(this.textBoxAxisY);
            this.groupBox12.Controls.Add(this.label58);
            this.groupBox12.Controls.Add(this.textBoxAxisX);
            this.groupBox12.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.groupBox12.Location = new System.Drawing.Point(1071, 419);
            this.groupBox12.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox12.Name = "groupBox12";
            this.groupBox12.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox12.Size = new System.Drawing.Size(190, 208);
            this.groupBox12.TabIndex = 3;
            this.groupBox12.TabStop = false;
            this.groupBox12.Text = "Axis* (normalized)";
            // 
            // textBoxAxisZ
            // 
            this.textBoxAxisZ.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxAxisZ.Location = new System.Drawing.Point(66, 121);
            this.textBoxAxisZ.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxAxisZ.Name = "textBoxAxisZ";
            this.textBoxAxisZ.Size = new System.Drawing.Size(97, 29);
            this.textBoxAxisZ.TabIndex = 2;
            // 
            // label61
            // 
            this.label61.AutoSize = true;
            this.label61.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label61.Location = new System.Drawing.Point(34, 128);
            this.label61.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label61.Name = "label61";
            this.label61.Size = new System.Drawing.Size(16, 24);
            this.label61.TabIndex = 264;
            this.label61.Text = "Z";
            this.label61.UseCompatibleTextRendering = true;
            // 
            // label59
            // 
            this.label59.AutoSize = true;
            this.label59.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label59.Location = new System.Drawing.Point(34, 88);
            this.label59.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label59.Name = "label59";
            this.label59.Size = new System.Drawing.Size(17, 24);
            this.label59.TabIndex = 263;
            this.label59.Text = "Y";
            this.label59.UseCompatibleTextRendering = true;
            // 
            // textBoxAxisY
            // 
            this.textBoxAxisY.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxAxisY.Location = new System.Drawing.Point(66, 81);
            this.textBoxAxisY.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxAxisY.Name = "textBoxAxisY";
            this.textBoxAxisY.Size = new System.Drawing.Size(97, 29);
            this.textBoxAxisY.TabIndex = 1;
            // 
            // label58
            // 
            this.label58.AutoSize = true;
            this.label58.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label58.Location = new System.Drawing.Point(33, 44);
            this.label58.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label58.Name = "label58";
            this.label58.Size = new System.Drawing.Size(17, 24);
            this.label58.TabIndex = 262;
            this.label58.Text = "X";
            this.label58.UseCompatibleTextRendering = true;
            // 
            // textBoxAxisX
            // 
            this.textBoxAxisX.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxAxisX.Location = new System.Drawing.Point(66, 41);
            this.textBoxAxisX.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxAxisX.Name = "textBoxAxisX";
            this.textBoxAxisX.Size = new System.Drawing.Size(97, 29);
            this.textBoxAxisX.TabIndex = 0;
            // 
            // groupBox13
            // 
            this.groupBox13.Controls.Add(this.numericUpDownLimitUpper);
            this.groupBox13.Controls.Add(this.numericUpDownLimitLower);
            this.groupBox13.Controls.Add(this.labelEffort);
            this.groupBox13.Controls.Add(this.textBoxLimitEffort);
            this.groupBox13.Controls.Add(this.labelLimitUpper);
            this.groupBox13.Controls.Add(this.labelLowerLimit);
            this.groupBox13.Controls.Add(this.labelVelocity);
            this.groupBox13.Controls.Add(this.textBoxLimitVelocity);
            this.groupBox13.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.groupBox13.Location = new System.Drawing.Point(1270, 419);
            this.groupBox13.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox13.Name = "groupBox13";
            this.groupBox13.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox13.Size = new System.Drawing.Size(291, 208);
            this.groupBox13.TabIndex = 4;
            this.groupBox13.TabStop = false;
            this.groupBox13.Text = "Limits**";
            // 
            // numericUpDownLimitUpper
            // 
            this.numericUpDownLimitUpper.DecimalPlaces = 10;
            this.numericUpDownLimitUpper.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.numericUpDownLimitUpper.Increment = new decimal(new int[] {
            872665,
            0,
            0,
            458752});
            this.numericUpDownLimitUpper.Location = new System.Drawing.Point(184, 71);
            this.numericUpDownLimitUpper.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.numericUpDownLimitUpper.Maximum = new decimal(new int[] {
            -1,
            -1,
            -1,
            0});
            this.numericUpDownLimitUpper.Minimum = new decimal(new int[] {
            -1,
            -1,
            -1,
            -2147483648});
            this.numericUpDownLimitUpper.Name = "numericUpDownLimitUpper";
            this.numericUpDownLimitUpper.Size = new System.Drawing.Size(97, 29);
            this.numericUpDownLimitUpper.TabIndex = 282;
            // 
            // numericUpDownLimitLower
            // 
            this.numericUpDownLimitLower.DecimalPlaces = 10;
            this.numericUpDownLimitLower.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.numericUpDownLimitLower.Increment = new decimal(new int[] {
            872665,
            0,
            0,
            458752});
            this.numericUpDownLimitLower.Location = new System.Drawing.Point(184, 31);
            this.numericUpDownLimitLower.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.numericUpDownLimitLower.Maximum = new decimal(new int[] {
            -1,
            -1,
            -1,
            0});
            this.numericUpDownLimitLower.Minimum = new decimal(new int[] {
            -1,
            -1,
            -1,
            -2147483648});
            this.numericUpDownLimitLower.Name = "numericUpDownLimitLower";
            this.numericUpDownLimitLower.Size = new System.Drawing.Size(97, 29);
            this.numericUpDownLimitLower.TabIndex = 281;
            // 
            // labelEffort
            // 
            this.labelEffort.AutoSize = true;
            this.labelEffort.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelEffort.Location = new System.Drawing.Point(74, 114);
            this.labelEffort.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelEffort.Name = "labelEffort";
            this.labelEffort.Size = new System.Drawing.Size(92, 24);
            this.labelEffort.TabIndex = 277;
            this.labelEffort.Text = "Effort (N-m)";
            this.labelEffort.TextAlign = System.Drawing.ContentAlignment.TopRight;
            this.labelEffort.UseCompatibleTextRendering = true;
            // 
            // textBoxLimitEffort
            // 
            this.textBoxLimitEffort.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxLimitEffort.Location = new System.Drawing.Point(187, 110);
            this.textBoxLimitEffort.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxLimitEffort.Name = "textBoxLimitEffort";
            this.textBoxLimitEffort.Size = new System.Drawing.Size(97, 29);
            this.textBoxLimitEffort.TabIndex = 2;
            // 
            // labelLimitUpper
            // 
            this.labelLimitUpper.AutoSize = true;
            this.labelLimitUpper.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelLimitUpper.Location = new System.Drawing.Point(76, 74);
            this.labelLimitUpper.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelLimitUpper.Name = "labelLimitUpper";
            this.labelLimitUpper.Size = new System.Drawing.Size(90, 24);
            this.labelLimitUpper.TabIndex = 276;
            this.labelLimitUpper.Text = "Upper (rad)";
            this.labelLimitUpper.TextAlign = System.Drawing.ContentAlignment.TopRight;
            this.labelLimitUpper.UseCompatibleTextRendering = true;
            // 
            // labelLowerLimit
            // 
            this.labelLowerLimit.AutoSize = true;
            this.labelLowerLimit.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelLowerLimit.Location = new System.Drawing.Point(76, 34);
            this.labelLowerLimit.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelLowerLimit.Name = "labelLowerLimit";
            this.labelLowerLimit.Size = new System.Drawing.Size(90, 24);
            this.labelLowerLimit.TabIndex = 275;
            this.labelLowerLimit.Text = "Lower (rad)";
            this.labelLowerLimit.TextAlign = System.Drawing.ContentAlignment.TopRight;
            this.labelLowerLimit.UseCompatibleTextRendering = true;
            // 
            // labelVelocity
            // 
            this.labelVelocity.AutoSize = true;
            this.labelVelocity.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelVelocity.Location = new System.Drawing.Point(60, 152);
            this.labelVelocity.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelVelocity.Name = "labelVelocity";
            this.labelVelocity.Size = new System.Drawing.Size(106, 24);
            this.labelVelocity.TabIndex = 280;
            this.labelVelocity.Text = "Velocity (m/s)";
            this.labelVelocity.TextAlign = System.Drawing.ContentAlignment.TopRight;
            this.labelVelocity.UseCompatibleTextRendering = true;
            // 
            // textBoxLimitVelocity
            // 
            this.textBoxLimitVelocity.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxLimitVelocity.Location = new System.Drawing.Point(187, 150);
            this.textBoxLimitVelocity.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxLimitVelocity.Name = "textBoxLimitVelocity";
            this.textBoxLimitVelocity.Size = new System.Drawing.Size(97, 29);
            this.textBoxLimitVelocity.TabIndex = 3;
            // 
            // groupBox14
            // 
            this.groupBox14.Controls.Add(this.textBoxFriction);
            this.groupBox14.Controls.Add(this.labelDamping);
            this.groupBox14.Controls.Add(this.textBoxDamping);
            this.groupBox14.Controls.Add(this.labelFriction);
            this.groupBox14.Font = new System.Drawing.Font("Segoe UI Variable Display", 9F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.groupBox14.Location = new System.Drawing.Point(714, 635);
            this.groupBox14.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox14.Name = "groupBox14";
            this.groupBox14.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox14.Size = new System.Drawing.Size(351, 152);
            this.groupBox14.TabIndex = 5;
            this.groupBox14.TabStop = false;
            this.groupBox14.Text = "Dynamics";
            // 
            // textBoxFriction
            // 
            this.textBoxFriction.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxFriction.Location = new System.Drawing.Point(236, 50);
            this.textBoxFriction.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxFriction.Name = "textBoxFriction";
            this.textBoxFriction.Size = new System.Drawing.Size(97, 31);
            this.textBoxFriction.TabIndex = 0;
            // 
            // labelDamping
            // 
            this.labelDamping.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelDamping.Location = new System.Drawing.Point(20, 108);
            this.labelDamping.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelDamping.Name = "labelDamping";
            this.labelDamping.Size = new System.Drawing.Size(208, 28);
            this.labelDamping.TabIndex = 289;
            this.labelDamping.Text = "Damping (N*m*s*rad^-1)";
            this.labelDamping.TextAlign = System.Drawing.ContentAlignment.TopRight;
            this.labelDamping.UseCompatibleTextRendering = true;
            // 
            // textBoxDamping
            // 
            this.textBoxDamping.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxDamping.Location = new System.Drawing.Point(236, 101);
            this.textBoxDamping.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.textBoxDamping.Name = "textBoxDamping";
            this.textBoxDamping.Size = new System.Drawing.Size(97, 31);
            this.textBoxDamping.TabIndex = 1;
            // 
            // labelFriction
            // 
            this.labelFriction.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelFriction.Location = new System.Drawing.Point(45, 58);
            this.labelFriction.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelFriction.Name = "labelFriction";
            this.labelFriction.Size = new System.Drawing.Size(183, 24);
            this.labelFriction.TabIndex = 288;
            this.labelFriction.Text = "Friction (N*m)";
            this.labelFriction.TextAlign = System.Drawing.ContentAlignment.TopRight;
            this.labelFriction.UseCompatibleTextRendering = true;
            // 
            // tabControl
            // 
            this.tabControl.Controls.Add(this.tabPageJointProperties);
            this.tabControl.Controls.Add(this.tabPageLinkProperties);
            this.tabControl.Controls.Add(this.tabPageKinematicsSummary);
            this.tabControl.Controls.Add(this.tabPageTendonsSummary);
            this.tabControl.Font = new System.Drawing.Font("Segoe UI", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.tabControl.Location = new System.Drawing.Point(12, 52);
            this.tabControl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.tabControl.Name = "tabControl";
            this.tabControl.SelectedIndex = 0;
            this.tabControl.Size = new System.Drawing.Size(1587, 1041);
            this.tabControl.TabIndex = 0;
            // 
            // tabPageKinematicsSummary
            // 
            this.tabPageKinematicsSummary.Controls.Add(this.checkBoxSummaryDegrees);
            this.tabPageKinematicsSummary.Controls.Add(this.dataGridViewKinematicsSummary);
            this.tabPageKinematicsSummary.Controls.Add(this.label43);
            this.tabPageKinematicsSummary.Controls.Add(this.label44);
            this.tabPageKinematicsSummary.Location = new System.Drawing.Point(4, 30);
            this.tabPageKinematicsSummary.Name = "tabPageKinematicsSummary";
            this.tabPageKinematicsSummary.Size = new System.Drawing.Size(1579, 1007);
            this.tabPageKinematicsSummary.TabIndex = 2;
            this.tabPageKinematicsSummary.Text = "Kinematics Summary";
            this.tabPageKinematicsSummary.UseVisualStyleBackColor = true;
            // 
            // checkBoxSummaryDegrees
            // 
            this.checkBoxSummaryDegrees.AutoSize = true;
            this.checkBoxSummaryDegrees.Checked = true;
            this.checkBoxSummaryDegrees.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkBoxSummaryDegrees.Location = new System.Drawing.Point(1448, 45);
            this.checkBoxSummaryDegrees.Name = "checkBoxSummaryDegrees";
            this.checkBoxSummaryDegrees.Size = new System.Drawing.Size(123, 25);
            this.checkBoxSummaryDegrees.TabIndex = 106;
            this.checkBoxSummaryDegrees.Text = "Use Degrees";
            this.checkBoxSummaryDegrees.UseVisualStyleBackColor = true;
            // 
            // dataGridViewKinematicsSummary
            // 
            this.dataGridViewKinematicsSummary.AllowUserToOrderColumns = true;
            this.dataGridViewKinematicsSummary.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.dataGridViewKinematicsSummary.Location = new System.Drawing.Point(14, 91);
            this.dataGridViewKinematicsSummary.Name = "dataGridViewKinematicsSummary";
            this.dataGridViewKinematicsSummary.RowHeadersWidth = 62;
            this.dataGridViewKinematicsSummary.RowTemplate.Height = 28;
            this.dataGridViewKinematicsSummary.Size = new System.Drawing.Size(1557, 903);
            this.dataGridViewKinematicsSummary.TabIndex = 105;
            // 
            // label43
            // 
            this.label43.AutoSize = true;
            this.label43.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label43.Location = new System.Drawing.Point(10, 45);
            this.label43.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label43.Name = "label43";
            this.label43.Size = new System.Drawing.Size(746, 42);
            this.label43.TabIndex = 104;
            this.label43.Text = "This page is a summary of Link and Joint properties, to edit values, please use t" +
    "heir respective tabs. \r\nDouble-clicking a cell will switch to the Joint or Link " +
    "tab for editing.\r\n";
            this.label43.UseCompatibleTextRendering = true;
            // 
            // label44
            // 
            this.label44.AutoSize = true;
            this.label44.Font = new System.Drawing.Font("Microsoft Sans Serif", 10F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label44.Location = new System.Drawing.Point(9, 18);
            this.label44.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label44.Name = "label44";
            this.label44.Size = new System.Drawing.Size(261, 29);
            this.label44.TabIndex = 103;
            this.label44.Text = "Kinematics Summary View";
            this.label44.UseCompatibleTextRendering = true;
            // 
            // tabPageTendonsSummary
            // 
            this.tabPageTendonsSummary.Controls.Add(this.groupBox15);
            this.tabPageTendonsSummary.Controls.Add(this.dataGridViewTendonsSummary);
            this.tabPageTendonsSummary.Controls.Add(this.label54);
            this.tabPageTendonsSummary.Controls.Add(this.label68);
            this.tabPageTendonsSummary.Location = new System.Drawing.Point(4, 30);
            this.tabPageTendonsSummary.Name = "tabPageTendonsSummary";
            this.tabPageTendonsSummary.Padding = new System.Windows.Forms.Padding(3);
            this.tabPageTendonsSummary.Size = new System.Drawing.Size(1579, 1007);
            this.tabPageTendonsSummary.TabIndex = 3;
            this.tabPageTendonsSummary.Text = "Tendons Summary";
            this.tabPageTendonsSummary.UseVisualStyleBackColor = true;
            // 
            // groupBox15
            // 
            this.groupBox15.Controls.Add(this.checkBoxShowAllTendons);
            this.groupBox15.Controls.Add(this.checkBoxShowTendonVisualization);
            this.groupBox15.Font = new System.Drawing.Font("Segoe UI Variable Display", 9F, System.Drawing.FontStyle.Bold);
            this.groupBox15.Location = new System.Drawing.Point(1135, 11);
            this.groupBox15.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox15.Name = "groupBox15";
            this.groupBox15.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox15.Size = new System.Drawing.Size(437, 70);
            this.groupBox15.TabIndex = 313;
            this.groupBox15.TabStop = false;
            this.groupBox15.Text = "Visualization";
            // 
            // checkBoxShowAllTendons
            // 
            this.checkBoxShowAllTendons.AutoSize = true;
            this.checkBoxShowAllTendons.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxShowAllTendons.Location = new System.Drawing.Point(225, 34);
            this.checkBoxShowAllTendons.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxShowAllTendons.Name = "checkBoxShowAllTendons";
            this.checkBoxShowAllTendons.Size = new System.Drawing.Size(162, 24);
            this.checkBoxShowAllTendons.TabIndex = 318;
            this.checkBoxShowAllTendons.Text = "Show All Tendons";
            this.checkBoxShowAllTendons.UseVisualStyleBackColor = true;
            // 
            // checkBoxShowTendonVisualization
            // 
            this.checkBoxShowTendonVisualization.AutoSize = true;
            this.checkBoxShowTendonVisualization.Checked = true;
            this.checkBoxShowTendonVisualization.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkBoxShowTendonVisualization.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxShowTendonVisualization.Location = new System.Drawing.Point(17, 33);
            this.checkBoxShowTendonVisualization.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxShowTendonVisualization.Name = "checkBoxShowTendonVisualization";
            this.checkBoxShowTendonVisualization.Size = new System.Drawing.Size(200, 24);
            this.checkBoxShowTendonVisualization.TabIndex = 317;
            this.checkBoxShowTendonVisualization.Text = "Show Selected Tendon";
            this.checkBoxShowTendonVisualization.UseVisualStyleBackColor = true;
            // 
            // dataGridViewTendonsSummary
            // 
            this.dataGridViewTendonsSummary.AllowUserToOrderColumns = true;
            this.dataGridViewTendonsSummary.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.dataGridViewTendonsSummary.Location = new System.Drawing.Point(14, 91);
            this.dataGridViewTendonsSummary.Name = "dataGridViewTendonsSummary";
            this.dataGridViewTendonsSummary.RowHeadersWidth = 62;
            this.dataGridViewTendonsSummary.RowTemplate.Height = 28;
            this.dataGridViewTendonsSummary.Size = new System.Drawing.Size(1557, 903);
            this.dataGridViewTendonsSummary.TabIndex = 107;
            // 
            // label54
            // 
            this.label54.AutoSize = true;
            this.label54.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label54.Location = new System.Drawing.Point(10, 45);
            this.label54.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label54.Name = "label54";
            this.label54.Size = new System.Drawing.Size(733, 24);
            this.label54.TabIndex = 106;
            this.label54.Text = "This page is a sumamry of the Tendons. Values cannot be edited directly, this is " +
    "only for previewing.";
            this.label54.UseCompatibleTextRendering = true;
            // 
            // label68
            // 
            this.label68.AutoSize = true;
            this.label68.Font = new System.Drawing.Font("Microsoft Sans Serif", 10F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label68.Location = new System.Drawing.Point(9, 18);
            this.label68.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label68.Name = "label68";
            this.label68.Size = new System.Drawing.Size(239, 29);
            this.label68.TabIndex = 105;
            this.label68.Text = "Tendons Summary View";
            this.label68.UseCompatibleTextRendering = true;
            // 
            // AssemblyExportForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 20F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1600, 1151);
            this.Controls.Add(this.textBoxRobotName);
            this.Controls.Add(this.label29);
            this.Controls.Add(this.buttonLinksExportUrdfOnly);
            this.Controls.Add(this.tabControl);
            this.Controls.Add(this.buttonNext);
            this.Controls.Add(this.buttonLinksFinish);
            this.Controls.Add(this.buttonClose);
            this.Controls.Add(this.buttonPrevious);
            this.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.MaximizeBox = false;
            this.MaximumSize = new System.Drawing.Size(1622, 1207);
            this.MinimumSize = new System.Drawing.Size(1622, 1207);
            this.Name = "AssemblyExportForm";
            this.ShowIcon = false;
            this.SizeGripStyle = System.Windows.Forms.SizeGripStyle.Hide;
            this.Text = "SuperDex CAD Exporter V2.2.0 - Export";
            this.Load += new System.EventHandler(this.AssemblyExportFormLoad);
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownTargetEdgeLength)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownTargetEdgeLengthCollision)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownEdgeLengthFraction)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownEdgeLengthFractionCollision)).EndInit();
            this.tabPageLinkProperties.ResumeLayout(false);
            this.tabPageLinkProperties.PerformLayout();
            this.groupBox11.ResumeLayout(false);
            this.groupBox11.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLinkGizmoSize)).EndInit();
            this.groupBox5.ResumeLayout(false);
            this.groupBox5.PerformLayout();
            this.groupBox9.ResumeLayout(false);
            this.groupBox9.PerformLayout();
            this.groupBox8.ResumeLayout(false);
            this.groupBox8.PerformLayout();
            this.groupBox4.ResumeLayout(false);
            this.groupBox4.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            this.groupBox7.ResumeLayout(false);
            this.groupBox7.PerformLayout();
            this.groupBox16.ResumeLayout(false);
            this.groupBox16.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownScaleCollision)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownScale)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownAngularDeflectionCollision)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownLinearDeflectionCollision)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownAngularDeflection)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownLinearDeflection)).EndInit();
            this.groupBox6.ResumeLayout(false);
            this.groupBox6.PerformLayout();
            this.groupBox3.ResumeLayout(false);
            this.groupBox3.PerformLayout();
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.tableLayoutPanel1.ResumeLayout(false);
            this.tableLayoutPanel1.PerformLayout();
            this.tabPageJointProperties.ResumeLayout(false);
            this.tabPageJointProperties.PerformLayout();
            this.groupBox18.ResumeLayout(false);
            this.groupBox18.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarJointGizmoSize)).EndInit();
            this.groupBox10.ResumeLayout(false);
            this.groupBox10.PerformLayout();
            this.groupBoxJointOrigin.ResumeLayout(false);
            this.groupBoxJointOrigin.PerformLayout();
            this.groupBox12.ResumeLayout(false);
            this.groupBox12.PerformLayout();
            this.groupBox13.ResumeLayout(false);
            this.groupBox13.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownLimitUpper)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownLimitLower)).EndInit();
            this.groupBox14.ResumeLayout(false);
            this.groupBox14.PerformLayout();
            this.tabControl.ResumeLayout(false);
            this.tabPageKinematicsSummary.ResumeLayout(false);
            this.tabPageKinematicsSummary.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.dataGridViewKinematicsSummary)).EndInit();
            this.tabPageTendonsSummary.ResumeLayout(false);
            this.tabPageTendonsSummary.PerformLayout();
            this.groupBox15.ResumeLayout(false);
            this.groupBox15.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.dataGridViewTendonsSummary)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.SaveFileDialog saveFileDialog1;
        private System.Windows.Forms.Button buttonLinksExportUrdfOnly;
        private System.Windows.Forms.Button buttonLinksFinish;
        private System.Windows.Forms.Button buttonPrevious;
        private System.Windows.Forms.ToolTip toolTips;
        private System.Windows.Forms.Button buttonNext;
        private System.Windows.Forms.Button buttonClose;
        private System.Windows.Forms.Label label29;
        private System.Windows.Forms.TextBox textBoxRobotName;
        private System.Windows.Forms.TabPage tabPageLinkProperties;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.GroupBox groupBox5;
        private System.Windows.Forms.Button buttonRecalculateInertial;
        private System.Windows.Forms.GroupBox groupBox9;
        private System.Windows.Forms.TextBox textBoxIxx;
        private System.Windows.Forms.Label label49;
        private System.Windows.Forms.Label label48;
        private System.Windows.Forms.TextBox textBoxIxy;
        private System.Windows.Forms.TextBox textBoxIyz;
        private System.Windows.Forms.TextBox textBoxIxz;
        private System.Windows.Forms.TextBox textBoxIzz;
        private System.Windows.Forms.Label label18;
        private System.Windows.Forms.Label label11;
        private System.Windows.Forms.Label label50;
        private System.Windows.Forms.Label label14;
        private System.Windows.Forms.TextBox textBoxIyy;
        private System.Windows.Forms.GroupBox groupBox8;
        private System.Windows.Forms.TextBox textBoxInertialOriginX;
        private System.Windows.Forms.Label label16;
        private System.Windows.Forms.TextBox textBoxInertialOriginYaw;
        private System.Windows.Forms.TextBox textBoxInertialOriginPitch;
        private System.Windows.Forms.Label label13;
        private System.Windows.Forms.TextBox textBoxInertialOriginRoll;
        private System.Windows.Forms.TextBox textBoxInertialOriginY;
        private System.Windows.Forms.Label label17;
        private System.Windows.Forms.Label label45;
        private System.Windows.Forms.Label label46;
        private System.Windows.Forms.TextBox textBoxInertialOriginZ;
        private System.Windows.Forms.Label label47;
        private System.Windows.Forms.Label label15;
        private System.Windows.Forms.Label label12;
        private System.Windows.Forms.TextBox textBoxMass;
        private System.Windows.Forms.GroupBox groupBox4;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.Label label34;
        private System.Windows.Forms.ComboBox comboBoxExporterConfigurationPreset;
        private System.Windows.Forms.GroupBox groupBox7;
        private System.Windows.Forms.Button buttonExportLinkMesh;
        private System.Windows.Forms.ComboBox comboBoxBackend;
        private System.Windows.Forms.ComboBox comboBoxBackendCollision;
        private System.Windows.Forms.ComboBox comboBoxEdgeSampling;
        private System.Windows.Forms.ComboBox comboBoxEdgeSamplingCollision;
        private System.Windows.Forms.NumericUpDown numericUpDownTargetEdgeLength;
        private System.Windows.Forms.NumericUpDown numericUpDownTargetEdgeLengthCollision;
        private System.Windows.Forms.NumericUpDown numericUpDownEdgeLengthFraction;
        private System.Windows.Forms.NumericUpDown numericUpDownEdgeLengthFractionCollision;
        private System.Windows.Forms.Label labelBackend;
        private System.Windows.Forms.Label labelEdgeSampling;
        private System.Windows.Forms.Label labelTargetEdgeLength;
        private System.Windows.Forms.Label labelEdgeLengthFraction;
        private System.Windows.Forms.CheckBox checkBoxPerLinkMeshing;
        private System.Windows.Forms.Button buttonResetMeshingToDefaults;
        private System.Windows.Forms.CheckBox checkBoxCollisionMeshing;
        private System.Windows.Forms.NumericUpDown numericUpDownScaleCollision;
        private System.Windows.Forms.NumericUpDown numericUpDownAngularDeflectionCollision;
        private System.Windows.Forms.NumericUpDown numericUpDownLinearDeflectionCollision;
        private System.Windows.Forms.Label label28;
        private System.Windows.Forms.NumericUpDown numericUpDownScale;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.NumericUpDown numericUpDownAngularDeflection;
        private System.Windows.Forms.NumericUpDown numericUpDownLinearDeflection;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.GroupBox groupBox6;
        private System.Windows.Forms.DomainUpDown domainUpDownAlpha;
        private System.Windows.Forms.Label label33;
        private System.Windows.Forms.DomainUpDown domainUpDownRed;
        private System.Windows.Forms.Label label32;
        private System.Windows.Forms.DomainUpDown domainUpDownGreen;
        private System.Windows.Forms.Label label31;
        private System.Windows.Forms.DomainUpDown domainUpDownBlue;
        private System.Windows.Forms.Label label30;
        private System.Windows.Forms.GroupBox groupBox3;
        private System.Windows.Forms.Label label20;
        private System.Windows.Forms.Label label21;
        private System.Windows.Forms.Label label24;
        private System.Windows.Forms.Label label22;
        private System.Windows.Forms.Label label25;
        private System.Windows.Forms.Label label26;
        private System.Windows.Forms.TextBox textBoxVisualOriginZ;
        private System.Windows.Forms.TextBox textBoxVisualOriginYaw;
        private System.Windows.Forms.TextBox textBoxVisualOriginPitch;
        private System.Windows.Forms.TextBox textBoxVisualOriginRoll;
        private System.Windows.Forms.TextBox textBoxVisualOriginX;
        private System.Windows.Forms.TextBox textBoxVisualOriginY;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.TableLayoutPanel tableLayoutPanel1;
        private System.Windows.Forms.Label label60;
        private System.Windows.Forms.Label labelStep;
        private System.Windows.Forms.Label labelObjCAD;
        private System.Windows.Forms.Label labelGlbCAD;
        private System.Windows.Forms.Label labelStlCAD;
        private System.Windows.Forms.Label label41;
        private System.Windows.Forms.Label label40;
        private System.Windows.Forms.CheckBox checkBoxStepVisual;
        private System.Windows.Forms.CheckBox checkBoxObjCADVisual;
        private System.Windows.Forms.CheckBox checkBoxStepCollision;
        private System.Windows.Forms.TextBox textBoxMeshFileExtensionCollision;
        private System.Windows.Forms.CheckBox checkBoxObjCADCollision;
        private System.Windows.Forms.CheckBox checkBoxGlbCADVisual;
        private System.Windows.Forms.CheckBox checkBoxGlbCADCollision;
        private System.Windows.Forms.CheckBox checkBoxStlCADVisual;
        private System.Windows.Forms.CheckBox checkBoxGlbOpenCascadeCollision;
        private System.Windows.Forms.CheckBox checkBoxStlCADCollision;
        private System.Windows.Forms.TextBox textBoxMeshFileExtension;
        private System.Windows.Forms.CheckBox checkBoxStlOpenCascadeVisual;
        private System.Windows.Forms.CheckBox checkBoxStlOpenCascadeCollision;
        private System.Windows.Forms.CheckBox checkBoxObjOpenCascadeVisual;
        private System.Windows.Forms.CheckBox checkBoxObjOpenCascadeCollision;
        private System.Windows.Forms.CheckBox checkBoxGlbOpenCascadeVisual;
        private System.Windows.Forms.Label label23;
        private System.Windows.Forms.Label label38;
        private System.Windows.Forms.Label label39;
        private System.Windows.Forms.Label label37;
        private System.Windows.Forms.LinkLabel linkLabelHelpMeChoose;
        private System.Windows.Forms.Label label19;
        private System.Windows.Forms.TabPage tabPageJointProperties;
        private System.Windows.Forms.Label label35;
        private System.Windows.Forms.GroupBox groupBox18;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.CheckBox checkBoxShowJointVisualization;
        private System.Windows.Forms.TrackBar trackBarJointGizmoSize;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label27;
        private System.Windows.Forms.TreeView treeViewJointTree;
        private System.Windows.Forms.Label label69;
        private System.Windows.Forms.GroupBox groupBox10;
        private System.Windows.Forms.TextBox textBoxParentLink;
        private System.Windows.Forms.TextBox textBoxChildLink;
        private System.Windows.Forms.TextBox textBoxRefAxis;
        private System.Windows.Forms.Label label62;
        private System.Windows.Forms.TextBox textBoxCoordSys;
        private System.Windows.Forms.ComboBox comboBoxJointType;
        private System.Windows.Forms.TextBox textBoxJointName;
        private System.Windows.Forms.Label label63;
        private System.Windows.Forms.Label label64;
        private System.Windows.Forms.Label label65;
        private System.Windows.Forms.Label label67;
        private System.Windows.Forms.Label label66;
        private System.Windows.Forms.GroupBox groupBoxJointOrigin;
        private System.Windows.Forms.TextBox textBoxJointY;
        private System.Windows.Forms.Label label57;
        private System.Windows.Forms.TextBox textBoxJointZ;
        private System.Windows.Forms.Label label56;
        private System.Windows.Forms.Label label55;
        private System.Windows.Forms.Label label53;
        private System.Windows.Forms.TextBox textBoxJointRoll;
        private System.Windows.Forms.Label label52;
        private System.Windows.Forms.TextBox textBoxJointPitch;
        private System.Windows.Forms.TextBox textBoxJointYaw;
        private System.Windows.Forms.Label label51;
        private System.Windows.Forms.TextBox textBoxJointX;
        private System.Windows.Forms.GroupBox groupBox12;
        private System.Windows.Forms.TextBox textBoxAxisZ;
        private System.Windows.Forms.Label label61;
        private System.Windows.Forms.Label label59;
        private System.Windows.Forms.TextBox textBoxAxisY;
        private System.Windows.Forms.Label label58;
        private System.Windows.Forms.TextBox textBoxAxisX;
        private System.Windows.Forms.GroupBox groupBox13;
        private System.Windows.Forms.NumericUpDown numericUpDownLimitUpper;
        private System.Windows.Forms.NumericUpDown numericUpDownLimitLower;
        private System.Windows.Forms.Label labelEffort;
        private System.Windows.Forms.TextBox textBoxLimitEffort;
        private System.Windows.Forms.Label labelLimitUpper;
        private System.Windows.Forms.Label labelLowerLimit;
        private System.Windows.Forms.Label labelVelocity;
        private System.Windows.Forms.TextBox textBoxLimitVelocity;
        private System.Windows.Forms.GroupBox groupBox14;
        private System.Windows.Forms.TextBox textBoxFriction;
        private System.Windows.Forms.Label labelDamping;
        private System.Windows.Forms.TextBox textBoxDamping;
        private System.Windows.Forms.Label labelFriction;
        private System.Windows.Forms.TabControl tabControl;
        private System.Windows.Forms.CheckBox checkBoxUseDegrees;
        private System.Windows.Forms.ComboBox comboBoxFolderStructure;
        private System.Windows.Forms.Label label42;
        private System.Windows.Forms.TabPage tabPageKinematicsSummary;
        private System.Windows.Forms.Label label43;
        private System.Windows.Forms.Label label44;
        private System.Windows.Forms.DataGridView dataGridViewKinematicsSummary;
        private System.Windows.Forms.CheckBox checkBoxSummaryDegrees;
        private System.Windows.Forms.TabPage tabPageTendonsSummary;
        private System.Windows.Forms.Label label54;
        private System.Windows.Forms.Label label68;
        private System.Windows.Forms.DataGridView dataGridViewTendonsSummary;
        private System.Windows.Forms.GroupBox groupBox11;
        private System.Windows.Forms.CheckBox checkBoxShowLinkVisualization;
        private System.Windows.Forms.GroupBox groupBox15;
        private System.Windows.Forms.CheckBox checkBoxShowTendonVisualization;
        private System.Windows.Forms.CheckBox checkBoxShowJointHighlights;
        private System.Windows.Forms.Label label70;
        private System.Windows.Forms.TrackBar trackBarLinkGizmoSize;
        private System.Windows.Forms.CheckBox checkBoxLinkHighlights;
        private System.Windows.Forms.CheckBox checkBoxShowAllTendons;
        private System.Windows.Forms.TreeView treeViewLinkProperties;
        private System.Windows.Forms.GroupBox groupBox16;
    }
}
