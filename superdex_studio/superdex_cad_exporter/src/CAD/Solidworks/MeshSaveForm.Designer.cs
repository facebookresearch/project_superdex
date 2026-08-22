/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

namespace CADRobotExporter.UI
{
    partial class MeshSaveForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
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
            this.comboBoxCoordinateSystem = new System.Windows.Forms.ComboBox();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.checkBoxGlb = new System.Windows.Forms.CheckBox();
            this.checkBoxStl = new System.Windows.Forms.CheckBox();
            this.checkBoxObj = new System.Windows.Forms.CheckBox();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.numericUpDownScale = new System.Windows.Forms.NumericUpDown();
            this.numericUpDownAngularDeflection = new System.Windows.Forms.NumericUpDown();
            this.numericUpDownLinearDeflection = new System.Windows.Forms.NumericUpDown();
            this.label6 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.labelBackend = new System.Windows.Forms.Label();
            this.comboBoxBackend = new System.Windows.Forms.ComboBox();
            this.labelEdgeSampling = new System.Windows.Forms.Label();
            this.comboBoxEdgeSampling = new System.Windows.Forms.ComboBox();
            this.labelTargetEdgeLength = new System.Windows.Forms.Label();
            this.numericUpDownTargetEdgeLength = new System.Windows.Forms.NumericUpDown();
            this.labelEdgeLengthFraction = new System.Windows.Forms.Label();
            this.numericUpDownEdgeLengthFraction = new System.Windows.Forms.NumericUpDown();
            this.label2 = new System.Windows.Forms.Label();
            this.buttonSave = new System.Windows.Forms.Button();
            this.buttonCancel = new System.Windows.Forms.Button();
            this.checkBoxAlwaysOverwrite = new System.Windows.Forms.CheckBox();
            this.checkBoxStep = new System.Windows.Forms.CheckBox();
            this.toolTip1 = new System.Windows.Forms.ToolTip(this.components);
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownScale)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownAngularDeflection)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownLinearDeflection)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownTargetEdgeLength)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownEdgeLengthFraction)).BeginInit();
            this.SuspendLayout();
            // 
            // comboBoxCoordinateSystem
            // 
            this.comboBoxCoordinateSystem.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxCoordinateSystem.FormattingEnabled = true;
            this.comboBoxCoordinateSystem.Items.AddRange(new object[] {
            "-- default --"});
            this.comboBoxCoordinateSystem.Location = new System.Drawing.Point(220, 386);
            this.comboBoxCoordinateSystem.Name = "comboBoxCoordinateSystem";
            this.comboBoxCoordinateSystem.Size = new System.Drawing.Size(299, 28);
            this.comboBoxCoordinateSystem.TabIndex = 0;
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.checkBoxStep);
            this.groupBox1.Controls.Add(this.checkBoxGlb);
            this.groupBox1.Controls.Add(this.checkBoxStl);
            this.groupBox1.Controls.Add(this.checkBoxObj);
            this.groupBox1.Location = new System.Drawing.Point(12, 12);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(134, 355);
            this.groupBox1.TabIndex = 0;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Export Format";
            // 
            // checkBoxGlb
            // 
            this.checkBoxGlb.AutoSize = true;
            this.checkBoxGlb.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxGlb.Location = new System.Drawing.Point(23, 29);
            this.checkBoxGlb.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxGlb.Name = "checkBoxGlb";
            this.checkBoxGlb.Size = new System.Drawing.Size(68, 24);
            this.checkBoxGlb.TabIndex = 0;
            this.checkBoxGlb.Text = "GLB";
            this.checkBoxGlb.UseVisualStyleBackColor = true;
            // 
            // checkBoxStl
            // 
            this.checkBoxStl.AutoSize = true;
            this.checkBoxStl.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxStl.Location = new System.Drawing.Point(23, 113);
            this.checkBoxStl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxStl.Name = "checkBoxStl";
            this.checkBoxStl.Size = new System.Drawing.Size(64, 24);
            this.checkBoxStl.TabIndex = 2;
            this.checkBoxStl.Text = "STL";
            this.checkBoxStl.UseVisualStyleBackColor = true;
            // 
            // checkBoxObj
            // 
            this.checkBoxObj.AutoSize = true;
            this.checkBoxObj.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxObj.Location = new System.Drawing.Point(23, 71);
            this.checkBoxObj.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxObj.Name = "checkBoxObj";
            this.checkBoxObj.Size = new System.Drawing.Size(66, 24);
            this.checkBoxObj.TabIndex = 1;
            this.checkBoxObj.Text = "OBJ";
            this.checkBoxObj.UseVisualStyleBackColor = true;
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.numericUpDownScale);
            this.groupBox2.Controls.Add(this.numericUpDownAngularDeflection);
            this.groupBox2.Controls.Add(this.numericUpDownLinearDeflection);
            this.groupBox2.Controls.Add(this.label6);
            this.groupBox2.Controls.Add(this.label3);
            this.groupBox2.Controls.Add(this.label1);
            this.groupBox2.Controls.Add(this.labelBackend);
            this.groupBox2.Controls.Add(this.comboBoxBackend);
            this.groupBox2.Controls.Add(this.labelEdgeSampling);
            this.groupBox2.Controls.Add(this.comboBoxEdgeSampling);
            this.groupBox2.Controls.Add(this.labelTargetEdgeLength);
            this.groupBox2.Controls.Add(this.numericUpDownTargetEdgeLength);
            this.groupBox2.Controls.Add(this.labelEdgeLengthFraction);
            this.groupBox2.Controls.Add(this.numericUpDownEdgeLengthFraction);
            this.groupBox2.Location = new System.Drawing.Point(152, 12);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(367, 355);
            this.groupBox2.TabIndex = 2;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Meshing Options";
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
            this.numericUpDownScale.Location = new System.Drawing.Point(230, 113);
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
            this.numericUpDownScale.TabIndex = 2;
            this.numericUpDownScale.Value = new decimal(new int[] {
            10,
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
            this.numericUpDownAngularDeflection.Location = new System.Drawing.Point(230, 70);
            this.numericUpDownAngularDeflection.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.numericUpDownAngularDeflection.Maximum = new decimal(new int[] {
            180,
            0,
            0,
            0});
            this.numericUpDownAngularDeflection.Name = "numericUpDownAngularDeflection";
            this.numericUpDownAngularDeflection.Size = new System.Drawing.Size(120, 29);
            this.numericUpDownAngularDeflection.TabIndex = 1;
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
            this.numericUpDownLinearDeflection.Location = new System.Drawing.Point(230, 27);
            this.numericUpDownLinearDeflection.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.numericUpDownLinearDeflection.Name = "numericUpDownLinearDeflection";
            this.numericUpDownLinearDeflection.Size = new System.Drawing.Size(120, 29);
            this.numericUpDownLinearDeflection.TabIndex = 0;
            this.numericUpDownLinearDeflection.Value = new decimal(new int[] {
            1,
            0,
            0,
            65536});
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label6.Location = new System.Drawing.Point(23, 115);
            this.label6.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(129, 24);
            this.label6.TabIndex = 59;
            this.label6.Text = "Scale (multiplier)";
            this.label6.UseCompatibleTextRendering = true;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label3.Location = new System.Drawing.Point(23, 73);
            this.label3.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(180, 24);
            this.label3.TabIndex = 58;
            this.label3.Text = "Angular Deflection (rad)";
            this.label3.UseCompatibleTextRendering = true;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.Location = new System.Drawing.Point(23, 29);
            this.label1.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(172, 24);
            this.label1.TabIndex = 57;
            this.label1.Text = "Linear Deflection (mm)";
            this.label1.UseCompatibleTextRendering = true;
            // 
            // labelBackend
            // 
            this.labelBackend.AutoSize = true;
            this.labelBackend.Location = new System.Drawing.Point(23, 159);
            this.labelBackend.Name = "labelBackend";
            this.labelBackend.Size = new System.Drawing.Size(62, 20);
            this.labelBackend.TabIndex = 30;
            this.labelBackend.Text = "Mesher";
            // 
            // comboBoxBackend
            // 
            this.comboBoxBackend.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxBackend.FormattingEnabled = true;
            this.comboBoxBackend.Items.AddRange(new object[] {
            "Isotropic",
            "Delabella"});
            this.comboBoxBackend.Location = new System.Drawing.Point(230, 156);
            this.comboBoxBackend.Name = "comboBoxBackend";
            this.comboBoxBackend.Size = new System.Drawing.Size(120, 28);
            this.comboBoxBackend.TabIndex = 31;
            // 
            // labelEdgeSampling
            // 
            this.labelEdgeSampling.AutoSize = true;
            this.labelEdgeSampling.Location = new System.Drawing.Point(23, 202);
            this.labelEdgeSampling.Name = "labelEdgeSampling";
            this.labelEdgeSampling.Size = new System.Drawing.Size(117, 20);
            this.labelEdgeSampling.TabIndex = 32;
            this.labelEdgeSampling.Text = "Edge Sampling";
            // 
            // comboBoxEdgeSampling
            // 
            this.comboBoxEdgeSampling.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxEdgeSampling.FormattingEnabled = true;
            this.comboBoxEdgeSampling.Items.AddRange(new object[] {
            "Adaptive",
            "Uniform"});
            this.comboBoxEdgeSampling.Location = new System.Drawing.Point(230, 199);
            this.comboBoxEdgeSampling.Name = "comboBoxEdgeSampling";
            this.comboBoxEdgeSampling.Size = new System.Drawing.Size(120, 28);
            this.comboBoxEdgeSampling.TabIndex = 33;
            // 
            // labelTargetEdgeLength
            // 
            this.labelTargetEdgeLength.AutoSize = true;
            this.labelTargetEdgeLength.Location = new System.Drawing.Point(23, 245);
            this.labelTargetEdgeLength.Name = "labelTargetEdgeLength";
            this.labelTargetEdgeLength.Size = new System.Drawing.Size(195, 20);
            this.labelTargetEdgeLength.TabIndex = 34;
            this.labelTargetEdgeLength.Text = "Edge Length mm (0=auto)";
            // 
            // numericUpDownTargetEdgeLength
            // 
            this.numericUpDownTargetEdgeLength.DecimalPlaces = 3;
            this.numericUpDownTargetEdgeLength.Increment = new decimal(new int[] {
            1,
            0,
            0,
            65536});
            this.numericUpDownTargetEdgeLength.Location = new System.Drawing.Point(230, 243);
            this.numericUpDownTargetEdgeLength.Maximum = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            this.numericUpDownTargetEdgeLength.Name = "numericUpDownTargetEdgeLength";
            this.numericUpDownTargetEdgeLength.Size = new System.Drawing.Size(120, 26);
            this.numericUpDownTargetEdgeLength.TabIndex = 35;
            // 
            // labelEdgeLengthFraction
            // 
            this.labelEdgeLengthFraction.AutoSize = true;
            this.labelEdgeLengthFraction.Location = new System.Drawing.Point(23, 288);
            this.labelEdgeLengthFraction.Name = "labelEdgeLengthFraction";
            this.labelEdgeLengthFraction.Size = new System.Drawing.Size(163, 20);
            this.labelEdgeLengthFraction.TabIndex = 36;
            this.labelEdgeLengthFraction.Text = "Edge Length Fraction";
            // 
            // numericUpDownEdgeLengthFraction
            // 
            this.numericUpDownEdgeLengthFraction.DecimalPlaces = 4;
            this.numericUpDownEdgeLengthFraction.Increment = new decimal(new int[] {
            5,
            0,
            0,
            196608});
            this.numericUpDownEdgeLengthFraction.Location = new System.Drawing.Point(230, 286);
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
            this.numericUpDownEdgeLengthFraction.Size = new System.Drawing.Size(120, 26);
            this.numericUpDownEdgeLengthFraction.TabIndex = 37;
            this.numericUpDownEdgeLengthFraction.Value = new decimal(new int[] {
            2,
            0,
            0,
            131072});
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.Location = new System.Drawing.Point(13, 386);
            this.label2.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(200, 24);
            this.label2.TabIndex = 63;
            this.label2.Text = "Output Coordinate System";
            this.label2.UseCompatibleTextRendering = true;
            // 
            // buttonSave
            // 
            this.buttonSave.Enabled = false;
            this.buttonSave.Location = new System.Drawing.Point(238, 442);
            this.buttonSave.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.buttonSave.Name = "buttonSave";
            this.buttonSave.Size = new System.Drawing.Size(136, 35);
            this.buttonSave.TabIndex = 1;
            this.buttonSave.Text = "Save";
            this.buttonSave.UseVisualStyleBackColor = true;
            // 
            // buttonCancel
            // 
            this.buttonCancel.Location = new System.Drawing.Point(382, 442);
            this.buttonCancel.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.buttonCancel.Name = "buttonCancel";
            this.buttonCancel.Size = new System.Drawing.Size(136, 35);
            this.buttonCancel.TabIndex = 3;
            this.buttonCancel.Text = "Cancel";
            this.buttonCancel.UseVisualStyleBackColor = true;
            // 
            // checkBoxAlwaysOverwrite
            // 
            this.checkBoxAlwaysOverwrite.AutoSize = true;
            this.checkBoxAlwaysOverwrite.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxAlwaysOverwrite.Location = new System.Drawing.Point(13, 448);
            this.checkBoxAlwaysOverwrite.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxAlwaysOverwrite.Name = "checkBoxAlwaysOverwrite";
            this.checkBoxAlwaysOverwrite.Size = new System.Drawing.Size(208, 24);
            this.checkBoxAlwaysOverwrite.TabIndex = 3;
            this.checkBoxAlwaysOverwrite.Text = "Always overwrite existing";
            this.checkBoxAlwaysOverwrite.UseVisualStyleBackColor = true;
            // 
            // checkBoxStep
            // 
            this.checkBoxStep.AutoSize = true;
            this.checkBoxStep.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxStep.Location = new System.Drawing.Point(23, 155);
            this.checkBoxStep.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxStep.Name = "checkBoxStep";
            this.checkBoxStep.Size = new System.Drawing.Size(76, 24);
            this.checkBoxStep.TabIndex = 3;
            this.checkBoxStep.Text = "STEP";
            this.toolTip1.SetToolTip(this.checkBoxStep, "Retains the temporary STEP file");
            this.checkBoxStep.UseVisualStyleBackColor = true;
            // 
            // MeshSaveForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 20F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(531, 491);
            this.Controls.Add(this.checkBoxAlwaysOverwrite);
            this.Controls.Add(this.buttonCancel);
            this.Controls.Add(this.buttonSave);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.comboBoxCoordinateSystem);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "MeshSaveForm";
            this.SizeGripStyle = System.Windows.Forms.SizeGripStyle.Hide;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "SuperDex Mesh Exporter";
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownScale)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownAngularDeflection)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownLinearDeflection)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownTargetEdgeLength)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDownEdgeLengthFraction)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.ComboBox comboBoxCoordinateSystem;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.NumericUpDown numericUpDownScale;
        private System.Windows.Forms.ComboBox comboBoxBackend;
        private System.Windows.Forms.ComboBox comboBoxEdgeSampling;
        private System.Windows.Forms.NumericUpDown numericUpDownTargetEdgeLength;
        private System.Windows.Forms.NumericUpDown numericUpDownEdgeLengthFraction;
        private System.Windows.Forms.Label labelBackend;
        private System.Windows.Forms.Label labelEdgeSampling;
        private System.Windows.Forms.Label labelTargetEdgeLength;
        private System.Windows.Forms.Label labelEdgeLengthFraction;
        private System.Windows.Forms.NumericUpDown numericUpDownAngularDeflection;
        private System.Windows.Forms.NumericUpDown numericUpDownLinearDeflection;
        private System.Windows.Forms.CheckBox checkBoxGlb;
        private System.Windows.Forms.CheckBox checkBoxStl;
        private System.Windows.Forms.CheckBox checkBoxObj;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Button buttonSave;
        private System.Windows.Forms.Button buttonCancel;
        private System.Windows.Forms.CheckBox checkBoxAlwaysOverwrite;
        private System.Windows.Forms.CheckBox checkBoxStep;
        private System.Windows.Forms.ToolTip toolTip1;
    }
}
