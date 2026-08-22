/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

namespace CADRobotExporter.UI
{
    partial class RobotImportForm
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
            this.label2 = new System.Windows.Forms.Label();
            this.comboBoxCoordinateSystem = new System.Windows.Forms.ComboBox();
            this.buttonCancel = new System.Windows.Forms.Button();
            this.buttonOK = new System.Windows.Forms.Button();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.checkBoxCreateCSYS = new System.Windows.Forms.CheckBox();
            this.checkBoxCreateRobotConfiguration = new System.Windows.Forms.CheckBox();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.buttonBrowse = new System.Windows.Forms.Button();
            this.openFileDialog = new System.Windows.Forms.OpenFileDialog();
            this.textBoxFilePath = new System.Windows.Forms.TextBox();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.SuspendLayout();
            //
            // label2
            //
            this.label2.AutoSize = true;
            this.label2.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.Location = new System.Drawing.Point(12, 36);
            this.label2.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(235, 24);
            this.label2.TabIndex = 65;
            this.label2.Text = "Base/World Coordinate System";
            this.label2.UseCompatibleTextRendering = true;
            //
            // comboBoxCoordinateSystem
            //
            this.comboBoxCoordinateSystem.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxCoordinateSystem.FormattingEnabled = true;
            this.comboBoxCoordinateSystem.Items.AddRange(new object[] {
            "-- default --"});
            this.comboBoxCoordinateSystem.Location = new System.Drawing.Point(254, 33);
            this.comboBoxCoordinateSystem.Name = "comboBoxCoordinateSystem";
            this.comboBoxCoordinateSystem.Size = new System.Drawing.Size(299, 28);
            this.comboBoxCoordinateSystem.TabIndex = 64;
            //
            // buttonCancel
            //
            this.buttonCancel.Location = new System.Drawing.Point(442, 328);
            this.buttonCancel.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.buttonCancel.Name = "buttonCancel";
            this.buttonCancel.Size = new System.Drawing.Size(136, 35);
            this.buttonCancel.TabIndex = 67;
            this.buttonCancel.Text = "Cancel";
            this.buttonCancel.UseVisualStyleBackColor = true;
            //
            // buttonOK
            //
            this.buttonOK.Enabled = false;
            this.buttonOK.Location = new System.Drawing.Point(298, 328);
            this.buttonOK.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.buttonOK.Name = "buttonOK";
            this.buttonOK.Size = new System.Drawing.Size(136, 35);
            this.buttonOK.TabIndex = 66;
            this.buttonOK.Text = "Import";
            this.buttonOK.UseVisualStyleBackColor = true;
            //
            // groupBox1
            //
            this.groupBox1.Controls.Add(this.checkBoxCreateRobotConfiguration);
            this.groupBox1.Controls.Add(this.checkBoxCreateCSYS);
            this.groupBox1.Controls.Add(this.comboBoxCoordinateSystem);
            this.groupBox1.Controls.Add(this.label2);
            this.groupBox1.Location = new System.Drawing.Point(12, 12);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(566, 183);
            this.groupBox1.TabIndex = 60;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Import Options";
            //
            // checkBoxCreateCSYS
            //
            this.checkBoxCreateCSYS.AutoSize = true;
            this.checkBoxCreateCSYS.Checked = true;
            this.checkBoxCreateCSYS.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkBoxCreateCSYS.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxCreateCSYS.Location = new System.Drawing.Point(12, 75);
            this.checkBoxCreateCSYS.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxCreateCSYS.Name = "checkBoxCreateCSYS";
            this.checkBoxCreateCSYS.Size = new System.Drawing.Size(230, 24);
            this.checkBoxCreateCSYS.TabIndex = 1;
            this.checkBoxCreateCSYS.Text = "Create Coordinate Systems";
            this.checkBoxCreateCSYS.UseVisualStyleBackColor = true;
            //
            // checkBoxCreateRobotConfiguration
            //
            this.checkBoxCreateRobotConfiguration.AutoSize = true;
            this.checkBoxCreateRobotConfiguration.Checked = true;
            this.checkBoxCreateRobotConfiguration.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkBoxCreateRobotConfiguration.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxCreateRobotConfiguration.Location = new System.Drawing.Point(12, 112);
            this.checkBoxCreateRobotConfiguration.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.checkBoxCreateRobotConfiguration.Name = "checkBoxCreateRobotConfiguration";
            this.checkBoxCreateRobotConfiguration.Size = new System.Drawing.Size(230, 24);
            this.checkBoxCreateRobotConfiguration.TabIndex = 66;
            this.checkBoxCreateRobotConfiguration.Text = "Create Robot Configuration";
            this.checkBoxCreateRobotConfiguration.UseVisualStyleBackColor = true;
            //
            // groupBox2
            //
            this.groupBox2.Controls.Add(this.textBoxFilePath);
            this.groupBox2.Controls.Add(this.buttonBrowse);
            this.groupBox2.Location = new System.Drawing.Point(12, 201);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(566, 107);
            this.groupBox2.TabIndex = 68;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "File";
            //
            // buttonBrowse
            //
            this.buttonBrowse.Enabled = false;
            this.buttonBrowse.Location = new System.Drawing.Point(423, 59);
            this.buttonBrowse.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.buttonBrowse.Name = "buttonBrowse";
            this.buttonBrowse.Size = new System.Drawing.Size(136, 35);
            this.buttonBrowse.TabIndex = 69;
            this.buttonBrowse.Text = "Browse...";
            this.buttonBrowse.UseVisualStyleBackColor = true;
            //
            // textBoxFilePath
            //
            this.textBoxFilePath.Location = new System.Drawing.Point(12, 25);
            this.textBoxFilePath.Name = "textBoxFilePath";
            this.textBoxFilePath.Size = new System.Drawing.Size(547, 26);
            this.textBoxFilePath.TabIndex = 70;
            //
            // RobotImportForm
            //
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 20F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(590, 377);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.buttonCancel);
            this.Controls.Add(this.buttonOK);
            this.Name = "RobotImportForm";
            this.ShowIcon = false;
            this.Text = "Robot Importer";
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.ComboBox comboBoxCoordinateSystem;
        private System.Windows.Forms.Button buttonCancel;
        private System.Windows.Forms.Button buttonOK;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.CheckBox checkBoxCreateCSYS;
        private System.Windows.Forms.CheckBox checkBoxCreateRobotConfiguration;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.TextBox textBoxFilePath;
        private System.Windows.Forms.Button buttonBrowse;
        private System.Windows.Forms.OpenFileDialog openFileDialog;
    }
}
