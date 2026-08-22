/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

namespace CADRobotExporter.UI
{
    partial class ProgressIndicatorForm
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
            this.panel1 = new System.Windows.Forms.Panel();
            this.labelCancelNotification = new System.Windows.Forms.Label();
            this.buttonCancel = new System.Windows.Forms.Button();
            this.label5 = new System.Windows.Forms.Label();
            this.labelTimeElapsed = new System.Windows.Forms.Label();
            this.labelSubSteps = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.labelStep = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.progressBar = new System.Windows.Forms.ProgressBar();
            this.timer1 = new System.Windows.Forms.Timer(this.components);
            this.panel1.SuspendLayout();
            this.SuspendLayout();
            //
            // panel1
            //
            this.panel1.Controls.Add(this.labelCancelNotification);
            this.panel1.Controls.Add(this.buttonCancel);
            this.panel1.Controls.Add(this.label5);
            this.panel1.Controls.Add(this.labelTimeElapsed);
            this.panel1.Controls.Add(this.labelSubSteps);
            this.panel1.Controls.Add(this.label2);
            this.panel1.Controls.Add(this.labelStep);
            this.panel1.Controls.Add(this.label6);
            this.panel1.Controls.Add(this.progressBar);
            this.panel1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel1.Location = new System.Drawing.Point(0, 0);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(698, 356);
            this.panel1.TabIndex = 0;
            //
            // labelCancelNotification
            //
            this.labelCancelNotification.AutoSize = true;
            this.labelCancelNotification.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelCancelNotification.Location = new System.Drawing.Point(45, 314);
            this.labelCancelNotification.Name = "labelCancelNotification";
            this.labelCancelNotification.Size = new System.Drawing.Size(0, 20);
            this.labelCancelNotification.TabIndex = 67;
            //
            // buttonCancel
            //
            this.buttonCancel.Location = new System.Drawing.Point(549, 307);
            this.buttonCancel.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.buttonCancel.Name = "buttonCancel";
            this.buttonCancel.Size = new System.Drawing.Size(136, 35);
            this.buttonCancel.TabIndex = 66;
            this.buttonCancel.Text = "Cancel";
            this.buttonCancel.UseVisualStyleBackColor = true;
            //
            // label5
            //
            this.label5.AutoSize = true;
            this.label5.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label5.Location = new System.Drawing.Point(49, 119);
            this.label5.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(111, 24);
            this.label5.TabIndex = 65;
            this.label5.Text = "Time Elapsed:";
            this.label5.UseCompatibleTextRendering = true;
            //
            // labelTimeElapsed
            //
            this.labelTimeElapsed.AutoSize = true;
            this.labelTimeElapsed.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelTimeElapsed.Location = new System.Drawing.Point(170, 119);
            this.labelTimeElapsed.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelTimeElapsed.Name = "labelTimeElapsed";
            this.labelTimeElapsed.Size = new System.Drawing.Size(166, 24);
            this.labelTimeElapsed.TabIndex = 64;
            this.labelTimeElapsed.Text = "1 minutes 30 seconds";
            this.labelTimeElapsed.UseCompatibleTextRendering = true;
            //
            // labelSubSteps
            //
            this.labelSubSteps.AutoSize = true;
            this.labelSubSteps.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelSubSteps.Location = new System.Drawing.Point(170, 167);
            this.labelSubSteps.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelSubSteps.Name = "labelSubSteps";
            this.labelSubSteps.Size = new System.Drawing.Size(303, 78);
            this.labelSubSteps.TabIndex = 63;
            this.labelSubSteps.Text = "☑ Creating top level coordinate system\r\n☐ Calculating intertia ←\r\n☑ Calculating j" +
    "oint axis\r\n☑ Exporting SuperDex";
            this.labelSubSteps.UseCompatibleTextRendering = true;
            //
            // label2
            //
            this.label2.AutoSize = true;
            this.label2.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.Location = new System.Drawing.Point(49, 167);
            this.label2.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(53, 24);
            this.label2.TabIndex = 62;
            this.label2.Text = "Steps:";
            this.label2.UseCompatibleTextRendering = true;
            //
            // labelStep
            //
            this.labelStep.AutoSize = true;
            this.labelStep.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelStep.Location = new System.Drawing.Point(170, 143);
            this.labelStep.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelStep.Name = "labelStep";
            this.labelStep.Size = new System.Drawing.Size(176, 24);
            this.labelStep.TabIndex = 61;
            this.labelStep.Text = "joint_1 of link_1 (1 of 8)";
            this.labelStep.UseCompatibleTextRendering = true;
            //
            // label6
            //
            this.label6.AutoSize = true;
            this.label6.Font = new System.Drawing.Font("Microsoft Sans Serif", 8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label6.Location = new System.Drawing.Point(49, 143);
            this.label6.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(92, 24);
            this.label6.TabIndex = 60;
            this.label6.Text = "Processing:";
            this.label6.UseCompatibleTextRendering = true;
            //
            // progressBar
            //
            this.progressBar.Location = new System.Drawing.Point(49, 41);
            this.progressBar.Name = "progressBar";
            this.progressBar.Size = new System.Drawing.Size(600, 40);
            this.progressBar.TabIndex = 1;
            //
            // ProgressIndicatorForm
            //
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 20F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(698, 356);
            this.Controls.Add(this.panel1);
            this.MaximizeBox = false;
            this.Name = "ProgressIndicatorForm";
            this.ShowIcon = false;
            this.SizeGripStyle = System.Windows.Forms.SizeGripStyle.Hide;
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "Superdex CAD Exporter is Processing...";
            this.TopMost = true;
            this.panel1.ResumeLayout(false);
            this.panel1.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.ProgressBar progressBar;
        private System.Windows.Forms.Label labelSubSteps;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label labelStep;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label labelTimeElapsed;
        private System.Windows.Forms.Button buttonCancel;
        private System.Windows.Forms.Timer timer1;
        private System.Windows.Forms.Label labelCancelNotification;
    }
}
