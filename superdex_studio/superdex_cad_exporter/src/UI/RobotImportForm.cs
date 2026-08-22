/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Windows.Forms;

using SolidWorks.Interop.sldworks;

namespace CADRobotExporter.UI
{
    public partial class RobotImportForm : Form
    {
        private static readonly string DefaultCoordSysName = "-- default --";

        private ModelDoc2 ActiveSWModel;

        public string UrdfFilePath => textBoxFilePath.Text;

        public bool CreateCoordinateSystems => checkBoxCreateCSYS.Checked;

        public bool CreateRobotConfiguration => checkBoxCreateRobotConfiguration.Checked;

        public string SelectedCoordinateSystem
        {
            get
            {
                string selected = comboBoxCoordinateSystem.Text;
                if (selected == DefaultCoordSysName)
                    return null;
                return selected;
            }
        }

        public RobotImportForm(SldWorks swApp)
        {
            InitializeComponent();

            ActiveSWModel = (ModelDoc2)swApp.ActiveDoc;

            openFileDialog.Title = "Select URDF File to Import";
            openFileDialog.Filter = "URDF files (*.urdf)|*.urdf|XML files (*.xml)|*.xml|All files (*.*)|*.*";
            openFileDialog.FilterIndex = 1;

            buttonBrowse.Enabled = true;
            buttonBrowse.Click += ButtonBrowse_Click;
            buttonOK.Click += ButtonOK_Click;
            buttonCancel.Click += ButtonCancel_Click;
            textBoxFilePath.TextChanged += TextBoxFilePath_TextChanged;

            AddCoordinateSystems();
        }

        private void AddCoordinateSystems()
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

        private void ButtonBrowse_Click(object sender, EventArgs e)
        {
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                textBoxFilePath.Text = openFileDialog.FileName;
            }
        }

        private void ButtonOK_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.OK;
            Close();
        }

        private void ButtonCancel_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
            Close();
        }

        private void TextBoxFilePath_TextChanged(object sender, EventArgs e)
        {
            buttonOK.Enabled = !string.IsNullOrWhiteSpace(textBoxFilePath.Text);
        }
    }
}
