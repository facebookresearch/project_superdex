/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System.Windows.Forms;

namespace CADRobotExporter.UI
{
    public class InputDialog : Form
    {
        private TextBox textBox;
        private Button okButton;
        private Button cancelButton;
        public string InputText => textBox.Text;
        public InputDialog(string title, string prompt)
        {
            Text = title;
            TopMost = true;
            Width = 300;
            Height = 150;
            FormBorderStyle = FormBorderStyle.FixedDialog;
            StartPosition = FormStartPosition.CenterParent;
            MaximizeBox = false;
            MinimizeBox = false;
            var label = new Label { Left = 10, Top = 10, Text = prompt, AutoSize = true };
            textBox = new TextBox { Left = 10, Top = 35, Width = 260 };
            okButton = new Button { Text = "OK", Left = 110, Top = 70, DialogResult = DialogResult.OK };
            cancelButton = new Button { Text = "Cancel", Left = 195, Top = 70, DialogResult = DialogResult.Cancel };
            AcceptButton = okButton;
            CancelButton = cancelButton;
            Controls.AddRange(new Control[] { label, textBox, okButton, cancelButton });
        }
    }
}
