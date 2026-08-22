/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System.Drawing;
using System.Windows.Forms;

namespace CADRobotExporter.UI
{
    public class ExportCompleteDialog : Form
    {
        private Button openFolderButton;
        private Button closeButton;

        public ExportCompleteDialog(string urdfPath, string mjcfPath, string superdexBotPath)
        {
            Text = "SuperDex CAD Exporter";
            TopMost = true;
            FormBorderStyle = FormBorderStyle.FixedDialog;
            StartPosition = FormStartPosition.CenterScreen;
            MaximizeBox = false;
            MinimizeBox = false;

            int formWidth = 720;

            string message = "Export complete!\n\n" +
                "URDF file saved to:\n" + urdfPath + "\n\n" +
                "MJCF file saved to:\n" + mjcfPath + "\n\n" +
                "SuperDex Bot file saved to:\n" + superdexBotPath;

            var label = new Label
            {
                Left = 10,
                Top = 10,
                MaximumSize = new Size(formWidth - 30, 0),
                Text = message,
                AutoSize = true
            };

            Controls.Add(label);

            int buttonY = label.Bottom + 20;

            openFolderButton = new Button
            {
                Text = "Open Folder",
                Left = formWidth - 320,
                Top = buttonY,
                Width = 200,
                Height = 32,
                DialogResult = DialogResult.Yes
            };

            closeButton = new Button
            {
                Text = "Close",
                Left = formWidth - 100,
                Top = buttonY,
                Width = 80,
                Height = 32,
                DialogResult = DialogResult.No
            };

            AcceptButton = openFolderButton;
            CancelButton = closeButton;
            Controls.Add(openFolderButton);
            Controls.Add(closeButton);

            ClientSize = new Size(formWidth, buttonY + 50);
        }
    }
}
