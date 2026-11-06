/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;

using CADRobotExporter.Utilities;

namespace CADRobotExporter.UI
{
    /// <summary>
    /// Reports where the exported files were written.
    /// </summary>
    public class ExportCompleteDialog : Window
    {
        public ExportCompleteDialog(
            string urdfPath, string mjcfPath, string superdexBotPath, string superdexPrefabPath)
        {
            Build(
                "SuperDex CAD Exporter",
                "Export complete!\n\n" +
                "URDF file saved to:\n" + urdfPath + "\n\n" +
                "MJCF file saved to:\n" + mjcfPath + "\n\n" +
                "SuperDex Bot file saved to:\n" + superdexBotPath + "\n\n" +
                "SuperDex Prefab file saved to:\n" + superdexPrefabPath,
                "Close");
        }

        private ExportCompleteDialog()
        {
        }

        private void Build(string title, string message, string closeCaption)
        {
            Theme.FluentTheme.Apply(this);

            Title = title;
            Topmost = true;
            Width = 720;
            SizeToContent = SizeToContent.Height;
            ResizeMode = ResizeMode.NoResize;
            WindowStartupLocation = WindowStartupLocation.CenterScreen;
            ShowInTaskbar = false;

            TextBlock label = new TextBlock
            {
                Text = message,
                TextWrapping = TextWrapping.Wrap,
            };

            Button openFolderButton = new Button
            {
                Content = "Open Folder",
                IsDefault = true,
                Width = 200,
                Height = 32,
            };
            openFolderButton.Click += (sender, e) => { DialogResult = true; };

            Button closeButton = new Button
            {
                Content = closeCaption,
                IsCancel = true,
                Width = 80,
                Height = 32,
                Margin = new Thickness(20, 0, 0, 0),
            };

            StackPanel buttons = new StackPanel
            {
                Orientation = Orientation.Horizontal,
                HorizontalAlignment = HorizontalAlignment.Right,
                Margin = new Thickness(0, 20, 0, 0),
            };
            buttons.Children.Add(openFolderButton);
            buttons.Children.Add(closeButton);

            StackPanel root = new StackPanel { Margin = new Thickness(10) };
            root.Children.Add(label);
            root.Children.Add(buttons);

            Content = root;
        }

        /// <summary>
        /// Lists the mesh files written and, if the user asks, opens Explorer with the first one
        /// selected.
        /// </summary>
        public static void ShowMeshExportComplete(
            IList<string> writtenPaths, Window owner = null, IntPtr ownerHandle = default(IntPtr))
        {
            ExportCompleteDialog dialog = new ExportCompleteDialog();
            dialog.Build(
                "SuperDex Mesh Exporter",
                "Mesh export complete!\n\nSaved to:\n" + string.Join("\n", writtenPaths),
                "OK");

            if (owner != null)
            {
                dialog.Owner = owner;
            }
            else
            {
                WpfWindowOwner.OwnTo(dialog, ownerHandle);
            }

            if (dialog.ShowDialog() == true && writtenPaths.Count > 0)
            {
                Process.Start("explorer.exe", $"/select,\"{writtenPaths[0]}\"");
            }
        }
    }
}
