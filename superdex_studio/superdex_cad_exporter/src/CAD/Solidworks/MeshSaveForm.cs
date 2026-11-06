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
using System.Windows;
using System.Windows.Controls;

using Meshing;

using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;

using CADRobotExporter.RobotExport;
using CADRobotExporter.CAD;
using CADRobotExporter.Export;
using CADRobotExporter.UI.ViewModels;
using CADRobotExporter.UI.Views;

using MessageBox = System.Windows.Forms.MessageBox;
using MessageBoxButtons = System.Windows.Forms.MessageBoxButtons;
using MessageBoxIcon = System.Windows.Forms.MessageBoxIcon;

namespace CADRobotExporter.UI
{
    public class MeshSaveForm : Window
    {
        public ModelDoc2 ActiveSWModel;
        public SldWorks SWApp = null;
        public string MeshFilename = "";

        public static string DefaultCoordSysName = "-- default --";

        private readonly MeshingColumnView meshing;
        private readonly ComboBox comboBoxCoordinateSystem;
        private readonly CheckBox checkBoxGlb;
        private readonly CheckBox checkBoxStl;
        private readonly CheckBox checkBoxObj;
        private readonly CheckBox checkBoxStep;
        private readonly CheckBox checkBoxAlwaysOverwrite;
        private readonly Button buttonSave;

        public MeshSaveForm(SldWorks iSldWorksApp, string solidworksSaveAsString)
        {
            // without this Tab, Enter and mnemonics are swallowed before they reach the window.
            System.Windows.Forms.Integration.ElementHost.EnableModelessKeyboardInterop(this);
            Theme.FluentTheme.Apply(this);

            Title = "SuperDex Mesh Exporter";
            Width = 560;
            SizeToContent = SizeToContent.Height;
            ResizeMode = ResizeMode.NoResize;
            WindowStartupLocation = WindowStartupLocation.CenterScreen;
            ShowInTaskbar = false;
            Background = (System.Windows.Media.Brush)Resources["LayerFill"];

            SWApp = iSldWorksApp;
            ActiveSWModel = (ModelDoc2)SWApp.ActiveDoc;

            meshing = new MeshingColumnView(false);
            meshing.SettingChanged += (s, e) => meshing.ApplyEnableState(true, true);

            checkBoxGlb = MakeFormatCheckBox("GLB");
            checkBoxStl = MakeFormatCheckBox("STL");
            checkBoxObj = MakeFormatCheckBox("OBJ");
            checkBoxStep = MakeFormatCheckBox("STEP");
            checkBoxStep.ToolTip = "Retains the temporary STEP file";

            checkBoxAlwaysOverwrite = new CheckBox
            {
                Content = "Always overwrite existing",
                Margin = new Thickness(0, 10, 0, 0),
            };

            comboBoxCoordinateSystem = new ComboBox { Margin = new Thickness(0, 6, 0, 0) };

            buttonSave = new Button
            {
                Content = "Save",
                IsDefault = true,
                MinWidth = 110,
                IsEnabled = false,
                Style = (Style)Resources[Theme.FluentTheme.AccentButtonKey],
            };
            buttonSave.Click += ButtonSave_Click;

            Button buttonCancel = new Button
            {
                Content = "Close",
                IsCancel = true,
                MinWidth = 110,
                Margin = new Thickness(8, 0, 0, 0),
            };
            buttonCancel.Click += ButtonCancel_Click;

            StackPanel root = new StackPanel { Margin = new Thickness(12) };
            root.Children.Add(BuildFormatGroup());
            root.Children.Add(BuildMeshingGroup());
            root.Children.Add(BuildCoordinateSystemPanel());

            StackPanel buttons = new StackPanel
            {
                Orientation = Orientation.Horizontal,
                HorizontalAlignment = HorizontalAlignment.Right,
                Margin = new Thickness(0, 14, 0, 0),
            };
            buttons.Children.Add(buttonSave);
            buttons.Children.Add(buttonCancel);
            root.Children.Add(buttons);

            Content = root;

            meshing.ApplyEnableState(true, true);
            AddCoordinateSystems();
            MeshFilename = ExtractFilePathWithoutExtension(solidworksSaveAsString);

            SubscribeToDocumentDestroyed();
            Closed += (s, e) => UnsubscribeFromDocumentDestroyed();
        }

        private void SubscribeToDocumentDestroyed()
        {
            try
            {
                if (ActiveSWModel is PartDoc part)
                {
                    part.DestroyNotify += OnDocumentDestroyed;
                }
                else if (ActiveSWModel is AssemblyDoc assembly)
                {
                    assembly.DestroyNotify += OnDocumentDestroyed;
                }
            }
            catch (Exception)
            {
                // Best effort
            }
        }

        private void UnsubscribeFromDocumentDestroyed()
        {
            try
            {
                if (ActiveSWModel is PartDoc part)
                {
                    part.DestroyNotify -= OnDocumentDestroyed;
                }
                else if (ActiveSWModel is AssemblyDoc assembly)
                {
                    assembly.DestroyNotify -= OnDocumentDestroyed;
                }
            }
            catch (Exception)
            {
                // The document may already be gone
            }
        }

        private int OnDocumentDestroyed()
        {
            Dispatcher.BeginInvoke(new Action(Close));
            return 0;
        }

        private CheckBox MakeFormatCheckBox(string caption)
        {
            CheckBox box = new CheckBox
            {
                Content = caption,
                Margin = new Thickness(0, 0, 20, 0),
            };
            box.Checked += CheckBox_CheckedChanged;
            box.Unchecked += CheckBox_CheckedChanged;
            return box;
        }

        private UIElement BuildFormatGroup()
        {
            StackPanel formats = new StackPanel { Orientation = Orientation.Horizontal };
            formats.Children.Add(checkBoxGlb);
            formats.Children.Add(checkBoxStl);
            formats.Children.Add(checkBoxObj);
            formats.Children.Add(checkBoxStep);

            StackPanel body = new StackPanel();
            body.Children.Add(formats);
            body.Children.Add(checkBoxAlwaysOverwrite);

            return new GroupBox
            {
                Header = "Export Format",
                Padding = new Thickness(10),
                Content = body,
            };
        }

        private UIElement BuildMeshingGroup()
        {
            Grid grid = new Grid();
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });

            AddMeshingRow(grid, 0, "Linear Deflection (mm)", meshing.LinearDeflection);
            AddMeshingRow(grid, 1, "Angular Deflection (rad)", meshing.AngularDeflection);
            AddMeshingRow(grid, 2, "Scale (multiplier)", meshing.Scale);
            AddMeshingRow(grid, 3, "Mesher", meshing.Backend);
            AddMeshingRow(grid, 4, "Edge Sampling", meshing.EdgeSampling);
            AddMeshingRow(grid, 5, "Edge Length mm (0=auto)", meshing.TargetEdgeLength);
            AddMeshingRow(grid, 6, "Edge Length Fraction", meshing.EdgeLengthFraction);

            return new GroupBox
            {
                Header = "Meshing Options",
                Padding = new Thickness(10),
                Margin = new Thickness(0, 10, 0, 0),
                Content = grid,
            };
        }

        private static void AddMeshingRow(Grid grid, int row, string caption, UIElement editor)
        {
            grid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });

            TextBlock label = new TextBlock
            {
                Text = caption,
                VerticalAlignment = VerticalAlignment.Center,
                Margin = new Thickness(0, 0, 12, 0),
            };
            Grid.SetRow(label, row);
            Grid.SetColumn(label, 0);
            grid.Children.Add(label);

            Grid.SetRow(editor, row);
            Grid.SetColumn(editor, 1);
            grid.Children.Add(editor);
        }

        private UIElement BuildCoordinateSystemPanel()
        {
            StackPanel panel = new StackPanel { Margin = new Thickness(0, 12, 0, 0) };
            panel.Children.Add(new TextBlock { Text = "Output Coordinate System" });
            panel.Children.Add(comboBoxCoordinateSystem);
            return panel;
        }

        private void CheckBox_CheckedChanged(object sender, RoutedEventArgs e)
        {
            buttonSave.IsEnabled = checkBoxGlb.IsChecked == true
                || checkBoxObj.IsChecked == true
                || checkBoxStl.IsChecked == true
                || checkBoxStep.IsChecked == true;
        }

        private void ButtonSave_Click(object sender, RoutedEventArgs e)
        {
            if (SaveMesh(out List<string> writtenPaths))
            {
                ExportCompleteDialog.ShowMeshExportComplete(writtenPaths, owner: this);
            }
            Activate();
        }

        private void ButtonCancel_Click(object sender, RoutedEventArgs e)
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

        private bool SaveMesh(out List<string> writtenPaths)
        {
            writtenPaths = new List<string>();
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
                    case System.Windows.Forms.DialogResult.Cancel:
                        return false;
                    case System.Windows.Forms.DialogResult.Yes:
                        break;
                    case System.Windows.Forms.DialogResult.No:
                        ActiveSWModel.ClearSelection2(true);
                        break;
                    default:
                        break;
                }
            }

            string coordSysName = comboBoxCoordinateSystem.SelectedItem as string;

            List<MeshFormat> exportFormats = new List<MeshFormat>();

            if (checkBoxStl.IsChecked == true)
            {
                exportFormats.Add(MeshFormat.stlOpenCascade);
            }
            if (checkBoxObj.IsChecked == true)
            {
                exportFormats.Add(MeshFormat.objOpenCascade);
            }
            if (checkBoxGlb.IsChecked == true)
            {
                exportFormats.Add(MeshFormat.glbOpenCascade);
            }

            float linearDeflection = (float)meshing.LinearDeflection.Value;
            float angularDeflection = (float)meshing.AngularDeflection.Value;
            float scale = (float)meshing.Scale.Value;
            bool overwrite = checkBoxAlwaysOverwrite.IsChecked == true;

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
            bool keepStep = checkBoxStep.IsChecked == true
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
                Backend = MeshingSettings.BackendFromIndex(meshing.Backend.SelectedIndex),
                EdgeSampling = MeshingSettings.EdgeSamplingFromIndex(meshing.EdgeSampling.SelectedIndex),
                TargetEdgeLength = (double)meshing.TargetEdgeLength.Value,
                TargetEdgeLengthFraction = (double)meshing.EdgeLengthFraction.Value,
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

            if (exported)
            {
                foreach (MeshExportOutput output in outputs)
                {
                    writtenPaths.Add(output.Path);
                }
                if (keepStep)
                {
                    writtenPaths.Add(stepPath);
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

                System.Windows.Forms.DialogResult result = MessageBox.Show(
                    message,
                    caption,
                    MessageBoxButtons.YesNo,
                    MessageBoxIcon.Warning
                );

                return result == System.Windows.Forms.DialogResult.Yes;
            }

            // File doesn't exist, OK to proceed
            return true;
        }
    }
}
