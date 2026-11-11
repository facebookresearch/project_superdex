/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Interop;
using System.Windows.Media;

using CADRobotExporter.UI.ViewModels;
using CADRobotExporter.Utilities;

namespace CADRobotExporter.UI.Views
{
    /// <summary>
    /// The upper strip of "Visual and Collision Mesh Export Properties": the visual origin, the
    /// preset picker, and the per-link color channels.
    /// </summary>
    public class VisualPropertiesView : UserControl
    {
        private readonly NumericField originX = new NumericField();
        private readonly NumericField originY = new NumericField();
        private readonly NumericField originZ = new NumericField();
        private readonly NumericField originRoll = new NumericField();
        private readonly NumericField originPitch = new NumericField();
        private readonly NumericField originYaw = new NumericField();

        // The WinForms originals were DomainUpDown controls with an empty Items collection, so
        // their spin arrows never did anything; plain text entry is what users actually had.
        private readonly NumericField red = new NumericField();
        private readonly NumericField green = new NumericField();
        private readonly NumericField blue = new NumericField();
        private readonly NumericField alpha = new NumericField();
        private readonly Border swatch = new Border
        {
            Width = 22,
            Height = 16,
            BorderThickness = new Thickness(1),
            BorderBrush = Brushes.Gray,
        };

        private readonly ComboBox presets = new ComboBox();

        public event EventHandler PresetSelectionChanged;

        public VisualPropertiesView()
        {
            presets.SelectionChanged += (s, e) =>
            {
                EventHandler handler = PresetSelectionChanged;
                if (handler != null)
                {
                    handler(this, EventArgs.Empty);
                }
            };

            GroupBox originGroup = MakeGroup("Origin (m, rad)", BuildOrigin());
            GroupBox presetGroup = MakeGroup("Presets", BuildPresets());
            GroupBox colorGroup = MakeGroup("Per-Link Color*", BuildColor());

            StackPanel rightColumn = new StackPanel { Margin = new Thickness(8, 0, 0, 0) };
            colorGroup.Margin = new Thickness(0, 8, 0, 0);
            rightColumn.Children.Add(presetGroup);
            rightColumn.Children.Add(colorGroup);

            Grid root = new Grid();
            root.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            root.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1.6, GridUnitType.Star) });
            Grid.SetColumn(originGroup, 0);
            Grid.SetColumn(rightColumn, 1);
            root.Children.Add(originGroup);
            root.Children.Add(rightColumn);

            Content = root;
        }

        private static GroupBox MakeGroup(string header, UIElement content)
        {
            return new GroupBox
            {
                Header = header,
                Content = content,
                Padding = new Thickness(6),
                VerticalAlignment = VerticalAlignment.Top,
            };
        }

        private static TextBlock MakeLabel(string text)
        {
            return new TextBlock
            {
                Text = text,
                VerticalAlignment = VerticalAlignment.Center,
                HorizontalAlignment = HorizontalAlignment.Right,
                Margin = new Thickness(0, 0, 6, 0),
            };
        }

        private static void Place(Grid grid, UIElement element, int row, int column)
        {
            Grid.SetRow(element, row);
            Grid.SetColumn(element, column);
            grid.Children.Add(element);
        }

        private UIElement BuildOrigin()
        {
            Grid grid = new Grid();
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });

            string[] leftCaptions = { "X", "Y", "Z" };
            TextBox[] leftBoxes = { originX, originY, originZ };
            string[] rightCaptions = { "Roll", "Pitch", "Yaw" };
            TextBox[] rightBoxes = { originRoll, originPitch, originYaw };

            for (int row = 0; row < 3; row++)
            {
                grid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });

                TextBlock rightLabel = MakeLabel(rightCaptions[row]);
                rightLabel.Margin = new Thickness(8, 0, 6, 0);
                leftBoxes[row].Margin = new Thickness(0, 1, 0, 1);
                rightBoxes[row].Margin = new Thickness(0, 1, 0, 1);

                Place(grid, MakeLabel(leftCaptions[row]), row, 0);
                Place(grid, leftBoxes[row], row, 1);
                Place(grid, rightLabel, row, 2);
                Place(grid, rightBoxes[row], row, 3);
            }

            return grid;
        }

        private UIElement BuildPresets()
        {
            Grid grid = new Grid();
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            grid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });

            TextBlock label = MakeLabel("Select Preset");
            Place(grid, label, 0, 0);
            Place(grid, presets, 0, 1);
            return grid;
        }

        private UIElement BuildColor()
        {
            Grid grid = new Grid();
            TextBox[] boxes = { red, green, blue, alpha };
            string[] captions = { "R", "G", "B", "A" };

            grid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            for (int i = 0; i < boxes.Length; i++)
            {
                grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
                grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });

                TextBlock label = MakeLabel(captions[i]);
                label.Margin = new Thickness(i == 0 ? 0 : 8, 0, 6, 0);
                boxes[i].MinWidth = 60;

                Place(grid, label, 0, i * 2);
                Place(grid, boxes[i], 0, (i * 2) + 1);
            }

            // The Windows color dialog has no alpha channel, so it only fills in R, G and B.
            Button pick = new Button
            {
                Content = swatch,
                Margin = new Thickness(8, 0, 0, 0),
                Padding = new Thickness(4, 2, 4, 2),
                ToolTip = "Pick the Link's color",
            };
            pick.Click += (s, e) => PickColor();
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            Place(grid, pick, 0, boxes.Length * 2);

            red.TextChanged += (s, e) => UpdateSwatch();
            green.TextChanged += (s, e) => UpdateSwatch();
            blue.TextChanged += (s, e) => UpdateSwatch();
            UpdateSwatch();

            return grid;
        }

        private ColorFields DisplayedColor()
        {
            return new ColorFields
            {
                Red = red.Value,
                Green = green.Value,
                Blue = blue.Value,
                Alpha = alpha.Value,
            };
        }

        private void UpdateSwatch()
        {
            byte r, g, b;
            swatch.Background = DisplayedColor().TryGetRgb(out r, out g, out b)
                ? new SolidColorBrush(Color.FromRgb(r, g, b))
                : Brushes.Transparent;
        }

        private void PickColor()
        {
            ColorFields color = DisplayedColor();

            using (var dialog = new System.Windows.Forms.ColorDialog { FullOpen = true, AnyColor = true })
            {
                byte r, g, b;
                if (color.TryGetRgb(out r, out g, out b))
                {
                    dialog.Color = System.Drawing.Color.FromArgb(r, g, b);
                }

                // Owned by the export window so the dialog stays above it rather than behind the host.
                Window owner = Window.GetWindow(this);
                System.Windows.Forms.DialogResult result = owner == null
                    ? dialog.ShowDialog()
                    : dialog.ShowDialog(new NativeWindowWrapper(new WindowInteropHelper(owner).Handle));
                if (result != System.Windows.Forms.DialogResult.OK)
                {
                    return;
                }

                color.SetRgb(dialog.Color.R, dialog.Color.G, dialog.Color.B);
            }

            red.Value = color.Red;
            green.Value = color.Green;
            blue.Value = color.Blue;
        }

        public void ClearPresets()
        {
            presets.Items.Clear();
        }

        public void AddPreset(object item)
        {
            presets.Items.Add(item);
        }

        public object SelectedPreset
        {
            get { return presets.SelectedItem; }
        }

        public int SelectedPresetIndex
        {
            get { return presets.SelectedIndex; }
            set { presets.SelectedIndex = value; }
        }

        public void Clear()
        {
            originX.Value = "";
            originY.Value = "";
            originZ.Value = "";
            originRoll.Value = "";
            originPitch.Value = "";
            originYaw.Value = "";
            red.Value = "";
            green.Value = "";
            blue.Value = "";
            alpha.Value = "";
        }

        public void Show(LinkPropertiesViewModel properties)
        {
            originX.Value = properties.VisualOrigin.X;
            originY.Value = properties.VisualOrigin.Y;
            originZ.Value = properties.VisualOrigin.Z;
            originRoll.Value = properties.VisualOrigin.Roll;
            originPitch.Value = properties.VisualOrigin.Pitch;
            originYaw.Value = properties.VisualOrigin.Yaw;

            red.Value = properties.Color.Red;
            green.Value = properties.Color.Green;
            blue.Value = properties.Color.Blue;
            alpha.Value = properties.Color.Alpha;
        }

        public void ReadInto(LinkPropertiesViewModel properties)
        {
            properties.VisualOrigin.X = originX.Value;
            properties.VisualOrigin.Y = originY.Value;
            properties.VisualOrigin.Z = originZ.Value;
            properties.VisualOrigin.Roll = originRoll.Value;
            properties.VisualOrigin.Pitch = originPitch.Value;
            properties.VisualOrigin.Yaw = originYaw.Value;

            properties.Color.Red = red.Value;
            properties.Color.Green = green.Value;
            properties.Color.Blue = blue.Value;
            properties.Color.Alpha = alpha.Value;
        }
    }
}
