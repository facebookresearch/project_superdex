/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Windows;
using System.Windows.Controls;

using CADRobotExporter.UI.ViewModels;

namespace CADRobotExporter.UI.Views
{
    /// <summary>
    /// The "Inertial Properties*" block of the Link Properties tab: inertial origin, the upper
    /// triangle of the inertia tensor, mass, and the recalculate action.
    /// </summary>
    public class InertialPropertiesView : UserControl
    {
        private readonly NumericField originX = new NumericField();
        private readonly NumericField originY = new NumericField();
        private readonly NumericField originZ = new NumericField();
        private readonly NumericField originRoll = new NumericField();
        private readonly NumericField originPitch = new NumericField();
        private readonly NumericField originYaw = new NumericField();

        private readonly NumericField ixx = new NumericField();
        private readonly NumericField ixy = new NumericField();
        private readonly NumericField ixz = new NumericField();
        private readonly NumericField iyy = new NumericField();
        private readonly NumericField iyz = new NumericField();
        private readonly NumericField izz = new NumericField();

        private readonly NumericField mass = new NumericField { MinWidth = 120 };
        private readonly Button recalculate = new Button
        {
            Content = "Recalculate",
            Padding = new Thickness(10, 2, 10, 2),
        };

        public event EventHandler RecalculateRequested;

        public InertialPropertiesView()
        {
            recalculate.Click += (s, e) =>
            {
                EventHandler handler = RecalculateRequested;
                if (handler != null)
                {
                    handler(this, EventArgs.Empty);
                }
            };

            Grid groups = new Grid();
            groups.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            // Wide enough for a compact tensor value with its exponent ("-5.21575E-05").
            groups.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1.6, GridUnitType.Star) });

            GroupBox originGroup = MakeGroup("Origin (m, rad)", BuildOrigin());
            GroupBox inertiaGroup = MakeGroup("Moment of Inertia (Kg * m^2)", BuildInertia());
            inertiaGroup.Margin = new Thickness(8, 0, 0, 0);
            Grid.SetColumn(originGroup, 0);
            Grid.SetColumn(inertiaGroup, 1);
            groups.Children.Add(originGroup);
            groups.Children.Add(inertiaGroup);

            StackPanel massRow = new StackPanel
            {
                Orientation = Orientation.Horizontal,
                HorizontalAlignment = HorizontalAlignment.Right,
                Margin = new Thickness(0, 8, 0, 0),
            };
            massRow.Children.Add(new TextBlock
            {
                Text = "Mass (kg)",
                VerticalAlignment = VerticalAlignment.Center,
                Margin = new Thickness(0, 0, 8, 0),
            });
            massRow.Children.Add(mass);
            recalculate.Margin = new Thickness(16, 0, 0, 0);
            massRow.Children.Add(recalculate);

            StackPanel body = new StackPanel();
            body.Children.Add(groups);
            body.Children.Add(massRow);

            Content = new GroupBox
            {
                Header = "Inertial Properties*",
                Padding = new Thickness(6),
                Content = body,
            };
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

        /// <summary>
        /// The tensor is symmetric, so only the upper triangle is editable and each row is
        /// indented one column further than the last.
        /// </summary>
        private UIElement BuildInertia()
        {
            Grid grid = new Grid();
            for (int column = 0; column < 3; column++)
            {
                grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
                grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            }

            AddTensorCell(grid, 0, 0, "ixx", ixx);
            AddTensorCell(grid, 0, 1, "ixy", ixy);
            AddTensorCell(grid, 0, 2, "ixz", ixz);
            AddTensorCell(grid, 1, 1, "iyy", iyy);
            AddTensorCell(grid, 1, 2, "iyz", iyz);
            AddTensorCell(grid, 2, 2, "izz", izz);

            for (int row = 0; row < 3; row++)
            {
                grid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            }

            return grid;
        }

        private static void AddTensorCell(Grid grid, int row, int cell, string caption, TextBox box)
        {
            TextBlock label = MakeLabel(caption);
            label.Margin = new Thickness(cell == 0 ? 0 : 8, 0, 6, 0);
            box.Margin = new Thickness(0, 1, 0, 1);
            box.MinWidth = 60;

            Place(grid, label, row, cell * 2);
            Place(grid, box, row, (cell * 2) + 1);
        }

        public void Clear()
        {
            originX.Value = "";
            originY.Value = "";
            originZ.Value = "";
            originRoll.Value = "";
            originPitch.Value = "";
            originYaw.Value = "";
            ixx.Value = "";
            ixy.Value = "";
            ixz.Value = "";
            iyy.Value = "";
            iyz.Value = "";
            izz.Value = "";
            mass.Value = "";
        }

        public void Show(LinkPropertiesViewModel properties)
        {
            originX.Value = properties.InertialOrigin.X;
            originY.Value = properties.InertialOrigin.Y;
            originZ.Value = properties.InertialOrigin.Z;
            originRoll.Value = properties.InertialOrigin.Roll;
            originPitch.Value = properties.InertialOrigin.Pitch;
            originYaw.Value = properties.InertialOrigin.Yaw;

            ixx.Value = properties.Inertia.Ixx;
            ixy.Value = properties.Inertia.Ixy;
            ixz.Value = properties.Inertia.Ixz;
            iyy.Value = properties.Inertia.Iyy;
            iyz.Value = properties.Inertia.Iyz;
            izz.Value = properties.Inertia.Izz;

            mass.Value = properties.Mass;
        }

        public void ReadInto(LinkPropertiesViewModel properties)
        {
            properties.InertialOrigin.X = originX.Value;
            properties.InertialOrigin.Y = originY.Value;
            properties.InertialOrigin.Z = originZ.Value;
            properties.InertialOrigin.Roll = originRoll.Value;
            properties.InertialOrigin.Pitch = originPitch.Value;
            properties.InertialOrigin.Yaw = originYaw.Value;

            properties.Inertia.Ixx = ixx.Value;
            properties.Inertia.Ixy = ixy.Value;
            properties.Inertia.Ixz = ixz.Value;
            properties.Inertia.Iyy = iyy.Value;
            properties.Inertia.Iyz = iyz.Value;
            properties.Inertia.Izz = izz.Value;

            properties.Mass = mass.Value;
        }
    }
}
