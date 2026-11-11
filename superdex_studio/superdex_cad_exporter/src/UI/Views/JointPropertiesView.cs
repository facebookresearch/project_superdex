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
    /// The right-hand property panel of the Joint Properties tab. The link tree stays a WinForms
    /// control on the left, so this hosts everything from "Basic Properties" rightwards.
    /// </summary>
    public class JointPropertiesView : UserControl
    {
        private readonly TextBox jointName = new TextBox();
        private readonly ComboBox jointType = new ComboBox { IsEditable = false };
        private readonly TextBox parentLink = MakeReadOnly();
        private readonly TextBox childLink = MakeReadOnly();
        private readonly TextBox coordSys = MakeReadOnly();
        private readonly TextBox refAxis = MakeReadOnly();

        private readonly NumericField originX = new NumericField();
        private readonly NumericField originY = new NumericField();
        private readonly NumericField originZ = new NumericField();
        private readonly NumericField originRoll = new NumericField();
        private readonly NumericField originPitch = new NumericField();
        private readonly NumericField originYaw = new NumericField();

        private readonly NumericField axisX = new NumericField();
        private readonly NumericField axisY = new NumericField();
        private readonly NumericField axisZ = new NumericField();

        private readonly NumericSpinner limitLower = MakeLimitSpinner();
        private readonly NumericSpinner limitUpper = MakeLimitSpinner();
        private readonly NumericField limitEffort = new NumericField();
        private readonly NumericField limitVelocity = new NumericField();

        private readonly NumericField damping = new NumericField();
        private readonly NumericField friction = new NumericField();

        private readonly CheckBox useDegreesBox =
            new CheckBox { Content = "Use Degrees for Editing", IsChecked = true };

        private readonly TextBlock lowerLabel = MakeFieldLabel();
        private readonly TextBlock upperLabel = MakeFieldLabel();
        private readonly TextBlock effortLabel = MakeFieldLabel();
        private readonly TextBlock velocityLabel = MakeFieldLabel();
        private readonly TextBlock frictionLabel = MakeFieldLabel();
        private readonly TextBlock dampingLabel = MakeFieldLabel();

        private readonly GroupBox originGroup;

        public event EventHandler UseDegreesChanged;
        public event EventHandler LimitsChanged;

        public bool UseDegrees
        {
            get { return useDegreesBox.IsChecked == true; }
            set { useDegreesBox.IsChecked = value; }
        }

        /// <summary>The joint type strings offered in the combo box.</summary>
        public void SetJointTypes(System.Collections.Generic.IEnumerable<string> types)
        {
            jointType.Items.Clear();
            foreach (string type in types)
            {
                jointType.Items.Add(type);
            }
        }

        public JointPropertiesView()
        {
            useDegreesBox.Checked += (s, e) => Raise(UseDegreesChanged);
            useDegreesBox.Unchecked += (s, e) => Raise(UseDegreesChanged);
            limitLower.ValueChanged += (s, e) => Raise(LimitsChanged);
            limitUpper.ValueChanged += (s, e) => Raise(LimitsChanged);

            //StackPanel notes = new StackPanel { HorizontalAlignment = HorizontalAlignment.Right };
            //notes.Children.Add(new TextBlock { Text = "Entries that are blank will not be written to URDF/MJCF" });
            //notes.Children.Add(new TextBlock { Text = "* Field will be overridden from CAD on next export" });
            //notes.Children.Add(new TextBlock { Text = "** Field group is required" });

            originGroup = MakeGroup("Origin* (m, rad)", BuildOrigin());

            Grid middleRow = new Grid();
            middleRow.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(2.1, GridUnitType.Star) });
            middleRow.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1.0, GridUnitType.Star) });
            middleRow.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1.5, GridUnitType.Star) });
            GroupBox axisGroup = MakeGroup("Axis* (normalized)", BuildAxis());
            GroupBox limitsGroup = MakeGroup("Limits**", BuildLimits());
            axisGroup.Margin = new Thickness(8, 0, 0, 0);
            limitsGroup.Margin = new Thickness(8, 0, 0, 0);
            Grid.SetColumn(originGroup, 0);
            Grid.SetColumn(axisGroup, 1);
            Grid.SetColumn(limitsGroup, 2);
            middleRow.Children.Add(originGroup);
            middleRow.Children.Add(axisGroup);
            middleRow.Children.Add(limitsGroup);

            GroupBox dynamicsGroup = MakeGroup("Dynamics", BuildDynamics());
            dynamicsGroup.HorizontalAlignment = HorizontalAlignment.Left;
            dynamicsGroup.Margin = new Thickness(0, 8, 0, 0);

            StackPanel body = new StackPanel();
            body.Children.Add(MakeGroup("Basic Properties", BuildBasicProperties()));
            body.Children.Add(new Border { Height = 8 });
            body.Children.Add(middleRow);
            body.Children.Add(dynamicsGroup);

            useDegreesBox.HorizontalAlignment = HorizontalAlignment.Right;
            useDegreesBox.VerticalAlignment = VerticalAlignment.Bottom;

            // The panel is narrower than the original absolute layout on smaller displays, so the
            // content scrolls rather than being clipped.
            //ScrollViewer bodyScroller = new ScrollViewer
            //{
            //    Content = body,
            //    VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
            //    HorizontalScrollBarVisibility = ScrollBarVisibility.Auto,
            //};

            Grid root = new Grid { Margin = new Thickness(8) };
            root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            //root.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
            root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            //Grid.SetRow(notes, 0);
            Grid.SetRow(body, 1);
            Grid.SetRow(useDegreesBox, 2);
            //root.Children.Add(notes);
            root.Children.Add(body);
            root.Children.Add(useDegreesBox);

            Content = root;
        }

        private static TextBox MakeReadOnly()
        {
            return new TextBox { IsReadOnly = true, Background = SystemColors.ControlBrush };
        }

        private static TextBlock MakeFieldLabel()
        {
            return new TextBlock
            {
                VerticalAlignment = VerticalAlignment.Center,
                HorizontalAlignment = HorizontalAlignment.Right,
                Margin = new Thickness(0, 0, 8, 0),
            };
        }

        private static NumericSpinner MakeLimitSpinner()
        {
            return new NumericSpinner
            {
                Minimum = decimal.MinValue,
                Maximum = decimal.MaxValue,
                DecimalPlaces = 10,
                MinWidth = 70,
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

        /// <summary>Adds a right-aligned caption and its editor as one row of a two-column grid.</summary>
        private static void AddRow(Grid grid, int row, UIElement caption, UIElement editor)
        {
            grid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            Grid.SetRow(caption, row);
            Grid.SetColumn(caption, 0);
            Grid.SetRow(editor, row);
            Grid.SetColumn(editor, 1);
            grid.Children.Add(caption);
            grid.Children.Add(editor);
        }

        private static void AddRow(Grid grid, int row, string caption, UIElement editor)
        {
            TextBlock label = MakeFieldLabel();
            label.Text = caption;
            AddRow(grid, row, label, editor);
        }

        private static Grid MakeFieldGrid(double editorWidth)
        {
            Grid grid = new Grid();
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            grid.ColumnDefinitions.Add(new ColumnDefinition
            {
                Width = editorWidth > 0 ? new GridLength(editorWidth) : new GridLength(1, GridUnitType.Star),
            });
            return grid;
        }

        private UIElement BuildBasicProperties()
        {
            Grid grid = MakeFieldGrid(0);

            AddRow(grid, 0, "Parent Link", parentLink);
            AddRow(grid, 1, "Child Link", childLink);

            // Joint Name and Joint Type share a row in the original layout.
            Grid nameAndType = new Grid();
            nameAndType.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            nameAndType.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            nameAndType.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(160) });
            TextBlock typeLabel = MakeFieldLabel();
            typeLabel.Text = "Joint Type";
            typeLabel.Margin = new Thickness(16, 0, 8, 0);
            Grid.SetColumn(jointName, 0);
            Grid.SetColumn(typeLabel, 1);
            Grid.SetColumn(jointType, 2);
            nameAndType.Children.Add(jointName);
            nameAndType.Children.Add(typeLabel);
            nameAndType.Children.Add(jointType);
            AddRow(grid, 2, "Joint Name", nameAndType);

            AddRow(grid, 3, "Coordinate System", coordSys);
            AddRow(grid, 4, "Joint Axis", refAxis);

            foreach (UIElement child in grid.Children)
            {
                FrameworkElement element = child as FrameworkElement;
                if (element != null)
                {
                    element.Margin = new Thickness(element.Margin.Left, 1, element.Margin.Right, 1);
                }
            }

            return grid;
        }

        private UIElement BuildOrigin()
        {
            Grid grid = new Grid();
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });

            AddOriginPair(grid, 0, "X", originX, "Roll", originRoll);
            AddOriginPair(grid, 1, "Y", originY, "Pitch", originPitch);
            AddOriginPair(grid, 2, "Z", originZ, "Yaw", originYaw);

            return grid;
        }

        private static void AddOriginPair(
            Grid grid, int row, string leftCaption, TextBox left, string rightCaption, TextBox right)
        {
            grid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });

            TextBlock leftLabel = MakeFieldLabel();
            leftLabel.Text = leftCaption;
            TextBlock rightLabel = MakeFieldLabel();
            rightLabel.Text = rightCaption;
            rightLabel.Margin = new Thickness(12, 0, 8, 0);

            left.Margin = new Thickness(0, 1, 0, 1);
            right.Margin = new Thickness(0, 1, 0, 1);

            Grid.SetRow(leftLabel, row);
            Grid.SetColumn(leftLabel, 0);
            Grid.SetRow(left, row);
            Grid.SetColumn(left, 1);
            Grid.SetRow(rightLabel, row);
            Grid.SetColumn(rightLabel, 2);
            Grid.SetRow(right, row);
            Grid.SetColumn(right, 3);

            grid.Children.Add(leftLabel);
            grid.Children.Add(left);
            grid.Children.Add(rightLabel);
            grid.Children.Add(right);
        }

        private UIElement BuildAxis()
        {
            Grid grid = MakeFieldGrid(0);
            AddRow(grid, 0, "X", axisX);
            AddRow(grid, 1, "Y", axisY);
            AddRow(grid, 2, "Z", axisZ);
            SpaceRows(grid);
            return grid;
        }

        private UIElement BuildLimits()
        {
            Grid grid = MakeFieldGrid(0);
            AddRow(grid, 0, lowerLabel, limitLower);
            AddRow(grid, 1, upperLabel, limitUpper);
            AddRow(grid, 2, effortLabel, limitEffort);
            AddRow(grid, 3, velocityLabel, limitVelocity);
            SpaceRows(grid);
            return grid;
        }

        private UIElement BuildDynamics()
        {
            Grid grid = MakeFieldGrid(140);
            AddRow(grid, 0, frictionLabel, friction);
            AddRow(grid, 1, dampingLabel, damping);
            SpaceRows(grid);
            return grid;
        }

        private static void SpaceRows(Grid grid)
        {
            foreach (UIElement child in grid.Children)
            {
                FrameworkElement element = child as FrameworkElement;
                if (element != null)
                {
                    element.Margin = new Thickness(element.Margin.Left, 1, element.Margin.Right, 1);
                }
            }
        }

        private void Raise(EventHandler handler)
        {
            if (handler != null)
            {
                handler(this, EventArgs.Empty);
            }
        }

        public void Clear()
        {
            jointName.Text = "";
            jointType.Text = "";
            parentLink.Text = "";
            childLink.Text = "";
            coordSys.Text = "";
            refAxis.Text = "";
            originX.Value = "";
            originY.Value = "";
            originZ.Value = "";
            originRoll.Value = "";
            originPitch.Value = "";
            originYaw.Value = "";
            axisX.Value = "";
            axisY.Value = "";
            axisZ.Value = "";
            limitEffort.Value = "";
            limitVelocity.Value = "";
            damping.Value = "";
            friction.Value = "";
        }

        public void Show(JointPropertiesViewModel properties)
        {
            ShowLabels(properties.Labels);

            if (!properties.HasJoint)
            {
                return;
            }

            jointName.Text = properties.Name;
            jointType.Text = properties.Type;
            parentLink.Text = properties.ParentLink;
            childLink.Text = properties.ChildLink;
            coordSys.Text = properties.CoordinateSystemName;
            refAxis.Text = properties.AxisName;

            originX.Value = properties.Origin.X;
            originY.Value = properties.Origin.Y;
            originZ.Value = properties.Origin.Z;
            originRoll.Value = properties.Origin.Roll;
            originPitch.Value = properties.Origin.Pitch;
            originYaw.Value = properties.Origin.Yaw;

            axisX.Value = properties.Axis.X;
            axisY.Value = properties.Axis.Y;
            axisZ.Value = properties.Axis.Z;

            limitLower.Increment = properties.LimitIncrement;
            limitUpper.Increment = properties.LimitIncrement;
            limitLower.Value = properties.LimitLower;
            limitUpper.Value = properties.LimitUpper;
            limitEffort.Value = properties.LimitEffort;
            limitVelocity.Value = properties.LimitVelocity;

            damping.Value = properties.Damping;
            friction.Value = properties.Friction;

            SetMotionControlsEnabled(properties.AllowsMotion);
        }

        public JointPropertiesViewModel Read()
        {
            JointPropertiesViewModel properties = new JointPropertiesViewModel();

            properties.Name = jointName.Text;
            properties.Type = jointType.Text;
            properties.ParentLink = parentLink.Text;
            properties.ChildLink = childLink.Text;

            properties.Origin.X = originX.Value;
            properties.Origin.Y = originY.Value;
            properties.Origin.Z = originZ.Value;
            properties.Origin.Roll = originRoll.Value;
            properties.Origin.Pitch = originPitch.Value;
            properties.Origin.Yaw = originYaw.Value;

            properties.Axis.X = axisX.Value;
            properties.Axis.Y = axisY.Value;
            properties.Axis.Z = axisZ.Value;

            properties.LimitLower = limitLower.Value;
            properties.LimitUpper = limitUpper.Value;
            properties.LimitEffort = limitEffort.Value;
            properties.LimitVelocity = limitVelocity.Value;

            properties.Damping = damping.Value;
            properties.Friction = friction.Value;

            return properties;
        }

        public string JointTypeText
        {
            get { return jointType.Text; }
        }

        public void ShowLabels(JointLabels labels)
        {
            lowerLabel.Text = labels.Lower;
            upperLabel.Text = labels.Upper;
            effortLabel.Text = labels.Effort;
            velocityLabel.Text = labels.Velocity;
            frictionLabel.Text = labels.Friction;
            dampingLabel.Text = labels.Damping;
            originGroup.Header = UseDegrees ? "Origin* (m, deg)" : "Origin* (m, rad)";
        }

        public void SetMotionControlsEnabled(bool enabled)
        {
            axisX.IsEnabled = enabled;
            axisY.IsEnabled = enabled;
            axisZ.IsEnabled = enabled;
            limitLower.IsEnabled = enabled;
            limitUpper.IsEnabled = enabled;
            limitEffort.IsEnabled = enabled;
            limitVelocity.IsEnabled = enabled;
            damping.IsEnabled = enabled;
            friction.IsEnabled = enabled;
        }
    }
}
