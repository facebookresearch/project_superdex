/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Globalization;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Input;
using System.Windows.Threading;

using CADRobotExporter.UI.ViewModels;

namespace CADRobotExporter.UI.Views
{
    /// <summary>
    /// WPF replacement for WinForms <c>NumericUpDown</c>, which has no WPF equivalent.
    ///
    /// Unlike <c>NumericUpDown</c>, assigning an out-of-range <see cref="Value"/> clamps rather
    /// than throwing.
    /// </summary>
    public class NumericSpinner : UserControl
    {
        private readonly TextBox textBox;
        private decimal value;
        private int decimalPlaces;
        private bool suppressTextSync;

        public event EventHandler ValueChanged;

        public decimal Minimum { get; set; }
        public decimal Maximum { get; set; }
        public decimal Increment { get; set; }

        public int DecimalPlaces
        {
            get { return decimalPlaces; }
            set
            {
                decimalPlaces = value;
                SyncTextFromValue();
            }
        }

        public decimal Value
        {
            get { return value; }
            set { SetValue(value, true); }
        }

        public void SetTextWeight(FontWeight weight)
        {
            textBox.FontWeight = weight;
        }

        public bool DiffersAsShownFrom(double baseline)
        {
            return MeshingSettings.DiffersAsShown(value, baseline, Minimum, Maximum, decimalPlaces);
        }

        public NumericSpinner()
        {
            Minimum = decimal.MinValue;
            Maximum = decimal.MaxValue;
            Increment = 1M;
            decimalPlaces = 2;

            textBox = new TextBox
            {
                BorderThickness = new Thickness(0),
                VerticalContentAlignment = VerticalAlignment.Center,
            };
            textBox.LostFocus += (s, e) => CommitText();
            textBox.TextChanged += OnTextChanged;
            textBox.PreviewKeyDown += OnTextBoxKeyDown;
            textBox.PreviewMouseLeftButtonDown += OnTextBoxMouseDown;

            RepeatButton up = MakeButton("\u25B2");
            RepeatButton down = MakeButton("\u25BC");
            up.Click += (s, e) => Step(1);
            down.Click += (s, e) => Step(-1);

            Grid buttons = new Grid();
            buttons.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
            buttons.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
            Grid.SetRow(up, 0);
            Grid.SetRow(down, 1);
            buttons.Children.Add(up);
            buttons.Children.Add(down);

            Grid root = new Grid();
            root.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            root.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            Grid.SetColumn(textBox, 0);
            Grid.SetColumn(buttons, 1);
            root.Children.Add(textBox);
            root.Children.Add(buttons);

            // The spinner draws its own chrome around a borderless TextBox, so it has to match the
            // themed TextBox: same corner radius, same stroke, same fill.
            Content = new Border
            {
                BorderBrush = new System.Windows.Media.SolidColorBrush(
                    System.Windows.Media.Color.FromRgb(0xCF, 0xCF, 0xCF)),
                BorderThickness = new Thickness(1),
                Background = new System.Windows.Media.SolidColorBrush(
                    System.Windows.Media.Color.FromRgb(0xFF, 0xFF, 0xFF)),
                CornerRadius = new CornerRadius(4),
                Child = root,
            };

            SyncTextFromValue();
        }

        private static RepeatButton MakeButton(string glyph)
        {
            return new RepeatButton
            {
                Content = new TextBlock { Text = glyph, FontSize = 6 },
                Width = 14,
                Padding = new Thickness(0),
                Focusable = false,
            };
        }

        private void Step(int direction)
        {
            SetValue(value + (Increment * direction), true);
        }

        private void OnTextBoxKeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Up)
            {
                Step(1);
                e.Handled = true;
            }
            else if (e.Key == Key.Down)
            {
                Step(-1);
                e.Handled = true;
            }
        }

        // Triple-click selects the whole value, as in NumericField; double-click stops at the
        // decimal point. Either way the field then scrolls back to the start of the value, which
        // the selection's caret would otherwise leave scrolled to its end.
        private void OnTextBoxMouseDown(object sender, MouseButtonEventArgs e)
        {
            if (e.ClickCount == 3)
            {
                textBox.SelectAll();
                e.Handled = true;
            }

            if (e.ClickCount >= 2)
            {
                Dispatcher.BeginInvoke(new Action(textBox.ScrollToHome), DispatcherPriority.Background);
            }
        }

        private void OnTextChanged(object sender, TextChangedEventArgs e)
        {
            if (suppressTextSync)
            {
                return;
            }

            decimal parsed;
            if (decimal.TryParse(textBox.Text, NumberStyles.Float, CultureInfo.CurrentCulture, out parsed))
            {
                // Don't reformat while the user is mid-edit; just track the value.
                SetValue(parsed, false);
            }
        }

        private void CommitText()
        {
            decimal parsed;
            if (decimal.TryParse(textBox.Text, NumberStyles.Float, CultureInfo.CurrentCulture, out parsed))
            {
                SetValue(parsed, true);
            }
            else
            {
                SyncTextFromValue();
            }
        }

        private void SetValue(decimal candidate, bool rewriteText)
        {
            decimal clamped = MeshingSettings.Clamp(candidate, Minimum, Maximum);
            bool changed = clamped != value;
            value = clamped;

            if (rewriteText)
            {
                SyncTextFromValue();
            }

            if (changed)
            {
                EventHandler handler = ValueChanged;
                if (handler != null)
                {
                    handler(this, EventArgs.Empty);
                }
            }
        }

        private void SyncTextFromValue()
        {
            if (textBox == null)
            {
                return;
            }

            suppressTextSync = true;
            textBox.Text = value.ToString("F" + decimalPlaces.ToString(CultureInfo.InvariantCulture), CultureInfo.CurrentCulture);
            suppressTextSync = false;
        }
    }
}
