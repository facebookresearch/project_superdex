/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Threading;

namespace CADRobotExporter.UI.Views
{
    /// <summary>
    /// A text box for a number held at full precision. Unfocused, it shows the number to six
    /// significant digits, so a long value keeps its exponent in view, with the full value in the
    /// tooltip; focused, it shows the full value for editing. Read and write it through
    /// <see cref="Value"/>: <see cref="TextBox.Text"/> is only the display. Triple-click selects
    /// the whole field.
    /// </summary>
    public class NumericField : TextBox
    {
        private const string CompactFormat = "G6";

        private string value = "";

        public NumericField()
        {
            // Implicit styles match the exact type, so a subclass would otherwise skip the theme's
            // TextBox style.
            SetResourceReference(StyleProperty, typeof(TextBox));
        }

        /// <summary>The full-precision text, including any edit in progress.</summary>
        public string Value
        {
            get { return IsKeyboardFocused ? Text : value; }
            set
            {
                this.value = value ?? "";
                ToolTip = this.value.Length == 0 ? null : this.value;
                Text = IsKeyboardFocused ? this.value : Compact(this.value);
            }
        }

        protected override void OnGotKeyboardFocus(KeyboardFocusChangedEventArgs e)
        {
            Text = value;
            base.OnGotKeyboardFocus(e);
        }

        protected override void OnPreviewMouseLeftButtonDown(MouseButtonEventArgs e)
        {
            if (e.ClickCount == 3)
            {
                SelectAll();
                e.Handled = true;
            }
            else
            {
                base.OnPreviewMouseLeftButtonDown(e);
            }

            if (e.ClickCount >= 2)
            {
                // A selection leaves the caret at its end, which scrolls a long value to its tail.
                // Queued so it runs after the double-click word selection, which happens later in
                // the same input event.
                Dispatcher.BeginInvoke(new Action(ScrollToHome), DispatcherPriority.Background);
            }
        }

        protected override void OnLostKeyboardFocus(KeyboardFocusChangedEventArgs e)
        {
            Value = Text;
            base.OnLostKeyboardFocus(e);
        }

        private static string Compact(string text)
        {
            double number;
            return double.TryParse(text, out number) ? number.ToString(CompactFormat) : text;
        }
    }
}
