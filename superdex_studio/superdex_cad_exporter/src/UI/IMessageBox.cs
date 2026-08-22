/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System.Windows;

namespace CADRobotExporter.UI
{
    public interface IMessageBox
    {
        MessageBoxResult Show(string message);
        MessageBoxResult Show(string message, string caption, MessageBoxButton buttons);
    }
}
