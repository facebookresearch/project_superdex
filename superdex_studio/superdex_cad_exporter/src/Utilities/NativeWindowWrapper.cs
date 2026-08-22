/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace CADRobotExporter.Utilities
{
    /// <summary>
    /// Wraps a native window handle (HWND) as an IWin32Window so a WinForms
    /// form can be owned by the host CAD application's main window (e.g. NX or SolidWorks).
    /// Owning the form to the host window keeps it above the host and lets it
    /// hide/restore together with the host, without staying on top of every application.
    /// </summary>
    [ComVisible(false)]
    public sealed class NativeWindowWrapper : IWin32Window
    {
        public NativeWindowWrapper(IntPtr handle)
        {
            Handle = handle;
        }

        public IntPtr Handle { get; }

        /// <summary>
        /// Returns the main window handle of the current (host) process, or
        /// IntPtr.Zero if it cannot be resolved. Since the exporter runs in-process
        /// inside the CAD application, this is the host application's main window.
        /// </summary>
        public static IntPtr GetHostMainWindowHandle()
        {
            return System.Diagnostics.Process.GetCurrentProcess().MainWindowHandle;
        }
    }
}
