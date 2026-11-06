/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.IO;
using System.Runtime.InteropServices;

namespace CADRobotExporter.Utilities
{
    /// <summary>
    /// Shows the better Explorer-style folder picker
    /// </summary>
    [ComVisible(false)]
    public static class FolderPicker
    {
        /// <summary>
        /// Asks the user for a folder.
        /// </summary>
        /// <param name="ownerHandle">Window the dialog is modal to, or IntPtr.Zero.</param>
        /// <param name="title">Dialog title.</param>
        /// <param name="initialFolder">Folder to open in; ignored if empty or missing.</param>
        /// <returns>The chosen folder, or null if the user cancelled.</returns>
        public static string Show(IntPtr ownerHandle, string title, string initialFolder)
        {
            IFileDialog dialog;
            try
            {
                dialog = (IFileDialog)new FileOpenDialogCoClass();
            }
            catch (Exception ex) when (ex is COMException || ex is InvalidCastException)
            {
                // The common item dialog is unavailable; fall back to the legacy picker.
                return ShowLegacyDialog(ownerHandle, title, initialFolder);
            }

            return ShowCommonItemDialog(dialog, ownerHandle, title, initialFolder);
        }

        private static string ShowCommonItemDialog(
            IFileDialog dialog, IntPtr ownerHandle, string title, string initialFolder)
        {
            try
            {
                dialog.GetOptions(out uint options);
                dialog.SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST | FOS_NOCHANGEDIR);

                if (!string.IsNullOrEmpty(title))
                {
                    dialog.SetTitle(title);
                }

                if (!string.IsNullOrEmpty(initialFolder) && Directory.Exists(initialFolder))
                {
                    Guid shellItemGuid = typeof(IShellItem).GUID;
                    if (SHCreateItemFromParsingName(initialFolder, IntPtr.Zero, ref shellItemGuid, out IShellItem folder) == 0
                        && folder != null)
                    {
                        dialog.SetFolder(folder);
                        Marshal.ReleaseComObject(folder);
                    }
                }

                int hr = dialog.Show(ownerHandle);
                if (hr == ERROR_CANCELLED_HRESULT)
                {
                    return null;
                }
                Marshal.ThrowExceptionForHR(hr);

                dialog.GetResult(out IShellItem result);
                try
                {
                    result.GetDisplayName(SIGDN_FILESYSPATH, out IntPtr pathPtr);
                    try
                    {
                        return Marshal.PtrToStringUni(pathPtr);
                    }
                    finally
                    {
                        Marshal.FreeCoTaskMem(pathPtr);
                    }
                }
                finally
                {
                    Marshal.ReleaseComObject(result);
                }
            }
            finally
            {
                Marshal.ReleaseComObject(dialog);
            }
        }

        private static string ShowLegacyDialog(IntPtr ownerHandle, string title, string initialFolder)
        {
            using (var dialog = new System.Windows.Forms.FolderBrowserDialog())
            {
                dialog.Description = title;
                if (!string.IsNullOrEmpty(initialFolder))
                {
                    dialog.SelectedPath = initialFolder;
                }

                System.Windows.Forms.DialogResult result = ownerHandle == IntPtr.Zero
                    ? dialog.ShowDialog()
                    : dialog.ShowDialog(new NativeOwner(ownerHandle));

                return result == System.Windows.Forms.DialogResult.OK ? dialog.SelectedPath : null;
            }
        }

        private sealed class NativeOwner : System.Windows.Forms.IWin32Window
        {
            public NativeOwner(IntPtr handle)
            {
                Handle = handle;
            }

            public IntPtr Handle { get; }
        }

        private const uint FOS_NOCHANGEDIR = 0x00000008;
        private const uint FOS_PICKFOLDERS = 0x00000020;
        private const uint FOS_FORCEFILESYSTEM = 0x00000040;
        private const uint FOS_PATHMUSTEXIST = 0x00000800;
        private const uint SIGDN_FILESYSPATH = 0x80058000;

        // HRESULT_FROM_WIN32(ERROR_CANCELLED)
        private const int ERROR_CANCELLED_HRESULT = unchecked((int)0x800704C7);

        [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = true)]
        private static extern int SHCreateItemFromParsingName(
            [MarshalAs(UnmanagedType.LPWStr)] string pszPath,
            IntPtr pbc,
            ref Guid riid,
            [MarshalAs(UnmanagedType.Interface)] out IShellItem ppv);

        [ComImport]
        [Guid("DC1C5A9C-E88A-4dde-A5A1-60F82A20AEF7")]
        private class FileOpenDialogCoClass
        {
        }

        [ComImport]
        [Guid("42f85136-db7e-439c-85f1-e4075d135fc8")]
        [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        private interface IFileDialog
        {
            // IModalWindow
            [PreserveSig]
            int Show(IntPtr hwndOwner);

            // IFileDialog, in vtable order
            void SetFileTypes(uint cFileTypes, IntPtr rgFilterSpec);
            void SetFileTypeIndex(uint iFileType);
            void GetFileTypeIndex(out uint piFileType);
            void Advise(IntPtr pfde, out uint pdwCookie);
            void Unadvise(uint dwCookie);
            void SetOptions(uint fos);
            void GetOptions(out uint pfos);
            void SetDefaultFolder(IShellItem psi);
            void SetFolder(IShellItem psi);
            void GetFolder(out IShellItem ppsi);
            void GetCurrentSelection(out IShellItem ppsi);
            void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string pszName);
            void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string pszName);
            void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string pszTitle);
            void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string pszText);
            void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string pszLabel);
            void GetResult(out IShellItem ppsi);
            void AddPlace(IShellItem psi, int fdap);
            void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string pszDefaultExtension);
            void Close(int hr);
            void SetClientGuid(ref Guid guid);
            void ClearClientData();
            void SetFilter(IntPtr pFilter);
        }

        [ComImport]
        [Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE")]
        [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        private interface IShellItem
        {
            void BindToHandler(IntPtr pbc, ref Guid bhid, ref Guid riid, out IntPtr ppv);
            void GetParent(out IShellItem ppsi);
            void GetDisplayName(uint sigdnName, out IntPtr ppszName);
            void GetAttributes(uint sfgaoMask, out uint psfgaoAttribs);
            void Compare(IShellItem psi, uint hint, out int piOrder);
        }
    }
}
