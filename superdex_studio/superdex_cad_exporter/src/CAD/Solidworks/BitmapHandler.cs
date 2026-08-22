/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using System;
using System.Collections;
using System.Drawing;
using System.IO;
using System.Reflection;

namespace CADRobotExporter.SW
{
    public class BitmapHandler : IDisposable
    {
        private static readonly Lazy<BitmapHandler> _instance =
            new Lazy<BitmapHandler>(() => new BitmapHandler());

        public static BitmapHandler Instance => _instance.Value;

        private ArrayList files;
        private bool _disposed = false;

        private BitmapHandler()
        {
            files = new ArrayList();
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
            {
                return;
            }

            if (disposing)
            {
                CleanFiles();
                _disposed = true;
            }
        }

        public string CreateFileFromResourceBitmap(string bitmapName, Assembly callingAssy)
        {
            string extension = Path.GetExtension(bitmapName);
            string tempFileName = Path.GetTempPath() + Guid.NewGuid().ToString() + extension;
            Bitmap bitmap;
            Stream manifestResourceStream;
            try
            {
                manifestResourceStream = callingAssy.GetManifestResourceStream(bitmapName);
                bitmap = new Bitmap(manifestResourceStream);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return "";
            }

            try
            {
                bitmap.Save(tempFileName);
                files.Add(tempFileName);
                return tempFileName;
            }
            catch (Exception ex2)
            {
                Console.WriteLine(ex2.Message);
                return "";
            }
            finally
            {
                bitmap.Dispose();
                manifestResourceStream.Close();
                manifestResourceStream = null;
            }
        }

        public bool CleanFiles()
        {
            if (files == null) return true;

            foreach (string file in files)
            {
                try
                {
                    File.Delete(file);
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.Message);
                }
            }

            files.Clear();
            return true;
        }
    }
}
