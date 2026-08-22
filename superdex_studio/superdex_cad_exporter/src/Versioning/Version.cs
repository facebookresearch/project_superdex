/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

using CADRobotExporter.Utilities;
using System.Diagnostics;

namespace CADRobotExporter.Versioning
{
    internal class Version
    {
        public static string GetCommitVersion()
        {
            // Getting commit version which is attached to the latest git commit
            return FileVersionInfo.GetVersionInfo(typeof(Logger).Assembly.Location).ProductVersion;
        }

        public static string GetBuildVersion()
        {
            // Getting AssemblyVersion which is auto incremented for each build. See the AssemblyInfo.cs
            return typeof(Logger).Assembly.GetName().Version.ToString();
        }
    }
}
